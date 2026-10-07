//! Continuous worker and bounded, nonblocking per-client output mailboxes.
use super::*;
use orinfer_engine::{
    model::{GenerationInput, RequestState},
    scheduler,
};
use std::{collections::VecDeque, sync::atomic::AtomicUsize};

#[derive(Default)]
pub(super) struct Activity {
    pub lifecycle: Arc<lifecycle::Lifecycle>,
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
    pub statistics: std::sync::Mutex<scheduler::Statistics>,
    pub admission: std::sync::Mutex<AdmissionStatistics>,
}

/// Wall time in admission includes policy, memory checks and request startup.
#[derive(Default, Clone, serde::Serialize)]
pub(super) struct AdmissionStatistics {
    rounds: usize,
    requests: usize,
    cached_text_requests: usize,
    wall_s: f64,
    max_round_s: f64,
    histogram: std::collections::BTreeMap<usize, usize>,
    cold_deferrals: usize,
}

type Decoder<'a> = tokenizers::tokenizer::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

fn token_score(codec: &ChatCodec, score: &orinfer_engine::sampling::TokenLogprob) -> Value {
    let entry = |id: u32, logprob: f64| {
        let bytes = codec.grammar.token_bytes(id).unwrap_or_default();
        json!({"token":String::from_utf8_lossy(&bytes),"logprob":logprob,"bytes":bytes})
    };
    let mut result = entry(score.token, score.logprob);
    result["top_logprobs"] = json!(
        score
            .top
            .iter()
            .map(|&(id, p)| entry(id, p))
            .collect::<Vec<_>>()
    );
    result
}

struct Mailbox {
    job: Job,
    pending: VecDeque<(ModelEvent, usize)>,
    bytes: usize,
    failed: bool,
    progress: Instant,
}
impl Mailbox {
    fn new(job: Job) -> Self {
        Self {
            job,
            pending: VecDeque::new(),
            bytes: 0,
            failed: false,
            progress: Instant::now(),
        }
    }
    fn flush(&mut self) -> bool {
        if !self.pending.is_empty()
            && self.progress.elapsed().as_millis() >= self.job.limits.output_ms as u128
        {
            self.failed = true;
            self.pending.clear();
            let _ = self
                .job
                .events
                .try_send(ModelEvent::Failed("Client output deadline exceeded".into()));
            return false;
        }
        while let Some((event, size)) = self.pending.pop_front() {
            match self.job.events.try_send(event) {
                Ok(()) => {
                    self.bytes -= size;
                    self.progress = Instant::now();
                }
                Err(mpsc::error::TrySendError::Full(event)) => {
                    self.pending.push_front((event, size));
                    break;
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    self.pending.clear();
                    self.bytes = 0;
                    return false;
                }
            }
        }
        !self.job.events.is_closed()
    }
    fn send(&mut self, event: ModelEvent) -> bool {
        if self.failed || !self.flush() {
            return false;
        }
        let size = match &event {
            ModelEvent::Chunk(v) | ModelEvent::Complete(v) => v.to_string().len(),
            ModelEvent::Failed(s) => s.len(),
        };
        let ceiling = match &event {
            ModelEvent::Complete(_) if self.job.prepared.sampling.top_logprobs.is_some() => self
                .job
                .prepared
                .max_tokens
                .saturating_mul(self.job.prepared.sampling.top_logprobs.unwrap() + 1)
                .saturating_mul(1024)
                .saturating_add(512 * 1024),
            _ => 512 * 1024,
        };
        if self.bytes.saturating_add(size) > ceiling || self.pending.len() >= 1024 {
            self.pending.clear();
            self.bytes = 0;
            self.failed = true;
            self.pending
                .push_back((ModelEvent::Failed("Client output buffer is full".into()), 0));
            return false;
        }
        if self.pending.is_empty() {
            self.progress = Instant::now();
        }
        self.pending.push_back((event, size));
        self.bytes += size;
        self.flush()
    }
    fn delta(&mut self, model: &str, delta: Value) -> bool {
        self.delta_scored(model, delta, None)
    }
    fn delta_scored(&mut self, model: &str, delta: Value, logprobs: Option<Value>) -> bool {
        let mut payload = chunk(&self.job, model, delta, None);
        if let Some(logprobs) = logprobs {
            payload["choices"][0]["logprobs"] = logprobs;
        }
        self.send(ModelEvent::Chunk(payload))
    }
}
enum Ending {
    Eos,
    Stop,
    Cancelled,
    Failed(String),
}
impl Ending {
    fn cacheable(&self) -> bool {
        matches!(self, Self::Eos | Self::Stop)
    }
}
struct Active<'a> {
    request: RequestState,
    mailbox: Mailbox,
    parser: output::Output,
    decoder: Decoder<'a>,
    count: usize,
    ending: Option<Ending>,
    started: Instant,
    queue_s: f64,
    reasoning_tokens: usize,
    raw_bytes: usize,
    pending_scores: VecDeque<(std::ops::Range<usize>, Value)>,
    content_scores: Vec<Value>,
}
impl<'a> Active<'a> {
    fn new(request: RequestState, job: Job, codec: &'a ChatCodec) -> Self {
        let mut parser = output::Output::new(
            job.prepared.thinking,
            job.prepared.tools.clone(),
            job.prepared.stops.clone(),
            &job.id,
        );
        parser.structured_json(job.prepared.structured);
        parser.constrained_tools(job.prepared.constrained_tools);
        let queue_s = job.queued.elapsed().as_secs_f64();
        Self {
            request,
            mailbox: Mailbox::new(job),
            parser,
            decoder: codec.tokenizer.decode_stream(false),
            count: 0,
            ending: None,
            started: Instant::now(),
            queue_s,
            reasoning_tokens: 0,
            raw_bytes: 0,
            pending_scores: VecDeque::new(),
            content_scores: vec![],
        }
    }
    fn consume(&mut self, tokens: &[u32], codec: &ChatCodec, model: &str) {
        let scores = self.request.take_logprobs();
        for (index, &token) in tokens.iter().enumerate() {
            if self.mailbox.job.events.is_closed() || self.mailbox.failed {
                self.ending = Some(Ending::Cancelled);
                break;
            }
            self.count += 1;
            if codec.eos.contains(&token) {
                self.ending = Some(Ending::Eos);
                break;
            }
            if let Some(score) = scores.get(index) {
                let bytes = match codec.grammar.token_bytes(token) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        self.ending = Some(Ending::Failed(error));
                        break;
                    }
                };
                let span = self.raw_bytes..self.raw_bytes + bytes.len();
                self.raw_bytes = span.end;
                self.pending_scores
                    .push_back((span, token_score(codec, score)));
            }
            let was_reasoning = self.parser.is_reasoning();
            let parsed = self
                .decoder
                .step(token)
                .map_err(|e| e.to_string())
                .and_then(|text| self.parser.push(text.as_deref().unwrap_or(""), false));
            match parsed {
                Ok(deltas) => {
                    if (was_reasoning && codec.tokenizer.token_to_id("</think>") != Some(token))
                        || deltas.iter().any(|d| d.get("reasoning_content").is_some())
                    {
                        self.reasoning_tokens += 1;
                    }
                    for delta in deltas {
                        let logprobs = self.delta_scores(&delta);
                        if !self.mailbox.delta_scored(model, delta, logprobs) {
                            self.ending = Some(Ending::Cancelled);
                            break;
                        }
                    }
                }
                Err(error) => {
                    self.ending = Some(Ending::Failed(error));
                    break;
                }
            }
            while self
                .pending_scores
                .front()
                .is_some_and(|(span, _)| span.end <= self.parser.processed_bytes())
            {
                self.pending_scores.pop_front();
            }
            if self.ending.is_some() {
                break;
            }
            if self.parser.stopped {
                self.ending = Some(Ending::Stop);
                break;
            }
        }
    }
    fn delta_scores(&mut self, delta: &Value) -> Option<Value> {
        delta.get("content")?;
        // Output spans are consumed in content-delta order, including text
        // withheld across UTF-8, XML markers and stop-string boundaries.
        let spans = self.parser.take_content_span();
        let mut content = vec![];
        if let Some(span) = spans {
            while self
                .pending_scores
                .front()
                .is_some_and(|(s, _)| s.end <= span.start)
            {
                self.pending_scores.pop_front();
            }
            while self
                .pending_scores
                .front()
                .is_some_and(|(s, _)| s.start < span.end)
            {
                let (range, mut score) = self.pending_scores.pop_front().unwrap();
                let bytes: Vec<u8> = serde_json::from_value(score["bytes"].clone()).unwrap();
                let start = span.start.saturating_sub(range.start);
                let end = (span.end - range.start).min(bytes.len());
                let visible = &bytes[start..end];
                let mut fragment = score.clone();
                fragment["token"] = json!(String::from_utf8_lossy(visible));
                fragment["bytes"] = json!(visible);
                content.push(fragment);
                if range.end > span.end {
                    score["bytes"] = json!(&bytes[end..]);
                    self.pending_scores.push_front((span.end..range.end, score));
                    break;
                }
            }
        }
        self.mailbox.job.prepared.sampling.top_logprobs?;
        if !self.mailbox.job.prepared.stream {
            self.content_scores.extend(content.clone());
        }
        Some(json!({"content":content,"refusal":null}))
    }
    fn finished(&self) -> bool {
        self.request.is_finished() || self.ending.is_some()
    }
    fn finish(&mut self, model: &str) -> Result<()> {
        if let Some(error) = self.request.failure() {
            return Err(error.into());
        }
        match &self.ending {
            Some(Ending::Cancelled) => return Ok(()),
            Some(Ending::Failed(error)) => return Err(error.clone()),
            _ => {}
        }
        let exhausted = self.ending.is_none()
            && self.count == self.mailbox.job.prepared.max_tokens
            && !self.request.constraint_finished();
        for delta in self.parser.finish(
            exhausted,
            &self.mailbox.job.prepared.tool_choice,
            self.mailbox.job.prepared.parallel,
        )? {
            let logprobs = self.delta_scores(&delta);
            if !self.mailbox.delta_scored(model, delta, logprobs) {
                return Ok(());
            }
        }
        let reason = if exhausted || self.parser.incomplete_call {
            "length"
        } else if !self.parser.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        let job = &self.mailbox.job;
        let usage = json!({"prompt_tokens":job.prepared.input.len(),"completion_tokens":self.count,
            "total_tokens":job.prepared.input.len()+self.count,
            "completion_tokens_details":{"reasoning_tokens":self.reasoning_tokens},
            "prompt_tokens_details":{"cached_tokens":self.request.prefix_statistics().cached_tokens}});
        let logprobs = job
            .prepared
            .sampling
            .top_logprobs
            .map(|_| json!({"content":self.content_scores,"refusal":null}));
        let response = if job.prepared.stream {
            // SSE has already delivered content and scores. Complete is only
            // its terminal signal; do not retain a second full response.
            json!({})
        } else {
            json!({"id":job.id,"object":"chat.completion","created":job.created,"model":model,
                "choices":[{"index":0,"message":self.parser.message(),"finish_reason":reason,"logprobs":logprobs}],"usage":usage})
        };
        let ending = chunk(job, model, json!({}), Some(reason));
        let usage_chunk=job.prepared.include_usage.then(||json!({"id":job.id,"object":"chat.completion.chunk","created":job.created,"model":model,"choices":[],"usage":usage}));
        self.mailbox.send(ModelEvent::Chunk(ending));
        if let Some(usage) = usage_chunk {
            self.mailbox.send(ModelEvent::Chunk(usage));
        }
        self.mailbox.send(ModelEvent::Complete(response));
        eprintln!(
            "{}: {} prompt, {} completion, queue {:.3}s, generation {:.3}s, finish {reason}",
            self.mailbox.job.id,
            self.mailbox.job.prepared.input.len(),
            self.count,
            self.queue_s,
            self.started.elapsed().as_secs_f64()
        );
        eprintln!(
            "{}: prefix {}; MTP {}",
            self.mailbox.job.id,
            serde_json::to_string(self.request.prefix_statistics()).map_err(|e| e.to_string())?,
            serde_json::to_string(self.request.speculation_statistics())
                .map_err(|e| e.to_string())?
        );
        Ok(())
    }
}

pub(super) fn worker(
    model: &mut Model,
    mut jobs: mpsc::Receiver<Job>,
    codec: &ChatCodec,
    model_id: &str,
    shutdown: &AtomicBool,
    options: scheduler::Options,
    activity: &Activity,
) {
    let lifecycle = &activity.lifecycle;
    let mut waiting: Vec<Job> = vec![];
    let mut active: Vec<Active<'_>> = vec![];
    let mut completed: Vec<Mailbox> = vec![];
    let mut admission_costs = scheduler::AdmissionCosts::default();
    loop {
        if shutdown.load(Ordering::Relaxed) || !lifecycle.is_ready() {
            break;
        }
        completed.retain_mut(|box_| box_.flush() && !box_.pending.is_empty());
        receive_waiting(&mut waiting, &mut jobs);
        for a in &mut active {
            if !a.mailbox.flush() || a.mailbox.failed || shutdown.load(Ordering::Relaxed) {
                a.ending = Some(Ending::Cancelled);
            }
        }
        let mut index = 0;
        while index < active.len() {
            if active[index].finished() {
                let mut a = active.remove(index);
                let cache = a.ending.as_ref().is_none_or(Ending::cacheable)
                    && a.count == a.request.generated_tokens()
                    && a.request.failure().is_none();
                if lifecycle.is_ready()
                    && let Err(error) = model.finish_request(&mut a.request, cache)
                {
                    let failure = orinfer_engine::error::EngineError::take(error.clone(), true);
                    lifecycle.fail(failure);
                    a.ending = Some(Ending::Failed(error));
                }
                if let Err(error) = a.finish(model_id) {
                    a.mailbox.send(ModelEvent::Failed(error));
                }
                completed.push(a.mailbox);
            } else {
                index += 1;
            }
        }
        if shutdown.load(Ordering::Relaxed) || !lifecycle.is_ready() {
            break;
        }
        let admission = Instant::now();
        let has_decoders = active.iter().any(|a| !a.request.is_prefilling());
        let has_prefills = active.iter().any(|a| a.request.is_prefilling());
        let decoder_count = active.iter().filter(|a| !a.request.is_prefilling()).count();
        let remaining_decode_tokens: usize = active
            .iter()
            .filter(|a| !a.request.is_prefilling())
            .map(|a| a.mailbox.job.prepared.max_tokens.saturating_sub(a.count))
            .sum();
        let mut admitted_count = 0;
        let mut cached_count = 0;
        let mut cold_deferrals = 0;
        while lifecycle.is_ready() && active.len() < options.max_active && !waiting.is_empty() {
            let costs: Vec<_> = waiting
                .iter()
                .map(|job| {
                    let mut cost = model
                        .estimated_request_cost(&job.prepared.input, &job.prepared.images)
                        .unwrap_or(scheduler::Waiting {
                            age_s: 0.,
                            remaining_s: f64::MAX,
                            restore_s: 0.,
                            remaining_tokens: 0,
                        });
                    cost.age_s = job.queued.elapsed().as_secs_f64();
                    cost
                })
                .collect();
            let mut admitted = false;
            let mut cached_text = false;
            let mut untried: Vec<_> = (0..waiting.len()).collect();
            while !untried.is_empty() {
                let relative = scheduler::select_waiting(
                    &untried.iter().map(|&i| costs[i]).collect::<Vec<_>>(),
                )
                .expect("nonempty candidates");
                let i = untried.remove(relative);
                if admission_costs.defer_cold(
                    costs[i],
                    decoder_count,
                    remaining_decode_tokens,
                    has_prefills,
                ) {
                    cold_deferrals += 1;
                    continue;
                }
                let job = &waiting[i];
                let input = GenerationInput {
                    input_tokens: job.prepared.input.clone(),
                    images: job.prepared.images.clone(),
                    max_new_tokens: job.prepared.max_tokens,
                    sampling: job.prepared.sampling.clone(),
                    prefix_hints: job.prepared.prefix_hints.clone(),
                };
                match model.can_admit(&input, &options) {
                    Ok(true) => {
                        let mut job = waiting.remove(i);
                        let text_only = input.images.is_empty();
                        let prompt_tokens = input.input_tokens.len();
                        match model.start_request(input, || {
                            job.events.is_closed() || shutdown.load(Ordering::Relaxed)
                        }) {
                            Ok(mut request) => {
                                if let Some(constraint) = job.prepared.constraint.take() {
                                    request
                                        .set_constraint(constraint)
                                        .expect("Request has not generated tokens");
                                }
                                cached_text = text_only
                                    && request.prefix_statistics().cached_tokens == prompt_tokens;
                                let mut a = Active::new(request, job, codec);
                                a.mailbox
                                    .delta(model_id, json!({"role":"assistant","content":""}));
                                active.push(a);
                            }
                            Err(error) => {
                                let failure =
                                    orinfer_engine::error::EngineError::take(error.clone(), false);
                                if failure.is_fatal() {
                                    lifecycle.fail(failure);
                                }
                                let mut mailbox = Mailbox::new(job);
                                mailbox.send(ModelEvent::Failed(error));
                                completed.push(mailbox);
                            }
                        }
                        admitted = true;
                        break;
                    }
                    Err(error) => {
                        let failure =
                            orinfer_engine::error::EngineError::take(error.clone(), false);
                        if failure.is_fatal() {
                            lifecycle.fail(failure);
                        }
                        let job = waiting.remove(i);
                        let mut mailbox = Mailbox::new(job);
                        mailbox.send(ModelEvent::Failed(error));
                        completed.push(mailbox);
                        admitted = true;
                        break;
                    }
                    Ok(false) => {}
                }
            }
            if admitted {
                admitted_count += 1;
                cached_count += usize::from(cached_text);
                // HTTP arrivals continue while CUDA restores a prefix. Include
                // already-arrived work in the cohort without a batching timer.
                if cached_text || !has_decoders {
                    receive_waiting(&mut waiting, &mut jobs);
                }
            }
            if !admitted
                || scheduler::admission_should_yield(
                    has_decoders,
                    admitted_count,
                    admission.elapsed().as_secs_f64(),
                    cached_text,
                )
            {
                break;
            }
        }
        if admitted_count > 0 {
            let seconds = admission.elapsed().as_secs_f64();
            if let Ok(mut statistics) = activity.admission.lock() {
                statistics.rounds += 1;
                statistics.requests += admitted_count;
                statistics.cached_text_requests += cached_count;
                statistics.wall_s += seconds;
                statistics.max_round_s = statistics.max_round_s.max(seconds);
                *statistics.histogram.entry(admitted_count).or_default() += 1;
            }
        }
        if cold_deferrals > 0
            && let Ok(mut statistics) = activity.admission.lock()
        {
            statistics.cold_deferrals += cold_deferrals;
        }
        activity.active.store(active.len(), Ordering::Relaxed);
        activity
            .queued
            .store(waiting.len() + jobs.len(), Ordering::Relaxed);
        if !lifecycle.is_ready() {
            break;
        }
        if !active.is_empty() {
            let before = model.scheduler_statistics();
            let step_at = Instant::now();
            let decoders = active.iter().filter(|a| !a.request.is_prefilling()).count();
            let mut requests: Vec<_> = active.iter_mut().map(|a| &mut a.request).collect();
            match model.advance_requests(&mut requests, &options) {
                Ok(outputs) => {
                    for output in outputs {
                        active[output.request].consume(&output.tokens, codec, model_id);
                    }
                }
                Err(error) => {
                    lifecycle.fail(orinfer_engine::error::EngineError::take(
                        error.clone(),
                        true,
                    ));
                    for a in &mut active {
                        a.ending = Some(Ending::Failed(error.clone()));
                    }
                }
            }
            let after = model.scheduler_statistics();
            admission_costs.observe(
                decoders,
                after.decode_tokens.saturating_sub(before.decode_tokens),
                after.prefill_tokens.saturating_sub(before.prefill_tokens),
                step_at.elapsed().as_secs_f64(),
            );
            if let Ok(mut statistics) = activity.statistics.lock() {
                *statistics = after;
            }
        } else if waiting.is_empty() && completed.is_empty() {
            let Some(job) = jobs.blocking_recv() else {
                break;
            };
            waiting.push(job);
        } else {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }
    jobs.close();
    for job in waiting
        .drain(..)
        .chain(std::iter::from_fn(|| jobs.try_recv().ok()))
    {
        let _ = job
            .events
            .try_send(ModelEvent::Failed("GPU worker stopped".into()));
    }
    for mut a in active.drain(..) {
        if lifecycle.name() != "failed"
            && let Err(error) = model.finish_request(&mut a.request, false)
        {
            lifecycle.fail(orinfer_engine::error::EngineError::take(error, true));
        }
        let _ = a
            .mailbox
            .job
            .events
            .try_send(ModelEvent::Failed("GPU worker stopped".into()));
    }
    for mailbox in &mut completed {
        mailbox.flush();
    }
    activity.active.store(0, Ordering::Relaxed);
    activity.queued.store(0, Ordering::Relaxed);
    if let Ok(stats) = serde_json::to_string(&model.scheduler_statistics()) {
        eprintln!("SCHEDULER {stats}");
    }
}

fn receive_waiting(waiting: &mut Vec<Job>, jobs: &mut mpsc::Receiver<Job>) {
    while waiting.len() < 128 {
        match jobs.try_recv() {
            Ok(job) => waiting.push(job),
            Err(_) => break,
        }
    }
    waiting.retain(|job| {
        if job.limits.queue_ms != 0
            && job.queued.elapsed().as_millis() >= job.limits.queue_ms as u128
        {
            let _ = job
                .events
                .try_send(ModelEvent::Failed("Request queue deadline exceeded".into()));
            return false;
        }
        !job.events.is_closed()
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mailbox() -> (Mailbox, mpsc::Receiver<ModelEvent>) {
        let (events, receiver) = mpsc::channel(1);
        let job = Job {
            _slot: Arc::new(
                Arc::new(tokio::sync::Semaphore::new(1))
                    .try_acquire_owned()
                    .unwrap(),
            ),
            _memory: None,
            limits: Limits::default(),
            prepared: Prepared {
                input: vec![1],
                images: vec![],
                max_tokens: 16,
                sampling: Default::default(),
                tools: vec![],
                tool_choice: json!("auto"),
                parallel: false,
                stops: vec![],
                thinking: false,
                structured: false,
                constrained_tools: false,
                stream: true,
                include_usage: false,
                constraint: None,
                prefix_hints: vec![],
            },
            events,
            id: "test".into(),
            created: 0,
            queued: Instant::now(),
        };
        (Mailbox::new(job), receiver)
    }

    #[test]
    fn long_prefill_does_not_start_the_slow_reader_deadline_early() {
        let (mut mailbox, _receiver) = mailbox();
        mailbox.progress = Instant::now() - std::time::Duration::from_secs(120);
        assert!(mailbox.send(ModelEvent::Chunk(json!(1))));
        assert!(!mailbox.failed);
        assert!(mailbox.send(ModelEvent::Chunk(json!(2))));
        mailbox.progress = Instant::now() - std::time::Duration::from_secs(120);
        assert!(!mailbox.flush());
        assert!(mailbox.failed);
    }
    #[test]
    fn admission_collects_arrivals_and_discards_disconnected_waiters() {
        let (sender, mut jobs) = mpsc::channel(4);
        let (first, first_receiver) = mailbox();
        let (cancelled, cancelled_receiver) = mailbox();
        sender.try_send(first.job).unwrap();
        sender.try_send(cancelled.job).unwrap();
        drop(cancelled_receiver);
        let mut waiting = Vec::new();
        receive_waiting(&mut waiting, &mut jobs);
        assert_eq!(waiting.len(), 1);
        let (arrival, arrival_receiver) = mailbox();
        sender.try_send(arrival.job).unwrap();
        receive_waiting(&mut waiting, &mut jobs);
        assert_eq!(waiting.len(), 2);
        drop((first_receiver, arrival_receiver));
        receive_waiting(&mut waiting, &mut jobs);
        assert!(waiting.is_empty());
    }

    #[test]
    fn full_transport_retains_event_order_without_blocking_worker() {
        let (mut mailbox, mut receiver) = mailbox();
        for id in 0..3 {
            assert!(mailbox.send(ModelEvent::Chunk(json!(id))));
        }
        assert_eq!(mailbox.pending.len(), 2);
        for id in 0..3 {
            assert!(matches!(receiver.try_recv().unwrap(), ModelEvent::Chunk(v) if v == json!(id)));
            assert!(mailbox.flush());
        }
        assert!(mailbox.pending.is_empty());
        assert_eq!(mailbox.bytes, 0);
    }

    #[test]
    fn slow_or_disconnected_client_has_bounded_retained_output() {
        let (mut mailbox, receiver) = mailbox();
        assert!(mailbox.send(ModelEvent::Chunk(json!(0))));
        assert!(!mailbox.send(ModelEvent::Chunk(json!("x".repeat(512 * 1024)))));
        assert!(mailbox.failed);
        assert_eq!(mailbox.pending.len(), 1);
        assert!(!mailbox.send(ModelEvent::Complete(json!({}))));
        drop(receiver);
        assert!(!mailbox.flush());
        assert!(mailbox.pending.is_empty());
    }
    #[test]
    fn large_probability_completion_uses_reserved_budget_without_unbounding_chunks() {
        let (mut mailbox, _receiver) = mailbox();
        mailbox.job.prepared.sampling.top_logprobs = Some(20);
        mailbox.job.prepared.max_tokens = 1024;
        assert!(mailbox.send(ModelEvent::Chunk(json!(0))));
        assert!(mailbox.send(ModelEvent::Complete(json!("x".repeat(640 * 1024)))));
        assert_eq!(mailbox.pending.len(), 1);
        assert!(!mailbox.failed);
        let (mut mailbox, _receiver) = self::mailbox();
        mailbox.job.prepared.sampling.top_logprobs = Some(20);
        assert!(mailbox.send(ModelEvent::Chunk(json!(0))));
        assert!(!mailbox.send(ModelEvent::Chunk(json!("x".repeat(640 * 1024)))));
    }
}
