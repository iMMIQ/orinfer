//! Continuous worker and bounded, nonblocking per-client output mailboxes.
use super::*;
use orin_engine::{
    model::{GenerationInput, RequestState},
    scheduler,
};
use std::{collections::VecDeque, sync::atomic::AtomicUsize};

#[derive(Default)]
pub(super) struct Activity {
    pub active: AtomicUsize,
    pub queued: AtomicUsize,
    pub statistics: std::sync::Mutex<scheduler::Statistics>,
}

type Decoder<'a> = tokenizers::tokenizer::DecodeStream<
    'a,
    tokenizers::models::ModelWrapper,
    tokenizers::normalizers::NormalizerWrapper,
    tokenizers::pre_tokenizers::PreTokenizerWrapper,
    tokenizers::processors::PostProcessorWrapper,
    tokenizers::decoders::DecoderWrapper,
>;

struct Mailbox {
    job: Job,
    pending: VecDeque<(ModelEvent, usize)>,
    bytes: usize,
    failed: bool,
}
impl Mailbox {
    fn new(job: Job) -> Self {
        Self {
            job,
            pending: VecDeque::new(),
            bytes: 0,
            failed: false,
        }
    }
    fn flush(&mut self) -> bool {
        while let Some((event, size)) = self.pending.pop_front() {
            match self.job.events.try_send(event) {
                Ok(()) => self.bytes -= size,
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
        if self.bytes.saturating_add(size) > 512 * 1024 || self.pending.len() >= 1024 {
            self.pending.clear();
            self.bytes = 0;
            self.failed = true;
            self.pending
                .push_back((ModelEvent::Failed("Client output buffer is full".into()), 0));
            return false;
        }
        self.pending.push_back((event, size));
        self.bytes += size;
        self.flush()
    }
    fn delta(&mut self, model: &str, delta: Value) -> bool {
        self.send(ModelEvent::Chunk(chunk(&self.job, model, delta, None)))
    }
}
struct Active<'a> {
    request: RequestState,
    mailbox: Mailbox,
    parser: output::Output,
    decoder: Decoder<'a>,
    count: usize,
    eos: bool,
    stopped: bool,
    cancelled: bool,
    error: Option<String>,
    started: Instant,
    queue_s: f64,
}
impl<'a> Active<'a> {
    fn new(request: RequestState, job: Job, codec: &'a ChatCodec) -> Self {
        let parser = output::Output::new(
            job.prepared.thinking,
            job.prepared.tools.clone(),
            job.prepared.stops.clone(),
            &job.id,
        );
        let queue_s = job.queued.elapsed().as_secs_f64();
        Self {
            request,
            mailbox: Mailbox::new(job),
            parser,
            decoder: codec.tokenizer.decode_stream(false),
            count: 0,
            eos: false,
            stopped: false,
            cancelled: false,
            error: None,
            started: Instant::now(),
            queue_s,
        }
    }
    fn consume(&mut self, tokens: &[u32], codec: &ChatCodec, model: &str) {
        for &token in tokens {
            if self.mailbox.job.events.is_closed() || self.mailbox.failed {
                self.cancelled = true;
                break;
            }
            self.count += 1;
            if codec.eos.contains(&token) {
                self.eos = true;
                self.stopped = true;
                break;
            }
            let parsed = self
                .decoder
                .step(token)
                .map_err(|e| e.to_string())
                .and_then(|text| self.parser.push(text.as_deref().unwrap_or(""), false));
            match parsed {
                Ok(deltas) => {
                    for delta in deltas {
                        if !self.mailbox.delta(model, delta) {
                            self.cancelled = true;
                            break;
                        }
                    }
                }
                Err(error) => {
                    self.error = Some(error);
                    break;
                }
            }
            if self.parser.stopped {
                self.stopped = true;
                break;
            }
        }
    }
    fn finished(&self) -> bool {
        self.request.is_finished() || self.stopped || self.cancelled || self.error.is_some()
    }
    fn finish(&mut self, model: &str) -> Result<()> {
        if self.cancelled {
            return Ok(());
        }
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let exhausted =
            !self.eos && !self.parser.stopped && self.count == self.mailbox.job.prepared.max_tokens;
        for delta in self.parser.finish(
            exhausted,
            &self.mailbox.job.prepared.tool_choice,
            self.mailbox.job.prepared.parallel,
        )? {
            if !self.mailbox.delta(model, delta) {
                return Ok(());
            }
        }
        let reason = if exhausted {
            "length"
        } else if !self.parser.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        };
        let job = &self.mailbox.job;
        let usage = json!({"prompt_tokens":job.prepared.input.len(),"completion_tokens":self.count,
            "total_tokens":job.prepared.input.len()+self.count,
            "prompt_tokens_details":{"cached_tokens":self.request.prefix_statistics().cached_tokens}});
        let response = json!({"id":job.id,"object":"chat.completion","created":job.created,"model":model,
            "choices":[{"index":0,"message":self.parser.message(),"finish_reason":reason,"logprobs":null}],"usage":usage});
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
    let mut waiting: Vec<Job> = vec![];
    let mut active: Vec<Active<'_>> = vec![];
    let mut completed: Vec<Mailbox> = vec![];
    loop {
        completed.retain_mut(|box_| box_.flush() && !box_.pending.is_empty());
        while waiting.len() < 128 {
            match jobs.try_recv() {
                Ok(job) => waiting.push(job),
                Err(_) => break,
            }
        }
        waiting.retain(|job| !job.events.is_closed());
        for a in &mut active {
            if !a.mailbox.flush() || a.mailbox.failed || shutdown.load(Ordering::Relaxed) {
                a.cancelled = true;
            }
        }
        let mut index = 0;
        while index < active.len() {
            if active[index].finished() {
                let mut a = active.remove(index);
                let cache =
                    !a.cancelled && a.error.is_none() && a.count == a.request.generated_tokens();
                if let Err(error) = model.finish_request(&mut a.request, cache) {
                    eprintln!("{}: cleanup {error}", a.mailbox.job.id);
                }
                if let Err(error) = a.finish(model_id) {
                    a.mailbox.send(ModelEvent::Failed(error));
                }
                completed.push(a.mailbox);
            } else {
                index += 1;
            }
        }
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        let admission = Instant::now();
        while active.len() < options.max_active && !waiting.is_empty() {
            let costs: Vec<_> = waiting
                .iter()
                .map(|job| {
                    let mut cost = model
                        .estimated_request_cost(&job.prepared.input, &job.prepared.images)
                        .unwrap_or(scheduler::Waiting {
                            age_s: 0.,
                            remaining_s: f64::MAX,
                            restore_s: 0.,
                        });
                    cost.age_s = job.queued.elapsed().as_secs_f64();
                    cost
                })
                .collect();
            let mut admitted = false;
            let mut untried: Vec<_> = (0..waiting.len()).collect();
            while !untried.is_empty() {
                let relative = scheduler::select_waiting(
                    &untried.iter().map(|&i| costs[i]).collect::<Vec<_>>(),
                )
                .expect("nonempty candidates");
                let i = untried.remove(relative);
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
                        let job = waiting.remove(i);
                        match model.start_request(input, || {
                            job.events.is_closed() || shutdown.load(Ordering::Relaxed)
                        }) {
                            Ok(request) => {
                                let mut a = Active::new(request, job, codec);
                                a.mailbox
                                    .delta(model_id, json!({"role":"assistant","content":""}));
                                active.push(a);
                            }
                            Err(error) => {
                                let mut mailbox = Mailbox::new(job);
                                mailbox.send(ModelEvent::Failed(error));
                                completed.push(mailbox);
                            }
                        }
                        admitted = true;
                        break;
                    }
                    Err(error) => {
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
            if !admitted || admission.elapsed().as_millis() >= 10 {
                break;
            }
        }
        activity.active.store(active.len(), Ordering::Relaxed);
        activity
            .queued
            .store(waiting.len() + jobs.len(), Ordering::Relaxed);
        if !active.is_empty() {
            let mut requests: Vec<_> = active.iter_mut().map(|a| &mut a.request).collect();
            match model.advance_requests(&mut requests, &options) {
                Ok(outputs) => {
                    for output in outputs {
                        active[output.request].consume(&output.tokens, codec, model_id);
                    }
                }
                Err(error) => {
                    for a in &mut active {
                        a.error = Some(error.clone());
                    }
                }
            }
            if let Ok(mut statistics) = activity.statistics.lock() {
                *statistics = model.scheduler_statistics().clone();
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
    activity.active.store(0, Ordering::Relaxed);
    activity.queued.store(0, Ordering::Relaxed);
    if let Ok(stats) = serde_json::to_string(model.scheduler_statistics()) {
        eprintln!("SCHEDULER {stats}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mailbox() -> (Mailbox, mpsc::Receiver<ModelEvent>) {
        let (events, receiver) = mpsc::channel(1);
        let job = Job {
            _slot: Arc::new(tokio::sync::Semaphore::new(1))
                .try_acquire_owned()
                .unwrap(),
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
                stream: true,
                include_usage: false,
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
}
