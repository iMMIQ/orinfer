//! Local JSONL inference uses the same protocol and continuous GPU worker as HTTP.
mod files;
mod planning;
#[cfg(test)]
mod tests;

use super::*;
use crate::{BatchConfig, BatchOrder};
use files::{Input, Journal, Outcome, Record};
use futures_util::stream::FuturesUnordered;

#[derive(Default, Debug, serde::Serialize)]
pub struct BatchReport {
    pub scheduler_policy: orinfer_engine::scheduler::Policy,
    pub batch_order: BatchOrder,
    pub planned_requests: usize,
    pub planning_prefix_tokens: usize,
    pub planning_s: f64,
    pub total: usize,
    pub succeeded: usize,
    pub truncated: usize,
    pub failed: usize,
    pub unfinished: usize,
    pub skipped: usize,
    pub processed: usize,
    /// Tokens generated in this invocation, including truncated responses.
    pub completion_tokens: usize,
    pub elapsed_s: f64,
    pub output_tps: f64,
    pub peak_active: usize,
    pub interrupted: bool,
}

pub fn run(config: BatchConfig) -> Result<BatchReport> {
    let started = Instant::now();
    let settings = config.settings();
    settings.validate()?;
    // Offline outputs are complete Chat responses even if a deployment's HTTP
    // defaults request streaming. Explicit row-level stream=true is rejected.
    let mut defaults = config.default_request_params.clone();
    defaults.insert("stream".into(), json!(false));
    defaults.remove("stream_options");
    let mut input = Input::open(&config.input)?;
    let descriptor = std::fs::read(config.model_dir.join("cache/model.json"))
        .map_err(|e| format!("Read model identity: {e}"))?;
    let identity = json!({"schema":1,"input_sha256":input.sha256,
        "model_sha256":orinfer_engine::artifact::sha256(&descriptor),"model":config.model,
        "default_request_params":defaults,"mtp_drafts":config.load.mtp_drafts,
        "prompt_lookup":config.load.prompt_lookup});
    let mut journal = Journal::open(&config.output, config.resume, &identity, &input)?;
    let mut report = BatchReport {
        scheduler_policy: config.scheduler.policy,
        batch_order: config.order,
        total: input.ids.len(),
        skipped: journal
            .states
            .values()
            .filter(|s| **s != Outcome::Failed)
            .count(),
        ..Default::default()
    };
    if report.skipped < report.total {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let result = runtime.block_on(async {
            let worker = startup::start(&settings).await?;
            let result = async {
                if config.order == BatchOrder::Auto
                    && config.scheduler.policy == orinfer_engine::scheduler::Policy::Throughput
                {
                    tokio::select! {
                        result = planning::prepare(&mut input, &worker.state, &defaults,
                            config.load.prefix_cache_bytes > 0 || config.load.kv_offload_bytes > 0, &mut report) => result?,
                        _ = shutdown_signal() => { report.interrupted = true; return Ok(()); }
                    }
                }
                execute(
                    &mut input,
                    &mut journal,
                    worker.state.clone(),
                    &defaults,
                    &mut report,
                )
                .await
            }
            .await;
            let stopped = worker.stop(settings.limits.drain_ms).await;
            result.and(stopped)
        });
        runtime.shutdown_timeout(std::time::Duration::from_secs(2));
        let synced = journal.sync();
        result.and(synced)?;
    }
    for status in journal.states.values() {
        match status {
            Outcome::Success => report.succeeded += 1,
            Outcome::Truncated => report.truncated += 1,
            Outcome::Failed => report.failed += 1,
        }
    }
    report.unfinished = report.total - journal.states.len();
    report.elapsed_s = started.elapsed().as_secs_f64();
    report.output_tps = report.completion_tokens as f64 / report.elapsed_s.max(1e-9);
    Ok(report)
}

struct Completed {
    value: Value,
    bytes: usize,
    // Keep preprocessing/output memory charged until the journal accepts it.
    _memory: Option<Arc<preparation::Lease>>,
}
fn failure(id: &str, code: &str, message: String) -> Value {
    json!({"id":format!("batch-{id}"),"custom_id":id,"response":null,
        "error":{"code":code,"message":message}})
}
fn request(
    mut body: Value,
    model: &str,
    defaults: &serde_json::Map<String, Value>,
) -> Result<ChatRequest> {
    let object = body
        .as_object_mut()
        .ok_or("Expected a Chat request object")?;
    object.entry("model").or_insert_with(|| json!(model));
    let request = defaults::decode(body, defaults)?;
    if request.stream || request.stream_options.is_some() {
        return Err("Offline requests require stream=false and no stream_options".into());
    }
    Ok(request)
}

async fn submit(
    state: Service,
    record: Record,
    defaults: &serde_json::Map<String, Value>,
) -> Completed {
    let mut memory = None;
    let result = async {
        let request = request(record.body?, &state.model, defaults)?;
        let estimate = preparation::request_memory(
            &request,
            state.context,
            state.vision.as_ref(),
            state.limits.memory_mib,
        );
        let memory_mib = estimate.mib;
        if memory_mib as usize > state.limits.memory_mib {
            return Err("Request exceeds preprocessing memory budget".into());
        }
        let reserved = Arc::clone(&state.preparation.memory)
            .acquire_many_owned(memory_mib)
            .await
            .map_err(|e| e.to_string())?;
        let cpu = Arc::clone(&state.preparation.workers)
            .acquire_owned()
            .await
            .map_err(|e| e.to_string())?;
        let slot = Arc::new(
            Arc::clone(&state.slots)
                .acquire_owned()
                .await
                .map_err(|e| e.to_string())?,
        );
        let codec = Arc::clone(&state.codec);
        let model = Arc::clone(&state.model);
        let vision = state.vision.clone();
        let context = state.context;
        let cancellation = preparation::Cancellation::default();
        let flag = cancellation.flag();
        let prefix_hint = record.prefix_hint;
        let prepared = tokio::task::spawn_blocking(move || {
            let (_cpu, mut memory) = (cpu, reserved);
            let mut context_budget = estimate.context(flag);
            let mut prepared = codec.prepare(
                request,
                &model,
                context,
                vision.as_ref(),
                &mut context_budget,
            )?;
            if let Some(prefix_hint) = prefix_hint
                && prefix_hint >= 32
                && prefix_hint < prepared.input.len()
            {
                prepared.prefix_hints.push(prefix_hint);
                prepared.prefix_hints.sort_unstable();
                prepared.prefix_hints.dedup();
            }
            context_budget.checkpoint(0)?;
            if !prepared.images.is_empty() || !prepared.audios.is_empty() {
                context_budget.release_media_scratch(&mut memory);
            }
            Ok::<_, String>((prepared, slot, Arc::new(context_budget.lease(memory))))
        })
        .await
        .map_err(|e| e.to_string())??;
        let (prepared, slot, reserved) = prepared;
        memory = Some(Arc::clone(&reserved));
        let memory = reserved;
        if !state.activity.lifecycle.is_ready() {
            return Err("GPU worker unavailable".into());
        }
        let (events, mut responses) = mpsc::channel(1);
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        state
            .jobs
            .send(Job {
                _slot: slot,
                _memory: Some(memory),
                limits: state.limits,
                prepared,
                events,
                id: format!("chatcmpl-batch-{}-{}", std::process::id(), record.line),
                created,
                queued: Instant::now(),
                hint: None,
                shared_prefix: record.prefix_hint,
            })
            .await
            .map_err(|_| "GPU worker disconnected")?;
        while let Some(event) = responses.recv().await {
            match event {
                ModelEvent::Complete(body) => return Ok(body),
                ModelEvent::Failed(error) | ModelEvent::Unavailable(error) => return Err(error),
                ModelEvent::Chunk(_) => {}
            }
        }
        Err("GPU worker disconnected before completion".into())
    }
    .await;
    let value = match result {
        Ok(body) => json!({"id":format!("batch-{}",record.id),"custom_id":record.id,
            "response":{"status_code":200,"request_id":body["id"],"body":body},"error":null}),
        Err(error) => failure(&record.id, "request_failed", error),
    };
    Completed {
        value,
        bytes: record.bytes,
        _memory: memory,
    }
}

async fn execute(
    input: &mut Input,
    journal: &mut Journal,
    state: Service,
    defaults: &serde_json::Map<String, Value>,
    report: &mut BatchReport,
) -> Result<()> {
    let mut pending = FuturesUnordered::new();
    let window = state.scheduler.max_active + 128;
    let writer = files::Writer::start(journal, window)?;
    let byte_budget = state.limits.memory_mib * (1 << 20) / 4;
    let mut bytes = 0;
    let mut eof = false;
    let signal = shutdown_signal();
    tokio::pin!(signal);
    let result = async {
        loop {
            while !eof && pending.len() < window && bytes < byte_budget {
                match input.next()? {
                    Some(record) if !journal.done(&record.id) => {
                        bytes += record.bytes;
                        pending.push(submit(state.clone(), record, defaults));
                    }
                    Some(_) => {}
                    None => eof = true,
                }
            }
            if pending.is_empty() {
                break;
            }
            let completed = tokio::select! {
                _ = &mut signal => { report.interrupted = true; break; },
                completed = pending.next() => completed.unwrap(),
            };
            bytes -= completed.bytes;
            let tokens = completed.value["response"]["body"]["usage"]["completion_tokens"]
                .as_u64()
                .unwrap_or(0) as usize;
            writer.append(journal, completed).await?;
            report.processed += 1;
            report.completion_tokens += tokens;
            if report.processed.is_multiple_of(100) {
                eprintln!(
                    "BATCH {}/{} records completed",
                    report.skipped + report.processed,
                    report.total
                );
            }
            if state.activity.lifecycle.name() == "failed" {
                return Err(format!(
                    "GPU worker failed: {:?}",
                    state.activity.lifecycle.failure()
                ));
            }
        }
        Ok(())
    }
    .await;
    drop(pending);
    // Dropping pending futures closes receivers and cancels in-flight CPU work;
    // the worker owns/reclaims every GPU arena before its caller joins it.
    report.peak_active = state
        .activity
        .statistics
        .lock()
        .map(|s| s.peak_active)
        .unwrap_or(0);
    writer.finish().await.and(result)
}
