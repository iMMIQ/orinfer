mod chat;
mod image;
mod output;

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, State, rejection::JsonRejection},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use chat::{ChatCodec, ChatRequest, Prepared};
use orin_engine::model::Model;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};

type Result<T> = std::result::Result<T, String>;
struct Settings {
    manifest: PathBuf,
    tokenizer: PathBuf,
    model: String,
    listen: String,
    gpu_lock: PathBuf,
}
impl Settings {
    fn parse(args: &[String]) -> Result<Self> {
        if args.is_empty() || !std::path::Path::new(&args[0]).is_dir() {
            return Err("Usage: orin-llm serve MODEL_DIR [--listen 127.0.0.1:8088] [--model qwen3.8-27b] [--gpu-lock artifacts/gpu-experiment.lock]; MODEL_DIR must contain the prepared cache and checkpoint tokenizer".into());
        }
        let mut settings = Self {
            manifest: (&args[0]).into(),
            tokenizer: (&args[0]).into(),
            model: "qwen3.8-27b".into(),
            listen: "127.0.0.1:8088".into(),
            gpu_lock: "artifacts/gpu-experiment.lock".into(),
        };
        for pair in args[1..].chunks(2) {
            if pair.len() != 2 {
                return Err("Server option needs a value".into());
            }
            match pair[0].as_str() {
                "--listen" => settings.listen = pair[1].clone(),
                "--model" => settings.model = pair[1].clone(),
                "--gpu-lock" => settings.gpu_lock = (&pair[1]).into(),
                _ => return Err(format!("Unknown server option {}", pair[0])),
            }
        }
        if settings.model.is_empty() {
            return Err("Model ID cannot be empty".into());
        }
        Ok(settings)
    }
}

#[cfg(test)]
mod settings_tests {
    use super::*;

    #[test]
    fn directory_is_both_model_and_tokenizer_and_options_are_unambiguous() {
        let directory = std::env::temp_dir().to_string_lossy().into_owned();
        let settings = Settings::parse(&[
            directory.clone(),
            "--listen".into(),
            "127.0.0.1:9999".into(),
        ])
        .unwrap();
        assert_eq!(settings.manifest, PathBuf::from(&directory));
        assert_eq!(settings.tokenizer, PathBuf::from(&directory));
        assert_eq!(settings.listen, "127.0.0.1:9999");
        assert!(Settings::parse(&[directory.clone(), directory.clone()]).is_err());
        assert!(Settings::parse(&[directory, "--listen".into()]).is_err());
        assert!(Settings::parse(&["legacy-model.json".into(), "tokenizer".into()]).is_err());
    }
}
#[derive(Clone)]
struct Service {
    jobs: mpsc::Sender<Job>,
    codec: Arc<ChatCodec>,
    model: Arc<str>,
    context: usize,
    vision: Option<orin_engine::vision::VisionSpec>,
    api_key: Option<Arc<str>>,
    ids: Arc<AtomicU64>,
}
struct Job {
    prepared: Prepared,
    events: mpsc::Sender<ModelEvent>,
    id: String,
    created: u64,
    queued: Instant,
}
enum ModelEvent {
    Chunk(Value),
    Complete(Value),
    Failed(String),
}

pub fn run(args: &[String]) -> Result<()> {
    let settings = Settings::parse(args)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(serve(settings))
}
async fn serve(settings: Settings) -> Result<()> {
    let codec = Arc::new(ChatCodec::load(&settings.tokenizer)?);
    let (sender, receiver) = mpsc::channel(128);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let model_id: Arc<str> = settings.model.into();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_codec = Arc::clone(&codec);
    let worker_model = Arc::clone(&model_id);
    let worker_shutdown = Arc::clone(&shutdown);
    let worker = std::thread::Builder::new()
        .name("orin-gpu".into())
        .spawn(move || {
            let initialize = || -> Result<_> {
                use std::os::fd::AsRawFd;
                if let Some(parent) = settings
                    .gpu_lock
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                {
                    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                let lock = std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(false)
                    .read(true)
                    .write(true)
                    .open(&settings.gpu_lock)
                    .map_err(|e| e.to_string())?;
                // SAFETY: flock borrows this live file descriptor. The file remains
                // owned by this worker for the complete CUDA model lifetime.
                if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                    return Err("GPU experiment lock is busy".into());
                }
                let model = Model::load(&settings.manifest)?;
                if worker_codec.tokenizer.get_vocab_size(true) > model.vocab() {
                    return Err("Tokenizer exceeds model vocabulary".into());
                }
                let context = model.max_context();
                let vision = model.vision().cloned();
                Ok((lock, model, context, vision))
            };
            match initialize() {
                Ok((_lock, mut model, context, vision)) => {
                    if ready_sender.send(Ok((context, vision))).is_ok() {
                        worker_loop(
                            &mut model,
                            receiver,
                            &worker_codec,
                            &worker_model,
                            &worker_shutdown,
                        );
                    }
                }
                Err(error) => {
                    let _ = ready_sender.send(Err(error));
                }
            }
        })
        .map_err(|e| e.to_string())?;
    let (context, vision) = ready_receiver.await.map_err(|e| e.to_string())??;
    let state = Service {
        jobs: sender,
        codec,
        model: model_id,
        context,
        vision,
        api_key: std::env::var("ORIN_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(Arc::from),
        ids: Arc::new(AtomicU64::new(0)),
    };
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completions))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&settings.listen)
        .await
        .map_err(|e| e.to_string())?;
    eprintln!(
        "API READY http://{}/v1; context {context}; one GPU worker, queue 128",
        settings.listen
    );
    let signal_shutdown = Arc::clone(&shutdown);
    let result = axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            signal_shutdown.store(true, Ordering::Relaxed);
        })
        .await
        .map_err(|e| e.to_string());
    shutdown.store(true, Ordering::Relaxed);
    worker
        .join()
        .map_err(|_| "GPU worker panicked".to_owned())?;
    result
}
fn authorized(state: &Service, headers: &HeaderMap) -> bool {
    state.api_key.as_ref().is_none_or(|key| {
        headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            == Some(key.as_ref())
    })
}
fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::INTERNAL_SERVER_ERROR => "server_error",
        _ => "invalid_request_error",
    };
    (status, Json(json!({"error":{"message":message.into(),"type":kind,"param":null,"code":status.as_u16().to_string()}}))).into_response()
}
async fn health(State(state): State<Service>) -> Json<Value> {
    Json(json!({"status":"ready","model":state.model.as_ref(),"max_context":state.context}))
}
async fn models(State(state): State<Service>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    Json(json!({"object":"list","data":[{"id":state.model.as_ref(),"object":"model","created":0,"owned_by":"orin-llm"}]})).into_response()
}
async fn completions(
    State(state): State<Service>,
    headers: HeaderMap,
    body: std::result::Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    let request = match body {
        Ok(Json(v)) => v,
        Err(e) => return error(e.status(), e.body_text()),
    };
    let codec = Arc::clone(&state.codec);
    let model = Arc::clone(&state.model);
    let context = state.context;
    let vision = state.vision.clone();
    let prepared = match tokio::task::spawn_blocking(move || {
        codec.prepare(request, &model, context, vision.as_ref())
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let stream = prepared.stream;
    let (events, mut responses) = mpsc::channel(64);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let serial = state.ids.fetch_add(1, Ordering::Relaxed);
    let job = Job {
        prepared,
        events,
        id: format!("chatcmpl-{created}-{}-{serial}", std::process::id()),
        created,
        queued: Instant::now(),
    };
    if let Err(e) = state.jobs.try_send(job) {
        return match e {
            mpsc::error::TrySendError::Full(_) => {
                error(StatusCode::TOO_MANY_REQUESTS, "Request queue is full")
            }
            _ => error(StatusCode::INTERNAL_SERVER_ERROR, "GPU worker unavailable"),
        };
    }
    if stream {
        let events = futures_util::stream::unfold(
            (responses, false),
            |(mut receiver, done)| async move {
                if done {
                    return None;
                }
                match receiver.recv().await {
                Some(ModelEvent::Chunk(chunk)) => Some((Ok::<_, Infallible>(Event::default().data(chunk.to_string())), (receiver, false))),
                Some(ModelEvent::Complete(_)) => Some((Ok(Event::default().data("[DONE]")), (receiver, true))),
                Some(ModelEvent::Failed(message)) => Some((Ok(Event::default().data(json!({"error":{"message":message,"type":"server_error","code":"generation_error"}}).to_string())), (receiver, false))),
                None => Some((Ok(Event::default().data("[DONE]")), (receiver, true))),
            }
            },
        );
        Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        while let Some(event) = responses.recv().await {
            match event {
                ModelEvent::Complete(response) => return Json(response).into_response(),
                ModelEvent::Failed(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
                ModelEvent::Chunk(_) => {}
            }
        }
        error(StatusCode::INTERNAL_SERVER_ERROR, "GPU worker disconnected")
    }
}
fn chunk(job: &Job, model: &str, delta: Value, finish: Option<&str>) -> Value {
    json!({"id":job.id,"object":"chat.completion.chunk","created":job.created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish,"logprobs":null}]})
}
fn send_delta(job: &Job, model: &str, delta: Value) -> bool {
    job.events
        .blocking_send(ModelEvent::Chunk(chunk(job, model, delta, None)))
        .is_ok()
}
fn worker_loop(
    model: &mut Model,
    mut jobs: mpsc::Receiver<Job>,
    codec: &ChatCodec,
    model_id: &str,
    shutdown: &AtomicBool,
) {
    while let Some(job) = jobs.blocking_recv() {
        if job.events.is_closed() {
            continue;
        }
        let result = generate_job(model, codec, model_id, &job, shutdown);
        if let Err(error) = result {
            eprintln!("{}: {error}", job.id);
            let _ = job.events.blocking_send(ModelEvent::Failed(error));
        }
    }
}
fn generate_job(
    model: &mut Model,
    codec: &ChatCodec,
    model_id: &str,
    job: &Job,
    shutdown: &AtomicBool,
) -> Result<()> {
    if shutdown.load(Ordering::Relaxed) {
        return Err("Server shutting down".into());
    }
    let start = Instant::now();
    let queue_s = job.queued.elapsed().as_secs_f64();
    let mut parser = output::Output::new(
        job.prepared.thinking,
        job.prepared.tools.clone(),
        job.prepared.stops.clone(),
        &job.id,
    );
    let mut decoder = codec.tokenizer.decode_stream(false);
    let mut parse_error = None;
    let mut eos = false;
    let mut cancelled = false;
    if !send_delta(job, model_id, json!({"role":"assistant","content":""})) {
        return Ok(());
    }
    let count = model.generate_visual(
        &job.prepared.input,
        &job.prepared.images,
        job.prepared.max_tokens,
        &job.prepared.sampling,
        || job.events.is_closed() || shutdown.load(Ordering::Relaxed),
        |id| {
            if job.events.is_closed() || shutdown.load(Ordering::Relaxed) {
                cancelled = true;
                return false;
            }
            if codec.eos.contains(&id) {
                eos = true;
                return false;
            }
            let parsed = decoder
                .step(id)
                .map_err(|e| e.to_string())
                .and_then(|text| parser.push(text.as_deref().unwrap_or(""), false));
            match parsed {
                Ok(deltas) => {
                    for delta in deltas {
                        if !send_delta(job, model_id, delta) {
                            cancelled = true;
                            return false;
                        }
                    }
                }
                Err(e) => {
                    parse_error = Some(e);
                    return false;
                }
            }
            !parser.stopped
        },
    )?;
    if cancelled {
        return Ok(());
    }
    if let Some(e) = parse_error {
        return Err(e);
    }
    let exhausted = !eos && !parser.stopped && count == job.prepared.max_tokens;
    for delta in parser.finish(exhausted, &job.prepared.tool_choice, job.prepared.parallel)? {
        if !send_delta(job, model_id, delta) {
            return Ok(());
        }
    }
    let finish = if exhausted {
        "length"
    } else if !parser.calls.is_empty() {
        "tool_calls"
    } else {
        "stop"
    };
    let usage = json!({"prompt_tokens":job.prepared.input.len(),"completion_tokens":count,"total_tokens":job.prepared.input.len()+count});
    let response = json!({"id":job.id,"object":"chat.completion","created":job.created,"model":model_id,
        "choices":[{"index":0,"message":parser.message(),"finish_reason":finish,"logprobs":null}],"usage":usage});
    let _ = job.events.blocking_send(ModelEvent::Chunk(chunk(
        job,
        model_id,
        json!({}),
        Some(finish),
    )));
    if job.prepared.include_usage {
        let _ = job.events.blocking_send(ModelEvent::Chunk(json!({"id":job.id,"object":"chat.completion.chunk","created":job.created,"model":model_id,"choices":[],"usage":usage})));
    }
    let _ = job.events.blocking_send(ModelEvent::Complete(response));
    eprintln!(
        "{}: {} prompt, {count} completion tokens, queue {queue_s:.3}s, generation {:.3}s, finish {finish}",
        job.id,
        job.prepared.input.len(),
        start.elapsed().as_secs_f64()
    );
    if let Some(stats) = model.speculation_statistics() {
        eprintln!(
            "{}: MTP {}",
            job.id,
            serde_json::to_string(stats).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}
