mod chat;
mod continuous;
mod grammar;
mod image;
mod lifecycle;
mod output;
mod preparation;
mod thinking;

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
use futures_util::StreamExt;
use orinfer_engine::model::Model;
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    future::IntoFuture,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{mpsc, oneshot};

type Result<T> = std::result::Result<T, String>;
use crate::{PreprocessingLimits as Limits, ServerConfig as Settings};
use orinfer_engine::execution::LoadOptions;
#[derive(Clone)]
struct Service {
    jobs: mpsc::Sender<Job>,
    slots: Arc<tokio::sync::Semaphore>,
    codec: Arc<ChatCodec>,
    model: Arc<str>,
    context: usize,
    mtp_drafts: usize,
    vision: Option<orinfer_engine::vision::VisionSpec>,
    api_key: Option<Arc<str>>,
    ids: Arc<AtomicU64>,
    scheduler: orinfer_engine::scheduler::Options,
    preparation: Arc<preparation::Pool>,
    limits: Limits,
    activity: Arc<continuous::Activity>,
}
struct Job {
    _slot: Arc<tokio::sync::OwnedSemaphorePermit>,
    _memory: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    limits: Limits,
    prepared: Prepared,
    events: mpsc::Sender<ModelEvent>,
    id: String,
    created: u64,
    queued: Instant,
    hint: Option<orinfer_engine::scheduler::RequestHint>,
}
enum ModelEvent {
    Chunk(Value),
    Complete(Value),
    Failed(String),
    Unavailable(String),
}

fn stream_payload(event: Option<ModelEvent>) -> (String, bool) {
    match event {
        Some(ModelEvent::Chunk(value)) => (value.to_string(), false),
        Some(ModelEvent::Complete(_)) => ("[DONE]".into(), true),
        Some(ModelEvent::Failed(message)) => (json!({"error":{"message":message,"type":"server_error","code":"generation_error"}}).to_string(), true),
        Some(ModelEvent::Unavailable(message)) => (json!({"error":{"message":message,"type":"server_error","code":"service_unavailable"}}).to_string(), true),
        None => (json!({"error":{"message":"GPU worker disconnected before completion","type":"server_error","code":"worker_disconnected"}}).to_string(), true),
    }
}

pub fn run(settings: Settings) -> Result<()> {
    settings.validate()?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let result = runtime.block_on(serve(settings));
    // Dropping a runtime normally waits forever for cancelled blocking tasks.
    runtime.shutdown_timeout(std::time::Duration::from_secs(2));
    result
}
async fn serve(settings: Settings) -> Result<()> {
    let codec = Arc::new(ChatCodec::load(&settings.model_dir)?);
    let (sender, receiver) = mpsc::channel(128);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let model_id: Arc<str> = settings.model.into();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_codec = Arc::clone(&codec);
    let worker_model = Arc::clone(&model_id);
    let worker_shutdown = Arc::clone(&shutdown);
    let scheduler = settings.scheduler;
    let activity = Arc::new(continuous::Activity::default());
    let lifecycle = Arc::clone(&activity.lifecycle);
    let worker_activity = Arc::clone(&activity);
    let worker_lifecycle = Arc::clone(&lifecycle);
    let limits = settings.limits;
    let worker = std::thread::Builder::new()
        .name("orinfer-gpu".into())
        .spawn(move || {
            let _ = worker_lifecycle.supervise(|| {
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
                    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0
                    {
                        return Err("GPU experiment lock is busy".into());
                    }
                    let model = Model::load_with_options(
                        &settings.model_dir,
                        LoadOptions {
                            cuda_graph: settings.cuda_graph,
                            prefix_cache_bytes: settings.prefix_cache_bytes,
                            mtp_drafts: settings.mtp_drafts,
                        },
                    )?;
                    if &worker_codec.asset_hashes != model.frontend_assets() {
                        return Err("Frontend assets changed while loading model".into());
                    }
                    if worker_codec.tokenizer.get_vocab_size(true) > model.vocab() {
                        return Err("Tokenizer exceeds model vocabulary".into());
                    }
                    let context = model.max_context();
                    let vision = model.vision().cloned();
                    if !model.batching_supported() && scheduler.max_active != 1 { return Err("This execution package schedules one active request; set --max-active-requests 1 (additional requests queue)".into()); }
                    Ok((lock, model, context, vision))
                };
                match initialize() {
                    Ok((_lock, mut model, context, vision)) => {
                        worker_lifecycle.ready();
                        let mtp_drafts = model.mtp_drafts();
                        if ready_sender.send(Ok((context, vision, mtp_drafts))).is_ok() {
continuous::worker(
                                    &mut model,
                                    receiver,
                                    &worker_codec,
                                    &worker_model,
                                    &worker_shutdown,
                                    scheduler,
                                    &worker_activity,
                                );
                        }
                    }
                    Err(error) => {
                        worker_lifecycle
                            .fail(orinfer_engine::error::EngineError::take(error.clone(), true));
                        let _ = ready_sender.send(Err(error));
                    }
                }
            });
            if !worker_shutdown.load(Ordering::Relaxed) && worker_lifecycle.is_ready() {
                worker_lifecycle.fail(orinfer_engine::error::EngineError::take(
                    "GPU worker exited unexpectedly".into(),
                    true,
                ));
            }
        })
        .map_err(|e| e.to_string())?;
    let (context, vision, mtp_drafts) = ready_receiver.await.map_err(|e| e.to_string())??;
    let state = Service {
        jobs: sender,
        slots: Arc::new(tokio::sync::Semaphore::new(128 + scheduler.max_active)),
        codec,
        model: model_id,
        context,
        mtp_drafts,
        vision,
        api_key: std::env::var("ORINFER_API_KEY")
            .ok()
            .filter(|s| !s.is_empty())
            .map(Arc::from),
        ids: Arc::new(AtomicU64::new(0)),
        scheduler,
        activity,
        preparation: Arc::new(preparation::Pool::new(limits)),
        limits,
    };
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route(
            "/v1/chat/completions",
            post(completions).layer(axum::middleware::from_fn_with_state(
                state.clone(),
                preparation::ingress,
            )),
        )
        .fallback(|| async { error(StatusCode::NOT_FOUND, "Unknown API endpoint") })
        .method_not_allowed_fallback(|| async {
            error(StatusCode::METHOD_NOT_ALLOWED, "Unsupported HTTP method")
        })
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            request_id,
        ))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&settings.listen)
        .await
        .map_err(|e| e.to_string())?;
    eprintln!(
        "API READY http://{}/v1; context {context}; continuous_batching=true, max_active={}, queue 128",
        settings.listen, scheduler.max_active
    );
    let shutdown_at = Arc::new(std::sync::Mutex::new(None));
    let signal_shutdown_at = Arc::clone(&shutdown_at);
    let signal_shutdown = Arc::clone(&shutdown);
    let draining = Arc::new(tokio::sync::Notify::new());
    let signal_draining = Arc::clone(&draining);
    let signal_lifecycle = Arc::clone(&lifecycle);
    let result = {
        let server = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                shutdown_signal().await;
                if let Ok(mut started) = signal_shutdown_at.lock() {
                    *started = Some(Instant::now());
                }
                signal_lifecycle.drain();
                signal_shutdown.store(true, Ordering::Relaxed);
                signal_draining.notify_one();
            })
            .into_future();
        tokio::pin!(server);
        tokio::select! {
            result = &mut server => result.map_err(|e| e.to_string()),
            _ = draining.notified() => {
                match tokio::time::timeout(std::time::Duration::from_millis(limits.drain_ms), &mut server).await {
                    Ok(result) => result.map_err(|e| e.to_string()),
                    Err(_) => Err("HTTP drain deadline exceeded".into()),
                }
            }
        }
    };
    shutdown.store(true, Ordering::Relaxed);
    let remaining_ms = shutdown_at
        .lock()
        .ok()
        .and_then(|v| *v)
        .map(|at| {
            limits
                .drain_ms
                .saturating_sub(at.elapsed().as_millis().min(u64::MAX as u128) as u64)
        })
        .unwrap_or(limits.drain_ms);
    if !wait_worker(&worker, remaining_ms).await {
        return Err("GPU worker did not exit before drain deadline".into());
    }
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
async fn request_id(
    State(state): State<Service>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let serial = state.ids.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let id = format!("req-{now:x}-{:x}-{serial:x}", std::process::id());
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert("x-request-id", id.parse().unwrap());
    response
}
fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let kind = match status {
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::INTERNAL_SERVER_ERROR | StatusCode::SERVICE_UNAVAILABLE => "server_error",
        _ => "invalid_request_error",
    };
    let message = message.into();
    let candidate = message.split("target type: ").nth(1).unwrap_or(&message);
    let param = candidate
        .split_once(": ")
        .map(|(p, _)| p)
        .filter(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || "_.[]".contains(c))
        })
        .or_else(|| {
            ["unknown field `", "missing field `"]
                .iter()
                .find_map(|prefix| candidate.split_once(prefix)?.1.split_once('`').map(|p| p.0))
        });
    let code = match status {
        StatusCode::BAD_REQUEST if message.contains("exceeds the context window") => {
            "context_length_exceeded"
        }
        StatusCode::BAD_REQUEST => "invalid_request",
        StatusCode::UNAUTHORIZED => "invalid_api_key",
        StatusCode::TOO_MANY_REQUESTS => "queue_full",
        StatusCode::SERVICE_UNAVAILABLE => "service_unavailable",
        StatusCode::NOT_FOUND => "not_found",
        StatusCode::METHOD_NOT_ALLOWED => "method_not_allowed",
        _ => "server_error",
    };
    (
        status,
        Json(json!({"error":{"message":message,"type":kind,"param":param,"code":code}})),
    )
        .into_response()
}
async fn health(State(state): State<Service>) -> Response {
    (if state.activity.lifecycle.is_ready() { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }, Json(
        json!({"status":state.activity.lifecycle.name(),"failure":state.activity.lifecycle.failure(),"model":state.model.as_ref(),"max_context":state.context,
        "frontend_assets":state.codec.asset_hashes,
        "chat_capabilities":{"streaming":true,"default_stream":false,"max_choices":1,
            "response_formats":["text","json_object","json_schema"],"constraint_backend":"llguidance",
            "tools":{"function":true,"strict":true,"required":true,"incremental_arguments":true},
            "logprobs":true,"max_top_logprobs":20,"logit_bias":true,"stored_completions":false,
            "images":state.vision.is_some(),"video":false,"audio":false,"default_max_completion_tokens":8192,
            "thinking_token_budget":true,"prompt_cache":{"key":true,"retention":["in-memory","24h"],"persistent":false,"ttl_guaranteed":false}},
        "mtp":{"enabled":state.mtp_drafts>0,"max_drafts":state.mtp_drafts},
        "continuous_batching":true,"scheduler":state.scheduler,
        "scheduler_statistics":state.activity.statistics.lock().ok().map(|s| s.clone()),
        "admission_statistics":state.activity.admission.lock().ok().map(|s| s.clone()),
        "active_requests":state.activity.active.load(Ordering::Relaxed),
        "queued_requests":state.activity.queued.load(Ordering::Relaxed)}),
    )).into_response()
}
async fn models(State(state): State<Service>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    Json(json!({"object":"list","data":[{"id":state.model.as_ref(),"object":"model","created":0,"owned_by":"orinfer"}]})).into_response()
}
async fn completions(
    State(state): State<Service>,
    axum::Extension(admission): axum::Extension<preparation::Ingress>,
    body: std::result::Result<Json<ChatRequest>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(v)) => v,
        Err(e) => {
            return error(
                if e.status() == StatusCode::UNPROCESSABLE_ENTITY {
                    StatusCode::BAD_REQUEST
                } else {
                    e.status()
                },
                e.body_text(),
            );
        }
    };
    let preparation::Ingress { slot, body } = admission;
    let memory_mib = preparation::request_memory_mib(&request, state.context);
    if memory_mib as usize > state.limits.memory_mib {
        return error(
            StatusCode::BAD_REQUEST,
            "Request exceeds preprocessing memory budget",
        );
    }
    let memory = match Arc::clone(&state.preparation.memory)
        .acquire_many_owned(memory_mib)
        .await
    {
        Ok(permit) => permit,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Preprocessing memory unavailable",
            );
        }
    };
    let cpu = match Arc::clone(&state.preparation.workers).acquire_owned().await {
        Ok(permit) => permit,
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "Preprocessing unavailable"),
    };
    let codec = Arc::clone(&state.codec);
    let model = Arc::clone(&state.model);
    let context = state.context;
    let vision = state.vision.clone();
    let cancellation = preparation::Cancellation::default();
    let flag = cancellation.flag();
    let prepared = match tokio::task::spawn_blocking(move || {
        // Admission and memory belong to actual work, including after handler cancellation.
        let _cpu = cpu;
        let _body = body;
        let mut preparation = preparation::Context::new(flag, memory_mib as usize * 1024 * 1024);
        preparation.checkpoint(0)?;
        let prepared =
            codec.prepare(request, &model, context, vision.as_ref(), &mut preparation)?;
        preparation.checkpoint(0)?;
        Ok::<_, String>((prepared, slot, memory))
    })
    .await
    {
        Ok(Ok(prepared)) => prepared,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let (prepared, slot, memory) = prepared;
    let memory = Arc::new(memory);
    if !state.activity.lifecycle.is_ready() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not ready");
    }
    let stream = prepared.stream;
    let (events, mut responses) = mpsc::channel(64);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let serial = state.ids.fetch_add(1, Ordering::Relaxed);
    let job = Job {
        _slot: slot,
        _memory: Some(Arc::clone(&memory)),
        limits: state.limits,
        prepared,
        events,
        id: format!("chatcmpl-{created}-{}-{serial}", std::process::id()),
        created,
        queued: Instant::now(),
        hint: None,
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
        let events =
            futures_util::stream::unfold((responses, false), |(mut receiver, done)| async move {
                if done {
                    return None;
                }
                let (payload, terminal) = stream_payload(receiver.recv().await);
                Some((
                    Ok::<_, Infallible>(Event::default().data(payload)),
                    (receiver, terminal),
                ))
            });
        retain_output_memory(
            Sse::new(events)
                .keep_alive(KeepAlive::default())
                .into_response(),
            memory,
        )
    } else {
        while let Some(event) = responses.recv().await {
            match event {
                ModelEvent::Complete(response) => {
                    return retain_output_memory(Json(response).into_response(), memory);
                }
                ModelEvent::Failed(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
                ModelEvent::Unavailable(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
                ModelEvent::Chunk(_) => {}
            }
        }
        error(StatusCode::INTERNAL_SERVER_ERROR, "GPU worker disconnected")
    }
}
fn retain_output_memory(
    response: Response,
    memory: Arc<tokio::sync::OwnedSemaphorePermit>,
) -> Response {
    let (parts, body) = response.into_parts();
    let stream = futures_util::stream::unfold(
        (body.into_data_stream(), memory),
        |(mut body, memory)| async move { body.next().await.map(|data| (data, (body, memory))) },
    );
    Response::from_parts(parts, axum::body::Body::from_stream(stream))
}
fn chunk(job: &Job, model: &str, delta: Value, finish: Option<&str>) -> Value {
    let mut chunk = json!({"id":job.id,"object":"chat.completion.chunk","created":job.created,"model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish,"logprobs":null}]});
    if job.prepared.include_usage {
        chunk["usage"] = Value::Null;
    }
    chunk
}
async fn wait_worker(worker: &std::thread::JoinHandle<()>, timeout_ms: u64) -> bool {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    while !worker.is_finished() {
        if tokio::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
    true
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
            }
            Err(_) => {
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    #[tokio::test]
    async fn context_overflow_has_a_client_recognizable_code_and_parameter() {
        let response = error(
            StatusCode::BAD_REQUEST,
            "max_completion_tokens: Request exceeds the context window of 100 tokens: 90 prompt tokens + 20 output tokens",
        );
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["code"], "context_length_exceeded");
        assert_eq!(body["error"]["param"], "max_completion_tokens");
    }
    #[tokio::test]
    async fn output_memory_is_held_until_http_body_finishes_or_is_dropped() {
        for consume in [false, true] {
            let permits = Arc::new(tokio::sync::Semaphore::new(1));
            let memory = Arc::new(Arc::clone(&permits).try_acquire_owned().unwrap());
            let response = retain_output_memory(Json(json!({"ok":true})).into_response(), memory);
            assert_eq!(permits.available_permits(), 0);
            if consume {
                axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap();
            } else {
                drop(response);
            }
            assert_eq!(permits.available_permits(), 1);
        }
    }
    #[tokio::test]
    async fn stuck_worker_cannot_block_shutdown_indefinitely() {
        let (release, gate) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            gate.recv().unwrap();
        });
        assert!(!wait_worker(&worker, 5).await);
        release.send(()).unwrap();
        assert!(wait_worker(&worker, 1000).await);
        worker.join().unwrap();
    }
    #[tokio::test]
    async fn capacity_rejection_is_a_service_error_in_http_and_sse() {
        use axum::body::to_bytes;
        let (payload, terminal) = stream_payload(Some(ModelEvent::Unavailable("capacity".into())));
        assert!(terminal);
        assert_eq!(
            serde_json::from_str::<Value>(&payload).unwrap()["error"]["code"],
            "service_unavailable"
        );
        let response = error(StatusCode::SERVICE_UNAVAILABLE, "capacity");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&body).unwrap()["error"]["code"],
            "service_unavailable"
        );
    }
    #[test]
    fn abnormal_worker_eof_cannot_look_like_successful_partial_output() {
        let (payload, terminal) = stream_payload(None);
        assert!(terminal);
        assert!(serde_json::from_str::<Value>(&payload).unwrap()["error"].is_object());
        assert_ne!(payload, "[DONE]");
        assert!(stream_payload(Some(ModelEvent::Failed("fault".into()))).1);
        assert_eq!(
            stream_payload(Some(ModelEvent::Complete(json!({})))).0,
            "[DONE]"
        );
    }
}
