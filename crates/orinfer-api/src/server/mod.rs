pub(crate) mod audio;
pub(crate) mod batch;
mod chat;
mod continuous;
pub(crate) mod defaults;
mod grammar;
mod image;
mod lifecycle;
mod metrics;
mod output;
mod preparation;
pub(crate) mod startup;
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
use futures_util::{FutureExt, StreamExt};
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
    prompt_lookup: bool,
    verify_weights: bool,
    resident_experts: orinfer_engine::execution::ResidentExperts,
    vision: Option<orinfer_engine::vision::VisionSpec>,
    api_key: Option<Arc<str>>,
    ids: Arc<AtomicU64>,
    scheduler: orinfer_engine::scheduler::Options,
    preparation: Arc<preparation::Pool>,
    limits: Limits,
    default_request_params: Arc<serde_json::Map<String, Value>>,
    activity: Arc<continuous::Activity>,
}
struct Job {
    _slot: Arc<tokio::sync::OwnedSemaphorePermit>,
    _memory: Option<Arc<preparation::Lease>>,
    limits: Limits,
    prepared: Prepared,
    events: mpsc::Sender<ModelEvent>,
    id: String,
    created: u64,
    queued: Instant,
    hint: Option<orinfer_engine::scheduler::RequestHint>,
    /// Offline lookahead: Some retains only known shared prompt branches.
    shared_prefix: Option<usize>,
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
    let descriptor: Value = serde_json::from_slice(
        &std::fs::read(settings.model_dir.join("cache/model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if !descriptor["metadata"]["embedding"].is_null() {
        return crate::embeddings::serve(settings);
    }
    if !descriptor["metadata"]["reranking"].is_null() {
        return crate::reranking::serve(settings);
    }
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
    let started = Instant::now();
    let worker = startup::start(&settings).await?;
    let state = worker.state.clone();
    let shutdown = Arc::clone(&worker.shutdown);
    let lifecycle = Arc::clone(&state.activity.lifecycle);
    let context = state.context;
    let scheduler = state.scheduler;
    let limits = state.limits;
    let router = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics::endpoint))
        .route("/v1/models", get(models))
        .route(
            "/v1/chat/completions",
            post(completions).layer(axum::middleware::from_fn_with_state(
                state.clone(),
                preparation::ingress,
            )),
        )
        .route(
            "/v1/audio/transcriptions",
            post(audio::transcriptions).layer(axum::middleware::from_fn_with_state(
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
    let (tcp, unix) = if let Some(path) = settings.listen.strip_prefix("unix:") {
        (
            None,
            Some(tokio::net::UnixListener::bind(path).map_err(|e| e.to_string())?),
        )
    } else {
        (
            Some(
                tokio::net::TcpListener::bind(&settings.listen)
                    .await
                    .map_err(|e| e.to_string())?,
            ),
            None,
        )
    };
    eprintln!(
        "API READY http://{}/v1; context {context}; continuous_batching=true, max_active={}, queue 128; startup={:.3}s",
        settings.listen,
        scheduler.max_active,
        started.elapsed().as_secs_f64()
    );
    let shutdown_at = Arc::new(std::sync::Mutex::new(None));
    let signal_shutdown_at = Arc::clone(&shutdown_at);
    let signal_shutdown = Arc::clone(&shutdown);
    let draining = Arc::new(tokio::sync::Notify::new());
    let signal_draining = Arc::clone(&draining);
    let signal_lifecycle = Arc::clone(&lifecycle);
    let result = {
        let shutdown = async move {
            shutdown_signal().await;
            if let Ok(mut started) = signal_shutdown_at.lock() {
                *started = Some(Instant::now());
            }
            signal_lifecycle.drain();
            signal_shutdown.store(true, Ordering::Relaxed);
            signal_draining.notify_one();
        };
        let server = if let Some(listener) = tcp {
            axum::serve(listener, router)
                .with_graceful_shutdown(shutdown)
                .into_future()
                .boxed()
        } else {
            axum::serve(unix.unwrap(), router)
                .with_graceful_shutdown(shutdown)
                .into_future()
                .boxed()
        };
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
    worker.stop(remaining_ms).await?;
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
pub(crate) fn error(status: StatusCode, message: impl Into<String>) -> Response {
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
        "verify_weights":state.verify_weights,
        "resident_experts":state.resident_experts,
        "default_request_params":state.default_request_params.as_ref(),
        "chat_capabilities":{"streaming":true,"default_stream":state.default_request_params.get("stream").and_then(Value::as_bool).unwrap_or(false),"max_choices":1,
            "response_formats":["text","json_object","json_schema"],"constraint_backend":"llguidance",
            "tools":{"function":state.codec.tools_supported,"strict":state.codec.tools_supported,"required":state.codec.tools_supported,"incremental_arguments":state.codec.tools_supported},
            "logprobs":true,"max_top_logprobs":20,"logit_bias":true,"stored_completions":false,
            "images":state.vision.is_some(),"video":false,"audio":state.codec.audio.is_some(),"default_max_completion_tokens":state.default_request_params.get("max_completion_tokens").and_then(Value::as_u64).or_else(|| state.default_request_params.get("max_tokens").and_then(Value::as_u64)).unwrap_or(8192),
            "thinking_token_budget":true,"prompt_cache":{"key":true,"retention":["in-memory","24h"],"persistent":false,"ttl_guaranteed":false}},
        "mtp":{"enabled":state.mtp_drafts>0,"max_drafts":state.mtp_drafts},
        "prompt_lookup":{"enabled":state.prompt_lookup && state.mtp_drafts>0},
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
    body: std::result::Result<Json<Value>, JsonRejection>,
) -> Response {
    let request = match body {
        Ok(Json(v)) => match defaults::decode(v, &state.default_request_params) {
            Ok(request) => request,
            Err(e) => return error(StatusCode::BAD_REQUEST, e),
        },
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
    let estimate = preparation::request_memory(
        &request,
        state.context,
        state.vision.as_ref(),
        state.limits.memory_mib,
    );
    let memory_mib = estimate.mib;
    if memory_mib as usize > state.limits.memory_mib {
        return error(
            StatusCode::BAD_REQUEST,
            "Request exceeds preprocessing memory budget",
        );
    }
    let mut memory = match Arc::clone(&state.preparation.memory)
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
        let mut preparation = estimate.context(flag);
        preparation.checkpoint(0)?;
        let prepared =
            codec.prepare(request, &model, context, vision.as_ref(), &mut preparation)?;
        preparation.checkpoint(0)?;
        if !prepared.audios.is_empty() || !prepared.images.is_empty() {
            preparation.release_media_scratch(&mut memory);
        }
        Ok::<_, String>((prepared, slot, preparation.lease(memory)))
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
        shared_prefix: None,
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
fn retain_output_memory(response: Response, memory: Arc<preparation::Lease>) -> Response {
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

pub(crate) async fn shutdown_signal() {
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
            let context = preparation::Context::new(Arc::new(AtomicBool::new(false)), 1 << 20);
            let memory = Arc::new(context.lease(Arc::clone(&permits).try_acquire_owned().unwrap()));
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
