//! Shared bounded HTTP ingress and an owner-thread GPU lifecycle for encoders.
use crate::server::{api_key, bearer_authorized, lifecycle::Lifecycle, listener, shutdown_signal};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Component, Path},
    time::Instant,
};
use tokenizers::Tokenizer;
type Result<T> = std::result::Result<T, String>;
pub(crate) struct Frontend {
    pub tokenizer: Tokenizer,
    pub hashes: BTreeMap<String, String>,
    pub context: usize,
    pub metadata: Value,
}
pub(crate) fn load_frontend_descriptor(root: &Path) -> Result<Frontend> {
    let bytes = std::fs::read(root.join("cache/model.json")).map_err(|e| e.to_string())?;
    let mut descriptor: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let hashes: BTreeMap<String, String> =
        serde_json::from_value(descriptor["frontend_assets"].take()).map_err(|e| e.to_string())?;
    for name in hashes.keys() {
        let mut components = Path::new(name).components();
        if components.next() != Some(Component::Normal(name.as_ref()))
            || components.next().is_some()
        {
            return Err("Invalid encoder frontend filename".into());
        }
    }
    let metadata = descriptor["metadata"].take();
    let context = metadata["max_context"]
        .as_u64()
        .ok_or("Missing encoder context")? as usize;
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).map_err(|e| e.to_string())?;
    Ok(Frontend {
        tokenizer,
        hashes,
        context,
        metadata,
    })
}

pub(crate) fn unbounded_tokenizer(mut tokenizer: Tokenizer) -> Result<Tokenizer> {
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

pub(crate) trait Backend {
    fn respond(&mut self, request: Value) -> Result<Value>;
}
#[derive(Clone)]
struct Service {
    sender: tokio::sync::mpsc::Sender<Job>,
    slots: std::sync::Arc<tokio::sync::Semaphore>,
    model: std::sync::Arc<str>,
    key: Option<std::sync::Arc<str>>,
    timeout_ms: u64,
    lifecycle: std::sync::Arc<Lifecycle>,
}
struct Job {
    _slot: tokio::sync::OwnedSemaphorePermit,
    request: Value,
    result: tokio::sync::oneshot::Sender<Result<Value>>,
    deadline: Instant,
}

pub(crate) fn serve<B, F>(
    settings: crate::ServerConfig,
    routes: Vec<&'static str>,
    load: F,
) -> Result<()>
where
    B: Backend + 'static,
    F: FnOnce(&crate::ServerConfig) -> Result<B> + Send + 'static,
{
    if !settings.default_request_params.is_empty() {
        return Err(
            "Encoder services accept request fields directly; Chat defaults are unsupported".into(),
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(serve_async(settings, routes, load))
}
async fn serve_async<B, F>(
    settings: crate::ServerConfig,
    routes: Vec<&'static str>,
    load: F,
) -> Result<()>
where
    B: Backend + 'static,
    F: FnOnce(&crate::ServerConfig) -> Result<B> + Send + 'static,
{
    use axum::{
        Router,
        extract::DefaultBodyLimit,
        routing::{get, post},
    };
    let (listener, _socket_path) = listener::bind(&settings.listen)
        .await
        .map_err(|error| format!("Listener bind failed: {error}"))?;
    let (sender, mut receiver) = tokio::sync::mpsc::channel::<Job>(4);
    let (ready, initialized) = tokio::sync::oneshot::channel();
    let state = Service {
        sender,
        slots: std::sync::Arc::new(tokio::sync::Semaphore::new(8)),
        model: settings.model.clone().into(),
        key: api_key(),
        timeout_ms: settings.limits.output_ms,
        lifecycle: Default::default(),
    };
    let listen = settings.listen.clone();
    let lifecycle = state.lifecycle.clone();
    let worker = std::thread::Builder::new()
        .name("orinfer-encoder-gpu".into())
        .spawn(move || {
            lifecycle.supervise(|| {
                let initialize = || -> Result<_> {
                    let lock = crate::manager::locked_file(&settings.gpu_lock)?;
                    let backend = load(&settings)?;
                    struct Loaded<B> {
                        backend: B,
                        _lock: std::fs::File,
                    }
                    Ok(Loaded {
                        backend,
                        _lock: lock,
                    })
                };
                match initialize() {
                    Err(error) => {
                        let _ = ready.send(Err(error));
                    }
                    Ok(mut loaded) => {
                        lifecycle.ready();
                        if ready.send(Ok(())).is_err() {
                            return;
                        }
                        while let Some(job) = receiver.blocking_recv() {
                            if job.result.is_closed() {
                                continue;
                            }
                            let result = if Instant::now() >= job.deadline {
                                Err("Encoder request expired in queue".into())
                            } else {
                                loaded.backend.respond(job.request)
                            };
                            let _ = job.result.send(result);
                        }
                    }
                }
            })
        })
        .map_err(|e| e.to_string())?;
    let initialized = initialized
        .await
        .map_err(|_| "Encoder GPU worker disconnected during startup".to_owned())
        .and_then(|r| r);
    if let Err(error) = initialized {
        drop(state);
        let _ = worker.join();
        return Err(error);
    }
    let mut router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models));
    for route in routes {
        router = router.route(route, post(inference_request));
    }
    let lifecycle = state.lifecycle.clone();
    let router = router
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(state);
    eprintln!("API READY encoders at {listen}; one complete independent sequence at a time");
    let result = listener
        .serve(router, async move {
            shutdown_signal().await;
            lifecycle.drain();
        })
        .await
        .map_err(|e| e.to_string());
    tokio::task::spawn_blocking(move || worker.join())
        .await
        .map_err(|e| e.to_string())?
        .and_then(|outcome| outcome)
        .map_err(|_| "Encoder GPU worker panicked".to_owned())?;
    result
}

async fn health(
    axum::extract::State(state): axum::extract::State<Service>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let status = if state.lifecycle.is_ready() {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        axum::Json(json!({"status":state.lifecycle.name(),"model":state.model.as_ref()})),
    )
        .into_response()
}
fn failure(status: axum::http::StatusCode, message: impl Into<String>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        status,
        axum::Json(json!({"error":{"message":message.into(),"type":"invalid_request_error"}})),
    )
        .into_response()
}
async fn models(
    axum::extract::State(state): axum::extract::State<Service>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !bearer_authorized(state.key.as_deref(), &headers) {
        return failure(axum::http::StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    axum::Json(json!({"object":"list","data":[{"id":state.model.as_ref(),"object":"model","owned_by":"orinfer"}]})).into_response()
}
async fn inference_request(
    axum::extract::State(state): axum::extract::State<Service>,
    headers: axum::http::HeaderMap,
    body: axum::body::Body,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse};
    if !bearer_authorized(state.key.as_deref(), &headers) {
        return failure(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    let Ok(slot) = std::sync::Arc::clone(&state.slots).try_acquire_owned() else {
        return failure(StatusCode::TOO_MANY_REQUESTS, "Encoder ingress is full");
    };
    let bytes = match axum::body::to_bytes(body, 32 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => return failure(StatusCode::PAYLOAD_TOO_LARGE, e.to_string()),
    };
    let request = match serde_json::from_slice::<Value>(&bytes) {
        Ok(value) => value,
        Err(e) => return failure(StatusCode::BAD_REQUEST, e.to_string()),
    };
    drop(bytes);
    let (result, receiver) = tokio::sync::oneshot::channel();
    let timeout = std::time::Duration::from_millis(state.timeout_ms);
    if state
        .sender
        .try_send(Job {
            _slot: slot,
            request,
            result,
            deadline: Instant::now() + timeout,
        })
        .is_err()
    {
        return failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Encoder request queue is full",
        );
    }
    match tokio::time::timeout(timeout, receiver).await {
        Ok(Ok(Ok(value))) => axum::Json(value).into_response(),
        Ok(Ok(Err(error))) => failure(StatusCode::BAD_REQUEST, error),
        Ok(Err(_)) => failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "Encoder GPU worker disconnected",
        ),
        Err(_) => failure(StatusCode::GATEWAY_TIMEOUT, "Encoder request timed out"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn encoder_health_matches_the_model_manager_readiness_contract() {
        let (sender, _receiver) = tokio::sync::mpsc::channel(1);
        let state = Service {
            sender,
            slots: std::sync::Arc::new(tokio::sync::Semaphore::new(1)),
            model: "test-rerank".into(),
            key: Some("test-key".into()),
            timeout_ms: 10000,
            lifecycle: Default::default(),
        };
        state.lifecycle.ready();
        let response = health(axum::extract::State(state)).await;
        assert_eq!(response.status(), axum::http::StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value, json!({"status":"ready","model":"test-rerank"}));
    }
    #[tokio::test]
    async fn cancelled_encoder_handler_retains_its_ingress_permit_until_job_drops() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let state = Service {
            sender,
            slots: std::sync::Arc::clone(&slots),
            model: "embed".into(),
            key: None,
            timeout_ms: 10000,
            lifecycle: Default::default(),
        };
        let handler = tokio::spawn(inference_request(
            axum::extract::State(state),
            axum::http::HeaderMap::new(),
            axum::body::Body::from(r#"{"model":"embed","input":"hello"}"#),
        ));
        let job = receiver.recv().await.unwrap();
        assert_eq!(job.request["input"], "hello");
        assert_eq!(slots.available_permits(), 0);
        handler.abort();
        let _ = handler.await;
        assert!(job.result.is_closed());
        assert_eq!(slots.available_permits(), 0);
        drop(job);
        assert_eq!(slots.available_permits(), 1);
    }
}
