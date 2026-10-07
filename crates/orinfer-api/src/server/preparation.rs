//! Budgets follow blocking work and queued image tensors, not HTTP handler lifetime.
use super::*;

pub(super) struct Pool {
    pub workers: Arc<tokio::sync::Semaphore>,
    pub memory: Arc<tokio::sync::Semaphore>,
    bodies: Arc<tokio::sync::Semaphore>,
}
impl Pool {
    pub fn new(limits: Limits) -> Self {
        Self {
            bodies: Arc::new(tokio::sync::Semaphore::new(16)),
            workers: Arc::new(tokio::sync::Semaphore::new(limits.preprocess_workers)),
            memory: Arc::new(tokio::sync::Semaphore::new(limits.memory_mib)),
        }
    }
}
#[derive(Default)]
pub(super) struct Cancellation(Arc<AtomicBool>);
impl Cancellation {
    pub fn flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.0)
    }
}
impl Drop for Cancellation {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
pub(super) struct Context {
    cancelled: Arc<AtomicBool>,
    budget: usize,
    retained: usize,
}
impl Context {
    pub fn new(cancelled: Arc<AtomicBool>, budget: usize) -> Self {
        Self {
            cancelled,
            budget,
            retained: 0,
        }
    }
    #[cfg(test)]
    pub fn unbounded() -> Self {
        Self::new(Arc::new(AtomicBool::new(false)), usize::MAX)
    }
    pub fn checkpoint(&self, temporary: usize) -> Result<()> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err("Request cancelled during preprocessing".into());
        }
        if self
            .retained
            .checked_add(temporary)
            .is_none_or(|n| n > self.budget)
        {
            return Err("Request preprocessing memory budget exceeded".into());
        }
        Ok(())
    }
    pub fn retain(&mut self, bytes: usize) -> Result<()> {
        self.checkpoint(bytes)?;
        self.retained += bytes;
        Ok(())
    }
}
pub(super) fn request_memory_mib(request: &ChatRequest, context: usize) -> u32 {
    let bytes = serde_json::to_vec(&request.messages)
        .map(|v| v.len())
        .unwrap_or(32 << 20);
    let image = request.messages.iter().any(|m| {
        m["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|p| p["type"] == "image_url"))
    });
    // JSON, rendered text, tokenizer intermediates and final IDs; image decoder
    // additionally checks actual dimensions before allocation under this ceiling.
    let text = bytes
        .saturating_mul(8)
        .saturating_add(context.saturating_mul(16))
        .saturating_add(4 << 20);
    // Reserve nested probability JSON and serialization copies through delivery.
    let probabilities = if request.logprobs == Some(true) {
        let output = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(8192)
            .min(context);
        // SSE emits scores incrementally and retains only bounded mailboxes.
        (if request.stream {
            output.min(128)
        } else {
            output
        })
        .saturating_mul(request.top_logprobs.unwrap_or(0).min(20) + 1)
        .saturating_mul(3072)
        .saturating_add(64 << 20)
    } else {
        0
    };
    (text
        .saturating_add(probabilities)
        .saturating_add(if image { 512 << 20 } else { 0 })
        .div_ceil(1 << 20)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn probability_budget_counts_completion_limit_and_streaming_retention() {
        let mut request: ChatRequest = serde_json::from_value(json!({"model":"test",
            "messages":[{"role":"user","content":"hello"}],"max_completion_tokens":512,
            "logprobs":true,"top_logprobs":20}))
        .unwrap();
        let complete = request_memory_mib(&request, 1024);
        request.stream = true;
        let streaming = request_memory_mib(&request, 1024);
        assert!(complete > streaming);
        request.logprobs = Some(false);
        assert!(streaming > request_memory_mib(&request, 1024));
        request.logprobs = Some(true);
        request.stream = false;
        request.max_completion_tokens = Some(usize::MAX);
        let capped = request_memory_mib(&request, 1024);
        request.max_completion_tokens = Some(1024);
        assert_eq!(capped, request_memory_mib(&request, 1024));
    }
    #[tokio::test]
    async fn cancelled_handler_cannot_return_permit_while_blocking_work_runs() {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let owned = Arc::clone(&permits).acquire_owned().await.unwrap();
        let (started, ready) = oneshot::channel();
        let (release, gate) = std::sync::mpsc::channel();
        let handler = tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                let _owned = owned;
                started.send(()).unwrap();
                gate.recv().unwrap();
            })
            .await
            .unwrap();
        });
        ready.await.unwrap();
        handler.abort();
        let _ = handler.await;
        assert_eq!(permits.available_permits(), 0);
        release.send(()).unwrap();
        let _permit = tokio::time::timeout(std::time::Duration::from_secs(2), permits.acquire())
            .await
            .unwrap()
            .unwrap();
    }
    #[test]
    fn cancellation_and_aggregate_memory_are_checked_between_stages() {
        let flag = Arc::new(AtomicBool::new(false));
        let mut context = Context::new(Arc::clone(&flag), 100);
        context.retain(60).unwrap();
        assert!(context.checkpoint(41).is_err());
        flag.store(true, Ordering::Release);
        assert!(context.checkpoint(0).is_err());
        assert!(Context::unbounded().checkpoint(usize::MAX).is_ok());
    }
}

#[derive(Clone)]
pub(super) struct Ingress {
    pub slot: Arc<tokio::sync::OwnedSemaphorePermit>,
    pub body: Arc<tokio::sync::OwnedSemaphorePermit>,
}
pub(super) async fn ingress(
    State(state): State<Service>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if !authorized(&state, request.headers()) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    if !state.activity.lifecycle.is_ready() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not ready");
    }
    let slot = match Arc::clone(&state.slots).try_acquire_owned() {
        Ok(v) => v,
        Err(_) => return error(StatusCode::TOO_MANY_REQUESTS, "Request queue is full"),
    };
    let body = match Arc::clone(&state.preparation.bodies).acquire_owned().await {
        Ok(v) => v,
        Err(_) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Body admission unavailable",
            );
        }
    };
    request.extensions_mut().insert(Ingress {
        slot: Arc::new(slot),
        body: Arc::new(body),
    });
    next.run(request).await
}
