//! Budgets follow blocking work and queued image tensors, not HTTP handler lifetime.
use super::*;

pub(super) struct Pool {
    pub workers: Arc<tokio::sync::Semaphore>,
    pub memory: Arc<tokio::sync::Semaphore>,
    bodies: Arc<tokio::sync::Semaphore>,
    pub images: Arc<super::image::cache::Cache>,
    pub worker_count: usize,
    statistics: std::sync::Mutex<Statistics>,
}
impl Pool {
    pub fn new(limits: Limits) -> Self {
        let worker_count = limits.workers();
        Self {
            bodies: Arc::new(tokio::sync::Semaphore::new(16)),
            workers: Arc::new(tokio::sync::Semaphore::new(worker_count)),
            memory: Arc::new(tokio::sync::Semaphore::new(limits.memory_mib)),
            images: Arc::new(super::image::cache::Cache::new(
                limits.image_cache_mib << 20,
            )),
            worker_count,
            statistics: Default::default(),
        }
    }
    pub fn observe(&self, timing: &Timing, failed: bool) {
        let mut statistics = self.statistics.lock().unwrap_or_else(|e| e.into_inner());
        statistics.requests += 1;
        statistics.failures += usize::from(failed);
        statistics.timing.memory_wait_s += timing.memory_wait_s;
        statistics.timing.worker_wait_s += timing.worker_wait_s;
        statistics.timing.normalize_s += timing.normalize_s;
        statistics.timing.render_s += timing.render_s;
        statistics.timing.tokenize_s += timing.tokenize_s;
        statistics.timing.prepare_s += timing.prepare_s;
    }
    pub fn statistics(&self) -> serde_json::Value {
        let statistics = self.statistics.lock().unwrap_or_else(|e| e.into_inner());
        json!({"workers":self.worker_count,"requests":statistics.requests,"failures":statistics.failures,"timing":statistics.timing})
    }
}
#[derive(Clone, Default, serde::Serialize)]
pub(super) struct Timing {
    pub memory_wait_s: f64,
    pub worker_wait_s: f64,
    pub normalize_s: f64,
    pub render_s: f64,
    pub tokenize_s: f64,
    pub prepare_s: f64,
}
#[derive(Default)]
struct Statistics {
    requests: usize,
    failures: usize,
    timing: Timing,
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
    pub timing: Timing,
    pub images: Option<Arc<super::image::cache::Cache>>,
    cancelled: Arc<AtomicBool>,
    budget: usize,
    retained: usize,
    worker_scratch: usize,
    base: Option<usize>,
}
impl Context {
    pub fn new(cancelled: Arc<AtomicBool>, budget: usize) -> Self {
        Self {
            timing: Timing::default(),
            images: None,
            cancelled,
            budget,
            retained: 0,
            worker_scratch: 0,
            base: None,
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
        let retained = self
            .retained
            .checked_mul(if self.base.is_some() { 2 } else { 1 });
        let required = retained
            .and_then(|n| n.checked_add(self.base.unwrap_or(0)))
            .and_then(|n| n.checked_add(temporary.max(self.worker_scratch)));
        if required.is_none_or(|n| n > self.budget) {
            return Err(format!(
                "Request preprocessing memory budget exceeded (required {} MiB, reserved {} MiB; prepared media {} MiB)",
                required.unwrap_or(usize::MAX).div_ceil(1 << 20),
                self.budget / (1 << 20),
                self.retained.div_ceil(1 << 20)
            ));
        }
        Ok(())
    }
    pub fn retain(&mut self, bytes: usize) -> Result<()> {
        self.checkpoint(bytes.saturating_mul(if self.base.is_some() { 2 } else { 1 }))?;
        self.retained += bytes;
        Ok(())
    }
    pub fn release(&mut self, bytes: usize) {
        self.retained = self
            .retained
            .checked_sub(bytes)
            .expect("preparation accounting");
    }
    pub fn worker_scratch(&mut self, bytes: usize) -> Result<()> {
        self.worker_scratch = self.worker_scratch.max(bytes);
        self.checkpoint(0)
    }
    pub fn lease(&self, permit: tokio::sync::OwnedSemaphorePermit) -> Lease {
        Lease {
            permit: std::sync::Mutex::new(permit),
            base_mib: self.base.unwrap_or(self.budget).div_ceil(1 << 20),
        }
    }
    pub fn release_media_scratch(&self, permit: &mut tokio::sync::OwnedSemaphorePermit) {
        // Prepared media and the worker's pending copy stay charged. Decoded
        // files, PCM and frontend scratch end with blocking preparation.
        let live_mib = self
            .retained
            .saturating_mul(2)
            .saturating_add(self.worker_scratch)
            .div_ceil(1 << 20)
            .saturating_add(16);
        let base_mib = self
            .base
            .unwrap_or_else(|| self.budget.saturating_sub(512 << 20))
            .div_ceil(1 << 20);
        let release = permit
            .num_permits()
            .saturating_sub(base_mib.saturating_add(live_mib));
        drop(permit.split(release));
    }
}
/// Shared by the GPU job and HTTP delivery, with independently releasable media.
pub(super) struct Lease {
    permit: std::sync::Mutex<tokio::sync::OwnedSemaphorePermit>,
    base_mib: usize,
}
impl Lease {
    pub fn release_images(&self) {
        let mut permit = self.permit.lock().unwrap_or_else(|e| e.into_inner());
        let release = permit.num_permits().saturating_sub(self.base_mib);
        drop(permit.split(release));
    }
}

/// Reserve the shared pool for aggregate image tensors as well as decoder scratch.
/// The configured ceiling bounds each request; actual retained bytes release
/// unused reservation as soon as blocking preparation finishes.
#[derive(Clone, Copy)]
pub(super) struct Memory {
    pub mib: u32,
    base_mib: usize,
}
impl Memory {
    pub fn context(self, cancelled: Arc<AtomicBool>) -> Context {
        let mut context = Context::new(cancelled, self.mib as usize * (1 << 20));
        context.base = Some(self.base_mib * (1 << 20));
        context
    }
}
pub(super) fn request_memory(
    request: &ChatRequest,
    context: usize,
    vision: Option<&orinfer_engine::vision::VisionSpec>,
    limit_mib: usize,
    default_max_tokens: usize,
) -> Memory {
    let minimum = request_memory_mib(request, context, default_max_tokens) as usize;
    let images = request
        .messages
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|part| part["type"] == "image_url")
        .count();
    let media = request
        .messages
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .any(|part| {
            matches!(
                part["type"].as_str(),
                Some("image_url" | "input_audio" | "audio")
            )
        });
    let extra = vision.filter(|_| images > 1).map_or(0, |v| {
        let per_image = request
            .image_processing
            .max_pixels
            .map_or(v.max_patches, |n| {
                (n / v.patch_size / v.patch_size).min(v.max_patches)
            });
        let features = request
            .image_processing
            .max_tokens
            .unwrap_or(v.max_features)
            .min(v.max_features);
        let patches = images.saturating_mul(per_image).min(
            features
                .saturating_mul(v.merge_size)
                .saturating_mul(v.merge_size),
        );
        patches
            .saturating_mul(v.patch_size)
            .saturating_mul(v.patch_size)
            // Keep RGB8 across the queue; expand one image directly to the
            // encoder's 16-bit input while its 512 MiB scratch is still held.
            .saturating_mul(3 * 2)
            .div_ceil(1 << 20)
    });
    let mib = if minimum > limit_mib {
        minimum
    } else {
        minimum.saturating_add(extra).min(limit_mib)
    };
    Memory {
        mib: u32::try_from(mib).unwrap_or(u32::MAX),
        base_mib: minimum.saturating_sub(if media { 512 } else { 0 }),
    }
}

pub(super) fn request_memory_mib(
    request: &ChatRequest,
    context: usize,
    default_max_tokens: usize,
) -> u32 {
    let bytes = serde_json::to_vec(&request.messages)
        .map(|v| v.len())
        .unwrap_or(32 << 20);
    let image = request.messages.iter().any(|m| {
        m["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().any(|p| p["type"] == "image_url"))
    });
    let audio = request.messages.iter().any(|m| {
        m["content"].as_array().is_some_and(|parts| {
            parts
                .iter()
                .any(|p| matches!(p["type"].as_str(), Some("input_audio" | "audio")))
        })
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
            .unwrap_or(default_max_tokens)
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
        .saturating_add(if image || audio { 512 << 20 } else { 0 })
        .div_ceil(1 << 20)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn queued_media_keeps_live_tensors_and_releases_frontend_peak() {
        let pool = Arc::new(tokio::sync::Semaphore::new(1024));
        let mut permit = Arc::clone(&pool).acquire_many_owned(522).await.unwrap();
        let mut context = Context::new(Arc::new(AtomicBool::new(false)), 522 << 20);
        context.retain(32 << 20).unwrap();
        context.release_media_scratch(&mut permit);
        assert_eq!(permit.num_permits(), 10 + 2 * 32 + 16);
        assert_eq!(pool.available_permits(), 1024 - permit.num_permits());
        drop(permit);
        assert_eq!(pool.available_permits(), 1024);
    }
    fn vision() -> orinfer_engine::vision::VisionSpec {
        serde_json::from_value(
            json!({"hidden":5120,"patch_size":16,"temporal_patch_size":2,
            "merge_size":2,"max_patches":32768,"max_features":32768,"pixels":"p","grid":"g",
            "length":"l","output":"o","features":"f","feature_index":"i","mrope_positions":"r",
            "plans":[{"patches":32768,"program":"vision"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}),
        )
        .unwrap()
    }
    #[tokio::test]
    async fn many_images_use_aggregate_bounded_memory_and_return_decoder_scratch() {
        let request: ChatRequest = serde_json::from_value(json!({"model":"test",
            "messages":[{"role":"user","content":(0..12).map(|_|json!({"type":"image_url",
                "image_url":{"url":"https://example.com/image.png"}})).collect::<Vec<_>>()}]}))
        .unwrap();
        let estimate = request_memory(&request, 262144, Some(&vision()), 2048, 8192);
        assert_eq!(
            estimate.mib as usize,
            request_memory_mib(&request, 262144, 8192) as usize + 192
        );
        let mut old = Context::new(
            Arc::new(AtomicBool::new(false)),
            request_memory_mib(&request, 262144, 8192) as usize * (1 << 20),
        );
        assert!(old.retain(576 << 20).is_err());
        let pool = Arc::new(tokio::sync::Semaphore::new(2048));
        let mut permit = Arc::clone(&pool)
            .acquire_many_owned(estimate.mib)
            .await
            .unwrap();
        let mut context = estimate.context(Arc::new(AtomicBool::new(false)));
        for _ in 0..12 {
            context.checkpoint(80 << 20).unwrap();
            context.retain(6 << 20).unwrap();
            context.worker_scratch(24 << 20).unwrap();
        }
        context.release_media_scratch(&mut permit);
        assert_eq!(permit.num_permits(), estimate.base_mib + 2 * 72 + 24 + 16);
        assert_eq!(pool.available_permits(), 2048 - permit.num_permits());
        assert!(Arc::clone(&pool).try_acquire_many_owned(512).is_ok());
        assert!(context.checkpoint(1024 << 20).is_err());
        drop(permit);
        assert_eq!(pool.available_permits(), 2048);
        let limited = request_memory(&request, 262144, Some(&vision()), 600, 8192);
        assert_eq!(limited.mib, 600);
        let mut context = limited.context(Arc::new(AtomicBool::new(false)));
        assert!(
            context
                .retain(576 << 20)
                .unwrap_err()
                .contains("reserved 600 MiB")
        );
    }
    #[tokio::test]
    async fn encoded_images_return_budget_while_http_delivery_keeps_its_baseline() {
        let pool = Arc::new(tokio::sync::Semaphore::new(2048));
        let permit = Arc::clone(&pool).acquire_many_owned(1200).await.unwrap();
        let mut context = Context::new(Arc::new(AtomicBool::new(false)), 1200 << 20);
        context.base = Some(8 << 20);
        let worker = Arc::new(context.lease(permit));
        let response = Arc::clone(&worker);
        worker.release_images();
        assert_eq!(pool.available_permits(), 2048 - 8);
        worker.release_images();
        assert_eq!(pool.available_permits(), 2048 - 8);
        drop(worker);
        assert_eq!(pool.available_permits(), 2048 - 8);
        drop(response);
        assert_eq!(pool.available_permits(), 2048);
    }
    #[test]
    fn single_image_and_text_admission_keep_their_original_bounds() {
        let mut request: ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
            {"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]})).unwrap();
        assert_eq!(
            request_memory(&request, 1024, Some(&vision()), 2048, 8192).mib,
            request_memory_mib(&request, 1024, 8192)
        );
        request.messages = vec![json!({"role":"user","content":"hello"})];
        assert_eq!(
            request_memory(&request, 1024, Some(&vision()), 2048, 8192).mib,
            request_memory_mib(&request, 1024, 8192)
        );
        request.messages = vec![json!({"role":"user","content":"large text".repeat(1024*1024)})];
        assert!(request_memory(&request, 1024, Some(&vision()), 16, 8192).mib > 16);
    }
    #[test]
    fn probability_budget_counts_completion_limit_and_streaming_retention() {
        let mut request: ChatRequest = serde_json::from_value(json!({"model":"test",
            "messages":[{"role":"user","content":"hello"}],"max_completion_tokens":512,
            "logprobs":true,"top_logprobs":20}))
        .unwrap();
        let complete = request_memory_mib(&request, 1024, 8192);
        request.stream = true;
        let streaming = request_memory_mib(&request, 1024, 8192);
        assert!(complete > streaming);
        request.logprobs = Some(false);
        assert!(streaming > request_memory_mib(&request, 1024, 8192));
        request.logprobs = Some(true);
        request.stream = false;
        request.max_completion_tokens = Some(usize::MAX);
        let capped = request_memory_mib(&request, 1024, 8192);
        request.max_completion_tokens = Some(1024);
        assert_eq!(capped, request_memory_mib(&request, 1024, 8192));
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
    pub slot: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
    pub body: Arc<tokio::sync::OwnedSemaphorePermit>,
}
pub(super) async fn ingress(
    State(state): State<Service>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    ingress_inner(state, request, next, true).await
}
pub(super) async fn render_ingress(
    State(state): State<Service>,
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    ingress_inner(state, request, next, false).await
}
async fn ingress_inner(
    state: Service,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
    generation: bool,
) -> Response {
    if !authorized(&state, request.headers()) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    if !state.activity.lifecycle.is_ready() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "GPU worker is not ready");
    }
    let slot = if generation {
        match Arc::clone(&state.slots).try_acquire_owned() {
            Ok(v) => Some(Arc::new(v)),
            Err(_) => return error(StatusCode::TOO_MANY_REQUESTS, "Request queue is full"),
        }
    } else {
        None
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
        slot,
        body: Arc::new(body),
    });
    next.run(request).await
}
