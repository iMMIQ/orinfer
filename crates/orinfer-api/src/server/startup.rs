//! Shared model loading and GPU worker lifecycle for HTTP and offline jobs.
use super::*;
use std::sync::atomic::AtomicUsize;

static LIVE_MODEL_WORKERS: AtomicUsize = AtomicUsize::new(0);

/// Native resources can still be live after an API startup/drain deadline.
/// Standalone runners must avoid native process exit hooks in this state.
pub fn has_live_model_workers() -> bool {
    LIVE_MODEL_WORKERS.load(Ordering::Acquire) != 0
}
struct WorkerLifetime;
impl WorkerLifetime {
    fn new() -> Self {
        LIVE_MODEL_WORKERS.fetch_add(1, Ordering::AcqRel);
        Self
    }
}
impl Drop for WorkerLifetime {
    fn drop(&mut self) {
        LIVE_MODEL_WORKERS.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(super) struct Worker {
    pub state: Service,
    pub shutdown: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
}
impl Worker {
    pub async fn stop(self, timeout_ms: u64) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        // Closing the last ingress wakes an idle blocking receiver.
        drop(self.state);
        if !wait_worker(&self.handle, timeout_ms).await {
            return Err("GPU worker did not exit before drain deadline".into());
        }
        self.handle
            .join()
            .map_err(|_| "GPU worker panicked".to_owned())
    }
}

pub(super) async fn start(settings: &Settings) -> Result<Worker> {
    let descriptor: Value = serde_json::from_slice(
        &std::fs::read(settings.model_dir.join("cache/model.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    validate_shutdown_environment(
        descriptor["metadata"]["target"]
            .as_str()
            .unwrap_or_default(),
        std::env::var_os("ASCEND_COREDUMP_SIGNAL").as_deref(),
    )?;
    let started = Instant::now();
    let settings = settings.clone();
    let default_request_params = Arc::new(settings.default_request_params.clone());
    let (sender, receiver) = mpsc::channel(128);
    let (ready_sender, ready_receiver) = oneshot::channel();
    let model_id: Arc<str> = settings.model.clone().into();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_model = Arc::clone(&model_id);
    let worker_shutdown = Arc::clone(&shutdown);
    let scheduler = settings.scheduler;
    let verify_weights = settings.verify_weights;
    let resident_experts = settings.resident_experts;
    let prompt_lookup = settings.prompt_lookup;
    let activity = Arc::new(continuous::Activity::default());
    let lifecycle = Arc::clone(&activity.lifecycle);
    let worker_activity = Arc::clone(&activity);
    let worker_lifecycle = Arc::clone(&lifecycle);
    let limits = settings.limits;
    let worker_lifetime = WorkerLifetime::new();
    let worker = std::thread::Builder::new()
        .name("orinfer-gpu".into())
        .spawn(move || {
            // Declared before model ownership; cleared only after native
            // destruction, including error and panic cleanup, has completed.
            let _worker_lifetime = worker_lifetime;
            let _ = worker_lifecycle.supervise(|| {
                let initialize = || -> Result<_> {
                    let ((worker_codec, frontend_s), loaded) = load_parallel(
                        || {
                            let at = Instant::now();
                            load_frontend(&settings).map(|codec| (codec, at.elapsed().as_secs_f64()))
                        },
                        || load_model(&settings),
                    )?;
                    let model = &loaded.model;
                    if &worker_codec.asset_hashes != model.frontend_assets() {
                        return Err("Frontend assets changed while loading model".into());
                    }
                    if worker_codec.tokenizer.get_vocab_size(true) > model.vocab() {
                        return Err("Tokenizer exceeds model vocabulary".into());
                    }
                    let context = model.max_context();
                    let vision = model.vision().cloned();
                    if !model.batching_supported() && scheduler.max_active != 1 {
                        return Err("This execution package schedules one active request; set --max-active-requests 1 (additional requests queue)".into());
                    }
                    eprintln!(
                        "WORKER READY after {:.3}s; frontend={frontend_s:.3}s; model={:.3}s; parallel_load=true",
                        started.elapsed().as_secs_f64(), loaded.seconds
                    );
                    Ok((loaded, context, vision, worker_codec))
                };
                match initialize() {
                    Ok((mut loaded, context, vision, worker_codec)) => {
                        // The first user request would otherwise pay lease
                        // mapping, packet plans and module loading: measured
                        // ~0.6s of extra first-prompt TTFT on Flash Next.
                        let warm_started = Instant::now();
                        let greedy = orinfer_engine::sampling::Options {
                            temperature: 0.0,
                            ..Default::default()
                        };
                        if let Err(error) =
                            loaded.model.generate(&[0], 1, &greedy, || false, |_| true)
                        {
                            let message = format!("Model warmup failed: {error}");
                            worker_lifecycle.fail(
                                orinfer_engine::error::EngineError::take(message.clone(), true),
                            );
                            let _ = ready_sender.send(Err(message));
                            return;
                        }
                        eprintln!(
                            "WARMUP READY after {:.3}s",
                            warm_started.elapsed().as_secs_f64()
                        );
                        worker_lifecycle.ready();
                        let mtp_drafts = loaded.model.mtp_drafts();
                        let ready = (context, vision, mtp_drafts, Arc::clone(&worker_codec));
                        if ready_sender.send(Ok(ready)).is_ok() {
                            continuous::worker(
                                &mut loaded.model,
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
    let ready = ready_receiver
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r);
    let (context, vision, mtp_drafts, codec) = match ready {
        Ok(ready) => ready,
        Err(error) => {
            shutdown.store(true, Ordering::Relaxed);
            drop(sender);
            if wait_worker(&worker, limits.drain_ms).await {
                let _ = worker.join();
            }
            return Err(error);
        }
    };
    let state = Service {
        jobs: sender,
        slots: Arc::new(tokio::sync::Semaphore::new(128 + scheduler.max_active)),
        codec,
        model: model_id,
        context,
        mtp_drafts,
        prompt_lookup,
        verify_weights,
        resident_experts,
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
        default_request_params,
    };
    Ok(Worker {
        state,
        shutdown,
        handle: worker,
    })
}

fn validate_shutdown_environment(target: &str, capture: Option<&std::ffi::OsStr>) -> Result<()> {
    if target == "ascend910b" && capture != Some(std::ffi::OsStr::new("none")) {
        return Err("A2 service requires ASCEND_COREDUMP_SIGNAL=none before process startup: CANN trace signal handlers bypass graceful shutdown".into());
    }
    Ok(())
}

// Fields drop in declaration order, including when frontend preparation fails:
// finish model teardown before releasing GPU ownership.
struct LoadedModel<T = Model> {
    model: T,
    _lock: std::fs::File,
    seconds: f64,
}

fn load_model(settings: &Settings) -> Result<LoadedModel> {
    use std::os::fd::AsRawFd;
    let at = Instant::now();
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
    // SAFETY: The descriptor remains owned by this GPU worker through model teardown.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err("GPU experiment lock is busy".into());
    }
    let model = Model::load_with_options(
        &settings.model_dir,
        LoadOptions {
            cuda_graph: settings.cuda_graph,
            verify_weights: settings.verify_weights,
            load_workers: settings.load_workers,
            resident_experts: settings.resident_experts,
            prefix_cache_bytes: settings.prefix_cache_bytes,
            mtp_drafts: settings.mtp_drafts,
            prompt_lookup: settings.prompt_lookup,
        },
    )?;
    Ok(LoadedModel {
        model,
        _lock: lock,
        seconds: at.elapsed().as_secs_f64(),
    })
}

fn load_frontend(settings: &Settings) -> Result<Arc<ChatCodec>> {
    let codec = Arc::new(ChatCodec::load(&settings.model_dir)?);
    if !settings.default_request_params.is_empty() {
        codec
            .prepare(
                defaults::validation_request(&settings.default_request_params)?,
                "defaults-validation",
                usize::MAX,
                None,
                &mut preparation::Context::new(
                    Arc::new(AtomicBool::new(false)),
                    settings.limits.memory_mib * 1024 * 1024,
                ),
            )
            .map_err(|e| format!("default_request_params: {e}"))?;
    }
    Ok(codec)
}

/// Only the frontend crosses threads; model construction and destruction stay
/// on the caller's CUDA owner, including frontend errors and panics.
pub(crate) fn load_parallel<T: Send, U>(
    frontend: impl FnOnce() -> Result<T> + Send,
    model: impl FnOnce() -> Result<U>,
) -> Result<(T, U)> {
    std::thread::scope(|scope| {
        let frontend = std::thread::Builder::new()
            .name("orinfer-frontend".into())
            .spawn_scoped(scope, frontend)
            .map_err(|e| e.to_string())?;
        let model = model();
        let frontend = frontend
            .join()
            .map_err(|_| "Frontend loader panicked".to_owned())
            .and_then(|r| r);
        Ok((frontend?, model?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_liveness_covers_native_resource_destruction() {
        use std::sync::mpsc;
        struct NativeDrop(mpsc::Sender<()>, mpsc::Receiver<()>);
        impl Drop for NativeDrop {
            fn drop(&mut self) {
                self.0.send(()).unwrap();
                self.1.recv().unwrap();
            }
        }
        let (entered, destroyed) = mpsc::channel();
        let (release, gate) = mpsc::channel();
        let lifetime = WorkerLifetime::new();
        let worker = std::thread::spawn(move || {
            let _lifetime = lifetime;
            let _native = NativeDrop(entered, gate);
        });
        destroyed
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(has_live_model_workers());
        release.send(()).unwrap();
        worker.join().unwrap();
        assert!(!has_live_model_workers());
    }
    #[test]
    fn a2_signal_capture_conflicts_fail_before_worker_start() {
        use std::ffi::OsStr;
        assert!(validate_shutdown_environment("ascend910b", Some(OsStr::new("none"))).is_ok());
        for capture in [None, Some(OsStr::new("all")), Some(OsStr::new(""))] {
            let error = validate_shutdown_environment("ascend910b", capture).unwrap_err();
            assert!(error.contains("ASCEND_COREDUMP_SIGNAL=none"));
        }
        assert!(validate_shutdown_environment("cuda", None).is_ok());
    }

    #[test]
    fn frontend_overlaps_model_load_and_model_stays_on_owner() {
        let owner = std::thread::current().id();
        let (front_tx, front_rx) = std::sync::mpsc::channel();
        let (model_tx, model_rx) = std::sync::mpsc::channel();
        let (frontend, model) = load_parallel(
            move || {
                front_tx.send(std::thread::current().id()).unwrap();
                model_rx
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                Ok(7)
            },
            || {
                assert_ne!(
                    front_rx
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap(),
                    owner
                );
                model_tx.send(()).unwrap();
                // Rc cannot cross threads, just like the CUDA model.
                Ok(std::rc::Rc::new(std::thread::current().id()))
            },
        )
        .unwrap();
        assert_eq!(frontend, 7);
        assert_eq!(*model, owner);
    }

    #[test]
    fn frontend_error_or_panic_drops_model_on_owner() {
        struct Owned(std::rc::Rc<std::cell::Cell<bool>>, std::thread::ThreadId);
        impl Drop for Owned {
            fn drop(&mut self) {
                assert_eq!(std::thread::current().id(), self.1);
                self.0.set(true);
            }
        }
        for panic in [false, true] {
            let dropped = std::rc::Rc::new(std::cell::Cell::new(false));
            let result = load_parallel(
                || -> Result<()> {
                    if panic {
                        panic!("frontend failure");
                    }
                    Err("frontend failure".into())
                },
                || Ok(Owned(dropped.clone(), std::thread::current().id())),
            );
            assert!(result.is_err());
            assert!(dropped.get());
        }
    }

    #[test]
    fn frontend_failure_keeps_gpu_lock_through_model_teardown() {
        use std::os::fd::AsRawFd;
        struct AssertLocked<'a>(&'a std::fs::File);
        impl Drop for AssertLocked<'_> {
            fn drop(&mut self) {
                // SAFETY: Independent live descriptor to the same test lock.
                let code =
                    unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
                assert_eq!(code, -1);
                assert_eq!(
                    std::io::Error::last_os_error().raw_os_error(),
                    Some(libc::EWOULDBLOCK)
                );
            }
        }
        let path =
            std::env::temp_dir().join(format!("orinfer-startup-lock-{}", std::process::id()));
        let probe = std::fs::File::create_new(&path).unwrap();
        for panic in [false, true] {
            let lock = std::fs::File::open(&path).unwrap();
            assert_eq!(
                // SAFETY: Both descriptors are live; operations are nonblocking.
                unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                0
            );
            let result = load_parallel(
                || -> Result<()> {
                    if panic {
                        panic!("frontend failure");
                    }
                    Err("frontend failure".into())
                },
                || {
                    Ok(LoadedModel {
                        model: AssertLocked(&probe),
                        _lock: lock,
                        seconds: 0.,
                    })
                },
            );
            assert!(result.is_err());
            assert_eq!(
                // SAFETY: Teardown completed; this live descriptor can acquire ownership.
                unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
                0
            );
            // SAFETY: Release the lock acquired above before the next iteration.
            assert_eq!(unsafe { libc::flock(probe.as_raw_fd(), libc::LOCK_UN) }, 0);
        }
        drop(probe);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn model_error_waits_for_frontend_cleanup() {
        let finished = AtomicBool::new(false);
        let completed = &finished;
        let (tx, rx) = std::sync::mpsc::channel();
        let result = load_parallel(
            move || {
                rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                completed.store(true, Ordering::Relaxed);
                Ok(())
            },
            || -> Result<()> {
                tx.send(()).unwrap();
                Err("model failure".into())
            },
        );
        assert_eq!(result.unwrap_err(), "model failure");
        assert!(finished.load(Ordering::Relaxed));
    }
}
