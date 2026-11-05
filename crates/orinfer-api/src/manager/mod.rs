//! CPU-only model catalog and lifecycle supervision. CUDA stays in child workers.
pub mod client;
mod ollama;
mod options;
mod pull;
mod registry;
mod routes;
#[cfg(test)]
mod tests;

pub use options::{ManagerConfig, WorkerOptions, default_data_dir, runtime_dir};
pub use registry::ModelRecord;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    os::{fd::AsRawFd, unix::fs::PermissionsExt},
    path::PathBuf,
    sync::{
        Arc, Mutex as SyncMutex,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use tokio::{
    process::{Child, Command},
    sync::{Mutex, Semaphore},
};

type Result<T> = std::result::Result<T, String>;

struct Runner {
    key: String,
    model: String,
    child: Mutex<Child>,
    client: reqwest::Client,
    active: AtomicUsize,
    last_used: SyncMutex<Instant>,
    keep_alive_s: AtomicU64,
    draining: AtomicBool,
    loading: AtomicBool,
    directory: PathBuf,
    pid: u32,
    batch: bool,
}
#[derive(Default)]
struct BudgetTotals {
    requests: usize,
    loading: usize,
}
#[derive(Clone, Copy)]
enum BudgetKind {
    Request,
    Loading,
}
struct MemoryBudget {
    bytes: usize,
    kind: BudgetKind,
    total: Arc<SyncMutex<BudgetTotals>>,
}
impl Drop for MemoryBudget {
    fn drop(&mut self) {
        let mut total = self.total.lock().unwrap_or_else(|e| e.into_inner());
        let used = match self.kind {
            BudgetKind::Request => &mut total.requests,
            BudgetKind::Loading => &mut total.loading,
        };
        *used = used.saturating_sub(self.bytes);
    }
}
struct Lease(Arc<Runner>, Option<MemoryBudget>);
impl Drop for Lease {
    fn drop(&mut self) {
        *self.0.last_used.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}
enum Acquisition {
    Ready(Lease, ModelRecord),
    Loading(Lease, ModelRecord, MemoryBudget),
}
struct Manager {
    config: ManagerConfig,
    directory: PathBuf,
    executable: PathBuf,
    records: Mutex<BTreeMap<String, ModelRecord>>,
    runners: Mutex<BTreeMap<String, Arc<Runner>>>,
    failures: Mutex<BTreeMap<String, (Instant, String)>>,
    transition: Semaphore,
    requests: Arc<Semaphore>,
    body_memory: Arc<Semaphore>,
    budgets: Arc<SyncMutex<BudgetTotals>>,
    gpu_owner: SyncMutex<Option<File>>,
    stopping: AtomicBool,
}

pub(crate) fn locked_file(path: &std::path::Path) -> Result<File> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    // SAFETY: The live descriptor stays owned until model teardown completes.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(format!("Resource lock is busy: {}", path.display()));
    }
    Ok(file)
}
fn private_directory(path: &std::path::Path) -> Result<()> {
    fs::create_dir_all(path).map_err(|e| e.to_string())?;
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    use std::os::unix::fs::MetadataExt;
    // SAFETY: getuid reads this process's numeric identity.
    if !metadata.is_dir() || metadata.uid() != unsafe { libc::getuid() } {
        return Err("Runtime directory must be owned by the current user".into());
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|e| e.to_string())
}
fn available_bytes() -> Result<usize> {
    let raw = fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    raw.lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse::<usize>()
                .ok()
        })
        .and_then(|kb| kb.checked_mul(1024))
        .ok_or_else(|| "Cannot read unified memory availability".into())
}
fn runner_key(record: &ModelRecord) -> String {
    let mut options = record.options.clone();
    options.default_request_params.clear();
    orinfer_engine::artifact::sha256(
        &serde_json::to_vec(&(
            record.identity.as_str(),
            record.execution_package.as_str(),
            options,
        ))
        .unwrap(),
    )
}

fn retryable(message: &str) -> bool {
    message.starts_with("Insufficient unified memory")
        || message.starts_with("Resource lock is busy:")
        || matches!(
            message,
            "Model is loading" | "Model is draining" | "Model is executing an offline batch job"
        )
}

impl Manager {
    async fn acquire(self: &Arc<Self>, name: &str) -> Result<(Lease, ModelRecord)> {
        let manager = Arc::clone(self);
        let name = name.to_owned();
        let (mut reply, receive) = tokio::sync::oneshot::channel();
        // Finish or roll back startup even if its initiating HTTP client leaves.
        // Waiting for capacity or another startup must remain cancellable.
        tokio::spawn(async move {
            let started = Instant::now();
            loop {
                if reply.is_closed() {
                    return;
                }
                // Ready models can accept work while another model starts or
                // unloads. Only lifecycle changes need the transition permit.
                let result = match manager.ready(&name).await {
                    Ok(Some((lease, record))) => Ok(Acquisition::Ready(lease, record)),
                    Err(error) => Err(error),
                    Ok(None) => {
                        let transition = tokio::select! {
                            biased;
                            _ = reply.closed() => return,
                            permit = manager.transition.acquire() => permit,
                        };
                        match transition {
                            Ok(_permit) if !reply.is_closed() => manager.acquire_inner(&name).await,
                            Ok(_) => return,
                            Err(_) => Err("Service is stopping".into()),
                        }
                    }
                };
                // The lifecycle permit has been released: different models
                // may load concurrently against separately reserved budgets.
                let result = match result {
                    Ok(Acquisition::Ready(lease, record)) => Ok((lease, record)),
                    Ok(Acquisition::Loading(lease, record, budget)) => {
                        let result = manager.finish_start(&lease.0).await;
                        drop(budget);
                        if result.is_ok() {
                            lease.0.loading.store(false, Ordering::Release);
                        }
                        result.map(|()| (lease, record))
                    }
                    Err(error) => Err(error),
                };
                let wait = result.as_ref().err().is_some_and(|e| retryable(e));
                if !wait
                    || manager.stopping.load(Ordering::Acquire)
                    || started.elapsed() >= Duration::from_secs(manager.config.load_timeout_s)
                {
                    let _ = reply.send(result);
                    return;
                }
                tokio::select! {
                    _ = reply.closed() => return,
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {}
                }
            }
        });
        receive.await.map_err(|e| e.to_string())?
    }
    async fn ready(&self, name: &str) -> Result<Option<(Lease, ModelRecord)>> {
        if self.stopping.load(Ordering::Acquire) {
            return Err("Service is stopping".into());
        }
        let record = self
            .records
            .lock()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Unknown model: {name}"))?;
        let lease = {
            let runners = self.runners.lock().await;
            let Some(runner) = runners.get(&runner_key(&record)) else {
                return Ok(None);
            };
            if runner.loading.load(Ordering::Acquire) {
                return Err("Model is loading".into());
            }
            if runner.batch {
                return Err("Model is executing an offline batch job".into());
            }
            if runner.draining.load(Ordering::Acquire) {
                return Err("Model is draining".into());
            }
            // Claiming a lease and marking a worker for eviction share this
            // lock, so an idle eviction cannot race a newly active request.
            runner.active.fetch_add(1, Ordering::AcqRel);
            Lease(Arc::clone(runner), None)
        };
        let exited = lease
            .0
            .child
            .lock()
            .await
            .try_wait()
            .map_err(|e| e.to_string())?
            .is_some();
        if exited {
            return Ok(None);
        }
        Ok(Some((lease, record)))
    }
    // The caller holds transition until the memory reservation and child are
    // published. Waiting for CUDA readiness happens outside this critical section.
    async fn acquire_inner(&self, name: &str) -> Result<Acquisition> {
        if let Some((lease, record)) = self.ready(name).await? {
            return Ok(Acquisition::Ready(lease, record));
        }
        let record = self
            .records
            .lock()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Unknown model: {name}"))?;
        let key = runner_key(&record);
        // Only an exited worker can remain after the ready check. Startup and
        // sweep also hold transition, so no replacement can be removed here.
        self.runners.lock().await.remove(&key);
        if let Some((at, message)) = self.failures.lock().await.get(&key)
            && at.elapsed() < Duration::from_secs(10)
        {
            return Err(message.clone());
        }
        let result = self.start(&record, &key).await;
        match result {
            Ok((runner, budget)) => {
                runner.active.store(1, Ordering::Release);
                self.runners
                    .lock()
                    .await
                    .insert(key.clone(), Arc::clone(&runner));
                self.failures.lock().await.remove(&key);
                Ok(Acquisition::Loading(Lease(runner, None), record, budget))
            }
            Err(error) => {
                if !retryable(&error) {
                    self.failures
                        .lock()
                        .await
                        .insert(key, (Instant::now(), error.clone()));
                }
                self.release_gpu_if_idle().await;
                Err(error)
            }
        }
    }

    async fn start(&self, record: &ModelRecord, key: &str) -> Result<(Arc<Runner>, MemoryBudget)> {
        let actual = orinfer_engine::artifact::sha256(
            &fs::read(record.path.join("cache/model.json")).map_err(|e| e.to_string())?,
        );
        if actual != record.identity {
            return Err("Model identity changed; import the new version before loading".into());
        }
        self.reserve_space(record).await?;
        let budget = self.reserve_budget(record.load_bytes, BudgetKind::Loading)?;
        let directory = self.directory.join(&key[..16]);
        private_directory(&directory)?;
        let socket = directory.join("worker.sock");
        if socket.exists() {
            fs::remove_file(&socket).map_err(|e| e.to_string())?;
        }
        let log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("worker.log"))
            .map_err(|e| e.to_string())?;
        let mut options = record.options.clone();
        // Alias-specific sampling defaults are applied by the gateway.
        options.default_request_params.clear();
        let mut command = self.command();
        command
            .arg("worker")
            .arg(&record.path)
            .arg("--model")
            .arg(&record.name)
            .arg("--listen")
            .arg(format!("unix:{}", socket.display()))
            .arg("--gpu-lock")
            .arg(directory.join("gpu.lock"))
            .args(options.arguments())
            .stdin(std::process::Stdio::null())
            .stdout(log.try_clone().map_err(|e| e.to_string())?)
            .stderr(log);
        let client = reqwest::Client::builder()
            .no_proxy()
            .unix_socket(socket)
            .connect_timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| e.to_string())?;
        let child = command.spawn().map_err(|e| e.to_string())?;
        let pid = child.id().ok_or("Worker did not start")?;
        Ok((
            Arc::new(Runner {
                key: key.into(),
                model: record.name.clone(),
                child: Mutex::new(child),
                client,
                active: AtomicUsize::new(0),
                last_used: SyncMutex::new(Instant::now()),
                batch: false,
                keep_alive_s: AtomicU64::new(self.config.keep_alive_s),
                draining: AtomicBool::new(false),
                loading: AtomicBool::new(true),
                directory,
                pid,
            }),
            budget,
        ))
    }
    async fn finish_start(&self, runner: &Arc<Runner>) -> Result<()> {
        let result = self.wait_ready(runner).await;
        if let Err(error) = &result {
            let _transition = self
                .transition
                .acquire()
                .await
                .map_err(|_| "Service is stopping")?;
            self.failures
                .lock()
                .await
                .insert(runner.key.clone(), (Instant::now(), error.clone()));
            self.remove_runner(runner).await?;
        }
        result
    }
    async fn wait_ready(&self, runner: &Runner) -> Result<()> {
        let at = Instant::now();
        loop {
            if let Some(status) = runner
                .child
                .lock()
                .await
                .try_wait()
                .map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "Worker exited with {status}; see {}",
                    runner.directory.join("worker.log").display()
                ));
            }
            if at.elapsed() >= Duration::from_secs(self.config.load_timeout_s)
                || self.stopping.load(Ordering::Acquire)
            {
                return Err("Model loading timed out or service stopped".into());
            }
            if let Ok(response) = runner
                .client
                .get("http://localhost/health")
                .timeout(Duration::from_secs(1))
                .send()
                .await
                && let Ok(health) = response.json::<Value>().await
                && health["status"] == "ready"
                && health["model"] == runner.model
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        eprintln!(
            "MODEL LOADED {} pid={} after {:.3}s",
            runner.model,
            runner.pid,
            at.elapsed().as_secs_f64()
        );
        Ok(())
    }
    fn memory_headroom(&self) -> Result<usize> {
        let reserve = self
            .config
            .memory_reserve_mib
            .checked_mul(1 << 20)
            .ok_or("Memory reserve overflow")?;
        Ok(available_bytes()?.saturating_sub(reserve))
    }
    fn reserve_budget(&self, bytes: usize, kind: BudgetKind) -> Result<MemoryBudget> {
        let mut total = self.budgets.lock().unwrap_or_else(|e| e.into_inner());
        // Keep pending loads fully reserved until ready, even while their
        // partial allocations already appear in MemAvailable. This errs on
        // the conservative side during overlapping CUDA startup.
        let proposed = total
            .requests
            .checked_add(total.loading)
            .and_then(|n| n.checked_add(bytes))
            .ok_or("Global memory budget overflow")?;
        if self.memory_headroom()? < proposed {
            return Err("Insufficient unified memory for allocation budget".into());
        }
        match kind {
            BudgetKind::Request => total.requests += bytes,
            BudgetKind::Loading => total.loading += bytes,
        }
        Ok(MemoryBudget {
            bytes,
            kind,
            total: Arc::clone(&self.budgets),
        })
    }
    async fn reserve_space(&self, record: &ModelRecord) -> Result<()> {
        loop {
            let runners: Vec<_> = self.runners.lock().await.values().cloned().collect();
            let count_full = self
                .config
                .max_loaded_models
                .is_some_and(|n| runners.len() >= n);
            let fits = {
                // Both request state and overlapping loads reserve against one
                // ledger. A final atomic reservation in start() rechecks races.
                let total = self.budgets.lock().unwrap_or_else(|e| e.into_inner());
                self.memory_headroom()?
                    .saturating_sub(total.requests)
                    .saturating_sub(total.loading)
                    >= record.load_bytes
            };
            if !count_full && fits {
                break;
            }
            let Some(victim) = self.claim_idle().await else {
                return Err("Insufficient unified memory or loaded-model capacity; active models are retained".into());
            };
            self.remove_runner(&victim).await?;
        }
        {
            let mut owner = self.gpu_owner.lock().unwrap_or_else(|e| e.into_inner());
            if owner.is_none() {
                *owner = Some(locked_file(&self.config.gpu_lock)?);
            }
        }
        Ok(())
    }
    // Caller holds transition. Claim under the same lock as ready(), before
    // waiting for process teardown; keep_alive never prevents pressure eviction.
    async fn claim_idle(&self) -> Option<Arc<Runner>> {
        let runners = self.runners.lock().await;
        let victim = runners
            .values()
            .filter(|r| {
                r.active.load(Ordering::Acquire) == 0
                    && !r.draining.load(Ordering::Acquire)
                    && !r.loading.load(Ordering::Acquire)
            })
            .min_by_key(|r| *r.last_used.lock().unwrap_or_else(|e| e.into_inner()))
            .cloned()?;
        victim.draining.store(true, Ordering::Release);
        Some(victim)
    }
    async fn reserve_request_with_eviction(
        &self,
        record: &ModelRecord,
        request: &Value,
        input_bytes: usize,
    ) -> Result<MemoryBudget> {
        match self.reserve_request(record, request, input_bytes) {
            Ok(budget) => return Ok(budget),
            Err(error) if !error.starts_with("Insufficient unified memory") => return Err(error),
            Err(_) => {}
        }
        let _transition = self
            .transition
            .acquire()
            .await
            .map_err(|_| "Service is stopping")?;
        loop {
            if self.stopping.load(Ordering::Acquire) {
                return Err("Service is stopping".into());
            }
            let error = match self.reserve_request(record, request, input_bytes) {
                Ok(budget) => return Ok(budget),
                Err(error) if error.starts_with("Insufficient unified memory") => error,
                Err(error) => return Err(error),
            };
            let Some(victim) = self.claim_idle().await else {
                return Err(error);
            };
            self.remove_runner(&victim).await?;
        }
    }
    fn reserve_request(
        &self,
        record: &ModelRecord,
        request: &Value,
        input_bytes: usize,
    ) -> Result<MemoryBudget> {
        let output = request
            .get("max_completion_tokens")
            .or_else(|| request.get("max_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(8192);
        let tokens = input_bytes
            .saturating_add(usize::try_from(output).unwrap_or(usize::MAX))
            .min(record.max_context);
        let bytes = record
            .kv_bytes_per_token
            .checked_mul(tokens)
            .and_then(|n| n.checked_add(record.sequence_bytes))
            .ok_or("Request memory estimate overflow")?;
        self.reserve_budget(bytes, BudgetKind::Request)
    }
    fn command(&self) -> Command {
        let mut command = Command::new(&self.executable);
        command.kill_on_drop(true);
        let parent = std::process::id();
        // SAFETY: The post-fork hook performs only async-signal-safe Linux calls.
        // A worker cannot outlive a crashed supervisor; CUDA is initialized after exec.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() as u32 != parent {
                    return Err(std::io::Error::other(
                        "Supervisor exited before worker exec",
                    ));
                }
                Ok(())
            });
        }
        command
    }

    async fn remove_runner(&self, runner: &Arc<Runner>) -> Result<()> {
        {
            let _runners = self.runners.lock().await;
            runner.draining.store(true, Ordering::Release);
        }
        let mut child = runner.child.lock().await;
        if child.try_wait().map_err(|e| e.to_string())?.is_none() {
            // SAFETY: This still-running child is retained and cannot have its PID reused.
            if unsafe { libc::kill(runner.pid as i32, libc::SIGTERM) } != 0 {
                return Err(std::io::Error::last_os_error().to_string());
            }
            match tokio::time::timeout(Duration::from_secs(35), child.wait()).await {
                Ok(status) => {
                    status.map_err(|e| e.to_string())?;
                }
                Err(_) => {
                    child.kill().await.map_err(|e| e.to_string())?;
                }
            }
        }
        let mut runners = self.runners.lock().await;
        if runners
            .get(&runner.key)
            .is_some_and(|current| Arc::ptr_eq(current, runner))
        {
            runners.remove(&runner.key);
            let _ = fs::remove_file(runner.directory.join("worker.sock"));
        }
        drop(runners);
        self.release_gpu_if_idle().await;
        Ok(())
    }
    async fn release_gpu_if_idle(&self) {
        if self.runners.lock().await.is_empty() {
            self.gpu_owner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take();
        }
    }
    async fn stop(&self, name: &str, force: bool) -> Result<()> {
        let record = self
            .records
            .lock()
            .await
            .get(name)
            .cloned()
            .ok_or_else(|| format!("Unknown model: {name}"))?;
        let transition = self
            .transition
            .acquire()
            .await
            .map_err(|_| "Service is stopping")?;
        let runner = {
            let runners = self.runners.lock().await;
            let Some(runner) = runners.get(&runner_key(&record)).cloned() else {
                return Ok(());
            };
            runner.draining.store(true, Ordering::Release);
            runner
        };
        drop(transition);
        let at = Instant::now();
        while !force && runner.active.load(Ordering::Acquire) != 0 {
            if at.elapsed() > Duration::from_secs(30) {
                let _transition = self
                    .transition
                    .acquire()
                    .await
                    .map_err(|_| "Service is stopping")?;
                let runners = self.runners.lock().await;
                if runners
                    .get(&runner.key)
                    .is_some_and(|current| Arc::ptr_eq(current, &runner))
                {
                    runner.draining.store(false, Ordering::Release);
                }
                return Err(
                    "Model still has active requests; stop again after completion or use --force"
                        .into(),
                );
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        let _transition = self
            .transition
            .acquire()
            .await
            .map_err(|_| "Service is stopping")?;
        self.remove_runner(&runner).await
    }
    async fn sweep(&self) {
        let Ok(_transition) = self.transition.try_acquire() else {
            return;
        };
        let runners: Vec<_> = self.runners.lock().await.values().cloned().collect();
        for runner in runners {
            let exited = runner
                .child
                .lock()
                .await
                .try_wait()
                .ok()
                .flatten()
                .is_some();
            let expired = {
                if runner.loading.load(Ordering::Acquire) {
                    continue;
                }
                let runners = self.runners.lock().await;
                if !runners
                    .get(&runner.key)
                    .is_some_and(|current| Arc::ptr_eq(current, &runner))
                {
                    continue;
                }
                let expired = runner.active.load(Ordering::Acquire) == 0
                    && runner
                        .last_used
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .elapsed()
                        >= Duration::from_secs(runner.keep_alive_s.load(Ordering::Relaxed));
                if expired || exited {
                    runner.draining.store(true, Ordering::Release);
                }
                expired
            };
            if (expired || exited)
                && let Err(e) = self.remove_runner(&runner).await
            {
                eprintln!("Worker cleanup: {e}");
            }
        }
    }
    async fn status(&self) -> Value {
        let runners: Vec<_> = self.runners.lock().await.values().cloned().collect();
        let records = self.records.lock().await;
        json!(runners.iter().map(|r| json!({"model":r.model,"pid":r.pid,"active_requests":r.active.load(Ordering::Acquire),
            "state":if r.draining.load(Ordering::Acquire) {"draining"} else if r.loading.load(Ordering::Acquire) {"loading"} else if r.batch {"batch"} else {"ready"},
            "estimated_load_bytes":records.values().find(|record| runner_key(record)==r.key).map(|record| record.load_bytes),
            "idle_s":r.last_used.lock().unwrap_or_else(|e| e.into_inner()).elapsed().as_secs_f64(),
            "keep_alive_s":r.keep_alive_s.load(Ordering::Relaxed),"log":r.directory.join("worker.log")})).collect::<Vec<_>>())
    }
}

pub fn run(mut config: ManagerConfig) -> Result<()> {
    if config.max_loaded_models == Some(0)
        || config.load_timeout_s == 0
        || config.memory_reserve_mib > 65536
        || !(1..=2048).contains(&config.worker_options.request_body_mib)
    {
        return Err("Invalid model manager limits".into());
    }
    private_directory(&config.data_dir)?;
    config.data_dir = config.data_dir.canonicalize().map_err(|e| e.to_string())?;
    let directory = runtime_dir(&config.data_dir);
    private_directory(directory.parent().ok_or("Invalid runtime directory")?)?;
    private_directory(&directory)?;
    let _owner = locked_file(&directory.join("daemon.lock"))?;
    let records = registry::read(&config.data_dir)?;
    let body_memory_mib = 512.max(config.worker_options.request_body_mib * 4);
    let manager = Arc::new(Manager {
        config,
        directory,
        executable: std::env::current_exe().map_err(|e| e.to_string())?,
        records: Mutex::new(records),
        runners: Mutex::new(BTreeMap::new()),
        failures: Mutex::new(BTreeMap::new()),
        transition: Semaphore::new(1),
        requests: Arc::new(Semaphore::new(128)),
        body_memory: Arc::new(Semaphore::new(body_memory_mib)),
        budgets: Arc::new(SyncMutex::new(BudgetTotals::default())),
        gpu_owner: SyncMutex::new(None),
        stopping: AtomicBool::new(false),
    });
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let result = runtime.block_on(routes::serve(manager));
    runtime.shutdown_timeout(Duration::from_secs(2));
    result
}
