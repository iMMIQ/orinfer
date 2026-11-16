//! Request snapshots prepared once after device work has been submitted.
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

type Result<T> = std::result::Result<T, String>;
type Work<T> = Box<dyn FnOnce() -> Result<T> + Send>;
pub(crate) struct Prepared<T> {
    value: OnceLock<Result<(T, f64)>>,
    work: Mutex<Option<Work<T>>>,
    started: AtomicBool,
}
impl<T: Default> Default for Prepared<T> {
    fn default() -> Self {
        Self {
            value: OnceLock::from(Ok((T::default(), 0.0))),
            work: Mutex::new(None),
            started: AtomicBool::new(true),
        }
    }
}
impl<T> Prepared<T> {
    pub fn new(work: impl FnOnce() -> Result<T> + Send + 'static) -> Self {
        Self {
            value: OnceLock::new(),
            work: Mutex::new(Some(Box::new(work))),
            started: AtomicBool::new(false),
        }
    }
    pub fn deferred() -> Self {
        Self {
            value: OnceLock::new(),
            work: Mutex::new(None),
            started: AtomicBool::new(false),
        }
    }
    pub fn get(&self) -> Result<&T> {
        self.get_or(|| Err("Missing CPU preparation".into()))
    }
    pub fn get_or(&self, fallback: impl FnOnce() -> Result<T>) -> Result<&T> {
        self.value
            .get_or_init(|| {
                let started = Instant::now();
                let work = self
                    .work
                    .lock()
                    .map_err(|_| "CPU preparation lock poisoned")?
                    .take();
                // A preparation failure must become an ordinary result so callers
                // still wait for the outstanding device completion before returning.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match work {
                    Some(work) => work(),
                    None => fallback(),
                }))
                .unwrap_or_else(|_| Err("CPU preparation panicked".into()))
                .map(|value| (value, started.elapsed().as_secs_f64()))
            })
            .as_ref()
            .map(|value| &value.0)
            .map_err(Clone::clone)
    }
    pub fn start(self: &Arc<Self>, active: &Arc<AtomicBool>) -> bool
    where
        T: Send + Sync + 'static,
    {
        if rayon::current_num_threads() < 2
            || self.is_ready()
            || active
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return false;
        }
        if self.started.swap(true, Ordering::AcqRel) {
            active.store(false, Ordering::Release);
            return false;
        }
        let prepared = Arc::clone(self);
        let active = Arc::clone(active);
        rayon::spawn(move || {
            let _ = prepared.get();
            active.store(false, Ordering::Release);
        });
        true
    }
    pub fn seconds(&self) -> f64 {
        self.value
            .get()
            .and_then(|v| v.as_ref().ok())
            .map_or(0.0, |v| v.1)
    }
    pub fn is_ready(&self) -> bool {
        self.value.get().is_some()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn one_worker_pool_keeps_preparation_lazy() {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                let prepared = Arc::new(Prepared::new(|| Ok(3usize)));
                let active = Arc::new(AtomicBool::new(false));
                assert!(!prepared.start(&active));
                assert!(!prepared.is_ready());
                assert!(!active.load(Ordering::Relaxed));
                assert_eq!(*prepared.get().unwrap(), 3);
            });
    }
    #[test]
    fn eager_start_returns_without_waiting_and_only_one_rank_launches_work() {
        let (send, started) = std::sync::mpsc::channel();
        let (release, gate) = std::sync::mpsc::channel();
        let count = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&count);
        let value = Arc::new(Prepared::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
            send.send(()).unwrap();
            gate.recv().unwrap();
            Ok(7usize)
        }));
        let active = Arc::new(AtomicBool::new(false));
        std::thread::scope(|s| {
            for _ in 0..8 {
                let value = Arc::clone(&value);
                let active = Arc::clone(&active);
                s.spawn(move || value.start(&active));
            }
        });
        started
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert!(!value.is_ready());
        let other = Arc::new(Prepared::new(|| Ok(11usize)));
        assert!(!other.start(&active));
        assert!(!other.is_ready());
        release.send(()).unwrap();
        assert_eq!(*value.get().unwrap(), 7);
        assert_eq!(count.load(Ordering::Relaxed), 1);
        assert_eq!(*Prepared::deferred().get_or(|| Ok(9usize)).unwrap(), 9);
    }
    #[test]
    fn ranks_share_one_preparation_and_failures_do_not_leave_waiters() {
        for panic in [false, true] {
            let count = std::sync::Arc::new(AtomicUsize::new(0));
            let calls = count.clone();
            let value = Prepared::new(move || -> Result<usize> {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(!panic, "controlled preparation failure");
                Err("controlled grammar error".into())
            });
            std::thread::scope(|s| {
                for _ in 0..8 {
                    let value = &value;
                    s.spawn(move || assert!(value.get().is_err()));
                }
            });
            assert_eq!(count.load(Ordering::Relaxed), 1);
        }
    }
}
