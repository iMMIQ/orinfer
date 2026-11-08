//! Request snapshots prepared once after device work has been submitted.
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

type Result<T> = std::result::Result<T, String>;
type Work<T> = Box<dyn FnOnce() -> Result<T> + Send>;
pub(crate) struct Prepared<T> {
    value: OnceLock<Result<(T, f64)>>,
    work: Mutex<Option<Work<T>>>,
}
impl<T: Default> Default for Prepared<T> {
    fn default() -> Self {
        Self {
            value: OnceLock::from(Ok((T::default(), 0.0))),
            work: Mutex::new(None),
        }
    }
}
impl<T> Prepared<T> {
    pub fn new(work: impl FnOnce() -> Result<T> + Send + 'static) -> Self {
        Self {
            value: OnceLock::new(),
            work: Mutex::new(Some(Box::new(work))),
        }
    }
    pub fn get(&self) -> Result<&T> {
        self.value
            .get_or_init(|| {
                let started = Instant::now();
                let work = self
                    .work
                    .lock()
                    .map_err(|_| "Sampling preparation lock poisoned")?
                    .take()
                    .ok_or("Missing sampling preparation")?;
                // A preparation failure must become an ordinary result so callers
                // still wait for the outstanding device completion before returning.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
                    .unwrap_or_else(|_| Err("Sampling preparation panicked".into()))
                    .map(|value| (value, started.elapsed().as_secs_f64()))
            })
            .as_ref()
            .map(|value| &value.0)
            .map_err(Clone::clone)
    }
    pub fn seconds(&self) -> f64 {
        self.value
            .get()
            .and_then(|v| v.as_ref().ok())
            .map_or(0.0, |v| v.1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
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
