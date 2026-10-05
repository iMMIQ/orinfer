use std::sync::{
    Mutex,
    atomic::{AtomicU8, Ordering},
};

#[derive(Default)]
pub(super) struct Lifecycle {
    phase: AtomicU8,
    failure: Mutex<Option<orin_engine::error::EngineError>>,
}
impl Lifecycle {
    pub fn supervise(&self, work: impl FnOnce()) -> std::thread::Result<()> {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        if outcome.is_err() {
            self.fail(orin_engine::error::EngineError::take(
                "GPU worker panicked".into(),
                true,
            ));
        }
        outcome
    }

    pub fn ready(&self) {
        let _ = self
            .phase
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire);
    }
    pub fn drain(&self) {
        let _ = self
            .phase
            .compare_exchange(1, 2, Ordering::AcqRel, Ordering::Acquire);
    }
    pub fn fail(&self, failure: orin_engine::error::EngineError) {
        if let Ok(mut value) = self.failure.lock() {
            if value.is_some() {
                return;
            }
            *value = Some(failure);
            self.phase.store(3, Ordering::Release);
        } else {
            self.phase.store(3, Ordering::Release);
        }
    }
    pub fn is_ready(&self) -> bool {
        self.phase.load(Ordering::Acquire) == 1
    }
    pub fn name(&self) -> &'static str {
        match self.phase.load(Ordering::Acquire) {
            0 => "starting",
            1 => "ready",
            2 => "draining",
            _ => "failed",
        }
    }
    pub fn failure(&self) -> Option<orin_engine::error::EngineError> {
        self.failure.lock().ok().and_then(|v| v.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_panic_immediately_invalidates_readiness() {
        let state = Lifecycle::default();
        state.ready();
        assert!(
            state
                .supervise(|| panic!("simulated worker failure"))
                .is_err()
        );
        assert!(!state.is_ready());
        assert_eq!(state.name(), "failed");
    }
    #[test]
    fn failure_cannot_be_overwritten_by_ready_or_shutdown() {
        let state = Lifecycle::default();
        assert!(!state.is_ready());
        state.ready();
        assert!(state.is_ready());
        state.drain();
        assert!(!state.is_ready());
        state.fail(orin_engine::error::EngineError::take("panic".into(), true));
        state.ready();
        state.drain();
        assert_eq!(state.name(), "failed");
        assert!(state.failure().is_some());
        state.fail(orin_engine::error::EngineError::take(
            "later cleanup failure".into(),
            true,
        ));
        assert_eq!(state.failure().unwrap().message, "panic");
    }
}
