//! Structured failure information at the runtime/transport boundary.
//! Driver calls record numeric errors on the thread owning the CUDA context;
//! classification never depends on matching human-readable error strings.
use std::cell::{Cell, RefCell};

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Request,
    Capacity,
    OutOfMemory,
    Cuda,
    Invariant,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub message: String,
    pub cuda_code: Option<i32>,
}
thread_local! {
    static DRIVER_FAILURE: RefCell<Option<i32>> = const { RefCell::new(None) };
    static CAPACITY_FAILURE: Cell<bool> = const { Cell::new(false) };
}
pub(crate) fn record_cuda(code: i32) {
    DRIVER_FAILURE.with(|v| {
        // Preserve the initiating error through rollback/cleanup errors.
        if v.borrow().is_none() {
            *v.borrow_mut() = Some(code);
        }
    });
}
pub(crate) fn record_capacity() {
    CAPACITY_FAILURE.with(|v| v.set(true));
}
impl EngineError {
    pub fn take(message: String, invariant: bool) -> Self {
        let cuda_code = DRIVER_FAILURE.with(|v| v.borrow_mut().take());
        let capacity = CAPACITY_FAILURE.with(|v| v.replace(false));
        let kind = match cuda_code {
            Some(2) => ErrorKind::OutOfMemory,
            Some(_) => ErrorKind::Cuda,
            None if invariant => ErrorKind::Invariant,
            None if capacity => ErrorKind::Capacity,
            None => ErrorKind::Request,
        };
        Self {
            kind,
            message,
            cuda_code,
        }
    }
    /// Allocation failure can also interrupt a multi-step state transition.
    /// Until recovery is proved, every driver/invariant failure retires the worker.
    pub fn is_fatal(&self) -> bool {
        !matches!(self.kind, ErrorKind::Request | ErrorKind::Capacity)
    }
}
impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for EngineError {}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_capacity_is_recoverable_and_does_not_leak() {
        record_capacity();
        let error = EngineError::take("opaque message".into(), false);
        assert_eq!(error.kind, ErrorKind::Capacity);
        assert!(!error.is_fatal());
        assert_eq!(
            EngineError::take("next".into(), false).kind,
            ErrorKind::Request
        );
        record_capacity();
        record_cuda(2);
        assert!(EngineError::take("allocation failed".into(), false).is_fatal());
    }
    #[test]
    fn driver_code_survives_cleanup_and_does_not_leak_between_errors() {
        record_cuda(700);
        record_cuda(2);
        let error = EngineError::take("opaque message".into(), false);
        assert!(error.is_fatal());
        assert_eq!(error.cuda_code, Some(700));
        assert_eq!(
            EngineError::take("CUDA error text alone".into(), false).kind,
            ErrorKind::Request
        );
        record_cuda(2);
        assert_eq!(
            EngineError::take("allocation".into(), false).kind,
            ErrorKind::OutOfMemory
        );
    }
}
