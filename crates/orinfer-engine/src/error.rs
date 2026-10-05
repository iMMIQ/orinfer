//! Structured failure information at the runtime/transport boundary.
//! Driver calls record numeric errors on the thread owning the CUDA context;
//! classification never depends on matching human-readable error strings.
use std::cell::RefCell;

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Request,
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
}
pub(crate) fn record_cuda(code: i32) {
    DRIVER_FAILURE.with(|v| {
        // Preserve the initiating error through rollback/cleanup errors.
        if v.borrow().is_none() {
            *v.borrow_mut() = Some(code);
        }
    });
}
impl EngineError {
    pub fn take(message: String, invariant: bool) -> Self {
        let cuda_code = DRIVER_FAILURE.with(|v| v.borrow_mut().take());
        let kind = match cuda_code {
            Some(2) => ErrorKind::OutOfMemory,
            Some(_) => ErrorKind::Cuda,
            None if invariant => ErrorKind::Invariant,
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
        self.kind != ErrorKind::Request
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
