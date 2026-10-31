//! Structured failure information at the runtime/transport boundary.
//! Driver calls record numeric errors on the thread owning the device context;
//! classification never depends on matching human-readable error strings.
use std::cell::{Cell, RefCell};

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Request,
    Capacity,
    OutOfMemory,
    Cuda,
    Ascend,
    Invariant,
}
#[derive(Debug, Clone, serde::Serialize)]
pub struct EngineError {
    pub kind: ErrorKind,
    pub message: String,
    pub cuda_code: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ascend_code: Option<i32>,
}
thread_local! {
    static DRIVER_FAILURE: RefCell<Option<i32>> = const { RefCell::new(None) };
    static ASCEND_FAILURE: Cell<Option<i32>> = const { Cell::new(None) };
    static CAPACITY_FAILURE: Cell<bool> = const { Cell::new(false) };
    static FORWARDED_FAILURE: RefCell<Option<EngineError>> = const { RefCell::new(None) };
}
/// Preserve the initiating device failure across a worker-thread reply.
pub(crate) fn forward(error: EngineError) -> String {
    let message = error.message.clone();
    FORWARDED_FAILURE.with(|value| {
        if value.borrow().is_none() {
            *value.borrow_mut() = Some(error);
        }
    });
    message
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
pub(crate) fn record_ascend(code: i32) {
    ASCEND_FAILURE.with(|v| {
        if v.get().is_none() {
            v.set(Some(code));
        }
    });
}
impl EngineError {
    pub fn take(message: String, invariant: bool) -> Self {
        let cuda_code = DRIVER_FAILURE.with(|v| v.borrow_mut().take());
        let capacity = CAPACITY_FAILURE.with(|v| v.replace(false));
        let ascend_code = ASCEND_FAILURE.with(|v| v.replace(None));
        if let Some(mut error) = FORWARDED_FAILURE.with(|v| v.borrow_mut().take()) {
            error.message = message;
            if invariant && !error.is_fatal() {
                error.kind = ErrorKind::Invariant;
            }
            return error;
        }
        let kind = match cuda_code {
            Some(2) => ErrorKind::OutOfMemory,
            Some(_) => ErrorKind::Cuda,
            None if ascend_code == Some(207001) => ErrorKind::OutOfMemory,
            None if ascend_code.is_some() => ErrorKind::Ascend,
            None if invariant => ErrorKind::Invariant,
            None if capacity => ErrorKind::Capacity,
            None => ErrorKind::Request,
        };
        Self {
            kind,
            message,
            cuda_code,
            ascend_code,
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
    fn worker_device_error_crosses_threads_and_cleanup_does_not_replace_it() {
        let error = std::thread::spawn(|| {
            record_ascend(507015);
            EngineError::take("rank failed".into(), true)
        })
        .join()
        .unwrap();
        let message = forward(error);
        record_cuda(700);
        let received = EngineError::take(message, false);
        assert_eq!(received.kind, ErrorKind::Ascend);
        assert_eq!(received.ascend_code, Some(507015));
        assert!(received.is_fatal());
        assert_eq!(
            EngineError::take("next".into(), false).kind,
            ErrorKind::Request
        );
    }
    #[test]
    fn ascend_failures_retire_workers_and_keep_the_initiating_code() {
        record_ascend(507015);
        record_ascend(207001);
        let error = EngineError::take("opaque message".into(), false);
        assert_eq!(error.kind, ErrorKind::Ascend);
        assert_eq!(error.ascend_code, Some(507015));
        assert!(error.is_fatal());
        assert_eq!(
            EngineError::take("next".into(), false).kind,
            ErrorKind::Request
        );
        record_ascend(207001);
        assert_eq!(
            EngineError::take("allocation".into(), false).kind,
            ErrorKind::OutOfMemory
        );
    }
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
