//! Request deadlines are recoverable outcomes, separate from worker failures.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum Deadline {
    Queue,
    Output,
}
impl Deadline {
    pub fn message(self) -> &'static str {
        match self {
            Self::Queue => "Request queue deadline exceeded",
            Self::Output => "Client output deadline exceeded",
        }
    }
    pub fn payload(self) -> Value {
        let (code, retryable) = match self {
            Self::Queue => ("request_queue_timeout", true),
            Self::Output => ("client_output_timeout", false),
        };
        let mut value = json!({"error": {
            "message": self.message(), "type": "timeout_error", "code": code,
            "param": null, "retryable": retryable,
        }});
        if retryable {
            value["error"]["retry_after_ms"] = json!(1000);
        }
        value
    }
    pub fn response(self) -> Response {
        let mut response = (StatusCode::REQUEST_TIMEOUT, Json(self.payload())).into_response();
        if matches!(self, Self::Queue) {
            response
                .headers_mut()
                .insert(axum::http::header::RETRY_AFTER, "1".parse().unwrap());
        }
        response
    }
}
