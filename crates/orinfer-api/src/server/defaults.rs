//! Deployment defaults are merged before deserialization so false, null and
//! empty values remain explicit request overrides.
use super::chat::ChatRequest;
use serde_json::{Map, Value, json};

pub(crate) fn validation_request(defaults: &Map<String, Value>) -> Result<ChatRequest, String> {
    for name in ["model", "messages"] {
        if defaults.contains_key(name) {
            return Err(format!(
                "default_request_params.{name}: Must be supplied by the client"
            ));
        }
    }
    let mut request = defaults.clone();
    request.insert("model".into(), json!("defaults-validation"));
    request.insert(
        "messages".into(),
        json!([{"role":"user","content":"Hello"}]),
    );
    serde_json::from_value(Value::Object(request))
        .map_err(|e| format!("default_request_params: {e}"))
}

pub(super) fn decode(
    mut body: Value,
    defaults: &Map<String, Value>,
) -> Result<ChatRequest, String> {
    let request = body
        .as_object_mut()
        .ok_or("Expected a JSON request object")?;
    // Related controls are one override group: an explicit alias must not
    // conflict with, or be defeated by, a deployment default for its partner.
    let output_override =
        request.contains_key("max_tokens") || request.contains_key("max_completion_tokens");
    let thinking_override =
        request.contains_key("enable_thinking") || request.contains_key("reasoning_effort");
    let stream_disabled = request
        .get("stream")
        .is_some_and(|v| v == false || v.is_null());
    let logprobs_disabled = request
        .get("logprobs")
        .is_some_and(|v| v == false || v.is_null());
    for (name, value) in defaults {
        if request.contains_key(name)
            || output_override && matches!(name.as_str(), "max_tokens" | "max_completion_tokens")
            || thinking_override && matches!(name.as_str(), "enable_thinking" | "reasoning_effort")
            || stream_disabled && name == "stream_options"
            || logprobs_disabled && name == "top_logprobs"
        {
            continue;
        }
        request.insert(name.clone(), value.clone());
    }
    serde_path_to_error::deserialize(body).map_err(|e| {
        let path = e.path().to_string();
        if path == "." {
            e.inner().to_string()
        } else {
            format!("{path}: {}", e.inner())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: Value, defaults: Value) -> ChatRequest {
        let mut body = json!({"model":"test","messages":[{"role":"user","content":"Hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        decode(body, defaults.as_object().unwrap()).unwrap()
    }

    #[test]
    fn defaults_only_fill_missing_fields() {
        let defaults =
            json!({"enable_thinking":true,"temperature":0.7,"stream":true,"stop":["END"]});
        let r = request(json!({}), defaults.clone());
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.temperature, Some(0.7));
        assert!(r.stream);
        let r = request(
            json!({"enable_thinking":false,"temperature":0,"stream":false,"stop":[]}),
            defaults.clone(),
        );
        assert_eq!(r.enable_thinking, Some(false));
        assert_eq!(r.temperature, Some(0.0));
        assert!(!r.stream);
        assert_eq!(r.stop, Some(json!([])));
        let r = request(
            json!({"enable_thinking":null,"temperature":null,"stream":null}),
            defaults,
        );
        assert_eq!(r.enable_thinking, None);
        assert_eq!(r.temperature, None);
        assert!(!r.stream);
        assert_eq!(request(json!({}), json!({})).enable_thinking, None);
    }

    #[test]
    fn related_controls_override_together_without_masking_request_conflicts() {
        let r = request(
            json!({"reasoning_effort":"none","max_tokens":32,"stream":false,"logprobs":false}),
            json!({"enable_thinking":true,"max_completion_tokens":8192,"stream":true,"stream_options":{"include_usage":true},"logprobs":true,"top_logprobs":3}),
        );
        assert_eq!(r.enable_thinking, None);
        assert_eq!(r.reasoning_effort.as_deref(), Some("none"));
        assert_eq!(r.max_tokens, Some(32));
        assert_eq!(r.max_completion_tokens, None);
        assert_eq!(r.stream_options, None);
        assert_eq!(r.top_logprobs, None);
        let r = request(
            json!({"enable_thinking":true,"reasoning_effort":"none","max_tokens":4,"max_completion_tokens":8}),
            json!({"enable_thinking":false}),
        );
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.reasoning_effort.as_deref(), Some("none"));
        assert_eq!(r.max_tokens, Some(4));
        assert_eq!(r.max_completion_tokens, Some(8));
        let r = request(
            json!({"enable_thinking":false,"max_completion_tokens":64}),
            json!({"reasoning_effort":"high","max_tokens":8192}),
        );
        assert_eq!(r.reasoning_effort, None);
        assert_eq!(r.max_tokens, None);
        assert_eq!(r.max_completion_tokens, Some(64));
    }

    #[test]
    fn invalid_defaults_and_requests_are_rejected() {
        for defaults in [
            json!({"model":"x"}),
            json!({"messages":[]}),
            json!({"unknown":true}),
            json!({"enable_thinking":"true"}),
        ] {
            assert!(validation_request(defaults.as_object().unwrap()).is_err());
        }
        assert!(decode(json!([]), &Map::new()).is_err());
        assert!(
            decode(
                json!({"model":"x","messages":[],"unknown":true}),
                &Map::new()
            )
            .is_err()
        );
        assert!(decode(json!({"messages":[]}), &Map::new()).is_err());
        let error = decode(
            json!({"model":"x","messages":[],"temperature":"invalid"}),
            &Map::new(),
        )
        .unwrap_err();
        assert!(error.starts_with("temperature: "), "{error}");
    }
}
