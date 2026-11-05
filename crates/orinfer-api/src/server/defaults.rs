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
    decode(Value::Object(request), &Map::new()).map_err(|e| format!("default_request_params: {e}"))
}

pub(super) fn decode(
    mut body: Value,
    defaults: &Map<String, Value>,
) -> Result<ChatRequest, String> {
    merge(&mut body, defaults)?;
    let mut request: ChatRequest = serde_path_to_error::deserialize(body).map_err(|e| {
        let path = e.path().to_string();
        if path == "." {
            e.inner().to_string()
        } else {
            format!("{path}: {}", e.inner())
        }
    })?;
    request.resolve_template_options()?;
    Ok(request)
}

pub(crate) fn merge(body: &mut Value, defaults: &Map<String, Value>) -> Result<(), String> {
    let request = body
        .as_object_mut()
        .ok_or("Expected a JSON request object")?;
    // Related controls are one override group: an explicit alias must not
    // conflict with, or be defeated by, a deployment default for its partner.
    let output_override =
        request.contains_key("max_tokens") || request.contains_key("max_completion_tokens");
    let kwargs = request
        .get("chat_template_kwargs")
        .and_then(Value::as_object);
    let thinking_override = request.contains_key("enable_thinking")
        || request.contains_key("reasoning_effort")
        || kwargs.is_some_and(|k| {
            super::chat::THINKING_FIELDS
                .iter()
                .any(|name| k.contains_key(*name))
        });
    let budget_override = request.contains_key("thinking_token_budget")
        || kwargs.is_some_and(|k| k.contains_key("thinking_token_budget"));
    let stream_disabled = request
        .get("stream")
        .is_some_and(|v| v == false || v.is_null());
    let logprobs_disabled = request
        .get("logprobs")
        .is_some_and(|v| v == false || v.is_null());
    for (name, value) in defaults {
        if name == "chat_template_kwargs" {
            // Merge variables individually; an explicit null clears only this
            // map. Request thinking aliases suppress defaults in either form.
            let Some(defaults) = value.as_object() else {
                request.entry(name.clone()).or_insert_with(|| value.clone());
                continue;
            };
            let Some(kwargs) = request
                .entry(name.clone())
                .or_insert_with(|| json!({}))
                .as_object_mut()
            else {
                continue;
            };
            for (key, value) in defaults {
                if thinking_override && super::chat::THINKING_FIELDS.contains(&key.as_str())
                    || budget_override && key == "thinking_token_budget"
                {
                    continue;
                }
                kwargs.entry(key.clone()).or_insert_with(|| value.clone());
            }
            continue;
        }
        if request.contains_key(name)
            || output_override && matches!(name.as_str(), "max_tokens" | "max_completion_tokens")
            || thinking_override && matches!(name.as_str(), "enable_thinking" | "reasoning_effort")
            || budget_override && name == "thinking_token_budget"
            || stream_disabled && name == "stream_options"
            || logprobs_disabled && name == "top_logprobs"
        {
            continue;
        }
        request.insert(name.clone(), value.clone());
    }
    Ok(())
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
    fn dsh_template_thinking_controls_override_defaults_in_both_directions() {
        let r = request(
            json!({"store":false,"chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":true},"stream":true,"stream_options":{"include_usage":true}}),
            json!({"enable_thinking":true,"reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":true,"custom":"default"}}),
        );
        assert_eq!(r.enable_thinking, Some(false));
        assert_eq!(r.reasoning_effort, None);
        assert_eq!(r.chat_template_kwargs["preserve_thinking"], true);
        assert_eq!(r.chat_template_kwargs["custom"], "default");
        let r = request(
            json!({"enable_thinking":false,"chat_template_kwargs":{"custom":"request"}}),
            json!({"chat_template_kwargs":{"thinking":true,"reasoning_effort":"high","custom":"default","preserve_thinking":true}}),
        );
        assert_eq!(r.enable_thinking, Some(false));
        assert_eq!(r.reasoning_effort, None);
        assert_eq!(r.chat_template_kwargs["custom"], "request");
        assert_eq!(r.chat_template_kwargs["preserve_thinking"], true);
        assert!(!r.chat_template_kwargs.contains_key("thinking"));
        let r = request(
            json!({"chat_template_kwargs":null}),
            json!({"chat_template_kwargs":{"enable_thinking":true}}),
        );
        assert!(r.chat_template_kwargs.is_empty());
        assert_eq!(r.enable_thinking, None);
        let r = request(
            json!({"chat_template_kwargs":{"thinking":true,"thinking_token_budget":32}}),
            json!({"enable_thinking":false,"thinking_token_budget":512}),
        );
        assert_eq!(r.enable_thinking, Some(true));
        assert_eq!(r.thinking_token_budget, Some(32));
        for kwargs in [
            json!([]),
            json!({"enable_thinking":"false"}),
            json!({"preserve_thinking":1}),
            json!({"enable_thinking":true,"thinking":false}),
        ] {
            assert!(
                decode(
                    json!({"model":"test","messages":[],"chat_template_kwargs":kwargs}),
                    &Map::new()
                )
                .is_err()
            );
        }
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
