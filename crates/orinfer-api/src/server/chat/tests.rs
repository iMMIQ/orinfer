use super::*;
#[test]
fn ollama_tool_history_vectors_preserve_text_and_call_identity() {
    for case in crate::test_fixtures::ollama()["requests"]
        .as_array()
        .unwrap()
    {
        let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
        let messages = normalize_messages(request.messages).unwrap();
        assert_eq!(json!(messages), case["expected_messages"], "{}", case["id"]);
    }
}
#[test]
fn template_json_matches_python_separators_unicode_and_string_escaping() {
    let value = json!({"z":["世界", "a,b:c", "\"\\", "<>&'"],"a":{"b":true},"empty":[]});
    let result = template_json(minijinja::Value::from(Serde(&value))).unwrap();
    assert_eq!(
        result.as_str().unwrap(),
        r#"{"a": {"b": true}, "empty": [], "z": ["世界", "a,b:c", "\"\\", "<>&'"]}"#
    );
}
#[test]
fn template_python_methods_support_real_python_argument_shapes() {
    let mut environment = Environment::new();
    environment.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    assert_eq!(environment.render_str("{{ 'ABC'.lower() }} {{ 'abcdef'.startswith(['xy', 'ab']) }} {{ 'abc'.endswith('bc') }} {{ ' a '.strip() }}", minijinja::context!()).unwrap(), "abc True True a");
}
#[test]
fn nullable_optional_fields_and_signed_seed_match_the_wire_contract() {
    let request: ChatRequest=serde_json::from_value(json!({"model":"test","messages":[],"tools":null,"stream":null,"seed":-1,"store":false,"metadata":{"case":"test"},"logprobs":true,"top_logprobs":3,"logit_bias":{"42":-100}})).unwrap();
    assert!(!request.stream && request.tools.is_empty());
    assert_eq!(request.seed, Some(-1));
    assert_eq!(request.logit_bias.unwrap()[&42], -100.0);
    assert!(
        serde_json::from_value::<ChatRequest>(
            json!({"model":"test","messages":[],"stream":"true"})
        )
        .is_err()
    );
}
#[test]
fn saved_calibration_truncation_does_not_discard_chat_history() {
    use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
    let directory = std::env::temp_dir().join(format!(
        "orin-chat-truncation-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("one".into(), 1),
                ("two".into(), 2),
                ("three".into(), 3),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 2,
            ..Default::default()
        }))
        .unwrap();
    assert_eq!(tokenizer.encode("one two three", false).unwrap().len(), 2);
    tokenizer
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        directory.join("generation_config.json"),
        r#"{"eos_token_id":0,"temperature":0.7,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}"#,
    )
    .unwrap();
    std::fs::write(
        directory.join("chat_template.jinja"),
        "{{ messages[0].content }}",
    )
    .unwrap();
    let codec = ChatCodec::load(&directory).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    let request = || {
        serde_json::from_value(json!({
            "model":"test", "messages":[{"role":"user","content":"one two three"}],
            "max_tokens":1,"seed":20261002
        }))
        .unwrap()
    };
    assert_eq!(
        codec
            .prepare(
                request(),
                "test",
                4,
                None,
                &mut crate::server::preparation::Context::unbounded()
            )
            .unwrap()
            .input,
        [1, 2, 3]
    );
    assert!(
        codec
            .prepare(
                request(),
                "test",
                3,
                None,
                &mut crate::server::preparation::Context::unbounded()
            )
            .is_err()
    );
    let prepared = codec
        .prepare(
            request(),
            "test",
            4,
            None,
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
    for retention in ["in-memory", "24h"] {
        let mut cached = request();
        cached.prompt_cache_key = Some("session-key".into());
        cached.prompt_cache_retention = Some(retention.into());
        let result = codec
            .prepare(
                cached,
                "test",
                4,
                None,
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
        assert_eq!(result.input, prepared.input);
        assert_eq!(result.sampling.seed, prepared.sampling.seed);
    }
    for (key, retention, param) in [
        ("", "24h", "prompt_cache_key"),
        ("key", "forever", "prompt_cache_retention"),
    ] {
        let mut invalid = request();
        invalid.prompt_cache_key = Some(key.into());
        invalid.prompt_cache_retention = Some(retention.into());
        let error = codec
            .prepare(
                invalid,
                "test",
                4,
                None,
                &mut crate::server::preparation::Context::unbounded(),
            )
            .err()
            .unwrap();
        assert!(error.starts_with(param), "{error}");
    }
    assert_eq!(prepared.sampling.temperature, 0.7);
    assert_eq!(prepared.sampling.top_p, 0.95);
    assert_eq!(prepared.sampling.top_k, 20);
    assert_eq!(prepared.sampling.repetition_penalty, 1.05);
    let override_request = serde_json::from_value(json!({
        "model":"test", "messages":[{"role":"user","content":"one"}],
        "max_tokens":1,"temperature":0,"top_k":0,"top_p":1,
        "repetition_penalty":1
    }))
    .unwrap();
    assert!(
        codec
            .prepare(
                override_request,
                "test",
                4,
                None,
                &mut crate::server::preparation::Context::unbounded()
            )
            .unwrap()
            .sampling
            .is_greedy()
    );
}

#[test]
fn history_and_multimodal_validation() {
    let history = vec![
        json!({"role":"user","content":"read"}),
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"x\"}"}}]}),
        json!({"role":"tool","tool_call_id":"a","content":"data"}),
    ];
    let normalized = normalize_messages(history.clone()).unwrap();
    assert_eq!(
        normalized[1]["tool_calls"][0]["function"]["arguments"]["path"],
        "x"
    );
    assert!(normalize_messages(history[..2].to_vec()).is_err());
    for raw in ["{\"path\":", "[]", "null"] {
        let mut failed = history.clone();
        failed[1]["tool_calls"][0]["function"]["arguments"] = json!(raw);
        failed[2]["content"] = json!("Invalid arguments; retry with path.");
        let result = normalize_messages(failed).unwrap();
        assert_eq!(
            result[1]["tool_calls"][0]["function"]["arguments"]["__orinfer_raw_arguments"],
            raw
        );
    }
    assert!(
        normalize_messages(vec![
            json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]})
        ])
        .is_err()
    );
}
#[test]
fn leading_instruction_messages_are_combined_in_order() {
    let normalized = normalize_messages(vec![
        json!({"role":"system","content":"one"}),
        json!({"role":"developer","content":"two"}),
        json!({"role":"user","content":"query"}),
    ])
    .unwrap();
    assert_eq!(normalized.len(), 2);
    assert_eq!(normalized[0]["content"], "one\n\ntwo");
    assert!(
        normalize_messages(vec![
            json!({"role":"user","content":"q"}),
            json!({"role":"system","content":"late"})
        ])
        .is_err()
    );
}

#[test]
#[ignore = "Needs local checkpoint and Transformers-generated reference cases"]
fn checkpoint_matches_reference_tokens() {
    let directory = std::env::var("ORINFER_TOKENIZER_DIR").expect("ORINFER_TOKENIZER_DIR");
    let cases = std::env::var("ORINFER_CHAT_REFERENCE").expect("ORINFER_CHAT_REFERENCE");
    let codec = ChatCodec::load(Path::new(&directory)).unwrap();
    let cases: Value = serde_json::from_slice(&std::fs::read(cases).unwrap()).unwrap();
    for case in cases.as_array().unwrap() {
        let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
        let vision: Option<orinfer_engine::vision::VisionSpec> = case
            .get("vision")
            .map(|v| serde_json::from_value(v.clone()).unwrap());
        let model = request.model.clone();
        let prepared = codec
            .prepare(
                request,
                &model,
                case["context_limit"].as_u64().unwrap_or(8704) as usize,
                vision.as_ref(),
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            prepared.input, expected,
            "Checkpoint template mismatch: {}",
            case["request"]["messages"]
        );
        if let Some(expected) = case.get("mrope_positions") {
            let expected: Vec<u32> = serde_json::from_value(expected.clone()).unwrap();
            let (_, positions) = orinfer_engine::model::media_layout(
                Path::new(&directory),
                &prepared.input,
                &prepared.images,
                prepared.input.len() + 16,
            )
            .unwrap();
            assert_eq!(
                positions, expected,
                "Official multimodal positions mismatch"
            );
        }
    }
}
