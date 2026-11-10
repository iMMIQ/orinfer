//! Serving policy is distinct from fixed-seed numerical evaluation fixtures.
use super::*;
use crate::server::preparation::Context;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

fn codec(deepseek: bool) -> ChatCodec {
    configured_codec(
        json!({"temperature":0,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}),
        if deepseek {
            json!({"chat_format":"deepseek_v41"})
        } else {
            json!({})
        },
        Value::Null,
        "{{ messages[0].content }}",
    )
}

fn configured_codec(
    mut generation: Value,
    metadata: Value,
    config: Value,
    template: &str,
) -> ChatCodec {
    let directory = std::env::temp_dir().join(format!(
        "orinfer-defaults-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(directory.join("cache")).unwrap();
    let model = WordLevel::builder()
        .vocab(
            [
                "[UNK]", "one", "off", "on", "xhigh", "medium", "low", "kept", "dropped",
            ]
            .into_iter()
            .enumerate()
            .map(|(id, token)| (token.to_string(), id as u32))
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    tokenizer
        .save(directory.join("tokenizer.json"), false)
        .unwrap();
    generation
        .as_object_mut()
        .unwrap()
        .insert("eos_token_id".into(), json!(0));
    if !config.is_null() {
        std::fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        directory.join("generation_config.json"),
        serde_json::to_vec(&generation).unwrap(),
    )
    .unwrap();
    std::fs::write(directory.join("chat_template.jinja"), template).unwrap();
    std::fs::write(
        directory.join("cache/model.json"),
        serde_json::to_vec(&json!({
            "frontend_assets":{},
            "metadata": metadata
        }))
        .unwrap(),
    )
    .unwrap();
    let codec = ChatCodec::load(&directory).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    codec
}

fn prepare(codec: &ChatCodec, extra: Value, context: usize) -> Result<Prepared> {
    let mut body = json!({"model":"test","messages":[{"role":"user","content":"one"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    codec.prepare(
        serde_json::from_value(body).unwrap(),
        "test",
        context,
        None,
        &mut Context::unbounded(),
    )
}

#[test]
fn deepinfer_sampling_overrides_legacy_assets_and_preserves_explicit_controls() {
    let codec = codec(true);
    let prepared = prepare(&codec, json!({}), 10_000).unwrap();
    assert_eq!(prepared.sampling.temperature, 1.0);
    assert_eq!(prepared.sampling.top_p, 1.0);
    assert_eq!(prepared.sampling.top_k, 0);
    assert_eq!(prepared.sampling.repetition_penalty, 1.0);
    assert!(!prepared.ignore_eos);
    assert!(!prepared.thinking);
    assert_eq!(prepared.max_tokens, 8192);
    assert_eq!(prepare(&codec, json!({}), 100).unwrap().max_tokens, 99);
    assert!(prepare(&codec, json!({"max_tokens":100}), 100).is_err());
    let explicit = prepare(
        &codec,
        json!({
            "temperature":0,"top_p":0.8,"top_k":7,"repetition_penalty":1.1,
            "seed":-1,"enable_thinking":true,"max_tokens":32
        }),
        100,
    )
    .unwrap();
    assert_eq!(explicit.sampling.temperature, 0.0);
    assert_eq!(explicit.sampling.top_p, 0.8);
    assert_eq!(explicit.sampling.top_k, 7);
    assert_eq!(explicit.sampling.repetition_penalty, 1.1);
    assert_eq!(explicit.sampling.seed, u64::MAX);
    assert!(explicit.thinking);
    assert_eq!(explicit.max_tokens, 32);
    assert!(
        !prepare(&codec, json!({"reasoning_effort":"none"}), 100)
            .unwrap()
            .thinking
    );
}

#[test]
fn deepseek_output_defaults_follow_thinking_effort_and_remaining_context() {
    let generic = codec(false);
    let codec = codec(true);
    for (extra, expected) in [
        (json!({}), 8192),
        (json!({"enable_thinking":true}), 65536),
        (json!({"reasoning_effort":"minimal"}), 65536),
        (json!({"reasoning_effort":"low"}), 65536),
        (json!({"reasoning_effort":"medium"}), 65536),
        (json!({"reasoning_effort":"high"}), 65536),
        (json!({"reasoning_effort":"xhigh"}), 65536),
        (json!({"reasoning_effort":"max"}), 131072),
        (json!({"reasoning_effort":"none"}), 8192),
        (
            json!({"enable_thinking":false,"reasoning_effort":"max"}),
            8192,
        ),
        (
            json!({"chat_template_kwargs":{"thinking":true,"reasoning_effort":"max"}}),
            131072,
        ),
        (json!({"enable_thinking":true,"max_tokens":null}), 65536),
        (
            json!({"reasoning_effort":"max","max_completion_tokens":null}),
            131072,
        ),
    ] {
        let full = prepare(&codec, extra.clone(), 262144).unwrap();
        assert_eq!(full.max_tokens, expected, "{extra}");
        let short = prepare(&codec, extra.clone(), 4096).unwrap();
        assert_eq!(short.max_tokens, 4096 - short.input.len(), "{extra}");
        assert!(prepare(&codec, extra, 1).is_err());
    }
    for limit in ["max_tokens", "max_completion_tokens"] {
        let explicit = json!({"reasoning_effort":"max",limit:64});
        assert_eq!(
            prepare(&codec, explicit.clone(), 100).unwrap().max_tokens,
            64
        );
        assert!(prepare(&codec, explicit, 64).is_err());
    }
    // A request's effort does not change another model family's output policy.
    assert_eq!(
        prepare(&generic, json!({"reasoning_effort":"max"}), 262144)
            .unwrap()
            .max_tokens,
        8192
    );
}

#[test]
fn deployment_budget_and_probability_admission_share_the_resolved_default() {
    let codec = codec(true);
    for (defaults, expected) in [
        (json!({"enable_thinking":true}), 65536),
        (
            json!({"chat_template_kwargs":{"reasoning_effort":"max"}}),
            131072,
        ),
        (
            json!({"reasoning_effort":"max","max_completion_tokens":4096}),
            4096,
        ),
    ] {
        let request =
            crate::server::defaults::validation_request(defaults.as_object().unwrap()).unwrap();
        assert_eq!(codec.max_tokens(&request), expected);
        let prepared = codec
            .prepare(
                request,
                "defaults-validation",
                262144,
                None,
                &mut Context::unbounded(),
            )
            .unwrap();
        assert_eq!(prepared.max_tokens, expected);
    }
    let defaults = json!({"reasoning_effort":"max","max_completion_tokens":4096});
    let request = crate::server::defaults::decode(
        json!({"model":"test","messages":[{"role":"user","content":"one"}],"max_tokens":32}),
        defaults.as_object().unwrap(),
    )
    .unwrap();
    assert_eq!(codec.max_tokens(&request), 32);

    let mut request = crate::server::defaults::validation_request(
        json!({"reasoning_effort":"max","logprobs":true,"top_logprobs":20})
            .as_object()
            .unwrap(),
    )
    .unwrap();
    let output = codec.max_tokens(&request);
    let memory = crate::server::preparation::request_memory_mib(&request, 262144, output);
    request.max_tokens = Some(output);
    assert_eq!(
        memory,
        crate::server::preparation::request_memory_mib(&request, 262144, 8192)
    );
    request.max_tokens = Some(8192);
    assert!(memory > crate::server::preparation::request_memory_mib(&request, 262144, 8192));
    request.max_tokens = None;
    request.stream = true;
    assert!(memory > crate::server::preparation::request_memory_mib(&request, 262144, output));
}

#[test]
fn omitted_seed_is_fresh_and_explicit_seed_is_stable() {
    let codec = codec(false);
    let seeds = (0..16)
        .map(|_| prepare(&codec, json!({}), 100).unwrap().sampling.seed)
        .collect::<BTreeSet<_>>();
    assert_eq!(seeds.len(), 16);
    for seed in [0, 20_261_002, -1, i64::MIN, i64::MAX] {
        let a = prepare(&codec, json!({"seed":seed}), 100).unwrap();
        let b = prepare(&codec, json!({"seed":seed}), 100).unwrap();
        assert_eq!(a.sampling.seed, seed as u64);
        assert_eq!(a.sampling.seed, b.sampling.seed);
    }
    let prepared = prepare(&codec, json!({}), 100).unwrap();
    assert_eq!(prepared.sampling.temperature, 0.0);
    assert_eq!(prepared.sampling.top_p, 0.95);
    assert_eq!(prepared.sampling.top_k, 20);
    assert_eq!(prepared.sampling.repetition_penalty, 1.05);
}

#[test]
fn deepinfer_top_k_alias_and_invalid_values() {
    let codec = codec(true);
    for value in [-1, 0, 1, i64::from(i32::MAX)] {
        let prepared = prepare(&codec, json!({"top_k":value}), 100).unwrap();
        assert_eq!(prepared.sampling.top_k, value.max(0) as usize);
    }
    for value in [-2, i64::from(i32::MAX) + 1] {
        assert!(prepare(&codec, json!({"top_k":value}), 100).is_err());
    }
}

const MODE_TEMPLATE: &str = "{{ messages[0].content }} {% if enable_thinking %}on{% else %}off{% endif %} {{ reasoning_effort }} {% if preserve_thinking %}kept{% else %}dropped{% endif %}";

#[test]
fn qwen_defaults_and_mode_switches_reach_sampling_and_template() {
    for (source, presence) in [
        ("Qwen3.8-27B-Q4A8", 0.0),
        ("Qwen3.8-Flash-Next", 0.0),
        ("Qwen/Qwen3.5-4B-e8p_q4-A8", 1.5),
    ] {
        let codec = configured_codec(
            json!({}),
            json!({"model":source}),
            Value::Null,
            MODE_TEMPLATE,
        );
        let default = prepare(&codec, json!({}), 10_000).unwrap();
        assert!(default.thinking, "{source}");
        assert_eq!(default.input, [1, 3, 4, 7]);
        assert_eq!(default.sampling.temperature, 1.0);
        assert_eq!(default.sampling.top_p, 0.95);
        assert_eq!(default.sampling.top_k, 20);
        assert_eq!(default.sampling.presence_penalty, presence);
        assert_eq!(default.sampling.repetition_penalty, 1.0);
        assert_eq!(default.max_tokens, 8192);
        for control in [
            json!({"enable_thinking":false}),
            json!({"reasoning_effort":"none"}),
            json!({"chat_template_kwargs":{"enable_thinking":false}}),
            json!({"chat_template_kwargs":{"thinking":false}}),
        ] {
            let off = prepare(&codec, control, 10_000).unwrap();
            assert!(!off.thinking);
            assert_eq!(off.input, [1, 2, 4, 7]);
            assert_eq!(off.sampling.temperature, 0.7);
            assert_eq!(off.sampling.top_p, 0.8);
            assert_eq!(off.sampling.top_k, 20);
            assert_eq!(off.sampling.presence_penalty, 1.5);
        }
        let medium = prepare(&codec, json!({"reasoning_effort":"medium"}), 100).unwrap();
        assert!(medium.thinking);
        assert_eq!(medium.input, [1, 3, 5, 7]);
        assert_eq!(medium.sampling.presence_penalty, presence);
        assert!(
            prepare(
                &codec,
                json!({"enable_thinking":true,"reasoning_effort":"none"}),
                100
            )
            .is_err()
        );
    }
}

#[test]
fn source_identity_and_explicit_controls_override_model_defaults() {
    let codec = configured_codec(
        json!({"temperature":0.9,"top_p":0.92,"top_k":17,"repetition_penalty":1.1}),
        json!({"model":"unrelated-deployment-name"}),
        json!({"quantization_config":{"source":"Qwen/Qwen3.8-27B"}}),
        MODE_TEMPLATE,
    );
    let inherited = prepare(&codec, json!({}), 100).unwrap();
    assert!(inherited.thinking);
    assert_eq!(inherited.sampling.temperature, 0.9);
    assert_eq!(inherited.sampling.top_p, 0.92);
    assert_eq!(inherited.sampling.top_k, 17);
    assert_eq!(inherited.sampling.repetition_penalty, 1.1);
    let body = json!({"model":"test","messages":[{"role":"user","content":"one"}],
        "chat_template_kwargs":{"enable_thinking":false},"temperature":0,
        "presence_penalty":0,"max_tokens":32});
    let defaults = json!({"enable_thinking":true,"temperature":0.5,"top_p":0.6,"top_k":7,
        "presence_penalty":1,"max_completion_tokens":64});
    let request = crate::server::defaults::decode(body, defaults.as_object().unwrap()).unwrap();
    let prepared = codec
        .prepare(request, "test", 100, None, &mut Context::unbounded())
        .unwrap();
    assert!(!prepared.thinking);
    assert_eq!(prepared.input, [1, 2, 4, 7]);
    assert_eq!(prepared.sampling.temperature, 0.0);
    assert_eq!(prepared.sampling.top_p, 0.6);
    assert_eq!(prepared.sampling.top_k, 7);
    assert_eq!(prepared.sampling.presence_penalty, 0.0);
    assert_eq!(prepared.max_tokens, 32);
}

#[test]
fn index_defaults_are_greedy_for_all_sizes_with_bounded_output() {
    for source in [
        "IndexTeam/Index-Translate-2B-e8p_q4-A8",
        "IndexTeam/Index-Translate-9B-e8p_q4-A8",
        "IndexTeam/Index-Translate-35B-A3B-preview-e8p_q4-A8",
    ] {
        let codec = configured_codec(
            json!({}),
            json!({"model":source}),
            Value::Null,
            "{{ messages[0].content }}",
        );
        let default = prepare(&codec, json!({}), 10_000).unwrap();
        assert!(!default.thinking);
        assert!(default.sampling.is_greedy());
        assert_eq!(default.sampling.top_p, 1.0);
        assert_eq!(default.sampling.top_k, 0);
        assert_eq!(default.max_tokens, 1024);
        assert_eq!(prepare(&codec, json!({}), 100).unwrap().max_tokens, 99);
        let explicit = prepare(
            &codec,
            json!({"temperature":0.8,"max_completion_tokens":2048}),
            10_000,
        )
        .unwrap();
        assert_eq!(explicit.sampling.temperature, 0.8);
        assert_eq!(explicit.max_tokens, 2048);
        assert!(prepare(&codec, json!({"max_tokens":100}), 100).is_err());
    }
}

#[test]
fn configured_penalties_and_greedy_semantics_are_inherited() {
    let codec = configured_codec(
        json!({"do_sample":false,"temperature":0.000001,"presence_penalty":0.3,"frequency_penalty":0.2}),
        json!({"model":"Qwen/Qwen3-ASR-1.7B-hf-e8p_q4-A8"}),
        Value::Null,
        "{{ messages[0].content }}",
    );
    let default = prepare(&codec, json!({}), 10_000).unwrap();
    assert!(!default.thinking);
    assert_eq!(default.sampling.temperature, 0.0);
    assert_eq!(default.sampling.presence_penalty, 0.3);
    assert_eq!(default.sampling.frequency_penalty, 0.2);
    assert_eq!(default.max_tokens, 8192);
    let explicit = prepare(
        &codec,
        json!({"temperature":0.5,"presence_penalty":0,"frequency_penalty":0}),
        100,
    )
    .unwrap();
    assert_eq!(explicit.sampling.temperature, 0.5);
    assert_eq!(explicit.sampling.presence_penalty, 0.0);
    assert_eq!(explicit.sampling.frequency_penalty, 0.0);
}
