//! Serving policy is distinct from fixed-seed numerical evaluation fixtures.
use super::*;
use crate::server::preparation::Context;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};

static SERIAL: AtomicUsize = AtomicUsize::new(0);

fn codec(deepseek: bool) -> ChatCodec {
    let directory = std::env::temp_dir().join(format!(
        "orinfer-defaults-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(directory.join("cache")).unwrap();
    let model = WordLevel::builder()
        .vocab(
            [("[UNK]".into(), 0), ("one".into(), 1)]
                .into_iter()
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
    std::fs::write(
        directory.join("generation_config.json"),
        r#"{"eos_token_id":0,"temperature":0,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}"#,
    )
    .unwrap();
    std::fs::write(
        directory.join("chat_template.jinja"),
        "{{ messages[0].content }}",
    )
    .unwrap();
    std::fs::write(
        directory.join("cache/model.json"),
        serde_json::to_vec(&json!({
            "frontend_assets":{},
            "metadata": if deepseek {json!({"chat_format":"deepseek_v41"})} else {json!({})}
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
