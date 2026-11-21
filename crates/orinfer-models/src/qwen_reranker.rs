//! Qwen3 causal reranking: complete sequences and a two-row final-position head.
use crate::{
    CreateRequest,
    adapter::Adapter,
    artifact::{Dtype, Result},
    encoder::{
        check_e8p_q4_quant, check_fields, check_signature, finalize, kernel, require_matrix,
        require_weight, transformer_ops,
    },
};
use orinfer_model_sdk::{architecture::BatchSegment, packet};
use serde_json::{Value, json};
use std::collections::BTreeMap;

struct Reranker;
impl Adapter for Reranker {
    fn batch(&self, _: &[BatchSegment], _: bool) -> Result<packet::BatchPlan> {
        Err("Reranking executes independent complete causal sequences".into())
    }
}

pub(crate) fn create(request: CreateRequest) -> Result<crate::registry::Built> {
    if request.architecture != "qwen3_reranker" || !request.prefill_profiles.is_empty() {
        return Err("Unsupported Qwen3 reranker execution contract".into());
    }
    let config = &request.config;
    check_signature(config, &request.expected_signature, "Reranker")?;
    check_fields(
        config,
        &[
            ("model_type", json!("qwen3")),
            ("architectures", json!(["Qwen3ForCausalLM"])),
            ("hidden_size", json!(1024)),
            ("intermediate_size", json!(3072)),
            ("num_hidden_layers", json!(28)),
            ("num_attention_heads", json!(16)),
            ("num_key_value_heads", json!(8)),
            ("head_dim", json!(128)),
            ("vocab_size", json!(151669)),
            ("rms_norm_eps", json!(1e-6)),
            ("hidden_act", json!("silu")),
            ("attention_bias", json!(false)),
            ("tie_word_embeddings", json!(true)),
            ("rope_theta", json!(1000000)),
            ("rope_scaling", Value::Null),
            ("sliding_window", Value::Null),
            ("use_sliding_window", json!(false)),
        ],
        "Qwen3 reranker",
    )?;
    let quant = &config["quantization_config"];
    check_e8p_q4_quant(
        quant,
        "qwen3-reranker-e8p-q4a8-v1",
        &[("classifier", json!("fp16-original-two-rows"))],
        "reranker",
    )?;
    let protected = quant.get("rotated_w8_layers").cloned().unwrap_or(json!([]));
    if protected != json!([])
        && protected != json!([0, 1, 2, 3])
        && protected != json!([0, 1, 2, 3, 8])
    {
        return Err("Unsupported Qwen3 reranker protected layers".into());
    }
    let protected: Vec<usize> = serde_json::from_value(protected).map_err(|e| e.to_string())?;
    let m = request.metadata;
    let spec = m
        .reranking
        .as_ref()
        .ok_or("Missing reranker scoring contract")?;
    if m.input != "Input"
        || m.vocab != 151669
        || !(512..=32768).contains(&m.chunk_tokens)
        || m.max_context != m.chunk_tokens
        || spec.program != "rerank"
        || spec.length != "Length"
        || spec.output != "ClassLogits"
        || quant["classifier_token_ids"] != json!([spec.false_token_id, spec.true_token_id])
    {
        return Err("Invalid reranker input or scoring roles".into());
    }
    let packed_rows = spec.rows.is_some();
    let buffers = &m.buffers;
    require_matrix(
        buffers,
        "model.embed_tokens.weight",
        151669,
        1024,
        false,
        "reranker",
    )?;
    require_weight(
        buffers,
        "orinfer.reranker.classifier.weight",
        Dtype::F16,
        &[2, 1024],
        "native_contiguous",
        "reranker",
    )?;
    require_weight(
        buffers,
        "Runtime.model.norm.weight.float16",
        Dtype::F16,
        &[1024],
        "native_contiguous",
        "reranker",
    )?;
    for layer in 0..28 {
        let rotated_w8 = protected.contains(&layer);
        let root = format!("model.layers.{layer}");
        let attention = if packed_rows {
            vec![("qkv", 4096, 1024, false), ("o", 1024, 2048, true)]
        } else {
            vec![
                ("q", 2048, 1024, false),
                ("k", 1024, 1024, false),
                ("v", 1024, 1024, false),
                ("o", 1024, 2048, true),
            ]
        };
        for (role, n, k, w8) in attention {
            require_matrix(
                buffers,
                &format!("{root}.self_attn.{role}_proj.weight"),
                n,
                k,
                w8 || rotated_w8,
                "reranker",
            )?;
        }
        let ffn = if packed_rows {
            vec![("gate_up", 6144, 1024), ("down", 1024, 3072)]
        } else {
            vec![
                ("gate", 3072, 1024),
                ("up", 3072, 1024),
                ("down", 1024, 3072),
            ]
        };
        for (role, n, k) in ffn {
            require_matrix(
                buffers,
                &format!("{root}.mlp.{role}_proj.weight"),
                n,
                k,
                rotated_w8,
                "reranker",
            )?;
        }
        for role in ["input_layernorm", "post_attention_layernorm"] {
            require_weight(
                buffers,
                &format!("Runtime.{root}.{role}.weight.float16"),
                Dtype::F16,
                &[1024],
                "native_contiguous",
                "reranker",
            )?;
        }
        for role in ["q_norm", "k_norm"] {
            require_weight(
                buffers,
                &format!("Runtime.{root}.self_attn.{role}.weight.float16"),
                Dtype::F16,
                &[128],
                "native_contiguous",
                "reranker",
            )?;
        }
    }
    let mut program = vec![kernel("qwen_reranker", "embedding")];
    program.extend(transformer_ops("qwen_reranker", 28, packed_rows, "qk_rope"));
    program.extend([
        kernel("qwen_reranker", "final_norm"),
        kernel("qwen_reranker", "classifier"),
    ]);
    let capacity = m.chunk_tokens;
    let (adapter, built) = finalize(
        Box::new(Reranker),
        m,
        BTreeMap::from([("rerank".into(), program)]),
        capacity,
        Default::default(),
    )?;
    built
        .metadata
        .reranking
        .as_ref()
        .unwrap()
        .validate(&built.metadata)?;
    Ok((adapter, built))
}
