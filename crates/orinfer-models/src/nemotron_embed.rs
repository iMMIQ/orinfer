//! Nemotron 3.5: independent bidirectional text and Pixtral embedding programs.
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

struct Embedding;
impl Adapter for Embedding {
    fn batch(&self, _: &[BatchSegment], _: bool) -> Result<packet::BatchPlan> {
        Err("Embedding requests execute complete bidirectional sequences independently".into())
    }
}

fn validate_config(config: &Value, signature: &Value) -> Result<()> {
    check_signature(config, signature, "Nemotron embedding")?;
    check_fields(
        config,
        &[
            ("model_type", json!("mistral3")),
            ("architectures", json!(["Mistral3Model"])),
            ("is_causal", json!(false)),
            ("pooling", json!("avg")),
            ("spatial_merge_size", json!(2)),
            ("image_token_index", json!(10)),
            ("projector_hidden_act", json!("gelu")),
            ("multimodal_projector_bias", json!(false)),
            ("vision_feature_layer", json!(-1)),
        ],
        "Nemotron",
    )?;
    check_fields(
        &config["text_config"],
        &[
            ("model_type", json!("ministral3")),
            ("hidden_size", json!(2048)),
            ("intermediate_size", json!(6144)),
            ("num_hidden_layers", json!(16)),
            ("num_attention_heads", json!(24)),
            ("num_key_value_heads", json!(8)),
            ("head_dim", json!(128)),
            ("vocab_size", json!(131072)),
            ("rms_norm_eps", json!(1e-5)),
            ("is_causal", json!(false)),
            ("pooling", json!("avg")),
            ("hidden_act", json!("silu")),
            ("sliding_window", Value::Null),
        ],
        "Ministral3",
    )?;
    check_fields(
        &config["vision_config"],
        &[
            ("model_type", json!("pixtral")),
            ("hidden_size", json!(1024)),
            ("intermediate_size", json!(4096)),
            ("num_hidden_layers", json!(24)),
            ("num_attention_heads", json!(16)),
            ("head_dim", json!(64)),
            ("patch_size", json!(14)),
        ],
        "Pixtral",
    )?;
    if config["vision_config"]["rope_parameters"]
        != json!({"rope_type":"default","rope_theta":10000.0})
    {
        return Err("Unsupported Pixtral axial RoPE".into());
    }
    check_fields(
        &config["text_config"]["rope_parameters"],
        &[
            ("rope_type", json!("yarn")),
            ("rope_theta", json!(1000000.0)),
            ("factor", json!(16.0)),
            ("original_max_position_embeddings", json!(16384)),
            ("beta_fast", json!(32.0)),
            ("beta_slow", json!(1.0)),
            ("apply_yarn_scaling", json!(false)),
            ("llama_4_scaling_beta", json!(0.1)),
        ],
        "Ministral3 RoPE",
    )?;
    check_e8p_q4_quant(
        &config["quantization_config"],
        "nemotron35-embed-e8p-q4a8-v1",
        &[
            ("basis_tensor", json!("orinfer.e8p_q4.basis")),
            ("vision", json!("e8p_q4")),
        ],
        "embedding",
    )
}

pub(crate) fn create(request: CreateRequest) -> Result<crate::registry::Built> {
    if request.architecture != "nemotron_embed" || !request.prefill_profiles.is_empty() {
        return Err("Unsupported Nemotron embedding execution contract".into());
    }
    validate_config(&request.config, &request.expected_signature)?;
    let protected = match request.config["quantization_config"].get("rotated_w8_layers") {
        None => false,
        Some(v) if v == &json!([]) => false,
        Some(v) if v == &json!([2]) => true,
        _ => return Err("Unsupported Nemotron rotated-W8 layer policy".into()),
    };
    let m = request.metadata;
    let spec = m
        .embedding
        .as_ref()
        .ok_or("Missing embedding output contract")?;
    for (actual, expected) in [
        (&spec.output, "Embedding"),
        (&spec.length, "Length"),
        (&spec.feature_index, "FeatureIndex"),
        (&spec.pixels, "Pixels"),
        (&spec.grid, "Grid"),
        (&spec.vision_length, "VisionLength"),
        (&spec.vision_output, "VisionOutput"),
        (&spec.features, "Features"),
    ] {
        if actual != expected {
            return Err("Embedding buffer roles differ from the native program".into());
        }
    }
    if m.input != "Input"
        || m.vocab != 131072
        || !(512..=32768).contains(&m.chunk_tokens)
        || m.max_context != m.chunk_tokens
        || spec.dimension != 2048
        || spec.program != "embed"
        || spec.vision_program != "embed_vision"
        || spec.max_patches > 12100
        || spec.max_patches % 4 != 0
        || spec.patch_size != 14
        || spec.merge_size != 2
        || spec.image_token_id != 10
    {
        return Err("Invalid Nemotron embedding dimensions".into());
    }
    let packed_rows = spec.rows.is_some();
    let buffers = &m.buffers;
    require_matrix(
        buffers,
        "language_model.embed_tokens.weight",
        131072,
        2048,
        false,
        "embedding",
    )?;
    for (root, layers, h, f, q, kv, vision) in [
        ("language_model.layers", 16, 2048, 6144, 3072, 1024, false),
        (
            "vision_tower.transformer.layers",
            24,
            1024,
            4096,
            1024,
            1024,
            true,
        ),
    ] {
        for layer in 0..layers {
            let rotated_w8 = protected && !vision && layer == 2;
            let prefix = format!("{root}.{layer}");
            let attn = if vision { "attention" } else { "self_attn" };
            let mlp = if vision { "feed_forward" } else { "mlp" };
            let attention = if packed_rows {
                vec![("qkv", q + 2 * kv, h, false), ("o", h, q, true)]
            } else {
                vec![
                    ("q", q, h, false),
                    ("k", kv, h, false),
                    ("v", kv, h, false),
                    ("o", h, q, true),
                ]
            };
            for (role, n, k, w8) in attention {
                require_matrix(
                    buffers,
                    &format!("{prefix}.{attn}.{role}_proj.weight"),
                    n,
                    k,
                    w8 || rotated_w8,
                    "embedding",
                )?;
            }
            let ffn = if packed_rows {
                vec![("gate_up", 2 * f, h), ("down", h, f)]
            } else {
                vec![("gate", f, h), ("up", f, h), ("down", h, f)]
            };
            for (role, n, k) in ffn {
                require_matrix(
                    buffers,
                    &format!("{prefix}.{mlp}.{role}_proj.weight"),
                    n,
                    k,
                    rotated_w8,
                    "embedding",
                )?;
            }
            for role in if vision {
                ["attention_norm", "ffn_norm"]
            } else {
                ["input_layernorm", "post_attention_layernorm"]
            } {
                require_weight(
                    buffers,
                    &format!("Runtime.{prefix}.{role}.weight.float16"),
                    Dtype::F16,
                    &[h],
                    "native_contiguous",
                    "embedding",
                )?;
            }
        }
    }
    for (role, n, k, w8) in [
        ("patch_merger.merging_layer", 1024, 4096, false),
        ("linear_1", 2048, 1024, false),
        ("linear_2", 2048, 2048, true),
    ] {
        require_matrix(
            buffers,
            &format!("multi_modal_projector.{role}.weight"),
            n,
            k,
            w8,
            "embedding",
        )?;
    }
    let mut text = vec![
        kernel("nemotron/text", "embedding"),
        kernel("nemotron/text", "overlay"),
    ];
    text.extend(transformer_ops("nemotron/text", 16, packed_rows, "rope"));
    text.extend(["final_norm", "pool", "normalize"].map(|r| kernel("nemotron/text", r)));
    let mut vision = vec![
        kernel("nemotron/vision", "patch"),
        kernel("nemotron/vision", "pre_norm"),
    ];
    vision.extend(transformer_ops("nemotron/vision", 24, packed_rows, "rope"));
    vision.extend(
        [
            "projector_norm",
            "merge",
            "merger_input",
            "merger",
            "linear1_input",
            "linear1",
            "gelu",
            "linear2_input",
            "linear2",
        ]
        .map(|r| kernel("nemotron/vision", r)),
    );
    let patches = spec.max_patches;
    let capacity = m.chunk_tokens.max(patches);
    let program_capacities = BTreeMap::from([
        ("embed".into(), m.chunk_tokens),
        ("embed_vision".into(), patches),
    ]);
    let (adapter, built) = finalize(
        Box::new(Embedding),
        m,
        BTreeMap::from([("embed".into(), text), ("embed_vision".into(), vision)]),
        capacity,
        program_capacities,
    )?;
    built
        .metadata
        .embedding
        .as_ref()
        .unwrap()
        .validate(&built.metadata)?;
    Ok((adapter, built))
}
