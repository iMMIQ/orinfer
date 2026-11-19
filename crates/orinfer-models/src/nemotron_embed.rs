//! Nemotron 3.5: independent bidirectional text and Pixtral embedding programs.
use crate::{CreateRequest, CreatedPlan};
use crate::{adapter::Adapter, artifact::Result, model::Operation};
use orinfer_model_sdk::{architecture::BatchSegment, packet};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

struct Embedding;
impl Adapter for Embedding {
    fn batch(&self, _: &[BatchSegment], _: bool) -> Result<packet::BatchPlan> {
        Err("Embedding requests execute complete bidirectional sequences independently".into())
    }
}

fn operation(name: impl Into<String>) -> Operation {
    Operation::Kernel {
        name: format!("nemotron/{}", name.into()),
    }
}
fn transformer(section: &str, layers: usize, packed_rows: bool) -> Vec<Operation> {
    let mut ops = Vec::new();
    for layer in 0..layers {
        let roles = if packed_rows {
            vec![
                "norm1",
                "rotate_qkv",
                "qkv",
                "rope",
                "attention",
                "quantize_o",
                "o",
                "add1",
                "norm2",
                "rotate_gateup",
                "gate_up",
                "swiglu",
                "rotate_down",
                "down",
                "add2",
            ]
        } else {
            vec![
                "norm1",
                "rotate_qkv",
                "q",
                "k",
                "v",
                "rope",
                "attention",
                "quantize_o",
                "o",
                "add1",
                "norm2",
                "rotate_gateup",
                "gate",
                "up",
                "swiglu",
                "rotate_down",
                "down",
                "add2",
            ]
        };
        for role in roles {
            ops.push(operation(format!("{section}/layer{layer}/{role}")));
        }
    }
    ops
}

fn validate_config(config: &Value, signature: &Value) -> Result<()> {
    let mut actual = config
        .as_object()
        .ok_or("Invalid embedding config")?
        .clone();
    actual.remove("quantization_config");
    if Value::Object(actual) != *signature {
        return Err("Nemotron embedding configuration signature differs".into());
    }
    for (key, expected) in [
        ("model_type", json!("mistral3")),
        ("architectures", json!(["Mistral3Model"])),
        ("is_causal", json!(false)),
        ("pooling", json!("avg")),
        ("spatial_merge_size", json!(2)),
        ("image_token_index", json!(10)),
        ("projector_hidden_act", json!("gelu")),
        ("multimodal_projector_bias", json!(false)),
        ("vision_feature_layer", json!(-1)),
    ] {
        if config[key] != expected {
            return Err(format!("Unsupported Nemotron {key}"));
        }
    }
    for (key, expected) in [
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
    ] {
        if config["text_config"][key] != expected {
            return Err(format!("Unsupported Ministral3 {key}"));
        }
    }
    for (key, expected) in [
        ("model_type", json!("pixtral")),
        ("hidden_size", json!(1024)),
        ("intermediate_size", json!(4096)),
        ("num_hidden_layers", json!(24)),
        ("num_attention_heads", json!(16)),
        ("head_dim", json!(64)),
        ("patch_size", json!(14)),
    ] {
        if config["vision_config"][key] != expected {
            return Err(format!("Unsupported Pixtral {key}"));
        }
    }
    if config["vision_config"]["rope_parameters"]
        != json!({"rope_type":"default","rope_theta":10000.0})
    {
        return Err("Unsupported Pixtral axial RoPE".into());
    }
    for (key, expected) in [
        ("rope_type", json!("yarn")),
        ("rope_theta", json!(1000000.0)),
        ("factor", json!(16.0)),
        ("original_max_position_embeddings", json!(16384)),
        ("beta_fast", json!(32.0)),
        ("beta_slow", json!(1.0)),
        ("apply_yarn_scaling", json!(false)),
        ("llama_4_scaling_beta", json!(0.1)),
    ] {
        if config["text_config"]["rope_parameters"][key] != expected {
            return Err(format!("Unsupported Ministral3 RoPE {key}"));
        }
    }
    let q = &config["quantization_config"];
    for (key, expected) in [
        ("quant_method", json!("orinfer_e8p_q4_int8")),
        ("version", json!(1)),
        ("coefficients", json!([4, 1])),
        ("rotation", json!("signed-block128")),
        ("scale", json!("fp16-output-row")),
        ("compute_dtype", json!("int8_quality")),
        ("basis_tensor", json!("orinfer.e8p_q4.basis")),
        ("layout", json!("u32-low-coarse-high-residual")),
        ("embedding", json!("e8p_q4")),
        ("vision", json!("e8p_q4")),
        ("policy", json!("nemotron35-embed-e8p-q4a8-v1")),
    ] {
        if q[key] != expected {
            return Err(format!("Unsupported embedding quantization {key}"));
        }
    }
    Ok(())
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
    let mut m = request.metadata;
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
    let require =
        |name: &str, dtype: crate::artifact::Dtype, shape: &[usize], layout: &str| -> Result<()> {
            let b = m
                .buffers
                .iter()
                .find(|b| b.name == name)
                .ok_or_else(|| format!("Missing embedding weight {name}"))?;
            if b.dtype != dtype
                || b.shape != shape
                || b.layout != layout
                || b.data.is_none()
                || b.access != crate::artifact::Access::Read
            {
                return Err(format!("Invalid embedding weight {name}"));
            }
            Ok(())
        };
    use crate::artifact::Dtype;
    let matrix = |name: &str, n: usize, k: usize, w8: bool| -> Result<()> {
        if w8 {
            require(
                &format!("{name}.int8"),
                Dtype::I8,
                &[n, k],
                "native_contiguous",
            )?;
        } else {
            require(
                &format!("{name}.packed"),
                Dtype::U32,
                &[k / 128, n, 16],
                "e8p-q4-k128-u32",
            )?;
        }
        require(
            &format!("{name}.scale"),
            Dtype::F16,
            &[n],
            "native_contiguous",
        )
    };
    let packed_rows = spec.rows.is_some();
    matrix("language_model.embed_tokens.weight", 131072, 2048, false)?;
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
                matrix(
                    &format!("{prefix}.{attn}.{role}_proj.weight"),
                    n,
                    k,
                    w8 || rotated_w8,
                )?;
            }
            let ffn = if packed_rows {
                vec![("gate_up", 2 * f, h), ("down", h, f)]
            } else {
                vec![("gate", f, h), ("up", f, h), ("down", h, f)]
            };
            for (role, n, k) in ffn {
                matrix(
                    &format!("{prefix}.{mlp}.{role}_proj.weight"),
                    n,
                    k,
                    rotated_w8,
                )?;
            }
            for role in if vision {
                ["attention_norm", "ffn_norm"]
            } else {
                ["input_layernorm", "post_attention_layernorm"]
            } {
                require(
                    &format!("Runtime.{prefix}.{role}.weight.float16"),
                    Dtype::F16,
                    &[h],
                    "native_contiguous",
                )?;
            }
        }
    }
    for (role, n, k, w8) in [
        ("patch_merger.merging_layer", 1024, 4096, false),
        ("linear_1", 2048, 1024, false),
        ("linear_2", 2048, 2048, true),
    ] {
        matrix(&format!("multi_modal_projector.{role}.weight"), n, k, w8)?;
    }
    let mut text = vec![operation("text/embedding"), operation("text/overlay")];
    text.extend(transformer("text", 16, packed_rows));
    text.extend(["final_norm", "pool", "normalize"].map(|r| operation(format!("text/{r}"))));
    let mut vision = vec![operation("vision/patch"), operation("vision/pre_norm")];
    vision.extend(transformer("vision", 24, packed_rows));
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
        .map(|r| operation(format!("vision/{r}"))),
    );
    let patches = spec.max_patches;
    m.programs = BTreeMap::from([("embed".into(), text), ("embed_vision".into(), vision)]);
    m.row_execution = Some(orinfer_model_sdk::operators::rows::RowExecution {
        capacity: m.chunk_tokens.max(patches),
        programs: BTreeSet::from(["embed".into(), "embed_vision".into()]),
        program_capacities: BTreeMap::from([
            ("embed".into(), m.chunk_tokens),
            ("embed_vision".into(), patches),
        ]),
        variants: Default::default(),
    });
    m.row_execution.as_mut().unwrap().register_variants(
        &m.programs,
        m.dynamic_batch_kernels.keys().map(String::as_str),
    )?;
    m.embedding.as_ref().unwrap().validate(&m)?;
    Ok((
        Box::new(Embedding),
        CreatedPlan {
            expert_banks: vec![],
            metadata: m,
        },
    ))
}
