//! Qwen3 causal reranking: complete sequences and a two-row final-position head.
use crate::{CreateRequest, CreatedPlan};
use crate::{adapter::Adapter, artifact::Result, model::Operation};
use orinfer_model_sdk::{architecture::BatchSegment, packet};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
struct Reranker;
impl Adapter for Reranker {
    fn batch(&self, _: &[BatchSegment], _: bool) -> Result<packet::BatchPlan> {
        Err("Reranking executes independent complete causal sequences".into())
    }
}
fn operation(name: impl Into<String>) -> Operation {
    Operation::Kernel {
        name: format!("qwen_reranker/{}", name.into()),
    }
}
pub(crate) fn create(request: CreateRequest) -> Result<crate::registry::Built> {
    if request.architecture != "qwen3_reranker" || !request.prefill_profiles.is_empty() {
        return Err("Unsupported Qwen3 reranker execution contract".into());
    }
    let config = &request.config;
    let mut signature = config
        .as_object()
        .ok_or("Invalid reranker configuration")?
        .clone();
    signature.remove("quantization_config");
    if Value::Object(signature) != request.expected_signature {
        return Err("Reranker configuration signature changed".into());
    }
    for (key, value) in [
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
    ] {
        if config[key] != value {
            return Err(format!("Unsupported Qwen3 reranker {key}"));
        }
    }
    let quant = &config["quantization_config"];
    for (key, value) in [
        ("quant_method", json!("orinfer_e8p_q4_int8")),
        ("version", json!(1)),
        ("coefficients", json!([4, 1])),
        ("rotation", json!("signed-block128")),
        ("scale", json!("fp16-output-row")),
        ("compute_dtype", json!("int8_quality")),
        ("layout", json!("u32-low-coarse-high-residual")),
        ("policy", json!("qwen3-reranker-e8p-q4a8-v1")),
        ("embedding", json!("e8p_q4")),
        ("classifier", json!("fp16-original-two-rows")),
    ] {
        if quant[key] != value {
            return Err(format!("Unsupported reranker quantization {key}"));
        }
    }
    let protected = quant.get("rotated_w8_layers").cloned().unwrap_or(json!([]));
    if protected != json!([])
        && protected != json!([0, 1, 2, 3])
        && protected != json!([0, 1, 2, 3, 8])
    {
        return Err("Unsupported Qwen3 reranker protected layers".into());
    }
    let protected: Vec<usize> = serde_json::from_value(protected).map_err(|e| e.to_string())?;
    let mut m = request.metadata;
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
    use crate::artifact::{Access, Dtype};
    let require = |name: &str, dtype: Dtype, shape: &[usize], layout: &str| -> Result<()> {
        let b = m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or_else(|| format!("Missing reranker weight {name}"))?;
        if b.dtype != dtype
            || b.shape != shape
            || b.layout != layout
            || b.data.is_none()
            || b.access != Access::Read
        {
            return Err(format!("Invalid reranker weight {name}"));
        }
        Ok(())
    };
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
    matrix("model.embed_tokens.weight", 151669, 1024, false)?;
    require(
        "orinfer.reranker.classifier.weight",
        Dtype::F16,
        &[2, 1024],
        "native_contiguous",
    )?;
    require(
        "Runtime.model.norm.weight.float16",
        Dtype::F16,
        &[1024],
        "native_contiguous",
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
            matrix(
                &format!("{root}.self_attn.{role}_proj.weight"),
                n,
                k,
                w8 || rotated_w8,
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
            matrix(&format!("{root}.mlp.{role}_proj.weight"), n, k, rotated_w8)?;
        }
        for role in ["input_layernorm", "post_attention_layernorm"] {
            require(
                &format!("Runtime.{root}.{role}.weight.float16"),
                Dtype::F16,
                &[1024],
                "native_contiguous",
            )?;
        }
        for role in ["q_norm", "k_norm"] {
            require(
                &format!("Runtime.{root}.self_attn.{role}.weight.float16"),
                Dtype::F16,
                &[128],
                "native_contiguous",
            )?;
        }
    }
    let mut program = vec![operation("embedding")];
    for layer in 0..28 {
        let roles = if packed_rows {
            vec![
                "norm1",
                "rotate_qkv",
                "qkv",
                "qk_rope",
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
                "qk_rope",
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
        program.extend(
            roles
                .into_iter()
                .map(|role| operation(format!("layer{layer}/{role}"))),
        );
    }
    program.extend([operation("final_norm"), operation("classifier")]);
    m.programs = BTreeMap::from([("rerank".into(), program)]);
    m.row_execution = Some(orinfer_model_sdk::operators::rows::RowExecution {
        capacity: m.chunk_tokens,
        programs: BTreeSet::from(["rerank".into()]),
        program_capacities: Default::default(),
        variants: Default::default(),
    });
    m.row_execution.as_mut().unwrap().register_variants(
        &m.programs,
        m.dynamic_batch_kernels.keys().map(String::as_str),
    )?;
    m.reranking.as_ref().unwrap().validate(&m)?;
    Ok((
        Box::new(Reranker),
        CreatedPlan {
            expert_banks: vec![],
            metadata: m,
        },
    ))
}
