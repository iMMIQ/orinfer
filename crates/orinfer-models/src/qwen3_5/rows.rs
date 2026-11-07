//! Table-Q4A8: shared row computation with request-owned causal state.
use super::config::Configuration;
use crate::packet::append;
use crate::{
    adapter::Adapter,
    artifact::{Access, Dtype, Result},
    execution::{BufferView, Invocation},
    model::{KvGrowth, Manifest, Operation, PrefillPlan, SegmentControls},
    policy::Policy,
};
use orinfer_model_sdk::{
    abi::{self, CreateRequest, CreatedPlan},
    architecture::{BatchLayout, BatchSegment},
    operators::rows::RowExecution,
};
use std::collections::{BTreeMap, BTreeSet};

const CAPACITY: usize = 4096;
const LANES: usize = 128;
// Appended after the layer state table; python/orinfer_build/models/qwen35/packet.py shares this order.
const PACKET_COLUMNS: &[&str] = &[
    "Input",
    "Token",
    "TokenStatus",
    "Logits",
    "Step",
    "Positions",
    "SeqLength",
    "SegmentLength",
    "LastIndex",
    "Requests",
    "MRopePositions",
    "Features",
    "FeatureIndex",
    "AudioFeatures",
    "AudioFeatureIndex",
    "MtpTargetHidden",
];
fn head_capacity(m: &Manifest) -> usize {
    m.buffers
        .iter()
        .find(|b| b.name == "HeadHidden")
        .and_then(|b| b.shape.first())
        .copied()
        .unwrap_or(LANES)
}
const PREFIX: &str = "qwen_rows";
// This order is the address-table contract compiled into the private kernels.
const STATE_COLUMNS: &[&str] = &[
    "History",
    "KPages",
    "KPagesScale",
    "MRopePositions",
    "Positions",
    "PrepareStatus",
    "QwenRowScratch",
    "Requests",
    "SegmentLength",
    "SeqLength",
    "State",
    "Step",
    "VPages",
    "VPagesScale",
];
#[path = "mtp.rs"]
mod mtp;
const PRE_GDN: &[&str] = &["norm_rotate", "qkv", "z", "ab"];
const PRE_ATTN: &[&str] = &["norm_rotate", "q", "k", "v"];
const PRIVATE_GDN: &[&str] = &["gates", "conv", "history", "recurrent", "gated_norm"];
const DEFERRED_COLUMNS: &[&str] = &["GdnLogCount", "GdnLogDelta", "GdnLogGate", "GdnLogKey"];
fn deferred_gdn(m: &Manifest) -> bool {
    m.gdn_execution
        .as_ref()
        .is_some_and(|g| g.deferred_capacity().is_some())
}
fn chunked_gdn(m: &Manifest) -> bool {
    m.gdn_execution
        .as_ref()
        .is_some_and(|g| g.prefill == orinfer_model_sdk::operators::gdn::Prefill::Chunked)
}
const PRIVATE_ATTN: &[&str] = &["prepare", "attention", "attention_merge"];
fn ffn_names(
    moe: bool,
    draft: bool,
    expanded: bool,
    planes: bool,
    cold: bool,
) -> Vec<&'static str> {
    let mut names = Vec::new();
    if moe {
        names.extend([
            "router",
            "topk",
            "histogram_local",
            "histogram_offsets",
            "histogram_finish",
            "offsets",
            "tiles",
            "dispatch",
        ]);
    }
    names.extend(["gate", "up"]);
    names.extend(if draft {
        &["down_quant"][..]
    } else {
        &["swiglu_rotate", "swiglu_quantize"][..]
    });
    names.push("down");
    if moe {
        if cold {
            names.extend(["expert_up_cold_planes", "expert_up_cold"]);
        }
        if expanded {
            names.push("expert_up_expand");
        }
        if planes {
            names.push("expert_up_planes");
        }
        names.extend(["expert_up", "expert_rotate"]);
        if cold {
            names.extend(["expert_down_cold_planes", "expert_down_cold"]);
        }
        if expanded {
            names.push("expert_down_expand");
        }
        if planes {
            names.push("expert_down_planes");
        }
        names.extend(["expert_down", "combine"]);
    }
    names
}
const HEAD: &[&str] = &["rotate", "project", "partials", "merge"];

struct QwenRows {
    manifest: Manifest,
    private_roles: Vec<Option<String>>,
    packet_kernels: Vec<crate::artifact::Kernel>,
}
fn private_roles(m: &Manifest) -> Vec<Option<String>> {
    let layout = m.batch_layout.as_ref().unwrap();
    let buffers: BTreeSet<_> = m.buffers.iter().map(|b| b.name.as_str()).collect();
    (0..layout.layers.len())
        .flat_map(|layer| {
            let buffers = &buffers;
            layout.state_columns.iter().map(move |column| {
                let per_layer = format!("L{layer}_{column}");
                if buffers.contains(per_layer.as_str()) {
                    Some(per_layer)
                } else {
                    buffers.contains(column.as_str()).then(|| column.clone())
                }
            })
        })
        .collect()
}
fn kernels(prefix: &str, names: &[&str]) -> Vec<Operation> {
    names
        .iter()
        .map(|name| Operation::Kernel {
            name: format!("{prefix}/{name}"),
        })
        .collect()
}
fn section(m: &mut Manifest, suffix: &str, names: &[&str]) -> Vec<Operation> {
    let name = format!("{PREFIX}/{suffix}");
    // Joined banks share the same projection implementation and row planner.
    // Collapse consecutive logical projections when their combined binding exists.
    let mut names = names.to_vec();
    let layer = suffix.split('/').next().filter(|s| s.starts_with("layer"));
    for (parts, combined, role) in [
        (&["q", "k", "v"][..], "qkv", "self_attn.qkv_proj"),
        (&["gate", "up"][..], "gate_up", "mlp.gate_up_proj"),
    ] {
        if layer.is_some_and(|layer| {
            let prefix = format!(
                "model.language_model.layers.{}.{}.weight.",
                &layer[5..],
                role
            );
            let packed = format!("{prefix}packed");
            let int8 = format!("{prefix}int8");
            m.buffers.iter().any(|b| b.name == packed || b.name == int8)
        }) && let Some(at) = names.windows(parts.len()).position(|p| p == parts)
        {
            names.splice(at..at + parts.len(), [combined]);
        }
    }
    let ops = kernels(&name, &names);
    m.programs.insert(name, ops.clone());
    ops
}
fn validate_weights(
    m: &Manifest,
    config: &Configuration,
    e8p: bool,
    embedding_q4: bool,
) -> Result<()> {
    let t = config.text();
    let map: BTreeMap<_, _> = m.buffers.iter().map(|b| (b.name.as_str(), b)).collect();
    let gdn = m.gdn_execution.as_ref();
    if !config.standard && gdn.is_none() {
        return Err("Missing explicit Qwen GDN execution contract".into());
    }
    if !chunked_gdn(m) && map.keys().any(|name| name.starts_with("GdnChunk")) {
        return Err("Chunkwise GDN buffers disagree with the execution contract".into());
    }
    if !deferred_gdn(m)
        && map
            .keys()
            .any(|name| DEFERRED_COLUMNS.iter().any(|role| name.ends_with(role)))
    {
        return Err("Deferred GDN buffers disagree with the execution contract".into());
    }
    let staged = ["AttentionDecodedK", "AttentionDecodedV"];
    if staged.iter().any(|name| map.contains_key(name)) {
        for name in staged {
            if map.get(name).is_none_or(|b| {
                b.dtype != Dtype::F16
                    || b.shape != [m.max_context, t.num_key_value_heads * t.head_dim]
                    || b.data.is_some()
                    || b.access != Access::ReadWrite
                    || m.reset_buffers.iter().any(|reset| reset == name)
                    || m.kv_cache
                        .as_ref()
                        .is_some_and(|kv| kv.buffers.contains_key(name))
            }) {
                return Err(format!("Invalid temporary attention workspace: {name}"));
            }
        }
    }
    if chunked_gdn(m) {
        for (name, dtype, shape) in [
            (
                "GdnChunkW",
                Dtype::F16,
                vec![t.linear_num_value_heads, CAPACITY, 128],
            ),
            (
                "GdnChunkU",
                Dtype::F16,
                vec![t.linear_num_value_heads, CAPACITY, 128],
            ),
            (
                "GdnChunkD",
                Dtype::F16,
                vec![t.linear_num_value_heads, CAPACITY, 32],
            ),
            (
                "GdnChunkG",
                Dtype::F32,
                vec![t.linear_num_value_heads, CAPACITY, 2],
            ),
        ] {
            if map.get(name).is_none_or(|b| {
                b.dtype != dtype
                    || b.shape != shape
                    || b.data.is_some()
                    || b.access != Access::ReadWrite
            }) {
                return Err(format!("Invalid chunkwise GDN workspace: {name}"));
            }
        }
    }
    if deferred_gdn(m) {
        for (layer, kind) in t.layer_types.iter().enumerate() {
            if kind != "linear_attention" {
                continue;
            }
            let capacity = gdn
                .and_then(|g| g.deferred_capacity())
                .filter(|n| [2, 4, 8, 16, 32].contains(n))
                .ok_or("Invalid deferred GDN capacity")?;
            for (suffix, dtype, shape) in [
                (
                    "GdnLogCount",
                    Dtype::I32,
                    vec![t.linear_num_value_heads, 128],
                ),
                (
                    "GdnLogDelta",
                    Dtype::F32,
                    vec![capacity, t.linear_num_value_heads, 128],
                ),
                (
                    "GdnLogGate",
                    Dtype::F32,
                    vec![capacity, t.linear_num_value_heads],
                ),
                (
                    "GdnLogKey",
                    Dtype::F16,
                    vec![capacity, t.linear_num_key_heads, 128],
                ),
            ] {
                let name = format!("L{layer}_{suffix}");
                if map.get(name.as_str()).is_none_or(|b| {
                    b.dtype != dtype
                        || b.shape != shape
                        || b.data.is_some()
                        || b.access != Access::ReadWrite
                }) || !m.reset_buffers.contains(&name)
                {
                    return Err(format!("Invalid request-owned deferred GDN state: {name}"));
                }
            }
        }
    }
    if ["MoeLow", "MoeHigh", "MoeSum"]
        .iter()
        .any(|name| map.contains_key(name))
    {
        if t.num_experts == 0 {
            return Err("E8P expert planes require routed experts".into());
        }
        for (name, dtype, size) in [
            (
                "MoeLow",
                Dtype::I8,
                CAPACITY * t.num_experts_per_tok * t.hidden_size.max(t.moe_intermediate_size) / 2,
            ),
            (
                "MoeHigh",
                Dtype::I8,
                CAPACITY * t.num_experts_per_tok * t.hidden_size.max(t.moe_intermediate_size) / 2,
            ),
            ("MoeSum", Dtype::I32, CAPACITY * t.num_experts_per_tok),
        ] {
            if map.get(name).is_none_or(|b| {
                b.dtype != dtype
                    || b.shape != [size]
                    || b.data.is_some()
                    || b.access != Access::ReadWrite
            }) {
                return Err(format!("Invalid E8P expert plane workspace: {name}"));
            }
        }
    }
    let cold_names = [
        "MoeColdTileOffsets",
        "MoeColdTileCount",
        "MoeColdTileExpert",
        "MoeColdTileRow",
    ];
    if cold_names.iter().any(|name| map.contains_key(name)) {
        if t.num_experts == 0 || !map.contains_key("MoeLow") {
            return Err("Expert partition requires routed E8P planes".into());
        }
        let slots = CAPACITY * t.num_experts_per_tok;
        let expert = map
            .get("MoeColdTileExpert")
            .ok_or("Missing cold expert tiles")?;
        if expert.shape.len() != 1
            || expert.shape[0] < slots.min(t.num_experts)
            || expert.shape[0] > slots + t.num_experts
        {
            return Err("Invalid cold expert tile capacity".into());
        }
        for (name, shape) in [
            (cold_names[0], vec![t.num_experts]),
            (cold_names[1], vec![1]),
            (cold_names[2], expert.shape.clone()),
            (cold_names[3], expert.shape.clone()),
        ] {
            if map.get(name).is_none_or(|b| {
                b.dtype != Dtype::I32
                    || b.shape != shape
                    || b.data.is_some()
                    || b.access != Access::ReadWrite
            }) {
                return Err(format!("Invalid expert partition workspace: {name}"));
            }
        }
    }
    let check = |name: String, dtype: Dtype, shape: Vec<usize>| -> Result<()> {
        if map.get(name.as_str()).is_none_or(|b| {
            b.dtype != dtype || b.shape != shape || b.access != Access::Read || b.data.is_none()
        }) {
            return Err(format!("Invalid Qwen Q4A8 operand: {name}"));
        }
        Ok(())
    };
    if e8p {
        check("orinfer.e8p_q4.basis".into(), Dtype::I8, vec![256, 8])?;
        check("orinfer.e8p_q4.selectors".into(), Dtype::U32, vec![256, 1])?;
    }
    let q4 = |name: String, n: usize, k: usize| -> Result<()> {
        if config
            .rotated_w8_layers
            .iter()
            .any(|layer| name.starts_with(&format!("model.language_model.layers.{layer}.")))
        {
            check(format!("{name}.int8"), Dtype::I8, vec![n, k])?;
            check(format!("{name}.scale"), Dtype::F16, vec![n])?;
            if map.contains_key(format!("{name}.packed").as_str()) {
                return Err("Protected projection has duplicate compressed weights".into());
            }
            return Ok(());
        }
        let (dtype, shape) = if e8p {
            (Dtype::U32, vec![k / 128, n, 16])
        } else {
            (Dtype::U8, vec![n, k / 2])
        };
        let packed = format!("{name}.packed");
        check(packed.clone(), dtype, shape)?;
        if e8p && map[packed.as_str()].layout != "e8p-q4-k128-u32" {
            return Err(format!("Invalid Qwen E8P physical layout: {packed}"));
        }
        check(format!("{name}.scale"), Dtype::F16, vec![n])
    };
    let has_projection = |name: &str| {
        map.contains_key(format!("{name}.packed").as_str())
            || map.contains_key(format!("{name}.int8").as_str())
    };
    if embedding_q4 {
        q4(
            "model.language_model.embed_tokens.weight".into(),
            t.vocab_size,
            t.hidden_size,
        )?;
    }
    if !t.tie_word_embeddings {
        q4("lm_head.weight".into(), t.vocab_size, t.hidden_size)?;
    }
    for (layer, kind) in t.layer_types.iter().enumerate() {
        let p = format!("model.language_model.layers.{layer}");
        let combined_ffn = format!("{p}.mlp.gate_up_proj.weight");
        let joined_ffn = t.num_experts == 0 && has_projection(&combined_ffn);
        if joined_ffn {
            q4(combined_ffn, 2 * t.intermediate_size, t.hidden_size)?;
        }
        for (name, n, k) in [
            ("gate_proj", t.intermediate_size, t.hidden_size),
            ("up_proj", t.intermediate_size, t.hidden_size),
            ("down_proj", t.hidden_size, t.intermediate_size),
        ] {
            if joined_ffn && name != "down_proj" {
                if has_projection(&format!("{p}.mlp.{name}.weight")) {
                    return Err("Combined FFN has duplicate source weights".into());
                }
                continue;
            }
            let branch = if t.num_experts > 0 {
                "shared_expert."
            } else {
                ""
            };
            q4(format!("{p}.mlp.{branch}{name}.weight"), n, k)?;
        }
        if t.num_experts > 0 {
            for (name, n, k) in [
                ("gate_up_proj", 2 * t.moe_intermediate_size, t.hidden_size),
                ("down_proj", t.hidden_size, t.moe_intermediate_size),
            ] {
                let name = format!("{p}.mlp.experts.{name}");
                let packed = format!("{name}.packed");
                check(
                    packed.clone(),
                    Dtype::U32,
                    vec![t.num_experts, k / 128, n, 16],
                )?;
                if map[packed.as_str()].layout != "e8p-q4-k128-u32" {
                    return Err("Invalid Qwen MoE E8P bank layout".into());
                }
                check(format!("{name}.scale"), Dtype::F16, vec![t.num_experts, n])?;
            }
        }
        let combined_qkv = format!("{p}.self_attn.qkv_proj.weight");
        let joined_qkv = kind == "full_attention" && has_projection(&combined_qkv);
        if joined_qkv {
            q4(combined_qkv, t.full_x(), t.hidden_size)?;
            for role in ["q", "k", "v"] {
                if has_projection(&format!("{p}.self_attn.{role}_proj.weight")) {
                    return Err("Combined QKV has duplicate source weights".into());
                }
            }
        }
        let shapes: &[(&str, usize)] = if joined_qkv {
            &[]
        } else if kind == "linear_attention" {
            &[
                ("linear_attn.in_proj_qkv", t.qkv()),
                ("linear_attn.in_proj_z", t.mixer()),
            ]
        } else {
            &[
                (
                    "self_attn.q_proj",
                    (if config.standard { 1 } else { 2 }) * t.full_q(),
                ),
                ("self_attn.k_proj", t.kv()),
                ("self_attn.v_proj", t.kv()),
            ]
        };
        for &(name, n) in shapes {
            q4(format!("{p}.{name}.weight"), n, t.hidden_size)?;
        }
    }
    Ok(())
}

fn vision(m: &mut Manifest, depth: usize, hidden: usize) -> Result<()> {
    let Some(v) = m.vision.as_ref() else {
        return Ok(());
    };
    if !matches!(depth, 24 | 27)
        || v.hidden != hidden
        || v.max_patches > 32768
        || v.plans.len() != 1
        || v.plans[0].min_patches != 4
        || v.plans[0].patches != v.max_patches
        || v.plans[0].program != "vision"
    {
        return Err("Unsupported Qwen compact vision contract".into());
    }
    let mut operations = kernels("qwen_rows/vision/begin", &["patch", "position"]);
    let append = |ops: &mut Vec<Operation>, section: &str, role: &str, weight: Option<String>| {
        if weight.is_some_and(|name| {
            m.buffers
                .iter()
                .any(|b| b.name == format!("model.visual.{name}.weight.packed"))
        }) {
            ops.extend(kernels(
                &format!("qwen_rows/vision/{section}"),
                &[&format!("{role}_rotate"), &format!("{role}_project")],
            ));
        }
        ops.extend(kernels(&format!("qwen_rows/vision/{section}"), &[role]));
    };
    for layer in 0..depth {
        for role in [
            "norm1",
            "qkv",
            "rope",
            "attention",
            "projection",
            "residual1",
            "norm2",
            "fc1",
            "fc2",
            "residual2",
        ] {
            let weight = match role {
                "qkv" => Some("attn.qkv"),
                "projection" => Some("attn.proj"),
                "fc1" => Some("mlp.linear_fc1"),
                "fc2" => Some("mlp.linear_fc2"),
                _ => None,
            }
            .map(|name| format!("blocks.{layer}.{name}"));
            append(&mut operations, &format!("layer{layer}"), role, weight);
        }
    }
    append(&mut operations, "end", "norm", None);
    append(
        &mut operations,
        "end",
        "merge1",
        Some("merger.linear_fc1".into()),
    );
    append(
        &mut operations,
        "end",
        "merge2",
        Some("merger.linear_fc2".into()),
    );
    m.programs.insert("vision".into(), operations);
    Ok(())
}

pub(crate) fn create(mut request: CreateRequest, policy: Policy) -> Result<crate::registry::Built> {
    if !matches!(request.architecture.as_str(), "qwen3_5" | "qwen3_asr")
        || policy != Policy::Int8Quality
    {
        return Err("Unsupported Qwen row precision policy".into());
    }
    let quant = &request.config["quantization_config"];
    let e8p = match quant["quant_method"].as_str() {
        Some("orinfer_q4_int8") => false,
        Some("orinfer_e8p_q4_int8") => true,
        _ => return Err("Unsupported Qwen Q4A8 codec".into()),
    };
    if e8p
        && (quant["coefficients"] != serde_json::json!([4, 1])
            || quant["basis_tensor"] != "orinfer.e8p_q4.basis"
            || quant["layout"] != "u32-low-coarse-high-residual")
    {
        return Err("Unsupported Qwen E8P-Q4 encoding".into());
    }
    for (key, value) in [
        ("version", serde_json::json!(1)),
        ("rotation", serde_json::json!("signed-block128")),
        ("scale", serde_json::json!("fp16-output-row")),
        ("compute_dtype", serde_json::json!("int8_quality")),
    ] {
        if quant[key] != value {
            return Err(format!("Unsupported Qwen Q4A8 {key}"));
        }
    }
    let embedding_q4 = quant["embedding"] == "e8p_q4";
    for role in ["embedding", "vision"] {
        if !quant[role].is_null() && (!e8p || quant[role] != "e8p_q4") {
            return Err(format!("Unsupported Qwen {role} precision"));
        }
    }
    let c = Configuration::parse(request.config)?;
    let architecture = if c.standard { "qwen3_asr" } else { "qwen3_5" };
    if request.architecture != architecture || c.signature != request.expected_signature {
        return Err("Qwen row architecture or signature differs".into());
    }
    let t = c.text();
    let standard_geometry = c.standard
        && t.hidden_size == 2048
        && t.intermediate_size == 6144
        && t.num_hidden_layers == 28
        && t.num_attention_heads == 16
        && t.num_key_value_heads == 8
        && t.head_dim == 128
        && t.vocab_size == 151936
        && t.tie_word_embeddings;
    let geometry = standard_geometry
        || matches!(
            (
                t.hidden_size,
                t.intermediate_size,
                t.num_hidden_layers,
                t.num_attention_heads,
                t.num_key_value_heads,
                t.linear_num_value_heads,
                t.tie_word_embeddings
            ),
            (5120, 17408, 64, 24, 4, 48, false)
                | (4096, 12288, 32, 16, 4, 32, false)
                | (2048, 6144, 24, 8, 2, 16, true)
                | (2560, 9216, 32, 16, 4, 32, true)
                | (2048, 512, 40, 16, 2, 32, false)
        );
    let moe_geometry = if t.num_experts > 0 {
        e8p && t.hidden_size == 2048
            && t.num_hidden_layers == 40
            && t.num_experts == 256
            && t.num_experts_per_tok == 8
            && t.moe_intermediate_size == 512
            && t.shared_expert_intermediate_size == 512
            && t.intermediate_size == 512
            && t.mlp_only_layers.is_empty()
    } else {
        t.intermediate_size != 512 && t.num_experts_per_tok == 0
    };
    if !geometry
        || !moe_geometry
        || (t.tie_word_embeddings && !embedding_q4)
        || (!c.standard
            && (t.head_dim != 256
                || t.linear_num_key_heads != 16
                || t.linear_key_head_dim != 128
                || t.linear_value_head_dim != 128
                || t.linear_conv_kernel_dim != 4
                || (t.vocab_size,) != (248320,)))
        || request.prefill_profiles.len() != 1
        || request.prefill_profiles[0].tokens != CAPACITY
        || request.prefill_profiles[0].min_tokens != 1
        || request.prefill_profiles[0].kind != "qwen_q4_rows"
    {
        return Err("Expected dense Qwen compact 1..4096 row contract".into());
    }
    let m = &mut request.metadata;
    validate_weights(m, &c, e8p, embedding_q4)?;
    m.programs.clear();
    let begin = if m.vision.is_some() || m.audio.is_some() {
        vec!["prepare", "embedding", "overlay"]
    } else {
        vec!["prepare", "embedding"]
    };
    let mut forward = section(m, "begin", &begin);
    let mut layer_ends = Vec::with_capacity(t.layer_types.len());
    let mut strides = BTreeMap::new();
    for (name, width, size) in [
        ("Hidden", t.hidden_size, 2),
        ("R0", t.hidden_size, 4),
        ("R1", t.hidden_size, 4),
        ("Norm", t.hidden_size, 2),
        ("QKV", t.qkv(), 2),
        ("Zout", t.mixer(), 2),
        ("AB", 2 * t.linear_num_value_heads, 2),
        ("FullX", t.full_x(), 2),
        ("FullQ", t.mixer(), 2),
        ("FullGate", t.mixer(), 2),
        ("MixerIn", t.mixer(), 2),
        ("Mix", t.hidden_size, 2),
        ("GateUp", 2 * t.intermediate_size, 2),
    ] {
        if width > 0 && (!c.standard || name != "Zout") {
            strides.insert(name.into(), width * size);
        }
    }
    for (layer, kind) in t.layer_types.iter().enumerate() {
        let gdn = kind == "linear_attention";
        forward.extend(section(
            m,
            &format!("layer{layer}/pre"),
            if gdn { PRE_GDN } else { PRE_ATTN },
        ));
        let mut private_names = if gdn { PRIVATE_GDN } else { PRIVATE_ATTN }.to_vec();
        if !gdn && m.buffers.iter().any(|b| b.name == "AttentionDecodedK") {
            private_names.insert(1, "attention_decode");
        }
        if gdn && deferred_gdn(m) {
            private_names.insert(3, "recurrent_materialize");
        }
        if gdn && chunked_gdn(m) {
            let index = private_names
                .iter()
                .position(|name| *name == "recurrent")
                .unwrap();
            private_names[index] = "recurrent_chunk";
            private_names.insert(index, "recurrent_prepare");
        }
        let mut private = section(m, &format!("layer{layer}/private"), &private_names);
        if gdn {
            // History is a fixed-size private state, independent of live rows.
            private[2] = Operation::Copy {
                source: "HistoryOut".into(),
                destination: format!("L{layer}_History"),
                bytes: 3 * t.qkv() * 2,
            };
            m.programs
                .insert(format!("{PREFIX}/layer{layer}/private"), private.clone());
        }
        forward.extend(private);
        let mut post = vec!["quantize", "out", "norm_rotate"];
        post.extend(ffn_names(
            t.num_experts > 0,
            false,
            m.buffers.iter().any(|b| b.name == "MoeExpanded"),
            m.buffers.iter().any(|b| b.name == "MoeLow"),
            m.buffers.iter().any(|b| b.name == "MoeColdTileCount"),
        ));
        forward.extend(section(m, &format!("layer{layer}/post"), &post));
        layer_ends.push(forward.len());
    }
    forward.extend(section(m, "end", &["advance"]));
    let gather = section(m, "gather", &["norm"]);
    section(m, "head", HEAD);
    let head = section(m, "head_last", HEAD);
    let select = section(m, "select", &["selection"]);
    let mut head_last = gather;
    head_last.extend(head);
    head_last.push(Operation::Copy {
        source: "HeadLogits".into(),
        destination: m.logits.clone(),
        bytes: m.vocab * 4,
    });
    head_last.extend(select);
    m.programs.insert("head".into(), head_last);
    m.programs.insert("prefill".into(), forward.clone());
    let mut decode = vec![Operation::Copy {
        source: m.token.clone(),
        destination: m.input.clone(),
        bytes: 4,
    }];
    decode.extend(forward.into_iter().filter_map(|mut operation| {
        if let Operation::Kernel { name } = &mut operation {
            if name.ends_with("/private/recurrent_prepare") {
                return None;
            }
            if let Some(prefix) = name.strip_suffix("/private/recurrent_chunk") {
                *name = format!("{prefix}/private/recurrent");
            }
        }
        Some(operation)
    }));
    decode.extend(m.programs["head"].clone());
    m.programs.insert("decode".into(), decode);
    m.prefill_plans = vec![PrefillPlan {
        chunk_tokens: CAPACITY,
        min_tokens: 1,
        prefill_program: "prefill".into(),
        head_program: "head".into(),
        stages: vec![],
        live_buffers: vec![],
    }];
    m.batch_profiles = vec![1, LANES];
    m.prefill_batch_profiles = request.prefill_profiles.clone();
    m.segment_controls = Some(SegmentControls {
        length: "SegmentLength".into(),
        last_index: "LastIndex".into(),
    });
    let state_columns = m
        .batch_layout
        .as_ref()
        .map_or_else(Vec::new, |l| l.state_columns.clone());
    m.batch_layout = Some(BatchLayout {
        layers: t.layer_types.clone(),
        row_strides: strides,
        profiles: BTreeMap::from([(CAPACITY, "qwen_q4_rows".into())]),
        hidden: t.hidden_size,
        small_mixed_shapes: vec![],
        segment_tokens: CAPACITY,
        state_columns,
        preparation_programs: BTreeMap::from([(1, "decode".into()), (CAPACITY, "prefill".into())]),
    });
    {
        let columns = &m.batch_layout.as_ref().unwrap().state_columns;
        let mut expected_columns = STATE_COLUMNS.to_vec();
        if c.standard {
            expected_columns
                .retain(|name| !["History", "State", "SegmentLength", "Step"].contains(name));
        }
        if deferred_gdn(m) {
            expected_columns.extend(DEFERRED_COLUMNS);
            expected_columns.sort_unstable();
        }
        let packet_columns = if request
            .kernel_contracts
            .iter()
            .any(|k| k.name == "qwen_rows/packet/tokens")
        {
            PACKET_COLUMNS.len()
        } else {
            0
        };
        if *columns != expected_columns
            || m.buffers
                .iter()
                .find(|b| b.name == "QwenRowPointers")
                .is_none_or(|b| {
                    b.dtype != Dtype::U64
                        || b.shape
                            != [LANES * (t.layer_types.len() * columns.len() + packet_columns)]
                })
        {
            return Err("Invalid Qwen private address table".into());
        }
        m.state_pointer_table = Some(crate::model::StatePointerTable {
            buffer: "QwenRowPointers".into(),
            max_rows: LANES,
        });
        for (layer, kind) in t.layer_types.iter().enumerate() {
            let mut names = if kind == "linear_attention" {
                &["gates", "conv", "recurrent", "gated_norm"][..]
            } else {
                &["prepare", "attention", "attention_merge"][..]
            }
            .to_vec();
            let name = format!("{PREFIX}/state/layer{layer}");
            // A separately bound history update belongs between convolution
            // and recurrence. In-place convolution packages omit this binding.
            if kind == "linear_attention"
                && request
                    .kernel_contracts
                    .iter()
                    .any(|kernel| kernel.name == format!("{name}/history"))
            {
                names.insert(2, "history");
            }
            m.programs.insert(
                name,
                kernels(&format!("{PREFIX}/state/layer{layer}"), &names),
            );
        }
    }
    mtp::register(m, t.num_experts > 0)?;
    if request
        .kernel_contracts
        .iter()
        .any(|k| k.name == "qwen_rows/packet/tokens")
    {
        let mut begin = vec!["tokens", "prepare", "embedding"];
        if m.vision.is_some() || m.audio.is_some() {
            begin.push("overlay");
        }
        m.programs.insert(
            "qwen_rows/packet/begin".into(),
            kernels("qwen_rows/packet", &begin),
        );
        let mut end = vec!["advance"];
        if m.mtp.is_some() {
            end.push("capture");
        }
        end.push("norm");
        m.programs.insert(
            "qwen_rows/packet/end".into(),
            kernels("qwen_rows/packet", &end),
        );
        m.programs.insert(
            "qwen_rows/packet/outputs".into(),
            kernels("qwen_rows/packet", &["outputs"]),
        );
    }
    vision(m, c.vision_depth, t.hidden_size)?;
    if m.audio.is_some() {
        m.programs.insert(
            "audio_stem".into(),
            kernels(
                "qwen_rows/audio/stem",
                &["conv1", "conv2", "conv3", "flatten", "out", "position"],
            ),
        );
        let mut ops = vec![];
        for layer in 0..24 {
            ops.extend(kernels(
                &format!("qwen_rows/audio/layer{layer}"),
                &[
                    "norm1",
                    "qkv",
                    "split",
                    "attention",
                    "out",
                    "add1",
                    "norm2",
                    "up",
                    "down",
                    "add2",
                ],
            ));
        }
        ops.extend(kernels(
            "qwen_rows/audio/end",
            &["norm", "project1", "project2"],
        ));
        m.programs.insert("audio_encoder".into(), ops);
    }
    crate::packet::register_indirect(m);
    let programs = m
        .programs
        .keys()
        .filter(|name| name.as_str() != "mtp_packet_indirect")
        .cloned()
        .collect::<BTreeSet<_>>();
    m.row_execution = Some(RowExecution {
        capacity: m
            .vision
            .as_ref()
            .map_or(CAPACITY, |v| CAPACITY.max(v.max_patches))
            .max(m.audio.as_ref().map_or(0, |a| a.max_features)),
        programs: programs.clone(),
        program_capacities: programs
            .iter()
            .map(|p| {
                (
                    p.clone(),
                    match p.as_str() {
                        "audio_stem" => m.audio.as_ref().unwrap().stem_chunks,
                        "audio_encoder" => m.audio.as_ref().unwrap().max_features,
                        "vision" => m.vision.as_ref().map_or(CAPACITY, |v| v.max_patches),
                        "decode"
                        | "mtp_draft"
                        | "mtp_head"
                        | "mtp_gather"
                        | "mtp_select"
                        | "qwen_rows/select"
                        | "qwen_rows/head_last" => 1,
                        "mtp_warm" | "mtp_begin" | "mtp_pre" | "mtp_private" | "mtp_post"
                        | "mtp_advance" => 512,
                        "verify" | "restore" => 8,
                        name if name.starts_with("qwen_rows/state/") => LANES,
                        name if name.starts_with("qwen_rows/packet/") => LANES,
                        name if name.starts_with("qwen_rows/verify_state/") => LANES,
                        name if name.starts_with("qwen_rows/verify/") => 8,
                        "qwen_rows/head" => head_capacity(m),
                        "qwen_rows/commit" => LANES,
                        _ => CAPACITY,
                    },
                )
            })
            .collect(),
        variants: BTreeMap::new(),
    });
    let slots = m
        .programs
        .iter()
        .flat_map(|(program, operations)| {
            let capacity = m.row_execution.as_ref().unwrap().program_capacity(program);
            operations.iter().filter_map(move |op| match op {
                Operation::Kernel { name } => Some((name.as_str(), capacity)),
                _ => None,
            })
        })
        .fold(
            BTreeMap::<_, usize>::new(),
            |mut slots, (name, capacity)| {
                slots
                    .entry(name)
                    .and_modify(|bound| *bound = (*bound).max(capacity))
                    .or_insert(capacity);
                slots
            },
        );
    let rows = m
        .row_execution
        .as_mut()
        .ok_or("Missing Qwen dynamic rows")?;
    for kernel in &request.kernel_contracts {
        let Some((slot, mut choice)) =
            orinfer_model_sdk::operators::rows::specialization(&kernel.name, CAPACITY)?
        else {
            continue;
        };
        let capacity = *slots
            .get(slot)
            .ok_or("Qwen local row choice has no computation slot")?;
        if choice.min_rows > capacity {
            continue;
        }
        choice.max_rows = choice.max_rows.min(capacity);
        rows.variants.entry(slot.into()).or_default().push(choice);
    }
    for choices in rows.variants.values_mut() {
        choices.sort_by_key(|v| v.min_rows);
    }
    let kv = m
        .kv_cache
        .as_mut()
        .ok_or("Qwen rows require INT8 KV storage")?;
    {
        kv.growth.insert(
            "prefill".to_owned(),
            KvGrowth {
                position: m.position.clone(),
                tokens: CAPACITY,
                length: Some("SegmentLength".into()),
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|name| !name.starts_with("Mtp"))
                    .cloned()
                    .collect(),
            },
        );
    }
    // The one-token decoder is captured independently of prompt segments.
    kv.growth.insert(
        "decode".into(),
        KvGrowth {
            position: m.position.clone(),
            tokens: 1,
            length: None,
            buffers: kv
                .buffers
                .keys()
                .filter(|name| !name.starts_with("Mtp"))
                .cloned()
                .collect(),
        },
    );
    *layer_ends.last_mut().ok_or("Missing Qwen layer boundary")? = m.programs["prefill"].len();
    let mut first = 0;
    let stages = layer_ends
        .into_iter()
        .map(|last| {
            let operations = m.programs["prefill"][first..last].to_vec();
            first = last;
            operations
        })
        .collect();
    crate::prefill::register(m, "prefill", stages, vec!["Hidden".into(), "R0".into()])?;
    let metadata = request.metadata;
    Ok((
        Box::new(QwenRows {
            private_roles: private_roles(&metadata),
            manifest: metadata.clone(),
            packet_kernels: if metadata.packet_table().is_some() {
                request.kernel_contracts
            } else {
                vec![]
            },
        }),
        CreatedPlan {
            metadata,
            decode_programs: BTreeSet::from(["decode".into()]),
            expert_banks: vec![],
        },
    ))
}

impl Adapter for QwenRows {
    fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: abi::BatchOperation,
        include_plan: bool,
    ) -> Result<abi::BatchPlan> {
        let result = match operation {
            abi::BatchOperation::Ordinary => self.packet(segments, None, include_plan),
            abi::BatchOperation::Verification => self.packet(segments, Some(0), include_plan),
            abi::BatchOperation::MixedRows { prefills }
                if prefills > 0 && usize::from(prefills) < segments.len() =>
            {
                self.packet(segments, Some(usize::from(prefills)), include_plan)
            }
            abi::BatchOperation::Draft
            | abi::BatchOperation::Warm
            | abi::BatchOperation::DraftHead => {
                mtp::draft(&self.manifest, segments, operation, include_plan)
            }
            abi::BatchOperation::Commit => mtp::commit(&self.manifest, segments, include_plan),
            _ => Err("Unsupported Qwen packet operation".into()),
        }?;
        crate::packet::indirect(&self.manifest, segments, &self.packet_kernels, result)
    }
    fn prepare(
        &self,
        program: &str,
        tokens: &[u32],
        _history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        if tokens.is_empty() {
            return Err("Empty Qwen causal segment".into());
        }
        let length =
            u32::try_from(tokens.len()).map_err(|_| "Qwen causal segment length overflow")?;
        let name = if program == "mtp_warm" || program == "mtp_draft" {
            "MtpSegmentLength"
        } else {
            "SegmentLength"
        };
        Ok(vec![(name.into(), length.to_le_bytes().to_vec())])
    }
    fn visual(
        &self,
        tokens: &[u32],
        images: &[abi::ImageGrid],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        crate::vision::layout(
            self.manifest
                .vision
                .as_ref()
                .ok_or("Model has no vision adapter")?,
            tokens,
            images,
            capacity,
        )
    }
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<abi::BatchPlan> {
        self.packet(segments, None, include_plan)
    }
}

impl QwenRows {
    fn state_bindings(&self, segments: &[BatchSegment]) -> abi::StateBindings {
        let decoders: Vec<_> = segments.iter().filter(|s| s.tokens == 1).cloned().collect();
        if self
            .manifest
            .programs
            .contains_key("qwen_rows/packet/begin")
        {
            let mut bindings =
                crate::packet::private_bindings(&decoders, LANES, &self.private_roles);
            let names: BTreeSet<_> = self
                .manifest
                .buffers
                .iter()
                .map(|b| b.name.as_str())
                .collect();
            let roles = PACKET_COLUMNS
                .iter()
                .map(|name| names.contains(name).then(|| (*name).to_owned()))
                .collect::<Vec<_>>();
            bindings.extend(crate::packet::private_bindings(&decoders, LANES, &roles));
            bindings
        } else {
            crate::packet::private_bindings(&decoders, decoders.len(), &self.private_roles)
        }
    }
    fn packet(
        &self,
        segments: &[BatchSegment],
        prefills: Option<usize>,
        include_plan: bool,
    ) -> Result<abi::BatchPlan> {
        let m = &self.manifest;
        let ordered: Vec<_> = if prefills.is_none() {
            segments
                .iter()
                .filter(|s| s.tokens == 1)
                .chain(segments.iter().filter(|s| s.tokens != 1))
                .cloned()
                .collect()
        } else {
            segments.to_vec()
        };
        let segments = ordered.as_slice();
        let rows = crate::packet::Rows::packet(segments, LANES, CAPACITY)?;
        let total = rows.total;
        let verified = |lane: usize| prefills.is_some_and(|n| lane >= n);
        let causal = prefills.and_then(|_| m.mtp.as_ref()?.verification_table.as_ref());
        let mut causal_groups = BTreeMap::<&str, Vec<BatchSegment>>::new();
        if let Some(table) = causal {
            for segment in &segments[prefills.unwrap()..] {
                let program = if table.attention.is_empty() {
                    ""
                } else {
                    table
                        .attention
                        .iter()
                        .find(|choice| {
                            (choice.min_tokens..=choice.max_tokens).contains(&segment.tokens)
                        })
                        .ok_or("Missing causal attention selection")?
                        .program
                        .as_str()
                };
                causal_groups
                    .entry(program)
                    .or_default()
                    .push(BatchSegment {
                        slot: segment.slot,
                        tokens: segment.tokens,
                    });
            }
        }
        let causal_bindings = || {
            let table = causal.unwrap();
            let roles = crate::packet::layer_roles(
                m,
                m.batch_layout.as_ref().unwrap().layers.len(),
                &table.columns,
                |role, layer| [format!("L{layer}_{role}"), role.to_owned()],
            );
            let decoders: Vec<_> = causal_groups.values().flatten().cloned().collect();
            crate::packet::private_bindings(&decoders, decoders.len(), &roles)
        };
        let heads = crate::packet::Rows::new(
            segments
                .iter()
                .enumerate()
                .map(|(lane, s)| if verified(lane) { s.tokens } else { 1 }),
            head_capacity(m),
        )?;
        let head_rows = heads.total;
        if segments.iter().enumerate().any(|(lane, s)| {
            verified(lane)
                && !m
                    .mtp
                    .as_ref()
                    .is_some_and(|spec| spec.batch_verification_tokens.contains(&s.tokens))
        }) {
            return Err("Invalid Qwen verification rows".into());
        }
        if !include_plan {
            return Ok((
                vec![],
                if causal.is_some() {
                    causal_bindings()
                } else if prefills.is_none() {
                    self.state_bindings(segments)
                } else {
                    vec![]
                },
            ));
        }
        let layout = m.batch_layout.as_ref().ok_or("Missing Qwen row arena")?;
        let decoders = segments.iter().take_while(|s| s.tokens == 1).count();
        let parallel = prefills.is_none() && decoders > 0;
        let packet = parallel && m.programs.contains_key("qwen_rows/packet/begin");
        let (mut out, mut views) = (Vec::new(), Vec::new());
        let empty = BTreeMap::new();
        if packet {
            append(
                m,
                &mut out,
                "qwen_rows/packet/begin",
                decoders,
                None,
                &empty,
            )?;
        }
        for (lane, segment) in segments.iter().enumerate() {
            let v = crate::packet::views(
                rows.starts[lane],
                layout
                    .row_strides
                    .iter()
                    .map(|(name, &stride)| (name.as_str(), stride)),
            )?;
            if packet && lane < decoders {
                views.push(v);
                continue;
            }
            if segment.tokens == 1 {
                out.push(Invocation {
                    operation: Operation::Copy {
                        source: m.token.clone(),
                        destination: m.input.clone(),
                        bytes: 4,
                    },
                    sequence: Some(segment.slot),
                    launch: None,
                    views: BTreeMap::new(),
                });
            }
            append(
                m,
                &mut out,
                &format!("{PREFIX}/begin"),
                segment.tokens,
                Some(segment.slot),
                &v,
            )?;
            views.push(v);
        }
        for layer in 0..layout.layers.len() {
            append(
                m,
                &mut out,
                &format!("{PREFIX}/layer{layer}/pre"),
                total,
                None,
                &empty,
            )?;
            if parallel {
                append(
                    m,
                    &mut out,
                    &format!("{PREFIX}/state/layer{layer}"),
                    decoders,
                    None,
                    &empty,
                )?;
            }
            if let Some(table) = causal {
                append(
                    m,
                    &mut out,
                    &format!("{PREFIX}/verify_state/layer{layer}"),
                    segments.len() - prefills.unwrap(),
                    None,
                    &empty,
                )?;
                if layout.layers[layer] != "linear_attention" && !table.attention.is_empty() {
                    let stride = layout.layers.len() * table.columns.len() * 8;
                    let mut first = 0;
                    for (program, group) in &causal_groups {
                        let view = BTreeMap::from([(
                            table.buffer.clone(),
                            BufferView {
                                buffer: table.buffer.clone(),
                                offset: first * stride,
                            },
                        )]);
                        append(
                            m,
                            &mut out,
                            &format!("{PREFIX}/verify_state/layer{layer}/{program}"),
                            group.len(),
                            None,
                            &view,
                        )?;
                        first += group.len();
                    }
                }
            }
            for (lane, (segment, v)) in
                segments
                    .iter()
                    .zip(&views)
                    .enumerate()
                    .filter(|(lane, (s, _))| {
                        (!parallel || s.tokens != 1) && !(causal.is_some() && verified(*lane))
                    })
            {
                append(
                    m,
                    &mut out,
                    &if verified(lane) {
                        format!("{PREFIX}/verify/layer{layer}/private")
                    } else {
                        format!("{PREFIX}/layer{layer}/private")
                    },
                    segment.tokens,
                    Some(segment.slot),
                    v,
                )?;
            }
            append(
                m,
                &mut out,
                &format!("{PREFIX}/layer{layer}/post"),
                total,
                None,
                &empty,
            )?;
        }
        if packet {
            append(m, &mut out, "qwen_rows/packet/end", decoders, None, &empty)?;
        }
        for (lane, (segment, v)) in segments.iter().zip(&views).enumerate() {
            if packet && lane < decoders {
                continue;
            }
            append(
                m,
                &mut out,
                &format!("{PREFIX}/end"),
                segment.tokens,
                Some(segment.slot),
                v,
            )?;
            if m.mtp.is_some() {
                append(
                    m,
                    &mut out,
                    "capture",
                    segment.tokens,
                    Some(segment.slot),
                    v,
                )?;
            }
            let mut gather = v.clone();
            gather.insert(
                "HeadHidden".into(),
                crate::packet::view("HeadHidden", heads.starts[lane], layout.hidden * 2)?,
            );
            append(
                m,
                &mut out,
                &if verified(lane) {
                    format!("{PREFIX}/verify/gather")
                } else {
                    format!("{PREFIX}/gather")
                },
                segment.tokens,
                Some(segment.slot),
                &gather,
            )?;
        }
        append(
            m,
            &mut out,
            &format!("{PREFIX}/head"),
            head_rows,
            None,
            &empty,
        )?;
        if packet {
            append(
                m,
                &mut out,
                "qwen_rows/packet/outputs",
                decoders,
                None,
                &empty,
            )?;
        }
        for (lane, segment) in segments.iter().enumerate() {
            if packet && lane < decoders {
                continue;
            }
            let count = if verified(lane) { segment.tokens } else { 1 };
            let v = crate::packet::views(
                heads.starts[lane],
                [("HeadLogits", m.vocab * 4), ("Pairs", 8)],
            )?;
            out.push(Invocation {
                operation: Operation::Copy {
                    source: "HeadLogits".into(),
                    destination: if verified(lane) {
                        m.mtp.as_ref().unwrap().verification_logits.clone()
                    } else {
                        m.logits.clone()
                    },
                    bytes: count * m.vocab * 4,
                },
                sequence: Some(segment.slot),
                launch: None,
                views: v.clone(),
            });
            append(
                m,
                &mut out,
                &if verified(lane) {
                    format!("{PREFIX}/verify/select")
                } else {
                    format!("{PREFIX}/select")
                },
                count,
                Some(segment.slot),
                &v,
            )?;
        }
        Ok((
            out,
            if causal.is_some() {
                causal_bindings()
            } else if prefills.is_none() {
                self.state_bindings(segments)
            } else {
                vec![]
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn joined_sections_follow_weight_banks_before_kernel_contracts_are_materialized() {
        let mut m = request().metadata;
        m.kernels.clear();
        for role in ["self_attn.qkv_proj", "mlp.gate_up_proj"] {
            let mut buffer = m.buffers.iter().find(|b| b.data.is_some()).unwrap().clone();
            buffer.name = format!("model.language_model.layers.3.{role}.weight.packed");
            m.buffers.push(buffer);
        }
        let pre = section(&mut m, "layer3/pre", PRE_ATTN);
        assert_eq!(
            pre,
            kernels("qwen_rows/layer3/pre", &["norm_rotate", "qkv"])
        );
        let post = section(
            &mut m,
            "layer3/post",
            &["norm_rotate", "gate", "up", "swiglu_rotate", "down"],
        );
        assert_eq!(
            post,
            kernels(
                "qwen_rows/layer3/post",
                &["norm_rotate", "gate_up", "swiglu_rotate", "down"]
            )
        );
        let unjoined = section(&mut m, "layer7/pre", PRE_ATTN);
        assert_eq!(unjoined, kernels("qwen_rows/layer7/pre", PRE_ATTN));
    }

    fn request() -> CreateRequest {
        request_geometry(false)
    }

    #[test]
    fn decode_packet_boundaries_keep_constant_launches_and_owner_addresses() {
        for small in [false, true] {
            let mut request = request_geometry(small);
            let layers = request.config["text_config"]["num_hidden_layers"]
                .as_u64()
                .unwrap() as usize;
            let columns = request
                .metadata
                .batch_layout
                .as_ref()
                .unwrap()
                .state_columns
                .len();
            request
                .metadata
                .buffers
                .iter_mut()
                .find(|b| b.name == "QwenRowPointers")
                .unwrap()
                .shape[0] += LANES * PACKET_COLUMNS.len();
            for name in ["Input", "Token", "TokenStatus", "Step"] {
                request.metadata.buffers.push(
                    serde_json::from_value(json!({
                        "name":name,"dtype":"i32","shape":[4096],"layout":"native_contiguous",
                        "alignment":256,"access":"read_write"
                    }))
                    .unwrap(),
                );
            }
            request.kernel_contracts.push(serde_json::from_value(json!({
                "name":"qwen_rows/packet/tokens","symbol":"test","grid":[1,1,1],"block":[32,1,1],
                "shared_memory_bytes":0,"cooperative":false,"args":[],
                "module":{"file":"test","sha256":"0".repeat(64)},
                "source":{"file":"test","sha256":"0".repeat(64)},
                "host_abi":{"file":"test","sha256":"0".repeat(64)}
            })).unwrap());
            let (adapter, _) = create(request, Policy::Int8Quality).unwrap();
            let mut count = None;
            for batch in [1, 2, 8, 128] {
                let segments = (0..batch)
                    .map(|i| BatchSegment {
                        slot: 200 - i,
                        tokens: 1,
                    })
                    .collect::<Vec<_>>();
                let (calls, bindings) = adapter.batch(&segments, true).unwrap();
                assert_eq!(*count.get_or_insert(calls.len()), calls.len());
                assert!(calls.iter().all(
                    |c| c.sequence.is_none() && matches!(c.operation, Operation::Kernel { .. })
                ));
                assert!(
                    matches!(&calls[0].operation, Operation::Kernel { name } if name == "qwen_rows/packet/tokens")
                );
                assert!(
                    matches!(&calls.last().unwrap().operation, Operation::Kernel { name } if name == "qwen_rows/packet/outputs")
                );
                let base = LANES * layers * columns;
                assert_eq!(bindings.len(), base + LANES * PACKET_COLUMNS.len());
                for (lane, segment) in segments.iter().enumerate() {
                    assert_eq!(
                        bindings[base + lane * PACKET_COLUMNS.len() + 1],
                        Some((segment.slot, "Token".into()))
                    );
                }
                if batch < LANES {
                    assert!(
                        bindings[base + batch * PACKET_COLUMNS.len()..]
                            .iter()
                            .all(Option::is_none)
                    );
                }
                assert_eq!(adapter.batch(&segments, false).unwrap().1, bindings);
                if batch == LANES {
                    continue;
                }
                let mut mixed = segments.clone();
                mixed.insert(
                    0,
                    BatchSegment {
                        slot: 999,
                        tokens: 3,
                    },
                );
                let (calls, mixed_bindings) = adapter.batch(&mixed, true).unwrap();
                assert_eq!(bindings, mixed_bindings);
                assert!(
                    calls
                        .iter()
                        .all(|c| c.sequence.is_none_or(|slot| slot == 999))
                );
                let gather = calls
                    .iter()
                    .find(|c| {
                        matches!(&c.operation,
                    Operation::Kernel { name } if name == "qwen_rows/gather/norm")
                    })
                    .unwrap();
                assert_eq!(gather.sequence, Some(999));
                assert_eq!(
                    gather.views["HeadHidden"].offset,
                    batch * if small { 2048 } else { 5120 } * 2
                );
            }
        }
    }

    fn deferred_request() -> CreateRequest {
        let mut r = request_geometry(true);
        r.metadata.gdn_execution.as_mut().unwrap().decode =
            orinfer_model_sdk::operators::gdn::Decode::Deferred { capacity: 8 };
        let layers = r.config["text_config"]["layer_types"].as_array().unwrap();
        for (layer, kind) in layers.iter().enumerate() {
            if kind != "linear_attention" {
                continue;
            }
            for (suffix, dtype, shape) in [
                ("GdnLogCount", "i32", vec![16, 128]),
                ("GdnLogDelta", "f32", vec![8, 16, 128]),
                ("GdnLogGate", "f32", vec![8, 16]),
                ("GdnLogKey", "f16", vec![8, 16, 128]),
            ] {
                let name = format!("L{layer}_{suffix}");
                r.metadata.reset_buffers.push(name.clone());
                r.metadata.buffers.push(
                    serde_json::from_value(json!({
                        "name":name,"dtype":dtype,"shape":shape,"layout":"native_contiguous",
                        "alignment":256,"access":"read_write"
                    }))
                    .unwrap(),
                );
            }
        }
        let columns = &mut r.metadata.batch_layout.as_mut().unwrap().state_columns;
        columns.extend(DEFERRED_COLUMNS.iter().map(|s| (*s).to_owned()));
        columns.sort_unstable();
        r.metadata
            .buffers
            .iter_mut()
            .find(|b| b.name == "QwenRowPointers")
            .unwrap()
            .shape = vec![LANES * layers.len() * columns.len()];
        r
    }

    #[test]
    fn pending_gdn_updates_are_private_and_materialized_before_dense_execution() {
        let (adapter, plan) = create(deferred_request(), Policy::Int8Quality).unwrap();
        let m = &plan.metadata;
        let names = |program: &str| {
            m.programs[program]
                .iter()
                .filter_map(|o| match o {
                    Operation::Kernel { name } => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let private = names("qwen_rows/layer0/private");
        assert!(
            private
                .iter()
                .position(|n| n.ends_with("/recurrent_materialize"))
                .unwrap()
                < private
                    .iter()
                    .position(|n| n.ends_with("/recurrent"))
                    .unwrap()
        );
        let parallel = names("qwen_rows/state/layer0");
        assert_eq!(
            parallel
                .iter()
                .filter(|n| n.ends_with("/recurrent"))
                .count(),
            1
        );
        let (_, state) = adapter
            .batch(&[BatchSegment { slot: 3, tokens: 1 }], true)
            .unwrap();
        assert!(state.iter().any(|s| {
            s.as_ref()
                .is_some_and(|(slot, name)| *slot == 3 && name == "L0_GdnLogDelta")
        }));
        let (_, speculative) = create(with_mtp(deferred_request()), Policy::Int8Quality).unwrap();
        let verify = &speculative.metadata.programs["qwen_rows/verify/layer0/private"];
        let materialize = verify
            .iter()
            .position(|o| {
                matches!(o, Operation::Kernel { name }
            if name.ends_with("/recurrent_materialize"))
            })
            .unwrap();
        let recurrent = verify
            .iter()
            .position(|o| {
                matches!(o, Operation::Kernel { name }
            if name.ends_with("/recurrent"))
            })
            .unwrap();
        assert!(materialize < recurrent);
        let mut bad = deferred_request();
        bad.metadata.reset_buffers.retain(|n| n != "L0_GdnLogCount");
        assert!(create(bad, Policy::Int8Quality).is_err());
        let mut bad = deferred_request();
        bad.metadata
            .buffers
            .iter_mut()
            .find(|b| b.name == "L0_GdnLogDelta")
            .unwrap()
            .dtype = Dtype::F16;
        assert!(create(bad, Policy::Int8Quality).is_err());
    }

    fn request_geometry(small: bool) -> CreateRequest {
        let dimensions = if small {
            (2048, 6144, 24, 16, 8, 2, true)
        } else {
            (5120, 17408, 64, 48, 24, 4, false)
        };
        dense_request(dimensions, small)
    }

    #[test]
    fn attention_staging_is_shared_scratch_with_request_owned_inputs() {
        let staged = || {
            let mut request = request_geometry(true);
            for name in ["AttentionDecodedK", "AttentionDecodedV"] {
                request.metadata.buffers.push(
                    serde_json::from_value(json!({
                        "name":name,"dtype":"f16","shape":[request.metadata.max_context,512],
                        "layout":"native_contiguous","alignment":256,"access":"read_write"
                    }))
                    .unwrap(),
                );
            }
            request
        };
        let (_, plan) = create(staged(), Policy::Int8Quality).unwrap();
        let private = &plan.metadata.programs["qwen_rows/layer3/private"];
        let roles = private
            .iter()
            .filter_map(|op| match op {
                Operation::Kernel { name } => name.rsplit('/').next(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            [
                "prepare",
                "attention_decode",
                "attention",
                "attention_merge"
            ]
        );
        assert!(
            !plan.metadata.programs["qwen_rows/state/layer3"].iter().any(
                |op| matches!(op,Operation::Kernel{name} if name.ends_with("attention_decode"))
            )
        );
        let mut missing = staged();
        missing
            .metadata
            .buffers
            .retain(|b| b.name != "AttentionDecodedV");
        assert!(create(missing, Policy::Int8Quality).is_err());
        let mut request = staged();
        request
            .metadata
            .reset_buffers
            .push("AttentionDecodedK".into());
        assert!(create(request, Policy::Int8Quality).is_err());
    }

    #[test]
    fn gdn_execution_is_required_and_cannot_be_inferred_from_buffers() {
        let mut missing = request();
        missing.metadata.gdn_execution = None;
        assert!(create(missing, Policy::Int8Quality).is_err());
        let mut mismatch = deferred_request();
        mismatch.metadata.gdn_execution.as_mut().unwrap().decode =
            orinfer_model_sdk::operators::gdn::Decode::Dense;
        assert!(create(mismatch, Policy::Int8Quality).is_err());
        let mut capacity = deferred_request();
        capacity.metadata.gdn_execution.as_mut().unwrap().decode =
            orinfer_model_sdk::operators::gdn::Decode::Deferred { capacity: 4 };
        assert!(create(capacity, Policy::Int8Quality).is_err());
    }

    #[test]
    fn request_parallel_history_update_precedes_recurrence_when_bound() {
        for separate_history in [false, true] {
            let mut request = request();
            if separate_history {
                let asset = json!({"file":"test", "sha256":"0".repeat(64)});
                request.kernel_contracts.push(
                    serde_json::from_value(json!({
                        "name":"qwen_rows/state/layer0/history",
                        "module":asset,"source":asset,"host_abi":asset,
                        "symbol":"history","grid":[1,1,1],"block":[1,1,1],
                        "shared_memory_bytes":0,"cooperative":false,"args":[]
                    }))
                    .unwrap(),
                );
            }
            let (adapter, _) = create(request, Policy::Int8Quality).unwrap();
            let segments = [
                BatchSegment { slot: 7, tokens: 1 },
                BatchSegment { slot: 2, tokens: 1 },
            ];
            let (calls, _) = adapter.batch(&segments, true).unwrap();
            let state: Vec<_> = calls
                .iter()
                .filter_map(|call| match &call.operation {
                    Operation::Kernel { name } => name.strip_prefix("qwen_rows/state/layer0/"),
                    _ => None,
                })
                .collect();
            let expected = if separate_history {
                vec!["gates", "conv", "history", "recurrent", "gated_norm"]
            } else {
                vec!["gates", "conv", "recurrent", "gated_norm"]
            };
            assert_eq!(state, expected);
        }
    }

    #[test]
    fn chunked_gdn_prefill_preserves_sequential_decode_and_verification() {
        let mut request = with_mtp(dense_request((2048, 6144, 24, 16, 8, 2, true), true));
        request.metadata.gdn_execution.as_mut().unwrap().prefill =
            orinfer_model_sdk::operators::gdn::Prefill::Chunked;
        for (name, dtype, width) in [
            ("GdnChunkW", "f16", 128),
            ("GdnChunkU", "f16", 128),
            ("GdnChunkD", "f16", 32),
            ("GdnChunkG", "f32", 2),
        ] {
            request.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":dtype,"shape":[16,4096,width],
                    "layout":"native_contiguous","alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        let mut invalid: CreateRequest =
            serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        invalid.metadata.buffers.retain(|b| b.name != "GdnChunkG");
        assert!(create(invalid, Policy::Int8Quality).is_err());
        let (adapter, plan) = create(request, Policy::Int8Quality).unwrap();
        let names = |program: &str| {
            plan.metadata.programs[program]
                .iter()
                .filter_map(|op| match op {
                    Operation::Kernel { name } => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let prefill = names("prefill");
        assert_eq!(
            prefill
                .iter()
                .filter(|n| n.ends_with("/recurrent_prepare"))
                .count(),
            18
        );
        assert_eq!(
            prefill
                .iter()
                .filter(|n| n.ends_with("/recurrent_chunk"))
                .count(),
            18
        );
        for program in ["decode", "verify"] {
            let calls = names(program);
            assert!(calls.iter().all(|n| !n.ends_with("/recurrent_prepare") && !n.ends_with("/recurrent_chunk")));
        }
        let (calls, _) = adapter
            .batch(&[BatchSegment { slot: 0, tokens: 1 }], false)
            .unwrap();
        assert!(calls.iter().all(|call| match &call.operation {
            Operation::Kernel { name } => !name.ends_with("/recurrent_chunk"),
            _ => true,
        }));
    }

    #[test]
    fn four_billion_reuses_tied_dense_row_and_mtp_programs() {
        let request = with_mtp(dense_request((2560, 9216, 32, 32, 16, 4, true), true));
        let (adapter, plan) = create(request, Policy::Int8Quality).unwrap();
        let layout = plan.metadata.batch_layout.as_ref().unwrap();
        assert_eq!(layout.hidden, 2560);
        assert_eq!(plan.metadata.prefill_plans[0].stages.len(), 32);
        assert!(plan.metadata.programs.contains_key("mtp_warm"));
        let (calls, _) = adapter
            .batch(
                &[
                    BatchSegment { slot: 2, tokens: 1 },
                    BatchSegment { slot: 5, tokens: 3 },
                ],
                true,
            )
            .unwrap();
        assert!(calls.iter().any(|c| c.sequence == Some(5)));
    }

    fn dense_request(
        (h, f, layers, hv, qh, kvh, tied): (usize, usize, usize, usize, usize, usize, bool),
        e8p: bool,
    ) -> CreateRequest {
        let mut config = json!({"model_type":"qwen3_5","text_config":{"hidden_size":h,
            "intermediate_size":f,"num_hidden_layers":layers,"vocab_size":248320,
            "linear_num_key_heads":16,"linear_num_value_heads":hv,"linear_key_head_dim":128,
            "linear_value_head_dim":128,"linear_conv_kernel_dim":4,
            "num_attention_heads":qh,"num_key_value_heads":kvh,"head_dim":256,"tie_word_embeddings":tied,
            "layer_types":(0..layers).map(|i|if i%4==3 {"full_attention"} else {"linear_attention"}).collect::<Vec<_>>()},
            "quantization_config":{"quant_method":"orinfer_q4_int8","version":1,"rotation":"signed-block128",
            "scale":"fp16-output-row","compute_dtype":"int8_quality"}});
        if e8p {
            config["quantization_config"]["quant_method"] = json!("orinfer_e8p_q4_int8");
            config["quantization_config"]["coefficients"] = json!([4, 1]);
            config["quantization_config"]["basis_tensor"] = json!("orinfer.e8p_q4.basis");
            config["quantization_config"]["layout"] = json!("u32-low-coarse-high-residual");
            config["quantization_config"]["embedding"] = json!("e8p_q4");
        }
        let signature = Configuration::parse(config.clone()).unwrap().signature;
        let mut buffers = vec![];
        let mut q4 = |name: String, n: usize, k: usize| {
            for (suffix, dtype, shape) in [
                (
                    "packed",
                    if e8p { "u32" } else { "u8" },
                    if e8p {
                        vec![k / 128, n, 16]
                    } else {
                        vec![n, k / 2]
                    },
                ),
                ("scale", "f16", vec![n]),
            ] {
                buffers.push(
                    json!({"name":format!("{name}.{suffix}"),"dtype":dtype,"shape":shape,
                    "layout":if e8p && suffix=="packed" {"e8p-q4-k128-u32"} else {"native_contiguous"},"alignment":256,"access":"read",
                    "data":{"tensor":format!("{name}.{suffix}"),"sha256":"0".repeat(64)}}),
                );
            }
        };
        q4(
            if tied {
                "model.language_model.embed_tokens.weight"
            } else {
                "lm_head.weight"
            }
            .into(),
            248320,
            h,
        );
        if e8p && !tied {
            q4("model.language_model.embed_tokens.weight".into(), 248320, h);
        }
        for i in 0..layers {
            let p = format!("model.language_model.layers.{i}");
            for (name, n, k) in [("gate_proj", f, h), ("up_proj", f, h), ("down_proj", h, f)] {
                q4(format!("{p}.mlp.{name}.weight"), n, k);
            }
            let projections: &[(&str, usize)] = if i % 4 == 3 {
                &[
                    ("self_attn.q_proj", 2 * qh * 256),
                    ("self_attn.k_proj", kvh * 256),
                    ("self_attn.v_proj", kvh * 256),
                ]
            } else {
                &[
                    ("linear_attn.in_proj_qkv", (32 + hv) * 128),
                    ("linear_attn.in_proj_z", hv * 128),
                ]
            };
            for (name, n) in projections {
                q4(format!("{p}.{name}.weight"), *n, h);
            }
        }
        if e8p {
            for (name, dtype, shape) in [
                ("basis", "i8", vec![256, 8]),
                ("selectors", "u32", vec![256, 1]),
            ] {
                buffers.push(
                    json!({"name":format!("orinfer.e8p_q4.{name}"),"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,"access":"read",
                    "data":{"tensor":format!("orinfer.e8p_q4.{name}"),"sha256":"0".repeat(64)}}),
                );
            }
        }
        buffers.extend([
            json!({"name":"Hidden","dtype":"f16","shape":[4096,h],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
            json!({"name":"R0","dtype":"f32","shape":[4096,h],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
            json!({"name":"QwenRowPointers","dtype":"u64","shape":[128*layers*STATE_COLUMNS.len()],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
            json!({"name":"QwenRowScratch","dtype":"u8","shape":[256],"layout":"native_contiguous","alignment":256,"access":"read_write"}),
        ]);
        serde_json::from_value(json!({"config":config,"expected_signature":signature,
            "architecture":"qwen3_5","compute_policy":"int8_quality",
            "prefill_profiles":[{"tokens":4096,"min_tokens":1,"kind":"qwen_q4_rows"}],
            "metadata":{"gdn_execution":{"prefill":"sequential","decode":{"kind":"dense"}},"schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,"max_context":262144,
                "vocab":248320,"toolchain":{},"buffers":buffers,"reset_buffers":["Step"],"input":"Input","token":"Token",
                "status":"TokenStatus","logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
                "batch_layout":{"layers":[],"row_strides":{},"profiles":{},"hidden":h,"state_columns":STATE_COLUMNS},
                "kv_cache":{"direct_prefill":true,"demand_mapping":true,"buffers":{},"prefill_workspace":{}},
                "dynamic_batch_kernels":{"qwen_rows/layer0/pre/norm_rotate":{"name":"qwen_rows/layer0/pre/norm_rotate","capacity":4096,
                    "grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],"arguments":[]}}}})).unwrap()
    }

    #[test]
    fn moe_reuses_dense_causal_rows_with_compressed_expert_banks() {
        let mut request = dense_request((2048, 512, 40, 32, 16, 2, false), true);
        request.config["model_type"] = json!("qwen3_5_moe");
        for (key, value) in [
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
            ("moe_intermediate_size", 512),
            ("shared_expert_intermediate_size", 512),
        ] {
            request.config["text_config"][key] = json!(value);
        }
        request.expected_signature = Configuration::parse(request.config.clone())
            .unwrap()
            .signature;
        for b in &mut request.metadata.buffers {
            if b.name.contains(".mlp.") {
                b.name = b.name.replace(".mlp.", ".mlp.shared_expert.");
                b.data.as_mut().unwrap().tensor = b.name.clone();
            }
        }
        for layer in 0..40 {
            for (role, n, k) in [("gate_up_proj", 1024, 2048), ("down_proj", 2048, 512)] {
                let name = format!("model.language_model.layers.{layer}.mlp.experts.{role}");
                for (suffix, dtype, shape, layout) in [
                    (
                        "packed",
                        "u32",
                        vec![256, k / 128, n, 16],
                        "e8p-q4-k128-u32",
                    ),
                    ("scale", "f16", vec![256, n], "native_contiguous"),
                ] {
                    request.metadata.buffers.push(
                        serde_json::from_value(json!({"name":format!("{name}.{suffix}"),
                        "dtype":dtype,"shape":shape,"layout":layout,"alignment":256,"access":"read",
                        "data":{"tensor":format!("{name}.{suffix}"),"sha256":"0".repeat(64)}}))
                        .unwrap(),
                    );
                }
            }
        }
        request.metadata.buffers.push(
            serde_json::from_value(json!({
                "name":"MoeLogits","dtype":"f32","shape":[4096,256],
                "layout":"native_contiguous","alignment":256,"access":"read_write"
            }))
            .unwrap(),
        );
        let copied = serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        let (adapter, plan) = create(with_mtp(copied), Policy::Int8Quality).unwrap();
        assert_eq!(plan.metadata.prefill_plans[0].stages.len(), 40);
        let post = &plan.metadata.programs["qwen_rows/layer0/post"];
        assert!(
            post.iter()
                .any(|op| matches!(op,Operation::Kernel{name} if name.ends_with("/expert_down")))
        );
        assert!(
            post.iter()
                .any(|op| matches!(op,Operation::Kernel{name} if name.ends_with("/combine")))
        );
        let (calls, _) = adapter
            .batch_operation(
                &[
                    BatchSegment { slot: 7, tokens: 3 },
                    BatchSegment { slot: 2, tokens: 2 },
                ],
                abi::BatchOperation::Warm,
                true,
            )
            .unwrap();
        let experts: Vec<_> = calls
            .iter()
            .filter(|c| {
                matches!(&c.operation,Operation::Kernel{name}
            if name=="qwen_rows/mtp/expert_up")
            })
            .collect();
        assert_eq!(experts.len(), 1);
        assert_eq!(experts[0].sequence, None);
        let conditions: Vec<_> = calls
            .iter()
            .filter_map(|c| {
                matches!(&c.operation,
            Operation::Copy{destination,..} if destination=="MtpCondition")
                .then_some(c.sequence)
            })
            .collect();
        assert_eq!(conditions, [Some(7), Some(2)]);
        let mut planes: CreateRequest =
            serde_json::from_value(serde_json::to_value(&request).unwrap()).unwrap();
        for (name, dtype, size) in [
            ("MoeLow", "i8", 32768 * 2048 / 2),
            ("MoeHigh", "i8", 32768 * 2048 / 2),
            ("MoeSum", "i32", 32768),
        ] {
            planes.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":dtype,"shape":[size],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        let copied = serde_json::from_value(serde_json::to_value(&planes).unwrap()).unwrap();
        let (_, plan) = create(with_mtp(copied), Policy::Int8Quality).unwrap();
        let post = &plan.metadata.programs["qwen_rows/layer0/post"];
        for role in ["expert_up", "expert_down"] {
            let index = |suffix: &str| {
                post.iter()
                    .position(|op| matches!(op,Operation::Kernel{name} if name.ends_with(suffix)))
                    .unwrap()
            };
            assert!(index(&format!("/{role}_planes")) < index(&format!("/{role}")));
        }
        assert!(
            plan.metadata.programs["mtp_post"]
                .iter()
                .all(|op| { !matches!(op,Operation::Kernel{name} if name.ends_with("_planes")) })
        );
        let mut partition: CreateRequest =
            serde_json::from_value(serde_json::to_value(&planes).unwrap()).unwrap();
        for (name, size) in [
            ("MoeColdTileOffsets", 256),
            ("MoeColdTileCount", 1),
            ("MoeColdTileExpert", 1280),
            ("MoeColdTileRow", 1280),
        ] {
            partition.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":"i32","shape":[size],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        let copied = serde_json::from_value(serde_json::to_value(&partition).unwrap()).unwrap();
        let (_, plan) = create(with_mtp(copied), Policy::Int8Quality).unwrap();
        let post = &plan.metadata.programs["qwen_rows/layer0/post"];
        for role in ["expert_up", "expert_down"] {
            let index = |suffix: &str| {
                post.iter()
                    .position(|op| matches!(op,Operation::Kernel{name} if name.ends_with(suffix)))
                    .unwrap()
            };
            assert!(index(&format!("/{role}_cold_planes")) < index(&format!("/{role}_cold")));
            assert!(index(&format!("/{role}_cold")) < index(&format!("/{role}")));
        }
        assert!(
            plan.metadata.programs["mtp_post"]
                .iter()
                .all(|op| { !matches!(op,Operation::Kernel{name} if name.contains("_cold")) })
        );
        partition
            .metadata
            .buffers
            .retain(|b| b.name != "MoeColdTileRow");
        assert!(create(partition, Policy::Int8Quality).is_err());
        planes.metadata.buffers.retain(|b| b.name != "MoeSum");
        assert!(create(planes, Policy::Int8Quality).is_err());
        request
            .metadata
            .buffers
            .retain(|b| !b.name.ends_with("layers.0.mlp.experts.down_proj.scale"));
        assert!(create(request, Policy::Int8Quality).is_err());
    }

    #[test]
    fn medium_dense_reuses_rows_with_separate_embedding_and_head_banks() {
        let request = dense_request((4096, 12288, 32, 32, 16, 4, false), true);
        let (adapter, plan) = create(request, Policy::Int8Quality).unwrap();
        let m = &plan.metadata;
        assert_eq!(m.prefill_plans[0].stages.len(), 32);
        let layout = m.batch_layout.as_ref().unwrap();
        assert_eq!(layout.hidden, 4096);
        assert_eq!(layout.row_strides["QKV"], 8192 * 2);
        assert_eq!(layout.row_strides["FullX"], 10240 * 2);
        for name in ["model.language_model.embed_tokens.weight", "lm_head.weight"] {
            assert!(m.buffers.iter().any(|b| b.name == format!("{name}.packed")));
        }
        let (calls, _) = adapter
            .batch(
                &[
                    BatchSegment { slot: 2, tokens: 1 },
                    BatchSegment { slot: 5, tokens: 3 },
                ],
                true,
            )
            .unwrap();
        assert!(calls.iter().any(|c| c.sequence == Some(5)));
        let mut wrong = dense_request((4096, 12288, 32, 32, 16, 4, false), true);
        wrong
            .metadata
            .buffers
            .retain(|b| b.name != "lm_head.weight.scale");
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }

    #[test]
    fn small_dense_reuses_row_state_and_tied_head_without_duplicate_weight() {
        let (adapter, plan) = create(request_geometry(true), Policy::Int8Quality).unwrap();
        let m = &plan.metadata;
        assert_eq!(m.prefill_plans[0].stages.len(), 24);
        let layout = m.batch_layout.as_ref().unwrap();
        assert_eq!(layout.hidden, 2048);
        assert_eq!(layout.row_strides["QKV"], 6144 * 2);
        assert!(!m.buffers.iter().any(|b| b.name.starts_with("lm_head.")));
        let (calls, _) = adapter
            .batch(
                &[
                    BatchSegment { slot: 3, tokens: 1 },
                    BatchSegment { slot: 7, tokens: 2 },
                ],
                true,
            )
            .unwrap();
        assert!(calls.iter().any(|c| c.sequence == Some(7)));
        let mut wrong = request_geometry(true);
        wrong
            .metadata
            .buffers
            .retain(|b| b.name != "model.language_model.embed_tokens.weight.scale");
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }

    #[test]
    fn quantized_vision_inserts_shared_projection_stages_before_the_epilogue() {
        let mut m = request_geometry(true).metadata;
        m.vision = Some(serde_json::from_value(json!({
            "hidden":2048,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
            "max_patches":32768,"max_features":8192,"pixels":"Pixels","grid":"Grid",
            "length":"Length","output":"Output","features":"Features","feature_index":"Index",
            "mrope_positions":"MRope","image_token_id":248056,"vision_start_token_id":248053,
            "vision_end_token_id":248054,"plans":[{"patches":32768,"min_patches":4,"program":"vision"}]
        })).unwrap());
        let mut packed = m.buffers[0].clone();
        packed.name = "model.visual.blocks.0.attn.qkv.weight.packed".into();
        m.buffers.push(packed);
        vision(&mut m, 24, 2048).unwrap();
        let names: Vec<_> = m.programs["vision"]
            .iter()
            .filter_map(|op| match op {
                Operation::Kernel { name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            &names[3..7],
            &[
                "qwen_rows/vision/layer0/qkv_rotate",
                "qwen_rows/vision/layer0/qkv_project",
                "qwen_rows/vision/layer0/qkv",
                "qwen_rows/vision/layer0/rope",
            ]
        );
        assert!(!names.contains(&"qwen_rows/vision/layer1/qkv_project"));
    }

    #[test]
    fn two_stage_codec_reuses_rows_and_rejects_scalar_storage() {
        let candidate = || {
            let mut r = request();
            r.config["quantization_config"]["quant_method"] = json!("orinfer_e8p_q4_int8");
            r.config["quantization_config"]["coefficients"] = json!([4, 1]);
            r.config["quantization_config"]["basis_tensor"] = json!("orinfer.e8p_q4.basis");
            r.config["quantization_config"]["layout"] = json!("u32-low-coarse-high-residual");
            r.expected_signature = Configuration::parse(r.config.clone()).unwrap().signature;
            for b in &mut r.metadata.buffers {
                if b.name.ends_with(".packed") {
                    b.dtype = Dtype::U32;
                    b.shape = vec![b.shape[1] * 2 / 128, b.shape[0], 16];
                    b.layout = "e8p-q4-k128-u32".into();
                }
            }
            for (name, dtype, shape) in [
                ("orinfer.e8p_q4.basis", "i8", vec![256, 8]),
                ("orinfer.e8p_q4.selectors", "u32", vec![256, 1]),
            ] {
                r.metadata.buffers.push(
                    serde_json::from_value(json!({
                        "name": name, "dtype": dtype, "shape": shape,
                        "layout": "native_contiguous", "alignment": 256, "access": "read",
                        "data": {"tensor": name, "sha256": "0".repeat(64)}
                    }))
                    .unwrap(),
                );
            }
            r
        };
        let (adapter, _) = create(candidate(), Policy::Int8Quality).unwrap();
        assert!(
            adapter
                .batch(&[BatchSegment { slot: 0, tokens: 1 }], false)
                .is_ok()
        );
        let mut wrong = candidate();
        wrong.metadata.buffers[0].dtype = Dtype::U8;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        let packed = wrong
            .metadata
            .buffers
            .iter_mut()
            .find(|b| b.name.ends_with(".packed"))
            .unwrap();
        packed.layout = "e8p-q4-row-u32".into();
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        let packed = wrong
            .metadata
            .buffers
            .iter_mut()
            .find(|b| b.name.ends_with(".packed"))
            .unwrap();
        packed.shape.swap(0, 1);
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = candidate();
        wrong.config["quantization_config"]["coefficients"] = json!([3, 1]);
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }

    #[test]
    fn mixed_rows_share_computation_without_padding_and_keep_private_owners() {
        let (adapter, plan) = create(request(), Policy::Int8Quality).unwrap();
        let segments = [
            BatchSegment { slot: 7, tokens: 1 },
            BatchSegment {
                slot: 2,
                tokens: 17,
            },
        ];
        let (calls, _) = adapter.batch(&segments, true).unwrap();
        let shared:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/pre/norm_rotate")).collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].sequence, None);
        assert_eq!(shared[0].launch.as_ref().unwrap().grid[0], 18);
        let conv:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/private/conv")).collect();
        assert_eq!(
            conv.iter().map(|call| call.sequence).collect::<Vec<_>>(),
            [Some(2)]
        );
        assert_eq!(conv[0].views["QKV"].offset, 10240 * 2);
        assert!(!conv[0].views.contains_key("Q")); // compact causal scratch is reused, never offset as token-major data
        assert!(calls.iter().any(|call| call.sequence.is_none() && matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/state/layer0/conv")));
        let gather:Vec<_>=calls.iter().filter(|call|matches!(&call.operation,Operation::Kernel{name} if name=="qwen_rows/gather/norm")).collect();
        assert_eq!(gather[1].views["HeadHidden"].offset, 5120 * 2);
        assert_eq!(plan.metadata.prefill_plans.len(), 1);
        assert!(
            plan.metadata.programs["head"].iter().any(
                |op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head_last/project")
            )
        );
        assert!(
            !plan.metadata.programs["head"]
                .iter()
                .any(|op| matches!(op,Operation::Kernel{name} if name=="qwen_rows/head/project"))
        );
    }

    #[test]
    fn private_pointer_columns_are_a_pinned_semantic_contract() {
        let mut r = request();
        r.metadata
            .batch_layout
            .as_mut()
            .unwrap()
            .state_columns
            .clear();
        assert!(create(r, Policy::Int8Quality).is_err());
        let mut r = request();
        r.metadata.batch_layout.as_mut().unwrap().state_columns[0] = "DifferentHistory".into();
        assert!(create(r, Policy::Int8Quality).is_err());
    }

    #[test]
    fn stages_preserve_the_full_forward_and_only_shared_live_values() {
        let (_, plan) = create(request(), Policy::Int8Quality).unwrap();
        let m = plan.metadata;
        let p = &m.prefill_plans[0];
        assert_eq!(p.stages.len(), 64);
        assert_eq!(p.live_buffers, ["Hidden", "R0"]);
        let joined: Vec<_> = p
            .stages
            .iter()
            .flat_map(|s| m.programs[s].clone())
            .collect();
        assert_eq!(joined, m.programs["prefill"]);
        for s in &p.stages {
            assert_eq!(m.row_execution.as_ref().unwrap().program_capacity(s), 4096);
            assert!(m.kv_cache.as_ref().unwrap().growth.contains_key(s));
        }
    }

    fn mtp_request() -> CreateRequest {
        with_mtp(request())
    }

    fn with_mtp(mut r: CreateRequest) -> CreateRequest {
        let configuration = Configuration::parse(r.config.clone()).unwrap();
        let t = configuration.text();
        for (name, width, dtype) in [
            ("MtpEmbedding", t.hidden_size, "f16"),
            ("MtpConcat", 2 * t.hidden_size, "f16"),
            ("MtpHidden", t.hidden_size, "f16"),
            ("MtpNorm", t.hidden_size, "f16"),
            ("MtpR0", t.hidden_size, "f32"),
            ("MtpR1", t.hidden_size, "f32"),
            ("MtpFullX", t.full_x(), "f16"),
            ("MtpFullQ", t.mixer(), "f16"),
            ("MtpFullGate", t.mixer(), "f16"),
            ("MtpMixer", t.mixer(), "f16"),
            ("MtpMix", t.hidden_size, "f16"),
            ("MtpGateUp", 2 * t.intermediate_size, "f16"),
        ] {
            r.metadata.buffers.push(
                serde_json::from_value(json!({
                    "name":name,"dtype":dtype,"shape":[512,width],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write"
                }))
                .unwrap(),
            );
        }
        r.metadata.mtp=Some(serde_json::from_value(json!({
            "position":"MtpStep","input":"MtpInput","token":"MtpToken","status":"MtpStatus",
            "verification_tokens":"SequenceTokens","verification_status":"SequenceStatus",
            "draft_logits":"MtpLogits","verification_logits":"SequenceLogits","feature_index":"MtpFeatureIndex",
            "accepted_inputs":"AcceptedInputs","target_length":"SeqLength","draft_program":"mtp_draft",
            "hidden_ring":"MtpTargetHidden","commit_always":true,"shared_verification_state":true,
            "default_verification_tokens":4,
            "warm_plans":[{"tokens":512,"min_tokens":1,"program":"mtp_warm","head_program":"mtp_head"}],
            "capture_plans":[{"tokens":4096,"program":"capture"}],
            "verification_plans":[{"tokens":8,"min_tokens":2,"program":"verify","restore_program":"restore","capture_program":"capture"}]
        })).unwrap());
        r
    }

    #[test]
    fn owner_parallel_commit_keeps_acceptance_and_compact_stride_private() {
        let mut r = mtp_request();
        let layers = r.config["text_config"]["num_hidden_layers"]
            .as_u64()
            .unwrap() as usize;
        for name in ["AcceptedInputs", "SegmentLength"] {
            if !r.metadata.buffers.iter().any(|b| b.name == name) {
                r.metadata.buffers.push(serde_json::from_value(json!({"name":name,"dtype":"i32","shape":[1],"layout":"native_contiguous","alignment":256,"access":"read_write"})).unwrap());
            }
        }
        r.metadata.buffers.push(serde_json::from_value(json!({"name":"QwenCommitPointers","dtype":"u64","shape":[128*layers*8],"layout":"native_contiguous","alignment":256,"access":"read_write"})).unwrap());
        let (a, plan) = create(r, Policy::Int8Quality).unwrap();
        assert_eq!(
            plan.metadata
                .mtp
                .as_ref()
                .unwrap()
                .commit_state_table
                .as_deref(),
            Some("QwenCommitPointers")
        );
        let segments = [
            BatchSegment { slot: 7, tokens: 3 },
            BatchSegment {
                slot: 31,
                tokens: 8,
            },
        ];
        let (calls, bindings) = a
            .batch_operation(&segments, abi::BatchOperation::Commit, true)
            .unwrap();
        assert!(calls.iter().all(|c| c.sequence.is_none()));
        assert!(calls.iter().any(|c|matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/commit/layer0/state")));
        assert_eq!(bindings.len(), 2 * layers * 8);
        assert_eq!(bindings[3], Some((7, "AcceptedInputs".into())));
        assert_eq!(bindings[4], Some((7, "SegmentLength".into())));
        assert_eq!(
            bindings[layers * 8 + 3],
            Some((31, "AcceptedInputs".into()))
        );
        assert_eq!(bindings[layers * 8 + 4], Some((31, "SegmentLength".into())));
    }

    #[test]
    fn mixed_roles_are_explicit_and_verification_heads_keep_all_positions() {
        let (a, _) = create(mtp_request(), Policy::Int8Quality).unwrap();
        // Equal lengths have different semantics; they cannot identify roles.
        let segments = [
            BatchSegment { slot: 7, tokens: 2 },
            BatchSegment { slot: 3, tokens: 2 },
        ];
        let (calls, _) = a
            .batch_operation(
                &segments,
                abi::BatchOperation::MixedRows { prefills: 1 },
                true,
            )
            .unwrap();
        let prefix:Vec<_>=calls.iter().filter(|c|matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/verify/layer0/history")).map(|c|c.sequence).collect();
        assert_eq!(prefix, [Some(3)]);
        for (owner, destination, count) in [(7, "Logits", 1), (3, "SequenceLogits", 2)] {
            assert!(calls.iter().any(|c| c.sequence==Some(owner) && matches!(&c.operation,Operation::Copy{destination:d,bytes,..} if d==destination && *bytes==count*248320*4)));
        }
        let shared:Vec<_>=calls.iter().filter(|c|matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/layer0/pre/qkv")).collect();
        assert_eq!(shared.len(), 1);
        assert_eq!(shared[0].sequence, None);
        assert!(
            a.batch_operation(
                &segments,
                abi::BatchOperation::MixedRows { prefills: 0 },
                true
            )
            .is_err()
        );
        assert!(
            a.batch_operation(
                &[BatchSegment { slot: 1, tokens: 1 }],
                abi::BatchOperation::Verification,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn indirect_packet_rebinds_owners_and_preserves_scalar_launches_and_views() {
        let (_, built) = create(mtp_request(), Policy::Int8Quality).unwrap();
        let mut m = built.metadata;
        m.packet_table = Some(serde_json::from_value(json!({
            "buffer":"PacketPointers", "columns":["State","SequenceLogits"],
            "kernels":{"private":"indirect"},
            "scalar_arguments":{"private":[[0,2]]},
            "copies":[{"source":"HeadLogits","destination":"SequenceLogits","kernel":"copy","row_bytes":400}]
        })).unwrap());
        let kernel = |name, args| {
            serde_json::from_value(json!({
            "name":name,"module":{"file":"x","sha256":"0".repeat(64)},
            "source":{"file":"x","sha256":"0".repeat(64)},
            "host_abi":{"file":"x","sha256":"0".repeat(64)},"symbol":"kernel",
            "grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":args
        })).unwrap()
        };
        let contracts = vec![
            kernel(
                "private",
                json!([{"kind":"i32","value":8},{"kind":"buffer","name":"State"}]),
            ),
            kernel(
                "indirect",
                json!([{"kind":"buffer","name":"PacketPointers"},{"kind":"i32","value":0},{"kind":"i32","value":8}]),
            ),
        ];
        m.dynamic_batch_kernels.insert("copy".into(), serde_json::from_value(json!({
            "name":"copy","capacity":8,"grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],
            "arguments":[{"index":2,"value":{"op":"rows"}}]
        })).unwrap());
        let lower = |m: &Manifest, slots: [usize; 2]| {
            let segments = [
                BatchSegment {
                    slot: slots[0],
                    tokens: 3,
                },
                BatchSegment {
                    slot: slots[1],
                    tokens: 8,
                },
            ];
            let plan = vec![
                Invocation {
                    operation: Operation::Kernel {
                        name: "private".into(),
                    },
                    sequence: Some(slots[0]),
                    views: BTreeMap::new(),
                    launch: Some(orinfer_model_sdk::operators::dynamic::RowLaunch {
                        grid: [3, 1, 1],
                        arguments: vec![(0, 3)],
                    }),
                },
                Invocation {
                    operation: Operation::Copy {
                        source: "HeadLogits".into(),
                        destination: "SequenceLogits".into(),
                        bytes: 800,
                    },
                    sequence: Some(slots[1]),
                    views: BTreeMap::from([(
                        "HeadLogits".into(),
                        BufferView {
                            buffer: "HeadLogits".into(),
                            offset: 1200,
                        },
                    )]),
                    launch: None,
                },
                Invocation {
                    operation: Operation::Copy {
                        source: "SharedSource".into(),
                        destination: "SharedDestination".into(),
                        bytes: 32,
                    },
                    sequence: Some(slots[1]),
                    views: BTreeMap::new(),
                    launch: None,
                },
            ];
            crate::packet::indirect(
                m,
                &segments,
                &contracts,
                (plan, vec![Some((slots[0], "ExistingState".into()))]),
            )
            .unwrap()
        };
        let (first, bindings) = lower(&m, [7, 31]);
        let (next, next_bindings) = lower(&m, [2, 19]);
        assert_eq!(first, next);
        assert_ne!(bindings, next_bindings);
        assert!(first.iter().all(|c| c.sequence.is_none()));
        assert_eq!(first[0].launch.as_ref().unwrap().arguments, [(2, 3)]);
        assert_eq!(first[1].launch.as_ref().unwrap().arguments, [(2, 2)]);
        assert_eq!(first[1].views["PacketPointers"].offset, 16);
        assert_eq!(first[1].views["HeadLogits"].offset, 1200);
        assert!(matches!(
            &first[2].operation,
            Operation::Copy { bytes: 32, .. }
        ));
        assert_eq!(
            bindings,
            [
                Some((7, "ExistingState".into())),
                Some((7, "State".into())),
                Some((7, "SequenceLogits".into())),
                Some((31, "State".into())),
                Some((31, "SequenceLogits".into()))
            ]
        );
        m.mtp = None;
        let (ordinary, ordinary_bindings) = lower(&m, [7, 31]);
        assert_eq!(ordinary, first);
        assert_eq!(ordinary_bindings, bindings);
    }

    #[test]
    fn draft_warm_joins_projections_and_keeps_conditions_private() {
        let (a, _) = create(mtp_request(), Policy::Int8Quality).unwrap();
        let segments = [
            BatchSegment { slot: 7, tokens: 3 },
            BatchSegment { slot: 3, tokens: 2 },
        ];
        let (calls, _) = a
            .batch_operation(&segments, abi::BatchOperation::Warm, true)
            .unwrap();
        let fc: Vec<_> = calls
            .iter()
            .filter(|c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/mtp/fc"))
            .collect();
        assert_eq!(fc.len(), 1);
        assert_eq!(fc[0].sequence, None);
        let conditions:Vec<_>=calls.iter().filter(|c| matches!(&c.operation,Operation::Copy{destination,..} if destination=="MtpCondition")).map(|c|c.sequence).collect();
        assert_eq!(conditions, [Some(7), Some(3)]);
        assert!(!calls.iter().any(
            |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/head/project")
        ));
        let endpoints = [
            BatchSegment { slot: 7, tokens: 1 },
            BatchSegment { slot: 3, tokens: 1 },
        ];
        let (head, _) = a
            .batch_operation(&endpoints, abi::BatchOperation::DraftHead, true)
            .unwrap();
        assert!(!head.iter().any(
            |c| matches!(&c.operation,Operation::Kernel{name} if name=="qwen_rows/mtp/final_norm")
        ));
        assert_eq!(head.iter().filter(|c|matches!(&c.operation,Operation::Copy{destination,..} if destination=="MtpLogits")).count(),2);
    }

    #[test]
    fn invalid_packet_and_unquantized_operands_fail_before_execution() {
        let (adapter, _) = create(request(), Policy::Int8Quality).unwrap();
        for segments in [
            vec![],
            vec![BatchSegment { slot: 1, tokens: 0 }],
            vec![BatchSegment {
                slot: 1,
                tokens: 4097,
            }],
            vec![
                BatchSegment { slot: 1, tokens: 1 },
                BatchSegment { slot: 1, tokens: 1 },
            ],
        ] {
            assert!(adapter.batch(&segments, true).is_err());
        }
        let mut wrong = request();
        wrong.metadata.buffers[0].dtype = Dtype::F16;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = request();
        wrong.prefill_profiles[0].min_tokens = 0;
        assert!(create(wrong, Policy::Int8Quality).is_err());
        let mut wrong = request();
        wrong.config["quantization_config"]["rotation"] = json!("none");
        assert!(create(wrong, Policy::Int8Quality).is_err());
    }
}
