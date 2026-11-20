//! Table-Q4A8: shared row computation with request-owned causal state.
use super::config::Configuration;
use crate::packet::append;
use crate::{CreateRequest, CreatedPlan};
use crate::{
    adapter::Adapter,
    artifact::{Access, Dtype, Result},
    execution::{BufferView, Invocation},
    model::{KvGrowth, Manifest, Operation, PrefillPlan, SegmentControls},
};
use orinfer_model_sdk::{
    architecture::{BatchLayout, BatchSegment},
    operators::rows::RowExecution,
    packet,
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
    operands: crate::packet::Operands,
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
fn validate_weights(m: &Manifest, config: &Configuration, embedding_q4: bool) -> Result<()> {
    let t = config.text();
    let map: BTreeMap<_, _> = m.buffers.iter().map(|b| (b.name.as_str(), b)).collect();
    let check = |name: &str, dtype: Dtype, shape: &[usize], access: Access| -> Result<()> {
        if map.get(name).is_none_or(|b| {
            b.dtype != dtype
                || b.shape != shape
                || b.access != access
                || b.data.is_some() != (access == Access::Read)
        }) {
            return Err(format!("Invalid Qwen buffer contract: {name}"));
        }
        Ok(())
    };
    let gdn = m.gdn_execution.as_ref();
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
            check(
                name,
                Dtype::F16,
                &[m.max_context, t.num_key_value_heads * t.head_dim],
                Access::ReadWrite,
            )?;
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
            check(name, dtype, &shape, Access::ReadWrite)?;
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
                check(&name, dtype, &shape, Access::ReadWrite)?;
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
            check(name, dtype, &[size], Access::ReadWrite)?;
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
            check(name, Dtype::I32, &shape, Access::ReadWrite)?;
        }
    }
    let weight = |name: String, dtype, shape: Vec<usize>| check(&name, dtype, &shape, Access::Read);
    weight("orinfer.e8p_q4.basis".into(), Dtype::I8, vec![256, 8])?;
    weight("orinfer.e8p_q4.selectors".into(), Dtype::U32, vec![256, 1])?;
    let q4 = |name: String, n: usize, k: usize| -> Result<()> {
        if config
            .rotated_w8_layers
            .iter()
            .any(|layer| name.starts_with(&format!("model.language_model.layers.{layer}.")))
        {
            weight(format!("{name}.int8"), Dtype::I8, vec![n, k])?;
            weight(format!("{name}.scale"), Dtype::F16, vec![n])?;
            if map.contains_key(format!("{name}.packed").as_str()) {
                return Err("Protected projection has duplicate compressed weights".into());
            }
            return Ok(());
        }
        let packed = format!("{name}.packed");
        weight(packed.clone(), Dtype::U32, vec![k / 128, n, 16])?;
        if map[packed.as_str()].layout != "e8p-q4-k128-u32" {
            return Err(format!("Invalid Qwen E8P physical layout: {packed}"));
        }
        weight(format!("{name}.scale"), Dtype::F16, vec![n])
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
                weight(
                    packed.clone(),
                    Dtype::U32,
                    vec![t.num_experts, k / 128, n, 16],
                )?;
                if map[packed.as_str()].layout != "e8p-q4-k128-u32" {
                    return Err("Invalid Qwen MoE E8P bank layout".into());
                }
                weight(format!("{name}.scale"), Dtype::F16, vec![t.num_experts, n])?;
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

pub(crate) fn create(mut request: CreateRequest) -> Result<crate::registry::Built> {
    let quant = &request.config["quantization_config"];
    if quant["quant_method"] != "orinfer_e8p_q4_int8" {
        return Err("Unsupported Qwen Q4A8 codec".into());
    }
    if quant["coefficients"] != serde_json::json!([4, 1])
        || quant["basis_tensor"] != "orinfer.e8p_q4.basis"
        || quant["layout"] != "u32-low-coarse-high-residual"
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
        if !quant[role].is_null() && quant[role] != "e8p_q4" {
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
        t.hidden_size == 2048
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
    if !c.standard && m.gdn_execution.is_none() {
        return Err("Missing explicit Qwen GDN execution contract".into());
    }
    if request.verify_weights || request.build_bindings {
        validate_weights(m, &c, embedding_q4)?;
    }
    let staged_attention = m.buffers.iter().any(|b| b.name == "AttentionDecodedK");
    if ["AttentionDecodedK", "AttentionDecodedV"]
        .iter()
        .any(|&name| {
            m.reset_buffers.iter().any(|reset| reset == name)
                || m.kv_cache
                    .as_ref()
                    .is_some_and(|kv| kv.buffers.contains_key(name))
        })
    {
        return Err("Attention decode scratch cannot be request state".into());
    }
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
        if !gdn && staged_attention {
            private_names.insert(1, "attention_decode");
        }
        if gdn && deferred_gdn(m) {
            for column in DEFERRED_COLUMNS {
                let name = format!("L{layer}_{column}");
                if !m.reset_buffers.contains(&name) {
                    return Err(format!(
                        "Missing reset for request-owned deferred GDN state: {name}"
                    ));
                }
            }
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
    #[cfg(test)]
    if request.operands.is_empty() {
        request.operands = crate::packet::Operands::from_kernels(&request.kernel_contracts);
    }
    Ok((
        Box::new(QwenRows {
            private_roles: private_roles(&metadata),
            manifest: metadata.clone(),
            operands: request.operands,
        }),
        CreatedPlan {
            metadata,
            expert_banks: vec![],
        },
    ))
}

impl Adapter for QwenRows {
    fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: packet::BatchOperation,
        include_plan: bool,
    ) -> Result<packet::BatchPlan> {
        let result = match operation {
            packet::BatchOperation::Ordinary => self.packet(segments, None, include_plan),
            packet::BatchOperation::Verification => self.packet(segments, Some(0), include_plan),
            packet::BatchOperation::MixedRows { prefills }
                if prefills > 0 && usize::from(prefills) < segments.len() =>
            {
                self.packet(segments, Some(usize::from(prefills)), include_plan)
            }
            packet::BatchOperation::Draft
            | packet::BatchOperation::Warm
            | packet::BatchOperation::DraftHead => {
                mtp::draft(&self.manifest, segments, operation, include_plan)
            }
            packet::BatchOperation::Commit => mtp::commit(&self.manifest, segments, include_plan),
            _ => Err("Unsupported Qwen packet operation".into()),
        }?;
        crate::packet::indirect(&self.manifest, segments, &self.operands, result)
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
        images: &[orinfer_model_sdk::architecture::ImageGrid],
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
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<packet::BatchPlan> {
        self.packet(segments, None, include_plan)
    }
}

impl QwenRows {
    fn state_bindings(&self, segments: &[BatchSegment]) -> packet::StateBindings {
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
    ) -> Result<packet::BatchPlan> {
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
#[path = "rows_tests.rs"]
mod tests;
