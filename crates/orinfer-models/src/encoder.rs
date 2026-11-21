//! Shared E8P Q4A8 validation and row-program builders for independent encoder models.
use crate::{
    CreatedPlan,
    adapter::Adapter,
    artifact::{Access, Buffer, Dtype, Result},
    model::{Manifest, Operation, TensorIdentity},
};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn kernel(prefix: &str, name: impl AsRef<str>) -> Operation {
    Operation::Kernel {
        name: format!("{prefix}/{}", name.as_ref()),
    }
}

pub(crate) fn check_signature(config: &Value, expected: &Value, label: &str) -> Result<()> {
    let mut actual = config
        .as_object()
        .ok_or_else(|| format!("Invalid {label} config"))?
        .clone();
    actual.remove("quantization_config");
    if Value::Object(actual) != *expected {
        return Err(format!("{label} configuration signature differs"));
    }
    Ok(())
}

pub(crate) fn check_fields(value: &Value, expected: &[(&str, Value)], prefix: &str) -> Result<()> {
    for (key, want) in expected {
        if value[key] != *want {
            return Err(format!("Unsupported {prefix} {key}"));
        }
    }
    Ok(())
}

pub(crate) fn check_e8p_q4_quant(
    quant: &Value,
    policy: &str,
    extra: &[(&str, Value)],
    label: &str,
) -> Result<()> {
    check_fields(
        quant,
        &[
            ("quant_method", json!("orinfer_e8p_q4_int8")),
            ("version", json!(1)),
            ("coefficients", json!([4, 1])),
            ("rotation", json!("signed-block128")),
            ("scale", json!("fp16-output-row")),
            ("compute_dtype", json!("int8_quality")),
            ("layout", json!("u32-low-coarse-high-residual")),
            ("embedding", json!("e8p_q4")),
            ("policy", json!(policy)),
        ],
        &format!("{label} quantization"),
    )?;
    check_fields(quant, extra, &format!("{label} quantization"))
}

pub(crate) fn require_weight(
    buffers: &[Buffer<TensorIdentity>],
    name: &str,
    dtype: Dtype,
    shape: &[usize],
    layout: &str,
    label: &str,
) -> Result<()> {
    let b = buffers
        .iter()
        .find(|b| b.name == name)
        .ok_or_else(|| format!("Missing {label} weight {name}"))?;
    if b.dtype != dtype
        || b.shape != shape
        || b.layout != layout
        || b.data.is_none()
        || b.access != Access::Read
    {
        return Err(format!("Invalid {label} weight {name}"));
    }
    Ok(())
}

pub(crate) fn require_matrix(
    buffers: &[Buffer<TensorIdentity>],
    name: &str,
    n: usize,
    k: usize,
    w8: bool,
    label: &str,
) -> Result<()> {
    if w8 {
        require_weight(
            buffers,
            &format!("{name}.int8"),
            Dtype::I8,
            &[n, k],
            "native_contiguous",
            label,
        )?;
    } else {
        require_weight(
            buffers,
            &format!("{name}.packed"),
            Dtype::U32,
            &[k / 128, n, 16],
            "e8p-q4-k128-u32",
            label,
        )?;
    }
    require_weight(
        buffers,
        &format!("{name}.scale"),
        Dtype::F16,
        &[n],
        "native_contiguous",
        label,
    )
}

pub(crate) fn transformer_ops(
    prefix: &str,
    layers: usize,
    packed_rows: bool,
    rope_role: &str,
) -> Vec<Operation> {
    let mut ops = Vec::new();
    for layer in 0..layers {
        let roles: &[&str] = if packed_rows {
            &[
                "norm1",
                "rotate_qkv",
                "qkv",
                rope_role,
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
            &[
                "norm1",
                "rotate_qkv",
                "q",
                "k",
                "v",
                rope_role,
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
            ops.push(kernel(prefix, format!("layer{layer}/{role}")));
        }
    }
    ops
}

pub(crate) fn finalize(
    adapter: Box<dyn Adapter>,
    mut metadata: Manifest,
    programs: BTreeMap<String, Vec<Operation>>,
    capacity: usize,
    program_capacities: BTreeMap<String, usize>,
) -> Result<crate::registry::Built> {
    let names: BTreeSet<String> = programs.keys().cloned().collect();
    metadata.programs = programs;
    let mut row_execution = orinfer_model_sdk::operators::rows::RowExecution {
        capacity,
        programs: names,
        program_capacities,
        variants: Default::default(),
    };
    row_execution.register_variants(
        &metadata.programs,
        metadata.dynamic_batch_kernels.keys().map(String::as_str),
    )?;
    metadata.row_execution = Some(row_execution);
    Ok((
        adapter,
        CreatedPlan {
            expert_banks: vec![],
            metadata,
        },
    ))
}
