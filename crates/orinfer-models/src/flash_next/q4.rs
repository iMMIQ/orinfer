//! Row-Q4 input transforms are part of the registered Flash execution recipe.
use super::contracts::{Storage, connect, kernel_buffer_arg as buffer};
use crate::{
    artifact::{Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use std::collections::{BTreeMap, BTreeSet};

/// Creation receives storage contracts before kernels are attached. Derive
/// precision variants from roles, never from package kernel enumeration.
pub(super) fn validate(manifest: &Manifest) -> Result<()> {
    let storage = Storage::new(&manifest.buffers);
    let mut matrices: Vec<(String, usize, usize, bool)> =
        vec![("Q4_output".to_owned(), 248320, 2560, true)];
    for layer in 0..48 {
        if layer % 4 == 3 {
            matrices.push((format!("Q4_blk.{layer}.attn_q"), 12288, 2560, true));
        } else {
            matrices.push((format!("Q4_blk.{layer}.attn_gate"), 6144, 2560, true));
            // QKV and Z deliberately share the Z sign vector.
            matrices.push((format!("Q4_blk.{layer}.attn_qkv"), 10240, 2560, false));
        }
    }
    let has_q4_out_proj = storage.names().any(|n| {
        n.starts_with("Q4_blk.")
            && (n.ends_with(".ssm_out_packed") || n.ends_with(".attn_output_packed"))
    });
    if has_q4_out_proj {
        for layer in 0..48 {
            let role = if layer % 4 == 3 {
                "attn_output"
            } else {
                "ssm_out"
            };
            let w8_weight = format!("W_blk.{layer}.{role}.weight_weight");
            let w8_scale = format!("W_blk.{layer}.{role}.weight_scale");
            if storage.names().any(|n| n == w8_weight || n == w8_scale) {
                return Err(format!(
                    "Duplicate resident W8 and Q4 banks for blk.{layer}.{role}"
                ));
            }
            matrices.push((format!("Q4_blk.{layer}.{role}"), 2560, 6144, true));
        }
    }
    let selected: BTreeSet<_> = matrices
        .iter()
        .map(|(prefix, _, _, _)| format!("{prefix}_packed"))
        .collect();
    for (prefix, outputs, inputs, signs) in matrices {
        for (suffix, dtype, shape) in [
            ("packed", Dtype::U16, vec![outputs, inputs / 4]),
            ("scale", Dtype::F16, vec![outputs]),
        ]
        .into_iter()
        .chain(signs.then_some(("signs", Dtype::I8, vec![inputs])))
        {
            storage.check(&format!("{prefix}_{suffix}"), dtype, true, |b| {
                Ok(b.shape == shape)
            })?;
        }
    }
    if storage
        .names()
        .any(|n| n.starts_with("Q4_") && n.ends_with("_packed") && !selected.contains(n))
    {
        return Err("Unexpected Q4 matrix outside the Flash mixed-precision policy".into());
    }
    Ok(())
}

/// Attach transforms published by the shared compute lowering. Recipe slot
/// counts describe logical operators; implementations can add prerequisites
/// without duplicating the model's single/batch/mixed execution recipes.
pub(super) fn prerequisites(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts = super::contracts::kernels(kernels);
    let mut rotations = BTreeMap::new();
    let mut declared = BTreeMap::new();
    for rotation in kernels {
        let Some(projection) = rotation.name.strip_suffix("/q4_rotation") else {
            continue;
        };
        let target = contracts
            .get(projection)
            .ok_or("Missing Q4 projection contract")?;
        if !(4..=5).contains(&rotation.args.len()) || !(5..=6).contains(&target.args.len()) {
            return Err("Unexpected Q4 prerequisite ABI".into());
        }
        connect(rotation, target, &[(0, 0), (1, 1)])?;
        let signs = buffer(rotation, 2)?;
        let packed = buffer(target, 4)?;
        if !signs.starts_with("Q4_")
            || !signs.ends_with("_signs")
            || !packed.starts_with("Q4_")
            || !packed.ends_with("_packed")
        {
            return Err("Invalid Q4 prerequisite operands".into());
        }
        let section = projection
            .rsplit_once('/')
            .ok_or("Missing Q4 projection section")?
            .0;
        if rotations
            .insert(
                (section, buffer(rotation, 0)?, buffer(rotation, 1)?, signs),
                rotation.name.as_str(),
            )
            .is_some()
        {
            return Err("Ambiguous Q4 input transform".into());
        }
        declared.insert(projection, rotation.name.as_str());
    }
    // Associate every consumer of a transformed operand, including QKV/Z's
    // shared rotation. Section and sign identities prevent reuse across layers
    // or between target and draft heads that share scratch storage.
    let mut pending = BTreeMap::new();
    for kernel in kernels {
        let (Some((section, _)), Ok(a), Ok(scale), Ok(packed)) = (
            kernel.name.rsplit_once('/'),
            buffer(kernel, 0),
            buffer(kernel, 1),
            buffer(kernel, 4),
        ) else {
            continue;
        };
        let Some(base) = packed
            .strip_suffix("_packed")
            .filter(|b| b.starts_with("Q4_"))
        else {
            continue;
        };
        let signs = if let Some(layer) = base.strip_suffix(".attn_qkv") {
            format!("{layer}.attn_gate_signs")
        } else {
            format!("{base}_signs")
        };
        if let Some(rotation) = rotations.get(&(section, a, scale, signs.as_str())) {
            pending.insert(kernel.name.as_str(), *rotation);
        }
    }
    if declared
        .keys()
        .any(|projection| pending.get(projection) != declared.get(projection))
    {
        return Err("Q4 transform does not match the projection rotation policy".into());
    }
    let mut unregistered: BTreeSet<_> = declared.values().copied().collect();
    let transforms = unregistered.clone();
    for operations in manifest.programs.values_mut() {
        // Each alias executes its own transform, once before its first consumer.
        let mut installed = BTreeSet::new();
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation {
                if transforms.contains(name.as_str()) {
                    unregistered.remove(name.as_str());
                    continue;
                }
                if let Some(rotation) = pending.get(name.as_str())
                    && installed.insert(*rotation)
                {
                    unregistered.remove(rotation);
                    expanded.push(Operation::Kernel {
                        name: (*rotation).into(),
                    });
                }
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    if !unregistered.is_empty() {
        return Err("Unregistered Q4 prerequisite".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "q4_tests.rs"]
mod tests;
