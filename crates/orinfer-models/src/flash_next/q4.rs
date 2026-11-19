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
    let mut matrices = vec![("Q4_output".to_owned(), 248320, true)];
    for layer in 0..48 {
        if layer % 4 == 3 {
            matrices.push((format!("Q4_blk.{layer}.attn_q"), 12288, true));
        } else {
            matrices.push((format!("Q4_blk.{layer}.attn_gate"), 6144, true));
            // QKV and Z deliberately share the Z sign vector.
            matrices.push((format!("Q4_blk.{layer}.attn_qkv"), 10240, false));
        }
    }
    let selected: BTreeSet<_> = matrices
        .iter()
        .map(|(prefix, _, _)| format!("{prefix}_packed"))
        .collect();
    for (prefix, outputs, signs) in matrices {
        for (suffix, dtype, shape) in [
            ("packed", Dtype::U16, vec![outputs, 640]),
            ("scale", Dtype::F16, vec![outputs]),
        ]
        .into_iter()
        .chain(signs.then_some(("signs", Dtype::I8, vec![2560])))
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
mod tests {
    use super::*;
    use crate::artifact::Argument;
    fn contract(name: &str, args: &[&str]) -> Kernel {
        let asset = serde_json::json!({"file":"test.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "name":name,"symbol":"main_kernel","module":asset,"source":asset,"host_abi":asset,
            "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    #[test]
    fn paired_qkv_z_rotate_once_before_the_first_consumer() {
        let mut m = manifest(0);
        let qkv = "decode/layer0/k6";
        let z = "decode/layer0/k7";
        let rotation = format!("{z}/q4_rotation");
        let kernels = vec![
            contract(
                qkv,
                &[
                    "A",
                    "AS",
                    "QKV",
                    "Q4_blk.0.attn_qkv_scale",
                    "Q4_blk.0.attn_qkv_packed",
                ],
            ),
            contract(
                z,
                &[
                    "A",
                    "AS",
                    "Z",
                    "Q4_blk.0.attn_gate_scale",
                    "Q4_blk.0.attn_gate_packed",
                ],
            ),
            contract(&rotation, &["A", "AS", "Q4_blk.0.attn_gate_signs", "Input"]),
        ];
        // Even a transform attached to the later projection is moved before
        // both consumers. Reconstructing a plan must remain idempotent.
        for program in ["decode", "alias"] {
            m.programs.insert(
                program.into(),
                [qkv, rotation.as_str(), z]
                    .into_iter()
                    .map(|name| Operation::Kernel { name: name.into() })
                    .collect(),
            );
        }
        for _ in 0..2 {
            prerequisites(&mut m, &kernels).unwrap();
            for operations in m.programs.values() {
                let names: Vec<_> = operations
                    .iter()
                    .map(|op| match op {
                        Operation::Kernel { name } => name.as_str(),
                        _ => panic!("Unexpected copy"),
                    })
                    .collect();
                assert_eq!(names, [rotation.as_str(), qkv, z]);
            }
        }
    }

    #[test]
    fn prerequisites_cover_every_program_alias_and_are_idempotent() {
        let mut m = manifest(0);
        m.programs.clear();
        let projection = "prefill_m4096/layer0/k6";
        let rotation = format!("{projection}/q4_rotation");
        let kernels = vec![
            contract(
                projection,
                &["A", "AS", "Out", "Q4_test_scale", "Q4_test_packed"],
            ),
            contract(&rotation, &["A", "AS", "Q4_test_signs", "Input"]),
        ];
        for preinstalled in [false, true] {
            m.programs.clear();
            // Cover both destructive lookup across aliases and the case where
            // just one alias already contains its prerequisite.
            for program in ["prefill", "prefill_m2048", "prefill_m4096"] {
                let mut ops = vec![];
                if preinstalled && program == "prefill" {
                    ops.push(Operation::Kernel {
                        name: (*rotation).into(),
                    });
                }
                ops.push(Operation::Kernel {
                    name: projection.into(),
                });
                m.programs.insert(program.into(), ops);
            }
            for _ in 0..2 {
                prerequisites(&mut m, &kernels).unwrap();
                for ops in m.programs.values() {
                    assert_eq!(ops.len(), 2);
                    assert!(matches!(&ops[0], Operation::Kernel { name } if name == &rotation));
                    assert!(matches!(&ops[1], Operation::Kernel { name } if name == projection));
                }
            }
        }
    }

    #[test]
    fn shared_lowering_expands_target_draft_and_prefill_recipes() {
        for section in [
            "flash_decode/layer0/pre",
            "draft_rows/pre",
            "prefill_m4096/layer0",
        ] {
            let mut m = manifest(0);
            m.programs.clear();
            let projection = format!("{section}/k6");
            let rotation = format!("{projection}/q4_rotation");
            m.programs.insert(
                section.into(),
                vec![Operation::Kernel {
                    name: projection.clone(),
                }],
            );
            let mut kernels = vec![
                contract(
                    &projection,
                    &[
                        "A",
                        "AS",
                        "Out",
                        "Q4_blk.0.attn_qkv_scale",
                        "Q4_blk.0.attn_qkv_packed",
                    ],
                ),
                contract(&rotation, &["A", "AS", "Q4_blk.0.attn_gate_signs", "Input"]),
            ];
            prerequisites(&mut m, &kernels).unwrap();
            assert!(
                matches!(&m.programs[section][0], Operation::Kernel {name} if name == &rotation)
            );
            assert!(
                matches!(&m.programs[section][1], Operation::Kernel {name} if name == &projection)
            );
            prerequisites(&mut m, &kernels).unwrap();
            assert_eq!(m.programs[section].len(), 2);
            m.programs.get_mut(section).unwrap().remove(0);
            kernels[1].args[0] = Argument::Buffer {
                name: "wrong".into(),
            };
            assert!(
                prerequisites(&mut m, &kernels)
                    .unwrap_err()
                    .contains("workspace mismatch")
            );
            kernels[1].args[0] = Argument::Buffer { name: "A".into() };
            m.programs.clear();
            assert!(
                prerequisites(&mut m, &kernels)
                    .unwrap_err()
                    .contains("Unregistered")
            );
        }
    }

    fn manifest(_layer: usize) -> Manifest {
        serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test"
        }))
        .unwrap()
    }
}
