//! BF16 HC prefill operands retain the original FP16 materialization boundaries.
use super::contracts::{Storage, connect, fuse, kernel_at, kernel_buffer_arg as buffer, retire};
use crate::{
    artifact::{Dtype, Kernel, Result},
    model::Manifest,
};
use std::collections::BTreeSet;

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel], exhaustive: bool) -> Result<()> {
    let contracts = super::contracts::kernels(kernels);
    let anchors = kernels.iter().filter(|k| {
        k.symbol == "hc_prefill_up_mix_kernel"
            && !k.name.ends_with("/aligned_rows")
            && !k.name.contains("/choice/")
    });
    let storage = exhaustive.then(|| Storage::new(&manifest.buffers));
    fuse(
        &mut manifest.programs,
        anchors.clone(),
        |program, operations, index, up, skip| {
            let name = &up.name;
            let rows = name
                .split('/')
                .next()
                .and_then(|s| s.strip_prefix("prefill_m"))
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| matches!(n, 512 | 4096))
                .ok_or("Unsupported HC prefill fusion program")?;
            if (program != format!("prefill_m{rows}") && program != "prefill") || index < 2 {
                return Err("Unsupported HC prefill fusion program".into());
            }
            let norm = kernel_at(&contracts, operations, index - 2)?;
            let down = kernel_at(&contracts, operations, index - 1)?;
            let mix = kernel_at(&contracts, operations, index + 1)?;
            let inject = kernel_at(&contracts, operations, index + 2)?;
            let count = |k: &Kernel| {
                k.args
                    .iter()
                    .filter(|a| matches!(a, crate::artifact::Argument::Buffer { .. }))
                    .count()
            };
            let (inv, normalized, residual, weight) = match (norm.symbol.as_str(), count(norm)) {
                ("hc_prefill_norm_kernel", 4) => (0, 1, 2, 3),
                ("hc_prefill_combine_norm_kernel", 7) => (2, 3, 4, 6),
                _ => return Err("Invalid HC prefill norm ABI".into()),
            };
            if down.symbol != "hc_prefill_projection_grouped_kernel"
                || inject.symbol != "hc_prefill_projection_kernel"
                || count(down) != 3
                || count(up) != 6
                || count(mix) != 3
                || count(inject) != 3
            {
                return Err("HC prefill fusion producer/consumer mismatch".into());
            }
            connect(down, norm, &[(0, normalized)])?;
            connect(up, down, &[(0, 1)])?;
            connect(up, norm, &[(1, inv), (3, residual), (5, weight)])?;
            connect(up, mix, &[(2, 0)])?;
            connect(inject, norm, &[(0, normalized)])?;
            if let Some(storage) = &storage {
                for (k, arg, dtype, shape, exact, immutable) in [
                    (
                        norm,
                        normalized,
                        Dtype::Bf16,
                        vec![rows, 4, 2560],
                        true,
                        false,
                    ),
                    (up, 0, Dtype::Bf16, vec![rows, 320], true, false),
                    (up, 1, Dtype::F32, vec![rows, 4], true, false),
                    (up, 2, Dtype::F16, vec![rows, 2560], false, false),
                    (up, 3, Dtype::F16, vec![rows, 4, 2560], false, false),
                    (up, 4, Dtype::Bf16, vec![10240, 320], true, true),
                    (up, 5, Dtype::F32, vec![4, 2560], true, true),
                ] {
                    storage.check(buffer(k, arg)?, dtype, immutable, |b| {
                        Ok(if exact {
                            b.shape == shape
                        } else {
                            b.bytes()? >= shape.iter().product::<usize>() * dtype.bytes()
                        })
                    })?;
                }
            }
            retire(skip, [index + 1])
        },
    )?;
    if anchors.clone().next().is_none() {
        return Ok(());
    }
    let canonical_names: BTreeSet<_> = anchors.map(|k| k.name.as_str()).collect();
    for alternative in kernels
        .iter()
        .filter(|k| k.name.ends_with("/aligned_rows") && k.symbol == "hc_prefill_up_mix_kernel")
    {
        let base = alternative.name.strip_suffix("/aligned_rows").unwrap();
        let canonical = contracts
            .get(base)
            .ok_or("Missing aligned HC canonical slot")?;
        let matching = alternative.args.len() == canonical.args.len()
            && alternative
                .args
                .iter()
                .zip(&canonical.args)
                .all(|(a, b)| match (a, b) {
                    (
                        crate::artifact::Argument::Buffer { name: a },
                        crate::artifact::Argument::Buffer { name: b },
                    ) => a == b,
                    (
                        crate::artifact::Argument::I32 { value: a },
                        crate::artifact::Argument::I32 { value: b },
                    ) => a == b,
                    _ => false,
                });
        if !canonical_names.contains(base) || !matching {
            return Err("Aligned HC changes the canonical operand contract".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Operation;

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let definitions = [
            ("BF", "bf16", vec![512, 4, 2560], false),
            ("Act", "bf16", vec![512, 320], false),
            ("Inv", "f32", vec![512, 4], false),
            ("Mixed", "f16", vec![512, 2560], false),
            ("Residual", "f16", vec![512, 4, 2560], false),
            ("Up", "bf16", vec![10240, 320], true),
            ("Weight", "f32", vec![4, 2560], true),
        ];
        let buffers: Vec<_> = definitions
            .into_iter()
            .map(|(name, dtype, shape, immutable)| {
                let mut b = serde_json::json!({"name":name,"dtype":dtype,"shape":shape,
                    "layout":"native_contiguous","alignment":256,
                    "access":if immutable {"read"} else {"read_write"}});
                if immutable {
                    b["data"] = serde_json::json!({"tensor":name,"sha256":"0".repeat(64)});
                }
                b
            })
            .collect();
        let m = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":buffers,
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test","programs":{"prefill_m512":
            (0..5).map(|i| serde_json::json!({"kind":"kernel","name":format!("prefill_m512/layer0/k{i}")})).collect::<Vec<_>>()}
        })).unwrap();
        let definitions: [(&str, &[&str]); 5] = [
            (
                "hc_prefill_norm_kernel",
                &["Inv", "BF", "Residual", "Weight"],
            ),
            (
                "hc_prefill_projection_grouped_kernel",
                &["BF", "Act", "Down"],
            ),
            (
                "hc_prefill_up_mix_kernel",
                &["Act", "Inv", "Mixed", "Residual", "Up", "Weight"],
            ),
            ("main_kernel", &["Mixed", "OldNormed", "OldUp"]),
            ("hc_prefill_projection_kernel", &["BF", "Inject", "IW"]),
        ];
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        let kernels = definitions.into_iter().enumerate().map(|(i,(symbol,args))|
            serde_json::from_value(serde_json::json!({"name":format!("prefill_m512/layer0/k{i}"),
                "symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
                "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()})).unwrap()
        ).collect();
        (m, kernels)
    }

    #[test]
    fn fusion_removes_only_the_adjacent_mix() {
        let (mut m, kernels) = fixture();
        build(&mut m, &kernels, true).unwrap();
        let names: Vec<_> = m.programs["prefill_m512"]
            .iter()
            .map(|op| {
                let Operation::Kernel { name } = op else {
                    unreachable!()
                };
                name.as_str()
            })
            .collect();
        assert_eq!(
            names,
            [
                "prefill_m512/layer0/k0",
                "prefill_m512/layer0/k1",
                "prefill_m512/layer0/k2",
                "prefill_m512/layer0/k4"
            ]
        );
    }

    #[test]
    fn fusion_rejects_wrong_output_and_inverse_shape() {
        let (mut m, mut kernels) = fixture();
        kernels[3].args[0] = crate::artifact::Argument::Buffer {
            name: "wrong".into(),
        };
        assert!(
            build(&mut m, &kernels, true)
                .unwrap_err()
                .contains("mismatch")
        );
        let (mut m, kernels) = fixture();
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Inv")
            .unwrap()
            .shape = vec![512, 8];
        assert!(build(&mut m, &kernels, true).unwrap_err().contains("shape"));
    }

    #[test]
    fn fusion_rejects_a_different_program() {
        let (mut m, kernels) = fixture();
        let ops = m.programs.remove("prefill_m512").unwrap();
        m.programs.insert("decode".into(), ops);
        assert!(
            build(&mut m, &kernels, true)
                .unwrap_err()
                .contains("program")
        );
    }

    #[test]
    fn dynamic_prefill_fuses_both_the_canonical_program_and_alias() {
        let (mut m, mut kernels) = fixture();
        let operations = m.programs.remove("prefill_m512").unwrap();
        let operations: Vec<_> = operations
            .into_iter()
            .map(|op| match op {
                Operation::Kernel { name } => Operation::Kernel {
                    name: name.replace("m512", "m4096"),
                },
                other => other,
            })
            .collect();
        m.programs
            .insert("prefill_m4096".into(), operations.clone());
        m.programs.insert("prefill".into(), operations);
        for b in &mut m.buffers {
            if b.data.is_none() && b.shape.first() == Some(&512) {
                b.shape[0] = 4096;
            }
        }
        for k in &mut kernels {
            k.name = k.name.replace("m512", "m4096");
            k.args.push(crate::artifact::Argument::I32 { value: 4096 });
        }
        build(&mut m, &kernels, true).unwrap();
        assert_eq!(m.programs["prefill"], m.programs["prefill_m4096"]);
        assert_eq!(m.programs["prefill"].len(), 4);
    }
}
