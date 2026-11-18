//! Compressed expert banks and exact A8 split steps owned by the Flash recipe.
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::abi::ExpertBank;
use std::collections::BTreeMap;

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    if kernels.is_empty() && manifest.buffers.iter().any(|b| b.name.starts_with("E8P_")) {
        return Err("Missing exact expert execution contracts".into());
    }
    let contracts: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut splits = BTreeMap::new();
    for kernel in kernels {
        if let Some(projection) = kernel.name.strip_suffix("/a8_split") {
            let target = contracts
                .get(projection)
                .ok_or("Missing exact expert projection")?;
            if !matches!(
                target.symbol.as_str(),
                "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
            ) || !(target.args.len() == 13
                || target.args.len() == 14
                    && matches!(target.args.last(), Some(Argument::I32 { .. })))
                || kernel.symbol != "pack_exact_a8_kernel"
                || !(kernel.args.len() == 4
                    || kernel.args.len() == 5
                        && matches!(kernel.args.last(), Some(Argument::I32 { .. })))
            {
                return Err("Unexpected exact expert ABI".into());
            }
            for (split_index, project_index) in [(1, 3), (2, 4), (3, 8)] {
                let (Argument::Buffer { name: a }, Argument::Buffer { name: b }) =
                    (&kernel.args[split_index], &target.args[project_index])
                else {
                    return Err("Exact expert workspace must be a whole buffer".into());
                };
                if a != b {
                    return Err("Exact expert split/projection workspace mismatch".into());
                }
            }
            splits.insert(projection.to_owned(), kernel.name.clone());
        }
    }
    for kernel in kernels.iter().filter(|k| {
        matches!(
            k.symbol.as_str(),
            "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
        )
    }) {
        if splits.contains_key(&kernel.name) {
            continue;
        }
        // An everywhere-selected fused dispatch produces the exact input
        // planes itself; no unused standalone pack needs to be published.
        if kernel.name.starts_with("prefill_m4096/")
            && let Some((section, _)) = kernel.name.rsplit_once('/')
            && kernels.iter().any(|producer| {
                producer.symbol == "dispatch_planes_kernel"
                    && producer.name.rsplit_once('/').is_some_and(|(parent, _)| parent == section)
                    && [(3, 2), (4, 4), (8, 9)].iter().all(|&(consumer, output)| {
                        matches!((kernel.args.get(consumer), producer.args.get(output)),
                            (Some(Argument::Buffer {name:a}),Some(Argument::Buffer {name:b})) if a==b)
                    })
            })
        {
            continue;
        }
        // A row specialization consumes the canonical split once. All pointer
        // operands must stay identical; only the row scalar and tile may differ.
        let canonical = kernel
            .name
            .split_once("/rows/")
            .or_else(|| kernel.name.split_once("/choice/"))
            .and_then(|(slot, _)| contracts.get(slot).copied())
            .filter(|k| splits.contains_key(&k.name));
        fn pointer(arg: &Argument) -> Option<(&str, usize)> {
            match arg {
                Argument::Buffer { name } => Some((name, 0)),
                Argument::BufferSlice { name, offset } => Some((name, *offset)),
                _ => None,
            }
        }
        if canonical.is_none_or(|base| {
            base.symbol != kernel.symbol
                || base
                    .args
                    .iter()
                    .filter_map(pointer)
                    .ne(kernel.args.iter().filter_map(pointer))
        }) {
            return Err("Missing exact expert A8 split or changed row operands".into());
        }
    }
    let mut registered = std::collections::BTreeSet::new();
    for operations in manifest.programs.values_mut() {
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation
                && let Some(split) = splits.get(name)
            {
                expanded.push(Operation::Kernel {
                    name: split.clone(),
                });
                registered.insert(name.clone());
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    if registered.len() != splits.len() {
        return Err("Unregistered exact expert execution step".into());
    }
    Ok(())
}

pub(super) fn banks(manifest: &Manifest, kernels: &[Kernel]) -> Result<Vec<ExpertBank>> {
    let buffers: BTreeMap<_, _> = manifest
        .buffers
        .iter()
        .map(|b| (b.name.as_str(), b))
        .collect();
    (0..48)
        .map(|layer| {
            let prefix = format!("W_blk.{layer}.");
            let names: Vec<_> = ["ffn_gate_up_exps", "ffn_down_exps"]
                .into_iter()
                .flat_map(|role| {
                    ["packed", "scale"].into_iter().map({
                        let prefix = prefix.clone();
                        move |part| format!("{prefix}{role}.weight_{part}")
                    })
                })
                .collect();
            let experts = buffers
                .get(names[0].as_str())
                .and_then(|b| b.shape.first())
                .copied()
                .ok_or("Missing expert bank")?;
            let mut consumers = BTreeMap::new();
            for k in kernels {
                if !k
                    .args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if name == &names[0]))
                {
                    continue;
                }
                // Exported host argument order, not the TileLang function's
                // declaration order. The exact path adds two nibble planes and sum.
                let histogram = if matches!(
                    k.symbol.as_str(),
                    "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
                ) && (k.args.len() == 13
                    || k.args.len() == 14 && matches!(k.args.last(), Some(Argument::I32 { .. })))
                {
                    2
                } else if k.args.len() == 11
                    || (k.args.len() == 12
                        && matches!(k.args.last(), Some(Argument::I32 { .. }))
                        && manifest.dynamic_batch_kernels.contains_key(&k.name))
                {
                    4
                } else {
                    return Err(format!("{}: unsupported expert consumer ABI", k.name));
                };
                let Some(Argument::Buffer { name }) = k.args.get(histogram) else {
                    return Err("Unsupported grouped expert histogram ABI".into());
                };
                let b = buffers
                    .get(name.as_str())
                    .ok_or("Missing expert histogram")?;
                if b.dtype != Dtype::I32 || b.shape != [experts] {
                    return Err(format!("{}: unsupported expert consumer ABI", k.name));
                }
                consumers.insert(k.name.clone(), histogram);
            }
            if consumers.is_empty() && !kernels.is_empty() {
                return Err("Missing expert gate/up consumers".into());
            }
            Ok(ExpertBank {
                experts,
                buffers: names,
                consumers,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(name: &str, symbol: &str, args: &[&str]) -> Kernel {
        let identity = serde_json::json!({"file":"kernel.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "name":name,"symbol":symbol,"module":identity,"source":identity,"host_abi":identity,
            "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    fn fixture() -> (Manifest, Vec<Kernel>) {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,
            "weight_parameters":0,"weight_scope":"test"
        }))
        .unwrap();
        m.programs.insert(
            "decode".into(),
            vec![Operation::Kernel {
                name: "decode/layer0/k29".into(),
            }],
        );
        let projection = kernel(
            "decode/layer0/k29",
            "exact_e8p_int4_kernel",
            &[
                "AS", "Book", "Counts", "High", "Low", "Offsets", "Out", "P", "Sum", "TC", "TE",
                "TR", "WS",
            ],
        );
        let split = kernel(
            "decode/layer0/k29/a8_split",
            "pack_exact_a8_kernel",
            &["A", "High", "Low", "Sum"],
        );
        (m, vec![projection, split])
    }

    #[test]
    fn sole_fused_dispatch_retires_pack_only_for_matching_expert_planes() {
        let (mut m, mut kernels) = fixture();
        kernels.truncate(1);
        kernels[0].name = "prefill_m4096/layer0/k29".into();
        m.programs.clear();
        m.programs.insert(
            "prefill".into(),
            vec![Operation::Kernel {
                name: kernels[0].name.clone(),
            }],
        );
        kernels.push(kernel(
            "prefill_m4096/layer0/k25",
            "dispatch_planes_kernel",
            &[
                "A", "DS", "High", "IDs", "Low", "Relative", "Offsets", "Scale", "Slots", "Sum",
            ],
        ));
        build(&mut m, &kernels).unwrap();
        assert_eq!(m.programs["prefill"].len(), 1);
        kernels[1].args[2] = Argument::Buffer {
            name: "another_owner".into(),
        };
        assert!(build(&mut m, &kernels).is_err());
    }

    #[test]
    fn exact_split_precedes_projection_and_validates_actual_abi() {
        let (mut m, kernels) = fixture();
        build(&mut m, &kernels).unwrap();
        assert!(
            matches!(&m.programs["decode"][0], Operation::Kernel {name} if name.ends_with("/a8_split"))
        );
        assert!(
            matches!(&m.programs["decode"][1], Operation::Kernel {name} if name.ends_with("/k29"))
        );
        let (mut m, mut kernels) = fixture();
        kernels[1].args[1] = Argument::Buffer {
            name: "wrong".into(),
        };
        assert!(
            build(&mut m, &kernels)
                .unwrap_err()
                .contains("workspace mismatch")
        );
        let (mut m, kernels) = fixture();
        assert!(
            build(&mut m, &kernels[..1])
                .unwrap_err()
                .contains("Missing")
        );
        let (mut m, kernels) = fixture();
        m.programs.clear();
        assert!(
            build(&mut m, &kernels)
                .unwrap_err()
                .contains("Unregistered")
        );
    }
}
