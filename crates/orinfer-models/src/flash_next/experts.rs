//! Compressed expert banks and exact A8 split steps owned by the Flash recipe.
use super::contracts::{connect, kernel_buffer_arg as buffer, pointer};
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::ExpertBank;
use std::collections::{BTreeMap, BTreeSet};

fn exact(kernel: &Kernel) -> bool {
    matches!(
        kernel.symbol.as_str(),
        "exact_e8p_int4_kernel" | "exact_e8p_shared_input_kernel"
    )
}

fn row_arguments(kernel: &Kernel, pointers: usize, dynamic: bool) -> bool {
    kernel.args.len() == pointers
        || dynamic
            && kernel.args.len() == pointers + 1
            && matches!(kernel.args.last(), Some(Argument::I32 { .. }))
}

fn planes(kernel: &Kernel, columns: [usize; 3]) -> Option<(&str, [&str; 3])> {
    let (section, _) = kernel.name.rsplit_once('/')?;
    let [a, b, c] = columns.map(|i| buffer(kernel, i).ok());
    Some((section, [a?, b?, c?]))
}

pub(super) fn build(manifest: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    if kernels.is_empty() && manifest.buffers.iter().any(|b| b.name.starts_with("E8P_")) {
        return Err("Missing exact expert execution contracts".into());
    }
    let contracts = super::contracts::kernels(kernels);
    let mut splits = BTreeMap::new();
    for kernel in kernels {
        if let Some(projection) = kernel.name.strip_suffix("/a8_split") {
            let target = contracts
                .get(projection)
                .ok_or("Missing exact expert projection")?;
            if !exact(target)
                || !row_arguments(target, 13, true)
                || kernel.symbol != "pack_exact_a8_kernel"
                || !row_arguments(kernel, 4, true)
            {
                return Err("Unexpected exact expert ABI".into());
            }
            connect(kernel, target, &[(1, 3), (2, 4), (3, 8)])?;
            splits.insert(projection, kernel.name.as_str());
        }
    }
    // The fused dispatch in large prefill can produce the planes itself.
    let fused = kernels
        .iter()
        .filter(|k| k.symbol == "dispatch_planes_kernel")
        .filter_map(|k| planes(k, [2, 4, 9]))
        .collect::<BTreeSet<_>>();
    for kernel in kernels.iter().filter(|k| exact(k)) {
        if splits.contains_key(kernel.name.as_str())
            || kernel.name.starts_with("prefill_m4096/")
                && planes(kernel, [3, 4, 8]).is_some_and(|p| fused.contains(&p))
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
            .filter(|k| splits.contains_key(k.name.as_str()));
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
    let mut missing: BTreeSet<_> = splits.keys().copied().collect();
    for operations in manifest.programs.values_mut() {
        let mut expanded = Vec::with_capacity(operations.len());
        for operation in std::mem::take(operations) {
            if let Operation::Kernel { name } = &operation
                && let Some(split) = splits.get(name.as_str())
            {
                expanded.push(Operation::Kernel {
                    name: (*split).into(),
                });
                missing.remove(name.as_str());
            }
            expanded.push(operation);
        }
        *operations = expanded;
    }
    if !missing.is_empty() {
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
    let mut banks = (0..48)
        .map(|layer| {
            let buffers_for_layer = ["ffn_gate_up_exps", "ffn_down_exps"]
                .into_iter()
                .flat_map(|role| {
                    ["packed", "scale"].map(|part| format!("W_blk.{layer}.{role}.weight_{part}"))
                })
                .collect::<Vec<_>>();
            let experts = buffers
                .get(buffers_for_layer[0].as_str())
                .and_then(|b| b.shape.first())
                .copied()
                .ok_or("Missing expert bank")?;
            Ok(ExpertBank {
                experts,
                buffers: buffers_for_layer,
                consumers: BTreeMap::new(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut gates: BTreeMap<_, _> = banks
        .iter_mut()
        .map(|bank| {
            (
                bank.buffers[0].as_str(),
                (bank.experts, &mut bank.consumers),
            )
        })
        .collect();
    for kernel in kernels {
        for arg in &kernel.args {
            let Argument::Buffer { name } = arg else {
                continue;
            };
            let Some((experts, consumers)) = gates.get_mut(name.as_str()) else {
                continue;
            };
            // Host ABI order: the exact path adds the two planes and sum.
            let histogram = if exact(kernel) && row_arguments(kernel, 13, true) {
                2
            } else if row_arguments(
                kernel,
                11,
                manifest.dynamic_batch_kernels.contains_key(&kernel.name),
            ) {
                4
            } else {
                return Err(format!("{}: unsupported expert consumer ABI", kernel.name));
            };
            let b = buffers
                .get(buffer(kernel, histogram)?)
                .ok_or("Missing expert histogram")?;
            if b.dtype != Dtype::I32 || b.shape != [*experts] {
                return Err(format!("{}: invalid expert histogram", kernel.name));
            }
            consumers.insert(kernel.name.clone(), histogram);
        }
    }
    if !kernels.is_empty() && banks.iter().any(|bank| bank.consumers.is_empty()) {
        return Err("Missing expert gate/up consumers".into());
    }
    Ok(banks)
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
