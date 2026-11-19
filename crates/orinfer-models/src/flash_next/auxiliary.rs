//! Fuse complete auxiliary producer chains in every target/draft program alias.
use super::contracts::{fuse, kernel_at, pointer, retire};
use crate::{
    artifact::{Argument, Kernel, Result},
    model::Manifest,
};
use std::collections::BTreeSet;

fn buffers(k: &Kernel) -> Vec<&Argument> {
    k.args
        .iter()
        .filter(|a| matches!(a, Argument::Buffer { .. } | Argument::BufferSlice { .. }))
        .collect()
}

fn fixed<const N: usize>(args: Vec<&Argument>) -> Result<[&Argument; N]> {
    args.try_into()
        .map_err(|_| "Invalid auxiliary fusion ABI".into())
}

fn connected<const N: usize>(links: [(&Argument, &Argument); N]) -> Result<()> {
    if links.into_iter().any(|(a, b)| !same_operand(a, b)) {
        return Err("Auxiliary fusion producer/consumer mismatch".into());
    }
    Ok(())
}

fn qsa_links(previous: &[&Kernel], args: Vec<&Argument>) -> Result<()> {
    let [merge, scores, result] = fixed(args)?;
    let [partial_scores, bounds, partial] = fixed(buffers(previous[0]))?;
    let [counts, count_scores, threshold, count_bounds] = fixed(buffers(previous[8]))?;
    let [offset_counts, _, _, offset_scores, offsets] = fixed(buffers(previous[9]))?;
    let [first_merge, first_scores, first_threshold, _] = fixed(buffers(previous[1]))?;
    connected([
        (merge, partial),
        (scores, partial_scores),
        (merge, first_merge),
        (scores, first_scores),
        (threshold, first_threshold),
        (counts, offset_counts),
        (scores, count_scores),
        (bounds, count_bounds),
        (scores, offset_scores),
        (result, offsets),
    ])?;
    for [hist, choose] in previous[2..8].as_chunks::<2>().0 {
        let [bins, hist_scores, hist_threshold, hist_bounds] = fixed(buffers(hist))?;
        let [choose_bins, choose_scores, choose_threshold, _] = fixed(buffers(choose))?;
        connected([
            (scores, hist_scores),
            (bounds, hist_bounds),
            (bins, choose_bins),
            (scores, choose_scores),
            (threshold, hist_threshold),
            (threshold, choose_threshold),
        ])?;
    }
    Ok(())
}

fn same_operand(a: &Argument, b: &Argument) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b) && pointer(a) == pointer(b)
}

pub(super) fn build(m: &mut Manifest, kernels: &[Kernel]) -> Result<()> {
    let contracts = super::contracts::kernels(kernels);
    let candidates = kernels.iter().filter(|k| !k.name.contains("/rows/"));
    let anchors = candidates.clone().filter(|k| {
        matches!(
            k.symbol.as_str(),
            "ple_norm_gate_kernel" | "qsa_select_merge_kernel"
        )
    });
    let mut partials: BTreeSet<_> = candidates
        .filter(|k| k.symbol == "qsa_select_partial_kernel")
        .map(|k| k.name.as_str())
        .collect();
    fuse(&mut m.programs, anchors, |_, operations, i, fused, skip| {
        let name = &fused.name;
        let selection = fused.symbol == "qsa_select_merge_kernel";
        let count = if fused.symbol == "ple_norm_gate_kernel" {
            2
        } else {
            10
        };
        let start = i
            .checked_sub(count)
            .ok_or("Incomplete auxiliary fusion chain")?;
        let parent = name.rsplit_once('/').ok_or("Missing auxiliary section")?.0;
        let previous = (start..i)
            .map(|at| {
                let k = kernel_at(&contracts, operations, at)?;
                if k.name.rsplit_once('/').map(|p| p.0) != Some(parent) {
                    return Err("Auxiliary fusion crosses sections".into());
                }
                Ok(k)
            })
            .collect::<Result<Vec<_>>>()?;
        let args = buffers(fused);
        if selection {
            if previous[0].symbol != "qsa_select_partial_kernel"
                || previous[0].name.contains("/rows/")
            {
                return Err("Missing QSA local selection producer".into());
            }
            partials.remove(previous[0].name.as_str());
            if let [table] = args.as_slice() {
                for old in previous {
                    let [owner_table] = fixed(buffers(old))?;
                    connected([(*table, owner_table)])?;
                }
            } else {
                qsa_links(&previous, args)?;
            }
        } else {
            // Sorted host ABI: Gated, Key, KeyWeight, Query, QueryWeight, Value.
            let [_, key, key_weight, query, query_weight, _] = fixed(args)?;
            let [_, key_input, key_scale] = fixed(buffers(previous[0]))?;
            let [_, query_input, query_scale] = fixed(buffers(previous[1]))?;
            connected([
                (key, key_input),
                (key_weight, key_scale),
                (query, query_input),
                (query_weight, query_scale),
            ])?;
        }
        retire(skip, start + usize::from(selection)..i)
    })?;
    if !partials.is_empty() {
        return Err("Unregistered auxiliary fusion".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Operation;

    fn kernel(name: &str, symbol: &str, args: &[&str]) -> Kernel {
        let asset = serde_json::json!({"file":"k","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({"name":name,"symbol":symbol,
            "module":asset,"source":asset,"host_abi":asset,"grid":[1,1,1],
            "block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
            "args":args.iter().map(|name| serde_json::json!({"kind":"buffer","name":name})).collect::<Vec<_>>()
        })).unwrap()
    }

    fn manifest(names: &[String]) -> Manifest {
        let ops: Vec<_> = names
            .iter()
            .map(|name| serde_json::json!({"kind":"kernel","name":name}))
            .collect();
        serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_87","model":"test",
            "chunk_tokens":4096,"max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status","logits":"Logits",
            "position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
            "programs":{"main":ops,"alias":ops}
        })).unwrap()
    }

    #[test]
    fn ple_fuses_every_alias_and_requires_both_norm_producers() {
        let mut kernels = vec![
            kernel("layer/kn", "main_kernel", &["NK", "Key", "KW"]),
            kernel("layer/qn", "main_kernel", &["NQ", "Query", "QW"]),
            kernel(
                "layer/gate",
                "ple_norm_gate_kernel",
                &["Gated", "Key", "KW", "Query", "QW", "Value"],
            ),
        ];
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 1));
        kernels[2].args[4] = Argument::Buffer {
            name: "OtherWeight".into(),
        };
        assert!(
            build(&mut manifest(&names), &kernels)
                .unwrap_err()
                .contains("producer/consumer")
        );
    }

    #[test]
    fn private_qsa_requires_one_owner_table_and_a_complete_section() {
        let mut kernels: Vec<_> = (0..11)
            .map(|i| {
                kernel(
                    &format!("private/k{i}"),
                    if i == 10 {
                        "qsa_select_merge_kernel"
                    } else if i == 0 {
                        "qsa_select_partial_kernel"
                    } else {
                        "main_kernel"
                    },
                    &["Owners"],
                )
            })
            .collect();
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 2));
        let mut orphan = kernels.clone();
        orphan.push(kernel(
            "orphan/partial",
            "qsa_select_partial_kernel",
            &["Owners"],
        ));
        assert!(build(&mut manifest(&names), &orphan).is_err());
        kernels[3].args[0] = Argument::Buffer {
            name: "OtherOwner".into(),
        };
        assert!(build(&mut manifest(&names), &kernels).is_err());
        assert!(build(&mut manifest(&names[1..]), &kernels).is_err());
        let mut m = manifest(&names);
        m.programs.get_mut("main").unwrap().insert(
            5,
            Operation::Copy {
                source: "A".into(),
                destination: "B".into(),
                bytes: 4,
            },
        );
        assert!(build(&mut m, &kernels).is_err());
    }

    #[test]
    fn local_qsa_keeps_both_stages_and_checks_the_bounded_union() {
        let mut kernels = vec![kernel(
            "qsa/k0",
            "qsa_select_partial_kernel",
            &["Position", "Scores", "Candidates"],
        )];
        for i in 1..8 {
            kernels.push(kernel(
                &format!("qsa/k{i}"),
                "main_kernel",
                if i % 2 == 0 {
                    &["Candidates", "Position", "Prefix", "Scores"]
                } else {
                    &["Candidates", "Position", "Prefix", "Remaining"]
                },
            ));
        }
        kernels.push(kernel(
            "qsa/k8",
            "main_kernel",
            &["Counts", "Position", "Prefix", "Scores"],
        ));
        kernels.push(kernel(
            "qsa/k9",
            "main_kernel",
            &["Counts", "Greater", "Offsets", "Position", "Selected"],
        ));
        kernels.push(kernel(
            "qsa/k10",
            "qsa_select_merge_kernel",
            &["Candidates", "Position", "Selected"],
        ));
        let names: Vec<_> = kernels.iter().map(|k| k.name.clone()).collect();
        let mut m = manifest(&names);
        build(&mut m, &kernels).unwrap();
        assert!(m.programs.values().all(|ops| ops.len() == 2));
        kernels[10].args[0] = Argument::Buffer {
            name: "OtherUnion".into(),
        };
        assert!(build(&mut manifest(&names), &kernels).is_err());
    }
}
