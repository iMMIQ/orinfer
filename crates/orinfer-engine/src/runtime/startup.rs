/// Select common profiles from architecture-owned plans, never kernel/model names.
pub(super) fn kernels(
    manifest: &crate::model::Manifest,
    adapter: &dyn orinfer_models::Adapter,
) -> crate::artifact::Result<std::collections::BTreeSet<String>> {
    use std::collections::BTreeSet;
    let mut kernels = program_kernels(manifest)?;
    if !manifest.batch_profiles.is_empty() {
        let mut shapes: BTreeSet<_> = manifest
            .batch_profiles
            .iter()
            .copied()
            .filter(|&n| n <= 8)
            .collect();
        if !manifest.dynamic_batch_kernels.is_empty() {
            shapes.insert(3);
        }
        for rows in shapes {
            let segments: Vec<_> = (0..rows)
                .map(|slot| orinfer_model_sdk::architecture::BatchSegment { slot, tokens: 1 })
                .collect();
            kernels.extend(
                adapter
                    .batch_operation(
                        &segments,
                        orinfer_model_sdk::packet::BatchOperation::Ordinary,
                        true,
                    )?
                    .0
                    .into_iter()
                    .filter_map(|op| match op.operation {
                        crate::model::Operation::Kernel { name } => Some(name),
                        _ => None,
                    }),
            );
        }
    }
    Ok(kernels)
}

fn program_kernels(
    manifest: &crate::model::Manifest,
) -> crate::artifact::Result<std::collections::BTreeSet<String>> {
    use crate::model::Operation;
    use std::collections::BTreeSet;
    let mut kernels = BTreeSet::new();
    let mut packets = BTreeSet::new();
    let table = manifest.packet_table();
    if let Some(spec) = manifest.mtp.as_ref().filter(|_| table.is_some()) {
        packets.insert(spec.draft_program.as_str());
        for plan in &spec.warm_plans {
            packets.extend([plan.program.as_str(), plan.head_program.as_str()]);
        }
        for plan in &spec.verification_plans {
            packets.extend([plan.program.as_str(), plan.restore_program.as_str()]);
        }
    }
    for program in programs(manifest) {
        let ops = manifest
            .programs
            .get(&program)
            .ok_or_else(|| format!("Missing startup program {program}"))?;
        let indirect = table.filter(|_| packets.contains(program.as_str()));
        for operation in ops {
            match operation {
                Operation::Kernel { name } => {
                    kernels.insert(
                        indirect
                            .and_then(|table| table.kernels.get(name))
                            .unwrap_or(name)
                            .clone(),
                    );
                }
                Operation::Copy {
                    source,
                    destination,
                    ..
                } => {
                    if let Some(copy) = indirect.and_then(|table| {
                        table
                            .copies
                            .iter()
                            .find(|copy| &copy.source == source && &copy.destination == destination)
                    }) {
                        kernels.insert(copy.kernel.clone());
                    }
                }
                _ => {}
            }
        }
    }
    Ok(kernels)
}

fn programs(manifest: &crate::model::Manifest) -> std::collections::BTreeSet<String> {
    use std::collections::BTreeSet;
    let mut programs = BTreeSet::from(["decode".to_string(), "head".to_string()]);
    let smallest = manifest
        .prefill_plans
        .iter()
        .map(|p| p.chunk_tokens)
        .min()
        .unwrap_or(manifest.chunk_tokens);
    if manifest.prefill_plans.is_empty() {
        programs.insert("prefill".into());
    } else {
        for plan in &manifest.prefill_plans {
            if plan.chunk_tokens <= 512 || plan.chunk_tokens == smallest {
                programs.extend([plan.prefill_program.clone(), plan.head_program.clone()]);
            }
        }
    }
    if let Some(vision) = &manifest.vision {
        programs.extend(vision.plans.iter().map(|p| p.program.clone()));
    }
    if let Some(spec) = &manifest.mtp {
        programs.insert(spec.draft_program.clone());
        programs.extend(spec.draft_snapshot_program.iter().cloned());
        programs.extend(spec.draft_restore_program.iter().cloned());
        for plan in &spec.warm_plans {
            if plan.tokens <= 512 || plan.tokens == smallest {
                programs.extend([plan.program.clone(), plan.head_program.clone()]);
            }
        }
        for plan in &spec.capture_plans {
            if plan.tokens <= 512 || plan.tokens == smallest {
                programs.insert(plan.program.clone());
            }
        }
        // Short remaining budgets may use fewer drafts than the configured count.
        for plan in &spec.verification_plans {
            if plan.tokens <= spec.default_verification_tokens || plan.min_tokens > 0 {
                programs.extend([
                    plan.program.clone(),
                    plan.restore_program.clone(),
                    plan.capture_program.clone(),
                ]);
            }
        }
    }
    programs
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn packet_preload_uses_indirect_kernels_and_copies_and_retains_direct_consumers() {
        let mut manifest: crate::model::Manifest = serde_json::from_str(r#"{
            "schema_version":1,"target":"sm_87","model":"unknown-family","chunk_tokens":1,
            "max_context":64,"vocab":32,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"i","token":"t","status":"s","logits":"l","position":"p",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "programs":{
                "decode":[{"kind":"kernel","name":"common_private"}],
                "head":[{"kind":"kernel","name":"shared_head"}],
                "prefill":[{"kind":"kernel","name":"shared_prefill"}],
                "draft":[{"kind":"kernel","name":"private_draft"}],
                "warm":[{"kind":"kernel","name":"private_warm"}],
                "warm_head":[{"kind":"copy","source":"Output","destination":"PrivateOutput","bytes":4}],
                "verify":[{"kind":"kernel","name":"common_private"}],
                "restore":[{"kind":"kernel","name":"private_restore"}],
                "capture":[{"kind":"kernel","name":"private_capture"}],
                "snapshot":[{"kind":"kernel","name":"private_snapshot"}]
            },
            "mtp":{
                "position":"p","input":"i","token":"t","status":"s","verification_tokens":"v",
                "verification_status":"vs","draft_logits":"d","verification_logits":"vl",
                "accepted_inputs":"a","target_length":"tl","draft_program":"draft",
                "draft_snapshot_program":"snapshot","default_verification_tokens":2,
                "warm_plans":[{"tokens":1,"program":"warm","head_program":"warm_head"}],
                "capture_plans":[{"tokens":1,"program":"capture"}],
                "verification_plans":[{"tokens":2,"program":"verify","restore_program":"restore","capture_program":"capture"}],
                "packet_table":{
                    "buffer":"Addresses","columns":["State"],
                    "kernels":{"common_private":"packet_common","private_draft":"packet_draft",
                        "private_warm":"packet_warm","private_restore":"packet_restore",
                        "private_capture":"packet_capture","private_snapshot":"packet_snapshot"},
                    "copies":[{"source":"Output","destination":"PrivateOutput","kernel":"packet_copy","row_bytes":4}]
                }
            }
        }"#).unwrap();
        assert_eq!(
            program_kernels(&manifest).unwrap(),
            [
                "common_private",
                "shared_head",
                "shared_prefill",
                "packet_common",
                "packet_draft",
                "packet_warm",
                "packet_restore",
                "packet_copy",
                "private_capture",
                "private_snapshot",
            ]
            .map(str::to_owned)
            .into_iter()
            .collect()
        );
        manifest.mtp.as_mut().unwrap().packet_table = None;
        let direct = program_kernels(&manifest).unwrap();
        assert!(direct.contains("private_draft") && direct.contains("private_warm"));
        assert!(!direct.iter().any(|name| name.starts_with("packet_")));
    }
    #[test]
    fn common_profiles_and_smallest_fallback_keep_architecture_program_names() {
        let mut manifest: crate::model::Manifest = serde_json::from_value(json!({
            "schema_version":1,"target":"sm_87","model":"unknown-family","chunk_tokens":4096,
            "max_context":262144,"vocab":32,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"i","token":"t","status":"s","logits":"l","position":"p",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "prefill_plans":[
                {"chunk_tokens":16,"prefill_program":"short_prompt","head_program":"short_head"},
                {"chunk_tokens":512,"prefill_program":"medium_prompt","head_program":"medium_head"},
                {"chunk_tokens":4096,"prefill_program":"large_prompt","head_program":"large_head"}
            ]
        }))
        .unwrap();
        let selected = programs(&manifest);
        assert!(selected.contains("short_prompt") && selected.contains("medium_head"));
        assert!(!selected.contains("large_prompt") && !selected.contains("prefill"));
        manifest.prefill_plans.drain(..2);
        assert!(programs(&manifest).contains("large_head"));
        manifest.prefill_plans.clear();
        assert!(programs(&manifest).contains("prefill"));
        manifest.mtp = Some(serde_json::from_value(json!({
            "position":"p","input":"i","token":"t","status":"s","verification_tokens":"v",
            "verification_status":"vs","draft_logits":"d","verification_logits":"vl",
            "accepted_inputs":"a","target_length":"tl","draft_program":"draft",
            "default_verification_tokens":4,
            "warm_plans":[{"tokens":4096,"program":"warm_large","head_program":"warm_head"}],
            "capture_plans":[{"tokens":4096,"program":"capture_large"}],
            "verification_plans":[
                {"tokens":2,"program":"verify2","restore_program":"restore2","capture_program":"capture2"},
                {"tokens":4,"program":"verify4","restore_program":"restore4","capture_program":"capture4"},
                {"tokens":8,"program":"verify8","restore_program":"restore8","capture_program":"capture8"}
            ]
        })).unwrap());
        let selected = programs(&manifest);
        assert!(selected.contains("verify2") && selected.contains("restore4"));
        assert!(!selected.contains("verify8"));
        assert!(selected.contains("warm_large") && selected.contains("capture_large"));
    }
}
