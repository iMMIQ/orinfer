/// Select common profiles from architecture-owned plans, never kernel/model names.
pub(super) fn kernels(
    manifest: &crate::model::Manifest,
    package: &crate::model_package::ModelPackage,
) -> crate::artifact::Result<std::collections::BTreeSet<String>> {
    use std::collections::BTreeSet;
    let programs = programs(manifest);
    let mut kernels = BTreeSet::new();
    for name in programs {
        let ops = manifest
            .programs
            .get(&name)
            .ok_or_else(|| format!("Missing startup program {name}"))?;
        kernels.extend(ops.iter().filter_map(|op| match op {
            crate::model::Operation::Kernel { name } => Some(name.clone()),
            _ => None,
        }));
    }
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
            kernels.extend(package.batch_plan(&segments)?.into_iter().filter_map(|op| {
                match op.operation {
                    crate::model::Operation::Kernel { name } => Some(name),
                    _ => None,
                }
            }));
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
            if plan.tokens <= spec.default_verification_tokens {
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
