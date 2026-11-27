use super::*;
#[test]
fn rejected_future_noise_is_not_reused_by_adaptive_rounds() {
    let position = 12345;
    assert_ne!(round_step(0, position).unwrap(), position);
    for round in 1..32 {
        let previous = round_step(round - 1, position).unwrap();
        let current = round_step(round, position).unwrap();
        assert_ne!(
            crate::sampling::categorical_key(20261002, previous as u64),
            crate::sampling::categorical_key(20261002, current as u64)
        );
        assert_eq!(round_step(round, position + 1).unwrap(), current + 1);
    }
    assert!(round_step(u32::MAX as usize, position).is_err());
    assert!(round_step(0, (1u64 << 32) as usize).is_err());
}
#[test]
fn rejection_preserves_target_distribution() {
    use crate::sampling::{Distribution, Options, counter_uniform};
    let q_logits = [0.7f32.ln(), 0.2f32.ln(), 0.1f32.ln()];
    let p_logits = [0.1f32.ln(), 0.3f32.ln(), 0.6f32.ln()];
    let mut counts = [0usize; 3];
    for seed in 0..20000 {
        let options = Options {
            seed,
            ..Options::default()
        };
        let q = Distribution::from_logits(&q_logits, &[], &options).unwrap();
        let token = q.draw(counter_uniform(seed, DRAFT_STREAM, 0)).unwrap();
        let drafts = [Proposal {
            token,
            distribution: q,
        }];
        let logits = [p_logits.as_slice(), p_logits.as_slice()].concat();
        let first = sampled_commit(&drafts, &logits, &[], &options, 0, 3).unwrap()[0];
        counts[first as usize] += 1;
    }
    for (actual, expected) in counts.iter().zip([0.1, 0.3, 0.6]) {
        assert!(
            (*actual as f64 / 20000.0 - expected).abs() < 0.015,
            "{counts:?}"
        );
    }
}
#[test]
fn rejection_ignores_later_rows_and_penalties_use_committed_history() {
    use crate::sampling::{Distribution, Options};
    let options = Options {
        temperature: 0.0,
        repetition_penalty: 2.0,
        ..Options::default()
    };
    let q = Distribution::from_logits(&[2., 1.], &[], &options).unwrap();
    let drafts = [Proposal {
        token: 0,
        distribution: q,
    }];
    assert_eq!(
        sampled_commit(&drafts, &[1., 2., f32::NAN, f32::NAN], &[], &options, 0, 2).unwrap(),
        [1]
    );
    let q = Distribution::from_logits(&[3., 2.], &[], &options).unwrap();
    let drafts = [Proposal {
        token: 0,
        distribution: q,
    }];
    assert_eq!(
        sampled_commit(&drafts, &[3., 2., 3., 2.], &[], &options, 0, 2).unwrap(),
        [0, 1]
    );
}
fn manifest() -> Manifest {
    let controls = [
        ("Step", 1),
        ("Input", 8),
        ("Token", 1),
        ("Status", 1),
        ("Length", 1),
        ("MtpStep", 1),
        ("MtpInput", 4),
        ("MtpToken", 1),
        ("MtpStatus", 1),
        ("Accepted", 1),
        ("SequenceTokens", 4),
        ("SequenceStatus", 4),
    ];
    let mut buffers: Vec<_> = controls
        .iter()
        .map(|&(name, words)| {
            serde_json::json!({
        "name":name,"dtype":"i32","shape":[words],"layout":"contiguous",
        "alignment":256,"access":"read_write","data":null})
        })
        .collect();
    for (name, rows) in [("MtpLogits", 1), ("SequenceLogits", 4)] {
        buffers.push(
            serde_json::json!({"name":name,"dtype":"f32","shape":[rows,100],
            "layout":"contiguous","alignment":256,"access":"read_write","data":null}),
        );
    }
    serde_json::from_value(serde_json::json!({
        "schema_version":1,"target":"sm_87","model":"test","chunk_tokens":8,
        "max_context":32,"vocab":100,"toolchain":{},"buffers":buffers,"kernels":[],
        "programs":{"execute":[{"kind":"zero","destination":"Step","bytes":4}]},
        "reset_buffers":["Step","MtpStep"],"input":"Input","token":"Token",
        "status":"Status","logits":"Logits","position":"Step",
        "weight_bytes":0,"weight_parameters":1,"weight_scope":"test"}))
    .unwrap()
}
fn spec() -> Spec {
    serde_json::from_value(serde_json::json!({
        "position":"MtpStep","input":"MtpInput","token":"MtpToken","status":"MtpStatus",
        "verification_tokens":"SequenceTokens","verification_status":"SequenceStatus",
        "draft_logits":"MtpLogits","verification_logits":"SequenceLogits","feature_index":null,
        "accepted_inputs":"Accepted","target_length":"Length","draft_program":"execute",
        "default_verification_tokens":4,
        "warm_plans":[{"tokens":1,"program":"execute","head_program":"execute"},
                      {"tokens":4,"program":"execute","head_program":"execute"}],
        "capture_plans":[{"tokens":1,"program":"execute"},{"tokens":8,"program":"execute"}],
        "verification_plans":[{"tokens":4,"program":"execute","restore_program":"execute","capture_program":"execute"}]
    })).unwrap()
}
#[test]
fn draft_override_uses_only_an_exact_package_profile() {
    use crate::execution::LoadOptions;
    let mut model = manifest();
    let options = |drafts| LoadOptions {
        mtp_drafts: Some(drafts),
        ..LoadOptions::default()
    };
    assert!(options(7).configure_mtp(&mut model).is_err());
    model.mtp = Some(spec());
    LoadOptions::default().configure_mtp(&mut model).unwrap();
    assert_eq!(model.mtp.as_ref().unwrap().default_verification_tokens, 4);
    assert!(options(7).configure_mtp(&mut model).is_err());
    assert!(options(8).configure_mtp(&mut model).is_err());
    let mut profile = model.mtp.as_ref().unwrap().verification_plans[0].clone();
    profile.tokens = 8;
    model.mtp.as_mut().unwrap().verification_plans.push(profile);
    options(7).configure_mtp(&mut model).unwrap();
    assert_eq!(model.mtp.as_ref().unwrap().default_verification_tokens, 8);
    options(3).configure_mtp(&mut model).unwrap();
    assert_eq!(model.mtp.as_ref().unwrap().default_verification_tokens, 4);
    options(0).configure_mtp(&mut model).unwrap();
    assert!(model.mtp.is_none());
    options(0).configure_mtp(&mut model).unwrap();
}
#[test]
fn disabling_mtp_preserves_target_capture_and_shared_head() {
    let mut model = manifest();
    let mut mtp = spec();
    mtp.draft_program = "draft".into();
    for p in &mut mtp.warm_plans {
        p.program = "draft".into();
        p.head_program = "shared_head".into();
    }
    for p in &mut mtp.capture_plans {
        p.program = "capture".into();
    }
    for p in &mut mtp.verification_plans {
        p.program = "verify".into();
        p.restore_program = "rollback".into();
        p.capture_program = "capture".into();
    }
    let ops = model.programs["execute"].clone();
    for name in ["draft", "verify", "rollback", "capture", "shared_head"] {
        model.programs.insert(name.into(), ops.clone());
    }
    model.prefill_plans.push(crate::model::PrefillPlan {
        min_tokens: 0,
        chunk_tokens: 8,
        prefill_program: "execute".into(),
        head_program: "shared_head".into(),
        stages: vec![],
        live_buffers: vec![],
    });
    model.mtp = Some(mtp);
    crate::execution::LoadOptions {
        mtp_drafts: Some(0),
        ..Default::default()
    }
    .configure_mtp(&mut model)
    .unwrap();
    assert!(model.mtp.is_none());
    for name in ["execute", "capture", "shared_head"] {
        assert!(model.programs.contains_key(name));
    }
    for name in ["draft", "verify", "rollback"] {
        assert!(!model.programs.contains_key(name));
    }
}
#[test]
fn disabling_mtp_excludes_unmapped_draft_kv_from_prefix_snapshots() {
    let mut model = manifest();
    let mut draft = spec();
    draft.draft_program = "draft".into();
    for plan in &mut draft.warm_plans {
        plan.program = "draft".into();
    }
    model.programs.insert("draft".into(), vec![]);
    model
        .reset_buffers
        .extend(["TargetKV", "DraftKV", "SharedKV"].map(String::from));
    model.kv_cache = Some(
        serde_json::from_value(serde_json::json!({
            "direct_prefill":true,"demand_mapping":true,
            "buffers":{"TargetKV":8,"DraftKV":8,"SharedKV":8},
            "growth":{
                "execute":{"position":"Step","tokens":1,"buffers":["TargetKV","SharedKV"]},
                "draft":{"position":"MtpStep","tokens":1,"buffers":["DraftKV","SharedKV"]}
            }
        }))
        .unwrap(),
    );
    model.mtp = Some(draft);
    crate::execution::LoadOptions {
        mtp_drafts: Some(0),
        ..Default::default()
    }
    .configure_mtp(&mut model)
    .unwrap();
    assert!(!model.reset_buffers.iter().any(|n| n == "DraftKV"));
    for name in ["TargetKV", "SharedKV"] {
        assert!(model.reset_buffers.iter().any(|n| n == name));
    }
    let kv = model.kv_cache.as_ref().unwrap();
    assert!(kv.buffers.contains_key("DraftKV")); // Retain virtual allocation.
    assert!(!kv.growth.contains_key("draft"));
    assert!(kv.growth.contains_key("execute"));
}
#[test]
fn draft_state_snapshot_requires_a_matching_restore_program() {
    let model = manifest();
    let mut spec = spec();
    assert!(!spec.commit_always);
    spec.draft_snapshot_program = Some("execute".into());
    assert!(spec.validate(&model).is_err());
    spec.draft_restore_program = Some("execute".into());
    assert!(spec.validate(&model).is_ok());
    spec.draft_restore_program = Some("absent".into());
    assert!(spec.validate(&model).is_err());
}
#[test]
fn rejects_hidden_ring_too_small_for_prefill() {
    let mut model = manifest();
    let mut spec = spec();
    spec.hidden_ring = Some("Ring".into());
    model.buffers.push(
        serde_json::from_value(serde_json::json!({
            "name":"Ring","dtype":"f16","shape":[8,16],"layout":"contiguous",
            "alignment":256,"access":"read_write","data":null
        }))
        .unwrap(),
    );
    assert!(spec.validate(&model).is_ok());
    model.buffers.last_mut().unwrap().shape[0] = 4;
    assert!(spec.validate(&model).is_err());
}

#[test]
fn rejects_control_aliases_missing_capture_and_request_reset() {
    let model = manifest();
    assert!(spec().validate(&model).is_ok());
    let mut broken = spec();
    broken.target_length = model.position.clone();
    assert!(broken.validate(&model).is_err());
    broken = spec();
    broken.token = broken.status.clone();
    assert!(broken.validate(&model).is_err());
    broken = spec();
    broken.capture_plans.pop();
    assert!(broken.validate(&model).is_err());
    let mut broken_model = manifest();
    broken_model.reset_buffers.clear();
    assert!(spec().validate(&broken_model).is_err());
    broken_model = manifest();
    broken_model
        .buffers
        .iter_mut()
        .find(|b| b.name == "SequenceTokens")
        .unwrap()
        .shape = vec![1];
    assert!(spec().validate(&broken_model).is_err());
}
#[test]
fn first_rejection_is_authoritative_even_if_later_tokens_match() {
    assert_eq!(greedy_commit(&[1, 2, 3], &[1, 9, 3, 4]).unwrap(), [1, 9]);
    assert_eq!(greedy_commit(&[1, 2, 3], &[9, 2, 3, 4]).unwrap(), [9]);
    assert_eq!(
        greedy_commit(&[1, 2, 3], &[1, 2, 3, 4]).unwrap(),
        [1, 2, 3, 4]
    );
    assert!(greedy_commit(&[1, 2], &[1, 2]).is_err());
}
