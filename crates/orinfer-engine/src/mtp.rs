//! Manifest-defined MTP programs and distribution-preserving speculation.
use crate::artifact::Result;
#[cfg(test)]
use crate::model::Manifest;
use serde::Serialize;

pub use orinfer_model_sdk::mtp::*;

pub fn greedy_commit(drafts: &[u32], target: &[u32]) -> Result<Vec<u32>> {
    if target.len() != drafts.len() + 1 || target.is_empty() {
        return Err("MTP verification must include one target token beyond drafts".into());
    }
    let accepted = drafts
        .iter()
        .zip(target)
        .take_while(|(a, b)| a == b)
        .count();
    let mut committed = drafts[..accepted].to_vec();
    committed.push(target[accepted]);
    Ok(committed)
}

// Separate counter streams keep proposal, acceptance and correction draws
// independent. Counters follow committed output positions, never graph rows.
pub const DRAFT_STREAM: u64 = 0x4d54_5001;
const ACCEPT_STREAM: u64 = 0x4d54_5002;
const CORRECTION_STREAM: u64 = 0x4d54_5003;

pub struct Proposal {
    pub token: u32,
    pub distribution: crate::sampling::Distribution,
}

/// Exact rejection sampling: accept x with min(1,p(x)/q(x)); otherwise
/// sample the positive residual. Discarded later rows have no authority.
pub fn sampled_commit(
    drafts: &[Proposal],
    target_logits: &[f32],
    history: &[u32],
    options: &crate::sampling::Options,
    step: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    if vocab == 0 || target_logits.len() != (drafts.len() + 1) * vocab {
        return Err("Invalid MTP verification logits extent".into());
    }
    let mut history = history.to_vec();
    let mut committed = Vec::with_capacity(drafts.len() + 1);
    for (i, draft) in drafts.iter().enumerate() {
        let p = crate::sampling::Distribution::from_logits(
            &target_logits[i * vocab..(i + 1) * vocab],
            &history,
            options,
        )?;
        let qx = draft.distribution.probability(draft.token);
        if qx <= 0.0 {
            return Err("MTP proposal outside its draft distribution".into());
        }
        let uniform =
            crate::sampling::counter_uniform(options.seed, ACCEPT_STREAM, (step + i) as u64);
        if uniform < (p.probability(draft.token) / qx).min(1.0) {
            committed.push(draft.token);
            history.push(draft.token);
        } else {
            let residual = p.residual(&draft.distribution)?;
            committed.push(residual.draw(crate::sampling::counter_uniform(
                options.seed,
                CORRECTION_STREAM,
                (step + i) as u64,
            ))?);
            return Ok(committed);
        }
    }
    let p = crate::sampling::Distribution::from_logits(
        &target_logits[drafts.len() * vocab..],
        &history,
        options,
    )?;
    committed.push(p.draw(crate::sampling::counter_uniform(
        options.seed,
        0,
        (step + drafts.len()) as u64,
    ))?);
    Ok(committed)
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Statistics {
    pub rounds: usize,
    pub proposed_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub committed_tokens: usize,
    pub initial_warm_s: f64,
    pub draft_s: f64,
    pub verification_s: f64,
    pub sampling_s: f64,
    pub restore_s: f64,
    pub refresh_s: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
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
            chunk_tokens: 8,
            prefill_program: "execute".into(),
            head_program: "shared_head".into(),
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
}
