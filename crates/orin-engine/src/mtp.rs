//! Manifest-defined MTP programs and distribution-preserving speculation.
use crate::{
    artifact::{Access, Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub position: String,
    pub input: String,
    pub token: String,
    pub status: String,
    pub verification_tokens: String,
    pub verification_status: String,
    pub draft_logits: String,
    pub verification_logits: String,
    pub feature_index: Option<String>,
    pub accepted_inputs: String,
    pub target_length: String,
    pub draft_program: String,
    pub default_verification_tokens: usize,
    pub warm_plans: Vec<WarmPlan>,
    pub capture_plans: Vec<CapturePlan>,
    pub verification_plans: Vec<VerificationPlan>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WarmPlan {
    pub tokens: usize,
    pub program: String,
    pub head_program: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapturePlan {
    pub tokens: usize,
    pub program: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationPlan {
    pub tokens: usize,
    pub program: String,
    pub restore_program: String,
    pub capture_program: String,
}

impl Spec {
    pub fn validate(&self, model: &Manifest) -> Result<()> {
        let program = |name: &str| -> Result<()> {
            if model.programs.get(name).is_none_or(Vec::is_empty) {
                return Err(format!("Missing MTP program {name}"));
            }
            Ok(())
        };
        program(&self.draft_program)?;
        let mut capture_sizes = BTreeSet::new();
        for p in &self.capture_plans {
            if p.tokens == 0 || p.tokens > model.chunk_tokens || !capture_sizes.insert(p.tokens) {
                return Err("Invalid/duplicate MTP capture plan".into());
            }
            program(&p.program)?;
        }
        if !capture_sizes.contains(&1)
            || (model.prefill_plans.is_empty() && !capture_sizes.contains(&model.chunk_tokens))
            || model
                .prefill_plans
                .iter()
                .any(|p| !capture_sizes.contains(&p.chunk_tokens))
        {
            return Err("MTP capture must cover every prefill plan and single-token decode".into());
        }
        let mut warm_sizes = BTreeSet::new();
        for p in &self.warm_plans {
            if p.tokens == 0 || p.tokens > model.max_context || !warm_sizes.insert(p.tokens) {
                return Err("Invalid/duplicate MTP warm plan".into());
            }
            program(&p.program)?;
            program(&p.head_program)?;
        }
        if !warm_sizes.contains(&1) {
            return Err("MTP needs a one-token warm fallback".into());
        }
        let mut verify_sizes = BTreeSet::new();
        for p in &self.verification_plans {
            if p.tokens < 2 || p.tokens > model.max_context || !verify_sizes.insert(p.tokens) {
                return Err("Invalid/duplicate MTP verification plan".into());
            }
            for name in [&p.program, &p.restore_program, &p.capture_program] {
                program(name)?;
            }
        }
        if !verify_sizes.contains(&self.default_verification_tokens) {
            return Err("Missing default MTP verification shape".into());
        }
        let max_warm = *warm_sizes.last().ok_or("No MTP warm plans")?;
        let max_verify = *verify_sizes.last().ok_or("No MTP verification plans")?;
        for (name, rows) in [
            (&self.draft_logits, 1),
            (&self.verification_logits, max_verify),
        ] {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP logits buffer")?;
            if buffer.dtype != Dtype::F32
                || buffer.access == Access::Read
                || buffer.shape != [rows, model.vocab]
            {
                return Err(format!("Invalid MTP logits {name}"));
            }
        }
        match (&model.vision, &self.feature_index) {
            (Some(vision), Some(name)) => {
                let buffer = model
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .ok_or("Missing shifted MTP feature index")?;
                if name == &vision.feature_index
                    || buffer.dtype != Dtype::I32
                    || buffer.access == Access::Read
                    || buffer.shape != [model.max_context]
                {
                    return Err("Invalid shifted MTP feature index".into());
                }
            }
            (None, None) => {}
            _ => return Err("MTP multimodal inputs must match the vision adapter".into()),
        }
        let mut controls = BTreeSet::new();
        for (name, words) in [
            (&self.position, 1),
            (&self.input, max_warm),
            (&self.token, 1),
            (&self.status, 1),
            (&self.accepted_inputs, 1),
            (&self.verification_tokens, max_verify),
            (&self.verification_status, max_verify),
        ] {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP control buffer")?;
            if buffer.dtype != Dtype::I32
                || buffer.access == Access::Read
                || buffer.bytes()? < words.checked_mul(4).ok_or("MTP size overflow")?
                || !controls.insert(name)
                || [&model.position, &model.input, &model.token, &model.status].contains(&name)
            {
                return Err(format!("Invalid/aliased MTP control {name}"));
            }
        }
        if !model.reset_buffers.contains(&self.position) {
            return Err("MTP position must reset between requests".into());
        }
        let length = model
            .buffers
            .iter()
            .find(|b| b.name == self.target_length)
            .ok_or("Missing target sequence length")?;
        if length.dtype != Dtype::I32
            || length.access == Access::Read
            || length.bytes()? != 4
            || controls.contains(&self.target_length)
            || [&model.position, &model.input, &model.token, &model.status]
                .contains(&&self.target_length)
        {
            return Err("Invalid/aliased target sequence length".into());
        }
        Ok(())
    }
}

/// Each round always commits a target-selected correction or bonus token.
/// The matching draft prefix is accepted; later proposals have no authority.
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
