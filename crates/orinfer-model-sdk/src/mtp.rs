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
    /// When present, consume each target prefill chunk before reusing this ring.
    #[serde(default)]
    pub hidden_ring: Option<String>,
    /// Compact recurrent verification needs a commit even on full acceptance.
    #[serde(default)]
    pub commit_always: bool,
    /// Draft physical state that cannot be repaired by rewinding its cursor.
    #[serde(default)]
    pub draft_snapshot_program: Option<String>,
    #[serde(default)]
    pub draft_restore_program: Option<String>,
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
        match (&self.draft_snapshot_program, &self.draft_restore_program) {
            (Some(save), Some(restore)) => {
                program(save)?;
                program(restore)?;
            }
            (None, None) => {}
            _ => return Err("MTP draft snapshot/restore must be paired".into()),
        }
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
        if let Some(name) = &self.hidden_ring {
            let buffer = model
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing MTP hidden ring")?;
            if buffer.dtype != Dtype::F16
                || buffer.access == Access::Read
                || buffer.shape.len() != 2
                || buffer.shape[1] == 0
                || buffer.shape[0] < model.chunk_tokens.max(max_warm).max(max_verify)
                || buffer.data.is_some()
            {
                return Err(
                    "MTP hidden ring must hold every capture/warm/verification chunk".into(),
                );
            }
        }
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
