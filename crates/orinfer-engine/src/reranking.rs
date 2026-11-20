//! Independent causal reranking through the shared native CUDA executor.
pub use crate::encoder_rows::Statistics;
use crate::{
    artifact::Result,
    encoder_rows::LoadedEncoder,
    execution::{ExecutionPhase, LoadOptions},
};
pub use orinfer_model_sdk::reranking::Spec;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};

#[derive(Debug, Serialize)]
pub struct Score {
    pub no_logit: f32,
    pub yes_logit: f32,
    pub logit_difference: f64,
    pub relevance_score: f64,
}
pub struct Model(LoadedEncoder<Spec>);
impl Model {
    pub fn load(path: &Path, options: LoadOptions) -> Result<Self> {
        LoadedEncoder::load(
            path,
            options,
            |m| m.reranking.clone(),
            "Model package does not provide reranking",
            "Reranking weight accounting overflow",
        )
        .map(Self)
    }
    pub fn max_context(&self) -> usize {
        self.0.context
    }
    pub fn spec(&self) -> &Spec {
        &self.0.spec
    }
    pub fn frontend_assets(&self) -> &BTreeMap<String, String> {
        &self.0.frontend_assets
    }
    pub fn statistics(&self) -> &Statistics {
        &self.0.statistics
    }
    pub fn score_many(&mut self, inputs: &[&[u32]]) -> Result<Vec<Score>> {
        let Some(rows) = self.0.spec.rows.clone() else {
            return inputs.iter().map(|tokens| self.score(tokens)).collect();
        };
        if inputs.iter().any(|t| !self.0.tokens_valid(t)) {
            return Err("Invalid reranking tokens or context length".into());
        }
        let packets = rows.packets(
            &inputs.iter().map(|t| t.len()).collect::<Vec<_>>(),
            self.0.context,
        )?;
        let mut output = Vec::with_capacity(inputs.len());
        for packet in packets {
            let mut tokens = vec![0; packet.rows];
            for (owner, &[start, end]) in packet.segments.iter().enumerate() {
                tokens[start as usize..end as usize].copy_from_slice(inputs[packet.first + owner]);
            }
            let e = &self.0.execution;
            e.upload_ids(&self.0.input, &tokens)?;
            crate::encoder_rows::upload(e, &rows, &packet)?;
            e.submit_program_rows(&self.0.spec.program, packet.rows, ExecutionPhase::Prefill)?;
            for pair in e
                .download_bytes(&self.0.spec.output, packet.segments.len() * 8)?
                .as_chunks::<8>()
                .0
            {
                output.push(score_from_bytes(pair)?);
            }
        }
        Ok(output)
    }
    pub fn score(&mut self, tokens: &[u32]) -> Result<Score> {
        if self.0.spec.rows.is_some() {
            return Ok(self.score_many(&[tokens])?.remove(0));
        }
        if !self.0.tokens_valid(tokens) {
            return Err("Invalid reranking tokens or context length".into());
        }
        self.0.execution.upload_ids(&self.0.input, tokens)?;
        self.0
            .execution
            .upload_ids(&self.0.spec.length, &[tokens.len() as u32])?;
        self.0.execution.submit_program_rows(
            &self.0.spec.program,
            tokens.len(),
            ExecutionPhase::Prefill,
        )?;
        let bytes = self.0.execution.download_bytes(&self.0.spec.output, 8)?;
        score_from_bytes(&bytes)
    }
}
fn score_from_bytes(bytes: &[u8]) -> Result<Score> {
    let no_logit = f32::from_le_bytes(bytes[..4].try_into().unwrap());
    let yes_logit = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if !no_logit.is_finite() || !yes_logit.is_finite() {
        return Err("Reranking produced nonfinite logits".into());
    }
    let difference = yes_logit as f64 - no_logit as f64;
    let relevance_score = probability(difference);
    Ok(Score {
        no_logit,
        yes_logit,
        logit_difference: difference,
        relevance_score,
    })
}
fn probability(difference: f64) -> f64 {
    if difference >= 0.0 {
        1.0 / (1.0 + (-difference).exp())
    } else {
        let e = difference.exp();
        e / (1.0 + e)
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn binary_probability_is_stable_for_extreme_logits() {
        assert_eq!(super::probability(0.0), 0.5);
        assert_eq!(super::probability(1000.0), 1.0);
        assert_eq!(super::probability(-1000.0), 0.0);
        assert!((super::probability(1.0) - 0.7310585786300049).abs() < 1e-15);
    }
}
