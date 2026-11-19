//! Independent causal reranking through the shared native CUDA executor.
pub use crate::embedding::Statistics;
use crate::{
    artifact::Result,
    backend::Executor,
    execution::{ExecutionPhase, LoadOptions},
};
pub use orinfer_model_sdk::reranking::Spec;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, time::Instant};

#[derive(Debug, Serialize)]
pub struct Score {
    pub no_logit: f32,
    pub yes_logit: f32,
    pub logit_difference: f64,
    pub relevance_score: f64,
}
pub struct Model {
    execution: Executor,
    spec: Spec,
    context: usize,
    vocab: usize,
    input: String,
    frontend_assets: BTreeMap<String, String>,
    statistics: Statistics,
}
impl Model {
    pub fn load(path: &Path, options: LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let prepared = crate::loader::load_with_experts(path, options.verify_weights, false)?;
        let m = &prepared.plan;
        let spec = m
            .reranking
            .clone()
            .ok_or("Model package does not provide reranking")?;
        let preload = m.kernels.iter().map(|k| k.name.clone()).collect();
        let (execution, stats) = Executor::load(&prepared, options, &preload)?;
        let padding = stats.load_stages.weight_arena_padding_bytes;
        let resident = m
            .weight_bytes
            .checked_add(padding)
            .ok_or("Reranking weight accounting overflow")?;
        let statistics = Statistics {
            load_to_ready_s: started.elapsed().as_secs_f64(),
            weight_bytes: m.weight_bytes,
            weight_arena_padding_bytes: padding,
            resident_weight_bytes: resident,
            effective_weight_bits: resident as f64 * 8.0 / m.weight_parameters as f64,
            allocated_bytes: stats.buffer_bytes,
        };
        Ok(Self {
            execution,
            spec,
            context: m.max_context,
            vocab: m.vocab,
            input: m.input.clone(),
            frontend_assets: prepared.frontend_assets,
            statistics,
        })
    }
    pub fn max_context(&self) -> usize {
        self.context
    }
    pub fn spec(&self) -> &Spec {
        &self.spec
    }
    pub fn frontend_assets(&self) -> &BTreeMap<String, String> {
        &self.frontend_assets
    }
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }
    pub fn score_many(&mut self, inputs: &[&[u32]]) -> Result<Vec<Score>> {
        let Some(rows) = self.spec.rows.clone() else {
            return inputs.iter().map(|tokens| self.score(tokens)).collect();
        };
        if inputs.iter().any(|t| {
            t.is_empty() || t.len() > self.context || t.iter().any(|&id| id as usize >= self.vocab)
        }) {
            return Err("Invalid reranking tokens or context length".into());
        }
        let packets = rows.packets(
            &inputs.iter().map(|t| t.len()).collect::<Vec<_>>(),
            self.context,
        )?;
        let mut output = Vec::with_capacity(inputs.len());
        for packet in packets {
            let mut tokens = vec![0; packet.rows];
            for (owner, &[start, end]) in packet.segments.iter().enumerate() {
                tokens[start as usize..end as usize].copy_from_slice(inputs[packet.first + owner]);
            }
            let e = &self.execution;
            e.upload_ids(&self.input, &tokens)?;
            crate::encoder_rows::upload(e, &rows, &packet)?;
            e.submit_program_rows(&self.spec.program, packet.rows, ExecutionPhase::Prefill)?;
            for pair in e
                .download_bytes(&self.spec.output, packet.segments.len() * 8)?
                .as_chunks::<8>()
                .0
            {
                output.push(score_from_bytes(pair)?);
            }
        }
        Ok(output)
    }
    pub fn score(&mut self, tokens: &[u32]) -> Result<Score> {
        if self.spec.rows.is_some() {
            return Ok(self.score_many(&[tokens])?.remove(0));
        }
        if tokens.is_empty()
            || tokens.len() > self.context
            || tokens.iter().any(|&t| t as usize >= self.vocab)
        {
            return Err("Invalid reranking tokens or context length".into());
        }
        self.execution.upload_ids(&self.input, tokens)?;
        self.execution
            .upload_ids(&self.spec.length, &[tokens.len() as u32])?;
        self.execution.submit_program_rows(
            &self.spec.program,
            tokens.len(),
            ExecutionPhase::Prefill,
        )?;
        let bytes = self.execution.download_bytes(&self.spec.output, 8)?;
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
