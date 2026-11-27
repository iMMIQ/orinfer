//! Manifest-defined MTP programs and distribution-preserving speculation.
use crate::artifact::Result;
#[cfg(test)]
use crate::model::Manifest;
use serde::Serialize;

pub use orinfer_model_sdk::mtp::*;
pub(crate) mod adaptive;

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
pub(crate) const ACCEPT_STREAM: u64 = 0x4d54_5002;
pub(crate) const CORRECTION_STREAM: u64 = 0x4d54_5003;

/// Adaptive controllers observe discarded proposal confidence and whole-round
/// cost. Fresh counters prevent those observations from selecting reused noise.
pub(crate) fn round_step(round: usize, position: usize) -> Result<usize> {
    if round >= u32::MAX as usize || position > u32::MAX as usize {
        return Err("Speculative sampling counter overflow".into());
    }
    // High word zero belongs to ordinary decode, including fallback rounds.
    usize::try_from((((round as u64) + 1) << 32) | position as u64)
        .map_err(|_| "Speculative sampling requires a 64-bit counter".into())
}

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
    pub lookup_rounds: usize,
    pub lookup_proposed_tokens: usize,
    pub lookup_accepted_draft_tokens: usize,
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
