//! Shared loading, statistics, and row-packet upload for independent encoders.
use crate::{artifact::Result, backend::Executor, execution::LoadOptions, model::Manifest};
use orinfer_model_sdk::encoder_rows::{Packet, Spec};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, time::Instant};

#[derive(Debug, Serialize)]
pub struct Statistics {
    pub load_to_ready_s: f64,
    pub weight_bytes: usize,
    pub weight_arena_padding_bytes: usize,
    pub resident_weight_bytes: usize,
    pub effective_weight_bits: f64,
    pub allocated_bytes: usize,
}

pub(crate) struct LoadedEncoder<S> {
    pub(crate) execution: Executor,
    pub(crate) spec: S,
    pub(crate) context: usize,
    pub(crate) vocab: usize,
    pub(crate) input: String,
    pub(crate) frontend_assets: BTreeMap<String, String>,
    pub(crate) statistics: Statistics,
}

impl<S> LoadedEncoder<S> {
    pub(crate) fn load(
        path: &Path,
        options: LoadOptions,
        extract: impl FnOnce(&Manifest) -> Option<S>,
        missing_error: &str,
        overflow_error: &str,
    ) -> Result<Self> {
        let started = Instant::now();
        let prepared = crate::loader::load_with_experts(path, options.verify_weights, false)?;
        let m = &prepared.plan;
        let spec = extract(m).ok_or(missing_error)?;
        let preload = m.kernels.iter().map(|k| k.name.clone()).collect();
        let (execution, stats) = Executor::load(&prepared, options, &preload)?;
        let padding = stats.load_stages.weight_arena_padding_bytes;
        let resident_weight_bytes = m.weight_bytes.checked_add(padding).ok_or(overflow_error)?;
        let statistics = Statistics {
            load_to_ready_s: started.elapsed().as_secs_f64(),
            weight_bytes: m.weight_bytes,
            weight_arena_padding_bytes: padding,
            resident_weight_bytes,
            effective_weight_bits: resident_weight_bytes as f64 * 8.0 / m.weight_parameters as f64,
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

    pub(crate) fn tokens_valid(&self, tokens: &[u32]) -> bool {
        !tokens.is_empty()
            && tokens.len() <= self.context
            && tokens.iter().all(|&t| (t as usize) < self.vocab)
    }
}

pub(crate) fn upload(e: &Executor, spec: &Spec, packet: &Packet) -> Result<()> {
    let encode = |rows: &[[u32; 2]]| -> Vec<u8> {
        rows.iter()
            .flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes()))
            .collect()
    };
    e.upload_bytes(&spec.ranges, &encode(&packet.ranges))?;
    e.upload_bytes(&spec.segments, &encode(&packet.segments))?;
    e.upload_ids(&spec.count, &[packet.segments.len() as u32])
}
