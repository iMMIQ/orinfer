//! Generation orchestration, independent of API transport and CUDA graph binding.
use crate::cuda::executor::{Executor, LoadStats};
use crate::execution::{ExecutionPhase, LoadOptions};
use crate::{
    artifact::Result,
    cuda::{Session, check, floats},
};
#[cfg(test)]
use std::collections::BTreeMap;
use std::time::Instant;
mod benchmark;
mod generation;
#[cfg(test)]
#[path = "../mtp_gpu_tests.rs"]
mod mtp_gpu_tests;
mod prefix;
mod speculation;
mod vision;

pub(crate) struct ModelRuntime {
    pub(crate) manifest: crate::model::Manifest,
    execution: Executor,
    stats: LoadStats,
    pub(crate) speculation_statistics: Option<crate::mtp::Statistics>,
    prefix_cache: crate::prefix::Cache<crate::cuda::snapshot::Snapshot>,
    pub(crate) prefix_statistics: crate::prefix::Statistics,
}
impl ModelRuntime {
    #[cfg(test)]
    pub(crate) fn load(path: &std::path::Path) -> Result<Self> {
        Self::load_with_options(path, LoadOptions::default())
    }
    pub(crate) fn load_with_options(path: &std::path::Path, options: LoadOptions) -> Result<Self> {
        let started = Instant::now();
        let prepared = crate::loader::load(path)?;
        let (execution, mut stats) = Executor::load(
            &prepared.plan,
            &prepared.weights_root,
            &prepared.kernel_root,
            prepared.fingerprint,
            &prepared.scopes,
            &prepared.decode_programs,
            options,
        )?;
        stats.load_to_ready_s = started.elapsed().as_secs_f64();
        Ok(Self {
            manifest: prepared.plan,
            execution,
            stats,
            speculation_statistics: None,
            prefix_cache: crate::prefix::Cache::new(options.prefix_cache_bytes),
            prefix_statistics: crate::prefix::Statistics::default(),
        })
    }
    fn launch_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        self.execution.launch_program(name, phase)
    }
    fn upload_ids(&self, name: &str, ids: &[u32]) -> Result<()> {
        self.execution.upload_ids(name, ids)
    }
    fn upload_bytes(&self, name: &str, data: &[u8]) -> Result<()> {
        self.execution.upload_bytes(name, data)
    }
    fn read_control(&self, name: &str) -> Result<i32> {
        self.execution.read_control(name)
    }
    fn read_controls(&self, name: &str, count: usize) -> Result<Vec<u32>> {
        self.execution.read_controls(name, count)
    }
}
pub(crate) fn run_model(
    manifest_path: &std::path::Path,
    requests_path: &std::path::Path,
    options: LoadOptions,
) -> Result<crate::model::Report> {
    let requests = crate::model::read(requests_path)?;
    ModelRuntime::load_with_options(manifest_path, options)?.benchmark(requests)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "Requires a real model, fixture and exclusive GPU experiment lock"]
    fn multimodal_token_probe() {
        #[derive(serde::Deserialize)]
        struct Probe {
            model: std::path::PathBuf,
            input_tokens: Vec<u32>,
            images: Vec<crate::vision::ImageInput>,
            prefixes: Vec<Vec<u32>>,
            target_tokens: Vec<u32>,
            output: std::path::PathBuf,
        }
        let path = std::env::var("ORIN_VISION_PROBE").expect("ORIN_VISION_PROBE");
        let probe: Probe = crate::model::read(std::path::Path::new(&path)).unwrap();
        let mut model = crate::runtime::ModelRuntime::load(&probe.model).unwrap();
        let options = crate::sampling::Options {
            temperature: 0.,
            ..Default::default()
        };
        let mut generated = vec![];
        model
            .generate(
                &probe.input_tokens,
                Some(&probe.images),
                128,
                &options,
                || false,
                |id| {
                    generated.push(id);
                    ![248046, 248044].contains(&id)
                },
            )
            .unwrap();
        let mut checks = vec![];
        for prefix in &probe.prefixes {
            let mut input = probe.input_tokens.clone();
            input.extend(prefix);
            let mut selected = 0;
            model
                .generate(
                    &input,
                    Some(&probe.images),
                    1,
                    &options,
                    || false,
                    |id| {
                        selected = id;
                        true
                    },
                )
                .unwrap();
            let spec = model
                .manifest
                .buffers
                .iter()
                .find(|b| b.name == model.manifest.logits)
                .unwrap();
            let mut raw = vec![0u8; spec.bytes().unwrap()];
            // SAFETY: generate synchronized the producing stream; the host
            // allocation covers the complete validated logits buffer.
            unsafe {
                check(
                    (model.execution.session.driver.download)(
                        raw.as_mut_ptr().cast(),
                        model.execution.pointers[&model.manifest.logits],
                        raw.len(),
                    ),
                    "probe logits",
                )
                .unwrap();
            }
            let logits = floats(&raw, spec.dtype);
            assert!(logits.iter().all(|v| v.is_finite()));
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z = f64::from(max)
                + logits
                    .iter()
                    .map(|&v| f64::from(v - max).exp())
                    .sum::<f64>()
                    .ln();
            let entry = |id: usize| {
                serde_json::json!({
                    "token_id":id,"logit":logits[id],"logprob":f64::from(logits[id])-log_z
                })
            };
            let mut ids: Vec<usize> = (0..logits.len()).collect();
            ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            checks.push(serde_json::json!({"prefix":prefix,"selected":selected,
                "top3":ids[..3].iter().map(|&id|entry(id)).collect::<Vec<_>>(),
                "targets":probe.target_tokens.iter().map(|&id|entry(id as usize)).collect::<Vec<_>>()
            }));
        }
        let result = serde_json::json!({"generated":generated,"checks":checks,
            "seed":crate::sampling::EVALUATION_SEED});
        use std::io::Write;
        let mut output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&probe.output)
            .unwrap();
        output
            .write_all(serde_json::to_string_pretty(&result).unwrap().as_bytes())
            .unwrap();
        println!("{result}");
    }
}
