//! Typed service configuration; argument parsing belongs to the CLI.
use orinfer_engine::{execution::CudaGraphMode, scheduler};
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_dir: PathBuf,
    pub model: String,
    pub listen: String,
    pub gpu_lock: PathBuf,
    pub cuda_graph: CudaGraphMode,
    pub verify_weights: bool,
    pub load_workers: Option<usize>,
    pub mtp_drafts: Option<usize>,
    pub prefix_cache_bytes: usize,
    pub scheduler: scheduler::Options,
    pub limits: PreprocessingLimits,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            model: "qwen3.8-27b".into(),
            listen: "0.0.0.0:8088".into(),
            gpu_lock: "artifacts/gpu-experiment.lock".into(),
            cuda_graph: CudaGraphMode::default(),
            verify_weights: false,
            load_workers: None,
            mtp_drafts: None,
            prefix_cache_bytes: 12usize << 30,
            scheduler: Default::default(),
            limits: Default::default(),
        }
    }
}
impl ServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.model_dir.is_dir() {
            return Err("MODEL_DIR must be a prepared model directory".into());
        }
        if self.load_workers.is_some_and(|n| !(1..=32).contains(&n)) {
            return Err("Weight load workers must be within 1..32".into());
        }
        if self.model.is_empty() {
            return Err("Model ID cannot be empty".into());
        }
        if self.mtp_drafts.is_some_and(|n| n > 7) {
            return Err("MTP drafts must be auto or 0..7".into());
        }
        if !(1..=32).contains(&self.limits.preprocess_workers)
            || !(512..=16384).contains(&self.limits.memory_mib)
            || self.limits.output_ms == 0
            || self.limits.drain_ms == 0
        {
            return Err("Invalid preprocessing or timeout limits".into());
        }
        self.scheduler.validate()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PreprocessingLimits {
    pub preprocess_workers: usize,
    pub memory_mib: usize,
    pub queue_ms: u64,
    pub output_ms: u64,
    pub drain_ms: u64,
}
impl Default for PreprocessingLimits {
    fn default() -> Self {
        Self {
            preprocess_workers: 2,
            memory_mib: 2048,
            queue_ms: 0,
            output_ms: 60000,
            drain_ms: 30000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programmatic_configuration_obeys_the_same_limits() {
        let mut config = ServerConfig {
            model_dir: std::env::temp_dir(),
            ..Default::default()
        };
        config.validate().unwrap();
        config.mtp_drafts = Some(8);
        assert!(config.validate().is_err());
        config.mtp_drafts = None;
        config.scheduler.max_active = 129;
        assert!(config.validate().is_err());
        config.scheduler = Default::default();
        config.limits.output_ms = 0;
        assert!(config.validate().is_err());
    }
}
