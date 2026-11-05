//! Typed service configuration; argument parsing belongs to the CLI.
use orinfer_engine::{
    execution::{CudaGraphMode, MAX_MTP_DRAFTS, ResidentExperts},
    scheduler,
};
use std::path::PathBuf;

/// File traversal only; GPU work still uses the shared continuous scheduler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchOrder {
    #[default]
    Auto,
    Input,
}
impl std::str::FromStr for BatchOrder {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "input" => Ok(Self::Input),
            _ => Err("Batch order must be auto or input".into()),
        }
    }
}
impl std::fmt::Display for BatchOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Input => "input",
        })
    }
}

/// File-based Chat inference. Execution and protocol semantics match `serve`.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub model_dir: PathBuf,
    pub model: String,
    pub input: PathBuf,
    pub output: PathBuf,
    pub resume: bool,
    pub order: BatchOrder,
    pub load: orinfer_engine::execution::LoadOptions,
    pub scheduler: scheduler::Options,
    pub limits: PreprocessingLimits,
    pub gpu_lock: PathBuf,
    pub default_request_params: serde_json::Map<String, serde_json::Value>,
}
impl BatchConfig {
    pub(crate) fn settings(&self) -> ServerConfig {
        ServerConfig {
            model_dir: self.model_dir.clone(),
            model: self.model.clone(),
            default_request_params: self.default_request_params.clone(),
            gpu_lock: self.gpu_lock.clone(),
            cuda_graph: self.load.cuda_graph,
            verify_weights: self.load.verify_weights,
            load_workers: self.load.load_workers,
            resident_experts: self.load.resident_experts,
            mtp_drafts: self.load.mtp_drafts,
            prompt_lookup: self.load.prompt_lookup,
            prefix_cache_bytes: self.load.prefix_cache_bytes,
            kv_offload_bytes: self.load.kv_offload_bytes,
            scheduler: self.scheduler,
            limits: self.limits,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfig {
    pub model_dir: PathBuf,
    pub model: String,
    pub listen: String,
    /// Missing Chat request fields supplied by this deployment.
    pub default_request_params: serde_json::Map<String, serde_json::Value>,
    pub gpu_lock: PathBuf,
    pub cuda_graph: CudaGraphMode,
    pub verify_weights: bool,
    pub load_workers: Option<usize>,
    pub resident_experts: ResidentExperts,
    pub mtp_drafts: Option<usize>,
    pub prompt_lookup: bool,
    pub prefix_cache_bytes: usize,
    pub kv_offload_bytes: usize,
    pub scheduler: scheduler::Options,
    pub limits: PreprocessingLimits,
}
impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::new(),
            model: "qwen3.8-27b".into(),
            listen: "[::]:8088".into(),
            default_request_params: Default::default(),
            gpu_lock: "artifacts/gpu-experiment.lock".into(),
            cuda_graph: CudaGraphMode::default(),
            verify_weights: false,
            load_workers: None,
            resident_experts: ResidentExperts::default(),
            mtp_drafts: None,
            prompt_lookup: false,
            prefix_cache_bytes: 12usize << 30,
            kv_offload_bytes: 0,
            scheduler: Default::default(),
            limits: Default::default(),
        }
    }
}
impl ServerConfig {
    pub fn validate(&self) -> Result<(), String> {
        crate::server::defaults::validation_request(&self.default_request_params)?;
        if !self.model_dir.is_dir() {
            return Err("MODEL_DIR must be a prepared model directory".into());
        }
        if self.load_workers.is_some_and(|n| !(1..=32).contains(&n)) {
            return Err("Weight load workers must be within 1..32".into());
        }
        if self.model.is_empty() {
            return Err("Model ID cannot be empty".into());
        }
        if self.mtp_drafts.is_some_and(|n| n > MAX_MTP_DRAFTS) {
            return Err("MTP drafts must be auto or 0..31".into());
        }
        if self.resident_experts == ResidentExperts::Count(0) {
            return Err("Resident expert count must be positive".into());
        }
        if !(1..=32).contains(&self.limits.preprocess_workers)
            || !(512..=16384).contains(&self.limits.memory_mib)
            || self.limits.image_cache_mib > 16384
            || !(1..=self.limits.memory_mib / 8).contains(&self.limits.request_body_mib)
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
    /// Retained immutable image assets; separate from in-flight preparation.
    pub image_cache_mib: usize,
    pub request_body_mib: usize,
    pub queue_ms: u64,
    pub output_ms: u64,
    pub drain_ms: u64,
}
impl Default for PreprocessingLimits {
    fn default() -> Self {
        Self {
            preprocess_workers: 2,
            memory_mib: 2048,
            image_cache_mib: 256,
            request_body_mib: 32,
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
        config.resident_experts = ResidentExperts::Count(0);
        assert!(config.validate().is_err());
        config.resident_experts = ResidentExperts::Full;
        config.mtp_drafts = Some(16);
        config.validate().unwrap();
        config.mtp_drafts = Some(MAX_MTP_DRAFTS + 1);
        assert!(config.validate().is_err());
        config.mtp_drafts = None;
        config.scheduler.max_active = 129;
        assert!(config.validate().is_err());
        config.scheduler = Default::default();
        config.limits.output_ms = 0;
        assert!(config.validate().is_err());
    }
}
