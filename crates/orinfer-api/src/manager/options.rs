use super::Result;
use crate::ServerConfig;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct WorkerOptions {
    pub cuda_graph: String,
    pub mtp_drafts: String,
    pub resident_experts: String,
    pub prefix_cache_mib: usize,
    pub kv_offload_mib: usize,
    pub max_active_requests: usize,
    pub max_batch_tokens: usize,
    pub max_speculative_tokens: usize,
    pub prefill_budget_ms: f64,
    pub target_tpot_ms: f64,
    pub memory_reserve_mib: usize,
    pub load_workers: Option<usize>,
    pub verify_weights: bool,
    pub prompt_lookup: bool,
    pub preprocess_workers: usize,
    pub preprocess_memory_mib: usize,
    pub default_request_params: Map<String, Value>,
}
impl Default for WorkerOptions {
    fn default() -> Self {
        Self {
            cuda_graph: "decode_only".into(),
            mtp_drafts: "auto".into(),
            resident_experts: "full".into(),
            prefix_cache_mib: 12288,
            kv_offload_mib: 0,
            max_active_requests: 32,
            max_batch_tokens: 128,
            max_speculative_tokens: 0,
            prefill_budget_ms: 200.,
            target_tpot_ms: 400.,
            memory_reserve_mib: 1024,
            load_workers: None,
            verify_weights: false,
            prompt_lookup: false,
            preprocess_workers: 2,
            preprocess_memory_mib: 2048,
            default_request_params: Map::new(),
        }
    }
}
impl WorkerOptions {
    pub fn from_settings(settings: &ServerConfig) -> Self {
        Self {
            cuda_graph: settings.cuda_graph.to_string(),
            mtp_drafts: settings
                .mtp_drafts
                .map_or_else(|| "auto".into(), |n| n.to_string()),
            resident_experts: settings.resident_experts.to_string(),
            prefix_cache_mib: settings.prefix_cache_bytes >> 20,
            kv_offload_mib: settings.kv_offload_bytes >> 20,
            max_active_requests: settings.scheduler.max_active,
            max_batch_tokens: settings.scheduler.max_batch_tokens,
            max_speculative_tokens: settings.scheduler.max_speculative_tokens,
            prefill_budget_ms: settings.scheduler.prefill_budget_ms,
            target_tpot_ms: settings.scheduler.target_tpot_ms,
            memory_reserve_mib: settings.scheduler.memory_reserve_bytes >> 20,
            load_workers: settings.load_workers,
            verify_weights: settings.verify_weights,
            prompt_lookup: settings.prompt_lookup,
            preprocess_workers: settings.limits.preprocess_workers,
            preprocess_memory_mib: settings.limits.memory_mib,
            default_request_params: settings.default_request_params.clone(),
        }
    }
    pub fn settings(&self, name: &str, path: &Path) -> Result<ServerConfig> {
        Ok(ServerConfig {
            model_dir: path.into(),
            model: name.into(),
            cuda_graph: self.cuda_graph.parse()?,
            mtp_drafts: orinfer_engine::execution::parse_mtp_drafts(&self.mtp_drafts)?,
            resident_experts: self.resident_experts.parse()?,
            prefix_cache_bytes: self
                .prefix_cache_mib
                .checked_mul(1 << 20)
                .ok_or("Prefix budget overflow")?,
            kv_offload_bytes: self
                .kv_offload_mib
                .checked_mul(1 << 20)
                .ok_or("CPU offload budget overflow")?,
            load_workers: self.load_workers,
            verify_weights: self.verify_weights,
            prompt_lookup: self.prompt_lookup,
            default_request_params: self.default_request_params.clone(),
            scheduler: orinfer_engine::scheduler::Options {
                max_active: self.max_active_requests,
                max_batch_tokens: self.max_batch_tokens,
                max_speculative_tokens: self.max_speculative_tokens,
                prefill_budget_ms: self.prefill_budget_ms,
                target_tpot_ms: self.target_tpot_ms,
                memory_reserve_bytes: self
                    .memory_reserve_mib
                    .checked_mul(1 << 20)
                    .ok_or("Memory reserve overflow")?,
                ..Default::default()
            },
            limits: crate::PreprocessingLimits {
                preprocess_workers: self.preprocess_workers,
                memory_mib: self.preprocess_memory_mib,
                ..Default::default()
            },
            ..Default::default()
        })
    }
    pub(super) fn arguments(&self) -> Vec<String> {
        let mut args = vec![
            "--cuda-graph".into(),
            self.cuda_graph.clone(),
            "--mtp-drafts".into(),
            self.mtp_drafts.clone(),
            "--resident-experts".into(),
            self.resident_experts.clone(),
            "--kv-offload-mib".into(),
            self.kv_offload_mib.to_string(),
            "--prefix-cache-mib".into(),
            self.prefix_cache_mib.to_string(),
            "--max-active-requests".into(),
            self.max_active_requests.to_string(),
            "--max-batch-tokens".into(),
            self.max_batch_tokens.to_string(),
            "--max-speculative-tokens".into(),
            self.max_speculative_tokens.to_string(),
            "--prefill-budget-ms".into(),
            self.prefill_budget_ms.to_string(),
            "--target-tpot-ms".into(),
            self.target_tpot_ms.to_string(),
            "--preprocess-workers".into(),
            self.preprocess_workers.to_string(),
            "--preprocess-memory-mib".into(),
            self.preprocess_memory_mib.to_string(),
            "--memory-reserve-mib".into(),
            self.memory_reserve_mib.to_string(),
            "--default-request-params".into(),
            serde_json::to_string(&self.default_request_params).unwrap(),
        ];
        if let Some(n) = self.load_workers {
            args.extend(["--load-workers".into(), n.to_string()]);
        }
        if self.verify_weights {
            args.push("--verify-weights".into());
        }
        if self.prompt_lookup {
            args.push("--prompt-lookup".into());
        }
        args
    }
}

#[derive(Clone, Debug)]
pub struct ManagerConfig {
    pub worker_options: WorkerOptions,
    pub data_dir: PathBuf,
    pub listen: String,
    pub keep_alive_s: u64,
    pub max_loaded_models: Option<usize>,
    pub memory_reserve_mib: usize,
    pub load_timeout_s: u64,
    pub gpu_lock: PathBuf,
}
impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            worker_options: WorkerOptions::default(),
            data_dir: default_data_dir(),
            listen: "[::]:8088".into(),
            keep_alive_s: 300,
            max_loaded_models: None,
            memory_reserve_mib: 2048,
            load_timeout_s: 120,
            gpu_lock: "artifacts/gpu-experiment.lock".into(),
        }
    }
}
pub fn default_data_dir() -> PathBuf {
    std::env::var_os("ORINFER_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
                        .join(".local/share")
                })
                .join("orinfer")
        })
}
pub fn runtime_dir(data_dir: &Path) -> PathBuf {
    let hash = orinfer_engine::artifact::sha256(data_dir.as_os_str().as_encoded_bytes());
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // SAFETY: getuid has no arguments or memory side effects.
            PathBuf::from(format!("/tmp/orinfer-{}", unsafe { libc::getuid() }))
        });
    base.join(format!("orinfer-{}", &hash[..12]))
}
