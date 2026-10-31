//! AOT model programs: architecture-specific plans, parameterized Rust runtime.
use crate::artifact::Result;
use crate::execution::{CudaGraphMode, LoadOptions};
pub use crate::weights::TensorIdentity;
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

pub use orinfer_model_sdk::model::{KvCache, KvGrowth, Manifest, Operation, PrefillPlan};

pub use orinfer_model_sdk::model::{Request, Requests};

#[derive(Debug, Serialize)]
pub struct RequestReport {
    pub id: String,
    pub input_tokens: usize,
    pub prefill_chunk_tokens: usize,
    pub prefill_program: String,
    pub output_tokens: Vec<u32>,
    pub reset_s: f64,
    pub prefill_s: f64,
    pub head_s: f64,
    pub ttft_s: f64,
    pub decode_s: f64,
    pub prefill_tps: f64,
    pub decode_tps: Option<f64>,
    pub final_position: usize,
    pub logits_files: Vec<String>,
    pub diagnostic: bool,
}
#[derive(Debug, Serialize)]
pub struct LoadStages {
    pub cuda_init_s: f64,
    pub buffer_alloc_s: f64,
    pub kernel_assets_s: f64,
    pub weight_read_wait_s: f64,
    pub weight_dma_wait_s: f64,
    pub weight_upload_chunks: usize,
    pub final_sync_s: f64,
    pub cuda_allocations: usize,
    pub weight_arena_padding_bytes: usize,
    pub deferred_workspace_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub resident_experts: crate::execution::ResidentExperts,
    pub expert_cache: Option<crate::execution::ExpertCacheStatistics>,
    pub manifest_sha256: String,
    pub model: String,
    pub device: crate::backend::DeviceInfo,
    pub load_to_ready_s: f64,
    pub load_stages: LoadStages,
    pub verify_weights: bool,
    pub load_workers: usize,
    pub preloaded_modules: usize,
    pub registered_modules: usize,
    pub weight_io_hash_s: f64,
    pub weight_upload_s: f64,
    pub module_load_bind_s: f64,
    pub graph_capture_s: f64,
    pub cuda_graph: CudaGraphMode,
    pub captured_programs: Vec<String>,
    /// Resident fixed buffers at load, excluding demand-mapped KV.
    pub buffer_bytes: usize,
    pub buffer_capacity_bytes: usize,
    pub peak_kv_bytes: usize,
    pub peak_prefill_workspace_bytes: usize,
    pub weight_bytes: usize,
    pub effective_weight_bits: f64,
    pub weight_scope: String,
    pub requests: Vec<RequestReport>,
    pub seed: u64,
    pub timing_scope: &'static str,
}

pub fn run(manifest: &Path, requests: &Path) -> Result<Report> {
    run_with_options(manifest, requests, LoadOptions::default())
}

pub fn run_with_options(manifest: &Path, requests: &Path, options: LoadOptions) -> Result<Report> {
    crate::runtime::run_model(manifest, requests, options)
}

/// A prepared HF-style directory contains the private execution cache.
pub fn resolve_manifest(path: &Path) -> Result<PathBuf> {
    let manifest = if path.is_dir() {
        path.join("cache/model.json")
    } else {
        path.to_owned()
    };
    manifest.canonicalize().map_err(|e| format!(
        "{}: {e}; build a native model with tools/model/qwen35/build.py or tools/model/flash_next/build.py",
        manifest.display()))
}

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub manifest_sha256: String,
    pub architecture: crate::architecture::Architecture,
    pub compute_policy: crate::architecture::ComputePolicy,
    pub execution_package: String,
    pub tensor_count: usize,
    pub shard_count: usize,
    pub weight_bytes: usize,
    pub validation_s: f64,
}

/// Read the native execution library's own version and registered capabilities.
pub fn inspect_execution_library(path: &Path) -> Result<orinfer_model_sdk::abi::PackageInfo> {
    crate::model_package::inspect_library(path)
}

/// Verify all container layouts and payloads without CUDA or resident weight copies.
pub fn validate_model(path: &Path) -> Result<ValidationReport> {
    let descriptor = resolve_manifest(path)?;
    let raw = read(&descriptor)?;
    if crate::backend::ascend::model::provider::Runtime::recognizes(&raw) {
        return crate::backend::ascend::model::validation::validate(&descriptor);
    }
    let start = Instant::now();
    let prepared = crate::loader::load(path, true)?;
    let manifest = &prepared.plan;
    let mut weights = crate::weights::Weights::open(&prepared.weights_root, true)?;
    let mut tensor_count = 0;
    for buffer in &manifest.buffers {
        if buffer.data.is_some() {
            weights.read(buffer)?;
            tensor_count += 1;
        }
    }
    let mut seen = BTreeSet::new();
    let mut reader = crate::artifact::AssetReader::default();
    for kernel in &manifest.kernels {
        for identity in [&kernel.module, &kernel.source, &kernel.host_abi] {
            if seen.insert(identity.key()) {
                reader.read(&prepared.kernel_root, identity)?;
            }
        }
    }
    Ok(ValidationReport {
        manifest_sha256: prepared.fingerprint,
        architecture: prepared.architecture,
        compute_policy: prepared.policy,
        execution_package: prepared.execution_package,
        tensor_count,
        shard_count: weights.shard_count(),
        weight_bytes: manifest.weight_bytes,
        validation_s: start.elapsed().as_secs_f64(),
    })
}

/// CPU-only inspection of the registered plan, without reading tensor payloads.
/// Profiling and package validation use the same loader as real inference.
#[derive(Debug, Serialize)]
pub struct PlanReport {
    pub plan_kernel_contracts: std::collections::BTreeSet<String>,
    pub manifest_sha256: String,
    pub execution_package: String,
    pub architecture: crate::architecture::Architecture,
    pub compute_policy: crate::architecture::ComputePolicy,
    pub allocation_bytes: BTreeMap<String, usize>,
    pub manifest: Manifest,
}
pub fn inspect_plan(path: &Path) -> Result<PlanReport> {
    let prepared = crate::loader::load(path, false)?;
    let mut allocation_bytes = BTreeMap::new();
    for buffer in &prepared.plan.buffers {
        let key = match prepared.scopes[&buffer.name] {
            crate::loader::BufferScope::Weights => "weights",
            crate::loader::BufferScope::Sequence => "sequence",
            crate::loader::BufferScope::Workspace => "workspace",
        };
        *allocation_bytes.entry(key.into()).or_insert(0) += buffer.bytes()?;
    }
    Ok(PlanReport {
        plan_kernel_contracts: prepared.execution_model.plan_contracts.borrow().clone(),
        manifest_sha256: prepared.fingerprint,
        execution_package: prepared.execution_package,
        architecture: prepared.architecture,
        compute_policy: prepared.policy,
        allocation_bytes,
        manifest: prepared.plan,
    })
}
pub(crate) fn read<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))
}

/// Thread-affine resident model. All CUDA resources remain on the creating thread.
enum Implementation {
    Cuda(Box<crate::runtime::ModelRuntime>),
    Ascend(Box<crate::backend::ascend::model::provider::Runtime>),
}
pub struct Model(Implementation);
mod request;
pub use crate::runtime::requests::{GenerationInput, StepOutput};
pub use crate::runtime::scoring::{Probe, ScoreCase, ScoreReport, ScoreRequests, TokenProbability};
pub use request::RequestState;

pub fn score_with_options(
    path: &Path,
    requests: &Path,
    options: LoadOptions,
) -> Result<ScoreReport> {
    let input = read(requests)?;
    crate::runtime::ModelRuntime::load_with_options(path, options)?.score(input)
}
impl Model {
    pub fn expert_cache_statistics(&self) -> Option<crate::execution::ExpertCacheStatistics> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.expert_cache_statistics(),
            Implementation::Ascend(inner) => inner.expert_cache_statistics(),
        }
    }
    pub fn frontend_assets(&self) -> &std::collections::BTreeMap<String, String> {
        match &self.0 {
            Implementation::Cuda(inner) => &inner.frontend_assets,
            Implementation::Ascend(inner) => inner.frontend_assets(),
        }
    }
    pub fn batching_supported(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => !inner.manifest.batch_profiles.is_empty(),
            Implementation::Ascend(inner) => inner.batching_supported(),
        }
    }
    pub fn prefill_packet_capacity(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner
                .manifest
                .batch_layout
                .as_ref()
                .map_or(1, |layout| layout.segment_tokens.max(1)),
            Implementation::Ascend(inner) => inner.prefill_packet_capacity(),
        }
    }
    pub fn scheduler_statistics(&self) -> crate::scheduler::Statistics {
        match &self.0 {
            Implementation::Cuda(inner) => inner.scheduling_statistics(),
            Implementation::Ascend(inner) => inner.scheduler_statistics(),
        }
    }
    pub fn estimated_request_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<crate::scheduler::Waiting> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.waiting_cost(input, images),
            Implementation::Ascend(inner) => inner.estimated_request_cost(input, images),
        }
    }
    pub fn request_hint(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<crate::scheduler::RequestHint> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.request_hint(input, images, audios),
            Implementation::Ascend(inner) => inner.request_hint(input, images, audios),
        }
    }
    pub fn estimated_hint_cost(
        &self,
        hint: &crate::scheduler::RequestHint,
    ) -> Result<crate::scheduler::Waiting> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.hint_cost(hint),
            Implementation::Ascend(inner) => inner.estimated_hint_cost(hint),
        }
    }
    pub fn share_prefill_checkpoint(
        &mut self,
        request: &mut RequestState,
        hint: &crate::scheduler::RequestHint,
    ) -> Result<bool> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                let request = request.cuda_mut()?;
                inner.share_prefill_checkpoint(request, hint)
            }
            Implementation::Ascend(inner) => {
                let request = request.ascend_mut()?;
                inner.share_prefill_checkpoint(request, hint)
            }
        }
    }
    pub fn prefix_cache_enabled(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => inner.prefix_cache_limit > 0,
            Implementation::Ascend(inner) => inner.prefix_cache_enabled(),
        }
    }
    pub fn decode_admission_allowed(
        &self,
        active: &[&RequestState],
        prompt_tokens: usize,
        options: &crate::scheduler::Options,
    ) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => {
                let active = active
                    .iter()
                    .filter_map(|request| request.cuda_ref())
                    .collect::<Vec<_>>();
                inner.decode_admission_allowed(&active, prompt_tokens, options)
            }
            Implementation::Ascend(inner) => {
                let active = active
                    .iter()
                    .filter_map(|request| request.ascend_ref())
                    .collect::<Vec<_>>();
                inner.decode_admission_allowed(&active, prompt_tokens, options)
            }
        }
    }
    pub fn can_admit(
        &mut self,
        input: &GenerationInput,
        options: &crate::scheduler::Options,
    ) -> Result<bool> {
        match &mut self.0 {
            Implementation::Cuda(inner) => inner.can_admit_request(input, options),
            Implementation::Ascend(inner) => inner.can_admit(input, options),
        }
    }
    pub fn start_request(
        &mut self,
        input: GenerationInput,
        cancelled: impl Fn() -> bool,
    ) -> Result<RequestState> {
        match &mut self.0 {
            Implementation::Cuda(inner) => inner
                .start_request(input, &cancelled)
                .map(RequestState::cuda),
            Implementation::Ascend(inner) => inner
                .start_request(input, cancelled)
                .map(RequestState::ascend),
        }
    }
    pub fn advance_requests(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &crate::scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                let mut requests = requests
                    .iter_mut()
                    .map(|request| request.cuda_mut())
                    .collect::<Result<Vec<_>>>()?;
                inner.advance_requests(&mut requests, options)
            }
            Implementation::Ascend(inner) => {
                let mut requests = requests
                    .iter_mut()
                    .map(|request| request.ascend_mut())
                    .collect::<Result<Vec<_>>>()?;
                inner.advance_requests(&mut requests, options)
            }
        }
    }
    pub fn advance_requests_with(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &crate::scheduler::Options,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                let mut requests = requests
                    .iter_mut()
                    .map(|request| request.cuda_mut())
                    .collect::<Result<Vec<_>>>()?;
                inner.advance_requests_with(&mut requests, options, maintenance)
            }
            Implementation::Ascend(inner) => {
                let mut requests = requests
                    .iter_mut()
                    .map(|request| request.ascend_mut())
                    .collect::<Result<Vec<_>>>()?;
                inner.advance_requests_with(&mut requests, options, maintenance)
            }
        }
    }
    pub fn finish_request(&mut self, request: &mut RequestState, cache: bool) -> Result<()> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                let request = request.cuda_mut()?;
                inner.finish_request(request, cache)
            }
            Implementation::Ascend(inner) => {
                let request = request.ascend_mut()?;
                inner.finish_request(request, cache)
            }
        }
    }
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_options(path, LoadOptions::default())
    }
    pub fn load_with_options(path: &Path, options: LoadOptions) -> Result<Self> {
        let descriptor = resolve_manifest(path)?;
        let raw: serde_json::Value = read(&descriptor)?;
        if crate::backend::ascend::model::provider::Runtime::recognizes(&raw) {
            crate::backend::ascend::model::provider::Runtime::load(path, options)
                .map(|inner| Self(Implementation::Ascend(Box::new(inner))))
        } else {
            crate::runtime::ModelRuntime::load_with_options(path, options)
                .map(|inner| Self(Implementation::Cuda(Box::new(inner))))
        }
    }
    pub fn max_context(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner.manifest.max_context,
            Implementation::Ascend(inner) => inner.max_context(),
        }
    }
    pub fn mtp_drafts(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner
                .manifest
                .mtp
                .as_ref()
                .map_or(0, |s| s.default_verification_tokens - 1),
            Implementation::Ascend(inner) => inner.mtp_drafts(),
        }
    }
    pub fn vocab(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner.manifest.vocab,
            Implementation::Ascend(inner) => inner.vocab(),
        }
    }
    pub fn vision(&self) -> Option<&crate::vision::VisionSpec> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.manifest.vision.as_ref(),
            Implementation::Ascend(inner) => inner.vision(),
        }
    }
    pub fn speculation_statistics(&self) -> Option<&crate::mtp::Statistics> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.speculation_statistics.as_ref(),
            Implementation::Ascend(inner) => inner.speculation_statistics(),
        }
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        match &self.0 {
            Implementation::Cuda(inner) => &inner.prefix_statistics,
            Implementation::Ascend(inner) => inner.prefix_statistics(),
        }
    }
    pub fn cached_prefix_tokens(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<usize> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.prefix_match_tokens(input, images),
            Implementation::Ascend(inner) => inner.cached_prefix_tokens(input, images),
        }
    }
    pub fn set_prefix_cache_hints(&mut self, positions: Vec<usize>) {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                inner.prefix_hints = positions;
            }
            Implementation::Ascend(inner) => inner.set_prefix_cache_hints(positions),
        }
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                inner.generate(input, None, limit, options, cancelled, emit)
            }
            Implementation::Ascend(inner) => inner.generate(input, limit, options, cancelled, emit),
        }
    }
    pub fn generate_visual(
        &mut self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        match &mut self.0 {
            Implementation::Cuda(inner) => {
                inner.generate(input, Some(images), limit, options, cancelled, emit)
            }
            Implementation::Ascend(inner) => {
                inner.generate_visual(input, images, limit, options, cancelled, emit)
            }
        }
    }
}

/// Prepare model-specific positions without initializing CUDA.
pub fn media_layout(
    path: &Path,
    tokens: &[u32],
    images: &[crate::vision::ImageInput],
    capacity: usize,
) -> Result<(Vec<i32>, Vec<u32>)> {
    let prepared = crate::loader::load(path, false)?;
    let vision = prepared
        .plan
        .vision
        .as_ref()
        .ok_or("Model has no vision adapter")?;
    for image in images {
        vision.feature_count(image)?;
    }
    prepared
        .execution_model
        .visual_layout(tokens, images, capacity)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Manifest {
        let buffers = [
            ("Weight", "f16", 2, "read"), ("Input", "i32", 2, "read_write"),
            ("Token", "i32", 1, "read_write"), ("Status", "i32", 1, "read_write"),
            ("Step", "i32", 1, "read_write"), ("Logits", "f16", 4, "read_write"),
        ].map(|(name,dtype,rows,access)|serde_json::json!({"name":name,"dtype":dtype,"shape":[rows],"layout":"contiguous","alignment":256,"access":access,"data":if name=="Weight" { serde_json::json!({"tensor":"Weight","sha256":"0".repeat(64)}) } else { serde_json::Value::Null }}));
        let id = serde_json::json!({"file":"a.cubin","sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({"schema_version":2,"target":"sm_87","model":"test","chunk_tokens":2,"max_context":8,"vocab":4,"toolchain":{"test":"test"},
            "buffers":buffers,"kernels":[{"name":"k","module":id,"source":id,"host_abi":id,"symbol":"k","grid":[1,1,1],"block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[{"kind":"buffer","name":"Input"}]}],
            "programs":{"prefill":[{"kind":"kernel","name":"k"}],"head":[{"kind":"kernel","name":"k"}],"decode":[{"kind":"kernel","name":"k"}]},
            "reset_buffers":["Step"],"input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Step","weight_bytes":4,"weight_parameters":2,"weight_scope":"test"})).unwrap()
    }
    #[test]
    fn rejects_state_aliases_out_of_bounds_and_weight_writes() {
        let mut m = fixture();
        assert!(m.validate().is_ok());
        m.programs.get_mut("decode").unwrap().push(Operation::Copy {
            source: "Input".into(),
            destination: "Input".into(),
            bytes: 4,
        });
        assert!(m.validate().is_err());
        m.programs.get_mut("decode").unwrap().pop();
        m.programs.get_mut("decode").unwrap().push(Operation::Zero {
            destination: "Token".into(),
            bytes: 8,
        });
        assert!(m.validate().is_err());
        m.programs.get_mut("decode").unwrap().pop();
        m.programs.get_mut("decode").unwrap().push(Operation::Zero {
            destination: "Weight".into(),
            bytes: 4,
        });
        assert!(m.validate().is_err());
    }
    #[test]
    fn rejects_legacy_schema_and_raw_buffer_identity() {
        let mut m = fixture();
        m.schema_version = 1;
        assert!(m.validate().is_err());
        let raw = serde_json::json!({"file":"weight.bin","sha256":"0".repeat(64)});
        assert!(serde_json::from_value::<TensorIdentity>(raw).is_err());
        m.schema_version = 2;
        m.buffers[0].data = None;
        assert!(m.validate().is_err());
    }
    #[test]
    fn rejects_conflicting_cached_module_identity_and_unreset_position() {
        let mut m = fixture();
        m.kernels[0].source.sha256 = "1".repeat(64);
        assert!(m.validate().is_err());
        m.kernels[0].source.sha256 = "0".repeat(64);
        m.reset_buffers.clear();
        assert!(m.validate().is_err());
    }
    #[test]
    fn rejects_invalid_histories_and_context_capacity() {
        let m = fixture();
        let mut r = Requests {
            requests: vec![Request {
                id: "one".into(),
                input_tokens: vec![0, 1],
                max_new_tokens: 2,
                forced_tokens: vec![2],
                logits_steps: vec![],
            }],
            logits_output: None,
        };
        assert!(m.validate_requests(&r).is_ok());
        r.requests[0].forced_tokens[0] = 4;
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].forced_tokens[0] = 2;
        r.requests[0].max_new_tokens = 8;
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].max_new_tokens = 2;
        r.requests[0].input_tokens.push(0);
        assert!(m.validate_requests(&r).is_err());
        r.requests[0].input_tokens.pop();
        r.requests[0].logits_steps.push(2);
        assert!(m.validate_requests(&r).is_err());
    }

    #[test]
    fn selects_largest_compatible_plan_and_rejects_invalid_metadata() {
        let mut m = fixture();
        assert_eq!(m.select_prefill_plan(4).unwrap(), (2, "prefill", "head"));
        m.buffers
            .iter_mut()
            .find(|b| b.name == "Input")
            .unwrap()
            .shape = vec![8];
        m.chunk_tokens = 8;
        m.prefill_plans = vec![
            PrefillPlan {
                min_tokens: 0,
                chunk_tokens: 2,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
                stages: vec![],
                live_buffers: vec![],
            },
            PrefillPlan {
                min_tokens: 0,
                chunk_tokens: 8,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
                stages: vec![],
                live_buffers: vec![],
            },
        ];
        assert!(m.validate().is_ok());
        assert_eq!(m.select_prefill_plan(8).unwrap().0, 8);
        m.prefill_plans[0].stages = vec!["prefill".into()];
        assert!(m.validate().is_ok());
        m.prefill_plans[0].stages.push("prefill".into());
        assert!(m.validate().is_err());
        m.prefill_plans[0].stages.clear();
        m.prefill_plans[0].live_buffers.push("Input".into());
        assert!(m.validate().is_err());
        m.prefill_plans[0].live_buffers.clear();
        assert_eq!(m.select_prefill_plan(6).unwrap().0, 2);
        assert!(m.select_prefill_plan(3).is_err());
        assert!(m.select_prefill_plan(0).is_err());
        m.prefill_plans[1].chunk_tokens = 2;
        assert!(m.validate().is_err());
        m.prefill_plans[1].chunk_tokens = 8;
        m.prefill_plans[1].prefill_program = "missing".into();
        assert!(m.validate().is_err());
        m.prefill_plans[1].prefill_program = "decode".into();
        assert!(m.validate().is_err());
    }
}
