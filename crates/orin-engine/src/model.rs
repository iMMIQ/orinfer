//! AOT model programs: architecture-specific plans, parameterized Rust runtime.
use crate::artifact::{Access, Argument, Buffer, Kernel, Result};
use crate::execution::{CudaGraphMode, LoadOptions};
pub use crate::weights::TensorIdentity;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub target: String,
    pub model: String,
    pub chunk_tokens: usize,
    #[serde(default)]
    pub prefill_plans: Vec<PrefillPlan>,
    pub max_context: usize,
    #[serde(default)]
    pub kv_cache: Option<KvCache>,
    pub vocab: usize,
    pub toolchain: BTreeMap<String, String>,
    pub buffers: Vec<Buffer<TensorIdentity>>,
    #[serde(default)]
    pub kernels: Vec<Kernel>,
    #[serde(default)]
    pub programs: BTreeMap<String, Vec<Operation>>,
    pub reset_buffers: Vec<String>,
    pub input: String,
    pub token: String,
    pub status: String,
    pub logits: String,
    pub position: String,
    pub weight_bytes: usize,
    pub weight_parameters: usize,
    pub weight_scope: String,
    #[serde(default)]
    pub vision: Option<crate::vision::VisionSpec>,
    #[serde(default)]
    pub mtp: Option<crate::mtp::Spec>,
    #[serde(skip)]
    pub(crate) batch_profiles: Vec<usize>,
    #[serde(skip)]
    pub(crate) greedy_sampling: bool,
    #[serde(skip)]
    pub(crate) batch_layout: Option<crate::architecture::BatchLayout>,
}

/// Storage contract for stable, demand-mapped token-major KV state.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvCache {
    pub direct_prefill: bool,
    pub demand_mapping: bool,
    /// Bytes per token for each payload or scale buffer (including metadata).
    pub buffers: BTreeMap<String, usize>,
    /// Shared per-layer prefill scratch, demand-mapped independently of KV state.
    #[serde(default)]
    pub prefill_workspace: BTreeMap<String, usize>,
    /// Derived by the architecture; never supplied by an operator package.
    #[serde(skip)]
    pub(crate) growth: BTreeMap<String, KvGrowth>,
}
#[derive(Clone, Debug)]
pub(crate) struct KvGrowth {
    pub position: String,
    pub tokens: usize,
    pub buffers: Vec<String>,
}

/// Fixed-shape graphs sharing one model's weights, workspace and private state.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillPlan {
    pub chunk_tokens: usize,
    pub prefill_program: String,
    pub head_program: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Kernel {
        name: String,
    },
    Copy {
        source: String,
        destination: String,
        bytes: usize,
    },
    Zero {
        destination: String,
        bytes: usize,
    },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Requests {
    pub requests: Vec<Request>,
    #[serde(default)]
    pub logits_output: Option<String>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub id: String,
    pub input_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    #[serde(default)]
    pub forced_tokens: Vec<u32>,
    #[serde(default)]
    pub logits_steps: Vec<usize>,
}

impl Manifest {
    pub fn validate<'a>(&'a self) -> Result<usize> {
        if self.schema_version != 2
            || self.target != "sm_87"
            || self.model.is_empty()
            || self.toolchain.is_empty()
        {
            return Err("Expected safetensors model schema 2 for SM87; prepare the AOT model directory first".into());
        }
        if self.chunk_tokens == 0
            || self.chunk_tokens > self.max_context
            || self.max_context > i32::MAX as usize
            || self.vocab == 0
            || self.vocab > i32::MAX as usize
        {
            return Err("Invalid model dimensions".into());
        }
        let mut identities = BTreeMap::<&str, &str>::new();
        let mut identity = |id: &'a crate::artifact::FileIdentity| -> Result<()> {
            if id.sha256.len() != 64
                || !id
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err("Malformed file hash".into());
            }
            if identities
                .insert(&id.file, &id.sha256)
                .is_some_and(|old| old != id.sha256)
            {
                return Err("Conflicting identities for one artifact file".into());
            }
            Ok(())
        };
        let mut buffers = BTreeMap::new();
        let mut tensors = BTreeMap::new();
        let mut total = 0usize;
        let mut weights = 0usize;
        for b in &self.buffers {
            if let Some(id) = &b.data {
                if id.tensor.is_empty()
                    || id.sha256.len() != 64
                    || !id
                        .sha256
                        .bytes()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                {
                    return Err("Invalid tensor identity".into());
                }
                if let Some(previous) =
                    tensors.insert(&id.tensor, (&id.sha256, b.dtype, &b.shape, &b.layout))
                    && previous != (&id.sha256, b.dtype, &b.shape, &b.layout)
                {
                    return Err("Conflicting tensor identities".into());
                }
            }
            let bytes = b.bytes()?;
            if b.access == Access::Read && b.data.is_none() {
                return Err(format!("{}: immutable tensor has no payload", b.name));
            }
            if b.name.is_empty()
                || b.layout.is_empty()
                || !b.alignment.is_power_of_two()
                || b.alignment > 256
                || buffers.insert(&b.name, b).is_some()
            {
                return Err("Invalid/duplicate model buffer".into());
            }
            total = total.checked_add(bytes).ok_or("Buffer total overflow")?;
            if b.access == Access::Read {
                weights = weights.checked_add(bytes).ok_or("Weight total overflow")?;
            }
        }
        if weights != self.weight_bytes || self.weight_parameters == 0 {
            return Err("Weight accounting mismatch".into());
        }
        let lookup = |name: &str| -> Result<&Buffer<TensorIdentity>> {
            buffers
                .get(&name.to_string())
                .copied()
                .ok_or_else(|| format!("Unknown buffer {name}"))
        };
        if let Some(kv) = &self.kv_cache {
            if kv.buffers.is_empty() {
                return Err("Empty KV allocation contract".into());
            }
            if kv
                .prefill_workspace
                .keys()
                .any(|n| kv.buffers.contains_key(n))
            {
                return Err("KV state and scratch allocations overlap".into());
            }
            for (name, stride) in kv.buffers.iter().chain(&kv.prefill_workspace) {
                let b = lookup(name)?;
                if *stride == 0
                    || stride.checked_mul(self.max_context) != Some(b.bytes()?)
                    || b.data.is_some()
                    || b.access == Access::Read
                {
                    return Err(format!(
                        "{name}: invalid token-major KV allocation contract"
                    ));
                }
            }
        }
        if let Some(vision) = &self.vision {
            vision.validate(self)?;
        }
        if let Some(mtp) = &self.mtp {
            mtp.validate(self)?;
        }
        for (name, minimum) in [
            (
                &self.input,
                self.chunk_tokens
                    .checked_mul(4)
                    .ok_or("Input size overflow")?,
            ),
            (&self.token, 4),
            (&self.status, 4),
            (&self.position, 4),
        ] {
            let b = lookup(name)?;
            if b.bytes()? < minimum
                || !matches!(b.dtype, crate::artifact::Dtype::I32)
                || b.access == Access::Read
            {
                return Err(format!("Invalid control buffer {name}"));
            }
        }
        let logits = lookup(&self.logits)?;
        if !matches!(
            logits.dtype,
            crate::artifact::Dtype::F16 | crate::artifact::Dtype::F32
        ) || logits.bytes()?
            != self
                .vocab
                .checked_mul(logits.dtype.bytes())
                .ok_or("Logits size overflow")?
        {
            return Err("Invalid full-vocabulary logits buffer".into());
        }
        let mut names = BTreeSet::new();
        for k in &self.kernels {
            identity(&k.module)?;
            identity(&k.source)?;
            identity(&k.host_abi)?;
            if k.name.is_empty()
                || !names.insert(&k.name)
                || k.symbol.is_empty()
                || k.symbol.contains('\0')
                || k.args.is_empty()
                || k.cooperative
                || k.grid.contains(&0)
                || k.block.contains(&0)
                || k.block
                    .iter()
                    .try_fold(1u32, |n, d| n.checked_mul(*d))
                    .is_none_or(|n| n > 1024)
            {
                return Err(format!("Invalid model kernel {}", k.name));
            }
            for a in &k.args {
                match a {
                    Argument::Buffer { name } => {
                        lookup(name)?;
                    }
                    Argument::F32 { value } if !value.is_finite() => {
                        return Err("Nonfinite scalar".into());
                    }
                    _ => (),
                }
            }
        }
        for required in ["prefill", "head", "decode"] {
            if self.programs.get(required).is_none_or(Vec::is_empty) {
                return Err(format!("Missing program {required}"));
            }
        }
        let mut plan_sizes = BTreeSet::new();
        for plan in &self.prefill_plans {
            if plan.chunk_tokens == 0
                || plan.chunk_tokens > self.chunk_tokens
                || !plan_sizes.insert(plan.chunk_tokens)
                || plan.prefill_program == "decode"
                || plan.head_program == "decode"
                || plan.prefill_program == plan.head_program
                || [&plan.prefill_program, &plan.head_program]
                    .iter()
                    .any(|name| self.programs.get(*name).is_none_or(Vec::is_empty))
            {
                return Err("Invalid/duplicate prefill plan or missing program".into());
            }
        }
        for ops in self.programs.values() {
            for op in ops {
                let writable = |name: &str, bytes: usize| -> Result<()> {
                    let b = lookup(name)?;
                    if bytes == 0 || bytes > b.bytes()? || b.access == Access::Read {
                        return Err(format!("Invalid destination range {name}"));
                    }
                    Ok(())
                };
                match op {
                    Operation::Kernel { name } if !names.contains(name) => {
                        return Err(format!("Unknown kernel {name}"));
                    }
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    } => {
                        writable(destination, *bytes)?;
                        if source == destination || *bytes > lookup(source)?.bytes()? {
                            return Err("Invalid copy source/alias".into());
                        }
                    }
                    Operation::Zero { destination, bytes } => writable(destination, *bytes)?,
                    _ => (),
                }
            }
        }
        if !self.reset_buffers.contains(&self.position) {
            return Err("Position must reset between requests".into());
        }
        for name in &self.reset_buffers {
            if lookup(name)?.access == Access::Read {
                return Err("Reset of immutable weight".into());
            }
        }
        Ok(total)
    }
    pub fn validate_requests(&self, r: &Requests) -> Result<()> {
        if r.requests.is_empty() {
            return Err("No requests".into());
        }
        let mut ids = BTreeSet::new();
        for q in &r.requests {
            if q.id.is_empty()
                || !ids.insert(&q.id)
                || q.id.contains('/')
                || q.id.contains('\\')
                || q.id == "."
                || q.id == ".."
            {
                return Err("Invalid/duplicate request identity".into());
            }
            if q.input_tokens.is_empty()
                || self.select_prefill_plan(q.input_tokens.len()).is_err()
                || q.max_new_tokens == 0
                || q.input_tokens
                    .len()
                    .checked_add(q.max_new_tokens - 1)
                    .is_none_or(|n| n > self.max_context)
            {
                return Err(format!(
                    "{}: input must fit a prefill plan, with generation fitting context",
                    q.id
                ));
            }
            if q.input_tokens
                .iter()
                .chain(&q.forced_tokens)
                .any(|id| *id as usize >= self.vocab)
                || q.forced_tokens.len() > q.max_new_tokens - 1
                || q.logits_steps.iter().any(|n| *n >= q.max_new_tokens)
            {
                return Err(format!("{}: invalid token/history/logits index", q.id));
            }
            if !q.logits_steps.is_empty() && r.logits_output.is_none() {
                return Err("Logits directory required".into());
            }
        }
        Ok(())
    }

    pub fn select_prefill_plan(&self, tokens: usize) -> Result<(usize, &str, &str)> {
        if tokens == 0 {
            return Err("Empty prefill".into());
        }
        if self.prefill_plans.is_empty() {
            if self.chunk_tokens == 0 || !tokens.is_multiple_of(self.chunk_tokens) {
                return Err("Input is not a multiple of the legacy chunk size".into());
            }
            return Ok((self.chunk_tokens, "prefill", "head"));
        }
        let plan = self
            .prefill_plans
            .iter()
            .filter(|plan| plan.chunk_tokens > 0 && tokens.is_multiple_of(plan.chunk_tokens))
            .max_by_key(|plan| plan.chunk_tokens)
            .ok_or("No compatible prefill plan")?;
        Ok((plan.chunk_tokens, &plan.prefill_program, &plan.head_program))
    }
}

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
pub struct Report {
    pub manifest_sha256: String,
    pub model: String,
    pub device: crate::cuda::DeviceInfo,
    pub load_to_ready_s: f64,
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
        "{}: {e}; prepare with tools/model/prepare.py, or split an existing prepared cache with tools/model/package.py",
        manifest.display()))
}

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub manifest_sha256: String,
    pub architecture: crate::architecture::Architecture,
    pub compute_policy: crate::architecture::ComputePolicy,
    pub operator_package: String,
    pub tensor_count: usize,
    pub shard_count: usize,
    pub weight_bytes: usize,
    pub validation_s: f64,
}

/// Verify all container layouts and payloads without CUDA or resident weight copies.
pub fn validate_model(path: &Path) -> Result<ValidationReport> {
    let start = Instant::now();
    let prepared = crate::loader::load(path)?;
    let manifest = &prepared.plan;
    let mut weights = crate::weights::Weights::open(&prepared.weights_root)?;
    let mut tensor_count = 0;
    for buffer in &manifest.buffers {
        if buffer.data.is_some() {
            weights.read(buffer)?;
            tensor_count += 1;
        }
    }
    let mut seen = BTreeSet::new();
    for kernel in &manifest.kernels {
        for identity in [&kernel.module, &kernel.source, &kernel.host_abi] {
            if seen.insert(&identity.file) {
                crate::artifact::read_identity(&prepared.kernel_root, identity)?;
            }
        }
    }
    Ok(ValidationReport {
        manifest_sha256: prepared.fingerprint,
        architecture: prepared.architecture,
        compute_policy: prepared.policy,
        operator_package: prepared.operator_package,
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
    pub manifest_sha256: String,
    pub operator_package: String,
    pub architecture: crate::architecture::Architecture,
    pub compute_policy: crate::architecture::ComputePolicy,
    pub allocation_bytes: BTreeMap<String, usize>,
    pub manifest: Manifest,
}
pub fn inspect_plan(path: &Path) -> Result<PlanReport> {
    let prepared = crate::loader::load(path)?;
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
        manifest_sha256: prepared.fingerprint,
        operator_package: prepared.operator_package,
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
pub struct Model(crate::runtime::ModelRuntime);
pub use crate::runtime::requests::{GenerationInput, RequestState, StepOutput};
impl Model {
    pub fn batching_supported(&self) -> bool {
        !self.0.manifest.batch_profiles.is_empty()
    }
    pub fn scheduler_statistics(&self) -> crate::scheduler::Statistics {
        self.0.scheduling_statistics()
    }
    pub fn estimated_request_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<crate::scheduler::Waiting> {
        self.0.waiting_cost(input, images)
    }
    pub fn can_admit(
        &mut self,
        input: &GenerationInput,
        options: &crate::scheduler::Options,
    ) -> Result<bool> {
        self.0.can_admit_request(input, options)
    }
    pub fn start_request(
        &mut self,
        input: GenerationInput,
        cancelled: impl Fn() -> bool,
    ) -> Result<RequestState> {
        self.0.start_request(input, &cancelled)
    }
    pub fn advance_requests(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &crate::scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
        self.0.advance_requests(requests, options)
    }
    pub fn finish_request(&mut self, request: &mut RequestState, cache: bool) -> Result<()> {
        self.0.finish_request(request, cache)
    }
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_with_options(path, LoadOptions::default())
    }
    pub fn load_with_options(path: &Path, options: LoadOptions) -> Result<Self> {
        crate::runtime::ModelRuntime::load_with_options(path, options).map(Self)
    }
    pub fn max_context(&self) -> usize {
        self.0.manifest.max_context
    }
    pub fn vocab(&self) -> usize {
        self.0.manifest.vocab
    }
    pub fn vision(&self) -> Option<&crate::vision::VisionSpec> {
        self.0.manifest.vision.as_ref()
    }
    pub fn speculation_statistics(&self) -> Option<&crate::mtp::Statistics> {
        self.0.speculation_statistics.as_ref()
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.0.prefix_statistics
    }
    /// CPU-only scheduling estimate; does not touch LRU or CUDA state.
    pub fn cached_prefix_tokens(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<usize> {
        self.0.prefix_match_tokens(input, images)
    }
    /// Exact token boundaries supplied by an API template. Consumed by the next request.
    pub fn set_prefix_cache_hints(&mut self, positions: Vec<usize>) {
        self.0.prefix_hints = positions;
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.0
            .generate(input, None, limit, options, cancelled, emit)
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
        self.0
            .generate(input, Some(images), limit, options, cancelled, emit)
    }
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
                chunk_tokens: 2,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
            },
            PrefillPlan {
                chunk_tokens: 8,
                prefill_program: "prefill".into(),
                head_program: "head".into(),
            },
        ];
        assert!(m.validate().is_ok());
        assert_eq!(m.select_prefill_plan(8).unwrap().0, 8);
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
