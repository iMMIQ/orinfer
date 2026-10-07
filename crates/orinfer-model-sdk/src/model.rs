use crate::artifact::{Access, Argument, Buffer, Kernel, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TensorIdentity {
    pub tensor: String,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Hash-pinned CPU input assets interpreted by the model implementation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_assets: Option<crate::artifact::FileIdentity>,
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
    #[serde(default)]
    pub batch_profiles: Vec<usize>,
    #[serde(default)]
    pub dynamic_batch_kernels: BTreeMap<String, crate::operators::dynamic::DynamicBatchKernel>,
    #[serde(default)]
    pub prefill_batch_profiles: Vec<crate::architecture::PrefillProfile>,
    #[serde(default)]
    pub greedy_sampling: bool,
    #[serde(default)]
    pub batch_gdn: bool,
    #[serde(default)]
    pub state_pointer_table: Option<StatePointerTable>,
    #[serde(default)]
    pub segment_controls: Option<SegmentControls>,
    #[serde(default)]
    pub batch_layout: Option<crate::architecture::BatchLayout>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SegmentControls {
    pub length: String,
    pub last_index: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StatePointerTable {
    pub buffer: String,
    pub max_rows: usize,
}

/// Storage contract for stable, demand-mapped token-major KV state.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvCache {
    #[serde(default)]
    pub prefix_divisors: BTreeMap<String, usize>,
    pub direct_prefill: bool,
    pub demand_mapping: bool,
    /// Bytes per token for each payload or scale buffer (including metadata).
    pub buffers: BTreeMap<String, usize>,
    /// Shared per-layer prefill scratch, demand-mapped independently of KV state.
    #[serde(default)]
    pub prefill_workspace: BTreeMap<String, usize>,
    /// Derived by the architecture; never supplied by an operator package.
    #[serde(default)]
    pub growth: BTreeMap<String, KvGrowth>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct KvGrowth {
    pub position: String,
    pub tokens: usize,
    pub buffers: Vec<String>,
}

/// Fixed-shape graphs sharing one model's weights, workspace and private state.
#[derive(Clone, Debug, Deserialize, Serialize)]
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
        if let Some(asset) = &self.input_assets {
            identity(asset)?;
        }
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
            for (name, divisor) in &kv.prefix_divisors {
                if *divisor == 0
                    || !kv.buffers.contains_key(name)
                    || !self.max_context.is_multiple_of(*divisor)
                {
                    return Err("Invalid compressed prefix row divisor".into());
                }
            }
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
                    Argument::BufferSlice { name, offset } => {
                        let b = lookup(name)?;
                        if *offset >= b.bytes()? || !offset.is_multiple_of(b.dtype.bytes()) {
                            return Err("Invalid kernel buffer slice".into());
                        }
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
