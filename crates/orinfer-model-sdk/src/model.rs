use crate::artifact::{Access, Argument, Buffer, Kernel, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Include canonical fallbacks and every selected replacement, even rare tails.
/// Build-only fusion inputs need not become executable host launch records.
pub fn execution_kernels(
    programs: &BTreeMap<String, Vec<Operation>>,
    rows: Option<&crate::operators::rows::RowExecution>,
) -> BTreeSet<String> {
    let mut names = programs
        .values()
        .flatten()
        .filter_map(|op| match op {
            Operation::Kernel { name } => Some(name.clone()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    if let Some(rows) = rows {
        names.extend(
            rows.variants
                .values()
                .flatten()
                .flat_map(|v| v.kernels.iter().cloned()),
        );
    }
    names
}
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
    /// Model-owned dynamic templates, independent of scheduler chunk choices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_execution: Option<crate::operators::rows::RowExecution>,
    /// Derived by the model adapter: fork, side start/end, join operation names.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub graph_branches: BTreeMap<String, Vec<[String; 4]>>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<crate::audio::AudioSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<crate::embedding::Spec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reranking: Option<crate::reranking::Spec>,
    #[serde(default)]
    pub mtp: Option<crate::mtp::Spec>,
    /// Shared row operand bindings, independent of speculative decoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_table: Option<std::sync::Arc<crate::packet::PacketTable>>,
    #[serde(default)]
    pub batch_profiles: Vec<usize>,
    #[serde(default)]
    pub dynamic_batch_kernels: BTreeMap<String, crate::operators::dynamic::DynamicBatchKernel>,
    #[serde(default)]
    pub prefill_batch_profiles: Vec<crate::architecture::PrefillProfile>,
    #[serde(default)]
    pub greedy_sampling: bool,
    /// Model-owned GDN construction policy; never inferred from scratch names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdn_execution: Option<crate::operators::gdn::Execution>,
    /// Derived by the native library from its shared sampling bindings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<Sampling>,
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
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub program: String,
    pub history: String,
    pub bias_ids: String,
    pub bias_values: String,
    pub mask: String,
    pub control: String,
    pub parameters: String,
    pub output: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub batch: Option<BatchedSampling>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BatchedSampling {
    pub program: String,
    pub capacity: usize,
    pub pointers: String,
    pub scratch: String,
    pub output: String,
    pub stride: usize,
    pub offsets: BTreeMap<String, usize>,
}

impl BatchedSampling {
    /// Private scratch layout shared by all six independent-row sampler bodies.
    /// Align each typed view; lane membership changes pointers, never arithmetic.
    pub fn layout(vocab: usize, context: usize) -> Result<(BTreeMap<String, usize>, usize)> {
        let blocks = vocab.div_ceil(1024);
        let shapes = BTreeMap::from([
            ("Counts", (vocab, 4)),
            ("Bias", (vocab, 8)),
            ("History", (context, 4)),
            ("BiasIds", (vocab, 4)),
            ("BiasValues", (vocab, 8)),
            ("Control", (4, 4)),
            ("Parameters", (6, 8)),
            ("Mask", (vocab.div_ceil(32), 4)),
            ("Scores", (vocab, 8)),
            ("Keys", (vocab, 8)),
            ("Maximum", (blocks, 8)),
            ("Invalid", (blocks, 4)),
            ("Mass", (vocab, 8)),
            ("State", (7, 8)),
            (
                "Hist",
                (
                    blocks
                        .checked_mul(256)
                        .ok_or("Sampling histogram overflow")?,
                    8,
                ),
            ),
        ]);
        let mut offsets = BTreeMap::new();
        let mut stride = 0usize;
        for (name, (elements, width)) in shapes {
            offsets.insert(name.into(), stride);
            let bytes = elements
                .checked_mul(width)
                .ok_or("Sampling scratch overflow")?;
            stride = stride
                .checked_add(
                    bytes
                        .checked_add(255)
                        .ok_or("Sampling alignment overflow")?
                        / 256
                        * 256,
                )
                .ok_or("Sampling stride overflow")?;
        }
        Ok((offsets, stride))
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<String>,
    pub buffers: Vec<String>,
}

/// Fixed-shape graphs sharing one model's weights, workspace and private state.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillPlan {
    pub chunk_tokens: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub min_tokens: usize,
    pub prefill_program: String,
    pub head_program: String,
    /// Ordered, resumable pieces of the same forward pass. Empty means atomic.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<String>,
    /// Workspace values that remain live between stages; never model state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub live_buffers: Vec<String>,
}

pub(crate) fn is_zero(value: &usize) -> bool {
    *value == 0
}

impl PrefillPlan {
    /// Largest real segment accepted by this plan within the remaining input.
    pub fn chunk(&self, remaining: usize) -> Option<usize> {
        let tokens = remaining.min(self.chunk_tokens);
        (tokens > 0 && self.accepts(tokens)).then_some(tokens)
    }

    pub fn accepts(&self, tokens: usize) -> bool {
        tokens == self.chunk_tokens
            || (self.min_tokens > 0 && (self.min_tokens..=self.chunk_tokens).contains(&tokens))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
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

/// Conservatively prove fork/join independence at whole-buffer granularity.
pub fn validate_graph_branches(manifest: &Manifest, kernels: &[Kernel]) -> Result<()> {
    let catalog: BTreeMap<_, _> = kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    for (program, branches) in &manifest.graph_branches {
        let ops = manifest
            .programs
            .get(program)
            .ok_or("Missing branch program")?;
        let positions: BTreeMap<_, _> = ops
            .iter()
            .enumerate()
            .filter_map(|(i, op)| {
                if let Operation::Kernel { name } = op {
                    Some((name.as_str(), i))
                } else {
                    None
                }
            })
            .collect();
        if positions.len()
            != ops
                .iter()
                .filter(|op| matches!(op, Operation::Kernel { .. }))
                .count()
        {
            return Err("Branch program has duplicate kernel names".into());
        }
        let io = |range: std::ops::Range<usize>| -> Result<(BTreeSet<&str>, BTreeSet<&str>)> {
            let (mut reads, mut writes) = (BTreeSet::new(), BTreeSet::new());
            for op in &ops[range] {
                match op {
                    Operation::Copy {
                        source,
                        destination,
                        ..
                    } => {
                        reads.insert(source.as_str());
                        writes.insert(destination.as_str());
                    }
                    Operation::Zero { destination, .. } => {
                        writes.insert(destination.as_str());
                    }
                    Operation::Kernel { name } => {
                        // Certify every compiled substitution, including fused
                        // variants whose writes differ from the canonical ABI.
                        let alternatives = manifest
                            .row_execution
                            .as_ref()
                            .and_then(|rows| rows.variants.get(name))
                            .into_iter()
                            .flatten()
                            .flat_map(|v| v.kernels.iter());
                        for identity in std::iter::once(name).chain(alternatives) {
                            let k = catalog
                                .get(identity.as_str())
                                .ok_or("Missing branch kernel contract")?;
                            if k.readonly_arguments.len() != k.args.len() {
                                return Err("Missing branch argument access contract".into());
                            }
                            for (arg, readonly) in k.args.iter().zip(&k.readonly_arguments) {
                                if let Argument::Buffer { name }
                                | Argument::BufferSlice { name, .. } = arg
                                {
                                    reads.insert(name.as_str());
                                    if !readonly {
                                        writes.insert(name.as_str());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok((reads, writes))
        };
        let mut regions = Vec::new();
        for branch in branches {
            let mut indices = [0; 4];
            for (i, name) in branch.iter().enumerate() {
                indices[i] = *positions
                    .get(name.as_str())
                    .ok_or("Missing fork/join endpoint")?;
            }
            if branch.iter().any(|name| {
                manifest
                    .row_execution
                    .as_ref()
                    .and_then(|rows| rows.variants.get(name))
                    .is_some_and(|variants| variants.iter().any(|v| v.kernels.len() != 1))
            }) {
                return Err("Parallel endpoints require one kernel per row choice".into());
            }
            let [fork, start, end, join] = indices;
            if !(fork < start && start <= end && end + 1 == join) {
                return Err("Invalid fork/join ordering".into());
            }
            let (main_reads, main_writes) = io(fork..start)?;
            let (side_reads, side_writes) = io(start..join)?;
            if !main_writes.is_disjoint(&side_reads)
                || !main_writes.is_disjoint(&side_writes)
                || !side_writes.is_disjoint(&main_reads)
            {
                return Err(format!("Dependent graph branches at {}", branch[0]));
            }
            regions.push(indices);
        }
        regions.sort();
        if regions.windows(2).any(|p| p[0][3] >= p[1][0]) {
            return Err("Overlapping fork/join regions".into());
        }
    }
    Ok(())
}

impl Manifest {
    pub fn packet_table(&self) -> Option<&std::sync::Arc<crate::packet::PacketTable>> {
        self.packet_table
            .as_ref()
            .or_else(|| self.mtp.as_ref().and_then(|s| s.packet_table.as_ref()))
    }

    pub fn independent(&self) -> bool {
        self.embedding.is_some() || self.reranking.is_some()
    }

    pub fn prefill_plan(&self, tokens: usize) -> Option<&PrefillPlan> {
        self.prefill_plans
            .iter()
            .filter(|p| p.accepts(tokens))
            .min_by_key(|p| p.chunk_tokens)
    }
    pub fn validate<'a>(&'a self) -> Result<usize> {
        if let Some(table) = self.packet_table() {
            table.validate(self)?;
        }
        if self.packet_table.is_some()
            && self.mtp.as_ref().is_some_and(|s| s.packet_table.is_some())
        {
            return Err("Duplicate packet operand tables".into());
        }

        let target = orinfer_device::Target::from_package(&self.target)?;
        if self.schema_version != 2 || self.model.is_empty() || self.toolchain.is_empty() {
            return Err(
                "Expected safetensors model schema 2; prepare the AOT model directory first".into(),
            );
        }
        if self.chunk_tokens == 0
            || self.chunk_tokens > self.max_context
            || self.max_context > i32::MAX as usize
            || self.vocab == 0
            || self.vocab > i32::MAX as usize
        {
            return Err("Invalid model dimensions".into());
        }
        let mut identities = BTreeMap::<crate::artifact::AssetKey, &str>::new();
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
                .insert(id.key(), &id.sha256)
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
        if let Some(sampling) = &self.sampling {
            use crate::artifact::Dtype;
            if self
                .programs
                .get(&sampling.program)
                .is_none_or(Vec::is_empty)
            {
                return Err("Missing GPU sampling program".into());
            }
            for (name, dtype, elements) in [
                (&sampling.history, Dtype::I32, self.max_context),
                (&sampling.bias_ids, Dtype::I32, self.vocab),
                (&sampling.bias_values, Dtype::F64, self.vocab),
                (&sampling.mask, Dtype::U32, self.vocab.div_ceil(32)),
                (&sampling.control, Dtype::I32, 4),
                (&sampling.parameters, Dtype::F64, 6),
                (&sampling.output, Dtype::I32, 2),
            ] {
                let b = lookup(name)?;
                if b.dtype != dtype || b.shape != [elements] || b.access != Access::ReadWrite {
                    return Err(format!("Invalid GPU sampling buffer {name}"));
                }
            }
            if let Some(batch) = &sampling.batch {
                let logits = lookup(&self.logits)?;
                if logits.dtype != Dtype::F32 || logits.shape != [self.vocab] {
                    return Err("Batched sampling requires FP32 vocabulary logits".into());
                }
                let (offsets, stride) = BatchedSampling::layout(self.vocab, self.max_context)?;
                if batch.capacity != 128
                    || batch.offsets != offsets
                    || batch.stride != stride
                    || self.row_execution.as_ref().is_none_or(|rows| {
                        !rows.programs.contains(&batch.program)
                            || rows.program_capacity(&batch.program) != batch.capacity
                    })
                    || self.programs.get(&batch.program).is_none_or(Vec::is_empty)
                {
                    return Err("Invalid batched sampling program/layout".into());
                }
                for (name, dtype, shape) in [
                    (&batch.pointers, Dtype::U64, [batch.capacity, 2]),
                    (&batch.scratch, Dtype::U8, [batch.capacity, stride]),
                    (&batch.output, Dtype::I32, [batch.capacity, 2]),
                ] {
                    let b = lookup(name)?;
                    if b.dtype != dtype
                        || b.shape != shape
                        || b.access != Access::ReadWrite
                        || b.data.is_some()
                    {
                        return Err(format!("Invalid batched sampling buffer {name}"));
                    }
                }
            }
        }
        if let Some(audio) = &self.audio {
            audio.validate(self)?;
        }
        if let Some(embedding) = &self.embedding {
            embedding.validate(self)?;
        }
        if let Some(reranking) = &self.reranking {
            reranking.validate(self)?;
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
        if !self.independent()
            && (!matches!(
                logits.dtype,
                crate::artifact::Dtype::F16 | crate::artifact::Dtype::F32
            ) || logits.bytes()?
                != self
                    .vocab
                    .checked_mul(logits.dtype.bytes())
                    .ok_or("Logits size overflow")?)
        {
            return Err("Invalid full-vocabulary logits buffer".into());
        }
        let mut names = BTreeSet::new();
        for k in &self.kernels {
            k.validate_launch(&target)?;
            if !k.readonly_arguments.is_empty() && k.readonly_arguments.len() != k.args.len() {
                return Err("Mismatched kernel argument access contract".into());
            }
            identity(&k.module)?;
            identity(&k.source)?;
            identity(&k.host_abi)?;
            if k.name.is_empty()
                || !names.insert(&k.name)
                || k.symbol.is_empty()
                || k.symbol.contains('\0')
                || k.args.is_empty()
                || k.cooperative
            {
                return Err(format!("Invalid model kernel {}", k.name));
            }
            for (index, a) in k.args.iter().enumerate() {
                match a {
                    Argument::Buffer { name } => {
                        if lookup(name)?.access == Access::Read
                            && k.readonly_arguments.get(index) == Some(&false)
                        {
                            return Err("Kernel declares a write to immutable weights".into());
                        }
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
        for required in if self.independent() {
            &[][..]
        } else {
            &["prefill", "head", "decode"][..]
        } {
            if self.programs.get(*required).is_none_or(Vec::is_empty) {
                return Err(format!("Missing program {required}"));
            }
        }
        if let Some(rows) = &self.row_execution {
            for program in &rows.programs {
                let allocation = self
                    .embedding
                    .as_ref()
                    .filter(|e| &e.vision_program == program)
                    .map(|e| e.max_patches)
                    .or_else(|| {
                        self.vision
                            .as_ref()
                            .and_then(|v| v.plans.iter().find(|p| &p.program == program))
                            .map(|p| p.patches)
                    })
                    .unwrap_or_else(|| {
                        self.audio
                            .as_ref()
                            .map_or(self.chunk_tokens, |a| match program.as_str() {
                                "audio_stem" => a.stem_chunks,
                                "audio_encoder" => a.max_features,
                                _ => self.chunk_tokens,
                            })
                    });
                if rows.program_capacity(program) > allocation {
                    return Err("Row execution exceeds the input allocation contract".into());
                }
            }
            rows.validate(
                &self.programs,
                &self.kernels.iter().map(|k| k.name.clone()).collect(),
                &self.dynamic_batch_kernels,
            )?;
        }
        let mut plan_sizes = BTreeSet::new();
        for plan in &self.prefill_plans {
            if plan.chunk_tokens == 0
                || plan.min_tokens > plan.chunk_tokens
                || (plan.min_tokens > 0 && self.segment_controls.is_none())
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
            if !plan.stages.is_empty() {
                let mut flattened = Vec::new();
                for stage in &plan.stages {
                    let ops = self
                        .programs
                        .get(stage)
                        .filter(|ops| !ops.is_empty())
                        .ok_or("Missing resumable prefill stage")?;
                    flattened.extend_from_slice(ops);
                }
                if flattened != self.programs[&plan.prefill_program] {
                    return Err("Prefill stages must exactly partition their forward pass".into());
                }
            } else if !plan.live_buffers.is_empty() {
                return Err("Atomic prefill cannot declare continuation buffers".into());
            }
            let mut live = BTreeSet::new();
            for name in &plan.live_buffers {
                let buffer = lookup(name)?;
                if !live.insert(name) || buffer.data.is_some() || buffer.access == Access::Read {
                    return Err("Invalid prefill continuation buffer".into());
                }
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
        validate_graph_branches(self, &self.kernels)?;
        if let Some(kv) = &self.kv_cache {
            for growth in kv.growth.values() {
                if let Some(name) = &growth.length {
                    let control = lookup(name)?;
                    if growth.tokens == 0
                        || control.dtype != crate::artifact::Dtype::I32
                        || control.bytes()? != 4
                        || control.data.is_some()
                        || control.access == Access::Read
                    {
                        return Err("Invalid logical KV length control".into());
                    }
                }
            }
        }
        if let Some(layout) = &self.batch_layout {
            for (&rows, program) in &layout.preparation_programs {
                let growth = self.kv_cache.as_ref().and_then(|kv| kv.growth.get(program));
                if (rows != 1 && !layout.profiles.contains_key(&rows))
                    || !self.programs.contains_key(program)
                    || growth.is_none_or(|g| {
                        g.tokens != rows && !(g.length.is_some() && g.tokens >= rows)
                    })
                {
                    return Err("Invalid batch packet preparation contract".into());
                }
            }
        }
        if !self.independent() && !self.reset_buffers.contains(&self.position) {
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
        if let Some(rows) = &self.row_execution {
            let capacity = self
                .prefill_plans
                .iter()
                .filter(|p| rows.programs.contains(&p.prefill_program))
                .map(|p| {
                    p.chunk_tokens
                        .min(rows.program_capacity(&p.prefill_program))
                })
                .max()
                .ok_or("Missing dynamic prefill template")?;
            let chunk = tokens.min(capacity);
            let plan = self
                .prefill_plan(chunk)
                .ok_or("Missing dynamic prefill template")?;
            return Ok((chunk, &plan.prefill_program, &plan.head_program));
        }
        if self.prefill_plans.is_empty() {
            if self.chunk_tokens == 0 || !tokens.is_multiple_of(self.chunk_tokens) {
                return Err("Input is not a multiple of the legacy chunk size".into());
            }
            return Ok((self.chunk_tokens, "prefill", "head"));
        }
        if let Some(plan) = self.prefill_plan(tokens).filter(|p| p.min_tokens > 0) {
            return Ok((tokens, &plan.prefill_program, &plan.head_program));
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

#[cfg(test)]
mod branch_tests {
    use super::*;
    #[test]
    fn ranged_prefill_keeps_capacity_and_rejects_uncovered_rows() {
        let plan: PrefillPlan = serde_json::from_value(serde_json::json!({
            "chunk_tokens":512, "min_tokens":129,
            "prefill_program":"body", "head_program":"head"
        }))
        .unwrap();
        for tokens in 129..=512 {
            assert!(plan.accepts(tokens));
        }
        for tokens in [0, 1, 128, 513] {
            assert!(!plan.accepts(tokens));
        }
        let exact: PrefillPlan = serde_json::from_value(serde_json::json!({
            "chunk_tokens":128, "prefill_program":"body", "head_program":"head"
        }))
        .unwrap();
        assert!(exact.accepts(128));
        assert!(!exact.accepts(127));
    }
    fn fixture() -> (Manifest, Vec<Kernel>) {
        let identity = serde_json::json!({"file":"k.cubin","sha256":"0".repeat(64)});
        let kernels: Vec<Kernel> = [
            ("fork", ["Input", "A"]),
            ("main", ["A", "C"]),
            ("side", ["Input", "B"]),
            ("join", ["C", "B"]),
        ]
        .into_iter()
        .map(|(name, args)| {
            serde_json::from_value(serde_json::json!({
                "name":name,"symbol":name,"module":identity,"source":identity,"host_abi":identity,
                "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,
                "readonly_arguments":[true,false],
                "args":args.map(|name|serde_json::json!({"kind":"buffer","name":name}))
            }))
            .unwrap()
        })
        .collect();
        let m = serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_87","model":"test",
            "chunk_tokens":512,"max_context":512,"vocab":16,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Position",
            "weight_bytes":0,"weight_parameters":0,"weight_scope":"test",
            "programs":{"decode":kernels.iter().map(|k|serde_json::json!({"kind":"kernel","name":k.name})).collect::<Vec<_>>()},
            "graph_branches":{"decode":[["fork","side","side","join"]]}
        })).unwrap();
        (m, kernels)
    }
    #[test]
    fn dynamic_prefill_uses_one_plan_for_single_rows_and_tails() {
        let (mut manifest, _) = fixture();
        manifest.prefill_plans = vec![PrefillPlan {
            chunk_tokens: 512,
            min_tokens: 1,
            prefill_program: "prefill".into(),
            head_program: "head".into(),
            stages: vec![],
            live_buffers: vec![],
        }];
        manifest.row_execution = Some(crate::operators::rows::RowExecution {
            capacity: 512,
            program_capacities: BTreeMap::new(),
            programs: BTreeSet::from(["prefill".into(), "head".into()]),
            variants: BTreeMap::new(),
        });
        for tokens in [1, 2, 17, 511, 512] {
            assert_eq!(manifest.prefill_plans[0].chunk(tokens), Some(tokens));
            assert_eq!(
                manifest.select_prefill_plan(tokens).unwrap(),
                (tokens, "prefill", "head")
            );
        }
        assert_eq!(
            manifest.select_prefill_plan(513).unwrap(),
            (512, "prefill", "head")
        );
        assert!(manifest.select_prefill_plan(0).is_err());
        assert_eq!(manifest.prefill_plans[0].chunk(0), None);
        assert_eq!(manifest.prefill_plans[0].chunk(513), Some(512));
        let rows = manifest.row_execution.as_mut().unwrap();
        rows.capacity = 32768;
        rows.programs.insert("vision".into());
        rows.program_capacities =
            BTreeMap::from([("prefill".into(), 512), ("vision".into(), 32768)]);
        for tokens in [513, 8192, 65536] {
            assert_eq!(
                manifest.select_prefill_plan(tokens).unwrap(),
                (512, "prefill", "head")
            );
        }
    }
    #[test]
    fn permits_shared_reads_and_rejects_cross_branch_dependencies() {
        let (m, mut k) = fixture();
        validate_graph_branches(&m, &k).unwrap();
        k[2].args[0] = Argument::Buffer { name: "A".into() };
        assert!(
            validate_graph_branches(&m, &k)
                .unwrap_err()
                .contains("Dependent")
        );
        let (m, mut k) = fixture();
        k[2].readonly_arguments.clear();
        assert!(
            validate_graph_branches(&m, &k)
                .unwrap_err()
                .contains("access")
        );
        let (m, mut k) = fixture();
        k[2].args[1] = Argument::BufferSlice {
            name: "A".into(),
            offset: 16,
        };
        assert!(validate_graph_branches(&m, &k).is_err());
    }
    #[test]
    fn row_substitutions_must_preserve_branch_independence() {
        use crate::operators::rows::{KernelVariant, RowExecution};
        let (mut m, mut kernels) = fixture();
        let mut alternate = kernels[2].clone();
        alternate.name = "side/aot".into();
        kernels.push(alternate);
        m.row_execution = Some(RowExecution {
            capacity: 128,
            programs: BTreeSet::from(["decode".into()]),
            program_capacities: BTreeMap::new(),
            variants: BTreeMap::from([(
                "side".into(),
                vec![KernelVariant {
                    min_rows: 1,
                    max_rows: 128,
                    multiple_of: 1,
                    kernels: vec!["side/aot".into()],
                }],
            )]),
        });
        validate_graph_branches(&m, &kernels).unwrap();
        assert_eq!(
            m.row_execution.as_ref().unwrap().kernel_slots()["side/aot"],
            "side"
        );
        kernels[4].args[1] = Argument::Buffer { name: "A".into() };
        assert!(
            validate_graph_branches(&m, &kernels)
                .unwrap_err()
                .contains("Dependent")
        );
        m.row_execution
            .as_mut()
            .unwrap()
            .variants
            .get_mut("side")
            .unwrap()[0]
            .kernels
            .clear();
        assert!(
            validate_graph_branches(&m, &kernels)
                .unwrap_err()
                .contains("endpoints")
        );
    }

    #[test]
    fn rejects_missing_endpoints_and_overlapping_regions() {
        let (mut m, k) = fixture();
        m.graph_branches.get_mut("decode").unwrap()[0][3] = "main".into();
        assert!(validate_graph_branches(&m, &k).is_err());
        let (mut m, k) = fixture();
        let r = m.graph_branches["decode"][0].clone();
        m.graph_branches.get_mut("decode").unwrap().push(r);
        assert!(
            validate_graph_branches(&m, &k)
                .unwrap_err()
                .contains("Overlapping")
        );
        let (mut m, k) = fixture();
        m.graph_branches.get_mut("decode").unwrap()[0][0] = "missing".into();
        assert!(validate_graph_branches(&m, &k).is_err());
    }
}
