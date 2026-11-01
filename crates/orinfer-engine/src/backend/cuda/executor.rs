use super::*;
use crate::execution::{CudaGraphMode, ExecutionPhase, LoadOptions};
use std::cell::RefCell;
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct BatchGraphKey {
    pub operation: orinfer_model_sdk::abi::BatchOperation,
    pub segments: Vec<(usize, usize)>,
    pub rows: usize,
    pub shared: Option<SharedGraphKey>,
}
impl From<Vec<(usize, usize)>> for BatchGraphKey {
    fn from(segments: Vec<(usize, usize)>) -> Self {
        Self {
            operation: orinfer_model_sdk::abi::BatchOperation::Ordinary,
            rows: segments.iter().map(|(_, tokens)| tokens).sum(),
            segments,
            shared: None,
        }
    }
}
impl BatchGraphKey {
    pub fn iter(&self) -> std::slice::Iter<'_, (usize, usize)> {
        self.segments.iter()
    }
}
pub(super) type BatchGraphCache = BTreeMap<BatchGraphKey, (Handle, u64, usize)>;

/// Shared pieces are keyed by program and resolved views, never membership.
/// Their session-owned addresses stay fixed; request state uses indirect tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SharedGraphKey {
    pub operations: std::rc::Rc<Vec<crate::execution::Invocation>>,
    pub addresses: Vec<u64>,
}
impl Ord for SharedGraphKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let operations = if std::rc::Rc::ptr_eq(&self.operations, &other.operations) {
            std::cmp::Ordering::Equal
        } else {
            self.operations.cmp(&other.operations)
        };
        operations.then_with(|| self.addresses.cmp(&other.addresses))
    }
}
impl PartialOrd for SharedGraphKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
pub(super) type SharedGraphCache = BTreeMap<SharedGraphKey, (Handle, u64)>;

/// Shared addresses stay fixed for this executor's lifetime. Only request
/// addresses can be rebound; shared graph capture rejects those names.
pub(crate) struct BufferAddresses {
    values: BTreeMap<String, u64>,
    private: std::collections::BTreeSet<String>,
}

impl BufferAddresses {
    fn new(values: BTreeMap<String, u64>, private: std::collections::BTreeSet<String>) -> Self {
        Self { values, private }
    }

    pub(super) fn is_private(&self, name: &str) -> bool {
        self.private.contains(name)
    }

    pub(super) fn bind_private(&mut self, name: &str, address: u64) -> Result<()> {
        if !self.is_private(name) {
            return Err("Cannot rebind a session-owned buffer".into());
        }
        *self.values.get_mut(name).ok_or("Unknown private buffer")? = address;
        Ok(())
    }
}

impl std::ops::Deref for BufferAddresses {
    type Target = BTreeMap<String, u64>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

#[cfg(test)]
mod address_tests {
    use super::BufferAddresses;

    #[test]
    fn only_request_addresses_can_be_rebound() {
        let mut addresses = BufferAddresses::new(
            [("weights", 10), ("workspace", 20), ("request", 30)]
                .map(|(name, address)| (name.into(), address))
                .into(),
            ["request".into()].into(),
        );
        for name in ["weights", "workspace", "unknown"] {
            assert!(!addresses.is_private(name));
            assert!(addresses.bind_private(name, 99).is_err());
        }
        assert_eq!(addresses.len(), 3);
        assert_eq!(addresses["weights"], 10);
        assert_eq!(addresses["workspace"], 20);
        assert_eq!(addresses["request"], 30);
        assert!(addresses.is_private("request"));
        addresses.bind_private("request", 40).unwrap();
        assert_eq!(addresses["request"], 40);
        addresses.bind_private("request", 0).unwrap();
        assert_eq!(addresses["request"], 0);
    }

    #[test]
    fn binding_never_creates_an_unknown_private_address() {
        let mut addresses =
            BufferAddresses::new([("weights".into(), 10)].into(), ["missing".into()].into());
        assert!(addresses.bind_private("missing", 99).is_err());
        assert_eq!(addresses.len(), 1);
        assert_eq!(addresses["weights"], 10);
    }
}

pub(super) struct DirectKernel {
    pub(super) spec: Kernel,
    pub(super) function: Option<sys::CUfunction>,
    pub(super) values: Vec<Value>,
    workspaces_ready: bool,
}
pub(super) type RowProgramCache = std::collections::VecDeque<(
    String,
    usize,
    std::rc::Rc<Vec<crate::execution::Invocation>>,
)>;

pub(super) struct DirectPrograms {
    pub(super) kernels: RefCell<BTreeMap<String, DirectKernel>>,
    pub(super) programs: BTreeMap<String, Vec<crate::model::Operation>>,
    pub(super) row_execution: Option<orinfer_model_sdk::operators::rows::RowExecution>,
    pub(super) row_kernels:
        BTreeMap<String, orinfer_model_sdk::operators::dynamic::DynamicBatchKernel>,
    pub(super) row_cache: RefCell<RowProgramCache>,
}

/// Execute an architecture adapter's explicit programs with stable allocations.
/// All handles, pointers and argument backing storage stay inside this session.
/// A CUDA model is thread-affine. Construct, use and drop it on its worker.
pub(crate) struct Executor {
    pub(crate) session: Session,
    pub(crate) pointers: BufferAddresses,
    pub(crate) sizes: BTreeMap<String, usize>,
    pub(crate) graphs: RefCell<BTreeMap<String, Handle>>,
    pub(super) direct: Option<DirectPrograms>,
    pub(super) kernels: RefCell<super::kernels::Catalog>,
    pub(super) cuda_graph: CudaGraphMode,
    pub(super) expert_cache: RefCell<Option<super::experts::Pager>>,
    pub(super) growth: BTreeMap<String, crate::model::KvGrowth>,
    pub(crate) peak_kv_bytes: std::cell::Cell<usize>,
    pub(crate) peak_prefill_workspace_bytes: std::cell::Cell<usize>,
    pub(super) prefill_workspace: std::collections::BTreeSet<String>,
    pub(crate) allocations: Allocations,
    workspace_dependencies: BTreeMap<String, Vec<String>>,
    text_workspaces: Vec<String>,
    pub(crate) parallel_regions: RefCell<Vec<[String; 4]>>,
    pub(super) row_kernel_slots: BTreeMap<String, String>,
    #[cfg(test)]
    pub(crate) parallel_gate_first: std::cell::Cell<bool>,
    #[cfg(test)]
    pub(crate) parallel_priority: RefCell<String>,
    #[cfg(test)]
    pub(crate) live_graph_events: std::cell::Cell<Option<[sys::CUevent; 2]>>,
    #[cfg(test)]
    pub(crate) live_graph_ms: RefCell<Vec<f32>>,
    pub(super) live_workspace: Option<(usize, std::rc::Weak<RefCell<super::snapshot::Workspace>>)>,
    pub(crate) workspace_save_s: f64,
    pub(crate) workspace_restore_s: f64,
    pub(super) sequences: Vec<super::sequence::Sequence>,
    pub(super) state_table: RefCell<super::state_table::StateTable>,
    pub(super) idle_selection: RefCell<Option<super::sequence::IdleSelection>>,
    pub(super) active_sequence: usize,
    pub(super) sequence_specs: Vec<crate::artifact::Buffer<crate::weights::TensorIdentity>>,
    pub(super) sequence_strides: BTreeMap<String, usize>,
    pub(super) batch_graphs: RefCell<BatchGraphCache>,
    pub(super) batch_graph_trials: RefCell<BTreeMap<BatchGraphKey, usize>>,
    pub(super) shared_graphs: RefCell<SharedGraphCache>,
    pub(super) shared_graph_trials: RefCell<BTreeMap<SharedGraphKey, usize>>,
    pub(super) batch_graph_clock: std::cell::Cell<u64>,
    pub(crate) batch_statistics: std::cell::Cell<crate::scheduler::BatchExecutionStatistics>,
}

/// Logical arenas share one CUDA owner today. Their addresses are separate so
/// future sequence scheduling can retain weights and replace request state.
#[derive(Default)]
pub(crate) struct BufferArena {
    pub addresses: BTreeMap<String, u64>,
    pub bytes: usize,
}
#[derive(Default)]
pub(crate) struct Allocations {
    pub weights: BufferArena,
    pub sequence: BufferArena,
    pub workspace: BufferArena,
}
pub(crate) use crate::backend::report::LoadStats;

fn deferred_workspace(
    buffer: &crate::artifact::Buffer<crate::weights::TensorIdentity>,
    scope: crate::loader::BufferScope,
) -> Result<bool> {
    // Small controls stay in ordinary allocations; VMM granularity would cost
    // more physical memory than their payload. Large scratch keeps stable VA.
    Ok(scope == crate::loader::BufferScope::Workspace
        && buffer.data.is_none()
        && buffer.bytes()? >= 64 * 1024)
}

fn initial_buffer_bytes(
    manifest: &crate::model::Manifest,
    buffer: &crate::artifact::Buffer<crate::weights::TensorIdentity>,
    scope: crate::loader::BufferScope,
) -> Result<usize> {
    // The bootstrap sequence has no images. Its first real lease selects an
    // arena sized to the actual request, including the complete feature bank.
    if scope == crate::loader::BufferScope::Sequence
        && let Some(vision) = &manifest.vision
        && buffer.name == vision.features
    {
        return Ok(vision.hidden * 2);
    }
    buffer.bytes()
}

impl Executor {
    pub(crate) fn load(
        prepared: &crate::loader::PreparedModel,
        options: LoadOptions,
        preload: &std::collections::BTreeSet<String>,
    ) -> Result<(Self, LoadStats)> {
        let manifest = &prepared.plan;
        let base = &prepared.weights_root;
        let kernel_base = &prepared.kernel_root;
        let scopes = &prepared.scopes;
        let decode_programs = &prepared.decode_programs;
        let fingerprint = prepared.fingerprint.clone();
        let expert_plan = super::experts::Plan::new(prepared, options.resident_experts)?;
        let cuda_graph = if expert_plan.is_some() {
            if options.cuda_graph != CudaGraphMode::Off {
                eprintln!(
                    "Finite expert residency uses direct execution for route-dependent SSD paging (requested CUDA graph mode: {})",
                    options.cuda_graph
                );
            }
            CudaGraphMode::Off
        } else {
            options.cuda_graph
        };
        let load_workers = options.weight_workers()?;
        if decode_programs
            .iter()
            .any(|name| !manifest.programs.contains_key(name))
        {
            return Err("Architecture declares a missing decode program".into());
        }
        let started = Instant::now();
        let capacity_bytes = manifest.validate()?;
        let lazy = manifest.kv_cache.as_ref().filter(|kv| kv.demand_mapping);
        let lazy_bytes = manifest
            .buffers
            .iter()
            .filter(|b| {
                lazy.is_some_and(|kv| {
                    kv.buffers.contains_key(&b.name) || kv.prefill_workspace.contains_key(&b.name)
                })
            })
            .try_fold(0usize, |sum, b| {
                b.bytes()
                    .and_then(|n| sum.checked_add(n).ok_or("KV sum overflow".into()))
            })?;
        let mut used = std::collections::BTreeSet::new();
        for kernel in &manifest.kernels {
            for arg in &kernel.args {
                match arg {
                    Argument::Buffer { name } | Argument::BufferSlice { name, .. } => {
                        used.insert(name.as_str());
                    }
                    _ => {}
                }
            }
        }
        for op in manifest.programs.values().flatten() {
            match op {
                crate::model::Operation::Copy {
                    source,
                    destination,
                    ..
                } => {
                    used.extend([source.as_str(), destination.as_str()]);
                }
                crate::model::Operation::Zero { destination, .. } => {
                    used.insert(destination.as_str());
                }
                _ => {}
            }
        }
        let unused_weights = manifest
            .buffers
            .iter()
            .filter(|b| {
                b.data.is_some()
                    && b.access == crate::artifact::Access::Read
                    && !used.contains(b.name.as_str())
            })
            .try_fold(0usize, |sum, b| {
                sum.checked_add(b.bytes()?)
                    .ok_or_else(|| "Unused weight sum overflow".to_string())
            })?;
        let paged_bytes = manifest
            .buffers
            .iter()
            .filter(|b| {
                expert_plan
                    .as_ref()
                    .is_some_and(|p| p.names.contains(&b.name))
            })
            .try_fold(0usize, |sum, b| {
                sum.checked_add(b.bytes()?)
                    .ok_or_else(|| "Expert weight sum overflow".to_string())
            })?;
        let weight_layout =
            super::weight_arena::ArenaLayout::new(manifest.buffers.iter().filter(|b| {
                b.data.is_some()
                    && b.access == crate::artifact::Access::Read
                    && used.contains(b.name.as_str())
                    && !expert_plan
                        .as_ref()
                        .is_some_and(|p| p.names.contains(&b.name))
            }))?;
        let capacity_bytes = capacity_bytes
            .checked_add(weight_layout.padding_bytes)
            .ok_or("Model allocation sum overflow")?;
        let deferred_workspace_bytes = manifest.buffers.iter().try_fold(0usize, |sum, b| {
            let existing = lazy.is_some_and(|kv| kv.prefill_workspace.contains_key(&b.name));
            sum.checked_add(if !existing && deferred_workspace(b, scopes[&b.name])? {
                b.bytes()?
            } else {
                0
            })
            .ok_or_else(|| "Workspace sum overflow".to_string())
        })?;
        let buffer_bytes = capacity_bytes
            - lazy_bytes
            - unused_weights
            - paged_bytes
            - deferred_workspace_bytes
            - manifest.buffers.iter().try_fold(0usize, |sum, b| {
                Ok::<_, String>(
                    sum + b.bytes()? - initial_buffer_bytes(manifest, b, scopes[&b.name])?,
                )
            })?;
        let init_started = Instant::now();
        let mut s = Session::new(Driver::load()?);
        let mut pointers = BTreeMap::new();
        let mut sizes = BTreeMap::new();
        let mut allocations = Allocations::default();
        let (mut major, mut minor, mut version) = (0, 0, 0);
        let mut name = [0u8; 256];
        let mut total = 0usize;
        // SAFETY: This owner initializes and retains the context on this thread.
        // The synchronous upload borrows live host storage until the driver returns.
        // Async memset touches separate session-owned allocations on our stream.
        unsafe {
            check(sys::cuInit(0), "cuInit model")?;
            check(sys::cuDeviceGet(&mut s.device, 0), "model device")?;
            check(
                sys::cuDeviceGetAttribute(
                    &mut major,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    s.device,
                ),
                "SM major",
            )?;
            check(
                sys::cuDeviceGetAttribute(
                    &mut minor,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    s.device,
                ),
                "SM minor",
            )?;
            if [major, minor] != [8, 7] {
                return Err("Model requires SM87".into());
            }
            check(sys::cuDriverGetVersion(&mut version), "driver version")?;
            check(
                sys::cuDeviceGetName(name.as_mut_ptr().cast(), 256, s.device),
                "device name",
            )?;
            check(
                sys::cuDeviceTotalMem_v2(&mut total, s.device),
                "total memory",
            )?;
            check(sys::cuCtxGetCurrent(&mut s.previous), "previous context")?;
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
                "retain context",
            )?;
            s.retained = true;
            check(sys::cuCtxSetCurrent(context), "current context")?;
            check(sys::cuStreamCreate(&mut s.stream, 1), "model stream")?;
            let (mut free, mut memtotal) = (0usize, 0usize);
            check(
                sys::cuMemGetInfo_v2(&mut free, &mut memtotal),
                "available memory",
            )?;
            if buffer_bytes > memtotal {
                return Err(format!(
                    "Model needs {buffer_bytes} bytes, device total {memtotal}"
                ));
            }
            // On Tegra the instantaneous free estimate excludes memory the OS
            // can reclaim (CUDA for Tegra, memory management). Let checked
            // allocations decide availability; Session rolls back on failure.
            if buffer_bytes > free {
                eprintln!(
                    "Model needs {buffer_bytes} bytes, CUDA free estimate {free}; checking allocations"
                );
            }
        }
        let mut load_stages = crate::model::LoadStages {
            cuda_init_s: init_started.elapsed().as_secs_f64(),
            buffer_alloc_s: 0.,
            kernel_assets_s: 0.,
            weight_read_wait_s: 0.,
            weight_dma_wait_s: 0.,
            weight_upload_chunks: 0,
            final_sync_s: 0.,
            cuda_allocations: 0,
            weight_arena_padding_bytes: weight_layout.padding_bytes,
            deferred_workspace_bytes,
        };
        let allocation_started = Instant::now();
        let mut weight_base = 0;
        if weight_layout.allocation_bytes != 0 {
            // SAFETY: Validated layout; register the single owner before any
            // fallible binding. Logical slices are never freed individually.
            unsafe {
                check(
                    sys::cuMemAlloc_v2(&mut weight_base, weight_layout.allocation_bytes),
                    "allocate immutable weight arena",
                )?;
            }
            s.buffers.push(weight_base);
            weight_base = weight_layout.base(weight_base)?;
            load_stages.cuda_allocations += 1;
        }
        load_stages.buffer_alloc_s += allocation_started.elapsed().as_secs_f64();
        allocations.weights.bytes = weight_layout.padding_bytes;
        // Buffers below that no weights/paged/lazy/deferred path claims would
        // each take an individual cuMemAlloc; carve them from one fixed arena
        // so load cost scales with bytes instead of the buffer count.
        let mut direct_offsets: BTreeMap<&str, u64> = BTreeMap::new();
        let mut direct_cursor: u64 = 0;
        for b in &manifest.buffers {
            if b.data.is_some()
                && b.access == crate::artifact::Access::Read
                && !used.contains(b.name.as_str())
            {
                continue;
            }
            let paged = expert_plan
                .as_ref()
                .is_some_and(|p| p.names.contains(&b.name));
            let stride = lazy
                .and_then(|kv| {
                    kv.buffers
                        .get(&b.name)
                        .or_else(|| kv.prefill_workspace.get(&b.name))
                })
                .copied();
            let deferred = stride.is_none() && deferred_workspace(b, scopes[&b.name])?;
            if weight_layout.offsets.contains_key(&b.name) || paged || stride.is_some() || deferred
            {
                continue;
            }
            let bytes = initial_buffer_bytes(manifest, b, scopes[&b.name])? as u64;
            direct_cursor = direct_cursor
                .checked_next_multiple_of(b.alignment)
                .ok_or("Fixed arena alignment overflow")?;
            direct_offsets.insert(b.name.as_str(), direct_cursor);
            direct_cursor = direct_cursor
                .checked_add(bytes)
                .ok_or("Fixed arena size overflow")?;
        }
        let mut fixed_base = 0;
        if direct_cursor != 0 {
            let allocation_started = Instant::now();
            // SAFETY: The checked layout above owns the extent; Session frees
            // the single base and never releases interior carve-outs.
            unsafe {
                check(
                    sys::cuMemAlloc_v2(&mut fixed_base, direct_cursor as usize),
                    "allocate fixed buffer arena",
                )?;
            }
            s.buffers.push(fixed_base);
            load_stages.cuda_allocations += 1;
            load_stages.buffer_alloc_s += allocation_started.elapsed().as_secs_f64();
        }
        let (mut weight_io_hash_s, mut weight_upload_s) = (0.0, 0.0);
        let mut weights = crate::weights::Weights::open(base, options.verify_weights)?;
        let mut uploads = Vec::new();
        let mut expert_sources = BTreeMap::new();
        for b in &manifest.buffers {
            if b.data.is_some()
                && b.access == crate::artifact::Access::Read
                && !used.contains(b.name.as_str())
            {
                continue;
            }
            let bytes = initial_buffer_bytes(manifest, b, scopes[&b.name])?;
            let address;
            let paged = expert_plan
                .as_ref()
                .is_some_and(|p| p.names.contains(&b.name));
            // SAFETY: Allocation length is checked by manifest validation and owned
            // by Session before any fallible operation can return.
            let stride = lazy
                .and_then(|kv| {
                    kv.buffers
                        .get(&b.name)
                        .or_else(|| kv.prefill_workspace.get(&b.name))
                })
                .copied();
            let deferred = stride.is_none() && deferred_workspace(b, scopes[&b.name])?;
            let allocation_started = Instant::now();
            if let Some(&offset) = weight_layout.offsets.get(&b.name) {
                address = weight_base
                    .checked_add(offset as u64)
                    .ok_or("Weight slice address overflow")?;
            } else if paged {
                let reservation = super::virtual_memory::Reservation::reserve(&s, bytes, 1)?;
                address = reservation.address;
                s.expert_buffers
                    .borrow_mut()
                    .insert(b.name.clone(), reservation);
            } else if let Some(stride) = stride {
                let reservation = super::virtual_memory::Reservation::reserve(&s, bytes, stride)?;
                address = reservation.address;
                s.virtual_buffers
                    .borrow_mut()
                    .insert(b.name.clone(), reservation);
            } else if deferred {
                let reservation = super::virtual_memory::Reservation::reserve(&s, bytes, 1)?;
                address = reservation.address;
                s.workspace_buffers
                    .borrow_mut()
                    .insert(b.name.clone(), reservation);
            } else {
                let offset = direct_offsets
                    .get(b.name.as_str())
                    .ok_or("Fixed buffer missing from its arena layout")?;
                address = fixed_base
                    .checked_add(*offset)
                    .ok_or("Fixed buffer address overflow")?;
            }
            load_stages.buffer_alloc_s += allocation_started.elapsed().as_secs_f64();
            if address % b.alignment != 0 {
                return Err("Model allocation alignment".into());
            }
            pointers.insert(b.name.clone(), address);
            sizes.insert(b.name.clone(), bytes);
            let arena = match scopes[&b.name] {
                crate::loader::BufferScope::Weights => &mut allocations.weights,
                crate::loader::BufferScope::Sequence => &mut allocations.sequence,
                crate::loader::BufferScope::Workspace => &mut allocations.workspace,
            };
            arena.addresses.insert(b.name.clone(), address);
            if !paged {
                arena.bytes += bytes;
            }
            if b.data.is_some() {
                let t = Instant::now();
                if paged {
                    if options.verify_weights {
                        weights.read(b)?;
                    }
                    expert_sources.insert(b.name.clone(), weights.source(b)?);
                    weight_io_hash_s += t.elapsed().as_secs_f64();
                } else if options.verify_weights {
                    let raw = weights.read(b)?;
                    weight_io_hash_s += t.elapsed().as_secs_f64();
                    let t = Instant::now();
                    // SAFETY: Validated immutable host/device extents; the
                    // synchronous full-verification upload keeps raw alive.
                    unsafe {
                        check(
                            sys::cuMemcpyHtoD_v2(address, raw.as_ptr().cast(), bytes),
                            "verified weight upload",
                        )?;
                    }
                    weight_upload_s += t.elapsed().as_secs_f64();
                } else {
                    uploads.push(super::weight_upload::Upload {
                        source: weights.source(b)?,
                        destination: address,
                    });
                    weight_io_hash_s += t.elapsed().as_secs_f64();
                }
            } else if stride.is_none() && !deferred {
                // SAFETY: Whole allocation is owned and not in use yet.
                unsafe {
                    check(
                        sys::cuMemsetD8Async(address, 0, bytes, s.stream),
                        "initialize workspace",
                    )?;
                }
            }
        }
        drop(weights);
        let mut catalog = None;
        let mut functions = BTreeMap::new();
        let mut pending_modules = manifest
            .kernels
            .iter()
            .filter(|k| preload.contains(&k.name))
            .peekable();
        let mut initialization_s = 0.;
        let upload_stats;
        {
            let mut initialize = || -> Result<bool> {
                if catalog.is_some() && pending_modules.peek().is_none() {
                    return Ok(false);
                }
                let at = Instant::now();
                if catalog.is_none() {
                    let assets_at = Instant::now();
                    catalog = Some(super::kernels::Catalog::new(
                        &s,
                        kernel_base,
                        &manifest.kernels,
                    )?);
                    load_stages.kernel_assets_s = assets_at.elapsed().as_secs_f64();
                }
                let worked = if let Some(kernel) = pending_modules.next() {
                    let function = catalog.as_mut().unwrap().resolve(&s, kernel)?;
                    functions.insert(kernel.name.clone(), function);
                    true
                } else {
                    false
                };
                initialization_s += at.elapsed().as_secs_f64();
                Ok(worked)
            };
            let t = Instant::now();
            upload_stats =
                super::weight_upload::upload_with_idle(&s, uploads, load_workers, &mut initialize)?;
            weight_upload_s += (t.elapsed().as_secs_f64() - upload_stats.idle_work_s).max(0.);
            // Empty/small models may finish reading before all modules load.
            // Complete initialization on the same CUDA owner before execution.
            while initialize()? {}
        }
        load_stages.weight_read_wait_s = upload_stats.read_wait_s;
        load_stages.weight_dma_wait_s = upload_stats.dma_wait_s;
        load_stages.weight_upload_chunks = upload_stats.chunks;
        let catalog = catalog.ok_or("Missing initialized kernel catalog")?;
        let module_started = Instant::now();
        let mut launches = BTreeMap::new();
        for k in &manifest.kernels {
            let function = functions.remove(&k.name);
            let values = k
                .args
                .iter()
                .map(|a| match a {
                    Argument::Buffer { name } => Value::Pointer(pointers[name]),
                    Argument::BufferSlice { name, offset } => {
                        Value::Pointer(pointers[name] + *offset as u64)
                    }
                    Argument::I32 { value } => Value::I32(*value),
                    Argument::U32 { value } => Value::U32(*value),
                    Argument::I64 { value } => Value::I64(*value),
                    Argument::U64 { value } => Value::U64(*value),
                    Argument::F32 { value } => Value::F32(*value),
                })
                .collect();
            launches.insert(
                k.name.clone(),
                DirectKernel {
                    spec: k.clone(),
                    function,
                    values,
                    workspaces_ready: false,
                },
            );
        }
        let preloaded_modules = catalog.loaded_modules();
        let registered_modules = catalog.registered_modules();
        let module_load_bind_s = initialization_s + module_started.elapsed().as_secs_f64();
        let sync_started = Instant::now();
        // SAFETY: Flush all default-stream uploads and workspace initialization
        // before capture on a nonblocking stream. No allocation occurs in graphs.
        unsafe {
            check(sys::cuCtxSynchronize(), "model uploads complete")?;
        }
        load_stages.final_sync_s = sync_started.elapsed().as_secs_f64();
        // Request limits can replace initial private buffers on the first lease.
        // Capture only when a program is submitted with its final arena bindings;
        // this also avoids retaining unused MTP and prefill graphs at startup.
        let graphs = BTreeMap::new();
        let graph_capture_s = 0.0;
        let direct = Some(DirectPrograms {
            kernels: RefCell::new(launches),
            programs: manifest.programs.clone(),
            row_execution: manifest.row_execution.clone(),
            row_kernels: manifest.dynamic_batch_kernels.clone(),
            row_cache: RefCell::default(),
        });
        let kernel_map: BTreeMap<_, _> = manifest
            .kernels
            .iter()
            .map(|k| (k.name.as_str(), k))
            .collect();
        let mut workspace_dependencies: BTreeMap<String, Vec<String>> = manifest
            .programs
            .iter()
            .map(|(name, ops)| {
                let mut buffers = std::collections::BTreeSet::new();
                for op in ops {
                    match op {
                        crate::model::Operation::Kernel { name } => {
                            for arg in &kernel_map[name.as_str()].args {
                                if let Argument::Buffer { name }
                                | Argument::BufferSlice { name, .. } = arg
                                {
                                    buffers.insert(name.clone());
                                }
                            }
                        }
                        crate::model::Operation::Copy {
                            source,
                            destination,
                            ..
                        } => {
                            buffers.extend([source.clone(), destination.clone()]);
                        }
                        crate::model::Operation::Zero { destination, .. } => {
                            buffers.insert(destination.clone());
                        }
                    }
                }
                buffers.retain(|n| s.workspace_buffers.borrow().contains_key(n));
                (name.clone(), buffers.into_iter().collect())
            })
            .collect();
        // Admission runs on every iteration; freeze shared dependency unions
        // once instead of scanning hundreds of native aliases per token.
        let text_workspaces: std::collections::BTreeSet<_> = manifest
            .programs
            .keys()
            .filter(|name| {
                !manifest
                    .vision
                    .as_ref()
                    .is_some_and(|v| v.plans.iter().any(|p| p.program == **name))
                    && !(manifest.audio.is_some()
                        && matches!(name.as_str(), "audio_stem" | "audio_encoder"))
                    && !manifest.sampling.as_ref().is_some_and(|s| {
                        s.program == **name || s.batch.as_ref().is_some_and(|b| b.program == **name)
                    })
            })
            .flat_map(|p| workspace_dependencies[p].iter().cloned())
            .collect();
        for names in workspace_dependencies.values_mut() {
            names.retain(|n| !text_workspaces.contains(n));
        }
        let load_to_ready_s = started.elapsed().as_secs_f64();
        let captured_programs: Vec<String> = graphs.keys().cloned().collect();
        eprintln!(
            "GPU LOAD complete after {load_to_ready_s:.3}s; {buffer_bytes} resident buffer bytes, {capacity_bytes} capacity bytes; cuda_graph={cuda_graph}, {} captured programs",
            captured_programs.len()
        );
        eprintln!(
            "LOAD STAGES {}",
            serde_json::to_string(&load_stages).map_err(|e| e.to_string())?
        );
        let device = DeviceInfo::Cuda {
            name: String::from_utf8_lossy(&name)
                .trim_end_matches('\0')
                .to_string(),
            sm: [major, minor],
            driver_version: version,
            total_bytes: total,
        };
        let executor = Self {
            row_kernel_slots: manifest
                .row_execution
                .as_ref()
                .map(|rows| rows.kernel_slots())
                .unwrap_or_default(),
            parallel_regions: RefCell::new(
                manifest
                    .graph_branches
                    .values()
                    .flatten()
                    .cloned()
                    .collect(),
            ),
            #[cfg(test)]
            parallel_gate_first: Default::default(),
            #[cfg(test)]
            parallel_priority: RefCell::new("small_side".into()),
            #[cfg(test)]
            live_graph_events: Default::default(),
            #[cfg(test)]
            live_graph_ms: Default::default(),
            session: s,
            pointers: BufferAddresses::new(
                pointers,
                manifest
                    .buffers
                    .iter()
                    .filter(|b| scopes[&b.name] == crate::loader::BufferScope::Sequence)
                    .map(|b| b.name.clone())
                    .collect(),
            ),
            sizes,
            graphs: RefCell::new(graphs),
            direct,
            kernels: RefCell::new(catalog),
            cuda_graph,
            expert_cache: RefCell::new(
                expert_plan.map(|p| super::experts::Pager::new(p, expert_sources, load_workers)),
            ),
            growth: manifest
                .kv_cache
                .as_ref()
                .map(|kv| kv.growth.clone())
                .unwrap_or_default(),
            allocations,
            workspace_dependencies,
            text_workspaces: text_workspaces.into_iter().collect(),
            peak_kv_bytes: std::cell::Cell::new(0),
            peak_prefill_workspace_bytes: std::cell::Cell::new(0),
            prefill_workspace: manifest
                .kv_cache
                .as_ref()
                .map(|kv| kv.prefill_workspace.keys().cloned().collect())
                .unwrap_or_default(),
            sequences: vec![super::sequence::Sequence::default()],
            state_table: Default::default(),
            idle_selection: Default::default(),
            live_workspace: None,
            workspace_save_s: 0.,
            workspace_restore_s: 0.,
            active_sequence: 0,
            sequence_specs: manifest
                .buffers
                .iter()
                .filter(|b| scopes[&b.name] == crate::loader::BufferScope::Sequence)
                .cloned()
                .collect(),
            sequence_strides: lazy.map(|kv| kv.buffers.clone()).unwrap_or_default(),
            batch_graphs: Default::default(),
            batch_graph_trials: Default::default(),
            shared_graphs: Default::default(),
            shared_graph_trials: Default::default(),
            batch_statistics: Default::default(),
            batch_graph_clock: std::cell::Cell::new(0),
        };
        let stats = LoadStats {
            resident_experts: options.resident_experts,
            manifest_sha256: fingerprint,
            device,
            load_to_ready_s,
            load_stages,
            verify_weights: options.verify_weights,
            load_workers,
            preloaded_modules,
            registered_modules,
            weight_io_hash_s,
            weight_upload_s,
            module_load_bind_s,
            graph_capture_s,
            cuda_graph,
            captured_programs,
            buffer_bytes,
            buffer_capacity_bytes: capacity_bytes,
        };
        let mut executor = executor;
        executor.sequences[0].addresses = executor.allocations.sequence.addresses.clone();
        executor.sequences[0].sizes = executor
            .allocations
            .sequence
            .addresses
            .keys()
            .map(|n| (n.clone(), executor.sizes[n]))
            .collect();
        // Sequence buffers carved from the fixed arena are interior slices of
        // one allocation: lease growth and idle reclaim must skip their free.
        for name in direct_offsets.keys() {
            if executor.allocations.sequence.addresses.contains_key(*name) {
                executor.sequences[0].mark_fixed_slice(name);
            }
        }
        Ok((executor, stats))
    }
    pub(crate) fn sync(&self) -> Result<()> {
        // SAFETY: This thread owns the live stream and all graph allocations.
        unsafe {
            check(
                sys::cuStreamSynchronize(self.session.stream),
                "model synchronize",
            )
        }
    }
    pub(crate) fn reset_sequence(&self, names: &[String]) -> Result<()> {
        self.reset_sequence_inner(names, false)
    }
    pub(crate) fn reset_sequence_reusing_pages(&self, names: &[String]) -> Result<()> {
        self.reset_sequence_inner(names, true)
    }
    fn reset_sequence_inner(&self, names: &[String], reuse: bool) -> Result<()> {
        self.idle_selection.borrow_mut().take();
        self.sync()?;
        // Keep at most one allocation granule per KV buffer for short requests.
        // Clear retained pages before reuse; larger contexts release their slabs.
        for (name, buffer) in self.session.virtual_buffers.borrow_mut().iter_mut() {
            if self.sequence_strides.contains_key(name) {
                if reuse
                    && buffer.mapped != 0
                    && buffer.mapped <= buffer.granularity
                    && !buffer.has_shared_slabs()
                {
                    // SAFETY: Completed stream, owned mapping and full mapped range.
                    unsafe {
                        check(
                            sys::cuMemsetD8Async(
                                buffer.address,
                                0,
                                buffer.mapped,
                                self.session.stream,
                            ),
                            "clear reusable KV page",
                        )?;
                    }
                } else {
                    buffer.release_slabs(&self.session.driver)?;
                }
            }
        }
        for name in names {
            if self.session.virtual_buffers.borrow().contains_key(name) {
                continue;
            }
            let address = *self.sequences[self.active_sequence]
                .addresses
                .get(name)
                .ok_or_else(|| format!("{name}: reset outside sequence arena"))?;
            // SAFETY: The completed request owns these validated mutable state
            // allocations. The next request cannot run until this stream completes.
            unsafe {
                check(
                    sys::cuMemsetD8Async(address, 0, self.sizes[name], self.session.stream),
                    "reset sequence",
                )?;
            }
        }
        self.sync()
    }
    pub(crate) fn download_bytes(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        if bytes > *self.sizes.get(name).ok_or("Unknown download buffer")? {
            return Err("Download exceeds buffer".into());
        }
        self.ensure_workspace(name, bytes)?;
        if self
            .session
            .virtual_buffers
            .borrow()
            .get(name)
            .is_some_and(|b| bytes > b.mapped)
        {
            return Err("Download exceeds resident KV extent".into());
        }
        self.sync()?;
        let mut raw = vec![0; bytes];
        // SAFETY: Both ranges cover bytes and the producing stream has completed.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(), self.pointers[name], bytes),
                "download tensor",
            )?;
        }
        Ok(raw)
    }
    /// Diagnostic read after the caller has synchronized the producing step.
    /// Uses the same synchronous copy as the original benchmark path.
    pub(crate) fn read_ready_bytes(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        if bytes > *self.sizes.get(name).ok_or("Unknown download buffer")? {
            return Err("Download exceeds buffer".into());
        }
        if self
            .session
            .virtual_buffers
            .borrow()
            .get(name)
            .is_some_and(|b| bytes > b.mapped)
        {
            return Err("Download exceeds resident KV extent".into());
        }
        let mut raw = vec![0; bytes];
        // SAFETY: The validated device and host spans cover bytes; this
        // synchronous copy preserves the diagnostic caller's completed-step order.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(raw.as_mut_ptr().cast(), self.pointers[name], bytes),
                "download completed tensor",
            )?;
        }
        Ok(raw)
    }
    pub(crate) fn copy_range(
        &self,
        source: &str,
        destination: &str,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        self.copy_range_async(source, 0, destination, offset, bytes)?;
        self.sync()
    }
    /// Checked copy on the execution stream; the caller owns its final fence.
    pub(crate) fn copy_range_async(
        &self,
        source: &str,
        source_offset: usize,
        destination: &str,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let source_bytes = *self.sizes.get(source).ok_or("Unknown copy source")?;
        if source_offset
            .checked_add(bytes)
            .is_none_or(|n| n > source_bytes)
            || offset
                .checked_add(bytes)
                .is_none_or(|n| n > self.sizes.get(destination).copied().unwrap_or(0))
        {
            return Err("Copy exceeds buffer".into());
        }
        self.ensure_workspace(source, source_offset + bytes)?;
        self.ensure_writable_range(destination, offset, bytes)?;
        // SAFETY: The validated ranges refer to live allocations owned by this
        // executor and the transfer is ordered with graph work on the same stream.
        unsafe {
            check(
                sys::cuMemcpyDtoDAsync_v2(
                    self.pointers[destination] + offset as u64,
                    self.pointers[source] + source_offset as u64,
                    bytes,
                    self.session.stream,
                ),
                "copy tensor range",
            )?;
        }
        Ok(())
    }
    pub(crate) fn launch_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        self.submit_program(name, phase)?;
        self.sync()?;
        #[cfg(test)]
        if name == "decode"
            && phase == ExecutionPhase::Decode
            && self.cuda_graph.uses_graph(phase)
            && let Some(events) = self.live_graph_events.get()
        {
            let mut ms = 0.;
            // SAFETY: Both events were recorded on the synchronized stream and are retained by Session.
            unsafe {
                check(
                    sys::cuEventElapsedTime(&mut ms, events[0], events[1]),
                    "live graph elapsed",
                )?;
            }
            self.live_graph_ms.borrow_mut().push(ms);
        }
        Ok(())
    }

    /// Submit the same registered program without adding synchronization points.
    pub(crate) fn submit_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        // A declared one-row entry may reuse kernels whose default export has
        // a much larger capacity. Resolve its real row arguments before launch.
        if self
            .direct
            .as_ref()
            .and_then(|d| d.row_execution.as_ref())
            .is_some_and(|rows| rows.programs.contains(name) && rows.program_capacity(name) == 1)
        {
            return self.submit_program_rows(name, 1, phase);
        }
        self.ensure_kv(name)?;
        if let Some(operations) = self.direct.as_ref().and_then(|d| d.programs.get(name)) {
            for operation in operations {
                if let crate::model::Operation::Copy { source, bytes, .. } = operation {
                    self.ensure_workspace(source, *bytes)?;
                }
                match operation {
                    crate::model::Operation::Copy {
                        destination, bytes, ..
                    }
                    | crate::model::Operation::Zero { destination, bytes } => {
                        self.ensure_writable_range(destination, 0, *bytes)?;
                    }
                    _ => {}
                }
            }
        }
        if self.cuda_graph.uses_graph(phase) {
            // Capture resolves all functions once; replay retains its modules
            // and addresses. KV growth above invalidates captures before reuse.
            let graph = self.program_graph(name)?;
            #[cfg(test)]
            let events = (name == "decode" && phase == ExecutionPhase::Decode)
                .then(|| self.live_graph_events.get())
                .flatten();
            // SAFETY: Graphs and their stable addresses live in this thread's session.
            unsafe {
                #[cfg(test)]
                if let Some(events) = events {
                    check(
                        sys::cuEventRecord(events[0], self.session.stream),
                        "live graph begin",
                    )?;
                }
                check(
                    sys::cuGraphLaunch(graph, self.session.stream),
                    "model graph",
                )?;
                #[cfg(test)]
                if let Some(events) = events {
                    check(
                        sys::cuEventRecord(events[1], self.session.stream),
                        "live graph end",
                    )?;
                }
            }
            return Ok(());
        }
        self.ensure_program_kernels(name)?;
        if let Some(direct) = &self.direct {
            use crate::model::Operation;
            let ops = direct
                .programs
                .get(name)
                .ok_or_else(|| format!("Missing program {name}"))?;
            let mut kernels = direct.kernels.borrow_mut();
            for op in ops {
                match op {
                    Operation::Kernel { name } => {
                        let kernel = kernels.get_mut(name).ok_or("Unbound kernel")?;
                        // Bind only this submitted kernel's private arguments.
                        // Switching requests must not walk every kernel in every
                        // compiled profile; weights and workspace stay fixed.
                        for (argument, value) in kernel.spec.args.iter().zip(&mut kernel.values) {
                            let (name, offset) = match argument {
                                Argument::Buffer { name } => (name, 0),
                                Argument::BufferSlice { name, offset } => (name, *offset),
                                _ => continue,
                            };
                            if let Some(&address) =
                                self.sequences[self.active_sequence].addresses.get(name)
                            {
                                *value = Value::Pointer(address + offset as u64);
                            }
                        }
                        self.ensure_experts(&kernel.spec, &kernel.values)?;
                        launch_kernel(
                            &kernel.spec,
                            kernel
                                .function
                                .ok_or("Kernel was not resolved before submission")?,
                            &mut kernel.values,
                            &self.session.driver,
                            self.session.stream,
                        )?;
                    }
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    } => {
                        // SAFETY: The same validated ranges and session-owned
                        // allocations are used by the captured program above.
                        unsafe {
                            check(
                                sys::cuMemcpyDtoDAsync_v2(
                                    self.pointers[destination],
                                    self.pointers[source],
                                    *bytes,
                                    self.session.stream,
                                ),
                                "direct state copy",
                            )?;
                        }
                    }
                    Operation::Zero { destination, bytes } => {
                        // SAFETY: Validated writable range on the same stream as
                        // its producers and consumers, with no extra sync.
                        unsafe {
                            check(
                                sys::cuMemsetD8Async(
                                    self.pointers[destination],
                                    0,
                                    *bytes,
                                    self.session.stream,
                                ),
                                "direct residual reset",
                            )?;
                        }
                    }
                }
            }
            return Ok(());
        }
        Err(format!("Missing direct execution bindings for {name}"))
    }
    pub(super) fn ensure_kernel(&self, name: &str) -> Result<()> {
        let direct = self.direct.as_ref().ok_or("Missing direct bindings")?;
        let mut bindings = direct.kernels.borrow_mut();
        let kernel = bindings
            .get_mut(name)
            .ok_or_else(|| format!("Unknown kernel {name}"))?;
        if !kernel.workspaces_ready {
            for argument in &kernel.spec.args {
                if let Argument::Buffer { name } | Argument::BufferSlice { name, .. } = argument {
                    self.ensure_workspace(name, self.sizes[name])?;
                }
            }
            // Shared mappings are retained for this executor's lifetime.
            // Track readiness separately from module preload: a preloaded
            // function can still reference an untouched workspace reservation.
            kernel.workspaces_ready = true;
        }
        if kernel.function.is_none() {
            let at = Instant::now();
            kernel.function = Some(
                self.kernels
                    .borrow_mut()
                    .resolve(&self.session, &kernel.spec)?,
            );
            let mut statistics = self.batch_statistics.get();
            statistics.kernel_load_s += at.elapsed().as_secs_f64();
            self.batch_statistics.set(statistics);
        }
        Ok(())
    }
    pub(super) fn ensure_program_kernels(&self, name: &str) -> Result<()> {
        let operations = self
            .direct
            .as_ref()
            .ok_or("Missing direct bindings")?
            .programs
            .get(name)
            .ok_or_else(|| format!("Unknown program {name}"))?;
        for operation in operations {
            if let crate::model::Operation::Kernel { name } = operation {
                self.ensure_kernel(name)?;
            }
        }
        Ok(())
    }
    pub(super) fn ensure_invocation_kernels(
        &self,
        operations: &[crate::execution::Invocation],
    ) -> Result<()> {
        for invocation in operations {
            for view in invocation.views.values() {
                self.ensure_workspace(&view.buffer, self.sizes[&view.buffer])?;
            }
            if let crate::model::Operation::Kernel { name } = &invocation.operation {
                self.ensure_kernel(name)?;
            } else {
                let names: Vec<&str> = match &invocation.operation {
                    crate::model::Operation::Copy {
                        source,
                        destination,
                        ..
                    } => vec![source, destination],
                    crate::model::Operation::Zero { destination, .. } => vec![destination],
                    _ => vec![],
                };
                for name in names {
                    let name = invocation
                        .views
                        .get(name)
                        .map_or(name, |v| v.buffer.as_str());
                    self.ensure_workspace(name, self.sizes[name])?;
                }
            }
        }
        Ok(())
    }

    pub(super) fn ensure_kv(&self, program: &str) -> Result<()> {
        let Some(growth) = self.growth.get(program) else {
            return Ok(());
        };
        if self.session.virtual_buffers.borrow().is_empty() {
            return Ok(());
        }
        let (position, length) = match &growth.length {
            Some(name) => {
                let [position, length] = self.read_control_group([&growth.position, name])?;
                (position, Some(length))
            }
            None => (self.read_control(&growth.position)?, None),
        };
        let position = usize::try_from(position).map_err(|_| "Negative KV write position")?;
        let count = match length {
            Some(length) => usize::try_from(length)
                .ok()
                .filter(|&n| n > 0 && n <= growth.tokens)
                .ok_or("Invalid logical KV write length")?,
            None => growth.tokens,
        };
        self.ensure_kv_extent(program, position, count)
    }
    /// The request state machine knows the committed write interval. Batch
    /// preparation can use it without downloading the same device position for
    /// every owner; execution still validates the resulting device position.
    pub(super) fn ensure_kv_extent(
        &self,
        program: &str,
        position: usize,
        count: usize,
    ) -> Result<()> {
        let Some(growth) = self.growth.get(program) else {
            return Ok(());
        };
        if count == 0
            || count > growth.tokens
            || (growth.length.is_none() && count != growth.tokens)
        {
            return Err("Invalid logical KV write length".into());
        }
        if self.session.virtual_buffers.borrow().is_empty() {
            return Ok(());
        }
        let tokens = position.checked_add(count).ok_or("KV position overflow")?;
        let mut buffers = self.session.virtual_buffers.borrow_mut();
        let mut additional = 0usize;
        for name in &growth.buffers {
            let b = buffers.get(name).ok_or("Missing demand KV buffer")?;
            let start = position.checked_mul(b.stride).ok_or("KV write overflow")?;
            let bytes = count.checked_mul(b.stride).ok_or("KV write overflow")?;
            additional = additional
                .checked_add(b.extent(tokens)?.saturating_sub(b.mapped))
                .and_then(|n| n.checked_add(b.writable_bytes(start, bytes).ok()?))
                .ok_or("KV growth sum overflow")?;
        }
        if additional == 0 {
            return Ok(());
        }
        // Physical mapping/COW changes require completed prior readers, even
        // when the caller supplied host positions and retained the active arena.
        self.sync()?;
        let (mut free, mut total) = (0, 0);
        // SAFETY: Live context; driver writes two scalar extents.
        unsafe {
            check(
                sys::cuMemGetInfo_v2(&mut free, &mut total),
                "KV available memory",
            )?;
        }
        if additional > free {
            return Err(format!("KV growth needs {additional} bytes, free {free}"));
        }
        for name in &growth.buffers {
            let buffer = buffers.get_mut(name).ok_or("Missing KV reservation")?;
            buffer.make_writable(
                &self.session,
                position
                    .checked_mul(buffer.stride)
                    .ok_or("KV write overflow")?,
                count
                    .checked_mul(buffer.stride)
                    .ok_or("KV write overflow")?,
            )?;
            buffer.grow(&self.session, tokens)?;
        }
        let scratch = buffers
            .iter()
            .filter(|(n, _)| self.prefill_workspace.contains(*n))
            .map(|(_, b)| b.mapped)
            .sum::<usize>();
        self.peak_prefill_workspace_bytes
            .set(self.peak_prefill_workspace_bytes.get().max(scratch));
        let resident = buffers.values().map(|b| b.physical_bytes()).sum::<usize>() - scratch;
        self.peak_kv_bytes
            .set(self.peak_kv_bytes.get().max(resident));
        self.sync()
    }
    pub(super) fn ensure_writable_range(
        &self,
        name: &str,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let end = offset
            .checked_add(bytes)
            .ok_or("Workspace range overflow")?;
        self.ensure_workspace(name, end)?;
        let mut buffers = self.session.virtual_buffers.borrow_mut();
        if let Some(buffer) = buffers.get_mut(name)
            && buffer.writable_bytes(offset, bytes)? != 0
        {
            self.sync()?;
            buffer.make_writable(&self.session, offset, bytes)?;
        }
        Ok(())
    }

    /// Map only before submission/capture. VA is immutable and earlier slabs
    /// are retained, so growing another row cannot invalidate existing graphs.
    pub(crate) fn ensure_workspace(&self, name: &str, bytes: usize) -> Result<()> {
        if bytes > self.sizes.get(name).copied().unwrap_or(0) {
            return Err("Workspace mapping exceeds buffer".into());
        }
        let mut buffers = self.session.workspace_buffers.borrow_mut();
        if let Some(buffer) = buffers.get_mut(name)
            && bytes > buffer.mapped
        {
            let at = Instant::now();
            let before = buffer.physical_bytes();
            self.sync()?;
            buffer.grow_bytes(&self.session, bytes)?;
            let mut stats = self.batch_statistics.get();
            stats.workspace_mappings += 1;
            stats.workspace_mapped_bytes += buffer.physical_bytes() - before;
            stats.workspace_mapping_s += at.elapsed().as_secs_f64();
            self.batch_statistics.set(stats);
        }
        Ok(())
    }

    pub(crate) fn pending_workspace_bytes(&self, programs: &[&str]) -> Result<usize> {
        let names: std::collections::BTreeSet<_> = programs
            .iter()
            .flat_map(|p| self.workspace_dependencies.get(*p).into_iter().flatten())
            .collect();
        names.into_iter().try_fold(0usize, |sum, name| {
            sum.checked_add(self.pending_workspace_range(name, self.sizes[name])?)
                .ok_or_else(|| "Pending workspace sum overflow".into())
        })
    }

    pub(crate) fn pending_workspace_range(&self, name: &str, bytes: usize) -> Result<usize> {
        if bytes > self.sizes.get(name).copied().unwrap_or(0) {
            return Err("Pending workspace exceeds buffer".into());
        }
        self.session
            .workspace_buffers
            .borrow()
            .get(name)
            .map_or(Ok(0), |buffer| {
                Ok(buffer.required_extent(bytes)?.saturating_sub(buffer.mapped))
            })
    }

    pub(crate) fn pending_text_workspace_bytes(&self) -> Result<usize> {
        self.text_workspaces.iter().try_fold(0usize, |sum, name| {
            sum.checked_add(self.pending_workspace_range(name, self.sizes[name])?)
                .ok_or_else(|| "Pending text workspace sum overflow".into())
        })
    }
    pub(crate) fn upload_ids(&self, name: &str, ids: &[u32]) -> Result<()> {
        let bytes = ids
            .len()
            .checked_mul(4)
            .ok_or("Token upload size overflow")?;
        // SAFETY: All bytes of the borrowed u32 slice are initialized and remain
        // live until upload_bytes has copied them into session-owned staging.
        self.upload_bytes(name, unsafe {
            std::slice::from_raw_parts(ids.as_ptr().cast(), bytes)
        })
    }
    pub(crate) fn read_control(&self, name: &str) -> Result<i32> {
        self.read_control_group([name]).map(|values| values[0])
    }
    pub(crate) fn read_control_group<const N: usize>(&self, names: [&str; N]) -> Result<[i32; N]> {
        let mut regions = [(0, 4); N];
        for (region, name) in regions.iter_mut().zip(names) {
            if self.sizes.get(name).is_none_or(|&size| size < 4) {
                return Err("Control exceeds allocation".into());
            }
            self.ensure_workspace(name, 4)?;
            region.0 = self.pointers[name];
        }
        let mut values = [0i32; N];
        // SAFETY: The initialized output array covers exactly N four-byte controls;
        // readback waits for the stream before copying into this borrowed output.
        let raw = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), N * 4) };
        self.session
            .transfers
            .borrow_mut()
            .readback(self.session.stream, &regions, raw)?;
        Ok(values)
    }
    pub(crate) fn upload_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        self.upload_bytes_at(name, 0, bytes)
    }
    pub(crate) fn upload_bytes_at(&self, name: &str, offset: usize, bytes: &[u8]) -> Result<()> {
        let size = self.sizes.get(name).ok_or("Missing input buffer")?;
        if offset
            .checked_add(bytes.len())
            .is_none_or(|end| end > *size)
        {
            return Err("Input upload exceeds buffer".into());
        }
        self.ensure_writable_range(name, offset, bytes.len())?;
        self.session.transfers.borrow_mut().upload(
            self.session.stream,
            self.pointers[name] + offset as u64,
            bytes,
        )
    }
    pub(crate) fn read_controls(&self, name: &str, count: usize) -> Result<Vec<u32>> {
        let bytes = count.checked_mul(4).ok_or("Control vector size overflow")?;
        if bytes > self.sizes[name] {
            return Err("Control vector exceeds allocation".into());
        }
        self.ensure_workspace(name, bytes)?;
        let mut values = vec![0u32; count];
        if bytes <= super::transfers::READBACK_BYTES {
            // SAFETY: The initialized u32 destination covers the checked byte
            // extent, and readback completes before returning this allocation.
            let raw = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), bytes) };
            self.session.transfers.borrow_mut().readback(
                self.session.stream,
                &[(self.pointers[name], bytes)],
                raw,
            )?;
            return Ok(values);
        }
        self.sync()?;
        // SAFETY: Producer is synchronized; both extents are checked above.
        unsafe {
            check(
                sys::cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), self.pointers[name], bytes),
                "MTP controls",
            )?;
        }
        Ok(values)
    }
}
