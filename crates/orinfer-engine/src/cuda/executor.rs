use super::*;
use crate::execution::{CudaGraphMode, ExecutionPhase, LoadOptions};
use std::cell::RefCell;
pub(super) type BatchGraphCache = BTreeMap<Vec<(usize, usize)>, (Handle, u64, usize, u64)>;

pub(super) struct DirectKernel {
    pub(super) spec: Kernel,
    pub(super) function: Handle,
    pub(super) values: Vec<Value>,
}

pub(super) struct DirectPrograms {
    pub(super) kernels: RefCell<BTreeMap<String, DirectKernel>>,
    pub(super) programs: BTreeMap<String, Vec<crate::model::Operation>>,
}

/// Execute an architecture adapter's explicit programs with stable allocations.
/// All handles, pointers and argument backing storage stay inside this session.
/// A CUDA model is thread-affine. Construct, use and drop it on its worker.
pub(crate) struct Executor {
    pub(crate) session: Session,
    pub(crate) pointers: BTreeMap<String, u64>,
    pub(crate) sizes: BTreeMap<String, usize>,
    pub(crate) graphs: RefCell<BTreeMap<String, Handle>>,
    pub(super) direct: Option<DirectPrograms>,
    pub(super) cuda_graph: CudaGraphMode,
    growth: BTreeMap<String, crate::model::KvGrowth>,
    pub(crate) peak_kv_bytes: std::cell::Cell<usize>,
    pub(crate) peak_prefill_workspace_bytes: std::cell::Cell<usize>,
    pub(super) prefill_workspace: std::collections::BTreeSet<String>,
    pub(crate) allocations: Allocations,
    pub(crate) snapshot_allocations: Vec<std::rc::Rc<super::snapshot::Allocation>>,
    pub(super) sequences: Vec<super::sequence::Sequence>,
    pub(super) active_sequence: usize,
    pub(super) sequence_specs: Vec<crate::artifact::Buffer<crate::weights::TensorIdentity>>,
    pub(super) sequence_strides: BTreeMap<String, usize>,
    pub(super) batch_graphs: RefCell<BatchGraphCache>,
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
pub(crate) struct LoadStats {
    pub(crate) manifest_sha256: String,
    pub(crate) device: DeviceInfo,
    pub(crate) load_to_ready_s: f64,
    pub(crate) weight_io_hash_s: f64,
    pub(crate) weight_upload_s: f64,
    pub(crate) module_load_bind_s: f64,
    pub(crate) graph_capture_s: f64,
    pub(crate) cuda_graph: CudaGraphMode,
    pub(crate) captured_programs: Vec<String>,
    pub(crate) buffer_bytes: usize,
    pub(crate) buffer_capacity_bytes: usize,
}
impl Executor {
    pub(crate) fn load(
        manifest: &crate::model::Manifest,
        base: &std::path::Path,
        kernel_base: &std::path::Path,
        fingerprint: String,
        scopes: &BTreeMap<String, crate::loader::BufferScope>,
        decode_programs: &std::collections::BTreeSet<String>,
        options: LoadOptions,
    ) -> Result<(Self, LoadStats)> {
        let cuda_graph = options.cuda_graph;
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
        let buffer_bytes = capacity_bytes - lazy_bytes - unused_weights;
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
            check((s.driver.init)(0), "cuInit model")?;
            check((s.driver.device)(&mut s.device, 0), "model device")?;
            check((s.driver.attribute)(&mut major, 75, s.device), "SM major")?;
            check((s.driver.attribute)(&mut minor, 76, s.device), "SM minor")?;
            if [major, minor] != [8, 7] {
                return Err("Model requires SM87".into());
            }
            check((s.driver.version)(&mut version), "driver version")?;
            check(
                (s.driver.name)(name.as_mut_ptr().cast(), 256, s.device),
                "device name",
            )?;
            check((s.driver.total)(&mut total, s.device), "total memory")?;
            check((s.driver.current)(&mut s.previous), "previous context")?;
            let mut context = ptr::null_mut();
            check((s.driver.retain)(&mut context, s.device), "retain context")?;
            s.retained = true;
            check((s.driver.set_current)(context), "current context")?;
            check((s.driver.stream_create)(&mut s.stream, 1), "model stream")?;
            let (mut free, mut memtotal) = (0usize, 0usize);
            check(
                (s.driver.memory_info)(&mut free, &mut memtotal),
                "available memory",
            )?;
            if buffer_bytes > free {
                return Err(format!("Model needs {buffer_bytes} bytes, free {free}"));
            }
        }
        let (mut weight_io_hash_s, mut weight_upload_s) = (0.0, 0.0);
        let mut weights = crate::weights::Weights::open(base)?;
        for b in &manifest.buffers {
            if b.data.is_some()
                && b.access == crate::artifact::Access::Read
                && !used.contains(b.name.as_str())
            {
                continue;
            }
            let bytes = b.bytes()?;
            let mut address = 0;
            // SAFETY: Allocation length is checked by manifest validation and owned
            // by Session before any fallible operation can return.
            let stride = lazy
                .and_then(|kv| {
                    kv.buffers
                        .get(&b.name)
                        .or_else(|| kv.prefill_workspace.get(&b.name))
                })
                .copied();
            if let Some(stride) = stride {
                let reservation = super::virtual_memory::Reservation::reserve(&s, bytes, stride)?;
                address = reservation.address;
                s.virtual_buffers
                    .borrow_mut()
                    .insert(b.name.clone(), reservation);
            } else {
                // SAFETY: Validated extent, immediately transferred to Session.
                unsafe {
                    check(
                        (s.driver.alloc)(&mut address, bytes),
                        &format!("allocate {}", b.name),
                    )?;
                }
                s.buffers.push(address);
            }
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
            arena.bytes += bytes;
            if b.data.is_some() {
                let t = Instant::now();
                let raw = weights.read(b)?;
                weight_io_hash_s += t.elapsed().as_secs_f64();
                if raw.len() != bytes {
                    return Err(format!("{} weight byte length mismatch", b.name));
                }
                let t = Instant::now();
                // SAFETY: Both ranges are valid for bytes and raw lives until the
                // synchronous transfer finishes. No online Python is involved.
                unsafe {
                    check(
                        (s.driver.upload)(address, raw.as_ptr().cast(), bytes),
                        &format!("upload {}", b.name),
                    )?;
                }
                weight_upload_s += t.elapsed().as_secs_f64();
            } else if stride.is_none() {
                // SAFETY: Whole allocation is owned and not in use yet.
                unsafe {
                    check(
                        (s.driver.memset)(address, 0, bytes, s.stream),
                        "initialize workspace",
                    )?;
                }
            }
        }
        drop(weights);
        let module_started = Instant::now();
        let mut modules = BTreeMap::<String, Handle>::new();
        let mut functions = BTreeMap::<(String, String), Handle>::new();
        let mut launches = BTreeMap::new();
        for k in &manifest.kernels {
            let module = if let Some(m) = modules.get(&k.module.file) {
                *m
            } else {
                let image = crate::artifact::read_identity(kernel_base, &k.module)?;
                crate::artifact::read_identity(kernel_base, &k.source)?;
                crate::artifact::read_identity(kernel_base, &k.host_abi)?;
                if !image.starts_with(b"\x7fELF") {
                    return Err("Expected cubin ELF".into());
                }
                let mut module = ptr::null_mut();
                // SAFETY: image contains a hash-verified cubin for the checked SM.
                // Driver copies module data; session owns the resulting handle.
                unsafe {
                    check(
                        (s.driver.module_load)(&mut module, image.as_ptr().cast()),
                        &format!("load {}", k.name),
                    )?;
                }
                s.modules.push(module);
                modules.insert(k.module.file.clone(), module);
                module
            };
            let key = (k.module.file.clone(), k.symbol.clone());
            let function = if let Some(f) = functions.get(&key) {
                *f
            } else {
                let mut function = ptr::null_mut();
                let symbol = CString::new(k.symbol.as_str()).map_err(|e| e.to_string())?;
                // SAFETY: Module remains live and symbol is a terminated C string.
                unsafe {
                    check(
                        (s.driver.function)(&mut function, module, symbol.as_ptr()),
                        "model function",
                    )?;
                }
                functions.insert(key, function);
                function
            };
            // SAFETY: Actual function/device bounds are checked before each launch
            // binding. Opt-in shared memory is set on the same live function.
            unsafe {
                let (mut threads, mut static_shared, mut max_shared, mut default_shared) =
                    (0, 0, 0, 0);
                check(
                    (s.driver.function_get)(&mut threads, 0, function),
                    "function threads",
                )?;
                check(
                    (s.driver.function_get)(&mut static_shared, 1, function),
                    "static shared memory",
                )?;
                check(
                    (s.driver.attribute)(&mut max_shared, 97, s.device),
                    "shared memory limit",
                )?;
                check(
                    (s.driver.attribute)(&mut default_shared, 8, s.device),
                    "default shared memory",
                )?;
                let shared = u64::from(k.shared_memory_bytes) + static_shared as u64;
                if k.block.iter().product::<u32>() > threads as u32 || shared > max_shared as u64 {
                    return Err(format!("{} resource limit", k.name));
                }
                for axis in 0..3 {
                    let (mut block, mut grid) = (0, 0);
                    check(
                        (s.driver.attribute)(&mut block, 2 + axis as i32, s.device),
                        "block limit",
                    )?;
                    check(
                        (s.driver.attribute)(&mut grid, 5 + axis as i32, s.device),
                        "grid limit",
                    )?;
                    if k.block[axis] > block as u32 || k.grid[axis] > grid as u32 {
                        return Err("Grid/block limit".into());
                    }
                }
                if shared > default_shared as u64 {
                    check(
                        (s.driver.function_set)(function, 8, k.shared_memory_bytes as i32),
                        "opt-in shared memory",
                    )?;
                }
            }
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
                Launch {
                    spec: k,
                    function,
                    values,
                },
            );
        }
        let module_load_bind_s = module_started.elapsed().as_secs_f64();
        // SAFETY: Flush all default-stream uploads and workspace initialization
        // before capture on a nonblocking stream. No allocation occurs in graphs.
        unsafe {
            check((s.driver.context_sync)(), "model uploads complete")?;
        }
        // Request limits can replace initial private buffers on the first lease.
        // Capture only when a program is submitted with its final arena bindings;
        // this also avoids retaining unused MTP and prefill graphs at startup.
        let graphs = BTreeMap::new();
        let graph_capture_s = 0.0;
        let direct = Some(DirectPrograms {
            kernels: RefCell::new(
                launches
                    .into_iter()
                    .map(|(name, launch)| {
                        (
                            name,
                            DirectKernel {
                                spec: launch.spec.clone(),
                                function: launch.function,
                                values: launch.values,
                            },
                        )
                    })
                    .collect(),
            ),
            programs: manifest.programs.clone(),
        });
        let load_to_ready_s = started.elapsed().as_secs_f64();
        let captured_programs: Vec<String> = graphs.keys().cloned().collect();
        eprintln!(
            "MODEL READY after {load_to_ready_s:.3}s; {buffer_bytes} resident buffer bytes, {capacity_bytes} capacity bytes; cuda_graph={cuda_graph}, {} captured programs",
            captured_programs.len()
        );
        let device = DeviceInfo {
            name: String::from_utf8_lossy(&name)
                .trim_end_matches('\0')
                .to_string(),
            sm: [major, minor],
            driver_version: version,
            total_bytes: total,
        };
        let executor = Self {
            snapshot_allocations: vec![],
            session: s,
            pointers,
            sizes,
            graphs: RefCell::new(graphs),
            direct,
            cuda_graph,
            growth: manifest
                .kv_cache
                .as_ref()
                .map(|kv| kv.growth.clone())
                .unwrap_or_default(),
            allocations,
            peak_kv_bytes: std::cell::Cell::new(0),
            peak_prefill_workspace_bytes: std::cell::Cell::new(0),
            prefill_workspace: manifest
                .kv_cache
                .as_ref()
                .map(|kv| kv.prefill_workspace.keys().cloned().collect())
                .unwrap_or_default(),
            sequences: vec![super::sequence::Sequence::default()],
            active_sequence: 0,
            sequence_specs: manifest
                .buffers
                .iter()
                .filter(|b| scopes[&b.name] == crate::loader::BufferScope::Sequence)
                .cloned()
                .collect(),
            sequence_strides: lazy.map(|kv| kv.buffers.clone()).unwrap_or_default(),
            batch_graphs: Default::default(),
            batch_statistics: Default::default(),
            batch_graph_clock: std::cell::Cell::new(0),
        };
        let stats = LoadStats {
            manifest_sha256: fingerprint,
            device,
            load_to_ready_s,
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
        Ok((executor, stats))
    }
    pub(crate) fn sync(&self) -> Result<()> {
        // SAFETY: This thread owns the live stream and all graph allocations.
        unsafe {
            check(
                (self.session.driver.stream_sync)(self.session.stream),
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
        self.sync()?;
        // Keep at most one allocation granule per KV buffer for short requests.
        // Clear retained pages before reuse; larger contexts release their slabs.
        for (name, buffer) in self.session.virtual_buffers.borrow_mut().iter_mut() {
            if self.sequence_strides.contains_key(name) {
                if reuse && buffer.mapped != 0 && buffer.mapped <= buffer.granularity {
                    // SAFETY: Completed stream, owned mapping and full mapped range.
                    unsafe {
                        check(
                            (self.session.driver.memset)(
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
                    (self.session.driver.memset)(address, 0, self.sizes[name], self.session.stream),
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
                (self.session.driver.download)(raw.as_mut_ptr().cast(), self.pointers[name], bytes),
                "download tensor",
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
        if bytes > *self.sizes.get(source).ok_or("Unknown copy source")?
            || offset
                .checked_add(bytes)
                .is_none_or(|n| n > self.sizes.get(destination).copied().unwrap_or(0))
        {
            return Err("Copy exceeds buffer".into());
        }
        // SAFETY: The validated ranges refer to live allocations owned by this
        // executor and the transfer is ordered with graph work on the same stream.
        unsafe {
            check(
                (self.session.driver.copy)(
                    self.pointers[destination] + offset as u64,
                    self.pointers[source],
                    bytes,
                    self.session.stream,
                ),
                "copy tensor range",
            )?;
        }
        self.sync()
    }
    pub(crate) fn launch_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        self.submit_program(name, phase)?;
        self.sync()
    }

    /// Submit the same registered program without adding synchronization points.
    pub(crate) fn submit_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        self.ensure_kv(name)?;
        if self.cuda_graph.uses_graph(phase) {
            let graph = self.program_graph(name)?;
            // SAFETY: Graphs and their stable addresses live in this thread's session.
            unsafe {
                check(
                    (self.session.driver.graph_launch)(graph, self.session.stream),
                    "model graph",
                )?;
            }
            return Ok(());
        }
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
                        launch_kernel(
                            &kernel.spec,
                            kernel.function,
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
                                (self.session.driver.copy)(
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
                                (self.session.driver.memset)(
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
    pub(super) fn ensure_kv(&self, program: &str) -> Result<()> {
        let Some(growth) = self.growth.get(program) else {
            return Ok(());
        };
        if self.session.virtual_buffers.borrow().is_empty() {
            return Ok(());
        }
        self.sync()?;
        let position = usize::try_from(self.read_control(&growth.position)?)
            .map_err(|_| "Negative KV write position")?;
        let tokens = position
            .checked_add(growth.tokens)
            .ok_or("KV position overflow")?;
        let mut buffers = self.session.virtual_buffers.borrow_mut();
        let mut additional = 0usize;
        for name in &growth.buffers {
            let b = buffers.get(name).ok_or("Missing demand KV buffer")?;
            additional = additional
                .checked_add(b.extent(tokens)?.saturating_sub(b.mapped))
                .ok_or("KV growth sum overflow")?;
        }
        if additional == 0 {
            return Ok(());
        }
        let (mut free, mut total) = (0, 0);
        // SAFETY: Live context; driver writes two scalar extents.
        unsafe {
            check(
                (self.session.driver.memory_info)(&mut free, &mut total),
                "KV available memory",
            )?;
        }
        if additional > free {
            return Err(format!("KV growth needs {additional} bytes, free {free}"));
        }
        for name in &growth.buffers {
            buffers
                .get_mut(name)
                .ok_or("Missing KV reservation")?
                .grow(&self.session, tokens)?;
        }
        let scratch = buffers
            .iter()
            .filter(|(n, _)| self.prefill_workspace.contains(*n))
            .map(|(_, b)| b.mapped)
            .sum::<usize>();
        self.peak_prefill_workspace_bytes
            .set(self.peak_prefill_workspace_bytes.get().max(scratch));
        let resident = buffers.values().map(|b| b.mapped).sum::<usize>() - scratch;
        self.peak_kv_bytes
            .set(self.peak_kv_bytes.get().max(resident));
        self.sync()
    }
    pub(crate) fn upload_ids(&self, name: &str, ids: &[u32]) -> Result<()> {
        if ids.len() * 4 > self.sizes[name] {
            return Err("Token upload exceeds buffer".into());
        }
        self.sync()?;
        // SAFETY: Synchronous copy borrows live IDs; validated allocation covers it.
        unsafe {
            check(
                (self.session.driver.upload)(
                    self.pointers[name],
                    ids.as_ptr().cast(),
                    ids.len() * 4,
                ),
                "model IDs",
            )?;
            check(
                (self.session.driver.context_sync)(),
                "model upload dependency",
            )
        }
    }
    pub(crate) fn read_control(&self, name: &str) -> Result<i32> {
        let mut value = 0i32;
        // SAFETY: Producer is synchronized, the control and destination cover 4 bytes.
        unsafe {
            check(
                (self.session.driver.download)(
                    (&mut value as *mut i32).cast(),
                    self.pointers[name],
                    4,
                ),
                "model control",
            )?;
        }
        Ok(value)
    }
    pub(crate) fn upload_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let size = self.sizes.get(name).ok_or("Missing image buffer")?;
        if bytes.len() > *size {
            return Err("Image upload exceeds buffer".into());
        }
        self.sync()?;
        // SAFETY: The synchronized destination is live and covers the borrowed bytes.
        unsafe {
            check(
                (self.session.driver.upload)(
                    self.pointers[name],
                    bytes.as_ptr().cast(),
                    bytes.len(),
                ),
                "image upload",
            )?;
            check(
                (self.session.driver.context_sync)(),
                "image upload dependency",
            )
        }
    }
    pub(crate) fn read_controls(&self, name: &str, count: usize) -> Result<Vec<u32>> {
        let bytes = count.checked_mul(4).ok_or("Control vector size overflow")?;
        if bytes > self.sizes[name] {
            return Err("Control vector exceeds allocation".into());
        }
        let mut values = vec![0u32; count];
        // SAFETY: Producer is synchronized; both extents are checked above.
        unsafe {
            check(
                (self.session.driver.download)(
                    values.as_mut_ptr().cast(),
                    self.pointers[name],
                    bytes,
                ),
                "MTP controls",
            )?;
        }
        Ok(values)
    }
}
