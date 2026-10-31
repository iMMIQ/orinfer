//! ABI-checked model libraries. The engine owns devices; packages own model plans.
use crate::{
    artifact::{FileIdentity, Result, read_identity, resolve_file},
    execution::{BufferView, Invocation},
    model::Operation,
};
use libloading::Library;
use orinfer_model_sdk::{
    abi::{self, BatchOutput, Bytes, CreateRequest, CreatedPlan, OwnedBytes},
    architecture::BatchSegment,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ffi::c_void, path::Path, ptr, slice};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionLibrary {
    pub abi_version: u32,
    pub library: FileIdentity,
    pub package: String,
    pub version: String,
}

/// Inspect an explicitly selected native library without creating a model or CUDA context.
pub(crate) fn inspect_library(path: &Path) -> Result<abi::PackageInfo> {
    let image = library_header(path)?;
    ModelPackage::open(path, &image)?.describe()
}

pub(crate) fn library_header(path: &Path) -> Result<[u8; 20]> {
    use std::io::Read;
    let mut image = [0; 20];
    std::fs::File::open(path)
        .and_then(|mut file| file.read_exact(&mut image))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(image)
}

pub(crate) fn validate_library(image: &[u8], host_arch: &str) -> Result<()> {
    let machine = match host_arch {
        "aarch64" => 183,
        "x86_64" => 62,
        _ => {
            return Err(format!(
                "Unsupported model-library host architecture {host_arch}"
            ));
        }
    };
    if image.len() < 20
        || &image[..4] != b"\x7fELF"
        || image[4] != 2
        || image[5] != 1
        || u16::from_le_bytes([image[16], image[17]]) != 3
        || u16::from_le_bytes([image[18], image[19]]) != machine
    {
        return Err(format!(
            "Model library must be an {host_arch} ELF shared object"
        ));
    }
    Ok(())
}

#[repr(C)]
struct Header {
    version: u32,
    size: usize,
}

type BindingCache = std::cell::RefCell<
    std::collections::VecDeque<(Vec<(usize, usize)>, std::rc::Rc<abi::StateBindings>)>,
>;
type CachedBatch = (
    std::rc::Rc<Vec<Invocation>>,
    std::rc::Rc<abi::StateBindings>,
);
type PlanEntries =
    std::collections::VecDeque<(abi::BatchOperation, Vec<(usize, usize)>, CachedBatch, usize)>;
type PlanCache = std::cell::RefCell<PlanEntries>;

pub(crate) struct ModelPackage {
    handle: *mut c_void,
    api: *const abi::Api,
    input_api: Option<*const abi::inputs::Api>,
    pub(crate) plan_contracts: std::cell::RefCell<std::collections::BTreeSet<String>>,
    // Retain code until the opaque model and all of its buffers are destroyed.
    _library: Library,
    bindings: BindingCache,
    batch_plans: PlanCache,
    batch_capabilities:
        std::cell::RefCell<std::collections::VecDeque<(abi::BatchOperation, Vec<usize>, bool)>>,
    kernel_buffers: BTreeMap<String, Vec<String>>,
    indirect_packets: bool,
}
impl ModelPackage {
    fn open(path: &Path, image: &[u8]) -> Result<Self> {
        validate_library(image, std::env::consts::ARCH)?;
        // SAFETY: this explicitly selected native library implements the documented
        // C ABI. Content verification is performed by validate-model.
        let library = unsafe { Library::new(path) }.map_err(|e| format!("Model library: {e}"))?;
        // SAFETY: symbol is the required versioned C entry point, not a Rust ABI.
        let entry =
            unsafe { library.get::<unsafe extern "C" fn(u32) -> *const abi::Api>(abi::ENTRYPOINT) }
                .map_err(|e| format!("Model entry point: {e}"))?;
        // SAFETY: the package retains its static function table for the library lifetime.
        let api = unsafe { entry(abi::ABI_VERSION) };
        if api.is_null() {
            return Err("Model library rejects engine ABI".into());
        }
        // SAFETY: every ABI entry returns at least this C prefix; larger tables
        // are read only after the declared size is checked.
        let header = unsafe { &*api.cast::<Header>() };
        if header.version != abi::ABI_VERSION || header.size != std::mem::size_of::<abi::Api>() {
            return Err("Incompatible model library function table".into());
        }
        Ok(Self {
            handle: ptr::null_mut(),
            api,
            input_api: None,
            plan_contracts: Default::default(),
            _library: library,
            bindings: Default::default(),
            batch_plans: Default::default(),
            batch_capabilities: Default::default(),
            kernel_buffers: Default::default(),
            indirect_packets: false,
        })
    }
    fn describe(&self) -> Result<abi::PackageInfo> {
        let mut description = OwnedBytes::default();
        // SAFETY: self retains the checked table and a live writable output slot.
        let status = unsafe { (self.table().describe)(&mut description) };
        let description = self.consume_bytes(description)?;
        if status != 0 {
            return Err("Model package cannot describe capabilities".into());
        }
        serde_json::from_slice(&description).map_err(|e| e.to_string())
    }
    pub(crate) fn create(
        base: &Path,
        spec: &ExecutionLibrary,
        request: CreateRequest,
    ) -> Result<(Self, CreatedPlan)> {
        orinfer_device::Target::from_package(&request.metadata.target)?
            .require_host(std::env::consts::ARCH)?;
        if spec.library.tensor.is_some() {
            return Err("Native execution library must be a standalone file".into());
        }
        if spec.abi_version != abi::ABI_VERSION
            || spec.package.is_empty()
            || spec.version.is_empty()
        {
            return Err("Incompatible model package ABI or identity".into());
        }
        let path = resolve_file(base, &spec.library.file)?;
        let image = if request.verify_weights {
            read_identity(base, &spec.library)?
        } else {
            library_header(&path)?.to_vec()
        };
        let mut model = Self::open(&path, &image)?;
        // Program preparation also supplies dynamic control lengths; it does
        // not require model-owned CPU weight assets.
        // SAFETY: explicitly selected, host-checked library; the optional symbol has a versioned C ABI.
        let entry = unsafe {
            model
                ._library
                .get::<unsafe extern "C" fn(u32) -> *const abi::inputs::Api>(
                    abi::inputs::ENTRYPOINT,
                )
        };
        let input_api = match entry {
            Ok(entry) => {
                // SAFETY: the static function table is retained by the library.
                let table = unsafe { entry(1) };
                if table.is_null() {
                    return Err("Model rejects input ABI".into());
                }
                // SAFETY: ABI v1 supplies the version and size prefix.
                let header = unsafe { &*table.cast::<Header>() };
                if header.version != 1 || header.size != std::mem::size_of::<abi::inputs::Api>() {
                    return Err("Incompatible model input function table".into());
                }
                Some(table)
            }
            Err(error) if request.metadata.input_assets.is_some() => {
                return Err(format!("Model input adapter: {error}"));
            }
            Err(_) => None,
        };
        let input = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
        model.input_api = input_api;
        let description = model.describe()?;
        if description.package != spec.package
            || description.version != spec.version
            || description.target != request.metadata.target
            || description.runtime_abi != abi::RUNTIME_ABI
            || !description.architectures.contains(&request.architecture)
            || !description
                .compute_policies
                .contains(&request.compute_policy)
        {
            return Err(
                "Model library identity, target or compute capabilities differ from package".into(),
            );
        }
        let mut output = OwnedBytes::default();
        // SAFETY: input and writable slots live until the synchronous call returns.
        let status = unsafe {
            (model.table().create)(input.as_ptr(), input.len(), &mut model.handle, &mut output)
        };
        let bytes = model.consume_bytes(output)?;
        if status != 0 {
            return Err(format!(
                "Model package: {}",
                String::from_utf8_lossy(&bytes)
            ));
        }
        if model.handle.is_null() {
            return Err("Model package returned a null handle".into());
        }
        let mut plan: CreatedPlan =
            serde_json::from_slice(&bytes).map_err(|e| format!("Model plan: {e}"))?;
        if let Some(catalog) = &request.catalog {
            model.export_execution(catalog, &mut plan.metadata)?;
        }
        model.indirect_packets = plan.metadata.packet_table().is_some();
        for kernel in &plan.metadata.kernels {
            let mut buffers: Vec<_> = kernel
                .args
                .iter()
                .filter_map(|a| match a {
                    crate::artifact::Argument::Buffer { name }
                    | crate::artifact::Argument::BufferSlice { name, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect();
            buffers.sort_unstable();
            buffers.dedup();
            model.kernel_buffers.insert(kernel.name.clone(), buffers);
        }
        Ok((model, plan))
    }
    fn export_execution(
        &self,
        catalog: &orinfer_model_sdk::catalog::Catalog,
        metadata: &mut crate::model::Manifest,
    ) -> Result<()> {
        use crate::artifact::Argument;
        if !metadata.kernels.is_empty()
            || !metadata.programs.is_empty()
            || !metadata.dynamic_batch_kernels.is_empty()
        {
            return Err(
                "Native catalog packages must export bulk bindings/programs through C arrays"
                    .into(),
            );
        }
        // SAFETY: the explicitly selected, host-checked library provides this versioned C entry.
        let entry = unsafe {
            self._library
                .get::<unsafe extern "C" fn(u32) -> *const abi::plans::Api>(abi::plans::ENTRYPOINT)
        }
        .map_err(|e| format!("Native plan adapter: {e}"))?;
        // SAFETY: the entry returns a static table retained by self._library.
        let api = unsafe { entry(1) };
        if api.is_null() {
            return Err("Model rejects plan ABI".into());
        }
        // SAFETY: every extension table begins with its version/size header.
        let header = unsafe { &*api.cast::<Header>() };
        if header.version != 1 || header.size != std::mem::size_of::<abi::plans::Api>() {
            return Err("Incompatible model plan function table".into());
        }
        // SAFETY: the size/version were checked and the library remains mapped.
        let table = unsafe { &*api };
        if table.kernel_size != std::mem::size_of::<abi::plans::Kernel>()
            || table.argument_size != std::mem::size_of::<abi::plans::Argument>()
            || table.program_size != std::mem::size_of::<abi::plans::Program>()
            || table.command_size != std::mem::size_of::<abi::plans::Command>()
        {
            return Err("Incompatible native plan record layout".into());
        }
        let mut output = abi::plans::Output::default();
        let mut error = OwnedBytes::default();
        // SAFETY: handle belongs to this library and output/error are live slots.
        let status = unsafe { (table.export)(self.handle, &mut output, &mut error) };
        let result = (|| {
            let error = self.consume_bytes(error)?;
            if status != 0 {
                return Err(format!("Native plans: {}", String::from_utf8_lossy(&error)));
            }
            let mut names = std::collections::BTreeSet::new();
            for k in span(output.kernels, output.kernel_count, 1_000_000)? {
                let name = string(k.name)?;
                match k.plan_contract {
                    0 => {}
                    1 => {
                        self.plan_contracts.borrow_mut().insert(name.clone());
                    }
                    _ => return Err("Invalid native plan contract flag".into()),
                }
                if !names.insert(name.clone()) {
                    return Err("Duplicate native kernel binding".into());
                }
                let args = span(k.arguments, k.argument_count, 256)?
                    .iter()
                    .map(|a| {
                        if (a.kind >= 2 && (a.name.len != 0 || a.offset != 0))
                            || (a.kind <= 1 && a.scalar != 0)
                            || (a.kind == 0 && a.offset != 0)
                        {
                            return Err("Invalid native argument payload".into());
                        }
                        let narrow = || {
                            u32::try_from(a.scalar)
                                .map_err(|_| "Native scalar exceeds u32".to_owned())
                        };
                        Ok(match a.kind {
                            0 => Argument::Buffer {
                                name: string(a.name)?,
                            },
                            1 => Argument::BufferSlice {
                                name: string(a.name)?,
                                offset: a.offset,
                            },
                            2 => Argument::I32 {
                                value: narrow()? as i32,
                            },
                            3 => Argument::U32 { value: narrow()? },
                            4 => Argument::I64 {
                                value: a.scalar as i64,
                            },
                            5 => Argument::U64 { value: a.scalar },
                            6 => Argument::F32 {
                                value: f32::from_bits(narrow()?),
                            },
                            _ => return Err("Unknown native argument kind".into()),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let binding = catalog.bind(k.definition, name.clone(), k.grid, args)?;
                if k.dynamic != usize::MAX {
                    let template = catalog
                        .dynamic
                        .get(k.dynamic)
                        .ok_or("Unknown dynamic launch definition")?
                        .bind(&name);
                    template.validate_binding(&binding)?;
                    metadata.dynamic_batch_kernels.insert(name, template);
                }
                metadata.kernels.push(binding);
            }
            for p in span(output.programs, output.program_count, 65536)? {
                let ops = span(p.commands, p.command_count, 1_000_000)?
                    .iter()
                    .map(|c| {
                        Ok(match c.kind {
                            0 if c.bytes == 0 && c.destination.len == 0 => Operation::Kernel {
                                name: string(c.name)?,
                            },
                            1 => Operation::Copy {
                                source: string(c.name)?,
                                destination: string(c.destination)?,
                                bytes: c.bytes,
                            },
                            2 if c.name.len == 0 => Operation::Zero {
                                destination: string(c.destination)?,
                                bytes: c.bytes,
                            },
                            _ => return Err("Invalid native plan command".into()),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                if metadata.programs.insert(string(p.name)?, ops).is_some() {
                    return Err("Duplicate native program".into());
                }
            }
            if metadata.kernels.is_empty() || metadata.programs.is_empty() {
                return Err("Empty native execution plan".into());
            }
            Ok(())
        })();
        // SAFETY: return the exact output, including error-path allocations, once.
        unsafe {
            (table.free)(output);
        }
        result
    }
    pub(crate) fn prepare_inputs(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
        mut upload: impl FnMut(&str, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let Some(api) = self.input_api else {
            return Ok(());
        };
        // SAFETY: self retains the validated extension and native model lifetime.
        let table = unsafe { &*api };
        let mut output = abi::inputs::Output::default();
        let mut error = OwnedBytes::default();
        // SAFETY: borrowed tokens/history and output slots are live for the call.
        let status = unsafe {
            (table.prepare_program)(
                self.handle,
                Bytes::borrowed(program),
                tokens.as_ptr(),
                tokens.len(),
                history.as_ptr(),
                history.len(),
                &mut output,
                &mut error,
            )
        };
        let result = (|| {
            let message = self.consume_bytes(error)?;
            if status != 0 {
                return Err(format!(
                    "Model inputs: {}",
                    String::from_utf8_lossy(&message)
                ));
            }
            let mut names = std::collections::BTreeSet::new();
            for item in span(output.uploads, output.count, 16)? {
                let name = string(item.buffer)?;
                if !names.insert(name.clone()) {
                    return Err("Duplicate input upload".into());
                }
                upload(
                    &name,
                    span(item.data.data, item.data.len, 128 * 1024 * 1024)?,
                )?;
            }
            Ok(())
        })();
        // SAFETY: all borrowed upload data have finished use; return producer ownership.
        unsafe {
            (table.free)(output);
        }
        result
    }
    fn table(&self) -> &abi::Api {
        // SAFETY: _library keeps the validated function table mapped.
        unsafe { &*self.api }
    }
    fn consume_bytes(&self, value: OwnedBytes) -> Result<Vec<u8>> {
        let bytes = if value.len > 128 * 1024 * 1024 || (value.data.is_null() && value.len != 0) {
            Err("Invalid model package response span".into())
        } else if value.len == 0 {
            Ok(vec![])
        } else {
            // SAFETY: the ABI provides len readable bytes until free_bytes.
            Ok(unsafe { slice::from_raw_parts(value.data, value.len) }.to_vec())
        };
        // SAFETY: release using the exact allocator/library that returned value.
        unsafe {
            (self.table().free_bytes)(value);
        }
        bytes
    }
    pub(crate) fn visual_layout(
        &self,
        tokens: &[u32],
        images: &[crate::vision::ImageInput],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let grids: Vec<_> = images
            .iter()
            .map(|i| abi::ImageGrid {
                grid_height: i.grid_height,
                grid_width: i.grid_width,
            })
            .collect();
        let mut output = abi::VisualOutput::default();
        let mut error = OwnedBytes::default();
        // SAFETY: all spans and this library-owned model remain alive for the call.
        let status = unsafe {
            (self.table().visual)(
                self.handle,
                tokens.as_ptr(),
                tokens.len(),
                grids.as_ptr(),
                grids.len(),
                capacity,
                &mut output,
                &mut error,
            )
        };
        let message = self.consume_bytes(error)?;
        let result = if status != 0 {
            Err(format!(
                "Visual layout: {}",
                String::from_utf8_lossy(&message)
            ))
        } else {
            (|| {
                if output.index_count != capacity
                    || output.position_count
                        != capacity.checked_mul(3).ok_or("Visual size overflow")?
                {
                    return Err("Invalid visual position geometry".into());
                }
                Ok((
                    span(output.indices, output.index_count, 262144)?.to_vec(),
                    span(output.positions, output.position_count, 3 * 262144)?.to_vec(),
                ))
            })()
        };
        // SAFETY: return buffers to their originating library exactly once.
        unsafe {
            (self.table().free_visual)(output);
        }
        result
    }
    pub(crate) fn batch_plan(&self, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
        self.cached_batch(segments, abi::BatchOperation::Ordinary)
            .map(|(plan, _)| (*plan).clone())
    }
    pub(crate) fn supports_batch(
        &self,
        segments: &[BatchSegment],
        operation: abi::BatchOperation,
    ) -> bool {
        let shape: Vec<_> = segments.iter().map(|s| s.tokens).collect();
        let mut cache = self.batch_capabilities.borrow_mut();
        if let Some((_, _, supported)) = cache
            .iter()
            .find(|(op, key, _)| *op == operation && *key == shape)
        {
            return *supported;
        }
        let supported = self.batch_operation(segments, operation, false).is_ok();
        if cache.len() >= 128 {
            cache.pop_front();
        }
        cache.push_back((operation, shape, supported));
        supported
    }
    pub(crate) fn cached_batch(
        &self,
        segments: &[BatchSegment],
        operation: abi::BatchOperation,
    ) -> Result<CachedBatch> {
        let key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        let mut cache = self.batch_plans.borrow_mut();
        if let Some(index) = cache
            .iter()
            .position(|(kind, k, _, _)| *kind == operation && *k == key)
        {
            let entry = cache.remove(index).expect("cached batch plan");
            let result = entry.2.clone();
            cache.push_back(entry);
            return Ok(result);
        }
        // The packet-table contract makes calls a function of phase and exact
        // segment lengths. A new owner cohort needs fresh bindings, not a new
        // native plan plus a decode of thousands of identical commands.
        if self.indirect_packets {
            let shaped = cache.iter().find_map(|(kind, previous, (plan, _), bytes)| {
                (*kind == operation
                    && previous.len() == key.len()
                    && previous.iter().zip(&key).all(|(a, b)| a.1 == b.1)
                    && plan.iter().all(|call| call.sequence.is_none()))
                .then(|| (std::rc::Rc::clone(plan), *bytes))
            });
            if let Some((plan, bytes)) = shaped {
                let (commands, bindings) = self.batch_operation(segments, operation, false)?;
                if !commands.is_empty() {
                    return Err("Bindings-only packet returned execution commands".into());
                }
                let result = (plan, std::rc::Rc::new(bindings));
                return Ok(Self::retain_batch(
                    &mut cache, operation, key, result, bytes,
                ));
            }
        }
        let (plan, bindings) = self.batch_operation(segments, operation, true)?;
        let bytes: usize = plan
            .iter()
            .map(|op| {
                let strings = match &op.operation {
                    Operation::Kernel { name } => name.len(),
                    Operation::Copy {
                        source,
                        destination,
                        ..
                    } => source.len() + destination.len(),
                    Operation::Zero { destination, .. } => destination.len(),
                };
                std::mem::size_of::<Invocation>()
                    + strings
                    + op.views
                        .iter()
                        .map(|(name, view)| {
                            std::mem::size_of::<(String, BufferView)>()
                                + 32
                                + name.len()
                                + view.buffer.len()
                        })
                        .sum::<usize>()
            })
            .sum();
        // Owner-independent calls can share their immutable plan while each
        // membership retains its own state bindings. Keep phase identity exact.
        let shared = plan
            .iter()
            .all(|call| call.sequence.is_none())
            .then(|| {
                cache.iter().find_map(|(kind, _, (cached, _), _)| {
                    (*kind == operation && cached.as_slice() == plan)
                        .then(|| std::rc::Rc::clone(cached))
                })
            })
            .flatten();
        let result = (
            shared.unwrap_or_else(|| std::rc::Rc::new(plan)),
            std::rc::Rc::new(bindings),
        );
        Ok(Self::retain_batch(
            &mut cache, operation, key, result, bytes,
        ))
    }
    fn retain_batch(
        cache: &mut PlanEntries,
        operation: abi::BatchOperation,
        key: Vec<(usize, usize)>,
        result: CachedBatch,
        bytes: usize,
    ) -> CachedBatch {
        const BUDGET: usize = 16 << 20;
        if bytes <= BUDGET {
            while cache.len() >= 32
                || cache.iter().map(|(_, _, _, bytes)| bytes).sum::<usize>() + bytes > BUDGET
            {
                cache.pop_front();
            }
            cache.push_back((operation, key, result.clone(), bytes));
        }
        result
    }
    pub(crate) fn state_bindings(
        &self,
        segments: &[BatchSegment],
    ) -> Result<std::rc::Rc<abi::StateBindings>> {
        let key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        if let Some((_, _, (_, bindings), _)) = self
            .batch_plans
            .borrow()
            .iter()
            .find(|(kind, k, _, _)| *kind == abi::BatchOperation::Ordinary && *k == key)
        {
            return Ok(std::rc::Rc::clone(bindings));
        }
        let mut cache = self.bindings.borrow_mut();
        if let Some(index) = cache.iter().position(|(k, _)| k == &key) {
            let entry = cache.remove(index).expect("existing bindings");
            let result = std::rc::Rc::clone(&entry.1);
            cache.push_back(entry);
            return Ok(result);
        }
        let bindings = std::rc::Rc::new(self.batch(segments, false)?.1);
        if cache.len() == 16 {
            cache.pop_front();
        }
        cache.push_back((key, std::rc::Rc::clone(&bindings)));
        Ok(bindings)
    }
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<abi::BatchPlan> {
        self.batch_operation(segments, abi::BatchOperation::Ordinary, include_plan)
    }
    pub(crate) fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: abi::BatchOperation,
        include_plan: bool,
    ) -> Result<abi::BatchPlan> {
        let mut output = BatchOutput::default();
        let mut error = OwnedBytes::default();
        // SAFETY: this live model and borrowed segments are retained for the call.
        let status = unsafe {
            (self.table().batch)(
                self.handle,
                segments.as_ptr(),
                segments.len(),
                operation.flags(include_plan),
                &mut output,
                &mut error,
            )
        };
        let message = self.consume_bytes(error)?;
        let result = if status != 0 {
            Err(format!(
                "Model batch: {}",
                String::from_utf8_lossy(&message)
            ))
        } else {
            self.decode_batch(&output)
        };
        // SAFETY: output belongs to this library and is released exactly once.
        unsafe {
            (self.table().free_batch)(output);
        }
        result
    }
    fn decode_batch(&self, output: &BatchOutput) -> Result<abi::BatchPlan> {
        let mut plan = vec![];
        for command in span(output.commands, output.command_count, 1_000_000)? {
            let operation = match command.kind {
                0 => Operation::Kernel {
                    name: string(command.name)?,
                },
                1 => Operation::Copy {
                    source: string(command.name)?,
                    destination: string(command.destination)?,
                    bytes: command.bytes,
                },
                2 => Operation::Zero {
                    destination: string(command.destination)?,
                    bytes: command.bytes,
                },
                _ => return Err("Unknown model package command".into()),
            };
            let mut views = BTreeMap::new();
            let mut seen = std::collections::BTreeSet::new();
            for view in span(command.views, command.view_count, 256)? {
                let name = borrowed_string(view.name)?;
                let buffer = borrowed_string(view.buffer)?;
                if !seen.insert(name) {
                    return Err("Duplicate model package buffer view".into());
                }
                // Adapters may give a whole boundary table to each operation.
                // Retain only its actual operands, including slices and copies.
                // This avoids cloning thousands of irrelevant view strings and
                // makes complete mixed/MTP plans fit the bounded CPU cache.
                let needed = match &operation {
                    Operation::Kernel { name: kernel } => self
                        .kernel_buffers
                        .get(kernel)
                        .is_none_or(|args| args.binary_search_by(|a| a.as_str().cmp(name)).is_ok()),
                    Operation::Copy {
                        source,
                        destination,
                        ..
                    } => name == source || name == destination,
                    Operation::Zero { destination, .. } => name == destination,
                };
                if needed {
                    views.insert(
                        name.to_owned(),
                        BufferView {
                            buffer: buffer.to_owned(),
                            offset: view.offset,
                        },
                    );
                }
            }
            let launch = match command.has_launch {
                0 => None,
                1 => Some(crate::operators::dynamic::RowLaunch {
                    grid: command.grid,
                    arguments: span(command.arguments, command.argument_count, 256)?
                        .iter()
                        .map(|a| (a.index, a.value))
                        .collect(),
                }),
                _ => return Err("Invalid model package launch flag".into()),
            };
            plan.push(Invocation {
                operation,
                sequence: (command.sequence != usize::MAX).then_some(command.sequence),
                launch,
                views,
            });
        }
        let bindings = span(output.bindings, output.binding_count, 1_000_000)?
            .iter()
            .map(|b| match b.present {
                0 => Ok(None),
                1 => Ok(Some((b.sequence, string(b.buffer)?))),
                _ => Err("Invalid model state binding flag".into()),
            })
            .collect::<Result<_>>()?;
        Ok((plan, bindings))
    }
}
fn span<'a, T>(data: *const T, len: usize, limit: usize) -> Result<&'a [T]> {
    if len > limit
        || (len != 0
            && (data.is_null() || !(data as usize).is_multiple_of(std::mem::align_of::<T>())))
    {
        return Err("Invalid model package array span".into());
    }
    if len == 0 {
        return Ok(&[]);
    }
    // SAFETY: trusted native package owns this array until free_batch. Bounds and
    // alignment are checked here; arbitrary native code is not a sandbox.
    Ok(unsafe { slice::from_raw_parts(data, len) })
}
fn string(value: Bytes) -> Result<String> {
    borrowed_string(value).map(str::to_owned)
}
fn borrowed_string<'a>(value: Bytes) -> Result<&'a str> {
    std::str::from_utf8(span(value.data, value.len, 65536)?).map_err(|e| e.to_string())
}
impl Drop for ModelPackage {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: the creating library remains mapped and destroys only its own handle.
            unsafe {
                (self.table().destroy)(self.handle);
            }
        }
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    fn fixture() -> (std::path::PathBuf, ExecutionLibrary, CreateRequest) {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "orinfer-native-model-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let engine = Path::new(env!("CARGO_MANIFEST_DIR"));
        let output = root.join("model.so");
        assert!(
            std::process::Command::new("cc")
                .args(["-shared", "-fPIC", "-Werror", "-Wall", "-Wextra"])
                .arg("-I")
                .arg(engine.join("../orinfer-model-sdk/include"))
                .arg(engine.join("tests/fixtures/model_package.c"))
                .arg("-o")
                .arg(&output)
                .status()
                .unwrap()
                .success()
        );
        let spec = ExecutionLibrary {
            abi_version: 1,
            library: FileIdentity {
                file: "model.so".into(),
                tensor: None,
                sha256: crate::artifact::sha256(&std::fs::read(output).unwrap()),
            },
            package: "test-model".into(),
            version: "1".into(),
        };
        let metadata=serde_json::from_value(serde_json::json!({"schema_version":2,"target":"sm_87","model":"fixture",
            "chunk_tokens":1,"max_context":8,"vocab":4,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"Input","token":"Token","status":"Status","logits":"Logits","position":"Position",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"fixture"})).unwrap();
        (
            root,
            spec,
            CreateRequest {
                catalog: None,
                expert_contracts: false,
                build_bindings: false,
                kernel_contracts: vec![],
                model_root: String::new(),
                verify_weights: false,
                config: serde_json::json!({"model_type":"test_family"}),
                architecture: "test_family".into(),
                compute_policy: "test_policy".into(),
                expected_signature: serde_json::Value::Null,
                metadata,
                prefill_profiles: vec![],
            },
        )
    }
    #[test]
    fn batch_views_keep_only_operands_and_still_reject_unused_duplicates() {
        let (root, spec, request) = fixture();
        let (mut model, _) = ModelPackage::create(&root, &spec, request).unwrap();
        model
            .kernel_buffers
            .insert("projection".into(), vec!["Input".into()]);
        let bytes = |s: &'static str| Bytes {
            data: s.as_ptr(),
            len: s.len(),
        };
        let views = [
            abi::View {
                name: bytes("Input"),
                buffer: bytes("Scratch"),
                offset: 128,
            },
            abi::View {
                name: bytes("Unused"),
                buffer: bytes("Other"),
                offset: 0,
            },
            abi::View {
                name: bytes("Unused"),
                buffer: bytes("Other"),
                offset: 0,
            },
        ];
        let mut command = abi::Command {
            kind: 0,
            name: bytes("projection"),
            destination: bytes(""),
            bytes: 0,
            sequence: usize::MAX,
            has_launch: 0,
            grid: [0; 3],
            arguments: ptr::null(),
            argument_count: 0,
            views: views.as_ptr(),
            view_count: 2,
        };
        let decode = |command: &abi::Command| {
            model.decode_batch(&BatchOutput {
                owner: ptr::null_mut(),
                commands: command,
                command_count: 1,
                bindings: ptr::null(),
                binding_count: 0,
            })
        };
        let plan = decode(&command).unwrap().0;
        assert_eq!(plan[0].views.len(), 1);
        assert_eq!(plan[0].views["Input"].offset, 128);
        command.kind = 1;
        command.name = bytes("Unused");
        command.destination = bytes("Input");
        assert_eq!(decode(&command).unwrap().0[0].views.len(), 2);
        command.view_count = 3;
        assert!(decode(&command).is_err());
    }
    #[test]
    fn program_controls_do_not_require_cpu_weight_assets() {
        let (root, spec, request) = fixture();
        assert!(request.metadata.input_assets.is_none());
        let (model, _) = ModelPackage::create(&root, &spec, request).unwrap();
        let mut uploads = Vec::new();
        model
            .prepare_inputs("warm", &[1, 2, 3], &[], |name, data| {
                uploads.push((name.to_owned(), data.to_vec()));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            uploads,
            [("SegmentLength".into(), 3u32.to_le_bytes().to_vec())]
        );
        drop(model);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn cached_batch_retains_owned_plans_and_isolates_phases_and_membership() {
        let (root, spec, request) = fixture();
        let (model, _) = ModelPackage::create(&root, &spec, request).unwrap();
        let segments = [
            BatchSegment { slot: 0, tokens: 2 },
            BatchSegment { slot: 3, tokens: 2 },
        ];
        let (a, _) = model
            .cached_batch(&segments, abi::BatchOperation::Ordinary)
            .unwrap();
        let (b, _) = model
            .cached_batch(&segments, abi::BatchOperation::Ordinary)
            .unwrap();
        assert!(std::rc::Rc::ptr_eq(&a, &b));
        let (verify, _) = model
            .cached_batch(&segments, abi::BatchOperation::Verification)
            .unwrap();
        assert!(!std::rc::Rc::ptr_eq(&a, &verify));
        let (changed, _) = model
            .cached_batch(
                &[BatchSegment { slot: 1, tokens: 2 }],
                abi::BatchOperation::Ordinary,
            )
            .unwrap();
        assert!(!std::rc::Rc::ptr_eq(&a, &changed));
        drop(model);
        // Data outlives the native output owner, free callback and library.
        assert_eq!(a[0].sequence, Some(7));
        assert_eq!(a[0].views["Input"].offset, 128);
        assert_eq!(a[0].launch.as_ref().unwrap().arguments, [(2, 3)]);
    }
    #[test]
    fn indirect_packet_shape_reuses_commands_and_refreshes_membership_bindings() {
        let (root, spec, request) = fixture();
        let (mut model, _) = ModelPackage::create(&root, &spec, request).unwrap();
        let phase = abi::BatchOperation::Ordinary;
        let initial = [
            BatchSegment { slot: 0, tokens: 2 },
            BatchSegment { slot: 3, tokens: 4 },
        ];
        let (commands, old_bindings) = model.cached_batch(&initial, phase).unwrap();
        // Install an owner-independent fixture plan, as certified by a real
        // packet-table adapter; the C fixture returns fresh owner bindings.
        let mut commands = commands.as_ref().clone();
        commands[0].sequence = None;
        let commands = std::rc::Rc::new(commands);
        model.batch_plans.borrow_mut().back_mut().unwrap().2.0 = commands.clone();
        model.indirect_packets = true;
        let replacement = [
            BatchSegment {
                slot: 11,
                tokens: 2,
            },
            BatchSegment { slot: 9, tokens: 4 },
        ];
        let (reused, new_bindings) = model.cached_batch(&replacement, phase).unwrap();
        assert!(std::rc::Rc::ptr_eq(&commands, &reused));
        assert!(!std::rc::Rc::ptr_eq(&old_bindings, &new_bindings));
        assert_eq!(old_bindings[1], Some((0, "PrivateState".into())));
        assert_eq!(new_bindings[1], Some((11, "PrivateState".into())));
        for (lengths, operation) in [
            ([2, 3], phase),
            ([4, 2], phase),
            ([2, 4], abi::BatchOperation::Commit),
            ([2, 4], abi::BatchOperation::MixedRows { prefills: 1 }),
        ] {
            let segments = std::array::from_fn::<_, 2, _>(|i| BatchSegment {
                slot: replacement[i].slot,
                tokens: lengths[i],
            });
            let (different, _) = model.cached_batch(&segments, operation).unwrap();
            assert!(!std::rc::Rc::ptr_eq(&commands, &different));
        }
        // Without the declared table contract, exact membership remains part
        // of plan identity even when a previous plan happens to be shared.
        model.indirect_packets = false;
        let (different, _) = model
            .cached_batch(
                &[
                    BatchSegment {
                        slot: 12,
                        tokens: 2,
                    },
                    BatchSegment {
                        slot: 13,
                        tokens: 4,
                    },
                ],
                phase,
            )
            .unwrap();
        assert!(!std::rc::Rc::ptr_eq(&commands, &different));
        drop(model);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn native_bulk_export_preserves_parameter_bits_and_program_commands() {
        use crate::artifact::Argument;
        let (root, spec, mut request) = fixture();
        let catalog: orinfer_model_sdk::catalog::Catalog =
            serde_json::from_value(serde_json::json!({
                "binding_signature":"0".repeat(64),
                "assets":[{"file":"kernel.cubin","sha256":"0".repeat(64)}],
                "definitions":[{"module":0,"source":0,"host_abi":0,"symbol":"main",
                    "block":[32,1,1],"shared_memory_bytes":0,"cooperative":false,
                    "parameters":["buffer","buffer_slice","i32","u32","i64","u64","f32"]}],
                "dynamic":[]
            }))
            .unwrap();
        request.catalog = Some(catalog);
        let (model, created) = ModelPackage::create(&root, &spec, request).unwrap();
        let k = &created.metadata.kernels[0];
        assert_eq!(k.name, "projection");
        assert_eq!(k.grid, [1, 1, 1]);
        assert!(matches!(&k.args[0], Argument::Buffer { name } if name == "Input"));
        assert!(matches!(&k.args[1], Argument::BufferSlice { name, offset:64 } if name == "State"));
        assert!(matches!(k.args[2], Argument::I32 { value: -1 }));
        assert!(matches!(k.args[3], Argument::U32 { value: u32::MAX }));
        assert!(matches!(k.args[4], Argument::I64 { value: -1 }));
        assert!(matches!(k.args[5], Argument::U64 { value: u64::MAX }));
        assert!(matches!(k.args[6], Argument::F32 { value } if value.to_bits() == 0x80000000));
        assert_eq!(created.metadata.programs["decode"].len(), 3);
        assert!(matches!(
            created.metadata.programs["decode"][1],
            Operation::Copy { bytes: 4, .. }
        ));
        assert!(matches!(
            created.metadata.programs["decode"][2],
            Operation::Zero { bytes: 4, .. }
        ));
        drop(model);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn independent_c_package_loads_unknown_family_and_preserves_binary_batch_contract() {
        let (root, spec, request) = fixture();
        let description = inspect_library(&root.join("model.so")).unwrap();
        assert_eq!(description.package, spec.package);
        assert_eq!(description.version, spec.version);
        let (model, plan) = ModelPackage::create(&root, &spec, request).unwrap();
        assert!(plan.decode_programs.contains("decode"));
        let segments = [BatchSegment { slot: 7, tokens: 3 }];
        let operations = model.batch_plan(&segments).unwrap();
        assert_eq!(operations[0].sequence, Some(7));
        assert_eq!(operations[0].launch.as_ref().unwrap().arguments, [(2, 3)]);
        assert_eq!(operations[0].views["Input"].offset, 128);
        let first = model.state_bindings(&segments).unwrap();
        let second = model.state_bindings(&segments).unwrap();
        assert!(std::rc::Rc::ptr_eq(&first, &second));
        assert_eq!(first[1], Some((7, "PrivateState".into())));
        assert!(
            model
                .visual_layout(&[1], &[], 8)
                .unwrap_err()
                .contains("unsupported")
        );
        drop(model);
        std::fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn explicit_hash_checks_and_runtime_abi_identity_checks() {
        for mode in 0..4 {
            let (root, mut spec, mut request) = fixture();
            match mode {
                0 => {
                    spec.library.sha256 = "0".repeat(64);
                    request.verify_weights = true;
                }
                1 => spec.abi_version = 99,
                2 => spec.version = "unexpected".into(),
                _ => request.compute_policy = "a4_not_supported".into(),
            }
            assert!(ModelPackage::create(&root, &spec, request).is_err());
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn normal_library_loading_leaves_content_verification_to_validate_model() {
        let (root, mut spec, request) = fixture();
        spec.library.sha256 = "0".repeat(64);
        let (model, _) = ModelPackage::create(&root, &spec, request).unwrap();
        drop(model);
        std::fs::remove_dir_all(root).unwrap();
    }
}

#[cfg(test)]
mod portable_tests {
    #[test]
    fn execution_library_machine_matches_its_host_abi_before_loading() {
        let mut image = [0u8; 20];
        image[..6].copy_from_slice(b"\x7fELF\x02\x01");
        image[16..18].copy_from_slice(&3u16.to_le_bytes());
        image[18..20].copy_from_slice(&62u16.to_le_bytes());
        assert!(super::validate_library(&image, "x86_64").is_ok());
        assert!(super::validate_library(&image, "aarch64").is_err());
        image[18..20].copy_from_slice(&183u16.to_le_bytes());
        assert!(super::validate_library(&image, "aarch64").is_ok());
        assert!(super::validate_library(&image, "x86_64").is_err());
        let other = if std::env::consts::ARCH == "aarch64" {
            62u16
        } else {
            183u16
        };
        image[18..20].copy_from_slice(&other.to_le_bytes());
        let error = super::ModelPackage::open(std::path::Path::new("unused.so"), &image)
            .err()
            .unwrap();
        assert!(error.contains("ELF shared object"));
    }
}
