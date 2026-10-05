//! Private, thread-affine CUDA Driver session for fixed AOT fixtures.
//! Handles never escape the session. All work is on one explicit stream; cleanup
//! synchronizes before destroying graphs/modules or freeing device allocations.
use crate::artifact::{Argument, Dtype, Kernel, Loaded, Result};
use libloading::Library;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    ffi::{CString, c_char, c_void},
    ptr,
    time::Instant,
};
mod sequence;
pub(crate) mod snapshot;
mod virtual_memory;
pub(crate) type Handle = *mut c_void;

pub(crate) struct Driver {
    pub(crate) _library: Library,
    pub(crate) init: unsafe extern "C" fn(u32) -> i32,
    pub(crate) version: unsafe extern "C" fn(*mut i32) -> i32,
    pub(crate) device: unsafe extern "C" fn(*mut i32, i32) -> i32,
    pub(crate) name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    pub(crate) attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    pub(crate) total: unsafe extern "C" fn(*mut usize, i32) -> i32,
    pub(crate) current: unsafe extern "C" fn(*mut Handle) -> i32,
    pub(crate) retain: unsafe extern "C" fn(*mut Handle, i32) -> i32,
    pub(crate) release: unsafe extern "C" fn(i32) -> i32,
    pub(crate) set_current: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) context_sync: unsafe extern "C" fn() -> i32,
    pub(crate) stream_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    pub(crate) stream_sync: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) stream_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) memory_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    pub(crate) alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    pub(crate) vmm_reserve: unsafe extern "C" fn(*mut u64, usize, usize, u64, u64) -> i32,
    pub(crate) vmm_address_free: unsafe extern "C" fn(u64, usize) -> i32,
    pub(crate) vmm_granularity:
        unsafe extern "C" fn(*mut usize, *const virtual_memory::AllocationProp, i32) -> i32,
    pub(crate) vmm_create:
        unsafe extern "C" fn(*mut u64, usize, *const virtual_memory::AllocationProp, u64) -> i32,
    pub(crate) vmm_map: unsafe extern "C" fn(u64, usize, usize, u64, u64) -> i32,
    pub(crate) vmm_access:
        unsafe extern "C" fn(u64, usize, *const virtual_memory::AccessDesc, usize) -> i32,
    pub(crate) vmm_unmap: unsafe extern "C" fn(u64, usize) -> i32,
    pub(crate) vmm_release: unsafe extern "C" fn(u64) -> i32,
    pub(crate) free: unsafe extern "C" fn(u64) -> i32,
    pub(crate) upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    pub(crate) copy: unsafe extern "C" fn(u64, u64, usize, Handle) -> i32,
    pub(crate) download: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    pub(crate) memset: unsafe extern "C" fn(u64, u8, usize, Handle) -> i32,
    pub(crate) module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> i32,
    pub(crate) module_unload: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> i32,
    pub(crate) function_get: unsafe extern "C" fn(*mut i32, i32, Handle) -> i32,
    pub(crate) function_set: unsafe extern "C" fn(Handle, i32, i32) -> i32,
    pub(crate) launch: unsafe extern "C" fn(
        Handle,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        Handle,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32,
    pub(crate) capture_begin: unsafe extern "C" fn(Handle, i32) -> i32,
    pub(crate) capture_end: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    pub(crate) graph_instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> i32,
    pub(crate) graph_launch: unsafe extern "C" fn(Handle, Handle) -> i32,
    pub(crate) graph_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) graph_exec_destroy: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) event_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    pub(crate) event_record: unsafe extern "C" fn(Handle, Handle) -> i32,
    pub(crate) event_sync: unsafe extern "C" fn(Handle) -> i32,
    pub(crate) event_elapsed: unsafe extern "C" fn(*mut f32, Handle, Handle) -> i32,
    pub(crate) event_destroy: unsafe extern "C" fn(Handle) -> i32,
}

impl Driver {
    pub(crate) fn load() -> Result<Self> {
        // SAFETY: This process loads the installed CUDA driver, whose exported C
        // signatures below are checked against the local CUDA 12.6 cuda.h. The
        // Library is retained for longer than every copied function pointer.
        let library = unsafe { Library::new("libcuda.so.1") }.map_err(|e| e.to_string())?;
        macro_rules! symbol {
            ($name:literal) => {{
                // SAFETY: Each inferred field type matches this named CUDA C symbol;
                // its library remains owned by Driver until session cleanup completes.
                unsafe {
                    *library
                        .get(concat!($name, "\0").as_bytes())
                        .map_err(|e| e.to_string())?
                }
            }};
        }
        Ok(Self {
            init: symbol!("cuInit"),
            version: symbol!("cuDriverGetVersion"),
            device: symbol!("cuDeviceGet"),
            name: symbol!("cuDeviceGetName"),
            attribute: symbol!("cuDeviceGetAttribute"),
            total: symbol!("cuDeviceTotalMem_v2"),
            current: symbol!("cuCtxGetCurrent"),
            retain: symbol!("cuDevicePrimaryCtxRetain"),
            release: symbol!("cuDevicePrimaryCtxRelease_v2"),
            set_current: symbol!("cuCtxSetCurrent"),
            context_sync: symbol!("cuCtxSynchronize"),
            stream_create: symbol!("cuStreamCreate"),
            stream_sync: symbol!("cuStreamSynchronize"),
            stream_destroy: symbol!("cuStreamDestroy_v2"),
            memory_info: symbol!("cuMemGetInfo_v2"),
            alloc: symbol!("cuMemAlloc_v2"),
            vmm_reserve: symbol!("cuMemAddressReserve"),
            vmm_address_free: symbol!("cuMemAddressFree"),
            vmm_granularity: symbol!("cuMemGetAllocationGranularity"),
            vmm_create: symbol!("cuMemCreate"),
            vmm_map: symbol!("cuMemMap"),
            vmm_access: symbol!("cuMemSetAccess"),
            vmm_unmap: symbol!("cuMemUnmap"),
            vmm_release: symbol!("cuMemRelease"),
            free: symbol!("cuMemFree_v2"),
            upload: symbol!("cuMemcpyHtoD_v2"),
            copy: symbol!("cuMemcpyDtoDAsync_v2"),
            download: symbol!("cuMemcpyDtoH_v2"),
            memset: symbol!("cuMemsetD8Async"),
            module_load: symbol!("cuModuleLoadData"),
            module_unload: symbol!("cuModuleUnload"),
            function: symbol!("cuModuleGetFunction"),
            function_get: symbol!("cuFuncGetAttribute"),
            function_set: symbol!("cuFuncSetAttribute"),
            launch: symbol!("cuLaunchKernel"),
            capture_begin: symbol!("cuStreamBeginCapture_v2"),
            capture_end: symbol!("cuStreamEndCapture"),
            graph_instantiate: symbol!("cuGraphInstantiateWithFlags"),
            graph_launch: symbol!("cuGraphLaunch"),
            graph_destroy: symbol!("cuGraphDestroy"),
            graph_exec_destroy: symbol!("cuGraphExecDestroy"),
            event_create: symbol!("cuEventCreate"),
            event_record: symbol!("cuEventRecord"),
            event_sync: symbol!("cuEventSynchronize"),
            event_elapsed: symbol!("cuEventElapsedTime"),
            event_destroy: symbol!("cuEventDestroy_v2"),
            _library: library,
        })
    }
}
pub(crate) fn check(code: i32, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        crate::error::record_cuda(code);
        Err(format!("{operation}: CUDA error {code}"))
    }
}

#[derive(Debug, Serialize)]
pub struct DeviceInfo {
    name: String,
    sm: [i32; 2],
    driver_version: i32,
    total_bytes: usize,
}
#[derive(Debug, Serialize)]
pub struct RunReport {
    pub manifest_sha256: String,
    pub device: DeviceInfo,
    pub buffer_bytes: usize,
    pub artifact_validation_s: f64,
    pub device_load_s: f64,
    pub first_launch_wall_s: f64,
    pub graph_capture_s: f64,
    pub hot_graph_cuda_event_ms: f32,
    pub relative_l2: f64,
    pub max_abs_error: f64,
    pub changed_input_replay_verified: bool,
    pub output_poisoning_replay_verified: bool,
    pub output_elements: usize,
    pub repetitions: u32,
    pub scope: &'static str,
}

// Raw handles make this private owner !Send/!Sync. No CUDA context or allocation
// is borrowed by outside code, and all resources belong to the retained context.
pub(crate) struct Session {
    pub(crate) driver: Driver,
    pub(crate) device: i32,
    pub(crate) previous: Handle,
    pub(crate) retained: bool,
    pub(crate) stream: Handle,
    pub(crate) buffers: Vec<u64>,
    pub(crate) virtual_buffers: std::cell::RefCell<virtual_memory::Reservations>,
    pub(crate) modules: Vec<Handle>,
    pub(crate) events: Vec<Handle>,
    pub(crate) graph: Handle,
    pub(crate) exec: Handle,
    pub(crate) graphs: std::cell::RefCell<Vec<(Handle, Handle)>>,
    pub(crate) capturing: bool,
}
impl Session {
    pub(crate) fn new(driver: Driver) -> Self {
        Self {
            driver,
            device: 0,
            previous: ptr::null_mut(),
            retained: false,
            stream: ptr::null_mut(),
            buffers: vec![],
            virtual_buffers: Default::default(),
            modules: vec![],
            events: vec![],
            graph: ptr::null_mut(),
            exec: ptr::null_mut(),
            graphs: Default::default(),
            capturing: false,
        }
    }
    pub(crate) fn cleanup(&mut self) -> Result<()> {
        let mut errors = vec![];
        let mut record = |code, op| {
            if let Err(e) = check(code, op) {
                errors.push(e)
            }
        };
        // SAFETY: These handles are solely owned by this session on this thread.
        // End failed captures before syncing. Work finishes before allocations,
        // modules and the retained context are released. Fields are cleared even
        // on CUDA errors to prevent double destruction in Drop.
        unsafe {
            if self.capturing {
                record(
                    (self.driver.capture_end)(self.stream, &mut self.graph),
                    "cleanup end capture",
                );
                self.capturing = false;
            }
            if !self.stream.is_null() {
                record(
                    (self.driver.stream_sync)(self.stream),
                    "cleanup stream synchronize",
                );
            }
            if !self.exec.is_null() {
                record(
                    (self.driver.graph_exec_destroy)(self.exec),
                    "destroy graph exec",
                );
                self.exec = ptr::null_mut();
            }
            if !self.graph.is_null() {
                record((self.driver.graph_destroy)(self.graph), "destroy graph");
                self.graph = ptr::null_mut();
            }
            for (graph, exec) in self.graphs.get_mut().drain(..) {
                if !exec.is_null() {
                    record(
                        (self.driver.graph_exec_destroy)(exec),
                        "destroy model graph exec",
                    );
                }
                record((self.driver.graph_destroy)(graph), "destroy model graph");
            }
            for event in self.events.drain(..) {
                record((self.driver.event_destroy)(event), "destroy event");
            }
            for (_, mut buffer) in std::mem::take(self.virtual_buffers.get_mut()) {
                if let Err(e) = buffer.release_slabs(&self.driver) {
                    eprintln!("CUDA KV cleanup: {e}");
                }
                if buffer.has_slabs() {
                    continue;
                }
                record(
                    (self.driver.vmm_address_free)(buffer.address, buffer.bytes),
                    "free KV address reservation",
                );
            }
            for buffer in self.buffers.drain(..) {
                record((self.driver.free)(buffer), "free buffer");
            }
            for module in self.modules.drain(..) {
                record((self.driver.module_unload)(module), "unload module");
            }
            if !self.stream.is_null() {
                record((self.driver.stream_destroy)(self.stream), "destroy stream");
                self.stream = ptr::null_mut();
            }
            if self.retained {
                record(
                    (self.driver.release)(self.device),
                    "release primary context",
                );
                self.retained = false;
                record(
                    (self.driver.set_current)(self.previous),
                    "restore prior context",
                );
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        if let Err(e) = self.cleanup() {
            eprintln!("CUDA cleanup: {e}");
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Value {
    Pointer(u64),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
    F32(f32),
}
impl Value {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::Pointer(v) | Self::U64(v) => (v as *mut u64).cast(),
            Self::I32(v) => (v as *mut i32).cast(),
            Self::U32(v) => (v as *mut u32).cast(),
            Self::I64(v) => (v as *mut i64).cast(),
            Self::F32(v) => (v as *mut f32).cast(),
        }
    }
}
pub(crate) struct Launch<'a> {
    spec: &'a Kernel,
    function: Handle,
    values: Vec<Value>,
}
impl Launch<'_> {
    fn execute(&mut self, driver: &Driver, stream: Handle) -> Result<()> {
        launch_kernel(self.spec, self.function, &mut self.values, driver, stream)
    }
}

fn launch_kernel(
    k: &Kernel,
    function: Handle,
    values: &mut [Value],
    driver: &Driver,
    stream: Handle,
) -> Result<()> {
    let mut args: Vec<*mut c_void> = values.iter_mut().map(Value::address).collect();
    // SAFETY: Values remain at stable Vec addresses until cuLaunchKernel has
    // copied each parameter. Device pointers refer to live session buffers;
    // generated ABI order/types are supplied by the verified manifest.
    // The module and explicit stream outlive all queued/captured work.
    unsafe {
        check(
            (driver.launch)(
                function,
                k.grid[0],
                k.grid[1],
                k.grid[2],
                k.block[0],
                k.block[1],
                k.block[2],
                k.shared_memory_bytes,
                stream,
                args.as_mut_ptr(),
                ptr::null_mut(),
            ),
            &format!("launch {}", k.name),
        )
    }
}

pub(crate) fn floats(raw: &[u8], dtype: Dtype) -> Vec<f32> {
    match dtype {
        Dtype::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect(),
        Dtype::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| half_to_float(u16::from_le_bytes(*b)))
            .collect(),
        Dtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect(),
        _ => unreachable!("validated floating point output"),
    }
}
fn half_to_float(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits >> 10) & 31;
    let mant = u32::from(bits & 1023);
    if exp == 0 {
        let value = (mant as f32) * 2f32.powi(-24);
        if sign == 0 { value } else { -value }
    } else {
        f32::from_bits(
            sign | if exp == 31 {
                0x7f800000 | (mant << 13)
            } else {
                ((u32::from(exp) + 112) << 23) | (mant << 13)
            },
        )
    }
}
fn errors(output: &[f32], reference: &[f32]) -> Result<(f64, f64)> {
    let (mut sum, mut norm, mut maximum) = (0f64, 0f64, 0f64);
    for (&a, &b) in output.iter().zip(reference) {
        if !a.is_finite() || !b.is_finite() {
            return Err("Nonfinite projection/reference output".into());
        }
        let d = f64::from(a) - f64::from(b);
        sum += d * d;
        norm += f64::from(b).powi(2);
        maximum = maximum.max(d.abs());
    }
    Ok(((sum / norm.max(1e-60)).sqrt(), maximum))
}

pub(crate) fn run(x: Loaded) -> Result<RunReport> {
    let started = Instant::now();
    let mut s = Session::new(Driver::load()?);
    let (mut major, mut minor, mut version, mut total) = (0, 0, 0, 0usize);
    let mut name = [0u8; 256];
    let mut device_buffers = BTreeMap::new();
    let mut launches = vec![];
    // SAFETY: All output pointers below refer to initialized host locals, all
    // byte copies use validated exact buffer sizes. Driver handles are registered
    // immediately in Session for cleanup on every early return. The session is
    // thread-affine and the primary context remains current for its entire life.
    unsafe {
        check((s.driver.init)(0), "cuInit")?;
        check((s.driver.device)(&mut s.device, 0), "cuDeviceGet")?;
        check((s.driver.attribute)(&mut major, 75, s.device), "SM major")?;
        check((s.driver.attribute)(&mut minor, 76, s.device), "SM minor")?;
        if [major, minor] != [8, 7] {
            return Err(format!("Expected SM87, got {major}.{minor}"));
        }
        check((s.driver.version)(&mut version), "driver version")?;
        check(
            (s.driver.name)(name.as_mut_ptr().cast(), 256, s.device),
            "device name",
        )?;
        check(
            (s.driver.total)(&mut total, s.device),
            "total device memory",
        )?;
        check((s.driver.current)(&mut s.previous), "get previous context")?;
        let mut context = ptr::null_mut();
        check(
            (s.driver.retain)(&mut context, s.device),
            "retain primary context",
        )?;
        s.retained = true;
        check((s.driver.set_current)(context), "set current context")?;
        check(
            (s.driver.stream_create)(&mut s.stream, 1),
            "create nonblocking stream",
        )?;
        let (mut free, mut available_total) = (0usize, 0usize);
        check(
            (s.driver.memory_info)(&mut free, &mut available_total),
            "free device memory",
        )?;
        if x.buffer_bytes > free {
            return Err(format!(
                "Fixture needs {} bytes; CUDA reports {free} free",
                x.buffer_bytes
            ));
        }
        for b in &x.manifest.buffers {
            let bytes = b.bytes()?;
            let mut address = 0;
            check(
                (s.driver.alloc)(&mut address, bytes),
                &format!("allocate {}", b.name),
            )?;
            s.buffers.push(address);
            if address % b.alignment != 0 {
                return Err(format!("{} allocation alignment mismatch", b.name));
            }
            if let Some(id) = &b.data {
                check(
                    (s.driver.upload)(address, x.files[&id.file].as_ptr().cast(), bytes),
                    &format!("upload {}", b.name),
                )?
            } else {
                check(
                    (s.driver.memset)(address, 0, bytes, s.stream),
                    &format!("initialize {}", b.name),
                )?
            }
            device_buffers.insert(b.name.clone(), address);
        }
        for k in &x.manifest.kernels {
            let mut module = ptr::null_mut();
            let image = &x.files[&k.module.file];
            if !image.starts_with(b"\x7fELF") {
                return Err("Fixture runner requires a cubin ELF module".into());
            }
            check(
                (s.driver.module_load)(&mut module, image.as_ptr().cast()),
                &format!("load {}", k.name),
            )?;
            s.modules.push(module);
            let symbol = CString::new(k.symbol.as_str()).map_err(|e| e.to_string())?;
            let mut function = ptr::null_mut();
            check(
                (s.driver.function)(&mut function, module, symbol.as_ptr()),
                &format!("find {}", k.symbol),
            )?;
            let (mut maxthreads, mut staticsmem, mut maxshared) = (0, 0, 0);
            check(
                (s.driver.function_get)(&mut maxthreads, 0, function),
                "function max threads",
            )?;
            check(
                (s.driver.function_get)(&mut staticsmem, 1, function),
                "function static shared memory",
            )?;
            check(
                (s.driver.attribute)(&mut maxshared, 97, s.device),
                "device shared memory limit",
            )?;
            if k.block.iter().product::<u32>() > maxthreads as u32
                || u64::from(k.shared_memory_bytes) + staticsmem as u64 > maxshared as u64
            {
                return Err(format!(
                    "{} exceeds actual function/device resources",
                    k.name
                ));
            }
            for axis in 0..3 {
                let (mut maxblock, mut maxgrid) = (0, 0);
                check(
                    (s.driver.attribute)(&mut maxblock, 2 + axis as i32, s.device),
                    "block dimension limit",
                )?;
                check(
                    (s.driver.attribute)(&mut maxgrid, 5 + axis as i32, s.device),
                    "grid dimension limit",
                )?;
                if k.block[axis] > maxblock as u32 || k.grid[axis] > maxgrid as u32 {
                    return Err("Launch dimensions exceed device limits".into());
                }
            }
            let mut default_shared = 0;
            check(
                (s.driver.attribute)(&mut default_shared, 8, s.device),
                "default shared memory limit",
            )?;
            if u64::from(k.shared_memory_bytes) + staticsmem as u64 > default_shared as u64 {
                check(
                    (s.driver.function_set)(function, 8, k.shared_memory_bytes as i32),
                    "opt-in dynamic shared memory",
                )?
            }
            let values = k
                .args
                .iter()
                .map(|a| match a {
                    Argument::Buffer { name } => Value::Pointer(device_buffers[name]),
                    Argument::I32 { value } => Value::I32(*value),
                    Argument::U32 { value } => Value::U32(*value),
                    Argument::I64 { value } => Value::I64(*value),
                    Argument::U64 { value } => Value::U64(*value),
                    Argument::F32 { value } => Value::F32(*value),
                })
                .collect();
            launches.push(Launch {
                spec: k,
                function,
                values,
            });
        }
    }
    // Pageable host uploads use the synchronous Driver API; finish all loader
    // transfers before consuming them on the nonblocking execution stream.
    // SAFETY: The session owns the current context and all uploaded allocations.
    unsafe {
        check((s.driver.context_sync)(), "finish loader transfers")?;
    }
    let load_s = started.elapsed().as_secs_f64();
    let first = Instant::now();
    for l in &mut launches {
        l.execute(&s.driver, s.stream)?;
    }
    // SAFETY: The same stream and allocations remain live until cleanup, copies
    // below run only after explicit stream synchronization. Capture records the
    // immutable allocations/ABI values; input changes alter data, never addresses.
    unsafe {
        check((s.driver.stream_sync)(s.stream), "first launch synchronize")?;
        let first_launch_wall_s = first.elapsed().as_secs_f64();
        let output_spec = x
            .manifest
            .buffers
            .iter()
            .find(|b| b.name == x.manifest.validation.output)
            .unwrap();
        let output = device_buffers[&output_spec.name];
        let mut expected = vec![0u8; output_spec.bytes()?];
        check(
            (s.driver.download)(expected.as_mut_ptr().cast(), output, expected.len()),
            "download first output",
        )?;
        let reference = floats(&x.files[&x.manifest.validation.reference.file], Dtype::F32);
        let (relative_l2, max_abs_error) =
            errors(&floats(&expected, output_spec.dtype), &reference)?;
        if relative_l2 > x.manifest.validation.relative_l2_tolerance {
            return Err(format!(
                "Projection relative L2 {relative_l2} exceeds {}",
                x.manifest.validation.relative_l2_tolerance
            ));
        }
        let capture = Instant::now();
        check(
            (s.driver.capture_begin)(s.stream, 1),
            "begin thread-local capture",
        )?;
        s.capturing = true;
        for l in &mut launches {
            l.execute(&s.driver, s.stream)?;
        }
        let code = (s.driver.capture_end)(s.stream, &mut s.graph);
        s.capturing = false;
        check(code, "end capture")?;
        check(
            (s.driver.graph_instantiate)(&mut s.exec, s.graph, 0),
            "instantiate graph",
        )?;
        let graph_capture_s = capture.elapsed().as_secs_f64();
        let mut observed = vec![0u8; expected.len()];
        check(
            (s.driver.memset)(output, 255, expected.len(), s.stream),
            "poison output",
        )?;
        check(
            (s.driver.graph_launch)(s.exec, s.stream),
            "replay poisoned output",
        )?;
        check((s.driver.stream_sync)(s.stream), "replay synchronize")?;
        check(
            (s.driver.download)(observed.as_mut_ptr().cast(), output, observed.len()),
            "download replay",
        )?;
        if expected != observed {
            return Err("Graph replay did not reproduce first output".into());
        }
        let input_spec = x
            .manifest
            .buffers
            .iter()
            .find(|b| b.name == x.manifest.validation.zero_input)
            .unwrap();
        let input = device_buffers[&input_spec.name];
        check(
            (s.driver.memset)(input, 0, input_spec.bytes()?, s.stream),
            "change actual input to zero",
        )?;
        check(
            (s.driver.memset)(output, 255, observed.len(), s.stream),
            "poison changed-input output",
        )?;
        check(
            (s.driver.graph_launch)(s.exec, s.stream),
            "replay changed input",
        )?;
        check(
            (s.driver.stream_sync)(s.stream),
            "changed-input synchronize",
        )?;
        check(
            (s.driver.download)(observed.as_mut_ptr().cast(), output, observed.len()),
            "download changed-input output",
        )?;
        if floats(&observed, output_spec.dtype)
            .iter()
            .any(|v| *v != 0.0)
        {
            return Err("Graph did not compute zero output from changed input".into());
        }
        let original = &x.files[&input_spec.data.as_ref().unwrap().file];
        check(
            (s.driver.upload)(input, original.as_ptr().cast(), original.len()),
            "restore actual input",
        )?;
        check((s.driver.context_sync)(), "finish restored input transfer")?;
        check(
            (s.driver.memset)(output, 255, observed.len(), s.stream),
            "poison restored output",
        )?;
        check(
            (s.driver.graph_launch)(s.exec, s.stream),
            "replay restored input",
        )?;
        check((s.driver.stream_sync)(s.stream), "restore synchronize")?;
        check(
            (s.driver.download)(observed.as_mut_ptr().cast(), output, observed.len()),
            "download restored output",
        )?;
        if observed != expected {
            return Err("Restored-input graph differs from first output".into());
        }
        let (mut begin, mut end) = (ptr::null_mut(), ptr::null_mut());
        check((s.driver.event_create)(&mut begin, 0), "create begin event")?;
        s.events.push(begin);
        check((s.driver.event_create)(&mut end, 0), "create end event")?;
        s.events.push(end);
        let mut times = vec![];
        for _ in 0..3 {
            check(
                (s.driver.event_record)(begin, s.stream),
                "record begin event",
            )?;
            for _ in 0..x.manifest.validation.repetitions {
                check(
                    (s.driver.graph_launch)(s.exec, s.stream),
                    "timed graph replay",
                )?
            }
            check((s.driver.event_record)(end, s.stream), "record end event")?;
            check((s.driver.event_sync)(end), "synchronize end event")?;
            let mut ms = 0f32;
            check(
                (s.driver.event_elapsed)(&mut ms, begin, end),
                "elapsed events",
            )?;
            times.push(ms / x.manifest.validation.repetitions as f32);
        }
        times.sort_by(f32::total_cmp);
        let report = RunReport {
            manifest_sha256: x.manifest_sha256,
            device: DeviceInfo {
                name: name
                    .iter()
                    .take_while(|c| **c != 0)
                    .map(|c| char::from(*c))
                    .collect(),
                sm: [major, minor],
                driver_version: version,
                total_bytes: total,
            },
            buffer_bytes: x.buffer_bytes,
            artifact_validation_s: x.validation_s,
            device_load_s: load_s,
            first_launch_wall_s,
            graph_capture_s,
            hot_graph_cuda_event_ms: times[1],
            relative_l2,
            max_abs_error,
            changed_input_replay_verified: true,
            output_poisoning_replay_verified: true,
            output_elements: reference.len(),
            repetitions: x.manifest.validation.repetitions,
            scope: "Fixed-shape real projection AOT fixture; not a model loader, model TPS or concurrent state validation",
        };
        s.cleanup()?;
        Ok(report)
    }
}

pub(crate) mod executor;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp16_conversion_handles_subnormal_and_special_values() {
        assert_eq!(half_to_float(0x3c00), 1.0);
        assert_eq!(half_to_float(0xbc00), -1.0);
        assert_eq!(half_to_float(1), 2f32.powi(-24));
        assert_eq!(half_to_float(0x0400), 2f32.powi(-14));
        assert!(half_to_float(0x7c00).is_infinite());
        assert!(half_to_float(0x7e00).is_nan());
        assert!(half_to_float(0x8000).is_sign_negative());
    }
    #[test]
    fn bf16_conversion_preserves_bits_and_special_values() {
        for bits in [
            0u16, 1, 0x0080, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x8000,
        ] {
            let output = floats(&bits.to_le_bytes(), Dtype::Bf16);
            assert_eq!(output[0].to_bits(), u32::from(bits) << 16);
        }
    }
    #[test]
    fn rejects_nonfinite_numerical_validation() {
        assert!(errors(&[f32::NAN], &[1.0]).is_err());
        assert_eq!(errors(&[1.0, 2.0], &[1.0, 2.0]).unwrap(), (0.0, 0.0));
    }
}
