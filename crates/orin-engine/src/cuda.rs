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
#[cfg(test)]
#[path = "mtp_gpu_tests.rs"]
mod mtp_gpu_tests;
type Handle = *mut c_void;

struct Driver {
    _library: Library,
    init: unsafe extern "C" fn(u32) -> i32,
    version: unsafe extern "C" fn(*mut i32) -> i32,
    device: unsafe extern "C" fn(*mut i32, i32) -> i32,
    name: unsafe extern "C" fn(*mut c_char, i32, i32) -> i32,
    attribute: unsafe extern "C" fn(*mut i32, i32, i32) -> i32,
    total: unsafe extern "C" fn(*mut usize, i32) -> i32,
    current: unsafe extern "C" fn(*mut Handle) -> i32,
    retain: unsafe extern "C" fn(*mut Handle, i32) -> i32,
    release: unsafe extern "C" fn(i32) -> i32,
    set_current: unsafe extern "C" fn(Handle) -> i32,
    context_sync: unsafe extern "C" fn() -> i32,
    stream_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    stream_sync: unsafe extern "C" fn(Handle) -> i32,
    stream_destroy: unsafe extern "C" fn(Handle) -> i32,
    memory_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
    alloc: unsafe extern "C" fn(*mut u64, usize) -> i32,
    free: unsafe extern "C" fn(u64) -> i32,
    upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    copy: unsafe extern "C" fn(u64, u64, usize, Handle) -> i32,
    download: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    memset: unsafe extern "C" fn(u64, u8, usize, Handle) -> i32,
    module_load: unsafe extern "C" fn(*mut Handle, *const c_void) -> i32,
    module_unload: unsafe extern "C" fn(Handle) -> i32,
    function: unsafe extern "C" fn(*mut Handle, Handle, *const c_char) -> i32,
    function_get: unsafe extern "C" fn(*mut i32, i32, Handle) -> i32,
    function_set: unsafe extern "C" fn(Handle, i32, i32) -> i32,
    launch: unsafe extern "C" fn(
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
    capture_begin: unsafe extern "C" fn(Handle, i32) -> i32,
    capture_end: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    graph_instantiate: unsafe extern "C" fn(*mut Handle, Handle, u64) -> i32,
    graph_launch: unsafe extern "C" fn(Handle, Handle) -> i32,
    graph_destroy: unsafe extern "C" fn(Handle) -> i32,
    graph_exec_destroy: unsafe extern "C" fn(Handle) -> i32,
    event_create: unsafe extern "C" fn(*mut Handle, u32) -> i32,
    event_record: unsafe extern "C" fn(Handle, Handle) -> i32,
    event_sync: unsafe extern "C" fn(Handle) -> i32,
    event_elapsed: unsafe extern "C" fn(*mut f32, Handle, Handle) -> i32,
    event_destroy: unsafe extern "C" fn(Handle) -> i32,
}

impl Driver {
    fn load() -> Result<Self> {
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
fn check(code: i32, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
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
struct Session {
    driver: Driver,
    device: i32,
    previous: Handle,
    retained: bool,
    stream: Handle,
    buffers: Vec<u64>,
    modules: Vec<Handle>,
    events: Vec<Handle>,
    graph: Handle,
    exec: Handle,
    graphs: Vec<(Handle, Handle)>,
    capturing: bool,
}
impl Session {
    fn new(driver: Driver) -> Self {
        Self {
            driver,
            device: 0,
            previous: ptr::null_mut(),
            retained: false,
            stream: ptr::null_mut(),
            buffers: vec![],
            modules: vec![],
            events: vec![],
            graph: ptr::null_mut(),
            exec: ptr::null_mut(),
            graphs: vec![],
            capturing: false,
        }
    }
    fn cleanup(&mut self) -> Result<()> {
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
            for (graph, exec) in self.graphs.drain(..) {
                record(
                    (self.driver.graph_exec_destroy)(exec),
                    "destroy model graph exec",
                );
                record((self.driver.graph_destroy)(graph), "destroy model graph");
            }
            for event in self.events.drain(..) {
                record((self.driver.event_destroy)(event), "destroy event");
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

enum Value {
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
struct Launch<'a> {
    spec: &'a Kernel,
    function: Handle,
    values: Vec<Value>,
}
impl Launch<'_> {
    fn execute(&mut self, driver: &Driver, stream: Handle) -> Result<()> {
        let mut args: Vec<*mut c_void> = self.values.iter_mut().map(Value::address).collect();
        let k = self.spec;
        // SAFETY: Values remain at stable Vec addresses until cuLaunchKernel has
        // copied each parameter. Device pointers refer to live session buffers;
        // generated ABI order/types are supplied by the verified manifest.
        // The module and explicit stream outlive all queued/captured work.
        unsafe {
            check(
                (driver.launch)(
                    self.function,
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
}

fn floats(raw: &[u8], dtype: Dtype) -> Vec<f32> {
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

/// Execute an architecture adapter's explicit programs with stable allocations.
/// All handles, pointers and argument backing storage stay inside this session.
/// A CUDA model is thread-affine. Construct, use and drop it on its worker.
pub(crate) struct ModelRuntime {
    pub(crate) manifest: crate::model::Manifest,
    session: Session,
    pointers: BTreeMap<String, u64>,
    sizes: BTreeMap<String, usize>,
    graphs: BTreeMap<String, Handle>,
    stats: LoadStats,
    pub(crate) speculation_statistics: Option<crate::mtp::Statistics>,
}
struct LoadStats {
    manifest_sha256: String,
    device: DeviceInfo,
    load_to_ready_s: f64,
    weight_io_hash_s: f64,
    weight_upload_s: f64,
    module_load_bind_s: f64,
    graph_capture_s: f64,
    buffer_bytes: usize,
}
impl ModelRuntime {
    pub(crate) fn load(manifest_path: &std::path::Path) -> Result<Self> {
        use crate::model::{Manifest, Operation};
        use sha2::{Digest, Sha256};
        use std::fs;
        let started = Instant::now();
        let manifest_raw = fs::read(manifest_path).map_err(|e| e.to_string())?;
        let manifest_sha256 = format!("{:x}", Sha256::digest(&manifest_raw));
        let manifest: Manifest =
            serde_json::from_slice(&manifest_raw).map_err(|e| e.to_string())?;
        let buffer_bytes = manifest.validate()?;
        let canonical_manifest = manifest_path.canonicalize().map_err(|e| e.to_string())?;
        let base = canonical_manifest
            .parent()
            .ok_or("Manifest needs parent directory")?;
        let mut s = Session::new(Driver::load()?);
        let mut pointers = BTreeMap::new();
        let mut sizes = BTreeMap::new();
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
        for b in &manifest.buffers {
            let bytes = b.bytes()?;
            let mut address = 0;
            // SAFETY: Allocation length is checked by manifest validation and owned
            // by Session before any fallible operation can return.
            unsafe {
                check(
                    (s.driver.alloc)(&mut address, bytes),
                    &format!("allocate {}", b.name),
                )?;
            }
            s.buffers.push(address);
            if address % b.alignment != 0 {
                return Err("Model allocation alignment".into());
            }
            pointers.insert(b.name.clone(), address);
            sizes.insert(b.name.clone(), bytes);
            if let Some(id) = &b.data {
                let t = Instant::now();
                let raw = crate::artifact::read_identity(base, id)?;
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
            } else {
                // SAFETY: Whole allocation is owned and not in use yet.
                unsafe {
                    check(
                        (s.driver.memset)(address, 0, bytes, s.stream),
                        "initialize workspace",
                    )?;
                }
            }
        }
        let module_started = Instant::now();
        let mut modules = BTreeMap::<String, Handle>::new();
        let mut functions = BTreeMap::<(String, String), Handle>::new();
        let mut launches = BTreeMap::new();
        for k in &manifest.kernels {
            let module = if let Some(m) = modules.get(&k.module.file) {
                *m
            } else {
                let image = crate::artifact::read_identity(base, &k.module)?;
                crate::artifact::read_identity(base, &k.source)?;
                crate::artifact::read_identity(base, &k.host_abi)?;
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
        let capture_started = Instant::now();
        let mut graphs = BTreeMap::new();
        for (phase, ops) in &manifest.programs {
            // SAFETY: Capture records operations against stable owned buffers.
            // Each copy/zero range and graph kernel reference was validated above.
            unsafe {
                check((s.driver.capture_begin)(s.stream, 0), "begin model capture")?;
            }
            s.capturing = true;
            for op in ops {
                match op {
                    Operation::Kernel { name } => launches
                        .get_mut(name)
                        .ok_or("Unbound kernel")?
                        .execute(&s.driver, s.stream)?,
                    Operation::Copy {
                        source,
                        destination,
                        bytes,
                    } => {
                        // SAFETY: Distinct allocations, validated byte ranges, same
                        // stream ordering as their producers and consumers.
                        unsafe {
                            check(
                                (s.driver.copy)(
                                    pointers[destination],
                                    pointers[source],
                                    *bytes,
                                    s.stream,
                                ),
                                "capture state copy",
                            )?;
                        }
                    }
                    Operation::Zero { destination, bytes } => {
                        // SAFETY: Validated writable range, ordered before use.
                        unsafe {
                            check(
                                (s.driver.memset)(pointers[destination], 0, *bytes, s.stream),
                                "capture residual reset",
                            )?;
                        }
                    }
                }
            }
            // SAFETY: Session records graph immediately, so all later failures
            // release capture products. Instantiate does not execute model state.
            unsafe {
                check(
                    (s.driver.capture_end)(s.stream, &mut s.graph),
                    "end model capture",
                )?;
                s.capturing = false;
                check(
                    (s.driver.graph_instantiate)(&mut s.exec, s.graph, 0),
                    "instantiate model graph",
                )?;
            }
            graphs.insert(phase.clone(), s.exec);
            s.graphs.push((s.graph, s.exec));
            s.graph = ptr::null_mut();
            s.exec = ptr::null_mut();
        }
        let graph_capture_s = capture_started.elapsed().as_secs_f64();
        let load_to_ready_s = started.elapsed().as_secs_f64();
        eprintln!("MODEL READY after {load_to_ready_s:.3}s; {buffer_bytes} buffer bytes");
        let device = DeviceInfo {
            name: String::from_utf8_lossy(&name)
                .trim_end_matches('\0')
                .to_string(),
            sm: [major, minor],
            driver_version: version,
            total_bytes: total,
        };
        Ok(Self {
            manifest,
            session: s,
            pointers,
            sizes,
            graphs,
            stats: LoadStats {
                manifest_sha256,
                device,
                load_to_ready_s,
                weight_io_hash_s,
                weight_upload_s,
                module_load_bind_s,
                graph_capture_s,
                buffer_bytes,
            },
            speculation_statistics: None,
        })
    }
    fn sync(&self) -> Result<()> {
        // SAFETY: This thread owns the live stream and all graph allocations.
        unsafe {
            check(
                (self.session.driver.stream_sync)(self.session.stream),
                "model synchronize",
            )
        }
    }
    fn launch_program(&self, name: &str) -> Result<()> {
        let graph = *self
            .graphs
            .get(name)
            .ok_or_else(|| format!("Missing program {name}"))?;
        // SAFETY: Graphs and their stable addresses live in this thread's session.
        unsafe {
            check(
                (self.session.driver.graph_launch)(graph, self.session.stream),
                "model graph",
            )?;
        }
        self.sync()
    }
    fn upload_ids(&self, name: &str, ids: &[u32]) -> Result<()> {
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
    fn read_control(&self, name: &str) -> Result<i32> {
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
    fn upload_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
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
    fn read_controls(&self, name: &str, count: usize) -> Result<Vec<u32>> {
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
    fn mtp_capture(&self, spec: &crate::mtp::Spec, tokens: usize) -> Result<()> {
        let plan = spec
            .capture_plans
            .iter()
            .find(|p| p.tokens == tokens)
            .ok_or("Missing MTP hidden capture shape")?;
        self.launch_program(&plan.program)
    }
    fn mtp_warm(
        &self,
        spec: &crate::mtp::Spec,
        shifted_ids: &[u32],
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let mut offset = 0;
        while offset < shifted_ids.len() {
            if cancelled() {
                return Err("Request cancelled during MTP warm/refresh".into());
            }
            let remaining = shifted_ids.len() - offset;
            let plan = spec
                .warm_plans
                .iter()
                .filter(|p| p.tokens <= remaining)
                .max_by_key(|p| p.tokens)
                .ok_or("No compatible MTP warm plan")?;
            self.upload_ids(&spec.input, &shifted_ids[offset..offset + plan.tokens])?;
            self.launch_program(&plan.program)?;
            offset += plan.tokens;
            if offset == shifted_ids.len() {
                self.launch_program(&plan.head_program)?;
            }
        }
        Ok(())
    }
    fn mtp_generate(
        &mut self,
        spec: &crate::mtp::Spec,
        input: &[u32],
        limit: usize,
        cancelled: &impl Fn() -> bool,
        emit: &mut impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        use std::time::Instant;
        let vocab = self.manifest.vocab;
        let mut stats = crate::mtp::Statistics::default();
        let first = self.read_control(&self.manifest.token)?;
        if self.read_control(&self.manifest.status)? != 0 || first < 0 || first as usize >= vocab {
            return Err("Invalid initial target token for MTP".into());
        }
        let mut pending = first as u32;
        let mut shifted = input[1..].to_vec();
        shifted.push(pending);
        let at = Instant::now();
        self.mtp_warm(spec, &shifted, cancelled)?;
        stats.initial_warm_s = at.elapsed().as_secs_f64();
        let mut generated = 1;
        if !emit(pending) || limit == 1 {
            stats.committed_tokens = generated;
            self.speculation_statistics = Some(stats);
            return Ok(generated);
        }
        while generated < limit {
            if cancelled() {
                return Err("Request cancelled during MTP decode".into());
            }
            let position = self.read_control(&self.manifest.position)? as usize;
            let capacity = self.manifest.max_context.saturating_sub(position);
            let remaining_outputs = limit - generated;
            let plan = spec
                .verification_plans
                .iter()
                .filter(|p| {
                    p.tokens <= capacity
                        && p.tokens <= remaining_outputs
                        && p.tokens <= spec.default_verification_tokens
                })
                .max_by_key(|p| p.tokens);
            let Some(plan) = plan else {
                // Final single-token tail or a context edge. The ordinary
                // target graph preserves existing sampling/state semantics.
                self.upload_ids(&self.manifest.token, &[pending])?;
                self.launch_program("decode")?;
                self.mtp_capture(spec, 1)?;
                let selected = self.read_control(&self.manifest.token)?;
                if self.read_control(&self.manifest.status)? != 0
                    || selected < 0
                    || selected as usize >= vocab
                {
                    return Err("Invalid MTP target fallback token".into());
                }
                pending = selected as u32;
                generated += 1;
                let stopped = !emit(pending);
                let at = Instant::now();
                self.mtp_warm(spec, &[pending], cancelled)?;
                stats.refresh_s += at.elapsed().as_secs_f64();
                if stopped {
                    break;
                }
                continue;
            };
            let at = Instant::now();
            let mut drafts = Vec::with_capacity(plan.tokens - 1);
            for i in 0..plan.tokens - 1 {
                if cancelled() {
                    return Err("Request cancelled during MTP draft".into());
                }
                if i != 0 {
                    self.launch_program(&spec.draft_program)?;
                }
                let token = self.read_control(&spec.token)?;
                if self.read_control(&spec.status)? != 0 || token < 0 || token as usize >= vocab {
                    return Err("Invalid MTP draft token".into());
                }
                drafts.push(token as u32);
            }
            stats.draft_s += at.elapsed().as_secs_f64();
            stats.proposed_tokens += drafts.len();
            let mut verification_input = vec![pending];
            verification_input.extend_from_slice(&drafts);
            let at = Instant::now();
            self.upload_ids(&self.manifest.input, &verification_input)?;
            self.launch_program(&plan.program)?;
            self.launch_program(&plan.capture_program)?;
            let target = self.read_controls(&spec.verification_tokens, plan.tokens)?;
            let status = self.read_controls(&spec.verification_status, plan.tokens)?;
            if target.iter().any(|&x| x as usize >= vocab) || status.iter().any(|&x| x != 0) {
                return Err("Invalid MTP target verification result".into());
            }
            stats.verification_s += at.elapsed().as_secs_f64();
            stats.rounds += 1;
            let mut committed = crate::mtp::greedy_commit(&drafts, &target)?;
            let accepted_drafts = committed.len() - 1;
            let mut emitted = 0;
            let mut stopped = false;
            for &token in &committed {
                emitted += 1;
                generated += 1;
                if !emit(token) {
                    stopped = true;
                    break;
                }
            }
            committed.truncate(emitted);
            stats.accepted_draft_tokens += accepted_drafts.min(emitted);
            let at = Instant::now();
            if emitted < plan.tokens {
                self.upload_ids(&spec.accepted_inputs, &[emitted as u32])?;
                self.launch_program(&plan.restore_program)?;
                self.upload_ids(&self.manifest.position, &[(position + emitted) as u32])?;
                self.upload_ids(&spec.target_length, &[(position + emitted) as u32])?;
            }
            stats.restore_s += at.elapsed().as_secs_f64();
            pending = *committed
                .last()
                .ok_or("MTP round committed no target token")?;
            // Keep the already correct MTP slot at position-1. Replace slots
            // from position onward with true target hidden states paired with
            // accepted tokens and the target correction/bonus. The final row
            // also produces the first draft of the next round.
            let at = Instant::now();
            self.upload_ids(&spec.position, &[position as u32])?;
            self.mtp_warm(spec, &committed, cancelled)?;
            stats.refresh_s += at.elapsed().as_secs_f64();
            if stopped || generated == limit {
                break;
            }
        }
        if self.read_control(&self.manifest.position)? as usize != input.len() + generated - 1 {
            return Err("MTP committed position mismatch".into());
        }
        self.upload_ids(&self.manifest.token, &[pending])?;
        stats.committed_tokens = generated;
        self.speculation_statistics = Some(stats);
        Ok(generated)
    }
    fn prepare_visual(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let Some(v) = &self.manifest.vision else {
            return if images.is_empty() {
                Ok(())
            } else {
                Err("Model has no vision adapter".into())
            };
        };
        let (index, positions) = v.layout(input, images, self.manifest.max_context)?;
        self.upload_bytes(
            &v.feature_index,
            &index
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.upload_ids(&v.mrope_positions, &positions)?;
        let mut offset = 0;
        for image in images {
            if cancelled() {
                return Err("Request cancelled during image encoding".into());
            }
            let features = v.feature_count(image)?;
            let patches = image.grid_height * image.grid_width;
            let plan = v
                .plans
                .iter()
                .filter(|p| p.patches >= patches)
                .min_by_key(|p| p.patches)
                .ok_or("No image graph")?;
            let raw: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|&x| match v.dtype {
                    crate::vision::Precision::F16 => half::f16::from_f32(x).to_bits().to_le_bytes(),
                    crate::vision::Precision::Bf16 => {
                        half::bf16::from_f32(x).to_bits().to_le_bytes()
                    }
                })
                .collect();
            self.upload_bytes(&v.pixels, &raw)?;
            self.upload_ids(
                &v.grid,
                &[image.grid_height as u32, image.grid_width as u32],
            )?;
            self.upload_ids(&v.length, &[patches as u32])?;
            self.launch_program(&plan.program)?;
            // SAFETY: Validated feature counts cover both nonoverlapping allocations;
            // the producing image graph completed and the stream owns this copy.
            unsafe {
                check(
                    (self.session.driver.copy)(
                        self.pointers[&v.features] + (offset * v.hidden * 2) as u64,
                        self.pointers[&v.output],
                        features * v.hidden * 2,
                        self.session.stream,
                    ),
                    "image features",
                )?;
            }
            self.sync()?;
            offset += features;
        }
        Ok(())
    }
    /// Full chunks use prefill graphs. A remaining tail is teacher-forced through
    /// the M=1 graph: no dummy IDs enter attention, convolution or GDN state.
    pub(crate) fn generate(
        &mut self,
        input: &[u32],
        images: Option<&[crate::vision::ImageInput]>,
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        let m = &self.manifest;
        if input.is_empty()
            || limit == 0
            || input
                .len()
                .checked_add(limit)
                .is_none_or(|n| n > m.max_context)
            || input.iter().any(|&id| id as usize >= m.vocab)
        {
            return Err("Invalid input, generation length or context budget".into());
        }
        options.validate()?;
        self.speculation_statistics = None;
        let mtp = self
            .manifest
            .mtp
            .as_ref()
            .filter(|_| options.is_greedy() && images.is_none_or(|images| images.is_empty()))
            .cloned();
        self.sync()?;
        for name in &m.reset_buffers {
            // SAFETY: Previous request is complete; these validated buffers are writable.
            unsafe {
                check(
                    (self.session.driver.memset)(
                        self.pointers[name],
                        0,
                        self.sizes[name],
                        self.session.stream,
                    ),
                    "reset request",
                )?;
            }
        }
        self.sync()?;
        self.prepare_visual(input, images.unwrap_or(&[]), &cancelled)?;
        let mut offset = 0;
        let mut last_head = None;
        while offset < input.len() {
            if cancelled() {
                return Err("Request cancelled during prefill".into());
            }
            let remaining = input.len() - offset;
            let plan = m
                .prefill_plans
                .iter()
                .filter(|p| p.chunk_tokens <= remaining)
                .max_by_key(|p| p.chunk_tokens);
            let selected = plan
                .map(|p| {
                    (
                        p.chunk_tokens,
                        p.prefill_program.as_str(),
                        p.head_program.as_str(),
                    )
                })
                .or_else(|| {
                    (m.prefill_plans.is_empty() && m.chunk_tokens <= remaining).then_some((
                        m.chunk_tokens,
                        "prefill",
                        "head",
                    ))
                });
            if let Some((chunk, program, head)) = selected {
                self.upload_ids(&m.input, &input[offset..offset + chunk])?;
                self.launch_program(program)?;
                if let Some(spec) = &mtp {
                    self.mtp_capture(spec, chunk)?;
                }
                offset += chunk;
                last_head = Some(head);
            } else {
                for id in &input[offset..] {
                    if cancelled() {
                        return Err("Request cancelled during prefill".into());
                    }
                    self.upload_ids(&m.token, std::slice::from_ref(id))?;
                    self.launch_program("decode")?;
                    if let Some(spec) = &mtp {
                        self.mtp_capture(spec, 1)?;
                    }
                }
                offset = input.len();
                last_head = None;
            }
        }
        if let Some(head) = last_head {
            self.launch_program(head)?;
        }
        if self.read_control(&m.position)? as usize != input.len() {
            return Err("Prefill position mismatch".into());
        }
        if let Some(spec) = &mtp {
            return self.mtp_generate(spec, input, limit, &cancelled, &mut emit);
        }
        let mut generated = 0;
        let mut history = input.to_vec();
        for step in 0..limit {
            if self.read_control(&m.status)? != 0 {
                return Err("Model token status failure".into());
            }
            let token = if options.is_greedy() {
                let value = self.read_control(&m.token)?;
                if value < 0 || value as usize >= m.vocab {
                    return Err("Selected token outside vocabulary".into());
                }
                value as u32
            } else {
                let spec = m
                    .buffers
                    .iter()
                    .find(|b| b.name == m.logits)
                    .ok_or("Missing logits")?;
                let mut raw = vec![0u8; spec.bytes()?];
                // SAFETY: The graph is complete and both buffers cover the logits tensor.
                unsafe {
                    check(
                        (self.session.driver.download)(
                            raw.as_mut_ptr().cast(),
                            self.pointers[&m.logits],
                            raw.len(),
                        ),
                        "sampling logits",
                    )?;
                }
                crate::sampling::sample(&floats(&raw, spec.dtype), &history, options, step)?
            };
            generated += 1;
            history.push(token);
            if !emit(token) {
                break;
            }
            if step + 1 < limit {
                // Also overwrite the greedy graph selection for stochastic sampling.
                self.upload_ids(&m.token, &[token])?;
                self.launch_program("decode")?;
            }
        }
        if self.read_control(&m.position)? as usize != input.len() + generated - 1 {
            return Err("Decode position mismatch".into());
        }
        Ok(generated)
    }
    fn benchmark(mut self, requests: crate::model::Requests) -> Result<crate::model::Report> {
        use crate::model::{Report, RequestReport};
        use std::fs;
        self.manifest.validate_requests(&requests)?;
        self.prepare_visual(&[], &[], &|| false)?;
        let manifest = &self.manifest;
        let s = &mut self.session;
        let pointers = &self.pointers;
        let sizes = &self.sizes;
        let graphs = &self.graphs;
        let stats = self.stats;
        let output_directory = requests
            .logits_output
            .as_ref()
            .map(std::path::PathBuf::from);
        if let Some(p) = &output_directory {
            fs::create_dir(p).map_err(|e| format!("New logits directory {}: {e}", p.display()))?;
        }
        let mut reports = vec![];
        let logits_spec = manifest
            .buffers
            .iter()
            .find(|b| b.name == manifest.logits)
            .ok_or("Logits missing")?;
        let dump_logits = |session: &Session,
                           q: &crate::model::Request,
                           step: usize,
                           files: &mut Vec<String>|
         -> Result<()> {
            if !q.logits_steps.contains(&step) {
                return Ok(());
            }
            let mut raw = vec![0u8; logits_spec.bytes()?];
            // SAFETY: Caller synchronized this stream, host range and device
            // allocation cover the complete logits tensor.
            unsafe {
                check(
                    (session.driver.download)(
                        raw.as_mut_ptr().cast(),
                        pointers[&manifest.logits],
                        raw.len(),
                    ),
                    "download logits",
                )?;
            }
            let values = floats(&raw, logits_spec.dtype);
            if values.iter().any(|v| !v.is_finite()) {
                return Err("Nonfinite logits".into());
            }
            let path = output_directory
                .as_ref()
                .ok_or("Logits directory absent")?
                .join(format!("{}-{step}.f32", q.id));
            let mut file = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .map_err(|e| e.to_string())?;
            use std::io::Write;
            let encoded: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            file.write_all(&encoded).map_err(|e| e.to_string())?;
            files.push(path.display().to_string());
            Ok(())
        };
        let read_i32 = |session: &Session, name: &str| -> Result<i32> {
            let mut value = 0i32;
            // SAFETY: Control allocations and stack destination are >=4 bytes.
            // Every caller has synchronized the producing stream.
            unsafe {
                check(
                    (session.driver.download)((&mut value as *mut i32).cast(), pointers[name], 4),
                    "download control",
                )?;
            }
            Ok(value)
        };
        for q in &requests.requests {
            let (chunk_tokens, prefill_program, head_program) =
                manifest.select_prefill_plan(q.input_tokens.len())?;
            eprintln!(
                "request {}: {} input, {} output tokens",
                q.id,
                q.input_tokens.len(),
                q.max_new_tokens
            );
            let reset_start = Instant::now();
            for b in &manifest.reset_buffers {
                // SAFETY: Validated writable buffers; previous request is complete.
                unsafe {
                    check(
                        (s.driver.memset)(pointers[b], 0, sizes[b], s.stream),
                        "reset model state",
                    )?;
                }
            }
            // SAFETY: Complete reset before timing the actual request.
            unsafe {
                check((s.driver.stream_sync)(s.stream), "state reset complete")?;
            }
            let reset_s = reset_start.elapsed().as_secs_f64();
            let request_start = Instant::now();
            let mut tokens = vec![];
            let mut logits_files = vec![];
            for chunk in q.input_tokens.chunks(chunk_tokens) {
                // SAFETY: Input IDs fit this allocation and were range checked.
                // Previous graph is complete before overwriting its input buffer.
                unsafe {
                    check((s.driver.stream_sync)(s.stream), "prefill input dependency")?;
                    check(
                        (s.driver.upload)(
                            pointers[&manifest.input],
                            chunk.as_ptr().cast(),
                            chunk.len() * 4,
                        ),
                        "prefill token upload",
                    )?;
                    check((s.driver.context_sync)(), "prefill upload dependency")?;
                    check(
                        (s.driver.graph_launch)(graphs[prefill_program], s.stream),
                        "prefill graph",
                    )?;
                }
            }
            // SAFETY: Synchronize before head timing and output download.
            unsafe {
                check((s.driver.stream_sync)(s.stream), "prefill complete")?;
            }
            let prefill_s = request_start.elapsed().as_secs_f64();
            let head_start = Instant::now();
            // SAFETY: Head consumes last prefill graph's buffers on the same stream.
            unsafe {
                check(
                    (s.driver.graph_launch)(graphs[head_program], s.stream),
                    "head graph",
                )?;
                check((s.driver.stream_sync)(s.stream), "head complete")?;
            }
            if read_i32(s, &manifest.status)? != 0 {
                return Err("Token selection status failure".into());
            }
            let token = read_i32(s, &manifest.token)?;
            if token < 0 || token as usize >= manifest.vocab {
                return Err("Invalid selected token".into());
            }
            tokens.push(token as u32);
            let head_s = head_start.elapsed().as_secs_f64();
            let ttft_s = request_start.elapsed().as_secs_f64();
            dump_logits(s, q, 0, &mut logits_files)?;
            let decode_start = Instant::now();
            for step in 1..q.max_new_tokens {
                if let Some(id) = q.forced_tokens.get(step - 1) {
                    // SAFETY: Previous graph completed, valid token ID and 4-byte
                    // destination; explicit synchronization separates upload/replay.
                    unsafe {
                        check(
                            (s.driver.upload)(
                                pointers[&manifest.token],
                                (id as *const u32).cast(),
                                4,
                            ),
                            "teacher-force token",
                        )?;
                        check((s.driver.context_sync)(), "teacher-force dependency")?;
                    }
                }
                // SAFETY: Stable graph addresses, state updates and next token
                // device-to-device copy are ordered inside this graph.
                unsafe {
                    check(
                        (s.driver.graph_launch)(graphs["decode"], s.stream),
                        "decode graph",
                    )?;
                    check((s.driver.stream_sync)(s.stream), "decode complete")?;
                }
                if read_i32(s, &manifest.status)? != 0 {
                    return Err("Decode token status failure".into());
                }
                let token = read_i32(s, &manifest.token)?;
                if token < 0 || token as usize >= manifest.vocab {
                    return Err("Invalid decode token".into());
                }
                tokens.push(token as u32);
                dump_logits(s, q, step, &mut logits_files)?;
            }
            let decode_s = decode_start.elapsed().as_secs_f64();
            let final_position = read_i32(s, &manifest.position)?;
            if final_position < 0
                || final_position as usize != q.input_tokens.len() + q.max_new_tokens - 1
            {
                return Err("Model position advancement mismatch".into());
            }
            let diagnostic = !q.logits_steps.is_empty() || !q.forced_tokens.is_empty();
            let r = RequestReport {
                prefill_chunk_tokens: chunk_tokens,
                prefill_program: prefill_program.to_string(),
                id: q.id.clone(),
                input_tokens: q.input_tokens.len(),
                output_tokens: tokens,
                reset_s,
                prefill_s,
                head_s,
                ttft_s,
                decode_s,
                prefill_tps: q.input_tokens.len() as f64 / prefill_s,
                decode_tps: (q.max_new_tokens > 1)
                    .then_some((q.max_new_tokens - 1) as f64 / decode_s),
                final_position: final_position as usize,
                logits_files,
                diagnostic,
            };
            eprintln!(
                "{} prefill {:.2} TPS, decode {:?} TPS",
                r.id, r.prefill_tps, r.decode_tps
            );
            reports.push(r);
            if let Ok(path) = std::env::var("ORIN_MODEL_PROGRESS") {
                let temporary = format!("{path}.tmp");
                fs::write(
                    &temporary,
                    serde_json::to_vec_pretty(&reports).map_err(|e| e.to_string())?,
                )
                .map_err(|e| e.to_string())?;
                fs::rename(&temporary, &path).map_err(|e| e.to_string())?;
            }
        }
        s.cleanup()?;
        Ok(Report {
            manifest_sha256: stats.manifest_sha256,
            model: manifest.model.clone(),
            device: stats.device,
            load_to_ready_s: stats.load_to_ready_s,
            weight_io_hash_s: stats.weight_io_hash_s,
            weight_upload_s: stats.weight_upload_s,
            module_load_bind_s: stats.module_load_bind_s,
            graph_capture_s: stats.graph_capture_s,
            buffer_bytes: stats.buffer_bytes,
            weight_bytes: manifest.weight_bytes,
            effective_weight_bits: 8.0 * manifest.weight_bytes as f64
                / manifest.weight_parameters as f64,
            weight_scope: manifest.weight_scope.clone(),
            requests: reports,
            seed: 20261002,
            timing_scope: "Rust wall clock; prefill includes input copies and synchronization; decode excludes first token, includes token D2H/synchronization; diagnostic requests include logits I/O; no MTP, fixed output length",
        })
    }
}
pub(crate) fn run_model(
    manifest_path: &std::path::Path,
    requests_path: &std::path::Path,
) -> Result<crate::model::Report> {
    let requests = crate::model::read(requests_path)?;
    ModelRuntime::load(manifest_path)?.benchmark(requests)
}

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

    #[test]
    #[ignore = "Requires a real model, fixture and exclusive GPU experiment lock"]
    fn multimodal_token_probe() {
        #[derive(serde::Deserialize)]
        struct Probe {
            model: std::path::PathBuf,
            input_tokens: Vec<u32>,
            images: Vec<crate::vision::ImageInput>,
            prefixes: Vec<Vec<u32>>,
            target_tokens: Vec<u32>,
            output: std::path::PathBuf,
        }
        let path = std::env::var("ORIN_VISION_PROBE").expect("ORIN_VISION_PROBE");
        let probe: Probe = crate::model::read(std::path::Path::new(&path)).unwrap();
        let mut model = ModelRuntime::load(&probe.model).unwrap();
        let options = crate::sampling::Options {
            temperature: 0.,
            ..Default::default()
        };
        let mut generated = vec![];
        model
            .generate(
                &probe.input_tokens,
                Some(&probe.images),
                128,
                &options,
                || false,
                |id| {
                    generated.push(id);
                    ![248046, 248044].contains(&id)
                },
            )
            .unwrap();
        let mut checks = vec![];
        for prefix in &probe.prefixes {
            let mut input = probe.input_tokens.clone();
            input.extend(prefix);
            let mut selected = 0;
            model
                .generate(
                    &input,
                    Some(&probe.images),
                    1,
                    &options,
                    || false,
                    |id| {
                        selected = id;
                        true
                    },
                )
                .unwrap();
            let spec = model
                .manifest
                .buffers
                .iter()
                .find(|b| b.name == model.manifest.logits)
                .unwrap();
            let mut raw = vec![0u8; spec.bytes().unwrap()];
            // SAFETY: generate synchronized the producing stream; the host
            // allocation covers the complete validated logits buffer.
            unsafe {
                check(
                    (model.session.driver.download)(
                        raw.as_mut_ptr().cast(),
                        model.pointers[&model.manifest.logits],
                        raw.len(),
                    ),
                    "probe logits",
                )
                .unwrap();
            }
            let logits = floats(&raw, spec.dtype);
            assert!(logits.iter().all(|v| v.is_finite()));
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z = f64::from(max)
                + logits
                    .iter()
                    .map(|&v| f64::from(v - max).exp())
                    .sum::<f64>()
                    .ln();
            let entry = |id: usize| {
                serde_json::json!({
                    "token_id":id,"logit":logits[id],"logprob":f64::from(logits[id])-log_z
                })
            };
            let mut ids: Vec<usize> = (0..logits.len()).collect();
            ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            checks.push(serde_json::json!({"prefix":prefix,"selected":selected,
                "top3":ids[..3].iter().map(|&id|entry(id)).collect::<Vec<_>>(),
                "targets":probe.target_tokens.iter().map(|&id|entry(id as usize)).collect::<Vec<_>>()
            }));
        }
        let result = serde_json::json!({"generated":generated,"checks":checks,
            "seed":crate::sampling::EVALUATION_SEED});
        use std::io::Write;
        let mut output = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&probe.output)
            .unwrap();
        output
            .write_all(serde_json::to_string_pretty(&result).unwrap().as_bytes())
            .unwrap();
        println!("{result}");
    }
}
