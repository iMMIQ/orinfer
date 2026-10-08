//! Private, thread-affine CUDA Driver session for fixed AOT fixtures.
//! Handles never escape the session. All work is on one explicit stream; cleanup
//! synchronizes before destroying graphs/modules or freeing device allocations.
use crate::artifact::{Argument, Dtype, Kernel, Loaded, Result};
use libloading::Library;
use serde::Serialize;
use std::{
    collections::BTreeMap,
    ffi::{CString, c_void},
    ptr,
    time::Instant,
};
mod kernels;
mod sequence;
pub(crate) mod snapshot;
mod virtual_memory;
mod weight_upload;
pub(crate) use cudarc::driver::sys;
pub(crate) type Handle = sys::CUgraphExec;

pub(crate) struct Driver {
    _library: Library,
}
impl Driver {
    pub(crate) fn load() -> Result<Self> {
        // SAFETY: Loading the installed CUDA driver; retaining it for the session.
        let library = unsafe { Library::new("libcuda.so.1") }.map_err(|e| e.to_string())?;
        // Preflight prevents cudarc's lazy symbol loader from panicking during execution.
        // Signatures and CUDA types come entirely from its generated CUDA 12.6 bindings.
        for name in [
            b"cuInit\0".as_slice(),
            b"cuDriverGetVersion\0".as_slice(),
            b"cuDeviceGet\0".as_slice(),
            b"cuDeviceGetName\0".as_slice(),
            b"cuDeviceGetAttribute\0".as_slice(),
            b"cuDeviceTotalMem_v2\0".as_slice(),
            b"cuCtxGetCurrent\0".as_slice(),
            b"cuDevicePrimaryCtxRetain\0".as_slice(),
            b"cuDevicePrimaryCtxRelease_v2\0".as_slice(),
            b"cuCtxSetCurrent\0".as_slice(),
            b"cuCtxSynchronize\0".as_slice(),
            b"cuStreamCreate\0".as_slice(),
            b"cuStreamSynchronize\0".as_slice(),
            b"cuStreamDestroy_v2\0".as_slice(),
            b"cuMemGetInfo_v2\0".as_slice(),
            b"cuMemAlloc_v2\0".as_slice(),
            b"cuMemAddressReserve\0".as_slice(),
            b"cuMemAddressFree\0".as_slice(),
            b"cuMemGetAllocationGranularity\0".as_slice(),
            b"cuMemCreate\0".as_slice(),
            b"cuMemMap\0".as_slice(),
            b"cuMemSetAccess\0".as_slice(),
            b"cuMemUnmap\0".as_slice(),
            b"cuMemRelease\0".as_slice(),
            b"cuMemFree_v2\0".as_slice(),
            b"cuMemcpyHtoD_v2\0".as_slice(),
            b"cuMemcpyHtoDAsync_v2\0".as_slice(),
            b"cuMemHostAlloc\0".as_slice(),
            b"cuMemFreeHost\0".as_slice(),
            b"cuMemcpyDtoDAsync_v2\0".as_slice(),
            b"cuMemcpyDtoH_v2\0".as_slice(),
            b"cuMemsetD8Async\0".as_slice(),
            b"cuModuleLoadData\0".as_slice(),
            b"cuModuleUnload\0".as_slice(),
            b"cuModuleGetFunction\0".as_slice(),
            b"cuFuncGetAttribute\0".as_slice(),
            b"cuFuncSetAttribute\0".as_slice(),
            b"cuLaunchKernel\0".as_slice(),
            b"cuStreamBeginCapture_v2\0".as_slice(),
            b"cuStreamEndCapture\0".as_slice(),
            b"cuGraphInstantiateWithFlags\0".as_slice(),
            b"cuGraphLaunch\0".as_slice(),
            b"cuGraphDestroy\0".as_slice(),
            b"cuGraphExecDestroy\0".as_slice(),
            b"cuEventCreate\0".as_slice(),
            b"cuEventRecord\0".as_slice(),
            b"cuEventSynchronize\0".as_slice(),
            b"cuEventElapsedTime\0".as_slice(),
            b"cuEventDestroy_v2\0".as_slice(),
            b"cuEventRecordWithFlags\0".as_slice(),
        ] {
            // SAFETY: Inspect the symbol address without calling it or dereferencing it.
            unsafe { library.get::<*const c_void>(name) }.map_err(|e| e.to_string())?;
        }
        Ok(Self { _library: library })
    }
}
pub(crate) fn check(code: sys::CUresult, operation: &str) -> Result<()> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        crate::error::record_cuda(code as i32);
        Err(format!("{operation}: CUDA error {}", code as i32))
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
    pub(crate) previous: sys::CUcontext,
    pub(crate) retained: bool,
    pub(crate) stream: sys::CUstream,
    pub(crate) buffers: Vec<u64>,
    pub(crate) virtual_buffers: std::cell::RefCell<virtual_memory::Reservations>,
    pub(crate) modules: std::cell::RefCell<Vec<sys::CUmodule>>,
    pub(crate) events: Vec<sys::CUevent>,
    pub(crate) graph: sys::CUgraph,
    pub(crate) exec: Handle,
    pub(crate) graphs: std::cell::RefCell<Vec<(sys::CUgraph, Handle)>>,
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
            modules: Default::default(),
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
                    sys::cuStreamEndCapture(self.stream, &mut self.graph),
                    "cleanup end capture",
                );
                self.capturing = false;
            }
            if !self.stream.is_null() {
                record(
                    sys::cuStreamSynchronize(self.stream),
                    "cleanup stream synchronize",
                );
            }
            if !self.exec.is_null() {
                record(sys::cuGraphExecDestroy(self.exec), "destroy graph exec");
                self.exec = ptr::null_mut();
            }
            if !self.graph.is_null() {
                record(sys::cuGraphDestroy(self.graph), "destroy graph");
                self.graph = ptr::null_mut();
            }
            for (graph, exec) in self.graphs.get_mut().drain(..) {
                if !exec.is_null() {
                    record(sys::cuGraphExecDestroy(exec), "destroy model graph exec");
                }
                record(sys::cuGraphDestroy(graph), "destroy model graph");
            }
            for event in self.events.drain(..) {
                record(sys::cuEventDestroy_v2(event), "destroy event");
            }
            for (_, mut buffer) in std::mem::take(self.virtual_buffers.get_mut()) {
                if let Err(e) = buffer.release_slabs(&self.driver) {
                    eprintln!("CUDA KV cleanup: {e}");
                }
                if buffer.has_slabs() {
                    continue;
                }
                record(
                    sys::cuMemAddressFree(buffer.address, buffer.bytes),
                    "free KV address reservation",
                );
            }
            for buffer in self.buffers.drain(..) {
                record(sys::cuMemFree_v2(buffer), "free buffer");
            }
            for module in self.modules.get_mut().drain(..) {
                record(sys::cuModuleUnload(module), "unload module");
            }
            if !self.stream.is_null() {
                record(sys::cuStreamDestroy_v2(self.stream), "destroy stream");
                self.stream = ptr::null_mut();
            }
            if self.retained {
                record(
                    sys::cuDevicePrimaryCtxRelease_v2(self.device),
                    "release primary context",
                );
                self.retained = false;
                record(sys::cuCtxSetCurrent(self.previous), "restore prior context");
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
    function: sys::CUfunction,
    values: Vec<Value>,
}
impl Launch<'_> {
    fn execute(&mut self, driver: &Driver, stream: sys::CUstream) -> Result<()> {
        launch_kernel(self.spec, self.function, &mut self.values, driver, stream)
    }
}

fn launch_kernel(
    k: &Kernel,
    function: sys::CUfunction,
    values: &mut [Value],
    _driver: &Driver,
    stream: sys::CUstream,
) -> Result<()> {
    let mut args: Vec<*mut c_void> = values.iter_mut().map(Value::address).collect();
    // SAFETY: Values remain at stable Vec addresses until cuLaunchKernel has
    // copied each parameter. Device pointers refer to live session buffers;
    // generated ABI order/types are supplied by the verified manifest.
    // The module and explicit stream outlive all queued/captured work.
    unsafe {
        check(
            sys::cuLaunchKernel(
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
    half::f16::from_bits(bits).to_f32()
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
        check(sys::cuInit(0), "cuInit")?;
        check(sys::cuDeviceGet(&mut s.device, 0), "cuDeviceGet")?;
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
            return Err(format!("Expected SM87, got {major}.{minor}"));
        }
        check(sys::cuDriverGetVersion(&mut version), "driver version")?;
        check(
            sys::cuDeviceGetName(name.as_mut_ptr().cast(), 256, s.device),
            "device name",
        )?;
        check(
            sys::cuDeviceTotalMem_v2(&mut total, s.device),
            "total device memory",
        )?;
        check(
            sys::cuCtxGetCurrent(&mut s.previous),
            "get previous context",
        )?;
        let mut context = ptr::null_mut();
        check(
            sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
            "retain primary context",
        )?;
        s.retained = true;
        check(sys::cuCtxSetCurrent(context), "set current context")?;
        check(
            sys::cuStreamCreate(&mut s.stream, 1),
            "create nonblocking stream",
        )?;
        let (mut free, mut available_total) = (0usize, 0usize);
        check(
            sys::cuMemGetInfo_v2(&mut free, &mut available_total),
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
                sys::cuMemAlloc_v2(&mut address, bytes),
                &format!("allocate {}", b.name),
            )?;
            s.buffers.push(address);
            if address % b.alignment != 0 {
                return Err(format!("{} allocation alignment mismatch", b.name));
            }
            if let Some(id) = &b.data {
                check(
                    sys::cuMemcpyHtoD_v2(address, x.files[&id.file].as_ptr().cast(), bytes),
                    &format!("upload {}", b.name),
                )?
            } else {
                check(
                    sys::cuMemsetD8Async(address, 0, bytes, s.stream),
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
                sys::cuModuleLoadData(&mut module, image.as_ptr().cast()),
                &format!("load {}", k.name),
            )?;
            s.modules.borrow_mut().push(module);
            let symbol = CString::new(k.symbol.as_str()).map_err(|e| e.to_string())?;
            let mut function = ptr::null_mut();
            check(
                sys::cuModuleGetFunction(&mut function, module, symbol.as_ptr()),
                &format!("find {}", k.symbol),
            )?;
            let (mut maxthreads, mut staticsmem, mut maxshared) = (0, 0, 0);
            check(
                sys::cuFuncGetAttribute(
                    &mut maxthreads,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                    function,
                ),
                "function max threads",
            )?;
            check(
                sys::cuFuncGetAttribute(
                    &mut staticsmem,
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                    function,
                ),
                "function static shared memory",
            )?;
            check(
                sys::cuDeviceGetAttribute(
                    &mut maxshared,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    s.device,
                ),
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
                    sys::cuDeviceGetAttribute(
                        &mut maxblock,
                        [
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                        ][axis],
                        s.device,
                    ),
                    "block dimension limit",
                )?;
                check(
                    sys::cuDeviceGetAttribute(
                        &mut maxgrid,
                        [
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                        ][axis],
                        s.device,
                    ),
                    "grid dimension limit",
                )?;
                if k.block[axis] > maxblock as u32 || k.grid[axis] > maxgrid as u32 {
                    return Err("Launch dimensions exceed device limits".into());
                }
            }
            let mut default_shared = 0;
            check(
                sys::cuDeviceGetAttribute(
                    &mut default_shared,
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                    s.device,
                ),
                "default shared memory limit",
            )?;
            if u64::from(k.shared_memory_bytes) + staticsmem as u64 > default_shared as u64 {
                check(
                    sys::cuFuncSetAttribute(
                        function,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        k.shared_memory_bytes as i32,
                    ),
                    "opt-in dynamic shared memory",
                )?
            }
            let values = k
                .args
                .iter()
                .map(|a| match a {
                    Argument::Buffer { name } => Value::Pointer(device_buffers[name]),
                    Argument::BufferSlice { name, offset } => {
                        Value::Pointer(device_buffers[name] + *offset as u64)
                    }
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
        check(sys::cuCtxSynchronize(), "finish loader transfers")?;
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
        check(
            sys::cuStreamSynchronize(s.stream),
            "first launch synchronize",
        )?;
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
            sys::cuMemcpyDtoH_v2(expected.as_mut_ptr().cast(), output, expected.len()),
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
            sys::cuStreamBeginCapture_v2(
                s.stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
            "begin thread-local capture",
        )?;
        s.capturing = true;
        for l in &mut launches {
            l.execute(&s.driver, s.stream)?;
        }
        let code = sys::cuStreamEndCapture(s.stream, &mut s.graph);
        s.capturing = false;
        check(code, "end capture")?;
        check(
            sys::cuGraphInstantiateWithFlags(&mut s.exec, s.graph, 0),
            "instantiate graph",
        )?;
        let graph_capture_s = capture.elapsed().as_secs_f64();
        let mut observed = vec![0u8; expected.len()];
        check(
            sys::cuMemsetD8Async(output, 255, expected.len(), s.stream),
            "poison output",
        )?;
        check(
            sys::cuGraphLaunch(s.exec, s.stream),
            "replay poisoned output",
        )?;
        check(sys::cuStreamSynchronize(s.stream), "replay synchronize")?;
        check(
            sys::cuMemcpyDtoH_v2(observed.as_mut_ptr().cast(), output, observed.len()),
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
            sys::cuMemsetD8Async(input, 0, input_spec.bytes()?, s.stream),
            "change actual input to zero",
        )?;
        check(
            sys::cuMemsetD8Async(output, 255, observed.len(), s.stream),
            "poison changed-input output",
        )?;
        check(sys::cuGraphLaunch(s.exec, s.stream), "replay changed input")?;
        check(
            sys::cuStreamSynchronize(s.stream),
            "changed-input synchronize",
        )?;
        check(
            sys::cuMemcpyDtoH_v2(observed.as_mut_ptr().cast(), output, observed.len()),
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
            sys::cuMemcpyHtoD_v2(input, original.as_ptr().cast(), original.len()),
            "restore actual input",
        )?;
        check(sys::cuCtxSynchronize(), "finish restored input transfer")?;
        check(
            sys::cuMemsetD8Async(output, 255, observed.len(), s.stream),
            "poison restored output",
        )?;
        check(
            sys::cuGraphLaunch(s.exec, s.stream),
            "replay restored input",
        )?;
        check(sys::cuStreamSynchronize(s.stream), "restore synchronize")?;
        check(
            sys::cuMemcpyDtoH_v2(observed.as_mut_ptr().cast(), output, observed.len()),
            "download restored output",
        )?;
        if observed != expected {
            return Err("Restored-input graph differs from first output".into());
        }
        let (mut begin, mut end) = (ptr::null_mut(), ptr::null_mut());
        check(sys::cuEventCreate(&mut begin, 0), "create begin event")?;
        s.events.push(begin);
        check(sys::cuEventCreate(&mut end, 0), "create end event")?;
        s.events.push(end);
        let mut times = vec![];
        for _ in 0..3 {
            check(sys::cuEventRecord(begin, s.stream), "record begin event")?;
            for _ in 0..x.manifest.validation.repetitions {
                check(sys::cuGraphLaunch(s.exec, s.stream), "timed graph replay")?
            }
            check(sys::cuEventRecord(end, s.stream), "record end event")?;
            check(sys::cuEventSynchronize(end), "synchronize end event")?;
            let mut ms = 0f32;
            check(
                sys::cuEventElapsedTime(&mut ms, begin, end),
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
