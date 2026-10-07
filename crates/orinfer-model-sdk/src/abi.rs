//! C ABI v1. JSON is confined to model creation; batch programs use borrowed C
//! arrays. Every allocation is released by the library that created it.
use crate::{
    architecture::{BatchSegment, PrefillProfile},
    artifact::Result,
    execution::Invocation,
    model::{Manifest, Operation},
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::c_void,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

pub type StateBindings = Vec<Option<(usize, String)>>;
pub type BatchPlan = (Vec<Invocation>, StateBindings);

pub const ABI_VERSION: u32 = 1;
pub const RUNTIME_ABI: u32 = 1;
pub const ENTRYPOINT: &[u8] = b"orinfer_model_v1\0";

#[derive(Deserialize, Serialize)]
pub struct PackageInfo {
    pub package: String,
    pub version: String,
    pub target: String,
    pub runtime_abi: u32,
    pub architectures: Vec<String>,
    pub compute_policies: Vec<String>,
}
#[derive(Deserialize, Serialize)]
pub struct CreateRequest {
    pub config: serde_json::Value,
    pub architecture: String,
    pub compute_policy: String,
    pub expected_signature: serde_json::Value,
    pub metadata: Manifest,
    pub prefill_profiles: Vec<PrefillProfile>,
}
#[derive(Deserialize, Serialize)]
pub struct CreatedPlan {
    pub metadata: Manifest,
    pub decode_programs: std::collections::BTreeSet<String>,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bytes {
    pub data: *const u8,
    pub len: usize,
}
impl Bytes {
    pub fn borrowed(value: &str) -> Self {
        Self {
            data: value.as_ptr(),
            len: value.len(),
        }
    }
}
#[repr(C)]
pub struct OwnedBytes {
    pub data: *mut u8,
    pub len: usize,
}
impl Default for OwnedBytes {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
        }
    }
}
impl OwnedBytes {
    fn new(value: Vec<u8>) -> Self {
        let mut value = value.into_boxed_slice();
        let result = Self {
            data: value.as_mut_ptr(),
            len: value.len(),
        };
        std::mem::forget(value);
        result
    }
}
#[repr(C)]
pub struct View {
    pub name: Bytes,
    pub buffer: Bytes,
    pub offset: usize,
}
#[repr(C)]
pub struct Argument {
    pub index: usize,
    pub value: i32,
}
#[repr(C)]
pub struct Command {
    /// 0=kernel, 1=copy, 2=zero; names reference the package's validated buffers.
    pub kind: u32,
    pub name: Bytes,
    pub destination: Bytes,
    pub bytes: usize,
    /// usize::MAX selects the shared workspace rather than a request slot.
    pub sequence: usize,
    pub has_launch: u32,
    pub grid: [u32; 3],
    pub arguments: *const Argument,
    pub argument_count: usize,
    pub views: *const View,
    pub view_count: usize,
}
#[repr(C)]
pub struct Binding {
    pub present: u32,
    pub sequence: usize,
    pub buffer: Bytes,
}
#[repr(C)]
pub struct BatchOutput {
    pub owner: *mut c_void,
    pub commands: *const Command,
    pub command_count: usize,
    pub bindings: *const Binding,
    pub binding_count: usize,
}
impl Default for BatchOutput {
    fn default() -> Self {
        Self {
            owner: ptr::null_mut(),
            commands: ptr::null(),
            command_count: 0,
            bindings: ptr::null(),
            binding_count: 0,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImageGrid {
    pub grid_height: usize,
    pub grid_width: usize,
}
#[repr(C)]
pub struct VisualOutput {
    pub owner: *mut c_void,
    pub indices: *const i32,
    pub index_count: usize,
    pub positions: *const u32,
    pub position_count: usize,
}
impl Default for VisualOutput {
    fn default() -> Self {
        Self {
            owner: ptr::null_mut(),
            indices: ptr::null(),
            index_count: 0,
            positions: ptr::null(),
            position_count: 0,
        }
    }
}
#[repr(C)]
pub struct Api {
    pub abi_version: u32,
    pub struct_size: usize,
    pub describe: unsafe extern "C" fn(*mut OwnedBytes) -> i32,
    pub create: unsafe extern "C" fn(*const u8, usize, *mut *mut c_void, *mut OwnedBytes) -> i32,
    pub destroy: unsafe extern "C" fn(*mut c_void),
    pub batch: unsafe extern "C" fn(
        *mut c_void,
        *const BatchSegment,
        usize,
        u32,
        *mut BatchOutput,
        *mut OwnedBytes,
    ) -> i32,
    pub free_bytes: unsafe extern "C" fn(OwnedBytes),
    pub free_batch: unsafe extern "C" fn(BatchOutput),
    pub visual: unsafe extern "C" fn(
        *mut c_void,
        *const u32,
        usize,
        *const ImageGrid,
        usize,
        usize,
        *mut VisualOutput,
        *mut OwnedBytes,
    ) -> i32,
    pub free_visual: unsafe extern "C" fn(VisualOutput),
}

/// Rust-only implementation interface. No Rust object or vtable crosses dlopen.
pub trait ModelImplementation: Sized {
    fn describe() -> PackageInfo;
    fn visual(
        &self,
        _tokens: &[u32],
        _images: &[ImageGrid],
        _capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        Err("Model package has no visual layout adapter".into())
    }
    fn create(request: CreateRequest) -> Result<(Self, CreatedPlan)>;
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<BatchPlan>;
}

struct BatchStorage {
    _operations: Vec<Invocation>,
    _binding_names: Vec<Option<(usize, String)>>,
    views: Vec<Vec<View>>,
    arguments: Vec<Vec<Argument>>,
    commands: Vec<Command>,
    bindings: Vec<Binding>,
}
impl BatchStorage {
    fn export(operations: Vec<Invocation>, bindings: Vec<Option<(usize, String)>>) -> BatchOutput {
        let mut owner = Box::new(Self {
            _operations: operations,
            _binding_names: bindings,
            views: vec![],
            arguments: vec![],
            commands: vec![],
            bindings: vec![],
        });
        for op in &owner._operations {
            owner.views.push(
                op.views
                    .iter()
                    .map(|(name, v)| View {
                        name: Bytes::borrowed(name),
                        buffer: Bytes::borrowed(&v.buffer),
                        offset: v.offset,
                    })
                    .collect(),
            );
            owner.arguments.push(
                op.launch
                    .as_ref()
                    .map(|v| {
                        v.arguments
                            .iter()
                            .map(|&(index, value)| Argument { index, value })
                            .collect()
                    })
                    .unwrap_or_default(),
            );
        }
        for (i, op) in owner._operations.iter().enumerate() {
            let (kind, name, destination, bytes) = match &op.operation {
                Operation::Kernel { name } => (0, name.as_str(), "", 0),
                Operation::Copy {
                    source,
                    destination,
                    bytes,
                } => (1, source.as_str(), destination.as_str(), *bytes),
                Operation::Zero { destination, bytes } => (2, "", destination.as_str(), *bytes),
            };
            owner.commands.push(Command {
                kind,
                name: Bytes::borrowed(name),
                destination: Bytes::borrowed(destination),
                bytes,
                sequence: op.sequence.unwrap_or(usize::MAX),
                has_launch: u32::from(op.launch.is_some()),
                grid: op.launch.as_ref().map(|l| l.grid).unwrap_or([0; 3]),
                arguments: owner.arguments[i].as_ptr(),
                argument_count: owner.arguments[i].len(),
                views: owner.views[i].as_ptr(),
                view_count: owner.views[i].len(),
            });
        }
        owner.bindings = owner
            ._binding_names
            .iter()
            .map(|v| match v {
                Some((sequence, name)) => Binding {
                    present: 1,
                    sequence: *sequence,
                    buffer: Bytes::borrowed(name),
                },
                None => Binding {
                    present: 0,
                    sequence: 0,
                    buffer: Bytes::borrowed(""),
                },
            })
            .collect();
        let result = BatchOutput {
            owner: (&mut *owner as *mut Self).cast(),
            commands: owner.commands.as_ptr(),
            command_count: owner.commands.len(),
            bindings: owner.bindings.as_ptr(),
            binding_count: owner.bindings.len(),
        };
        std::mem::forget(owner);
        result
    }
}

fn boundary(output: *mut OwnedBytes, f: impl FnOnce() -> Result<Vec<u8>>) -> i32 {
    if output.is_null() {
        return 1;
    }
    let result = catch_unwind(AssertUnwindSafe(f));
    let (status, bytes) = match result {
        Ok(Ok(bytes)) => (0, bytes),
        Ok(Err(error)) => (1, error.into_bytes()),
        Err(_) => (2, b"Model package panicked".to_vec()),
    };
    // SAFETY: caller supplies a writable output slot, whose value is owned by it.
    unsafe {
        ptr::write(output, OwnedBytes::new(bytes));
    }
    status
}

/// # Safety
/// Input spans and output slots must remain valid for this synchronous call.
pub unsafe extern "C" fn create<M: ModelImplementation>(
    input: *const u8,
    len: usize,
    handle: *mut *mut c_void,
    output: *mut OwnedBytes,
) -> i32 {
    boundary(output, || {
        if handle.is_null() || input.is_null() || len > 128 * 1024 * 1024 {
            return Err("Invalid create arguments".into());
        }
        // SAFETY: checked nonnull; caller provides len readable bytes.
        let request = serde_json::from_slice(unsafe { slice::from_raw_parts(input, len) })
            .map_err(|e| e.to_string())?;
        let (model, plan) = M::create(request)?;
        let bytes = serde_json::to_vec(&plan).map_err(|e| e.to_string())?;
        // SAFETY: the handle slot is writable; destroy owns this box thereafter.
        unsafe {
            ptr::write(handle, Box::into_raw(Box::new(model)).cast());
        }
        Ok(bytes)
    })
}
/// # Safety
/// handle must be a live handle returned by create<M>, destroyed exactly once.
pub unsafe extern "C" fn destroy<M>(handle: *mut c_void) {
    if !handle.is_null() {
        // SAFETY: ownership is transferred back to the library that allocated M.
        drop(unsafe { Box::from_raw(handle.cast::<M>()) });
    }
}
/// # Safety
/// All input/output spans must be valid; handle must belong to this library.
pub unsafe extern "C" fn batch<M: ModelImplementation>(
    handle: *mut c_void,
    input: *const BatchSegment,
    len: usize,
    include_plan: u32,
    result: *mut BatchOutput,
    error: *mut OwnedBytes,
) -> i32 {
    boundary(error, || {
        if handle.is_null()
            || input.is_null()
            || !(1..=128).contains(&len)
            || result.is_null()
            || include_plan > 1
        {
            return Err("Invalid batch arguments".into());
        }
        // SAFETY: caller retains the model and this readable segment array.
        let (model, segments) =
            unsafe { (&*handle.cast::<M>(), slice::from_raw_parts(input, len)) };
        if segments
            .iter()
            .any(|s| s.tokens == 0 || s.tokens > 262144 || s.slot == usize::MAX)
        {
            return Err("Invalid segment geometry".into());
        }
        let (ops, bindings) = model.batch(segments, include_plan != 0)?;
        // SAFETY: writable result slot; free_batch is the sole owner of its buffers.
        unsafe {
            ptr::write(result, BatchStorage::export(ops, bindings));
        }
        Ok(vec![])
    })
}
/// # Safety
/// value must be a still-owned buffer returned by this library.
pub unsafe extern "C" fn free_bytes(value: OwnedBytes) {
    if !value.data.is_null() {
        // SAFETY: original boxed slice uses exactly this pointer and length.
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(value.data, value.len)) });
    }
}
/// # Safety
/// value must be a still-owned batch returned by this library.
pub unsafe extern "C" fn free_batch(value: BatchOutput) {
    if !value.owner.is_null() {
        // SAFETY: private BatchStorage was boxed by export in this library.
        drop(unsafe { Box::from_raw(value.owner.cast::<BatchStorage>()) });
    }
}

struct VisualStorage {
    indices: Vec<i32>,
    positions: Vec<u32>,
}
/// # Safety
/// The model, borrowed token/grid spans and writable output slots must be valid.
pub unsafe extern "C" fn visual<M: ModelImplementation>(
    handle: *mut c_void,
    tokens: *const u32,
    count: usize,
    images: *const ImageGrid,
    image_count: usize,
    capacity: usize,
    result: *mut VisualOutput,
    error: *mut OwnedBytes,
) -> i32 {
    boundary(error, || {
        if handle.is_null()
            || result.is_null()
            || tokens.is_null()
            || count > capacity
            || capacity > 262144
            || (image_count != 0 && images.is_null())
            || image_count > 4096
        {
            return Err("Invalid visual layout arguments".into());
        }
        // SAFETY: caller owns the live model and readable spans for this call.
        let (model, tokens, images) = unsafe {
            (
                &*handle.cast::<M>(),
                slice::from_raw_parts(tokens, count),
                if image_count == 0 {
                    &[]
                } else {
                    slice::from_raw_parts(images, image_count)
                },
            )
        };
        let (indices, positions) = model.visual(tokens, images, capacity)?;
        let mut owner = Box::new(VisualStorage { indices, positions });
        let output = VisualOutput {
            owner: (&mut *owner as *mut VisualStorage).cast(),
            indices: owner.indices.as_ptr(),
            index_count: owner.indices.len(),
            positions: owner.positions.as_ptr(),
            position_count: owner.positions.len(),
        };
        std::mem::forget(owner);
        // SAFETY: result is writable and free_visual receives ownership.
        unsafe {
            ptr::write(result, output);
        }
        Ok(vec![])
    })
}
/// # Safety
/// output must be a still-owned visual result returned by this library.
pub unsafe extern "C" fn free_visual(output: VisualOutput) {
    if !output.owner.is_null() {
        // SAFETY: exact owner type allocated by visual above, freed once.
        drop(unsafe { Box::from_raw(output.owner.cast::<VisualStorage>()) });
    }
}

/// # Safety
/// output must point to a writable OwnedBytes slot.
pub unsafe extern "C" fn describe<M: ModelImplementation>(output: *mut OwnedBytes) -> i32 {
    boundary(output, || {
        serde_json::to_vec(&M::describe()).map_err(|e| e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{execution::BufferView, operators::dynamic::RowLaunch};
    use std::collections::BTreeMap;

    struct TestModel;
    impl ModelImplementation for TestModel {
        fn describe() -> PackageInfo {
            PackageInfo {
                package: "fixture".into(),
                version: "1".into(),
                target: "sm_87".into(),
                runtime_abi: RUNTIME_ABI,
                architectures: vec!["fixture".into()],
                compute_policies: vec!["fixture".into()],
            }
        }
        fn create(_: CreateRequest) -> Result<(Self, CreatedPlan)> {
            panic!("caught at ABI boundary")
        }
        fn batch(
            &self,
            segments: &[BatchSegment],
            plan: bool,
        ) -> Result<(Vec<Invocation>, Vec<Option<(usize, String)>>)> {
            assert_eq!(segments[0].slot, 7);
            let ops = if plan {
                vec![Invocation {
                    operation: Operation::Kernel {
                        name: "projection".into(),
                    },
                    sequence: Some(7),
                    launch: Some(RowLaunch {
                        grid: [3, 1, 1],
                        arguments: vec![(2, 3)],
                    }),
                    views: BTreeMap::from([(
                        "input".into(),
                        BufferView {
                            buffer: "workspace".into(),
                            offset: 128,
                        },
                    )]),
                }]
            } else {
                vec![]
            };
            Ok((ops, vec![None, Some((7, "recurrent-state".into()))]))
        }
    }
    #[test]
    fn binary_batch_retains_names_arguments_views_and_private_bindings() {
        let handle = Box::into_raw(Box::new(TestModel)).cast();
        let segments = [BatchSegment { slot: 7, tokens: 3 }];
        for include_plan in [0, 1] {
            let mut output = BatchOutput::default();
            let mut error = OwnedBytes::default();
            // SAFETY: locally owned model/segment/output slots match this implementation.
            let status = unsafe {
                batch::<TestModel>(
                    handle,
                    segments.as_ptr(),
                    1,
                    include_plan,
                    &mut output,
                    &mut error,
                )
            };
            assert_eq!(status, 0);
            assert_eq!(output.command_count, include_plan as usize);
            // SAFETY: returned arrays are owned by output until free_batch below.
            unsafe {
                let bindings = slice::from_raw_parts(output.bindings, output.binding_count);
                assert_eq!(bindings[0].present, 0);
                assert_eq!(bindings[1].sequence, 7);
                assert_eq!(
                    slice::from_raw_parts(bindings[1].buffer.data, bindings[1].buffer.len),
                    b"recurrent-state"
                );
                if include_plan != 0 {
                    let command = &*output.commands;
                    assert_eq!(command.sequence, 7);
                    assert_eq!(command.grid, [3, 1, 1]);
                    assert_eq!((*command.arguments).value, 3);
                    assert_eq!((*command.views).offset, 128);
                    assert_eq!(
                        slice::from_raw_parts(command.name.data, command.name.len),
                        b"projection"
                    );
                }
                free_bytes(error);
                free_batch(output);
            }
        }
        // SAFETY: model was allocated above and no longer borrowed by outputs.
        unsafe {
            destroy::<TestModel>(handle);
        }
    }
    #[test]
    fn failed_calls_return_library_owned_errors_and_do_not_cross_unwind_boundary() {
        let mut error = OwnedBytes::default();
        let mut output = BatchOutput::default();
        // SAFETY: intentionally invalid null model is rejected before dereference.
        let status = unsafe {
            batch::<TestModel>(ptr::null_mut(), ptr::null(), 0, 1, &mut output, &mut error)
        };
        assert_ne!(status, 0);
        assert!(output.owner.is_null());
        // SAFETY: this error buffer was returned by the same library implementation.
        unsafe {
            assert!(!slice::from_raw_parts(error.data, error.len).is_empty());
            free_bytes(error);
        }
        let mut error = OwnedBytes::default();
        let status = boundary(&mut error, || -> Result<Vec<u8>> {
            panic!("controlled failure")
        });
        assert_eq!(status, 2);
        // SAFETY: the boundary allocated exactly the reported boxed slice.
        unsafe {
            assert_eq!(
                slice::from_raw_parts(error.data, error.len),
                b"Model package panicked"
            );
            free_bytes(error);
        }
    }
}
