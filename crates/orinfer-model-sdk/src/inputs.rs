//! Optional ABI v1 extension for model-owned CPU lookup/decompression.
use super::{Bytes, ModelImplementation, OwnedBytes, boundary};
use std::{ffi::c_void, ptr, slice};
pub const ENTRYPOINT: &[u8] = b"orinfer_model_inputs_v1\0";
#[repr(C)]
pub struct Upload {
    pub buffer: Bytes,
    pub data: Bytes,
}
#[repr(C)]
pub struct Output {
    pub owner: *mut c_void,
    pub uploads: *const Upload,
    pub count: usize,
}
impl Default for Output {
    fn default() -> Self {
        Self {
            owner: ptr::null_mut(),
            uploads: ptr::null(),
            count: 0,
        }
    }
}
#[repr(C)]
pub struct Api {
    pub version: u32,
    pub struct_size: usize,
    pub prepare: unsafe extern "C" fn(
        *mut c_void,
        *const u32,
        usize,
        *const u32,
        usize,
        *mut Output,
        *mut OwnedBytes,
    ) -> i32,
    pub free: unsafe extern "C" fn(Output),
    /// Program-aware lookup for target verification and shifted draft inputs.
    pub prepare_program: unsafe extern "C" fn(
        *mut c_void,
        Bytes,
        *const u32,
        usize,
        *const u32,
        usize,
        *mut Output,
        *mut OwnedBytes,
    ) -> i32,
}
/// # Safety
/// The live model, input spans and writable outputs belong to the caller.
pub unsafe extern "C" fn prepare<M: ModelImplementation>(
    handle: *mut c_void,
    tokens: *const u32,
    count: usize,
    history: *const u32,
    history_count: usize,
    output: *mut Output,
    error: *mut OwnedBytes,
) -> i32 {
    // SAFETY: forwarding the caller's live spans with a static program identifier.
    unsafe {
        prepare_program::<M>(
            handle,
            Bytes::borrowed("target"),
            tokens,
            count,
            history,
            history_count,
            output,
            error,
        )
    }
}
struct Storage {
    values: Vec<(String, Vec<u8>)>,
    uploads: Vec<Upload>,
}
/// # Safety
/// The live model, token/history spans and writable outputs belong to the caller.
pub unsafe extern "C" fn prepare_program<M: ModelImplementation>(
    handle: *mut c_void,
    program: Bytes,
    tokens: *const u32,
    count: usize,
    history: *const u32,
    history_count: usize,
    output: *mut Output,
    error: *mut OwnedBytes,
) -> i32 {
    boundary(error, || {
        if handle.is_null()
            || tokens.is_null()
            || count == 0
            || count > 4096
            || history_count > 262144
            || (history_count != 0 && history.is_null())
            || program.data.is_null()
            || program.len > 256
            || output.is_null()
        {
            return Err("Invalid model input preparation spans".into());
        }
        // SAFETY: caller retains its model and the validated readable spans.
        let (model, tokens, history) = unsafe {
            (
                &*handle.cast::<M>(),
                slice::from_raw_parts(tokens, count),
                if history_count == 0 {
                    &[]
                } else {
                    slice::from_raw_parts(history, history_count)
                },
            )
        };
        // SAFETY: caller provides a bounded readable UTF-8 program span.
        let program =
            std::str::from_utf8(unsafe { slice::from_raw_parts(program.data, program.len) })
                .map_err(|e| e.to_string())?;
        let mut storage = Box::new(Storage {
            values: model.prepare(program, tokens, history)?,
            uploads: vec![],
        });
        storage.uploads = storage
            .values
            .iter()
            .map(|(name, data)| Upload {
                buffer: Bytes::borrowed(name),
                data: Bytes {
                    data: data.as_ptr(),
                    len: data.len(),
                },
            })
            .collect();
        let result = Output {
            owner: (&mut *storage as *mut Storage).cast(),
            uploads: storage.uploads.as_ptr(),
            count: storage.uploads.len(),
        };
        std::mem::forget(storage);
        // SAFETY: output is writable; the caller returns ownership through free.
        unsafe {
            ptr::write(output, result);
        }
        Ok(vec![])
    })
}
/// # Safety
/// Output must be an unfreed value created by this library's prepare function.
pub unsafe extern "C" fn free(output: Output) {
    if !output.owner.is_null() {
        // SAFETY: the exact boxed producer allocation is returned once.
        drop(unsafe { Box::from_raw(output.owner.cast::<Storage>()) });
    }
}
