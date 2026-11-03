//! CANN public C ABI from acl_rt.h. No link-time CANN dependency.
use super::Driver;
use crate::artifact::Result;
use std::ffi::c_void;

pub(super) type Handle = *mut c_void;

impl Api {
    pub fn current(&self) -> Result<Handle> {
        let mut current = std::ptr::null_mut();
        // SAFETY: Live ACL library and writable context output. CANN uses
        // ACL_ERROR_RT_CONTEXT_NULL (107002) for an unbound host thread.
        let status = unsafe { (self.get_context)(&mut current) };
        if status != 107002 {
            super::check(status, "get current ACL context")?;
        }
        Ok(current)
    }
}

macro_rules! functions {
    ($($field:ident : $name:ident ($($argument:ty),*) -> $result:ty;)+) => {
        #[derive(Clone, Copy)]
        pub(super) struct Api { $(pub $field: unsafe extern "C" fn($($argument),*) -> $result,)+ }
        impl Api {
            pub fn load(driver: &Driver) -> Result<Self> {
                // SAFETY: All signatures below are from CANN's public acl_rt.h;
                // the context retains Driver for every copied function pointer.
                unsafe { Ok(Self { $($field: *driver._library.get(concat!(stringify!($name),"\0").as_bytes())
                    .map_err(|e| format!("{}: {e}", stringify!($name)))?,)+ }) }
            }
        }
    };
}
functions! {
    create_context: aclrtCreateContext(*mut Handle, i32) -> i32;
    destroy_context: aclrtDestroyContext(Handle) -> i32;
    get_context: aclrtGetCurrentContext(*mut Handle) -> i32;
    set_context: aclrtSetCurrentContext(Handle) -> i32;
    create_stream: aclrtCreateStream(*mut Handle) -> i32;
    destroy_stream: aclrtDestroyStream(Handle) -> i32;
    sync_stream: aclrtSynchronizeStreamWithTimeout(Handle, i32) -> i32;
    create_event: aclrtCreateEventExWithFlag(*mut Handle, u32) -> i32;
    destroy_event: aclrtDestroyEvent(Handle) -> i32;
    record_event: aclrtRecordEvent(Handle, Handle) -> i32;
    query_event: aclrtQueryEventStatus(Handle, *mut i32) -> i32;
    sync_event: aclrtSynchronizeEventWithTimeout(Handle, i32) -> i32;
    wait_event: aclrtStreamWaitEventWithTimeout(Handle, Handle, i32) -> i32;
    malloc: aclrtMalloc(*mut Handle, usize, i32) -> i32;
    free: aclrtFree(Handle) -> i32;
    malloc_host: aclrtMallocHost(*mut Handle, usize) -> i32;
    free_host: aclrtFreeHost(Handle) -> i32;
    memcpy: aclrtMemcpyAsync(Handle, usize, *const c_void, usize, i32, Handle) -> i32;
    memset: aclrtMemsetAsync(Handle, usize, i32, usize, Handle) -> i32;
    memory_info: aclrtGetMemInfo(i32, *mut usize, *mut usize) -> i32;
}
