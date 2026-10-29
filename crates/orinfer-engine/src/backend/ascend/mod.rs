//! CANN discovery and thread-affine native execution resources.
mod api;
pub mod graph;
pub mod hccl;
pub mod kernels;
pub mod model;
pub mod operators;
pub mod runtime;
#[cfg(test)]
mod tests;
pub mod validation;
use super::Inventory;
use crate::artifact::Result;
use libloading::Library;
use std::{
    ffi::{CStr, c_char},
    path::Path,
    ptr,
    sync::{Arc, Mutex},
};

type Init = unsafe extern "C" fn(*const c_char) -> i32;
type Finalize = unsafe extern "C" fn() -> i32;
type Count = unsafe extern "C" fn(*mut u32) -> i32;
type Soc = unsafe extern "C" fn() -> *const c_char;

struct Driver {
    _library: Library,
    finalize: Finalize,
    initialized: bool,
    count: Count,
    soc: Soc,
    path: std::path::PathBuf,
}
impl Drop for Driver {
    fn drop(&mut self) {
        if self.initialized {
            // SAFETY: This owner successfully initialized ACL, owns no device
            // resources, and keeps the library loaded through finalization.
            unsafe {
                (self.finalize)();
            }
        }
    }
}
fn check(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        crate::error::record_ascend(status);
        Err(format!("{operation}: ACL error {status}"))
    }
}

fn open_library(path: &Path) -> Result<Driver> {
    // SAFETY: CANN is an explicitly selected native dependency. The Library
    // owner stays live for every function pointer and ACL lifetime below.
    let library = unsafe { Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: These four symbols and signatures are declared by CANN's public
    // acl_rt.h and acl_base_rt.h. No device allocation or kernel is launched.
    let (init, finalize, count, soc) = unsafe {
        (
            *library
                .get::<Init>(b"aclInit\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Finalize>(b"aclFinalize\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Count>(b"aclrtGetDeviceCount\0")
                .map_err(|e| e.to_string())?,
            *library
                .get::<Soc>(b"aclrtGetSocName\0")
                .map_err(|e| e.to_string())?,
        )
    };
    let mut runtime = Driver {
        _library: library,
        finalize,
        initialized: false,
        count,
        soc,
        path: path.into(),
    };
    // SAFETY: Null selects default ACL configuration. Only a successful init
    // transfers finalization ownership; an existing caller's ACL is not closed.
    check(unsafe { init(ptr::null()) }, "aclInit")?;
    runtime.initialized = true;
    Ok(runtime)
}

fn inventory(runtime: &Driver) -> Result<Inventory> {
    let mut devices = 0;
    check(
        // SAFETY: ACL initialized successfully and devices is a live writable u32.
        unsafe { (runtime.count)(&mut devices) },
        "aclrtGetDeviceCount",
    )?;
    let soc_name = if devices == 0 {
        None
    } else {
        // SAFETY: ACL owns the returned static NUL-terminated SoC name while
        // the initialized runtime is retained; copy it before finalization.
        let name = unsafe { (runtime.soc)() };
        if name.is_null() {
            return Err("aclrtGetSocName returned null".into());
        }
        // SAFETY: Non-null CANN string with the lifetime described above.
        Some(
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned(),
        )
    };
    let mut api_symbols = Vec::new();
    for name in [
        "aclrtReserveMemAddress",
        "aclrtMapMem",
        "aclmdlRICaptureBegin",
        "aclrtLaunchKernel",
    ] {
        let symbol = std::ffi::CString::new(name).map_err(|e| e.to_string())?;
        // SAFETY: Only resolve an opaque symbol address; it is never called or
        // interpreted as proof that this chip supports the operation.
        if unsafe {
            runtime
                ._library
                .get::<*const std::ffi::c_void>(symbol.as_bytes_with_nul())
        }
        .is_ok()
        {
            api_symbols.push(name.into());
        }
    }
    Ok(Inventory {
        backend: orinfer_device::Backend::Ascend,
        host_arch: std::env::consts::ARCH.into(),
        device_count: devices,
        soc: soc_name,
        runtime_library: runtime.path.display().to_string(),
        api_symbols,
        model_execution_available: true,
    })
}

struct Active {
    driver: Option<Arc<Driver>>,
    users: usize,
}
static ACTIVE: Mutex<Active> = Mutex::new(Active {
    driver: None,
    users: 0,
});
struct DriverLease {
    driver: Option<Arc<Driver>>,
    registered: bool,
}
impl std::ops::Deref for DriverLease {
    type Target = Driver;
    fn deref(&self) -> &Driver {
        self.driver.as_ref().unwrap()
    }
}
impl Drop for DriverLease {
    fn drop(&mut self) {
        if !self.registered {
            return;
        }
        if let Ok(mut active) = ACTIVE.lock() {
            // Finalization and the next initialization share the same lock.
            self.driver.take();
            active.users -= 1;
            if active.users == 0 {
                active.driver.take();
            }
        }
    }
}

fn acquire() -> Result<Arc<DriverLease>> {
    // Serialize process initialization. Every session and inventory query holds
    // a reference, so discovery cannot finalize a live model's ACL runtime.
    let mut active = ACTIVE
        .lock()
        .map_err(|_| "ACL initialization lock poisoned")?;
    if active.driver.is_none() {
        active.driver = Some(Arc::new(load_driver()?));
    }
    active.users += 1;
    Ok(Arc::new(DriverLease {
        driver: active.driver.clone(),
        registered: true,
    }))
}

fn load_driver() -> Result<Driver> {
    if let Some(path) = std::env::var_os("ORINFER_ASCEND_LIBRARY") {
        return open_library(Path::new(&path));
    }
    let mut errors = Vec::new();
    for path in [
        "libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
    ] {
        match open_library(Path::new(path)) {
            Ok(value) => return Ok(value),
            Err(error) => errors.push(error),
        }
    }
    Err(format!(
        "CANN discovery failed; initialize the CANN runtime library environment: {}",
        errors.join("; ")
    ))
}

pub(super) fn inspect() -> Result<Inventory> {
    let driver = acquire()?;
    inventory(&driver)
}

#[cfg(test)]
mod discovery_tests {
    use super::*;
    fn fixture(init_status: i32, count_status: i32) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("acl.c");
        let library = dir.path().join("libacl-test.so");
        std::fs::write(
            &source,
            format!(
                r#"
            #include <stdint.h>
            static int initialized=0,finalized=0;
            int aclInit(const char *p) {{ (void)p;initialized++;return {init_status}; }}
            int aclFinalize(void) {{ finalized++;return 0; }}
            int aclrtGetDeviceCount(uint32_t *n) {{ *n=8;return {count_status}; }}
            const char *aclrtGetSocName(void) {{ return "Ascend910B3"; }}
            int test_finalized(void) {{ return finalized; }}
            int test_initialized(void) {{ return initialized; }}
        "#
            ),
        )
        .unwrap();
        assert!(
            std::process::Command::new("cc")
                .args(["-shared", "-fPIC"])
                .arg(&source)
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        (dir, library)
    }
    #[test]
    fn successful_and_failed_queries_finalize_owned_acl() {
        for (init_status, count_status) in [(0, 0), (0, 7), (7, 0)] {
            let (_dir, path) = fixture(init_status, count_status);
            // SAFETY: The test builds this exact library and retains a second
            // reference so its counters remain accessible after probe cleanup.
            let library = unsafe { Library::new(&path) }.unwrap();
            let result = open_library(&path).and_then(|driver| inventory(&driver));
            if init_status == 0 && count_status == 0 {
                let info = result.unwrap();
                assert_eq!(info.device_count, 8);
                assert_eq!(info.soc.as_deref(), Some("Ascend910B3"));
                assert!(info.model_execution_available);
            } else {
                assert!(result.unwrap_err().contains("ACL error 7"));
            }
            // SAFETY: The fixture declares both functions with this signature.
            unsafe {
                assert_eq!(
                    library
                        .get::<unsafe extern "C" fn() -> i32>(b"test_finalized\0")
                        .unwrap()(),
                    i32::from(init_status == 0)
                );
                assert_eq!(
                    library
                        .get::<unsafe extern "C" fn() -> i32>(b"test_initialized\0")
                        .unwrap()(),
                    1
                );
            }
        }
    }
}
