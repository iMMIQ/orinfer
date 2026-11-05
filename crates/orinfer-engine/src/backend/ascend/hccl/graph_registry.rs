//! Device-scoped CANN 9.2 Graph callbacks, installed before loading native HCCL.
use crate::artifact::Result;
use libloading::Library;
use std::sync::OnceLock;

type GetDevice = unsafe extern "C" fn(*mut i32) -> i32;
type Configure = unsafe extern "C" fn(GetDevice) -> i32;
type Status = unsafe extern "C" fn() -> i32;
struct Adapter {
    _library: Library,
    configure: Configure,
    status: Status,
}
static ADAPTER: OnceLock<Result<Adapter>> = OnceLock::new();

pub(in crate::backend::ascend) fn install() -> Result<()> {
    ADAPTER
        .get_or_init(load)
        .as_ref()
        .map(|_| ())
        .map_err(Clone::clone)
}

#[cfg(target_os = "linux")]
fn load() -> Result<Adapter> {
    use std::{io::Write, os::fd::FromRawFd};
    // SAFETY: NOLOAD only acquires an existing library reference; balance that
    // reference before rejecting an already-bound vendor factory.
    let existing =
        unsafe { libc::dlopen(c"libhcomm.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD) };
    if !existing.is_null() {
        // SAFETY: Release only the NOLOAD reference acquired above.
        unsafe { libc::dlclose(existing) };
        return Err("HCOMM was loaded before the CANN Graph isolation adapter; start orinfer before loading HCCL".into());
    }
    // The compiled adapter is part of the engine, not a model asset. Keep it in
    // sealed anonymous memory: no compiler, writable .so path or payload hashing
    // at startup. Its library stays resident through all CANN teardown callbacks.
    // SAFETY: A fixed NUL-terminated name and Linux's documented memfd flags.
    let fd = unsafe {
        libc::memfd_create(
            c"orinfer-hccl-graph-registry".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if fd < 0 {
        return Err(format!(
            "HCCL registry memfd: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: The new owned descriptor is transferred exactly once to File.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(include_bytes!(env!("ORINFER_HCCL_REGISTRY")))
        .map_err(|e| format!("HCCL registry image: {e}"))?;
    // SAFETY: Seal only this private descriptor after writing the complete image.
    if unsafe {
        libc::fcntl(
            fd,
            libc::F_ADD_SEALS,
            libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL,
        )
    } < 0
    {
        return Err(format!(
            "Seal HCCL registry: {}",
            std::io::Error::last_os_error()
        ));
    }
    let path = std::path::PathBuf::from(format!("/proc/self/fd/{fd}"));
    // SAFETY: This exact build-time image exports only the device registry
    // factory and its two C entrypoints. GLOBAL must precede CANN library loading
    // so the vendor callback methods bind to the device-scoped factory.
    let library: Library = unsafe {
        libloading::os::unix::Library::open(
            Some(&path),
            libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
        )
    }
    .map_err(|e| format!("Load CANN Graph registry adapter: {e}"))?
    .into();
    // SAFETY: Both entrypoints are compiled from hccl_graph_registry.cc above.
    unsafe {
        Ok(Adapter {
            configure: *library
                .get(b"orinfer_hccl_graph_registry_configure\0")
                .map_err(|e| e.to_string())?,
            status: *library
                .get(b"orinfer_hccl_graph_registry_status\0")
                .map_err(|e| e.to_string())?,
            _library: library,
        })
    }
}

#[cfg(not(target_os = "linux"))]
fn load() -> Result<Adapter> {
    Err("Native Ascend execution requires Linux".into())
}

pub(super) fn configure(library: &Library) -> Result<()> {
    install()?;
    // CANN 9.2 must expose matching callback methods before the adapter can
    // allocate native objects. CPU-only C ABI fixtures have no such registry.
    const FACTORY: &[u8] = b"_ZN4hccl16AclgraphCallback11GetInstanceEv\0";
    // SAFETY: Resolve opaque addresses for ABI presence checks only.
    if unsafe { library.get::<*mut std::ffi::c_void>(FACTORY) }.is_err() {
        #[cfg(test)]
        return Ok(());
        #[cfg(not(test))]
        return Err("CANN 9.2 HCOMM Graph callback ABI is required".into());
    }
    for name in [
        b"_ZN4hccl16AclgraphCallback15RegisterModelIdEPNS_16HcclCommunicatorEPvy\0".as_slice(),
        b"_ZN4hccl16AclgraphCallback15CleanCaptureResEy\0".as_slice(),
    ] {
        // SAFETY: No invocation; CANN owns these symbols while Api is alive.
        unsafe { library.get::<*mut std::ffi::c_void>(name) }
            .map_err(|e| format!("CANN 9.2 Graph registry ABI: {e}"))?;
    }
    // The native object declaration is pinned to HCOMM 9.2.0. Query component
    // metadata once; a different private ABI must be reviewed before use.
    let mut version = 0;
    let mut package = *b"hcomm\0";
    // SAFETY: Public CANN package-version API and writable bounded outputs.
    let status = unsafe {
        let query = library
            .get::<unsafe extern "C" fn(*mut std::ffi::c_char, *mut i32) -> i32>(
                b"aclsysGetVersionNum\0",
            )
            .map_err(|e| e.to_string())?;
        query(package.as_mut_ptr().cast(), &mut version)
    };
    if status != 0 || version != 90_200_000 {
        return Err(format!(
            "CANN HCOMM 9.2.0 Graph registry ABI required: status {status}, version {version}"
        ));
    }
    // SAFETY: Public CANN signature. The owning HCCL Api retains its library.
    let query = unsafe {
        *library
            .get::<GetDevice>(b"aclrtGetDevice\0")
            .map_err(|e| e.to_string())?
    };
    let adapter = ADAPTER.get().unwrap().as_ref().map_err(Clone::clone)?;
    // SAFETY: Function ABI is compiled with this engine and query is retained.
    if unsafe { (adapter.configure)(query) } != 0 {
        return Err("Configure CANN Graph registry device query failed".into());
    }
    check()
}

pub(in crate::backend::ascend) fn check() -> Result<()> {
    if let Some(Ok(adapter)) = ADAPTER.get() {
        // SAFETY: Process-lifetime adapter and scalar C return value.
        let status = unsafe { (adapter.status)() };
        if status != 0 {
            crate::error::record_ascend(status);
            return Err(format!(
                "CANN Graph registry device lookup failed: ACL error {status}"
            ));
        }
    }
    Ok(())
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn native_registry_preserves_other_device_when_model_ids_are_reused() {
        install().unwrap();
        let directory = tempfile::tempdir().unwrap();
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let library = directory.path().join("libhccl-registry-fixture.so");
        assert!(
            std::process::Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
                .args([
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-D_GLIBCXX_USE_CXX11_ABI=0",
                    "-static-libstdc++",
                    "-static-libgcc",
                    "-Wl,--exclude-libs,ALL",
                    "-I",
                ])
                .arg(manifest.join("../../tools/operators/ascend"))
                .arg(manifest.join("tests/fixtures/hccl_graph_registry.cc"))
                .arg("-o")
                .arg(&library)
                .status()
                .unwrap()
                .success()
        );
        // SAFETY: This test compiled the fixed C ABI and retains its code below.
        let library = unsafe { Library::new(&library) }.unwrap();
        configure(&library).unwrap();
        // SAFETY: Exact scalar fixture signatures; no actual device handles.
        unsafe {
            let device = library
                .get::<unsafe extern "C" fn(i32)>(b"fixture_set_device\0")
                .unwrap();
            let capture = library
                .get::<unsafe extern "C" fn(u64, usize) -> i32>(b"fixture_capture\0")
                .unwrap();
            let owner = library
                .get::<unsafe extern "C" fn(u64) -> i32>(b"fixture_owner\0")
                .unwrap();
            let destroy = library
                .get::<unsafe extern "C" fn(usize)>(b"fixture_destroy\0")
                .unwrap();
            device(0);
            assert_eq!(capture(49, 101), 0);
            device(1);
            assert_eq!(capture(49, 202), 0);
            assert_eq!(owner(49), 202);
            device(0);
            assert_eq!(owner(49), 101);
            destroy(101);
            assert_eq!(owner(49), 0);
            assert_eq!(capture(49, 303), 0);
            device(1);
            assert_eq!(owner(49), 202);
            destroy(202);
            device(0);
            assert_eq!(owner(49), 303);
            destroy(303);
        }
        check().unwrap();
        // The process-lifetime adapter retains this fixture's device-query
        // pointer, just as real Api/Driver owners retain the CANN library.
        std::mem::forget(library);
    }
}
