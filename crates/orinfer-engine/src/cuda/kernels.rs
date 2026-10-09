//! Verified AOT assets with thread-affine, on-demand CUDA function resolution.
use super::{Kernel, Session, check, sys};
use crate::artifact::Result;
use std::{collections::BTreeMap, ffi::CString, path::Path, ptr};

struct DeviceBounds {
    shared: u32,
    default_shared: u32,
    block: [u32; 3],
    grid: [u32; 3],
}
#[derive(Clone, Copy)]
struct Function {
    handle: sys::CUfunction,
    threads: u32,
    static_shared: u32,
}
impl DeviceBounds {
    fn new(session: &Session) -> Result<Self> {
        let attribute = |attribute| -> Result<u32> {
            let mut value = 0;
            // SAFETY: Live checked device; driver writes one scalar.
            unsafe {
                check(
                    sys::cuDeviceGetAttribute(&mut value, attribute, session.device),
                    "kernel device limits",
                )?;
            }
            u32::try_from(value).map_err(|_| "Negative CUDA resource limit".into())
        };
        use sys::CUdevice_attribute::*;
        Ok(Self {
            shared: attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?,
            default_shared: attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)?,
            block: [
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)?,
            ],
            grid: [
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)?,
                attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z)?,
            ],
        })
    }
    fn validate(&self, spec: &Kernel, function: Function) -> Result<()> {
        let threads = spec.block.iter().try_fold(1u32, |n, d| n.checked_mul(*d));
        let shared = spec.shared_memory_bytes.checked_add(function.static_shared);
        if spec.block.contains(&0)
            || spec.grid.contains(&0)
            || threads.is_none_or(|n| n > function.threads)
            || shared.is_none_or(|n| n > self.shared)
            || (0..3).any(|axis| {
                spec.block[axis] > self.block[axis] || spec.grid[axis] > self.grid[axis]
            })
        {
            return Err(format!("{} exceeds function/device resources", spec.name));
        }
        Ok(())
    }
}

pub(super) struct Catalog {
    images: BTreeMap<crate::artifact::AssetKey, Vec<u8>>,
    modules: BTreeMap<crate::artifact::AssetKey, sys::CUmodule>,
    functions: BTreeMap<(crate::artifact::AssetKey, String), Function>,
    shared: BTreeMap<(crate::artifact::AssetKey, String), u32>,
    bounds: DeviceBounds,
}
impl Catalog {
    pub(super) fn new(session: &Session, base: &Path, kernels: &[Kernel]) -> Result<Self> {
        let mut images = BTreeMap::new();
        let mut verified = BTreeMap::<crate::artifact::AssetKey, String>::new();
        let mut shared = BTreeMap::new();
        let mut reader = crate::artifact::AssetReader::default();
        for spec in kernels {
            // Validate ALL assets now, even when driver loading is deferred.
            for identity in [&spec.module, &spec.source, &spec.host_abi] {
                if let Some(hash) = verified.get(&identity.key()) {
                    if hash != &identity.sha256 {
                        return Err("Conflicting kernel asset identities".into());
                    }
                    continue;
                }
                let image = reader.read(base, identity)?;
                if identity.key() == spec.module.key() {
                    if !image.starts_with(b"\x7fELF") {
                        return Err("Expected cubin ELF".into());
                    }
                    images.insert(identity.key(), image);
                }
                verified.insert(identity.key(), identity.sha256.clone());
            }
            shared
                .entry((spec.module.key(), spec.symbol.clone()))
                .and_modify(|value: &mut u32| *value = (*value).max(spec.shared_memory_bytes))
                .or_insert(spec.shared_memory_bytes);
        }
        Ok(Self {
            images,
            modules: BTreeMap::new(),
            functions: BTreeMap::new(),
            shared,
            bounds: DeviceBounds::new(session)?,
        })
    }
    pub(super) fn loaded_modules(&self) -> usize {
        self.modules.len()
    }
    pub(super) fn registered_modules(&self) -> usize {
        self.modules.len() + self.images.len()
    }
    pub(super) fn resolve(&mut self, session: &Session, spec: &Kernel) -> Result<sys::CUfunction> {
        let key = (spec.module.key(), spec.symbol.clone());
        let function = if let Some(function) = self.functions.get(&key) {
            *function
        } else {
            let module = if let Some(module) = self.modules.get(&spec.module.key()) {
                *module
            } else {
                let image = self
                    .images
                    .get(&spec.module.key())
                    .ok_or("Unverified CUDA module")?;
                let mut module = ptr::null_mut();
                // SAFETY: Verified SM87 ELF; the driver copies the image. Session
                // owns the module until all queued/captured work is destroyed.
                unsafe {
                    check(
                        sys::cuModuleLoadData(&mut module, image.as_ptr().cast()),
                        "load model module",
                    )?;
                }
                session.modules.borrow_mut().push(module);
                self.modules.insert(spec.module.key(), module);
                self.images.remove(&spec.module.key());
                module
            };
            let symbol = CString::new(spec.symbol.as_str()).map_err(|e| e.to_string())?;
            let mut handle = ptr::null_mut();
            let (mut threads, mut static_shared) = (0, 0);
            // SAFETY: The live module outlives the function and its resource queries.
            unsafe {
                check(
                    sys::cuModuleGetFunction(&mut handle, module, symbol.as_ptr()),
                    "model function",
                )?;
                check(
                    sys::cuFuncGetAttribute(
                        &mut threads,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                        handle,
                    ),
                    "function threads",
                )?;
                check(
                    sys::cuFuncGetAttribute(
                        &mut static_shared,
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                        handle,
                    ),
                    "function shared memory",
                )?;
            }
            let function = Function {
                handle,
                threads: u32::try_from(threads).map_err(|_| "Invalid function threads")?,
                static_shared: u32::try_from(static_shared)
                    .map_err(|_| "Invalid static shared memory")?,
            };
            let dynamic = self.shared[&key];
            let shared = dynamic
                .checked_add(function.static_shared)
                .ok_or("Shared memory overflow")?;
            if shared > self.bounds.shared {
                return Err(format!("{} shared memory limit", spec.name));
            }
            if shared > self.bounds.default_shared {
                // Configure the maximum of ALL bindings once. A later profile
                // must not lower the allowance of an already captured function.
                // SAFETY: Resource extent checked against the actual device.
                unsafe {
                    check(sys::cuFuncSetAttribute(handle, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic as i32), "opt-in shared memory")?;
                }
            }
            self.functions.insert(key, function);
            function
        };
        self.bounds.validate(spec, function)?;
        Ok(function.handle)
    }
    pub(super) fn validate_launch(&self, spec: &Kernel) -> Result<()> {
        let function = *self
            .functions
            .get(&(spec.module.key(), spec.symbol.clone()))
            .ok_or("Unresolved launch resources")?;
        self.bounds.validate(spec, function)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_resources_reject_dynamic_grid_and_shared_memory_overflows() {
        let identity = serde_json::json!({"file":"k.cubin", "sha256":"0".repeat(64)});
        let mut spec: Kernel = serde_json::from_value(serde_json::json!({
            "name":"k", "symbol":"main_kernel", "module":identity,"source":identity,"host_abi":identity,
            "grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":32768,"args":[],"cooperative":false
        })).unwrap();
        let bounds = DeviceBounds {
            shared: 65536,
            default_shared: 49152,
            block: [1024, 1024, 64],
            grid: [2147483647, 65535, 65535],
        };
        let function = Function {
            handle: ptr::null_mut(),
            threads: 256,
            static_shared: 1024,
        };
        assert!(bounds.validate(&spec, function).is_ok());
        spec.grid[1] = 65536;
        assert!(bounds.validate(&spec, function).is_err());
        spec.grid[1] = 1;
        spec.block[0] = 512;
        assert!(bounds.validate(&spec, function).is_err());
        spec.block[0] = 128;
        spec.shared_memory_bytes = 65536;
        assert!(bounds.validate(&spec, function).is_err());
        spec.shared_memory_bytes = u32::MAX;
        assert!(bounds.validate(&spec, function).is_err());
    }
}
