//! Hash-verified model libraries. The engine owns CUDA; packages own model plans.
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
    let image = std::fs::read(path).map_err(|e| e.to_string())?;
    ModelPackage::open(path, &image)?.describe()
}

#[repr(C)]
struct Header {
    version: u32,
    size: usize,
}

type BindingCache = std::cell::RefCell<
    std::collections::VecDeque<(Vec<(usize, usize)>, std::rc::Rc<abi::StateBindings>)>,
>;

pub(crate) struct ModelPackage {
    handle: *mut c_void,
    api: *const abi::Api,
    input_api: Option<*const abi::inputs::Api>,
    // Retain code until the opaque model and all of its buffers are destroyed.
    _library: Library,
    bindings: BindingCache,
}
impl ModelPackage {
    fn open(path: &Path, image: &[u8]) -> Result<Self> {
        if image.len() < 20
            || &image[..4] != b"\x7fELF"
            || image[4] != 2
            || image[5] != 1
            || u16::from_le_bytes([image[16], image[17]]) != 3
            || u16::from_le_bytes([image[18], image[19]]) != 183
        {
            return Err("Model library must be an aarch64 ELF shared object".into());
        }
        // SAFETY: this explicitly selected native library implements the documented
        // C ABI. The model loader verifies its hash before calling open.
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
            _library: library,
            bindings: Default::default(),
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
        if spec.library.tensor.is_some() {
            return Err("Native execution library must be a standalone file".into());
        }
        if spec.abi_version != abi::ABI_VERSION
            || spec.package.is_empty()
            || spec.version.is_empty()
        {
            return Err("Incompatible model package ABI or identity".into());
        }
        let image = read_identity(base, &spec.library)?;
        let path = resolve_file(base, &spec.library.file)?;
        let mut model = Self::open(&path, &image)?;
        let input_api = if request.metadata.input_assets.is_some() {
            // SAFETY: hash-verified native library exports the documented C extension.
            let entry = unsafe {
                model
                    ._library
                    .get::<unsafe extern "C" fn(u32) -> *const abi::inputs::Api>(
                        abi::inputs::ENTRYPOINT,
                    )
            }
            .map_err(|e| format!("Model input adapter: {e}"))?;
            // SAFETY: the extension returns a static table retained by the library.
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
        } else {
            None
        };
        let input = serde_json::to_vec(&request).map_err(|e| e.to_string())?;
        model.input_api = input_api;
        let description = model.describe()?;
        if description.package != spec.package
            || description.version != spec.version
            || description.target != "sm_87"
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
        let plan = serde_json::from_slice(&bytes).map_err(|e| format!("Model plan: {e}"))?;
        Ok((model, plan))
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
        self.batch(segments, true).map(|(plan, _)| plan)
    }
    pub(crate) fn state_bindings(
        &self,
        segments: &[BatchSegment],
    ) -> Result<std::rc::Rc<abi::StateBindings>> {
        let key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
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
        let mut output = BatchOutput::default();
        let mut error = OwnedBytes::default();
        // SAFETY: this live model and borrowed segments are retained for the call.
        let status = unsafe {
            (self.table().batch)(
                self.handle,
                segments.as_ptr(),
                segments.len(),
                u32::from(include_plan),
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
            Self::decode_batch(&output)
        };
        // SAFETY: output belongs to this library and is released exactly once.
        unsafe {
            (self.table().free_batch)(output);
        }
        result
    }
    fn decode_batch(output: &BatchOutput) -> Result<abi::BatchPlan> {
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
            for view in span(command.views, command.view_count, 256)? {
                if views
                    .insert(
                        string(view.name)?,
                        BufferView {
                            buffer: string(view.buffer)?,
                            offset: view.offset,
                        },
                    )
                    .is_some()
                {
                    return Err("Duplicate model package buffer view".into());
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
    std::str::from_utf8(span(value.data, value.len, 65536)?)
        .map(str::to_owned)
        .map_err(|e| e.to_string())
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
        let root = std::env::temp_dir().join(format!(
            "orinfer-native-model-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
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
    fn hash_abi_identity_and_compute_mismatches_fail_before_model_creation() {
        for mode in 0..4 {
            let (root, mut spec, mut request) = fixture();
            match mode {
                0 => spec.library.sha256 = "0".repeat(64),
                1 => spec.abi_version = 99,
                2 => spec.version = "unexpected".into(),
                _ => request.compute_policy = "a4_not_supported".into(),
            }
            assert!(ModelPackage::create(&root, &spec, request).is_err());
            std::fs::remove_dir_all(root).unwrap();
        }
    }
}

#[cfg(test)]
mod portable_tests {
    #[test]
    fn non_aarch64_execution_library_is_rejected_before_loading() {
        let mut image = [0u8; 20];
        image[..6].copy_from_slice(b"\x7fELF\x02\x01");
        image[16..18].copy_from_slice(&3u16.to_le_bytes());
        image[18..20].copy_from_slice(&62u16.to_le_bytes());
        let error = super::ModelPackage::open(std::path::Path::new("unused.so"), &image)
            .err()
            .unwrap();
        assert!(error.contains("aarch64 ELF shared object"));
    }
}
