//! Native model implementations built with the engine's common source workspace.
//! Only the public SDK links this library to the runtime; no engine dependency.
pub use orinfer_model_sdk::{artifact, execution, model, operators};
mod adapter;
mod flash_next;
mod policy;
mod qwen3_5;
mod registry;
use orinfer_model_sdk::{
    abi::{self, CreateRequest, CreatedPlan, ModelImplementation},
    architecture::BatchSegment,
};
struct Models {
    adapter: Box<dyn adapter::Adapter>,
}
impl ModelImplementation for Models {
    fn prepare(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
    ) -> artifact::Result<Vec<(String, Vec<u8>)>> {
        self.adapter.prepare(program, tokens, history)
    }
    fn describe() -> abi::PackageInfo {
        abi::PackageInfo {
            package: env!("CARGO_PKG_NAME").into(),
            version: env!("CARGO_PKG_VERSION").into(),
            target: "sm_87".into(),
            runtime_abi: abi::RUNTIME_ABI,
            architectures: registry::architectures(),
            compute_policies: registry::policies(),
        }
    }
    fn create(request: CreateRequest) -> artifact::Result<(Self, CreatedPlan)> {
        let (adapter, plan) = registry::create(request)?;
        Ok((Self { adapter }, plan))
    }
    fn batch(
        &self,
        segments: &[BatchSegment],
        include_plan: bool,
    ) -> artifact::Result<abi::BatchPlan> {
        self.adapter.batch(segments, include_plan)
    }
    fn visual(
        &self,
        tokens: &[u32],
        images: &[abi::ImageGrid],
        capacity: usize,
    ) -> artifact::Result<(Vec<i32>, Vec<u32>)> {
        self.adapter.visual(tokens, images, capacity)
    }
}
static API: abi::Api = abi::Api {
    abi_version: abi::ABI_VERSION,
    struct_size: std::mem::size_of::<abi::Api>(),
    describe: abi::describe::<Models>,
    create: abi::create::<Models>,
    destroy: abi::destroy::<Models>,
    batch: abi::batch::<Models>,
    free_bytes: abi::free_bytes,
    free_batch: abi::free_batch,
    visual: abi::visual::<Models>,
    free_visual: abi::free_visual,
};
#[unsafe(no_mangle)]
pub extern "C" fn orinfer_model_v1(version: u32) -> *const abi::Api {
    if version == abi::ABI_VERSION {
        &API
    } else {
        std::ptr::null()
    }
}
static INPUT_API: abi::inputs::Api = abi::inputs::Api {
    version: 1,
    struct_size: std::mem::size_of::<abi::inputs::Api>(),
    prepare: abi::inputs::prepare::<Models>,
    prepare_program: abi::inputs::prepare_program::<Models>,
    free: abi::inputs::free,
};
#[unsafe(no_mangle)]
pub extern "C" fn orinfer_model_inputs_v1(version: u32) -> *const abi::inputs::Api {
    if version == 1 {
        &INPUT_API
    } else {
        std::ptr::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn package_identity_and_registered_capabilities_match_the_exported_library() {
        assert!(orinfer_model_v1(0).is_null());
        let mut bytes = abi::OwnedBytes::default();
        // SAFETY: the table is static; bytes is a live writable output slot.
        assert_eq!(unsafe { (API.describe)(&mut bytes) }, 0);
        // SAFETY: describe supplies len readable bytes until its free callback.
        let info: abi::PackageInfo =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(bytes.data, bytes.len) })
                .unwrap();
        // SAFETY: return the exact library-owned allocation once.
        unsafe { (API.free_bytes)(bytes) };
        assert_eq!(info.package, "orinfer-models");
        assert_eq!(info.runtime_abi, 1);
        assert_eq!(info.architectures, ["flash_next", "qwen3_5"]);
        assert_eq!(info.compute_policies, ["int8_quality"]);
    }
}
