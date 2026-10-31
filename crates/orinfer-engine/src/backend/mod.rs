//! Device implementation boundary for model execution and diagnostics.
mod arena;
pub mod ascend;
pub(crate) mod cuda;
mod executor;
mod growth;
pub(crate) mod report;
pub(crate) mod snapshot;
mod values;
pub(crate) use executor::{Executor, LoadStats};
pub use orinfer_device::{Backend, DeviceInfo, GraphMode, Target};
pub(crate) use values::floats;

/// Symbol presence is diagnostic information, not a hardware capability promise.
#[derive(Debug, serde::Serialize)]
pub struct Inventory {
    pub backend: Backend,
    pub host_arch: String,
    pub device_count: u32,
    pub soc: Option<String>,
    pub runtime_library: String,
    pub api_symbols: Vec<String>,
    pub model_execution_available: bool,
}
pub fn inspect(backend: Backend) -> crate::artifact::Result<Inventory> {
    match backend {
        Backend::Ascend => ascend::inspect(),
        Backend::Cuda => cuda::inspect(),
    }
}
