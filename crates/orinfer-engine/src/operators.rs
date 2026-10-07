//! Versioned, weight-independent model execution packages. Model control flow is supplied by a versioned native library.
use crate::{
    architecture::{Architecture, ComputePolicy, PrefillProfile},
    artifact::{Access, Buffer, Dtype, Kernel, Result},
    weights::TensorIdentity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub mod dynamic;

pub const RUNTIME_ABI: u32 = orinfer_model_sdk::abi::RUNTIME_ABI;

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BufferContract {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub layout: String,
    pub alignment: u64,
    pub access: Access,
}
impl From<&Buffer<TensorIdentity>> for BufferContract {
    fn from(b: &Buffer<TensorIdentity>) -> Self {
        Self {
            name: b.name.clone(),
            dtype: b.dtype,
            shape: b.shape.clone(),
            layout: b.layout.clone(),
            alignment: b.alignment,
            access: b.access,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPackage {
    pub schema_version: u32,
    pub runtime_abi: u32,
    pub target: String,
    pub architecture: Architecture,
    pub compute_policy: ComputePolicy,
    pub config_signature: Value,
    pub prefill_profiles: Vec<PrefillProfile>,
    #[serde(default)]
    pub batch_profiles: Vec<usize>,
    #[serde(default)]
    pub dynamic_batch_kernels: Vec<dynamic::DynamicBatchKernel>,
    #[serde(default)]
    pub prefill_batch_profiles: Vec<PrefillProfile>,
    #[serde(default)]
    pub greedy_sampling: bool,
    #[serde(default)]
    pub batch_gdn: bool,
    pub buffer_contracts: Vec<BufferContract>,
    pub kernels: Vec<Kernel>,
    pub toolchain: BTreeMap<String, String>,
    pub execution: crate::model_package::ExecutionLibrary,
}
impl ExecutionPackage {
    pub(crate) fn validate(
        &self,
        architecture: &str,
        policy: &str,
        buffers: &[Buffer<TensorIdentity>],
    ) -> Result<()> {
        if self.schema_version != 1
            || self.runtime_abi != RUNTIME_ABI
            || self.target != "sm_87"
            || self.architecture != architecture
            || self.compute_policy != policy
        {
            return Err(
                "Incompatible model execution package architecture, policy, target or ABI".into(),
            );
        }
        let mut seen = std::collections::BTreeSet::new();
        if self
            .batch_profiles
            .iter()
            .any(|&b| !(2..=128).contains(&b) || !b.is_power_of_two() || !seen.insert(b))
        {
            return Err("Invalid or duplicate batch profile".into());
        }
        let kernels: BTreeMap<_, _> = self.kernels.iter().map(|k| (k.name.as_str(), k)).collect();
        let mut templates = std::collections::BTreeSet::new();
        for template in &self.dynamic_batch_kernels {
            if !templates.insert(template.name.as_str()) {
                return Err("Duplicate dynamic batch template".into());
            }
            template.validate(
                kernels
                    .get(template.name.as_str())
                    .ok_or("Missing dynamic batch kernel")?,
            )?;
        }
        if !templates.is_empty()
            && (!self.batch_profiles.contains(&128)
                || kernels
                    .keys()
                    .filter(|name| {
                        name.starts_with("batch_m128/") || name.starts_with("batch_gdn_m128/")
                    })
                    .any(|name| !templates.contains(name)))
        {
            return Err("Incomplete dynamic batch contracts".into());
        }
        seen.clear();
        if self
            .prefill_batch_profiles
            .iter()
            .any(|p| p.tokens <= 128 || !p.tokens.is_power_of_two() || !seen.insert(p.tokens))
            || (!self.prefill_batch_profiles.is_empty() && self.batch_profiles.is_empty())
        {
            return Err("Invalid joint prefill profile".into());
        }
        let actual: Vec<_> = buffers.iter().map(BufferContract::from).collect();
        if actual != self.buffer_contracts {
            return Err(
                "Model buffer dtype/shape/layout/access contract differs from model execution package"
                    .into(),
            );
        }
        Ok(())
    }
}

pub fn default_cache() -> PathBuf {
    if let Some(path) = std::env::var_os("ORINFER_EXECUTION_CACHE") {
        return path.into();
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("orinfer/packages");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
        .join(".cache/orinfer/packages")
}

pub(crate) fn resolve(model_cache: &Path, id: &str) -> Result<PathBuf> {
    if id.len() != 64
        || !id
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err("Malformed model execution package digest".into());
    }
    // An explicit shared cache takes precedence over a bundled package.
    // Missing entries fall back; a present corrupt entry fails validation.
    let installed = default_cache().join(id);
    if installed.exists() {
        return installed.canonicalize().map_err(|e| e.to_string());
    }
    crate::artifact::resolve_file(model_cache, &format!("packages/{id}/package.json"))?
        .parent()
        .map(Path::to_owned)
        .ok_or_else(|| "Operator package has no directory".into())
}
