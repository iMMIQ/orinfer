//! Versioned, weight-independent operator packages. Plans are registered in Rust.
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

pub const RUNTIME_ABI: u32 = 1;

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
pub struct OperatorPackage {
    pub schema_version: u32,
    pub runtime_abi: u32,
    pub target: String,
    pub architecture: Architecture,
    pub compute_policy: ComputePolicy,
    pub config_signature: Value,
    pub prefill_profiles: Vec<PrefillProfile>,
    pub buffer_contracts: Vec<BufferContract>,
    pub kernels: Vec<Kernel>,
    pub toolchain: BTreeMap<String, String>,
}
impl OperatorPackage {
    pub(crate) fn validate(
        &self,
        architecture: Architecture,
        policy: ComputePolicy,
        signature: &Value,
        buffers: &[Buffer<TensorIdentity>],
    ) -> Result<()> {
        if self.schema_version != 1
            || self.runtime_abi != RUNTIME_ABI
            || self.target != "sm_87"
            || self.architecture != architecture
            || self.compute_policy != policy
        {
            return Err("Incompatible operator package architecture, policy, target or ABI".into());
        }
        if &self.config_signature != signature {
            return Err("Operator package does not support this model configuration".into());
        }
        let actual: Vec<_> = buffers.iter().map(BufferContract::from).collect();
        if actual != self.buffer_contracts {
            return Err(
                "Model buffer dtype/shape/layout/access contract differs from operator package"
                    .into(),
            );
        }
        Ok(())
    }
}

pub fn default_cache() -> PathBuf {
    if let Some(path) = std::env::var_os("ORIN_OPERATOR_CACHE") {
        return path.into();
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(path).join("orin-llm/operators");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into()))
        .join(".cache/orin-llm/operators")
}

pub(crate) fn resolve(model_cache: &Path, id: &str) -> Result<PathBuf> {
    if id.len() != 64
        || !id
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        return Err("Malformed operator package digest".into());
    }
    // An explicit shared cache takes precedence over a bundled package.
    // Missing entries fall back; a present corrupt entry fails validation.
    let installed = default_cache().join(id);
    if installed.exists() {
        return installed.canonicalize().map_err(|e| e.to_string());
    }
    crate::artifact::resolve_file(model_cache, &format!("operators/{id}/package.json"))?
        .parent()
        .map(Path::to_owned)
        .ok_or_else(|| "Operator package has no directory".into())
}
