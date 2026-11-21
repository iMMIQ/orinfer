//! Versioned, weight-independent model execution packages. Rust adapters compose model control flow from data bindings.
use crate::{
    architecture::{Architecture, ComputePolicy, PrefillProfile},
    artifact::{Access, Buffer, Dtype, Result},
    weights::TensorIdentity,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

pub use orinfer_model_sdk::operators::dynamic;
pub mod generated;
pub mod install;
pub(crate) mod shared_json;

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
    pub target: String,
    pub architecture: Architecture,
    pub compute_policy: ComputePolicy,
    pub config_signature: Value,
    pub prefill_profiles: Vec<PrefillProfile>,
    #[serde(default)]
    pub batch_profiles: Vec<usize>,
    #[serde(default)]
    pub prefill_batch_profiles: Vec<PrefillProfile>,
    #[serde(default)]
    pub greedy_sampling: bool,
    #[serde(default)]
    pub batch_gdn: bool,
    pub buffer_contracts: Vec<BufferContract>,
    pub catalog: orinfer_model_sdk::catalog::Catalog,
    pub toolchain: BTreeMap<String, String>,
    pub bindings: crate::artifact::FileIdentity,
}
impl ExecutionPackage {
    pub(crate) fn validate(
        &self,
        architecture: &str,
        policy: &str,
        buffers: &[Buffer<TensorIdentity>],
    ) -> Result<()> {
        self.validate_metadata()?;
        if self.architecture != architecture || self.compute_policy != policy {
            return Err("Incompatible model execution package architecture or policy".into());
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

    fn validate_metadata(&self) -> Result<()> {
        let target = orinfer_model_sdk::device::Target::from_package(&self.target)?;
        if self.schema_version != 2
            || self.architecture.is_empty()
            || self.compute_policy.is_empty()
        {
            return Err(
                "Incompatible model execution package schema, architecture, policy or target"
                    .into(),
            );
        }
        if self.bindings.file != "bindings.json"
            || self.bindings.tensor.is_some()
            || !crate::artifact::is_sha256_hex(&self.bindings.sha256)
            || self.catalog.assets.iter().any(|asset| {
                asset.file != "assets.safetensors"
                    || asset.tensor.as_ref().is_none_or(String::is_empty)
                    || !crate::artifact::is_sha256_hex(&asset.sha256)
            })
        {
            return Err("Invalid binding or asset identity in execution package".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        if self
            .batch_profiles
            .iter()
            .any(|&b| !(2..=128).contains(&b) || !b.is_power_of_two() || !seen.insert(b))
        {
            return Err("Invalid or duplicate batch profile".into());
        }
        self.catalog.validate_target(&target)?;
        seen.clear();
        if self
            .prefill_batch_profiles
            .iter()
            .any(|p| p.tokens <= 128 || !p.tokens.is_power_of_two() || !seen.insert(p.tokens))
            || (!self.prefill_batch_profiles.is_empty() && self.batch_profiles.is_empty())
        {
            return Err("Invalid joint prefill profile".into());
        }
        Ok(())
    }

    pub(crate) fn read_bindings(
        &self,
        root: &Path,
        verify: bool,
    ) -> Result<orinfer_model_sdk::catalog::Bindings> {
        let mut reader = crate::artifact::AssetReader::default();
        let raw = if verify {
            let raw = reader.read(root, &self.bindings)?;
            let mut checked = std::collections::BTreeSet::new();
            for asset in &self.catalog.assets {
                if checked.insert((asset.key(), &asset.sha256)) {
                    reader.read(root, asset)?;
                }
            }
            raw
        } else {
            reader.read_unverified(root, &self.bindings)?
        };
        let registry: orinfer_model_sdk::catalog::BindingRegistry =
            shared_json::decode(&raw).map_err(|e| format!("Execution bindings: {e}"))?;
        registry.bind(&self.catalog)
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
    if !crate::artifact::is_sha256_hex(id) {
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
