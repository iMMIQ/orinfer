//! Custom model-directory loader. Only initialized tensor data lives with weights;
//! operator packages supply implementations, and architecture code builds plans.
use crate::{
    architecture::{self, Architecture, ComputePolicy, Configuration},
    artifact::{Access, Result, resolve_file, sha256},
    model::Manifest,
    operators::{self, OperatorPackage},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BufferScope {
    Weights,
    Sequence,
    Workspace,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    schema_version: u32,
    architecture: Architecture,
    compute_policy: ComputePolicy,
    operator_package: String,
    buffer_scopes: BTreeMap<String, BufferScope>,
    metadata: Manifest,
}

pub(crate) struct PreparedModel {
    pub plan: Manifest,
    pub weights_root: PathBuf,
    pub kernel_root: PathBuf,
    pub fingerprint: String,
    pub scopes: BTreeMap<String, BufferScope>,
    pub operator_package: String,
    pub architecture: Architecture,
    pub policy: ComputePolicy,
    pub decode_programs: std::collections::BTreeSet<String>,
}

pub(crate) fn load(path: &Path) -> Result<PreparedModel> {
    let descriptor = crate::model::resolve_manifest(path)?;
    let weights_root = descriptor
        .parent()
        .ok_or("Model needs cache directory")?
        .to_owned();
    let root = weights_root.parent().ok_or("Model needs root directory")?;
    let raw = fs::read(&descriptor).map_err(|e| e.to_string())?;
    let mut model: Descriptor =
        serde_json::from_slice(&raw).map_err(|e| format!("Model descriptor: {e}"))?;
    if model.schema_version != 1
        || !model.metadata.kernels.is_empty()
        || !model.metadata.programs.is_empty()
    {
        return Err("Expected custom model descriptor without embedded kernels or programs".into());
    }
    let config_path = resolve_file(root, "config.json")?;
    let config_raw = fs::read(config_path).map_err(|e| e.to_string())?;
    let config =
        Configuration::parse(serde_json::from_slice(&config_raw).map_err(|e| e.to_string())?)?;
    if config.architecture != model.architecture {
        return Err("Model architecture differs from config.json".into());
    }
    let kernel_root = operators::resolve(&weights_root, &model.operator_package)?;
    let package_raw =
        fs::read(resolve_file(&kernel_root, "package.json")?).map_err(|e| e.to_string())?;
    if sha256(&package_raw) != model.operator_package {
        return Err("Operator package digest mismatch".into());
    }
    let package: OperatorPackage =
        serde_json::from_slice(&package_raw).map_err(|e| format!("Operator package: {e}"))?;
    package.validate(
        model.architecture,
        model.compute_policy,
        &config.signature,
        &model.metadata.buffers,
    )?;
    validate_scopes(&model.metadata, &model.buffer_scopes)?;
    let decode_programs =
        architecture::build(&config, &mut model.metadata, &package.prefill_profiles)?;
    model.metadata.kernels = package.kernels;
    model.metadata.validate()?;
    // The descriptor pins the package; configuration also affects generated plans.
    let fingerprint = sha256(&[raw, package_raw, config_raw].concat());
    Ok(PreparedModel {
        plan: model.metadata,
        weights_root,
        kernel_root,
        fingerprint,
        scopes: model.buffer_scopes,
        operator_package: model.operator_package,
        architecture: model.architecture,
        policy: model.compute_policy,
        decode_programs,
    })
}

fn validate_scopes(model: &Manifest, scopes: &BTreeMap<String, BufferScope>) -> Result<()> {
    if scopes.len() != model.buffers.len() {
        return Err("Every buffer must have exactly one allocation scope".into());
    }
    for buffer in &model.buffers {
        let scope = scopes
            .get(&buffer.name)
            .ok_or("Missing buffer allocation scope")?;
        if (buffer.access == Access::Read) != (*scope == BufferScope::Weights) {
            return Err(format!("{}: invalid weight allocation scope", buffer.name));
        }
    }
    for name in model.reset_buffers.iter().chain([
        &model.input,
        &model.token,
        &model.status,
        &model.position,
    ]) {
        if scopes.get(name) != Some(&BufferScope::Sequence) {
            return Err(format!("{name}: request reset must target sequence state"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    #[test]
    fn rejects_embedded_user_execution_program_before_loading_resources() {
        let root = std::env::temp_dir().join(format!(
            "orin-loader-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(root.join("cache")).unwrap();
        let descriptor = serde_json::json!({"schema_version":1,"architecture":"qwen3_5",
            "compute_policy":"int8_quality","operator_package":"0".repeat(64),"buffer_scopes":{},
            "metadata":{"schema_version":2,"target":"sm_87","model":"test","chunk_tokens":2,
                "max_context":8,"vocab":4,"toolchain":{},"buffers":[],"programs":{"decode":[]},
                "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
                "logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":1,"weight_scope":"test"}});
        fs::write(
            root.join("cache/model.json"),
            serde_json::to_vec(&descriptor).unwrap(),
        )
        .unwrap();
        let error = load(&root).err().unwrap();
        fs::remove_dir_all(root).unwrap();
        assert!(error.contains("without embedded kernels or programs"));
    }
}
