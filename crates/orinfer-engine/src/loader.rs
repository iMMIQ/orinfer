//! Custom model-directory loader. Only initialized tensor data lives with weights;
//! native execution packages supply kernels and build model-specific plans.
use crate::{
    architecture::{Architecture, ComputePolicy},
    artifact::{Access, Result, resolve_file, sha256},
    model::Manifest,
    operators::{self, ExecutionPackage},
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
    execution_package: String,
    buffer_scopes: BTreeMap<String, BufferScope>,
    frontend_assets: BTreeMap<String, String>,
    metadata: Manifest,
}

pub(crate) struct PreparedModel {
    pub plan: Manifest,
    pub weights_root: PathBuf,
    pub kernel_root: PathBuf,
    pub fingerprint: String,
    pub scopes: BTreeMap<String, BufferScope>,
    pub execution_package: String,
    pub architecture: Architecture,
    pub policy: ComputePolicy,
    pub frontend_assets: BTreeMap<String, String>,
    pub decode_programs: std::collections::BTreeSet<String>,
    pub execution_model: crate::model_package::ModelPackage,
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
    if model.schema_version != 1 {
        return Err("Expected model data schema 1 with a native execution package".into());
    }
    if !model.metadata.kernels.is_empty() || !model.metadata.programs.is_empty() {
        return Err("Expected custom model descriptor without embedded kernels or programs".into());
    }
    validate_frontend_assets(root, &model.frontend_assets)?;
    let config_path = resolve_file(root, "config.json")?;
    let config_raw = fs::read(config_path).map_err(|e| e.to_string())?;
    let config: serde_json::Value =
        serde_json::from_slice(&config_raw).map_err(|e| e.to_string())?;
    let kernel_root = operators::resolve(&weights_root, &model.execution_package)?;
    let package_raw =
        fs::read(resolve_file(&kernel_root, "package.json")?).map_err(|e| e.to_string())?;
    if sha256(&package_raw) != model.execution_package {
        return Err("Model execution package digest mismatch".into());
    }
    let package: ExecutionPackage = serde_json::from_slice(&package_raw)
        .map_err(|e| format!("Model execution package: {e}"))?;
    package.validate(
        &model.architecture,
        &model.compute_policy,
        &model.metadata.buffers,
    )?;
    validate_scopes(&model.metadata, &model.buffer_scopes)?;
    model
        .metadata
        .batch_profiles
        .clone_from(&package.batch_profiles);
    model
        .metadata
        .prefill_batch_profiles
        .clone_from(&package.prefill_batch_profiles);
    model.metadata.dynamic_batch_kernels = package
        .dynamic_batch_kernels
        .iter()
        .map(|t| (t.name.clone(), t.clone()))
        .collect();
    model.metadata.greedy_sampling = package.greedy_sampling;
    model.metadata.batch_gdn = package.batch_gdn;
    let (execution_model, built) = crate::model_package::ModelPackage::create(
        &kernel_root,
        &package.execution,
        orinfer_model_sdk::abi::CreateRequest {
            model_root: root.display().to_string(),
            config,
            architecture: model.architecture.clone(),
            compute_policy: model.compute_policy.clone(),
            expected_signature: package.config_signature.clone(),
            metadata: model.metadata,
            prefill_profiles: package.prefill_profiles,
        },
    )?;
    model.metadata = built.metadata;
    validate_scopes(&model.metadata, &model.buffer_scopes)?;
    let decode_programs = built.decode_programs;
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
        execution_package: model.execution_package,
        architecture: model.architecture,
        policy: model.compute_policy,
        frontend_assets: model.frontend_assets,
        decode_programs,
        execution_model,
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
    if let Some(table) = &model.state_pointer_table
        && (table.max_rows == 0
            || table.max_rows > 128
            || scopes.get(&table.buffer) != Some(&BufferScope::Workspace)
            || model
                .buffers
                .iter()
                .find(|b| b.name == table.buffer)
                .is_none_or(|b| b.dtype != crate::artifact::Dtype::U64))
    {
        return Err("Invalid model state pointer table contract".into());
    }
    if let Some(controls) = &model.segment_controls {
        for name in [&controls.length, &controls.last_index] {
            if scopes.get(name) != Some(&BufferScope::Sequence)
                || model
                    .buffers
                    .iter()
                    .find(|b| &b.name == name)
                    .is_none_or(|b| b.dtype != crate::artifact::Dtype::I32 || b.shape != [1])
            {
                return Err("Invalid model segment control contract".into());
            }
        }
    }
    if let Some(kv) = &model.kv_cache {
        for name in kv.prefill_workspace.keys() {
            if scopes.get(name) != Some(&BufferScope::Workspace)
                || model.reset_buffers.contains(name)
            {
                return Err(format!("{name}: prefill scratch must be workspace"));
            }
        }
        for name in kv.buffers.keys() {
            if scopes.get(name) != Some(&BufferScope::Sequence)
                || !model.reset_buffers.contains(name)
            {
                return Err(format!(
                    "{name}: KV must be private resettable sequence state"
                ));
            }
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
        let descriptor = serde_json::json!({"schema_version":1,"architecture":"test_family",
            "compute_policy":"int8_quality","execution_package":"0".repeat(64),"buffer_scopes":{},"frontend_assets":{},
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

fn validate_frontend_assets(root: &Path, expected: &BTreeMap<String, String>) -> Result<()> {
    let names = [
        "tokenizer.json",
        "chat_template.jinja",
        "generation_config.json",
    ];
    if expected.len() != names.len() {
        return Err(
            "Prepared model requires all frontend asset identities; run package.py pin-assets"
                .into(),
        );
    }
    for name in names {
        let bytes = fs::read(resolve_file(root, name)?).map_err(|e| e.to_string())?;
        if expected.get(name) != Some(&sha256(&bytes)) {
            return Err(format!("Frontend asset identity mismatch: {name}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod asset_tests {
    use super::*;
    #[test]
    fn equal_size_semantic_edits_are_rejected() {
        let root = std::env::temp_dir().join(format!("orin-assets-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let mut hashes = BTreeMap::new();
        for name in [
            "tokenizer.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            fs::write(root.join(name), b"original").unwrap();
            hashes.insert(name.to_owned(), sha256(b"original"));
        }
        validate_frontend_assets(&root, &hashes).unwrap();
        fs::write(root.join("tokenizer.json"), b"modified").unwrap();
        assert!(
            validate_frontend_assets(&root, &hashes)
                .unwrap_err()
                .contains("tokenizer.json")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
