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
    pub expert_banks: Vec<orinfer_model_sdk::abi::ExpertBank>,
    pub execution_model: crate::model_package::ModelPackage,
}

pub(crate) fn load(path: &Path, verify_weights: bool) -> Result<PreparedModel> {
    load_impl(path, verify_weights, false, true)
}
pub(crate) fn load_with_experts(
    path: &Path,
    verify_weights: bool,
    expert_contracts: bool,
) -> Result<PreparedModel> {
    load_impl(path, verify_weights, expert_contracts, false)
}
fn load_impl(
    path: &Path,
    verify_weights: bool,
    expert_contracts: bool,
    build_bindings: bool,
) -> Result<PreparedModel> {
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
    model.metadata.greedy_sampling = package.greedy_sampling;
    model.metadata.batch_gdn = package.batch_gdn;
    // Fork/join order belongs to the native model implementation.
    model.metadata.graph_branches.clear();
    let (execution_model, built) = crate::model_package::ModelPackage::create(
        &kernel_root,
        &package.execution,
        orinfer_model_sdk::abi::CreateRequest {
            catalog: Some(package.catalog),
            expert_contracts,
            build_bindings,
            kernel_contracts: vec![],
            model_root: root.display().to_string(),
            verify_weights,
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
    let expert_banks = built.expert_banks;
    validate_dynamic_contracts(&model.metadata, build_bindings)?;
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
        expert_banks,
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
    if let Some(sampling) = &model.sampling {
        for name in [
            &sampling.history,
            &sampling.bias_ids,
            &sampling.bias_values,
            &sampling.mask,
            &sampling.control,
            &sampling.parameters,
            &sampling.output,
        ] {
            if scopes.get(name) != Some(&BufferScope::Workspace)
                || model.reset_buffers.contains(name)
            {
                return Err(format!("{name}: sampling scratch must be workspace"));
            }
        }
        if let Some(batch) = &sampling.batch {
            if scopes.get(&model.logits) != Some(&BufferScope::Sequence) {
                return Err("Batched sampling requires request-owned logits".into());
            }
            for name in [&batch.scratch, &batch.pointers, &batch.output] {
                if scopes.get(name) != Some(&BufferScope::Workspace)
                    || model.reset_buffers.contains(name)
                {
                    return Err(format!("{name}: batched sampling must be workspace"));
                }
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
    if model.independent() {
        if !model.reset_buffers.is_empty()
            || model.state_pointer_table.is_some()
            || model.sampling.is_some()
            || model.buffers.iter().any(|b| {
                b.access != Access::Read && scopes.get(&b.name) != Some(&BufferScope::Workspace)
            })
        {
            return Err(
                "Independent encoders require shared workspace and no persistent sequence state"
                    .into(),
            );
        }
        return Ok(());
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

fn validate_dynamic_contracts(model: &Manifest, exhaustive_rows: bool) -> Result<()> {
    let kernels: BTreeMap<_, _> = model.kernels.iter().map(|k| (k.name.as_str(), k)).collect();
    let mut row_contracts = std::collections::BTreeSet::new();
    for (name, template) in &model.dynamic_batch_kernels {
        if name != &template.name {
            return Err("Dynamic launch name differs from binding".into());
        }
        template.validate_binding(
            kernels
                .get(name.as_str())
                .ok_or("Missing dynamic batch kernel")?,
        )?;
        // Plans used by the offline builder and validate-model check the entire
        // domain once per expression contract, rather than once per model layer.
        if exhaustive_rows
            && row_contracts.insert((template.capacity, &template.grid, &template.arguments))
        {
            template.validate_rows()?;
        }
    }
    if !model.dynamic_batch_kernels.is_empty()
        && ((!model.batch_profiles.contains(&128) && model.row_execution.is_none())
            || kernels
                .keys()
                .filter(|name| {
                    name.starts_with("batch_m128/") || name.starts_with("batch_gdn_m128/")
                })
                .any(|name| !model.dynamic_batch_kernels.contains_key(*name)))
    {
        return Err("Incomplete dynamic batch contracts".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dynamic_fixture() -> Manifest {
        let identity = serde_json::json!({"file":"kernel.cubin", "sha256":"0".repeat(64)});
        serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":2,
            "max_context":128,"vocab":4,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"Input","token":"Token","status":"Status","position":"Step",
            "logits":"Logits","weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "batch_profiles":[128],
            "kernels":[{"name":"rows","module":identity,"source":identity,"host_abi":identity,
                "symbol":"test","grid":[128,1,1],"block":[32,1,1],"shared_memory_bytes":0,
                "cooperative":false,"args":[{"kind":"i32","value":128}]}],
            "dynamic_batch_kernels":{"rows":{"name":"rows","capacity":128,
                "grid":[{"op":"rows"},{"op":"constant","value":1},{"op":"constant","value":1}],
                "arguments":[{"index":0,"value":{"op":"rows"}}]}}
        }))
        .unwrap()
    }

    #[test]
    fn independent_row_templates_do_not_declare_autoregressive_batch_profiles() {
        let mut manifest = dynamic_fixture();
        manifest.batch_profiles.clear();
        assert!(validate_dynamic_contracts(&manifest, true).is_err());
        manifest.row_execution = Some(orinfer_model_sdk::operators::rows::RowExecution {
            capacity: 128,
            programs: std::collections::BTreeSet::from(["embed".into()]),
            program_capacities: Default::default(),
            variants: Default::default(),
        });
        validate_dynamic_contracts(&manifest, true).unwrap();
    }

    #[test]
    fn dynamic_row_domain_is_checked_offline_and_at_each_real_launch() {
        let mut manifest = dynamic_fixture();
        let template = manifest.dynamic_batch_kernels.get_mut("rows").unwrap();
        template.arguments[0].value = serde_json::from_value(serde_json::json!({
            "op":"divide","lhs":{"op":"rows"},"rhs":{"op":"remainder",
                "lhs":{"op":"rows"},"rhs":{"op":"constant","value":3}}
        }))
        .unwrap();
        manifest.kernels[0].args[0] = crate::artifact::Argument::I32 { value: 64 };
        validate_dynamic_contracts(&manifest, false).unwrap();
        assert!(validate_dynamic_contracts(&manifest, true).is_err());
        let template = &manifest.dynamic_batch_kernels["rows"];
        assert!(template.launch(2).is_ok());
        assert!(template.launch(3).is_err());
    }

    #[test]
    fn every_binding_keeps_its_abi_check_when_expressions_are_shared() {
        let mut manifest = dynamic_fixture();
        let mut template = manifest.dynamic_batch_kernels["rows"].clone();
        let mut binding = manifest.kernels[0].clone();
        binding.name = "rows_second_layer".into();
        template.name.clone_from(&binding.name);
        manifest
            .dynamic_batch_kernels
            .insert(template.name.clone(), template);
        manifest.kernels.push(binding);
        validate_dynamic_contracts(&manifest, true).unwrap();
        manifest.kernels[1].args[0] = crate::artifact::Argument::I32 { value: 127 };
        for exhaustive in [false, true] {
            assert!(validate_dynamic_contracts(&manifest, exhaustive).is_err());
        }
    }

    #[test]
    #[ignore = "Requires a prepared native model; CPU only, no GPU allocation"]
    fn validate_native_execution_bindings() {
        let path = std::path::PathBuf::from(std::env::var_os("ORINFER_MODEL").unwrap());
        let inspection = load(&path, false).unwrap();
        let runtime = load_with_experts(&path, false, false).unwrap();
        let used = orinfer_model_sdk::model::execution_kernels(
            &inspection.plan.programs,
            inspection.plan.row_execution.as_ref(),
        );
        let before: BTreeMap<_, _> = inspection
            .plan
            .kernels
            .iter()
            .filter(|k| used.contains(&k.name))
            .map(|k| (k.name.clone(), serde_json::to_value(k).unwrap()))
            .collect();
        let after: BTreeMap<_, _> = runtime
            .plan
            .kernels
            .iter()
            .map(|k| (k.name.clone(), serde_json::to_value(k).unwrap()))
            .collect();
        assert_eq!(
            before, after,
            "Execution export changed a binding or lost a replacement"
        );
        let mut expected = serde_json::to_value(&inspection.plan).unwrap();
        let mut actual = serde_json::to_value(&runtime.plan).unwrap();
        for value in [&mut expected, &mut actual] {
            value.as_object_mut().unwrap().remove("kernels");
            value
                .as_object_mut()
                .unwrap()
                .remove("dynamic_batch_kernels");
        }
        assert_eq!(
            expected, actual,
            "Execution export changed programs or request state"
        );
        let dynamic: BTreeMap<_, _> = inspection
            .plan
            .dynamic_batch_kernels
            .iter()
            .filter(|(name, _)| used.contains(*name))
            .map(|(name, k)| (name, serde_json::to_value(k).unwrap()))
            .collect();
        let runtime_dynamic: BTreeMap<_, _> = runtime
            .plan
            .dynamic_batch_kernels
            .iter()
            .map(|(name, k)| (name, serde_json::to_value(k).unwrap()))
            .collect();
        assert_eq!(dynamic, runtime_dynamic);
        // Finite residency must also retain every consumer's routing operand.
        let finite = load_with_experts(&path, false, true).unwrap();
        let kernels: std::collections::BTreeSet<_> =
            finite.plan.kernels.iter().map(|k| &k.name).collect();
        for bank in &finite.expert_banks {
            for name in bank.consumers.keys() {
                assert!(kernels.contains(name));
            }
        }
        eprintln!(
            "Native bindings: {} build, {} execution; programs/state/launches unchanged",
            inspection.plan.kernels.len(),
            runtime.plan.kernels.len()
        );
    }
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
        let error = load(&root, false).err().unwrap();
        fs::remove_dir_all(root).unwrap();
        assert!(error.contains("without embedded kernels or programs"));
    }
}

pub(crate) fn validate_frontend_assets(
    root: &Path,
    expected: &BTreeMap<String, String>,
) -> Result<()> {
    let names: &[&str] = if expected.contains_key("processor_config.json") {
        &[
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
        ]
    } else if expected.contains_key("tokenizer_config.json") {
        &[
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ]
    } else {
        &[
            "tokenizer.json",
            "chat_template.jinja",
            "generation_config.json",
        ]
    };
    if expected.len() != names.len() {
        return Err(
            "Prepared model requires all frontend asset identities; run package.py pin-assets"
                .into(),
        );
    }
    for name in names {
        let bytes = fs::read(resolve_file(root, name)?).map_err(|e| e.to_string())?;
        if expected.get(*name) != Some(&sha256(&bytes)) {
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
