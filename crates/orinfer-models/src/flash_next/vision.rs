//! Checkpoint-exact shared ViT and Flash visual embedding/MRoPE roles.
use super::plan::section;
use crate::CreateRequest;
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use std::collections::BTreeMap;

pub(super) fn validate(request: &CreateRequest) -> Result<()> {
    let Some(v) = &request.metadata.vision else {
        return Ok(());
    };
    let contract: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../configs/architecture-contract.json"
    ))
    .map_err(|e| e.to_string())?;
    let expected = &contract["families"]["flash_next"]["supported_vision"];
    for (key, value) in expected.as_object().ok_or("Missing vision contract")? {
        if &request.config["vision_config"][key] != value {
            return Err(format!("Unsupported Flash vision_config.{key}"));
        }
    }
    if request.config["text_config"]["rope_parameters"]["mrope_interleaved"] != true
        || request.config["text_config"]["rope_parameters"]["mrope_section"]
            != serde_json::json!([11, 11, 10])
        || v.hidden != 2560
        || v.pixels != "VPixels"
        || v.grid != "VGrid"
        || v.length != "VLength"
        || v.output != "VOutput"
        || v.features != "Features"
        || v.feature_index != "FeatureIndex"
        || v.mrope_positions != "MRopePositions"
        || [
            v.image_token_id,
            v.vision_start_token_id,
            v.vision_end_token_id,
        ] != [248056, 248053, 248054]
        || [
            "image_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ]
        .iter()
        .zip([
            v.image_token_id,
            v.vision_start_token_id,
            v.vision_end_token_id,
        ])
        .any(|(key, id)| request.config[key].as_u64() != Some(id.into()))
        || request.config["language_model_only"] == true
    {
        return Err("Flash visual roles or checkpoint image IDs differ".into());
    }
    Ok(())
}

pub(super) fn register(
    manifest: &Manifest,
    programs: &mut BTreeMap<String, Vec<Operation>>,
) -> Result<()> {
    let Some(v) = &manifest.vision else {
        return Ok(());
    };
    for plan in &v.plans {
        let program = if plan.min_patches > 0 {
            "vision".into()
        } else {
            format!("vision_m{}", plan.patches)
        };
        if plan.program != program {
            return Err("Unknown Flash vision profile".into());
        }
        let mut ops = section(&program, "begin", 2, None);
        for layer in 0..27 {
            ops.extend(section(&program, &format!("layer{layer}"), 10, None));
        }
        ops.extend(section(&program, "end", 3, None));
        programs.insert(program, ops);
    }
    Ok(())
}

/// Vision lengths use the same symbolic launch machinery as text rows.
pub(super) fn configure(m: &mut Manifest) -> Result<()> {
    let Some(v) = &m.vision else {
        return Ok(());
    };
    for plan in v.plans.iter().filter(|p| p.min_patches > 0) {
        let rows = m
            .row_execution
            .as_mut()
            .ok_or("Missing dynamic vision rows")?;
        // Raising the shared storage bound must not widen existing text programs.
        for name in &rows.programs {
            rows.program_capacities
                .entry(name.clone())
                .or_insert(rows.capacity);
        }
        rows.capacity = rows.capacity.max(plan.patches);
        rows.programs.insert(plan.program.clone());
        rows.program_capacities
            .insert(plan.program.clone(), plan.patches);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vision_capacity_does_not_widen_text_or_mtp_programs() {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":1,"target":"sm_87","model":"flash","chunk_tokens":4096,
            "max_context":262144,"vocab":32,"toolchain":{},"buffers":[],"reset_buffers":[],
            "input":"i","token":"t","status":"s","logits":"l","position":"p",
            "weight_bytes":0,"weight_parameters":1,"weight_scope":"test",
            "row_execution":{"capacity":4096,"programs":["prefill","verify"],"program_capacities":{"verify":8}},
            "vision":{"hidden":2560,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
                "max_patches":8192,"max_features":16384,"pixels":"p","grid":"g","length":"n",
                "output":"o","features":"f","feature_index":"x","mrope_positions":"r",
                "plans":[{"patches":8192,"min_patches":4,"program":"vision"}],
                "image_token_id":1,"vision_start_token_id":2,"vision_end_token_id":3}
        })).unwrap();
        configure(&mut m).unwrap();
        let rows = m.row_execution.unwrap();
        assert_eq!(rows.capacity, 8192);
        assert_eq!(rows.program_capacity("vision"), 8192);
        assert_eq!(rows.program_capacity("prefill"), 4096);
        assert_eq!(rows.program_capacity("verify"), 8);
    }
}
