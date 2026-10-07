//! Checkpoint-exact shared ViT and Flash visual embedding/MRoPE roles.
use super::plan::section;
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use orinfer_model_sdk::abi::CreateRequest;
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
        let program = format!("vision_m{}", plan.patches);
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
