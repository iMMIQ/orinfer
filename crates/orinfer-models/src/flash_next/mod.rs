//! Flash Next execution order and immutable CPU lookup owned by the model package.
mod batching;
mod inputs;
mod mtp;
mod plan;
mod vision;
use crate::{adapter::Adapter, artifact::Result, model::Manifest, policy::Policy};
use orinfer_model_sdk::{
    abi::{self, CreateRequest, CreatedPlan},
    architecture::BatchSegment,
};
struct Flash {
    manifest: Manifest,
    inputs: inputs::Inputs,
}
impl Adapter for Flash {
    fn visual(
        &self,
        tokens: &[u32],
        images: &[abi::ImageGrid],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        crate::vision::layout(
            self.manifest
                .vision
                .as_ref()
                .ok_or("Model has no vision adapter")?,
            tokens,
            images,
            capacity,
        )
    }
    fn prepare(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        self.inputs.prepare_for(program, tokens, history)
    }
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<abi::BatchPlan> {
        Ok((
            if include_plan {
                batching::plan(&self.manifest, segments)?
            } else {
                vec![]
            },
            vec![],
        ))
    }
}
pub(crate) fn create(mut request: CreateRequest, policy: Policy) -> Result<crate::registry::Built> {
    if policy != Policy::Int8Quality
        || request.config["model_type"] != "qwen4_exp"
        || request.architecture != "flash_next"
    {
        return Err("Unsupported Flash architecture or precision policy".into());
    }
    let contract: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../configs/architecture-contract.json"
    ))
    .map_err(|e| e.to_string())?;
    let text = &request.config["text_config"];
    let family = &contract["families"]["flash_next"];
    for (key, expected) in family["supported_text"]
        .as_object()
        .ok_or("Missing Flash contract")?
    {
        if &text[key] != expected {
            return Err(format!("Unsupported Flash text_config.{key}"));
        }
    }
    for (key, expected) in family["supported_rope"]
        .as_object()
        .ok_or("Missing Flash RoPE contract")?
    {
        if &text["rope_parameters"][key] != expected {
            return Err(format!("Unsupported Flash RoPE {key}"));
        }
    }
    let codec = serde_json::json!({"quant_method":"orinfer_e8p_int8","version":1,
        "basis":"integer-e8p-spread29-v1","expert_rotation":"signed-block128",
        "embedding_rotation":"paley20-walsh8","compute_dtype":"int8_quality"});
    for (key, expected) in codec.as_object().ok_or("Missing Flash codec contract")? {
        if &request.config["quantization_config"][key] != expected {
            return Err(format!("Unsupported Flash quantization_config.{key}"));
        }
    }
    vision::validate(&request)?;
    let mut signature =
        serde_json::json!({"text":text,"quantization":request.config["quantization_config"]});
    if request.metadata.vision.is_some() {
        signature["vision"] = request.config["vision_config"].clone();
        signature["image_tokens"] = serde_json::json!([
            request.config["image_token_id"],
            request.config["vision_start_token_id"],
            request.config["vision_end_token_id"]
        ]);
    }
    if signature != request.expected_signature
        || text.get("norm_topk_prob").is_some_and(|v| v != true)
        || request.config["quantization_config"]["quant_method"] != "orinfer_e8p_int8"
        || request.config["quantization_config"]
            .get("component")
            .is_some_and(|v| v != "text")
        || request.metadata.max_context > 262144
        || text["max_position_embeddings"]
            .as_u64()
            .is_none_or(|v| v < request.metadata.max_context as u64)
        || request.metadata.vocab != 248320
    {
        return Err("Flash package arithmetic/checkpoint signature mismatch".into());
    }
    plan::build(&mut request.metadata, &request.prefill_profiles)?;
    let inputs = inputs::Inputs::open(
        &request.model_root,
        request
            .metadata
            .input_assets
            .as_ref()
            .ok_or("Missing CPU input assets")?,
    )?;
    let metadata = request.metadata;
    let mut decode_programs = std::collections::BTreeSet::from(["decode".into()]);
    if let Some(spec) = &metadata.mtp {
        decode_programs.extend([
            spec.draft_program.clone(),
            "mtp_snapshot".into(),
            "mtp_restore_draft".into(),
        ]);
        for p in &spec.warm_plans {
            if p.tokens <= 8 {
                decode_programs.extend([p.program.clone(), p.head_program.clone()]);
            }
        }
        decode_programs.insert("mtp_capture_m1".into());
        for p in &spec.verification_plans {
            decode_programs.extend([
                p.program.clone(),
                p.capture_program.clone(),
                p.restore_program.clone(),
            ]);
        }
    }
    let created = CreatedPlan {
        metadata: metadata.clone(),
        decode_programs,
    };
    Ok((
        Box::new(Flash {
            manifest: metadata,
            inputs,
        }),
        created,
    ))
}
