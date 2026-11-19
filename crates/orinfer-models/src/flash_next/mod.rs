//! Flash Next execution order and immutable CPU lookup owned by the model package.
use crate::{CreateRequest, CreatedPlan};
mod auxiliary;
mod batching;
mod causal_rows;
mod contracts;
mod decode_fusions;
mod draft_batch;
mod draft_rows;
mod experts;
mod hc;
mod inputs;
mod mixed;
mod mtp;
mod parallel;
mod plan;
mod q4;
mod routing;
mod row_variants;
mod rows;
mod unified_decode;
mod vision;
use crate::{adapter::Adapter, artifact::Result, model::Manifest};
use orinfer_model_sdk::{architecture::BatchSegment, packet};
struct Flash {
    manifest: Manifest,
    inputs: inputs::Inputs,
    operands: crate::packet::Operands,
}
impl Adapter for Flash {
    fn prefetch_inputs(&self, tokens: &[u32], history: &[u32]) {
        self.inputs.prefetch(tokens, history, 4096);
    }
    fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: packet::BatchOperation,
        include_plan: bool,
    ) -> Result<packet::BatchPlan> {
        let result = match operation {
            packet::BatchOperation::Ordinary => self.batch(segments, include_plan),
            packet::BatchOperation::Verification => {
                let plan = mixed::verification_plan(&self.manifest, segments, include_plan)?;
                Ok((plan, vec![]))
            }
            packet::BatchOperation::MixedVerification => {
                let plan = mixed::mixed_verification_plan(&self.manifest, segments, include_plan)?;
                Ok((plan, vec![]))
            }
            packet::BatchOperation::MixedRows { prefills } => {
                let plan = mixed::mixed_rows_plan(
                    &self.manifest,
                    segments,
                    usize::from(prefills),
                    include_plan,
                )?;
                Ok((plan, vec![]))
            }
            packet::BatchOperation::Draft
            | packet::BatchOperation::Warm
            | packet::BatchOperation::DraftHead => {
                let plan = draft_batch::plan(&self.manifest, segments, operation, include_plan)?;
                Ok((plan, vec![]))
            }
            packet::BatchOperation::Commit => {
                let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
                let mut owners = std::collections::BTreeSet::new();
                if segments.is_empty() || segments.len() > 128 {
                    return Err("Invalid MTP commit packet size".into());
                }
                let mut plan = vec![];
                for segment in segments {
                    if !owners.insert(segment.slot)
                        || !spec.batch_verification_tokens.contains(&segment.tokens)
                    {
                        return Err("Invalid or repeated MTP commit segment".into());
                    }
                    let restore = spec
                        .verification_plan(segment.tokens)
                        .ok_or("Missing MTP commit profile")?;
                    if include_plan {
                        rows::append(
                            &self.manifest,
                            &mut plan,
                            &restore.restore_program,
                            segment.tokens,
                            Some(segment.slot),
                            &Default::default(),
                        )?;
                    }
                }
                Ok((plan, vec![]))
            }
        }?;
        crate::packet::indirect(&self.manifest, segments, &self.operands, result)
    }
    fn visual(
        &self,
        tokens: &[u32],
        images: &[orinfer_model_sdk::architecture::ImageGrid],
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
        if program == "batch_target"
            && tokens.len() > 1
            && self
                .manifest
                .batch_layout
                .as_ref()
                .is_some_and(|l| l.segment_tokens > 0)
        {
            return self.inputs.prepare_for("target", tokens, history, 4096);
        }
        if let Some(warm) = self.manifest.mtp.as_ref().and_then(|s| {
            s.warm_plans
                .iter()
                .find(|p| p.program == program && p.min_tokens > 0)
        }) {
            if !warm.accepts(tokens.len()) {
                return Err("Invalid dynamic warm extent".into());
            }
            return self
                .inputs
                .prepare_for("mtp_warm", tokens, history, warm.tokens);
        }
        let verification = self
            .manifest
            .mtp
            .as_ref()
            .map_or(&[][..], |s| s.verification_plans.as_slice());
        let (program, capacity) = inputs::layout(
            &self.manifest.prefill_plans,
            verification,
            program,
            tokens.len(),
        )?;
        self.inputs.prepare_for(program, tokens, history, capacity)
    }
    fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<packet::BatchPlan> {
        Ok((
            if include_plan {
                batching::plan(&self.manifest, segments)?
            } else {
                vec![]
            },
            batching::state_bindings(&self.manifest, segments)?,
        ))
    }
}
pub(crate) fn create(mut request: CreateRequest) -> Result<crate::registry::Built> {
    if request.config["model_type"] != "qwen4_exp" || request.architecture != "flash_next" {
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
    if !unified_decode::present(&request.metadata)
        || request.metadata.dynamic_batch_kernels["flash_decode/begin/k0"].capacity
            != unified_decode::COMPUTATION_CAPACITY
        || !causal_rows::present(&request.metadata)
        || request
            .metadata
            .batch_layout
            .as_ref()
            .is_none_or(|l| l.segment_tokens != 512)
        || (request.metadata.mtp.is_some() && !draft_rows::present(&request.metadata))
    {
        return Err("Flash requires shared target/draft rows and dynamic causal state".into());
    }
    plan::build(&mut request.metadata, &request.prefill_profiles)?;
    let exhaustive = request.verify_weights || request.build_bindings;
    if exhaustive {
        q4::validate(&request.metadata)?;
    }
    q4::prerequisites(&mut request.metadata, &request.kernel_contracts)?;
    experts::build(&mut request.metadata, &request.kernel_contracts)?;
    hc::build(&mut request.metadata, &request.kernel_contracts, exhaustive)?;
    routing::build(&mut request.metadata, &request.kernel_contracts, exhaustive)?;
    decode_fusions::build(&mut request.metadata, &request.kernel_contracts, exhaustive)?;
    parallel::build(&mut request.metadata, &request.kernel_contracts)?;
    plan::stages(&mut request.metadata, &request.operands)?;
    crate::packet::register_indirect(&mut request.metadata);

    let manifest = &mut request.metadata;
    // Source binding slots describe compiled implementations. Only the
    // canonical template and its resumable stages remain public programs.
    manifest.programs.remove("prefill_m4096");
    manifest.programs.remove("head_m4096");
    if let Some(kv) = &mut manifest.kv_cache {
        kv.growth.remove("prefill_m4096");
    }
    let plan = manifest
        .prefill_plans
        .first_mut()
        .ok_or("Missing dynamic prefill template")?;
    plan.prefill_program = "prefill".into();
    plan.head_program = "head".into();
    for stage in &mut plan.stages {
        let name = stage.replace("prefill_m4096/", "prefill/");
        let operations = manifest
            .programs
            .remove(stage)
            .ok_or("Missing dynamic prefill stage")?;
        manifest.programs.insert(name.clone(), operations);
        if let Some(kv) = &mut manifest.kv_cache
            && let Some(growth) = kv.growth.remove(stage)
        {
            kv.growth.insert(name.clone(), growth);
        }
        *stage = name;
    }
    let mut programs = std::collections::BTreeSet::from(["prefill".into(), "head".into()]);
    programs.extend(plan.stages.iter().cloned());
    if manifest.mtp.is_some() {
        programs.insert("mtp_capture_m4096".into());
    }
    let variants = row_variants::build(
        &request.kernel_contracts,
        exhaustive.then_some(manifest.buffers.as_slice()),
    )?;
    manifest.row_execution = Some(orinfer_model_sdk::operators::rows::RowExecution {
        capacity: 4096,
        program_capacities: std::collections::BTreeMap::new(),
        programs,
        variants,
    });
    if let Some(layout) = &mut manifest.batch_layout
        && !layout.profiles.is_empty()
    {
        let kv = manifest
            .kv_cache
            .as_mut()
            .ok_or("Missing batch KV metadata")?;
        let base = kv
            .growth
            .get("decode")
            .ok_or("Missing decode KV growth")?
            .clone();
        layout.preparation_programs.insert(1, "decode".into());
        for &tokens in layout.profiles.keys() {
            let program = "causal_rows/target/layer3/private".to_owned();
            if !manifest.programs.contains_key(&program) {
                return Err("Missing batch causal preparation program".into());
            }
            let mut growth = base.clone();
            growth.tokens = tokens;
            growth.length = (layout.segment_tokens > 0).then(|| "Length".into());
            growth
                .buffers
                .retain(|name| !kv.prefill_workspace.contains_key(name));
            growth.buffers.push(plan::score_role(512, "Target"));
            kv.growth.insert(program.clone(), growth);
            layout.preparation_programs.insert(tokens, program);
        }
    }

    if let Some(spec) = &mut request.metadata.mtp {
        spec.mixed_verification_tokens = if !spec.batch_verification_tokens.is_empty()
            && !request.metadata.prefill_batch_profiles.is_empty()
            && request
                .metadata
                .prefill_batch_profiles
                .iter()
                .all(|p| p.kind == "flash_mixed_grouped")
            && request
                .metadata
                .batch_layout
                .as_ref()
                .is_some_and(|layout| !layout.profiles.is_empty())
        {
            vec![2]
        } else {
            vec![]
        };
    }
    unified_decode::configure(&mut request.metadata, &request.kernel_contracts)?;
    causal_rows::configure(&mut request.metadata)?;
    vision::configure(&mut request.metadata)?;
    // Row construction also creates complete verification programs. Fuse only
    // after every alias exists so none can reintroduce replaced producers.
    auxiliary::build(&mut request.metadata, &request.kernel_contracts)?;
    parallel::build_rows(&mut request.metadata, &request.kernel_contracts)?;
    let expert_banks = experts::banks(&request.metadata, &request.kernel_contracts)?;
    let inputs = inputs::Inputs::open(
        &request.model_root,
        request
            .metadata
            .input_assets
            .as_ref()
            .ok_or("Missing CPU input assets")?,
        request.verify_weights,
    )?;
    let metadata = request.metadata;
    let created = CreatedPlan {
        expert_banks,
        metadata: metadata.clone(),
    };
    Ok((
        Box::new(Flash {
            manifest: metadata,
            inputs,
            operands: request.operands,
        }),
        created,
    ))
}
