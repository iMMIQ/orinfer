//! Qwen3.5 model semantics; quality policy chooses the registered mixed-precision recipe.
mod config;
mod plan;
use crate::{adapter::Adapter, artifact, model, policy::Policy, vision};
use model::Manifest;
use orinfer_model_sdk::{
    abi::{self, CreateRequest, CreatedPlan},
    architecture::BatchSegment,
};
struct Qwen {
    manifest: Manifest,
}
impl Adapter for Qwen {
    fn visual(
        &self,
        tokens: &[u32],
        images: &[abi::ImageGrid],
        capacity: usize,
    ) -> artifact::Result<(Vec<i32>, Vec<u32>)> {
        vision::layout(
            self.manifest
                .vision
                .as_ref()
                .ok_or("Model has no vision adapter")?,
            tokens,
            images,
            capacity,
        )
    }
    fn batch(
        &self,
        segments: &[BatchSegment],
        include_plan: bool,
    ) -> artifact::Result<abi::BatchPlan> {
        Ok((
            if include_plan {
                plan::batching::plan(&self.manifest, segments)?
            } else {
                vec![]
            },
            if self.manifest.batch_gdn {
                plan::batching::state_bindings(&self.manifest, segments)
            } else {
                vec![]
            },
        ))
    }
}
pub(crate) fn create(
    mut request: CreateRequest,
    policy: Policy,
) -> artifact::Result<crate::registry::Built> {
    if request.architecture != "qwen3_5" || policy != Policy::Int8Quality {
        return Err(
            "Qwen A8 package does not support this architecture or precision policy".into(),
        );
    }
    let config = config::Configuration::parse(request.config)?;
    if config.signature != request.expected_signature {
        return Err("Execution package does not support this model configuration".into());
    }
    let decode_programs = config::build(&config, &mut request.metadata, &request.prefill_profiles)?;
    if request.metadata.batch_gdn {
        request.metadata.state_pointer_table = Some(model::StatePointerTable {
            buffer: "BatchGdnPointers".into(),
            max_rows: 128,
        });
    }
    if ["BatchSegmentLength", "BatchLastIndex"]
        .iter()
        .all(|name| request.metadata.buffers.iter().any(|b| b.name == *name))
    {
        request.metadata.segment_controls = Some(model::SegmentControls {
            length: "BatchSegmentLength".into(),
            last_index: "BatchLastIndex".into(),
        });
    }
    let metadata = request.metadata;
    let plan = CreatedPlan {
        metadata: metadata.clone(),
        decode_programs,
    };
    Ok((Box::new(Qwen { manifest: metadata }), plan))
}
