//! Source registration is independent of deployable package identity and ABI.
use crate::adapter::Adapter;
use crate::{CreateRequest, CreatedPlan};
use orinfer_model_sdk::artifact::Result;

pub(crate) type Built = (Box<dyn Adapter>, CreatedPlan);
struct Factory {
    architecture: &'static str,
    create: fn(CreateRequest) -> Result<Built>,
}
const FACTORIES: &[Factory] = &[
    Factory {
        architecture: "qwen3_reranker",
        create: crate::qwen_reranker::create,
    },
    Factory {
        architecture: "nemotron_embed",
        create: crate::nemotron_embed::create,
    },
    Factory {
        architecture: "qwen3_asr",
        create: crate::qwen3_5::create,
    },
    Factory {
        architecture: "flash_next",
        create: crate::flash_next::create,
    },
    Factory {
        architecture: "qwen3_5",
        create: crate::qwen3_5::create,
    },
];
const COMPUTE_POLICY: &str = "int8_quality";

fn select(architecture: &str, policy: &str) -> Result<&'static Factory> {
    if policy != COMPUTE_POLICY {
        return Err(format!("Unsupported compute policy: {policy}"));
    }
    FACTORIES
        .iter()
        .find(|f| f.architecture == architecture)
        .ok_or_else(|| format!("Unregistered model architecture: {architecture}"))
}

pub(crate) fn create(request: CreateRequest) -> Result<Built> {
    (select(&request.architecture, &request.compute_policy)?.create)(request)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn family_and_policy_are_validated_as_separate_axes() {
        assert!(select("qwen3_5", COMPUTE_POLICY).is_ok());
        assert!(select("unknown", COMPUTE_POLICY).is_err());
        assert!(select("qwen3_5", "a4").is_err());
        assert!(select("qwen3_5", "force_int8").is_err());
    }
}
