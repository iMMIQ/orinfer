//! Model-family registration and parameterized execution plans.
//! Weight files contain data; operator packages contain bindings. Control flow
//! belongs to the registered architecture implementation.
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

mod qwen3_5;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Architecture {
    Qwen3_5,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ComputePolicy {
    Int8Quality,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrefillKind {
    ChunkLut4,
    ChunkExpanded,
    Sequence,
    Recurrent,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillProfile {
    pub tokens: usize,
    pub kind: PrefillKind,
}

#[derive(Debug)]
pub(crate) enum FamilyConfig {
    Qwen3_5(qwen3_5::TextConfig),
}

#[derive(Debug)]
pub(crate) struct BatchLayout {
    pub architecture: Architecture,
    pub layers: Vec<String>,
    pub row_strides: BTreeMap<String, usize>,
    pub profiles: BTreeMap<usize, PrefillKind>,
    pub hidden: usize,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchSegment {
    pub slot: usize,
    pub tokens: usize,
}

pub(crate) fn batch_plan(
    manifest: &Manifest,
    segments: &[BatchSegment],
) -> Result<Vec<crate::execution::Invocation>> {
    match manifest
        .batch_layout
        .as_ref()
        .ok_or("Missing registered batch layout")?
        .architecture
    {
        Architecture::Qwen3_5 => qwen3_5::batching::plan(manifest, segments),
    }
}

pub(crate) fn batch_state_bindings(
    manifest: &Manifest,
    segments: &[BatchSegment],
) -> Result<Vec<Option<(usize, String)>>> {
    match manifest
        .batch_layout
        .as_ref()
        .ok_or("Missing registered batch layout")?
        .architecture
    {
        Architecture::Qwen3_5 => Ok(qwen3_5::batching::state_bindings(manifest, segments)),
    }
}

pub(crate) struct Configuration {
    pub architecture: Architecture,
    family: FamilyConfig,
    pub vision_depth: usize,
    pub signature: Value,
}

impl Configuration {
    fn qwen3_5(&self) -> &qwen3_5::TextConfig {
        match &self.family {
            FamilyConfig::Qwen3_5(text) => text,
        }
    }

    pub fn parse(config: Value) -> Result<Self> {
        let architecture = match config["model_type"].as_str() {
            Some("qwen3_5" | "qwen3_5_text") => Architecture::Qwen3_5,
            _ => return Err("Unregistered model architecture".into()),
        };
        let text_value = config.get("text_config").unwrap_or(&config);
        let family = match architecture {
            Architecture::Qwen3_5 => FamilyConfig::Qwen3_5(qwen3_5::parse_text(text_value)?),
        };
        // These values affect compiled arithmetic, weight contracts or state.
        // Checkpoint identity, naming, dtype labels and quantization provenance
        // deliberately do not determine architecture/operator compatibility.
        let contract: Value = serde_json::from_str(include_str!(
            "../../../../configs/architecture-contract.json"
        ))
        .map_err(|e| e.to_string())?;
        let text_keys = contract["families"]["qwen3_5"]["text_keys"]
            .as_array()
            .ok_or("Missing architecture signature contract")?;
        let text_signature: serde_json::Map<String, Value> = text_keys
            .iter()
            .filter_map(|k| k.as_str())
            .filter_map(|k| text_value.get(k).map(|v| (k.into(), v.clone())))
            .collect();
        let vision_depth = config["vision_config"]["depth"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let signature = serde_json::json!({"text": text_signature,
            "vision": config.get("vision_config").cloned().unwrap_or(Value::Null)});
        Ok(Self {
            architecture,
            family,
            vision_depth,
            signature,
        })
    }
}

pub(crate) fn build(
    config: &Configuration,
    manifest: &mut Manifest,
    profiles: &[PrefillProfile],
) -> Result<std::collections::BTreeSet<String>> {
    match config.architecture {
        Architecture::Qwen3_5 => qwen3_5::build(config, manifest, profiles),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Value {
        serde_json::json!({"model_type":"qwen3_5_text","hidden_size":8,"vocab_size":16,
            "num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],
            "linear_num_key_heads":1,"linear_num_value_heads":2,
            "linear_key_head_dim":2,"linear_value_head_dim":3,"linear_conv_kernel_dim":4})
    }
    #[test]
    fn configuration_rejects_unknown_or_incomplete_architectures() {
        let mut raw = config();
        raw["model_type"] = "other".into();
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["layer_types"] = serde_json::json!(["linear_attention"]);
        assert!(Configuration::parse(raw).is_err());
        let mut raw = config();
        raw["linear_num_value_heads"] = 0.into();
        assert!(Configuration::parse(raw).is_err());
    }
    #[test]
    fn kernel_compatibility_tracks_math_not_checkpoint_provenance() {
        let original = Configuration::parse(config()).unwrap();
        let mut raw = config();
        raw["quantization_config"] = serde_json::json!({"source":"another checkpoint"});
        raw["dtype"] = "float16".into();
        assert_eq!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
        let mut raw = config();
        raw["rms_norm_eps"] = serde_json::json!(0.1);
        assert_ne!(
            original.signature,
            Configuration::parse(raw).unwrap().signature
        );
    }
}
