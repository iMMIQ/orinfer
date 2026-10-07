//! Qwen configuration semantics and compiled arithmetic signature.
//! Weight files contain data; operator packages contain bindings. Control flow
//! belongs to the registered architecture implementation.
use crate::{artifact::Result, model::Manifest};
use serde_json::Value;

use super::plan;
pub use orinfer_model_sdk::architecture::*;

pub(crate) struct Configuration {
    text: plan::TextConfig,
    pub vision_depth: usize,
    pub signature: Value,
}

impl Configuration {
    pub(super) fn text(&self) -> &plan::TextConfig {
        &self.text
    }

    pub fn parse(config: Value) -> Result<Self> {
        if !matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_text")
        ) {
            return Err("Unsupported Qwen model configuration".into());
        }
        let text_value = config.get("text_config").unwrap_or(&config);
        let text = plan::parse_text(text_value)?;
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
            text,
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
    plan::build(config, manifest, profiles)
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
