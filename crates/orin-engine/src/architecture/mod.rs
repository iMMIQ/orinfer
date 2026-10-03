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
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillProfile {
    pub tokens: usize,
    pub kind: PrefillKind,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

pub(crate) struct Configuration {
    pub architecture: Architecture,
    pub text: TextConfig,
    pub vision_depth: usize,
    pub signature: Value,
}

impl Configuration {
    pub fn parse(config: Value) -> Result<Self> {
        let architecture = match config["model_type"].as_str() {
            Some("qwen3_5" | "qwen3_5_text") => Architecture::Qwen3_5,
            _ => return Err("Unregistered model architecture".into()),
        };
        let text_value = config.get("text_config").unwrap_or(&config);
        let text: TextConfig =
            serde_json::from_value(text_value.clone()).map_err(|e| format!("Text config: {e}"))?;
        if text.hidden_size == 0
            || text.vocab_size == 0
            || text.num_hidden_layers == 0
            || text.layer_types.len() != text.num_hidden_layers
            || text
                .layer_types
                .iter()
                .any(|s| !matches!(s.as_str(), "linear_attention" | "full_attention"))
            || text.linear_num_key_heads == 0
            || text.linear_num_value_heads == 0
            || text.linear_key_head_dim == 0
            || text.linear_value_head_dim == 0
            || text.linear_conv_kernel_dim < 2
        {
            return Err("Invalid Qwen3_5 layer/state configuration".into());
        }
        // These values affect compiled arithmetic, weight contracts or state.
        // Checkpoint identity, naming, dtype labels and quantization provenance
        // deliberately do not determine architecture/operator compatibility.
        let text_keys = [
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "vocab_size",
            "layer_types",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "hidden_act",
            "rms_norm_eps",
            "rope_parameters",
            "partial_rotary_factor",
            "attn_output_gate",
            "output_gate_type",
            "attention_bias",
            "tie_word_embeddings",
            "mtp_num_hidden_layers",
            "mtp_use_dedicated_embeddings",
        ];
        let text_signature: serde_json::Map<String, Value> = text_keys
            .into_iter()
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
            text,
            vision_depth,
            signature,
        })
    }
}

#[derive(Clone, Copy)]
enum Extent {
    Residual,
    History,
    State,
    Hidden,
    Token,
}
#[derive(Clone, Copy)]
enum Step {
    Kernel(usize),
    Copy(&'static str, &'static str, Extent),
    Zero(&'static str, Extent),
}
struct Recipe {
    begin: &'static [Step],
    gdn: &'static [Step],
    attention: &'static [Step],
    end: &'static [Step],
}

struct Builder<'a> {
    config: &'a Configuration,
    programs: BTreeMap<String, Vec<Operation>>,
}
impl Builder<'_> {
    fn emit(
        &self,
        steps: &[Step],
        program: &str,
        section: &str,
        tokens: usize,
        layer: usize,
    ) -> Result<Vec<Operation>> {
        let t = &self.config.text;
        let bytes = |extent: Extent| -> Result<usize> {
            let factors: &[usize] = match extent {
                Extent::Residual => &[tokens, t.hidden_size, 4],
                Extent::Hidden => &[t.hidden_size, 2],
                Extent::Token => &[4],
                Extent::History => &[
                    t.linear_conv_kernel_dim - 1,
                    t.linear_num_key_heads
                        .checked_mul(t.linear_key_head_dim)
                        .and_then(|n| n.checked_mul(2))
                        .and_then(|n| {
                            t.linear_num_value_heads
                                .checked_mul(t.linear_value_head_dim)
                                .and_then(|v| n.checked_add(v))
                        })
                        .ok_or("History dimensions overflow")?,
                    2,
                ],
                Extent::State => &[
                    t.linear_num_value_heads,
                    t.linear_key_head_dim,
                    t.linear_value_head_dim,
                    4,
                ],
            };
            factors.iter().try_fold(1usize, |n, v| {
                n.checked_mul(*v)
                    .ok_or_else(|| "Plan extent overflow".into())
            })
        };
        let name = |s: &str| s.replace("{layer}", &layer.to_string());
        steps
            .iter()
            .map(|step| {
                Ok(match *step {
                    Step::Kernel(slot) => Operation::Kernel {
                        name: format!("{program}/{section}/k{slot}"),
                    },
                    Step::Copy(source, destination, extent) => Operation::Copy {
                        source: name(source),
                        destination: name(destination),
                        bytes: bytes(extent)?,
                    },
                    Step::Zero(destination, extent) => Operation::Zero {
                        destination: name(destination),
                        bytes: bytes(extent)?,
                    },
                })
            })
            .collect()
    }
    fn text(&mut self, program: &str, recipe: &Recipe, tokens: usize) -> Result<()> {
        let mut ops = self.emit(recipe.begin, program, "begin", tokens, 0)?;
        for (layer, kind) in self.config.text.layer_types.iter().enumerate() {
            let steps = if kind == "linear_attention" {
                recipe.gdn
            } else {
                recipe.attention
            };
            ops.extend(self.emit(steps, program, &format!("layer{layer}"), tokens, layer)?);
        }
        ops.extend(self.emit(recipe.end, program, "end", tokens, 0)?);
        self.programs.insert(program.into(), ops);
        Ok(())
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
    #[test]
    fn state_copy_extents_follow_configuration_and_reject_overflow() {
        let config = Configuration::parse(config()).unwrap();
        let builder = Builder {
            config: &config,
            programs: BTreeMap::new(),
        };
        let steps = &[
            Step::Copy("Ho", "L{layer}_History", Extent::History),
            Step::Copy("Sout", "L{layer}_State", Extent::State),
        ];
        let ops = builder.emit(steps, "prefill_m2", "layer1", 2, 1).unwrap();
        assert!(
            matches!(&ops[0], Operation::Copy { destination, bytes:60, .. } if destination=="L1_History")
        );
        assert!(matches!(&ops[1], Operation::Copy { bytes: 48, .. }));
        assert!(
            builder
                .emit(
                    &[Step::Zero("R0", Extent::Residual)],
                    "overflow",
                    "begin",
                    usize::MAX,
                    0
                )
                .is_err()
        );
    }
}
