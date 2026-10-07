//! Registered Qwen3_5 execution order. Package bindings supply implementations,
//! not control flow. Integer slot indices are local to each typed program section.
use super::config::Configuration;
use crate::{
    artifact::Result,
    model::{Manifest, Operation},
};
use Extent::{Hidden, History, Residual, State, Token};
use Step::{Copy as C, Kernel as K, Zero as Z};
use orinfer_model_sdk::architecture::*;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
#[path = "batching.rs"]
pub(crate) mod batching;

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
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
        let t = &self.config.text();
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
        for (layer, kind) in self.config.text().layer_types.iter().enumerate() {
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

const CHUNK_LUT4: Recipe = Recipe {
    begin: &[K(0), K(1), Z("R0", Residual)],
    gdn: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        C("Ho", "L{layer}_History", History),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        C("Sout", "L{layer}_State", State),
        K(12),
        K(13),
        K(14),
        K(15),
        K(16),
        K(17),
        K(18),
    ],
    attention: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        K(12),
    ],
    end: &[K(0)],
};

const CHUNK_EXPANDED: Recipe = Recipe {
    begin: &[K(0), K(1), Z("R0", Residual)],
    gdn: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        C("Ho", "L{layer}_History", History),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        C("Sout", "L{layer}_State", State),
        K(12),
        K(13),
        K(14),
        K(15),
        K(16),
        K(17),
        K(18),
        K(19),
        K(20),
    ],
    attention: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        K(12),
        K(13),
        K(14),
    ],
    end: &[K(0)],
};

const SEQUENCE: Recipe = Recipe {
    begin: &[K(0), K(1), Z("R0", Residual)],
    gdn: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        C("Ho", "L{layer}_History", History),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        K(12),
        K(13),
        K(14),
        K(15),
    ],
    attention: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
    ],
    end: &[K(0)],
};

const DECODE: Recipe = Recipe {
    begin: &[C("Token", "Input", Token), K(0), K(1), Z("R0", Residual)],
    gdn: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        C("Ho", "L{layer}_History", History),
        K(5),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
        K(12),
        K(13),
    ],
    attention: &[
        K(0),
        K(1),
        K(2),
        K(3),
        K(4),
        K(5),
        K(6),
        K(7),
        K(8),
        K(9),
        K(10),
        K(11),
    ],
    end: &[K(0), K(1), K(2), K(3), K(4)],
};

const HEAD: &[Step] = &[K(0), K(1), K(2), K(3)];
const VISION: Recipe = Recipe {
    begin: &[K(0), K(1)],
    gdn: &[],
    attention: &[K(0), K(1), K(2), K(3), K(4), K(5), K(6), K(7), K(8), K(9)],
    end: &[K(0), K(1), K(2)],
};
const MTP_WARM: &[Step] = &[
    K(0),
    K(1),
    K(2),
    K(3),
    K(4),
    K(5),
    Z("MtpR0", Residual),
    K(6),
    K(7),
    K(8),
    K(9),
    K(10),
    K(11),
    K(12),
    K(13),
    K(14),
    K(15),
    K(16),
    K(17),
    K(18),
];
const MTP_HEAD: &[Step] = &[
    K(0),
    K(1),
    K(2),
    K(3),
    C("MtpLastHidden", "MtpCondition", Hidden),
];

pub(super) fn build(
    config: &Configuration,
    manifest: &mut Manifest,
    profiles: &[PrefillProfile],
) -> Result<std::collections::BTreeSet<String>> {
    if manifest.vocab != config.text().vocab_size || profiles.is_empty() {
        return Err("Model vocabulary or prefill profiles differ from configuration".into());
    }
    let mut b = Builder {
        config,
        programs: BTreeMap::new(),
    };
    manifest.prefill_plans.clear();
    let mut sizes = std::collections::BTreeSet::new();
    for profile in profiles {
        if profile.tokens == 0
            || profile.tokens > manifest.chunk_tokens
            || !sizes.insert(profile.tokens)
        {
            return Err("Invalid/duplicate operator prefill profile".into());
        }
        let recipe = match profile.kind.as_str() {
            "chunk_lut4" => &CHUNK_LUT4,
            "chunk_expanded" => &CHUNK_EXPANDED,
            "sequence" => &SEQUENCE,
            "recurrent" => &Recipe {
                begin: SEQUENCE.begin,
                gdn: DECODE.gdn,
                attention: &[
                    K(0),
                    K(1),
                    K(2),
                    K(3),
                    K(5),
                    K(6),
                    K(7),
                    K(8),
                    K(9),
                    K(10),
                    K(11),
                ],
                end: SEQUENCE.end,
            },
            _ => return Err("Qwen A8 package does not support this prefill strategy".into()),
        };
        let prefill = format!("prefill_m{}", profile.tokens);
        let head = format!("head_m{}", profile.tokens);
        b.text(&prefill, recipe, profile.tokens)?;
        b.programs.insert(
            head.clone(),
            b.emit(HEAD, &head, "body", profile.tokens, 0)?,
        );
        manifest.prefill_plans.push(crate::model::PrefillPlan {
            chunk_tokens: profile.tokens,
            prefill_program: prefill,
            head_program: head,
        });
    }
    if !sizes.contains(&manifest.chunk_tokens) {
        return Err("Operator package lacks the model's maximum prefill chunk".into());
    }
    b.text("decode", &DECODE, 1)?;
    if manifest.greedy_sampling {
        b.programs.insert(
            "greedy_sampling".into(),
            vec![
                Operation::Zero {
                    destination: "SamplingCounts".into(),
                    bytes: manifest.vocab * 4,
                },
                Operation::Kernel {
                    name: "greedy_sampling/count".into(),
                },
                Operation::Kernel {
                    name: "greedy_sampling/partials".into(),
                },
                Operation::Kernel {
                    name: "greedy_sampling/merge".into(),
                },
            ],
        );
    }
    batching::register(config, manifest, profiles, &mut b)?;
    b.programs.insert(
        "prefill".into(),
        b.programs[&format!("prefill_m{}", manifest.chunk_tokens)].clone(),
    );
    b.programs.insert(
        "head".into(),
        b.programs[&format!("head_m{}", manifest.chunk_tokens)].clone(),
    );
    if let Some(vision) = &manifest.vision {
        if config.vision_depth == 0 {
            return Err("Vision enabled without vision configuration".into());
        }
        for plan in &vision.plans {
            let program = format!("vision_m{}", plan.patches);
            if plan.program != program {
                return Err("Vision program differs from registered plan".into());
            }
            let mut ops = b.emit(VISION.begin, &program, "begin", plan.patches, 0)?;
            for layer in 0..config.vision_depth {
                ops.extend(b.emit(
                    VISION.attention,
                    &program,
                    &format!("layer{layer}"),
                    plan.patches,
                    layer,
                )?);
            }
            ops.extend(b.emit(VISION.end, &program, "end", plan.patches, 0)?);
            b.programs.insert(program, ops);
        }
    }
    if let Some(mtp) = &manifest.mtp {
        if mtp.draft_program != "mtp_draft" {
            return Err("Unknown MTP draft program".into());
        }
        for plan in &mtp.capture_plans {
            let program = format!("mtp_capture_m{}", plan.tokens);
            if plan.program != program {
                return Err("Unknown MTP capture program".into());
            }
            b.programs.insert(
                program.clone(),
                b.emit(&[K(0)], &program, "body", plan.tokens, 0)?,
            );
        }
        for plan in &mtp.warm_plans {
            let program = format!("mtp_warm_m{}", plan.tokens);
            let head = format!("mtp_head_m{}", plan.tokens);
            if plan.program != program || plan.head_program != head {
                return Err("Unknown MTP warm program".into());
            }
            b.programs.insert(
                program.clone(),
                b.emit(MTP_WARM, &program, "body", plan.tokens, 0)?,
            );
            b.programs.insert(
                head.clone(),
                b.emit(MTP_HEAD, &head, "body", plan.tokens, 0)?,
            );
        }
        for plan in &mtp.verification_plans {
            let program = format!("verify_m{}", plan.tokens);
            let restore = format!("restore_m{}", plan.tokens);
            let capture = format!("mtp_capture_m{}", plan.tokens);
            if plan.program != program
                || plan.restore_program != restore
                || plan.capture_program != capture
            {
                return Err("Unknown MTP verification program".into());
            }
            let mut ops = b
                .programs
                .get(&format!("prefill_m{}", plan.tokens))
                .ok_or("Verification lacks a matching text prefill plan")?
                .clone();
            // Verification scores every proposal; ordinary prefill head scores
            // only the last token. They have different bindings and shapes.
            ops.extend(b.emit(HEAD, &program, "head", plan.tokens, 0)?);
            b.programs.insert(program, ops);
            let mut ops = vec![];
            for (layer, kind) in config.text().layer_types.iter().enumerate() {
                if kind == "linear_attention" {
                    ops.extend(b.emit(
                        &[K(0), K(1)],
                        &restore,
                        &format!("layer{layer}"),
                        plan.tokens,
                        layer,
                    )?);
                }
            }
            b.programs.insert(restore, ops);
        }
        let warm = b
            .programs
            .get("mtp_warm_m1")
            .ok_or("MTP draft requires warm_m1")?;
        let mut ops = vec![Operation::Copy {
            source: mtp.token.clone(),
            destination: mtp.input.clone(),
            bytes: 4,
        }];
        ops.extend(warm.iter().skip(1).cloned()); // draft uses the existing condition, not gather
        ops.extend(
            b.programs
                .get("mtp_head_m1")
                .ok_or("MTP draft requires head_m1")?
                .clone(),
        );
        b.programs.insert("mtp_draft".into(), ops);
    }
    if let Some(kv) = &mut manifest.kv_cache {
        if kv.direct_prefill && kv.prefill_workspace.is_empty() {
            // Chunk recipes retain their stable slot numbering but omit gather.
            for (program, ops) in &mut b.programs {
                let chunk = profiles.iter().any(|p| {
                    matches!(p.kind.as_str(), "chunk_expanded" | "chunk_lut4")
                        && program == &format!("prefill_m{}", p.tokens)
                });
                if chunk || program == "prefill" {
                    for (layer, kind) in config.text().layer_types.iter().enumerate() {
                        if kind == "full_attention" {
                            let gather = format!("{program}/layer{layer}/k4");
                            // The alias prefill uses bindings of its largest profile.
                            let gather_alias =
                                format!("prefill_m{}/layer{layer}/k4", manifest.chunk_tokens);
                            ops.retain(|op| !matches!(op, Operation::Kernel { name }
                                if name == &gather || (program == "prefill" && name == &gather_alias)));
                        }
                    }
                }
            }
        }
        for program in b.programs.keys() {
            let (position, tokens, mtp) = if program == "decode" {
                (manifest.position.clone(), 1, false)
            } else if program == "prefill" {
                (manifest.position.clone(), manifest.chunk_tokens, false)
            } else if let Some(n) = program
                .strip_prefix("prefill_m")
                .or_else(|| program.strip_prefix("verify_m"))
            {
                (
                    manifest.position.clone(),
                    n.parse().map_err(|_| "Invalid prefill size")?,
                    false,
                )
            } else if program == "mtp_draft" {
                (
                    manifest.mtp.as_ref().ok_or("Missing MTP")?.position.clone(),
                    1,
                    true,
                )
            } else if let Some(n) = program.strip_prefix("mtp_warm_m") {
                (
                    manifest.mtp.as_ref().ok_or("Missing MTP")?.position.clone(),
                    n.parse().map_err(|_| "Invalid MTP size")?,
                    true,
                )
            } else {
                continue;
            };
            let mut buffers: Vec<String> = kv
                .buffers
                .keys()
                .filter(|name| name.starts_with("Mtp") == mtp)
                .cloned()
                .collect();
            if (mtp && tokens >= 32)
                || (!mtp
                    && (program == "prefill"
                        || profiles.iter().any(|p| {
                            matches!(p.kind.as_str(), "chunk_expanded" | "chunk_lut4")
                                && program == &format!("prefill_m{}", p.tokens)
                        })))
            {
                buffers.extend(kv.prefill_workspace.keys().cloned());
            }
            kv.growth.insert(
                program.clone(),
                crate::model::KvGrowth {
                    position,
                    tokens,
                    buffers,
                },
            );
        }
    }
    manifest.programs = b.programs;
    // Declare decode candidates here, independent of kernel names. Some of
    // these plans also run during prefill; the runtime supplies the call phase.
    let mut decode = std::collections::BTreeSet::from(["decode".to_owned()]);
    if let Some(mtp) = &manifest.mtp {
        decode.insert(mtp.draft_program.clone());
        let max_refresh = mtp
            .verification_plans
            .iter()
            .map(|p| p.tokens)
            .max()
            .unwrap_or(1);
        for plan in &mtp.warm_plans {
            if plan.tokens <= max_refresh {
                decode.insert(plan.program.clone());
                decode.insert(plan.head_program.clone());
            }
        }
        for plan in &mtp.capture_plans {
            if plan.tokens == 1 {
                decode.insert(plan.program.clone());
            }
        }
        for plan in &mtp.verification_plans {
            decode.extend([
                plan.program.clone(),
                plan.restore_program.clone(),
                plan.capture_program.clone(),
            ]);
        }
    }
    Ok(decode)
}

pub(super) fn parse_text(text_value: &Value) -> Result<TextConfig> {
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
    Ok(text)
}

#[cfg(test)]
mod extent_tests {
    use super::*;
    fn config() -> Value {
        serde_json::json!({"model_type":"qwen3_5_text","hidden_size":8,"vocab_size":16,"num_hidden_layers":2,"layer_types":["linear_attention","full_attention"],"linear_num_key_heads":1,"linear_num_value_heads":2,"linear_key_head_dim":2,"linear_value_head_dim":3,"linear_conv_kernel_dim":4})
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
