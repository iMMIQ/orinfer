//! Registered Qwen3_5 execution order. Package bindings supply implementations,
//! not control flow. Integer slot indices are local to each typed program section.
use super::*;
use Extent::{Hidden, History, Residual, State, Token};
use Step::{Copy as C, Kernel as K, Zero as Z};

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
    if manifest.vocab != config.text.vocab_size || profiles.is_empty() {
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
        let recipe = match profile.kind {
            PrefillKind::ChunkLut4 => &CHUNK_LUT4,
            PrefillKind::ChunkExpanded => &CHUNK_EXPANDED,
            PrefillKind::Sequence => &SEQUENCE,
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
            for (layer, kind) in config.text.layer_types.iter().enumerate() {
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
        if kv.direct_prefill {
            // Chunk recipes retain their stable slot numbering but omit gather.
            for (program, ops) in &mut b.programs {
                let chunk = profiles.iter().any(|p| {
                    p.kind != PrefillKind::Sequence && program == &format!("prefill_m{}", p.tokens)
                });
                if chunk || program == "prefill" {
                    for (layer, kind) in config.text.layer_types.iter().enumerate() {
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
            let buffers = kv
                .buffers
                .keys()
                .filter(|name| name.starts_with("Mtp") == mtp)
                .cloned()
                .collect();
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
