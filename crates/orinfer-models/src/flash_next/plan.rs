//! Registered E8P/A8 Flash Next order. Package slots provide implementations;
//! model data supplies storage identities, never execution control flow.
use crate::{
    artifact::Result,
    model::{KvGrowth, Manifest, Operation, PrefillPlan, SegmentControls},
};
use orinfer_model_sdk::architecture::PrefillProfile;
use std::collections::{BTreeMap, BTreeSet};

// Large prefill profiles use compact views of one stable 4096-row arena.
pub(super) fn arena_width(tokens: usize) -> usize {
    if tokens >= 256 { 4096 } else { tokens }
}

pub(super) fn score_role(tokens: usize, prefix: &str) -> String {
    let width = if prefix == "Draft" && tokens >= 256 {
        512
    } else {
        arena_width(tokens)
    };
    format!("{prefix}IndexScoresM{width}")
}

struct Recipe {
    recurrent: usize,
    attention: usize,
    first: usize,
    ple: usize,
    history_at: usize,
    ple_history_at: usize,
    last: usize,
    head: usize,
}
fn recipe(tokens: usize) -> Result<Recipe> {
    Ok(match tokens {
        1 => Recipe {
            recurrent: 40,
            attention: 54,
            first: 41,
            ple: 49,
            history_at: 12,
            ple_history_at: 21,
            last: 54,
            head: 9,
        },
        4096 => Recipe {
            recurrent: 42,
            attention: 56,
            first: 44,
            ple: 51,
            history_at: 12,
            ple_history_at: 21,
            last: 57,
            head: 9,
        },
        _ => return Err("Unsupported Flash prefill width".into()),
    })
}

/// Freeze layer boundaries after prerequisites are inserted. Later local
/// fusions update both the full program and its stages to keep them identical.
pub(super) fn stages(manifest: &mut Manifest, operands: &crate::packet::Operands) -> Result<()> {
    let mut registrations = vec![];
    for plan in &mut manifest.prefill_plans {
        if plan.chunk_tokens < 512 {
            continue;
        }
        let program = &plan.prefill_program;
        let operations = &manifest.programs[program];
        let mut starts = vec![0];
        let mut layer = 0;
        for (index, operation) in operations.iter().enumerate() {
            let Operation::Kernel { name } = operation else {
                continue;
            };
            let Some(suffix) = name.strip_prefix(&format!("{program}/layer")) else {
                continue;
            };
            let Some(number) = suffix
                .split('/')
                .next()
                .and_then(|s| s.parse::<usize>().ok())
            else {
                continue;
            };
            if number != layer {
                if number != layer + 1 {
                    return Err("Non-contiguous prefill layer order".into());
                }
                starts.push(index);
                layer = number;
            }
        }
        if starts.len() != 48 {
            return Err("Incomplete Flash prefill layer partition".into());
        }
        starts.push(operations.len());
        let stages: Vec<_> = starts
            .windows(2)
            .enumerate()
            .map(|(i, span)| {
                (
                    format!("{program}/stage{i}"),
                    operations[span[0]..span[1]].to_vec(),
                )
            })
            .collect();
        let initialize = format!(
            "{program}/begin/k{}",
            usize::from(manifest.vision.is_some())
        );
        let args = operands.arguments(&initialize)?;
        let residual = if plan.min_tokens == 0 {
            args.get(1)
        } else {
            args.iter()
                .find(|arg| {
                    matches!(arg, crate::artifact::Argument::Buffer { name }
                    if name != "Length" && name != "M4096_Embedding")
                })
                .or_else(|| args.get(1))
        };
        let Some(crate::artifact::Argument::Buffer { name: residual }) = residual else {
            return Err("Continuation must bind a whole mutable buffer".into());
        };
        plan.live_buffers.push(residual.clone());
        for (_, operations) in stages.iter().skip(1) {
            let Some(Operation::Kernel { name }) = operations.first() else {
                return Err("Prefill continuation must begin with a layer kernel".into());
            };
            let args: Vec<_> = operands
                .arguments(name)?
                .iter()
                .filter(|a| {
                    matches!(
                        a,
                        crate::artifact::Argument::Buffer { .. }
                            | crate::artifact::Argument::BufferSlice { .. }
                    )
                })
                .collect();
            let indices: &[usize] = if name.contains("/layer1/") {
                // PLE's input A8 projection is prepared in the preceding
                // section. Its row scales remain live with the activation.
                &[0, 2]
            } else {
                // HC fuses the previous FFN's injection into the next layer's
                // normalization. Its Block/Inject operands cross this cut.
                match args.len() {
                    3 => &[1],
                    4 => &[2],
                    6 => &[0, 1, 4],
                    7 => &[0, 1, 5],
                    _ => return Err("Unsupported HC continuation boundary".into()),
                }
            };
            for &index in indices {
                let Some(crate::artifact::Argument::Buffer { name }) = args.get(index) else {
                    return Err("Missing HC/PLE continuation operand".into());
                };
                plan.live_buffers.push(name.clone());
            }
        }
        plan.live_buffers.push("Length".into());
        plan.live_buffers.sort();
        plan.live_buffers.dedup();
        // Position, MRope coordinates, and visual assets already belong to the
        // request's private arena. Only shared cross-layer values need copying.
        registrations.push((
            program.clone(),
            stages.into_iter().map(|(_, ops)| ops).collect(),
            plan.live_buffers.clone(),
        ));
    }
    for (program, stages, live) in registrations {
        crate::prefill::register(manifest, &program, stages, live)?;
    }
    Ok(())
}
pub(super) fn section(
    program: &str,
    name: &str,
    count: usize,
    history: Option<(usize, usize, usize)>,
) -> Vec<Operation> {
    let mut slot = 0;
    (0..count)
        .map(|index| {
            if let Some((at, tokens, layer)) = history
                && at == index
            {
                return Operation::Copy {
                    source: format!("M{}_HistoryOut", arena_width(tokens)),
                    destination: format!("State_{layer}_conv"),
                    bytes: 3 * 10240 * 2,
                };
            }
            let operation = Operation::Kernel {
                name: format!("{program}/{name}/k{slot}"),
            };
            slot += 1;
            operation
        })
        .collect()
}
pub(super) fn build(manifest: &mut Manifest, profiles: &[PrefillProfile]) -> Result<()> {
    if manifest.chunk_tokens != 4096 {
        return Err("Unsupported Flash scheduling/features contract".into());
    }
    let widths: BTreeSet<_> = profiles.iter().map(|p| p.tokens).collect();
    let dynamic = profiles.len() == 1
        && profiles[0].tokens == 4096
        && profiles[0].min_tokens == 1
        && profiles[0].kind == "flash_dynamic_recurrent";
    if !dynamic {
        return Err("Flash requires one dynamic 1..4096 prefill template".into());
    }
    let mut states = BTreeSet::from([
        "State_ple".to_owned(),
        manifest.position.clone(),
        manifest.token.clone(),
        manifest.status.clone(),
    ]);
    for layer in 0..48 {
        let suffixes: &[&str] = if layer % 4 == 3 {
            &[
                "key",
                "value",
                "key_scale",
                "value_scale",
                "index",
                "pending",
            ]
        } else {
            &["conv", "gdn"]
        };
        states.extend(suffixes.iter().map(|s| format!("State_{layer}_{s}")));
    }
    if let Some(spec) = &manifest.mtp {
        states.insert(spec.position.clone());
        states.insert("DraftM1_Condition".into());
        states.extend(
            [
                "key",
                "value",
                "key_scale",
                "value_scale",
                "index",
                "pending",
            ]
            .map(|s| format!("State_48_{s}")),
        );
    }
    if states != manifest.reset_buffers.iter().cloned().collect() {
        return Err("Flash reset/prefix state must retain every recurrent and QSA state".into());
    }
    let mut programs = BTreeMap::new();
    manifest.prefill_plans.clear();
    for tokens in std::iter::once(1).chain(widths.iter().copied()) {
        let r = recipe(tokens)?;
        let history = format!("M{}_HistoryOut", arena_width(tokens));
        if manifest
            .buffers
            .iter()
            .find(|b| b.name == history)
            .is_none_or(|b| b.bytes().ok() != Some(61440) || b.data.is_some())
        {
            return Err("Missing Flash convolution history output role".into());
        }
        let program = if tokens == 1 {
            "decode".into()
        } else {
            format!("prefill_m{tokens}")
        };
        let head = format!("head_m{tokens}");
        let mut ops = section(
            &program,
            "begin",
            1 + usize::from(manifest.vision.is_some()),
            None,
        );
        for layer in 0..48 {
            let count = match layer {
                0 => r.first,
                1 => r.ple,
                47 => r.last,
                _ if layer % 4 == 3 => r.attention,
                _ => r.recurrent,
            };
            let history = (layer % 4 != 3).then_some((
                if layer == 1 {
                    r.ple_history_at
                } else {
                    r.history_at
                },
                tokens,
                layer,
            ));
            ops.extend(section(&program, &format!("layer{layer}"), count, history));
        }
        ops.extend(section(&program, "end", 1, None));
        let mut head_ops = section(&head, "body", r.head, None);
        let min_tokens = profiles
            .iter()
            .find(|p| p.tokens == tokens)
            .map_or(0, |p| p.min_tokens);
        if min_tokens > 0 {
            head_ops.insert(
                4,
                Operation::Kernel {
                    name: format!("{head}/body/logical_gather"),
                },
            );
        }
        if tokens == 1 {
            ops.extend(head_ops.clone());
        } else {
            manifest.prefill_plans.push(PrefillPlan {
                min_tokens,
                chunk_tokens: tokens,
                prefill_program: program.clone(),
                head_program: head.clone(),
                stages: vec![],
                live_buffers: vec![],
            });
        }
        programs.insert(program, ops);
        programs.insert(head, head_ops);
    }
    programs.insert("prefill".into(), programs["prefill_m4096"].clone());
    programs.insert("head".into(), programs["head_m4096"].clone());
    manifest.segment_controls = Some(SegmentControls {
        length: "Length".into(),
        last_index: "LastIndex".into(),
    });
    let kv = manifest
        .kv_cache
        .as_mut()
        .ok_or("Missing Flash INT8 KV contract")?;
    if !kv.direct_prefill || !kv.demand_mapping {
        return Err("Flash requires demand-mapped direct INT8 KV".into());
    }
    let mut expected = BTreeMap::new();
    let mut divisors = BTreeMap::new();
    for layer in (3..48).step_by(4) {
        for (suffix, stride) in [
            ("key", 512),
            ("value", 512),
            ("key_scale", 16),
            ("value_scale", 16),
            ("index", 64),
        ] {
            expected.insert(format!("State_{layer}_{suffix}"), stride);
        }
        divisors.insert(format!("State_{layer}_index"), 4);
    }
    if manifest.mtp.is_some() {
        for (suffix, stride) in [
            ("key", 512),
            ("value", 512),
            ("key_scale", 16),
            ("value_scale", 16),
            ("index", 64),
        ] {
            expected.insert(format!("State_48_{suffix}"), stride);
        }
        divisors.insert("State_48_index".into(), 4);
    }
    if kv.buffers != expected || kv.prefix_divisors != divisors {
        return Err("Flash KV layout differs from group-64 INT8 and group-4 index storage".into());
    }
    let mut scores: BTreeMap<_, _> = [1, 16, 128, 4096]
        .into_iter()
        .map(|m| (score_role(m, "Target"), m))
        .collect();
    if manifest.mtp.is_some() {
        scores.extend(
            [1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 512]
                .into_iter()
                .map(|m| (score_role(m, "Draft"), m)),
        );
        scores.extend((2..=8).map(|m| (score_role(m, "Verify"), m)));
    }
    if kv.prefill_workspace != scores
        || scores.iter().any(|(name, stride)| {
            manifest
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .is_none_or(|b| {
                    b.dtype != crate::artifact::Dtype::F32
                        || b.bytes().ok() != Some(manifest.max_context * stride)
                        || b.data.is_some()
                })
        })
    {
        return Err("Flash column-major score workspace contract differs".into());
    }
    kv.growth.clear();
    for (name, tokens) in std::iter::once(("decode".to_owned(), 1))
        .chain(widths.iter().map(|&n| (format!("prefill_m{n}"), n)))
        .chain(std::iter::once(("prefill".to_owned(), 4096)))
    {
        kv.growth.insert(
            name,
            KvGrowth {
                length: profiles
                    .iter()
                    .any(|p| p.tokens == tokens && p.min_tokens > 0)
                    .then(|| "Length".to_owned()),
                position: manifest.position.clone(),
                tokens,
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|n| !n.starts_with("State_48_"))
                    .cloned()
                    .chain([score_role(tokens, "Target")])
                    .collect(),
            },
        );
    }
    super::vision::register(manifest, &mut programs)?;
    manifest.programs = programs;
    super::mtp::build(manifest)?;
    super::batching::register(manifest)?;
    Ok(())
}
