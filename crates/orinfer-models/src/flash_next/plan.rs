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
        16 | 128 => Recipe {
            recurrent: 44,
            attention: 58,
            first: 45,
            ple: 53,
            history_at: 13,
            ple_history_at: 22,
            last: 58,
            head: 10,
        },
        512 | 2048 | 4096 => Recipe {
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
    if widths != BTreeSet::from([16, 128, 512, 2048, 4096])
        || widths.len() != profiles.len()
        || profiles.iter().any(|p| p.kind != "flash_recurrent")
    {
        return Err(
            "Flash package needs registered 16/128/512/2048/4096 recurrent profiles".into(),
        );
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
    for tokens in [1, 16, 128, 512, 2048, 4096] {
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
        let head_ops = section(&head, "body", r.head, None);
        if tokens == 1 {
            ops.extend(head_ops.clone());
        } else {
            manifest.prefill_plans.push(PrefillPlan {
                chunk_tokens: tokens,
                prefill_program: program.clone(),
                head_program: head.clone(),
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
    for (name, tokens) in [
        ("decode", 1),
        ("prefill_m16", 16),
        ("prefill_m128", 128),
        ("prefill_m512", 512),
        ("prefill_m2048", 2048),
        ("prefill_m4096", 4096),
        ("prefill", 4096),
    ] {
        kv.growth.insert(
            name.into(),
            KvGrowth {
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
