//! Native Flash HC speculation and complete recurrent/index prefix commit.
use super::plan::section;
use crate::{
    artifact::Result,
    model::{KvGrowth, Manifest, Operation},
};
use std::collections::BTreeSet;

fn verification_layer(program: &str, layer: usize, tokens: usize) -> Vec<Operation> {
    let count = match layer {
        0 => 44,
        1 => 53,
        _ if layer % 4 == 3 => 55,
        _ => 43,
    };
    let history = if layer == 1 { 23 } else { 13 };
    let mut slot = 0;
    (0..count)
        .map(|index| {
            if layer % 4 != 3 {
                let copy = match index {
                    n if n == history => Some((
                        format!("M{tokens}_HistoryOut"),
                        format!("State_{layer}_conv"),
                        61440,
                    )),
                    n if n == history + 2 => Some((
                        format!("VerifyM{tokens}_Keys"),
                        format!("VerifyM{tokens}_Saved_{layer}_gdn_K"),
                        tokens * 16 * 128 * 2,
                    )),
                    n if n == history + 3 => Some((
                        format!("VerifyM{tokens}_Gates"),
                        format!("VerifyM{tokens}_Saved_{layer}_gdn_G"),
                        tokens * 48 * 4,
                    )),
                    _ => None,
                };
                if let Some((source, destination, bytes)) = copy {
                    return Operation::Copy {
                        source,
                        destination,
                        bytes,
                    };
                }
            }
            let name = format!("{program}/layer{layer}/k{slot}");
            slot += 1;
            Operation::Kernel { name }
        })
        .collect()
}

pub(super) fn build(manifest: &mut Manifest) -> Result<()> {
    let Some(spec) = manifest.mtp.clone() else {
        return Ok(());
    };
    if spec.position != "MtpPosition"
        || spec.input != "MtpInput"
        || spec.token != "MtpToken"
        || spec.status != "MtpStatus"
        || spec.verification_tokens != "VerificationTokens"
        || spec.verification_status != "VerificationStatus"
        || spec.draft_logits != "DraftLogits"
        || spec.verification_logits != "VerificationLogits"
        || spec.accepted_inputs != "AcceptedInputs"
        || spec.target_length != "PositionOut"
        || spec.draft_program != "mtp_draft"
        || spec.hidden_ring.as_deref() != Some("MtpHiddenRing")
        || spec.feature_index.is_some()
        || !spec.commit_always
        || spec.draft_snapshot_program.as_deref() != Some("mtp_snapshot")
        || spec.draft_restore_program.as_deref() != Some("mtp_restore_draft")
        || spec.default_verification_tokens != 4
    {
        return Err("Unsupported Flash MTP control/state contract".into());
    }
    let warm_widths: BTreeSet<_> = spec.warm_plans.iter().map(|p| p.tokens).collect();
    let capture_widths: BTreeSet<_> = spec.capture_plans.iter().map(|p| p.tokens).collect();
    let verify_widths: BTreeSet<_> = spec.verification_plans.iter().map(|p| p.tokens).collect();
    if warm_widths != BTreeSet::from([1, 2, 3, 4, 5, 6, 7, 8, 16, 128, 512])
        || warm_widths.len() != spec.warm_plans.len()
        || capture_widths != BTreeSet::from([1, 16, 128, 512])
        || capture_widths.len() != spec.capture_plans.len()
        || verify_widths != (2..=8).collect()
        || verify_widths.len() != spec.verification_plans.len()
    {
        return Err("Unsupported Flash MTP profile coverage".into());
    }
    let programs = &mut manifest.programs;
    for p in &spec.warm_plans {
        let program = format!("mtp_warm_m{}", p.tokens);
        let head = format!("mtp_head_m{}", p.tokens);
        if p.program != program || p.head_program != head {
            return Err("Unknown Flash draft program".into());
        }
        let (body_count, head_count) = match p.tokens {
            1..=8 => (54, 10),
            16 | 128 => (58, 11),
            512 => (57, 10),
            _ => unreachable!(),
        };
        let mut ops = section(&program, "begin", 7, None);
        ops.extend(section(&program, "layer48", body_count, None));
        ops.extend(section(&program, "end", 1, None));
        let head_ops = section(&head, "body", head_count, None);
        if p.tokens == 1 {
            let mut chain = ops[1..].to_vec();
            chain.extend(head_ops.clone());
            programs.insert("mtp_draft".into(), chain);
        }
        programs.insert(program, ops);
        programs.insert(head, head_ops);
    }
    for p in &spec.verification_plans {
        let program = format!("verify_m{}", p.tokens);
        let restore = format!("restore_m{}", p.tokens);
        let capture = format!("mtp_capture_m{}", p.tokens);
        if p.program != program || p.restore_program != restore || p.capture_program != capture {
            return Err("Unknown Flash verification program".into());
        }
        let mut ops = section(&program, "begin", 1, None);
        let mut commit = vec![];
        for layer in 0..48 {
            ops.extend(verification_layer(&program, layer, p.tokens));
            commit.extend(section(
                &restore,
                &format!("layer{layer}"),
                if layer == 1 {
                    3
                } else if layer % 4 == 3 {
                    1
                } else {
                    2
                },
                None,
            ));
        }
        ops.extend(section(&program, "end", 1, None));
        ops.extend(section(&program, "head", 9, None));
        programs.insert(program, ops);
        programs.insert(restore, commit);
        programs.insert(capture.clone(), section(&capture, "body", 1, None));
    }
    for p in &spec.capture_plans {
        let name = format!("mtp_capture_m{}", p.tokens);
        if p.program != name {
            return Err("Unknown Flash capture program".into());
        }
        programs.insert(name.clone(), section(&name, "body", 1, None));
    }
    for (program, source, destination) in [
        ("mtp_snapshot", "State_48_pending", "DraftPendingSaved"),
        ("mtp_restore_draft", "DraftPendingSaved", "State_48_pending"),
    ] {
        programs.insert(
            program.into(),
            vec![Operation::Copy {
                source: source.into(),
                destination: destination.into(),
                bytes: 1024,
            }],
        );
    }
    let kv = manifest.kv_cache.as_mut().ok_or("Missing Flash KV")?;
    for p in &spec.verification_plans {
        kv.growth.insert(
            p.program.clone(),
            KvGrowth {
                position: manifest.position.clone(),
                tokens: p.tokens,
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|n| !n.starts_with("State_48_"))
                    .cloned()
                    .collect(),
            },
        );
    }
    for p in &spec.warm_plans {
        kv.growth.insert(
            p.program.clone(),
            KvGrowth {
                position: spec.position.clone(),
                tokens: p.tokens,
                buffers: kv
                    .buffers
                    .keys()
                    .filter(|n| n.starts_with("State_48_"))
                    .cloned()
                    .collect(),
            },
        );
    }
    kv.growth.insert(
        spec.draft_program.clone(),
        KvGrowth {
            position: spec.position,
            tokens: 1,
            buffers: kv
                .buffers
                .keys()
                .filter(|n| n.starts_with("State_48_"))
                .cloned()
                .collect(),
        },
    );
    Ok(())
}
