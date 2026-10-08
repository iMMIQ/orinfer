//! Shared dense/MoE work with request-private recurrence and sparse attention.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::BatchSegment;
use std::collections::{BTreeMap, BTreeSet};

const PROFILES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];

fn stages(rows: usize) -> Vec<(String, usize)> {
    let small = rows <= 8;
    let pre = if small { 11 } else { 12 };
    let post = if small { 26 } else { 29 };
    let mut out = vec![
        ("begin".into(), 1),
        ("head".into(), if small { 8 } else { 9 }),
    ];
    for layer in 0..48 {
        let attention = layer % 4 == 3;
        if layer == 1 {
            out.push((format!("layer{layer}/ple"), 7));
        }
        out.extend([
            (
                format!("layer{layer}/pre"),
                pre - usize::from(attention) + usize::from(layer == 1),
            ),
            (format!("layer{layer}/post"), post - usize::from(attention)),
            (
                format!("layer{layer}/private"),
                if attention {
                    19
                } else if layer == 1 {
                    4
                } else {
                    2
                },
            ),
        ]);
    }
    out
}

pub(super) fn register(m: &mut Manifest) -> Result<()> {
    if m.batch_profiles.is_empty() {
        return Ok(());
    }
    if m.batch_profiles != PROFILES || !m.prefill_batch_profiles.is_empty() {
        return Err("Flash decode requires 2/4/8/16/32/64/128 profiles".into());
    }
    let layout = m
        .batch_layout
        .as_ref()
        .ok_or("Missing Flash batch layout")?;
    if layout.hidden != 2560
        || !layout.profiles.is_empty()
        || !layout.small_mixed_shapes.is_empty()
        || layout.layers
            != (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect::<Vec<String>>()
        || layout.row_strides.get("M1_Embedding") != Some(&5120)
        || layout.row_strides.get("M1_Ple") != Some(&5120)
        || layout.row_strides.get("Logits") != Some(&(m.vocab * 4))
        || layout.row_strides.get("Selected") != Some(&8)
    {
        return Err("Invalid Flash decode row/state layout".into());
    }
    for rows in PROFILES {
        for (name, stride) in &layout.row_strides {
            let role = format!("BatchM{rows}_{name}");
            if *stride == 0
                || m.buffers
                    .iter()
                    .find(|b| b.name == role)
                    .is_none_or(|b| b.data.is_some() || b.bytes().ok() != stride.checked_mul(rows))
            {
                return Err(format!(
                    "Missing or inconsistent Flash batch row role {role}"
                ));
            }
        }
        for (section, count) in stages(rows) {
            let program = format!("flash_batch_m{rows}/{section}");
            m.programs.insert(
                program.clone(),
                super::plan::section(&format!("flash_batch_m{rows}"), &section, count, None),
            );
        }
    }
    Ok(())
}

fn invocation(
    operation: Operation,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Invocation {
    Invocation {
        operation,
        sequence,
        views: views.clone(),
        launch: None,
    }
}
fn append(
    out: &mut Vec<Invocation>,
    m: &Manifest,
    name: &str,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    let ops = m
        .programs
        .get(name)
        .ok_or_else(|| format!("Missing Flash batch stage {name}"))?;
    out.extend(
        ops.iter()
            .cloned()
            .map(|op| invocation(op, sequence, views)),
    );
    Ok(())
}

pub(super) fn plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    if segments.is_empty()
        || segments.len() > 128
        || segments.iter().any(|s| s.tokens != 1)
        || segments
            .iter()
            .map(|s| s.slot)
            .collect::<BTreeSet<_>>()
            .len()
            != segments.len()
    {
        return Err("Flash batch needs 1..128 distinct single-token sequences".into());
    }
    if segments.len() == 1 {
        return Ok(m.programs["decode"]
            .iter()
            .cloned()
            .map(|op| invocation(op, Some(segments[0].slot), &BTreeMap::new()))
            .chain(m.mtp.as_ref().into_iter().flat_map(|_| {
                m.programs["mtp_capture_m1"]
                    .iter()
                    .cloned()
                    .map(|op| invocation(op, Some(segments[0].slot), &BTreeMap::new()))
            }))
            .collect());
    }
    let rows = *m
        .batch_profiles
        .iter()
        .find(|&&r| r >= segments.len())
        .ok_or("Missing Flash batch profile")?;
    let layout = m.batch_layout.as_ref().ok_or("Missing Flash row layout")?;
    let prefix = format!("flash_batch_m{rows}");
    let empty = BTreeMap::new();
    let views: Vec<_> = (0..segments.len())
        .map(|lane| {
            layout
                .row_strides
                .iter()
                .map(|(name, stride)| {
                    (
                        format!("BatchM{rows}_{name}"),
                        BufferView {
                            buffer: format!("BatchM{rows}_{name}"),
                            offset: lane * stride,
                        },
                    )
                })
                .collect::<BTreeMap<_, _>>()
        })
        .collect();
    let mut out = vec![];
    for (lane, segment) in segments.iter().enumerate() {
        if m.vision.is_some() {
            out.push(invocation(
                Operation::Kernel {
                    name: "decode/begin/k0".into(),
                },
                Some(segment.slot),
                &empty,
            ));
        }
        for name in ["M1_Embedding", "M1_Ple"] {
            out.push(invocation(
                Operation::Copy {
                    source: name.into(),
                    destination: format!("BatchM{rows}_{name}"),
                    bytes: 5120,
                },
                Some(segment.slot),
                &views[lane],
            ));
        }
    }
    if rows > segments.len() {
        // Private mixers do not write inactive lanes. Clear their row roles as
        // well, so shrinking a batch cannot reuse stale/nonfinite activations.
        for (name, stride) in &layout.row_strides {
            if matches!(name.as_str(), "Logits" | "Selected") {
                continue;
            }
            let name = format!("BatchM{rows}_{name}");
            let pad = BTreeMap::from([(
                name.clone(),
                BufferView {
                    buffer: name.clone(),
                    offset: segments.len() * stride,
                },
            )]);
            out.push(invocation(
                Operation::Zero {
                    destination: name,
                    bytes: (rows - segments.len()) * stride,
                },
                None,
                &pad,
            ));
        }
    }
    append(&mut out, m, &format!("{prefix}/begin"), None, &empty)?;
    for layer in 0..48 {
        let private = &m.programs[&format!("{prefix}/layer{layer}/private")];
        let mut first = 0;
        if layer == 1 {
            append(&mut out, m, &format!("{prefix}/layer1/ple"), None, &empty)?;
            for (lane, s) in segments.iter().enumerate() {
                out.extend(
                    private[..2]
                        .iter()
                        .cloned()
                        .map(|op| invocation(op, Some(s.slot), &views[lane])),
                );
            }
            first = 2;
        }
        append(
            &mut out,
            m,
            &format!("{prefix}/layer{layer}/pre"),
            None,
            &empty,
        )?;
        for (lane, s) in segments.iter().enumerate() {
            for (i, op) in private[first..].iter().enumerate() {
                out.push(invocation(op.clone(), Some(s.slot), &views[lane]));
                if layer % 4 != 3 && i == 0 {
                    out.push(invocation(
                        Operation::Copy {
                            source: "M1_HistoryOut".into(),
                            destination: format!("State_{layer}_conv"),
                            bytes: 61440,
                        },
                        Some(s.slot),
                        &empty,
                    ));
                }
            }
        }
        append(
            &mut out,
            m,
            &format!("{prefix}/layer{layer}/post"),
            None,
            &empty,
        )?;
    }
    for (lane, s) in segments.iter().enumerate() {
        out.push(invocation(
            Operation::Kernel {
                name: "decode/end/k0".into(),
            },
            Some(s.slot),
            &empty,
        ));
        if let Some(spec) = &m.mtp {
            let capture = spec
                .capture_plans
                .iter()
                .find(|p| p.tokens == 1)
                .ok_or("Missing Flash MTP capture")?;
            let capture_views = layout
                .row_strides
                .iter()
                .map(|(name, stride)| {
                    (
                        name.clone(),
                        BufferView {
                            buffer: format!("BatchM{rows}_{name}"),
                            offset: lane * stride,
                        },
                    )
                })
                .collect();
            append(&mut out, m, &capture.program, Some(s.slot), &capture_views)?;
        }
    }
    append(&mut out, m, &format!("{prefix}/head"), None, &empty)?;
    for (lane, s) in segments.iter().enumerate() {
        let mut copy = views[lane].clone();
        let pair = format!("BatchM{rows}_Selected");
        out.push(invocation(
            Operation::Copy {
                source: format!("BatchM{rows}_Logits"),
                destination: m.logits.clone(),
                bytes: m.vocab * 4,
            },
            Some(s.slot),
            &copy,
        ));
        for (offset, destination) in [(0, &m.token), (4, &m.status)] {
            copy.insert(
                pair.clone(),
                BufferView {
                    buffer: pair.clone(),
                    offset: lane * 8 + offset,
                },
            );
            out.push(invocation(
                Operation::Copy {
                    source: pair.clone(),
                    destination: destination.clone(),
                    bytes: 4,
                },
                Some(s.slot),
                &copy,
            ));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> Manifest {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":4096,
            "max_context":262144,"vocab":16,"toolchain":{},"buffers":[],
            "reset_buffers":[],"input":"Input","token":"Token","status":"Status",
            "logits":"Logits","position":"Position","weight_bytes":0,"weight_parameters":0,"weight_scope":"test"
        })).unwrap();
        m.batch_profiles = PROFILES.into();
        m.batch_layout = Some(orinfer_model_sdk::architecture::BatchLayout {
            hidden: 2560,
            layers: (0..48)
                .map(|i| {
                    if i % 4 == 3 {
                        "full_attention".into()
                    } else {
                        "linear_attention".into()
                    }
                })
                .collect(),
            row_strides: BTreeMap::from([
                ("M1_Embedding".into(), 5120),
                ("M1_Ple".into(), 5120),
                ("Logits".into(), 64),
                ("Selected".into(), 8),
                ("Residual".into(), 20480),
            ]),
            profiles: Default::default(),
            small_mixed_shapes: vec![],
        });
        for rows in PROFILES {
            for (name, stride) in &m.batch_layout.as_ref().unwrap().row_strides {
                m.buffers.push(serde_json::from_value(serde_json::json!({
                    "name":format!("BatchM{rows}_{name}"),"dtype":"u8","shape":[rows*stride],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write","data":null
                })).unwrap());
            }
        }
        m.programs.insert(
            "decode".into(),
            vec![Operation::Kernel {
                name: "decode/end/k0".into(),
            }],
        );
        register(&mut m).unwrap();
        m
    }
    #[test]
    fn batching_preserves_common_profiles_and_pads_only_shared_rows() {
        let m = manifest();
        for count in [2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 64, 127, 128] {
            let segments: Vec<_> = (0..count)
                .map(|i| BatchSegment {
                    slot: i * 3 + 7,
                    tokens: 1,
                })
                .collect();
            let p = plan(&m, &segments).unwrap();
            let rows = count.next_power_of_two();
            let recurrences:Vec<_>=p.iter().filter(|i| matches!(&i.operation,Operation::Kernel{name} if name==&format!("flash_batch_m{rows}/layer0/private/k1"))).collect();
            assert_eq!(recurrences.len(), count);
            for (lane, i) in recurrences.iter().enumerate() {
                assert_eq!(i.sequence, Some(lane * 3 + 7));
                assert_eq!(
                    i.views[&format!("BatchM{rows}_Residual")].offset,
                    lane * 20480
                );
            }
            let zeroes = p
                .iter()
                .filter(|i| matches!(i.operation, Operation::Zero { .. }))
                .count();
            assert_eq!(zeroes, if rows == count { 0 } else { 3 });
            assert_eq!(p.iter().filter(|i| matches!(&i.operation,Operation::Kernel{name} if name==&format!("flash_batch_m{rows}/layer0/pre/k6")) && i.sequence.is_none()).count(),1);
            assert_eq!(p.iter().filter(|i| matches!(&i.operation,Operation::Copy{destination,..} if destination=="Logits")).count(),count);
        }
    }
    #[test]
    fn slot_order_does_not_change_ownership_and_invalid_work_is_rejected() {
        let m = manifest();
        let p = plan(
            &m,
            &[
                BatchSegment {
                    slot: 19,
                    tokens: 1,
                },
                BatchSegment { slot: 2, tokens: 1 },
            ],
        )
        .unwrap();
        let owners:Vec<_>=p.iter().filter(|i| matches!(&i.operation,Operation::Copy{destination,..} if destination=="State_0_conv")).map(|i|i.sequence).collect();
        assert_eq!(owners, vec![Some(19), Some(2)]);
        for segments in [
            vec![],
            vec![BatchSegment { slot: 0, tokens: 2 }],
            vec![BatchSegment { slot: 0, tokens: 1 }; 2],
            (0..129)
                .map(|slot| BatchSegment { slot, tokens: 1 })
                .collect(),
        ] {
            assert!(plan(&m, &segments).is_err());
        }
        let mut broken = manifest();
        broken
            .batch_layout
            .as_mut()
            .unwrap()
            .row_strides
            .insert("M1_Ple".into(), 20480);
        assert!(register(&mut broken).is_err());
    }
}
