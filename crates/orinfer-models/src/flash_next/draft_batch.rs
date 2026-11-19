//! Independent draft rows share projections; every QSA update owns a request.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::{architecture::BatchSegment, packet::BatchOperation};
use std::collections::BTreeMap;

pub(super) fn plan(
    m: &Manifest,
    segments: &[BatchSegment],
    operation: BatchOperation,
    include_plan: bool,
) -> Result<Vec<Invocation>> {
    let body = operation != BatchOperation::DraftHead;
    let head = operation != BatchOperation::Warm;
    let spec = m.mtp.as_ref().ok_or("Missing draft adapter")?;
    let capacity = spec.batch_draft_profiles.last().copied().unwrap_or(0);
    let rows = crate::packet::Rows::packet(segments, capacity, capacity)?;
    let used = rows.total;
    if segments
        .iter()
        .any(|s| operation != BatchOperation::Warm && s.tokens != 1)
    {
        return Err("Draft packet requires unique independent one-token rows".into());
    }
    if !include_plan {
        return Ok(vec![]);
    }
    let offsets = rows.starts;
    let frame_capacity = 512;
    let prefix = "draft_rows";
    let arena = "DraftRows_";
    let mut out = vec![];
    fn push(
        out: &mut Vec<Invocation>,
        operation: Operation,
        sequence: Option<usize>,
        views: BTreeMap<String, BufferView>,
    ) {
        out.push(Invocation {
            operation,
            sequence,
            views,
            launch: None,
        });
    }
    let copy = |source: &str, destination: &str, bytes: usize| Operation::Copy {
        source: source.into(),
        destination: destination.into(),
        bytes,
    };
    let row = |name: &str, index: usize| -> Result<BufferView> {
        let buffer = m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or("Missing draft row arena")?;
        let bytes = buffer.bytes()?;
        if !bytes.is_multiple_of(frame_capacity) {
            return Err("Unaligned draft row arena".into());
        }
        crate::packet::view(name, index, bytes / frame_capacity)
    };
    let embedding = "DraftRows_Embedding".to_owned();
    let condition = "DraftRows_Condition".to_owned();
    let hidden = "DraftRows_Hidden".to_owned();
    // Initialize only live scratch rows, without touching request state.
    for (name, stride) in if body {
        vec![(&embedding, 5120), (&condition, 20480)]
    } else {
        vec![(&hidden, 20480)]
    } {
        push(
            &mut out,
            Operation::Zero {
                destination: name.clone(),
                bytes: used * stride,
            },
            None,
            BTreeMap::new(),
        );
    }
    for (lane, segment) in segments.iter().enumerate() {
        if body {
            for operation in super::draft_rows::causal_begin(
                m,
                segment.tokens,
                operation == BatchOperation::Warm,
            ) {
                let launch = if let Operation::Kernel { name } = &operation {
                    Some(
                        m.dynamic_batch_kernels
                            .get(name)
                            .ok_or("Missing draft causal control")?
                            .launch(segment.tokens)?,
                    )
                } else {
                    None
                };
                out.push(Invocation {
                    operation,
                    sequence: Some(segment.slot),
                    views: BTreeMap::new(),
                    launch,
                });
            }

            for (source, destination, bytes) in [
                (
                    "DraftM512_Embedding".to_owned(),
                    &embedding,
                    segment.tokens * 5120,
                ),
                (
                    "DraftM512_Condition".to_owned(),
                    &condition,
                    segment.tokens * 20480,
                ),
            ] {
                push(
                    &mut out,
                    copy(&source, destination, bytes),
                    Some(segment.slot),
                    BTreeMap::from([(destination.clone(), row(destination, offsets[lane])?)]),
                );
            }
        } else {
            push(
                &mut out,
                copy("DraftM1_Condition", &hidden, 20480),
                Some(segment.slot),
                BTreeMap::from([(hidden.clone(), row(&hidden, lane)?)]),
            );
        }
    }
    if body {
        for stage in ["begin", "pre"] {
            super::rows::append(
                m,
                &mut out,
                &format!("{prefix}/{stage}"),
                used,
                None,
                &BTreeMap::new(),
            )?;
        }
        // Dedicated draft arenas are row-major. Only interface buffers get a lane
        // view; causal scratch/control bindings still resolve within the owner slot.
        for (lane, segment) in segments.iter().enumerate() {
            let views: BTreeMap<String, BufferView> = m
                .buffers
                .iter()
                .filter(|b| {
                    b.name.starts_with(arena)
                        && b.bytes().is_ok_and(|n| n.is_multiple_of(frame_capacity))
                })
                .map(|b| Ok((b.name.clone(), row(&b.name, offsets[lane])?)))
                .collect::<Result<_>>()?;
            super::rows::append(
                m,
                &mut out,
                "causal_rows/draft",
                segment.tokens,
                Some(segment.slot),
                &views,
            )?;

            out.push(Invocation {
                operation: Operation::Kernel {
                    name: "causal_rows/warm/advance".into(),
                },
                sequence: Some(segment.slot),
                views: BTreeMap::new(),
                launch: {
                    Some(
                        m.dynamic_batch_kernels["causal_rows/warm/advance"]
                            .launch(segment.tokens)?,
                    )
                },
            });
        }
        super::rows::append(
            m,
            &mut out,
            &format!("{prefix}/post"),
            used,
            None,
            &BTreeMap::new(),
        )?;
    }
    if head {
        super::rows::append(
            m,
            &mut out,
            "draft_rows/head",
            segments.len(),
            None,
            &BTreeMap::new(),
        )?;
    }
    let logits = "FlashDecode_Logits".to_owned();
    let pairs = "FlashDecode_Selected".to_owned();
    for (lane, segment) in segments.iter().enumerate() {
        for (source, destination, bytes, extra) in [
            (&logits, spec.draft_logits.as_str(), m.vocab * 4, 0),
            (&pairs, spec.token.as_str(), 4, 0),
            (&pairs, spec.status.as_str(), 4, 4),
            (&hidden, "DraftM1_Condition", 20480, 0),
        ] {
            if !head && source != &hidden {
                continue;
            }
            if !body && source == &hidden {
                continue;
            }
            let endpoint = if source == &hidden {
                offsets[lane] + segment.tokens - 1
            } else {
                lane
            };
            let mut view = if source != &hidden {
                BufferView {
                    buffer: source.clone(),
                    offset: endpoint * (if source == &logits { m.vocab * 4 } else { 8 }),
                }
            } else {
                row(source, endpoint)?
            };
            view.offset += extra;
            out.push(Invocation {
                operation: copy(source, destination, bytes),
                sequence: Some(segment.slot),
                views: BTreeMap::from([(source.clone(), view)]),
                launch: None,
            });
        }
    }
    Ok(out)
}
