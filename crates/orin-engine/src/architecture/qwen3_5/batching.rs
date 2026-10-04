//! Selective batching: shared row-wise projections, isolated causal mixers.
//! State kernels retain their validated per-sequence layout and FP32 recurrence.
use super::*;
use crate::execution::{BufferView, Invocation};

pub(super) fn register(
    config: &Configuration,
    m: &mut Manifest,
    profiles: &[PrefillProfile],
    builder: &mut Builder<'_>,
) -> Result<()> {
    if m.batch_profiles.is_empty() {
        return Ok(());
    }
    let t = &config.text;
    if t.intermediate_size == 0 {
        return Err("Batching requires intermediate_size".into());
    }
    let gdn_width = t.linear_num_value_heads * t.linear_value_head_dim;
    let qkv = 2 * t.linear_num_key_heads * t.linear_key_head_dim + gdn_width;
    let mut strides = BTreeMap::from([
        ("Hidden".into(), t.hidden_size * 2),
        ("Norm".into(), t.hidden_size * 2),
        ("R0".into(), t.hidden_size * 4),
        ("R1".into(), t.hidden_size * 4),
        ("Mix".into(), t.hidden_size * 2),
        ("QKV".into(), qkv * 2),
        ("Zout".into(), gdn_width * 2),
        ("Y".into(), gdn_width * 2),
        ("MixerIn".into(), gdn_width * 2),
        ("AB".into(), t.linear_num_value_heads * 4),
        ("g".into(), t.linear_num_value_heads * 4),
        ("Beta".into(), t.linear_num_value_heads * 4),
        ("GateUp".into(), t.intermediate_size * 4),
        ("Activated".into(), t.intermediate_size * 2),
        ("Positions".into(), 4),
    ]);
    // Full projection dimensions are supplied by the package's logical buffers.
    for name in ["FullX", "FullQ", "FullGate"] {
        let buffer = m
            .buffers
            .iter()
            .find(|b| b.name == name)
            .ok_or("Missing batch row buffer")?;
        strides.insert(name.into(), buffer.bytes()? / m.chunk_tokens);
    }
    let recipe = Recipe {
        begin: &[],
        gdn: &[
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
        attention: &[K(0), K(1), K(2), K(3), K(4), K(5), K(6), K(7), K(8)],
        end: &[K(0)],
    };
    for &rows in &m.batch_profiles {
        builder.text(&format!("batch_m{rows}"), &recipe, rows)?;
    }
    m.batch_layout = Some(BatchLayout {
        layers: t.layer_types.clone(),
        row_strides: strides,
        profiles: profiles.iter().map(|p| (p.tokens, p.kind)).collect(),
        hidden: t.hidden_size,
    });
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
    }
}
fn kernel(
    name: String,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Invocation {
    invocation(Operation::Kernel { name }, sequence, views)
}
fn section(m: &Manifest, program: &str, section: &str) -> Vec<Operation> {
    let prefix = format!("{program}/{section}/");
    m.programs[program]
        .iter()
        .filter(|op| matches!(op, Operation::Kernel { name } if name.starts_with(&prefix)))
        .cloned()
        .collect()
}

pub(crate) fn plan(m: &Manifest, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
    let layout = m
        .batch_layout
        .as_ref()
        .ok_or("Operator package lacks batching")?;
    if segments.is_empty() || segments.len() > 128 {
        return Err("Invalid batch segment count".into());
    }
    let mut unique = std::collections::BTreeSet::new();
    let total = segments.iter().try_fold(0usize, |n, s| -> Result<usize> {
        if s.tokens == 0
            || !unique.insert(s.slot)
            || (s.tokens != 1 && !layout.profiles.contains_key(&s.tokens))
        {
            return Err("Invalid or repeated request segment".into());
        }
        n.checked_add(s.tokens)
            .ok_or("Batch token count overflow".into())
    })?;
    let rows = m
        .batch_profiles
        .iter()
        .copied()
        .filter(|&n| n >= total)
        .min()
        .ok_or("Batch exceeds compiled row capacity")?;
    let head_rows = m
        .batch_profiles
        .iter()
        .copied()
        .filter(|&n| n >= segments.len())
        .min()
        .ok_or("Missing batch head capacity")?;
    let batch = format!("batch_m{rows}");
    let empty = BTreeMap::new();
    let mut out = vec![invocation(
        Operation::Zero {
            destination: "R0".into(),
            bytes: rows * layout.hidden * 4,
        },
        None,
        &empty,
    )];
    if rows > total {
        let padding = BTreeMap::from([(
            "Hidden".into(),
            BufferView {
                buffer: "Hidden".into(),
                offset: total * layout.hidden * 2,
            },
        )]);
        out.push(invocation(
            Operation::Zero {
                destination: "Hidden".into(),
                bytes: (rows - total) * layout.hidden * 2,
            },
            None,
            &padding,
        ));
    }
    let mut starts = vec![];
    let mut start = 0;
    for s in segments {
        let program = if s.tokens == 1 {
            "decode".into()
        } else {
            format!("prefill_m{}", s.tokens)
        };
        let views: BTreeMap<_, _> = layout
            .row_strides
            .iter()
            .map(|(n, stride)| {
                (
                    n.clone(),
                    BufferView {
                        buffer: n.clone(),
                        offset: start * stride,
                    },
                )
            })
            .collect();
        if s.tokens == 1 {
            out.push(invocation(
                Operation::Copy {
                    source: m.token.clone(),
                    destination: m.input.clone(),
                    bytes: 4,
                },
                Some(s.slot),
                &views,
            ));
        }
        for op in section(m, &program, "begin") {
            out.push(invocation(op, Some(s.slot), &views));
        }
        starts.push((program, views));
        start += s.tokens;
    }
    for (layer, kind) in layout.layers.iter().enumerate() {
        let is_gdn = kind == "linear_attention";
        let shared_prefix = if is_gdn { 4 } else { 2 };
        let shared_end = if is_gdn { 12 } else { 9 };
        for slot in 0..shared_prefix {
            out.push(kernel(
                format!("{batch}/layer{layer}/k{slot}"),
                None,
                &empty,
            ));
        }
        if rows > total {
            let stride = layout.row_strides["MixerIn"];
            let pad = BTreeMap::from([(
                "MixerIn".into(),
                BufferView {
                    buffer: "MixerIn".into(),
                    offset: total * stride,
                },
            )]);
            out.push(invocation(
                Operation::Zero {
                    destination: "MixerIn".into(),
                    bytes: (rows - total) * stride,
                },
                None,
                &pad,
            ));
        }
        for (s, (program, views)) in segments.iter().zip(&starts) {
            let sequence_kind = layout.profiles.get(&s.tokens);
            let ops = section(m, program, &format!("layer{layer}"));
            let selected: &[usize] = if is_gdn {
                if sequence_kind == Some(&PrefillKind::Sequence) {
                    &[4, 5, 6, 7]
                } else if s.tokens == 1 || sequence_kind == Some(&PrefillKind::Recurrent) {
                    &[4, 5]
                } else {
                    return Err("Chunk prefill cannot use the small-row batch mixer".into());
                }
            } else if sequence_kind == Some(&PrefillKind::Recurrent) {
                &[2, 3]
            } else {
                &[2, 3, 4]
            };
            for &i in selected {
                out.push(invocation(
                    ops.get(i).ok_or("Missing segment mixer")?.clone(),
                    Some(s.slot),
                    views,
                ));
                if is_gdn
                    && ((sequence_kind == Some(&PrefillKind::Sequence) && i == 5)
                        || (sequence_kind != Some(&PrefillKind::Sequence) && i == 4))
                {
                    out.push(invocation(
                        Operation::Copy {
                            source: "Ho".into(),
                            destination: format!("L{layer}_History"),
                            bytes: m
                                .buffers
                                .iter()
                                .find(|b| b.name == format!("L{layer}_History"))
                                .ok_or("Missing history")?
                                .bytes()?,
                        },
                        Some(s.slot),
                        views,
                    ));
                }
            }
        }
        for slot in shared_prefix..shared_end {
            out.push(kernel(
                format!("{batch}/layer{layer}/k{slot}"),
                None,
                &empty,
            ));
        }
    }
    for (lane, (s, (program, views))) in segments.iter().zip(&starts).enumerate() {
        out.push(invocation(
            section(m, program, "end")
                .into_iter()
                .next()
                .ok_or("Missing segment advance")?,
            Some(s.slot),
            views,
        ));
        if let Some(spec) = &m.mtp {
            let capture = spec
                .capture_plans
                .iter()
                .find(|p| p.tokens == s.tokens)
                .ok_or("Missing batch hidden capture")?;
            for op in &m.programs[&capture.program] {
                out.push(invocation(op.clone(), Some(s.slot), views));
            }
        }
        let head = if s.tokens == 1 {
            "decode".into()
        } else {
            format!("head_m{}", s.tokens)
        };
        let norm = if s.tokens == 1 {
            section(m, &head, "end").get(1).cloned()
        } else {
            m.programs[&head].first().cloned()
        }
        .ok_or("Missing segment head norm")?;
        let mut head_views = views.clone();
        head_views.insert(
            "LastHidden".into(),
            BufferView {
                buffer: "BatchHeadHidden".into(),
                offset: lane * layout.hidden * 2,
            },
        );
        out.push(invocation(norm, Some(s.slot), &head_views));
    }
    if head_rows > segments.len() {
        let pad = BTreeMap::from([(
            "BatchHeadHidden".into(),
            BufferView {
                buffer: "BatchHeadHidden".into(),
                offset: segments.len() * layout.hidden * 2,
            },
        )]);
        out.push(invocation(
            Operation::Zero {
                destination: "BatchHeadHidden".into(),
                bytes: (head_rows - segments.len()) * layout.hidden * 2,
            },
            None,
            &pad,
        ));
    }
    out.push(kernel(format!("batch_m{head_rows}/end/k0"), None, &empty));
    for (lane, s) in segments.iter().enumerate() {
        let views = BTreeMap::from([(
            "BatchHeadLogits".into(),
            BufferView {
                buffer: "BatchHeadLogits".into(),
                offset: lane * m.vocab * 4,
            },
        )]);
        out.push(invocation(
            Operation::Copy {
                source: "BatchHeadLogits".into(),
                destination: m.logits.clone(),
                bytes: m.vocab * 4,
            },
            Some(s.slot),
            &views,
        ));
        for i in [3, 4] {
            out.push(kernel(format!("decode/end/k{i}"), Some(s.slot), &empty));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn manifest() -> Manifest {
        let mut m: Manifest = serde_json::from_value(serde_json::json!({
            "schema_version":2,"target":"sm_87","model":"test","chunk_tokens":128,
            "max_context":1024,"vocab":16,"toolchain":{},"buffers":[{
                "name":"L0_History","dtype":"f16","shape":[30],"layout":"contiguous",
                "alignment":256,"access":"read_write","data":null}],
            "reset_buffers":[],"input":"Input","token":"Token","status":"TokenStatus",
            "logits":"Logits","position":"Step","weight_bytes":0,"weight_parameters":0,"weight_scope":"test"
        })).unwrap();
        m.batch_profiles = vec![2, 4, 8, 16, 32, 64, 128];
        m.batch_layout = Some(BatchLayout {
            layers: vec!["linear_attention".into(), "full_attention".into()],
            row_strides: BTreeMap::from([
                ("Hidden".into(), 16),
                ("R0".into(), 32),
                ("MixerIn".into(), 12),
                ("QKV".into(), 20),
            ]),
            profiles: BTreeMap::from([(2, PrefillKind::Sequence), (32, PrefillKind::Recurrent)]),
            hidden: 8,
        });
        let mut section = |program: &str, name: &str, count: usize| {
            m.programs
                .entry(program.into())
                .or_default()
                .extend((0..count).map(|i| Operation::Kernel {
                    name: format!("{program}/{name}/k{i}"),
                }));
        };
        for program in ["decode", "prefill_m2", "prefill_m32"] {
            section(program, "begin", 2);
            section(program, "layer0", 16);
            section(program, "layer1", 12);
            section(program, "end", 5);
        }
        section("head_m2", "body", 1);
        section("head_m32", "body", 1);
        m
    }
    #[test]
    fn mixed_segments_have_private_mixers_shared_projections_and_safe_padding() {
        let m = manifest();
        let p = plan(
            &m,
            &[
                BatchSegment { slot: 9, tokens: 1 },
                BatchSegment {
                    slot: 3,
                    tokens: 32,
                },
            ],
        )
        .unwrap();
        assert!(p.iter().any(
            |i| matches!(&i.operation,Operation::Kernel{name} if name=="batch_m64/layer0/k1")
                && i.sequence.is_none()
        ));
        let mixer = p
            .iter()
            .find(
                |i| matches!(&i.operation,Operation::Kernel{name} if name=="prefill_m32/layer0/k5"),
            )
            .unwrap();
        assert_eq!(mixer.sequence, Some(3));
        assert_eq!(mixer.views["QKV"].offset, 20);
        let gdn = p.iter().filter(|i| matches!(&i.operation,Operation::Kernel{name} if name.ends_with("layer0/k5") && i.sequence.is_some())).count();
        assert_eq!(gdn, 2);
        assert!(p.iter().any(|i| matches!(&i.operation,Operation::Zero{destination,bytes:496} if destination=="Hidden") && i.views["Hidden"].offset==528));
        assert!(p.iter().any(
            |i| matches!(&i.operation,Operation::Copy{destination,..} if destination=="Logits")
                && i.sequence == Some(3)
        ));
    }
    #[test]
    fn duplicate_lanes_unsupported_shapes_and_oversized_batches_are_rejected() {
        let m = manifest();
        for segments in [
            vec![],
            vec![BatchSegment { slot: 0, tokens: 0 }],
            vec![BatchSegment { slot: 0, tokens: 3 }],
            vec![
                BatchSegment { slot: 0, tokens: 1 },
                BatchSegment { slot: 0, tokens: 1 },
            ],
            (0..5)
                .map(|slot| BatchSegment { slot, tokens: 32 })
                .collect(),
        ] {
            assert!(plan(&m, &segments).is_err());
        }
    }
}
