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
        if m.batch_gdn {
            return Err("Batch GDN requires batch profiles".into());
        }
        return Ok(());
    }
    let t = config.text();
    if m.batch_gdn {
        let pointers = m
            .buffers
            .iter()
            .find(|b| b.name == "BatchGdnPointers")
            .ok_or("Missing batch GDN address table")?;
        if pointers.dtype != crate::artifact::Dtype::U64
            || pointers.shape != [t.num_hidden_layers, 128, 3]
            || pointers.access != crate::artifact::Access::ReadWrite
        {
            return Err("Invalid batch GDN address table contract".into());
        }
        for &rows in &m.batch_profiles {
            builder.text(
                &format!("batch_gdn_m{rows}"),
                &Recipe {
                    begin: &[],
                    gdn: &[K(0), K(1)],
                    attention: &[],
                    end: &[],
                },
                rows,
            )?;
        }
    }
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
    for profile in &m.prefill_batch_profiles {
        if profile.tokens > m.chunk_tokens {
            return Err("Joint prefill exceeds shared workspace capacity".into());
        }
        let (gdn, attention): (&[Step], &[Step]) = match profile.kind.as_str() {
            "chunk_lut4" => (
                &[
                    K(0),
                    K(1),
                    K(2),
                    K(3),
                    K(4),
                    K(12),
                    K(13),
                    K(14),
                    K(15),
                    K(16),
                    K(17),
                    K(18),
                ],
                &[
                    K(0),
                    K(1),
                    K(2),
                    K(6),
                    K(7),
                    K(8),
                    K(9),
                    K(10),
                    K(11),
                    K(12),
                ],
            ),
            "chunk_expanded" => (
                &[
                    K(0),
                    K(1),
                    K(2),
                    K(3),
                    K(4),
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
                &[
                    K(0),
                    K(1),
                    K(2),
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
            ),
            _ => return Err("Joint prefill requires a chunk projection recipe".into()),
        };
        let program = format!("prefill_batch_m{}", profile.tokens);
        let mut ops = vec![];
        for (layer, kind) in t.layer_types.iter().enumerate() {
            ops.extend(builder.emit(
                if kind == "linear_attention" {
                    gdn
                } else {
                    attention
                },
                &program,
                &format!("layer{layer}"),
                profile.tokens,
                layer,
            )?);
        }
        builder.programs.insert(program, ops);
    }
    m.batch_layout = Some(BatchLayout {
        layers: t.layer_types.clone(),
        row_strides: strides,
        profiles: profiles
            .iter()
            .map(|p| (p.tokens, p.kind.clone()))
            .collect(),
        hidden: t.hidden_size,
        small_mixed_shapes: profiles
            .iter()
            .filter(|p| matches!(p.kind.as_str(), "sequence" | "recurrent"))
            .map(|p| p.tokens)
            .collect(),
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
        launch: None,
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
    let small_rows = m
        .batch_profiles
        .iter()
        .copied()
        .filter(|&n| n >= total)
        .min();
    let long_profile = if small_rows.is_none() {
        Some(
            m.prefill_batch_profiles
                .iter()
                .filter(|p| p.tokens >= total)
                .min_by_key(|p| p.tokens)
                .ok_or("Batch exceeds compiled row capacity")?,
        )
    } else {
        None
    };
    let dynamic = !m.dynamic_batch_kernels.is_empty();
    let rows = if dynamic && (2..=128).contains(&total) {
        total
    } else {
        small_rows.unwrap_or_else(|| long_profile.unwrap().tokens)
    };
    let source_rows = if m.batch_profiles.contains(&rows) {
        rows
    } else {
        128
    };
    let head_rows = m
        .batch_profiles
        .iter()
        .copied()
        .filter(|&n| n >= segments.len())
        .min()
        .ok_or("Missing batch head capacity")?;
    let head_rows = if dynamic && segments.len() >= 2 {
        segments.len()
    } else {
        head_rows
    };
    let head_source_rows = if m.batch_profiles.contains(&head_rows) {
        head_rows
    } else {
        128
    };
    let batch = if long_profile.is_some() {
        format!("prefill_batch_m{rows}")
    } else {
        format!("batch_m{source_rows}")
    };
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
        let long = long_profile.is_some();
        let shared_prefix = if is_gdn {
            if long { 5 } else { 4 }
        } else if long {
            3
        } else {
            2
        };
        let shared_suffix = if long {
            if is_gdn { 12 } else { 6 }
        } else {
            shared_prefix
        };
        let shared_end = if let Some(profile) = long_profile {
            match (is_gdn, profile.kind.as_str()) {
                (true, "chunk_lut4") => 19,
                (true, _) => 21,
                (false, "chunk_lut4") => 13,
                (false, _) => 15,
            }
        } else if is_gdn {
            12
        } else {
            9
        };
        for slot in 0..shared_prefix {
            out.push(kernel(
                format!("{batch}/layer{layer}/k{slot}"),
                None,
                &empty,
            ));
        }
        if rows > total {
            let destination = if long && is_gdn { "Y" } else { "MixerIn" };
            let stride = layout.row_strides[destination];
            let pad = BTreeMap::from([(
                destination.into(),
                BufferView {
                    buffer: destination.into(),
                    offset: total * stride,
                },
            )]);
            out.push(invocation(
                Operation::Zero {
                    destination: destination.into(),
                    bytes: (rows - total) * stride,
                },
                None,
                &pad,
            ));
        }
        // Decode rows can share one recurrence even alongside a prompt chunk.
        // Prompt rows are null in the table; their causal mixer runs afterward
        // using its own scratch and writes only that segment's Y rows.
        let batch_gdn = !long && m.batch_gdn && is_gdn && segments.iter().any(|s| s.tokens == 1);
        if batch_gdn {
            let views = BTreeMap::from([(
                "BatchGdnPointers".into(),
                BufferView {
                    buffer: "BatchGdnPointers".into(),
                    offset: layer * 128 * 3 * 8,
                },
            )]);
            for slot in 0..2 {
                out.push(kernel(
                    format!("batch_gdn_m{source_rows}/layer{layer}/k{slot}"),
                    None,
                    &views,
                ));
            }
        }
        for (s, (program, views)) in segments.iter().zip(&starts) {
            if batch_gdn && s.tokens == 1 {
                continue;
            }
            let sequence_kind = layout.profiles.get(&s.tokens).map(String::as_str);
            let ops = section(m, program, &format!("layer{layer}"));
            let selected: &[usize] = if is_gdn {
                if sequence_kind == Some("sequence") {
                    &[4, 5, 6, 7]
                } else if s.tokens == 1 || sequence_kind == Some("recurrent") {
                    &[4, 5]
                } else if long {
                    &[5, 6, 7, 8, 9, 10, 11]
                } else {
                    return Err("Chunk prefill cannot use the small-row batch mixer".into());
                }
            } else if sequence_kind == Some("recurrent") {
                &[2, 3]
            } else if matches!(sequence_kind, Some("chunk_lut4" | "chunk_expanded")) {
                &[3, 4, 5]
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
                    && ((sequence_kind == Some("sequence") && i == 5)
                        || (matches!(sequence_kind, Some("chunk_lut4" | "chunk_expanded"))
                            && i == 5)
                        || ((s.tokens == 1 || sequence_kind == Some("recurrent")) && i == 4))
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
                if is_gdn
                    && i == 11
                    && matches!(sequence_kind, Some("chunk_lut4" | "chunk_expanded"))
                {
                    out.push(invocation(
                        Operation::Copy {
                            source: "Sout".into(),
                            destination: format!("L{layer}_State"),
                            bytes: m
                                .buffers
                                .iter()
                                .find(|b| b.name == format!("L{layer}_State"))
                                .ok_or("Missing recurrent state")?
                                .bytes()?,
                        },
                        Some(s.slot),
                        views,
                    ));
                }
            }
        }
        for slot in shared_suffix..shared_end {
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
    out.push(kernel(
        format!("batch_m{head_source_rows}/end/k0"),
        None,
        &empty,
    ));
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
    for invocation in &mut out {
        if let Operation::Kernel { name } = &invocation.operation {
            let count = if name == "batch_m128/end/k0" && head_source_rows != head_rows {
                Some(head_rows)
            } else if source_rows != rows
                && (name.starts_with("batch_m128/layer")
                    || name.starts_with("batch_gdn_m128/layer"))
            {
                Some(rows)
            } else {
                None
            };
            if let Some(count) = count {
                invocation.launch = Some(
                    m.dynamic_batch_kernels
                        .get(name)
                        .ok_or("Missing dynamic launch contract")?
                        .launch(count)?,
                );
            }
        }
    }
    Ok(out)
}

/// GPU pointer table ABI: layer, token row, (state, history, position).
/// Prompt chunks, padding and attention-only layers stay null. A decoder after
/// a prompt chunk occupies its packed token row, not its request ordinal.
pub(crate) fn state_bindings(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Vec<Option<(usize, String)>> {
    let layout = m.batch_layout.as_ref().expect("validated batch layout");
    let mut bindings = vec![None; layout.layers.len() * 128 * 3];
    for (layer, kind) in layout.layers.iter().enumerate() {
        if kind != "linear_attention" {
            continue;
        }
        let mut row = 0;
        for segment in segments {
            if segment.tokens == 1 {
                for (column, name) in [
                    format!("L{layer}_State"),
                    format!("L{layer}_History"),
                    m.position.clone(),
                ]
                .into_iter()
                .enumerate()
                {
                    bindings[(layer * 128 + row) * 3 + column] = Some((segment.slot, name));
                }
            }
            row += segment.tokens;
        }
    }
    bindings
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
            profiles: BTreeMap::from([(2, "sequence".into()), (32, "recurrent".into())]),
            hidden: 8,
            small_mixed_shapes: vec![2, 32],
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
    fn symbolic_rows_compile_exact_launches_and_preserve_common_profiles() {
        use crate::operators::dynamic::{DynamicBatchKernel, RowExpression};
        let mut m = manifest();
        for (section, count) in [("layer0", 12), ("layer1", 9), ("end", 1)] {
            for slot in 0..count {
                let name = format!("batch_m128/{section}/k{slot}");
                m.dynamic_batch_kernels.insert(
                    name.clone(),
                    DynamicBatchKernel {
                        name,
                        grid: [
                            RowExpression::Rows,
                            RowExpression::Constant { value: 1 },
                            RowExpression::Constant { value: 1 },
                        ],
                        arguments: vec![],
                    },
                );
            }
        }
        for count in [3, 5, 6, 7, 9, 15, 31, 33, 63, 127] {
            let segments: Vec<_> = (0..count)
                .map(|slot| BatchSegment { slot, tokens: 1 })
                .collect();
            let p = plan(&m, &segments).unwrap();
            assert!(!p.iter().any(|i| matches!(&i.operation, Operation::Zero { destination, .. }
                if destination == "Hidden" || destination == "MixerIn" || destination == "BatchHeadHidden")));
            let launches: Vec<_> = p.iter().filter_map(|i| i.launch.as_ref()).collect();
            assert_eq!(launches.len(), 22);
            assert!(launches.iter().all(|launch| launch.grid[0] == count as u32));
        }
        let p = plan(
            &m,
            &(0..4)
                .map(|slot| BatchSegment { slot, tokens: 1 })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(p.iter().all(|i| i.launch.is_none()));
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
        assert!(
            p.iter()
                .filter_map(|i| i.launch.as_ref())
                .all(|launch| launch.grid[0] == 33)
        );
        assert!(p.iter().any(
            |i| matches!(&i.operation, Operation::Kernel { name } if name == "batch_m2/end/k0")
                && i.launch.is_none()
        ));
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

    #[test]
    fn dense_joint_prefill_shares_projections_but_keeps_chunk_state_private() {
        let mut m = manifest();
        m.prefill_batch_profiles = vec![PrefillProfile {
            tokens: 2048,
            kind: "chunk_expanded".into(),
        }];
        m.batch_layout
            .as_mut()
            .unwrap()
            .profiles
            .insert(512, "chunk_lut4".into());
        m.batch_layout
            .as_mut()
            .unwrap()
            .row_strides
            .insert("Y".into(), 12);
        m.buffers.push(
            serde_json::from_value(serde_json::json!({
                "name":"L0_State","dtype":"f32","shape":[20],"layout":"contiguous",
                "alignment":256,"access":"read_write","data":null
            }))
            .unwrap(),
        );
        for (section, count) in [("begin", 2), ("layer0", 19), ("layer1", 13), ("end", 1)] {
            m.programs
                .entry("prefill_m512".into())
                .or_default()
                .extend((0..count).map(|i| Operation::Kernel {
                    name: format!("prefill_m512/{section}/k{i}"),
                }));
        }
        m.programs.insert(
            "head_m512".into(),
            vec![Operation::Kernel {
                name: "head_m512/body/k0".into(),
            }],
        );
        let p = plan(
            &m,
            &[
                BatchSegment {
                    slot: 7,
                    tokens: 512,
                },
                BatchSegment {
                    slot: 2,
                    tokens: 512,
                },
            ],
        )
        .unwrap();
        assert!(p.iter().any(|i| i.sequence.is_none()
            && matches!(&i.operation,
            Operation::Kernel { name } if name == "prefill_batch_m2048/layer0/k17")));
        for slot in [7, 2] {
            let conv = p
                .iter()
                .find(|i| {
                    i.sequence == Some(slot)
                        && matches!(&i.operation,
                Operation::Kernel { name } if name == "prefill_m512/layer0/k5")
                })
                .unwrap();
            assert_eq!(conv.views["QKV"].offset, usize::from(slot == 2) * 512 * 20);
            assert!(
                !conv.views.contains_key("Q"),
                "Head-major scratch must not get a token-row offset"
            );
            assert!(p.iter().any(|i| i.sequence == Some(slot) && matches!(&i.operation,
                Operation::Copy { source, destination, bytes:80 } if source == "Sout" && destination == "L0_State")));
            assert!(p.iter().any(|i| i.sequence == Some(slot)
                && matches!(&i.operation,
                Operation::Kernel { name } if name == "prefill_m512/layer1/k5")));
        }
        assert!(p.iter().any(|i| matches!(&i.operation,
            Operation::Zero { destination, bytes:12288 } if destination == "Y")
            && i.views["Y"].offset == 12288));
        assert!(!p.iter().any(|i| matches!(&i.operation,
            Operation::Kernel { name } if name.starts_with("batch_gdn_"))));
    }
    #[test]
    fn batched_recurrence_isolates_decode_rows_in_mixed_work() {
        let mut m = manifest();
        m.batch_gdn = true;
        let segments = [
            BatchSegment { slot: 9, tokens: 1 },
            BatchSegment { slot: 3, tokens: 1 },
        ];
        let p = plan(&m, &segments).unwrap();
        let mixers: Vec<_> = p
            .iter()
            .filter(|i| {
                matches!(&i.operation,
            Operation::Kernel { name } if name.starts_with("batch_gdn_m2/"))
            })
            .collect();
        assert_eq!(mixers.len(), 2);
        assert!(
            mixers
                .iter()
                .all(|i| i.sequence.is_none() && i.views["BatchGdnPointers"].offset == 0)
        );
        assert!(!p.iter().any(|i| matches!(&i.operation,
            Operation::Copy { destination, .. } if destination.ends_with("_History"))));
        let table = state_bindings(&m, &segments);
        assert_eq!(table.len(), 2 * 128 * 3);
        assert_eq!(table[0], Some((9, "L0_State".into())));
        assert_eq!(table[3], Some((3, "L0_State".into())));
        assert_eq!(table[5], Some((3, "Step".into())));
        assert!(table[6..].iter().all(Option::is_none));
        let mixed = plan(&m, &[segments[0], BatchSegment { slot: 3, tokens: 2 }]).unwrap();
        assert!(mixed.iter().any(|i| matches!(&i.operation,
            Operation::Kernel { name } if name.starts_with("batch_gdn_"))));
        assert!(mixed.iter().any(|i| matches!(&i.operation,
            Operation::Copy { destination, .. } if destination.ends_with("_History"))));
        let segments = [
            BatchSegment {
                slot: 9,
                tokens: 32,
            },
            BatchSegment { slot: 3, tokens: 1 },
            BatchSegment { slot: 7, tokens: 1 },
        ];
        let table = state_bindings(&m, &segments);
        assert!(table[..32 * 3].iter().all(Option::is_none));
        assert_eq!(table[32 * 3], Some((3, "L0_State".into())));
        assert_eq!(table[33 * 3 + 2], Some((7, "Step".into())));
        assert!(table[34 * 3..].iter().all(Option::is_none));
        let p = plan(&m, &segments).unwrap();
        assert!(p.iter().any(|i| matches!(&i.operation,
            Operation::Kernel { name } if name.starts_with("batch_gdn_m64/"))));
        assert!(!p.iter().any(|i| i.sequence == Some(3)
            && matches!(&i.operation,
            Operation::Kernel { name } if name.ends_with("layer0/k5"))));
        assert!(p.iter().any(|i| i.sequence == Some(9)
            && matches!(&i.operation,
            Operation::Kernel { name } if name == "prefill_m32/layer0/k5")));
    }
}
