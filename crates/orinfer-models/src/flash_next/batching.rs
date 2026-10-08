//! Shared dense/MoE work with request-private recurrence and sparse attention.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::architecture::BatchSegment;
use std::collections::{BTreeMap, BTreeSet};

const PROFILES: [usize; 7] = [2, 4, 8, 16, 32, 64, 128];
const POINTERS: &str = "FlashBatchPointers";

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
    let columns = layout.state_columns.clone();
    if !columns.is_empty() {
        if columns.windows(2).any(|p| p[0] >= p[1])
            || m.buffers
                .iter()
                .find(|b| b.name == POINTERS)
                .is_none_or(|b| {
                    b.dtype != crate::artifact::Dtype::U64
                        || b.shape != [128, 48, columns.len()]
                        || b.data.is_some()
                })
            || columns.iter().any(|name| {
                !(0..48).any(|layer| {
                    let target = private_role(name, layer);
                    m.buffers
                        .iter()
                        .any(|b| b.name == target && b.data.is_none())
                })
            })
        {
            return Err("Invalid Flash private address-table layout".into());
        }
        m.state_pointer_table = Some(crate::model::StatePointerTable {
            buffer: POINTERS.into(),
            max_rows: 128,
        });
    } else if m.buffers.iter().any(|b| b.name == POINTERS) {
        return Err("Flash private address table needs column identities".into());
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
        if !columns.is_empty() {
            for layer in 0..48 {
                let prefix = format!("flash_private_m{rows}");
                m.programs.insert(
                    format!("{prefix}/layer{layer}"),
                    super::plan::section(
                        &prefix,
                        &format!("layer{layer}"),
                        if layer % 4 == 3 { 19 } else { 3 },
                        None,
                    ),
                );
            }
        }
    }
    for rows in [4, 8, 16, 32, 64, 128] {
        let prefix = format!("flash_dynamic_m{rows}");
        if m.dynamic_batch_kernels
            .contains_key(&format!("{prefix}/begin/k0"))
        {
            if columns.is_empty() {
                return Err("Dynamic Flash batches require parallel private mixers".into());
            }
            for (section, count) in stages(rows) {
                if section.ends_with("/private") {
                    continue;
                }
                let program = format!("{prefix}/{section}");
                let ops = super::plan::section(&prefix, &section, count, None);
                for op in &ops {
                    if let Operation::Kernel { name } = op
                        && !m.dynamic_batch_kernels.contains_key(name)
                    {
                        return Err(format!("Missing dynamic Flash launch {name}"));
                    }
                }
                m.programs.insert(program, ops);
            }
            for layer in 0..48 {
                for op in &m.programs[&format!("flash_private_m{rows}/layer{layer}")] {
                    if let Operation::Kernel { name } = op
                        && !m.dynamic_batch_kernels.contains_key(name)
                    {
                        return Err(format!("Missing dynamic private launch {name}"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn private_role(column: &str, layer: usize) -> String {
    column.strip_prefix("State_").map_or_else(
        || column.to_owned(),
        |suffix| format!("State_{layer}_{suffix}"),
    )
}

pub(super) fn state_bindings(
    m: &Manifest,
    segments: &[BatchSegment],
) -> Result<orinfer_model_sdk::abi::StateBindings> {
    let Some(layout) = &m.batch_layout else {
        return Ok(vec![]);
    };
    if layout.state_columns.is_empty() {
        return Ok(vec![]);
    }
    if segments.len() > 128
        || segments.iter().any(|s| s.tokens != 1)
        || segments
            .iter()
            .map(|s| s.slot)
            .collect::<BTreeSet<_>>()
            .len()
            != segments.len()
    {
        return Err("Flash private table requires distinct single-token requests".into());
    }
    if segments.len() <= 1 {
        return Ok(vec![]);
    }
    let names: BTreeSet<_> = m.buffers.iter().map(|b| b.name.as_str()).collect();
    let roles: Vec<_> = (0..48)
        .flat_map(|layer| {
            let names = &names;
            layout.state_columns.iter().map(move |column| {
                let name = private_role(column, layer);
                names.contains(name.as_str()).then_some(name)
            })
        })
        .collect();
    let rows = segments.len().next_power_of_two();
    let mut result = Vec::with_capacity(rows * roles.len());
    for lane in 0..rows {
        if let Some(segment) = segments.get(lane) {
            result.extend(
                roles
                    .iter()
                    .map(|name| name.as_ref().map(|n| (segment.slot, n.clone()))),
            );
        } else {
            result.extend(std::iter::repeat_n(None, roles.len()));
        }
    }
    Ok(result)
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
    let dynamic_capacity = segments.len().next_power_of_two();
    let dynamic = !PROFILES.contains(&segments.len())
        && m.dynamic_batch_kernels
            .contains_key(&format!("flash_dynamic_m{dynamic_capacity}/begin/k0"));
    let rows = if dynamic {
        dynamic_capacity
    } else {
        *m.batch_profiles
            .iter()
            .find(|&&r| r >= segments.len())
            .ok_or("Missing Flash batch profile")?
    };
    let layout = m.batch_layout.as_ref().ok_or("Missing Flash row layout")?;
    let prefix = format!(
        "flash_{}_m{rows}",
        if dynamic { "dynamic" } else { "batch" }
    );
    let fixed_prefix = format!("flash_batch_m{rows}");
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
    if !dynamic && rows > segments.len() {
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
        let private = &m.programs[&format!("{fixed_prefix}/layer{layer}/private")];
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
        if !layout.state_columns.is_empty() {
            append(
                &mut out,
                m,
                &format!("flash_private_m{rows}/layer{layer}"),
                None,
                &empty,
            )?;
        } else {
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
    if dynamic {
        for op in &mut out {
            if let Operation::Kernel { name } = &op.operation
                && (name.starts_with("flash_dynamic_") || name.starts_with("flash_private_"))
            {
                op.launch = Some(
                    m.dynamic_batch_kernels
                        .get(name)
                        .ok_or("Missing dynamic Flash invocation contract")?
                        .launch(segments.len())?,
                );
            }
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
            state_columns: vec![],
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
    #[test]
    fn parallel_private_stages_keep_state_rows_and_padding_separate() {
        let mut m = manifest();
        let columns = ["M1_HistoryOut", "Position", "State_conv", "State_gdn"];
        m.batch_layout.as_mut().unwrap().state_columns =
            columns.iter().map(|n| (*n).into()).collect();
        for name in ["M1_HistoryOut", "Position"] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":name,"dtype":"u8","shape":[4],"layout":"native_contiguous",
                    "alignment":256,"access":"read_write","data":null
                }))
                .unwrap(),
            );
        }
        for layer in (0..48).filter(|l| l % 4 != 3) {
            for suffix in ["conv", "gdn"] {
                m.buffers.push(serde_json::from_value(serde_json::json!({
                    "name":format!("State_{layer}_{suffix}"),"dtype":"u8","shape":[4],
                    "layout":"native_contiguous","alignment":256,"access":"read_write","data":null
                })).unwrap());
            }
        }
        m.buffers.push(
            serde_json::from_value(serde_json::json!({
            "name":POINTERS,"dtype":"u64","shape":[128,48,4],"layout":"native_contiguous",
                "alignment":256,"access":"read_write","data":null
            }))
            .unwrap(),
        );
        register(&mut m).unwrap();
        assert_eq!(m.state_pointer_table.as_ref().unwrap().max_rows, 128);
        for count in [2, 3, 8, 15, 128] {
            let segments: Vec<_> = (0..count)
                .map(|lane| BatchSegment {
                    slot: 129 - lane,
                    tokens: 1,
                })
                .collect();
            let bindings = state_bindings(&m, &segments).unwrap();
            assert_eq!(bindings.len(), 48 * count.next_power_of_two() * 4);
            for layer in 0..48 {
                for (lane, s) in segments.iter().enumerate() {
                    let start = (lane * 48 + layer) * 4;
                    assert_eq!(bindings[start], Some((s.slot, "M1_HistoryOut".into())));
                    assert_eq!(bindings[start + 1], Some((s.slot, "Position".into())));
                    assert_eq!(
                        bindings[start + 2],
                        (layer % 4 != 3).then(|| (s.slot, format!("State_{layer}_conv")))
                    );
                }
            }
            assert!(bindings[count * 48 * 4..].iter().all(Option::is_none));
            let p = plan(&m, &segments).unwrap();
            let private: Vec<_> = p
                .iter()
                .filter(|i| {
                    matches!(&i.operation,
                Operation::Kernel{name} if name.starts_with("flash_private_"))
                })
                .collect();
            assert_eq!(private.len(), 36 * 3 + 12 * 19);
            assert!(
                private
                    .iter()
                    .all(|i| i.sequence.is_none() && i.views.is_empty())
            );
            assert!(!p.iter().any(|i| matches!(&i.operation,
                Operation::Copy{destination,..} if destination=="State_0_conv")));
        }
        m.batch_layout.as_mut().unwrap().state_columns.swap(0, 1);
        assert!(register(&mut m).is_err());
    }
    #[test]
    fn dynamic_rows_keep_fixed_profiles_and_never_create_padding_requests() {
        use crate::operators::dynamic::{DynamicBatchKernel, RowExpression};
        let mut m = manifest();
        m.batch_layout.as_mut().unwrap().state_columns = vec!["Position".into()];
        for (name, dtype, shape) in [
            ("Position", "i32", vec![1]),
            (POINTERS, "u64", vec![128, 48, 1]),
        ] {
            m.buffers.push(
                serde_json::from_value(serde_json::json!({
                    "name":name,"dtype":dtype,"shape":shape,"layout":"native_contiguous",
                    "alignment":256,"access":"read_write","data":null
                }))
                .unwrap(),
            );
        }
        register(&mut m).unwrap();
        for capacity in [4, 8, 16, 32, 64, 128] {
            let prefix = format!("flash_dynamic_m{capacity}");
            let shared = stages(capacity)
                .into_iter()
                .filter(|(s, _)| !s.ends_with("/private"))
                .flat_map(|(section, count)| {
                    super::super::plan::section(&prefix, &section, count, None)
                });
            let private = (0..48).flat_map(|layer| {
                m.programs[&format!("flash_private_m{capacity}/layer{layer}")].clone()
            });
            for op in shared.chain(private) {
                if let Operation::Kernel { name } = op {
                    m.dynamic_batch_kernels.insert(
                        name.clone(),
                        DynamicBatchKernel {
                            name,
                            capacity,
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
        }
        register(&mut m).unwrap();
        for count in [
            2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 127, 128,
        ] {
            let segments: Vec<_> = (0..count)
                .map(|slot| BatchSegment { slot, tokens: 1 })
                .collect();
            let p = plan(&m, &segments).unwrap();
            let fixed = PROFILES.contains(&count);
            assert!(
                !p.iter()
                    .any(|i| matches!(i.operation, Operation::Zero { .. }))
            );
            let overridden: Vec<_> = p.iter().filter_map(|i| i.launch.as_ref()).collect();
            if fixed {
                assert!(overridden.is_empty());
            } else {
                assert!(!overridden.is_empty());
                assert!(overridden.iter().all(|l| l.grid[0] == count as u32));
            }
            assert_eq!(p.iter().filter(|i| matches!(&i.operation, Operation::Copy { destination, .. } if destination == "Logits")).count(), count);
        }
        m.dynamic_batch_kernels.remove("flash_dynamic_m128/head/k0");
        assert!(register(&mut m).is_err());
    }
}
