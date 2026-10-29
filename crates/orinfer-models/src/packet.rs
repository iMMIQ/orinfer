//! Common row invocation binding; model programs retain their arithmetic order.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::{Manifest, Operation},
};
use orinfer_model_sdk::{abi::StateBindings, architecture::BatchSegment};
use std::collections::{BTreeMap, BTreeSet};

/// Publish wrapped operands through one common program/export boundary.
pub(crate) fn register_indirect(m: &mut Manifest) {
    let Some(table) = m.packet_table() else {
        return;
    };
    let operations = table
        .kernels
        .values()
        .chain(table.copies.iter().map(|c| &c.kernel))
        .map(|name| Operation::Kernel { name: name.clone() })
        .collect();
    m.programs.insert("mtp_packet_indirect".into(), operations);
    if let Some(spec) = &mut m.mtp
        && !spec
            .batch_programs
            .iter()
            .any(|p| p == "mtp_packet_indirect")
    {
        spec.batch_programs.push("mtp_packet_indirect".into());
    }
}

/// Keep packet addresses stable while rebinding each owner in data.
/// Existing causal/commit table bindings precede the boundary table's bindings.
pub(crate) fn indirect(
    m: &Manifest,
    segments: &[BatchSegment],
    contracts: &[crate::artifact::Kernel],
    (mut plan, mut bindings): orinfer_model_sdk::abi::BatchPlan,
) -> Result<orinfer_model_sdk::abi::BatchPlan> {
    let Some(table) = m.packet_table() else {
        return Ok((plan, bindings));
    };
    let stride = table
        .columns
        .len()
        .checked_mul(8)
        .ok_or("Packet table stride overflow")?;
    let owners: BTreeMap<_, _> = segments
        .iter()
        .enumerate()
        .map(|(i, s)| (s.slot, i))
        .collect();
    let kernels: BTreeMap<_, _> = contracts.iter().map(|k| (k.name.as_str(), k)).collect();
    let arguments = |name: &str| -> Result<Vec<crate::artifact::Argument>> {
        if let Some(kernel) = kernels.get(name) {
            Ok(kernel.args.clone())
        } else {
            crate::bindings::kernel_arguments(name)
        }
    };
    for call in &mut plan {
        let Some(owner) = call.sequence else { continue };
        if call.views.iter().any(|(name, view)| {
            (table.columns.contains(name) || table.columns.contains(&view.buffer))
                && (name != &view.buffer || view.offset != 0)
        }) {
            return Err("Private packet views require an explicit indirect offset".into());
        }
        let lane = *owners
            .get(&owner)
            .ok_or("Packet invocation owner is absent")?;
        let mut indirect = false;
        match &call.operation {
            Operation::Kernel { name } => {
                if let Some(target) = table.kernels.get(name) {
                    let scalar_indices = |args: &[crate::artifact::Argument]| {
                        args.iter()
                            .enumerate()
                            .filter_map(|(i, a)| {
                                matches!(a, crate::artifact::Argument::I32 { .. }).then_some(i)
                            })
                            .collect::<Vec<_>>()
                    };
                    let mapping = if let Some(mapping) = table.scalar_arguments.get(name) {
                        mapping.clone()
                    } else {
                        let before = scalar_indices(&arguments(name)?);
                        let after = scalar_indices(&arguments(target)?);
                        if before.len() != after.len() {
                            return Err("Missing indirect scalar ABI mapping".into());
                        }
                        before.into_iter().zip(after).collect()
                    };
                    let target_arguments = arguments(target)?;
                    if !target_arguments.iter().any(|a| matches!(a,
                        crate::artifact::Argument::Buffer { name } | crate::artifact::Argument::BufferSlice { name, .. } if name == &table.buffer))
                        || target_arguments.iter().any(|a| matches!(a,
                        crate::artifact::Argument::Buffer { name } | crate::artifact::Argument::BufferSlice { name, .. } if table.columns.contains(name)))
                        || mapping.iter().any(|(_,index)| !matches!(target_arguments.get(*index),Some(crate::artifact::Argument::I32{..})))
                    { return Err("Invalid indirect packet ABI".into()); }
                    if let Some(launch) = &mut call.launch {
                        for (index, _) in &mut launch.arguments {
                            *index = mapping
                                .iter()
                                .find(|(original, _)| original == index)
                                .map(|(_, target)| *target)
                                .ok_or("Unknown packet scalar")?;
                        }
                    }
                    call.operation = Operation::Kernel {
                        name: target.clone(),
                    };
                    indirect = true;
                } else {
                    if arguments(name)?.iter().any(|a| match a {
                        crate::artifact::Argument::Buffer { name }
                        | crate::artifact::Argument::BufferSlice { name, .. } => {
                            let target = call
                                .views
                                .get(name)
                                .map_or(name.as_str(), |v| v.buffer.as_str());
                            table.columns.iter().any(|column| column == target)
                        }
                        _ => false,
                    }) {
                        return Err(format!("Missing indirect packet implementation {name}"));
                    }
                }
            }
            Operation::Copy {
                source,
                destination,
                bytes,
            } => {
                if !table.columns.contains(source) && !table.columns.contains(destination) {
                    call.sequence = None;
                    continue;
                }
                let copy = table
                    .copies
                    .iter()
                    .find(|c| &c.source == source && &c.destination == destination)
                    .ok_or("Missing indirect packet copy")?;
                if copy.row_bytes == 0 || !bytes.is_multiple_of(copy.row_bytes) {
                    return Err("Unaligned indirect packet copy".into());
                }
                call.launch = Some(
                    m.dynamic_batch_kernels
                        .get(&copy.kernel)
                        .ok_or("Missing packet copy launch")?
                        .launch(bytes / copy.row_bytes)?,
                );
                call.operation = Operation::Kernel {
                    name: copy.kernel.clone(),
                };
                indirect = true;
            }
            Operation::Zero { .. } => {
                return Err("Private packet zero requires an indirect implementation".into());
            }
        }
        if indirect {
            call.views.insert(
                table.buffer.clone(),
                BufferView {
                    buffer: table.buffer.clone(),
                    offset: lane
                        .checked_mul(stride)
                        .ok_or("Packet table view overflow")?,
                },
            );
        }
        call.sequence = None;
    }
    let roles: Vec<_> = table.columns.iter().cloned().map(Some).collect();
    bindings.extend(private_bindings(segments, segments.len(), &roles));
    Ok((plan, bindings))
}

pub(crate) use orinfer_model_sdk::packet::Rows;

pub(crate) fn view(buffer: &str, row: usize, stride: usize) -> Result<BufferView> {
    Ok(BufferView {
        buffer: buffer.into(),
        offset: row
            .checked_mul(stride)
            .ok_or("Packet byte offset overflow")?,
    })
}

pub(crate) fn views<'a>(
    row: usize,
    strides: impl IntoIterator<Item = (&'a str, usize)>,
) -> Result<BTreeMap<String, BufferView>> {
    strides
        .into_iter()
        .map(|(name, stride)| Ok((name.into(), view(name, row, stride)?)))
        .collect()
}

/// Resolve each model's ordered role candidates; absent roles stay null.
pub(crate) fn layer_roles<S: AsRef<str>, I: IntoIterator<Item = String>>(
    m: &Manifest,
    layers: usize,
    columns: &[S],
    resolve: impl Fn(&str, usize) -> I,
) -> Vec<Option<String>> {
    let names: BTreeSet<_> = m.buffers.iter().map(|b| b.name.as_str()).collect();
    (0..layers)
        .flat_map(|layer| {
            let (names, resolve) = (&names, &resolve);
            columns.iter().map(move |column| {
                resolve(column.as_ref(), layer)
                    .into_iter()
                    .find(|name| names.contains(name.as_str()))
            })
        })
        .collect()
}

pub(crate) fn invocation(
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

pub(crate) fn private_bindings(
    segments: &[BatchSegment],
    capacity: usize,
    roles: &[Option<String>],
) -> StateBindings {
    (0..capacity)
        .flat_map(|lane| {
            roles.iter().map(move |role| {
                segments
                    .get(lane)
                    .and_then(|s| role.as_ref().map(|name| (s.slot, name.clone())))
            })
        })
        .collect()
}

pub(crate) fn append(
    m: &Manifest,
    out: &mut Vec<Invocation>,
    program: &str,
    count: usize,
    sequence: Option<usize>,
    views: &BTreeMap<String, BufferView>,
) -> Result<()> {
    let rows = m.row_execution.as_ref().ok_or("Missing row execution")?;
    let operations = m.programs.get(program).ok_or("Missing row program")?;
    let mut calls = rows.specialize(program, operations, &m.dynamic_batch_kernels, count)?;
    for call in &mut calls {
        call.sequence = sequence;
        call.views = views.clone();
    }
    out.extend(calls);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_rows_and_heads_keep_independent_owner_order() {
        let segments = [
            BatchSegment { slot: 9, tokens: 1 },
            BatchSegment { slot: 2, tokens: 7 },
            BatchSegment { slot: 7, tokens: 3 },
        ];
        let rows = Rows::packet(&segments, 128, 1024).unwrap();
        let heads = Rows::new([1, 7, 1], 128).unwrap();
        assert_eq!(rows.starts, [0, 1, 8]);
        assert_eq!(rows.total, 11);
        assert_eq!(heads.starts, [0, 1, 8]);
        assert_eq!(heads.total, 9);
        let tail = views(rows.starts[2], [("Hidden", 5120), ("Scale", 2)]).unwrap();
        assert_eq!(tail["Hidden"].offset, 40960);
        assert_eq!(tail["Scale"].offset, 16);
        let roles = [Some("KV".to_owned()), None];
        let forward = private_bindings(&segments, 4, &roles);
        assert_eq!(forward[0], Some((9, "KV".into())));
        assert_eq!(forward[2], Some((2, "KV".into())));
        assert_eq!(&forward[6..], &[None, None]);
        let reversed = private_bindings(
            &segments[..2].iter().rev().cloned().collect::<Vec<_>>(),
            4,
            &roles,
        );
        assert_eq!(reversed[0], Some((2, "KV".into())));
        assert_eq!(&reversed[4..], &[None, None, None, None]);
    }

    #[test]
    fn packet_bounds_fail_before_creating_views() {
        assert!(Rows::new([], 128).is_err());
        assert!(Rows::new([1, 0], 128).is_err());
        assert!(Rows::new([usize::MAX, 1], usize::MAX).is_err());
        assert!(Rows::new([127, 2], 128).is_err());
        assert!(view("Hidden", usize::MAX, 2).is_err());
        let segments = [
            BatchSegment { slot: 3, tokens: 1 },
            BatchSegment { slot: 3, tokens: 1 },
        ];
        assert!(Rows::packet(&segments, 128, 128).is_err());
        assert!(Rows::packet(&segments[..1], 0, 128).is_err());
    }
}
