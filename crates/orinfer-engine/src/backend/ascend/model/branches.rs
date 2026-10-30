//! Validate branch structure and actual physical read/write dependencies.
use super::{Instruction, SegmentOp};
use crate::{artifact::Result, backend::ascend::runtime::Buffer};

pub(super) fn validate_instructions(instructions: &[Instruction]) -> Result<()> {
    let mut stack = Vec::new();
    for instruction in instructions {
        match instruction {
            Instruction::Fork => {
                if stack.len() == 8 {
                    return Err("ACL model branch nesting exceeds eight streams".into());
                }
                stack.push(false);
            }
            Instruction::Switch => match stack.last_mut() {
                Some(state @ false) => *state = true,
                _ => return Err("ACL branch switch requires an open side branch".into()),
            },
            Instruction::Join => {
                if stack.pop() != Some(true) {
                    return Err("ACL branch join requires a switched branch".into());
                }
            }
            Instruction::Reduce { .. }
            | Instruction::Gather { .. }
            | Instruction::Exchange { .. }
                if stack.contains(&false) =>
            {
                return Err("HCCL collectives remain on the main branch".into());
            }
            Instruction::Embed { .. }
            | Instruction::LimitRows { .. }
            | Instruction::Lookup { .. }
            | Instruction::Route { .. }
            | Instruction::Snapshot { .. }
                if !stack.is_empty() =>
            {
                return Err("ACL branch crosses a CPU or snapshot boundary".into());
            }
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("ACL model contains an unjoined branch".into());
    }
    Ok(())
}

#[derive(Default)]
struct Frame {
    index: usize,
    main: bool,
    side_accesses: Vec<(Buffer, bool)>,
    main_accesses: Vec<(Buffer, bool)>,
}

pub(super) fn validate(ops: &[SegmentOp]) -> Result<Vec<usize>> {
    let mut depths = Vec::new();
    let mut stack: Vec<Frame> = Vec::new();
    for op in ops {
        match op {
            SegmentOp::Fork(index) => {
                if *index != depths.len() || stack.len() == 8 {
                    return Err("ACL Graph branch index or nesting differs".into());
                }
                depths.push(stack.len());
                stack.push(Frame {
                    index: *index,
                    ..Frame::default()
                });
                continue;
            }
            SegmentOp::Switch(index) => match stack.last_mut() {
                Some(frame) if frame.index == *index && !frame.main => frame.main = true,
                _ => return Err("ACL Graph switch differs from its fork".into()),
            },
            SegmentOp::Join(index) => {
                let frame = stack.pop().ok_or("ACL Graph join has no fork")?;
                if !frame.main
                    || frame.index != *index
                    || frame.side_accesses.is_empty()
                    || frame.main_accesses.is_empty()
                {
                    return Err("ACL Graph requires two nonempty, matched branches".into());
                }
                for (side, side_writes) in &frame.side_accesses {
                    for (main, main_writes) in &frame.main_accesses {
                        if (*side_writes || *main_writes) && side.overlaps(main) {
                            return Err(
                                "ACL Graph parallel branches have overlapping writable spans"
                                    .into(),
                            );
                        }
                    }
                }
            }
            _ => {
                if matches!(
                    op,
                    SegmentOp::Reduce { .. } | SegmentOp::Gather { .. } | SegmentOp::Exchange(_)
                ) && stack.iter().any(|frame| !frame.main)
                {
                    return Err("HCCL cannot run on an auxiliary branch".into());
                }
                let accesses: Vec<_> = match op {
                    SegmentOp::Kernel(kernel) => kernel
                        .accesses()
                        .map(|(buffer, writes)| (buffer.clone(), writes))
                        .collect(),
                    SegmentOp::Reduce {
                        source,
                        destination,
                        ..
                    } => vec![(source.clone(), false), (destination.clone(), true)],
                    SegmentOp::Gather {
                        source,
                        destination,
                        ..
                    } => vec![(source.clone(), false), (destination.clone(), true)],
                    SegmentOp::Exchange(exchange) => exchange.branch_accesses(),
                    SegmentOp::W4(plan) => plan.branch_accesses(),
                    _ => unreachable!(),
                };
                for frame in &mut stack {
                    if frame.main {
                        frame.main_accesses.extend(accesses.iter().cloned());
                    } else {
                        frame.side_accesses.extend(accesses.iter().cloned());
                    }
                }
            }
        }
    }
    if !stack.is_empty() {
        return Err("ACL Graph contains an unjoined branch".into());
    }
    Ok(depths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::{kernels::Kernel, tests::Fixture};

    #[test]
    fn reject_unjoined_branches_and_cpu_boundaries() {
        assert!(
            validate_instructions(&[Instruction::Fork, Instruction::Switch, Instruction::Join])
                .is_ok()
        );
        assert!(
            validate_instructions(&[
                Instruction::Fork,
                Instruction::Fork,
                Instruction::Switch,
                Instruction::Join,
                Instruction::Switch,
                Instruction::Join
            ])
            .is_ok()
        );
        for invalid in [
            vec![Instruction::Join],
            vec![Instruction::Switch],
            vec![Instruction::Fork],
            vec![Instruction::Fork, Instruction::Join],
            vec![Instruction::Fork, Instruction::Switch, Instruction::Switch],
            vec![
                Instruction::Fork,
                Instruction::LimitRows { rows: 128 },
                Instruction::Switch,
                Instruction::Join,
            ],
            vec![
                Instruction::Fork,
                Instruction::Snapshot {
                    name: "layer00".into(),
                    buffer: "x".into(),
                },
            ],
        ] {
            assert!(validate_instructions(&invalid).is_err());
        }
    }

    #[test]
    fn nested_branches_allow_shared_reads_and_reject_physical_view_aliases() {
        let mut fixture = Fixture::new();
        {
            let session = fixture.session();
            let input = session.allocate(4).unwrap();
            let arena = session.allocate(12).unwrap();
            let a = arena.slice(0, 4).unwrap();
            let b = arena.slice(4, 4).unwrap();
            let c = arena.slice(8, 4).unwrap();
            let overlap = arena.slice(2, 4).unwrap();
            let (path, _) = fixture.aot();
            let side = Kernel::load(&session, &path, &[input.clone(), a]).unwrap();
            let middle = Kernel::load(&session, &path, &[input.clone(), b]).unwrap();
            let main = Kernel::load(&session, &path, &[input.clone(), c.clone()]).unwrap();
            let race = Kernel::load(&session, &path, &[overlap, c]).unwrap();
            let valid = [
                SegmentOp::Fork(0),
                SegmentOp::Kernel(side.clone()),
                SegmentOp::Switch(0),
                SegmentOp::Fork(1),
                SegmentOp::Kernel(middle),
                SegmentOp::Switch(1),
                SegmentOp::Kernel(main),
                SegmentOp::Join(1),
                SegmentOp::Join(0),
            ];
            assert_eq!(validate(&valid).unwrap(), [0, 1]);
            let invalid = [
                SegmentOp::Fork(0),
                SegmentOp::Kernel(side),
                SegmentOp::Switch(0),
                SegmentOp::Kernel(race),
                SegmentOp::Join(0),
            ];
            assert!(
                validate(&invalid)
                    .unwrap_err()
                    .contains("overlapping writable")
            );
        }
        fixture.assert_clean();
    }
}
