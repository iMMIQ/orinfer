use super::{
    Element, Instruction, Packet, bounded_rows, branches, gather_element, validate_dispatch,
    validate_row_ranges,
};

#[test]
fn direct_segments_retain_parallel_resources_and_fence_partial_errors() {
    use super::{Segment, SegmentOp};
    use crate::backend::ascend::{
        graph::Branches,
        hccl::{Communicator, RootInfo},
        kernels::Kernel,
        tests::Fixture,
    };
    use std::rc::Rc;
    for fail_before_join in [false, true] {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let (path, _) = fixture.aot();
        let root =
            RootInfo::fixture(&session, &path.parent().unwrap().join("libacl-fixture.so")).unwrap();
        let comm = Communicator::join(&session, &root, 0).unwrap();
        let input = session.allocate(4).unwrap();
        let output = session.allocate(4).unwrap();
        let reduced = session.allocate(8).unwrap();
        session
            .upload(&input, 0, &[1, 2, 3, 4])
            .unwrap()
            .wait()
            .unwrap();
        let kernel = Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
        let mut ops = vec![
            SegmentOp::Fork(0),
            SegmentOp::Kernel(kernel),
            SegmentOp::Switch(0),
            SegmentOp::Reduce {
                source: input.clone(),
                destination: reduced.clone(),
                elements: 1,
                element: Element::Int32,
                scatter: false,
            },
        ];
        if fail_before_join {
            // The fixture fails a valid native collective after both branches queue work.
            ops.push(SegmentOp::Gather {
                source: input.clone(),
                destination: reduced.clone(),
                elements: 1,
                element: Element::Int32,
            });
        }
        ops.push(SegmentOp::Join(0));
        let depths = branches::validate(&ops).unwrap();
        let ops = Rc::new(ops);
        let parallel =
            Branches::new(&session, &depths, Box::new((Rc::clone(&ops), comm.clone()))).unwrap();
        let segment = Rc::new(Segment {
            ops,
            comm,
            parallel: Some(parallel),
        });
        if fail_before_join {
            fixture.failure(14);
        }
        let submitted = segment.enqueue(&session);
        if fail_before_join {
            assert!(submitted.is_err());
            // Segment remains cached/alive; its Drop cannot supply this fence.
            assert_eq!(fixture.pending_jobs(), 0);
        }
        drop((segment, input, reduced, root));
        if fail_before_join {
            // A native HCCL failure retires the context. The failed packet
            // keeps all native backing even after our explicit branch fence.
            assert_eq!(fixture.counter(8), 1);
            assert_eq!(session.allocated_bytes(), 16);
            drop((output, session));
            assert_eq!(fixture.counter(0), 3);
            assert_eq!(fixture.counter(4), 1);
            assert_eq!(fixture.counter(6), 0);
            continue;
        }
        assert_eq!(session.allocated_bytes(), 16);
        assert_eq!(fixture.counter(8), 1);
        submitted.unwrap().wait().unwrap();
        assert_eq!(fixture.counter(8), 0);
        assert_eq!(session.allocated_bytes(), 4);
        assert_eq!(session.download(&output, 0, 4).unwrap(), [1, 2, 3, 4]);
        drop((output, session));
        fixture.assert_clean();
    }
}

#[test]
fn owner_batch_ranges_share_dispatch_validation_from_one_through_capacity() {
    let short = packet(2048, 1, Some(511));
    let long = packet(2048, 512, None);
    validate_row_ranges(&[&short, &long], 1, Some(2048)).unwrap();
    for q in 1..=2048 {
        assert_eq!(
            [&short, &long]
                .iter()
                .filter(|p| p.min_rows <= q && q <= p.dispatch_max())
                .count(),
            1
        );
    }
    for invalid in [
        packet(2048, 513, None),
        packet(2048, 511, None),
        packet(2048, 512, Some(2047)),
        packet(1024, 512, None),
    ] {
        assert!(validate_row_ranges(&[&short, &invalid], 1, Some(2048)).is_err());
    }
}

#[test]
fn compacted_suffix_rows_keep_short_packets_and_reject_invalid_extents() {
    for actual in 1..=2048 {
        assert_eq!(bounded_rows(actual, 2048, 140).unwrap(), actual.min(140));
    }
    assert_eq!(bounded_rows(2048, 2048, 128).unwrap(), 128);
    for (actual, capacity, limit) in [
        (0, 2048, 128),
        (2049, 2048, 128),
        (17, 2048, 0),
        (17, 2048, 2049),
    ] {
        assert!(bounded_rows(actual, capacity, limit).is_err());
    }
    let instruction: Instruction = serde_json::from_value(serde_json::json!({
        "op": "limit_rows", "rows": 140
    }))
    .unwrap();
    assert!(matches!(instruction, Instruction::LimitRows { rows: 140 }));
}

fn packet(rows: usize, min_rows: usize, max_rows: Option<usize>) -> Packet {
    let mut coordinates = serde_json::Map::new();
    for name in [
        "base_cos",
        "base_sin",
        "long_cos",
        "long_sin",
        "source_cos",
        "source_sin",
        "slots",
        "c2_slots",
        "swa_indices",
        "c1_indices",
        "c2_indices",
        "c1_lengths",
        "c2_lengths",
        "ring",
        "premix",
    ] {
        coordinates.insert(name.into(), "test".into());
    }
    serde_json::from_value(serde_json::json!({
        "rows": rows, "min_rows": min_rows, "max_rows": max_rows,
        "coordinates": coordinates, "body": [], "head": [], "reset": [],
        "best_value": "v", "best_index": "i", "logits": null, "vocab_first": 0
    }))
    .unwrap()
}

#[test]
fn prefill_dispatch_keeps_capacity_and_covers_boundaries() {
    let legacy = vec![packet(1, 0, None), packet(2048, 2, None)];
    validate_dispatch(&legacy).unwrap();
    let packets = vec![
        packet(1, 0, None),
        packet(2048, 2, Some(511)),
        packet(2048, 512, None),
    ];
    validate_dispatch(&packets).unwrap();
    assert_eq!(packets[1].rows, 2048);
    assert_eq!(packets[1].prefill_plan().chunk_tokens, 511);
    assert_eq!(packets[2].prefill_plan().min_tokens, 512);
    for rows in 2..=2048 {
        assert_eq!(
            packets
                .iter()
                .filter(|p| p.min_rows > 0 && (p.min_rows..=p.dispatch_max()).contains(&rows))
                .count(),
            1
        );
    }
}

#[test]
fn prefill_dispatch_rejects_holes_overlap_and_invalid_capacity() {
    for (end, start) in [(511, 513), (512, 512), (0, 512), (2049, 2050)] {
        let packets = vec![
            packet(1, 0, None),
            packet(2048, 2, Some(end)),
            packet(2048, start, None),
        ];
        assert!(validate_dispatch(&packets).is_err());
    }
    assert!(validate_dispatch(&[]).is_err());
    assert!(validate_dispatch(&[packet(1, 0, None)]).is_err());
}

#[test]
fn collective_and_local_kernel_contracts_are_backward_compatible() {
    assert!(matches!(
        gather_element(false, false),
        Ok(Element::Bfloat16)
    ));
    assert!(matches!(gather_element(true, false), Ok(Element::Float32)));
    assert!(matches!(gather_element(false, true), Ok(Element::Int32)));
    assert!(gather_element(true, true).is_err());
    let old: Instruction = serde_json::from_value(serde_json::json!({
        "op": "gather", "source": "a", "destination": "b", "elements": 16, "fp32": false
    }))
    .unwrap();
    assert!(matches!(
        old,
        Instruction::Gather {
            int32: false,
            fixed_rows: false,
            ..
        }
    ));
    let old: Instruction = serde_json::from_value(serde_json::json!({
        "op": "reduce", "source": "a", "destination": "b", "elements": 16, "fp32": false
    }))
    .unwrap();
    assert!(matches!(
        old,
        Instruction::Reduce {
            scatter: false,
            fixed_rows: false,
            ..
        }
    ));
    let scatter: Instruction = serde_json::from_value(serde_json::json!({
        "op": "reduce", "source": "a", "destination": "b", "elements": 16,
        "fp32": true, "scatter": true, "fixed_rows": true
    }))
    .unwrap();
    assert!(matches!(
        scatter,
        Instruction::Reduce {
            scatter: true,
            fixed_rows: true,
            ..
        }
    ));
    let exchange: Instruction = serde_json::from_value(serde_json::json!({
        "op": "exchange", "source": "a", "destination": "b", "elements": 16,
        "fp32": false
    }))
    .unwrap();
    assert!(matches!(
        &exchange,
        Instruction::Exchange {
            int32: false,
            fixed_rows: false,
            ..
        }
    ));
    assert!(
        branches::validate_instructions(&[
            Instruction::Fork,
            exchange.clone(),
            Instruction::Switch,
            Instruction::Join
        ])
        .is_err()
    );
    assert!(
        branches::validate_instructions(&[
            Instruction::Fork,
            Instruction::Switch,
            exchange,
            Instruction::Join
        ])
        .is_ok()
    );
    let old: Instruction = serde_json::from_value(serde_json::json!({
        "op": "kernel", "manifest": "kernel.json", "buffers": []
    }))
    .unwrap();
    assert!(matches!(
        old,
        Instruction::Kernel {
            fixed_rows: None,
            ..
        }
    ));
}

#[test]
fn tuned_w4_reuses_capacity_only_for_device_routing() {
    use super::w4_plan_rows;
    assert_eq!(
        w4_plan_rows(12288, 6, Some(16), true, false).unwrap(),
        12288
    );
    assert_eq!(w4_plan_rows(12288, 6, None, true, false).unwrap(), 6);
    assert_eq!(w4_plan_rows(12288, 6, None, true, true).unwrap(), 6);
    assert!(w4_plan_rows(12288, 6, Some(16), false, false).is_err());
    assert!(w4_plan_rows(12288, 6, Some(16), true, true).is_err());
    assert!(w4_plan_rows(12288, 6, Some(0), true, false).is_err());
}
