use super::{Element, Instruction, Packet, bounded_rows, gather_element, validate_dispatch};

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
