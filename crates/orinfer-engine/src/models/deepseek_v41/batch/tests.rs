use super::*;
fn layout(lengths: &[usize], owners: &[usize]) -> Result<Layout> {
    Layout::new(
        lengths
            .iter()
            .zip(owners)
            .map(|(&tokens, &slot)| BatchSegment { tokens, slot })
            .collect(),
        vec![0],
        false,
        4096,
        128,
    )
}
fn contract() -> Contract {
    serde_json::from_value(serde_json::json!({
        "owners": 32, "columns": [], "private": [], "pointers": {},
        "active": "a", "endpoints": "e", "work": {}
    }))
    .unwrap()
}
fn key(layout: &Layout) -> Key {
    shape_key(
        &contract(),
        layout,
        (layout.role, layout.decode, 1),
        &BTreeSet::new(),
    )
    .unwrap()
}
#[test]
fn source_work_padding_preserves_order_and_published_capacity() {
    let c = contract();
    let l = layout(
        &[6, 6, 6, 6, 6, 6, 6, 6, 752],
        &[0, 1, 2, 3, 4, 5, 6, 7, 17],
    )
    .unwrap();
    let grid: RowExpression = serde_json::from_value(
        serde_json::json!({"op":"multiply","lhs":{"op":"rows"},"rhs":{"op":"constant","value":4}}),
    )
    .unwrap();
    let raw = work_items(&l, &grid, None).unwrap();
    let padded = c.work_items(&l, &grid, "source", 0).unwrap();
    assert!(padded.starts_with(&raw));
    assert!(padded[raw.len()..].iter().all(|&x| x == [-1, 0]));
    assert!((padded.len() / 2).is_power_of_two());
    let launches = [WorkLaunch {
        role: l.role,
        grid: grid.clone(),
        range: None,
        partition: 1,
        decoder: false,
    }]
    .into();
    let k = shape_key(&c, &l, (l.role, false, 1), &launches).unwrap();
    assert_eq!(k.work_rows, vec![padded.len() / 2]);
    for owners in [1usize, 3, 20, 32, 128] {
        for count in 0..=owners * 64 {
            let rows = work_launch_rows(count, owners, true).unwrap();
            assert!(rows * 2 >= count && rows <= owners * 32);
        }
    }
    assert!(work_launch_rows(8193, 128, true).is_err());
}
#[test]
fn explicit_verification_capacity_keeps_legacy_selection_and_checks_head_bounds() {
    let mut c = contract();
    c.output_capacity = 544;
    assert_eq!(c.verification_rows(127, 2048).unwrap(), 127);
    assert_eq!(c.verification_rows(2048, 2048).unwrap(), 2048);
    c.verification_capacity = Some(192);
    assert_eq!(c.verification_rows(127, 2048).unwrap(), 192);
    for rows in [0, 545, 2049] {
        c.verification_capacity = Some(rows);
        assert!(c.verification_rows(127, 2048).is_err());
    }
    c.verification_capacity = Some(256);
    assert!(c.verification_rows(127, 255).is_err());
}
#[test]
fn successful_shape_streaks_are_family_local_and_reset_on_eviction() {
    let mut runtime = Runtime::new(contract(), 32, &[], &BTreeSet::new());
    let a = key(&layout(&[16], &[7]).unwrap());
    let b = key(&layout(&[17], &[7]).unwrap());
    let mut draft = a.clone();
    draft.family.0 = super::super::PacketRole::Draft;
    assert_eq!(runtime.consecutive_uses(&a), 0);
    runtime.record_use(a.clone());
    runtime.record_use(draft.clone());
    assert_eq!(runtime.consecutive_uses(&a), 1);
    runtime.record_use(a.clone());
    assert_eq!(runtime.consecutive_uses(&a), 2);
    assert_eq!(runtime.consecutive_uses(&draft), 1);
    runtime.record_use(b.clone());
    assert_eq!(runtime.consecutive_uses(&a), 0);
    runtime.record_use(a.clone());
    runtime.remove(&b);
    assert_eq!(runtime.consecutive_uses(&a), 1);
    runtime.remove(&a);
    assert_eq!(runtime.consecutive_uses(&a), 0);
    assert_eq!(runtime.consecutive_uses(&draft), 1);
    for _ in 0..300 {
        runtime.record_use(a.clone());
    }
    assert_eq!(runtime.consecutive_uses(&a), u8::MAX);
    assert_eq!(runtime.recent.len(), 2);
}
#[test]
fn launch_identity_matches_the_compilers_owner_work_extents() {
    let mut contract = contract();
    contract.decoder_replay = Some(
        serde_json::from_value(serde_json::json!({
            "keep":128,"final_prompt_only":true,"tables":["w"],"maps":[]
        }))
        .unwrap(),
    );
    let grid: RowExpression = serde_json::from_value(serde_json::json!({
        "op":"divide","lhs":{"op":"add","lhs":{"op":"rows"},
            "rhs":{"op":"constant","value":1}},"rhs":{"op":"constant","value":2}
    }))
    .unwrap();
    for lengths in [
        &[1, 15][..],
        &[33, 95],
        &[35, 93],
        &[769, 1279],
        &[771, 1277],
        &[2048, 1],
    ] {
        let mut layout = layout(lengths, &[7, 17]).unwrap();
        layout.outputs = vec![1];
        layout.output_positions = vec![layout.segments[1].tokens - 1];
        layout.decoder_roles = vec![
            super::super::replay::Role::Producer,
            super::super::replay::Role::Final,
        ];
        for decoder in [false, true] {
            for partition in [1, 8] {
                for range in [None, Some((1, 8)), Some((9, 2048))] {
                    contract.partitioned_tables.clear();
                    contract.work_ranges.clear();
                    contract.decoder_replay.as_mut().unwrap().tables.clear();
                    if decoder {
                        contract
                            .decoder_replay
                            .as_mut()
                            .unwrap()
                            .tables
                            .insert("w".into());
                    }
                    if partition > 1 {
                        contract.partitioned_tables.insert("w".into(), partition);
                    }
                    if let Some((min_rows, max_rows)) = range {
                        contract
                            .work_ranges
                            .insert("w".into(), WorkRange { min_rows, max_rows });
                    }
                    let launches = [WorkLaunch {
                        role: super::super::PacketRole::Target,
                        grid: grid.clone(),
                        range,
                        partition,
                        decoder,
                    }]
                    .into();
                    let key = shape_key(
                        &contract,
                        &layout,
                        (super::super::PacketRole::Target, false, 1),
                        &launches,
                    )
                    .unwrap();
                    let expected = (0..partition)
                        .map(|rank| {
                            contract
                                .work_items(&layout, &grid, "w", rank)
                                .unwrap()
                                .len()
                                / 2
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(key.work_rows, expected);
                }
            }
        }
    }
}
#[test]
fn partitioned_work_keeps_original_owner_and_graph_order() {
    let mut contract: Contract = serde_json::from_value(serde_json::json!({
        "owners": 32, "columns": [], "private": [], "pointers": {},
        "active": "a", "endpoints": "e", "work": {}
    }))
    .unwrap();
    let a = layout(&[1, 15], &[91, 4]).unwrap();
    let b = layout(&[15, 1], &[8, 77]).unwrap();
    assert_eq!(key(&a), key(&b));
    contract.partitioned_tables.insert("w".into(), 8);
    assert_eq!(contract.token_capacity("w", 2048).unwrap(), 256);
    assert_eq!(contract.token_capacity("w", 2047).unwrap(), 256);
    assert_eq!(contract.token_capacity("original", 2048).unwrap(), 2048);
    let launches = [WorkLaunch {
        role: super::super::PacketRole::Target,
        grid: RowExpression::Constant { value: 64 },
        range: None,
        partition: 8,
        decoder: false,
    }]
    .into();
    assert_ne!(
        shape_key(
            &contract,
            &a,
            (super::super::PacketRole::Target, false, 1),
            &launches
        )
        .unwrap(),
        shape_key(
            &contract,
            &b,
            (super::super::PacketRole::Target, false, 1),
            &launches
        )
        .unwrap()
    );
    let reordered_ids = layout(&[1, 15], &[33, 91]).unwrap();
    assert_eq!(
        shape_key(
            &contract,
            &a,
            (super::super::PacketRole::Target, false, 1),
            &launches
        )
        .unwrap(),
        shape_key(
            &contract,
            &reordered_ids,
            (super::super::PacketRole::Target, false, 1),
            &launches
        )
        .unwrap()
    );
    let grid: RowExpression =
        serde_json::from_value(serde_json::json!({"op":"constant","value":64})).unwrap();
    let work = contract.work_items(&a, &grid, "w", 7).unwrap();
    assert_eq!(work.len(), 64);
    assert!(work.iter().all(|p| p[0] == 1));
    let empty = layout(&[1, 2], &[9, 4]).unwrap();
    assert_eq!(
        contract.work_items(&empty, &grid, "w", 7).unwrap(),
        [[-1, 0], [-1, 0]]
    );
}
#[test]
fn address_tables_keep_owner_and_output_order_when_graphs_share_a_shape() {
    let a = layout(&[1, 1], &[91, 8]).unwrap();
    let b = layout(&[1, 1], &[8, 91]).unwrap();
    assert_eq!(key(&a), key(&b));
    assert_ne!(a.table_key(1), b.table_key(1));
    let mut output = a.clone();
    output.outputs = vec![1];
    assert_eq!(key(&a), key(&output));
    assert_ne!(a.table_key(1), output.table_key(1));
    assert_ne!(a.table_key(1), a.table_key(512));
    let mut decode = a.clone();
    decode.decode = true;
    assert_ne!(a.table_key(1), decode.table_key(1));
}
#[test]
fn arbitrary_batch_work_covers_each_logical_owner_tile() {
    let grid: RowExpression = serde_json::from_value(
        serde_json::json!({"op":"multiply","lhs":{"op":"rows"},"rhs":{"op":"constant","value":3}}),
    )
    .unwrap();
    for batch in 1..=128 {
        let lengths = (0..batch).map(|i| 1 + i % 31).collect::<Vec<_>>();
        let ids = (0..batch).map(|i| i * 17 + 9).collect::<Vec<_>>();
        let l = layout(&lengths, &ids).unwrap();
        let work = work_items(&l, &grid, None).unwrap();
        assert_eq!(work.len() % 2, 0);
        assert_eq!(
            work.iter().take(batch.min(64)).copied().collect::<Vec<_>>(),
            (0..batch.min(64))
                .map(|owner| [owner as i32, 0])
                .collect::<Vec<_>>()
        );
        for (owner, &q) in lengths.iter().enumerate() {
            let physical = work
                .iter()
                .filter(|p| p[0] == owner as i32)
                .map(|p| p[1] as usize)
                .collect::<Vec<_>>();
            assert_eq!(physical, (0..(3 * q).min(64)).collect::<Vec<_>>());
            let mut logical = physical
                .into_iter()
                .flat_map(|p| (p..3 * q).step_by(64))
                .collect::<Vec<_>>();
            logical.sort_unstable();
            assert_eq!(logical, (0..3 * q).collect::<Vec<_>>());
        }
    }
}
#[test]
fn bounded_work_distributes_each_owner_across_all_physical_slots() {
    let grid: RowExpression =
        serde_json::from_value(serde_json::json!({"op":"constant","value":64})).unwrap();
    for batch in 1..=128 {
        let work = work_lengths(vec![1; batch], &grid, None).unwrap();
        assert_eq!(work.len(), batch * 64);
        let mut slots = vec![0u64; batch];
        for (index, &[owner, _]) in work.iter().enumerate() {
            slots[owner as usize] |= 1u64 << (index % 64);
        }
        assert_eq!(slots, vec![u64::MAX; batch]);
    }
}
#[test]
fn interleaved_work_keeps_empty_views_and_odd_padding_isolated() {
    let grid: RowExpression = serde_json::from_value(serde_json::json!({"op":"rows"})).unwrap();
    assert_eq!(
        work_lengths([0, 3, 1, 0, 2], &grid, None).unwrap(),
        [[1, 0], [2, 0], [4, 0], [1, 1], [4, 1], [1, 2]]
    );
    assert_eq!(
        work_lengths([1, 0, 2], &grid, None).unwrap(),
        [[0, 0], [2, 0], [2, 1], [-1, 0]]
    );
    assert_eq!(
        work_lengths([0, 0], &grid, None).unwrap(),
        [[-1, 0], [-1, 0]]
    );
}
#[test]
fn owner_work_selection_keeps_query_policy_across_mixed_packets() {
    let grid: RowExpression = serde_json::from_value(serde_json::json!({"op":"rows"})).unwrap();
    let mixed = layout(&[1, 511, 512, 513], &[91, 8, 44, 7]).unwrap();
    let short = WorkRange {
        min_rows: 1,
        max_rows: 511,
    };
    let long = WorkRange {
        min_rows: 512,
        max_rows: 4096,
    };
    let a = work_items(&mixed, &grid, Some(&short)).unwrap();
    let b = work_items(&mixed, &grid, Some(&long)).unwrap();
    assert!(a.iter().all(|p| p[0] < 2));
    assert!(b.iter().all(|p| p[0] >= 2));
    let one = layout(&[511], &[99]).unwrap();
    assert_eq!(
        work_items(&one, &grid, Some(&long)).unwrap(),
        [[-1, 0], [-1, 0]]
    );
    assert_eq!(work_items(&one, &grid, Some(&short)).unwrap().len(), 64);
}
#[test]
fn graph_identity_follows_shape_and_role_not_owner_order() {
    let a = layout(&[1, 7, 3], &[91, 4, 33]).unwrap();
    let b = layout(&[3, 1, 7], &[8, 77, 5]).unwrap();
    assert_eq!(key(&a), key(&b));
    assert_eq!(a.rows.starts, vec![0, 1, 8]);
    assert_eq!(b.rows.starts, vec![0, 3, 4]);
    assert!(layout(&[1, 7], &[9, 9]).is_err());
    assert!(layout(&[0], &[8]).is_err());
    assert!(Layout::new(a.segments.clone(), vec![0], true, 4096, 128).is_err());
    assert!(Layout::new(a.segments.clone(), vec![2, 0], false, 4096, 128).is_err());
    let mut c = a.clone();
    c.outputs.push(2);
    assert_ne!(key(&a), key(&c));
    let mut d = a.clone();
    d.decode = true;
    assert_ne!(key(&a), key(&d));
}
#[test]
fn scheduling_quantum_is_independent_of_attention_work_selection() {
    let base = serde_json::json!({
        "owners": 128, "columns": [], "private": [], "pointers": {},
        "active": "a", "endpoints": "e", "work": {}, "work_ranges": {}
    });
    let mut contract: Contract = serde_json::from_value(base).unwrap();
    assert_eq!(contract.preferred_span(), 1);
    contract.work_ranges.insert(
        "old".into(),
        WorkRange {
            min_rows: 512,
            max_rows: 2048,
        },
    );
    assert_eq!(contract.preferred_span(), 512);
    contract.prefill_quantum = Some(256);
    assert_eq!(contract.preferred_span(), 256);
    contract.work_ranges.clear();
    assert_eq!(contract.preferred_span(), 256);
}

#[test]
fn verification_uses_every_position_and_keeps_reverse_owner_endpoints() {
    let segments = vec![
        BatchSegment {
            slot: 29,
            tokens: 6,
        },
        BatchSegment { slot: 7, tokens: 2 },
    ];
    let view = Layout::verification(segments, 192, 32).unwrap();
    assert_eq!(view.outputs, [0, 0, 0, 0, 0, 0, 1, 1]);
    assert_eq!(view.output_positions, [0, 1, 2, 3, 4, 5, 0, 1]);
    let clipped = view.decoder(128, true).unwrap();
    assert_eq!(view.output_positions, clipped.output_positions);
    let mut contract = contract();
    contract.output_capacity = 192;
    let family = (view.role, view.decode, 1);
    let target = shape_key(&contract, &view, family, &BTreeSet::new()).unwrap();
    let mut draft = view.clone();
    draft.role = super::super::PacketRole::Draft;
    assert_ne!(
        target,
        shape_key(&contract, &draft, (draft.role, false, 1), &BTreeSet::new()).unwrap()
    );
    let mut endpoints = view.clone();
    endpoints.output_positions[0] = 1;
    assert_ne!(endpoints.table_key(1), view.table_key(1));
    contract.output_capacity = 7;
    assert!(shape_key(&contract, &view, family, &BTreeSet::new()).is_err());
}
