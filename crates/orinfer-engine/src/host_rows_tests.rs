use super::*;
#[test]
fn resident_chunks_keep_offsets_tails_and_owned_bytes() {
    use std::os::unix::fs::FileExt;
    let rows = (64 << 20) + 17;
    let mut plane = Plane {
        parts: vec![Part {
            file: tempfile::tempfile().unwrap(),
            start: 7,
            first_byte: 0,
            bytes: rows + 3,
        }],
        row_bytes: 1,
        rows: rows + 3,
        resident: None,
    };
    plane.parts[0]
        .file
        .set_len((plane.parts[0].start + plane.rows) as u64)
        .unwrap();
    let positions = [0, (32 << 20) - 1, 32 << 20, (64 << 20) - 1, rows - 1];
    for (i, position) in positions.into_iter().enumerate() {
        plane.parts[0]
            .file
            .write_all_at(&[i as u8 + 1], (10 + position) as u64)
            .unwrap();
    }
    plane.load_rows(3, rows).unwrap();
    plane.parts[0].file.set_len(0).unwrap();
    for (i, position) in positions.into_iter().enumerate() {
        let mut actual = [0];
        plane.read_row_at(3 + position, &mut actual).unwrap();
        assert_eq!(actual, [i as u8 + 1]);
    }
}
#[test]
fn segmented_tables_preserve_global_ids_and_resident_crossings() {
    use safetensors::tensor::{TensorView, serialize};
    let dir = tempfile::tempdir().unwrap();
    let mut weights = vec![];
    let mut scales = vec![];
    for (part, first, rows) in [(0, 0, 3), (1, 3, 5)] {
        let w: Vec<u8> = (first * 32..(first + rows) * 32).map(|i| i as u8).collect();
        let scale: Vec<u8> = (first..first + rows)
            .flat_map(|i| (i as f32 + 0.5).to_le_bytes())
            .collect();
        let path = dir.path().join(format!("{part}.safetensors"));
        std::fs::write(
            &path,
            serialize(
                [
                    ("w", TensorView::new(Dtype::I8, vec![rows, 32], &w).unwrap()),
                    (
                        "s",
                        TensorView::new(Dtype::F32, vec![rows, 1], &scale).unwrap(),
                    ),
                ],
                None,
            )
            .unwrap(),
        )
        .unwrap();
        weights.push((path.clone(), "w".to_owned()));
        scales.push((path, "s".to_owned()));
    }
    let rows = Int8Rows::open_sharded(&weights, &scales, Shard { first: 1, rows: 6 }).unwrap();
    let ids = [6, 2, 3, 1, 3];
    let mut expected = vec![];
    rows.gather_into(&ids, &mut expected, 180).unwrap();
    for (&id, bytes) in ids.iter().zip(expected.as_chunks::<36>().0) {
        assert_eq!(bytes[0], (id * 32) as u8);
        assert_eq!(&bytes[32..], &(id as f32 + 0.5).to_le_bytes());
    }
    let budget = rows.required_resident_bytes().unwrap();
    let resident = rows.into_resident(budget).unwrap();
    for (path, _) in &weights {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(0)
            .unwrap();
    }
    let mut actual = vec![];
    resident.gather_into(&ids, &mut actual, 180).unwrap();
    assert_eq!(actual, expected);
}
#[test]
fn packed_records_preserve_queries_shard_edges_and_resident_lifetime() {
    use safetensors::tensor::{TensorView, serialize};
    let dir = tempfile::tempdir().unwrap();
    let mut sources = vec![];
    let mut expected_rows = vec![];
    for (part, first, rows) in [(0, 0, 3), (1, 3, 5)] {
        let mut records = vec![];
        for row in first..first + rows {
            let mut packet: Vec<u8> = (row * 64..(row + 1) * 64).map(|i| i as u8).collect();
            packet.extend((row as f32 + 0.5).to_le_bytes());
            packet.extend((-0.0f32).to_le_bytes());
            records.extend(&packet);
            expected_rows.push(packet);
        }
        let path = dir.path().join(format!("{part}.safetensors"));
        std::fs::write(
            &path,
            serialize(
                [(
                    "record",
                    TensorView::new(Dtype::U8, vec![rows, 72], &records).unwrap(),
                )],
                None,
            )
            .unwrap(),
        )
        .unwrap();
        sources.push((path, "record".to_owned()));
    }
    let store = Int8Rows::open_records(&sources, Shard { first: 1, rows: 6 }).unwrap();
    let ids = [6, 2, 3, 1, 3];
    let expected: Vec<u8> = ids
        .iter()
        .flat_map(|&id| expected_rows[id as usize].clone())
        .collect();
    let mut actual = vec![];
    store
        .gather_into(&ids, &mut actual, expected.len())
        .unwrap();
    assert_eq!(actual, expected);
    assert!(
        store
            .gather_into(&[0], &mut actual, expected.len())
            .is_err()
    );
    assert_eq!(actual, expected);
    let budget = store.required_resident_bytes().unwrap();
    let resident = store.into_resident(budget).unwrap();
    assert!(resident.resident_bytes() <= budget);
    for (path, _) in &sources {
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(0)
            .unwrap();
    }
    let shared = SharedInt8Rows::new(resident);
    assert_eq!(&*shared.gather(&ids, expected.len()).unwrap(), &expected);
    assert_eq!(&*shared.gather(&ids, expected.len()).unwrap(), &expected);
    assert_eq!(shared.statistics().unwrap().packet_hits, 1);
    let split = shared.gather_split(&ids, expected.len()).unwrap();
    let expected_values: Vec<u8> = expected
        .as_chunks::<72>()
        .0
        .iter()
        .flat_map(|row| &row[..64])
        .copied()
        .collect();
    let expected_scales: Vec<u8> = expected
        .as_chunks::<72>()
        .0
        .iter()
        .flat_map(|row| &row[64..])
        .copied()
        .collect();
    assert_eq!(split.0, expected_values);
    assert_eq!(split.1, expected_scales);
    // Second identical gather_split hits the shared split packet cache.
    assert_eq!(
        &*shared.gather_split(&ids, expected.len()).unwrap(),
        &*split
    );
    assert_eq!(shared.statistics().unwrap().packet_hits, 2);
    // Large batch (>1024) exercises bounded parallel chunked resident gather.
    let big_ids: Vec<u64> = (0..2500).map(|i| 1 + (i % 6) as u64).collect();
    let big_expected: Vec<u8> = big_ids
        .iter()
        .flat_map(|&id| expected_rows[id as usize].clone())
        .collect();
    let big_split = shared.gather_split(&big_ids, big_ids.len() * 72).unwrap();
    let big_exp_v: Vec<u8> = big_expected
        .as_chunks::<72>()
        .0
        .iter()
        .flat_map(|row| &row[..64])
        .copied()
        .collect();
    let big_exp_s: Vec<u8> = big_expected
        .as_chunks::<72>()
        .0
        .iter()
        .flat_map(|row| &row[64..])
        .copied()
        .collect();
    assert_eq!(big_split.0, big_exp_v);
    assert_eq!(big_split.1, big_exp_s);
    assert!(shared.gather_split(&[0], 72).is_err());
}
#[test]
fn stable_deduplication_preserves_query_order_and_empty_ranks() {
    let ids = [9, 0, 9, 4, 1, 4, 8];
    let plan = RowPlan::build(&ids, 10, 4, 32).unwrap();
    assert_eq!(plan.rows, [0, 1, 4, 8, 9]);
    assert_eq!(plan.starts, [0, 2, 3, 4, 5]);
    assert_eq!(
        plan.restore
            .iter()
            .map(|&i| plan.rows[i] as i64)
            .collect::<Vec<_>>(),
        ids
    );
    assert_eq!(plan.for_rank(0).unwrap(), [0, 1]);
    let empty = RowPlan::build(&[0, 0], 1, 8, 8).unwrap();
    assert!(empty.for_rank(7).unwrap().is_empty());
    assert!(RowPlan::build(&[-1], 10, 8, 8).is_err());
    assert!(RowPlan::build(&[10], 10, 8, 8).is_err());
    assert!(RowPlan::build(&ids, 10, 8, 4).is_err());
}
#[test]
fn host_rows_keep_quantized_payload_and_scales_byte_exact() {
    use safetensors::tensor::{TensorView, serialize};
    let dir = tempfile::tempdir().unwrap();
    let weights: Vec<u8> = (0..4 * 32).map(|i| i as u8).collect();
    let scales: Vec<u8> = [0.5f32, 1., 2., 4.]
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let w = dir.path().join("weight.safetensors");
    let s = dir.path().join("scale.safetensors");
    std::fs::write(
        &w,
        serialize(
            [(
                "w",
                TensorView::new(Dtype::I8, vec![4, 32], &weights).unwrap(),
            )],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &s,
        serialize(
            [(
                "s",
                TensorView::new(Dtype::F32, vec![4, 1], &scales).unwrap(),
            )],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let rows = Int8Rows::open(&w, "w", &s, "s", Shard::for_rank(4, 2, 1).unwrap()).unwrap();
    assert_eq!(rows.row_bytes(), 36);
    let mut packet = vec![];
    rows.gather_into(&[3, 2, 3], &mut packet, 108).unwrap();
    let expected: Vec<u8> = [3, 2, 3]
        .into_iter()
        .flat_map(|row| {
            weights[row * 32..(row + 1) * 32]
                .iter()
                .chain(&scales[row * 4..(row + 1) * 4])
                .copied()
        })
        .collect();
    assert_eq!(packet, expected);
    assert!(rows.gather_into(&[1], &mut packet, 108).is_err());
    assert_eq!(packet, expected);
    assert!(rows.gather_into(&[3], &mut packet, 35).is_err());
    assert_eq!(packet, expected);
    let all = Int8Rows::open(&w, "w", &s, "s", Shard::for_rank(4, 1, 0).unwrap()).unwrap();
    for ids in [
        vec![0, 0, 1, 2, 3],
        vec![3, 3, 0, 3, 2, 1, 0, 2],
        vec![2; 257],
        vec![3, 2, 1, 0],
        vec![],
    ] {
        all.gather_into(&ids, &mut packet, ids.len() * 36).unwrap();
        let wanted: Vec<u8> = ids
            .iter()
            .flat_map(|&row| {
                let row = row as usize;
                weights[row * 32..(row + 1) * 32]
                    .iter()
                    .chain(&scales[row * 4..(row + 1) * 4])
                    .copied()
            })
            .collect();
        assert_eq!(packet, wanted);
    }
    let shared = SharedInt8Rows::new(rows);
    let batches = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| scope.spawn(|| shared.gather(&[3, 2, 3], 108).unwrap()))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    for batch in &batches {
        assert_eq!(**batch, expected);
        assert!(Arc::ptr_eq(batch, &batches[0]));
    }
    assert!(shared.gather(&[3, 2, 3], 107).is_err());
    assert!(shared.gather(&[1], 108).is_err());
    let changed = shared.gather(&[2, 3], 72).unwrap();
    assert_eq!(&changed[..36], &expected[36..72]);
    assert_eq!(&changed[36..], &expected[..36]);
    assert_eq!(*batches[0], expected);

    let cached = SharedInt8Rows::with_cache(all, 65536).unwrap();
    let first = cached.gather(&[3, 0, 2, 3], 144).unwrap();
    assert_eq!(cached.statistics().unwrap().unique_row_reads, 3);
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let cached = &cached;
                scope.spawn(move || {
                    let ids = if i % 2 == 0 {
                        vec![1, 3, 1, 0, 2]
                    } else {
                        vec![2, 1, 3]
                    };
                    let bytes = cached.gather(&ids, 180).unwrap();
                    (ids, bytes)
                })
            })
            .collect();
        for handle in handles {
            let (ids, bytes) = handle.join().unwrap();
            let wanted: Vec<u8> = ids
                .into_iter()
                .flat_map(|id| {
                    let row = id as usize;
                    weights[row * 32..(row + 1) * 32]
                        .iter()
                        .chain(&scales[row * 4..(row + 1) * 4])
                        .copied()
                })
                .collect();
            assert_eq!(*bytes, wanted);
        }
    });
    assert_eq!(cached.statistics().unwrap().unique_row_reads, 4);
    let shard = Shard::for_rank(4, 2, 1).unwrap();
    assert!(
        Int8Rows::open(&w, "w", &s, "s", shard)
            .unwrap()
            .into_resident(35)
            .is_err()
    );
    let rows = Int8Rows::open(&w, "w", &s, "s", shard).unwrap();
    let budget = rows.required_resident_bytes().unwrap();
    let resident = rows.into_resident(budget).unwrap();
    assert!(resident.resident_bytes() <= budget);
    let resident = resident.into_resident(budget).unwrap();
    // Destroy the test payload, including any still-open file descriptors.
    File::create(&w).unwrap();
    File::create(&s).unwrap();
    std::fs::remove_file(w).unwrap();
    std::fs::remove_file(s).unwrap();
    let shared = Arc::new(SharedInt8Rows::with_cache(resident, 0).unwrap());
    let workers: Vec<_> = (0..8)
        .map(|i| {
            let shared = Arc::clone(&shared);
            std::thread::spawn(move || {
                let ids = if i % 2 == 0 { [3, 2, 3] } else { [2, 3, 2] };
                let bytes = shared.gather(&ids, 108).unwrap();
                (ids, bytes)
            })
        })
        .collect();
    for worker in workers {
        let (ids, bytes) = worker.join().unwrap();
        let wanted: Vec<u8> = ids
            .iter()
            .flat_map(|&row| {
                let row = row as usize;
                weights[row * 32..(row + 1) * 32]
                    .iter()
                    .chain(&scales[row * 4..(row + 1) * 4])
                    .copied()
            })
            .collect();
        assert_eq!(*bytes, wanted);
    }
    assert!(shared.gather(&[0], 36).is_err());
    assert_eq!(shared.statistics().unwrap().reserved_bytes, 0);
    assert_eq!(shared.statistics().unwrap().resident_bytes, budget);
    let reordered = cached.gather(&[2, 3, 0, 2], 144).unwrap();
    assert_eq!(&reordered[..36], &first[72..108]);
    assert_eq!(&reordered[36..72], &first[..36]);
    assert_eq!(&reordered[72..108], &first[36..72]);
    assert_eq!(&reordered[108..], &first[72..108]);
    let statistics = cached.statistics().unwrap();
    assert_eq!(statistics.unique_row_reads, 4);
    assert!(statistics.unique_row_hits >= 3);
    assert_eq!(statistics.entries, 4);
    assert!(statistics.reserved_bytes <= 65536);
    assert!(cached.gather(&[4], 144).is_err());
    assert!(cached.gather(&[2, 3, 0, 2], 143).is_err());
    let same = cached.gather(&[2, 3, 0, 2], 144).unwrap();
    assert!(Arc::ptr_eq(&reordered, &same));
    assert!(cached.statistics().unwrap().packet_hits >= 1);
    assert_eq!(cached.statistics().unwrap().unique_row_reads, 4);
}
