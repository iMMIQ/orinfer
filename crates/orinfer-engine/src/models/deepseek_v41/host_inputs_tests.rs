use super::*;
use crate::host_rows::{Int8Rows, Shard};
use orinfer_model_sdk::conditional_memory::{HashLayer, HashLayout};
use safetensors::tensor::{TensorView, serialize};
use std::{collections::BTreeMap, sync::atomic::AtomicUsize};

fn assets() -> (Arc<Hasher>, Arc<SharedTables>) {
    assets_with_residency(false)
}
fn assets_with_residency(resident: bool) -> (Arc<Hasher>, Arc<SharedTables>) {
    let layout = HashLayout {
        schema_version: 1,
        config_sha256: "0".repeat(64),
        tokenizer_sha256: "1".repeat(64),
        token_map: (0..16).collect(),
        pad_token: 0,
        barriers: vec![15],
        heads: 8,
        layers: [1, 14]
            .into_iter()
            .map(|layer| HashLayer {
                layer,
                rows: 48,
                multipliers: [1, 3, 5, 7],
                primes: vec![2; 24],
            })
            .collect(),
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rows.safetensors");
    let values = (0..48).flat_map(|r| vec![r as u8; 256]).collect::<Vec<_>>();
    let scales = (0..384)
        .flat_map(|i| (i as f32 + 0.25).to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(
        &path,
        serialize(
            [
                (
                    "w",
                    TensorView::new(safetensors::Dtype::I8, vec![48, 256], &values).unwrap(),
                ),
                (
                    "s",
                    TensorView::new(safetensors::Dtype::F32, vec![48, 8], &scales).unwrap(),
                ),
            ],
            None,
        )
        .unwrap(),
    )
    .unwrap();
    let mut store1 = Int8Rows::open(&path, "w", &path, "s", Shard { first: 0, rows: 48 }).unwrap();
    let mut store14 = Int8Rows::open(&path, "w", &path, "s", Shard { first: 0, rows: 48 }).unwrap();
    if resident {
        store1 = store1.into_resident(1 << 20).unwrap();
        store14 = store14.into_resident(1 << 20).unwrap();
    }
    let tables = Arc::new(SharedTables::new());
    tables
        .set(Ok(Arc::new(BTreeMap::from([
            (1, SharedInt8Rows::new(store1)),
            (14, SharedInt8Rows::new(store14)),
        ]))))
        .ok()
        .unwrap();
    (Arc::new(Hasher::new(layout).unwrap()), tables)
}
fn query(
    hasher: &Hasher,
    layer_index: usize,
    history: &[u32],
    tokens: &[u32],
) -> (Vec<u64>, Vec<bool>) {
    let (mut hashes, mut valid) = (vec![], vec![]);
    hasher
        .hash_into(history, tokens, &mut hashes, &mut valid)
        .unwrap();
    (layer_ids_for(hasher, layer_index, &hashes, &valid), valid)
}
fn check(rows: &Rows, ids: &[u64], valid: &[bool]) {
    let values = ids
        .iter()
        .flat_map(|&r| vec![r as u8; 256])
        .collect::<Vec<_>>();
    let scales = ids
        .iter()
        .flat_map(|&r| (0..8).flat_map(move |g| ((r * 8 + g) as f32 + 0.25).to_le_bytes()))
        .collect::<Vec<_>>();
    assert_eq!(rows.rows, valid.len());
    let mut actual = [
        vec![0; values.len()],
        vec![0; scales.len()],
        vec![0; valid.len()],
    ];
    for (plane, offset, source) in rows.regions() {
        actual[plane][offset..offset + source.len()].copy_from_slice(source);
    }
    assert_eq!(actual[0], values);
    assert_eq!(actual[1], scales);
    assert_eq!(
        actual[2],
        valid.iter().map(|&v| u8::from(v)).collect::<Vec<_>>()
    );
}
#[test]
fn partial_window_joins_rank_readers_and_preserves_mixed_owner_bytes() {
    let (hasher, tables) = assets();
    let store1 = &tables.get().unwrap().as_ref().unwrap()[&1];
    let store14 = &tables.get().unwrap().as_ref().unwrap()[&14];
    let window = Window::new(
        hasher.clone(),
        tables.clone(),
        3,
        16,
        vec![1, 2, 3],
        vec![4, 5, 15, 6],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    window.start();
    let (mut ids1, mut valid) = query(&hasher, 0, &[9], &[10]);
    let (x, v) = query(&hasher, 0, &[2, 3, 4], &[5, 15]);
    ids1.extend(x);
    valid.extend(v);
    let (x, v) = query(&hasher, 0, &[10, 11], &[7]);
    ids1.extend(x);
    valid.extend(v);
    let (mut ids14, _) = query(&hasher, 1, &[9], &[10]);
    ids14.extend(query(&hasher, 1, &[2, 3, 4], &[5, 15]).0);
    ids14.extend(query(&hasher, 1, &[10, 11], &[7]).0);
    let first = First::new(
        &[1, 14],
        vec![Hint {
            window,
            destination: 1..3,
            source: 1..3,
        }],
    );
    let calls = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let (first, ids1, ids14, valid, calls) = (&first, &ids1, &ids14, &valid, &calls);
            scope.spawn(move || {
                let rows = first
                    .get(
                        1,
                        store1,
                        || {
                            calls.fetch_add(1, Ordering::Relaxed);
                            Ok(ids1.clone())
                        },
                        valid,
                    )
                    .unwrap();
                check(rows, ids1, valid);
                assert_eq!(rows.reused_rows, 2);
                let rows14 = first.get(14, store14, || Ok(ids14.clone()), valid).unwrap();
                check(rows14, ids14, valid);
                assert_eq!(rows14.reused_rows, 2);
            });
        }
    });
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    // Replacing the table's packet cache cannot change this frame's data.
    store1.gather_split(&[47, 3], 2 * RECORD).unwrap();
    check(first.prepared.get().unwrap(), &ids1, &valid);
}
#[test]
fn changed_input_or_media_does_not_wait_for_an_unrelated_forecast() {
    let (hasher, tables) = assets();
    let store = &tables.get().unwrap().as_ref().unwrap()[&1];
    for mask in [false, true] {
        let window = Window::new(
            hasher.clone(),
            tables.clone(),
            2,
            20,
            vec![1, 2, 3],
            vec![4, 5],
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
        let (mut ids, mut valid) = query(&hasher, 0, &[1, 2, 3], &[4, 5]);
        if mask {
            valid[0] = !valid[0];
        } else {
            ids[0] ^= 1;
        }
        let first = First::new(
            &[1, 14],
            vec![Hint {
                window: window.clone(),
                destination: 0..2,
                source: 0..2,
            }],
        );
        let rows = first.get(1, store, || Ok(ids.clone()), &valid).unwrap();
        check(rows, &ids, &valid);
        assert_eq!(rows.reused_rows, 0);
        assert!(!window.is_ready());
    }
}
#[test]
fn speculative_failure_retries_live_rows_and_live_failure_propagates() {
    let (hasher, tables) = assets();
    let store = &tables.get().unwrap().as_ref().unwrap()[&1];
    let missing = Arc::new(SharedTables::new());
    missing.set(Ok(Arc::new(BTreeMap::new()))).ok().unwrap();
    let window = Window::new(
        hasher.clone(),
        missing,
        1,
        0,
        vec![],
        vec![4],
        Arc::new(AtomicBool::new(false)),
    )
    .unwrap();
    let (ids, valid) = query(&hasher, 0, &[], &[4]);
    let first = First::new(
        &[1, 14],
        vec![Hint {
            window,
            destination: 0..1,
            source: 0..1,
        }],
    );
    let rows = first.get(1, store, || Ok(ids.clone()), &valid).unwrap();
    check(rows, &ids, &valid);
    assert_eq!(rows.reused_rows, 0);
    let bad = First::new(&[1, 14], vec![]);
    assert!(bad.get(1, store, || Ok(vec![49; LANES]), &[true]).is_err());
    assert!(
        bad.get(
            1,
            store,
            || panic!("failed frame must not run again"),
            &[true]
        )
        .is_err()
    );
}
#[test]
fn concurrent_window_and_extra_prefetch_never_self_deadlock() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for (resident, rounds, tokens_len) in [(true, 128usize, 256usize), (false, 32, 128)] {
            let (hasher, tables) = assets_with_residency(resident);
            let map = Arc::clone(tables.get().unwrap().as_ref().unwrap());
            let store1 = &map[&1];
            let store14 = &map[&14];
            let history = vec![1, 2, 3];
            let tokens: Vec<u32> = (0..tokens_len).map(|i| (i % 14 + 1) as u32).collect();
            let (ids1, valid) = query(&hasher, 0, &history, &tokens);
            let (ids14, _) = query(&hasher, 1, &history, &tokens);
            let active = Arc::new(AtomicBool::new(false));
            for round in 0..rounds {
                let window = Window::new(
                    Arc::clone(&hasher),
                    Arc::clone(&tables),
                    round % 4,
                    0,
                    history.clone(),
                    tokens.clone(),
                    Arc::clone(&active),
                )
                .unwrap();
                let extra = Window::new(
                    Arc::clone(&hasher),
                    Arc::clone(&tables),
                    (round + 1) % 4,
                    0,
                    history.clone(),
                    tokens.clone(),
                    Arc::clone(&active),
                )
                .unwrap();
                let first = First::new(
                    &[1, 14],
                    vec![Hint {
                        window: Arc::clone(&window),
                        destination: 0..tokens_len,
                        source: 0..tokens_len,
                    }],
                );
                if round % 2 == 0 {
                    window.start();
                } else {
                    Window::start_all(&[Arc::clone(&window), extra]);
                }
                first.prefetch_extra(14, Arc::clone(&map), || Ok(ids14.clone()), &valid);
                std::thread::scope(|scope| {
                    for _ in 0..4 {
                        let (first, ids1, ids14, valid) = (&first, &ids1, &ids14, &valid);
                        scope.spawn(move || {
                            let r1 = first.get(1, store1, || Ok(ids1.clone()), valid).unwrap();
                            assert_eq!(r1.reused_rows, tokens_len);
                            let r14 = first.get(14, store14, || Ok(ids14.clone()), valid).unwrap();
                            assert_eq!(r14.reused_rows, tokens_len);
                        });
                    }
                });
                // Ensure background batch clears active before next round if start_all ran.
                let _ = window.prepared.get().unwrap();
            }
        }
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(std::time::Duration::from_secs(15))
        .expect("concurrent Window::start/start_all + First::prefetch_extra deadlocked");
}
