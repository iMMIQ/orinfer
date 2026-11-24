//! Immutable multi-layer Engram inputs shared by one packet and one bounded forecast.
use super::{SharedTables, team::Team};
use crate::{
    artifact::Result,
    host_rows::SharedInt8Rows,
    sampling::{BackgroundLane, Prepared, prepare_lane},
};
use orinfer_model_sdk::conditional_memory::Hasher;
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

static EXTRA_PREFETCH_LANE: OnceLock<Option<BackgroundLane>> = OnceLock::new();

fn extra_prefetch_lane() -> Option<&'static BackgroundLane> {
    EXTRA_PREFETCH_LANE
        .get_or_init(|| BackgroundLane::spawn("orinfer-engram-prefetch", 16).ok())
        .as_ref()
}

struct ResetActive(Arc<AtomicBool>);
impl Drop for ResetActive {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

const LANES: usize = 24;
const VALUES: usize = 256;
const SCALES: usize = 32;
const RECORD: usize = VALUES + SCALES;

struct Payload {
    planes: Arc<(Vec<u8>, Vec<u8>)>,
    mask: Vec<u8>,
}
struct Part {
    destination: usize,
    source: Range<usize>,
    data: Arc<Payload>,
}
pub(super) struct Rows {
    parts: Vec<Part>,
    pub rows: usize,
    pub reused_rows: usize,
    pub wait_s: f64,
}
impl Rows {
    pub fn regions(&self) -> impl Iterator<Item = (usize, usize, &[u8])> {
        self.parts.iter().flat_map(|part| {
            let s = &part.source;
            [
                (
                    0,
                    part.destination * LANES * VALUES,
                    &part.data.planes.0[s.start * LANES * VALUES..s.end * LANES * VALUES],
                ),
                (
                    1,
                    part.destination * LANES * SCALES,
                    &part.data.planes.1[s.start * LANES * SCALES..s.end * LANES * SCALES],
                ),
                (2, part.destination, &part.data.mask[s.clone()]),
            ]
        })
    }
}
fn read(store: &SharedInt8Rows, ids: &[u64], valid: &[bool]) -> Result<Arc<Payload>> {
    let planes = store.gather_split(ids, ids.len() * RECORD)?;
    Ok(Arc::new(Payload {
        planes,
        mask: valid.iter().map(|&v| u8::from(v)).collect(),
    }))
}
pub(super) struct Window {
    pub owner: usize,
    pub offset: usize,
    pub rows: usize,
    layer_ids: BTreeMap<usize, Arc<Vec<u64>>>,
    valid: Arc<Vec<bool>>,
    prepared: Arc<Prepared<BTreeMap<usize, Arc<Payload>>>>,
    active: Arc<AtomicBool>,
}
impl Window {
    fn new(
        hasher: Arc<Hasher>,
        tables: Arc<SharedTables>,
        owner: usize,
        offset: usize,
        history: Vec<u32>,
        tokens: Vec<u32>,
        active: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        let rows = tokens.len();
        let (mut hashes, mut valid) = (vec![], vec![]);
        hasher.hash_into(&history, &tokens, &mut hashes, &mut valid)?;
        let mut layer_ids = BTreeMap::new();
        for (layer_index, layer) in hasher.layout().layers.iter().enumerate() {
            layer_ids.insert(
                layer.layer,
                Arc::new(layer_ids_for(&hasher, layer_index, &hashes, &valid)),
            );
        }
        let valid = Arc::new(valid);
        let input_layers = layer_ids.clone();
        let input_valid = Arc::clone(&valid);
        let prepared = Prepared::new(move || {
            let stores = tables
                .get()
                .ok_or("Host tables not ready")?
                .as_ref()
                .map_err(Clone::clone)?;
            let mut payloads = BTreeMap::new();
            for (layer, ids) in &input_layers {
                payloads.insert(
                    *layer,
                    read(
                        stores.get(layer).ok_or("Engram table missing")?,
                        ids,
                        &input_valid,
                    )?,
                );
            }
            Ok(payloads)
        });
        Ok(Arc::new(Self {
            owner,
            offset,
            rows,
            layer_ids,
            valid,
            prepared: Arc::new(prepared),
            active,
        }))
    }
    pub fn start(&self) {
        self.prepared.start(&self.active);
    }
    pub fn start_all(windows: &[Arc<Self>]) {
        let Some(first) = windows.first() else {
            return;
        };
        if windows.len() == 1 {
            first.start();
            return;
        }
        if rayon::current_num_threads() < 2
            || first
                .active
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_err()
        {
            return;
        }
        let Some(lane) = prepare_lane() else {
            first.active.store(false, Ordering::Release);
            return;
        };
        let batch = windows.to_vec();
        let active = Arc::clone(&first.active);
        if !lane.try_submit(move || {
            let _reset = ResetActive(active);
            for window in batch {
                let _ = window.prepared.get();
            }
        }) {
            first.active.store(false, Ordering::Release);
        }
    }
    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        self.prepared.is_ready()
    }
}
#[derive(Clone)]
pub(super) struct Hint {
    pub window: Arc<Window>,
    pub destination: Range<usize>,
    pub source: Range<usize>,
}
pub(super) struct First {
    pub layer: usize,
    hints: Arc<Vec<Hint>>,
    pub prepared: Prepared<Rows>,
    extras: BTreeMap<usize, (Arc<Prepared<Rows>>, AtomicBool)>,
}
impl First {
    pub fn new(layers: &[usize], hints: Vec<Hint>) -> Arc<Self> {
        let primary = layers.first().copied().unwrap_or(0);
        let extras = layers
            .iter()
            .skip(1)
            .map(|&layer| {
                (
                    layer,
                    (Arc::new(Prepared::deferred()), AtomicBool::new(false)),
                )
            })
            .collect();
        Arc::new(Self {
            layer: primary,
            hints: Arc::new(hints),
            prepared: Prepared::deferred(),
            extras,
        })
    }
    pub fn handles_layer(&self, layer: usize) -> bool {
        layer == self.layer || self.extras.contains_key(&layer)
    }
    /// Launch background gather for a later Engram layer (e.g. layer 14) once per
    /// forward so it overlaps with layers 0..13 running on the NPU.
    pub fn prefetch_extra(
        &self,
        layer: usize,
        tables: Arc<BTreeMap<usize, SharedInt8Rows>>,
        ids: impl FnOnce() -> Result<Vec<u64>>,
        valid: &[bool],
    ) {
        if rayon::current_num_threads() < 2 {
            return;
        }
        let Some((slot, started)) = self.extras.get(&layer) else {
            return;
        };
        if slot.is_ready()
            || started
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_err()
        {
            return;
        }
        let Ok(ids) = ids() else {
            return;
        };
        let Some(lane) = extra_prefetch_lane() else {
            return;
        };
        let slot = Arc::clone(slot);
        let hints = Arc::clone(&self.hints);
        let valid = valid.to_vec();
        let _ = lane.try_submit(move || {
            let _ = slot.get_or(|| {
                let store = tables.get(&layer).ok_or("Engram table missing")?;
                assemble(layer, store, &ids, &valid, &hints)
            });
        });
    }
    pub fn get(
        &self,
        layer: usize,
        store: &SharedInt8Rows,
        ids: impl FnOnce() -> Result<Vec<u64>>,
        valid: &[bool],
    ) -> Result<&Rows> {
        let slot: &Prepared<Rows> = if layer == self.layer {
            &self.prepared
        } else {
            self.extras
                .get(&layer)
                .map(|(p, _)| p.as_ref())
                .ok_or("Engram layer not prepared in host inputs")?
        };
        slot.get_or(|| assemble(layer, store, &ids()?, valid, &self.hints))
    }
}
fn layer_ids_for(hasher: &Hasher, layer_index: usize, hashes: &[i64], valid: &[bool]) -> Vec<u64> {
    let width = hasher.values_per_token();
    let base = layer_index * LANES;
    valid
        .iter()
        .enumerate()
        .flat_map(|(row, &active)| {
            (0..LANES).map(move |lane| {
                if active {
                    hashes[row * width + base + lane] as u64
                } else {
                    0
                }
            })
        })
        .collect()
}
fn assemble(
    layer: usize,
    store: &SharedInt8Rows,
    ids: &[u64],
    valid: &[bool],
    hints: &[Hint],
) -> Result<Rows> {
    let mut matched: Vec<(Range<usize>, Range<usize>, Arc<Payload>)> = vec![];
    let mut wait_s = 0.;
    let mut candidates: Vec<&Hint> = hints.iter().collect();
    candidates.sort_by_key(|h| h.destination.start);
    let mut cursor = 0usize;
    for hint in candidates {
        let d = hint.destination.clone();
        let s = hint.source.clone();
        // Reject overlapping, out-of-range, or changed row identities/media before waiting.
        if d.start >= cursor
            && d.len() == s.len()
            && !d.is_empty()
            && d.end <= valid.len()
            && s.end <= hint.window.rows
            && hint.window.layer_ids.get(&layer).is_some_and(|w_ids| {
                ids[d.start * LANES..d.end * LANES] == w_ids[s.start * LANES..s.end * LANES]
            })
            && valid[d.clone()] == hint.window.valid[s.clone()]
        {
            let started = Instant::now();
            if let Ok(data) = hint.window.prepared.get()
                && let Some(payload) = data.get(&layer)
            {
                cursor = d.end;
                matched.push((d, s, Arc::clone(payload)));
            }
            wait_s += started.elapsed().as_secs_f64();
        }
    }
    let mut parts = vec![];
    let reused_rows: usize = matched.iter().map(|(d, _, _)| d.len()).sum();
    if matched.is_empty() {
        parts.push(Part {
            destination: 0,
            source: 0..valid.len(),
            data: read(store, ids, valid)?,
        });
    } else {
        let mut gaps: Vec<Range<usize>> = vec![];
        let mut pos = 0usize;
        for (d, _, _) in &matched {
            if pos < d.start {
                gaps.push(pos..d.start);
            }
            pos = d.end;
        }
        if pos < valid.len() {
            gaps.push(pos..valid.len());
        }
        let gap_rows: usize = gaps.iter().map(|g| g.len()).sum();
        if gap_rows > 0 {
            let mut needed = Vec::with_capacity(gap_rows * LANES);
            let mut masks = Vec::with_capacity(gap_rows);
            for g in &gaps {
                needed.extend_from_slice(&ids[g.start * LANES..g.end * LANES]);
                masks.extend_from_slice(&valid[g.clone()]);
            }
            let fallback = read(store, &needed, &masks)?;
            let mut src_pos = 0usize;
            for g in gaps {
                let len = g.len();
                parts.push(Part {
                    destination: g.start,
                    source: src_pos..src_pos + len,
                    data: Arc::clone(&fallback),
                });
                src_pos += len;
            }
        }
        for (d, s, data) in matched {
            parts.push(Part {
                destination: d.start,
                source: s,
                data,
            });
        }
    }
    Ok(Rows {
        parts,
        rows: valid.len(),
        reused_rows,
        wait_s,
    })
}
impl Team {
    pub fn first_input(&self, hints: Vec<Hint>) -> Option<Arc<First>> {
        self.host_hasher.as_ref().map(|hasher| {
            let layers: Vec<usize> = hasher.layout().layers.iter().map(|l| l.layer).collect();
            First::new(&layers, hints)
        })
    }
    pub fn forecast_input(
        &self,
        owner: usize,
        offset: usize,
        input: &[u32],
    ) -> Option<Arc<Window>> {
        let hasher = Arc::clone(self.host_hasher.as_ref()?);
        let tables = Arc::clone(self.host_tables.as_ref()?);
        let end = input.len().min(offset + self.rows);
        (offset < end)
            .then(|| {
                Window::new(
                    hasher,
                    tables,
                    owner,
                    offset,
                    input[offset.saturating_sub(3)..offset].to_vec(),
                    input[offset..end].to_vec(),
                    Arc::clone(&self.host_prepare),
                )
                .ok()
            })
            .flatten()
    }
}

#[cfg(test)]
mod tests {
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
        let mut store1 =
            Int8Rows::open(&path, "w", &path, "s", Shard { first: 0, rows: 48 }).unwrap();
        let mut store14 =
            Int8Rows::open(&path, "w", &path, "s", Shard { first: 0, rows: 48 }).unwrap();
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
                                let r14 =
                                    first.get(14, store14, || Ok(ids14.clone()), valid).unwrap();
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
}
