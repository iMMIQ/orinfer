//! Immutable first-Engram inputs shared by one packet and one bounded forecast.
use super::{SharedTables, team::Team};
use crate::{artifact::Result, host_rows::SharedInt8Rows, sampling::Prepared};
use orinfer_model_sdk::conditional_memory::Hasher;
use std::{
    ops::Range,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

const LANES: usize = 24;
const VALUES: usize = 256;
const SCALES: usize = 32;
const RECORD: usize = VALUES + SCALES;

struct Payload {
    values: Vec<u8>,
    scales: Vec<u8>,
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
                    &part.data.values[s.start * LANES * VALUES..s.end * LANES * VALUES],
                ),
                (
                    1,
                    part.destination * LANES * SCALES,
                    &part.data.scales[s.start * LANES * SCALES..s.end * LANES * SCALES],
                ),
                (2, part.destination, &part.data.mask[s.clone()]),
            ]
        })
    }
}
fn read(store: &SharedInt8Rows, ids: &[u64], valid: &[bool]) -> Result<Arc<Payload>> {
    let packed = store.gather(ids, ids.len() * RECORD)?;
    let mut values = Vec::with_capacity(ids.len() * VALUES);
    let mut scales = Vec::with_capacity(ids.len() * SCALES);
    for record in packed.as_chunks::<RECORD>().0 {
        values.extend_from_slice(&record[..VALUES]);
        scales.extend_from_slice(&record[VALUES..]);
    }
    Ok(Arc::new(Payload {
        values,
        scales,
        mask: valid.iter().map(|&v| u8::from(v)).collect(),
    }))
}
pub(super) struct Window {
    pub owner: usize,
    pub offset: usize,
    pub rows: usize,
    ids: Arc<Vec<u64>>,
    valid: Arc<Vec<bool>>,
    prepared: Arc<Prepared<Arc<Payload>>>,
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
        let ids = Arc::new(first_ids(&hasher, &hashes, &valid));
        let valid = Arc::new(valid);
        let layer = hasher.layout().layers[0].layer;
        let input_ids = Arc::clone(&ids);
        let input_valid = Arc::clone(&valid);
        let prepared = Prepared::new(move || {
            let stores = tables
                .get()
                .ok_or("Host tables not ready")?
                .as_ref()
                .map_err(Clone::clone)?;
            read(
                stores.get(&layer).ok_or("Engram table missing")?,
                &input_ids,
                &input_valid,
            )
        });
        Ok(Arc::new(Self {
            owner,
            offset,
            rows,
            ids,
            valid,
            prepared: Arc::new(prepared),
            active,
        }))
    }
    pub fn start(&self) {
        self.prepared.start(&self.active);
    }
    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        self.prepared.is_ready()
    }
}
pub(super) struct Hint {
    pub window: Arc<Window>,
    pub destination: Range<usize>,
    pub source: Range<usize>,
}
pub(super) struct First {
    pub layer: usize,
    hint: Option<Hint>,
    pub prepared: Prepared<Rows>,
}
impl First {
    pub fn new(layer: usize, hint: Option<Hint>) -> Arc<Self> {
        Arc::new(Self {
            layer,
            hint,
            prepared: Prepared::deferred(),
        })
    }
    pub fn get(
        &self,
        store: &SharedInt8Rows,
        ids: impl FnOnce() -> Result<Vec<u64>>,
        valid: &[bool],
    ) -> Result<&Rows> {
        self.prepared
            .get_or(|| assemble(store, &ids()?, valid, self.hint.as_ref()))
    }
}
fn first_ids(hasher: &Hasher, hashes: &[i64], valid: &[bool]) -> Vec<u64> {
    let width = hasher.values_per_token();
    valid
        .iter()
        .enumerate()
        .flat_map(|(row, &active)| {
            (0..LANES).map(move |lane| {
                if active {
                    hashes[row * width + lane] as u64
                } else {
                    0
                }
            })
        })
        .collect()
}
fn assemble(
    store: &SharedInt8Rows,
    ids: &[u64],
    valid: &[bool],
    hint: Option<&Hint>,
) -> Result<Rows> {
    let mut reused = None;
    let mut wait_s = 0.;
    if let Some(hint) = hint {
        let d = hint.destination.clone();
        let s = hint.source.clone();
        // Reject changed row identities or media before waiting on a forecast.
        if d.len() == s.len()
            && d.end <= valid.len()
            && s.end <= hint.window.rows
            && ids[d.start * LANES..d.end * LANES]
                == hint.window.ids[s.start * LANES..s.end * LANES]
            && valid[d.clone()] == hint.window.valid[s.clone()]
        {
            let started = Instant::now();
            if let Ok(data) = hint.window.prepared.get() {
                reused = Some((d, s, Arc::clone(data)));
            }
            wait_s = started.elapsed().as_secs_f64();
        }
    }
    let mut parts = vec![];
    let reused_rows = reused.as_ref().map_or(0, |(d, _, _)| d.len());
    if let Some((d, s, data)) = reused {
        let mut needed = Vec::with_capacity(ids.len() - d.len() * LANES);
        needed.extend_from_slice(&ids[..d.start * LANES]);
        needed.extend_from_slice(&ids[d.end * LANES..]);
        if !needed.is_empty() {
            let mut masks = valid[..d.start].to_vec();
            masks.extend_from_slice(&valid[d.end..]);
            let fallback = read(store, &needed, &masks)?;
            if d.start > 0 {
                parts.push(Part {
                    destination: 0,
                    source: 0..d.start,
                    data: Arc::clone(&fallback),
                });
            }
            if d.end < valid.len() {
                parts.push(Part {
                    destination: d.end,
                    source: d.start..masks.len(),
                    data: fallback,
                });
            }
        }
        parts.push(Part {
            destination: d.start,
            source: s,
            data,
        });
    } else {
        parts.push(Part {
            destination: 0,
            source: 0..valid.len(),
            data: read(store, ids, valid)?,
        });
    }
    Ok(Rows {
        parts,
        rows: valid.len(),
        reused_rows,
        wait_s,
    })
}
impl Team {
    pub fn first_input(&self, hint: Option<Hint>) -> Option<Arc<First>> {
        self.host_hasher
            .as_ref()
            .map(|hasher| First::new(hasher.layout().layers[0].layer, hint))
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
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    fn assets() -> (Arc<Hasher>, Arc<SharedTables>) {
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
        let store = Int8Rows::open(&path, "w", &path, "s", Shard { first: 0, rows: 48 }).unwrap();
        let tables = Arc::new(SharedTables::new());
        tables
            .set(Ok(Arc::new(BTreeMap::from([(
                1,
                SharedInt8Rows::new(store),
            )]))))
            .ok()
            .unwrap();
        (Arc::new(Hasher::new(layout).unwrap()), tables)
    }
    fn query(hasher: &Hasher, history: &[u32], tokens: &[u32]) -> (Vec<u64>, Vec<bool>) {
        let (mut hashes, mut valid) = (vec![], vec![]);
        hasher
            .hash_into(history, tokens, &mut hashes, &mut valid)
            .unwrap();
        (first_ids(hasher, &hashes, &valid), valid)
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
        let store = &tables.get().unwrap().as_ref().unwrap()[&1];
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
        let (mut ids, mut valid) = query(&hasher, &[9], &[10]);
        let (x, v) = query(&hasher, &[2, 3, 4], &[5, 15]);
        ids.extend(x);
        valid.extend(v);
        let (x, v) = query(&hasher, &[10, 11], &[7]);
        ids.extend(x);
        valid.extend(v);
        let first = First::new(
            1,
            Some(Hint {
                window,
                destination: 1..3,
                source: 1..3,
            }),
        );
        let calls = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                let (first, ids, valid, calls) = (&first, &ids, &valid, &calls);
                scope.spawn(move || {
                    let rows = first
                        .get(
                            store,
                            || {
                                calls.fetch_add(1, Ordering::Relaxed);
                                Ok(ids.clone())
                            },
                            valid,
                        )
                        .unwrap();
                    check(rows, ids, valid);
                    assert_eq!(rows.reused_rows, 2);
                });
            }
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        // Replacing the table's packet cache cannot change this frame's data.
        store.gather(&[47, 3], 2 * RECORD).unwrap();
        check(first.prepared.get().unwrap(), &ids, &valid);
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
            let (mut ids, mut valid) = query(&hasher, &[1, 2, 3], &[4, 5]);
            if mask {
                valid[0] = !valid[0];
            } else {
                ids[0] ^= 1;
            }
            let first = First::new(
                1,
                Some(Hint {
                    window: window.clone(),
                    destination: 0..2,
                    source: 0..2,
                }),
            );
            let rows = first.get(store, || Ok(ids.clone()), &valid).unwrap();
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
        let (ids, valid) = query(&hasher, &[], &[4]);
        let first = First::new(
            1,
            Some(Hint {
                window,
                destination: 0..1,
                source: 0..1,
            }),
        );
        let rows = first.get(store, || Ok(ids.clone()), &valid).unwrap();
        check(rows, &ids, &valid);
        assert_eq!(rows.reused_rows, 0);
        let bad = First::new(1, None);
        assert!(bad.get(store, || Ok(vec![49; LANES]), &[true]).is_err());
        assert!(
            bad.get(store, || panic!("failed frame must not run again"), &[true])
                .is_err()
        );
    }
}
