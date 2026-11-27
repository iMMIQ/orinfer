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
type PrimaryPrefetch = (Vec<u64>, Vec<bool>, Rows);

pub(super) struct First {
    pub layer: usize,
    hints: Arc<Vec<Hint>>,
    pub prepared: Arc<Prepared<Rows>>,
    primary_prefetch: Arc<Prepared<PrimaryPrefetch>>,
    primary_started: AtomicBool,
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
            prepared: Arc::new(Prepared::deferred()),
            primary_prefetch: Arc::new(Prepared::deferred()),
            primary_started: AtomicBool::new(false),
            extras,
        })
    }
    pub fn handles_layer(&self, layer: usize) -> bool {
        layer == self.layer || self.extras.contains_key(&layer)
    }
    /// Launch background gather for the primary Engram layer (e.g. layer 1) so
    /// decode verification RAM reads overlap with verification IPC and layer 0.
    /// `get()` validates exact `(ids, valid)` identity before reusing it.
    pub fn prefetch_primary(
        &self,
        tables: Arc<BTreeMap<usize, SharedInt8Rows>>,
        ids: Vec<u64>,
        valid: Vec<bool>,
    ) {
        if rayon::current_num_threads() < 2
            || self.prepared.is_ready()
            || self
                .primary_started
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
        let Some(lane) = extra_prefetch_lane() else {
            return;
        };
        let layer = self.layer;
        let slot = Arc::clone(&self.primary_prefetch);
        let hints = Arc::clone(&self.hints);
        let _ = lane.try_submit(move || {
            let _ = slot.get_or(|| {
                let store = tables.get(&layer).ok_or("Engram table missing")?;
                let rows = assemble(layer, store, &ids, &valid, &hints)?;
                Ok((ids, valid, rows))
            });
        });
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
        if layer == self.layer {
            if self.primary_started.load(Ordering::Acquire)
                && let Ok((cached_ids, cached_valid, cached_rows)) = self.primary_prefetch.get()
            {
                let live_ids = ids()?;
                if *cached_ids == live_ids && cached_valid.as_slice() == valid {
                    return Ok(cached_rows);
                }
                return self
                    .prepared
                    .get_or(|| assemble(layer, store, &live_ids, valid, &self.hints));
            }
            return self
                .prepared
                .get_or(|| assemble(layer, store, &ids()?, valid, &self.hints));
        }
        let slot = self
            .extras
            .get(&layer)
            .map(|(p, _)| p.as_ref())
            .ok_or("Engram layer not prepared in host inputs")?;
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
    /// Pre-hash and start background Engram gather for the primary layer (layer 1)
    /// on small decode/verification packets before NPU submission starts.
    pub fn prefetch_first_primary(&self, first: &First, segments: &[(&[u32], &[u32])]) {
        let Some(hasher) = self.host_hasher.as_ref() else {
            return;
        };
        let Some(tables) = self
            .host_tables
            .as_ref()
            .and_then(|shared| shared.get())
            .and_then(|res| res.as_ref().ok())
            .cloned()
        else {
            return;
        };
        let mut hashes = Vec::new();
        let mut valid = Vec::new();
        for &(history, tokens) in segments {
            let mut h = Vec::new();
            let mut v = Vec::new();
            if hasher.hash_into(history, tokens, &mut h, &mut v).is_err() {
                return;
            }
            hashes.extend(h);
            valid.extend(v);
        }
        if valid.is_empty() {
            return;
        }
        if let Some(layer_index) = hasher
            .layout()
            .layers
            .iter()
            .position(|l| l.layer == first.layer)
        {
            let ids = layer_ids_for(hasher, layer_index, &hashes, &valid);
            first.prefetch_primary(tables, ids, valid);
        }
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
#[path = "host_inputs_tests.rs"]
mod tests;
