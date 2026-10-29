//! Immutable conditional-memory row stores and deterministic ownership plans.
use crate::artifact::Result;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::HashMap,
    fs::File,
    path::Path,
    sync::{Arc, Mutex},
};
mod cache;
pub mod validation;
use cache::RowCache;

#[derive(Clone, Copy, Debug)]
pub struct Shard {
    pub first: u64,
    pub rows: u64,
}
impl Shard {
    pub fn for_rank(rows: u64, ranks: usize, rank: usize) -> Result<Self> {
        if rows == 0 || ranks == 0 || rank >= ranks {
            return Err("Invalid row shard topology".into());
        }
        let span = rows.div_ceil(ranks as u64);
        let first = span
            .checked_mul(rank as u64)
            .ok_or("Row shard overflow")?
            .min(rows);
        Ok(Self {
            first,
            rows: span.min(rows - first),
        })
    }
    fn local(self, row: u64) -> Result<usize> {
        let local = row
            .checked_sub(self.first)
            .filter(|&i| i < self.rows)
            .ok_or("Row is outside this host shard")?;
        usize::try_from(local).map_err(|_| "Host row offset overflow".into())
    }
}

/// Unique rows grouped by owner. restore maps every original query to its
/// unique row, preserving repetitions and input order across communication.
pub struct RowPlan {
    pub rows: Vec<u64>,
    pub restore: Vec<usize>,
    pub starts: Vec<usize>,
}
impl RowPlan {
    pub fn build(ids: &[i64], table_rows: u64, ranks: usize, max_queries: usize) -> Result<Self> {
        if ids.len() > max_queries || ranks == 0 || ranks > 1024 || table_rows == 0 {
            return Err("Invalid host-row plan dimensions or query budget".into());
        }
        let span = table_rows.div_ceil(ranks as u64);
        let mut unique = Vec::with_capacity(ids.len());
        let mut slots = HashMap::with_capacity(ids.len());
        let mut original = Vec::with_capacity(ids.len());
        let mut counts = vec![0usize; ranks];
        for &id in ids {
            let row = u64::try_from(id)
                .ok()
                .filter(|&n| n < table_rows)
                .ok_or("Conditional-memory row outside table")?;
            let slot = *slots.entry(row).or_insert_with(|| {
                let slot = unique.len();
                unique.push(row);
                counts[(row / span) as usize] += 1;
                slot
            });
            original.push(slot);
        }
        let mut starts = vec![0usize; ranks + 1];
        for i in 0..ranks {
            starts[i + 1] = starts[i] + counts[i];
        }
        let mut cursor = starts[..ranks].to_vec();
        let mut rows = vec![0u64; unique.len()];
        let mut positions = vec![0usize; unique.len()];
        for (slot, &row) in unique.iter().enumerate() {
            let owner = (row / span) as usize;
            let position = cursor[owner];
            cursor[owner] += 1;
            rows[position] = row;
            positions[slot] = position;
        }
        let restore = original.into_iter().map(|slot| positions[slot]).collect();
        Ok(Self {
            rows,
            restore,
            starts,
        })
    }
    pub fn for_rank(&self, rank: usize) -> Result<&[u64]> {
        if rank >= self.starts.len().saturating_sub(1) {
            return Err("Host-row rank outside plan".into());
        }
        Ok(&self.rows[self.starts[rank]..self.starts[rank + 1]])
    }
}

struct Plane {
    file: File,
    start: usize,
    row_bytes: usize,
    rows: usize,
}
impl Plane {
    fn open(path: &Path, tensor: &str, dtype: Dtype) -> Result<Self> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: Deployment assets are immutable for this read-only mapping's
        // lifetime. The header mapping is dropped before any row reads.
        let mapping = unsafe { MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
        let (header, metadata) = SafeTensors::read_metadata(&mapping).map_err(|e| e.to_string())?;
        let info = metadata.info(tensor).ok_or("Missing host-row tensor")?;
        if info.dtype != dtype || info.shape.len() != 2 || info.shape.contains(&0) {
            return Err("Invalid host-row tensor dtype/shape".into());
        }
        let rows = info.shape[0];
        let row_bytes = (info.data_offsets.1 - info.data_offsets.0) / rows;
        let start = 8 + header + info.data_offsets.0;
        drop(mapping);
        Ok(Self {
            file,
            start,
            row_bytes,
            rows,
        })
    }
    /// pread one row into an exact slice. ZFS mmap page-fault readiness
    /// contends badly when several rank threads randomly read a
    /// multi-hundred-GiB table, while positional reads resolve through the
    /// ARC without that contention.
    fn read_row_at(&self, row: usize, target: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        if target.len() != self.row_bytes {
            return Err("Host-row read target extent differs".into());
        }
        let begin = self.start + row * self.row_bytes;
        self.file
            .read_exact_at(target, begin as u64)
            .map_err(|e| e.to_string())
    }
}

/// Exact byte-preserving INT8 rows plus FP32 group scales, read positionally.
pub struct Int8Rows {
    weight: Plane,
    scale: Plane,
    shard: Shard,
    row_bytes: usize,
}

/// Replicated device workers share one immutable row packet. Keep only the
/// latest query per store; changing query order, repetitions or owner rows
/// replaces the cache without invalidating packets still held by consumers.
/// An optional bounded row cache survives these packet replacements.
pub struct SharedInt8Rows {
    store: Int8Rows,
    state: Mutex<SharedState>,
    report: bool,
}
#[derive(Clone, Default, serde::Serialize)]
pub struct HostRowCacheStatistics {
    pub gathers: u64,
    pub packet_hits: u64,
    pub unique_row_hits: u64,
    pub unique_row_reads: u64,
    pub entries: usize,
    pub reserved_bytes: usize,
}
struct SharedState {
    last: Option<RowBatch>,
    cache: Option<RowCache>,
    statistics: HostRowCacheStatistics,
}
struct RowBatch {
    ids: Vec<u64>,
    bytes: Arc<Vec<u8>>,
}
impl SharedInt8Rows {
    pub fn new(store: Int8Rows) -> Self {
        Self {
            store,
            state: Mutex::new(SharedState {
                last: None,
                cache: None,
                statistics: HostRowCacheStatistics::default(),
            }),
            report: std::env::var_os("ORINFER_HOST_ROW_CACHE_REPORT").is_some(),
        }
    }
    /// The row budget includes payload, tags, ages and allocation overhead.
    /// The most recent packet and caller-owned staging remain separately bounded.
    pub fn with_cache(store: Int8Rows, budget: usize) -> Result<Self> {
        let cache = RowCache::new(store.row_bytes(), budget)?;
        let mut result = Self::new(store);
        result
            .state
            .get_mut()
            .map_err(|_| "Host cache poisoned")?
            .cache = cache;
        Ok(result)
    }
    pub fn statistics(&self) -> Result<HostRowCacheStatistics> {
        let state = self.state.lock().map_err(|_| "Host cache poisoned")?;
        let mut statistics = state.statistics.clone();
        if let Some(cache) = &state.cache {
            statistics.entries = cache.entries();
            statistics.reserved_bytes = cache.bytes();
        }
        Ok(statistics)
    }
    pub fn gather(&self, ids: &[u64], max_bytes: usize) -> Result<Arc<Vec<u8>>> {
        ids.len()
            .checked_mul(self.store.row_bytes())
            .filter(|&n| n <= max_bytes)
            .ok_or("Host-row staging budget exceeded")?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Shared host-row reader poisoned")?;
        if let Some(bytes) = state
            .last
            .as_ref()
            .filter(|batch| batch.ids == ids)
            .map(|batch| Arc::clone(&batch.bytes))
        {
            state.statistics.gathers += 1;
            state.statistics.packet_hits += 1;
            return Ok(bytes);
        }
        // Hold the cache lock across the parallel gather so simultaneous rank
        // misses join one read rather than submitting duplicate random I/O.
        let mut bytes = vec![];
        let (hits, reads) =
            self.store
                .gather_cached_into(ids, &mut bytes, max_bytes, state.cache.as_mut())?;
        state.statistics.gathers += 1;
        state.statistics.unique_row_hits += hits as u64;
        state.statistics.unique_row_reads += reads as u64;
        let bytes = Arc::new(bytes);
        state.last = Some(RowBatch {
            ids: ids.to_vec(),
            bytes: Arc::clone(&bytes),
        });
        Ok(bytes)
    }
}
impl Drop for SharedInt8Rows {
    fn drop(&mut self) {
        if self.report
            && let Ok(statistics) = self.statistics()
        {
            eprintln!(
                "HOST_ROWS_CACHE table_rows={} {}",
                self.store.shard.rows,
                serde_json::to_string(&statistics).unwrap_or_default()
            );
        }
    }
}
impl Int8Rows {
    pub fn open(
        weight: &Path,
        weight_tensor: &str,
        scale: &Path,
        scale_tensor: &str,
        shard: Shard,
    ) -> Result<Self> {
        let weight = Plane::open(weight, weight_tensor, Dtype::I8)?;
        let scale = Plane::open(scale, scale_tensor, Dtype::F32)?;
        if weight.rows != scale.rows
            || weight.row_bytes % 32 != 0
            || scale.row_bytes != weight.row_bytes / 32 * 4
            || shard
                .first
                .checked_add(shard.rows)
                .is_none_or(|end| end > weight.rows as u64)
        {
            return Err("Host INT8 row/scale/shard geometry differs".into());
        }
        let row_bytes = weight
            .row_bytes
            .checked_add(scale.row_bytes)
            .ok_or("Host row packet size overflow")?;
        Ok(Self {
            weight,
            scale,
            shard,
            row_bytes,
        })
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    /// Packet layout follows query order, with INT8 values followed by FP32 scales.
    /// All IDs and the budget are checked before modifying the destination.
    /// Rows resolve in parallel: the backing table is a random-read store,
    /// and cache-miss stalls from sequential reads would idle the device.
    pub fn gather_into(&self, ids: &[u64], out: &mut Vec<u8>, max_bytes: usize) -> Result<()> {
        self.gather_cached_into(ids, out, max_bytes, None)
            .map(|_| ())
    }
    fn gather_cached_into(
        &self,
        ids: &[u64],
        out: &mut Vec<u8>,
        max_bytes: usize,
        mut cache: Option<&mut RowCache>,
    ) -> Result<(usize, usize)> {
        use rayon::prelude::*;
        let bytes = ids
            .len()
            .checked_mul(self.row_bytes)
            .filter(|&n| n <= max_bytes)
            .ok_or("Host-row staging budget exceeded")?;
        let mut slots = HashMap::with_capacity(ids.len().min(4096));
        let mut unique = Vec::new();
        let mut restore = Vec::with_capacity(ids.len());
        for &row in ids {
            self.shard.local(row)?;
            let slot = *slots.entry(row).or_insert_with(|| {
                let slot = unique.len();
                unique.push(row);
                slot
            });
            restore.push(slot);
        }
        out.clear();
        out.resize(bytes, 0);
        let weight_bytes = self.weight.row_bytes;
        let mut hits = vec![false; unique.len()];
        for (index, &row) in unique.iter().enumerate() {
            if let Some(data) = cache.as_deref_mut().and_then(|cache| cache.get(row)) {
                out[index * self.row_bytes..(index + 1) * self.row_bytes].copy_from_slice(data);
                hits[index] = true;
            }
        }
        let result: Result<()> = out[..unique.len() * self.row_bytes]
            .par_chunks_mut(self.row_bytes)
            .enumerate()
            .try_for_each(|(index, packet)| {
                if hits[index] {
                    return Ok(());
                }
                let row = usize::try_from(unique[index]).map_err(|_| "Host row offset overflow")?;
                let (values, scales) = packet.split_at_mut(weight_bytes);
                self.weight.read_row_at(row, values)?;
                self.scale.read_row_at(row, scales)
            });
        result?;
        if let Some(cache) = cache {
            for (index, &row) in unique.iter().enumerate() {
                if !hits[index] {
                    cache.insert(
                        row,
                        &out[index * self.row_bytes..(index + 1) * self.row_bytes],
                    );
                }
            }
        }
        // First-appearance slots never exceed their destination indices.
        // Expanding backwards preserves every unread compact source row and
        // needs no second payload allocation.
        for (destination, &source) in restore.iter().enumerate().rev() {
            if destination != source {
                let begin = source * self.row_bytes;
                out.copy_within(begin..begin + self.row_bytes, destination * self.row_bytes);
            }
        }
        let hit_count = hits.iter().filter(|&&hit| hit).count();
        Ok((hit_count, unique.len() - hit_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
