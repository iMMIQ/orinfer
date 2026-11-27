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
mod resident;
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

struct Part {
    file: File,
    start: usize,
    first_byte: usize,
    bytes: usize,
}
struct Plane {
    parts: Vec<Part>,
    row_bytes: usize,
    rows: usize,
    resident: Option<(usize, Vec<u8>)>,
}
impl Drop for Plane {
    fn drop(&mut self) {
        if let Some((_, data)) = &mut self.resident {
            resident::discard(data);
        }
    }
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
            parts: vec![Part {
                file,
                start,
                first_byte: 0,
                bytes: rows * row_bytes,
            }],
            row_bytes,
            rows,
            resident: None,
        })
    }
    fn open_parts(sources: &[(std::path::PathBuf, String)], dtype: Dtype) -> Result<Self> {
        let mut result: Option<Self> = None;
        for (path, name) in sources {
            let mut plane = Self::open(path, name, dtype)?;
            if let Some(total) = &mut result {
                if total.row_bytes != plane.row_bytes {
                    return Err("Host-row shard strides differ".into());
                }
                plane.parts[0].first_byte = total.rows * total.row_bytes;
                total.rows = total
                    .rows
                    .checked_add(plane.rows)
                    .ok_or("Host-row count overflow")?;
                total
                    .rows
                    .checked_mul(total.row_bytes)
                    .ok_or("Host-row extent overflow")?;
                total.parts.append(&mut plane.parts);
            } else {
                result = Some(plane);
            }
        }
        result.ok_or_else(|| "Empty host-row shard list".into())
    }
    fn read_bytes(&self, mut offset: usize, mut target: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        while !target.is_empty() {
            let index = self
                .parts
                .partition_point(|p| p.first_byte <= offset)
                .checked_sub(1)
                .ok_or("Host-row offset outside shards")?;
            let part = &self.parts[index];
            let local = offset - part.first_byte;
            if local >= part.bytes {
                return Err("Host-row extent outside shards".into());
            }
            let count = target.len().min(part.bytes - local);
            part.file
                .read_exact_at(&mut target[..count], (part.start + local) as u64)
                .map_err(|e| e.to_string())?;
            offset += count;
            target = &mut target[count..];
        }
        Ok(())
    }
    fn load_rows(&mut self, first: usize, rows: usize) -> Result<()> {
        use rayon::prelude::*;
        if first > self.rows || rows > self.rows - first {
            return Err("Resident host table range exceeds source".into());
        }
        let bytes = rows * self.row_bytes;
        let offset = first * self.row_bytes;
        let mut data = resident::zeroed(bytes)?;
        // Bound first-touch/file-I/O concurrency while device weights load.
        // Stripes retain sequential reads and reuse the existing Rayon pool.
        let readers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(8);
        let chunk = 32 << 20;
        let stripe = bytes
            .div_ceil(chunk)
            .div_ceil(readers)
            .max(1)
            .checked_mul(chunk)
            .ok_or("Host row stripe extent overflow")?;
        data.par_chunks_mut(stripe)
            .enumerate()
            .try_for_each(|(index, target)| {
                for (part, target) in target.chunks_mut(chunk).enumerate() {
                    self.read_bytes(offset + index * stripe + part * chunk, target)?;
                }
                Ok::<_, String>(())
            })?;
        self.resident = Some((first, data));
        Ok(())
    }
    fn resident_slice(&self, row: usize) -> Result<&[u8]> {
        let (first, data) = self
            .resident
            .as_ref()
            .ok_or("Host-row plane is not resident")?;
        let begin = row
            .checked_sub(*first)
            .and_then(|v| v.checked_mul(self.row_bytes))
            .ok_or("Resident host row outside table")?;
        data.get(begin..begin + self.row_bytes)
            .ok_or_else(|| "Resident host row outside table".into())
    }
    /// pread one row into an exact slice. ZFS mmap page-fault readiness
    /// contends badly when several rank threads randomly read a
    /// multi-hundred-GiB table, while positional reads resolve through the
    /// ARC without that contention.
    fn read_row_at(&self, row: usize, target: &mut [u8]) -> Result<()> {
        if row >= self.rows || target.len() != self.row_bytes {
            return Err("Host-row read exceeds source or target extent".into());
        }
        if self.resident.is_some() {
            target.copy_from_slice(self.resident_slice(row)?);
            return Ok(());
        }
        self.read_bytes(row * self.row_bytes, target)
    }
}

/// Exact byte-preserving INT8 rows plus FP32 group scales, read positionally.
pub struct Int8Rows {
    storage: RowStorage,
    shard: Shard,
    row_bytes: usize,
}
enum RowStorage {
    Split(Plane, Plane),
    Records(Plane),
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
    pub resident_bytes: usize,
}
struct SharedState {
    last: Option<RowBatch>,
    last_split: Option<SplitBatch>,
    cache: Option<RowCache>,
    statistics: HostRowCacheStatistics,
}
struct RowBatch {
    ids: Vec<u64>,
    bytes: Arc<Vec<u8>>,
}
struct SplitBatch {
    ids: Vec<u64>,
    planes: Arc<(Vec<u8>, Vec<u8>)>,
}
impl SharedInt8Rows {
    pub fn new(store: Int8Rows) -> Self {
        Self {
            store,
            state: Mutex::new(SharedState {
                last: None,
                last_split: None,
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
        statistics.resident_bytes = self.store.resident_bytes();
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
    /// Gather separate INT8 values and FP32 scales directly, reusing the latest
    /// query packet across replicated TP workers without an intermediate record copy.
    pub fn gather_split(&self, ids: &[u64], max_bytes: usize) -> Result<Arc<(Vec<u8>, Vec<u8>)>> {
        ids.len()
            .checked_mul(self.store.row_bytes())
            .filter(|&n| n <= max_bytes)
            .ok_or("Host-row staging budget exceeded")?;
        let use_lockfree_resident = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "Shared host-row reader poisoned")?;
            if let Some(planes) = state
                .last_split
                .as_ref()
                .filter(|batch| batch.ids == ids)
                .map(|batch| Arc::clone(&batch.planes))
            {
                state.statistics.gathers += 1;
                state.statistics.packet_hits += 1;
                return Ok(planes);
            }
            self.store.is_resident() && state.cache.is_none()
        };
        if use_lockfree_resident {
            // Resident tables are immutable; release the mutex during parallel
            // gather so forecast prefetches never block live rank gathers.
            let (values, scales) = self.store.gather_split_resident(ids, max_bytes)?;
            let planes = Arc::new((values, scales));
            let mut state = self
                .state
                .lock()
                .map_err(|_| "Shared host-row reader poisoned")?;
            state.statistics.gathers += 1;
            state.statistics.unique_row_reads += ids.len() as u64;
            state.last_split = Some(SplitBatch {
                ids: ids.to_vec(),
                planes: Arc::clone(&planes),
            });
            return Ok(planes);
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| "Shared host-row reader poisoned")?;
        if let Some(planes) = state
            .last_split
            .as_ref()
            .filter(|batch| batch.ids == ids)
            .map(|batch| Arc::clone(&batch.planes))
        {
            state.statistics.gathers += 1;
            state.statistics.packet_hits += 1;
            return Ok(planes);
        }
        let mut packed = vec![];
        let (hits, reads) =
            self.store
                .gather_cached_into(ids, &mut packed, max_bytes, state.cache.as_mut())?;
        let (val_width, scale_width) = self.store.split_widths();
        let row_width = self.store.row_bytes();
        let mut values = Vec::with_capacity(ids.len() * val_width);
        let mut scales = Vec::with_capacity(ids.len() * scale_width);
        for record in packed.chunks_exact(row_width) {
            values.extend_from_slice(&record[..val_width]);
            scales.extend_from_slice(&record[val_width..]);
        }
        state.statistics.gathers += 1;
        state.statistics.unique_row_hits += hits as u64;
        state.statistics.unique_row_reads += reads as u64;
        let planes = Arc::new((values, scales));
        state.last_split = Some(SplitBatch {
            ids: ids.to_vec(),
            planes: Arc::clone(&planes),
        });
        Ok(planes)
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
        Self::open_sharded(
            &[(weight.to_owned(), weight_tensor.to_owned())],
            &[(scale.to_owned(), scale_tensor.to_owned())],
            shard,
        )
    }
    pub(crate) fn open_sharded(
        weight: &[(std::path::PathBuf, String)],
        scale: &[(std::path::PathBuf, String)],
        shard: Shard,
    ) -> Result<Self> {
        let weight = Plane::open_parts(weight, Dtype::I8)?;
        let scale = Plane::open_parts(scale, Dtype::F32)?;
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
            storage: RowStorage::Split(weight, scale),
            shard,
            row_bytes,
        })
    }
    /// Offline records contain exact INT8 group-32 values followed by FP32 scales.
    pub(crate) fn open_records(
        sources: &[(std::path::PathBuf, String)],
        shard: Shard,
    ) -> Result<Self> {
        let records = Plane::open_parts(sources, Dtype::U8)?;
        if !records.row_bytes.is_multiple_of(36)
            || shard
                .first
                .checked_add(shard.rows)
                .is_none_or(|end| end > records.rows as u64)
        {
            return Err("Host INT8 record/shard geometry differs".into());
        }
        let row_bytes = records.row_bytes;
        Ok(Self {
            storage: RowStorage::Records(records),
            shard,
            row_bytes,
        })
    }
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub fn split_widths(&self) -> (usize, usize) {
        match &self.storage {
            RowStorage::Split(weight, scale) => (weight.row_bytes, scale.row_bytes),
            RowStorage::Records(records) => {
                let groups = records.row_bytes / 36;
                (groups * 32, groups * 4)
            }
        }
    }
    pub fn is_resident(&self) -> bool {
        match &self.storage {
            RowStorage::Split(weight, scale) => {
                weight.resident.is_some() && scale.resident.is_some()
            }
            RowStorage::Records(records) => records.resident.is_some(),
        }
    }
    /// Direct parallel gather from RAM-resident tables straight into separated
    /// INT8 values and FP32 scales, bypassing HashMap dedup and intermediate staging.
    fn gather_split_resident(&self, ids: &[u64], max_bytes: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        use rayon::prelude::*;
        use std::mem::MaybeUninit;
        ids.len()
            .checked_mul(self.row_bytes)
            .filter(|&n| n <= max_bytes)
            .ok_or("Host-row staging budget exceeded")?;
        let (val_width, scale_width) = self.split_widths();
        let val_bytes = ids.len() * val_width;
        let scale_bytes = ids.len() * scale_width;
        if ids.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        for &row in ids {
            self.shard.local(row)?;
        }
        #[cfg(target_arch = "aarch64")]
        #[inline(always)]
        unsafe fn prefetch_read(ptr: *const u8) {
            // SAFETY: PRFM is a non-faulting architectural prefetch hint.
            unsafe {
                core::arch::asm!(
                    "prfm pldl1keep, [{ptr}]",
                    ptr = in(reg) ptr,
                    options(readonly, nostack, preserves_flags),
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        #[inline(always)]
        unsafe fn prefetch_read(_ptr: *const u8) {}

        let mut values = Vec::<u8>::with_capacity(val_bytes);
        let mut scales = Vec::<u8>::with_capacity(scale_bytes);
        let val_spare = &mut values.spare_capacity_mut()[..val_bytes];
        let scale_spare = &mut scales.spare_capacity_mut()[..scale_bytes];
        const PREFETCH_DIST: usize = 8;

        let fill_chunk = |chunk_ids: &[u64],
                          val_out: &mut [MaybeUninit<u8>],
                          scale_out: &mut [MaybeUninit<u8>]|
         -> Result<()> {
            match &self.storage {
                RowStorage::Records(records) => {
                    let (first, data) = records
                        .resident
                        .as_ref()
                        .ok_or("Host-row plane is not resident")?;
                    let row_bytes = records.row_bytes;
                    let base_ptr = data.as_ptr();
                    let data_len = data.len();
                    for (idx, ((&row, v_dst), s_dst)) in chunk_ids
                        .iter()
                        .zip(val_out.chunks_exact_mut(val_width))
                        .zip(scale_out.chunks_exact_mut(scale_width))
                        .enumerate()
                    {
                        if let Some(&next_row) = chunk_ids.get(idx + PREFETCH_DIST)
                            && let Ok(nr) = usize::try_from(next_row)
                            && let Some(nb) = nr
                                .checked_sub(*first)
                                .and_then(|v| v.checked_mul(row_bytes))
                            && nb + row_bytes <= data_len
                        {
                            // SAFETY: nb + row_bytes <= data_len bounds the prefetch addresses.
                            unsafe {
                                let p = base_ptr.add(nb);
                                prefetch_read(p);
                                if row_bytes > 64 {
                                    prefetch_read(p.add(64));
                                }
                                if row_bytes > 128 {
                                    prefetch_read(p.add(128));
                                }
                                if row_bytes > 192 {
                                    prefetch_read(p.add(192));
                                }
                                if row_bytes > 256 {
                                    prefetch_read(p.add(256));
                                }
                            }
                        }
                        let src = records.resident_slice(
                            usize::try_from(row).map_err(|_| "Host row offset overflow")?,
                        )?;
                        // SAFETY: src has length val_width + scale_width; destinations are exact chunks.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                src.as_ptr(),
                                v_dst.as_mut_ptr().cast::<u8>(),
                                val_width,
                            );
                            std::ptr::copy_nonoverlapping(
                                src.as_ptr().add(val_width),
                                s_dst.as_mut_ptr().cast::<u8>(),
                                scale_width,
                            );
                        }
                    }
                }
                RowStorage::Split(weight, scale) => {
                    let (w_first, w_data) = weight
                        .resident
                        .as_ref()
                        .ok_or("Host-row plane is not resident")?;
                    let (s_first, s_data) = scale
                        .resident
                        .as_ref()
                        .ok_or("Host-row plane is not resident")?;
                    for (idx, ((&row, v_dst), s_dst)) in chunk_ids
                        .iter()
                        .zip(val_out.chunks_exact_mut(val_width))
                        .zip(scale_out.chunks_exact_mut(scale_width))
                        .enumerate()
                    {
                        if let Some(&next_row) = chunk_ids.get(idx + PREFETCH_DIST)
                            && let Ok(nr) = usize::try_from(next_row)
                        {
                            if let Some(wb) = nr
                                .checked_sub(*w_first)
                                .and_then(|v| v.checked_mul(val_width))
                                && wb + val_width <= w_data.len()
                            {
                                // SAFETY: wb + val_width <= w_data.len().
                                unsafe {
                                    let p = w_data.as_ptr().add(wb);
                                    prefetch_read(p);
                                    if val_width > 64 {
                                        prefetch_read(p.add(64));
                                    }
                                    if val_width > 128 {
                                        prefetch_read(p.add(128));
                                    }
                                    if val_width > 192 {
                                        prefetch_read(p.add(192));
                                    }
                                }
                            }
                            if let Some(sb) = nr
                                .checked_sub(*s_first)
                                .and_then(|v| v.checked_mul(scale_width))
                                && sb + scale_width <= s_data.len()
                            {
                                // SAFETY: sb + scale_width <= s_data.len().
                                unsafe {
                                    prefetch_read(s_data.as_ptr().add(sb));
                                }
                            }
                        }
                        let r = usize::try_from(row).map_err(|_| "Host row offset overflow")?;
                        let w_src = weight.resident_slice(r)?;
                        let s_src = scale.resident_slice(r)?;
                        // SAFETY: w_src/s_src lengths match val_width/scale_width.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                w_src.as_ptr(),
                                v_dst.as_mut_ptr().cast::<u8>(),
                                val_width,
                            );
                            std::ptr::copy_nonoverlapping(
                                s_src.as_ptr(),
                                s_dst.as_mut_ptr().cast::<u8>(),
                                scale_width,
                            );
                        }
                    }
                }
            }
            Ok(())
        };
        if ids.len() < 384 {
            fill_chunk(ids, val_spare, scale_spare)?;
        } else {
            // Bound parallelism to <=16 contiguous stripes to avoid waking all
            // host cores across NUMA nodes on a few megabytes of RAM copies.
            let stripe = (ids.len() / 8).clamp(128, 2048);
            ids.par_chunks(stripe)
                .zip(val_spare.par_chunks_mut(stripe * val_width))
                .zip(scale_spare.par_chunks_mut(stripe * scale_width))
                .try_for_each(|((chunk_ids, v_out), s_out)| fill_chunk(chunk_ids, v_out, s_out))?;
        }
        // SAFETY: Every byte of val_spare[..val_bytes] and scale_spare[..scale_bytes]
        // was initialized by copy_nonoverlapping above without error.
        unsafe {
            values.set_len(val_bytes);
            scales.set_len(scale_bytes);
        }
        Ok((values, scales))
    }
    /// Payload and bounded allocation overhead for this shard, without staging.
    pub fn required_resident_bytes(&self) -> Result<usize> {
        usize::try_from(self.shard.rows)
            .ok()
            .and_then(|rows| rows.checked_mul(self.row_bytes))
            .and_then(|bytes| bytes.checked_add(4096 + std::mem::size_of::<Self>()))
            .ok_or_else(|| "Resident host table budget overflow".into())
    }
    pub fn resident_bytes(&self) -> usize {
        let payload = match &self.storage {
            RowStorage::Split(weight, scale) => match (&weight.resident, &scale.resident) {
                (Some((_, weight)), Some((_, scale))) => weight.capacity() + scale.capacity(),
                _ => return 0,
            },
            RowStorage::Records(records) => match &records.resident {
                Some((_, data)) => data.capacity(),
                None => return 0,
            },
        };
        payload + 4096 + std::mem::size_of::<Self>()
    }
    /// Read the immutable shard once at startup. Both planes keep their exact
    /// checkpoint bytes; shared workers retain this one owner through Arc.
    pub fn into_resident(mut self, budget: usize) -> Result<Self> {
        if self.required_resident_bytes()? > budget {
            return Err("Resident host table exceeds its budget".into());
        }
        if self.resident_bytes() != 0 {
            return if self.resident_bytes() <= budget {
                Ok(self)
            } else {
                Err("Resident host table allocation exceeds its budget".into())
            };
        }
        let first = usize::try_from(self.shard.first).map_err(|_| "Host shard offset overflow")?;
        let rows = usize::try_from(self.shard.rows).map_err(|_| "Host shard extent overflow")?;
        match &mut self.storage {
            RowStorage::Split(weight, scale) => {
                weight.load_rows(first, rows)?;
                scale.load_rows(first, rows)?;
            }
            RowStorage::Records(records) => records.load_rows(first, rows)?,
        }
        if self.resident_bytes() > budget {
            return Err("Resident host table allocation exceeds its budget".into());
        }
        Ok(self)
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
                match &self.storage {
                    RowStorage::Split(weight, scale) => {
                        let (values, scales) = packet.split_at_mut(weight.row_bytes);
                        weight.read_row_at(row, values)?;
                        scale.read_row_at(row, scales)
                    }
                    RowStorage::Records(records) => records.read_row_at(row, packet),
                }
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
#[path = "host_rows_tests.rs"]
mod tests;
