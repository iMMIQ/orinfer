//! Bounded CPU row cache for an immutable, previously verified IQ4_NL table.
//!
//! Only the safetensors header is touched when opening. The temporary metadata
//! mapping is dropped; rows are read with positional I/O, without pinning or
//! expanding the whole table. The cache budget includes its packed bytes and
//! fixed indexing arrays. Reclaimable OS file cache and per-batch output/I/O
//! buffers are separate from this explicit cache budget.
use crate::artifact::Result;
use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, fs::File, os::unix::fs::FileExt, path::Path};

const WAYS: usize = 4;
const EMPTY: u64 = u64::MAX;
// IQ4_NL's standardized nonlinear codebook, GGML type 20.
const VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Debug)]
pub struct Lookup {
    /// BF16 rows in the requested order, including duplicates.
    pub features: Vec<bf16>,
    pub cache_hits: usize,
    /// Unique rows read from the file; duplicate misses issue one read.
    pub disk_rows: usize,
}

pub struct Table {
    file: File,
    start: u64,
    rows: u64,
    width: usize,
    row_bytes: usize,
    workers: usize,
    cache: Vec<u8>,
    tags: Vec<u64>,
    used: Vec<u64>,
    clock: u64,
}

impl Table {
    /// The caller must verify the immutable artifact's identity before use.
    /// Logical dimensions and physical quantization metadata are checked here.
    pub fn open(
        path: &Path,
        tensor: &str,
        rows: u64,
        width: usize,
        cache_bytes: usize,
        workers: usize,
    ) -> Result<Self> {
        if rows == 0
            || rows == EMPTY
            || width == 0
            || !width.is_multiple_of(32)
            || !(1..=32).contains(&workers)
        {
            return Err("Invalid PLE table dimensions or I/O worker count".into());
        }
        let row_bytes = (width / 32)
            .checked_mul(18)
            .ok_or("PLE row size overflow")?;
        let file = File::open(path).map_err(|e| format!("PLE table: {e}"))?;
        // SAFETY: The caller keeps the prepared artifact immutable. No tensor
        // view escapes this scope, and the mapping is discarded after parsing.
        let mapping = unsafe { MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
        let (header, metadata) = SafeTensors::read_metadata(&mapping).map_err(|e| e.to_string())?;
        let info = metadata.info(tensor).ok_or("Missing PLE table tensor")?;
        let logical_rows = usize::try_from(rows).map_err(|_| "PLE row count overflow")?;
        let attributes = metadata
            .metadata()
            .as_ref()
            .ok_or("Missing PLE table metadata")?;
        let dimensions: Vec<u64> = serde_json::from_str(
            attributes
                .get(&format!("ggml.dimensions.{tensor}"))
                .ok_or("Missing PLE dimensions")?,
        )
        .map_err(|e| e.to_string())?;
        if info.dtype != Dtype::U8
            || info.shape != [logical_rows, row_bytes]
            || dimensions != [width as u64, rows]
            || attributes
                .get(&format!("ggml.type.{tensor}"))
                .map(String::as_str)
                != Some("20")
            || attributes
                .get(&format!("orin.layout.{tensor}"))
                .map(String::as_str)
                != Some("ggml_20")
        {
            return Err("PLE table must have the declared IQ4_NL row layout".into());
        }
        let start = (8 + header + info.data_offsets.0) as u64;
        drop(mapping);
        let entry_bytes = row_bytes.checked_add(16).ok_or("PLE cache size overflow")?;
        let entries = (cache_bytes / entry_bytes / WAYS) * WAYS;
        Ok(Self {
            file,
            start,
            rows,
            width,
            row_bytes,
            workers,
            cache: vec![0; entries * row_bytes],
            tags: vec![EMPTY; entries],
            used: vec![0; entries],
            clock: 0,
        })
    }

    pub fn cache_bytes(&self) -> usize {
        self.cache.len() + (self.tags.len() + self.used.len()) * size_of::<u64>()
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    fn set(&self, row: u64) -> std::ops::Range<usize> {
        // Mix sequential and prime-modulo row IDs before selecting a cache set.
        let hash = super::splitmix64(row);
        let start = (hash as usize % (self.tags.len() / WAYS)) * WAYS;
        start..start + WAYS
    }

    fn tick(&mut self) -> u64 {
        if self.clock == u64::MAX {
            self.used.fill(0);
            self.clock = 0;
        }
        self.clock += 1;
        self.clock
    }

    fn cached(&mut self, row: u64) -> Option<usize> {
        if self.tags.is_empty() {
            return None;
        }
        let slot = self.set(row).find(|&slot| self.tags[slot] == row)?;
        self.used[slot] = self.tick();
        Some(slot * self.row_bytes)
    }

    fn insert(&mut self, row: u64, bytes: &[u8]) {
        if self.tags.is_empty() {
            return;
        }
        let slot = self.set(row).min_by_key(|&slot| self.used[slot]).unwrap();
        let start = slot * self.row_bytes;
        self.cache[start..start + self.row_bytes].copy_from_slice(bytes);
        self.tags[slot] = row;
        self.used[slot] = self.tick();
    }

    /// Request order is preserved. Sorting and deduplicating misses improves
    /// locality; bounded parallel reads overlap independent storage requests.
    /// Invalid row IDs are rejected before reads or changes to the cache.
    pub fn lookup(&mut self, ids: &[u64]) -> Result<Lookup> {
        if ids.iter().any(|&row| row >= self.rows) {
            return Err("PLE row outside table".into());
        }
        let length = ids
            .len()
            .checked_mul(self.width)
            .ok_or("PLE batch size overflow")?;
        let mut result = Lookup {
            features: vec![bf16::ZERO; length],
            cache_hits: 0,
            disk_rows: 0,
        };
        let mut missing = BTreeMap::<u64, Vec<usize>>::new();
        for (index, &row) in ids.iter().enumerate() {
            if let Some(start) = self.cached(row) {
                decode(
                    &self.cache[start..start + self.row_bytes],
                    &mut result.features[index * self.width..(index + 1) * self.width],
                )?;
                result.cache_hits += 1;
            } else {
                missing.entry(row).or_default().push(index);
            }
        }
        let tasks: Vec<u64> = missing.keys().copied().collect();
        result.disk_rows = tasks.len();
        if tasks.is_empty() {
            return Ok(result);
        }
        let file = &self.file;
        let row_bytes = self.row_bytes;
        let start = self.start;
        let chunk = tasks.len().div_ceil(self.workers);
        let reads = std::thread::scope(|scope| {
            let jobs: Vec<_> = tasks
                .chunks(chunk)
                .map(|rows| {
                    scope.spawn(move || {
                        let mut data = vec![0; rows.len() * row_bytes];
                        for (&row, bytes) in rows.iter().zip(data.chunks_mut(row_bytes)) {
                            file.read_exact_at(bytes, start + row * row_bytes as u64)
                                .map_err(|e| format!("PLE row {row}: {e}"))?;
                        }
                        Ok::<_, String>(data)
                    })
                })
                .collect();
            jobs.into_iter()
                .map(|job| {
                    job.join()
                        .map_err(|_| "PLE I/O worker panicked".to_string())?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for (rows, data) in tasks.chunks(chunk).zip(reads) {
            for (&row, bytes) in rows.iter().zip(data.chunks(row_bytes)) {
                for &index in &missing[&row] {
                    decode(
                        bytes,
                        &mut result.features[index * self.width..(index + 1) * self.width],
                    )?;
                }
                self.insert(row, bytes);
            }
        }
        Ok(result)
    }
}

fn decode(bytes: &[u8], output: &mut [bf16]) -> Result<()> {
    for (packed, values) in bytes
        .as_chunks::<18>()
        .0
        .iter()
        .zip(output.as_chunks_mut::<32>().0.iter_mut())
    {
        let scale = f16::from_bits(u16::from_le_bytes([packed[0], packed[1]])).to_f32();
        if !scale.is_finite() {
            return Err("Nonfinite PLE IQ4_NL scale".into());
        }
        for (j, &code) in packed[2..].iter().enumerate() {
            values[j] = bf16::from_f32(scale * f32::from(VALUES[usize::from(code & 15)]));
            values[j + 16] = bf16::from_f32(scale * f32::from(VALUES[usize::from(code >> 4)]));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::{
        collections::HashMap,
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new(kind: u32) -> Self {
            Self::with_rows(kind, 8)
        }
        fn with_rows(kind: u32, rows: usize) -> Self {
            let path = std::env::temp_dir().join(format!(
                "orin-ple-{}-{}.safetensors",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut data = Vec::new();
            for row in 0..rows {
                data.extend(
                    f16::from_f32(if row % 2 == 0 { 0.5 } else { -0.25 })
                        .to_bits()
                        .to_le_bytes(),
                );
                data.extend((0u8..16).map(|j| j | ((15 - j) << 4)));
            }
            let tensor = TensorView::new(Dtype::U8, vec![rows, 18], &data).unwrap();
            let metadata = HashMap::from([
                ("ggml.type.P".into(), kind.to_string()),
                ("ggml.dimensions.P".into(), format!("[32,{rows}]")),
                ("orin.layout.P".into(), format!("ggml_{kind}")),
            ]);
            fs::write(
                &path,
                safetensors::serialize([("P", tensor)], Some(metadata)).unwrap(),
            )
            .unwrap();
            Self(path)
        }
        fn table(&self, budget: usize, workers: usize) -> Table {
            Table::open(&self.0, "P", 8, 32, budget, workers).unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_file(&self.0).unwrap();
        }
    }

    #[test]
    fn nonlinear_codes_signed_scales_order_and_duplicate_reads() {
        let fixture = Fixture::new(20);
        let mut table = fixture.table(1024, 3);
        let result = table.lookup(&[1, 0, 1]).unwrap();
        assert_eq!(result.disk_rows, 2);
        assert_eq!(result.cache_hits, 0);
        let expected = [
            -127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0,
            69.0, 89.0, 113.0,
        ];
        for (row, scale) in [-0.25, 0.5, -0.25].into_iter().enumerate() {
            for j in 0..16 {
                assert_eq!(
                    result.features[row * 32 + j],
                    bf16::from_f32(expected[j] * scale)
                );
                assert_eq!(
                    result.features[row * 32 + j + 16],
                    bf16::from_f32(expected[15 - j] * scale)
                );
            }
        }
        let warm = table.lookup(&[1, 0, 1]).unwrap();
        assert_eq!(warm.cache_hits, 3);
        assert_eq!(warm.disk_rows, 0);
        assert_eq!(warm.features, result.features);
    }

    #[test]
    fn bounded_lru_eviction_zero_budget_and_parallel_equivalence() {
        let fixture = Fixture::new(20);
        let mut table = fixture.table(136, 2); // Exactly one four-way set.
        assert_eq!(table.cache_bytes(), 136);
        table.lookup(&[0, 1, 2, 3]).unwrap();
        assert_eq!(table.lookup(&[0]).unwrap().cache_hits, 1);
        table.lookup(&[4]).unwrap();
        assert_eq!(table.lookup(&[0]).unwrap().cache_hits, 1);
        assert_eq!(table.lookup(&[1]).unwrap().disk_rows, 1);
        let ids = [7, 3, 2, 0, 7, 1];
        let mut uncached = fixture.table(0, 1);
        let mut parallel = fixture.table(0, 8);
        assert_eq!(uncached.cache_bytes(), 0);
        assert_eq!(
            uncached.lookup(&ids).unwrap().features,
            parallel.lookup(&ids).unwrap().features
        );
        assert_eq!(uncached.lookup(&ids).unwrap().disk_rows, 5);
        assert!(uncached.lookup(&[]).unwrap().features.is_empty());
    }

    #[test]
    fn rejects_wrong_format_extent_ids_and_truncated_reads() {
        let fixture = Fixture::new(2); // Q4_0 has the same size, different meaning.
        assert!(Table::open(&fixture.0, "P", 8, 32, 1024, 2).is_err());
        let fixture = Fixture::new(20);
        assert!(Table::open(&fixture.0, "P", 7, 32, 1024, 2).is_err());
        assert!(Table::open(&fixture.0, "P", 8, 32, 1024, 0).is_err());
        let mut table = fixture.table(1024, 2);
        assert!(table.lookup(&[0, 8]).is_err());
        assert_eq!(table.lookup(&[0]).unwrap().disk_rows, 1);
        // A prepared artifact must be immutable; positional I/O still reports
        // truncation as an error rather than returning fabricated zero rows.
        fs::OpenOptions::new()
            .write(true)
            .open(&fixture.0)
            .unwrap()
            .set_len(8)
            .unwrap();
        assert!(table.lookup(&[7]).is_err());
    }

    #[test]
    fn preparing_features_is_transactional_for_request_history() {
        use crate::ple::{History, NgramConfig, NgramHasher};
        let config = NgramConfig {
            vocab_size: 128,
            eos_token_id: 127,
            ngram_size: 3,
            heads_per_ngram: 1,
            vocab_base: 5,
            vocab_alignment: 8,
            layer_index: 0,
            seed: 1234,
        };
        let hasher = NgramHasher::new(config).unwrap();
        assert_eq!(hasher.table_rows(), 16);
        let fixture = Fixture::with_rows(20, 16);
        let mut table = Table::open(&fixture.0, "P", 16, 32, 0, 2).unwrap();
        let mut history = History::default();
        hasher.append(10, &mut history).unwrap();
        let before = history.clone();
        let prepared = hasher
            .prepare(&[11, 127, 12], &history, &mut table)
            .unwrap();
        let mut expected = before.clone();
        let rows: Vec<_> = [11, 127, 12]
            .into_iter()
            .flat_map(|t| hasher.append(t, &mut expected).unwrap())
            .collect();
        assert_eq!(
            prepared.lookup.features,
            table.lookup(&rows).unwrap().features
        );
        assert_eq!(prepared.next_history, expected);
        assert_eq!(history, before);
        assert!(hasher.prepare(&[13, 128], &history, &mut table).is_err());
        assert_eq!(history, before);
        fs::OpenOptions::new()
            .write(true)
            .open(&fixture.0)
            .unwrap()
            .set_len(8)
            .unwrap();
        assert!(hasher.prepare(&[13], &history, &mut table).is_err());
        assert_eq!(history, before);
    }
}
