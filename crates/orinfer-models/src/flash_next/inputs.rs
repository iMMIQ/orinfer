//! Bounded CPU E8P embedding/PLE lookup. History comes from the request owner;
//! immutable decoded rows are shared, never mutable n-gram state.
use crate::artifact::Result;
use half::f16;
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    fs::File,
    io::Read,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    rc::Rc,
};

/// Execution purpose selects the arena, independently of the logical row count.
/// Shared batch packets and exact MTP verification retain their own input views.
pub(super) fn layout<'a>(
    prefill: &[crate::model::PrefillPlan],
    verification: &[orinfer_model_sdk::mtp::VerificationPlan],
    program: &'a str,
    tokens: usize,
) -> Result<(&'a str, usize)> {
    if let Some(plan) = verification.iter().find(|p| p.program == program) {
        if !plan.accepts(tokens) {
            return Err("Verification input extent does not match its program".into());
        }
        return Ok(("target", plan.tokens));
    }
    if program == "batch_target" {
        return Ok(("target", tokens));
    }
    if let Some(plan) = prefill
        .iter()
        .find(|p| p.prefill_program == program || p.stages.iter().any(|stage| stage == program))
    {
        if !plan.accepts(tokens) {
            return Err("Prefill input extent does not match its program".into());
        }
        return Ok(("target", plan.chunk_tokens));
    }
    Ok((program, tokens))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Book {
    table: Vec<i8>,
    signs: Vec<i8>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    file: String,
    sha256: String,
    offset: u64,
    first: usize,
    rows: usize,
    #[serde(default)]
    book: Option<usize>,
    #[serde(default)]
    scale_offset: Option<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Assets {
    version: u32,
    vocab: usize,
    eos: u32,
    multipliers: Vec<u64>,
    sizes: Vec<u64>,
    offsets: Vec<u64>,
    books: Vec<Book>,
    embedding: Vec<Part>,
    ple: Vec<Part>,
}
struct Table {
    values: Vec<[i8; 8]>,
    signs: Vec<i8>,
}
struct Source {
    file: File,
    offset: u64,
    first: usize,
    rows: usize,
    book: Option<usize>,
    scale_offset: Option<u64>,
    width: usize,
}
type Row = Rc<Vec<u16>>;
#[derive(Default)]
struct Cache {
    clock: u64,
    bytes: usize,
    values: HashMap<(usize, usize), (u64, Row)>,
    order: BTreeMap<u64, (usize, usize)>,
}
impl Cache {
    fn get(&mut self, key: (usize, usize)) -> Option<Row> {
        let (stamp, value) = self.values.get_mut(&key)?;
        self.order.remove(stamp);
        self.clock += 1;
        *stamp = self.clock;
        self.order.insert(*stamp, key);
        Some(Rc::clone(value))
    }
    fn put(&mut self, key: (usize, usize), value: Row, budget: usize) {
        let size = value.len() * 2 + 512;
        if size > budget {
            return;
        }
        while self.bytes + size > budget {
            let (_, key) = self.order.pop_first().expect("nonempty bounded row cache");
            let (_, old) = self.values.remove(&key).expect("paired cache entry");
            self.bytes -= old.len() * 2 + 512;
        }
        self.clock += 1;
        self.bytes += size;
        self.values.insert(key, (self.clock, value));
        self.order.insert(self.clock, key);
    }
}
pub(super) struct Inputs {
    assets: Assets,
    tables: Vec<Table>,
    embedding: Vec<Source>,
    ple: Vec<Source>,
    embedding_cache: RefCell<Cache>,
    ple_cache: RefCell<Cache>,
}
fn path(root: &Path, name: &str) -> Result<PathBuf> {
    let relative = Path::new(name);
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        return Err("Unsafe model input asset path".into());
    }
    let file = root
        .join(relative)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    if !file.starts_with(root) {
        return Err("Model input asset leaves model directory".into());
    }
    Ok(file)
}
fn digest(file: &mut File) -> Result<String> {
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hex::encode(hash.finalize()))
}

fn validate_rows(file: &mut File, part: &Part, width: usize, verify_weights: bool) -> Result<()> {
    if part.sha256.len() != 64 || !part.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Invalid CPU row shard hash".into());
    }
    // SAFETY: Prepared assets remain immutable while open. Only the container
    // header is accessed here; the mapping is dropped before row lookup begins.
    let mapping = unsafe { MmapOptions::new().map(&*file) }.map_err(|e| e.to_string())?;
    let (header, metadata) = SafeTensors::read_metadata(&mapping).map_err(|e| e.to_string())?;
    let start = 8 + header;
    let tensors = metadata.tensors();
    let contains = |offset: u64, dtype: Dtype, shape: &[usize]| {
        tensors.values().any(|info| {
            (start + info.data_offsets.0) as u64 == offset
                && info.dtype == dtype
                && info.shape == shape
        })
    };
    let rows_match = if part.book.is_some() {
        contains(part.offset, Dtype::U8, &[part.rows, 2 + width / 4])
    } else {
        contains(part.offset, Dtype::I8, &[part.rows, width])
            && part
                .scale_offset
                .is_some_and(|offset| contains(offset, Dtype::F16, &[part.rows]))
    };
    if !rows_match {
        return Err("CPU row safetensors dtype, shape or offset mismatch".into());
    }
    drop(mapping);
    if verify_weights && digest(file)? != part.sha256 {
        return Err("CPU row shard hash mismatch".into());
    }
    Ok(())
}
impl Inputs {
    pub(super) fn open(
        root: &str,
        identity: &crate::artifact::FileIdentity,
        verify_weights: bool,
    ) -> Result<Self> {
        if identity.tensor.is_some() {
            return Err("CPU input metadata must be a standalone file".into());
        }
        let root = Path::new(root).canonicalize().map_err(|e| e.to_string())?;
        let raw = std::fs::read(path(&root, &identity.file)?).map_err(|e| e.to_string())?;
        if verify_weights && hex::encode(Sha256::digest(&raw)) != identity.sha256 {
            return Err("CPU input metadata hash mismatch".into());
        }
        let assets: Assets = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if assets.version != 1
            || assets.vocab != 248320
            || assets.eos as usize >= assets.vocab
            || assets.multipliers.len() != 3
            || assets.sizes.len() != 16
            || assets.offsets.len() != 16
            || assets.multipliers.iter().any(|&m| {
                m.checked_mul((assets.vocab - 1) as u64)
                    .is_none_or(|x| x > i64::MAX as u64)
            })
        {
            return Err("Unsupported CPU embedding/PLE metadata".into());
        }
        let mut tables = vec![];
        for book in &assets.books {
            let width = book.signs.len();
            if book.table.len() != 2048
                || !matches!(width, 160 | 2560)
                || book.signs.iter().any(|s| !matches!(s, -1 | 1))
                || book
                    .table
                    .iter()
                    .any(|&t| t % 2 != 0 || t.unsigned_abs() > 126)
            {
                return Err("Invalid E8P book or rotation".into());
            }
            let bit = [0, 4, 1, 5, 2, 6, 3, 7];
            let values = (0..65536u32)
                .map(|code| {
                    let parity = (code & 255).count_ones() & 1;
                    let signs = (code & 255) ^ parity;
                    std::array::from_fn(|j| {
                        let base = book.table[((code >> 8) * 8) as usize + j] as i16;
                        (base * if (signs >> bit[j]) & 1 == 0 { 1 } else { -1 }
                            + if parity == 0 { 1 } else { -1 }) as i8
                    })
                })
                .collect();
            tables.push(Table {
                values,
                signs: book.signs.clone(),
            });
        }
        let load = |parts: &[Part], width: usize| -> Result<Vec<Source>> {
            let mut out = vec![];
            let mut first = 0usize;
            for part in parts {
                if part.first != first
                    || part.rows == 0
                    || match (part.book, part.scale_offset) {
                        (Some(book), None) => {
                            tables.get(book).is_none_or(|b| b.signs.len() != width)
                        }
                        (None, Some(_)) => width != 2560,
                        _ => true,
                    }
                {
                    return Err("Invalid or discontinuous CPU row slices".into());
                }
                let mut file = File::open(path(&root, &part.file)?).map_err(|e| e.to_string())?;
                validate_rows(&mut file, part, width, verify_weights)?;
                let bytes = (part.rows as u64)
                    .checked_mul(if part.book.is_some() {
                        (2 + width / 4) as u64
                    } else {
                        width as u64
                    })
                    .and_then(|n| n.checked_add(part.offset))
                    .ok_or("CPU row extent overflow")?;
                if bytes > file.metadata().map_err(|e| e.to_string())?.len() {
                    return Err("CPU rows exceed file".into());
                }
                if part.scale_offset.is_some_and(|offset| {
                    offset
                        .checked_add(part.rows as u64 * 2)
                        .is_none_or(|end| end > file.metadata().map(|m| m.len()).unwrap_or(0))
                }) {
                    return Err("CPU row scales exceed file".into());
                }
                out.push(Source {
                    file,
                    offset: part.offset,
                    first,
                    rows: part.rows,
                    book: part.book,
                    scale_offset: part.scale_offset,
                    width,
                });
                first = first
                    .checked_add(part.rows)
                    .ok_or("CPU row count overflow")?;
            }
            if out.is_empty() {
                return Err("Missing CPU row shards".into());
            }
            Ok(out)
        };
        let embedding = load(&assets.embedding, 2560)?;
        let ple = load(&assets.ple, 160)?;
        if embedding.last().map(|p| p.first + p.rows) != Some(assets.vocab) {
            return Err("Embedding vocabulary mismatch".into());
        }
        let total = ple
            .last()
            .map(|p| p.first + p.rows)
            .ok_or("Empty PLE table")? as u64;
        if assets
            .sizes
            .iter()
            .zip(&assets.offsets)
            .any(|(&s, &o)| s == 0 || o.checked_add(s).is_none_or(|end| end > total))
        {
            return Err("PLE head outside CPU row table".into());
        }
        Ok(Self {
            assets,
            tables,
            embedding,
            ple,
            embedding_cache: Default::default(),
            ple_cache: Default::default(),
        })
    }
    fn row(
        &self,
        sources: &[Source],
        cache: &RefCell<Cache>,
        row: usize,
        budget: usize,
    ) -> Result<Row> {
        let key = shard_of(sources, row)?;
        if let Some(value) = cache.borrow_mut().get(key) {
            return Ok(value);
        }
        let value = load_row(sources, &self.tables, key.0, key.1)?;
        cache.borrow_mut().put(key, Rc::clone(&value), budget);
        Ok(value)
    }
    /// Read and decode every uncached row one packet needs before assembly.
    /// Keys are sorted per table so each worker reads ascending file offsets;
    /// decoded rows are inserted into the owning cache on the calling thread.
    fn prefetch_rows(&self, embedding_rows: &[usize], ple_rows: &[usize]) -> Result<()> {
        let missing = |sources: &[Source],
                       rows: &[usize],
                       cache: &RefCell<Cache>|
         -> Result<Vec<(usize, usize)>> {
            let mut need: Vec<(usize, usize)> = Vec::new();
            {
                let mut cache = cache.borrow_mut();
                for &row in rows {
                    let key = shard_of(sources, row)?;
                    if cache.get(key).is_none() {
                        need.push(key);
                    }
                }
            }
            need.sort_unstable();
            need.dedup();
            Ok(need)
        };
        let embedding_need = missing(&self.embedding, embedding_rows, &self.embedding_cache)?;
        let ple_need = missing(&self.ple, ple_rows, &self.ple_cache)?;
        let total = embedding_need.len() + ple_need.len();
        if total == 0 {
            return Ok(());
        }
        // Work items carry the owning table so one sorted list feeds a single
        // parallel pass: false selects embedding shards, true selects PLE.
        let keys: Vec<(bool, usize, usize)> = embedding_need
            .iter()
            .map(|&(shard, local)| (false, shard, local))
            .chain(ple_need.iter().map(|&(shard, local)| (true, shard, local)))
            .collect();
        let mut decoded: Vec<Option<Vec<u16>>> = vec![None; total];
        let threads = total.min(std::thread::available_parallelism().map_or(1, |n| n.get()));
        let Inputs {
            embedding,
            ple,
            tables,
            ..
        } = self;
        if threads <= 1 {
            for (key, out) in keys.iter().zip(decoded.iter_mut()) {
                *out = Some(if key.0 {
                    decode_row(ple, tables, key.1, key.2)?
                } else {
                    decode_row(embedding, tables, key.1, key.2)?
                });
            }
        } else {
            let chunk = total.div_ceil(threads);
            std::thread::scope(|scope| -> Result<()> {
                let mut handles = vec![];
                for (keys, out) in keys.chunks(chunk).zip(decoded.chunks_mut(chunk)) {
                    handles.push(scope.spawn(move || -> Result<()> {
                        for (key, out) in keys.iter().zip(out.iter_mut()) {
                            *out = Some(if key.0 {
                                decode_row(ple, tables, key.1, key.2)?
                            } else {
                                decode_row(embedding, tables, key.1, key.2)?
                            });
                        }
                        Ok(())
                    }));
                }
                for handle in handles {
                    handle.join().map_err(|_| "CPU row loader panicked")??;
                }
                Ok(())
            })?;
        }
        for (key, value) in keys.iter().zip(decoded) {
            let row = Rc::new(value.expect("every missing row was decoded"));
            if key.0 {
                self.ple_cache
                    .borrow_mut()
                    .put((key.1, key.2), row, PLE_CACHE_BUDGET);
            } else {
                self.embedding_cache
                    .borrow_mut()
                    .put((key.1, key.2), row, EMBEDDING_CACHE_BUDGET);
            }
        }
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn prepare(
        &self,
        tokens: &[u32],
        history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        self.prepare_for("target", tokens, history, tokens.len())
    }
    pub(super) fn prepare_for(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
        capacity: usize,
    ) -> Result<Vec<(String, Vec<u8>)>> {
        let draft = program.starts_with("mtp_warm") || program == "mtp_draft";
        if !draft && program != "target" {
            return Err("Unknown Flash input program".into());
        }
        if tokens.is_empty()
            || tokens.len() > capacity
            || !(1..=4096).contains(&capacity)
            || (draft && program != "mtp_warm" && capacity != tokens.len())
            || tokens
                .iter()
                .chain(history)
                .any(|&t| t as usize >= self.assets.vocab)
        {
            return Err("Unsupported Flash input shape/token".into());
        }
        let mut previous: Vec<u32> = history
            .iter()
            .rev()
            .take_while(|&&t| t != self.assets.eos)
            .take(2)
            .copied()
            .collect();
        previous.reverse();
        // Pass 1 resolves every row this packet needs without I/O; the missing
        // set is then read and decoded in parallel before assembly, so cold
        // prompts no longer pay one synchronous row read per lookup.
        let mut embedding_rows: Vec<usize> = Vec::with_capacity(tokens.len());
        let mut ple_rows: Vec<usize> =
            Vec::with_capacity(if draft { 0 } else { tokens.len() * 16 });
        for &token in tokens {
            embedding_rows.push(token as usize);
            if draft {
                continue;
            }
            let mut mixed = token as u64 * self.assets.multipliers[0];
            for shift in 1..=2 {
                let prior = previous
                    .len()
                    .checked_sub(shift)
                    .map(|i| previous[i])
                    .unwrap_or(self.assets.eos);
                mixed ^= prior as u64 * self.assets.multipliers[shift];
                for head in (shift - 1) * 8..shift * 8 {
                    ple_rows.push(
                        (mixed % self.assets.sizes[head] + self.assets.offsets[head]) as usize,
                    );
                }
            }
            if token == self.assets.eos {
                previous.clear();
            } else {
                previous.push(token);
                if previous.len() > 2 {
                    previous.remove(0);
                }
            }
        }
        self.prefetch_rows(&embedding_rows, &ple_rows)?;
        let mut embedding = Vec::with_capacity(tokens.len() * 5120);
        let mut ple = Vec::with_capacity(if draft { 0 } else { tokens.len() * 5120 });
        let mut ple_cursor = 0;
        for &token in tokens {
            let row = self.row(
                &self.embedding,
                &self.embedding_cache,
                token as usize,
                EMBEDDING_CACHE_BUDGET,
            )?;
            for &value in row.iter() {
                embedding.extend(value.to_le_bytes());
            }
            if draft {
                continue;
            }
            for _ in 0..16 {
                let row = self.row(
                    &self.ple,
                    &self.ple_cache,
                    ple_rows[ple_cursor],
                    PLE_CACHE_BUDGET,
                )?;
                ple_cursor += 1;
                for &value in row.iter() {
                    ple.extend(value.to_le_bytes());
                }
            }
        }
        if draft {
            let mut prepared = vec![(
                format!(
                    "DraftM{}_Embedding",
                    if program == "mtp_warm" {
                        capacity
                    } else {
                        tokens.len()
                    }
                ),
                embedding,
            )];
            if program == "mtp_warm" {
                prepared.push((
                    "MtpLength".into(),
                    (tokens.len() as u32).to_le_bytes().to_vec(),
                ));
            }
            return Ok(prepared);
        }
        Ok(vec![
            (
                format!("M{}_Embedding", super::plan::arena_width(capacity)),
                embedding,
            ),
            (format!("M{}_Ple", super::plan::arena_width(capacity)), ple),
        ])
    }
}
/// Decoded-row LRU budgets in bytes. Repetitive prompts stay cached across
/// requests; cold rows are read and decoded in parallel instead.
const EMBEDDING_CACHE_BUDGET: usize = 64 * 1024 * 1024;
const PLE_CACHE_BUDGET: usize = 128 * 1024 * 1024;

fn shard_of(sources: &[Source], row: usize) -> Result<(usize, usize)> {
    let shard = sources
        .partition_point(|s| s.first <= row)
        .checked_sub(1)
        .ok_or("CPU row outside table")?;
    let local = row - sources[shard].first;
    if local >= sources[shard].rows {
        return Err("CPU row outside table".into());
    }
    Ok((shard, local))
}
fn decode_row(
    sources: &[Source],
    tables: &[Table],
    shard: usize,
    local: usize,
) -> Result<Vec<u16>> {
    let source = &sources[shard];
    let width = source.width;
    let stride = if source.book.is_some() {
        2 + width / 4
    } else {
        width
    };
    let mut packet = vec![0u8; stride];
    source
        .file
        .read_exact_at(&mut packet, source.offset + (local * stride) as u64)
        .map_err(|e| e.to_string())?;
    if let Some(book) = source.book {
        return decode(&packet, &tables[book]);
    }
    let mut scale = [0u8; 2];
    source
        .file
        .read_exact_at(
            &mut scale,
            source.scale_offset.ok_or("Missing INT8 row scale")? + local as u64 * 2,
        )
        .map_err(|e| e.to_string())?;
    let scale = f16::from_bits(u16::from_le_bytes(scale)).to_f32();
    if !scale.is_finite() || scale <= 0.0 {
        return Err("Invalid INT8 row scale".into());
    }
    Ok(packet
        .iter()
        .map(|&v| f16::from_f32((v as i8) as f32 * scale).to_bits())
        .collect())
}
fn load_row(sources: &[Source], tables: &[Table], shard: usize, local: usize) -> Result<Row> {
    Ok(Rc::new(decode_row(sources, tables, shard, local)?))
}
fn decode(packet: &[u8], table: &Table) -> Result<Vec<u16>> {
    let scale = f16::from_bits(u16::from_le_bytes([packet[0], packet[1]])).to_f32();
    if !scale.is_finite() || scale <= 0. {
        return Err("Invalid E8P row scale".into());
    }
    let width = table.signs.len();
    let columns = width / 20;
    let mut rotated = vec![0f32; width];
    for (i, code) in packet[2..].as_chunks::<2>().0.iter().enumerate() {
        let values = &table.values[u16::from_le_bytes([code[0], code[1]]) as usize];
        for j in 0..8 {
            rotated[i * 8 + j] = values[j] as f32 * scale;
        }
    }
    let mut out = vec![0f32; width];
    let norm = (20f32).sqrt().recip();
    for i in 0..20 {
        for j in 0..20 {
            let sign = if j == 0 {
                1.
            } else if i == 0 {
                -1.
            } else {
                let delta = (j + 19 - i) % 19;
                if delta == 0 || (1..19).any(|k| k * k % 19 == delta) {
                    1.
                } else {
                    -1.
                }
            };
            let coefficient = sign * norm;
            for col in 0..columns {
                out[i * columns + col] += coefficient * rotated[j * columns + col];
            }
        }
    }
    for row in out.chunks_exact_mut(columns) {
        let mut half = 1;
        while half < columns {
            for block in row.chunks_exact_mut(half * 2) {
                for j in 0..half {
                    let a = block[j];
                    let b = block[j + half];
                    block[j] = a + b;
                    block[j + half] = a - b;
                }
            }
            half *= 2;
        }
    }
    let norm = (columns as f32).sqrt().recip();
    Ok(out
        .iter()
        .zip(&table.signs)
        .map(|(&x, &s)| f16::from_f32((x * norm) * s as f32).to_bits())
        .collect())
}
#[cfg(test)]
mod tests {
    #[test]
    fn execution_purpose_separates_overlapping_input_shapes() {
        let prefill = [crate::model::PrefillPlan {
            chunk_tokens: 4096,
            min_tokens: 1,
            prefill_program: "prefill".into(),
            head_program: "head".into(),
            stages: vec!["prefill/stage0".into()],
            live_buffers: vec![],
        }];
        let verification: Vec<_> = (2..=8)
            .map(|tokens| orinfer_model_sdk::mtp::VerificationPlan {
                tokens,
                min_tokens: 0,
                program: format!("verify_m{tokens}"),
                restore_program: format!("restore_m{tokens}"),
                capture_program: format!("capture_m{tokens}"),
            })
            .collect();
        for plan in &verification {
            assert_eq!(
                layout(&prefill, &verification, "prefill", plan.tokens).unwrap(),
                ("target", 4096)
            );
            assert_eq!(
                layout(&prefill, &verification, &plan.program, plan.tokens).unwrap(),
                ("target", plan.tokens)
            );
            assert!(layout(&prefill, &verification, &plan.program, plan.tokens + 1).is_err());
        }
        assert_eq!(
            layout(&prefill, &verification, "batch_target", 128).unwrap(),
            ("target", 128)
        );
        assert_eq!(
            layout(&prefill, &verification, "prefill", 421).unwrap(),
            ("target", 4096)
        );
        assert_eq!(
            layout(&prefill, &verification, "target", 1).unwrap(),
            ("target", 1)
        );
        for program in ["prefill", "prefill/stage0"] {
            assert_eq!(
                layout(&prefill, &verification, program, 1).unwrap(),
                ("target", 4096)
            );
            assert!(layout(&prefill, &verification, program, 0).is_err());
            assert!(layout(&prefill, &verification, program, 4097).is_err());
        }
        assert_eq!(
            layout(&prefill, &verification, "mtp_warm_m4", 4).unwrap(),
            ("mtp_warm_m4", 4)
        );
    }

    use super::*;
    use safetensors::tensor::TensorView;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Rows(PathBuf);
    impl Rows {
        fn new(encoded: bool) -> (Self, Part, usize) {
            let path = std::env::temp_dir().join(format!(
                "orin-cpu-rows-{}-{}.safetensors",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let width = if encoded { 160 } else { 2560 };
            let columns = if encoded { 2 + width / 4 } else { width };
            let data = vec![0u8; 2 * columns];
            let scales = [0u8; 4];
            let dtype = if encoded { Dtype::U8 } else { Dtype::I8 };
            let mut tensors = vec![(
                "rows",
                TensorView::new(dtype, vec![2, columns], &data).unwrap(),
            )];
            if !encoded {
                tensors.push((
                    "scale",
                    TensorView::new(Dtype::F16, vec![2], &scales).unwrap(),
                ));
            }
            let bytes = safetensors::serialize(tensors, None).unwrap();
            let (header, metadata) = SafeTensors::read_metadata(&bytes).unwrap();
            let offset = |name| (8 + header + metadata.info(name).unwrap().data_offsets.0) as u64;
            let part = Part {
                file: path.file_name().unwrap().to_str().unwrap().into(),
                sha256: hex::encode(Sha256::digest(&bytes)),
                offset: offset("rows"),
                first: 0,
                rows: 2,
                book: encoded.then_some(0),
                scale_offset: (!encoded).then(|| offset("scale")),
            };
            std::fs::write(&path, bytes).unwrap();
            (Self(path), part, width)
        }
        fn validate(&self, part: &Part, width: usize, verify: bool) -> Result<()> {
            validate_rows(&mut File::open(&self.0).unwrap(), part, width, verify)
        }
    }
    impl Drop for Rows {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }

    #[test]
    fn cached_rows_share_immutable_storage_and_survive_eviction() {
        let mut cache = Cache::default();
        let row = Rc::new(vec![0x3c00; 64]);
        let budget = row.len() * 2 + 512;
        cache.put((0, 0), Rc::clone(&row), budget);
        let hit = cache.get((0, 0)).unwrap();
        assert!(Rc::ptr_eq(&hit, &row));
        cache.put((0, 1), Rc::new(vec![0x4000; 64]), budget);
        assert!(cache.get((0, 0)).is_none());
        assert_eq!(hit.as_slice(), &[0x3c00; 64]);
        assert_eq!(cache.get((0, 1)).unwrap().as_slice(), &[0x4000; 64]);
        assert!(cache.bytes <= budget);
    }

    #[test]
    fn cpu_rows_verify_content_only_when_requested() {
        for encoded in [false, true] {
            let (file, part, width) = Rows::new(encoded);
            file.validate(&part, width, false).unwrap();
            file.validate(&part, width, true).unwrap();
            let mut bytes = std::fs::read(&file.0).unwrap();
            bytes[part.offset as usize] ^= 1;
            std::fs::write(&file.0, bytes).unwrap();
            file.validate(&part, width, false).unwrap();
            assert!(file.validate(&part, width, true).is_err());
        }
    }

    #[test]
    fn cpu_rows_reject_invalid_structure_in_both_modes() {
        for encoded in [false, true] {
            let (file, mut part, width) = Rows::new(encoded);
            for verify in [false, true] {
                assert!(file.validate(&part, width + 8, verify).is_err());
                part.rows = 1;
                assert!(file.validate(&part, width, verify).is_err());
                part.rows = 2;
                part.offset += 1;
                assert!(file.validate(&part, width, verify).is_err());
                part.offset -= 1;
                if let Some(offset) = part.scale_offset {
                    part.scale_offset = Some(offset + 1);
                    assert!(file.validate(&part, width, verify).is_err());
                    part.scale_offset = Some(offset);
                }
                part.book = if encoded { None } else { Some(0) };
                assert!(file.validate(&part, width, verify).is_err());
                part.book = encoded.then_some(0);
            }
            let mut bytes = std::fs::read(&file.0).unwrap();
            bytes.pop();
            std::fs::write(&file.0, bytes).unwrap();
            for verify in [false, true] {
                assert!(file.validate(&part, width, verify).is_err());
            }
        }
    }

    #[test]
    #[ignore = "Requires prepared Flash CPU row assets and frozen Python input rows"]
    fn reference_rows_match_with_eos_and_request_history() {
        let root = std::env::var("ORINFER_FLASH_MODEL").unwrap();
        let golden = std::env::var("ORINFER_FLASH_GOLDEN").unwrap();
        let raw = std::fs::read(Path::new(&root).join("cache/cpu/inputs.json")).unwrap();
        let inputs = Inputs::open(
            &root,
            &crate::artifact::FileIdentity {
                file: "cache/cpu/inputs.json".into(),
                tensor: None,
                sha256: hex::encode(Sha256::digest(&raw)),
            },
            true,
        )
        .unwrap();
        let golden: serde_json::Value =
            serde_json::from_slice(&std::fs::read(golden).unwrap()).unwrap();
        let tokens: Vec<u32> = serde_json::from_value(golden["tokens"].clone()).unwrap();
        let mut history: Vec<u32> = serde_json::from_value(golden["history"].clone()).unwrap();
        let expected: Vec<Vec<u16>> = ["embedding", "ple"]
            .into_iter()
            .map(|k| serde_json::from_value(golden[k].clone()).unwrap())
            .collect();
        for (index, &token) in tokens.iter().enumerate() {
            let rows = inputs.prepare(&[token], &history).unwrap();
            for (kind, (_, raw)) in rows.iter().enumerate() {
                let actual: Vec<u16> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| u16::from_le_bytes([p[0], p[1]]))
                    .collect();
                let expected = &expected[kind][index * 2560..(index + 1) * 2560];
                let max = actual
                    .iter()
                    .zip(expected)
                    .map(|(&a, &b)| (f16::from_bits(a).to_f32() - f16::from_bits(b).to_f32()).abs())
                    .fold(0f32, f32::max);
                assert!(max <= 0.0005, "input {index}/{kind}: max FP16 error {max}");
            }
            history.push(token);
        }
        assert!(inputs.embedding_cache.borrow().bytes <= EMBEDDING_CACHE_BUDGET);
        assert!(inputs.ple_cache.borrow().bytes <= PLE_CACHE_BUDGET);
        assert!(inputs.prepare(&[248320], &[]).is_err());
    }
}
