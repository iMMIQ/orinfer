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
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread::JoinHandle,
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
type Row = Arc<Vec<u16>>;
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
        Some(Arc::clone(value))
    }
    fn put(&mut self, key: (usize, usize), value: Row, budget: usize) {
        let size = value.len() * 2 + 512;
        if size > budget || self.get(key).is_some() {
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
type Uploads = Vec<(String, Vec<u8>)>;
type DecodeTask = (bool, usize, usize);

/// Dispatch across the persistent worker pool whenever at least two cold rows
/// need reading so concurrent pread calls overlap SSD latency without spawning
/// fresh OS threads.
const PARALLEL_DECODE_THRESHOLD: usize = 2;

struct LookupData {
    assets: Assets,
    tables: Vec<Table>,
    embedding: Vec<Source>,
    ple: Vec<Source>,
}

struct ActiveJob {
    keys: Vec<DecodeTask>,
    chunk: usize,
    next: AtomicUsize,
    done_mu: Mutex<JobCollector>,
    done_cv: Condvar,
}

struct JobCollector {
    decoded: Vec<Option<Vec<u16>>>,
    error: Option<String>,
    workers_done: usize,
}

#[derive(Default)]
struct PoolState {
    job_id: u64,
    job: Option<Arc<ActiveJob>>,
    shutdown: bool,
}

/// Persistent worker pool for decoding large cold prefill batches without
/// spawning fresh OS threads on every packet.
struct RowPool {
    data: Arc<LookupData>,
    run_lock: Mutex<()>,
    state: Arc<(Mutex<PoolState>, Condvar)>,
    handles: Mutex<Vec<JoinHandle<()>>>,
}

impl RowPool {
    fn new(data: Arc<LookupData>) -> Self {
        Self {
            data,
            run_lock: Mutex::new(()),
            state: Arc::new((Mutex::new(PoolState::default()), Condvar::new())),
            handles: Mutex::new(Vec::new()),
        }
    }

    fn ensure_workers(&self) -> usize {
        let mut handles = self.handles.lock().unwrap();
        if handles.is_empty() {
            let count = std::thread::available_parallelism()
                .map_or(1, |n| n.get())
                .saturating_sub(1)
                .min(11);
            for index in 0..count {
                let data = Arc::clone(&self.data);
                let state = Arc::clone(&self.state);
                if let Ok(handle) = std::thread::Builder::new()
                    .name(format!("flash-row-{index}"))
                    .spawn(move || {
                        let mut seen = 0u64;
                        loop {
                            let job = {
                                let (lock, cv) = &*state;
                                let mut guard = lock.lock().unwrap();
                                while !guard.shutdown && guard.job_id == seen {
                                    guard = cv.wait(guard).unwrap();
                                }
                                if guard.shutdown {
                                    return;
                                }
                                seen = guard.job_id;
                                guard.job.clone()
                            };
                            let Some(job) = job else {
                                continue;
                            };
                            let (local, error) = Self::drain_job(&data, &job);
                            let mut collector = job.done_mu.lock().unwrap();
                            if collector.error.is_none() {
                                if let Some(err) = error {
                                    collector.error = Some(err);
                                } else {
                                    for (idx, row) in local {
                                        collector.decoded[idx] = Some(row);
                                    }
                                }
                            }
                            collector.workers_done += 1;
                            job.done_cv.notify_one();
                        }
                    })
                {
                    handles.push(handle);
                }
            }
        }
        handles.len()
    }

    fn drain_job(data: &LookupData, job: &ActiveJob) -> (Vec<(usize, Vec<u16>)>, Option<String>) {
        let mut local = Vec::new();
        let total = job.keys.len();
        loop {
            let start = job.next.fetch_add(job.chunk, Ordering::Relaxed);
            if start >= total {
                break;
            }
            let end = (start + job.chunk).min(total);
            for idx in start..end {
                let (is_ple, shard, row) = job.keys[idx];
                let res = if is_ple {
                    decode_row(&data.ple, &data.tables, shard, row)
                } else {
                    decode_row(&data.embedding, &data.tables, shard, row)
                };
                match res {
                    Ok(values) => local.push((idx, values)),
                    Err(err) => {
                        job.next.store(total, Ordering::Relaxed);
                        return (local, Some(err));
                    }
                }
            }
        }
        (local, None)
    }

    fn decode(&self, keys: Vec<DecodeTask>) -> Result<Vec<Vec<u16>>> {
        let total = keys.len();
        if total < PARALLEL_DECODE_THRESHOLD {
            let mut decoded = Vec::with_capacity(total);
            for &(is_ple, shard, row) in &keys {
                decoded.push(if is_ple {
                    decode_row(&self.data.ple, &self.data.tables, shard, row)?
                } else {
                    decode_row(&self.data.embedding, &self.data.tables, shard, row)?
                });
            }
            return Ok(decoded);
        }
        let _run = self.run_lock.lock().unwrap();
        let workers = self.ensure_workers();
        if workers == 0 {
            let mut decoded = Vec::with_capacity(total);
            for &(is_ple, shard, row) in &keys {
                decoded.push(if is_ple {
                    decode_row(&self.data.ple, &self.data.tables, shard, row)?
                } else {
                    decode_row(&self.data.embedding, &self.data.tables, shard, row)?
                });
            }
            return Ok(decoded);
        }
        let chunk = (total / ((workers + 1) * 4)).clamp(1, 16);
        let job = Arc::new(ActiveJob {
            keys,
            chunk,
            next: AtomicUsize::new(0),
            done_mu: Mutex::new(JobCollector {
                decoded: vec![None; total],
                error: None,
                workers_done: 0,
            }),
            done_cv: Condvar::new(),
        });
        {
            let (lock, cv) = &*self.state;
            let mut guard = lock.lock().unwrap();
            guard.job = Some(Arc::clone(&job));
            guard.job_id += 1;
            cv.notify_all();
        }
        let (local, local_err) = Self::drain_job(&self.data, &job);
        let mut collector = job.done_mu.lock().unwrap();
        while collector.workers_done < workers {
            collector = job.done_cv.wait(collector).unwrap();
        }
        {
            let (lock, _) = &*self.state;
            lock.lock().unwrap().job = None;
        }
        if let Some(err) = local_err.or_else(|| collector.error.take()) {
            return Err(err);
        }
        for (idx, row) in local {
            collector.decoded[idx] = Some(row);
        }
        Ok(collector
            .decoded
            .drain(..)
            .map(|row| row.expect("every missing row was decoded"))
            .collect())
    }
}

impl Drop for RowPool {
    fn drop(&mut self) {
        let _run = self.run_lock.lock().unwrap();
        {
            let (lock, cv) = &*self.state;
            let mut guard = lock.lock().unwrap();
            guard.shutdown = true;
            guard.job = None;
            cv.notify_all();
        }
        for handle in self.handles.get_mut().unwrap().drain(..) {
            let _ = handle.join();
        }
    }
}

struct Lookup {
    data: Arc<LookupData>,
    embedding_cache: Mutex<Cache>,
    ple_cache: Mutex<Cache>,
    pool: RowPool,
}
// The GPU worker owns the pending job. Only immutable token spans and CPU
// lookup assets cross into the preparation thread.
pub(super) struct Inputs {
    lookup: Arc<Lookup>,
    pending: RefCell<Option<Pending>>,
}

struct Pending {
    tokens: Vec<u32>,
    history: Vec<u32>,
    capacity: usize,
    task: JoinHandle<Result<Uploads>>,
}

fn history_tail(history: &[u32]) -> &[u32] {
    &history[history.len().saturating_sub(2)..]
}

impl Inputs {
    pub(super) fn open(
        root: &Path,
        identity: &crate::artifact::FileIdentity,
        verify_weights: bool,
    ) -> Result<Self> {
        Ok(Self {
            lookup: Arc::new(Lookup::open(root, identity, verify_weights)?),
            pending: RefCell::default(),
        })
    }

    pub(super) fn prefetch(&self, tokens: &[u32], history: &[u32], capacity: usize) {
        let mut pending = self.pending.borrow_mut();
        if pending.as_ref().is_some_and(|job| !job.task.is_finished()) {
            return;
        }
        if let Some(job) = pending.take() {
            let _ = job.task.join();
        }
        let lookup = Arc::clone(&self.lookup);
        let tokens = tokens.to_vec();
        let history = history_tail(history).to_vec();
        let work_tokens = tokens.clone();
        let work_history = history.clone();
        if let Ok(task) = std::thread::Builder::new()
            .name("flash-inputs".into())
            .spawn(move || lookup.prepare_for("target", &work_tokens, &work_history, capacity))
        {
            *pending = Some(Pending {
                tokens,
                history,
                capacity,
                task,
            });
        }
    }

    pub(super) fn prepare_for(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
        capacity: usize,
    ) -> Result<Uploads> {
        if program == "target"
            && history
                .iter()
                .all(|&token| (token as usize) < self.lookup.data.assets.vocab)
        {
            let mut pending = self.pending.borrow_mut();
            if pending.as_ref().is_some_and(|job| {
                job.capacity == capacity
                    && job.tokens == tokens
                    && job.history == history_tail(history)
            }) {
                return pending
                    .take()
                    .unwrap()
                    .task
                    .join()
                    .map_err(|_| "Flash input preparation panicked")?;
            }
        }
        self.lookup.prepare_for(program, tokens, history, capacity)
    }
    #[cfg(test)]
    pub(super) fn prepare(
        &self,
        tokens: &[u32],
        history: &[u32],
    ) -> Result<Vec<(String, Vec<u8>)>> {
        self.prepare_for("target", tokens, history, tokens.len())
    }
}

impl Drop for Inputs {
    fn drop(&mut self) {
        // The native library must keep the thread's code mapped until it exits.
        if let Some(job) = self.pending.get_mut().take() {
            let _ = job.task.join();
        }
    }
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
    if !crate::artifact::is_sha256_hex(&part.sha256.to_ascii_lowercase()) {
        return Err("Invalid CPU row shard hash".into());
    }
    // SAFETY: Prepared assets remain immutable while open. Only the container
    // header is accessed here; the mapping is dropped before row lookup begins
    // so multi-GiB PLE shards never trigger mmap readahead or page-table thrash.
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
impl Lookup {
    pub(super) fn open(
        root: &Path,
        identity: &crate::artifact::FileIdentity,
        verify_weights: bool,
    ) -> Result<Self> {
        if identity.tensor.is_some() {
            return Err("CPU input metadata must be a standalone file".into());
        }
        let root = root.canonicalize().map_err(|e| e.to_string())?;
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
                let len = file.metadata().map_err(|e| e.to_string())?.len();
                let bytes = (part.rows as u64)
                    .checked_mul(if part.book.is_some() {
                        (2 + width / 4) as u64
                    } else {
                        width as u64
                    })
                    .and_then(|n| n.checked_add(part.offset))
                    .ok_or("CPU row extent overflow")?;
                if bytes > len {
                    return Err("CPU rows exceed file".into());
                }
                if part.scale_offset.is_some_and(|offset| {
                    offset
                        .checked_add(part.rows as u64 * 2)
                        .is_none_or(|end| end > len)
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
        let data = Arc::new(LookupData {
            assets,
            tables,
            embedding,
            ple,
        });
        Ok(Self {
            pool: RowPool::new(Arc::clone(&data)),
            data,
            embedding_cache: Default::default(),
            ple_cache: Default::default(),
        })
    }
    /// Resolve every row needed by one packet with batched cache locking and
    /// pread-backed decoding. Single-row misses run inline; multi-row misses
    /// reuse the persistent worker pool to overlap SSD reads and E8P decoding.
    fn fetch_rows(
        &self,
        embedding_rows: &[usize],
        ple_rows: &[usize],
    ) -> Result<(Vec<Row>, Vec<Row>)> {
        type RowKey = (usize, usize);
        type TableLookup = (Vec<Option<Row>>, Vec<RowKey>, Vec<RowKey>);
        let lookup_table =
            |sources: &[Source], rows: &[usize], cache: &Mutex<Cache>| -> Result<TableLookup> {
                let mut keys = Vec::with_capacity(rows.len());
                for &row in rows {
                    keys.push(shard_of(sources, row)?);
                }
                let mut resolved = Vec::with_capacity(rows.len());
                let mut need = Vec::new();
                {
                    let mut cache = cache.lock().unwrap();
                    for &key in &keys {
                        if let Some(hit) = cache.get(key) {
                            resolved.push(Some(hit));
                        } else {
                            resolved.push(None);
                            need.push(key);
                        }
                    }
                }
                need.sort_unstable();
                need.dedup();
                Ok((resolved, keys, need))
            };
        let (mut emb_resolved, emb_keys, emb_need) =
            lookup_table(&self.data.embedding, embedding_rows, &self.embedding_cache)?;
        let (mut ple_resolved, ple_keys, ple_need) = if ple_rows.is_empty() {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            lookup_table(&self.data.ple, ple_rows, &self.ple_cache)?
        };
        let total = emb_need.len() + ple_need.len();
        if total != 0 {
            let tasks: Vec<DecodeTask> = emb_need
                .iter()
                .map(|&(shard, local)| (false, shard, local))
                .chain(ple_need.iter().map(|&(shard, local)| (true, shard, local)))
                .collect();
            let decoded = self.pool.decode(tasks)?;
            let mut iter = decoded.into_iter();
            if !emb_need.is_empty() {
                let mut freshly = HashMap::with_capacity(emb_need.len());
                {
                    let mut cache = self.embedding_cache.lock().unwrap();
                    for &key in &emb_need {
                        let row = Arc::new(iter.next().expect("decoded embedding row"));
                        cache.put(key, Arc::clone(&row), EMBEDDING_CACHE_BUDGET);
                        freshly.insert(key, row);
                    }
                }
                for (slot, &key) in emb_resolved.iter_mut().zip(&emb_keys) {
                    if slot.is_none() {
                        *slot = Some(Arc::clone(&freshly[&key]));
                    }
                }
            }
            if !ple_need.is_empty() {
                let mut freshly = HashMap::with_capacity(ple_need.len());
                {
                    let mut cache = self.ple_cache.lock().unwrap();
                    for &key in &ple_need {
                        let row = Arc::new(iter.next().expect("decoded PLE row"));
                        cache.put(key, Arc::clone(&row), PLE_CACHE_BUDGET);
                        freshly.insert(key, row);
                    }
                }
                for (slot, &key) in ple_resolved.iter_mut().zip(&ple_keys) {
                    if slot.is_none() {
                        *slot = Some(Arc::clone(&freshly[&key]));
                    }
                }
            }
        }
        Ok((
            emb_resolved
                .into_iter()
                .map(|row| row.expect("embedding row resolved"))
                .collect(),
            ple_resolved
                .into_iter()
                .map(|row| row.expect("PLE row resolved"))
                .collect(),
        ))
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
        let assets = &self.data.assets;
        if tokens.is_empty()
            || tokens.len() > capacity
            || !(1..=4096).contains(&capacity)
            || (draft && program != "mtp_warm" && capacity != tokens.len())
            || tokens
                .iter()
                .chain(history)
                .any(|&t| t as usize >= assets.vocab)
        {
            return Err("Unsupported Flash input shape/token".into());
        }
        let mut previous: Vec<u32> = history
            .iter()
            .rev()
            .take_while(|&&t| t != assets.eos)
            .take(2)
            .copied()
            .collect();
        previous.reverse();
        let mut embedding_rows: Vec<usize> = Vec::with_capacity(tokens.len());
        let mut ple_rows: Vec<usize> =
            Vec::with_capacity(if draft { 0 } else { tokens.len() * 16 });
        for &token in tokens {
            embedding_rows.push(token as usize);
            if draft {
                continue;
            }
            let mut mixed = token as u64 * assets.multipliers[0];
            for shift in 1..=2 {
                let prior = previous
                    .len()
                    .checked_sub(shift)
                    .map(|i| previous[i])
                    .unwrap_or(assets.eos);
                mixed ^= prior as u64 * assets.multipliers[shift];
                for head in (shift - 1) * 8..shift * 8 {
                    ple_rows.push((mixed % assets.sizes[head] + assets.offsets[head]) as usize);
                }
            }
            if token == assets.eos {
                previous.clear();
            } else {
                previous.push(token);
                if previous.len() > 2 {
                    previous.remove(0);
                }
            }
        }
        let (emb_values, ple_values) = self.fetch_rows(&embedding_rows, &ple_rows)?;
        let mut embedding = Vec::with_capacity(tokens.len() * 5120);
        for row in &emb_values {
            extend_u16_le(&mut embedding, row);
        }
        let mut ple = Vec::with_capacity(if draft { 0 } else { tokens.len() * 5120 });
        for row in &ple_values {
            extend_u16_le(&mut ple, row);
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

#[inline]
fn extend_u16_le(dst: &mut Vec<u8>, src: &[u16]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: u16 has no padding bytes and target byte order is little-endian.
        let bytes = unsafe {
            std::slice::from_raw_parts(src.as_ptr().cast::<u8>(), std::mem::size_of_val(src))
        };
        dst.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &value in src {
            dst.extend_from_slice(&value.to_le_bytes());
        }
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
const fn paley_sign(i: usize, j: usize) -> f32 {
    if j == 0 {
        1.
    } else if i == 0 {
        -1.
    } else {
        let delta = (j + 19 - i) % 19;
        if delta == 0 {
            return 1.;
        }
        let mut k = 1;
        while k < 19 {
            if k * k % 19 == delta {
                return 1.;
            }
            k += 1;
        }
        -1.
    }
}

const PALEY_20: [[f32; 20]; 20] = {
    let mut table = [[0.; 20]; 20];
    let mut i = 0;
    while i < 20 {
        let mut j = 0;
        while j < 20 {
            table[i][j] = paley_sign(i, j);
            j += 1;
        }
        i += 1;
    }
    table
};

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
    let mut stack = [0u8; 2560];
    let packet = &mut stack[..stride];
    source
        .file
        .read_exact_at(packet, source.offset + (local * stride) as u64)
        .map_err(|e| e.to_string())?;
    if let Some(book) = source.book {
        return decode(packet, &tables[book]);
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
fn decode(packet: &[u8], table: &Table) -> Result<Vec<u16>> {
    match table.signs.len() {
        160 => decode_fixed::<160, 8>(packet, table),
        2560 => decode_fixed::<2560, 128>(packet, table),
        _ => Err("Invalid E8P row width".into()),
    }
}
fn decode_fixed<const WIDTH: usize, const COLS: usize>(
    packet: &[u8],
    table: &Table,
) -> Result<Vec<u16>> {
    let scale = f16::from_bits(u16::from_le_bytes([packet[0], packet[1]])).to_f32();
    if !scale.is_finite() || scale <= 0. {
        return Err("Invalid E8P row scale".into());
    }
    let mut rotated = [0f32; WIDTH];
    for (i, code) in packet[2..].as_chunks::<2>().0.iter().enumerate() {
        let values = &table.values[u16::from_le_bytes([code[0], code[1]]) as usize];
        let base = i * 8;
        for j in 0..8 {
            rotated[base + j] = values[j] as f32 * scale;
        }
    }
    let paley_norm = (20f32).sqrt().recip();
    let col_norm = (COLS as f32).sqrt().recip();
    let mut result = vec![0u16; WIDTH];
    for (i, signs_row) in PALEY_20.iter().enumerate() {
        let mut acc = [0f32; COLS];
        for (j, &sign) in signs_row.iter().enumerate() {
            let coefficient = sign * paley_norm;
            let src = &rotated[j * COLS..(j + 1) * COLS];
            for col in 0..COLS {
                acc[col] += coefficient * src[col];
            }
        }
        let mut half = 1;
        while half < COLS {
            for block in acc.chunks_exact_mut(half * 2) {
                for j in 0..half {
                    let a = block[j];
                    let b = block[j + half];
                    block[j] = a + b;
                    block[j + half] = a - b;
                }
            }
            half *= 2;
        }
        let base = i * COLS;
        let dst = &mut result[base..base + COLS];
        let sign_slice = &table.signs[base..base + COLS];
        for col in 0..COLS {
            dst[col] = f16::from_f32((acc[col] * col_norm) * sign_slice[col] as f32).to_bits();
        }
    }
    Ok(result)
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
        let row = Arc::new(vec![0x3c00; 64]);
        let budget = row.len() * 2 + 512;
        cache.put((0, 0), Arc::clone(&row), budget);
        let hit = cache.get((0, 0)).unwrap();
        assert!(Arc::ptr_eq(&hit, &row));
        cache.put((0, 1), Arc::new(vec![0x4000; 64]), budget);
        assert!(cache.get((0, 0)).is_none());
        assert_eq!(hit.as_slice(), &[0x3c00; 64]);
        assert_eq!(cache.get((0, 1)).unwrap().as_slice(), &[0x4000; 64]);
        assert!(cache.bytes <= budget);
    }

    #[test]
    fn fast_e8p_decode_matches_reference_bit_for_bit() {
        let reference_decode = |packet: &[u8], table: &Table| -> Vec<u16> {
            let scale = f16::from_bits(u16::from_le_bytes([packet[0], packet[1]])).to_f32();
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
            out.iter()
                .zip(&table.signs)
                .map(|(&x, &s)| f16::from_f32((x * norm) * s as f32).to_bits())
                .collect()
        };
        let values: Vec<[i8; 8]> = (0..65536u32)
            .map(|code| {
                std::array::from_fn(|j| {
                    (((code.wrapping_mul(17).wrapping_add(j as u32 * 31)) % 127) as i8) - 63
                })
            })
            .collect();
        for width in [160usize, 2560usize] {
            let table = Table {
                values: values.clone(),
                signs: (0..width)
                    .map(|i| if (i * 7 + 3) % 3 == 0 { -1 } else { 1 })
                    .collect(),
            };
            let mut packet = vec![0u8; 2 + width / 4];
            let scale_bits = f16::from_f32(0.03125).to_bits().to_le_bytes();
            packet[0] = scale_bits[0];
            packet[1] = scale_bits[1];
            for (i, chunk) in packet[2..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let code = ((i as u32).wrapping_mul(4099).wrapping_add(13)) as u16;
                chunk.copy_from_slice(&code.to_le_bytes());
            }
            let actual = decode(&packet, &table).unwrap();
            let expected = reference_decode(&packet, &table);
            assert_eq!(actual, expected, "width={width} mismatch");
        }
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
            Path::new(&root),
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
            let synchronous = inputs.prepare(&[token], &history).unwrap();
            inputs.prefetch(&[token], &history, 1);
            let rows = inputs.prepare(&[token], &history).unwrap();
            assert_eq!(synchronous, rows);
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
        assert!(inputs.lookup.embedding_cache.lock().unwrap().bytes <= EMBEDDING_CACHE_BUDGET);
        assert!(inputs.lookup.ple_cache.lock().unwrap().bytes <= PLE_CACHE_BUDGET);
        assert!(inputs.prepare(&[248320], &[]).is_err());
    }
}
