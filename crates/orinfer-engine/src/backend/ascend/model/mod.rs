//! Original-checkpoint single-sequence A2 execution and latency measurements.
mod affinity;
mod batch;
mod batch_benchmark;
mod benchmark;
mod branches;
#[cfg(test)]
mod dispatch_tests;
mod packet;
pub(crate) mod provider;
mod replay;
mod row_contract;
mod state;
mod team;
mod weights;

use super::{
    hccl::{Communicator, Counts, Element, Exchange, RootInfo},
    kernels::{Kernel, Template},
    operators::{GroupedInputs, GroupedShape, GroupedW4A8},
    runtime::{Buffer, Limits, Session, Ticket},
};
use crate::{
    artifact::{FileIdentity, Result, read_identity, resolve_file},
    host_rows::{Int8Rows, Shard, SharedInt8Rows},
    weights::checkpoint::Checkpoint,
};
use orinfer_model_sdk::conditional_memory::{HashLayout, Hasher};
use orinfer_model_sdk::operators::dynamic::RowExpression;
use orinfer_model_sdk::operators::rows::KernelVariant;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    rc::{Rc, Weak},
    sync::{Arc, OnceLock},
    time::Instant,
};

pub use batch_benchmark::run as run_batch;
pub use benchmark::{Report, run};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Model {
    schema_version: u32,
    config: FileIdentity,
    index: FileIdentity,
    engram: FileIdentity,
    capacity: usize,
    layers: usize,
    ranks: Vec<Rank>,
    rope: BTreeMap<String, FileIdentity>,
    #[serde(default)]
    row_selections: BTreeMap<String, Vec<KernelVariant>>,
    #[serde(default = "default_packet_cache_capacity")]
    packet_cache_capacity: usize,
    #[serde(default)]
    batch: Option<batch::Contract>,
    #[serde(skip)]
    graph_launches: std::collections::BTreeSet<batch::WorkLaunch>,
}
fn default_packet_cache_capacity() -> usize {
    64
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Rank {
    rank: u32,
    buffers: Vec<Definition>,
    packets: Vec<Packet>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Definition {
    name: String,
    bytes: usize,
    view: Option<View>,
    weight: Option<weights::Weight>,
    data: Option<FileIdentity>,
    prepared_weight: Option<String>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct View {
    parent: String,
    offset: usize,
    #[serde(default)]
    row_capacity: Option<usize>,
    #[serde(default)]
    offset_expression: Option<RowExpression>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Packet {
    rows: usize,
    #[serde(default)]
    min_rows: usize,
    #[serde(default)]
    max_rows: Option<usize>,
    coordinates: packet::Coordinates,
    #[serde(default)]
    decoder_coordinates: Option<packet::Coordinates>,
    body: Vec<Instruction>,
    head: Vec<Instruction>,
    reset: Vec<String>,
    best_value: String,
    best_index: String,
    logits: Option<String>,
    vocab_first: u32,
    #[serde(default)]
    batch_decode: bool,
}
impl Packet {
    fn dispatch_max(&self) -> usize {
        self.max_rows.unwrap_or(self.rows)
    }
    fn prefill_plan(&self) -> orinfer_model_sdk::model::PrefillPlan {
        orinfer_model_sdk::model::PrefillPlan {
            chunk_tokens: self.dispatch_max(),
            min_tokens: self.min_rows,
            prefill_program: "prefill".into(),
            head_program: "head".into(),
            stages: vec![],
            live_buffers: vec![],
        }
    }
}
fn embedding_width() -> usize {
    5120
}

#[derive(Clone, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Instruction {
    Fork,
    Switch,
    Join,
    /// Subsequent row-dependent launches consume an already compacted suffix.
    /// Coordinates and live state must be compacted by preceding kernels.
    LimitRows {
        rows: usize,
    },
    /// Select each owner's bounded suffix, preserving the packet capacity.
    DecoderRows {
        keep: usize,
        prepare: usize,
        select: usize,
    },
    Gather {
        source: String,
        destination: String,
        elements: u64,
        fp32: bool,
        #[serde(default)]
        int32: bool,
        #[serde(default)]
        fixed_rows: bool,
        #[serde(default)]
        elements_expression: Option<RowExpression>,
        #[serde(default)]
        trace_only: bool,
    },
    Exchange {
        source: String,
        destination: String,
        /// Element count per peer, evaluated for the current global row count.
        elements: u64,
        fp32: bool,
        #[serde(default)]
        int32: bool,
        #[serde(default)]
        fixed_rows: bool,
        #[serde(default)]
        elements_expression: Option<RowExpression>,
    },
    Kernel {
        manifest: String,
        #[serde(default)]
        uncompacted_manifest: Option<String>,
        buffers: Vec<String>,
        #[serde(default)]
        fixed_rows: Option<usize>,
        #[serde(default)]
        row_expression: Option<RowExpression>,
        #[serde(default)]
        owner_work: Option<RowExpression>,
        #[serde(default)]
        owners: bool,
    },
    Reduce {
        source: String,
        destination: String,
        elements: u64,
        fp32: bool,
        #[serde(default)]
        scatter: bool,
        #[serde(default)]
        fixed_rows: bool,
        #[serde(default)]
        elements_expression: Option<RowExpression>,
    },
    W4 {
        name: String,
        rows: usize,
        input: usize,
        output: usize,
        experts: usize,
        buffers: [String; 6],
        workspace: String,
        groups: Option<String>,
    },
    Route {
        indices: String,
        gather: String,
        restore: String,
        groups: Vec<String>,
        rows: usize,
    },
    Embed {
        destination: String,
        #[serde(default)]
        first_column: usize,
        #[serde(default = "embedding_width")]
        columns: usize,
    },
    Lookup {
        layer: usize,
        values: String,
        scales: String,
        mask: String,
    },
    Snapshot {
        name: String,
        buffer: String,
    },
}
#[derive(Clone)]
enum Step {
    Graph(Rc<super::graph::Graph>),
    Segment(Rc<Segment>),
    W4(Rc<GroupedW4A8>),
    Route {
        indices: Buffer,
        gather: Buffer,
        restore: Buffer,
        groups: Vec<String>,
        rows: usize,
    },
    Embed {
        destination: Buffer,
        first_column: usize,
        columns: usize,
    },
    Lookup {
        layer: usize,
        values: Buffer,
        scales: Buffer,
        mask: Buffer,
    },
    Snapshot {
        name: String,
        buffer: Buffer,
    },
}
/// One launch inside a captured segment: kernels, HCCL SUM collectives and
/// repeatable ACLNN grouped matmuls replay together inside one ACL Graph.
#[derive(Clone)]
enum SegmentOp {
    Fork(usize),
    Switch(usize),
    Join(usize),
    Gather {
        source: Buffer,
        destination: Buffer,
        elements: u64,
        element: Element,
    },
    Exchange(Exchange),
    Kernel(Kernel),
    Reduce {
        source: Buffer,
        destination: Buffer,
        elements: u64,
        element: Element,
        scatter: bool,
    },
    W4(Rc<GroupedW4A8>),
}

fn enqueue_segment(
    ops: &[SegmentOp],
    comm: &Communicator,
    parallel: Option<&super::graph::Branches>,
    main: super::api::Handle,
) -> Result<()> {
    let mut stream = main;
    let mut parents = Vec::new();
    for op in ops {
        match op {
            SegmentOp::Fork(index) => {
                parents.push(stream);
                stream = parallel
                    .ok_or("ACL branch resources missing")?
                    .fork(*index, stream)?;
            }
            SegmentOp::Switch(index) => {
                parallel
                    .ok_or("ACL branch resources missing")?
                    .switch(*index)?;
                stream = *parents.last().ok_or("ACL branch parent missing")?;
            }
            SegmentOp::Join(index) => {
                parallel
                    .ok_or("ACL branch resources missing")?
                    .join(*index, stream)?;
                stream = parents.pop().ok_or("ACL branch parent missing")?;
            }
            SegmentOp::Exchange(exchange) => exchange.execute_on_stream(stream)?,
            SegmentOp::Kernel(kernel) => kernel.launch_on_stream(stream)?,
            SegmentOp::Gather {
                source,
                destination,
                elements,
                element,
            } => {
                comm.gather_on_stream(stream, source, destination, *elements, *element)?;
            }
            SegmentOp::Reduce {
                source,
                destination,
                elements,
                element,
                scatter,
            } => {
                if *scatter {
                    comm.scatter_on_stream(stream, source, destination, *elements, *element)?;
                } else {
                    comm.reduce_on_stream(stream, source, destination, *elements, *element)?;
                }
            }
            SegmentOp::W4(plan) => plan.execute_on_stream(stream)?,
        }
    }
    Ok(())
}
/// Bound launches and branch resources shared by direct execution and capture.
struct Segment {
    ops: Rc<Vec<SegmentOp>>,
    comm: Communicator,
    parallel: Option<Rc<super::graph::Branches>>,
}
impl Segment {
    fn fence_result<T>(&self, result: Result<T>) -> Result<T> {
        result.or_else(|error| {
            if let Some(parallel) = &self.parallel {
                parallel
                    .fence()
                    .map_err(|fence| format!("{error}; {fence}"))?;
            }
            Err(error)
        })
    }
    fn execute_on_stream(&self, stream: super::api::Handle) -> Result<()> {
        self.fence_result(enqueue_segment(
            &self.ops,
            &self.comm,
            self.parallel.as_deref(),
            stream,
        ))
    }
    fn enqueue(self: &Rc<Self>, session: &Session) -> Result<Ticket> {
        // SAFETY: This bound segment owns every operand, executor, communicator
        // and branch. The completion retains it and fences partial submission.
        unsafe {
            session.submit_compute(vec![], Box::new(Rc::clone(self)), |stream| {
                self.execute_on_stream(stream)
            })
        }
    }
}
struct Execution {
    packet: Packet,
    body: Vec<Step>,
    head: Vec<Step>,
    captured: bool,
}
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct W4Key {
    name: String,
    rows: usize,
    input: usize,
    output: usize,
    experts: usize,
    pointers: [usize; 6],
    groups: Option<usize>,
    workspace: usize,
}
type SharedTables = OnceLock<Result<Arc<BTreeMap<usize, SharedInt8Rows>>>>;

fn host_row_cache_budget() -> Result<usize> {
    host_memory_budget("ORINFER_A2_HOST_ROW_CACHE_MIB", 2048)
}
fn host_memory_budget(name: &str, default_mib: usize) -> Result<usize> {
    let mib = match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("Invalid {name} MiB"))?,
        Err(std::env::VarError::NotPresent) => default_mib,
        Err(_) => return Err(format!("Invalid {name} environment")),
    };
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| format!("{name} budget overflow"))
}
struct Worker {
    session: Session,
    comm: Communicator,
    buffers: BTreeMap<String, Buffer>,
    views: BTreeMap<String, (View, usize)>,
    row_selections: BTreeMap<String, Vec<KernelVariant>>,
    w4: BTreeMap<W4Key, Weak<GroupedW4A8>>,
    routed_w4: BTreeMap<String, (usize, Weak<GroupedW4A8>)>,
    packets: BTreeMap<usize, Rc<Execution>>,
    packet_order: row_contract::PacketLru,
    packet_templates: Vec<Packet>,
    package: PathBuf,
    snapshots: bool,
    checkpoint: Checkpoint,
    hasher: Hasher,
    tables: Arc<BTreeMap<usize, SharedInt8Rows>>,
    rope: BTreeMap<String, Vec<u8>>,
    capacity: usize,
    rank: u32,
    templates: BTreeMap<PathBuf, Rc<Template>>,
    phases: LoadPhases,
    steps: BTreeMap<usize, packet::StepStats>,
    batch: Option<batch::Runtime>,
}

/// Host-to-device chunk and pipeline depth; in-flight pinned memory stays
/// within the session budget while DMA overlaps the next host copy.
const CHUNK: usize = 8 << 20;
const DEPTH: usize = 6;

/// Per-phase cold-start accounting for one rank; every phase is wall time
/// spent on that rank's load thread.
#[derive(Default)]
struct Phase {
    bytes: usize,
    seconds: f64,
}
impl Phase {
    fn add(&mut self, bytes: usize, seconds: f64) {
        self.bytes += bytes;
        self.seconds += seconds;
    }
    fn report(&self) -> String {
        if self.seconds <= 0.0 {
            return format!("{:.2}GiB/instant", self.bytes as f64 / (1 << 30) as f64);
        }
        format!(
            "{:.2}GiB/{:.1}s({:.2}GiB/s)",
            self.bytes as f64 / (1 << 30) as f64,
            self.seconds,
            self.bytes as f64 / (1 << 30) as f64 / self.seconds
        )
    }
}
#[derive(Default)]
struct LoadPhases {
    hccl_s: f64,
    alloc_s: f64,
    prepared: Phase,
    matrix: Phase,
    data: Phase,
    zero: Phase,
    tables_s: f64,
    templates: (usize, f64),
    binds: (usize, f64),
    aclnn: (usize, f64),
    warmups: (usize, f64),
    captures: (usize, f64),
}

fn wait_in_flight(tickets: &mut VecDeque<Ticket>) -> Result<()> {
    while let Some(ticket) = tickets.pop_front() {
        ticket.wait()?;
    }
    Ok(())
}

/// Pipelined upload: submitted chunks overlap host copies with in-flight DMA.
fn upload(session: &Session, buffer: &Buffer, data: &[u8]) -> Result<()> {
    if data.len() != buffer.bytes() {
        return Err("Model upload byte extent differs".into());
    }
    let mut tickets = VecDeque::new();
    for (chunk, bytes) in data.chunks(CHUNK).enumerate() {
        tickets.push_back(session.upload(buffer, chunk * CHUNK, bytes)?);
        if tickets.len() > DEPTH {
            tickets.pop_front().unwrap().wait()?;
        }
    }
    wait_in_flight(&mut tickets)
}

/// Streams a prepared weight file without materializing it in host memory;
/// storage reads overlap device DMA instead of preceding it.
fn upload_file(session: &Session, buffer: &Buffer, path: &Path, bytes: usize) -> Result<()> {
    use std::io::Read;
    let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() != bytes as u64 {
        return Err("Prepared weight metadata differs".into());
    }
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut staging = vec![0u8; CHUNK];
    let mut tickets = VecDeque::new();
    let mut offset = 0;
    while offset < bytes {
        let want = CHUNK.min(bytes - offset);
        file.read_exact(&mut staging[..want])
            .map_err(|e| e.to_string())?;
        // The session copies staging into pinned storage before returning.
        tickets.push_back(session.upload(buffer, offset, &staging[..want])?);
        offset += want;
        if tickets.len() > DEPTH {
            tickets.pop_front().unwrap().wait()?;
        }
    }
    wait_in_flight(&mut tickets)
}

/// Debug-only integrity check: chunked download and comparison against the
/// exact bytes that were uploaded. On mismatch, maps nearby device chunks
/// back to source chunk indices to reveal duplication, swaps or loss.
fn verify_upload(
    session: &Session,
    buffer: &Buffer,
    name: &str,
    mut source: impl FnMut(usize, &mut [u8]) -> Result<()>,
) -> Result<()> {
    let chunks = buffer.bytes().div_ceil(CHUNK);
    let mut expected = vec![0u8; CHUNK];
    for index in 0..chunks {
        let want = CHUNK.min(buffer.bytes() - index * CHUNK);
        source(index, &mut expected[..want])?;
        let actual = session.download(buffer, index * CHUNK, want)?;
        if actual != expected[..want] {
            let mut first = 0;
            while first < want && actual[first] == expected[first] {
                first += 1;
            }
            let mut mapping = vec![];
            for probe in index..(index + 4).min(chunks) {
                let chunk = session.download(
                    buffer,
                    probe * CHUNK,
                    CHUNK.min(buffer.bytes() - probe * CHUNK),
                )?;
                let mut matches = vec![];
                for candidate in 0..(chunks.min(8)) {
                    let mut candidate_data = vec![0u8; CHUNK];
                    let candidate_want = CHUNK.min(buffer.bytes() - candidate * CHUNK);
                    source(candidate, &mut candidate_data[..candidate_want])?;
                    if chunk == candidate_data[..candidate_want] {
                        matches.push(candidate);
                    }
                }
                mapping.push((probe, matches));
            }
            return Err(format!(
                "{name}: upload verification differs at chunk {index} byte {first}, expected {:02x?}, actual {:02x?}, chunk map {mapping:?}",
                &expected[first..(first + 16).min(want)],
                &actual[first..(first + 16).min(want)]
            ));
        }
    }
    eprintln!("A2 verified {name}");
    Ok(())
}

fn verify_file_upload(session: &Session, buffer: &Buffer, path: &Path, name: &str) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    verify_upload(session, buffer, name, |index, out| {
        file.seek(SeekFrom::Start((index * CHUNK) as u64))
            .and_then(|_| file.read_exact(out))
            .map_err(|e| e.to_string())
    })
}

fn verify_data_upload(session: &Session, buffer: &Buffer, data: &[u8], name: &str) -> Result<()> {
    verify_upload(session, buffer, name, |index, out| {
        let begin = index * CHUNK;
        out.copy_from_slice(&data[begin..begin + out.len()]);
        Ok(())
    })
}
fn get(buffers: &BTreeMap<String, Buffer>, name: &str) -> Result<Buffer> {
    buffers
        .get(name)
        .cloned()
        .ok_or_else(|| format!("Model buffer is not defined: {name}"))
}
impl Worker {
    fn row_buffer(&self, name: &str, rows: usize) -> Result<Buffer> {
        fn resolve(worker: &Worker, name: &str, rows: usize, depth: usize) -> Result<Buffer> {
            if depth > 64 {
                return Err("A2 view chain exceeds nesting limit".into());
            }
            if let Some((view, bytes)) = worker.views.get(name) {
                resolve(worker, &view.parent, rows, depth + 1)?
                    .slice(row_contract::view_offset(view, rows)?, *bytes)
            } else {
                get(&worker.buffers, name)
            }
        }
        resolve(self, name, rows, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn load(
        session: Session,
        root: &RootInfo,
        rank: Rank,
        model: &Model,
        model_dir: &Path,
        package: &Path,
        layout: HashLayout,
        snapshots: bool,
        shared_tables: &SharedTables,
    ) -> Result<Self> {
        let start = Instant::now();
        let mut phases = LoadPhases::default();
        let t = Instant::now();
        let comm = Communicator::join(&session, root, rank.rank)?;
        phases.hccl_s = t.elapsed().as_secs_f64();
        let mut checkpoint = Checkpoint::open(model_dir, &model.index.file)?;
        let views = rank
            .buffers
            .iter()
            .filter_map(|definition| {
                definition
                    .view
                    .clone()
                    .map(|view| (definition.name.clone(), (view, definition.bytes)))
            })
            .collect();
        let mut buffers = BTreeMap::new();
        let needed = rank
            .buffers
            .iter()
            .filter(|b| b.view.is_none())
            .try_fold(0usize, |n, b| {
                n.checked_add(b.bytes).ok_or("Model byte estimate overflow")
            })?;
        if needed > session.memory_info()?.0.saturating_sub(1 << 30) {
            return Err(format!(
                "Rank {} needs {needed} bytes plus 1 GiB headroom",
                rank.rank
            ));
        }
        let mut fills: Vec<Ticket> = Vec::new();
        for definition in rank.buffers {
            if buffers.contains_key(&definition.name) || definition.bytes == 0 {
                return Err("Duplicate/empty model buffer".into());
            }
            let buffer = if let Some(view) = definition.view {
                if definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                {
                    return Err("Model view cannot own weight data".into());
                }
                get(&buffers, &view.parent)?.slice(view.offset, definition.bytes)?
            } else {
                let t = Instant::now();
                let buffer = session.allocate(definition.bytes)?;
                phases.alloc_s += t.elapsed().as_secs_f64();
                let verify = std::env::var_os("ORINFER_A2_VERIFY_UPLOAD").is_some();
                if let Some(file) = definition.prepared_weight {
                    let path = resolve_file(package, &file)?;
                    let t = Instant::now();
                    upload_file(&session, &buffer, &path, definition.bytes)?;
                    phases
                        .prepared
                        .add(definition.bytes, t.elapsed().as_secs_f64());
                    if verify {
                        verify_file_upload(&session, &buffer, &path, &definition.name)?;
                    }
                } else if let Some(weight) = definition.weight {
                    let t = Instant::now();
                    let data = weights::load(&mut checkpoint, &weight)?;
                    upload(&session, &buffer, &data)?;
                    phases
                        .matrix
                        .add(definition.bytes, t.elapsed().as_secs_f64());
                    if verify {
                        verify_data_upload(&session, &buffer, &data, &definition.name)?;
                    }
                } else if let Some(identity) = definition.data {
                    let t = Instant::now();
                    let data = read_identity(package, &identity)?;
                    upload(&session, &buffer, &data)?;
                    phases.data.add(definition.bytes, t.elapsed().as_secs_f64());
                    if verify {
                        verify_data_upload(&session, &buffer, &data, &definition.name)?;
                    }
                } else {
                    // Fills are compute-stream ordered before every later use;
                    // deferring their waits keeps the transfer stream busy.
                    let t = Instant::now();
                    fills.push(session.fill(&buffer, 0)?);
                    phases.zero.add(definition.bytes, t.elapsed().as_secs_f64());
                }
                buffer
            };
            buffers.insert(definition.name, buffer);
        }
        let t = Instant::now();
        for ticket in &fills {
            ticket.wait()?;
        }
        phases.zero.seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let tables = shared_tables
            .get_or_init(|| {
                let mut stores = BTreeMap::new();
                for layer in &layout.layers {
                    let prefix = format!("layers.{}.engram.embed", layer.layer);
                    stores.insert(
                        layer.layer,
                        Int8Rows::open(
                            &checkpoint.path(&format!("{prefix}.weight"))?,
                            &format!("{prefix}.weight"),
                            &checkpoint.path(&format!("{prefix}.scale"))?,
                            &format!("{prefix}.scale"),
                            Shard {
                                first: 0,
                                rows: layer.rows,
                            },
                        )?,
                    );
                }
                let mut resident_budget = host_memory_budget("ORINFER_A2_HOST_TABLE_MIB", 0)?;
                let resident = resident_budget != 0;
                let required =
                    stores
                        .values()
                        .try_fold(0usize, |total, store| -> Result<usize> {
                            total
                                .checked_add(store.required_resident_bytes()?)
                                .ok_or("Host table total extent overflow".into())
                        })?;
                if resident && required > resident_budget {
                    return Err("A2 resident host tables exceed configured budget".into());
                }
                let table_budget = if resident {
                    0
                } else {
                    host_row_cache_budget()? / layout.layers.len().max(1)
                };
                let mut tables = BTreeMap::new();
                for (layer, mut store) in stores {
                    if resident {
                        store = store.into_resident(resident_budget)?;
                        resident_budget -= store.resident_bytes();
                    }
                    tables.insert(layer, SharedInt8Rows::with_cache(store, table_budget)?);
                }
                Ok(Arc::new(tables))
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let tables = Arc::clone(tables);
        let rope = model
            .rope
            .iter()
            .map(|(name, id)| Ok((name.clone(), read_identity(package, id)?)))
            .collect::<Result<_>>()?;
        phases.tables_s = t.elapsed().as_secs_f64();
        let mut worker = Self {
            session,
            comm,
            buffers,
            views,
            row_selections: model.row_selections.clone(),
            w4: BTreeMap::new(),
            routed_w4: BTreeMap::new(),
            packets: BTreeMap::new(),
            packet_order: row_contract::PacketLru::new(model.packet_cache_capacity),
            packet_templates: rank.packets.clone(),
            package: package.to_owned(),
            snapshots,
            checkpoint,
            hasher: Hasher::new(layout)?,
            tables,
            rope,
            capacity: model.capacity,
            rank: rank.rank,
            templates: BTreeMap::new(),
            phases,
            steps: BTreeMap::new(),
            batch: model.batch.as_ref().map(|contract| {
                batch::Runtime::new(
                    contract.clone(),
                    model.packet_cache_capacity,
                    &rank.packets,
                    &model.graph_launches,
                )
            }),
        };
        if model.batch.is_some() {
            return Ok(worker);
        }
        for packet in rank.packets {
            let actual = packet.dispatch_max();
            let body = worker.compile(&packet.body, package, snapshots, (packet.rows, actual))?;
            let head = worker.compile(&packet.head, package, snapshots, (packet.rows, actual))?;
            if worker
                .packets
                .insert(
                    actual,
                    Rc::new(Execution {
                        packet,
                        body,
                        head,
                        captured: true,
                    }),
                )
                .is_some()
            {
                return Err("Duplicate packet row count".into());
            }
            worker.touch_packet(actual);
        }
        let phases = &worker.phases;
        eprintln!(
            "A2 rank {} load {:.1}s: hccl={:.1}s alloc={:.1}s prepared={} matrix={} data={} zero={} tables={:.1}s templates={}/{:.1}s binds={}/{:.1}s aclnn={}/{:.1}s warmup={}/{:.1}s captures={}/{:.1}s",
            rank.rank,
            start.elapsed().as_secs_f64(),
            phases.hccl_s,
            phases.alloc_s,
            phases.prepared.report(),
            phases.matrix.report(),
            phases.data.report(),
            phases.zero.report(),
            phases.tables_s,
            phases.templates.0,
            phases.templates.1,
            phases.binds.0,
            phases.binds.1,
            phases.aclnn.0,
            phases.aclnn.1,
            phases.warmups.0,
            phases.warmups.1,
            phases.captures.0,
            phases.captures.1,
        );
        Ok(worker)
    }
    fn compile(
        &mut self,
        instructions: &[Instruction],
        base: &Path,
        snapshots: bool,
        extent: (usize, usize),
    ) -> Result<Vec<Step>> {
        self.compile_bound(instructions, base, snapshots, extent, None, true)
    }
    fn compile_bound(
        &mut self,
        instructions: &[Instruction],
        base: &Path,
        snapshots: bool,
        extent: (usize, usize),
        layout: Option<&batch::Layout>,
        capture: bool,
    ) -> Result<Vec<Step>> {
        let (mut capacity, actual) = extent;
        let mut active_rows = actual;
        branches::validate_instructions(instructions)?;
        let boundary = replay::Boundary::find(instructions)?;
        let decoder_layout = boundary
            .as_ref()
            .map(|b| {
                layout
                    .ok_or("Decoder rows require an owner layout")?
                    .decoder(
                        b.keep,
                        self.batch.as_ref().is_some_and(|r| {
                            r.contract
                                .decoder_replay
                                .as_ref()
                                .is_some_and(|r| r.final_prompt_only)
                        }),
                    )
            })
            .transpose()?;
        let clipped = decoder_layout
            .as_ref()
            .is_some_and(|l| l.rows.total < actual);
        let empty_decoder = decoder_layout.as_ref().is_some_and(|l| l.rows.total == 0);
        let mut current_layout = layout;
        let mut steps = vec![];
        let mut pending: Vec<SegmentOp> = vec![];
        let mut fork_count = 0;
        let mut fork_stack = Vec::new();
        let routed: std::collections::BTreeSet<_> = instructions
            .iter()
            .filter_map(|i| {
                if let Instruction::Route { groups, .. } = i {
                    Some(groups)
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        for (index, instruction) in instructions.iter().enumerate() {
            if (!clipped || empty_decoder) && boundary.as_ref().is_some_and(|b| b.skip(index)) {
                continue;
            }
            let scaled = |n: usize| -> Result<usize> {
                if capacity == 0 || !n.is_multiple_of(capacity) {
                    return Err("Packet row expression differs from capacity".into());
                }
                (n / capacity)
                    .checked_mul(active_rows)
                    .ok_or_else(|| "Packet row extent overflow".into())
            };
            match instruction {
                Instruction::DecoderRows { .. } => {
                    current_layout = decoder_layout.as_ref();
                    active_rows = current_layout.ok_or("Missing decoder layout")?.rows.total;
                    if active_rows == 0 {
                        break;
                    }
                    continue;
                }
                Instruction::LimitRows { rows } => {
                    active_rows = bounded_rows(active_rows, capacity, *rows)?;
                    capacity = *rows;
                    continue;
                }
                Instruction::Fork => {
                    pending.push(SegmentOp::Fork(fork_count));
                    fork_stack.push(fork_count);
                    fork_count += 1;
                    continue;
                }
                Instruction::Switch => {
                    pending.push(SegmentOp::Switch(
                        *fork_stack.last().ok_or("ACL switch has no fork")?,
                    ));
                    continue;
                }
                Instruction::Join => {
                    pending.push(SegmentOp::Join(
                        fork_stack.pop().ok_or("ACL join has no fork")?,
                    ));
                    continue;
                }
                _ => {}
            }
            if let Instruction::Snapshot { name, buffer } = instruction {
                // Snapshots only observe device state for tracing; without
                // tracing the step is inert and kernels merge across it.
                if snapshots {
                    self.capture_pending(&mut pending, &mut steps, capture)?;
                    fork_count = 0;
                    steps.push(Step::Snapshot {
                        name: name.clone(),
                        buffer: {
                            let b = get(&self.buffers, buffer)?;
                            if current_layout.is_some() {
                                b.slice(0, b.bytes() / capacity * active_rows)?
                            } else {
                                b
                            }
                        },
                    });
                }
                continue;
            }
            if let Instruction::Kernel {
                manifest,
                uncompacted_manifest,
                buffers,
                fixed_rows,
                row_expression,
                owner_work,
                owners,
            } = instruction
            {
                let rows = if let Some(grid) = owner_work {
                    self.batch
                        .as_ref()
                        .ok_or("Missing batch runtime")?
                        .contract
                        .work_items(
                            current_layout.ok_or("Owner kernel needs batch layout")?,
                            grid,
                            &buffers[0],
                            self.rank as usize,
                        )?
                        .len()
                        .div_ceil(2)
                } else if *owners {
                    current_layout
                        .ok_or("Head kernel needs batch layout")?
                        .outputs
                        .len()
                } else {
                    row_contract::kernel_rows(
                        active_rows,
                        capacity,
                        *fixed_rows,
                        row_expression.as_ref(),
                    )?
                };
                let manifest = if clipped {
                    manifest
                } else {
                    uncompacted_manifest.as_ref().unwrap_or(manifest)
                };
                let manifest = self
                    .row_selections
                    .get(manifest)
                    .and_then(|variants| variants.iter().find(|v| v.accepts(rows)))
                    .map_or(manifest.as_str(), |v| v.kernels[0].as_str());
                let mut operands = buffers
                    .iter()
                    .map(|n| self.row_buffer(n, active_rows))
                    .collect::<Result<Vec<_>>>()?;
                let path = resolve_file(base, manifest)?;
                let template = match self.templates.get(&path) {
                    Some(template) => Rc::clone(template),
                    None => {
                        let t = Instant::now();
                        let template = Template::load(&self.session, &path)?;
                        self.phases.templates.0 += 1;
                        self.phases.templates.1 += t.elapsed().as_secs_f64();
                        self.templates.insert(path, Rc::clone(&template));
                        template
                    }
                };
                let t = Instant::now();
                for (buffer, operand) in operands.iter_mut().zip(&template.spec().operands) {
                    if operand.access == super::kernels::Access::Scratch {
                        *buffer = buffer.slice(0, operand.bytes()?)?;
                    }
                }
                let mut kernel = if template.spec().row_capacity()?.is_some() {
                    Kernel::bind_rows(&self.session, &template, &operands, rows)?
                } else {
                    Kernel::bind(&self.session, &template, &operands)?
                };
                if owner_work.is_some() {
                    let runtime = self
                        .batch
                        .as_ref()
                        .ok_or("Owner kernel needs a batch contract")?;
                    let accesses: Vec<batch::Access> =
                        serde_json::from_value(template.spec().parameters["_owner_access"].clone())
                            .map_err(|e| e.to_string())?;
                    let hazards = accesses
                        .iter()
                        .map(|a| {
                            Ok((
                                get(
                                    &self.buffers,
                                    &runtime
                                        .contract
                                        .columns
                                        .get(
                                            *runtime
                                                .contract
                                                .pointers
                                                .get(&buffers[1])
                                                .ok_or("Owner pointer mapping missing")?
                                                .get(a.column)
                                                .ok_or("Local owner column out of range")?,
                                        )
                                        .ok_or("Owner column out of range")?
                                        .buffer,
                                )?,
                                a.write,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    kernel = kernel.with_hazards(&self.session, hazards)?;
                }
                pending.push(SegmentOp::Kernel(kernel));
                self.phases.binds.0 += 1;
                self.phases.binds.1 += t.elapsed().as_secs_f64();
                continue;
            }
            if let Instruction::Reduce {
                source,
                destination,
                elements,
                fp32,
                scatter,
                fixed_rows,
                elements_expression,
            } = instruction
            {
                let source = self.row_buffer(source, active_rows)?;
                let destination = self.row_buffer(destination, active_rows)?;
                let element = if *fp32 {
                    Element::Float32
                } else {
                    Element::Bfloat16
                };
                let count = usize::try_from(*elements).map_err(|_| "Reduce size overflow")?;
                let elements = row_contract::elements(
                    count,
                    active_rows,
                    capacity,
                    *fixed_rows,
                    elements_expression.as_ref(),
                )? as u64;
                if *scatter {
                    self.comm.validate_scatter(
                        &self.session,
                        &source,
                        &destination,
                        elements,
                        element,
                    )?;
                } else {
                    self.check_reduce(&source, &destination, elements, element)?;
                }
                pending.push(SegmentOp::Reduce {
                    source,
                    destination,
                    elements,
                    element,
                    scatter: *scatter,
                });
                continue;
            }
            if let Instruction::Gather {
                source,
                destination,
                elements,
                fp32,
                int32,
                fixed_rows,
                trace_only,
                elements_expression,
            } = instruction
            {
                if *trace_only && !self.snapshots {
                    continue;
                }
                let source = self.row_buffer(source, active_rows)?;
                let destination = self.row_buffer(destination, active_rows)?;
                let element = gather_element(*fp32, *int32)?;
                self.session.require(&source)?;
                self.session.require(&destination)?;
                let count = usize::try_from(*elements).map_err(|_| "Gather size overflow")?;
                let elements = row_contract::elements(
                    count,
                    active_rows,
                    capacity,
                    *fixed_rows,
                    elements_expression.as_ref(),
                )? as u64;
                let bytes = elements
                    .checked_mul(element.bytes())
                    .ok_or("Model gather extent overflow")?;
                let output = bytes.checked_mul(8).ok_or("Model gather output overflow")?;
                if source.overlaps(&destination)
                    || elements == 0
                    || bytes > source.bytes() as u64
                    || output > destination.bytes() as u64
                {
                    return Err("Model AllGather spans differ from TP8 geometry".into());
                }
                pending.push(SegmentOp::Gather {
                    source,
                    destination,
                    elements,
                    element,
                });
                continue;
            }
            if let Instruction::Exchange {
                source,
                destination,
                elements,
                fp32,
                int32,
                fixed_rows,
                elements_expression,
            } = instruction
            {
                let source = self.row_buffer(source, active_rows)?;
                let destination = self.row_buffer(destination, active_rows)?;
                let elements = row_contract::elements(
                    usize::try_from(*elements).map_err(|_| "Exchange size overflow")?,
                    active_rows,
                    capacity,
                    *fixed_rows,
                    elements_expression.as_ref(),
                )? as u64;
                if elements == 0 {
                    return Err("Model exchange requires nonzero peer counts".into());
                }
                let peers = vec![elements; self.comm.ranks() as usize];
                pending.push(SegmentOp::Exchange(self.comm.prepare_exchange(
                    &self.session,
                    &source,
                    &destination,
                    Counts::new(peers.clone())?,
                    Counts::new(peers)?,
                    gather_element(*fp32, *int32)?,
                )?));
                continue;
            }
            if let Instruction::W4 {
                name,
                rows,
                input,
                output,
                experts,
                buffers,
                workspace,
                groups,
            } = instruction
            {
                let mut b = buffers
                    .iter()
                    .map(|n| self.row_buffer(n, active_rows))
                    .collect::<Result<Vec<_>>>()?;
                let rows = scaled(*rows)?;
                let input_bytes = rows.checked_mul(*input).ok_or("W4 input extent overflow")?;
                let output_bytes = rows
                    .checked_mul(*output)
                    .and_then(|n| n.checked_mul(2))
                    .ok_or("W4 output extent overflow")?;
                for (slot, bytes) in [(0, input_bytes), (4, rows * 4), (5, output_bytes)] {
                    b[slot] = b[slot].slice(0, bytes)?;
                }
                let mut ends = vec![0; *experts];
                *ends.last_mut().ok_or("Empty W4 expert bank")? = rows as i64;
                let counts = match groups {
                    Some(name) => Some(self.row_buffer(name, active_rows)?),
                    None => None,
                };
                let capture_w4 = std::env::var_os("ORINFER_A2_W4_NO_CAPTURE").is_none();
                let shared_workspace = self.row_buffer(workspace, active_rows)?;
                let key = W4Key {
                    name: name.clone(),
                    rows,
                    input: *input,
                    output: *output,
                    experts: *experts,
                    pointers: std::array::from_fn(|i| b[i].pointer() as usize),
                    groups: counts.as_ref().map(|b| b.pointer() as usize),
                    workspace: shared_workspace.pointer() as usize,
                };
                let plan = if let Some(plan) = self.w4.get(&key).and_then(Weak::upgrade) {
                    plan
                } else {
                    let t = Instant::now();
                    let plan = Rc::new(GroupedW4A8::prepare_with_workspace(
                        &self.session,
                        GroupedShape {
                            rows,
                            input: *input,
                            output: *output,
                            experts: *experts,
                        },
                        &ends,
                        GroupedInputs {
                            activation: &b[0],
                            weight: &b[1],
                            weight_scale: &b[2],
                            assist: &b[3],
                            token_scale: &b[4],
                            output: &b[5],
                        },
                        Some(&shared_workspace),
                        counts.as_ref(),
                    )?);
                    self.phases.aclnn.0 += 1;
                    self.phases.aclnn.1 += t.elapsed().as_secs_f64();
                    self.w4.insert(key, Rc::downgrade(&plan));
                    plan
                };
                if routed.contains(name) {
                    let binding = if capacity == actual {
                        name.clone()
                    } else {
                        format!("{name}/rows/{actual}")
                    };
                    self.routed_w4
                        .insert(binding, (actual, Rc::downgrade(&plan)));
                }
                // W4 graph nodes use the same stream-ordered scratch as direct
                // enqueues. The surrounding kernels restore activation bytes:
                // ACLNN's A8/W4 implementation transforms its input in place.
                if capture_w4 {
                    pending.push(SegmentOp::W4(plan));
                } else {
                    self.capture_pending(&mut pending, &mut steps, capture)?;
                    fork_count = 0;
                    steps.push(Step::W4(plan));
                }
                continue;
            }
            if !pending.is_empty() {
                self.capture_pending(&mut pending, &mut steps, capture)?;
                fork_count = 0;
            }
            steps.push(match instruction {
                Instruction::Route {
                    indices,
                    gather,
                    restore,
                    groups,
                    rows,
                } => Step::Route {
                    indices: get(&self.buffers, indices)?,
                    gather: get(&self.buffers, gather)?,
                    restore: get(&self.buffers, restore)?,
                    groups: groups
                        .iter()
                        .map(|name| {
                            if capacity == actual {
                                name.clone()
                            } else {
                                format!("{name}/rows/{actual}")
                            }
                        })
                        .collect(),
                    rows: scaled(*rows)?,
                },
                Instruction::Embed {
                    destination,
                    first_column,
                    columns,
                } => Step::Embed {
                    destination: get(&self.buffers, destination)?,
                    first_column: *first_column,
                    columns: *columns,
                },
                Instruction::Lookup {
                    layer,
                    values,
                    scales,
                    mask,
                } => Step::Lookup {
                    layer: *layer,
                    values: get(&self.buffers, values)?,
                    scales: get(&self.buffers, scales)?,
                    mask: get(&self.buffers, mask)?,
                },
                Instruction::Snapshot { .. }
                | Instruction::LimitRows { .. }
                | Instruction::DecoderRows { .. }
                | Instruction::Fork
                | Instruction::Switch
                | Instruction::Join
                | Instruction::Gather { .. }
                | Instruction::Exchange { .. }
                | Instruction::Kernel { .. }
                | Instruction::Reduce { .. }
                | Instruction::W4 { .. } => unreachable!(),
            });
        }
        if !pending.is_empty() {
            self.capture_pending(&mut pending, &mut steps, capture)?;
        }
        Ok(steps)
    }

    fn check_reduce(
        &self,
        source: &Buffer,
        destination: &Buffer,
        elements: u64,
        element: Element,
    ) -> Result<()> {
        self.session.require(source)?;
        self.session.require(destination)?;
        if source.overlaps(destination) && source.pointer() != destination.pointer() {
            return Err("HCCL AllReduce requires identical or disjoint spans".into());
        }
        let bytes = elements
            .checked_mul(element.bytes())
            .ok_or("HCCL reduction byte size overflow")?;
        if elements == 0 || bytes > source.bytes() as u64 || bytes > destination.bytes() as u64 {
            return Err("HCCL reduction outside buffer".into());
        }
        Ok(())
    }

    /// Captures one contiguous mixed run of kernels, collectives and grouped
    /// matmuls. Every segment executes once before capture: CANN's first-use
    /// lazy state must exist before the capture region records the launches,
    /// and skipping repeats of already-executed templates produced invalid
    /// replay output.
    fn capture_pending(
        &mut self,
        pending: &mut Vec<SegmentOp>,
        steps: &mut Vec<Step>,
        capture: bool,
    ) -> Result<()> {
        let depths = branches::validate(pending)?;
        let ops = Rc::new(pending.clone());
        let comm = self.comm.clone();
        let parallel = if depths.is_empty() {
            None
        } else {
            Some(super::graph::Branches::new(
                &self.session,
                &depths,
                Box::new((Rc::clone(&ops), comm.clone())),
            )?)
        };
        let segment = Rc::new(Segment {
            ops,
            comm,
            parallel,
        });
        steps.push(if capture {
            Step::Graph(Rc::new(self.capture_segment(segment)?))
        } else {
            Step::Segment(segment)
        });
        pending.clear();
        Ok(())
    }
    fn capture_segment(&mut self, segment: Rc<Segment>) -> Result<super::graph::Graph> {
        let t = Instant::now();
        // Warm every bound segment immediately before capture, even when its
        // direct form has executed before: CANN lazy capture state requires it.
        segment.enqueue(&self.session)?.wait()?;
        self.phases.warmups.0 += 1;
        self.phases.warmups.1 += t.elapsed().as_secs_f64();
        let t = Instant::now();
        // SAFETY: The segment retains all stable native operands, executors,
        // collective resources and branch events through graph destruction.
        let result = unsafe {
            super::graph::Graph::capture(&self.session, Box::new(Rc::clone(&segment)), |stream| {
                enqueue_segment(
                    &segment.ops,
                    &segment.comm,
                    segment.parallel.as_deref(),
                    stream,
                )
            })
        };
        // Capture rolls back before branch fencing; synchronizing a stream
        // while it is still being captured is not a valid cleanup operation.
        let graph = segment.fence_result(result)?;
        self.phases.captures.0 += 1;
        self.phases.captures.1 += t.elapsed().as_secs_f64();
        Ok(graph)
    }
    fn capture_steps(&mut self, steps: &[Step]) -> Result<Vec<Step>> {
        steps
            .iter()
            .map(|step| match step {
                Step::Segment(segment) => self
                    .capture_segment(Rc::clone(segment))
                    .map(|graph| Step::Graph(Rc::new(graph))),
                _ => Ok(step.clone()),
            })
            .collect()
    }
    fn reset(&self) -> Result<()> {
        let names: std::collections::BTreeSet<_> = self
            .packet_templates
            .iter()
            .flat_map(|packet| &packet.reset)
            .collect();
        for name in names {
            self.session.fill(&get(&self.buffers, name)?, 0)?.wait()?;
        }
        Ok(())
    }
    fn touch_packet(&mut self, rows: usize) {
        if let Some(evicted) = self.packet_order.touch(rows) {
            self.packets.remove(&evicted);
            self.w4.retain(|_, plan| plan.strong_count() > 0);
            let packets = &self.packets;
            self.routed_w4
                .retain(|_, (rows, plan)| packets.contains_key(rows) && plan.strong_count() > 0);
        }
    }
}

fn load_spec(path: &Path, model_dir: &Path) -> Result<(Model, PathBuf, HashLayout)> {
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    let base = path.parent().ok_or("Model plan has no parent")?.to_owned();
    let mut model: Model = crate::model::read(&path)?;
    if model.schema_version != 1
        || model.layers != 40
        || model.ranks.len() != 8
        || !(128..=262144).contains(&model.capacity)
        || !model.capacity.is_multiple_of(32)
        || !(4..=4096).contains(&model.packet_cache_capacity)
    {
        return Err("Expected complete 40-layer TP8 A2 plan with capacity <=262144".into());
    }
    read_identity(model_dir, &model.config)?;
    read_identity(model_dir, &model.index)?;
    let layout: HashLayout =
        serde_json::from_slice(&read_identity(&base, &model.engram)?).map_err(|e| e.to_string())?;
    if layout.config_sha256 != model.config.sha256 {
        return Err("Engram/config identity differs".into());
    }
    if crate::artifact::sha256(
        &std::fs::read(model_dir.join("tokenizer.json")).map_err(|e| e.to_string())?,
    ) != layout.tokenizer_sha256
    {
        return Err("Engram/tokenizer identity differs".into());
    }
    preflight(&model, &base)?;
    model.graph_launches = batch::graph_launches(&model);
    Ok((model, base, layout))
}

/// Reject incomplete or mismatched packages before allocating resident weights.
fn preflight(model: &Model, base: &Path) -> Result<()> {
    batch::validate(model)?;
    let mut kernels = BTreeMap::<String, super::kernels::Spec>::new();
    let used: std::collections::BTreeSet<_> = model
        .ranks
        .iter()
        .flat_map(|r| &r.packets)
        .flat_map(|p| p.body.iter().chain(&p.head))
        .filter_map(|i| match i {
            Instruction::Kernel {
                manifest,
                uncompacted_manifest,
                ..
            } => Some(
                [Some(manifest.clone()), uncompacted_manifest.clone()]
                    .into_iter()
                    .flatten(),
            ),
            _ => None,
        })
        .flatten()
        .collect();
    let mut names = used.clone();
    for (slot, variants) in &model.row_selections {
        if !used.contains(slot) || variants.is_empty() {
            return Err("A2 row selection requires a used canonical slot".into());
        }
        let mut previous = 0;
        for variant in variants {
            if variant.min_rows <= previous
                || variant.min_rows > variant.max_rows
                || variant.max_rows > 4096
                || variant.multiple_of != 1
                || variant.kernels.len() != 1
            {
                return Err("Invalid or overlapping A2 row selections".into());
            }
            previous = variant.max_rows;
            names.extend(variant.kernels.iter().cloned());
        }
    }
    for name in names {
        let path = resolve_file(base, &name)?;
        let spec: super::kernels::Spec = crate::model::read(&path)?;
        spec.validate()?;
        spec.target.require_host(std::env::consts::ARCH)?;
        read_identity(
            path.parent().ok_or("Kernel path has no parent")?,
            &spec.library,
        )?;
        kernels.insert(name, spec);
    }
    for (slot, variants) in &model.row_selections {
        let source = &kernels[slot];
        let capacity = source
            .row_capacity()?
            .ok_or("A2 row selections require dynamic kernels")?;
        for variant in variants {
            let replacement = &kernels[&variant.kernels[0]];
            row_contract::validate_replacement(source, replacement)?;
            if variant.max_rows > capacity {
                return Err("A2 row selection exceeds its operator capacity".into());
            }
        }
    }
    for (rank_index, rank) in model.ranks.iter().enumerate() {
        if rank.rank as usize != rank_index {
            return Err("Rank order differs from TP topology".into());
        }
        let mut buffers = BTreeMap::new();
        for definition in &rank.buffers {
            if definition.bytes == 0 || buffers.contains_key(&definition.name) {
                return Err("Invalid/duplicate model buffer".into());
            }
            if let Some(view) = &definition.view {
                row_contract::view_offset(view, view.row_capacity.unwrap_or(1))?;
                let bytes = *buffers
                    .get(&view.parent)
                    .ok_or("Model view precedes its allocation")?;
                if view
                    .offset
                    .checked_add(definition.bytes)
                    .is_none_or(|end| end > bytes)
                    || definition.weight.is_some()
                    || definition.data.is_some()
                    || definition.prepared_weight.is_some()
                {
                    return Err("Invalid model tensor view".into());
                }
            }
            buffers.insert(definition.name.clone(), definition.bytes);
            if let Some(file) = &definition.prepared_weight {
                let path = resolve_file(base, file)?;
                let metadata = std::fs::metadata(path).map_err(|e| e.to_string())?;
                if !metadata.is_file()
                    || metadata.len() != definition.bytes as u64
                    || definition.weight.is_some()
                    || definition.data.is_some()
                {
                    return Err("Prepared weight metadata differs".into());
                }
            }
        }
        if model.batch.is_none() {
            validate_dispatch(&rank.packets)?;
        }
        if rank
            .packets
            .iter()
            .map(|p| (p.rows, p.min_rows, p.max_rows))
            .collect::<Vec<_>>()
            != model.ranks[0]
                .packets
                .iter()
                .map(|p| (p.rows, p.min_rows, p.max_rows))
                .collect::<Vec<_>>()
        {
            return Err("TP ranks have different row contracts".into());
        }
        for packet in &rank.packets {
            replay::validate_packet(packet, model.batch.as_ref())?;
            for coordinates in
                std::iter::once(&packet.coordinates).chain(packet.decoder_coordinates.as_ref())
            {
                if coordinates
                    .names()
                    .iter()
                    .any(|name| !buffers.contains_key(*name))
                {
                    return Err("Packet coordinate buffer is not defined".into());
                }
            }
            branches::validate_instructions(&packet.body)?;
            branches::validate_instructions(&packet.head)?;
            if packet.rows == 0
                || packet.rows > 4096
                || packet.rows > model.capacity
                || packet.dispatch_max() > packet.rows
            {
                return Err("Invalid packet row capacity".into());
            }
            let layers = packet
                .body
                .iter()
                .filter_map(|step| {
                    if let Instruction::Snapshot { name, .. } = step
                        && !model
                            .batch
                            .as_ref()
                            .and_then(|c| c.decoder_replay.as_ref())
                            .and_then(|c| c.source_ready.as_ref())
                            .is_some_and(|s| s == name)
                    {
                        Some(name.clone())
                    } else {
                        None
                    }
                })
                .collect::<std::collections::BTreeSet<_>>();
            if layers != (0..40).map(|i| format!("layer{i:02}")).collect() {
                return Err("A2 execution packet does not cover all 40 layers".into());
            }
            let mut current_capacity = packet.rows;
            for instruction in packet.body.iter().chain(&packet.head) {
                if let Instruction::Embed {
                    destination,
                    first_column,
                    columns,
                } = instruction
                {
                    let span = packet::embedding_span(*first_column, *columns)?;
                    if buffers.get(destination).copied() != Some(current_capacity * span.len()) {
                        return Err("Embedding input geometry differs".into());
                    }
                }
                if let Instruction::LimitRows { rows } = instruction {
                    if packet.min_rows == 0 {
                        return Err("Decode packets cannot truncate their rows".into());
                    }
                    bounded_rows(current_capacity, current_capacity, *rows)?;
                    current_capacity = *rows;
                }
                if let Instruction::Kernel {
                    manifest,
                    uncompacted_manifest,
                    buffers: bindings,
                    fixed_rows,
                    row_expression,
                    owner_work,
                    owners,
                } = instruction
                {
                    let spec = &kernels[manifest];
                    if let Some(fallback) = uncompacted_manifest {
                        replay::validate_fallback(spec, &kernels[fallback])?;
                    }
                    let expected = if owner_work.is_some() {
                        4096
                    } else if *owners {
                        model
                            .batch
                            .as_ref()
                            .ok_or("Owner head requires batch contract")?
                            .owners
                    } else {
                        row_contract::kernel_rows(
                            current_capacity,
                            current_capacity,
                            *fixed_rows,
                            row_expression.as_ref(),
                        )?
                    };
                    if owner_work.is_some() != spec.op.starts_with("private_rows.") {
                        return Err("Owner dispatch and AOT identity differ".into());
                    }
                    if let Some(grid) = owner_work {
                        let c = model.batch.as_ref().ok_or("Missing owner contract")?;
                        let owner_capacity = bindings
                            .first()
                            .map(|table| c.token_capacity(table, packet.rows))
                            .transpose()?
                            .unwrap_or(packet.rows);
                        if *owners
                            || fixed_rows.is_some()
                            || row_expression.is_some()
                            || bindings.len() < 3
                            || c.work.get(&bindings[0]) != Some(grid)
                            || !c.pointers.contains_key(&bindings[1])
                            || bindings[2] != c.active
                            || spec.parameters["_owner_token_capacity"].as_u64()
                                != Some(owner_capacity as u64)
                        {
                            return Err("Owner kernel/table/row contract differs".into());
                        }
                        let accesses: Vec<batch::Access> =
                            serde_json::from_value(spec.parameters["_owner_access"].clone())
                                .map_err(|e| e.to_string())?;
                        let mapping = &c.pointers[&bindings[1]];
                        if accesses.is_empty()
                            || !accesses.iter().any(|a| a.write)
                            || accesses.iter().any(|a| {
                                a.column >= mapping.len()
                                    || c.columns[mapping[a.column]]
                                        .decode
                                        .is_some_and(|d| d != packet.batch_decode)
                            })
                        {
                            return Err("Invalid owner pointee access contract".into());
                        }
                    }

                    if spec
                        .row_capacity()?
                        .is_some_and(|cap| packet.min_rows == 0 || cap != expected)
                        || ((fixed_rows.is_some() || row_expression.is_some())
                            && spec.row_capacity()?.is_none())
                    {
                        return Err("Packet/AOT dynamic row capacities differ".into());
                    }
                    if bindings.len() != spec.operands.len() {
                        return Err("Packet/AOT operand count differs".into());
                    }
                    for (binding, operand) in bindings.iter().zip(&spec.operands) {
                        let bytes = buffers.get(binding).copied();
                        let required = operand.bytes()?;
                        let compatible = if operand.access == super::kernels::Access::Scratch {
                            bytes.is_some_and(|size| size >= required)
                        } else {
                            bytes == Some(required)
                        };
                        if !compatible {
                            return Err(format!("{binding}: packet/AOT byte extent differs"));
                        }
                    }
                    for variant in model.row_selections.get(manifest).into_iter().flatten() {
                        for (binding, operand) in
                            bindings.iter().zip(&kernels[&variant.kernels[0]].operands)
                        {
                            if operand.access == super::kernels::Access::Scratch
                                && buffers[binding] < operand.bytes()?
                            {
                                return Err(format!(
                                    "{binding}: selected scratch exceeds allocation"
                                ));
                            }
                        }
                    }
                }
                if let Instruction::Gather { fp32, int32, .. }
                | Instruction::Exchange { fp32, int32, .. } = instruction
                {
                    gather_element(*fp32, *int32)?;
                }
                if let Instruction::Gather {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                }
                | Instruction::Reduce {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                }
                | Instruction::Exchange {
                    elements,
                    fixed_rows,
                    elements_expression,
                    ..
                } = instruction
                {
                    row_contract::elements(
                        usize::try_from(*elements).map_err(|_| "Collective count overflow")?,
                        current_capacity,
                        current_capacity,
                        *fixed_rows,
                        elements_expression.as_ref(),
                    )?;
                }
                if let Instruction::Exchange {
                    source,
                    destination,
                    elements,
                    fp32,
                    int32,
                    ..
                } = instruction
                {
                    let bytes = elements
                        .checked_mul(gather_element(*fp32, *int32)?.bytes())
                        .and_then(|v| v.checked_mul(model.ranks.len() as u64))
                        .ok_or("Model exchange extent overflow")?;
                    if *elements == 0
                        || buffers
                            .get(source.as_str())
                            .is_none_or(|&v| bytes > v as u64)
                        || buffers
                            .get(destination.as_str())
                            .is_none_or(|&v| bytes > v as u64)
                    {
                        return Err("Model exchange exceeds its buffers".into());
                    }
                }
                if let Instruction::W4 {
                    rows,
                    input,
                    output,
                    experts,
                    buffers: bindings,
                    workspace,
                    groups,
                    ..
                } = instruction
                {
                    if *experts != 48
                        || *rows != current_capacity * 6
                        || !input.is_multiple_of(256)
                        || !output.is_multiple_of(32)
                    {
                        return Err("W4 model geometry differs from the routed packet".into());
                    }
                    let extents = [
                        rows * input,
                        experts * input * output / 2,
                        experts * output * 8,
                        experts * output * 4,
                        rows * 4,
                        rows * output * 2,
                    ];
                    if bindings
                        .iter()
                        .zip(extents)
                        .any(|(name, bytes)| buffers.get(name).copied() != Some(bytes))
                        || buffers
                            .get(workspace)
                            .is_none_or(|&bytes| bytes < 100663808)
                        || groups
                            .as_ref()
                            .is_some_and(|name| buffers.get(name).copied() != Some(experts * 8))
                    {
                        return Err("W4 packet operand/workspace extent differs".into());
                    }
                }
            }
        }
    }
    Ok(())
}

fn gather_element(fp32: bool, int32: bool) -> Result<Element> {
    match (fp32, int32) {
        (true, false) => Ok(Element::Float32),
        (false, true) => Ok(Element::Int32),
        (false, false) => Ok(Element::Bfloat16),
        (true, true) => Err("AllGather has conflicting element types".into()),
    }
}

fn bounded_rows(actual: usize, capacity: usize, limit: usize) -> Result<usize> {
    if actual == 0 || actual > capacity || limit == 0 || limit > capacity {
        return Err("Compacted row extent exceeds its packet capacity".into());
    }
    Ok(actual.min(limit))
}

fn validate_dispatch(packets: &[Packet]) -> Result<()> {
    let Some(decode) = packets.first() else {
        return Err("Missing A2 decode packet".into());
    };
    if decode.rows != 1 || decode.min_rows != 0 || decode.dispatch_max() != 1 || packets.len() < 2 {
        return Err("Expected single-row decode followed by prefill ranges".into());
    }
    validate_row_ranges(&packets[1..].iter().collect::<Vec<_>>(), 2, None)
}

fn validate_row_ranges(packets: &[&Packet], first: usize, capacity: Option<usize>) -> Result<()> {
    let mut next = first;
    for packet in packets {
        if packet.min_rows != next
            || packet.dispatch_max() < next
            || packet.dispatch_max() > packet.rows
            || capacity.is_some_and(|c| packet.rows != c)
        {
            return Err("A2 prefill ranges must be ordered, contiguous and disjoint".into());
        }
        next = packet.dispatch_max() + 1;
    }
    if packets.is_empty() || capacity.is_some_and(|c| next != c + 1) {
        return Err("A2 row ranges do not cover their declared capacity".into());
    }
    Ok(())
}
