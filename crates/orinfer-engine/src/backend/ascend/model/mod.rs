//! Shared native A2 contracts and execution for serving and diagnostic requests.
mod affinity;
mod batch;
mod batch_benchmark;
mod benchmark;
mod branches;
#[cfg(test)]
mod dispatch_tests;
mod load;
mod memory;
mod packet;
mod plan;
pub(crate) mod provider;
mod replay;
mod row_contract;
mod state;
mod team;
mod weights;

use super::{
    hccl::{Communicator, Counts, Element, Exchange, RootInfo},
    kernels::{Kernel, Template},
    operators::{GroupedConfig, GroupedInputs, GroupedShape, GroupedW4A8, W4Layout},
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
    cell::Cell,
    collections::{BTreeMap, VecDeque},
    path::{Path, PathBuf},
    rc::{Rc, Weak},
    sync::{Arc, OnceLock},
    time::Instant,
};

use load::upload;
#[cfg(test)]
use plan::validate_dispatch;
use plan::{bounded_rows, gather_element, load_spec, validate_row_ranges, w4_plan_rows};

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
    prepared_tensor: Option<PreparedTensor>,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PreparedTensor {
    name: String,
    dtype: String,
    shape: Vec<usize>,
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
        #[serde(default)]
        expected_expert_rows: Option<usize>,
        #[serde(default)]
        weight_layout: W4Layout,
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
    direct_uses: Cell<u8>,
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
    expected_expert_rows: Option<usize>,
    weight_layout: W4Layout,
}
type SharedTables = OnceLock<Result<Arc<BTreeMap<usize, SharedInt8Rows>>>>;

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
    template_aliases: BTreeMap<PathBuf, BTreeMap<String, Rc<Template>>>,
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
    branches: (usize, f64),
    dependencies: (usize, f64),
    table_build_s: f64,
    table_upload_s: f64,
    template_paths_s: f64,
    kernel_geometry_s: f64,
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
                let geometry_started = Instant::now();
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
                    .map_or(manifest.as_str(), |v| v.kernels[0].as_str())
                    .to_owned();
                let mut operands = buffers
                    .iter()
                    .map(|n| self.row_buffer(n, active_rows))
                    .collect::<Result<Vec<_>>>()?;
                self.phases.kernel_geometry_s += geometry_started.elapsed().as_secs_f64();
                let template = self.template(base, &manifest)?;
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
                expected_expert_rows,
                weight_layout,
            } = instruction
            {
                let mut b = buffers
                    .iter()
                    .map(|n| self.row_buffer(n, active_rows))
                    .collect::<Result<Vec<_>>>()?;
                // Device group counts bound actual work. A tuned instruction
                // keeps its capacity descriptor across row counts; pointers and
                // the tuning identity still participate in executor reuse.
                let rows = w4_plan_rows(
                    *rows,
                    scaled(*rows)?,
                    *expected_expert_rows,
                    groups.is_some(),
                    routed.contains(name),
                )?;
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
                    expected_expert_rows: *expected_expert_rows,
                    weight_layout: *weight_layout,
                };
                let plan = if let Some(plan) = self.w4.get(&key).and_then(Weak::upgrade) {
                    plan
                } else {
                    let t = Instant::now();
                    let plan = Rc::new(GroupedW4A8::prepare_with_config(
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
                        GroupedConfig {
                            expected_rows: *expected_expert_rows,
                            weight_layout: *weight_layout,
                        },
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

    /// Package assets are immutable for the worker's lifetime. Resolve each
    /// package-local name once, while retaining canonical-path deduplication
    /// and the first load's identity/ABI checks.
    fn template(&mut self, base: &Path, manifest: &str) -> Result<Rc<Template>> {
        if let Some(template) = self
            .template_aliases
            .get(base)
            .and_then(|a| a.get(manifest))
        {
            return Ok(Rc::clone(template));
        }
        let started = Instant::now();
        let path = resolve_file(base, manifest)?;
        self.phases.template_paths_s += started.elapsed().as_secs_f64();
        let template = match self.templates.get(&path) {
            Some(template) => Rc::clone(template),
            None => {
                let started = Instant::now();
                let template = Template::load(&self.session, &path)?;
                self.phases.templates.0 += 1;
                self.phases.templates.1 += started.elapsed().as_secs_f64();
                self.templates.insert(path, Rc::clone(&template));
                template
            }
        };
        self.template_aliases
            .entry(base.to_path_buf())
            .or_default()
            .insert(manifest.to_owned(), Rc::clone(&template));
        Ok(template)
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
        let validation_started = Instant::now();
        let depths = branches::validate(pending)?;
        self.phases.dependencies.0 += 1;
        self.phases.dependencies.1 += validation_started.elapsed().as_secs_f64();
        let ops = Rc::new(pending.clone());
        let comm = self.comm.clone();
        let branch_started = Instant::now();
        let parallel = if depths.is_empty() {
            None
        } else {
            Some(super::graph::Branches::new(
                &self.session,
                &depths,
                Box::new((Rc::clone(&ops), comm.clone())),
            )?)
        };
        self.phases.branches.0 += usize::from(parallel.is_some());
        self.phases.branches.1 += branch_started.elapsed().as_secs_f64();
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
