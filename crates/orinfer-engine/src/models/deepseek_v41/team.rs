//! Thread-affine TP workers shared by benchmarks and continuous generation.
use super::{
    Model, SharedTables, Worker, affinity,
    batch::{CacheState, Layout, PartAction, SplitPlan, SplitStates},
    load_spec,
};
use crate::backend::ascend::{
    hccl::RootInfo,
    runtime::{Limits, Session},
};
use crate::{artifact::Result, error::EngineError};
mod channel;
mod host_release;
mod preparation;
use channel::{Peer, Transport, reply_channel};
pub(super) use preparation::{CaptureBudget, capture_credit, capture_interval};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone)]
pub(super) enum Command {
    Admit(Vec<usize>),
    StartOwner {
        slot: usize,
        limit: usize,
        snapshot: Option<u64>,
    },
    BeginImages(Arc<Vec<crate::vision::ImageInput>>),
    AdvanceImages(usize, super::vision::CapturePolicy),
    CancelImages,
    InstallImages(usize, Arc<Vec<super::vision::Span>>),
    GrowCached(Layout),
    GrowCachedAndBeginVerification(Layout, Vec<(usize, usize)>),
    OwnerRequirement(usize, usize, usize),
    Reset(Vec<usize>),
    Release(Option<usize>),
    #[cfg(test)]
    Cached(Layout),
    #[cfg(test)]
    Invalidate(Layout),
    InvalidateParts(Layout, bool, bool, bool),
    #[cfg(test)]
    Prepare(Layout, bool),
    PrepareParts(Layout, SplitPlan),
    Shared(Layout),
    CaptureShared(Layout, String),
    Forward(Layout, Vec<Vec<u32>>, Readout, Vec<String>),
    FastForward(Layout, Vec<Vec<u32>>, Readout),
    FastVerifyForward(Layout, Vec<(usize, usize)>, Vec<Vec<u32>>, Readout),
    BeginVerification(Vec<(usize, usize)>),
    FinishVerification(Vec<(usize, usize)>),
    Save {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    #[cfg(test)]
    Restore {
        owner: usize,
        snapshot: u64,
    },
    Evict(u64),
    EvictMany(Vec<u64>),
    RollbackSave {
        owner: usize,
        snapshot: u64,
    },
    Offload(u64),
    PlanOffload(u64),
    SaveHost {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    PlanHost {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    Budget(super::memory::Requirement),
    Reclaim,
    ReclaimForAdmission,
    PlanSnapshot {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    PlanAndSave {
        owner: usize,
        snapshot: u64,
        complete: bool,
        max_incoming_bytes: usize,
    },
    RoutingGroups(bool),
    Stop,
}
pub(super) enum Reply {
    Ready,
    Image(Option<super::vision::Encoded>),
    #[cfg(test)]
    Cached(CacheState),
    SplitCached(SplitStates, bool),
    SplitCachedAndVerification(SplitStates, bool, bool),
    Outputs(Vec<Output>, bool),
    Shared(Vec<super::shared::State>),
    Allocations(BTreeMap<u64, usize>),
    SaveSkipped(String),
    SnapshotPlan {
        allocations: BTreeMap<u64, usize>,
        device_free: usize,
        session_free: usize,
    },
    RoutingGroups(Vec<RoutingGroup>),
    Capacity(bool),
    BudgetStatus {
        fits: bool,
        device_free: usize,
    },
    Requirement(super::memory::Requirement),
}
#[derive(serde::Serialize)]
pub(super) struct RoutingGroup {
    pub buffer: String,
    pub counts: Vec<i64>,
}
#[derive(Clone, Default)]
pub(super) struct Output {
    pub owner: usize,
    pub token: u32,
    pub value: f32,
    pub logits: Vec<f32>,
    pub summary: Option<crate::backend::ascend::sampling::Summary>,
    pub report: Option<crate::sampling::TokenLogprob>,
    pub error: Option<String>,
}
#[derive(Clone, Default)]
pub(super) struct Readout {
    pub full: bool,
    /// Only generated continuations may consume a retained device input.
    /// Explicit/teacher-forced callers keep authoritative host tokens.
    pub continuations: BTreeSet<usize>,
    pub draws: BTreeMap<usize, crate::backend::ascend::sampling::Draw>,
    pub store_draft_distributions: bool,
    pub rejection: BTreeMap<usize, Vec<crate::backend::ascend::sampling::RejectionProposal>>,
    #[cfg(test)]
    pub capture_sampled_logits: bool,
    /// None retains the benchmark/legacy full-packet diagnostic contract.
    pub full_owners: Option<BTreeSet<usize>>,
    pub parameters: Arc<
        crate::sampling::Prepared<
            BTreeMap<usize, Vec<crate::backend::ascend::sampling::Parameters>>,
        >,
    >,
    pub first_input: Option<Arc<super::host_inputs::First>>,
    pub next_inputs: Vec<Arc<super::host_inputs::Window>>,
}
impl From<bool> for Readout {
    fn from(full: bool) -> Self {
        Self {
            full,
            ..Self::default()
        }
    }
}
type RejectionRows = (
    Vec<[i32; 3]>,
    Vec<Option<crate::backend::ascend::sampling::RejectionProposal>>,
);
impl Readout {
    fn validate_draw_roles(&self) -> Result<()> {
        #[cfg(test)]
        if self.capture_sampled_logits {
            return Ok(());
        }
        if self.draws.keys().any(|owner| self.wants_logits(*owner)) {
            return Err("An owner cannot request sampled and raw outputs together".into());
        }
        Ok(())
    }
    pub fn sampling_steps(&mut self, steps: &BTreeMap<usize, usize>) {
        for (owner, draw) in &mut self.draws {
            if let Some(&step) = steps.get(owner) {
                draw.step = step;
            }
        }
    }
    pub fn rejection_rows(&self, layout: &Layout) -> Result<RejectionRows> {
        self.validate_draw_roles()?;
        let mut positions = BTreeMap::<usize, usize>::new();
        let mut controls = Vec::with_capacity(layout.outputs.len());
        let mut proposals = Vec::with_capacity(layout.outputs.len());
        for &lane in &layout.outputs {
            let owner = layout.segments[lane].slot;
            let position = positions.entry(owner).or_default();
            let proposal = self
                .rejection
                .get(&owner)
                .and_then(|p| p.get(*position))
                .copied();
            let draw = self.draws.get(&owner).cloned().map(|mut d| {
                if proposal.is_some() {
                    d.seed ^= crate::mtp::CORRECTION_STREAM;
                }
                d
            });
            controls.push(
                draw.map(|d| d.control(*position))
                    .transpose()?
                    .unwrap_or([0; 3]),
            );
            proposals.push(proposal);
            *position += 1;
        }
        Ok((controls, proposals))
    }
    pub fn row_parameters(
        &self,
        layout: &Layout,
    ) -> Result<Vec<crate::backend::ascend::sampling::Parameters>> {
        let parameters = self.parameters.get()?;
        if parameters.is_empty() {
            return Ok(vec![]);
        }
        let mut positions = BTreeMap::<usize, usize>::new();
        layout
            .outputs
            .iter()
            .map(|&lane| {
                let owner = layout.segments[lane].slot;
                let position = positions.entry(owner).or_default();
                let value = parameters
                    .get(&owner)
                    .map(|rows| {
                        rows.get(*position)
                            .cloned()
                            .ok_or("Missing owner sampling parameters".to_string())
                    })
                    .transpose()?
                    .unwrap_or_default();
                *position += 1;
                Ok(value)
            })
            .collect()
    }
    pub fn wants_logits(&self, owner: usize) -> bool {
        self.full
            && self
                .full_owners
                .as_ref()
                .is_none_or(|owners| owners.contains(&owner))
    }
    pub fn logit_ranges(
        &self,
        outputs: impl Iterator<Item = (usize, usize)>,
    ) -> Vec<std::ops::Range<usize>> {
        let mut ranges: Vec<std::ops::Range<usize>> = vec![];
        for (row, owner) in outputs {
            if self.wants_logits(owner) {
                if let Some(last) = ranges.last_mut()
                    && last.end == row
                {
                    last.end += 1;
                } else {
                    ranges.push(row..row + 1);
                }
            }
        }
        ranges
    }
    pub fn controls(&self, layout: &Layout) -> Result<Vec<[i32; 3]>> {
        self.validate_draw_roles()?;
        let mut positions = BTreeMap::<usize, usize>::new();
        layout
            .outputs
            .iter()
            .map(|&lane| {
                let slot = layout.segments[lane].slot;
                let position = positions.entry(slot).or_default();
                let value = self
                    .draws
                    .get(&slot)
                    .map(|d| d.control(*position))
                    .transpose()?
                    .unwrap_or([0; 3]);
                *position += 1;
                Ok(value)
            })
            .collect()
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ResidentKey {
    role: super::PacketRole,
    decode: bool,
    bucketed: bool,
    total_rows: usize,
    outputs: usize,
    decoder_rows: usize,
    clipped: bool,
    pad_slot: bool,
    work_rows: Vec<usize>,
    segments: Vec<(usize, super::replay::Role)>,
}

pub(super) struct Team {
    peers: Transport,
    graphs_enabled: bool,
    handles: Vec<(JoinHandle<()>, Arc<std::sync::atomic::AtomicBool>)>,
    pub model: Arc<Model>,
    pub load_s: f64,
    pub rows: usize,
    pub target_domains: Vec<(bool, usize, usize)>,
    pub verification_rows: usize,
    pub draft_blocks: Vec<usize>,
    pub strided_draft: bool,
    pub work_launches: BTreeSet<super::batch::WorkLaunch>,
    // Drop worker contexts and communication groups before the bootstrap context.
    _bootstrap: Option<Session>,
    pub(super) host_tables: Option<Arc<SharedTables>>,
    pub(super) host_hasher: Option<Arc<orinfer_model_sdk::conditional_memory::Hasher>>,
    pub(super) host_prepare: Arc<std::sync::atomic::AtomicBool>,
    rank_free_range: std::cell::Cell<(usize, usize)>,
    resident_graphs: std::cell::RefCell<BTreeSet<ResidentKey>>,
    /// Per-owner maximum verification rows already backed by a pooled journal slab.
    verified_journal_rows: std::cell::RefCell<BTreeMap<usize, usize>>,
}

/// Execution mode and preparation are measured separately; direct is a valid steady path.
#[derive(Default)]
pub(super) struct Timing {
    pub seconds: f64,
    pub preparation: f64,
    pub replay: bool,
    pub setup: f64,
    pub shared: bool,
    pub shared_setup: f64,
    pub mask_rows: usize,
    pub report_rows: usize,
    pub mask_prepare_s: f64,
    pub captured: bool,
    pub invalidated: bool,
}
impl Timing {
    /// One completed TP packet, independent of its number of ranks or owners.
    pub fn record(&self, statistics: &mut crate::scheduler::BatchExecutionStatistics) {
        statistics.direct_iterations += usize::from(!self.replay);
        statistics.graph_hits += usize::from(self.replay && !self.captured);
        statistics.graph_misses += usize::from(!self.replay || self.captured);
        statistics.graph_invalidations += usize::from(self.invalidated);
        statistics.dynamic_graph_captures += usize::from(self.captured);
        statistics.shared_graph_hits += usize::from(self.shared);
        statistics.shared_graph_captures += usize::from(self.shared_setup > 0.);
        // Capture preparation includes the coordinated host work and warmup;
        // separately priced shared capture is already inside that interval.
        statistics.capture_s += if self.captured {
            self.preparation
        } else {
            self.shared_setup
        };
        if self.replay {
            statistics.replay_s += self.seconds;
        } else {
            statistics.direct_s += self.seconds;
        }
    }
}
impl Team {
    pub fn load(plan: &Path, model_dir: &Path) -> Result<Self> {
        Self::load_with_drafts(plan, model_dir, None, true)
    }
    pub fn load_with_drafts(
        plan: &Path,
        model_dir: &Path,
        requested: Option<usize>,
        graphs_enabled: bool,
    ) -> Result<Self> {
        let model_dir = model_dir
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let (mut model, package, layout) = load_spec(plan, &model_dir)?;
        if model.batch.is_none() {
            return Err("A2 service requires a native rows plan".into());
        }
        let draft_capacity = model.ranks[0]
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Draft)
            .filter_map(|p| p.coordinates.draft_block)
            .max()
            .unwrap_or(0);
        if requested.is_some_and(|limit| limit > draft_capacity) {
            return Err("Requested A2 drafts exceed the execution package capacity".into());
        }
        let mut draft_blocks = model.ranks[0]
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Draft)
            .filter_map(|p| p.coordinates.draft_block)
            .collect::<Vec<_>>();
        draft_blocks.sort_unstable();
        draft_blocks.dedup();
        let strided_draft = model.ranks[0]
            .packets
            .iter()
            .any(|p| p.draft_stride.is_some());
        let rows = model.ranks[0]
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Target && !p.batch_decode)
            .map(|p| p.rows)
            .max()
            .ok_or("Missing A2 prefill rows")?;
        let verification_rows = model.ranks[0]
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Target && !p.batch_decode)
            .filter(|p| {
                p.body
                    .iter()
                    .any(|step| matches!(step, super::Instruction::Exchange { .. }))
            })
            .map(|p| p.min_rows.saturating_sub(1))
            .min()
            .unwrap_or(rows);
        let verification_rows = model
            .batch
            .as_ref()
            .ok_or("Missing batch contract")?
            .verification_rows(verification_rows, rows)?;
        let timeout = channel::command_timeout(
            std::env::var("ORINFER_A2_COMMAND_TIMEOUT_MS")
                .ok()
                .as_deref(),
        )?;
        let heavy_timeout = channel::heavy_command_timeout(
            std::env::var("ORINFER_A2_HEAVY_COMMAND_TIMEOUT_MS")
                .ok()
                .as_deref(),
            timeout,
        )?;
        let work_launches = super::batch::graph_launches(&model);
        // Rank plans move to their workers; retain only target row geometry
        // needed by the host scheduler's measured cost domains.
        let target_domains = model.ranks[0]
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Target)
            .map(|p| (p.batch_decode, p.min_rows, p.max_rows.unwrap_or(p.rows)))
            .collect();
        let ranks = std::mem::take(&mut model.ranks);
        let snapshots = std::env::var_os("ORINFER_A2_BATCH_TRACE").is_some();
        let cpus = affinity::from_env(ranks.len())?;
        let host_cpus = affinity::host_cpus(cpus.as_deref())?;
        affinity::configure_host_pool(host_cpus.as_deref());
        let started = Instant::now();
        let bootstrap = Session::new(0, Limits::default())?;
        let root = RootInfo::new(&bootstrap, ranks.len() as u32)?;
        let shared = Arc::new(SharedTables::new());
        let model = Arc::new(model);
        let mut team = Self {
            peers: Transport::with_heavy_timeout(timeout, heavy_timeout),
            graphs_enabled,
            handles: vec![],
            model: Arc::clone(&model),
            load_s: 0.,
            rows,
            target_domains,
            verification_rows,
            draft_blocks,
            strided_draft,
            work_launches,
            _bootstrap: Some(bootstrap),
            host_tables: Some(shared),
            host_hasher: Some(Arc::new(
                orinfer_model_sdk::conditional_memory::Hasher::new(layout.clone())?,
            )),
            host_prepare: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            rank_free_range: std::cell::Cell::new((0, 0)),
            resident_graphs: std::cell::RefCell::new(BTreeSet::new()),
            verified_journal_rows: std::cell::RefCell::new(BTreeMap::new()),
        };
        // One shared CPU load owns no ACL handles. Join it with rank workers
        // on every startup/teardown path, including partial rank failures.
        let table_shared = Arc::clone(team.host_tables.as_ref().unwrap());
        let table_dir = model_dir.to_owned();
        let table_index = model.index.file.clone();
        let table_layout = layout.clone();
        let table_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let table_done_flag = Arc::clone(&table_done);
        team.handles.push((
            std::thread::Builder::new()
                .name("orinfer-a2-tables".into())
                .spawn(move || {
                    table_shared.get_or_init(|| {
                        super::load::tables(&table_dir, &table_index, &table_layout)
                    });
                    table_done_flag.store(true, std::sync::atomic::Ordering::Release);
                })
                .map_err(|e| e.to_string())?,
            table_done,
        ));
        for rank in ranks {
            let cpus = cpus.as_ref().map(|v| v[rank.rank as usize].clone());
            let root = root.clone();
            let shared = Arc::clone(team.host_tables.as_ref().unwrap());
            let model = Arc::clone(&model);
            let package = package.clone();
            let layout = layout.clone();
            let model_dir = model_dir.to_owned();
            let (send, commands) = mpsc::channel();
            let (reply, receive) = reply_channel();
            let index = rank.rank;
            let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let done_flag = Arc::clone(&done);
            let handle = std::thread::Builder::new()
                .name(format!("orinfer-a2-{index}"))
                .spawn(move || {
                    struct MarkDone(Arc<std::sync::atomic::AtomicBool>);
                    impl Drop for MarkDone {
                        fn drop(&mut self) {
                            self.0.store(true, std::sync::atomic::Ordering::Release);
                        }
                    }
                    let _mark_done = MarkDone(done_flag);
                    let result = (|| {
                        if let Some(cpus) = cpus.as_deref() {
                            affinity::apply(cpus)?;
                        }
                        let session = Session::new(
                            index,
                            Limits {
                                device_bytes: 58 << 30,
                                pinned_bytes: 64 << 20,
                                timeout_ms: heavy_timeout.as_millis() as i32,
                            },
                        )?;
                        session.set_graphs_enabled(graphs_enabled);
                        Worker::load(
                            session, &root, rank, &model, &model_dir, &package, layout, snapshots,
                            &shared,
                        )
                    })();
                    // Keep the table owner on Team, rather than extending its
                    // lifetime through the rank closure's native destructors.
                    drop(shared);
                    let mut worker = match result {
                        Ok(worker) => worker,
                        Err(error) => {
                            let _ = reply.send(Err(EngineError::take(error, true)));
                            return;
                        }
                    };
                    if reply.send(Ok(Reply::Ready)).is_err() {
                        return;
                    }
                    let mut snapshots = BTreeMap::new();
                    let mut lineage = BTreeMap::new();
                    let mut host_snapshots = BTreeMap::new();
                    let mut host_pool = super::state::HostPool::default();
                    let mut journal = BTreeMap::new();
                    let mut journal_pool = BTreeMap::new();
                    let mut idle_journals: Vec<super::Buffer> = vec![];
                    const MAX_IDLE_JOURNALS: usize = 4;
                    let recycle_journal =
                        |j: super::state::Journal, pool: &mut Vec<super::Buffer>| {
                            if pool.len() < MAX_IDLE_JOURNALS {
                                pool.push(j.bank);
                            }
                        };
                    while let Ok(command) = commands.recv() {
                        let result = match command {
                            Command::Stop => break,
                            Command::BeginVerification(owners) => {
                                journal.clear();
                                let mut copies = vec![];
                                let mut oom = false;
                                let mut outcome: Result<()> = Ok(());
                                for (id, rows) in owners {
                                    match worker.save_mutable(
                                        id,
                                        rows,
                                        journal_pool.remove(&id),
                                        &mut idle_journals,
                                        &mut copies,
                                    ) {
                                        Ok(saved) => {
                                            journal.insert(id, saved);
                                        }
                                        Err(error)
                                            if !worker.session.is_poisoned()
                                                && error.contains("Insufficient A2 memory") =>
                                        {
                                            oom = true;
                                            break;
                                        }
                                        Err(error) => {
                                            outcome = Err(error);
                                            break;
                                        }
                                    }
                                }
                                if oom {
                                    journal_pool.append(&mut journal);
                                    Ok(Reply::Capacity(false))
                                } else {
                                    outcome.and_then(|()| {
                                        // Compute-stream ordered before verification forward.
                                        if !copies.is_empty() {
                                            worker.copy_many(&copies)?;
                                        }
                                        Ok(Reply::Ready)
                                    })
                                }
                            }
                            Command::FinishVerification(owners) => {
                                let mut copies = vec![];
                                let result = owners
                                    .into_iter()
                                    .try_for_each(|(id, keep)| {
                                        worker.restore_mutable(
                                            id,
                                            journal
                                                .get(&id)
                                                .ok_or("Missing DSpark state journal")?,
                                            keep,
                                            &mut copies,
                                        )
                                    })
                                    .and_then(|_| {
                                        // Compute-stream ordered before subsequent draft/verify;
                                        // copy_many retains journal buffers through completion.
                                        if !copies.is_empty() {
                                            worker.copy_many(&copies)?;
                                        }
                                        Ok(Reply::Ready)
                                    });
                                journal_pool.append(&mut journal);
                                result
                            }
                            Command::Budget(requirement) => {
                                worker.check_budget(requirement).and_then(|fits| {
                                    let (device_free, _) = worker.available_memory()?;
                                    Ok(Reply::BudgetStatus { fits, device_free })
                                })
                            }
                            Command::Reclaim => {
                                if let Some(bank) = idle_journals.pop() {
                                    drop(bank);
                                    Ok(Reply::Capacity(true))
                                } else {
                                    worker.evict_batch_program().map(Reply::Capacity)
                                }
                            }
                            Command::ReclaimForAdmission => {
                                if let Some(bank) = idle_journals.pop() {
                                    drop(bank);
                                    Ok(Reply::Capacity(true))
                                } else {
                                    worker
                                        .evict_batch_program_for_admission()
                                        .map(Reply::Capacity)
                                }
                            }
                            Command::PlanSnapshot {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent = lineage.get(&owner).and_then(|id| snapshots.get(id));
                                worker
                                    .plan_owner_snapshot(owner, complete, snapshot, parent)
                                    .and_then(|allocations| {
                                        let incoming = allocations
                                            .get(&crate::backend::kv::checkpoint_id(snapshot)?)
                                            .copied()
                                            .unwrap_or(0);
                                        let (device_free, session_free) = if incoming != 0 {
                                            worker.available_memory()?
                                        } else {
                                            (usize::MAX, usize::MAX)
                                        };
                                        Ok(Reply::SnapshotPlan {
                                            allocations,
                                            device_free,
                                            session_free,
                                        })
                                    })
                            }
                            Command::PlanAndSave {
                                owner,
                                snapshot,
                                complete,
                                max_incoming_bytes,
                            } => {
                                let planned = {
                                    let parent =
                                        lineage.get(&owner).and_then(|id| snapshots.get(id));
                                    worker.plan_owner_snapshot(owner, complete, snapshot, parent)
                                };
                                match planned {
                                    Err(error) => Err(error),
                                    Ok(allocations) => (|| {
                                        let incoming = allocations
                                            .get(&crate::backend::kv::checkpoint_id(snapshot)?)
                                            .copied()
                                            .unwrap_or(0);
                                        let (device_free, session_free) = if incoming != 0 {
                                            worker.available_memory()?
                                        } else {
                                            (usize::MAX, usize::MAX)
                                        };
                                        let fits = incoming <= max_incoming_bytes
                                            && (incoming == 0
                                                || super::memory::Requirement::snapshot(incoming)?
                                                    .fits(device_free, session_free));
                                        if !fits {
                                            return Ok(Reply::SnapshotPlan {
                                                allocations,
                                                device_free,
                                                session_free,
                                            });
                                        }
                                        let saved = {
                                            let parent = lineage
                                                .get(&owner)
                                                .and_then(|id| snapshots.get(id));
                                            worker.save_owner(owner, complete, snapshot, parent)
                                        };
                                        match saved {
                                            Ok(saved) => {
                                                let saved_allocs = saved.allocations();
                                                snapshots.insert(snapshot, saved);
                                                lineage.insert(owner, snapshot);
                                                Ok(Reply::Allocations(saved_allocs))
                                            }
                                            Err(error) if !worker.session.is_poisoned() => {
                                                Ok(Reply::SaveSkipped(error))
                                            }
                                            Err(error) => Err(error),
                                        }
                                    })(),
                                }
                            }
                            Command::PlanHost {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent =
                                    lineage.get(&owner).and_then(|id| host_snapshots.get(id));
                                worker
                                    .plan_host_snapshot(owner, complete, snapshot, parent)
                                    .map(Reply::Allocations)
                            }
                            Command::SaveHost {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent =
                                    lineage.get(&owner).and_then(|id| host_snapshots.get(id));
                                match worker.save_host_snapshot(
                                    owner,
                                    complete,
                                    snapshot,
                                    parent,
                                    &mut host_pool,
                                ) {
                                    Ok(saved) => {
                                        let allocations = saved.allocations();
                                        host_snapshots.insert(snapshot, saved);
                                        lineage.insert(owner, snapshot);
                                        Ok(Reply::Allocations(allocations))
                                    }
                                    Err(error) if !worker.session.is_poisoned() => {
                                        Ok(Reply::SaveSkipped(error))
                                    }
                                    Err(error) => Err(error),
                                }
                            }
                            Command::PlanOffload(id) => snapshots
                                .get(&id)
                                .ok_or_else(|| "Missing A2 offload plan".into())
                                .map(|s: &super::state::Snapshot| {
                                    Reply::Allocations(s.host_allocations())
                                }),
                            Command::Offload(id) => match snapshots.get(&id) {
                                None => Err("Missing A2 offload snapshot".to_string()),
                                Some(saved) => match super::state::HostSnapshot::offload(
                                    &worker.session,
                                    saved,
                                    &mut host_pool,
                                ) {
                                    Ok(saved) => {
                                        host_snapshots.insert(id, saved);
                                        snapshots.remove(&id);
                                        Ok(Reply::Ready)
                                    }
                                    Err(error) if !worker.session.is_poisoned() => {
                                        Ok(Reply::SaveSkipped(error))
                                    }
                                    Err(error) => Err(error),
                                },
                            },
                            Command::InstallImages(owner, spans) => worker
                                .batch
                                .as_mut()
                                .ok_or_else(|| "Missing batch runtime".to_owned())
                                .and_then(|batch| {
                                    batch
                                        .owners
                                        .get_mut(&owner)
                                        .ok_or_else(|| "Missing image owner".to_owned())
                                })
                                .map(|owner| {
                                    owner.vision = spans;
                                    Reply::Ready
                                }),
                            Command::BeginImages(images) => {
                                match images.get(worker.rank as usize) {
                                    Some(image) => worker
                                        .vision
                                        .as_mut()
                                        .ok_or_else(|| "Missing native vision encoder".to_owned())
                                        .and_then(|encoder| encoder.begin(&worker.session, image))
                                        .map(|()| Reply::Ready),
                                    None => Ok(Reply::Ready),
                                }
                            }
                            Command::AdvanceImages(segments, capture) => {
                                let effective_capture = if capture
                                    != super::vision::CapturePolicy::Never
                                    && worker.available_memory().is_ok_and(|(d, s)| {
                                        super::memory::Requirement::workspace().fits(d, s)
                                    }) {
                                    capture
                                } else {
                                    super::vision::CapturePolicy::Never
                                };
                                worker
                                    .vision
                                    .as_mut()
                                    .ok_or_else(|| "Missing native vision encoder".to_owned())
                                    .and_then(|encoder| {
                                        encoder.advance(
                                            &worker.session,
                                            segments,
                                            effective_capture,
                                        )
                                    })
                                    .map(Reply::Image)
                            }
                            Command::CancelImages => {
                                if let Some(encoder) = &mut worker.vision {
                                    encoder.cancel();
                                }
                                Ok(Reply::Ready)
                            }
                            Command::Admit(ids) => worker.admit(&ids).map(|_| Reply::Ready),
                            Command::StartOwner {
                                slot,
                                limit,
                                snapshot,
                            } => worker.admit(&[slot]).and_then(|()| {
                                // Restore shared prefix pages before reserving future KV budget
                                // so zero-copy device prefix hits are not double-charged.
                                let step = if let Some(snapshot) = snapshot {
                                    let restored = if let Some(saved) = snapshots.get(&snapshot) {
                                        worker.restore_owner(slot, saved)
                                    } else if let Some(saved) = host_snapshots.get(&snapshot) {
                                        worker.restore_host_snapshot(slot, saved)
                                    } else {
                                        Err("Missing A2 prefix snapshot".into())
                                    };
                                    restored.map(|()| {
                                        lineage.insert(slot, snapshot);
                                    })
                                } else {
                                    Ok(())
                                }
                                .and_then(|()| worker.limit_owner(slot, limit));
                                if step.is_err() {
                                    lineage.remove(&slot);
                                    let _ = worker.release_owner(slot);
                                }
                                step.map(|()| Reply::Ready)
                            }),
                            Command::GrowCached(layout) => worker
                                .grow_batch(&layout)
                                .and_then(|()| worker.batch_split_cached(&layout))
                                .and_then(|states| {
                                    let workspace_fits = if states.all_graphs() {
                                        true
                                    } else {
                                        let (device, session) = worker.available_memory()?;
                                        super::memory::Requirement::workspace()
                                            .fits(device, session)
                                    };
                                    Ok(Reply::SplitCached(states, workspace_fits))
                                }),
                            Command::GrowCachedAndBeginVerification(layout, owners) => {
                                worker.layout_protection(&layout).and_then(|(mono, split)| {
                                    let protected = super::batch::ProtectedKeys::for_layout(
                                        mono.as_ref(),
                                        split.as_ref(),
                                    );
                                    worker.grow_batch_protected(&layout, &protected)?;
                                    let states = worker.batch_split_cached(&layout)?;
                                    let workspace_fits = if states.all_graphs() {
                                        true
                                    } else {
                                        let (device, session) = worker.available_memory()?;
                                        super::memory::Requirement::workspace()
                                            .fits(device, session)
                                    };
                                    journal.clear();
                                    let mut copies = vec![];
                                    let mut oom = false;
                                    let mut outcome: Result<()> = Ok(());
                                    for (id, rows) in owners {
                                        match worker.save_mutable_protected(
                                            id,
                                            rows,
                                            journal_pool.remove(&id),
                                            &mut idle_journals,
                                            &mut copies,
                                            &protected,
                                        ) {
                                            Ok(saved) => {
                                                journal.insert(id, saved);
                                            }
                                            Err(error)
                                                if !worker.session.is_poisoned()
                                                    && error.contains("Insufficient A2 memory") =>
                                            {
                                                oom = true;
                                                break;
                                            }
                                            Err(error) => {
                                                outcome = Err(error);
                                                break;
                                            }
                                        }
                                    }
                                    if oom {
                                        journal_pool.append(&mut journal);
                                        Ok(Reply::SplitCachedAndVerification(
                                            states,
                                            workspace_fits,
                                            false,
                                        ))
                                    } else {
                                        outcome.and_then(|()| {
                                            if !copies.is_empty() {
                                                worker.copy_many(&copies)?;
                                            }
                                            Ok(Reply::SplitCachedAndVerification(
                                                states,
                                                workspace_fits,
                                                true,
                                            ))
                                        })
                                    }
                                })
                            }
                            Command::OwnerRequirement(tokens, shared_tokens, reserve) => worker
                                .owner_requirement(tokens, shared_tokens, reserve)
                                .map(Reply::Requirement),
                            Command::Reset(ids) => ids
                                .into_iter()
                                .try_for_each(|id| {
                                    if let Some(j) = journal_pool.remove(&id) {
                                        recycle_journal(j, &mut idle_journals);
                                    }
                                    if let Some(j) = journal.remove(&id) {
                                        recycle_journal(j, &mut idle_journals);
                                    }
                                    lineage.remove(&id);
                                    worker.reset_owner(id)
                                })
                                .map(|_| Reply::Ready),
                            Command::Release(None) => {
                                journal_pool.clear();
                                journal.clear();
                                idle_journals.clear();
                                lineage.clear();
                                worker.release_owners().map(|_| Reply::Ready)
                            }
                            Command::Release(Some(owner)) => {
                                if let Some(j) = journal_pool.remove(&owner) {
                                    recycle_journal(j, &mut idle_journals);
                                }
                                if let Some(j) = journal.remove(&owner) {
                                    recycle_journal(j, &mut idle_journals);
                                }
                                lineage.remove(&owner);
                                worker.release_owner(owner).map(|_| Reply::Ready)
                            }
                            #[cfg(test)]
                            Command::Cached(layout) => {
                                worker.batch_cached(&layout).map(Reply::Cached)
                            }
                            #[cfg(test)]
                            Command::Invalidate(layout) => {
                                worker.invalidate_batch(&layout).map(|_| Reply::Ready)
                            }
                            Command::InvalidateParts(layout, trunk, tail, head) => worker
                                .invalidate_parts(&layout, trunk, tail, head)
                                .map(|_| Reply::Ready),
                            #[cfg(test)]
                            Command::Prepare(layout, capture) => {
                                worker.prepare_batch(&layout, capture).map(|_| Reply::Ready)
                            }
                            Command::PrepareParts(layout, plan) => {
                                worker.prepare_parts(&layout, plan).map(|_| Reply::Ready)
                            }
                            Command::Shared(layout) => {
                                worker.shared_state(&layout).map(Reply::Shared)
                            }
                            Command::CaptureShared(layout, id) => {
                                worker.capture_shared(&layout, &id).map(Reply::Shared)
                            }
                            Command::Forward(layout, tokens, readout, shared) => worker
                                .select_shared(&layout, &shared)
                                .and_then(|()| worker.forward_batch(&layout, &tokens, &readout))
                                .map(|values| Reply::Outputs(values, !shared.is_empty())),
                            Command::FastForward(layout, tokens, readout) => worker
                                .layout_protection(&layout)
                                .and_then(|(mono, split)| {
                                    let protected = super::batch::ProtectedKeys::for_layout(
                                        mono.as_ref(),
                                        split.as_ref(),
                                    );
                                    worker.grow_batch_protected(&layout, &protected)?;
                                    worker.forward_batch(&layout, &tokens, &readout)
                                })
                                .map(|values| Reply::Outputs(values, false)),
                            Command::FastVerifyForward(layout, owners, tokens, readout) => worker
                                .layout_protection(&layout)
                                .and_then(|(mono, split)| {
                                    let protected = super::batch::ProtectedKeys::for_layout(
                                        mono.as_ref(),
                                        split.as_ref(),
                                    );
                                    worker.grow_batch_protected(&layout, &protected)?;
                                    journal.clear();
                                    let mut copies = vec![];
                                    for (id, rows) in owners {
                                        let saved = worker.save_mutable_protected(
                                            id,
                                            rows,
                                            journal_pool.remove(&id),
                                            &mut idle_journals,
                                            &mut copies,
                                            &protected,
                                        )?;
                                        journal.insert(id, saved);
                                    }
                                    if !copies.is_empty() {
                                        worker.copy_many(&copies)?;
                                    }
                                    worker.forward_batch(&layout, &tokens, &readout)
                                })
                                .map(|values| Reply::Outputs(values, false)),
                            Command::Save {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent = lineage.get(&owner).and_then(|id| snapshots.get(id));
                                match worker.save_owner(owner, complete, snapshot, parent) {
                                    Ok(saved) => {
                                        let allocations = saved.allocations();
                                        snapshots.insert(snapshot, saved);
                                        lineage.insert(owner, snapshot);
                                        Ok(Reply::Allocations(allocations))
                                    }
                                    Err(error) if !worker.session.is_poisoned() => {
                                        Ok(Reply::SaveSkipped(error))
                                    }
                                    Err(error) => Err(error),
                                }
                            }
                            #[cfg(test)]
                            Command::Restore { owner, snapshot } => {
                                let restored = if let Some(saved) = snapshots.get(&snapshot) {
                                    worker.restore_owner(owner, saved)
                                } else if let Some(saved) = host_snapshots.get(&snapshot) {
                                    worker.restore_host_snapshot(owner, saved)
                                } else {
                                    Err("Missing A2 prefix snapshot".into())
                                };
                                restored.map(|_| {
                                    lineage.insert(owner, snapshot);
                                    Reply::Ready
                                })
                            }
                            Command::Evict(snapshot) => {
                                snapshots.remove(&snapshot);
                                host_snapshots.remove(&snapshot);
                                lineage.retain(|_, id| *id != snapshot);
                                Ok(Reply::Ready)
                            }
                            Command::EvictMany(ids) => {
                                let removed: BTreeSet<u64> = ids.into_iter().collect();
                                for id in &removed {
                                    snapshots.remove(id);
                                    host_snapshots.remove(id);
                                }
                                lineage.retain(|_, id| !removed.contains(id));
                                Ok(Reply::Ready)
                            }
                            Command::RollbackSave { owner, snapshot } => {
                                snapshots.remove(&snapshot);
                                host_snapshots.remove(&snapshot);
                                // Clear the owner's lineage on every rank so an
                                // asymmetric SaveSkipped cannot leave ranks with
                                // divergent parent snapshots on future plans.
                                lineage.remove(&owner);
                                lineage.retain(|_, id| *id != snapshot);
                                Ok(Reply::Ready)
                            }
                            Command::RoutingGroups(decode) => {
                                worker.routing_groups(decode).map(Reply::RoutingGroups)
                            }
                        };
                        let result = result.map_err(|mut error| {
                            if let Err(hccl_err) = worker.comm.check_async_error() {
                                eprintln!("A2 rank {index} HCCL async error: {hccl_err}");
                                error = format!("{error}; HCCL async: {hccl_err}");
                            }
                            EngineError::take(error, true)
                        });
                        let failed = result.is_err();
                        if let Err(error) = &result {
                            eprintln!("A2 rank {index}: {error}");
                        }
                        let evicted = worker.take_programs_evicted();
                        if reply.send_evicted(result, evicted).is_err() || failed {
                            break;
                        }
                    }
                    let _ = worker.session.synchronize();
                })
                .map_err(|error| error.to_string())?;
            team.handles.push((handle, done));
            team.peers.push(Peer { send, receive });
        }
        let ready = team.peers.receive(Duration::from_secs(1800))?;
        if ready.iter().any(|reply| !matches!(reply, Reply::Ready)) {
            return Err("Invalid A2 startup reply".into());
        }
        team.load_s = started.elapsed().as_secs_f64();
        Ok(team)
    }
    fn sync_resident_graphs(&self) {
        if self.peers.take_programs_evicted() {
            self.resident_graphs.borrow_mut().clear();
        }
    }
    pub fn exchange(&self, command: Command) -> Result<Vec<Reply>> {
        #[cfg(test)]
        if matches!(command, Command::Prepare(_, _) | Command::Invalidate(_)) {
            self.resident_graphs.borrow_mut().clear();
        }
        match &command {
            Command::Release(None) => self.verified_journal_rows.borrow_mut().clear(),
            Command::Release(Some(slot)) => {
                self.verified_journal_rows.borrow_mut().remove(slot);
            }
            Command::Reset(ids) => {
                let mut rows = self.verified_journal_rows.borrow_mut();
                for id in ids {
                    rows.remove(id);
                }
            }
            _ => {}
        }
        let replies = exchange(&self.peers, command)?;
        self.sync_resident_graphs();
        Ok(replies)
    }
    pub fn has_fused_small_target(&self) -> bool {
        self.target_domains
            .iter()
            .any(|&(decode, min_rows, max_rows)| !decode && min_rows <= 1 && max_rows != 31)
    }
    pub fn can_fast_verify(&self, layout: &Layout, journal: &[(usize, usize)]) -> bool {
        if !self.is_resident(layout) {
            return false;
        }
        let pooled = self.verified_journal_rows.borrow();
        journal
            .iter()
            .all(|&(slot, rows)| pooled.get(&slot).is_some_and(|&cap| cap >= rows))
    }
    pub fn finish_verification(
        &self,
        rollback: Vec<(usize, usize)>,
        verified: &[(usize, usize)],
    ) -> Result<()> {
        self.exchange(Command::FinishVerification(rollback))?;
        let mut pooled = self.verified_journal_rows.borrow_mut();
        for &(slot, rows) in verified {
            pooled
                .entry(slot)
                .and_modify(|cap| *cap = (*cap).max(rows))
                .or_insert(rows);
        }
        Ok(())
    }
    pub fn exchange_heavy(&self, command: Command) -> Result<Vec<Reply>> {
        let replies = self.peers.exchange_heavy(command)?;
        self.sync_resident_graphs();
        Ok(replies)
    }
    pub fn rank_free_range(&self) -> (usize, usize) {
        self.rank_free_range.get()
    }
    pub fn try_offload(&self, id: u64) -> Result<bool> {
        let mut skipped = None;
        for reply in self.exchange_heavy(Command::Offload(id))? {
            match reply {
                Reply::Ready => {}
                Reply::SaveSkipped(reason) => {
                    skipped.get_or_insert(reason);
                }
                _ => return Err("Invalid A2 offload reply".into()),
            }
        }
        if let Some(reason) = skipped {
            if std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
                eprintln!("A2 offload skipped for snapshot {id}: {reason}");
            }
            return Ok(false);
        }
        Ok(true)
    }
    pub fn try_host_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
        plan: bool,
    ) -> Result<Option<BTreeMap<u64, usize>>> {
        let command = if plan {
            Command::PlanHost {
                owner,
                snapshot,
                complete,
            }
        } else {
            Command::SaveHost {
                owner,
                snapshot,
                complete,
            }
        };
        let replies = if plan {
            self.exchange(command)?
        } else {
            self.exchange_heavy(command)?
        };
        self.try_combine_allocations(replies, owner, snapshot)
    }
    pub fn try_snapshot_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
    ) -> Result<Option<BTreeMap<u64, usize>>> {
        let replies = self.exchange(Command::Save {
            owner,
            snapshot,
            complete,
        })?;
        self.try_combine_allocations(replies, owner, snapshot)
    }
    fn try_combine_allocations(
        &self,
        replies: Vec<Reply>,
        owner: usize,
        snapshot: u64,
    ) -> Result<Option<BTreeMap<u64, usize>>> {
        try_combine_allocations_with(&self.peers, replies, owner, snapshot)
    }
    pub fn forward(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        full: bool,
    ) -> Result<(Vec<Output>, f64, f64)> {
        forward(&self.peers, layout, tokens, full, self.graphs_enabled)
    }
    #[cfg(test)]
    pub fn forward_with(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        readout: impl Into<Readout>,
        budget: Option<CaptureBudget>,
        maintenance: impl FnOnce(),
    ) -> Result<(Vec<Output>, Timing)> {
        forward_with(
            &self.peers,
            layout,
            tokens,
            readout.into(),
            budget,
            maintenance,
            self.graphs_enabled,
        )
    }

    pub fn profile_forward(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        readout: Readout,
        graph: bool,
    ) -> Result<(Vec<Output>, Timing)> {
        forward_with(&self.peers, layout, tokens, readout, None, || {}, graph)
    }

    pub fn offload_allocations(&self, id: u64) -> Result<BTreeMap<u64, usize>> {
        self.allocations(Command::PlanOffload(id))
    }
    pub fn ranks(&self) -> usize {
        self.peers.iter().count()
    }
    pub fn host_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
        plan: bool,
    ) -> Result<BTreeMap<u64, usize>> {
        self.allocations(if plan {
            Command::PlanHost {
                owner,
                snapshot,
                complete,
            }
        } else {
            Command::SaveHost {
                owner,
                snapshot,
                complete,
            }
        })
    }
    #[cfg(test)]
    pub fn snapshot_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
    ) -> Result<BTreeMap<u64, usize>> {
        self.allocations(Command::Save {
            owner,
            snapshot,
            complete,
        })
    }
    fn resident_key(&self, layout: &Layout) -> Option<ResidentKey> {
        if !self.graphs_enabled || layout.has_work_shape() {
            return None;
        }
        let contract = self.model.batch.as_ref()?;
        let (clipped, work_rows) =
            super::batch::resident_work_rows(contract, layout, &self.work_launches).ok()?;
        let replay = contract.decoder_replay.as_ref();
        let final_only = replay.is_some_and(|r| r.final_prompt_only);
        let keep = replay.filter(|r| r.final_prompt_only).map_or(0, |r| r.keep);
        let decoder_rows = if keep > 0 && layout.role == super::PacketRole::Target && !layout.decode
        {
            layout
                .segments
                .iter()
                .zip(&layout.decoder_roles)
                .map(|(s, role)| {
                    if *role == super::replay::Role::Producer {
                        0
                    } else {
                        s.tokens.min(keep)
                    }
                })
                .sum()
        } else {
            0
        };
        let use_bucket_key = layout.bucketed && (final_only || replay.is_none());
        let segments = if use_bucket_key {
            vec![]
        } else {
            layout
                .segments
                .iter()
                .zip(&layout.decoder_roles)
                .map(|(s, &r)| (s.tokens, r))
                .collect()
        };
        Some(ResidentKey {
            role: layout.role,
            decode: layout.decode,
            bucketed: use_bucket_key,
            total_rows: layout.rows.total,
            outputs: layout.outputs.len(),
            decoder_rows,
            clipped,
            pad_slot: layout.pad_slot.is_some(),
            work_rows,
            segments,
        })
    }
    pub fn is_resident(&self, layout: &Layout) -> bool {
        self.sync_resident_graphs();
        self.resident_key(layout)
            .is_some_and(|key| self.resident_graphs.borrow().contains(&key))
    }
    /// Pre-capture the canonical serving ACL Graphs (pure prefill Trunk, mixed
    /// Trunk/Tail/Head, decode-only verification buckets, and Draft block
    /// shapes) before `/health` reports ready, removing cold ACLNN/TileLang
    /// initialization and on-path capture pauses from the serving window.
    pub fn prewarm_graphs(&self) -> Result<()> {
        if !self.graphs_enabled || std::env::var_os("ORINFER_A2_NO_PREWARM").is_some() {
            return Ok(());
        }
        let Some(contract) = self.model.batch.as_ref() else {
            return Ok(());
        };
        let owners = contract.owners;
        let capacity = self.rows;
        if owners < 2 || capacity < 512 {
            return Ok(());
        }
        let started = Instant::now();
        let draft_block = self.draft_blocks.first().copied();
        let draft_owners = if draft_block.is_some() {
            owners.min(14)
        } else {
            2
        };
        let mut started_slots = Vec::with_capacity(draft_owners);
        for slot in 0..draft_owners {
            let limit = if slot < 3 { capacity } else { 128 };
            self.exchange(Command::StartOwner {
                slot,
                limit,
                snapshot: None,
            })?;
            started_slots.push(slot);
        }
        let capture_one = |layout: Layout| -> Result<bool> {
            let prepared = self.begin_forward(&layout)?;
            if !prepared.workspace_fits {
                return Ok(false);
            }
            if prepared.all_graphs() {
                return Ok(true);
            }
            let has_tail = prepared.states.first().is_some_and(|s| s.tail.is_some());
            let has_head = prepared.states.first().is_some_and(|s| s.head.is_some());
            let need = |states: Vec<CacheState>| super::batch::PartAction {
                invalidate: false,
                prepare: if states.iter().all(|s| *s == CacheState::Graph) {
                    None
                } else {
                    Some(true)
                },
            };
            let plan = SplitPlan {
                trunk: need(prepared.states.iter().map(|s| s.trunk).collect()),
                tail: has_tail
                    .then(|| need(prepared.states.iter().map(|s| s.tail.unwrap()).collect())),
                head: has_head
                    .then(|| need(prepared.states.iter().map(|s| s.head.unwrap()).collect())),
            };
            if plan.any_prepare() {
                self.peers
                    .exchange_heavy(Command::PrepareParts(layout.clone(), plan))?;
                self.sync_resident_graphs();
            }
            Ok(true)
        };
        let mut captured = 0usize;
        // 1. Pure prefill 2048- and 1024-row Trunks (also pays one-time ACLNN/TileLang load).
        for pure_rows in [capacity, 1024usize] {
            if pure_rows > capacity {
                continue;
            }
            let pure = Layout::new(
                vec![orinfer_model_sdk::architecture::BatchSegment {
                    slot: 0,
                    tokens: pure_rows,
                }],
                vec![],
                false,
                capacity,
                owners,
            );
            if let Ok(mut pure) = pure {
                pure.enable_bucketing();
                if capture_one(pure).unwrap_or(false) {
                    captured += 1;
                }
            }
        }
        // 2. Decode-only canonical verification buckets (1, 6, 8, 9, 12, 16, 20, 24, 32, 48, 64, 80, 96),
        // plus 9-owner (WorkPad::Bucketed=1024) decode verification buckets (64, 80, 96).
        let output_cap = contract.output_capacity();
        for rows in [1usize, 6, 8, 9, 12, 16, 20, 24, 32, 48, 64, 80, 96] {
            if rows > capacity || rows > output_cap {
                continue;
            }
            if let Ok(mut dec) = Layout::verification(
                vec![orinfer_model_sdk::architecture::BatchSegment {
                    slot: 0,
                    tokens: rows,
                }],
                capacity,
                owners,
            ) {
                dec.enable_bucketing();
                if capture_one(dec).unwrap_or(false) {
                    captured += 1;
                }
            }
        }
        if draft_owners >= 9 {
            for rows in [64usize, 80, 96] {
                if rows > capacity || rows > output_cap {
                    continue;
                }
                let per = (rows / 9).max(1);
                let rem = rows.saturating_sub(per * 8);
                if rem == 0 {
                    continue;
                }
                let segs = (0..8)
                    .map(|slot| orinfer_model_sdk::architecture::BatchSegment { slot, tokens: per })
                    .collect();
                if let Ok(mut dec9) = Layout::verification(segs, capacity, owners)
                    && dec9.append_pad(8, rem, rem, None, capacity, owners).is_ok()
                    && capture_one(dec9).unwrap_or(false)
                {
                    captured += 1;
                }
            }
        }
        // 3. Mixed 2048/1024 Trunks + canonical Tail/Head buckets (both <=8 and >8 owners).
        if let Some(replay) = contract.decoder_replay.as_ref() {
            for (total_rows, dec_rows) in [
                (capacity, 12usize),
                (capacity, 20usize),
                (capacity, 24usize),
                (capacity, 32usize),
                (capacity, 64usize),
                (capacity, 127usize),
                (1024usize, 24usize),
                (1024usize, 32usize),
                (1024usize, 64usize),
            ] {
                if total_rows > capacity || dec_rows >= total_rows || dec_rows > output_cap {
                    continue;
                }
                if let Ok(mut mixed) = Layout::new(
                    vec![orinfer_model_sdk::architecture::BatchSegment {
                        slot: 0,
                        tokens: total_rows - dec_rows,
                    }],
                    vec![],
                    false,
                    capacity,
                    owners,
                ) && mixed
                    .append_pad(1, dec_rows, dec_rows, None, capacity, owners)
                    .is_ok()
                    && capture_one(mixed).unwrap_or(false)
                {
                    captured += 1;
                }
            }
            // 9- and 10-owner mixed 2048 shapes capture WorkPad::Bucketed=1024 Trunk
            // (with both <=8 and >8 decoder Tail graphs) when >8 owners are active.
            if draft_owners >= 9 {
                for dec_rows in [64usize, 127usize] {
                    if dec_rows >= capacity || dec_rows > output_cap {
                        continue;
                    }
                    for total_owners in [9usize, 10usize] {
                        if total_owners > draft_owners {
                            continue;
                        }
                        let dec_owners = total_owners - 1;
                        let real_dec = dec_owners - 1;
                        let per = (dec_rows / dec_owners).max(1);
                        let rem = dec_rows.saturating_sub(per * real_dec);
                        if rem == 0 {
                            continue;
                        }
                        let mut segs = Vec::with_capacity(real_dec + 1);
                        segs.push(orinfer_model_sdk::architecture::BatchSegment {
                            slot: 0,
                            tokens: capacity - dec_rows,
                        });
                        for slot in 1..=real_dec {
                            segs.push(orinfer_model_sdk::architecture::BatchSegment {
                                slot,
                                tokens: per,
                            });
                        }
                        if let Ok(mut mixed_n) = Layout::new(segs, vec![], false, capacity, owners)
                        {
                            for lane in 1..=real_dec {
                                mixed_n.decoder_roles[lane] = super::replay::Role::Continue;
                                for pos in 0..per {
                                    mixed_n.outputs.push(lane);
                                    mixed_n.output_positions.push(pos);
                                }
                            }
                            if mixed_n
                                .append_pad(real_dec + 1, rem, rem, None, capacity, owners)
                                .is_ok()
                                && capture_one(mixed_n).unwrap_or(false)
                            {
                                captured += 1;
                            }
                        }
                    }
                }
            }
            // Final-window mixed buckets (1 Final prefill keeping `keep` decoder rows + decode/pad).
            let keep = replay.keep;
            if replay.final_prompt_only && keep > 0 && keep < capacity {
                for (total_rows, target_dec, target_out) in [
                    (capacity, 192usize, 32usize),
                    (capacity, 192usize, 64usize),
                    (capacity, 256usize, 127usize),
                    (1024usize, 192usize, 32usize),
                ] {
                    if total_rows > capacity || target_out > output_cap || target_dec <= keep {
                        continue;
                    }
                    let extra_dec = target_dec - keep;
                    let dec_toks = target_out.saturating_sub(1).min(extra_dec);
                    let pad_len = extra_dec - dec_toks;
                    let pad_out = target_out.saturating_sub(1 + dec_toks);
                    let prefill_toks = total_rows.saturating_sub(dec_toks + pad_len);
                    if prefill_toks < keep {
                        continue;
                    }
                    let mut segs = vec![orinfer_model_sdk::architecture::BatchSegment {
                        slot: 0,
                        tokens: prefill_toks,
                    }];
                    if dec_toks > 0 {
                        segs.push(orinfer_model_sdk::architecture::BatchSegment {
                            slot: 1,
                            tokens: dec_toks,
                        });
                    }
                    if let Ok(mut fin) = Layout::new(segs, vec![0], false, capacity, owners) {
                        if dec_toks > 0 {
                            fin.decoder_roles[1] = super::replay::Role::Continue;
                            for pos in 0..dec_toks {
                                fin.outputs.push(1);
                                fin.output_positions.push(pos);
                            }
                        }
                        let ready = if pad_len > 0 {
                            fin.append_pad(2, pad_len, pad_out, None, capacity, owners)
                                .is_ok()
                        } else {
                            fin.enable_bucketing();
                            true
                        };
                        if ready && capture_one(fin).unwrap_or(false) {
                            captured += 1;
                        }
                    }
                }
                // 9-owner Final-window mixed buckets capture the WorkPad::Bucketed=1024
                // graphs used when >8 owners are active alongside a final prefill window.
                if draft_owners >= 9 {
                    for (target_dec, target_out) in [(192usize, 64usize), (256usize, 127usize)] {
                        if target_out > output_cap || target_dec <= keep {
                            continue;
                        }
                        let extra_dec = target_dec - keep;
                        let dec_total = target_out.saturating_sub(1).min(extra_dec);
                        let per = (dec_total / 8).max(1);
                        let rem = extra_dec.saturating_sub(per * 7);
                        let pad_out = target_out.saturating_sub(1 + per * 7);
                        let prefill_toks = capacity.saturating_sub(per * 7 + rem);
                        if rem == 0 || prefill_toks < keep || pad_out > rem {
                            continue;
                        }
                        let mut segs = Vec::with_capacity(8);
                        segs.push(orinfer_model_sdk::architecture::BatchSegment {
                            slot: 0,
                            tokens: prefill_toks,
                        });
                        for slot in 1..8 {
                            segs.push(orinfer_model_sdk::architecture::BatchSegment {
                                slot,
                                tokens: per,
                            });
                        }
                        if let Ok(mut fin9) = Layout::new(segs, vec![0], false, capacity, owners) {
                            for lane in 1..8 {
                                fin9.decoder_roles[lane] = super::replay::Role::Continue;
                                for pos in 0..per {
                                    fin9.outputs.push(lane);
                                    fin9.output_positions.push(pos);
                                }
                            }
                            if fin9
                                .append_pad(8, rem, pad_out, None, capacity, owners)
                                .is_ok()
                                && capture_one(fin9).unwrap_or(false)
                            {
                                captured += 1;
                            }
                        }
                    }
                }
                // Two-final-window mixed bucket (384 decoder rows = 2x128 Final + 128 decode/pad,
                // 64 outputs) removes Direct fallbacks when two short final prefills pack together.
                // Place the large producer chunk on slot 0 (which has full `capacity` token budget).
                if capacity >= 2048 && keep * 3 <= capacity && 64 <= output_cap {
                    let dec_toks = 62usize;
                    let pad_len = keep.saturating_sub(dec_toks);
                    let prod_toks = capacity - keep * 2 - dec_toks - pad_len;
                    let segs = vec![
                        orinfer_model_sdk::architecture::BatchSegment {
                            slot: 0,
                            tokens: prod_toks,
                        },
                        orinfer_model_sdk::architecture::BatchSegment {
                            slot: 1,
                            tokens: keep,
                        },
                        orinfer_model_sdk::architecture::BatchSegment {
                            slot: 2,
                            tokens: keep,
                        },
                        orinfer_model_sdk::architecture::BatchSegment {
                            slot: 3,
                            tokens: dec_toks,
                        },
                    ];
                    if let Ok(mut fin2) = Layout::new(segs, vec![1, 2], false, capacity, owners) {
                        fin2.decoder_roles[3] = super::replay::Role::Continue;
                        for pos in 0..dec_toks {
                            fin2.outputs.push(3);
                            fin2.output_positions.push(pos);
                        }
                        if fin2
                            .append_pad(4, pad_len, 0, None, capacity, owners)
                            .is_ok()
                            && capture_one(fin2).unwrap_or(false)
                        {
                            captured += 1;
                        }
                    }
                }
            }
        }
        // 4. DSpark Draft block buckets (1..=14 owners).
        if let Some(block) = draft_block {
            for n in 1..=draft_owners {
                let segments = (0..n)
                    .map(|slot| orinfer_model_sdk::architecture::BatchSegment {
                        slot,
                        tokens: block,
                    })
                    .collect::<Vec<_>>();
                let built = if self.strided_draft {
                    Layout::new(segments, (0..n).collect(), false, owners * block, owners)
                } else {
                    Layout::verification(segments, owners * block, owners)
                };
                if let Ok(mut draft) = built {
                    draft.role = super::PacketRole::Draft;
                    if capture_one(draft).unwrap_or(false) {
                        captured += 1;
                    }
                }
            }
        }
        if !started_slots.is_empty() {
            self.exchange(Command::Release(None))?;
        }
        eprintln!(
            "A2 PREWARM READY after {:.3}s; graphs={captured}",
            started.elapsed().as_secs_f64()
        );
        Ok(())
    }
    pub fn fast_forward(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        readout: impl Into<Readout>,
        maintenance: impl FnOnce(),
    ) -> Result<(Vec<Output>, Timing)> {
        let started = Instant::now();
        let readout = readout.into();
        let replicated = layout.role == super::PacketRole::Draft;
        let sampled = !readout.draws.is_empty();
        let readout_rejection_empty = readout.rejection.is_empty();
        let parameters = Arc::clone(&readout.parameters);
        let first_input = readout.first_input.clone();
        self.peers
            .send(Command::FastForward(layout, tokens, readout))?;
        maintenance();
        let results = self.peers.collect()?;
        self.sync_resident_graphs();
        collect_forward_outputs(
            results,
            replicated,
            false,
            sampled,
            readout_rejection_empty,
            parameters,
            first_input,
            started,
            0.,
            true,
            false,
            false,
            false,
            0.,
            0.,
        )
    }
    pub fn fast_verify_forward(
        &self,
        layout: Layout,
        journal: Vec<(usize, usize)>,
        tokens: Vec<Vec<u32>>,
        readout: impl Into<Readout>,
    ) -> Result<(Vec<Output>, Timing)> {
        let started = Instant::now();
        let readout = readout.into();
        let sampled = !readout.draws.is_empty();
        let readout_rejection_empty = readout.rejection.is_empty();
        let parameters = Arc::clone(&readout.parameters);
        let first_input = readout.first_input.clone();
        self.peers
            .send(Command::FastVerifyForward(layout, journal, tokens, readout))?;
        let results = self.peers.collect()?;
        self.sync_resident_graphs();
        collect_forward_outputs(
            results,
            false,
            false,
            sampled,
            readout_rejection_empty,
            parameters,
            first_input,
            started,
            0.,
            true,
            false,
            false,
            false,
            0.,
            0.,
        )
    }
    pub fn begin_forward(&self, layout: &Layout) -> Result<PreparedForward> {
        let prepared = begin_forward(&self.peers, layout)?;
        self.sync_resident_graphs();
        if prepared.all_graphs()
            && let Some(key) = self.resident_key(layout)
        {
            self.resident_graphs.borrow_mut().insert(key);
        }
        Ok(prepared)
    }
    pub fn send_verification_forward(
        &self,
        layout: &Layout,
        journal: Vec<(usize, usize)>,
    ) -> Result<Instant> {
        let started = Instant::now();
        self.peers.send(Command::GrowCachedAndBeginVerification(
            layout.clone(),
            journal,
        ))?;
        Ok(started)
    }
    pub fn collect_verification_forward(
        &self,
        layout: &Layout,
        started: Instant,
    ) -> Result<(PreparedForward, bool)> {
        let replies = self.peers.collect()?;
        let mut states = Vec::with_capacity(replies.len());
        let mut workspace_fits = true;
        let mut journal_ready = true;
        for reply in replies {
            let Reply::SplitCachedAndVerification(state, fits, ready) = reply else {
                return Err("Invalid A2 verification cache state reply".to_string());
            };
            states.push(state);
            workspace_fits &= fits;
            journal_ready &= ready;
        }
        let prepared = PreparedForward {
            started,
            states,
            workspace_fits,
        };
        self.sync_resident_graphs();
        if prepared.all_graphs()
            && let Some(key) = self.resident_key(layout)
        {
            self.resident_graphs.borrow_mut().insert(key);
        }
        Ok((prepared, journal_ready))
    }
    pub fn begin_verification_forward(
        &self,
        layout: &Layout,
        journal: Vec<(usize, usize)>,
    ) -> Result<(PreparedForward, bool)> {
        let started = self.send_verification_forward(layout, journal)?;
        self.collect_verification_forward(layout, started)
    }
    pub fn try_plan_and_save(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
        max_incoming_bytes: usize,
    ) -> Result<Option<BTreeMap<u64, usize>>> {
        let replies = self.exchange(Command::PlanAndSave {
            owner,
            snapshot,
            complete,
            max_incoming_bytes,
        })?;
        let mut any_not_saved = false;
        let mut any_saved = false;
        let mut min_device = usize::MAX;
        let mut max_device = 0usize;
        for reply in &replies {
            match reply {
                Reply::Allocations(_) => any_saved = true,
                Reply::SnapshotPlan { device_free, .. } => {
                    any_not_saved = true;
                    if *device_free != usize::MAX {
                        min_device = min_device.min(*device_free);
                        max_device = max_device.max(*device_free);
                    }
                }
                Reply::SaveSkipped(_) => any_not_saved = true,
                _ => return Err("Invalid A2 plan-and-save reply".into()),
            }
        }
        if min_device != usize::MAX {
            self.rank_free_range.set((min_device, max_device));
        }
        if any_not_saved {
            if any_saved {
                self.exchange(Command::RollbackSave { owner, snapshot })?;
            }
            return Ok(None);
        }
        self.try_combine_allocations(replies, owner, snapshot)
    }
    pub fn finish_forward(
        &self,
        prepared: PreparedForward,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        readout: impl Into<Readout>,
        budget: Option<CaptureBudget>,
        maintenance: impl FnOnce(),
    ) -> Result<(Vec<Output>, Timing)> {
        let key = self.resident_key(&layout);
        let (outputs, timing) = finish_forward(
            &self.peers,
            prepared,
            layout,
            tokens,
            readout.into(),
            budget,
            maintenance,
            self.graphs_enabled,
        )?;
        let evicted = self.peers.take_programs_evicted();
        if evicted || timing.invalidated || timing.captured || (!timing.replay && timing.setup > 0.)
        {
            let mut resident = self.resident_graphs.borrow_mut();
            resident.clear();
            if timing.replay
                && let Some(key) = key
            {
                resident.insert(key);
            }
        } else if timing.replay
            && let Some(key) = key
        {
            self.resident_graphs.borrow_mut().insert(key);
        }
        Ok((outputs, timing))
    }
    pub fn plan_snapshot_with_budget(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
    ) -> Result<(BTreeMap<u64, usize>, usize, usize)> {
        let mut result: Option<BTreeMap<u64, usize>> = None;
        let mut min_device = usize::MAX;
        let mut max_device = 0usize;
        let mut min_session = usize::MAX;
        for (rank, reply) in self
            .exchange(Command::PlanSnapshot {
                owner,
                snapshot,
                complete,
            })?
            .into_iter()
            .enumerate()
        {
            let Reply::SnapshotPlan {
                allocations,
                device_free,
                session_free,
            } = reply
            else {
                return Err("Invalid A2 snapshot plan reply".into());
            };
            min_device = min_device.min(device_free);
            if device_free != usize::MAX {
                max_device = max_device.max(device_free);
            }
            min_session = min_session.min(session_free);
            if let Some(combined) = &mut result {
                if !combined.keys().eq(allocations.keys()) {
                    return Err(format!(
                        "TP snapshot allocation lineage differs on rank {rank}"
                    ));
                }
                for (id, bytes) in allocations {
                    *combined.get_mut(&id).unwrap() = combined[&id].max(bytes);
                }
            } else {
                result = Some(allocations);
            }
        }
        if min_device != usize::MAX {
            self.rank_free_range.set((min_device, max_device));
        }
        Ok((
            result.ok_or_else(|| "Empty A2 snapshot team".to_string())?,
            min_device,
            min_session,
        ))
    }
    pub fn has_budget(&self, requirement: super::memory::Requirement) -> Result<bool> {
        let mut all_fit = true;
        let mut min_free = usize::MAX;
        let mut max_free = 0usize;
        for reply in self.exchange(Command::Budget(requirement))? {
            let Reply::BudgetStatus { fits, device_free } = reply else {
                return Err("Invalid A2 budget reply".into());
            };
            all_fit &= fits;
            min_free = min_free.min(device_free);
            max_free = max_free.max(device_free);
        }
        if min_free != usize::MAX {
            self.rank_free_range.set((min_free, max_free));
        }
        Ok(all_fit)
    }
    pub fn owner_requirement(
        &self,
        tokens: usize,
        shared_tokens: usize,
        reserve: usize,
    ) -> Result<super::memory::Requirement> {
        let mut maximum = super::memory::Requirement {
            device: 0,
            session: 0,
        };
        for reply in self.exchange(Command::OwnerRequirement(tokens, shared_tokens, reserve))? {
            let Reply::Requirement(required) = reply else {
                return Err("Invalid A2 owner requirement reply".into());
            };
            maximum.device = maximum.device.max(required.device);
            maximum.session = maximum.session.max(required.session);
        }
        Ok(maximum)
    }
    pub fn reclaim_programs(&self) -> Result<bool> {
        self.resident_graphs.borrow_mut().clear();
        Ok(self.capacities(Command::Reclaim)?.into_iter().any(|v| v))
    }
    pub fn reclaim_programs_for_admission(&self) -> Result<bool> {
        self.resident_graphs.borrow_mut().clear();
        Ok(self
            .capacities(Command::ReclaimForAdmission)?
            .into_iter()
            .any(|v| v))
    }
    fn capacities(&self, command: Command) -> Result<Vec<bool>> {
        self.exchange(command)?
            .into_iter()
            .map(|reply| match reply {
                Reply::Capacity(value) => Ok(value),
                _ => Err("Invalid A2 capacity reply".into()),
            })
            .collect()
    }
    fn allocations(&self, command: Command) -> Result<BTreeMap<u64, usize>> {
        let mut result: Option<BTreeMap<u64, usize>> = None;
        for (rank, reply) in self.exchange(command)?.into_iter().enumerate() {
            let Reply::Allocations(allocations) = reply else {
                return Err("Invalid A2 snapshot allocation reply".into());
            };
            if let Some(combined) = &mut result {
                if !combined.keys().eq(allocations.keys()) {
                    let missing = combined
                        .keys()
                        .filter(|k| !allocations.contains_key(k))
                        .take(8)
                        .collect::<Vec<_>>();
                    let extra = allocations
                        .keys()
                        .filter(|k| !combined.contains_key(k))
                        .take(8)
                        .collect::<Vec<_>>();
                    return Err(format!(
                        "TP snapshot allocation lineage differs on rank {rank}: missing {missing:?}, extra {extra:?}"
                    ));
                }
                for (id, bytes) in allocations {
                    *combined.get_mut(&id).unwrap() = combined[&id].max(bytes);
                }
            } else {
                result = Some(allocations);
            }
        }
        result.ok_or_else(|| "Empty A2 snapshot team".into())
    }
}

impl Worker {
    /// Benchmark diagnostics read completed groups on their owning worker.
    /// The online request path never sends this command.
    fn routing_groups(&self, decode: bool) -> Result<Vec<RoutingGroup>> {
        let mut groups = BTreeMap::new();
        for packet in self
            .packet_templates
            .iter()
            .filter(|p| p.role == super::PacketRole::Target && p.batch_decode == decode)
        {
            for step in &packet.body {
                if let super::Instruction::W4 {
                    experts,
                    groups: Some(name),
                    ..
                }
                | super::Instruction::W8 {
                    experts,
                    groups: Some(name),
                    ..
                } = step
                    && groups.insert(name, *experts).is_some_and(|n| n != *experts)
                {
                    return Err("Routing group expert counts differ across consumers".into());
                }
            }
        }
        groups
            .into_iter()
            .map(|(name, experts)| {
                let buffer = super::get(&self.buffers, name)?;
                if experts.checked_mul(8) != Some(buffer.bytes()) {
                    return Err("Routing group buffer extent differs from its W4 contract".into());
                }
                let bytes = self.session.download(&buffer, 0, buffer.bytes())?;
                let counts = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect::<Vec<_>>();
                if counts.iter().any(|&n| n < 0) {
                    return Err("Routing group contains a negative count".into());
                }
                Ok(RoutingGroup {
                    buffer: name.clone(),
                    counts,
                })
            })
            .collect()
    }
}
impl Drop for Team {
    fn drop(&mut self) {
        for peer in self.peers.iter() {
            let _ = peer.send.send(Command::Stop);
        }
        let host_release = self.host_tables.take().and_then(host_release::start);
        let failed = self.peers.has_failed();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut detached = 0usize;
        for (handle, done) in self.handles.drain(..) {
            if failed {
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    if Instant::now() >= deadline {
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                if !done.load(std::sync::atomic::Ordering::Acquire) {
                    detached += 1;
                    continue;
                }
            }
            if handle.join().is_err() {
                eprintln!("A2 worker panicked during teardown");
            }
        }
        if detached > 0 {
            eprintln!(
                "A2 teardown detached {detached} wedged rank thread(s) after failure deadline"
            );
            // Do not destroy the bootstrap ACL context or run CANN process-exit
            // destructors while a wedged rank thread is still inside native calls.
            crate::backend::ascend::runtime::mark_quarantined();
            std::mem::forget(self._bootstrap.take());
        } else {
            // CPU table reclamation owns no ACL/HCCL state and overlaps both rank
            // teardown and the final bootstrap context. Join before releasing the
            // model owner's GPU lock or returning from destruction.
            drop(self._bootstrap.take());
        }
        if let Some(handle) = host_release
            && handle.join().is_err()
        {
            eprintln!("CPU table release thread panicked during teardown");
        }
    }
}

fn exchange(peers: &Transport, command: Command) -> Result<Vec<Reply>> {
    peers.exchange(command)
}

fn try_combine_allocations_with(
    peers: &Transport,
    replies: Vec<Reply>,
    owner: usize,
    snapshot: u64,
) -> Result<Option<BTreeMap<u64, usize>>> {
    if let Some(reason) = replies.iter().find_map(|reply| match reply {
        Reply::SaveSkipped(reason) => Some(reason.as_str()),
        _ => None,
    }) {
        if std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
            eprintln!("A2 prefix save skipped for snapshot {snapshot}: {reason}");
        }
        peers.exchange(Command::RollbackSave { owner, snapshot })?;
        return Ok(None);
    }
    let mut result: Option<BTreeMap<u64, usize>> = None;
    for (rank, reply) in replies.into_iter().enumerate() {
        let Reply::Allocations(allocations) = reply else {
            return Err("Invalid A2 snapshot allocation reply".into());
        };
        if let Some(combined) = &mut result {
            if !combined.keys().eq(allocations.keys()) {
                let missing = combined
                    .keys()
                    .filter(|k| !allocations.contains_key(k))
                    .take(8)
                    .collect::<Vec<_>>();
                let extra = allocations
                    .keys()
                    .filter(|k| !combined.contains_key(k))
                    .take(8)
                    .collect::<Vec<_>>();
                return Err(format!(
                    "TP snapshot allocation lineage differs on rank {rank}: missing {missing:?}, extra {extra:?}"
                ));
            }
            for (id, bytes) in allocations {
                *combined.get_mut(&id).unwrap() = combined[&id].max(bytes);
            }
        } else {
            result = Some(allocations);
        }
    }
    Ok(Some(
        result.ok_or_else(|| "Empty A2 snapshot team".to_string())?,
    ))
}

fn forward(
    peers: &Transport,
    layout: Layout,
    tokens: Vec<Vec<u32>>,
    full: bool,
    graphs_enabled: bool,
) -> Result<(Vec<Output>, f64, f64)> {
    let (outputs, timing) = forward_with(
        peers,
        layout,
        tokens,
        full.into(),
        None,
        || {},
        graphs_enabled,
    )?;
    Ok((outputs, timing.seconds, timing.preparation))
}
#[cfg(test)]
fn preparation(states: &[CacheState], repeated: bool) -> Result<(bool, bool)> {
    let plan = preparation::plan(states, repeated, None)?;
    Ok((plan.invalidate, plan.capture()))
}

pub(super) struct PreparedForward {
    started: Instant,
    states: Vec<SplitStates>,
    pub workspace_fits: bool,
}
impl PreparedForward {
    pub fn all_graphs(&self) -> bool {
        self.states.iter().all(SplitStates::all_graphs)
    }
}

fn begin_forward(peers: &Transport, layout: &Layout) -> Result<PreparedForward> {
    let started = Instant::now();
    let cached = exchange(peers, Command::GrowCached(layout.clone()))?;
    let mut states = Vec::with_capacity(cached.len());
    let mut workspace_fits = true;
    for reply in cached {
        let Reply::SplitCached(state, fits) = reply else {
            return Err("Invalid A2 cache state reply".to_string());
        };
        states.push(state);
        workspace_fits &= fits;
    }
    Ok(PreparedForward {
        started,
        states,
        workspace_fits,
    })
}

fn part_action(plan: &preparation::Plan) -> PartAction {
    PartAction {
        invalidate: plan.invalidate,
        prepare: match plan.mode {
            preparation::Mode::BuildDirect => Some(false),
            preparation::Mode::CaptureReplay => Some(true),
            preparation::Mode::Direct | preparation::Mode::Replay => None,
        },
    }
}

fn forward_with(
    peers: &Transport,
    layout: Layout,
    tokens: Vec<Vec<u32>>,
    readout: Readout,
    budget: Option<CaptureBudget>,
    maintenance: impl FnOnce(),
    graphs_enabled: bool,
) -> Result<(Vec<Output>, Timing)> {
    let prepared = begin_forward(peers, &layout)?;
    finish_forward(
        peers,
        prepared,
        layout,
        tokens,
        readout,
        budget,
        maintenance,
        graphs_enabled,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_forward(
    peers: &Transport,
    prepared: PreparedForward,
    layout: Layout,
    tokens: Vec<Vec<u32>>,
    readout: Readout,
    mut budget: Option<CaptureBudget>,
    maintenance: impl FnOnce(),
    graphs_enabled: bool,
) -> Result<(Vec<Output>, Timing)> {
    let mut maintenance = Some(maintenance);
    let PreparedForward {
        started, states, ..
    } = prepared;
    let repeated = layout.decode
        || layout.bucketed
        || layout.role == super::PacketRole::Draft
        || layout.outputs.len() > layout.segments.len()
            && layout
                .decoder_roles
                .iter()
                .all(|r| *r == super::replay::Role::Continue);
    if let Some(budget) = &mut budget {
        budget.slack_s -= started.elapsed().as_secs_f64();
    }
    let trunk_states: Vec<CacheState> = states.iter().map(|s| s.trunk).collect();
    let has_tail = states.first().is_some_and(|s| s.tail.is_some());
    let has_head = states.first().is_some_and(|s| s.head.is_some());
    if states
        .iter()
        .any(|s| s.tail.is_some() != has_tail || s.head.is_some() != has_head)
    {
        return Err("TP split packet shapes disagree across ranks".into());
    }
    let split_packet = has_tail || has_head;
    let min_warmup = if layout.bucketed || layout.role == super::PacketRole::Draft {
        1
    } else if repeated {
        2
    } else {
        8
    };
    let plan_part = |part_states: &[CacheState],
                     part_repeated: bool,
                     part_budget: Option<CaptureBudget>,
                     allow_new_capture: bool| {
        let mut plan = if graphs_enabled {
            preparation::plan_with_warmup(part_states, part_repeated, part_budget, min_warmup)?
        } else {
            preparation::direct_plan(part_states)?
        };
        if !allow_new_capture && plan.mode == preparation::Mode::CaptureReplay {
            plan.mode = if plan.invalidate {
                preparation::Mode::BuildDirect
            } else {
                preparation::Mode::Direct
            };
            plan.reason = "trunk_direct";
        }
        Ok::<_, String>(plan)
    };
    let bucket_budget = |base: Option<CaptureBudget>, capture_scale: f64| {
        base.map(|mut b| {
            if layout.bucketed || layout.role == super::PacketRole::Draft {
                if b.max_uses.is_none() {
                    b.uses = b.uses.max(24.);
                    // Context variance across mixed cohorts can occasionally record a
                    // low-context Direct observation below a high-context Replay mean.
                    // Do not strand a canonical reusable bucket in Direct forever;
                    // fall back to the bounded replay-trial rule (still honoring slack_s).
                    if b.direct_s
                        .zip(b.replay_s)
                        .is_some_and(|(direct, replay)| direct <= replay * 1.02)
                    {
                        b.replay_s = None;
                    }
                }
                if let Some(c) = &mut b.capture_s {
                    *c *= capture_scale;
                }
            }
            b
        })
    };
    // Reuse compiled Trunk Graphs across large mixed cohorts (>=512 rows).
    // Bucketed canonical shapes also capture bounded Tail and Head slices;
    // unbucketed large cohorts keep cohort-specific Tail/Head slices direct.
    let allow_tail_head_capture =
        layout.bucketed || !(has_tail && !layout.decode && layout.rows.total >= 512);
    let trunk_plan = plan_part(
        &trunk_states,
        repeated,
        bucket_budget(budget, if has_tail { 0.55 } else { 1.0 }),
        true,
    )?;
    let stagger_tail = layout.bucketed
        && trunk_plan.mode == preparation::Mode::CaptureReplay
        && budget.is_some_and(|b| {
            b.slack_s.is_finite()
                && b.capture_s.unwrap_or(0.) * 1.5 + b.replay_s.or(b.direct_s).unwrap_or(0.)
                    > b.slack_s
        });
    let tail_plan = if has_tail {
        let tail_states: Vec<CacheState> = states.iter().map(|s| s.tail.unwrap()).collect();
        let tail_scale = if trunk_plan.mode == preparation::Mode::Replay {
            0.45
        } else {
            1.0
        };
        Some(plan_part(
            &tail_states,
            repeated,
            bucket_budget(budget, tail_scale),
            allow_tail_head_capture && trunk_plan.capture() && !stagger_tail,
        )?)
    } else {
        None
    };
    let head_plan = if has_head {
        let head_states: Vec<CacheState> = states.iter().map(|s| s.head.unwrap()).collect();
        let head_scale = if trunk_plan.mode == preparation::Mode::Replay
            && tail_plan
                .as_ref()
                .is_none_or(|p| p.mode == preparation::Mode::Replay)
        {
            0.10
        } else {
            1.0
        };
        let allow_head_capture = allow_tail_head_capture
            && !stagger_tail
            && trunk_plan.capture()
            && tail_plan.as_ref().is_none_or(preparation::Plan::capture);
        Some(plan_part(
            &head_states,
            repeated,
            bucket_budget(budget, head_scale),
            allow_head_capture,
        )?)
    } else {
        None
    };
    let split_plan = SplitPlan {
        trunk: part_action(&trunk_plan),
        tail: tail_plan.as_ref().map(part_action),
        head: head_plan.as_ref().map(part_action),
    };
    let invalidate = split_plan.any_invalidate();
    let capture = trunk_plan.capture()
        && tail_plan.as_ref().is_none_or(preparation::Plan::capture)
        && head_plan.as_ref().is_none_or(preparation::Plan::capture);
    let any_captured = trunk_plan.mode == preparation::Mode::CaptureReplay
        || tail_plan
            .as_ref()
            .is_some_and(|p| p.mode == preparation::Mode::CaptureReplay)
        || head_plan
            .as_ref()
            .is_some_and(|p| p.mode == preparation::Mode::CaptureReplay);
    let cached_prepare = !invalidate && !split_plan.any_prepare();
    if budget.is_some_and(|b| b.trace) {
        super::log_async(format!(
            "A2_EXECUTION_PLAN {}",
            serde_json::json!({
                "role":format!("{:?}",layout.role), "rows":layout.rows.total,
                "mode":trunk_plan.mode,"reason":trunk_plan.reason,"invalidate":invalidate,
                "tail_mode":tail_plan.as_ref().map(|p| p.mode),
                "head_mode":head_plan.as_ref().map(|p| p.mode),
                "budget":budget,
            })
        ));
    }
    // A deferred response must not wait behind a new graph's warmup/capture.
    // Stable shapes run this callback after submission, below.
    if !cached_prepare && let Some(maintenance) = maintenance.take() {
        maintenance();
    }
    if invalidate {
        exchange(
            peers,
            Command::InvalidateParts(
                layout.clone(),
                split_plan.trunk.invalidate,
                split_plan.tail.is_some_and(|a| a.invalidate),
                split_plan.head.is_some_and(|a| a.invalidate),
            ),
        )?;
    }
    if !cached_prepare {
        peers.exchange_heavy(Command::PrepareParts(layout.clone(), split_plan))?;
    }
    let mut shared = vec![];
    let mut shared_setup = 0.;
    if graphs_enabled && !capture && super::shared::eligible(&layout) && super::shared::enabled() {
        let read = |replies: Vec<Reply>| -> Result<Vec<Vec<super::shared::State>>> {
            replies
                .into_iter()
                .map(|r| match r {
                    Reply::Shared(s) => Ok(s),
                    _ => Err("Invalid shared Graph state reply".into()),
                })
                .collect()
        };
        let states = read(exchange(peers, Command::Shared(layout.clone()))?)?;
        let slack = budget.map_or(f64::INFINITY, |b| b.slack_s - b.direct_s.unwrap_or(0.))
            - started.elapsed().as_secs_f64();
        let (ready, candidate) = super::shared::plan(&states, slack)?;
        shared = ready;
        if let Some(id) = candidate {
            if let Some(maintenance) = maintenance.take() {
                maintenance();
            }
            let capture_started = Instant::now();
            let states = read(peers.exchange_heavy(Command::CaptureShared(layout.clone(), id))?)?;
            shared_setup = capture_started.elapsed().as_secs_f64();
            shared = super::shared::plan(&states, -1.)?.0;
        }
    }
    let prepare_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let replicated = layout.role == super::PacketRole::Draft;
    let shared_mode = !shared.is_empty();
    let partial_trunk_graph = split_packet && trunk_plan.capture() && !capture;
    if shared_mode
        && (budget.is_some_and(|b| b.trace)
            || std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some())
    {
        super::log_async(format!(
            "A2_SHARED_PLAN {}",
            serde_json::json!({"rows":layout.rows.total,"pieces":shared.len(),"setup_s":shared_setup})
        ));
    }
    let sampled = !readout.draws.is_empty();
    let readout_rejection_empty = readout.rejection.is_empty();
    let parameters = Arc::clone(&readout.parameters);
    let first_input = readout.first_input.clone();
    let command = Command::Forward(layout, tokens, readout, shared);
    peers.send(command)?;
    if let Some(maintenance) = maintenance {
        maintenance();
    }
    let results = peers.collect()?;
    collect_forward_outputs(
        results,
        replicated,
        shared_mode,
        sampled,
        readout_rejection_empty,
        parameters,
        first_input,
        started,
        prepare_s,
        capture,
        any_captured,
        invalidate,
        partial_trunk_graph,
        shared_setup,
        if cached_prepare {
            shared_setup
        } else {
            prepare_s
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_forward_outputs(
    results: Vec<Reply>,
    replicated: bool,
    shared_mode: bool,
    sampled: bool,
    readout_rejection_empty: bool,
    parameters: Arc<
        crate::sampling::Prepared<
            BTreeMap<usize, Vec<crate::backend::ascend::sampling::Parameters>>,
        >,
    >,
    first_input: Option<Arc<super::host_inputs::First>>,
    started: Instant,
    prepare_s: f64,
    capture: bool,
    any_captured: bool,
    invalidate: bool,
    partial_trunk_graph: bool,
    shared_setup: f64,
    setup: f64,
) -> Result<(Vec<Output>, Timing)> {
    if std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some()
        && let Some(first) = first_input
        && first.prepared.seconds() > 0.
        && let Ok(data) = first.prepared.get()
    {
        super::log_async(format!(
            "A2_HOST_INPUT {}",
            serde_json::json!({
                "rows": data.rows, "reused_rows": data.reused_rows,
                "forecast_wait_s": data.wait_s,
                "prepare_s": first.prepared.seconds(),
            })
        ));
    }
    let ranks = results
        .into_iter()
        .map(|reply| match reply {
            Reply::Outputs(outputs, actual_shared) if actual_shared == shared_mode => Ok(outputs),
            _ => Err("Invalid A2 output reply".into()),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = ranks.first().cloned().ok_or("Missing A2 TP output")?;
    if ranks.iter().any(|rank| rank.len() != outputs.len()) {
        return Err("A2 TP output count differs".into());
    }
    let mut owner_positions = BTreeMap::<usize, usize>::new();
    let prepared_parameters = parameters.get()?;
    for (i, output) in outputs.iter_mut().enumerate() {
        output.logits.clear();
        output.error = None;
        let position = owner_positions.entry(output.owner).or_default();
        let row_error = prepared_parameters
            .get(&output.owner)
            .and_then(|rows| rows.get(*position))
            .and_then(|p| p.error.clone());
        *position += 1;
        for rank in &ranks {
            let value = rank.get(i).ok_or("A2 TP output count differs")?;
            if value.owner != output.owner {
                return Err("A2 TP output owner differs".into());
            }
            if replicated && value.token != output.token {
                return Err("DSpark TP proposal pools disagree".into());
            }
            if value.value > output.value
                || value.value == output.value && value.token < output.token
            {
                output.token = value.token;
                output.value = value.value;
                output.summary.clone_from(&value.summary);
            }
            output.logits.extend_from_slice(&value.logits);
        }
        if let Some(error) = row_error {
            output.error = Some(error);
            output.summary = None;
            continue;
        }
        let summaries = ranks
            .iter()
            .filter_map(|r| r[i].summary.as_ref())
            .collect::<Vec<_>>();
        if !summaries.is_empty() {
            if summaries.len() != ranks.len() {
                return Err("TP probability summaries differ".into());
            }
            match crate::backend::ascend::sampling::report(
                &summaries,
                output
                    .summary
                    .as_ref()
                    .ok_or("Missing chosen sampling summary")?,
                output.token,
            ) {
                Ok(report) => output.report = Some(report),
                Err(error) if error == "Grammar permits no tokens" => {
                    output.error = Some(error);
                    output.summary = None;
                    continue;
                }
                Err(error) => return Err(error),
            }
        }
        if sampled && !output.value.is_finite() {
            output.error = Some("Grammar permits no tokens".into());
        } else if !readout_rejection_empty && output.value == f32::MIN {
            output.error = Some("Rejected proposal has no residual mass".into());
        }
        output.summary = None;
    }
    Ok((
        outputs,
        Timing {
            mask_rows: parameters
                .get()?
                .values()
                .flatten()
                .filter(|p| p.mask.is_some())
                .count(),
            report_rows: parameters
                .get()?
                .values()
                .flatten()
                .filter(|p| p.top_logprobs.is_some())
                .count(),
            mask_prepare_s: parameters.seconds(),
            captured: any_captured,
            invalidated: invalidate,
            seconds: started.elapsed().as_secs_f64(),
            preparation: prepare_s,
            replay: capture,
            shared: shared_mode || partial_trunk_graph,
            shared_setup: if partial_trunk_graph && any_captured {
                prepare_s
            } else {
                shared_setup
            },
            setup,
        },
    ))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod state_tests;
