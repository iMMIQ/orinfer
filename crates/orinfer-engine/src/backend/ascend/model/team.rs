//! Thread-affine TP workers shared by benchmarks and continuous generation.
use super::super::{
    hccl::RootInfo,
    runtime::{Limits, Session},
};
use super::{
    Model, SharedTables, Worker, affinity,
    batch::{CacheState, Layout},
    load_spec,
};
use crate::{artifact::Result, error::EngineError};
mod channel;
mod host_release;
use channel::{Peer, Transport, reply_channel};
use std::{
    collections::BTreeMap,
    path::Path,
    sync::{Arc, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[derive(Clone)]
pub(super) enum Command {
    Admit(Vec<usize>),
    Reset(Vec<usize>),
    Release(Option<usize>),
    Cached(Layout),
    Invalidate(Layout),
    Prepare(Layout, bool),
    Forward(Layout, Vec<Vec<u32>>, bool),
    Save {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    Restore {
        owner: usize,
        snapshot: u64,
    },
    Evict(u64),
    Budget(super::memory::Requirement),
    Reclaim,
    PlanSnapshot {
        owner: usize,
        snapshot: u64,
        complete: bool,
    },
    RoutingGroups(bool),
    Stop,
}
pub(super) enum Reply {
    Ready,
    Cached(CacheState),
    Outputs(Vec<Output>),
    Allocations(BTreeMap<u64, usize>),
    RoutingGroups(Vec<RoutingGroup>),
    Capacity(bool),
}
#[derive(serde::Serialize)]
pub(super) struct RoutingGroup {
    pub buffer: String,
    pub counts: Vec<i64>,
}
#[derive(Clone)]
pub(super) struct Output {
    pub owner: usize,
    pub token: u32,
    pub value: f32,
    pub logits: Vec<f32>,
}
pub(super) struct Team {
    peers: Transport,
    handles: Vec<JoinHandle<()>>,
    pub model: Arc<Model>,
    pub load_s: f64,
    pub rows: usize,
    pub owner_allocation: super::memory::Allocation,
    // Drop worker contexts and communication groups before the bootstrap context.
    _bootstrap: Option<Session>,
    host_tables: Option<Arc<SharedTables>>,
}
impl Team {
    pub fn load(plan: &Path, model_dir: &Path) -> Result<Self> {
        let model_dir = model_dir
            .canonicalize()
            .map_err(|error| error.to_string())?;
        let (mut model, package, layout) = load_spec(plan, &model_dir)?;
        if model.batch.is_none() {
            return Err("A2 service requires a native rows plan".into());
        }
        let private = &model.batch.as_ref().unwrap().private;
        let owner_allocation = model.ranks.iter().try_fold(
            super::memory::Allocation {
                bytes: 0,
                resident: 0,
            },
            |maximum, rank| -> Result<_> {
                let (_, current) = super::memory::owner_layout(
                    private
                        .iter()
                        .filter_map(|name| rank.buffers.iter().find(|b| &b.name == name))
                        .filter(|b| b.view.is_none())
                        .map(|b| (b.name.as_str(), b.bytes, 256)),
                )?;
                Ok(super::memory::Allocation {
                    bytes: maximum.bytes.max(current.bytes),
                    resident: maximum.resident.max(current.resident),
                })
            },
        )?;
        let rows = model.ranks[0]
            .packets
            .iter()
            .filter(|p| !p.batch_decode)
            .map(|p| p.rows)
            .max()
            .ok_or("Missing A2 prefill rows")?;
        let timeout = channel::command_timeout(
            std::env::var("ORINFER_A2_COMMAND_TIMEOUT_MS")
                .ok()
                .as_deref(),
        )?;
        let ranks = std::mem::take(&mut model.ranks);
        let cpus = affinity::from_env(ranks.len())?;
        let started = Instant::now();
        let bootstrap = Session::new(0, Limits::default())?;
        let root = RootInfo::new(&bootstrap, ranks.len() as u32)?;
        let shared = Arc::new(SharedTables::new());
        let model = Arc::new(model);
        let mut team = Self {
            peers: Transport::new(timeout),
            handles: vec![],
            model: Arc::clone(&model),
            load_s: 0.,
            rows,
            owner_allocation,
            _bootstrap: Some(bootstrap),
            host_tables: Some(shared),
        };
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
            let handle = std::thread::Builder::new()
                .name(format!("orinfer-a2-{index}"))
                .spawn(move || {
                    let result = (|| {
                        if let Some(cpus) = cpus.as_deref() {
                            affinity::apply(cpus)?;
                        }
                        Worker::load(
                            Session::new(
                                index,
                                Limits {
                                    device_bytes: 58 << 30,
                                    pinned_bytes: 64 << 20,
                                    timeout_ms: timeout.as_millis() as i32,
                                },
                            )?,
                            &root,
                            rank,
                            &model,
                            &model_dir,
                            &package,
                            layout,
                            false,
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
                    while let Ok(command) = commands.recv() {
                        let result = match command {
                            Command::Stop => break,
                            Command::Budget(requirement) => worker.session.memory_info().map(|m| {
                                Reply::Capacity(
                                    requirement.fits(m.0, worker.session.remaining_device_bytes()),
                                )
                            }),
                            Command::Reclaim => worker.evict_batch_program().map(Reply::Capacity),
                            Command::PlanSnapshot {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent = lineage.get(&owner).and_then(|id| snapshots.get(id));
                                worker
                                    .plan_owner_snapshot(owner, complete, snapshot, parent)
                                    .map(Reply::Allocations)
                            }
                            Command::Admit(ids) => worker.admit(&ids).map(|_| Reply::Ready),
                            Command::Reset(ids) => ids
                                .into_iter()
                                .try_for_each(|id| {
                                    lineage.remove(&id);
                                    worker.reset_owner(id)
                                })
                                .map(|_| Reply::Ready),
                            Command::Release(None) => {
                                lineage.clear();
                                worker.release_owners().map(|_| Reply::Ready)
                            }
                            Command::Release(Some(owner)) => {
                                lineage.remove(&owner);
                                worker.release_owner(owner).map(|_| Reply::Ready)
                            }
                            Command::Cached(layout) => {
                                worker.batch_cached(&layout).map(Reply::Cached)
                            }
                            Command::Invalidate(layout) => {
                                worker.invalidate_batch(&layout).map(|_| Reply::Ready)
                            }
                            Command::Prepare(layout, capture) => {
                                worker.prepare_batch(&layout, capture).map(|_| Reply::Ready)
                            }
                            Command::Forward(layout, tokens, full) => {
                                worker.forward_batch(&layout, &tokens, full).map(|values| {
                                    Reply::Outputs(
                                        values
                                            .into_iter()
                                            .map(|(owner, best, logits)| Output {
                                                owner,
                                                token: best.token,
                                                value: best.value,
                                                logits,
                                            })
                                            .collect(),
                                    )
                                })
                            }
                            Command::Save {
                                owner,
                                snapshot,
                                complete,
                            } => {
                                let parent = lineage.get(&owner).and_then(|id| snapshots.get(id));
                                worker
                                    .save_owner(owner, complete, snapshot, parent)
                                    .map(|saved| {
                                        let allocations = saved.allocations();
                                        snapshots.insert(snapshot, saved);
                                        lineage.insert(owner, snapshot);
                                        Reply::Allocations(allocations)
                                    })
                            }
                            Command::Restore { owner, snapshot } => snapshots
                                .get(&snapshot)
                                .ok_or_else(|| "Missing A2 prefix snapshot".into())
                                .and_then(|saved| worker.restore_owner(owner, saved))
                                .map(|_| {
                                    lineage.insert(owner, snapshot);
                                    Reply::Ready
                                }),
                            Command::Evict(snapshot) => {
                                snapshots.remove(&snapshot);
                                lineage.retain(|_, id| *id != snapshot);
                                Ok(Reply::Ready)
                            }
                            Command::RoutingGroups(decode) => {
                                worker.routing_groups(decode).map(Reply::RoutingGroups)
                            }
                        };
                        let result = result.map_err(|error| EngineError::take(error, true));
                        let failed = result.is_err();
                        if let Err(error) = &result {
                            eprintln!("A2 rank {index}: {error}");
                        }
                        if reply.send(result).is_err() || failed {
                            break;
                        }
                    }
                    let _ = worker.session.synchronize();
                })
                .map_err(|error| error.to_string())?;
            team.handles.push(handle);
            team.peers.push(Peer { send, receive });
        }
        let ready = team.peers.receive(Duration::from_secs(1800))?;
        if ready.iter().any(|reply| !matches!(reply, Reply::Ready)) {
            return Err("Invalid A2 startup reply".into());
        }
        team.load_s = started.elapsed().as_secs_f64();
        Ok(team)
    }
    pub fn exchange(&self, command: Command) -> Result<Vec<Reply>> {
        exchange(&self.peers, command)
    }
    pub fn forward(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        full: bool,
    ) -> Result<(Vec<Output>, f64, f64)> {
        forward(&self.peers, layout, tokens, full)
    }
    pub fn forward_with(
        &self,
        layout: Layout,
        tokens: Vec<Vec<u32>>,
        full: bool,
        maintenance: impl FnOnce(),
    ) -> Result<(Vec<Output>, f64, f64)> {
        forward_with(&self.peers, layout, tokens, full, maintenance)
    }

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
    pub fn plan_snapshot_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
    ) -> Result<BTreeMap<u64, usize>> {
        self.allocations(Command::PlanSnapshot {
            owner,
            snapshot,
            complete,
        })
    }
    pub fn has_budget(&self, requirement: super::memory::Requirement) -> Result<bool> {
        Ok(self
            .capacities(Command::Budget(requirement))?
            .into_iter()
            .all(|v| v))
    }
    pub fn reclaim_programs(&self) -> Result<bool> {
        Ok(self.capacities(Command::Reclaim)?.into_iter().any(|v| v))
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
        for reply in self.exchange(command)? {
            let Reply::Allocations(allocations) = reply else {
                return Err("Invalid A2 snapshot allocation reply".into());
            };
            if let Some(combined) = &mut result {
                if !combined.keys().eq(allocations.keys()) {
                    return Err("TP snapshot allocation lineage differs".into());
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
            .filter(|p| p.batch_decode == decode)
        {
            for step in &packet.body {
                if let super::Instruction::W4 {
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
        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                eprintln!("A2 worker panicked during teardown");
            }
        }
        // CPU table reclamation owns no ACL/HCCL state and overlaps both rank
        // teardown and the final bootstrap context. Join before releasing the
        // model owner's GPU lock or returning from destruction.
        drop(self._bootstrap.take());
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

fn forward(
    peers: &Transport,
    layout: Layout,
    tokens: Vec<Vec<u32>>,
    full: bool,
) -> Result<(Vec<Output>, f64, f64)> {
    forward_with(peers, layout, tokens, full, || {})
}
/// Every rank changes execution mode together. Any miss resets the shared
/// class to direct. Successful uses are counted on every rank; promotion waits
/// for the slowest rank. Offline TP8 measurements put prefill amortization at
/// 8–20 uses (0.31–0.46 s capture versus 24–41 ms saved per execution).
fn preparation(states: &[CacheState], decode: bool) -> Result<(bool, bool)> {
    if states.is_empty() {
        return Err("Missing A2 TP cache states".into());
    }
    if states.contains(&CacheState::Missing) {
        return Ok((true, false));
    }
    let graphs = states.iter().filter(|s| **s == CacheState::Graph).count();
    if graphs > 0 {
        return Ok((graphs != states.len(), true));
    }
    let required = if decode { 1 } else { 8 };
    Ok((
        false,
        states
            .iter()
            .all(|s| matches!(s, CacheState::Direct(n) if *n >= required)),
    ))
}

fn forward_with(
    peers: &Transport,
    layout: Layout,
    tokens: Vec<Vec<u32>>,
    full: bool,
    maintenance: impl FnOnce(),
) -> Result<(Vec<Output>, f64, f64)> {
    let started = Instant::now();
    let cached = exchange(peers, Command::Cached(layout.clone()))?;
    let states = cached
        .into_iter()
        .map(|reply| match reply {
            Reply::Cached(state) => Ok(state),
            _ => Err("Invalid A2 cache state reply".to_string()),
        })
        .collect::<Result<Vec<_>>>()?;
    let (invalidate, capture) = preparation(&states, layout.decode)?;
    if invalidate {
        exchange(peers, Command::Invalidate(layout.clone()))?;
    }
    exchange(peers, Command::Prepare(layout.clone(), capture))?;
    let prepare_s = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let command = Command::Forward(layout, tokens, full);
    peers.send(command)?;
    maintenance();
    let results = peers.collect()?;
    let ranks = results
        .into_iter()
        .map(|reply| match reply {
            Reply::Outputs(outputs) => Ok(outputs),
            _ => Err("Invalid A2 output reply".into()),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = ranks.first().cloned().ok_or("Missing A2 TP output")?;
    if ranks.iter().any(|rank| rank.len() != outputs.len()) {
        return Err("A2 TP output count differs".into());
    }
    for (i, output) in outputs.iter_mut().enumerate() {
        output.logits.clear();
        for rank in &ranks {
            let value = rank.get(i).ok_or("A2 TP output count differs")?;
            if value.owner != output.owner {
                return Err("A2 TP output owner differs".into());
            }
            if value.value > output.value
                || value.value == output.value && value.token < output.token
            {
                output.token = value.token;
                output.value = value.value;
            }
            output.logits.extend_from_slice(&value.logits);
        }
    }
    Ok((outputs, started.elapsed().as_secs_f64(), prepare_s))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod state_tests;
