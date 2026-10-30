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
use crate::artifact::Result;
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
    Memory,
    RoutingGroups(bool),
    Stop,
}
pub(super) enum Reply {
    Ready,
    Cached(CacheState),
    Outputs(Vec<Output>),
    Bytes(usize),
    Allocations(BTreeMap<u64, usize>),
    RoutingGroups(Vec<RoutingGroup>),
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
struct Peer {
    send: mpsc::Sender<Command>,
    receive: mpsc::Receiver<Result<Reply>>,
}
pub(super) struct Team {
    peers: Vec<Peer>,
    handles: Vec<JoinHandle<()>>,
    pub model: Arc<Model>,
    pub load_s: f64,
    pub rows: usize,
    pub owner_bytes: usize,
    // Drop worker contexts and communication groups before the bootstrap context.
    _bootstrap: Session,
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
        let owner_bytes = model.ranks[0]
            .buffers
            .iter()
            .filter(|b| private.contains(&b.name) && b.view.is_none())
            .map(|b| b.bytes)
            .sum();
        let rows = model.ranks[0]
            .packets
            .iter()
            .filter(|p| !p.batch_decode)
            .map(|p| p.rows)
            .max()
            .ok_or("Missing A2 prefill rows")?;
        let ranks = std::mem::take(&mut model.ranks);
        let cpus = affinity::from_env(ranks.len())?;
        let started = Instant::now();
        let bootstrap = Session::new(0, Limits::default())?;
        let root = RootInfo::new(&bootstrap, ranks.len() as u32)?;
        let shared = Arc::new(SharedTables::new());
        let model = Arc::new(model);
        let mut team = Self {
            peers: vec![],
            handles: vec![],
            model: Arc::clone(&model),
            load_s: 0.,
            rows,
            owner_bytes,
            _bootstrap: bootstrap,
        };
        for rank in ranks {
            let cpus = cpus.as_ref().map(|v| v[rank.rank as usize].clone());
            let root = root.clone();
            let shared = Arc::clone(&shared);
            let model = Arc::clone(&model);
            let package = package.clone();
            let layout = layout.clone();
            let model_dir = model_dir.to_owned();
            let (send, commands) = mpsc::channel();
            let (reply, receive) = mpsc::channel();
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
                                    timeout_ms: 30000,
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
                    let mut worker = match result {
                        Ok(worker) => worker,
                        Err(error) => {
                            let _ = reply.send(Err(error));
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
                            Command::Memory => worker
                                .session
                                .memory_info()
                                .map(|(free, _)| Reply::Bytes(free)),
                            Command::RoutingGroups(decode) => {
                                worker.routing_groups(decode).map(Reply::RoutingGroups)
                            }
                        };
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
        for peer in &team.peers {
            peer.receive
                .recv_timeout(Duration::from_secs(1800))
                .map_err(|error| format!("A2 load: {error}"))??;
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
    pub fn minimum_bytes(&self, command: Command) -> Result<usize> {
        self.bytes(command)
    }
    pub fn snapshot_allocations(
        &self,
        owner: usize,
        snapshot: u64,
        complete: bool,
    ) -> Result<BTreeMap<u64, usize>> {
        let mut result: Option<BTreeMap<u64, usize>> = None;
        for reply in self.exchange(Command::Save {
            owner,
            snapshot,
            complete,
        })? {
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
    fn bytes(&self, command: Command) -> Result<usize> {
        self.exchange(command)?
            .into_iter()
            .map(|reply| match reply {
                Reply::Bytes(bytes) => Ok(bytes),
                _ => Err("Invalid A2 byte reply".into()),
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .reduce(usize::min)
            .ok_or_else(|| "Empty A2 team".into())
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
        for peer in &self.peers {
            let _ = peer.send.send(Command::Stop);
        }
        for handle in self.handles.drain(..) {
            if handle.join().is_err() {
                eprintln!("A2 worker panicked during teardown");
            }
        }
    }
}

fn exchange(peers: &[Peer], command: Command) -> Result<Vec<Reply>> {
    for peer in peers {
        peer.send
            .send(command.clone())
            .map_err(|error| error.to_string())?;
    }
    let results = peers
        .iter()
        .map(|peer| {
            peer.receive
                .recv_timeout(Duration::from_secs(1800))
                .map_err(|error| format!("A2 worker: {error}"))?
        })
        .collect::<Vec<_>>();
    results.into_iter().collect()
}

fn forward(
    peers: &[Peer],
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
    peers: &[Peer],
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
    for peer in peers {
        peer.send
            .send(command.clone())
            .map_err(|error| error.to_string())?;
    }
    maintenance();
    let results = peers
        .iter()
        .map(|peer| {
            peer.receive
                .recv_timeout(Duration::from_secs(1800))
                .map_err(|error| format!("A2 forward: {error}"))?
        })
        .collect::<Result<Vec<_>>>()?;
    let ranks = results
        .into_iter()
        .map(|reply| match reply {
            Reply::Outputs(outputs) => Ok(outputs),
            _ => Err("Invalid A2 output reply".into()),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = ranks.first().cloned().ok_or("Missing A2 TP output")?;
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
mod tests {
    use super::*;
    #[test]
    fn promotion_waits_for_successful_uses_on_every_rank() {
        assert!(preparation(&[], false).is_err());
        for decode in [false, true] {
            assert_eq!(
                preparation(&[CacheState::Missing, CacheState::Graph], decode).unwrap(),
                (true, false)
            );
            assert_eq!(
                preparation(&[CacheState::Graph, CacheState::Direct(0)], decode).unwrap(),
                (true, true)
            );
            assert_eq!(
                preparation(&[CacheState::Graph; 2], decode).unwrap(),
                (false, true)
            );
        }
        assert_eq!(
            preparation(&[CacheState::Direct(0), CacheState::Direct(1)], true).unwrap(),
            (false, false)
        );
        assert_eq!(
            preparation(&[CacheState::Direct(1); 2], true).unwrap(),
            (false, true)
        );
        assert_eq!(
            preparation(&[CacheState::Direct(7), CacheState::Direct(8)], false).unwrap(),
            (false, false)
        );
        assert_eq!(
            preparation(&[CacheState::Direct(8), CacheState::Direct(9)], false).unwrap(),
            (false, true)
        );
    }
    #[test]
    fn collective_capture_is_coordinated_after_an_asymmetric_cache_eviction() {
        for statuses in [
            [CacheState::Graph, CacheState::Graph],
            [CacheState::Direct(1), CacheState::Direct(1)],
            [CacheState::Direct(7), CacheState::Direct(8)],
            [CacheState::Direct(8), CacheState::Direct(9)],
            [CacheState::Graph, CacheState::Missing],
            [CacheState::Direct(1), CacheState::Graph],
            [CacheState::Missing, CacheState::Missing],
        ] {
            std::thread::scope(|scope| {
                let mut peers = vec![];
                let mut maintained = vec![];
                let (recapture, capture) = preparation(&statuses, false).unwrap();
                for hit in statuses {
                    let (send, commands) = mpsc::channel();
                    let (reply, receive) = mpsc::channel();
                    let (maintenance, completed) = mpsc::channel();
                    maintained.push(maintenance);
                    peers.push(Peer { send, receive });
                    scope.spawn(move || {
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Cached(_)
                        ));
                        reply.send(Ok(Reply::Cached(hit))).unwrap();
                        if recapture {
                            assert!(matches!(
                                commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                                Command::Invalidate(_)
                            ));
                            reply.send(Ok(Reply::Ready)).unwrap();
                        }
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Prepare(_, mode) if mode == capture
                        ));
                        reply.send(Ok(Reply::Ready)).unwrap();
                        assert!(matches!(
                            commands.recv_timeout(Duration::from_secs(5)).unwrap(),
                            Command::Forward(_, _, false)
                        ));
                        // Workers cannot complete until host maintenance has run.
                        completed.recv_timeout(Duration::from_secs(5)).unwrap();
                        reply.send(Ok(Reply::Outputs(vec![]))).unwrap();
                    });
                }
                let layout = Layout::new(
                    vec![orinfer_model_sdk::architecture::BatchSegment {
                        slot: 17,
                        tokens: 2,
                    }],
                    vec![],
                    false,
                    8,
                    2,
                )
                .unwrap();
                assert!(
                    forward_with(&peers, layout, vec![vec![0, 1]], false, || {
                        for maintenance in maintained {
                            maintenance.send(()).unwrap();
                        }
                    })
                    .unwrap()
                    .0
                    .is_empty()
                );
            });
        }
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;
    use orinfer_model_sdk::architecture::BatchSegment;
    #[test]
    #[ignore = "Offline preparation-stage probe; needs the real A2 package"]
    fn measure_packet_preparation_stages() {
        use super::super::replay::Role;
        use std::os::fd::AsRawFd;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: The owned descriptor outlives all native resources.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let directory =
            std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
        let descriptor: serde_json::Value =
            crate::model::read(&directory.join("cache/model.json")).unwrap();
        let plan = directory
            .join("cache/packages")
            .join(descriptor["execution_package"].as_str().unwrap())
            .join("model.json");
        let team = Team::load(&plan, &directory).unwrap();
        team.exchange(Command::Admit(vec![7, 17])).unwrap();
        for (lengths, roles) in [
            (vec![128], vec![Role::Final]),
            (vec![2048], vec![Role::Final]),
            (vec![1, 799], vec![Role::Continue, Role::Producer]),
            (vec![128], vec![Role::Final]),
        ] {
            let mut l = Layout::new(
                lengths
                    .iter()
                    .zip([7, 17])
                    .map(|(&tokens, slot)| BatchSegment { tokens, slot })
                    .collect(),
                roles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                    .collect(),
                false,
                team.rows,
                32,
            )
            .unwrap();
            l.decoder_roles = roles;
            team.exchange(Command::Invalidate(l.clone())).unwrap();
            team.exchange(Command::Prepare(l, false)).unwrap();
        }
        drop(team);
        drop(lock);
    }
    #[test]
    #[ignore = "Offline release-mode TP8 timing probe; needs the real A2 package"]
    // Reject only an explicitly selected debug probe, not the debug build.
    #[allow(clippy::assertions_on_constants)]
    fn measure_bound_direct_and_graph_packet_costs() {
        use super::super::replay::Role;
        use std::os::fd::AsRawFd;
        assert!(
            !cfg!(debug_assertions),
            "Measure the optimized current-build test executable"
        );
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: This owned descriptor outlives every native object below.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        {
            let session = Session::new(0, Limits::default()).unwrap();
            for count in [1, 20, 40] {
                let mut created = vec![];
                let mut destroyed = vec![];
                for _ in 0..5 {
                    let start = Instant::now();
                    let branches = super::super::super::graph::Branches::new(
                        &session,
                        &vec![0; count],
                        Box::new(()),
                    )
                    .unwrap();
                    created.push(start.elapsed().as_secs_f64());
                    let start = Instant::now();
                    drop(branches);
                    destroyed.push(start.elapsed().as_secs_f64());
                }
                eprintln!(
                    "A2 branch probe {}",
                    serde_json::json!({"forks":count,
                    "create_s":created,"destroy_s":destroyed})
                );
            }
        }
        let directory =
            std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
        let descriptor: serde_json::Value =
            crate::model::read(&directory.join("cache/model.json")).unwrap();
        let plan = directory
            .join("cache/packages")
            .join(descriptor["execution_package"].as_str().unwrap())
            .join("model.json");
        let team = Team::load(&plan, &directory).unwrap();
        team.exchange(Command::Admit(vec![7, 17])).unwrap();
        let layout = |lengths: &[usize], slots: &[usize], roles: &[Role], decode| {
            let mut l = Layout::new(
                lengths
                    .iter()
                    .zip(slots)
                    .map(|(&tokens, &slot)| BatchSegment { tokens, slot })
                    .collect(),
                roles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                    .collect(),
                decode,
                if decode { 32 } else { team.rows },
                32,
            )
            .unwrap();
            l.decoder_roles = roles.to_vec();
            l
        };
        let tokens = |l: &Layout| {
            l.segments
                .iter()
                .map(|s| {
                    (0..s.tokens)
                        .map(|i| ((i * 73 + s.slot * 19 + 11) % 129280) as u32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let prefix = layout(&[128], &[7], &[Role::Final], false);
        team.forward(prefix.clone(), tokens(&prefix), true).unwrap();
        team.snapshot_allocations(7, 1, true).unwrap();
        for (name, l) in [
            ("decode1", layout(&[1], &[7], &[Role::Continue], true)),
            ("final128", layout(&[128], &[7], &[Role::Final], false)),
            (
                "mixed800",
                layout(
                    &[1, 799],
                    &[7, 17],
                    &[Role::Continue, Role::Producer],
                    false,
                ),
            ),
            ("prefill2048", layout(&[2048], &[7], &[Role::Final], false)),
        ] {
            team.exchange(Command::Invalidate(l.clone())).unwrap();
            team.exchange(Command::Prepare(l.clone(), false)).unwrap();
            let mut timings = vec![];
            let mut promotion_s = 0.;
            for capture in [false, true] {
                if capture {
                    let start = Instant::now();
                    team.exchange(Command::Prepare(l.clone(), true)).unwrap();
                    promotion_s = start.elapsed().as_secs_f64();
                }
                let mut times = vec![];
                for _ in 0..5 {
                    for owner in [7, 17] {
                        team.exchange(Command::Restore { owner, snapshot: 1 })
                            .unwrap();
                    }
                    let start = Instant::now();
                    let result = team
                        .exchange(Command::Forward(l.clone(), tokens(&l), false))
                        .unwrap();
                    assert!(result.iter().all(|r| matches!(r, Reply::Outputs(_))));
                    times.push(start.elapsed().as_secs_f64());
                }
                timings.push(times);
            }
            eprintln!(
                "A2 packet mode probe {}",
                serde_json::json!({"name":name,
                "direct_s":timings[0],"graph_s":timings[1],"promotion_s":promotion_s})
            );
        }
        team.exchange(Command::Evict(1)).unwrap();
        drop(team);
        drop(lock);
    }
    #[test]
    #[ignore = "Requires the native TP8 A2 package and eight NPUs"]
    fn graph_launch_identity_reuses_changed_lengths_and_decoder_roles() {
        use super::super::replay::Role;
        use std::os::fd::AsRawFd;
        assert_eq!(std::env::var("HCCL_DETERMINISTIC").unwrap(), "strict");
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: The descriptor outlives all native workers and device resources.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let directory =
            std::path::PathBuf::from(std::env::var_os("ORINFER_A2_STATE_MODEL").unwrap());
        let descriptor: serde_json::Value =
            crate::model::read(&directory.join("cache/model.json")).unwrap();
        let plan = directory
            .join("cache/packages")
            .join(descriptor["execution_package"].as_str().unwrap())
            .join("model.json");
        let team = Team::load(&plan, &directory).unwrap();
        team.exchange(Command::Admit(vec![7, 17])).unwrap();
        let layout = |lengths: &[usize], slots: &[usize], roles: &[Role]| {
            let mut layout = Layout::new(
                lengths
                    .iter()
                    .zip(slots)
                    .map(|(&tokens, &slot)| BatchSegment { slot, tokens })
                    .collect(),
                roles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (*r != Role::Producer).then_some(i))
                    .collect(),
                false,
                team.rows,
                32,
            )
            .unwrap();
            layout.decoder_roles = roles.to_vec();
            layout
        };
        let tokens = |l: &Layout| {
            l.segments
                .iter()
                .map(|s| {
                    (0..s.tokens)
                        .map(|i| ((i * 73 + s.slot * 19 + 11) % 129280) as u32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let same = |a: &[Output], b: &[Output]| {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.owner, b.owner);
                assert_eq!(a.logits, b.logits);
            }
        };
        // These layouts have identical local launches on some ranks and
        // different launches on others. Every rank must keep them distinct.
        let partial_a = layout(&[1, 15], &[7, 17], &[Role::Final; 2]);
        let partial_b = layout(&[15, 1], &[7, 17], &[Role::Final; 2]);
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.forward(partial_a.clone(), tokens(&partial_a), true)
            .unwrap();
        assert!(
            team.exchange(Command::Cached(partial_a.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Direct(1))))
        );
        assert!(
            team.exchange(Command::Cached(partial_b.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Missing))),
            "Partial rank aliases must never share a global cache class"
        );
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.forward(partial_b.clone(), tokens(&partial_b), true)
            .unwrap();
        assert!(
            team.exchange(Command::Cached(partial_a.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph)))
        );
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        for expected in 2..=8 {
            team.forward(partial_a.clone(), tokens(&partial_a), false)
                .unwrap();
            assert!(
                team.exchange(Command::Cached(partial_a.clone()))
                    .unwrap()
                    .iter()
                    .all(|r| matches!(r, Reply::Cached(CacheState::Direct(n)) if *n == expected))
            );
            team.exchange(Command::Reset(vec![7, 17])).unwrap();
        }
        let reused = team
            .forward(partial_a.clone(), tokens(&partial_a), true)
            .unwrap()
            .0;
        assert!(
            team.exchange(Command::Cached(partial_a.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Graph)))
        );
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        team.exchange(Command::Invalidate(partial_a.clone()))
            .unwrap();
        let fresh = team
            .forward(partial_a.clone(), tokens(&partial_a), true)
            .unwrap()
            .0;
        same(&reused, &fresh);
        eprintln!(
            "A2 graph partial-rank aliases remain distinct; alternating graph cache hits equal fresh direct execution"
        );
        for (name, a, b, roles) in [
            ("short", [33, 95], [35, 93], [Role::Final; 2]),
            (
                "partitioned producer",
                [769, 1279],
                [785, 1263],
                [Role::Producer; 2],
            ),
        ] {
            let a = layout(&a, &[7, 17], &roles);
            let b = layout(&b, &[7, 17], &roles);
            let tail = layout(&[128, 128], &[7, 17], &[Role::Final; 2]);
            team.exchange(Command::Reset(vec![7, 17])).unwrap();
            team.forward(a.clone(), tokens(&a), true).unwrap();
            assert!(
                team.exchange(Command::Cached(b.clone()))
                    .unwrap()
                    .iter()
                    .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph))),
                "{name} must reuse the captured launches"
            );
            team.exchange(Command::Reset(vec![7, 17])).unwrap();
            team.exchange(Command::Prepare(b.clone(), true)).unwrap();
            let mut reused = team.forward(b.clone(), tokens(&b), true).unwrap().0;
            if roles[0] == Role::Producer {
                reused = team.forward(tail.clone(), tokens(&tail), true).unwrap().0;
            }
            team.exchange(Command::Reset(vec![7, 17])).unwrap();
            team.exchange(Command::Invalidate(b.clone())).unwrap();
            let mut fresh = team.forward(b.clone(), tokens(&b), true).unwrap().0;
            if roles[0] == Role::Producer {
                fresh = team.forward(tail.clone(), tokens(&tail), true).unwrap().0;
            }
            same(&reused, &fresh);
            eprintln!(
                "A2 graph {name}: changed length reuse equals a fresh direct execution, complete logits"
            );
        }
        team.exchange(Command::Reset(vec![7, 17])).unwrap();
        let prefix = layout(&[128], &[7], &[Role::Final]);
        team.forward(prefix.clone(), tokens(&prefix), true).unwrap();
        team.snapshot_allocations(7, 1, true).unwrap();
        let restore = || {
            for owner in [7, 17] {
                team.exchange(Command::Restore { owner, snapshot: 1 })
                    .unwrap();
            }
        };
        let a = layout(&[1, 128], &[7, 17], &[Role::Continue, Role::Final]);
        let b = layout(&[128, 1], &[17, 7], &[Role::Final, Role::Continue]);
        restore();
        team.forward(a.clone(), tokens(&a), true).unwrap();
        assert!(
            team.exchange(Command::Cached(b.clone()))
                .unwrap()
                .iter()
                .all(|r| matches!(r, Reply::Cached(CacheState::Direct(_) | CacheState::Graph)))
        );
        restore();
        team.exchange(Command::Prepare(b.clone(), true)).unwrap();
        let reused = team.forward(b.clone(), tokens(&b), true).unwrap().0;
        restore();
        team.exchange(Command::Invalidate(b.clone())).unwrap();
        let fresh = team.forward(b.clone(), tokens(&b), true).unwrap().0;
        same(&reused, &fresh);
        restore();
        let mut changed = tokens(&b);
        changed[0].fill(999);
        let changed = team.forward(b, changed, true).unwrap().0;
        assert!(
            changed
                .iter()
                .zip(&fresh)
                .any(|(a, b)| a.logits != b.logits)
        );
        team.exchange(Command::Evict(1)).unwrap();
        eprintln!(
            "A2 graph mixed: reversed owner roles reuse equals fresh direct execution; changed input executes"
        );
        drop(team);
        drop(lock);
    }
    #[test]
    #[ignore = "Requires the complete native TP8 A2 service package and eight NPUs"]
    fn native_prefix_restore_keeps_both_compression_ratios_and_owner_identity() {
        use std::os::fd::AsRawFd;
        assert!(
            std::env::var("HCCL_DETERMINISTIC")
                .is_ok_and(|value| value.eq_ignore_ascii_case("strict")),
            "Exact cross-request logits diagnostic uses explicit strict communication; normal serving uses the default"
        );
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: the owned descriptor outlives the model and its worker threads.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let directory = std::path::PathBuf::from(
            std::env::var_os("ORINFER_A2_STATE_MODEL")
                .expect("Set ORINFER_A2_STATE_MODEL to the published native service directory"),
        );
        let descriptor: serde_json::Value =
            crate::model::read(&directory.join("cache/model.json")).unwrap();
        let plan = directory
            .join("cache/packages")
            .join(descriptor["execution_package"].as_str().unwrap())
            .join("model.json");
        let team = Team::load(&plan, &directory).unwrap();
        if team
            .model
            .batch
            .as_ref()
            .unwrap()
            .decoder_replay
            .as_ref()
            .is_some_and(|r| r.final_prompt_only)
        {
            verify_prompt_replay(&team);
            return;
        }
        let forward = |slot, tokens: Vec<u32>, full| {
            let layout = Layout::new(
                vec![BatchSegment {
                    slot,
                    tokens: tokens.len(),
                }],
                vec![0],
                false,
                team.rows,
                128,
            )
            .unwrap();
            team.forward(layout, vec![tokens], full)
                .unwrap()
                .0
                .remove(0)
        };
        for length in [17usize, 129, 2049, 8193] {
            team.exchange(Command::Admit(vec![17, 7])).unwrap();
            let input = (0..length)
                .map(|i| ((i * 73 + 11) % 129280) as u32)
                .collect::<Vec<_>>();
            for chunk in input.chunks(team.rows) {
                forward(17, chunk.to_vec(), false);
            }
            team.exchange(Command::Save {
                owner: 17,
                snapshot: length as u64,
                complete: false,
            })
            .unwrap();
            let expected = forward(17, vec![13, 17, 23], true);
            team.exchange(Command::Release(Some(17))).unwrap();
            team.exchange(Command::Admit(vec![91])).unwrap();
            team.exchange(Command::Restore {
                owner: 91,
                snapshot: length as u64,
            })
            .unwrap();
            let actual = forward(91, vec![13, 17, 23], true);
            assert_eq!(expected.token, actual.token, "prefix length {length}");
            assert_eq!(expected.logits, actual.logits, "prefix length {length}");
            for chunk in input.chunks(team.rows) {
                forward(7, chunk.iter().map(|&v| (v + 1) % 129280).collect(), false);
            }
            let changed = forward(7, vec![13, 17, 23], true);
            assert_ne!(
                actual.logits, changed.logits,
                "changed input must execute with the reused Graph"
            );
            team.exchange(Command::Evict(length as u64)).unwrap();
            team.exchange(Command::Release(None)).unwrap();
            eprintln!(
                "A2 prefix {length}: complete logits equal after owner remap; changed input differs"
            );
        }
        drop(team);
        drop(lock);
    }

    fn verify_prompt_replay(team: &Team) {
        let contract = team.model.batch.as_ref().unwrap();
        assert!(!contract.state.iter().any(|s| {
            s.buffer
                .strip_prefix("cache.swa.")
                .and_then(|s| s.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
                .is_some_and(|layer| layer >= 21)
        }));
        let forward = |slot, tokens: Vec<u32>, output, decode| {
            let layout = Layout::new(
                vec![BatchSegment {
                    slot,
                    tokens: tokens.len(),
                }],
                if output { vec![0] } else { vec![] },
                decode,
                if decode { contract.owners } else { team.rows },
                contract.owners,
            )
            .unwrap();
            team.forward(layout, vec![tokens], output).unwrap().0
        };
        for length in [17usize, 2049, 8193] {
            team.exchange(Command::Admit(vec![17, 7])).unwrap();
            let input = (0..length)
                .map(|i| ((i * 73 + 11) % 129280) as u32)
                .collect::<Vec<_>>();
            let mut previous: Option<(u64, BTreeMap<u64, usize>)> = None;
            for (index, chunk) in input.chunks(team.rows).enumerate() {
                assert!(forward(17, chunk.to_vec(), false, false).is_empty());
                let id = if (index + 1) * team.rows >= length {
                    length as u64
                } else {
                    (2u64 << 32) + ((length as u64) << 16) + index as u64
                };
                let banks = team.snapshot_allocations(17, id, false).unwrap();
                if let Some((old_id, old_banks)) = previous {
                    if contract.state.iter().any(|s| s.append_only) {
                        assert!(
                            banks.keys().any(|id| old_banks.contains_key(id)),
                            "closed KV banks must survive ancestor eviction"
                        );
                    }
                    team.exchange(Command::Evict(old_id)).unwrap();
                }
                previous = Some((id, banks));
            }
            let tail = (0..128).map(|i| (i * 13 + 17) as u32).collect::<Vec<_>>();
            let mut expected = forward(17, tail.clone(), true, false);
            expected.extend(forward(17, vec![13], true, true));
            team.exchange(Command::Release(Some(17))).unwrap();
            team.exchange(Command::Admit(vec![91])).unwrap();
            team.exchange(Command::Restore {
                owner: 91,
                snapshot: length as u64,
            })
            .unwrap();
            let mut actual = forward(91, tail.clone(), true, false);
            let complete = !contract
                .decoder_replay
                .as_ref()
                .unwrap()
                .decoder_state
                .is_empty();
            let complete_id = (1u64 << 32) + length as u64;
            if complete {
                team.exchange(Command::Save {
                    owner: 91,
                    snapshot: complete_id,
                    complete: true,
                })
                .unwrap();
                team.exchange(Command::Evict(length as u64)).unwrap();
            }
            actual.extend(forward(91, vec![13], true, true));
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(
                    a.logits, b.logits,
                    "prompt cache {length}, final/decode logits"
                );
            }
            if complete {
                team.exchange(Command::Admit(vec![92])).unwrap();
                team.exchange(Command::Restore {
                    owner: 92,
                    snapshot: complete_id,
                })
                .unwrap();
                let resumed = forward(92, vec![13], true, true);
                assert_eq!(
                    resumed[0].logits, actual[1].logits,
                    "complete prompt {length}: consumer SWA must survive restore"
                );
                for token in [17, 23] {
                    let expected = forward(91, vec![token], true, true);
                    let resumed = forward(92, vec![token], true, true);
                    assert_eq!(
                        expected[0].logits, resumed[0].logits,
                        "complete prompt {length}, decode {token}"
                    );
                }
                team.exchange(Command::Evict(complete_id)).unwrap();
            }
            for chunk in input.chunks(team.rows) {
                forward(
                    7,
                    chunk.iter().map(|&v| (v + 1) % 129280).collect(),
                    false,
                    false,
                );
            }
            let changed = forward(7, tail, true, false);
            assert_ne!(
                actual[0].logits, changed[0].logits,
                "changed prefix must change output"
            );
            team.exchange(Command::Evict(length as u64)).unwrap();
            team.exchange(Command::Release(None)).unwrap();
            eprintln!(
                "A2 prompt replay prefix {length}: empty consumer, rebuilt final128 and decode agree after owner remap"
            );
        }
    }
}
