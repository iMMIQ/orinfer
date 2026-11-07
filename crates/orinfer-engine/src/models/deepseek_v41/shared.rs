//! Stable, owner-independent pieces of a bound packet. Kernels stay unchanged.
use super::*;
use std::{cell::RefCell, ops::Range};

pub(super) fn enabled() -> bool {
    std::env::var("ORINFER_A2_SHARED_GRAPH").is_ok_and(|v| v == "1")
}

pub(super) fn eligible(layout: &batch::Layout) -> bool {
    // Small verifier/draft cohorts use the existing whole-packet path. The
    // shared FFN capture budget is reserved for large source-row packets.
    layout.role == PacketRole::Target && !layout.decode && layout.rows.total >= 512
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Binding {
    Kernel(usize, usize, Vec<(usize, usize)>),
    Gather(usize, usize, u64, Element),
    Reduce(usize, usize, u64, Element, bool),
    W4(usize),
    Fork,
    Switch,
    Join,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key(String, Vec<Binding>);

pub(super) struct Pending {
    pub ops: Vec<SegmentOp>,
    labels: BTreeMap<usize, String>,
    common_end: usize,
    pub common: bool,
}
impl Pending {
    pub fn new(common: bool) -> Self {
        Self {
            ops: vec![],
            labels: BTreeMap::new(),
            common_end: 0,
            common,
        }
    }
    pub fn push(&mut self, op: SegmentOp) {
        self.ops.push(op);
        if self.common {
            self.common_end = self.ops.len();
        }
    }
    pub fn push_w4(&mut self, plan: Rc<GroupedW4A8>, name: &str) {
        if self.common {
            self.labels.insert(self.ops.len(), name.to_owned());
        }
        self.push(SegmentOp::W4(plan));
    }
    pub fn clear(&mut self) {
        self.ops.clear();
        self.labels.clear();
        self.common_end = 0;
    }
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

pub(super) struct Entry {
    key: Key,
    pub id: String,
    segment: Rc<Segment>,
    graph: RefCell<Option<Rc<crate::backend::ascend::graph::Graph>>>,
    uses: Cell<usize>,
    active: Cell<bool>,
    tick: Cell<u64>,
}
impl Entry {
    fn graph(&self) -> Option<Rc<crate::backend::ascend::graph::Graph>> {
        self.active
            .get()
            .then(|| self.graph.borrow().clone())
            .flatten()
    }
}
#[derive(Clone)]
pub(super) struct Piece {
    range: Range<usize>,
    entry: Rc<Entry>,
}
#[derive(Clone, Debug)]
pub(super) struct State {
    pub id: String,
    pub ready: bool,
    pub uses: usize,
    pub capture_s: f64,
}
pub(super) fn plan(states: &[Vec<State>], slack_s: f64) -> Result<(Vec<String>, Option<String>)> {
    let first = states.first().ok_or("Missing shared Graph ranks")?;
    if states
        .iter()
        .any(|s| s.len() != first.len() || s.iter().zip(first).any(|(a, b)| a.id != b.id))
    {
        return Err("Shared Graph logical boundaries differ across ranks".into());
    }
    let mut ready = vec![];
    let mut capture = None;
    for (i, state) in first.iter().enumerate() {
        if states.iter().all(|s| s[i].ready) {
            ready.push(state.id.clone());
        } else if capture.is_none()
            && states
                .iter()
                .all(|s| s[i].uses >= 3 && s[i].capture_s.is_finite() && s[i].capture_s <= slack_s)
        {
            capture = Some(state.id.clone());
        }
    }
    Ok((ready, capture))
}
pub(super) struct Cache {
    pub enabled: bool,
    entries: BTreeMap<Key, Rc<Entry>>,
    slots: BTreeMap<Key, Weak<Entry>>,
    clock: u64,
    nodes: usize,
    capture_s: f64,
}
impl Cache {
    pub fn new() -> Self {
        Self {
            enabled: enabled(),
            entries: BTreeMap::new(),
            slots: BTreeMap::new(),
            clock: 0,
            nodes: 0,
            capture_s: 0.01,
        }
    }
    pub fn evict(&mut self) -> bool {
        let key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.tick.get())
            .map(|(k, _)| k.clone());
        let Some(key) = key else {
            return false;
        };
        let entry = self.entries.remove(&key).unwrap();
        self.nodes -= entry.segment.ops.len();
        entry.active.set(false);
        entry.uses.set(0);
        entry.graph.borrow_mut().take();
        true
    }
    fn touch(&mut self, entry: Rc<Entry>) {
        if !self.entries.contains_key(&entry.key) {
            while self.nodes + entry.segment.ops.len() > 4096 || self.entries.len() >= 128 {
                if !self.evict() {
                    break;
                }
            }
            self.nodes += entry.segment.ops.len();
            self.entries.insert(entry.key.clone(), Rc::clone(&entry));
        }
        self.clock = self.clock.wrapping_add(1);
        entry.tick.set(self.clock);
    }
    pub fn pieces(&mut self, pending: &Pending, comm: &Communicator) -> Result<Vec<Piece>> {
        if !self.enabled || pending.common_end == 0 {
            return Ok(vec![]);
        }
        let ops = &pending.ops[..pending.common_end];
        let ranges = ranges(
            ops.iter()
                .map(|op| match op {
                    SegmentOp::Fork(_) => Kind::Fork,
                    SegmentOp::Switch(_) => Kind::Switch,
                    SegmentOp::Join(_) => Kind::Join,
                    SegmentOp::Kernel(k)
                        if k.spec().op.starts_with("private_rows.")
                            || k.spec().parameters.contains_key("_owner_access") =>
                    {
                        Kind::Private
                    }
                    SegmentOp::Exchange(_) => Kind::Private,
                    SegmentOp::W4(_) => Kind::Matmul,
                    _ => Kind::Shared,
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )?;
        let mut pieces = vec![];
        let mut pruned = false;
        for range in ranges {
            let id = pending
                .labels
                .range(range.clone())
                .next()
                .ok_or("Shared piece has no native label")?
                .1
                .clone();
            let mut normalized = ops[range.clone()].to_vec();
            let mut indices = BTreeMap::new();
            let mut bindings = vec![];
            for op in &mut normalized {
                bindings.push(match op {
                    SegmentOp::Fork(index) => {
                        let next = indices.len();
                        indices.insert(*index, next);
                        *index = next;
                        Binding::Fork
                    }
                    SegmentOp::Switch(index) | SegmentOp::Join(index) => {
                        *index = *indices.get(index).ok_or("Shared piece crosses a branch")?;
                        if matches!(op, SegmentOp::Switch(_)) {
                            Binding::Switch
                        } else {
                            Binding::Join
                        }
                    }
                    SegmentOp::Kernel(k) => {
                        let (t, r, p) = k.binding_key();
                        Binding::Kernel(t, r, p)
                    }
                    SegmentOp::Gather {
                        source,
                        destination,
                        elements,
                        element,
                    } => Binding::Gather(
                        source.pointer() as usize,
                        destination.pointer() as usize,
                        *elements,
                        *element,
                    ),
                    SegmentOp::Reduce {
                        source,
                        destination,
                        elements,
                        element,
                        scatter,
                    } => Binding::Reduce(
                        source.pointer() as usize,
                        destination.pointer() as usize,
                        *elements,
                        *element,
                        *scatter,
                    ),
                    SegmentOp::W4(p) => Binding::W4(Rc::as_ptr(p) as usize),
                    SegmentOp::Exchange(_) => {
                        return Err("Shared piece contains a private exchange".into());
                    }
                });
            }
            let key = Key(id.clone(), bindings);
            self.clock = self.clock.wrapping_add(1);
            let entry = if let Some(e) = self.slots.get(&key).and_then(Weak::upgrade) {
                e
            } else {
                branches::validate(&normalized)?;
                let ops = Rc::new(normalized);
                let e = Rc::new(Entry {
                    key: key.clone(),
                    id,
                    segment: Rc::new(Segment {
                        ops,
                        comm: comm.clone(),
                        // Auxiliary streams belong to a captured instance.
                        // Merely recognizing a reusable piece allocates none.
                        parallel: None,
                        shared: vec![],
                    }),
                    graph: RefCell::new(None),
                    uses: Cell::new(0),
                    active: Cell::new(false),
                    tick: Cell::new(self.clock),
                });
                if !pruned && self.slots.len() >= 2048 {
                    self.slots.retain(|_, e| e.strong_count() > 0);
                    pruned = true;
                }
                self.slots.insert(key, Rc::downgrade(&e));
                e
            };
            self.touch(Rc::clone(&entry));
            pieces.push(Piece { range, entry });
        }
        Ok(pieces)
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Fork,
    Switch,
    Join,
    Shared,
    Private,
    Matmul,
}
fn ranges(kinds: &[Kind]) -> Result<Vec<Range<usize>>> {
    let mut units = vec![];
    let mut i = 0;
    while i < kinds.len() {
        let start = i;
        let mut depth = 0usize;
        let mut private = false;
        let mut matmul = false;
        loop {
            match kinds[i] {
                Kind::Fork => depth += 1,
                Kind::Join => depth = depth.checked_sub(1).ok_or("Unmatched shared branch")?,
                Kind::Private => private = true,
                Kind::Matmul => matmul = true,
                _ => {}
            }
            i += 1;
            if depth == 0 {
                break;
            }
            if i == kinds.len() {
                return Err("Incomplete shared branch".into());
            }
        }
        units.push((start..i, private, matmul));
    }
    let mut out = vec![];
    let mut start = 0;
    let mut matmul = false;
    for (range, private, has_matmul) in units {
        if private {
            if matmul && range.start - start >= 8 {
                out.push(start..range.start);
            }
            start = range.end;
            matmul = false;
        } else {
            matmul |= has_matmul;
        }
    }
    if matmul && kinds.len() - start >= 8 {
        out.push(start..kinds.len());
    }
    Ok(out)
}

impl Execution {
    fn shared_entries(&self) -> Vec<Rc<Entry>> {
        self.body
            .iter()
            .chain(&self.head)
            .flat_map(|s| match s {
                Step::Segment(segment) => segment
                    .shared
                    .iter()
                    .map(|p| Rc::clone(&p.entry))
                    .collect::<Vec<_>>(),
                _ => vec![],
            })
            .collect()
    }
    pub fn shared_graphs(&self) -> Vec<Rc<crate::backend::ascend::graph::Graph>> {
        self.shared_entries()
            .into_iter()
            .filter_map(|e| e.graph())
            .collect()
    }
    pub fn record_shared_use(&self) {
        for e in self.shared_entries() {
            e.uses.set(e.uses.get().saturating_add(1));
        }
    }
}
impl Segment {
    pub(super) fn shared_graphs(&self) -> Vec<Rc<crate::backend::ascend::graph::Graph>> {
        self.shared.iter().filter_map(|p| p.entry.graph()).collect()
    }
    pub(super) fn enqueue_shared(&self, stream: crate::backend::ascend::api::Handle) -> Result<()> {
        let mut cursor = 0;
        for p in &self.shared {
            let Some(graph) = p.entry.graph() else {
                continue;
            };
            enqueue_segment(
                &self.ops[cursor..p.range.start],
                &self.comm,
                self.parallel.as_deref(),
                stream,
            )?;
            // SAFETY: the enclosing packet retains the snapshot of active graphs.
            unsafe {
                graph.execute_on_stream(stream)?;
            }
            cursor = p.range.end;
        }
        enqueue_segment(
            &self.ops[cursor..],
            &self.comm,
            self.parallel.as_deref(),
            stream,
        )
    }
}

impl Worker {
    pub fn shared_state(&mut self, layout: &batch::Layout) -> Result<Vec<State>> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let execution = runtime
            .cache
            .get(&runtime.key(layout)?)
            .ok_or("Missing shared packet")?;
        let entries = execution.shared_entries();
        for e in &entries {
            self.shared.touch(Rc::clone(e));
        }
        // Optional fragments must not displace the normal packet/owner
        // workspace. Coordinate a defer if any rank lacks capture headroom.
        let capture_s = if entries.is_empty() {
            self.shared.capture_s
        } else {
            let (device, session) = self.available_memory()?;
            let mut needed = memory::Requirement::workspace();
            needed.device += 256 << 20;
            if needed.fits(device, session) {
                self.shared.capture_s
            } else {
                f64::INFINITY
            }
        };
        Ok(entries
            .into_iter()
            .map(|e| State {
                id: e.id.clone(),
                ready: e.graph.borrow().is_some(),
                uses: e.uses.get(),
                capture_s,
            })
            .collect())
    }
    pub fn select_shared(&self, layout: &batch::Layout, ids: &[String]) -> Result<()> {
        if !self.shared.enabled {
            return if ids.is_empty() {
                Ok(())
            } else {
                Err("Shared graphs are disabled".into())
            };
        }
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let execution = runtime
            .cache
            .get(&runtime.key(layout)?)
            .ok_or("Missing shared packet")?;
        let entries = execution.shared_entries();
        if ids.iter().any(|id| {
            !entries
                .iter()
                .any(|e| &e.id == id && e.graph.borrow().is_some())
        }) {
            return Err("Shared graph selection is not resident".into());
        }
        for e in entries {
            e.active.set(ids.contains(&e.id));
        }
        Ok(())
    }
    pub fn capture_shared(&mut self, layout: &batch::Layout, id: &str) -> Result<Vec<State>> {
        let start = Instant::now();
        let (key, execution, entry) = {
            let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
            let key = runtime.key(layout)?;
            let execution = Rc::clone(runtime.cache.get(&key).ok_or("Missing shared packet")?);
            let entry = execution
                .shared_entries()
                .into_iter()
                .find(|e| e.id == id)
                .ok_or("Missing shared piece")?;
            (key, execution, entry)
        };
        self.reserve_batch_memory(memory::Requirement::workspace(), "shared graph capture")?;
        let runtime = self.batch.as_mut().unwrap();
        runtime.cache.insert(key.clone(), execution);
        runtime.touch_shared_packet(key);
        self.shared.touch(Rc::clone(&entry));
        entry.active.set(false);
        self.session.synchronize()?;
        entry.graph.borrow_mut().take();
        upload(
            &self.session,
            &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
            &0i32.to_le_bytes(),
        )?;
        let ops = Rc::clone(&entry.segment.ops);
        let depths = branches::validate(&ops)?;
        let parallel = if depths.is_empty() {
            None
        } else {
            Some(crate::backend::ascend::graph::Branches::new(
                &self.session,
                &depths,
                Box::new((Rc::clone(&ops), self.comm.clone())),
            )?)
        };
        let segment = Rc::new(Segment {
            ops,
            comm: self.comm.clone(),
            parallel,
            shared: vec![],
        });
        let graph = self.capture_segment(segment)?;
        self.shared.capture_s = self.shared.capture_s.max(start.elapsed().as_secs_f64());
        *entry.graph.borrow_mut() = Some(Rc::new(graph));
        self.shared_state(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eviction_keeps_submitted_graph_alive_then_returns_to_raw_segment() {
        use crate::backend::ascend::{hccl::RootInfo, kernels::Kernel, tests::Fixture};
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let (path, _) = fixture.aot();
        let root =
            RootInfo::fixture(&session, &path.parent().unwrap().join("libacl-fixture.so")).unwrap();
        let comm = Communicator::join(&session, &root, 0).unwrap();
        let input = session.allocate(4).unwrap();
        let output = session.allocate(4).unwrap();
        let kernel = Kernel::load(&session, &path, &[input.clone(), output.clone()]).unwrap();
        let graph = Rc::new(kernel.capture(&session).unwrap());
        let raw = Rc::new(Segment {
            ops: Rc::new(vec![SegmentOp::Kernel(kernel)]),
            comm: comm.clone(),
            parallel: None,
            shared: vec![],
        });
        let entry = Rc::new(Entry {
            key: Key("fixture".into(), vec![]),
            id: "fixture".into(),
            segment: Rc::clone(&raw),
            graph: RefCell::new(Some(graph)),
            uses: Cell::new(3),
            active: Cell::new(true),
            tick: Cell::new(0),
        });
        let segment = Rc::new(Segment {
            ops: Rc::clone(&raw.ops),
            comm,
            parallel: None,
            shared: vec![Piece {
                range: 0..1,
                entry: Rc::clone(&entry),
            }],
        });
        let mut cache = Cache::new();
        cache.touch(Rc::clone(&entry));
        upload(&session, &input, &[1, 2, 3, 4]).unwrap();
        segment.enqueue(&session).unwrap().wait().unwrap();
        upload(&session, &input, &[9, 8, 7, 6]).unwrap();
        let ticket = segment.enqueue(&session).unwrap();
        assert!(cache.evict());
        assert!(entry.graph().is_none());
        assert_eq!(entry.uses.get(), 0);
        assert_eq!(fixture.counter(7), 1);
        ticket.wait().unwrap();
        assert_eq!(fixture.counter(7), 0);
        // The CPU fixture models Graph lifetimes; actual replay arithmetic is
        // covered by the native changed-input comparison. Eviction must also
        // leave the original operations usable without that Graph.
        segment.enqueue(&session).unwrap().wait().unwrap();
        assert_eq!(session.download(&output, 0, 4).unwrap(), [9, 8, 7, 6]);
        drop((
            segment, entry, raw, input, root, cache, ticket, output, session,
        ));
        fixture.assert_clean();
    }
    #[test]
    fn ranks_select_only_common_graphs_and_capture_one_missing_piece() {
        let state = |id: &str, ready, uses| State {
            id: id.into(),
            ready,
            uses,
            capture_s: 0.02,
        };
        let mut ranks = vec![
            vec![state("a", true, 8), state("b", false, 3)],
            vec![state("a", false, 8), state("b", false, 3)],
        ];
        assert_eq!(plan(&ranks, 0.03).unwrap(), (vec![], Some("a".into())));
        assert_eq!(plan(&ranks, 0.01).unwrap(), (vec![], None));
        ranks[1][0].ready = true;
        assert_eq!(
            plan(&ranks, 0.03).unwrap(),
            (vec!["a".into()], Some("b".into()))
        );
        ranks[1][1].uses = 2;
        assert_eq!(plan(&ranks, 0.03).unwrap(), (vec!["a".into()], None));
        ranks[1][1].uses = 3;
        ranks[1][1].capture_s = f64::INFINITY;
        assert_eq!(
            plan(&ranks, f64::INFINITY).unwrap(),
            (vec!["a".into()], None)
        );
        ranks[1][1].id = "other".into();
        assert!(plan(&ranks, 0.03).is_err());
    }
    #[test]
    fn fragments_keep_nested_branches_and_reject_private_sides() {
        use Kind::*;
        let public = [
            Shared, Fork, Shared, Switch, Matmul, Shared, Shared, Join, Shared,
        ];
        let public_ranges = ranges(&public).unwrap();
        assert_eq!(public_ranges.len(), 1);
        assert_eq!(public_ranges[0], 0..9);
        let mut private = public;
        private[2] = Private;
        assert!(ranges(&private).unwrap().is_empty());
        assert!(ranges(&[Fork, Matmul]).is_err());
    }
}
