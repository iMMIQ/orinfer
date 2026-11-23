//! Owner-private state and shared token rows for native A2 packets.
mod memory;
use super::{Buffer, Execution, Instruction, SubProgram, Worker, get};
use crate::artifact::Result;
use orinfer_model_sdk::{
    architecture::BatchSegment, operators::dynamic::RowExpression, packet::Rows,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    rc::Rc,
};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Column {
    #[serde(default)]
    pub role: super::PacketRole,
    pub buffer: String,
    pub row_bytes: Option<usize>,
    pub row_alignment: usize,
    pub decode: Option<bool>,
    #[serde(default)]
    pub row_partition: Option<usize>,
    #[serde(default)]
    pub decoder: bool,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Contract {
    pub owners: usize,
    #[serde(default)]
    pub output_capacity: usize,
    /// Explicitly validated aggregate verification rows, independent of SP selection.
    #[serde(default)]
    pub verification_capacity: Option<usize>,
    pub columns: Vec<Column>,
    pub private: Vec<String>,
    pub pointers: BTreeMap<String, Vec<usize>>,
    pub active: String,
    pub endpoints: String,
    pub work: BTreeMap<String, RowExpression>,
    #[serde(default)]
    pub work_ranges: BTreeMap<String, WorkRange>,
    /// Pointer/work tables selecting global rows rank%stride + n*stride.
    #[serde(default)]
    pub partitioned_tables: BTreeMap<String, usize>,
    #[serde(default)]
    pub prefill_quantum: Option<usize>,
    #[serde(default)]
    pub state: Vec<super::state::StateLayout>,
    #[serde(default)]
    pub verification_windows: BTreeMap<String, super::state::AppendWindow>,
    #[serde(default)]
    pub decoder_replay: Option<super::replay::Contract>,
}
impl Contract {
    pub(super) fn verification_rows(&self, legacy: usize, capacity: usize) -> Result<usize> {
        match self.verification_capacity {
            Some(rows) if rows == 0 || rows > capacity.min(self.output_capacity()) => {
                Err("Verification capacity exceeds target rows or output capacity".into())
            }
            Some(rows) => Ok(rows),
            None => Ok(legacy),
        }
    }
    pub(super) fn output_capacity(&self) -> usize {
        if self.output_capacity == 0 {
            self.owners
        } else {
            self.output_capacity
        }
    }
    pub(super) fn draft_capacity(&self) -> usize {
        (self.output_capacity() / self.owners).saturating_sub(1)
    }
    pub(super) fn token_capacity(&self, table: &str, rows: usize) -> Result<usize> {
        match self.partitioned_tables.get(table) {
            None => Ok(rows),
            Some(&stride) if stride > 1 => Ok(rows.div_ceil(stride)),
            _ => Err("Invalid owner row partition stride".into()),
        }
    }
    pub(super) fn work_items(
        &self,
        layout: &Layout,
        grid: &RowExpression,
        name: &str,
        rank: usize,
    ) -> Result<Vec<[i32; 2]>> {
        let padded = work_pad(layout);
        let decoder = self
            .decoder_replay
            .as_ref()
            .filter(|r| {
                layout.role == super::PacketRole::Target
                    && !layout.decode
                    && r.tables.contains(name)
            })
            .map(|r| layout.decoder(r.keep, r.final_prompt_only))
            .transpose()?;
        let layout = decoder.as_ref().unwrap_or(layout);
        let range = self.work_ranges.get(name);
        let mut items = if let Some(&stride) = self.partitioned_tables.get(name) {
            let view = layout.rows.partition(stride, rank % stride)?;
            work_lengths(view.lengths, grid, range)?
        } else {
            work_items(layout, grid, range)?
        };
        if layout.work_shape.is_some() {
            let reference = layout.work_layout()?;
            let capacity =
                work_capacity(self.work_items(&reference, grid, name, rank)?.len(), padded);
            if items.len() > capacity {
                return Err(format!(
                    "Owner work table {name} needs {} items but its retained Graph holds {capacity}",
                    items.len()
                ));
            }
            items.resize(capacity, [-1, 0]);
        }
        items.resize(work_capacity(items.len(), padded), [-1, 0]);
        Ok(items)
    }
    pub(super) fn preferred_span(&self) -> usize {
        self.prefill_quantum.unwrap_or_else(|| {
            self.work_ranges
                .values()
                .map(|r| r.min_rows)
                .filter(|&n| n > 1)
                .min()
                .unwrap_or(1)
        })
    }
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkRange {
    pub min_rows: usize,
    pub max_rows: usize,
}
#[derive(Deserialize)]
pub(super) struct Access {
    pub column: usize,
    pub write: bool,
}
pub(super) struct Owner {
    pub bank: Option<Buffer>,
    pub buffers: BTreeMap<String, Buffer>,
    pub history: Vec<u32>,
    pub limit: Option<usize>,
    pub vision: std::sync::Arc<Vec<super::vision::Span>>,
    pub clean: bool,
}
pub(super) struct IdleOwner {
    pub bank: Buffer,
    pub buffers: BTreeMap<String, Buffer>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheState {
    Missing,
    Direct { uses: u8, consecutive: u8 },
    Graph,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SplitStates {
    pub trunk: CacheState,
    pub tail: Option<CacheState>,
    pub head: Option<CacheState>,
}
impl SplitStates {
    pub(super) fn monolithic(state: CacheState) -> Self {
        Self {
            trunk: state,
            tail: None,
            head: None,
        }
    }
    pub(super) fn all_graphs(&self) -> bool {
        self.trunk == CacheState::Graph
            && self.tail.is_none_or(|s| s == CacheState::Graph)
            && self.head.is_none_or(|s| s == CacheState::Graph)
    }
    #[cfg(test)]
    pub(super) fn summary(&self) -> CacheState {
        let mut min_uses = u8::MAX;
        let mut min_consecutive = u8::MAX;
        let mut all_graph = true;
        for state in std::iter::once(self.trunk)
            .chain(self.tail)
            .chain(self.head)
        {
            match state {
                CacheState::Missing => return CacheState::Missing,
                CacheState::Graph => {}
                CacheState::Direct { uses, consecutive } => {
                    all_graph = false;
                    min_uses = min_uses.min(uses);
                    min_consecutive = min_consecutive.min(consecutive);
                }
            }
        }
        if all_graph {
            CacheState::Graph
        } else {
            CacheState::Direct {
                uses: min_uses,
                consecutive: min_consecutive,
            }
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PartAction {
    pub invalidate: bool,
    /// None = already resident in requested mode; Some(false) = BuildDirect; Some(true) = CaptureReplay.
    pub prepare: Option<bool>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SplitPlan {
    pub trunk: PartAction,
    pub tail: Option<PartAction>,
    pub head: Option<PartAction>,
}
impl SplitPlan {
    pub(super) fn any_invalidate(&self) -> bool {
        self.trunk.invalidate
            || self.tail.is_some_and(|a| a.invalidate)
            || self.head.is_some_and(|a| a.invalidate)
    }
    pub(super) fn any_prepare(&self) -> bool {
        self.trunk.prepare.is_some()
            || self.tail.is_some_and(|a| a.prepare.is_some())
            || self.head.is_some_and(|a| a.prepare.is_some())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    family: (super::PacketRole, bool, usize),
    rows: usize,
    decoder_rows: Option<usize>,
    outputs: usize,
    work_rows: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TrunkKey {
    family: (super::PacketRole, bool, usize),
    rows: usize,
    clipped: bool,
    work_rows: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct TailKey {
    family: (super::PacketRole, bool, usize),
    trunk_family: (super::PacketRole, bool, usize),
    rows: usize,
    decoder_rows: usize,
    clipped: bool,
    work_rows: Vec<usize>,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct HeadKey {
    family: (super::PacketRole, bool, usize),
    rows: usize,
    decoder_rows: Option<usize>,
    outputs: usize,
}
pub(super) struct SplitKeys {
    pub family: (super::PacketRole, bool, usize),
    pub trunk: TrunkKey,
    pub tail: Option<TailKey>,
    pub head: Option<HeadKey>,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct WorkLaunch {
    role: super::PacketRole,
    family: (bool, usize, usize),
    grid: RowExpression,
    range: Option<(usize, usize)>,
    partition: usize,
    decoder: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkPad {
    Exact,
    Padded,
    Bucketed,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct TableKey {
    family: (super::PacketRole, bool, usize),
    tail_min_rows: Option<usize>,
    segments: Vec<(usize, usize)>,
    outputs: Vec<usize>,
    output_positions: Vec<usize>,
    decoder_roles: Vec<super::replay::Role>,
    work_shape: Option<Vec<usize>>,
    work_padding: WorkPad,
}
pub(super) struct BoundTables {
    key: TableKey,
    retained: Vec<Buffer>,
}
#[derive(Default)]
pub(in crate::models::deepseek_v41) struct ProtectedKeys<'a> {
    pub mono: Option<&'a Key>,
    pub trunk: Option<&'a TrunkKey>,
    pub tail: Option<&'a TailKey>,
    pub head: Option<&'a HeadKey>,
    pub keep_idle_owner: bool,
}
pub(super) struct Runtime {
    pub contract: Contract,
    pub owners: BTreeMap<usize, Owner>,
    pub(super) idle_owners: Vec<IdleOwner>,
    pub(super) cache: BTreeMap<Key, Rc<Execution>>,
    recent: BTreeMap<(super::PacketRole, bool, usize), (Key, u8)>,
    lru: VecDeque<Key>,
    pub(super) trunk_cache: BTreeMap<TrunkKey, Rc<SubProgram>>,
    recent_trunk: BTreeMap<(super::PacketRole, bool, usize), (TrunkKey, u8)>,
    trunk_lru: VecDeque<TrunkKey>,
    pub(super) tail_cache: BTreeMap<TailKey, Rc<SubProgram>>,
    recent_tail: BTreeMap<(super::PacketRole, bool, usize), (TailKey, u8)>,
    tail_lru: VecDeque<TailKey>,
    pub(super) head_cache: BTreeMap<HeadKey, Rc<SubProgram>>,
    recent_head: BTreeMap<(super::PacketRole, bool, usize), (HeadKey, u8)>,
    head_lru: VecDeque<HeadKey>,
    capacity: usize,
    tables: BTreeMap<(super::PacketRole, bool, usize), TableUse>,
    families: Vec<(super::PacketRole, bool, usize)>,
    launches: BTreeSet<WorkLaunch>,
    pub(super) bound_tables: Option<BoundTables>,
}
struct TableUse {
    max_rows: usize,
    draft_block: Option<usize>,
    draft_stride: Option<usize>,
    decoder_split: Option<usize>,
    decoder_boundary: Option<usize>,
    decoder_prepare: usize,
    head_independent: bool,
    trunk_uses_clipped: bool,
    tail_downgrade_targets: BTreeSet<(super::PacketRole, bool, usize)>,
    pointers: BTreeSet<String>,
    work: BTreeSet<String>,
    columns: Vec<bool>,
    trunk_pointers: BTreeSet<String>,
    trunk_work: BTreeSet<String>,
    trunk_columns: Vec<bool>,
    tail_pointers: BTreeSet<String>,
    tail_work: BTreeSet<String>,
    tail_columns: Vec<bool>,
}
impl Runtime {
    pub(super) fn touch_shared_packet(&mut self, key: Key) {
        self.lru.retain(|k| k != &key);
        self.lru.push_back(key);
    }
    fn touch_trunk(&mut self, key: TrunkKey) {
        self.trunk_lru.retain(|k| k != &key);
        self.trunk_lru.push_back(key);
    }
    fn touch_tail(&mut self, key: TailKey) {
        self.tail_lru.retain(|k| k != &key);
        self.tail_lru.push_back(key);
    }
    fn touch_head(&mut self, key: HeadKey) {
        self.head_lru.retain(|k| k != &key);
        self.head_lru.push_back(key);
    }
    fn consecutive_uses(&self, key: &Key) -> u8 {
        self.recent
            .get(&key.family)
            .filter(|(previous, _)| previous == key)
            .map_or(0, |(_, uses)| *uses)
    }
    fn record_use(&mut self, key: Key) {
        let uses = self.consecutive_uses(&key).saturating_add(1);
        self.recent.insert(key.family, (key, uses));
    }
    fn consecutive_trunk(&self, key: &TrunkKey) -> u8 {
        self.recent_trunk
            .get(&key.family)
            .filter(|(previous, _)| previous == key)
            .map_or(0, |(_, uses)| *uses)
    }
    fn record_trunk_use(&mut self, key: TrunkKey) {
        let uses = self.consecutive_trunk(&key).saturating_add(1);
        self.recent_trunk.insert(key.family, (key, uses));
    }
    fn consecutive_tail(&self, key: &TailKey) -> u8 {
        self.recent_tail
            .get(&key.family)
            .filter(|(previous, _)| previous == key)
            .map_or(0, |(_, uses)| *uses)
    }
    fn record_tail_use(&mut self, key: TailKey) {
        let uses = self.consecutive_tail(&key).saturating_add(1);
        self.recent_tail.insert(key.family, (key, uses));
    }
    fn consecutive_head(&self, key: &HeadKey) -> u8 {
        self.recent_head
            .get(&key.family)
            .filter(|(previous, _)| previous == key)
            .map_or(0, |(_, uses)| *uses)
    }
    fn record_head_use(&mut self, key: HeadKey) {
        let uses = self.consecutive_head(&key).saturating_add(1);
        self.recent_head.insert(key.family, (key, uses));
    }
    fn remove(&mut self, key: &Key) -> bool {
        let removed = self.cache.remove(key).is_some();
        if self
            .recent
            .get(&key.family)
            .is_some_and(|(last, _)| last == key)
        {
            self.recent.remove(&key.family);
        }
        removed
    }
    fn remove_trunk(&mut self, key: &TrunkKey) -> bool {
        let removed = self.trunk_cache.remove(key).is_some();
        if self
            .recent_trunk
            .get(&key.family)
            .is_some_and(|(last, _)| last == key)
        {
            self.recent_trunk.remove(&key.family);
        }
        removed
    }
    fn remove_tail(&mut self, key: &TailKey) -> bool {
        let removed = self.tail_cache.remove(key).is_some();
        if self
            .recent_tail
            .get(&key.family)
            .is_some_and(|(last, _)| last == key)
        {
            self.recent_tail.remove(&key.family);
        }
        removed
    }
    fn remove_head(&mut self, key: &HeadKey) -> bool {
        let removed = self.head_cache.remove(key).is_some();
        if self
            .recent_head
            .get(&key.family)
            .is_some_and(|(last, _)| last == key)
        {
            self.recent_head.remove(&key.family);
        }
        removed
    }
    /// Cap captured heavy graphs across monolithic, trunk, and tail caches so
    /// 910B hardware stream/event/model quotas stay bounded. Uncaptured direct
    /// entries hold no ACL Graph resources, and tiny 1-kernel heads stay
    /// bounded independently by `self.capacity` so they are never thrash-evicted.
    fn trim_total_programs(&mut self, protected: &ProtectedKeys<'_>) {
        let graph_cap = self.capacity.saturating_add(self.capacity / 2).max(1);
        loop {
            let captured = self.cache.values().filter(|e| e.all_captured()).count()
                + self.trunk_cache.values().filter(|e| e.captured).count()
                + self.tail_cache.values().filter(|e| e.captured).count();
            if captured <= graph_cap {
                break;
            }
            let evicted = Self::evict_oldest_unprotected(
                &mut self.lru,
                protected.mono,
                |k| self.cache.remove(k).is_some(),
                &mut self.recent,
                |k| k.family,
            ) || Self::evict_oldest_unprotected(
                &mut self.tail_lru,
                protected.tail,
                |k| self.tail_cache.remove(k).is_some(),
                &mut self.recent_tail,
                |k| k.family,
            ) || Self::evict_oldest_unprotected(
                &mut self.trunk_lru,
                protected.trunk,
                |k| self.trunk_cache.remove(k).is_some(),
                &mut self.recent_trunk,
                |k| k.family,
            );
            if !evicted {
                break;
            }
        }
    }
    fn evict_oldest_unprotected<K: PartialEq, F: Ord>(
        lru: &mut VecDeque<K>,
        protected: Option<&K>,
        mut remove_entry: impl FnMut(&K) -> bool,
        recent: &mut BTreeMap<F, (K, u8)>,
        family: impl Fn(&K) -> F,
    ) -> bool {
        while let Some(pos) = lru.iter().position(|k| Some(k) != protected) {
            let key = lru.remove(pos).unwrap();
            if remove_entry(&key) {
                let fam = family(&key);
                if recent.get(&fam).is_some_and(|(last, _)| last == &key) {
                    recent.remove(&fam);
                }
                return true;
            }
        }
        false
    }
    fn resolve_family(&self, layout: &Layout) -> Result<(super::PacketRole, bool, usize)> {
        self.families
            .iter()
            .find_map(|family| {
                let used = &self.tables[family];
                (family.0 == layout.role
                    && family.1 == layout.decode
                    && (layout.role != super::PacketRole::Draft
                        || used.draft_block.is_some_and(|block| {
                            layout.segments.iter().all(|s| s.tokens == block)
                                && layout.outputs.len()
                                    == if used.draft_stride.is_some() {
                                        layout.segments.len()
                                    } else {
                                        layout.rows.total
                                    }
                        }))
                    && family.2 <= layout.rows.total
                    && layout.rows.total <= used.max_rows)
                    .then_some(*family)
            })
            .ok_or_else(|| "Missing batch family".into())
    }
    pub(super) fn key(&self, layout: &Layout) -> Result<Key> {
        let family = self.resolve_family(layout)?;
        shape_key(&self.contract, layout, family, &self.launches)
    }
    fn resolve_tail_family(
        &self,
        trunk_family: (super::PacketRole, bool, usize),
        decoder_rows: usize,
        clipped: bool,
    ) -> (super::PacketRole, bool, usize) {
        if !clipped || decoder_rows == 0 {
            return trunk_family;
        }
        let trunk_used = &self.tables[&trunk_family];
        if trunk_used.tail_downgrade_targets.is_empty() {
            return trunk_family;
        }
        self.families
            .iter()
            .find_map(|family| {
                let used = &self.tables[family];
                (trunk_used.tail_downgrade_targets.contains(family)
                    && family.2 <= decoder_rows
                    && decoder_rows <= used.max_rows)
                    .then_some(*family)
            })
            .unwrap_or(trunk_family)
    }
    pub(super) fn split_keys(&self, layout: &Layout) -> Result<Option<SplitKeys>> {
        let family = self.resolve_family(layout)?;
        let used = &self.tables[&family];
        if used.decoder_split.is_none() {
            return Ok(None);
        }
        let decoder_rows = self
            .contract
            .decoder_replay
            .as_ref()
            .filter(|_| layout.role == super::PacketRole::Target && !layout.decode)
            .map(|r| layout.decoder(r.keep, r.final_prompt_only))
            .transpose()?
            .map(|d| d.rows.total);
        let clipped = decoder_rows.is_some_and(|d| d < layout.rows.total);
        let tail_family = decoder_rows
            .filter(|&d| d > 0)
            .map(|d| self.resolve_tail_family(family, d, clipped));
        let (_, trunk_work, decoder_work) = split_work_rows(
            &self.contract,
            layout,
            family,
            tail_family.unwrap_or(family),
            &self.launches,
        )?;
        let trunk = TrunkKey {
            family,
            rows: layout.rows.total,
            clipped: used.trunk_uses_clipped && clipped,
            work_rows: trunk_work,
        };
        let tail = decoder_rows.filter(|&d| d > 0).map(|d| {
            let tf = tail_family.unwrap_or(family);
            TailKey {
                family: tf,
                trunk_family: family,
                rows: if used.decoder_prepare > 0 {
                    layout.rows.total
                } else {
                    0
                },
                decoder_rows: d,
                clipped,
                work_rows: decoder_work,
            }
        });
        let head = (!layout.outputs.is_empty()).then_some(HeadKey {
            family: if used.head_independent {
                tail_family.unwrap_or(family)
            } else {
                family
            },
            rows: if used.head_independent {
                0
            } else {
                layout.rows.total
            },
            decoder_rows,
            outputs: layout.outputs.len(),
        });
        Ok(Some(SplitKeys {
            family,
            trunk,
            tail,
            head,
        }))
    }
    pub(super) fn execution(
        &self,
        layout: &Layout,
        templates: &[Rc<super::Packet>],
    ) -> Result<Option<Rc<Execution>>> {
        if let Some(split) = self.split_keys(layout)? {
            let Some(trunk) = self.trunk_cache.get(&split.trunk).cloned() else {
                return Ok(None);
            };
            let tail = match &split.tail {
                Some(k) => match self.tail_cache.get(k) {
                    Some(t) => Some(Rc::clone(t)),
                    None => return Ok(None),
                },
                None => None,
            };
            let head = match &split.head {
                Some(k) => match self.head_cache.get(k) {
                    Some(h) => Some(Rc::clone(h)),
                    None => return Ok(None),
                },
                None => None,
            };
            let packet = templates
                .iter()
                .find(|p| {
                    p.role == layout.role
                        && p.batch_decode == layout.decode
                        && p.min_rows == split.family.2
                })
                .cloned()
                .ok_or("Missing batch family")?;
            let tail_packet = split
                .tail
                .as_ref()
                .filter(|k| k.family != split.family)
                .map(|k| {
                    templates
                        .iter()
                        .find(|p| {
                            p.role == k.family.0
                                && p.batch_decode == k.family.1
                                && p.min_rows == k.family.2
                        })
                        .cloned()
                        .ok_or_else(|| "Missing tail batch family".to_string())
                })
                .transpose()?;
            return Ok(Some(Rc::new(Execution {
                packet,
                tail_packet,
                trunk,
                tail,
                head,
            })));
        }
        let key = self.key(layout)?;
        Ok(self.cache.get(&key).cloned())
    }
    pub fn new(
        contract: Contract,
        capacity: usize,
        packets: &[super::Packet],
        launches: &BTreeSet<WorkLaunch>,
    ) -> Self {
        let mut tables = BTreeMap::new();
        let mut boundaries = BTreeMap::new();
        for packet in packets {
            let boundary = super::replay::Boundary::find(&packet.body)
                .ok()
                .flatten()
                .filter(|b| {
                    // Only split when no Fork/Switch/Join crosses the boundary
                    // and no instruction in body uses owner-output counts.
                    let split_idx = b.split_index();
                    let mut depth = 0usize;
                    for step in &packet.body[..split_idx] {
                        match step {
                            Instruction::Fork => depth += 1,
                            Instruction::Join => depth = depth.saturating_sub(1),
                            _ => {}
                        }
                    }
                    let body_uses_owners = packet.body.iter().any(|s| match s {
                        Instruction::Kernel { owners, .. } | Instruction::Gather { owners, .. } => {
                            *owners
                        }
                        _ => false,
                    });
                    depth == 0 && !body_uses_owners
                });
            let split = boundary.as_ref().map(super::replay::Boundary::split_index);
            let trunk_uses_clipped = split.is_some_and(|split_idx| {
                packet.body[..split_idx].iter().any(|s| {
                    matches!(
                        s,
                        Instruction::Kernel {
                            uncompacted_manifest: Some(_),
                            ..
                        }
                    )
                })
            });
            let head_independent = matches!(
                packet.head.first(),
                Some(Instruction::DecoderRows {
                    prepare: 0,
                    select: 0,
                    ..
                })
            ) && !packet.head.iter().any(|s| {
                matches!(
                    s,
                    Instruction::Kernel {
                        uncompacted_manifest: Some(_),
                        ..
                    }
                )
            });
            let mut used = TableUse {
                max_rows: packet.dispatch_max(),
                draft_block: packet.coordinates.draft_block,
                draft_stride: packet.draft_stride,
                decoder_split: split,
                decoder_boundary: boundary.as_ref().map(|b| b.index),
                decoder_prepare: boundary.as_ref().map_or(0, |b| b.prepare),
                head_independent,
                trunk_uses_clipped,
                tail_downgrade_targets: BTreeSet::new(),
                pointers: BTreeSet::new(),
                work: BTreeSet::new(),
                columns: vec![false; contract.columns.len()],
                trunk_pointers: BTreeSet::new(),
                trunk_work: BTreeSet::new(),
                trunk_columns: vec![false; contract.columns.len()],
                tail_pointers: BTreeSet::new(),
                tail_work: BTreeSet::new(),
                tail_columns: vec![false; contract.columns.len()],
            };
            let record_step = |step: &Instruction,
                               work: &mut BTreeSet<String>,
                               pointers: &mut BTreeSet<String>,
                               columns: &mut [bool]| {
                if let Instruction::Kernel {
                    owner_work: Some(_),
                    buffers,
                    ..
                } = step
                {
                    work.insert(buffers[0].clone());
                    pointers.insert(buffers[1].clone());
                    for &column in &contract.pointers[&buffers[1]] {
                        columns[column] = true;
                    }
                }
            };
            let split_at = split.unwrap_or(packet.body.len());
            for step in &packet.body[..split_at] {
                record_step(step, &mut used.work, &mut used.pointers, &mut used.columns);
                record_step(
                    step,
                    &mut used.trunk_work,
                    &mut used.trunk_pointers,
                    &mut used.trunk_columns,
                );
            }
            for step in packet.body[split_at..].iter().chain(&packet.head) {
                record_step(step, &mut used.work, &mut used.pointers, &mut used.columns);
                record_step(
                    step,
                    &mut used.tail_work,
                    &mut used.tail_pointers,
                    &mut used.tail_columns,
                );
            }
            let key = (packet.role, packet.batch_decode, packet.min_rows);
            if let Some(b) = boundary {
                boundaries.insert(key, b);
            }
            tables.insert(key, used);
        }
        // Pre-validate cross-family tail downgrade: an SP trunk (prepare > 0)
        // that gathers all sequence-partitioned live rows into global buffers
        // can hand off directly to a smaller non-SP family's Tail + Head.
        for trunk in packets {
            let tk = (trunk.role, trunk.batch_decode, trunk.min_rows);
            let Some(tb) = boundaries.get(&tk) else {
                continue;
            };
            if trunk.role != super::PacketRole::Target || trunk.batch_decode || tb.prepare == 0 {
                continue;
            }
            let mut trunk_global = BTreeSet::new();
            for step in &trunk.body[..tb.index] {
                match step {
                    Instruction::Gather { destination, .. } => {
                        trunk_global.insert(destination.as_str());
                    }
                    Instruction::Kernel { buffers, .. } => {
                        for b in buffers {
                            trunk_global.insert(b.as_str());
                        }
                    }
                    _ => {}
                }
            }
            for cand in packets {
                let ck = (cand.role, cand.batch_decode, cand.min_rows);
                if ck == tk
                    || cand.role != super::PacketRole::Target
                    || cand.batch_decode
                    || cand.min_rows >= trunk.min_rows
                    || cand.rows != trunk.rows
                    || cand.decoder_coordinates != trunk.decoder_coordinates
                    || cand.head != trunk.head
                    || cand.logits != trunk.logits
                    || cand.best_value != trunk.best_value
                    || cand.best_index != trunk.best_index
                    || cand.confidence != trunk.confidence
                    || cand.vocab_first != trunk.vocab_first
                    || cand.replicated_output != trunk.replicated_output
                {
                    continue;
                }
                let Some(cb) = boundaries.get(&ck) else {
                    continue;
                };
                if cb.prepare != 0 || cb.keep != tb.keep || cb.select == 0 {
                    continue;
                }
                // Every non-scratch source buffer read by the candidate's
                // decoder selection must already be populated in global order
                // by the trunk + its prepare gathers.
                let select_sources_ready = cand.body[cb.index + 1..=cb.index + cb.select]
                    .iter()
                    .all(|step| match step {
                        Instruction::Kernel { buffers, .. } => {
                            let src = buffers.first().map(String::as_str).unwrap_or("");
                            src.starts_with("batch.decoder.scratch.")
                                || src.ends_with(".residual.b")
                                || trunk_global.contains(src)
                        }
                        _ => false,
                    });
                if select_sources_ready {
                    tables
                        .get_mut(&tk)
                        .unwrap()
                        .tail_downgrade_targets
                        .insert(ck);
                }
            }
        }
        Self {
            contract,
            owners: BTreeMap::new(),
            idle_owners: Vec::new(),
            cache: BTreeMap::new(),
            recent: BTreeMap::new(),
            lru: VecDeque::new(),
            trunk_cache: BTreeMap::new(),
            recent_trunk: BTreeMap::new(),
            trunk_lru: VecDeque::new(),
            tail_cache: BTreeMap::new(),
            recent_tail: BTreeMap::new(),
            tail_lru: VecDeque::new(),
            head_cache: BTreeMap::new(),
            recent_head: BTreeMap::new(),
            head_lru: VecDeque::new(),
            capacity,
            tables,
            families: packets
                .iter()
                .map(|p| (p.role, p.batch_decode, p.min_rows))
                .collect(),
            launches: launches.clone(),
            bound_tables: None,
        }
    }
}
#[derive(Clone)]
pub(super) struct Layout {
    pub role: super::PacketRole,
    pub segments: Vec<BatchSegment>,
    pub rows: Rows,
    pub outputs: Vec<usize>,
    pub output_positions: Vec<usize>,
    pub decode: bool,
    pub decoder_roles: Vec<super::replay::Role>,
    pub pad_slot: Option<usize>,
    pub bucketed: bool,
    work_shape: Option<Vec<usize>>,
    work_padding: bool,
}
impl Layout {
    pub fn new(
        segments: Vec<BatchSegment>,
        outputs: Vec<usize>,
        decode: bool,
        capacity: usize,
        owners: usize,
    ) -> Result<Self> {
        let rows = Rows::packet(&segments, owners, capacity)?;
        if decode && segments.iter().any(|s| s.tokens != 1)
            || outputs.iter().any(|&i| i >= segments.len())
            || outputs.windows(2).any(|p| p[0] >= p[1])
        {
            return Err("Invalid owner output/role layout".into());
        }
        let decoder_roles = (0..segments.len())
            .map(|i| {
                if decode {
                    super::replay::Role::Continue
                } else if outputs.contains(&i) {
                    super::replay::Role::Final
                } else {
                    super::replay::Role::Producer
                }
            })
            .collect();
        let work_padding = !decode && rows.total >= 512;
        Ok(Self {
            role: super::PacketRole::Target,
            output_positions: outputs.iter().map(|&i| segments[i].tokens - 1).collect(),
            segments,
            rows,
            outputs,
            decode,
            decoder_roles,
            pad_slot: None,
            bucketed: false,
            work_shape: None,
            work_padding,
        })
    }
    pub(super) fn enable_bucketing(&mut self) {
        self.bucketed = true;
        if !self.decode {
            self.work_padding = true;
        }
    }
    /// Append a scratch padding segment (and optional dummy outputs) while
    /// preserving all real owner slices, roles, and retained work shapes.
    pub(super) fn append_pad(
        &mut self,
        slot: usize,
        pad_tokens: usize,
        pad_outputs: usize,
        shrink_producer: Option<usize>,
        capacity: usize,
        owners: usize,
    ) -> Result<()> {
        if pad_tokens == 0 || self.decode || self.pad_slot.is_some() {
            return Err("Invalid padding layout request".into());
        }
        if let Some(lane) = shrink_producer {
            if self.decoder_roles.get(lane) != Some(&super::replay::Role::Producer)
                || self.segments[lane].tokens <= pad_tokens
            {
                return Err("Invalid producer shrink for padding".into());
            }
            self.segments[lane].tokens -= pad_tokens;
            if let Some(shape) = &mut self.work_shape {
                shape[lane] = shape[lane].saturating_sub(pad_tokens);
            }
        }
        let pad_lane = self.segments.len();
        self.segments.push(BatchSegment {
            slot,
            tokens: pad_tokens,
        });
        self.decoder_roles.push(super::replay::Role::Continue);
        if let Some(shape) = &mut self.work_shape {
            shape.push(pad_tokens);
        }
        for pos in 0..pad_outputs {
            self.outputs.push(pad_lane);
            self.output_positions.push(pos % pad_tokens);
        }
        self.rows = Rows::packet(&self.segments, owners, capacity)?;
        self.pad_slot = Some(slot);
        self.enable_bucketing();
        Ok(())
    }
    /// Retain a proven launch envelope while row/address tables carry real lengths.
    /// Only short decoder lengths may change. Prompt membership and lengths
    /// remain fixed, including their final-window decoder replay roles.
    pub(super) fn bound_work(
        &mut self,
        reference: &Self,
        contract: &Contract,
        launches: &BTreeSet<WorkLaunch>,
    ) -> Result<bool> {
        if self.work_shape.is_some()
            || reference.work_shape.is_some()
            || self.role != super::PacketRole::Target
            || reference.role != self.role
            || self.decode
            || reference.decode
            || self.rows.total != reference.rows.total
            || self.outputs.len() != reference.outputs.len()
            || self.segments.len() != reference.segments.len()
            || self.decoder_roles != reference.decoder_roles
            || work_pad(self) != work_pad(reference)
            || self
                .segments
                .iter()
                .zip(&reference.segments)
                .zip(&self.decoder_roles)
                .any(|((a, b), role)| {
                    a.slot != b.slot
                        || if *role == super::replay::Role::Continue {
                            !(1..=8).contains(&a.tokens) || !(1..=8).contains(&b.tokens)
                        } else {
                            a.tokens != b.tokens
                        }
                })
        {
            return Ok(false);
        }
        let family = (self.role, false, 1);
        let actual = shape_key_with_scope(contract, self, family, launches, false)?;
        let retained = shape_key_with_scope(contract, reference, family, launches, false)?;
        if actual.decoder_rows != retained.decoder_rows
            || actual
                .work_rows
                .iter()
                .zip(&retained.work_rows)
                .any(|(a, b)| a > b)
        {
            return Ok(false);
        }
        self.work_shape = Some(reference.segments.iter().map(|s| s.tokens).collect());
        Ok(true)
    }
    fn work_layout(&self) -> Result<Self> {
        let mut reference = self.clone();
        if let Some(lengths) = reference.work_shape.take() {
            for (segment, tokens) in reference.segments.iter_mut().zip(lengths) {
                segment.tokens = tokens;
            }
            let rows = orinfer_model_sdk::packet::RowView {
                starts: vec![],
                lengths: reference.segments.iter().map(|s| s.tokens).collect(),
                total: 0,
            }
            .aligned(1)?;
            if rows.total != self.rows.total {
                return Err("Retained work changed the packet row count".into());
            }
            reference.rows = Rows {
                starts: rows.starts,
                total: rows.total,
            };
        }
        Ok(reference)
    }
    /// Verification shares the target body and head; only endpoint selection changes.
    pub(super) fn verification(
        segments: Vec<BatchSegment>,
        capacity: usize,
        owners: usize,
    ) -> Result<Self> {
        let mut view = Self::new(segments, vec![], false, capacity, owners)?;
        for (i, segment) in view.segments.iter().enumerate() {
            for position in 0..segment.tokens {
                view.outputs.push(i);
                view.output_positions.push(position);
            }
            view.decoder_roles[i] = super::replay::Role::Continue;
        }
        Ok(view)
    }
    pub(super) fn decoder(&self, keep: usize, final_prompt_only: bool) -> Result<Self> {
        if !final_prompt_only {
            return self.suffix(keep);
        }
        if self.decode
            || !(128..=512).contains(&keep)
            || self.decoder_roles.len() != self.segments.len()
        {
            return Err("Invalid final-prompt decoder layout".into());
        }
        let mut view = self.clone();
        if let Some(lengths) = &mut view.work_shape {
            for (length, role) in lengths.iter_mut().zip(&self.decoder_roles) {
                *length = if *role == super::replay::Role::Producer {
                    0
                } else {
                    (*length).min(keep)
                };
            }
        }
        // Preserve zero-row owners and their operand-table lanes.
        let lengths = self
            .segments
            .iter()
            .zip(&self.decoder_roles)
            .map(|(s, role)| {
                if *role == super::replay::Role::Producer {
                    0
                } else {
                    s.tokens.min(keep)
                }
            })
            .collect::<Vec<_>>();
        let selected = orinfer_model_sdk::packet::RowView {
            starts: vec![0; lengths.len()],
            lengths,
            total: 0,
        }
        .aligned(1)?;
        view.rows = Rows {
            starts: selected.starts,
            total: selected.total,
        };
        for (s, n) in view.segments.iter_mut().zip(selected.lengths) {
            s.tokens = n;
        }
        view.remap_endpoints(self)?;
        if view.outputs.iter().any(|&i| view.segments[i].tokens == 0) {
            return Err("Output owner has no decoder rows".into());
        }
        Ok(view)
    }
    pub(super) fn suffix(&self, keep: usize) -> Result<Self> {
        if self.decode || !(128..=512).contains(&keep) {
            return Err("Invalid decoder bounded replay layout".into());
        }
        let mut view = self.clone();
        if let Some(lengths) = &mut view.work_shape {
            for length in lengths {
                *length = (*length).min(keep);
            }
        }
        for segment in &mut view.segments {
            segment.tokens = segment.tokens.min(keep);
        }
        view.rows = Rows::packet(
            &view.segments,
            self.segments.iter().map(|s| s.slot + 1).max().unwrap_or(1),
            self.rows.total,
        )?;
        view.remap_endpoints(self)?;
        Ok(view)
    }
    fn remap_endpoints(&mut self, source: &Self) -> Result<()> {
        for (&owner, position) in self.outputs.iter().zip(&mut self.output_positions) {
            *position = position
                .checked_sub(source.segments[owner].tokens - self.segments[owner].tokens)
                .ok_or("Verification output precedes the retained decoder window")?;
        }
        Ok(())
    }
    pub(super) fn row_map(
        &self,
        decoder: &Self,
        source: usize,
        destination: usize,
        rank: usize,
        contiguous: bool,
    ) -> Result<Vec<i32>> {
        if self.decode
            || decoder.segments.len() != self.segments.len()
            || self
                .segments
                .iter()
                .zip(&decoder.segments)
                .any(|(s, d)| s.slot != d.slot || d.tokens > s.tokens)
            || source == 0
            || destination == 0
            || rank >= destination
        {
            return Err("Invalid decoder row map partition".into());
        }
        let mut selected = vec![];
        for (i, segment) in self.segments.iter().enumerate() {
            let end = self.rows.starts[i] + segment.tokens;
            selected.extend(end - decoder.segments[i].tokens..end);
        }
        let width = selected.len().div_ceil(destination);
        let selected = if contiguous {
            selected
                .into_iter()
                .skip(rank * width)
                .take(width)
                .collect::<Vec<_>>()
        } else {
            selected
                .into_iter()
                .skip(rank)
                .step_by(destination)
                .collect::<Vec<_>>()
        };
        let mut result = selected
            .into_iter()
            .map(|index| {
                i32::try_from(index % source * self.rows.total.div_ceil(source) + index / source)
                    .map_err(|_| "Decoder row map overflow".into())
            })
            .collect::<Result<Vec<_>>>()?;
        result.resize(width, -1);
        Ok(result)
    }
    #[cfg(test)]
    fn table_key(&self, min_rows: usize) -> TableKey {
        self.table_key_for(min_rows, None)
    }
    fn table_key_for(&self, min_rows: usize, tail_min_rows: Option<usize>) -> TableKey {
        TableKey {
            family: (self.role, self.decode, min_rows),
            tail_min_rows,
            segments: self.segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            outputs: self.outputs.clone(),
            output_positions: self.output_positions.clone(),
            decoder_roles: self.decoder_roles.clone(),
            work_shape: self.work_shape.clone(),
            work_padding: work_pad(self),
        }
    }
}
/// Keep every rank's launch descriptors within its packet family. All ranks
/// use the same union; unrelated families cannot fragment a cache class.
pub(super) fn graph_launches(model: &super::Model) -> BTreeSet<WorkLaunch> {
    let Some(contract) = &model.batch else {
        return BTreeSet::new();
    };
    let mut launches = BTreeSet::new();
    for rank in &model.ranks {
        for packet in &rank.packets {
            for program in [&packet.body, &packet.head] {
                let mut decoder_domain = false;
                for step in program {
                    if matches!(step, Instruction::DecoderRows { .. }) {
                        decoder_domain = true;
                    }
                    if let Instruction::Kernel {
                        owner_work: Some(grid),
                        buffers,
                        ..
                    } = step
                    {
                        launches.insert(WorkLaunch {
                            role: packet.role,
                            family: (packet.batch_decode, packet.min_rows, packet.dispatch_max()),
                            grid: grid.clone(),
                            range: contract
                                .work_ranges
                                .get(&buffers[0])
                                .map(|r| (r.min_rows, r.max_rows)),
                            partition: contract
                                .partitioned_tables
                                .get(&buffers[0])
                                .copied()
                                .unwrap_or(1),
                            decoder: decoder_domain
                                || contract
                                    .decoder_replay
                                    .as_ref()
                                    .is_some_and(|r| r.tables.contains(&buffers[0])),
                        });
                    }
                }
            }
        }
    }
    launches
}
/// Graph nodes depend on launch extents, while tables carry owner identities,
/// exact lengths and roles. Preserve every owner-work launch, including rank
/// partitions and policy ranges, without specializing on its table contents.
fn shape_key(
    contract: &Contract,
    layout: &Layout,
    family: (super::PacketRole, bool, usize),
    launches: &BTreeSet<WorkLaunch>,
) -> Result<Key> {
    shape_key_with_scope(contract, layout, family, launches, true)
}
fn split_work_rows(
    contract: &Contract,
    layout: &Layout,
    family: (super::PacketRole, bool, usize),
    tail_family: (super::PacketRole, bool, usize),
    launches: &BTreeSet<WorkLaunch>,
) -> Result<(Option<usize>, Vec<usize>, Vec<usize>)> {
    if layout.outputs.len() > contract.output_capacity() {
        return Err("Output rows exceed the published head capacity".into());
    }
    let work_layout;
    let layout = if layout.work_shape.is_some() {
        work_layout = layout.work_layout()?;
        &work_layout
    } else {
        layout
    };
    let padded = work_pad(layout);
    let decoder = contract
        .decoder_replay
        .as_ref()
        .filter(|_| layout.role == super::PacketRole::Target && !layout.decode)
        .map(|r| layout.decoder(r.keep, r.final_prompt_only))
        .transpose()?;
    let mut partitions = BTreeMap::new();
    let mut trunk_work = Vec::new();
    let mut decoder_work = Vec::new();
    for launch in launches {
        let applicable = launch.family.0 == layout.decode
            && if layout.role == super::PacketRole::Draft {
                launch.family.1 == family.2
            } else if launch.decoder {
                launch.family.1 == tail_family.2
            } else {
                launch.family.1 == family.2
            };
        if launch.role != layout.role || !applicable {
            continue;
        }
        if launch.decoder && decoder.as_ref().is_none_or(|d| d.rows.total == 0) {
            continue;
        }
        let selected = if launch.decoder {
            decoder.as_ref().ok_or("Missing decoder work layout")?
        } else {
            layout
        };
        for rank in 0..launch.partition {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                partitions.entry((launch.decoder, launch.partition, rank))
            {
                entry.insert(if launch.partition == 1 {
                    selected
                        .segments
                        .iter()
                        .map(|s| s.tokens)
                        .collect::<Vec<_>>()
                } else {
                    selected.rows.partition(launch.partition, rank)?.lengths
                });
            }
            let lengths = &partitions[&(launch.decoder, launch.partition, rank)];
            let range = launch
                .range
                .map(|(min_rows, max_rows)| WorkRange { min_rows, max_rows });
            let count = lengths.iter().copied().try_fold(0usize, |sum, tokens| {
                sum.checked_add(work_blocks(tokens, &launch.grid, range.as_ref())?)
                    .ok_or_else(|| "Owner work extent overflow".to_string())
            })?;
            if count > 8192 {
                return Err("Owner work table exceeds its capacity".into());
            }
            let rows = work_capacity(count, padded) / 2;
            if launch.decoder {
                decoder_work.push(rows);
            } else {
                trunk_work.push(rows);
            }
        }
    }
    Ok((
        decoder.as_ref().map(|d| d.rows.total),
        trunk_work,
        decoder_work,
    ))
}

fn shape_key_with_scope(
    contract: &Contract,
    layout: &Layout,
    family: (super::PacketRole, bool, usize),
    launches: &BTreeSet<WorkLaunch>,
    exact_family: bool,
) -> Result<Key> {
    if layout.outputs.len() > contract.output_capacity() {
        return Err("Output rows exceed the published head capacity".into());
    }
    let work_layout;
    let layout = if layout.work_shape.is_some() {
        work_layout = layout.work_layout()?;
        &work_layout
    } else {
        layout
    };
    let padded = work_pad(layout);
    let decoder = contract
        .decoder_replay
        .as_ref()
        .filter(|_| layout.role == super::PacketRole::Target && !layout.decode)
        .map(|r| layout.decoder(r.keep, r.final_prompt_only))
        .transpose()?;
    let mut partitions = BTreeMap::new();
    let mut work_rows = Vec::with_capacity(launches.iter().map(|l| l.partition).sum());
    for launch in launches {
        let applicable = launch.family.0 == layout.decode
            && if layout.role == super::PacketRole::Draft {
                launch.family.1 == family.2
            } else {
                (launch.family.1..=launch.family.2).contains(&layout.rows.total)
            };
        if launch.role != layout.role || exact_family && !applicable {
            continue;
        }
        if launch.decoder && decoder.as_ref().is_none_or(|d| d.rows.total == 0) {
            continue;
        }
        let selected = if launch.decoder {
            decoder.as_ref().ok_or("Missing decoder work layout")?
        } else {
            layout
        };
        for rank in 0..launch.partition {
            if let std::collections::btree_map::Entry::Vacant(entry) =
                partitions.entry((launch.decoder, launch.partition, rank))
            {
                entry.insert(if launch.partition == 1 {
                    selected
                        .segments
                        .iter()
                        .map(|s| s.tokens)
                        .collect::<Vec<_>>()
                } else {
                    selected.rows.partition(launch.partition, rank)?.lengths
                });
            }
            let lengths = &partitions[&(launch.decoder, launch.partition, rank)];
            let range = launch
                .range
                .map(|(min_rows, max_rows)| WorkRange { min_rows, max_rows });
            let count = lengths.iter().copied().try_fold(0usize, |sum, tokens| {
                sum.checked_add(work_blocks(tokens, &launch.grid, range.as_ref())?)
                    .ok_or_else(|| "Owner work extent overflow".to_string())
            })?;
            if count > 8192 {
                return Err("Owner work table exceeds its capacity".into());
            }
            work_rows.push(work_capacity(count, padded) / 2);
        }
    }
    Ok(Key {
        family,
        rows: layout.rows.total,
        decoder_rows: decoder.as_ref().map(|d| d.rows.total),
        outputs: layout.outputs.len(),
        work_rows,
    })
}
/// Owner tables carry real work; unused lanes already use the inactive sentinel.
/// Retain a launch envelope across nearby owner-work counts so changing
/// prompt splits need not rebuild an otherwise identical model packet.
fn work_pad(layout: &Layout) -> WorkPad {
    // Decoder/suffix views inherit their source packet's launch envelope.
    // Recomputing it from their shorter row count would disagree with the
    // retained-work check and with the uploaded owner-work table.
    if layout.role == super::PacketRole::Target && !layout.decode {
        if layout.bucketed {
            WorkPad::Bucketed
        } else if layout.work_padding {
            WorkPad::Padded
        } else {
            WorkPad::Exact
        }
    } else {
        WorkPad::Exact
    }
}
fn work_capacity(count: usize, mode: WorkPad) -> usize {
    match mode {
        WorkPad::Exact => count.max(1).div_ceil(2) * 2,
        WorkPad::Padded => count.max(2).next_power_of_two().min(8192),
        // Bucketed packets use a 256-item (128-pair) floor so 1..=4 active
        // owners per rank partition share one canonical Trunk/Tail ACL Graph;
        // unused lanes hold the [-1, 0] sentinel and return immediately.
        WorkPad::Bucketed => count.max(256).next_power_of_two().min(8192),
    }
}
fn work_blocks(tokens: usize, grid: &RowExpression, range: Option<&WorkRange>) -> Result<usize> {
    if tokens == 0 || range.is_some_and(|r| tokens < r.min_rows || tokens > r.max_rows) {
        Ok(0)
    } else {
        Ok(grid.resolve(tokens)?.min(64))
    }
}

pub(super) fn work_items(
    layout: &Layout,
    grid: &RowExpression,
    range: Option<&WorkRange>,
) -> Result<Vec<[i32; 2]>> {
    work_lengths(layout.segments.iter().map(|s| s.tokens), grid, range)
}
fn work_lengths(
    lengths: impl IntoIterator<Item = usize>,
    grid: &RowExpression,
    range: Option<&WorkRange>,
) -> Result<Vec<[i32; 2]>> {
    let blocks = lengths
        .into_iter()
        .map(|tokens| work_blocks(tokens, grid, range))
        .collect::<Result<Vec<_>>>()?;
    let mut owners = blocks
        .into_iter()
        .enumerate()
        .filter(|(_, count)| *count > 0)
        .map(|(owner, count)| (owner, count, 0))
        .collect::<Vec<_>>();
    let count = owners.iter().map(|(_, count, _)| count).sum::<usize>();
    if count == 0 {
        return Ok(vec![[-1, 0], [-1, 0]]);
    }
    if count > 8192 {
        return Err("Owner work table exceeds its capacity".into());
    }
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let mut stride = 64;
    while gcd(stride, owners.len()) != 1 {
        stride -= 1;
    }
    let mut out = Vec::with_capacity(count + count % 2);
    let mut remaining = count;
    // Rotate owners across bounded grid slots. With 64 tiles per owner,
    // the coprime wave stride gives each owner one tile on every grid slot.
    'waves: for wave in 0..owners.len() {
        for slot in 0..64 {
            let index = (slot + wave * stride) % owners.len();
            let (owner, count, next) = &mut owners[index];
            if *next < *count {
                out.push([*owner as i32, *next as i32]);
                *next += 1;
                remaining -= 1;
                if remaining == 0 {
                    break 'waves;
                }
            }
        }
    }
    if out.len() % 2 != 0 {
        out.push([-1, 0]);
    }
    Ok(out)
}
impl Worker {
    pub(super) fn batch_split_cached(&self, layout: &Layout) -> Result<SplitStates> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        if let Some(split) = runtime.split_keys(layout)? {
            let state = |sub: Option<&Rc<SubProgram>>, consecutive: u8| {
                sub.map_or(CacheState::Missing, |e| {
                    if e.captured {
                        CacheState::Graph
                    } else {
                        CacheState::Direct {
                            uses: e.direct_uses.get(),
                            consecutive,
                        }
                    }
                })
            };
            let trunk = state(
                runtime.trunk_cache.get(&split.trunk),
                runtime.consecutive_trunk(&split.trunk),
            );
            let tail = split
                .tail
                .as_ref()
                .map(|k| state(runtime.tail_cache.get(k), runtime.consecutive_tail(k)));
            let head = split
                .head
                .as_ref()
                .map(|k| state(runtime.head_cache.get(k), runtime.consecutive_head(k)));
            return Ok(SplitStates { trunk, tail, head });
        }
        let key = runtime.key(layout)?;
        let state = runtime.cache.get(&key).map_or(CacheState::Missing, |e| {
            if e.all_captured() {
                CacheState::Graph
            } else {
                CacheState::Direct {
                    uses: e.trunk.direct_uses.get(),
                    consecutive: runtime.consecutive_uses(&key),
                }
            }
        });
        Ok(SplitStates::monolithic(state))
    }
    #[cfg(test)]
    pub(super) fn batch_cached(&self, layout: &Layout) -> Result<CacheState> {
        Ok(self.batch_split_cached(layout)?.summary())
    }
    #[cfg(test)]
    pub(super) fn invalidate_batch(&mut self, layout: &Layout) -> Result<()> {
        self.invalidate_parts(layout, true, true, true)
    }
    pub(super) fn invalidate_parts(
        &mut self,
        layout: &Layout,
        trunk: bool,
        tail: bool,
        head: bool,
    ) -> Result<()> {
        self.session.synchronize()?;
        let (split_keys, mono_key) = {
            let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
            if let Some(split) = runtime.split_keys(layout)? {
                if trunk {
                    runtime.remove_trunk(&split.trunk);
                    runtime.trunk_lru.retain(|k| k != &split.trunk);
                }
                if tail && let Some(k) = &split.tail {
                    runtime.remove_tail(k);
                    runtime.tail_lru.retain(|x| x != k);
                }
                if head && let Some(k) = &split.head {
                    runtime.remove_head(k);
                    runtime.head_lru.retain(|x| x != k);
                }
                (Some(split), None)
            } else {
                let key = runtime.key(layout)?;
                if trunk {
                    runtime.remove(&key);
                    runtime.lru.retain(|k| k != &key);
                }
                (None, Some(key))
            }
        };
        self.w4.retain(|_, w| w.strong_count() > 0);
        let protected = ProtectedKeys {
            mono: mono_key.as_ref().filter(|_| !trunk),
            trunk: split_keys.as_ref().filter(|_| !trunk).map(|s| &s.trunk),
            tail: split_keys
                .as_ref()
                .filter(|_| !tail)
                .and_then(|s| s.tail.as_ref()),
            head: split_keys
                .as_ref()
                .filter(|_| !head)
                .and_then(|s| s.head.as_ref()),
            keep_idle_owner: false,
        };
        self.reserve_batch_memory_protected(
            super::memory::Requirement::workspace(),
            "coordinated Graph preparation",
            &protected,
        )
    }
    #[cfg(test)]
    pub(super) fn prepare_batch(&mut self, layout: &Layout, capture: bool) -> Result<()> {
        let states = self.batch_split_cached(layout)?;
        let action = |state: CacheState| PartAction {
            invalidate: false,
            prepare: match state {
                CacheState::Missing => Some(capture),
                CacheState::Direct { .. } if capture => Some(true),
                _ => None,
            },
        };
        self.prepare_parts(
            layout,
            SplitPlan {
                trunk: action(states.trunk),
                tail: states.tail.map(action),
                head: states.head.map(action),
            },
        )
    }
    pub(super) fn prepare_parts(&mut self, layout: &Layout, plan: SplitPlan) -> Result<()> {
        let split = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .split_keys(layout)?;
        if let Some(split) = split {
            return self.prepare_split_parts(layout, split, plan);
        }
        let capture = plan.trunk.prepare.unwrap_or(false);
        let key = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .key(layout)?;
        let previous = self.batch.as_ref().unwrap().cache.get(&key).cloned();
        if previous
            .as_ref()
            .is_none_or(|e| capture && !e.all_captured())
        {
            let started = std::time::Instant::now();
            let before = (
                self.phases.warmups.1,
                self.phases.captures.1,
                self.phases.binds.1,
                self.phases.aclnn.1,
                self.phases.templates.1,
                self.phases.branches.1,
                self.phases.dependencies.1,
                self.phases.table_build_s,
                self.phases.table_upload_s,
                self.phases.template_paths_s,
                self.phases.kernel_geometry_s,
            );
            self.reserve_batch_memory_protected(
                super::memory::Requirement::workspace(),
                "Graph preparation",
                &ProtectedKeys {
                    mono: Some(&key),
                    ..ProtectedKeys::default()
                },
            )?;
            let reserve_s = started.elapsed().as_secs_f64();
            let packet_started = std::time::Instant::now();
            let packet = self
                .packet_templates
                .iter()
                .find(|p| {
                    p.role == layout.role
                        && p.batch_decode == layout.decode
                        && p.min_rows == key.family.2
                })
                .cloned()
                .ok_or("Missing batch family")?;
            let packet_s = packet_started.elapsed().as_secs_f64();
            let guard_started = std::time::Instant::now();
            let mut tables_s = 0.;
            if capture {
                // Every indirect state access checks this stream-ordered word.
                // Capturing a new shape cannot overwrite live KV or C2 rings.
                super::upload(
                    &self.session,
                    &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
                    &0i32.to_le_bytes(),
                )?;
                let tables_started = std::time::Instant::now();
                self.update_batch_tables(layout, &packet, None)?;
                tables_s = tables_started.elapsed().as_secs_f64();
            }
            let guard_s = guard_started.elapsed().as_secs_f64() - tables_s;
            if layout.rows.total > packet.rows {
                return Err("Batch exceeds its family row capacity".into());
            }
            let base = self.package.clone();
            let compile_started = std::time::Instant::now();
            let (body, head) = if let Some(previous) = &previous {
                (
                    self.capture_steps(&previous.trunk.steps)?,
                    self.capture_steps(previous.head_steps())?,
                )
            } else {
                let body = self.compile_bound(
                    &packet.body,
                    &base,
                    self.snapshots,
                    (packet.rows, layout.rows.total),
                    Some(layout),
                    capture,
                )?;
                let head = if layout.outputs.is_empty() {
                    vec![]
                } else {
                    self.compile_bound(
                        &packet.head,
                        &base,
                        false,
                        (packet.rows, layout.rows.total),
                        Some(layout),
                        capture,
                    )?
                };
                (body, head)
            };
            let compile_s = compile_started.elapsed().as_secs_f64();
            self.batch.as_mut().unwrap().cache.insert(
                key.clone(),
                Rc::new(Execution::monolithic(packet, body, head, capture)),
            );
            if self.rank == 0 && std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
                eprintln!(
                    "A2 packet prepare {}",
                    serde_json::json!({
                        "rows": key.rows, "decode": layout.decode, "capture": capture,
                        "promotion": previous.is_some(),
                        "family": format!("{:?}", key.family),
                        "direct_uses": previous.as_ref().map_or(0, |e| e.trunk.direct_uses.get()),
                        "consecutive_uses": self.batch.as_ref().unwrap().consecutive_uses(&key),
                        "owner_lengths": layout.segments.iter().map(|s| s.tokens).collect::<Vec<_>>(),
                        "decoder_rows": key.decoder_rows, "outputs": key.outputs,
                        "work_rows": key.work_rows, "wall_s": started.elapsed().as_secs_f64(),
                        "warmup_s": self.phases.warmups.1-before.0,
                        "capture_s": self.phases.captures.1-before.1,
                        "bind_s": self.phases.binds.1-before.2,
                        "aclnn_s": self.phases.aclnn.1-before.3,
                        "template_s": self.phases.templates.1-before.4,
                        "branch_s": self.phases.branches.1-before.5,
                        "dependency_s": self.phases.dependencies.1-before.6,
                        "reserve_s": reserve_s, "packet_s": packet_s,
                        "guard_s": guard_s, "tables_s": tables_s,
                        "table_build_s": self.phases.table_build_s-before.7,
                        "table_upload_s": self.phases.table_upload_s-before.8,
                        "compile_s": compile_s,
                        "template_paths_s": self.phases.template_paths_s-before.9,
                        "kernel_geometry_s": self.phases.kernel_geometry_s-before.10,
                    })
                );
            }
        }
        let runtime = self.batch.as_mut().unwrap();
        let protected = ProtectedKeys {
            mono: Some(&key),
            ..ProtectedKeys::default()
        };
        runtime.lru.retain(|k| k != &key);
        runtime.lru.push_back(key.clone());
        while runtime.cache.len() > runtime.capacity {
            let Some(old) = runtime.lru.pop_front() else {
                break;
            };
            runtime.remove(&old);
        }
        runtime.trim_total_programs(&protected);
        self.w4.retain(|_, w| w.strong_count() > 0);
        Ok(())
    }
    fn prepare_split_parts(
        &mut self,
        layout: &Layout,
        split: SplitKeys,
        plan: SplitPlan,
    ) -> Result<()> {
        let need_trunk = plan.trunk.prepare.filter(|&cap| {
            self.batch
                .as_ref()
                .unwrap()
                .trunk_cache
                .get(&split.trunk)
                .is_none_or(|e| cap && !e.captured)
        });
        let need_tail = split.tail.as_ref().and_then(|k| {
            plan.tail.and_then(|a| a.prepare).filter(|&cap| {
                self.batch
                    .as_ref()
                    .unwrap()
                    .tail_cache
                    .get(k)
                    .is_none_or(|e| cap && !e.captured)
            })
        });
        let need_head = split.head.as_ref().and_then(|k| {
            plan.head.and_then(|a| a.prepare).filter(|&cap| {
                self.batch
                    .as_ref()
                    .unwrap()
                    .head_cache
                    .get(k)
                    .is_none_or(|e| cap && !e.captured)
            })
        });
        if need_trunk.is_some() || need_tail.is_some() || need_head.is_some() {
            let started = std::time::Instant::now();
            let any_capture =
                need_trunk == Some(true) || need_tail == Some(true) || need_head == Some(true);
            self.reserve_batch_memory_protected(
                super::memory::Requirement::workspace(),
                "Graph preparation",
                &ProtectedKeys {
                    mono: None,
                    trunk: Some(&split.trunk),
                    tail: split.tail.as_ref(),
                    head: split.head.as_ref(),
                    keep_idle_owner: false,
                },
            )?;
            let packet = self
                .packet_templates
                .iter()
                .find(|p| {
                    p.role == layout.role
                        && p.batch_decode == layout.decode
                        && p.min_rows == split.family.2
                })
                .cloned()
                .ok_or("Missing batch family")?;
            let tail_packet = split
                .tail
                .as_ref()
                .filter(|k| k.family != split.family)
                .map(|k| {
                    self.packet_templates
                        .iter()
                        .find(|p| {
                            p.role == k.family.0
                                && p.batch_decode == k.family.1
                                && p.min_rows == k.family.2
                        })
                        .cloned()
                        .ok_or_else(|| "Missing tail batch family".to_string())
                })
                .transpose()?;
            if layout.rows.total > packet.rows {
                return Err("Batch exceeds its family row capacity".into());
            }
            if any_capture {
                super::upload(
                    &self.session,
                    &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
                    &0i32.to_le_bytes(),
                )?;
                self.update_batch_tables(layout, &packet, tail_packet.as_deref())?;
            }
            let trunk_used = &self.batch.as_ref().unwrap().tables[&split.family];
            let split_idx = trunk_used
                .decoder_split
                .ok_or("Missing decoder split index")?;
            let trunk_boundary = trunk_used
                .decoder_boundary
                .ok_or("Missing decoder boundary index")?;
            let base = self.package.clone();
            if let Some(capture) = need_trunk {
                let prev = self
                    .batch
                    .as_ref()
                    .unwrap()
                    .trunk_cache
                    .get(&split.trunk)
                    .cloned();
                let steps = if let Some(prev) = prev {
                    self.capture_steps(&prev.steps)?
                } else {
                    self.compile_bound(
                        &packet.body[..split_idx],
                        &base,
                        self.snapshots,
                        (packet.rows, layout.rows.total),
                        Some(layout),
                        capture,
                    )?
                };
                self.batch.as_mut().unwrap().trunk_cache.insert(
                    split.trunk.clone(),
                    Rc::new(SubProgram {
                        steps,
                        captured: capture,
                        direct_uses: std::cell::Cell::new(0),
                    }),
                );
            }
            if let (Some(tail_key), Some(capture)) = (&split.tail, need_tail) {
                let prev = self
                    .batch
                    .as_ref()
                    .unwrap()
                    .tail_cache
                    .get(tail_key)
                    .cloned();
                let steps = if let Some(prev) = prev {
                    self.capture_steps(&prev.steps)?
                } else if let Some(tp) = &tail_packet {
                    let tail_boundary = self.batch.as_ref().unwrap().tables[&tail_key.family]
                        .decoder_boundary
                        .ok_or("Missing tail decoder boundary index")?;
                    let prepare_len = trunk_boundary - split_idx;
                    let mut spliced =
                        Vec::with_capacity(prepare_len + (tp.body.len() - tail_boundary));
                    spliced.extend_from_slice(&packet.body[split_idx..trunk_boundary]);
                    let mut dr = tp.body[tail_boundary].clone();
                    if let Instruction::DecoderRows { prepare, .. } = &mut dr {
                        *prepare = prepare_len;
                    }
                    spliced.push(dr);
                    spliced.extend_from_slice(&tp.body[tail_boundary + 1..]);
                    self.compile_bound(
                        &spliced,
                        &base,
                        self.snapshots,
                        (packet.rows, layout.rows.total),
                        Some(layout),
                        capture,
                    )?
                } else {
                    self.compile_bound(
                        &packet.body[split_idx..],
                        &base,
                        self.snapshots,
                        (packet.rows, layout.rows.total),
                        Some(layout),
                        capture,
                    )?
                };
                self.batch.as_mut().unwrap().tail_cache.insert(
                    tail_key.clone(),
                    Rc::new(SubProgram {
                        steps,
                        captured: capture,
                        direct_uses: std::cell::Cell::new(0),
                    }),
                );
            }
            if let (Some(head_key), Some(capture)) = (&split.head, need_head) {
                let prev = self
                    .batch
                    .as_ref()
                    .unwrap()
                    .head_cache
                    .get(head_key)
                    .cloned();
                let steps = if let Some(prev) = prev {
                    self.capture_steps(&prev.steps)?
                } else {
                    self.compile_bound(
                        &packet.head,
                        &base,
                        false,
                        (packet.rows, layout.rows.total),
                        Some(layout),
                        capture,
                    )?
                };
                self.batch.as_mut().unwrap().head_cache.insert(
                    head_key.clone(),
                    Rc::new(SubProgram {
                        steps,
                        captured: capture,
                        direct_uses: std::cell::Cell::new(0),
                    }),
                );
            }
            if self.rank == 0 && std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
                eprintln!(
                    "A2 split prepare {}",
                    serde_json::json!({
                        "rows": layout.rows.total,
                        "decoder_rows": split.tail.as_ref().map(|k| k.decoder_rows),
                        "tail_family": split.tail.as_ref().map(|k| k.family.2),
                        "outputs": layout.outputs.len(),
                        "trunk": need_trunk, "tail": need_tail, "head": need_head,
                        "wall_s": started.elapsed().as_secs_f64(),
                    })
                );
            }
        }
        let runtime = self.batch.as_mut().unwrap();
        let cap = runtime.capacity;
        runtime.touch_trunk(split.trunk.clone());
        while runtime.trunk_cache.len() > cap {
            let Some(old) = runtime.trunk_lru.pop_front() else {
                break;
            };
            runtime.remove_trunk(&old);
        }
        if let Some(k) = &split.tail {
            runtime.touch_tail(k.clone());
            while runtime.tail_cache.len() > cap {
                let Some(old) = runtime.tail_lru.pop_front() else {
                    break;
                };
                runtime.remove_tail(&old);
            }
        }
        if let Some(k) = &split.head {
            runtime.touch_head(k.clone());
            while runtime.head_cache.len() > cap {
                let Some(old) = runtime.head_lru.pop_front() else {
                    break;
                };
                runtime.remove_head(&old);
            }
        }
        runtime.trim_total_programs(&ProtectedKeys {
            mono: None,
            trunk: Some(&split.trunk),
            tail: split.tail.as_ref(),
            head: split.head.as_ref(),
            keep_idle_owner: false,
        });
        self.w4.retain(|_, w| w.strong_count() > 0);
        Ok(())
    }
    fn update_batch_tables(
        &mut self,
        layout: &Layout,
        packet: &super::Packet,
        tail_packet: Option<&super::Packet>,
    ) -> Result<Vec<Buffer>> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let tail_override = tail_packet.filter(|tp| tp.min_rows != packet.min_rows);
        let key = layout.table_key_for(packet.min_rows, tail_override.map(|tp| tp.min_rows));
        if let Some(bound) = &runtime.bound_tables
            && bound.key == key
        {
            return Ok(bound.retained.clone());
        }
        let build_started = std::time::Instant::now();
        let used = runtime
            .tables
            .get(&(packet.role, packet.batch_decode, packet.min_rows))
            .ok_or("Missing packet table usage")?;
        let tail_used = tail_override
            .map(|tp| {
                runtime
                    .tables
                    .get(&(tp.role, tp.batch_decode, tp.min_rows))
                    .ok_or("Missing tail packet table usage")
            })
            .transpose()?;
        let col_active = |index: usize| {
            if let Some(tu) = tail_used {
                used.trunk_columns[index] || tu.tail_columns[index]
            } else {
                used.columns[index]
            }
        };
        let decoder = runtime
            .contract
            .decoder_replay
            .as_ref()
            .filter(|_| layout.role == super::PacketRole::Target && !layout.decode)
            .map(|r| layout.decoder(r.keep, r.final_prompt_only))
            .transpose()?;
        let mut aligned = BTreeMap::new();
        aligned.insert((false, 1, 1), layout.rows.partition(1, 0)?);
        if let Some(l) = &decoder {
            aligned.insert((true, 1, 1), l.rows.partition(1, 0)?);
        }
        for (index, column) in runtime.contract.columns.iter().enumerate() {
            if col_active(index) && column.row_bytes.is_some() {
                let stride = column.row_partition.unwrap_or(1);
                let alignment = column.row_alignment;
                if let std::collections::btree_map::Entry::Vacant(entry) =
                    aligned.entry((column.decoder && !layout.decode, stride, 1))
                {
                    let rows = if column.decoder && !layout.decode {
                        &decoder.as_ref().ok_or("Missing decoder rows")?.rows
                    } else {
                        &layout.rows
                    };
                    entry.insert(rows.partition(stride, self.rank as usize % stride)?);
                }
                if alignment > 1
                    && !aligned.contains_key(&(column.decoder && !layout.decode, stride, alignment))
                {
                    let view = aligned[&(column.decoder && !layout.decode, stride, 1)]
                        .aligned(alignment)?;
                    aligned.insert((column.decoder && !layout.decode, stride, alignment), view);
                }
            }
        }
        let mut values = Vec::new();
        let mut retained = vec![];
        for (lane, segment) in layout.segments.iter().enumerate() {
            let owner = runtime
                .owners
                .get(&segment.slot)
                .ok_or("Owner must be admitted before capture")?;
            values.extend_from_slice(&(segment.tokens as u64).to_le_bytes());
            for (index, column) in runtime.contract.columns.iter().enumerate() {
                let buffer = if !col_active(index) {
                    None
                } else if let Some(stride) = column.row_bytes {
                    let view = &aligned[&(
                        column.decoder && !layout.decode,
                        column.row_partition.unwrap_or(1),
                        column.row_alignment,
                    )];
                    if view.lengths[lane] == 0 {
                        None
                    } else {
                        Some(
                            get(&self.buffers, &column.buffer)?.slice(
                                view.starts[lane]
                                    .checked_mul(stride)
                                    .ok_or("Owner offset overflow")?,
                                view.lengths[lane]
                                    .checked_mul(stride)
                                    .ok_or("Owner span overflow")?,
                            )?,
                        )
                    }
                } else {
                    Some(get(&owner.buffers, &column.buffer)?)
                };
                let address = buffer.as_ref().map_or(0, |b| b.pointer() as usize as u64);
                values.extend_from_slice(&address.to_le_bytes());
                if let Some(buffer) = buffer {
                    retained.push(buffer);
                }
            }
        }
        let width = runtime.contract.columns.len() + 1;
        let addresses = values.as_chunks::<8>().0;
        let mut inputs = vec![];
        let mut pointer_names: Vec<&String> = if let Some(tu) = tail_used {
            used.trunk_pointers.union(&tu.tail_pointers).collect()
        } else {
            used.pointers.iter().collect()
        };
        pointer_names.sort_unstable();
        for name in pointer_names {
            let mapping = &runtime.contract.pointers[name];
            let is_decoder = runtime
                .contract
                .decoder_replay
                .as_ref()
                .is_some_and(|r| !layout.decode && r.tables.contains(name));
            let table_layout = if is_decoder {
                decoder.as_ref().ok_or("Missing decoder table rows")?
            } else {
                layout
            };
            let mut data = Vec::with_capacity(layout.segments.len() * (mapping.len() + 1) * 8);
            for owner in 0..layout.segments.len() {
                if let Some(&stride) = runtime.contract.partitioned_tables.get(name) {
                    data.extend_from_slice(
                        &(aligned[&(is_decoder, stride, 1)].lengths[owner] as u64).to_le_bytes(),
                    );
                } else {
                    data.extend_from_slice(
                        &(table_layout.segments[owner].tokens as u64).to_le_bytes(),
                    );
                }
                for &column in mapping {
                    data.extend_from_slice(&addresses[owner * width + column + 1]);
                }
            }
            inputs.push((get(&self.buffers, name)?.slice(0, data.len())?, data));
        }
        let mut work_names: Vec<&String> = if let Some(tu) = tail_used {
            used.trunk_work.union(&tu.tail_work).collect()
        } else {
            used.work.iter().collect()
        };
        work_names.sort_unstable();
        for name in work_names {
            let grid = &runtime.contract.work[name];
            let data = runtime
                .contract
                .work_items(layout, grid, name, self.rank as usize)?
                .iter()
                .flat_map(|p| p.iter().flat_map(|x| x.to_le_bytes()))
                .collect::<Vec<_>>();
            inputs.push((get(&self.buffers, name)?.slice(0, data.len())?, data));
        }
        if let Some(replay) = &runtime.contract.decoder_replay
            && let Some(selected) = &decoder
            && selected.rows.total < layout.rows.total
        {
            for map in &replay.maps {
                let data = layout
                    .row_map(
                        selected,
                        map.source_partition,
                        map.destination_partition,
                        self.rank as usize % map.destination_partition,
                        map.contiguous,
                    )?
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>();
                if !data.is_empty() {
                    inputs.push((get(&self.buffers, &map.buffer)?.slice(0, data.len())?, data));
                }
            }
        }
        let output_layout = decoder.as_ref().unwrap_or(layout);
        let data = output_layout
            .outputs
            .iter()
            .zip(&output_layout.output_positions)
            .flat_map(|(&i, &position)| {
                ((output_layout.rows.starts[i] + position) as i32).to_le_bytes()
            })
            .collect::<Vec<_>>();
        if !data.is_empty() {
            inputs.push((
                get(&self.buffers, &runtime.contract.endpoints)?.slice(0, data.len())?,
                data,
            ));
        }
        self.phases.table_build_s += build_started.elapsed().as_secs_f64();
        let upload_started = std::time::Instant::now();
        // Both metadata and its consumers use the compute stream. The upload
        // completion retains the slab and destinations through DMA completion.
        self.session.upload_inputs(&inputs)?;
        self.phases.table_upload_s += upload_started.elapsed().as_secs_f64();
        self.batch.as_mut().unwrap().bound_tables = Some(BoundTables {
            key,
            retained: retained.clone(),
        });
        Ok(retained)
    }
    fn trace_rings(&self, path: &std::path::Path, stage: &str) -> Result<()> {
        if self.rank != 0 {
            return Ok(());
        }
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
        for (id, owner) in &self.batch.as_ref().unwrap().owners {
            for (name, b) in &owner.buffers {
                if name.starts_with("cache.ring.") {
                    let data = self.session.download(b, 0, b.bytes())?;
                    std::fs::write(path.join(format!("{stage}-owner{id}-{name}.bin")), data)
                        .map_err(|e| e.to_string())?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn forward_batch(
        &mut self,
        layout: &Layout,
        tokens: &[Vec<u32>],
        readout: &super::team::Readout,
    ) -> Result<Vec<super::team::Output>> {
        if !readout.draws.is_empty() && self.sampling.is_none() {
            return Err("Device sampling requires a sampling program".into());
        }
        if tokens.len() != layout.segments.len()
            || tokens
                .iter()
                .zip(&layout.segments)
                .any(|(t, s)| t.len() != s.tokens)
        {
            return Err("Batch input differs from row layout".into());
        }
        let (split_keys, mono_key, execution) = {
            let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
            let split = runtime.split_keys(layout)?;
            let mono = if split.is_none() {
                Some(runtime.key(layout)?)
            } else {
                None
            };
            let execution = runtime
                .execution(layout, &self.packet_templates)?
                .ok_or("Batch must be prepared before forward")?;
            (split, mono, execution)
        };
        {
            let runtime = self.batch.as_mut().unwrap();
            if let Some(split) = &split_keys {
                runtime.touch_trunk(split.trunk.clone());
                if let Some(k) = &split.tail {
                    runtime.touch_tail(k.clone());
                }
                if let Some(k) = &split.head {
                    runtime.touch_head(k.clone());
                }
            } else if let Some(k) = &mono_key {
                runtime.touch_shared_packet(k.clone());
            }
            for segment in &layout.segments {
                if let Some(owner) = runtime.owners.get_mut(&segment.slot) {
                    owner.clean = false;
                }
            }
        }
        let retained =
            self.update_batch_tables(layout, &execution.packet, execution.tail_packet.as_deref())?;
        let device_input = if std::env::var_os("ORINFER_A2_BATCH_TRACE").is_none() {
            if let Some(mut runtime) = self.device_inputs.take() {
                let result = runtime.prepare(self, &execution.packet, layout, tokens, readout);
                self.device_inputs = Some(runtime);
                Some(result?)
            } else {
                None
            }
        } else {
            None
        };
        let device_coordinates = device_input.as_ref().is_some_and(|d| d.device_coordinates);
        if !readout.draws.is_empty()
            && layout.role == super::PacketRole::Draft
            && execution.packet.sampling_controls.is_none()
        {
            return Err("Draft packet has no coupled sampling controls".into());
        }
        let mut joined = vec![];
        let mut visual_rows = vec![];
        let mut hashes = vec![];
        let mut valid = vec![];
        let needs_hashes = execution
            .packet
            .body
            .iter()
            .chain(&execution.packet.head)
            .any(|step| matches!(step, super::Instruction::Lookup { .. }));
        let mut inputs = vec![];
        if let Some(name) = &execution.packet.sampling_controls {
            let stride = execution
                .packet
                .draft_stride
                .ok_or("Sampling draft requires a physical owner stride")?;
            let mut controls = vec![0u8; layout.segments.len() * stride * 16];
            for (lane, segment) in layout.segments.iter().enumerate() {
                if let Some(draw) = readout.draws.get(&segment.slot) {
                    let bytes = draw.controls(self.rank * 16160, segment.tokens)?;
                    let start = lane * stride * 16;
                    controls[start..start + bytes.len()].copy_from_slice(&bytes);
                }
            }
            inputs.push((
                get(&self.buffers, name)?.slice(0, controls.len())?,
                controls,
            ));
        }
        let image_input = |name: Option<&String>| -> Result<Option<(Buffer, Vec<u8>)>> {
            name.map(|name| {
                let buffer = get(&self.buffers, name)?;
                let data = vec![0u8; buffer.bytes()];
                Ok((buffer, data))
            })
            .transpose()
        };
        // Include the padded sequence rows. Every rank reads its existing
        // contiguous view; an old image flag must not survive in a dummy row.
        let mut source_images = image_input(execution.packet.coordinates.image_rows.as_ref())?;
        let mut decoder_images = image_input(
            execution
                .packet
                .decoder_coordinates
                .as_ref()
                .and_then(|c| c.image_rows.as_ref()),
        )?;
        let decoder = execution
            .packet
            .decoder_coordinates
            .as_ref()
            .map(|_| -> Result<_> {
                let replay = self
                    .batch
                    .as_ref()
                    .unwrap()
                    .contract
                    .decoder_replay
                    .as_ref()
                    .ok_or("Missing decoder replay contract")?;
                Ok((
                    layout.decoder(replay.keep, replay.final_prompt_only)?,
                    replay.keep,
                    replay.final_prompt_only,
                ))
            })
            .transpose()?;
        if let Some(block) = execution
            .packet
            .coordinates
            .draft_block
            .filter(|_| device_coordinates)
        {
            let names = &execution.packet.coordinates;
            for (ratio, indices, lengths) in [
                (1, &names.c1_indices, &names.c1_lengths),
                (2, &names.c2_indices, &names.c2_lengths),
            ] {
                let width = 128 / ratio;
                let mut ids = vec![-1i32; layout.rows.total * width];
                for row in ids.chunks_mut(width) {
                    for (i, item) in row.iter_mut().take(block).enumerate() {
                        *item = i as i32;
                    }
                }
                let bytes = ids
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>();
                inputs.push((get(&self.buffers, indices)?.slice(0, bytes.len())?, bytes));
                let bytes = vec![block as i32; layout.rows.total]
                    .into_iter()
                    .flat_map(i32::to_le_bytes)
                    .collect::<Vec<_>>();
                inputs.push((get(&self.buffers, lengths)?.slice(0, bytes.len())?, bytes));
            }
        }
        for (i, (segment, input)) in layout.segments.iter().zip(tokens).enumerate() {
            let history = &self.batch.as_ref().unwrap().owners[&segment.slot].history;
            if !device_coordinates {
                inputs.extend(self.coordinate_inputs(
                    &execution.packet.coordinates,
                    history.len(),
                    segment.tokens,
                    layout.rows.starts[i],
                    None,
                )?);
            }
            if let Some((_, data)) = &mut source_images {
                super::vision::write_mask(
                    data,
                    layout.rows.starts[i],
                    &self.batch.as_ref().unwrap().owners[&segment.slot].vision,
                    history.len(),
                    segment.tokens,
                )?;
            }
            if let Some(names) = &execution.packet.decoder_coordinates {
                let (suffix, keep, final_prompt_only) = decoder.as_ref().unwrap();
                let (keep, final_prompt_only) = (*keep, *final_prompt_only);
                let count = suffix.segments[i].tokens;
                let skipped = segment.tokens - count;
                if final_prompt_only
                    && layout.decoder_roles[i] == super::replay::Role::Final
                    && count < keep
                    && history.len() + segment.tokens > keep
                {
                    return Err(
                        "Final decoder packet must contain the complete retained prompt window"
                            .into(),
                    );
                }
                if count > 0 {
                    if let Some((_, data)) = &mut decoder_images {
                        super::vision::write_mask(
                            data,
                            suffix.rows.starts[i],
                            &self.batch.as_ref().unwrap().owners[&segment.slot].vision,
                            history.len() + skipped,
                            count,
                        )?;
                    }
                    if !device_coordinates {
                        inputs.extend(self.coordinate_inputs(
                            names,
                            history.len() + skipped,
                            count,
                            suffix.rows.starts[i],
                            Some(
                                if skipped > 0
                                    || final_prompt_only
                                        && layout.decoder_roles[i] == super::replay::Role::Final
                                {
                                    history.len() + skipped
                                } else {
                                    0
                                },
                            ),
                        )?);
                    }
                }
            }
            if needs_hashes {
                let mut owner_hashes = vec![];
                let mut owner_valid = vec![];
                self.hasher
                    .hash_into(history, input, &mut owner_hashes, &mut owner_valid)?;
                hashes.extend(owner_hashes);
                valid.extend(owner_valid);
            }
            if layout.role == super::PacketRole::Target {
                for span in self.batch.as_ref().unwrap().owners[&segment.slot]
                    .vision
                    .iter()
                {
                    let start = history.len().max(span.start);
                    let end = (history.len() + input.len()).min(span.end());
                    if start < end {
                        visual_rows.push(super::vision::Overlay {
                            row: joined.len() + start - history.len(),
                            source_row: start - span.start,
                            rows: end - start,
                            data: std::sync::Arc::clone(&span.data),
                        });
                    }
                }
            }
            joined.extend_from_slice(input);
        }
        inputs.extend(source_images);
        inputs.extend(decoder_images);
        inputs.push((
            get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
            1i32.to_le_bytes().to_vec(),
        ));
        if let Some(name) = &execution.packet.coordinates.input_ids
            && device_input.is_none()
        {
            let bytes = joined
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>();
            inputs.push((get(&self.buffers, name)?.slice(0, bytes.len())?, bytes));
        }
        self.session.upload_inputs(&inputs)?;
        let sampled = if let Some(trace) = std::env::var_os("ORINFER_A2_BATCH_TRACE") {
            let name = layout
                .segments
                .iter()
                .map(|s| {
                    format!(
                        "{}-{}-{}",
                        s.slot,
                        self.batch.as_ref().unwrap().owners[&s.slot].history.len(),
                        s.tokens
                    )
                })
                .collect::<Vec<_>>()
                .join("_");
            let path = std::path::PathBuf::from(trace).join(name);
            self.trace_rings(&path, "before")?;
            self.trace_packet(
                &execution,
                &joined,
                &hashes,
                &valid,
                !layout.outputs.is_empty(),
                &path,
                &visual_rows,
            )?;
            self.trace_rings(&path, "after")?;
            self.sample_batch(&execution, layout, readout)?
        } else {
            let pending = self.enqueue_packet_retained(
                Rc::clone(&execution),
                super::packet::Inputs {
                    tokens: &joined,
                    hashes: &hashes,
                    valid: &valid,
                    visual_rows: &visual_rows,
                    device: device_input,
                    first: readout.first_input.as_deref(),
                    next: readout.next_input.as_deref(),
                },
                !layout.outputs.is_empty(),
                retained,
            )?;
            let sampled = readout.parameters.get().and_then(|_| {
                // Model, sampling and readback are ordered on the same stream.
                self.sample_batch(&execution, layout, readout)
            });
            if layout.role == super::PacketRole::Draft
                && let Some(mut runtime) = self.device_inputs.take()
            {
                let result = runtime.save_draft(self, &execution.packet, layout);
                self.device_inputs = Some(runtime);
                result?;
            }
            // Errors still drain every rank's submitted collective order. On
            // success the sampling readback has already fenced the model.
            pending.wait()?;
            sampled?
        };
        if !execution.all_captured() {
            execution.record_shared_use();
        }
        if readout.store_draft_distributions {
            if layout.role != super::PacketRole::Draft {
                return Err("Only a draft packet can populate proposal distributions".into());
            }
            let source = get(
                &self.buffers,
                execution
                    .packet
                    .logits
                    .as_ref()
                    .ok_or("Missing draft logits")?,
            )?;
            let owners = layout
                .segments
                .iter()
                .map(|s| (s.slot, s.tokens))
                .collect::<Vec<_>>();
            self.sampling
                .as_mut()
                .ok_or("Missing device sampler")?
                .store_proposals(
                    &self.session,
                    &source,
                    execution
                        .packet
                        .draft_stride
                        .ok_or("Missing draft stride")?,
                    &owners,
                )?;
        }
        let mut output = vec![];
        if !layout.outputs.is_empty() {
            let count = execution
                .packet
                .draft_stride
                .map_or(layout.outputs.len(), |stride| {
                    layout.segments.len() * stride
                });
            let (values, indices) = if sampled.is_some() {
                (vec![], vec![])
            } else {
                let values = get(
                    &self.buffers,
                    execution
                        .packet
                        .confidence
                        .as_ref()
                        .unwrap_or(&execution.packet.best_value),
                )?;
                let indices = get(&self.buffers, &execution.packet.best_index)?;
                let mut outputs = self
                    .session
                    .download_compute_many(&[(&values, 0, count * 4), (&indices, 0, count * 4)])?
                    .into_iter();
                (outputs.next().unwrap(), outputs.next().unwrap())
            };
            let outputs = if let Some(stride) = execution.packet.draft_stride {
                layout
                    .segments
                    .iter()
                    .enumerate()
                    .flat_map(|(lane, segment)| {
                        (0..segment.tokens).map(move |position| (lane * stride + position, lane))
                    })
                    .collect::<Vec<_>>()
            } else {
                layout
                    .outputs
                    .iter()
                    .enumerate()
                    .map(|(i, &lane)| (i, lane))
                    .collect()
            };
            let mut logits = BTreeMap::new();
            let ranges = readout.logit_ranges(
                outputs
                    .iter()
                    .map(|&(row, lane)| (row, layout.segments[lane].slot)),
            );
            if !ranges.is_empty() {
                let source = get(
                    &self.buffers,
                    execution
                        .packet
                        .logits
                        .as_ref()
                        .ok_or("Missing batch logits")?,
                )?;
                let total_bytes = ranges
                    .iter()
                    .try_fold(0usize, |sum, r| sum.checked_add(r.len() * 16160 * 4))
                    .ok_or("Logit readback extent overflow")?;
                let chunks = if total_bytes <= self.session.pinned_limit() {
                    let spans = ranges
                        .iter()
                        .map(|r| (&source, r.start * 16160 * 4, r.len() * 16160 * 4))
                        .collect::<Vec<_>>();
                    self.session.download_compute_many(&spans)?
                } else {
                    ranges
                        .iter()
                        .map(|r| {
                            self.session
                                .download(&source, r.start * 16160 * 4, r.len() * 16160 * 4)
                        })
                        .collect::<Result<Vec<_>>>()?
                };
                for (range, bytes) in ranges.into_iter().zip(chunks) {
                    for (offset, data) in bytes.as_chunks::<{ 16160 * 4 }>().0.iter().enumerate() {
                        logits.insert(
                            range.start + offset,
                            data.as_chunks::<4>()
                                .0
                                .iter()
                                .map(|v| f32::from_le_bytes(*v))
                                .collect::<Vec<_>>(),
                        );
                    }
                }
            }
            for (i, lane) in outputs {
                let (value, index) = if let Some(sampled) = &sampled {
                    let sample = sampled.get(i).ok_or("Missing sampled row")?;
                    (sample.score, sample.token as i32)
                } else {
                    (
                        f32::from_le_bytes(values[i * 4..i * 4 + 4].try_into().unwrap()),
                        i32::from_le_bytes(indices[i * 4..i * 4 + 4].try_into().unwrap()),
                    )
                };
                let vocabulary = if execution.packet.replicated_output {
                    129280
                } else {
                    16160
                };
                if !(value.is_finite() || (sampled.is_some() && value == f32::NEG_INFINITY))
                    || !(0..vocabulary).contains(&index)
                {
                    return Err("Invalid batched output".into());
                }
                let probs = logits.remove(&i).unwrap_or_default();
                output.push(super::team::Output {
                    owner: layout.segments[lane].slot,
                    value,
                    token: execution.packet.vocab_first + index as u32,
                    logits: probs,
                    summary: sampled.as_ref().and_then(|s| s[i].summary.clone()),
                    report: None,
                    error: None,
                });
            }
        }
        for (segment, input) in layout.segments.iter().zip(tokens).filter(|(s, _)| {
            layout.role == super::PacketRole::Target && layout.pad_slot != Some(s.slot)
        }) {
            self.batch
                .as_mut()
                .unwrap()
                .owners
                .get_mut(&segment.slot)
                .unwrap()
                .history
                .extend_from_slice(input);
        }
        if let Some(split) = split_keys {
            if !execution.trunk.captured {
                execution
                    .trunk
                    .direct_uses
                    .set(execution.trunk.direct_uses.get().saturating_add(1));
            }
            let runtime = self.batch.as_mut().unwrap();
            runtime.record_trunk_use(split.trunk);
            if let (Some(tail_key), Some(tail)) = (split.tail, &execution.tail) {
                if !tail.captured {
                    tail.direct_uses
                        .set(tail.direct_uses.get().saturating_add(1));
                }
                runtime.record_tail_use(tail_key);
            }
            if let (Some(head_key), Some(head)) = (split.head, &execution.head) {
                if !head.captured {
                    head.direct_uses
                        .set(head.direct_uses.get().saturating_add(1));
                }
                runtime.record_head_use(head_key);
            }
        } else if let Some(key) = mono_key {
            if !execution.all_captured() {
                execution
                    .trunk
                    .direct_uses
                    .set(execution.trunk.direct_uses.get().saturating_add(1));
            }
            self.batch.as_mut().unwrap().record_use(key);
        }
        Ok(output)
    }
    fn sample_batch(
        &mut self,
        execution: &Execution,
        layout: &Layout,
        readout: &super::team::Readout,
    ) -> Result<Option<Vec<crate::backend::ascend::sampling::Sample>>> {
        if layout.outputs.is_empty()
            || readout.draws.is_empty()
            || layout.role == super::PacketRole::Draft
        {
            if layout.role == super::PacketRole::Target
                && let Some(runtime) = &mut self.device_inputs
            {
                for segment in &layout.segments {
                    runtime.invalidate(segment.slot)?;
                }
            }
            return Ok(None);
        }
        let source = get(
            &self.buffers,
            execution
                .packet
                .logits
                .as_ref()
                .ok_or("Missing sampling logits")?,
        )?;
        if !readout.rejection.is_empty() {
            if let Some(runtime) = &mut self.device_inputs {
                for segment in &layout.segments {
                    runtime.invalidate(segment.slot)?;
                }
            }
            if readout
                .row_parameters(layout)?
                .iter()
                .any(crate::backend::ascend::sampling::Parameters::processed)
            {
                return Err("Probability rejection does not accept processed rows".into());
            }
            let (controls, proposals) = readout.rejection_rows(layout)?;
            return Ok(Some(self.sampling.as_mut().unwrap().sample_rejection(
                &self.session,
                &self.comm,
                &source,
                execution.packet.vocab_first,
                (&controls, &proposals),
                execution.all_captured(),
            )?));
        }
        let sampled = self.sampling.as_mut().unwrap().enqueue_sample(
            &self.session,
            &source,
            execution.packet.vocab_first,
            &readout.controls(layout)?,
            &readout.row_parameters(layout)?,
            execution.all_captured(),
        )?;
        // Return the already-produced local samples while device continuation
        // preparation proceeds on the same stream after this readback fence.
        let pending = sampled
            .is_none()
            .then(|| {
                self.sampling
                    .as_ref()
                    .unwrap()
                    .begin_read_samples(&self.session, layout.outputs.len())
            })
            .transpose()?;
        if let Some(mut runtime) = self.device_inputs.take() {
            let result =
                if sampled.is_none() && std::env::var_os("ORINFER_A2_BATCH_TRACE").is_none() {
                    let samples = self.sampling.as_ref().unwrap().output().clone();
                    runtime.commit(self, layout, &samples, readout)
                } else {
                    layout
                        .segments
                        .iter()
                        .try_for_each(|s| runtime.invalidate(s.slot))
                };
            self.device_inputs = Some(runtime);
            result?;
        }
        Ok(Some(match sampled {
            Some(sampled) => sampled,
            None => self
                .sampling
                .as_ref()
                .unwrap()
                .finish_read_samples(&self.session, pending.unwrap())?,
        }))
    }
}

pub(super) fn validate(model: &super::Model) -> Result<()> {
    let Some(c) = &model.batch else { return Ok(()) };
    if !(1..=128).contains(&c.owners)
        || c.columns.is_empty()
        || model.ranks.iter().any(|r| r.packets.len() < 2)
    {
        return Err("Invalid batch contract".into());
    }
    let valid_partition = |stride: usize| stride > 1 && model.ranks.len().is_multiple_of(stride);
    if c.partitioned_tables.iter().any(|(name, &stride)| {
        !valid_partition(stride) || (!c.pointers.contains_key(name) && !c.work.contains_key(name))
    }) {
        return Err("Invalid owner row partition table".into());
    }
    for rank in &model.ranks {
        let buffers = rank
            .buffers
            .iter()
            .map(|b| (b.name.as_str(), b))
            .collect::<BTreeMap<_, _>>();
        super::replay::validate_contract(c, &buffers, model.ranks.len())?;
        for column in &c.columns {
            let b = buffers
                .get(column.buffer.as_str())
                .ok_or("Unknown owner column")?;
            if b.weight.is_some()
                || b.data.is_some()
                || b.prepared_weight.is_some()
                || b.prepared_tensor.is_some()
                || column.row_bytes == Some(0)
                || !column.row_alignment.is_power_of_two()
                || column.row_alignment > 256
                || column
                    .row_partition
                    .is_some_and(|n| !valid_partition(n) || column.row_bytes.is_none())
            {
                return Err("Owner column cannot own immutable weights".into());
            }
            if column.row_bytes.is_none() && !c.private.contains(&column.buffer) {
                return Err("Fixed owner column is not private".into());
            }
        }
        let unique = c.private.iter().collect::<std::collections::BTreeSet<_>>();
        if unique.len() != c.private.len() {
            return Err("Duplicate private state role".into());
        }
        let mut state_roles = BTreeSet::new();
        for state in c
            .state
            .iter()
            .chain(c.decoder_replay.iter().flat_map(|r| &r.decoder_state))
        {
            if !state_roles.insert(&state.buffer) || !c.private.contains(&state.buffer) {
                return Err("Snapshot state must name unique owner-private buffers".into());
            }
            let buffer = buffers
                .get(state.buffer.as_str())
                .ok_or("Missing snapshot state buffer")?;
            if buffer.view.is_some()
                || state.tokens_per_row == 0 && (state.circular || state.window.is_some())
                || state.circular && state.window.is_none()
            {
                return Err("Invalid snapshot state geometry".into());
            }
            state.spans(model.capacity, buffer.bytes)?;
        }
        let mut parents = std::collections::BTreeSet::new();
        for (name, window) in &c.verification_windows {
            if !state_roles.contains(name) {
                return Err("Verification window must name persistent owner state".into());
            }
            window.overwritten_spans(0, c.draft_capacity() + 1, buffers[name.as_str()].bytes)?;
        }
        for name in &c.private {
            let b = buffers
                .get(name.as_str())
                .ok_or("Missing private owner state")?;
            if b.weight.is_some()
                || b.data.is_some()
                || b.prepared_weight.is_some()
                || b.prepared_tensor.is_some()
                || b.view
                    .as_ref()
                    .is_some_and(|v| !parents.contains(&v.parent))
            {
                return Err("Invalid private allocation/view order".into());
            }
            parents.insert(name.clone());
        }
        if rank
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Target && p.batch_decode)
            .count()
            != 1
            || rank
                .packets
                .iter()
                .filter(|p| p.role == super::PacketRole::Target)
                .all(|p| p.batch_decode)
        {
            return Err("Batch requires one decode and complete prefill dispatch".into());
        }
        let prefill = rank
            .packets
            .iter()
            .filter(|p| p.role == super::PacketRole::Target && !p.batch_decode)
            .collect::<Vec<_>>();
        let capacity = prefill[0].rows;
        super::validate_row_ranges(&prefill, 1, Some(capacity))?;
        c.verification_rows(1, capacity)?;
        if !(1..=capacity).contains(&c.preferred_span()) {
            return Err("Invalid prefill scheduling quantum".into());
        }
        for column in &c.columns {
            match (column.row_bytes, column.decode) {
                (None, None) => {}
                (Some(stride), Some(decode)) => {
                    let p = rank
                        .packets
                        .iter()
                        .find(|p| p.role == column.role && p.batch_decode == decode)
                        .ok_or("Missing owner column packet family")?;
                    if p.rows
                        .div_ceil(column.row_partition.unwrap_or(1))
                        .checked_add(c.owners * (column.row_alignment - 1))
                        .and_then(|q| stride.checked_mul(q))
                        != Some(buffers[column.buffer.as_str()].bytes)
                    {
                        return Err("Owner column physical row stride differs".into());
                    }
                }
                _ => return Err("Owner row column requires a family".into()),
            }
        }
        for (name, grid) in &c.work {
            if buffers.get(name.as_str()).map(|b| b.bytes) != Some(8192 * 2 * 4) {
                return Err("Invalid owner work capacity".into());
            }
            grid.resolve(1)?;
        }
        if c.work_ranges.iter().any(|(name, r)| {
            !c.work.contains_key(name)
                || r.min_rows == 0
                || r.min_rows > r.max_rows
                || r.max_rows > 4096
        }) {
            return Err("Invalid owner-local work selection".into());
        }
        if c.output_capacity() < c.owners
            || buffers.get(c.endpoints.as_str()).map(|b| b.bytes) != Some(c.output_capacity() * 4)
        {
            return Err("Invalid owner output endpoints".into());
        }
        if c.pointers.is_empty() || buffers.get(c.active.as_str()).map(|b| b.bytes) != Some(4) {
            return Err("Invalid owner table shape".into());
        }
        for (name, mapping) in &c.pointers {
            if mapping.is_empty()
                || mapping.iter().any(|&i| i >= c.columns.len())
                || buffers.get(name.as_str()).map(|b| b.bytes)
                    != Some(c.owners * (mapping.len() + 1) * 8)
            {
                return Err("Invalid owner pointer mapping".into());
            }
            let partition = c.partitioned_tables.get(name).copied().unwrap_or(1);
            if mapping.iter().any(|&i| {
                c.columns[i].row_bytes.is_some()
                    && c.columns[i].row_partition.unwrap_or(1) != partition
            }) {
                return Err("Owner pointer table mixes row partitions".into());
            }
        }
        for p in &rank.packets {
            if p.role == super::PacketRole::Draft
                && (p.coordinates.draft_block.is_none_or(|block| {
                    !(1..=crate::execution::MAX_MTP_DRAFTS).contains(&block)
                        || p.dispatch_max() != c.owners * block
                        || c.output_capacity() < c.owners * (block + 1)
                        || p.draft_stride
                            .is_some_and(|stride| stride < block || p.rows != c.owners * stride)
                        || p.draft_stride.is_none() && p.rows != c.owners * block
                        || p.draft_stride.is_some() && p.min_rows != block
                }) || p.coordinates.input_ids.is_none()
                    || !p.replicated_output
                    || p.confidence.is_none()
                    || p.vocab_first != 0)
            {
                return Err("Invalid DSpark draft packet contract".into());
            }
            if p.min_rows == 0
                || p.min_rows > p.dispatch_max()
                || p.dispatch_max() > p.rows
                || (p.batch_decode && p.rows != c.owners)
                || p.reset.iter().any(|n| !unique.contains(n))
            {
                return Err("Invalid batch family/reset contract".into());
            }
            for s in p.body.iter().chain(&p.head) {
                if let Instruction::Kernel {
                    owner_work: Some(grid),
                    buffers,
                    ..
                } = s
                {
                    if buffers.len() < 3
                        || c.partitioned_tables.get(&buffers[0])
                            != c.partitioned_tables.get(&buffers[1])
                    {
                        return Err("Owner work and pointers select different rows".into());
                    }
                    for q in [1, p.rows] {
                        grid.resolve(q)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
