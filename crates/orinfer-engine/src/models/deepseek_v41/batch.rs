//! Owner-private state and shared token rows for native A2 packets.
mod memory;
use super::{Buffer, Execution, Instruction, Worker, get};
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
            let capacity = self.work_items(&reference, grid, name, rank)?.len();
            if items.len() > capacity {
                return Err("Owner work exceeds its retained Graph capacity".into());
            }
            items.resize(capacity, [-1, 0]);
        }
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
    pub buffers: BTreeMap<String, Buffer>,
    pub history: Vec<u32>,
    pub limit: Option<usize>,
    pub vision: std::sync::Arc<Vec<super::vision::Span>>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheState {
    Missing,
    Direct { uses: u8, consecutive: u8 },
    Graph,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    family: (super::PacketRole, bool, usize),
    rows: usize,
    decoder_rows: Option<usize>,
    outputs: usize,
    work_rows: Vec<usize>,
}
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct WorkLaunch {
    role: super::PacketRole,
    grid: RowExpression,
    range: Option<(usize, usize)>,
    partition: usize,
    decoder: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct TableKey {
    family: (super::PacketRole, bool, usize),
    segments: Vec<(usize, usize)>,
    outputs: Vec<usize>,
    output_positions: Vec<usize>,
    decoder_roles: Vec<super::replay::Role>,
    work_shape: Option<Vec<usize>>,
}
pub(super) struct BoundTables {
    key: TableKey,
    retained: Vec<Buffer>,
}
pub(super) struct Runtime {
    pub contract: Contract,
    pub owners: BTreeMap<usize, Owner>,
    pub(super) cache: BTreeMap<Key, Rc<Execution>>,
    recent: BTreeMap<(super::PacketRole, bool, usize), (Key, u8)>,
    lru: VecDeque<Key>,
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
    pointers: BTreeSet<String>,
    work: BTreeSet<String>,
    columns: Vec<bool>,
}
impl Runtime {
    pub(super) fn touch_shared_packet(&mut self, key: Key) {
        self.lru.retain(|k| k != &key);
        self.lru.push_back(key);
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
    pub(super) fn key(&self, layout: &Layout) -> Result<Key> {
        let (family, _) = self
            .families
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
                    .then_some((*family, used))
            })
            .ok_or("Missing batch family")?;
        shape_key(&self.contract, layout, family, &self.launches)
    }
    pub fn new(
        contract: Contract,
        capacity: usize,
        packets: &[super::Packet],
        launches: &BTreeSet<WorkLaunch>,
    ) -> Self {
        let mut tables = BTreeMap::new();
        for packet in packets {
            let mut used = TableUse {
                max_rows: packet.dispatch_max(),
                draft_block: packet.coordinates.draft_block,
                draft_stride: packet.draft_stride,
                pointers: BTreeSet::new(),
                work: BTreeSet::new(),
                columns: vec![false; contract.columns.len()],
            };
            for step in packet.body.iter().chain(&packet.head) {
                if let Instruction::Kernel {
                    owner_work: Some(_),
                    buffers,
                    ..
                } = step
                {
                    used.work.insert(buffers[0].clone());
                    used.pointers.insert(buffers[1].clone());
                    for &column in &contract.pointers[&buffers[1]] {
                        used.columns[column] = true;
                    }
                }
            }
            tables.insert((packet.role, packet.batch_decode, packet.min_rows), used);
        }
        Self {
            contract,
            owners: BTreeMap::new(),
            cache: BTreeMap::new(),
            recent: BTreeMap::new(),
            lru: VecDeque::new(),
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
    work_shape: Option<Vec<usize>>,
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
        Ok(Self {
            role: super::PacketRole::Target,
            output_positions: outputs.iter().map(|&i| segments[i].tokens - 1).collect(),
            segments,
            rows,
            outputs,
            decode,
            decoder_roles,
            work_shape: None,
        })
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
        let actual = shape_key(contract, self, family, launches)?;
        let retained = shape_key(contract, reference, family, launches)?;
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
    fn table_key(&self, min_rows: usize) -> TableKey {
        TableKey {
            family: (self.role, self.decode, min_rows),
            segments: self.segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            outputs: self.outputs.clone(),
            output_positions: self.output_positions.clone(),
            decoder_roles: self.decoder_roles.clone(),
            work_shape: self.work_shape.clone(),
        }
    }
}
/// Use a common union of every rank/family launch descriptor, so equivalent
/// layouts form the same cache class on every TP rank, including future tunes.
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
    let decoder = contract
        .decoder_replay
        .as_ref()
        .filter(|_| layout.role == super::PacketRole::Target && !layout.decode)
        .map(|r| layout.decoder(r.keep, r.final_prompt_only))
        .transpose()?;
    let mut partitions = BTreeMap::new();
    let mut work_rows = Vec::with_capacity(launches.iter().map(|l| l.partition).sum());
    for launch in launches {
        if launch.role != layout.role {
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
            work_rows.push(count.max(1).div_ceil(2));
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
    pub(super) fn batch_cached(&self, layout: &Layout) -> Result<CacheState> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let key = runtime.key(layout)?;
        Ok(runtime.cache.get(&key).map_or(CacheState::Missing, |e| {
            if e.captured {
                CacheState::Graph
            } else {
                CacheState::Direct {
                    uses: e.direct_uses.get(),
                    consecutive: runtime.consecutive_uses(&key),
                }
            }
        }))
    }
    pub(super) fn invalidate_batch(&mut self, layout: &Layout) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        let key = runtime.key(layout)?;
        runtime.remove(&key);
        runtime.lru.retain(|k| k != &key);
        self.w4.retain(|_, w| w.strong_count() > 0);
        self.reserve_batch_memory(
            super::memory::Requirement::workspace(),
            "coordinated Graph preparation",
        )
    }
    pub(super) fn prepare_batch(&mut self, layout: &Layout, capture: bool) -> Result<()> {
        let key = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .key(layout)?;
        let previous = self.batch.as_ref().unwrap().cache.get(&key).cloned();
        if previous.as_ref().is_none_or(|e| capture && !e.captured) {
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
            self.reserve_batch_memory(
                super::memory::Requirement::workspace(),
                "Graph preparation",
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
            // Every indirect state access checks this stream-ordered word.
            // Capturing a new shape cannot overwrite live KV or C2 rings.
            super::upload(
                &self.session,
                &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
                &0i32.to_le_bytes(),
            )?;
            let guard_s = guard_started.elapsed().as_secs_f64();
            let tables_started = std::time::Instant::now();
            self.update_batch_tables(layout, &packet)?;
            let tables_s = tables_started.elapsed().as_secs_f64();
            if layout.rows.total > packet.rows {
                return Err("Batch exceeds its family row capacity".into());
            }
            let base = self.package.clone();
            let compile_started = std::time::Instant::now();
            let (body, head) = if let Some(previous) = &previous {
                (
                    self.capture_steps(&previous.body)?,
                    self.capture_steps(&previous.head)?,
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
                Rc::new(Execution {
                    packet,
                    body,
                    head,
                    captured: capture,
                    direct_uses: std::cell::Cell::new(0),
                }),
            );
            if self.rank == 0 && std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
                eprintln!(
                    "A2 packet prepare {}",
                    serde_json::json!({
                        "rows": key.rows, "decode": layout.decode, "capture": capture,
                        "promotion": previous.is_some(),
                        "family": format!("{:?}", key.family),
                        "direct_uses": previous.as_ref().map_or(0, |e| e.direct_uses.get()),
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
        runtime.lru.retain(|k| k != &key);
        runtime.lru.push_back(key);
        while runtime.cache.len() > runtime.capacity {
            let old = runtime.lru.pop_front().unwrap();
            runtime.remove(&old);
        }
        self.w4.retain(|_, w| w.strong_count() > 0);
        Ok(())
    }
    fn update_batch_tables(
        &mut self,
        layout: &Layout,
        packet: &super::Packet,
    ) -> Result<Vec<Buffer>> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let key = layout.table_key(packet.min_rows);
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
            if used.columns[index] && column.row_bytes.is_some() {
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
                let buffer = if !used.columns[index] {
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
        for name in &used.pointers {
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
        for name in &used.work {
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
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let key = runtime.key(layout)?;
        let execution = Rc::clone(
            runtime
                .cache
                .get(&key)
                .ok_or("Batch must be prepared before forward")?,
        );
        let retained = self.update_batch_tables(layout, &execution.packet)?;
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
        for (i, (segment, input)) in layout.segments.iter().zip(tokens).enumerate() {
            let history = &self.batch.as_ref().unwrap().owners[&segment.slot].history;
            inputs.extend(self.coordinate_inputs(
                &execution.packet.coordinates,
                history.len(),
                segment.tokens,
                layout.rows.starts[i],
                None,
            )?);
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
            let mut owner_hashes = vec![];
            let mut owner_valid = vec![];
            self.hasher
                .hash_into(history, input, &mut owner_hashes, &mut owner_valid)?;
            hashes.extend(owner_hashes);
            valid.extend(owner_valid);
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
        if let Some(name) = &execution.packet.coordinates.input_ids {
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
                },
                !layout.outputs.is_empty(),
                retained,
            )?;
            let sampled = readout.parameters.get().and_then(|_| {
                // Model, sampling and readback are ordered on the same stream.
                self.sample_batch(&execution, layout, readout)
            });
            // Errors still drain every rank's submitted collective order. On
            // success the sampling readback has already fenced the model.
            pending.wait()?;
            sampled?
        };
        if !execution.captured {
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
            for range in readout.logit_ranges(
                outputs
                    .iter()
                    .map(|&(row, lane)| (row, layout.segments[lane].slot)),
            ) {
                let source = get(
                    &self.buffers,
                    execution
                        .packet
                        .logits
                        .as_ref()
                        .ok_or("Missing batch logits")?,
                )?;
                let bytes = self.session.download(
                    &source,
                    range.start * 16160 * 4,
                    range.len() * 16160 * 4,
                )?;
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
                });
            }
        }
        for (segment, input) in layout
            .segments
            .iter()
            .zip(tokens)
            .filter(|_| layout.role == super::PacketRole::Target)
        {
            self.batch
                .as_mut()
                .unwrap()
                .owners
                .get_mut(&segment.slot)
                .unwrap()
                .history
                .extend_from_slice(input);
        }
        if !execution.captured {
            execution
                .direct_uses
                .set(execution.direct_uses.get().saturating_add(1));
        }
        self.batch.as_mut().unwrap().record_use(key);
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
                execution.captured,
            )?));
        }
        Ok(Some(self.sampling.as_mut().unwrap().sample(
            &self.session,
            &source,
            execution.packet.vocab_first,
            &readout.controls(layout)?,
            &readout.row_parameters(layout)?,
            execution.captured,
        )?))
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
