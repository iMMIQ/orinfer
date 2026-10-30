//! Owner-private state and shared token rows for native A2 packets.
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
    pub decoder_replay: Option<super::replay::Contract>,
}
impl Contract {
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
            .filter(|r| r.tables.contains(name))
            .map(|r| layout.decoder(r.keep, r.final_prompt_only))
            .transpose()?;
        let layout = decoder.as_ref().unwrap_or(layout);
        let range = self.work_ranges.get(name);
        if let Some(&stride) = self.partitioned_tables.get(name) {
            let view = layout.rows.partition(stride, rank % stride)?;
            work_lengths(view.lengths, grid, range)
        } else {
            work_items(layout, grid, range)
        }
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
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Key {
    decode: bool,
    lengths: Vec<usize>,
    outputs: usize,
    partitioned_lengths: Vec<usize>,
    decoder_roles: Vec<super::replay::Role>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
struct TableKey {
    family: (bool, usize),
    segments: Vec<(usize, usize)>,
    outputs: Vec<usize>,
    decoder_roles: Vec<super::replay::Role>,
}
pub(super) struct BoundTables {
    key: TableKey,
    retained: Vec<Buffer>,
}
pub(super) struct Runtime {
    pub contract: Contract,
    pub owners: BTreeMap<usize, Owner>,
    cache: BTreeMap<Key, Rc<Execution>>,
    lru: VecDeque<Key>,
    capacity: usize,
    tables: BTreeMap<(bool, usize), TableUse>,
    pub(super) bound_tables: Option<BoundTables>,
}
struct TableUse {
    pointers: BTreeSet<String>,
    work: BTreeSet<String>,
    columns: Vec<bool>,
}
impl Runtime {
    fn key(&self, layout: &Layout) -> Key {
        let mut key = layout.key();
        if self
            .contract
            .decoder_replay
            .as_ref()
            .is_some_and(|r| r.final_prompt_only)
        {
            key.decoder_roles = layout.decoder_roles.clone();
            key.partitioned_lengths = layout.segments.iter().map(|s| s.tokens).collect();
        }
        if !layout.decode && !self.contract.partitioned_tables.is_empty() {
            // Reordering owners changes the count of selected rows per owner,
            // and hence indirect launch extents, even for equal sorted lengths.
            key.partitioned_lengths = layout.segments.iter().map(|s| s.tokens).collect();
        }
        key
    }
    pub fn new(contract: Contract, capacity: usize, packets: &[super::Packet]) -> Self {
        let mut tables = BTreeMap::new();
        for packet in packets {
            let mut used = TableUse {
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
            tables.insert((packet.batch_decode, packet.min_rows), used);
        }
        Self {
            contract,
            owners: BTreeMap::new(),
            cache: BTreeMap::new(),
            lru: VecDeque::new(),
            capacity,
            tables,
            bound_tables: None,
        }
    }
}
#[derive(Clone)]
pub(super) struct Layout {
    pub segments: Vec<BatchSegment>,
    pub rows: Rows,
    pub outputs: Vec<usize>,
    pub decode: bool,
    pub decoder_roles: Vec<super::replay::Role>,
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
            segments,
            rows,
            outputs,
            decode,
            decoder_roles,
        })
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
        if view.outputs.iter().any(|&i| view.segments[i].tokens == 0) {
            return Err("Output owner has no decoder rows".into());
        }
        Ok(view)
    }
    pub(super) fn suffix(&self, keep: usize) -> Result<Self> {
        if self.decode || !(128..=512).contains(&keep) {
            return Err("Invalid decoder bounded replay layout".into());
        }
        Self::new(
            self.segments
                .iter()
                .map(|s| BatchSegment {
                    slot: s.slot,
                    tokens: s.tokens.min(keep),
                })
                .collect(),
            self.outputs.clone(),
            false,
            self.rows.total,
            self.segments.len(),
        )
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
    fn key(&self) -> Key {
        let mut lengths = self.segments.iter().map(|s| s.tokens).collect::<Vec<_>>();
        lengths.sort_unstable();
        Key {
            decode: self.decode,
            lengths,
            outputs: self.outputs.len(),
            partitioned_lengths: vec![],
            decoder_roles: vec![],
        }
    }
    fn table_key(&self, min_rows: usize) -> TableKey {
        TableKey {
            family: (self.decode, min_rows),
            segments: self.segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            outputs: self.outputs.clone(),
            decoder_roles: self.decoder_roles.clone(),
        }
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
    let mut out = vec![];
    for (owner, tokens) in lengths.into_iter().enumerate() {
        if tokens == 0 || range.is_some_and(|r| tokens < r.min_rows || tokens > r.max_rows) {
            continue;
        }
        for block in 0..grid.resolve(tokens)?.min(64) {
            out.push([owner as i32, block as i32]);
        }
    }
    if out.is_empty() {
        return Ok(vec![[-1, 0], [-1, 0]]);
    }
    if out.len() > 8192 {
        return Err("Owner work table exceeds its capacity".into());
    }
    if out.len() % 2 != 0 {
        out.push([-1, 0]);
    }
    Ok(out)
}
impl Worker {
    fn reserve_batch_memory(&mut self, needed: usize, stage: &str) -> Result<()> {
        if self.session.memory_info()?.0 >= needed {
            return Ok(());
        }
        self.session.synchronize()?;
        let mut evicted = 0;
        while self.session.memory_info()?.0 < needed {
            let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
            let Some(key) = runtime.lru.pop_front() else {
                return Err(format!(
                    "Insufficient A2 memory for {stage}, including capture headroom"
                ));
            };
            runtime.cache.remove(&key);
            self.w4.retain(|_, w| w.strong_count() > 0);
            self.session.synchronize()?;
            evicted += 1;
        }
        eprintln!(
            "A2 rank {} evicted {evicted} cached packet shapes for {stage}",
            self.rank
        );
        Ok(())
    }
    pub(super) fn admit(&mut self, ids: &[usize]) -> Result<()> {
        self.batch
            .as_mut()
            .ok_or("Missing native batch contract")?
            .bound_tables = None;
        let runtime = self.batch.as_ref().ok_or("Missing native batch contract")?;
        let names = runtime.contract.private.clone();
        let count = ids
            .iter()
            .copied()
            .filter(|id| *id != 0 && !runtime.owners.contains_key(id))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let entries = names
            .iter()
            .filter(|n| !self.views.contains_key(*n))
            .map(|n| Ok((n.as_str(), get(&self.buffers, n)?.bytes(), 256)))
            .collect::<Result<Vec<_>>>()?;
        let arena = crate::backend::arena::ArenaLayout::from_sizes(entries.into_iter())?;
        let arena_bytes = arena
            .allocation_bytes
            .checked_add(255)
            .ok_or("Owner arena alignment overflow")?;
        // HUGE_FIRST rounds independent allocations in the device page pool.
        // One aligned owner bank avoids charging a page for each small scale.
        let per_owner = arena_bytes
            .checked_add((2 << 20) - 1)
            .ok_or("Owner physical extent overflow")?
            / (2 << 20)
            * (2 << 20);
        let needed = per_owner
            .checked_mul(count)
            .and_then(|n| n.checked_add(3 << 30))
            .ok_or("Owner admission estimate overflow")?;
        self.reserve_batch_memory(needed, "owner admission")?;
        for &id in ids {
            if self.batch.as_ref().unwrap().owners.contains_key(&id) {
                continue;
            }
            if self.batch.as_ref().unwrap().owners.len()
                >= self.batch.as_ref().unwrap().contract.owners
            {
                return Err("Owner capacity exceeded".into());
            }
            let mut buffers = BTreeMap::new();
            let bank = if id == 0 {
                None
            } else {
                Some(self.session.allocate(arena_bytes)?)
            };
            let base = if let Some(bank) = &bank {
                self.session.fill(bank, 0)?.wait()?;
                usize::try_from(arena.base(bank.pointer() as usize as u64)?)
                    .map_err(|e| e.to_string())?
                    - (bank.pointer() as usize)
            } else {
                0
            };
            for name in &names {
                let buffer = if id == 0 {
                    get(&self.buffers, name)?
                } else if let Some((view, bytes)) = self.views.get(name) {
                    get(&buffers, &view.parent)?.slice(view.offset, *bytes)?
                } else {
                    bank.as_ref().unwrap().slice(
                        base + arena.offsets[name],
                        get(&self.buffers, name)?.bytes(),
                    )?
                };
                buffers.insert(name.clone(), buffer);
            }
            self.batch.as_mut().unwrap().owners.insert(
                id,
                Owner {
                    buffers,
                    history: vec![],
                },
            );
            self.reset_owner(id)?;
        }
        Ok(())
    }
    pub(super) fn reset_owner(&mut self, id: usize) -> Result<()> {
        let names = self
            .packet_templates
            .iter()
            .flat_map(|p| p.reset.iter())
            .collect::<std::collections::BTreeSet<_>>();
        let owner = self
            .batch
            .as_mut()
            .ok_or("Missing batch runtime")?
            .owners
            .get_mut(&id)
            .ok_or("Unknown owner")?;
        for name in names {
            self.session.fill(&get(&owner.buffers, name)?, 0)?.wait()?;
        }
        owner.history.clear();
        Ok(())
    }
    pub(super) fn release_owners(&mut self) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.clear();
        Ok(())
    }
    pub(super) fn batch_cached(&self, layout: &Layout) -> Result<bool> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        Ok(runtime.cache.contains_key(&runtime.key(layout)))
    }
    pub(super) fn invalidate_batch(&mut self, layout: &Layout) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        let key = runtime.key(layout);
        runtime.cache.remove(&key);
        runtime.lru.retain(|k| k != &key);
        self.w4.retain(|_, w| w.strong_count() > 0);
        self.reserve_batch_memory(3 << 30, "coordinated Graph preparation")
    }
    pub(super) fn prepare_batch(&mut self, layout: &Layout) -> Result<()> {
        let key = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .key(layout);
        if !self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .cache
            .contains_key(&key)
        {
            self.reserve_batch_memory(3 << 30, "Graph preparation")?;
            let packet = self
                .packet_templates
                .iter()
                .find(|p| {
                    p.batch_decode == layout.decode
                        && p.min_rows <= layout.rows.total
                        && layout.rows.total <= p.dispatch_max()
                })
                .cloned()
                .ok_or("Missing batch family")?;
            // Every indirect state access checks this stream-ordered word.
            // Capturing a new shape cannot overwrite live KV or C2 rings.
            super::upload(
                &self.session,
                &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
                &0i32.to_le_bytes(),
            )?;
            self.update_batch_tables(layout, &packet)?;
            if layout.rows.total > packet.rows {
                return Err("Batch exceeds its family row capacity".into());
            }
            let base = self.package.clone();
            let body = self.compile_bound(
                &packet.body,
                &base,
                self.snapshots,
                (packet.rows, layout.rows.total),
                Some(layout),
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
                )?
            };
            self.batch
                .as_mut()
                .unwrap()
                .cache
                .insert(key.clone(), Rc::new(Execution { packet, body, head }));
        }
        let runtime = self.batch.as_mut().unwrap();
        runtime.lru.retain(|k| k != &key);
        runtime.lru.push_back(key);
        while runtime.cache.len() > runtime.capacity {
            let old = runtime.lru.pop_front().unwrap();
            runtime.cache.remove(&old);
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
        let used = runtime
            .tables
            .get(&(packet.batch_decode, packet.min_rows))
            .ok_or("Missing packet table usage")?;
        let decoder = runtime
            .contract
            .decoder_replay
            .as_ref()
            .filter(|_| !layout.decode)
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
                    aligned.entry((column.decoder, stride, 1))
                {
                    let rows = if column.decoder {
                        &decoder.as_ref().ok_or("Missing decoder rows")?.rows
                    } else {
                        &layout.rows
                    };
                    entry.insert(rows.partition(stride, self.rank as usize % stride)?);
                }
                if alignment > 1 && !aligned.contains_key(&(column.decoder, stride, alignment)) {
                    let view = aligned[&(column.decoder, stride, 1)].aligned(alignment)?;
                    aligned.insert((column.decoder, stride, alignment), view);
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
                let buffer =
                    if !used.columns[index] || column.decode.is_some_and(|d| d != layout.decode) {
                        None
                    } else if let Some(stride) = column.row_bytes {
                        let view = &aligned[&(
                            column.decoder,
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
            if mapping.iter().any(|&c| {
                runtime.contract.columns[c]
                    .decode
                    .is_some_and(|d| d != layout.decode)
            }) {
                continue;
            }
            let is_decoder = runtime
                .contract
                .decoder_replay
                .as_ref()
                .is_some_and(|r| r.tables.contains(name));
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
            .flat_map(|&i| {
                ((output_layout.rows.starts[i] + output_layout.segments[i].tokens - 1) as i32)
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        if !data.is_empty() {
            inputs.push((
                get(&self.buffers, &runtime.contract.endpoints)?.slice(0, data.len())?,
                data,
            ));
        }
        self.session.upload_inputs(&inputs)?.wait()?;
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
        full_logits: bool,
    ) -> Result<Vec<(usize, super::packet::Best, Vec<f32>)>> {
        if tokens.len() != layout.segments.len()
            || tokens
                .iter()
                .zip(&layout.segments)
                .any(|(t, s)| t.len() != s.tokens)
        {
            return Err("Batch input differs from row layout".into());
        }
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let execution = Rc::clone(
            runtime
                .cache
                .get(&runtime.key(layout))
                .ok_or("Batch must be prepared before forward")?,
        );
        let retained = self.update_batch_tables(layout, &execution.packet)?;
        let mut joined = vec![];
        let mut hashes = vec![];
        let mut valid = vec![];
        for (i, (segment, input)) in layout.segments.iter().zip(tokens).enumerate() {
            let history = &self.batch.as_ref().unwrap().owners[&segment.slot].history;
            self.coordinate_at(
                &execution.packet.coordinates,
                history.len(),
                segment.tokens,
                layout.rows.starts[i],
            )?;
            if let Some(names) = &execution.packet.decoder_coordinates {
                let replay = self
                    .batch
                    .as_ref()
                    .unwrap()
                    .contract
                    .decoder_replay
                    .as_ref()
                    .ok_or("Missing decoder replay contract")?;
                let keep = replay.keep;
                let final_prompt_only = replay.final_prompt_only;
                let suffix = layout.decoder(keep, final_prompt_only)?;
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
                    self.coordinate_at_bounded(
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
                    )?;
                }
            }
            let mut owner_hashes = vec![];
            let mut owner_valid = vec![];
            self.hasher
                .hash_into(history, input, &mut owner_hashes, &mut owner_valid)?;
            hashes.extend(owner_hashes);
            valid.extend(owner_valid);
            joined.extend_from_slice(input);
        }
        super::upload(
            &self.session,
            &get(&self.buffers, &self.batch.as_ref().unwrap().contract.active)?,
            &1i32.to_le_bytes(),
        )?;
        if let Some(trace) = std::env::var_os("ORINFER_A2_BATCH_TRACE") {
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
            )?;
            self.trace_rings(&path, "after")?;
        } else {
            self.enqueue_packet_retained(
                Rc::clone(&execution),
                &joined,
                &hashes,
                &valid,
                !layout.outputs.is_empty(),
                retained,
            )?
            .wait()?;
        }
        let mut output = vec![];
        if !layout.outputs.is_empty() {
            let count = layout.outputs.len();
            let values = self.session.download(
                &get(&self.buffers, &execution.packet.best_value)?,
                0,
                count * 4,
            )?;
            let indices = self.session.download(
                &get(&self.buffers, &execution.packet.best_index)?,
                0,
                count * 4,
            )?;
            let logits = if full_logits {
                Some(
                    self.session.download(
                        &get(
                            &self.buffers,
                            execution
                                .packet
                                .logits
                                .as_ref()
                                .ok_or("Missing batch logits")?,
                        )?,
                        0,
                        count * 16160 * 4,
                    )?,
                )
            } else {
                None
            };
            for (i, &lane) in layout.outputs.iter().enumerate() {
                let value = f32::from_le_bytes(values[i * 4..i * 4 + 4].try_into().unwrap());
                let index = i32::from_le_bytes(indices[i * 4..i * 4 + 4].try_into().unwrap());
                if !value.is_finite() || !(0..16160).contains(&index) {
                    return Err("Invalid batched output".into());
                }
                let probs = logits.as_ref().map_or(vec![], |b| {
                    b[i * 16160 * 4..(i + 1) * 16160 * 4]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|v| f32::from_le_bytes(*v))
                        .collect()
                });
                output.push((
                    layout.segments[lane].slot,
                    super::packet::Best {
                        value,
                        token: execution.packet.vocab_first + index as u32,
                    },
                    probs,
                ));
            }
        }
        for (segment, input) in layout.segments.iter().zip(tokens) {
            self.batch
                .as_mut()
                .unwrap()
                .owners
                .get_mut(&segment.slot)
                .unwrap()
                .history
                .extend_from_slice(input);
        }
        Ok(output)
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
        for name in &c.private {
            let b = buffers
                .get(name.as_str())
                .ok_or("Missing private owner state")?;
            if b.weight.is_some()
                || b.data.is_some()
                || b.prepared_weight.is_some()
                || b.view
                    .as_ref()
                    .is_some_and(|v| !parents.contains(&v.parent))
            {
                return Err("Invalid private allocation/view order".into());
            }
            parents.insert(name.clone());
        }
        if rank.packets.iter().filter(|p| p.batch_decode).count() != 1
            || rank.packets.iter().all(|p| p.batch_decode)
        {
            return Err("Batch requires one decode and complete prefill dispatch".into());
        }
        let prefill = rank
            .packets
            .iter()
            .filter(|p| !p.batch_decode)
            .collect::<Vec<_>>();
        let capacity = prefill[0].rows;
        super::validate_row_ranges(&prefill, 1, Some(capacity))?;
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
                        .find(|p| p.batch_decode == decode)
                        .unwrap();
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
        if buffers.get(c.endpoints.as_str()).map(|b| b.bytes) != Some(c.owners * 4) {
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
mod tests {
    use super::*;
    fn layout(lengths: &[usize], owners: &[usize]) -> Result<Layout> {
        Layout::new(
            lengths
                .iter()
                .zip(owners)
                .map(|(&tokens, &slot)| BatchSegment { tokens, slot })
                .collect(),
            vec![0],
            false,
            4096,
            128,
        )
    }
    #[test]
    fn partitioned_work_keeps_original_owner_and_graph_order() {
        let mut contract: Contract = serde_json::from_value(serde_json::json!({
            "owners": 32, "columns": [], "private": [], "pointers": {},
            "active": "a", "endpoints": "e", "work": {}
        }))
        .unwrap();
        let a = layout(&[1, 7, 3], &[91, 4, 33]).unwrap();
        let b = layout(&[3, 1, 7], &[8, 77, 5]).unwrap();
        let original = Runtime::new(contract.clone(), 8, &[]);
        assert_eq!(original.key(&a), original.key(&b));
        contract.partitioned_tables.insert("w".into(), 8);
        assert_eq!(contract.token_capacity("w", 2048).unwrap(), 256);
        assert_eq!(contract.token_capacity("w", 2047).unwrap(), 256);
        assert_eq!(contract.token_capacity("original", 2048).unwrap(), 2048);
        let runtime = Runtime::new(contract.clone(), 8, &[]);
        assert_ne!(runtime.key(&a), runtime.key(&b));
        let reordered_ids = layout(&[1, 7, 3], &[33, 91, 4]).unwrap();
        assert_eq!(runtime.key(&a), runtime.key(&reordered_ids));
        let grid: RowExpression =
            serde_json::from_value(serde_json::json!({"op":"constant","value":64})).unwrap();
        let work = contract.work_items(&a, &grid, "w", 7).unwrap();
        assert_eq!(work.len(), 64);
        assert!(work.iter().all(|p| p[0] == 1));
        let empty = layout(&[1, 2], &[9, 4]).unwrap();
        assert_eq!(
            contract.work_items(&empty, &grid, "w", 7).unwrap(),
            [[-1, 0], [-1, 0]]
        );
    }
    #[test]
    fn address_tables_keep_owner_and_output_order_when_graphs_share_a_shape() {
        let a = layout(&[1, 1], &[91, 8]).unwrap();
        let b = layout(&[1, 1], &[8, 91]).unwrap();
        assert_eq!(a.key(), b.key());
        assert_ne!(a.table_key(1), b.table_key(1));
        let mut output = a.clone();
        output.outputs = vec![1];
        assert_eq!(a.key(), output.key());
        assert_ne!(a.table_key(1), output.table_key(1));
        assert_ne!(a.table_key(1), a.table_key(512));
        let mut decode = a.clone();
        decode.decode = true;
        assert_ne!(a.table_key(1), decode.table_key(1));
    }
    #[test]
    fn arbitrary_batch_work_covers_each_logical_owner_tile() {
        let grid:RowExpression=serde_json::from_value(serde_json::json!({"op":"multiply","lhs":{"op":"rows"},"rhs":{"op":"constant","value":3}})).unwrap();
        for batch in 1..=128 {
            let lengths = (0..batch).map(|i| 1 + i % 31).collect::<Vec<_>>();
            let ids = (0..batch).map(|i| i * 17 + 9).collect::<Vec<_>>();
            let l = layout(&lengths, &ids).unwrap();
            let work = work_items(&l, &grid, None).unwrap();
            assert_eq!(work.len() % 2, 0);
            for (owner, &q) in lengths.iter().enumerate() {
                let physical = work
                    .iter()
                    .filter(|p| p[0] == owner as i32)
                    .map(|p| p[1] as usize)
                    .collect::<Vec<_>>();
                assert_eq!(physical, (0..(3 * q).min(64)).collect::<Vec<_>>());
                let mut logical = physical
                    .into_iter()
                    .flat_map(|p| (p..3 * q).step_by(64))
                    .collect::<Vec<_>>();
                logical.sort_unstable();
                assert_eq!(logical, (0..3 * q).collect::<Vec<_>>());
            }
        }
    }
    #[test]
    fn owner_work_selection_keeps_query_policy_across_mixed_packets() {
        let grid: RowExpression = serde_json::from_value(serde_json::json!({"op":"rows"})).unwrap();
        let mixed = layout(&[1, 511, 512, 513], &[91, 8, 44, 7]).unwrap();
        let short = WorkRange {
            min_rows: 1,
            max_rows: 511,
        };
        let long = WorkRange {
            min_rows: 512,
            max_rows: 4096,
        };
        let a = work_items(&mixed, &grid, Some(&short)).unwrap();
        let b = work_items(&mixed, &grid, Some(&long)).unwrap();
        assert!(a.iter().all(|p| p[0] < 2));
        assert!(b.iter().all(|p| p[0] >= 2));
        let one = layout(&[511], &[99]).unwrap();
        assert_eq!(
            work_items(&one, &grid, Some(&long)).unwrap(),
            [[-1, 0], [-1, 0]]
        );
        assert_eq!(work_items(&one, &grid, Some(&short)).unwrap().len(), 64);
    }
    #[test]
    fn graph_identity_follows_shape_and_role_not_owner_order() {
        let a = layout(&[1, 7, 3], &[91, 4, 33]).unwrap();
        let b = layout(&[3, 1, 7], &[8, 77, 5]).unwrap();
        assert_eq!(a.key(), b.key());
        assert_eq!(a.rows.starts, vec![0, 1, 8]);
        assert_eq!(b.rows.starts, vec![0, 3, 4]);
        assert!(layout(&[1, 7], &[9, 9]).is_err());
        assert!(layout(&[0], &[8]).is_err());
        assert!(Layout::new(a.segments.clone(), vec![0], true, 4096, 128).is_err());
        assert!(Layout::new(a.segments.clone(), vec![2, 0], false, 4096, 128).is_err());
        let mut c = a.clone();
        c.outputs.push(2);
        assert_ne!(a.key(), c.key());
        let mut d = a.clone();
        d.decode = true;
        assert_ne!(a.key(), d.key());
    }
}
#[test]
fn scheduling_quantum_is_independent_of_attention_work_selection() {
    let base = serde_json::json!({
        "owners": 128, "columns": [], "private": [], "pointers": {},
        "active": "a", "endpoints": "e", "work": {}, "work_ranges": {}
    });
    let mut contract: Contract = serde_json::from_value(base).unwrap();
    assert_eq!(contract.preferred_span(), 1);
    contract.work_ranges.insert(
        "old".into(),
        WorkRange {
            min_rows: 512,
            max_rows: 2048,
        },
    );
    assert_eq!(contract.preferred_span(), 512);
    contract.prefill_quantum = Some(256);
    assert_eq!(contract.preferred_span(), 256);
    contract.work_ranges.clear();
    assert_eq!(contract.preferred_span(), 256);
}
