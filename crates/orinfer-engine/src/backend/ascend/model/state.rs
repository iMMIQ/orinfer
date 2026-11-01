//! Exact persistent-state snapshots; compute workspaces remain request-local.
use super::{Buffer, Worker, batch::Owner, get};
use crate::artifact::Result;
use crate::backend::ascend::runtime::Session;
use crate::backend::{
    kv,
    snapshot::{Allocation as SharedAllocation, Piece as SharedPiece},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::Range, rc::Rc};
mod host;
pub(super) use crate::backend::kv::host::{HostBuffer, HostPool};
pub(super) type HostSnapshot = Snapshot<HostBuffer>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StateLayout {
    pub buffer: String,
    pub row_bytes: usize,
    /// Zero denotes a fixed-size recurrent state, copied in full.
    pub tokens_per_row: usize,
    pub window: Option<usize>,
    #[serde(default)]
    pub circular: bool,
    /// Closed linear rows are immutable; windows and recurrent state remain private copies.
    #[serde(default)]
    pub append_only: bool,
}

/// Logical read history inside an append-only physical ring or ring page.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct AppendWindow {
    pub offset: usize,
    pub row_bytes: usize,
    pub rows: usize,
    pub history: usize,
}
impl AppendWindow {
    pub(super) fn overwritten_spans(
        &self,
        tokens: usize,
        steps: usize,
        bytes: usize,
    ) -> Result<Vec<Range<usize>>> {
        if self.row_bytes == 0
            || self.rows == 0
            || self.history > self.rows
            || steps == 0
            || steps > self.rows
            || self
                .rows
                .checked_mul(self.row_bytes)
                .and_then(|n| n.checked_add(self.offset))
                .is_none_or(|n| n > bytes)
        {
            return Err("Invalid verification append window".into());
        }
        if self.rows - steps >= self.history {
            return Ok(vec![]);
        }
        let first = tokens % self.rows;
        let left = steps.min(self.rows - first);
        let mut spans = Vec::with_capacity(2);
        spans.push(
            self.offset + first * self.row_bytes..self.offset + (first + left) * self.row_bytes,
        );
        if steps > left {
            spans.push(self.offset..self.offset + (steps - left) * self.row_bytes);
        }
        Ok(spans)
    }
}
impl StateLayout {
    fn overwritten_spans(
        &self,
        tokens: usize,
        steps: usize,
        bytes: usize,
    ) -> Result<Vec<Range<usize>>> {
        self.spans(tokens, bytes)?;
        if self.append_only || self.tokens_per_row > 0 && !self.circular {
            return Ok(vec![]);
        }
        if self.tokens_per_row == 0 {
            return Ok(std::iter::once(0..bytes).collect());
        }
        let rows = bytes / self.row_bytes;
        if self.tokens_per_row != 1 || steps == 0 || steps > rows {
            return Err("Invalid verification journal window".into());
        }
        AppendWindow {
            offset: 0,
            row_bytes: self.row_bytes,
            rows,
            history: self.window.unwrap_or(rows),
        }
        .overwritten_spans(tokens, steps, bytes)
    }
    pub(super) fn spans(&self, tokens: usize, bytes: usize) -> Result<Vec<Range<usize>>> {
        if self.row_bytes == 0
            || !bytes.is_multiple_of(self.row_bytes)
            || self.window == Some(0)
            || self.append_only
                && (self.tokens_per_row == 0 || self.window.is_some() || self.circular)
        {
            return Err("Invalid A2 state layout".into());
        }
        if self.tokens_per_row == 0 {
            return Ok(std::iter::once(0..bytes).collect());
        }
        let end = tokens / self.tokens_per_row;
        let start = self.window.map_or(0, |window| end.saturating_sub(window));
        let rows = bytes / self.row_bytes;
        if self.circular {
            if rows == 0 || end - start > rows {
                return Err("A2 state window exceeds ring capacity".into());
            }
            let first = start % rows;
            let count = end - start;
            let left = count.min(rows - first);
            let mut spans = vec![];
            if left > 0 {
                spans.push(first * self.row_bytes..(first + left) * self.row_bytes);
            }
            if count > left {
                spans.push(0..(count - left) * self.row_bytes);
            }
            Ok(spans)
        } else {
            if end > rows {
                return Err("A2 state endpoint exceeds capacity".into());
            }
            Ok((end > start)
                .then_some(start * self.row_bytes..end * self.row_bytes)
                .into_iter()
                .collect())
        }
    }
}
#[derive(Clone)]
struct Piece<B = Buffer> {
    name: String,
    offset: usize,
    buffer: B,
    bank_id: u64,
    bank_bytes: usize,
}
pub(super) struct Snapshot<B = DeviceBuffer> {
    history: Vec<u32>,
    pieces: Vec<Piece<B>>,
}
pub(super) struct Journal {
    history: usize,
    rows: usize,
    pieces: Vec<Piece>,
    bank: Buffer,
    circular: BTreeMap<String, usize>,
}
type PendingCopy = (String, Buffer, Range<usize>, usize);
struct CapturePlan<B> {
    pieces: Vec<Piece<B>>,
    pending: Vec<PendingCopy>,
    bytes: usize,
}
impl<B> CapturePlan<B> {
    fn append(&mut self, name: &str, source: Buffer, span: Range<usize>) -> Result<()> {
        if span.is_empty() {
            return Ok(());
        }
        let offset = self
            .bytes
            .checked_add(255)
            .ok_or("Snapshot alignment overflow")?
            / 256
            * 256;
        self.bytes = offset
            .checked_add(span.len())
            .ok_or("Snapshot size overflow")?;
        self.pending.push((name.to_owned(), source, span, offset));
        Ok(())
    }
    fn allocations(&self, bank_id: u64) -> Result<BTreeMap<u64, usize>> {
        let mut banks: BTreeMap<_, _> = self
            .pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect();
        if !self.pending.is_empty() {
            let bytes = self
                .bytes
                .checked_add(255)
                .ok_or("Snapshot size overflow")?;
            banks.insert(
                kv::checkpoint_id(bank_id)?,
                super::memory::round_pages(bytes)?,
            );
        }
        Ok(banks)
    }
}
pub(super) trait StateBytes: Clone {
    fn bytes(&self) -> usize;
    fn view(&self, range: orinfer_device::MemoryRange) -> Result<Self>;
    fn paged(&self) -> bool {
        false
    }
    fn host_capacity(&self) -> usize {
        self.bytes()
    }
}
impl StateBytes for Buffer {
    fn bytes(&self) -> usize {
        self.bytes()
    }
    fn view(&self, range: orinfer_device::MemoryRange) -> Result<Self> {
        self.slice(range.offset, range.bytes)
    }
}
impl<B: StateBytes> Snapshot<B> {
    pub fn allocations(&self) -> BTreeMap<u64, usize> {
        self.pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect()
    }
    fn plan<'a>(
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        parent: Option<&Self>,
    ) -> Result<CapturePlan<B>> {
        if parent.is_some_and(|p| !history.starts_with(&p.history)) {
            return Err("A2 snapshot parent history diverges".into());
        }
        let mut plan = CapturePlan {
            pieces: vec![],
            pending: vec![],
            bytes: 0,
        };
        for layout in layouts {
            let source = get(buffers, &layout.buffer)?;
            for mut span in layout.spans(history.len(), source.bytes())? {
                if layout.append_only
                    && let Some(parent) = parent
                {
                    let end = parent.history.len() / layout.tokens_per_row * layout.row_bytes;
                    let mut inherited = 0;
                    for piece in parent.pieces.iter().filter(|p| p.name == layout.buffer) {
                        if piece.offset != inherited
                            || inherited > end
                            || piece.buffer.bytes() > end - inherited
                        {
                            return Err("A2 snapshot parent has noncontiguous closed rows".into());
                        }
                        inherited += piece.buffer.bytes();
                        plan.pieces.push(piece.clone());
                    }
                    if inherited != 0 && inherited != end {
                        return Err("A2 snapshot parent does not cover its closed rows".into());
                    }
                    span.start = inherited;
                }
                plan.append(&layout.buffer, source.clone(), span)?;
            }
        }
        Ok(plan)
    }
}
impl<B: StateBytes> Snapshot<B> {
    fn plan_checkpoint<'a>(
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        parent: Option<&Self>,
    ) -> Result<(CapturePlan<B>, Vec<SharedPiece>)> {
        let layouts = layouts.collect::<Vec<_>>();
        let mut plan = Self::plan(
            layouts
                .iter()
                .copied()
                .filter(|l| !l.append_only || !buffers[&l.buffer].demand_mapped()),
            buffers,
            history,
            parent,
        )?;
        let mut shared = vec![];
        for layout in layouts
            .into_iter()
            .filter(|l| l.append_only && buffers[&l.buffer].demand_mapped())
        {
            let source = get(buffers, &layout.buffer)?;
            for span in layout.spans(history.len(), source.bytes())? {
                let regions = kv::checkpoint_regions(source.kv_preview(span.end)?)?;
                if let Some(tail) = regions.tail {
                    let candidates = parent
                        .into_iter()
                        .flat_map(|p| &p.pieces)
                        .filter(|p| p.name == layout.buffer && !p.buffer.paged())
                        .collect::<Vec<_>>();
                    let reuse = kv::reuse_tail(
                        tail,
                        candidates.iter().map(|p| orinfer_device::MemoryRange {
                            offset: p.offset,
                            bytes: p.buffer.bytes(),
                        }),
                    )?;
                    for part in reuse.parts {
                        let p = candidates[part.index];
                        plan.pieces.push(Piece {
                            name: p.name.clone(),
                            offset: part.range.offset,
                            buffer: p.buffer.view(orinfer_device::MemoryRange {
                                offset: part.source_offset,
                                bytes: part.range.bytes,
                            })?,
                            bank_id: p.bank_id,
                            bank_bytes: p.bank_bytes,
                        });
                    }
                    plan.append(
                        &layout.buffer,
                        source.clone(),
                        reuse.missing.offset..reuse.missing.offset + reuse.missing.bytes,
                    )?;
                }
                shared.extend(
                    regions
                        .shared
                        .into_iter()
                        .map(|(allocation, range, source)| SharedPiece {
                            allocation,
                            name: layout.buffer.clone(),
                            range,
                            source,
                        }),
                );
            }
        }
        Ok((plan, shared))
    }
}
/// Publish exactly the full allocations selected by the common checkpoint plan.
fn publish_shared(
    session: &Session,
    buffers: &BTreeMap<String, Buffer>,
    preview: &[SharedPiece],
) -> Result<Vec<Piece<DeviceBuffer>>> {
    let mut extents = BTreeMap::<&str, usize>::new();
    for p in preview {
        extents
            .entry(&p.name)
            .and_modify(|end| *end = (*end).max(p.range.offset + p.range.bytes))
            .or_insert(p.range.offset + p.range.bytes);
    }
    session.synchronize()?;
    let mut pieces = vec![];
    for (name, end) in extents {
        for (allocation, range, source) in session.publish_kv(&get(buffers, name)?, end)? {
            pieces.push(Piece {
                name: name.to_owned(),
                offset: range.offset,
                bank_id: allocation.id,
                bank_bytes: allocation.bytes,
                buffer: DeviceBuffer {
                    allocation,
                    source,
                    bytes: range.bytes,
                },
            });
        }
    }
    Ok(pieces)
}

#[derive(Clone)]
pub(super) struct DeviceBuffer {
    allocation: Rc<SharedAllocation>,
    source: usize,
    bytes: usize,
}
impl DeviceBuffer {
    fn native(&self) -> Result<Buffer> {
        self.allocation
            .native
            .as_ref()
            .and_then(|b| b.downcast_ref::<Buffer>())
            .ok_or("Foreign A2 snapshot backing")?
            .slice(self.source, self.bytes)
    }
    fn shared_piece(&self, name: &str, offset: usize) -> SharedPiece {
        SharedPiece {
            allocation: self.allocation.clone(),
            name: name.to_owned(),
            range: crate::backend::snapshot::Range {
                offset,
                bytes: self.bytes,
            },
            source: self.source,
        }
    }
}
impl StateBytes for DeviceBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn view(&self, range: orinfer_device::MemoryRange) -> Result<Self> {
        range.within(self.bytes)?;
        Ok(Self {
            allocation: self.allocation.clone(),
            source: self
                .source
                .checked_add(range.offset)
                .ok_or("Device KV view overflow")?,
            bytes: range.bytes,
        })
    }
    fn paged(&self) -> bool {
        self.allocation.handle.is_some()
    }
    fn host_capacity(&self) -> usize {
        self.allocation.used
    }
}
impl Snapshot {
    fn plan_device<'a>(
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        parent: Option<&Self>,
    ) -> Result<CapturePlan<DeviceBuffer>> {
        let (mut plan, shared) = Self::plan_checkpoint(layouts, buffers, history, parent)?;
        plan.pieces.extend(shared.into_iter().map(|p| Piece {
            name: p.name,
            offset: p.range.offset,
            bank_id: p.allocation.id,
            bank_bytes: p.allocation.bytes,
            buffer: DeviceBuffer {
                allocation: p.allocation,
                source: p.source,
                bytes: p.range.bytes,
            },
        }));
        Ok(plan)
    }
    fn capture<'a>(
        session: &Session,
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<Self> {
        let layouts = layouts.collect::<Vec<_>>();
        let CapturePlan {
            pending,
            bytes,
            pieces: planned,
        } = Self::plan_device(layouts.iter().copied(), buffers, history, parent)?;
        let shared = planned
            .iter()
            .filter(|p| p.buffer.allocation.handle.is_some())
            .map(|p| p.buffer.shared_piece(&p.name, p.offset))
            .collect::<Vec<_>>();
        let mut pieces = planned
            .into_iter()
            .filter(|p| p.buffer.allocation.handle.is_none())
            .collect::<Vec<_>>();
        if !pending.is_empty() {
            let bank = session.allocate(bytes.checked_add(255).ok_or("Snapshot size overflow")?)?;
            let base = (bank.pointer() as usize).div_ceil(256) * 256 - bank.pointer() as usize;
            let allocation = session.snapshot_allocation(&bank, kv::checkpoint_id(bank_id)?)?;
            let mut copies = vec![];
            for (name, source, span, offset) in pending {
                copies.push((source, span.start, bank.clone(), base + offset, span.len()));
                pieces.push(Piece {
                    name,
                    offset: span.start,
                    bank_id: allocation.id,
                    bank_bytes: allocation.bytes,
                    buffer: DeviceBuffer {
                        allocation: allocation.clone(),
                        source: base + offset,
                        bytes: span.len(),
                    },
                });
            }
            session.copy_many(&copies)?.wait()?;
        }
        pieces.extend(publish_shared(session, buffers, &shared)?);
        Ok(Self {
            history: history.to_vec(),
            pieces,
        })
    }
    pub fn host_allocations(&self) -> BTreeMap<u64, usize> {
        self.pieces
            .iter()
            .map(|p| (p.bank_id, p.buffer.host_capacity()))
            .collect()
    }
    fn restore_kv(&self, session: &Session, buffers: &BTreeMap<String, Buffer>) -> Result<()> {
        let mut groups = BTreeMap::<&str, Vec<SharedPiece>>::new();
        for piece in &self.pieces {
            if piece.buffer.allocation.handle.is_some() {
                groups
                    .entry(&piece.name)
                    .or_default()
                    .push(piece.buffer.shared_piece(&piece.name, piece.offset));
            }
        }
        session.synchronize()?;
        for (name, pieces) in groups {
            session.restore_kv(&get(buffers, name)?, &pieces)?;
        }
        Ok(())
    }
    #[cfg(test)]
    fn restore_into(&self, session: &Session, buffers: &BTreeMap<String, Buffer>) -> Result<()> {
        self.restore_kv(session, buffers)?;
        self.grow_restored(session, buffers)?;
        self.restore_mutable_into(session, buffers)
    }
    #[cfg(test)]
    fn grow_restored(&self, session: &Session, buffers: &BTreeMap<String, Buffer>) -> Result<()> {
        grow_restored(
            session,
            buffers,
            self.pieces
                .iter()
                .map(|p| (p.name.as_str(), p.offset + p.buffer.bytes())),
        )
    }
    fn restore_mutable_into(
        &self,
        session: &Session,
        buffers: &BTreeMap<String, Buffer>,
    ) -> Result<()> {
        let copies = self
            .pieces
            .iter()
            .filter(|p| p.buffer.allocation.handle.is_none())
            .map(|p| {
                Ok((
                    p.buffer.native()?,
                    0,
                    get(buffers, &p.name)?,
                    p.offset,
                    p.buffer.bytes(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if !copies.is_empty() {
            session.copy_many(&copies)?.wait()?;
        }
        Ok(())
    }
}
#[cfg(test)]
fn grow_restored<'a>(
    session: &Session,
    buffers: &BTreeMap<String, Buffer>,
    pieces: impl Iterator<Item = (&'a str, usize)>,
) -> Result<()> {
    let mut extents = BTreeMap::<&str, usize>::new();
    for (name, end) in pieces {
        if buffers[name].demand_mapped() {
            extents
                .entry(name)
                .and_modify(|n| *n = (*n).max(end))
                .or_insert(end);
        }
    }
    session.grow_many(
        &extents
            .into_iter()
            .map(|(name, end)| Ok((get(buffers, name)?, end)))
            .collect::<Result<Vec<_>>>()?,
    )
}

impl Worker {
    pub(super) fn release_owner(&mut self, id: usize) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.remove(&id).ok_or("Unknown A2 owner")?;
        if id == 0 {
            self.clear_base_growth()?;
        }
        Ok(())
    }
    fn snapshot_state(&self, id: usize, complete: bool) -> Result<(&Owner, Vec<&StateLayout>)> {
        let runtime = self.batch.as_ref().ok_or("Missing A2 batch runtime")?;
        if runtime.contract.state.is_empty() {
            return Err("A2 package does not declare persistent state".into());
        }
        let owner = runtime.owners.get(&id).ok_or("Unknown A2 owner")?;
        let decoder = runtime
            .contract
            .decoder_replay
            .as_ref()
            .filter(|r| complete && r.final_prompt_only && !r.decoder_state.is_empty());
        if complete && decoder.is_none() {
            return Err("A2 package does not declare complete-prompt decoder state".into());
        }
        Ok((
            owner,
            runtime
                .contract
                .state
                .iter()
                .chain(decoder.into_iter().flat_map(|r| &r.decoder_state))
                .collect(),
        ))
    }
    pub(super) fn plan_owner_snapshot(
        &self,
        id: usize,
        complete: bool,
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<BTreeMap<u64, usize>> {
        let (owner, layouts) = self.snapshot_state(id, complete)?;
        Snapshot::plan_device(layouts.into_iter(), &owner.buffers, &owner.history, parent)?
            .allocations(bank_id)
    }
    pub(super) fn save_owner(
        &self,
        id: usize,
        complete: bool,
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<Snapshot> {
        let (
            Owner {
                buffers, history, ..
            },
            layouts,
        ) = self.snapshot_state(id, complete)?;
        Snapshot::capture(
            &self.session,
            layouts.into_iter(),
            buffers,
            history,
            bank_id,
            parent,
        )
    }

    pub(super) fn restore_owner(&mut self, id: usize, snapshot: &Snapshot) -> Result<()> {
        self.reset_owner(id)?;
        {
            let runtime = self.batch.as_ref().ok_or("Missing A2 runtime")?;
            snapshot.restore_kv(
                &self.session,
                &runtime.owners.get(&id).ok_or("Unknown owner")?.buffers,
            )?;
        }
        self.grow_owner_readonly(id, snapshot.history.len())?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        let owner = runtime.owners.get_mut(&id).ok_or("Unknown A2 owner")?;
        snapshot.restore_mutable_into(&self.session, &owner.buffers)?;
        owner.history.clone_from(&snapshot.history);
        runtime.bound_tables = None;
        Ok(())
    }

    /// Verification only overwrites windows and recurrent rings. Closed linear
    /// KV rows keep their bytes and become invisible when history is restored.
    pub(super) fn save_mutable(
        &mut self,
        id: usize,
        rows: usize,
        previous: Option<Journal>,
    ) -> Result<Journal> {
        if !(1..=crate::execution::MAX_MTP_DRAFTS + 1).contains(&rows) {
            return Err("Verification rows exceed the C2 ring capacity".into());
        }
        let (owner, layouts) = self.snapshot_state(id, true)?;
        let history = owner.history.len();
        let mut pending = vec![];
        let mut bytes = 0usize;
        let mut circular = BTreeMap::new();
        for layout in layouts {
            let source = get(&owner.buffers, &layout.buffer)?;
            let window = self
                .batch
                .as_ref()
                .unwrap()
                .contract
                .verification_windows
                .get(&layout.buffer);
            let spans = if let Some(window) = window {
                window.overwritten_spans(history, rows, source.bytes())?
            } else {
                layout.overwritten_spans(history, rows, source.bytes())?
            };
            if !spans.is_empty() && (layout.circular || window.is_some()) {
                circular.insert(
                    layout.buffer.clone(),
                    window.map_or(layout.row_bytes, |w| w.row_bytes),
                );
            }
            for span in spans {
                bytes = bytes.checked_add(255).ok_or("Journal alignment overflow")? / 256 * 256;
                pending.push((layout.buffer.clone(), source.clone(), span.clone(), bytes));
                bytes = bytes
                    .checked_add(span.len())
                    .ok_or("Journal size overflow")?;
            }
        }
        let bytes = bytes.checked_add(255).ok_or("Journal size overflow")?;
        let bank = if let Some(saved) = previous.filter(|saved| saved.bank.bytes() >= bytes) {
            saved.bank
        } else {
            self.reserve_batch_memory(
                super::memory::Requirement::snapshot(bytes)?,
                "DSpark state journal",
            )?;
            self.session.allocate(bytes)?
        };
        let address = bank.pointer() as usize;
        let base = address.div_ceil(256) * 256 - address;
        let mut pieces = vec![];
        let mut copies = vec![];
        for (name, source, span, offset) in pending {
            let buffer = bank.slice(base + offset, span.len())?;
            copies.push((source, span.start, buffer.clone(), 0, span.len()));
            pieces.push(Piece {
                name,
                offset: span.start,
                buffer,
                bank_id: 0,
                bank_bytes: bank.bytes(),
            });
        }
        self.session.copy_many(&copies)?.wait()?;
        Ok(Journal {
            history,
            rows,
            pieces,
            bank,
            circular,
        })
    }

    pub(super) fn restore_mutable(
        &mut self,
        id: usize,
        snapshot: &Journal,
        keep: usize,
    ) -> Result<()> {
        if !(1..=snapshot.rows).contains(&keep) {
            return Err("Invalid DSpark accepted input count".into());
        }
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        let owner = runtime.owners.get_mut(&id).ok_or("Unknown A2 owner")?;
        let mut remaining = snapshot
            .circular
            .iter()
            .map(|(name, stride)| (name.clone(), keep * stride))
            .collect::<BTreeMap<_, _>>();
        let mut copies = vec![];
        for piece in &snapshot.pieces {
            let skip = remaining.get_mut(&piece.name).map_or(0, |left| {
                let count = (*left).min(piece.buffer.bytes());
                *left -= count;
                count
            });
            if skip < piece.buffer.bytes() {
                copies.push((
                    piece.buffer.clone(),
                    skip,
                    get(&owner.buffers, &piece.name)?,
                    piece.offset + skip,
                    piece.buffer.bytes() - skip,
                ));
            }
        }
        if !copies.is_empty() {
            self.session.copy_many(&copies)?.wait()?;
        }
        let confirmed = snapshot
            .history
            .checked_add(keep)
            .ok_or("Verification history overflow")?;
        if owner.history.len() < confirmed {
            return Err("Verification history precedes journal".into());
        }
        owner.history.truncate(confirmed);
        runtime.bound_tables = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;

    #[test]
    fn compressed_snapshot_restores_after_crossing_a_physical_page_boundary() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let grain = 2 << 20;
        let source = session.reserve(grain * 2).unwrap();
        let buffers = BTreeMap::from([("kv".into(), source.clone())]);
        let layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        session.grow(&source, 32).unwrap();
        session
            .upload(&source, 0, &[73; 32])
            .unwrap()
            .wait()
            .unwrap();
        let parent = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[9; 4],
            1,
            None,
        )
        .unwrap();
        session.grow(&source, grain + 32).unwrap();
        session
            .upload(&source, grain, &[91; 32])
            .unwrap()
            .wait()
            .unwrap();
        let history = vec![9; (grain / 16 + 2) * 2];
        let child = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &history,
            2,
            Some(&parent),
        )
        .unwrap();
        drop(parent);
        let target = session.reserve(grain * 2).unwrap();
        session.grow(&target, grain + 32).unwrap();
        child
            .restore_into(&session, &BTreeMap::from([("kv".into(), target.clone())]))
            .unwrap();
        assert_eq!(session.download(&target, 0, 32).unwrap(), vec![73; 32]);
        assert_eq!(session.download(&target, grain, 32).unwrap(), vec![91; 32]);
        drop(child);
        drop(buffers);
        drop(source);
        drop(target);
        drop(session);
        fixture.assert_clean();
    }

    #[test]
    fn evicting_a_parent_requires_replanning_the_new_bank() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let buffers = BTreeMap::from([("kv".into(), session.allocate(128).unwrap())]);
        let layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        let parent = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3],
            1,
            None,
        )
        .unwrap();
        let before = session.allocated_bytes();
        let child = Snapshot::<DeviceBuffer>::plan(
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            Some(&parent),
        )
        .unwrap();
        assert_eq!(session.allocated_bytes(), before);
        assert_eq!(child.bytes, 32);
        assert!(
            child
                .allocations(2)
                .unwrap()
                .contains_key(&kv::checkpoint_id(1).unwrap())
        );
        drop(child);
        drop(parent);
        let replanned = Snapshot::<DeviceBuffer>::plan(
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            None,
        )
        .unwrap();
        assert_eq!(replanned.bytes, 64);
        assert_eq!(replanned.allocations(2).unwrap().len(), 1);
        let saved = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            2,
            None,
        )
        .unwrap();
        assert_eq!(saved.allocations(), replanned.allocations(2).unwrap());
        drop(replanned);
        drop(saved);
        drop(buffers);
        drop(session);
        fixture.assert_clean();
    }

    #[test]
    fn inherited_closed_rows_survive_parent_eviction_and_keep_mutable_state_private() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let layouts = [
            StateLayout {
                buffer: "kv".into(),
                row_bytes: 16,
                tokens_per_row: 2,
                window: None,
                circular: false,
                append_only: true,
            },
            StateLayout {
                buffer: "swa".into(),
                row_bytes: 4,
                tokens_per_row: 1,
                window: Some(3),
                circular: true,
                append_only: false,
            },
            StateLayout {
                buffer: "ring".into(),
                row_bytes: 64,
                tokens_per_row: 0,
                window: None,
                circular: false,
                append_only: false,
            },
        ];
        let buffers = [("kv", 128), ("swa", 32), ("ring", 64)]
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), session.allocate(bytes).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let mut parent = None;
        let mut reference = None;
        for (index, length) in [3usize, 5, 9].into_iter().enumerate() {
            let kv = (0..128).map(|i| (i / 16 + 1) as u8).collect::<Vec<_>>();
            let mut swa = vec![0; 32];
            for token in 0..length {
                swa[token % 8 * 4..token % 8 * 4 + 4].fill(token as u8 + 1);
            }
            session
                .upload_inputs(&[
                    (buffers["kv"].clone(), kv),
                    (buffers["swa"].clone(), swa),
                    (buffers["ring"].clone(), vec![length as u8; 64]),
                ])
                .unwrap()
                .wait()
                .unwrap();
            let history = (0..length as u32).collect::<Vec<_>>();
            let before_plan = session.allocated_bytes();
            let planned =
                Snapshot::<DeviceBuffer>::plan(layouts.iter(), &buffers, &history, parent.as_ref())
                    .unwrap()
                    .allocations(index as u64 + 1)
                    .unwrap();
            assert_eq!(
                session.allocated_bytes(),
                before_plan,
                "Planning cannot allocate device memory"
            );
            let saved = Snapshot::capture(
                &session,
                layouts.iter(),
                &buffers,
                &history,
                index as u64 + 1,
                parent.as_ref(),
            )
            .unwrap();
            assert_eq!(saved.allocations(), planned);
            let kv_pieces = saved
                .pieces
                .iter()
                .filter(|p| p.name == "kv")
                .collect::<Vec<_>>();
            assert_eq!(kv_pieces.len(), index + 1);
            assert_eq!(kv_pieces[0].bank_id, kv::checkpoint_id(1).unwrap());
            assert_eq!(kv_pieces.last().unwrap().offset, [0, 16, 32][index]);
            assert!(
                saved
                    .pieces
                    .iter()
                    .filter(|p| p.name != "kv")
                    .all(|p| p.bank_id == kv::checkpoint_id(index as u64 + 1).unwrap())
            );
            assert!(
                Snapshot::capture(
                    &session,
                    layouts.iter(),
                    &buffers,
                    &[99; 9],
                    20,
                    Some(&saved)
                )
                .is_err()
            );
            let mut full_layouts = layouts.clone();
            full_layouts[0].append_only = false;
            reference = Some(
                Snapshot::capture(&session, full_layouts.iter(), &buffers, &history, 100, None)
                    .unwrap(),
            );
            // Dropping the parent must preserve the child's inherited banks;
            // mutable spans still point exclusively to this capture's bank.
            parent = Some(saved);
        }
        drop(buffers);
        let saved = parent.unwrap();
        assert_eq!(saved.allocations().len(), 3);
        let target = || {
            [("kv", 128), ("swa", 32), ("ring", 64)]
                .into_iter()
                .map(|(name, bytes)| {
                    let buffer = session.allocate(bytes).unwrap();
                    session.fill(&buffer, 0).unwrap().wait().unwrap();
                    (name.to_owned(), buffer)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let actual = target();
        let expected = target();
        saved.restore_into(&session, &actual).unwrap();
        reference
            .as_ref()
            .unwrap()
            .restore_into(&session, &expected)
            .unwrap();
        for name in ["kv", "swa", "ring"] {
            assert_eq!(
                session
                    .download(&actual[name], 0, actual[name].bytes())
                    .unwrap(),
                session
                    .download(&expected[name], 0, expected[name].bytes())
                    .unwrap(),
                "{name}"
            );
        }
        assert_eq!(session.download(&actual["kv"], 64, 64).unwrap(), [0; 64]);
        assert_eq!(session.download(&actual["swa"], 4, 20).unwrap(), [0; 20]);
        drop((saved, reference, actual, expected, session));
        fixture.assert_clean();
    }

    #[test]
    fn shared_rows_reject_mutable_geometry() {
        let mut layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        assert_eq!(layout.spans(5, 128).unwrap(), vec![0..32]);
        layout.tokens_per_row = 0;
        assert!(layout.spans(5, 128).is_err());
        layout.tokens_per_row = 1;
        layout.window = Some(3);
        assert!(layout.spans(5, 128).is_err());
        layout.window = None;
        layout.circular = true;
        assert!(layout.spans(5, 128).is_err());
    }

    #[test]
    fn state_spans_preserve_partial_compression_and_wrapped_window() {
        let mut layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: false,
        };
        assert_eq!(layout.spans(129, 2048).unwrap(), vec![0..1024]);
        assert!(layout.spans(258, 2048).is_err());
        layout.tokens_per_row = 1;
        layout.window = Some(3);
        layout.circular = true;
        assert_eq!(layout.spans(9, 128).unwrap(), vec![96..128, 0..16]);
        assert_eq!(layout.spans(8, 128).unwrap(), vec![80..128]);
        assert!(layout.spans(8, 0).is_err());
        layout.tokens_per_row = 0;
        assert_eq!(layout.spans(999, 128).unwrap(), vec![0..128]);
    }
}

#[cfg(test)]
mod journal_tests {
    use super::*;

    #[test]
    fn verification_journal_covers_only_overwritten_slots_across_wraps() {
        let layout = StateLayout {
            buffer: "swa".into(),
            row_bytes: 16,
            tokens_per_row: 1,
            window: Some(128),
            circular: true,
            append_only: false,
        };
        for count in [1, 6, 9, 17, 32] {
            for start in [0, 1, 122, 123, 126, 127, 128, 2047, 262138] {
                let spans = layout.overwritten_spans(start, count, 128 * 16).unwrap();
                assert_eq!(spans.iter().map(|s| s.len()).sum::<usize>(), count * 16);
                let rows = spans
                    .into_iter()
                    .flat_map(|s| (s.start / 16)..(s.end / 16))
                    .collect::<Vec<_>>();
                assert_eq!(
                    rows,
                    (start..start + count).map(|p| p % 128).collect::<Vec<_>>()
                );
            }
        }
        let mut linear = layout.clone();
        linear.circular = false;
        assert!(
            linear
                .overwritten_spans(120, 6, 128 * 16)
                .unwrap()
                .is_empty()
        );
        assert!(
            layout
                .overwritten_spans(2047, 6, 4096 * 16)
                .unwrap()
                .is_empty()
        );
        let c2 = AppendWindow {
            offset: 32 * 4096,
            row_bytes: 4096,
            rows: 32,
            history: 1,
        };
        for start in [0, 31, 32, 2047, 2048, 2049] {
            for count in [6, 9, 17, 31] {
                assert!(
                    c2.overwritten_spans(start, count, 64 * 4096)
                        .unwrap()
                        .is_empty()
                );
            }
            let spans = c2.overwritten_spans(start, 32, 64 * 4096).unwrap();
            assert_eq!(spans.iter().map(|s| s.len()).sum::<usize>(), 32 * 4096);
            assert!(c2.overwritten_spans(start, 33, 64 * 4096).is_err());
        }
        let recurrent = StateLayout {
            tokens_per_row: 0,
            window: None,
            circular: false,
            ..layout
        };
        assert_eq!(
            recurrent.overwritten_spans(2049, 6, 4096).unwrap(),
            vec![0..4096]
        );
    }
}
