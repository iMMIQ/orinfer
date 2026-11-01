//! Shared immutable KV extents plus private endpoint state. Restore keeps graph VAs.
use super::executor::Executor;
use super::{Result, check, sys};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
    time::Instant,
};

use crate::backend::snapshot::size;
pub(crate) use crate::backend::snapshot::{Allocation, Piece, Plan, Range, Snapshot, Workspace};
impl Executor {
    #[cfg(test)]
    pub(crate) fn snapshot_resident_bytes(&self) -> usize {
        self.session
            .snapshot_allocations
            .borrow()
            .iter()
            .map(|a| a.bytes)
            .sum()
    }
    pub(crate) fn plan_snapshot(
        &self,
        ranges: BTreeMap<String, Range>,
        kv_names: &BTreeSet<String>,
        base: Option<&Snapshot>,
    ) -> Result<Plan> {
        if let Some(base) = base {
            base.require_owner(self.session.owner)?;
        }
        let mut tails = BTreeMap::new();
        let (mut state_ranges, mut shared, mut kv) = (BTreeMap::new(), BTreeMap::new(), vec![]);
        for (name, range) in ranges {
            range.within(*self.sizes.get(&name).ok_or("Unknown snapshot buffer")?)?;
            if !kv_names.contains(&name) {
                state_ranges.insert(name, range);
                continue;
            }
            if range.offset != 0 {
                return Err("KV snapshot must start at zero".into());
            }
            if let Some(buffer) = self.session.virtual_buffers.borrow().get(&name) {
                let regions = crate::backend::kv::checkpoint_regions(
                    buffer.preview(self.session.owner, range)?,
                )?;
                if let Some(suffix) = regions.tail {
                    tails.insert(name.clone(), suffix);
                    let candidates = base
                        .into_iter()
                        .flat_map(|p| p.state.iter().chain(&p.kv))
                        .filter(|p| p.name == name && p.allocation.handle.is_none())
                        .collect::<Vec<_>>();
                    let reuse =
                        crate::backend::kv::reuse_tail(suffix, candidates.iter().map(|p| p.range))?;
                    for part in reuse.parts {
                        let p = candidates[part.index];
                        kv.push(Piece {
                            allocation: p.allocation.clone(),
                            name: name.clone(),
                            range: part.range,
                            source: p
                                .source
                                .checked_add(part.source_offset)
                                .ok_or("KV source overflow")?,
                        });
                    }
                    if reuse.missing.bytes > 0 {
                        state_ranges.insert(name.clone(), reuse.missing);
                    }
                }
                let mut end = 0;
                for (allocation, range, source) in regions.shared {
                    end = range.offset + range.bytes;
                    kv.push(Piece {
                        allocation,
                        name: name.clone(),
                        range,
                        source,
                    });
                }
                if end > 0 {
                    shared.insert(
                        name,
                        Range {
                            offset: 0,
                            bytes: end,
                        },
                    );
                }
            } else {
                state_ranges.insert(name, range);
            }
        }
        let new_bytes = size(&state_ranges)?;
        Ok(Plan {
            owner: self.session.owner,
            state: state_ranges,
            shared,
            tails,
            host_inherited: vec![],
            kv,
            new_bytes,
        })
    }
    pub(crate) fn snapshot(&mut self, plan: &Plan) -> Result<Option<Snapshot>> {
        self.session.owner.require(plan.owner)?;
        if !plan.host_inherited.is_empty() {
            return Err("CPU checkpoint plan cannot be captured in device tier".into());
        }
        self.sync()?;
        self.collect_snapshots()?;
        let Some(state) = self.save_ranges(plan.state.clone())? else {
            return Ok(None);
        };
        let Some(mut kv) = self.save_kv_ranges(&plan.shared)? else {
            drop(state);
            self.collect_snapshots()?;
            return Ok(None);
        };
        kv.extend(
            plan.kv
                .iter()
                .filter(|p| p.allocation.handle.is_none())
                .cloned(),
        );
        let bytes = state.iter().chain(&kv).map(|p| p.range.bytes).sum();
        self.sync()?;
        Ok(Some(Snapshot {
            owner: self.session.owner,
            bytes,
            state,
            kv,
        }))
    }
    fn save_ranges(&mut self, ranges: BTreeMap<String, Range>) -> Result<Option<Vec<Piece>>> {
        self.allocate_ranges(ranges, true)
    }
    fn allocate_ranges(
        &mut self,
        ranges: BTreeMap<String, Range>,
        copy: bool,
    ) -> Result<Option<Vec<Piece>>> {
        let bytes = size(&ranges)?;
        if bytes == 0 {
            return Ok(Some(vec![]));
        }
        let (mut free, mut total, mut address) = (0, 0, 0);
        // SAFETY: Live context, checked extents. Ownership transfers to Session
        // before any fallible copy; its destructor also handles error paths.
        unsafe {
            check(
                sys::cuMemGetInfo_v2(&mut free, &mut total),
                "snapshot memory",
            )?;
            if bytes > free.saturating_sub(64 * 1024 * 1024) {
                return Ok(None);
            }
            let status = sys::cuMemAlloc_v2(&mut address, bytes);
            if status == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
                return Ok(None);
            }
            check(status, "allocate prefix snapshot")?;
        }
        self.session.buffers.push(address);
        let allocation = Rc::new(Allocation {
            id: crate::backend::kv::allocation_id()?,
            used: bytes,
            native: None,
            owner: self.session.owner,
            address,
            bytes,
            handle: None,
        });
        self.session
            .snapshot_allocations
            .borrow_mut()
            .push(Rc::clone(&allocation));
        let (mut pieces, mut source) = (vec![], 0);
        for (name, range) in ranges {
            if range.bytes != 0 {
                if copy {
                    // SAFETY: Source/destination extents are validated; one stream
                    // orders all writes before the immutable snapshot is published.
                    unsafe {
                        check(
                            sys::cuMemcpyDtoDAsync_v2(
                                address + source as u64,
                                self.pointers[&name] + range.offset as u64,
                                range.bytes,
                                self.session.stream,
                            ),
                            "save prefix extent",
                        )?;
                    }
                }
                let count = range.bytes;
                pieces.push(Piece {
                    allocation: Rc::clone(&allocation),
                    name,
                    range,
                    source,
                });
                source = (source + count + 255) & !255;
            }
        }
        Ok(Some(pieces))
    }
    pub(crate) fn reserve_workspace(
        &mut self,
        ranges: BTreeMap<String, Range>,
    ) -> Result<Option<Rc<RefCell<Workspace>>>> {
        let plan = self.plan_snapshot(ranges, &BTreeSet::new(), None)?;
        self.collect_snapshots()?;
        let Some(state) = self.allocate_ranges(plan.state, false)? else {
            return Ok(None);
        };
        let bytes = state.iter().map(|p| p.range.bytes).sum();
        Ok(Some(Rc::new(RefCell::new(Workspace {
            snapshot: Snapshot {
                owner: self.session.owner,
                bytes,
                state,
                kv: vec![],
            },
            resident: true,
            saved_s: 0.,
        }))))
    }
    pub(crate) fn retain_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<()> {
        workspace
            .borrow()
            .snapshot
            .require_owner(self.session.owner)?;
        if self
            .live_workspace
            .as_ref()
            .and_then(|(_, w)| w.upgrade())
            .is_some()
        {
            return Err("Shared workspace already has a live continuation".into());
        }
        self.live_workspace = Some((self.active_sequence, Rc::downgrade(workspace)));
        Ok(())
    }
    pub(super) fn save_live_workspace(&mut self) -> Result<()> {
        let Some((owner, weak)) = &self.live_workspace else {
            return Ok(());
        };
        let Some(workspace) = weak.upgrade() else {
            self.live_workspace = None;
            return Ok(());
        };
        if *owner != self.active_sequence {
            return Err("Live workspace belongs to an inactive sequence".into());
        }
        let at = Instant::now();
        let mut workspace = workspace.borrow_mut();
        if !workspace.resident {
            return Err("Live workspace has already been saved".into());
        }
        for p in &workspace.snapshot.state {
            // SAFETY: The backing and source extents were validated at reserve;
            // they are private, live, and ordered with preceding layer writes.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        p.allocation.address + p.source as u64,
                        self.pointers[&p.name] + p.range.offset as u64,
                        p.range.bytes,
                        self.session.stream,
                    ),
                    "save live workspace",
                )?;
            }
        }
        self.sync()?;
        workspace.resident = false;
        let seconds = at.elapsed().as_secs_f64();
        workspace.saved_s += seconds;
        self.workspace_save_s += seconds;
        self.live_workspace = None;
        Ok(())
    }
    pub(crate) fn resume_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<f64> {
        workspace
            .borrow()
            .snapshot
            .require_owner(self.session.owner)?;
        if let Some((_, live)) = &self.live_workspace
            && live
                .upgrade()
                .is_some_and(|live| Rc::ptr_eq(&live, workspace))
        {
            self.live_workspace = None;
        } else {
            self.save_live_workspace()?;
        }
        let mut workspace = workspace.borrow_mut();
        let at = Instant::now();
        let seconds = if workspace.resident {
            0.
        } else {
            self.restore_snapshot(&workspace.snapshot)?;
            workspace.resident = true;
            let seconds = at.elapsed().as_secs_f64();
            self.workspace_restore_s += seconds;
            seconds
        };
        Ok(seconds + std::mem::take(&mut workspace.saved_s))
    }
    fn save_kv_ranges(&mut self, ranges: &BTreeMap<String, Range>) -> Result<Option<Vec<Piece>>> {
        let mut pieces = vec![];
        for (name, range) in ranges {
            let mut buffers = self.session.virtual_buffers.borrow_mut();
            let Some(buffer) = buffers.get_mut(name) else {
                continue;
            };
            for (allocation, range, source) in buffer.publish(&self.session, *range)? {
                pieces.push(Piece {
                    allocation,
                    name: name.clone(),
                    range,
                    source,
                });
            }
        }
        Ok(Some(pieces))
    }
    pub(crate) fn restore_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        snapshot.require_owner(self.session.owner)?;
        self.sync()?;
        // Map complete immutable KV allocations at fixed request VAs. Partial
        // allocation slices and endpoint state retain the byte-exact copy path.
        let mut shared = BTreeMap::<&str, Vec<&Piece>>::new();
        for p in &snapshot.kv {
            shared.entry(&p.name).or_default().push(p);
        }
        let mut aliased = BTreeSet::<(&str, usize)>::new();
        for (name, pieces) in &mut shared {
            pieces.sort_by_key(|p| p.range.offset);
            let mut buffers = self.session.virtual_buffers.borrow_mut();
            let Some(v) = buffers.get_mut(*name) else {
                continue;
            };
            if !pieces.iter().any(|p| can_alias(p, v.granularity)) {
                continue;
            }
            v.release_slabs(&self.session.driver)?;
            for p in pieces {
                if can_alias(p, v.granularity) {
                    let offset = crate::backend::kv::alias_offset(
                        &p.allocation,
                        p.range,
                        p.source,
                        v.granularity,
                        v.bytes,
                    )
                    .ok_or("Invalid shared KV alias")?;
                    v.grow_bytes(&self.session, offset)?;
                    v.map_shared(&self.session, offset, Rc::clone(&p.allocation))?;
                    aliased.insert((name, p.range.offset));
                } else {
                    v.grow_bytes(&self.session, p.range.offset + p.range.bytes)?;
                }
            }
        }
        // Grow once to each final extent, then restore private fragments/state.
        let mut extents: BTreeMap<&str, usize> = BTreeMap::new();
        for p in snapshot.kv.iter().chain(&snapshot.state) {
            if aliased.contains(&(p.name.as_str(), p.range.offset)) {
                continue;
            }
            if let Some(v) = self.session.virtual_buffers.borrow_mut().get_mut(&p.name) {
                v.make_writable(&self.session, p.range.offset, p.range.bytes)?;
            }
            let end = p
                .range
                .offset
                .checked_add(p.range.bytes)
                .ok_or("Restore extent overflow")?;
            extents
                .entry(&p.name)
                .and_modify(|e| *e = (*e).max(end))
                .or_insert(end);
        }
        for (name, count) in extents {
            if let Some(v) = self.session.virtual_buffers.borrow_mut().get_mut(name) {
                v.grow(&self.session, count.div_ceil(v.stride))?;
            }
        }
        for p in snapshot.kv.iter().chain(&snapshot.state) {
            if aliased.contains(&(p.name.as_str(), p.range.offset)) {
                continue;
            }
            // SAFETY: Immutable allocations remain referenced; ranges were checked
            // at capture, and graph-bound destinations retain their original VAs.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        self.pointers[&p.name] + p.range.offset as u64,
                        p.allocation.address + p.source as u64,
                        p.range.bytes,
                        self.session.stream,
                    ),
                    "restore prefix extent",
                )?;
            }
        }
        self.sync()
    }
    pub(crate) fn release_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        drop(snapshot);
        self.collect_snapshots()
    }
    pub(crate) fn collect_snapshots(&mut self) -> Result<()> {
        if self
            .session
            .snapshot_allocations
            .borrow()
            .iter()
            .all(|a| Rc::strong_count(a) > 1)
        {
            return Ok(());
        }
        self.sync()?;
        let mut allocations = self.session.snapshot_allocations.borrow_mut();
        let mut i = 0;
        while i < allocations.len() {
            let a = &allocations[i];
            if Rc::strong_count(a) != 1 {
                i += 1;
                continue;
            }
            if a.handle.is_some() {
                let key = a.address.to_string();
                let mut buffers = self.session.prefix_buffers.borrow_mut();
                let v = buffers
                    .get_mut(&key)
                    .ok_or("Unowned shared prefix allocation")?;
                v.release_slabs(&self.session.driver)?;
                // SAFETY: Registry is the only owner, mappings and consumers are retired.
                unsafe {
                    check(
                        sys::cuMemAddressFree(v.address, v.bytes),
                        "release shared prefix address",
                    )?;
                }
                buffers.remove(&key);
                allocations.swap_remove(i);
                continue;
            }
            let index = self
                .session
                .buffers
                .iter()
                .position(|&p| p == a.address)
                .ok_or("Unowned snapshot allocation")?;
            // SAFETY: Stream completed and registry is the only remaining owner.
            unsafe {
                check(sys::cuMemFree_v2(a.address), "release prefix allocation")?;
            }
            self.session.buffers.swap_remove(index);
            allocations.swap_remove(i);
        }
        Ok(())
    }
}
fn can_alias(p: &Piece, granularity: usize) -> bool {
    crate::backend::kv::alias_offset(&p.allocation, p.range, p.source, granularity, usize::MAX)
        .is_some()
}
impl Executor {
    fn read_host_regions(
        &self,
        regions: &[crate::backend::kv::host::ReadRegion],
        output: &mut [u8],
    ) -> Result<()> {
        for (allocation, source, dest, bytes) in regions {
            allocation.owner.require(self.session.owner)?;
            Range {
                offset: *source,
                bytes: *bytes,
            }
            .within(allocation.used)?;
            Range {
                offset: *dest,
                bytes: *bytes,
            }
            .within(output.len())?;
        }
        let mut index = 0;
        let mut offset = 0;
        let mut staging = vec![0; 4 << 20];
        while index < regions.len() {
            let mut spans = vec![];
            let mut destinations = vec![];
            let mut written = 0;
            while index < regions.len() && written < staging.len() {
                let (allocation, source, dest, bytes) = &regions[index];
                let count = (bytes - offset).min(staging.len() - written);
                if count > 0 {
                    spans.push((allocation.address + (source + offset) as u64, count));
                    destinations.push((dest + offset, written, count));
                }
                written += count;
                offset += count;
                if offset == *bytes {
                    index += 1;
                    offset = 0;
                }
            }
            self.session.transfers.borrow_mut().bulk_readback(
                self.session.stream,
                &spans,
                &mut staging[..written],
            )?;
            for (dest, start, bytes) in destinations {
                output[dest..dest + bytes].copy_from_slice(&staging[start..start + bytes]);
            }
        }
        Ok(())
    }
    pub(crate) fn offload_snapshot(
        &self,
        snapshot: &Snapshot,
        pool: &mut crate::backend::kv::host::HostPool,
    ) -> Result<crate::backend::kv::host::HostSnapshot> {
        snapshot.require_owner(self.session.owner)?;
        self.sync()?;
        pool.capture(snapshot, |regions, output| {
            self.read_host_regions(regions, output)
        })
    }
    pub(crate) fn host_snapshot(
        &mut self,
        plan: &Plan,
        pool: &mut crate::backend::kv::host::HostPool,
    ) -> Result<crate::backend::kv::host::HostSnapshot> {
        self.session.owner.require(plan.owner)?;
        self.sync()?;
        let mut kv = self
            .save_kv_ranges(&plan.shared)?
            .ok_or("KV publication failed")?;
        kv.extend(
            plan.kv
                .iter()
                .filter(|p| p.allocation.handle.is_none())
                .cloned(),
        );
        let state = plan
            .state
            .iter()
            .map(|(name, range)| Piece {
                name: name.clone(),
                range: *range,
                source: range.offset,
                allocation: Rc::new(Allocation {
                    owner: self.session.owner,
                    id: 0,
                    address: self.pointers[name],
                    bytes: self.sizes[name],
                    used: self.sizes[name],
                    handle: None,
                    native: None,
                }),
            })
            .collect::<Vec<_>>();
        let mut snapshot =
            pool.capture_components(self.session.owner, &kv, &state, |regions, output| {
                self.read_host_regions(regions, output)
            })?;
        snapshot.bytes += plan
            .host_inherited
            .iter()
            .map(|p| p.range.bytes)
            .sum::<usize>();
        snapshot.kv.extend(plan.host_inherited.iter().cloned());
        Ok(snapshot)
    }
    fn materialize_host_kv(
        &self,
        bank: &crate::backend::kv::host::HostBank,
    ) -> Result<Rc<Allocation>> {
        bank.materialize(self.session.owner, |descriptor, bytes| {
            super::virtual_memory::restore_host(&self.session, descriptor, bytes)
        })
    }
    pub(crate) fn restore_host_snapshot(
        &self,
        snapshot: &crate::backend::kv::host::HostSnapshot,
    ) -> Result<()> {
        let shared =
            snapshot.restore_shared(self.session.owner, |bank| self.materialize_host_kv(bank))?;
        self.restore_snapshot(&shared)?;
        for piece in snapshot
            .state
            .iter()
            .chain(&snapshot.kv)
            .filter(|p| !p.data.bank.descriptor().is_some_and(|d| d.paged))
        {
            let source = piece.data.bank.bytes.borrow();
            Range {
                offset: piece.data.offset,
                bytes: piece.range.bytes,
            }
            .within(source.len())?;
            Range {
                offset: piece.range.offset,
                bytes: piece.range.bytes,
            }
            .within(self.sizes[&piece.name])?;
            self.ensure_writable_range(&piece.name, piece.range.offset, piece.range.bytes)?;
            self.session.transfers.borrow_mut().upload(
                self.session.stream,
                self.pointers[&piece.name] + piece.range.offset as u64,
                &source[piece.data.offset..piece.data.offset + piece.range.bytes],
            )?;
        }
        self.sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_valid_prefix_maps_the_same_physical_allocation() {
        let allocation = Rc::new(Allocation {
            owner: orinfer_device::DeviceOwner::new(orinfer_device::Backend::Cuda, 0),
            id: 2,
            address: 0,
            bytes: 8192,
            used: 100,
            handle: Some(7),
            native: None,
        });
        let mut piece = Piece {
            allocation,
            name: "kv".into(),
            range: Range {
                offset: 8192,
                bytes: 100,
            },
            source: 0,
        };
        assert!(can_alias(&piece, 8192));
        piece.range.bytes = 101;
        assert!(!can_alias(&piece, 8192));
        piece.range.bytes = 100;
        piece.range.offset = 1;
        assert!(!can_alias(&piece, 8192));
    }
}
