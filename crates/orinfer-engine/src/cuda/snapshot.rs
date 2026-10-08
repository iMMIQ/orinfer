//! Shared immutable KV extents plus private endpoint state. Restore keeps graph VAs.
use super::executor::Executor;
use super::{Result, check, sys};
use std::{
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

#[derive(Clone, Debug)]
pub(crate) struct Range {
    pub offset: usize,
    pub bytes: usize,
}
pub(crate) struct Allocation {
    address: u64,
    bytes: usize,
}
#[derive(Clone)]
pub(crate) struct Piece {
    allocation: Rc<Allocation>,
    name: String,
    range: Range,
    source: usize,
}
#[derive(Clone)]
pub(crate) struct Snapshot {
    pub bytes: usize,
    state: Vec<Piece>,
    pub kv: Vec<Piece>,
}
pub(crate) struct Plan {
    state: BTreeMap<String, Range>,
    tail: BTreeMap<String, Range>,
    kv: Vec<Piece>,
    pub new_bytes: usize,
}
impl crate::prefix::Resident for Plan {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        for p in &self.kv {
            out.insert(p.allocation.address, p.allocation.bytes);
        }
    }
}
impl crate::prefix::Resident for Snapshot {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        for p in self.state.iter().chain(&self.kv) {
            out.insert(p.allocation.address, p.allocation.bytes);
        }
    }
}
fn size(ranges: &BTreeMap<String, Range>) -> Result<usize> {
    ranges.values().try_fold(0usize, |offset, r| {
        offset
            .checked_add(r.bytes)
            .and_then(|n| n.checked_add(255))
            .map(|n| n & !255)
            .ok_or("Snapshot size overflow".into())
    })
}
impl Executor {
    #[cfg(test)]
    pub(crate) fn snapshot_resident_bytes(&self) -> usize {
        self.snapshot_allocations.iter().map(|a| a.bytes).sum()
    }
    pub(crate) fn plan_snapshot(
        &self,
        ranges: BTreeMap<String, Range>,
        kv_names: &BTreeSet<String>,
        base: &[Piece],
    ) -> Result<Plan> {
        let (mut state_ranges, mut new_kv, mut kv) = (BTreeMap::new(), BTreeMap::new(), vec![]);
        for (name, range) in ranges {
            let end = range
                .offset
                .checked_add(range.bytes)
                .ok_or("Snapshot extent overflow")?;
            if self.sizes.get(&name).is_none_or(|&bytes| end > bytes)
                || self
                    .session
                    .virtual_buffers
                    .borrow()
                    .get(&name)
                    .is_some_and(|v| end > v.mapped)
            {
                return Err(format!("{name}: snapshot outside resident allocation"));
            }
            if !kv_names.contains(&name) {
                state_ranges.insert(name, range);
                continue;
            }
            if range.offset != 0 {
                return Err("KV snapshot must start at zero".into());
            }
            let mut covered = 0;
            for p in base.iter().filter(|p| p.name == name) {
                if covered == range.bytes {
                    break;
                }
                if p.range.offset != covered {
                    return Err("Non-contiguous shared KV extent".into());
                }
                let mut p = p.clone();
                p.range.bytes = p.range.bytes.min(range.bytes - covered);
                covered += p.range.bytes;
                kv.push(p);
            }
            if covered < range.bytes {
                new_kv.insert(
                    name,
                    Range {
                        offset: covered,
                        bytes: range.bytes - covered,
                    },
                );
            }
        }
        let new_bytes = size(&state_ranges)?
            .checked_add(size(&new_kv)?)
            .ok_or("Snapshot size overflow")?;
        Ok(Plan {
            state: state_ranges,
            tail: new_kv,
            kv,
            new_bytes,
        })
    }
    pub(crate) fn snapshot(&mut self, plan: &Plan) -> Result<Option<Snapshot>> {
        self.sync()?;
        self.collect_snapshots()?;
        let Some(state) = self.save_ranges(plan.state.clone())? else {
            return Ok(None);
        };
        let Some(tail) = self.save_ranges(plan.tail.clone())? else {
            drop(state);
            self.collect_snapshots()?;
            return Ok(None);
        };
        let mut kv = plan.kv.clone();
        kv.extend(tail);
        let bytes = state.iter().chain(&kv).map(|p| p.range.bytes).sum();
        self.sync()?;
        Ok(Some(Snapshot { bytes, state, kv }))
    }
    fn save_ranges(&mut self, ranges: BTreeMap<String, Range>) -> Result<Option<Vec<Piece>>> {
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
        let allocation = Rc::new(Allocation { address, bytes });
        self.snapshot_allocations.push(Rc::clone(&allocation));
        let (mut pieces, mut source) = (vec![], 0);
        for (name, range) in ranges {
            if range.bytes != 0 {
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
    pub(crate) fn restore_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        self.sync()?;
        // Grow once to each final extent, then restore shared KV and endpoint state.
        let mut extents: BTreeMap<&str, usize> = BTreeMap::new();
        for p in snapshot.kv.iter().chain(&snapshot.state) {
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
            .snapshot_allocations
            .iter()
            .all(|a| Rc::strong_count(a) > 1)
        {
            return Ok(());
        }
        self.sync()?;
        let mut i = 0;
        while i < self.snapshot_allocations.len() {
            let a = &self.snapshot_allocations[i];
            if Rc::strong_count(a) != 1 {
                i += 1;
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
            self.snapshot_allocations.swap_remove(i);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn range_accounting_checks_overflow_and_alignment() {
        let mut r = BTreeMap::new();
        r.insert(
            "a".into(),
            Range {
                offset: 100,
                bytes: 1,
            },
        );
        r.insert(
            "b".into(),
            Range {
                offset: 0,
                bytes: 257,
            },
        );
        assert_eq!(size(&r).unwrap(), 768);
        r.insert(
            "c".into(),
            Range {
                offset: 0,
                bytes: usize::MAX,
            },
        );
        assert!(size(&r).is_err());
    }
}
