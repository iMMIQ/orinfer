//! Shared KV ownership and mapping transitions. Device adapters issue native calls.
pub(crate) mod host;
use crate::artifact::Result;
use orinfer_model_sdk::device::{DeviceOwner, MemoryRange};
use std::{any::Any, cell::Cell, rc::Rc};

thread_local! { static NEXT: Cell<u64> = const { Cell::new(0) }; }
/// Even keys identify physical KV allocations; odd keys are private checkpoints.
/// Each TP worker runs the same ordered allocation commands in its own session.
pub(crate) fn allocation_id() -> Result<u64> {
    NEXT.with(|next| {
        let id = next
            .get()
            .checked_add(1)
            .filter(|n| *n <= u64::MAX / 8)
            .ok_or("KV allocation identity overflow")?;
        next.set(id);
        Ok(id * 2)
    })
}
pub(crate) fn checkpoint_id(id: u64) -> Result<u64> {
    id.checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .filter(|n| *n <= u64::MAX / 2)
        .ok_or_else(|| "KV checkpoint identity overflow".into())
}
/// Immutable allocation identity survives virtual aliases. `used` excludes the
/// unused physical suffix, so CPU backing and COW need only preserve valid bytes.
pub(crate) struct Allocation {
    pub owner: DeviceOwner,
    pub id: u64,
    pub address: u64,
    pub bytes: usize,
    pub used: usize,
    pub handle: Option<u64>,
    pub native: Option<Rc<dyn Any>>,
}
impl std::fmt::Debug for Allocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvAllocation")
            .field("id", &self.id)
            .field("bytes", &self.bytes)
            .field("used", &self.used)
            .finish()
    }
}
#[derive(Debug)]
pub(crate) struct Mapping {
    pub id: u64,
    pub offset: usize,
    pub bytes: usize,
    pub handle: u64,
    pub mapped: bool,
    pub shared: Option<Rc<Allocation>>,
}
impl Mapping {
    #[cfg(test)]
    pub fn owned(offset: usize, bytes: usize, handle: u64) -> Result<Self> {
        Ok(Self {
            id: allocation_id()?,
            offset,
            bytes,
            handle,
            mapped: false,
            shared: None,
        })
    }
}
pub(crate) trait MappingOps {
    fn allocate(&self, bytes: usize) -> Result<u64>;
    fn release(&self, handle: u64, bytes: usize) -> Result<()>;
    fn map(&self, offset: usize, bytes: usize, handle: u64, shared: bool) -> Result<()>;
    fn access(&self, _offset: usize, _bytes: usize, _shared: bool) -> Result<()> {
        Ok(())
    }
    fn unmap(&self, offset: usize, bytes: usize) -> Result<()>;
    fn zero(&self, offset: usize, bytes: usize) -> Result<()>;
    fn copy(&self, offset: usize, allocation: &Allocation, bytes: usize) -> Result<()>;
    /// False leaves source leases retained until the enclosing batch fence.
    fn complete(&self) -> Result<bool>;
    /// Adopt an owned physical handle into stable immutable backing, without copying.
    fn publish(&self, mapping: &Mapping, used: usize) -> Result<Rc<Allocation>>;
}
pub(crate) fn align(bytes: usize, granularity: usize) -> Result<usize> {
    if granularity == 0 {
        return Err("Empty VMM allocation granularity".into());
    }
    bytes
        .checked_next_multiple_of(granularity)
        .ok_or("VMM alignment overflow".into())
}

/// Device-independent virtual range and physical ownership. Device adapters
/// provide native operations and decide when their streams must be fenced.
pub(crate) struct Reservation {
    pub bytes: usize,
    pub granularity: usize,
    pub mapped: usize,
    pub mappings: Vec<Mapping>,
    pub retained: Vec<Rc<Allocation>>,
    pub poisoned: bool,
}

impl Reservation {
    pub fn new(bytes: usize, granularity: usize) -> Self {
        Self {
            bytes,
            granularity,
            mapped: 0,
            mappings: vec![],
            retained: vec![],
            poisoned: false,
        }
    }

    pub fn extent(&self, bytes: usize) -> Result<usize> {
        let extent = align(bytes, self.granularity)?;
        if extent > self.bytes {
            return Err("KV growth exceeds reservation".into());
        }
        Ok(extent)
    }

    /// CUDA initializes immediately. Ascend returns the span for initialization
    /// after all mappings in the stream batch have been installed.
    pub fn grow(
        &mut self,
        bytes: usize,
        ops: &impl MappingOps,
        initialize: bool,
    ) -> Result<Option<MemoryRange>> {
        self.require_healthy()?;
        let extent = self.extent(bytes)?;
        if extent <= self.mapped {
            return Ok(None);
        }
        let range = MemoryRange {
            offset: self.mapped,
            bytes: extent - self.mapped,
        };
        self.map_owned(range, ops, initialize)?;
        self.mapped = extent;
        Ok(Some(range))
    }

    pub fn map_sparse(&mut self, offset: usize, ops: &impl MappingOps) -> Result<()> {
        self.require_healthy()?;
        let bytes = self.granularity;
        self.check_mapping(offset, bytes)?;
        if self.mappings.iter().any(|page| page.offset == offset) {
            return Err("Invalid sparse weight page transition".into());
        }
        self.map_owned(MemoryRange { offset, bytes }, ops, false)
    }

    fn map_owned(
        &mut self,
        range: MemoryRange,
        ops: &impl MappingOps,
        initialize: bool,
    ) -> Result<()> {
        let id = allocation_id()?;
        let handle = ops.allocate(range.bytes)?;
        let mut page = Mapping {
            id,
            offset: range.offset,
            bytes: range.bytes,
            handle,
            mapped: false,
            shared: None,
        };
        let result = map(&mut page, ops).and_then(|()| {
            if initialize {
                ops.zero(page.offset, page.bytes)?;
            }
            Ok(())
        });
        if result.is_err() {
            self.poisoned = true;
            if initialize
                && release_mapping(&mut page, &mut |p| ops.unmap(p.offset, p.bytes), &mut |p| {
                    ops.release(p.handle, p.bytes)
                })
                .is_ok()
            {
                return result;
            }
        }
        // Retain ownership when rollback could not detach the mapping or the
        // enclosing Ascend batch is quarantined.
        self.mappings.push(page);
        result
    }

    pub fn map_shared(
        &mut self,
        offset: usize,
        allocation: Rc<Allocation>,
        owner: DeviceOwner,
        ops: &impl MappingOps,
    ) -> Result<()> {
        self.require_healthy()?;
        owner.require(allocation.owner)?;
        let bytes = allocation.bytes;
        self.check_mapping(offset, bytes)?;
        if offset != self.mapped {
            return Err("Noncontiguous shared KV mappings".into());
        }
        let handle = allocation
            .handle
            .ok_or("Prefix allocation is not VMM backed")?;
        self.mappings.push(Mapping {
            id: allocation.id,
            offset,
            bytes,
            handle,
            mapped: false,
            shared: Some(allocation),
        });
        let result = map(self.mappings.last_mut().unwrap(), ops);
        self.poisoned |= result.is_err();
        result?;
        self.mapped += bytes;
        Ok(())
    }

    pub fn clear(
        &mut self,
        unmap: impl FnMut(&Mapping) -> Result<()>,
        release: impl FnMut(&Mapping) -> Result<()>,
    ) -> Result<()> {
        let result = release_mappings(&mut self.mappings, &mut self.retained, unmap, release);
        self.poisoned = result.is_err();
        if result.is_ok() {
            self.mapped = 0;
        }
        result
    }

    fn check_mapping(&self, offset: usize, bytes: usize) -> Result<()> {
        if bytes == 0
            || !offset.is_multiple_of(self.granularity)
            || !bytes.is_multiple_of(self.granularity)
        {
            return Err("Invalid VMM mapping alignment".into());
        }
        MemoryRange { offset, bytes }.within(self.bytes)?;
        Ok(())
    }

    fn require_healthy(&self) -> Result<()> {
        if self.poisoned {
            return Err("KV reservation is quarantined after a failed transition".into());
        }
        Ok(())
    }
}

fn map(page: &mut Mapping, ops: &impl MappingOps) -> Result<()> {
    let shared = page.shared.is_some();
    ops.map(page.offset, page.bytes, page.handle, shared)?;
    page.mapped = true;
    ops.access(page.offset, page.bytes, shared)
}

pub(crate) fn overlaps(offset: usize, bytes: usize, start: usize, end: usize) -> bool {
    start < end && start < offset.saturating_add(bytes) && offset < end
}
pub(crate) fn writable_bytes(mappings: &[Mapping], range: MemoryRange) -> Result<usize> {
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or("KV write extent overflow")?;
    mappings
        .iter()
        .filter(|p| p.shared.is_some() && overlaps(p.offset, p.bytes, range.offset, end))
        .try_fold(0usize, |n, p| {
            n.checked_add(p.bytes)
                .ok_or_else(|| "KV COW budget overflow".into())
        })
}
/// Caller fences preceding readers before changing mappings. Failed transitions
/// retain both old and new allocations for the owning session's cleanup policy.
pub(crate) fn make_writable(
    mappings: &mut Vec<Mapping>,
    retained: &mut Vec<Rc<Allocation>>,
    range: MemoryRange,
    ops: &impl MappingOps,
) -> Result<()> {
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or("KV write extent overflow")?;
    let mut changed = false;
    let mut index = 0;
    while index < mappings.len() {
        if !overlaps(
            mappings[index].offset,
            mappings[index].bytes,
            range.offset,
            end,
        ) || mappings[index].shared.is_none()
        {
            index += 1;
            continue;
        }
        let source = Rc::clone(mappings[index].shared.as_ref().unwrap());
        let id = allocation_id()?;
        let handle = ops.allocate(mappings[index].bytes)?;
        let mut replacement = Mapping {
            id,
            offset: mappings[index].offset,
            bytes: mappings[index].bytes,
            handle,
            mapped: false,
            shared: None,
        };
        if let Err(error) = ops.unmap(mappings[index].offset, mappings[index].bytes) {
            if ops.release(handle, replacement.bytes).is_err() {
                replacement.offset = usize::MAX;
                mappings.push(replacement);
            }
            return Err(error);
        }
        retained.push(source.clone());
        std::mem::swap(&mut mappings[index], &mut replacement);
        let page = &mut mappings[index];
        map(page, ops)?;
        ops.zero(page.offset, page.bytes)?;
        ops.copy(page.offset, &source, source.used)?;
        changed = true;
        index += 1;
    }
    if changed && ops.complete()? {
        retained.clear();
    }
    Ok(())
}
/// Publish stable physical allocations, preserving exact valid prefix bounds.
/// The returned slices are (allocation, logical range, byte offset in allocation).
pub(crate) fn publish(
    mappings: &mut [Mapping],
    range: MemoryRange,
    ops: &impl MappingOps,
) -> Result<Vec<(Rc<Allocation>, MemoryRange, usize)>> {
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or("KV publication overflow")?;
    let mut out = vec![];
    let mut covered = range.offset;
    for page in mappings {
        if !page.mapped || !overlaps(page.offset, page.bytes, range.offset, end) {
            continue;
        }
        let start = page.offset.max(range.offset);
        let stop = (page.offset + page.bytes).min(end);
        if start != covered {
            return Err("KV publication has a mapping gap".into());
        }
        let used = stop - page.offset;
        let allocation = if let Some(allocation) = &page.shared {
            if used > allocation.used {
                return Err("Immutable KV grew without COW".into());
            }
            Rc::clone(allocation)
        } else {
            let allocation = ops.publish(page, used)?;
            if allocation.id != page.id || allocation.bytes != page.bytes || allocation.used != used
            {
                return Err("Native KV publication differs from its plan".into());
            }
            page.shared = Some(Rc::clone(&allocation));
            allocation
        };
        out.push((
            allocation,
            MemoryRange {
                offset: start,
                bytes: stop - start,
            },
            start - page.offset,
        ));
        covered = stop;
    }
    if covered != end {
        return Err("KV publication exceeds resident memory".into());
    }
    Ok(out)
}
pub(crate) fn release_mappings(
    mappings: &mut Vec<Mapping>,
    retained: &mut Vec<Rc<Allocation>>,
    mut unmap: impl FnMut(&Mapping) -> Result<()>,
    mut release: impl FnMut(&Mapping) -> Result<()>,
) -> Result<()> {
    let mut errors = vec![];
    mappings.retain_mut(|page| {
        if let Err(error) = release_mapping(page, &mut unmap, &mut release) {
            errors.push(error);
            true
        } else {
            false
        }
    });
    if errors.is_empty() {
        retained.clear();
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}
fn release_mapping(
    page: &mut Mapping,
    unmap: &mut impl FnMut(&Mapping) -> Result<()>,
    release: &mut impl FnMut(&Mapping) -> Result<()>,
) -> Result<()> {
    if page.mapped {
        unmap(page)?;
        page.mapped = false;
    }
    if page.shared.is_none() {
        release(page)?;
    }
    Ok(())
}

/// A valid slice can alias its whole physical allocation, including a private
/// zero/uncommitted suffix. Subsequent writes use the same COW transition.
pub(crate) fn alias_offset(
    allocation: &Allocation,
    range: MemoryRange,
    source: usize,
    grain: usize,
    capacity: usize,
) -> Option<usize> {
    let offset = range.offset.checked_sub(source)?;
    (grain > 0
        && allocation.handle.is_some()
        && offset.is_multiple_of(grain)
        && allocation.bytes.is_multiple_of(grain)
        && source.checked_add(range.bytes)? <= allocation.used
        && offset.checked_add(allocation.bytes)? <= capacity)
        .then_some(offset)
}

pub(crate) type Published = (Rc<Allocation>, MemoryRange, usize);
pub(crate) fn preview(
    mappings: &[Mapping],
    owner: DeviceOwner,
    address: u64,
    range: MemoryRange,
) -> Result<Vec<Published>> {
    let end = range
        .offset
        .checked_add(range.bytes)
        .ok_or("KV preview overflow")?;
    let mut out = vec![];
    let mut covered = range.offset;
    for page in mappings {
        if !page.mapped || !overlaps(page.offset, page.bytes, range.offset, end) {
            continue;
        }
        let start = page.offset.max(range.offset);
        let stop = (page.offset + page.bytes).min(end);
        if start != covered {
            return Err("KV preview has a mapping gap".into());
        }
        let allocation = page.shared.clone().unwrap_or_else(|| {
            Rc::new(Allocation {
                owner,
                id: page.id,
                address: address + page.offset as u64,
                bytes: page.bytes,
                used: stop - page.offset,
                handle: Some(page.handle),
                native: None,
            })
        });
        out.push((
            allocation,
            MemoryRange {
                offset: start,
                bytes: stop - start,
            },
            start - page.offset,
        ));
        covered = stop;
    }
    if covered != end {
        return Err("KV preview exceeds resident memory".into());
    }
    Ok(out)
}

/// Full native allocations share backing. A partial suffix stays compact in
/// the endpoint bank, so publishing a checkpoint does not freeze a writable
/// tail or retain a mostly unused native allocation for every endpoint.
pub(crate) struct CheckpointRegions {
    pub shared: Vec<Published>,
    pub tail: Option<MemoryRange>,
}
pub(crate) fn checkpoint_regions(preview: Vec<Published>) -> Result<CheckpointRegions> {
    let mut shared = vec![];
    let mut tail: Option<MemoryRange> = None;
    for (allocation, range, source) in preview {
        MemoryRange {
            offset: source,
            bytes: range.bytes,
        }
        .within(allocation.used)?;
        if tail.is_none() && source == 0 && range.bytes == allocation.bytes {
            shared.push((allocation, range, source));
        } else if let Some(tail) = &mut tail {
            if tail.offset.checked_add(tail.bytes) != Some(range.offset) {
                return Err("Noncontiguous checkpoint suffix".into());
            }
            tail.bytes = tail
                .bytes
                .checked_add(range.bytes)
                .ok_or("Checkpoint suffix overflow")?;
        } else {
            tail = Some(range);
        }
    }
    Ok(CheckpointRegions { shared, tail })
}

/// Bound scatter copies and old-bank retention. Eight pieces cover a 2MiB
/// key allocation at this A2 model's 4096-token checkpoints; sparser scale
/// histories periodically repack rather than pinning an unbounded chain.
const MAX_TAIL_PARTS: usize = 8;
pub(crate) struct TailPart {
    pub index: usize,
    pub range: MemoryRange,
    pub source_offset: usize,
}
pub(crate) struct TailReuse {
    pub parts: Vec<TailPart>,
    pub missing: MemoryRange,
}
/// Reuse only a contiguous immutable prefix of the requested private suffix.
/// Indices refer to caller metadata; device and CPU adapters supply bounded views.
pub(crate) fn reuse_tail(
    tail: MemoryRange,
    parent: impl IntoIterator<Item = MemoryRange>,
) -> Result<TailReuse> {
    let end = tail
        .offset
        .checked_add(tail.bytes)
        .ok_or("KV suffix overflow")?;
    let mut ranges = parent
        .into_iter()
        .enumerate()
        .map(|(index, range)| {
            let stop = range
                .offset
                .checked_add(range.bytes)
                .ok_or("Parent KV suffix overflow")?;
            Ok((index, range, stop))
        })
        .collect::<Result<Vec<_>>>()?;
    ranges.sort_by_key(|(_, range, _)| range.offset);
    let mut cursor = tail.offset;
    let mut parts = vec![];
    for (index, range, stop) in ranges {
        if cursor == end {
            break;
        }
        if stop <= cursor {
            continue;
        }
        if range.offset > cursor {
            break;
        }
        let next = stop.min(end);
        parts.push(TailPart {
            index,
            range: MemoryRange {
                offset: cursor,
                bytes: next - cursor,
            },
            source_offset: cursor - range.offset,
        });
        cursor = next;
    }
    let missing = MemoryRange {
        offset: cursor,
        bytes: end - cursor,
    };
    if parts.len() + usize::from(missing.bytes != 0) > MAX_TAIL_PARTS {
        return Ok(TailReuse {
            parts: vec![],
            missing: tail,
        });
    }
    Ok(TailReuse { parts, missing })
}

#[cfg(test)]
#[path = "kv_tests.rs"]
mod tests;
