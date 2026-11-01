//! Shared KV ownership and mapping transitions. Device adapters issue native calls.
pub(crate) mod host;
use crate::artifact::Result;
use orinfer_device::{DeviceOwner, MemoryRange};
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
    fn release(&self, handle: u64) -> Result<()>;
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
            if ops.release(handle).is_err() {
                replacement.offset = usize::MAX;
                mappings.push(replacement);
            }
            return Err(error);
        }
        retained.push(source.clone());
        std::mem::swap(&mut mappings[index], &mut replacement);
        let page = &mut mappings[index];
        ops.map(page.offset, page.bytes, page.handle, false)?;
        page.mapped = true;
        ops.access(page.offset, page.bytes, false)?;
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
    mut release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    let mut errors = vec![];
    mappings.retain_mut(|page| {
        if page.mapped {
            if let Err(e) = unmap(page) {
                errors.push(e);
                return true;
            }
            page.mapped = false;
        }
        if page.shared.is_none()
            && let Err(e) = release(page.handle)
        {
            errors.push(e);
            return true;
        }
        false
    });
    if errors.is_empty() {
        retained.clear();
        Ok(())
    } else {
        Err(errors.join("; "))
    }
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
mod tests {
    use super::*;
    use std::{cell::RefCell, collections::BTreeMap};
    #[test]
    fn tail_reuse_crops_promoted_prefix_and_bounds_fragment_chains() {
        let reuse = reuse_tail(
            MemoryRange {
                offset: 4,
                bytes: 16,
            },
            [
                MemoryRange {
                    offset: 8,
                    bytes: 8,
                },
                MemoryRange {
                    offset: 0,
                    bytes: 8,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            reuse
                .parts
                .iter()
                .map(|p| (p.index, p.source_offset, p.range))
                .collect::<Vec<_>>(),
            vec![
                (
                    1,
                    4,
                    MemoryRange {
                        offset: 4,
                        bytes: 4
                    }
                ),
                (
                    0,
                    0,
                    MemoryRange {
                        offset: 8,
                        bytes: 8
                    }
                ),
            ]
        );
        assert_eq!(
            reuse.missing,
            MemoryRange {
                offset: 16,
                bytes: 4
            }
        );
        let full = reuse_tail(
            MemoryRange {
                offset: 12,
                bytes: 4,
            },
            [MemoryRange {
                offset: 0,
                bytes: 16,
            }],
        )
        .unwrap();
        assert_eq!(full.parts[0].source_offset, 12);
        assert_eq!(full.missing.bytes, 0);
        let too_many = reuse_tail(
            MemoryRange {
                offset: 0,
                bytes: 9,
            },
            (0..8).map(|offset| MemoryRange { offset, bytes: 1 }),
        )
        .unwrap();
        assert!(too_many.parts.is_empty());
        assert_eq!(too_many.missing.bytes, 9);
        assert!(
            reuse_tail(
                MemoryRange {
                    offset: usize::MAX,
                    bytes: 1
                },
                []
            )
            .is_err()
        );
    }
    #[test]
    fn checkpoint_shares_full_allocations_and_packs_only_the_partial_suffix() {
        for grain in [64, 2 << 20] {
            let owner = DeviceOwner::new(orinfer_device::Backend::Ascend, 0);
            let allocation = |bytes| {
                Rc::new(Allocation {
                    owner,
                    id: 2,
                    address: 0,
                    bytes,
                    used: bytes,
                    handle: Some(9),
                    native: None,
                })
            };
            let regions = checkpoint_regions(vec![
                (
                    allocation(grain),
                    MemoryRange {
                        offset: 0,
                        bytes: grain,
                    },
                    0,
                ),
                (
                    allocation(grain),
                    MemoryRange {
                        offset: grain,
                        bytes: 17,
                    },
                    0,
                ),
            ])
            .unwrap();
            assert_eq!(regions.shared.len(), 1);
            assert_eq!(regions.tail.unwrap().offset, grain);
            assert_eq!(regions.tail.unwrap().bytes, 17);
            let full = checkpoint_regions(vec![(
                allocation(grain),
                MemoryRange {
                    offset: 0,
                    bytes: grain,
                },
                0,
            )])
            .unwrap();
            assert_eq!(full.shared.len(), 1);
            assert!(full.tail.is_none());
            let partial = checkpoint_regions(vec![(
                allocation(grain),
                MemoryRange {
                    offset: 0,
                    bytes: 17,
                },
                0,
            )])
            .unwrap();
            assert!(partial.shared.is_empty());
            assert_eq!(partial.tail.unwrap().bytes, 17);
        }
    }
    struct Fake {
        owner: DeviceOwner,
        data: RefCell<BTreeMap<u64, Vec<u8>>>,
        mapped: RefCell<BTreeMap<usize, u64>>,
        next: Cell<u64>,
        copies: Cell<usize>,
        creates: Cell<usize>,
        deferred: Cell<bool>,
    }
    impl Fake {
        fn new() -> Self {
            Self {
                owner: DeviceOwner::new(orinfer_device::Backend::Cuda, 0),
                data: Default::default(),
                mapped: Default::default(),
                next: Cell::new(0),
                copies: Cell::new(0),
                creates: Cell::new(0),
                deferred: Cell::new(false),
            }
        }
    }
    impl MappingOps for Fake {
        fn allocate(&self, bytes: usize) -> Result<u64> {
            let id = self.next.get() + 1;
            self.next.set(id);
            self.creates.set(self.creates.get() + 1);
            self.data.borrow_mut().insert(id, vec![0; bytes]);
            Ok(id)
        }
        fn release(&self, h: u64) -> Result<()> {
            assert!(!self.mapped.borrow().values().any(|v| *v == h));
            self.data
                .borrow_mut()
                .remove(&h)
                .ok_or("Double physical release")?;
            Ok(())
        }
        fn map(&self, o: usize, _b: usize, h: u64, _s: bool) -> Result<()> {
            assert!(self.mapped.borrow_mut().insert(o, h).is_none());
            Ok(())
        }
        fn unmap(&self, o: usize, _b: usize) -> Result<()> {
            self.mapped.borrow_mut().remove(&o).ok_or("Double unmap")?;
            Ok(())
        }
        fn zero(&self, o: usize, _b: usize) -> Result<()> {
            self.data
                .borrow_mut()
                .get_mut(&self.mapped.borrow()[&o])
                .unwrap()
                .fill(0);
            Ok(())
        }
        fn copy(&self, o: usize, a: &Allocation, n: usize) -> Result<()> {
            self.copies.set(self.copies.get() + 1);
            let bytes = self.data.borrow()[&a.handle.unwrap()][..n].to_vec();
            self.data
                .borrow_mut()
                .get_mut(&self.mapped.borrow()[&o])
                .unwrap()[..n]
                .copy_from_slice(&bytes);
            Ok(())
        }
        fn complete(&self) -> Result<bool> {
            Ok(!self.deferred.get())
        }
        fn publish(&self, p: &Mapping, used: usize) -> Result<Rc<Allocation>> {
            Ok(Rc::new(Allocation {
                owner: self.owner,
                id: p.id,
                address: p.handle,
                bytes: p.bytes,
                used,
                handle: Some(p.handle),
                native: None,
            }))
        }
    }
    #[test]
    fn publication_shares_physical_memory_and_cow_preserves_only_committed_prefix() {
        let ops = Fake::new();
        let h = ops.allocate(256).unwrap();
        ops.data.borrow_mut().get_mut(&h).unwrap().fill(73);
        let mut p = Mapping::owned(0, 256, h).unwrap();
        ops.map(0, 256, h, false).unwrap();
        p.mapped = true;
        let mut maps = vec![p];
        let mut retained = vec![];
        let plan = preview(
            &maps,
            ops.owner,
            0,
            MemoryRange {
                offset: 0,
                bytes: 17,
            },
        )
        .unwrap();
        let saved = publish(
            &mut maps,
            MemoryRange {
                offset: 0,
                bytes: 17,
            },
            &ops,
        )
        .unwrap();
        assert_eq!(plan[0].0.id, saved[0].0.id);
        assert_eq!(ops.creates.get(), 1);
        assert_eq!(ops.copies.get(), 0);
        let again = publish(
            &mut maps,
            MemoryRange {
                offset: 0,
                bytes: 17,
            },
            &ops,
        )
        .unwrap();
        assert!(Rc::ptr_eq(&saved[0].0, &again[0].0));
        assert!(
            publish(
                &mut maps,
                MemoryRange {
                    offset: 0,
                    bytes: 18
                },
                &ops
            )
            .is_err()
        );
        ops.deferred.set(true);
        make_writable(
            &mut maps,
            &mut retained,
            MemoryRange {
                offset: 17,
                bytes: 1,
            },
            &ops,
        )
        .unwrap();
        assert_eq!(
            retained.len(),
            1,
            "in-flight copy retains immutable backing"
        );
        let new = maps[0].handle;
        assert_ne!(new, h);
        assert_ne!(maps[0].id, saved[0].0.id);
        assert_eq!(&ops.data.borrow()[&new][..17], &[73; 17]);
        assert!(ops.data.borrow()[&new][17..].iter().all(|b| *b == 0));
        ops.data.borrow_mut().get_mut(&new).unwrap()[17] = 9;
        assert_eq!(
            ops.data.borrow()[&h][17],
            73,
            "branch does not modify shared cache"
        );
        retained.clear();
        release_mappings(
            &mut maps,
            &mut retained,
            |p| ops.unmap(p.offset, p.bytes),
            |h| ops.release(h),
        )
        .unwrap();
        assert!(
            ops.data.borrow().contains_key(&h),
            "saved immutable allocation remains owned by backend"
        );
    }
    #[test]
    fn logical_alias_and_budget_rules_are_independent_of_native_granularity() {
        for grain in [64, 256, 2 << 20] {
            let a = Allocation {
                owner: DeviceOwner::new(orinfer_device::Backend::Ascend, 0),
                id: 2,
                address: 3,
                bytes: grain,
                used: 17,
                handle: Some(9),
                native: None,
            };
            assert_eq!(
                alias_offset(
                    &a,
                    MemoryRange {
                        offset: grain * 2,
                        bytes: 17
                    },
                    0,
                    grain,
                    grain * 3
                ),
                Some(grain * 2)
            );
            assert_eq!(
                alias_offset(
                    &a,
                    MemoryRange {
                        offset: grain * 2,
                        bytes: 18
                    },
                    0,
                    grain,
                    grain * 3
                ),
                None
            );
            assert_eq!(
                alias_offset(
                    &a,
                    MemoryRange {
                        offset: grain * 2,
                        bytes: 17
                    },
                    0,
                    grain,
                    grain * 2
                ),
                None
            );
        }
    }
}
