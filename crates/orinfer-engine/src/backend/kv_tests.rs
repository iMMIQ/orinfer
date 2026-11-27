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
        let owner = DeviceOwner::new(orinfer_model_sdk::device::Backend::Ascend, 0);
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
    fail_access: Cell<bool>,
    fail_unmap: Cell<bool>,
}
impl Fake {
    fn new() -> Self {
        Self {
            owner: DeviceOwner::new(orinfer_model_sdk::device::Backend::Cuda, 0),
            data: Default::default(),
            mapped: Default::default(),
            next: Cell::new(0),
            copies: Cell::new(0),
            creates: Cell::new(0),
            deferred: Cell::new(false),
            fail_access: Cell::new(false),
            fail_unmap: Cell::new(false),
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
    fn release(&self, h: u64, _bytes: usize) -> Result<()> {
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
    fn access(&self, _offset: usize, _bytes: usize, _shared: bool) -> Result<()> {
        if self.fail_access.get() {
            return Err("injected access fault".into());
        }
        Ok(())
    }
    fn unmap(&self, o: usize, _b: usize) -> Result<()> {
        if self.fail_unmap.get() {
            return Err("injected unmap fault".into());
        }
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
fn growth_fault_preserves_committed_extent_and_retains_failed_cleanup() {
    let ops = Fake::new();
    let mut reservation = Reservation::new(256, 64);
    reservation.grow(1, &ops, true).unwrap();
    let committed = reservation.mappings[0].handle;

    ops.fail_access.set(true);
    ops.fail_unmap.set(true);
    assert!(reservation.grow(65, &ops, true).is_err());
    assert_eq!(reservation.mapped, 64);
    assert!(reservation.poisoned);
    assert_eq!(reservation.mappings.len(), 2);
    assert!(ops.data.borrow().contains_key(&committed));
    assert!(reservation.grow(129, &ops, true).is_err());

    ops.fail_unmap.set(false);
    reservation
        .clear(
            |page| ops.unmap(page.offset, page.bytes),
            |page| ops.release(page.handle, page.bytes),
        )
        .unwrap();
    assert!(ops.data.borrow().is_empty());
    assert_eq!(reservation.mapped, 0);

    // With a successful rollback, failed growth leaves no physical pages.
    assert!(reservation.grow(1, &ops, true).is_err());
    assert!(reservation.mappings.is_empty());
    assert!(ops.data.borrow().is_empty());
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
        |page| ops.release(page.handle, page.bytes),
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
            owner: DeviceOwner::new(orinfer_model_sdk::device::Backend::Ascend, 0),
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
