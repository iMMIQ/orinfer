//! One bounded CPU-bank and device-replica directory for CUDA and Ascend.
use super::*;
use std::{cell::RefCell, collections::BTreeMap, rc::Weak};
#[derive(Clone, Copy, Debug)]
pub(crate) struct Descriptor {
    pub owner: DeviceOwner,
    pub id: u64,
    pub bytes: usize,
    pub used: usize,
    pub paged: bool,
}
impl From<&Allocation> for Descriptor {
    fn from(a: &Allocation) -> Self {
        Self {
            owner: a.owner,
            id: a.id,
            bytes: a.bytes,
            used: a.used,
            paged: a.handle.is_some(),
        }
    }
}
pub(crate) struct HostBank {
    pub bytes: RefCell<Vec<u8>>,
    // Disjoint, nonadjacent [start, end) intervals, published after DMA completes.
    valid: RefCell<BTreeMap<usize, usize>>,
    descriptor: RefCell<Option<Descriptor>>,
    device: RefCell<Weak<Allocation>>,
}
impl HostBank {
    /// A completed immutable source can be reused while an active request or
    /// device cache keeps it resident. The CPU tier does not pin device memory.
    pub fn remember(&self, allocation: &Rc<Allocation>) -> Result<()> {
        if self.bytes.borrow().len() != allocation.used {
            return Err("Host/device KV extents differ".into());
        }
        let descriptor = Descriptor::from(allocation.as_ref());
        if let Some(previous) = *self.descriptor.borrow() {
            previous.owner.require(descriptor.owner)?;
            if previous.id != descriptor.id
                || previous.used != descriptor.used
                || previous.bytes != descriptor.bytes
                || previous.paged != descriptor.paged
            {
                return Err("KV replica identity changed".into());
            }
        }
        *self.descriptor.borrow_mut() = Some(descriptor);
        *self.device.borrow_mut() = Rc::downgrade(allocation);
        Ok(())
    }
    pub fn descriptor(&self) -> Option<Descriptor> {
        *self.descriptor.borrow()
    }
    pub fn resident(&self, owner: DeviceOwner) -> Result<Option<Rc<Allocation>>> {
        if let Some(descriptor) = *self.descriptor.borrow() {
            descriptor.owner.require(owner)?;
        }
        Ok(self.device.borrow().upgrade())
    }
    /// The common directory owns lookup and publication; adapters return only
    /// completed native allocations. Failed transfers never become cache hits.
    pub fn materialize(
        &self,
        owner: DeviceOwner,
        upload: impl FnOnce(Descriptor, &[u8]) -> Result<Rc<Allocation>>,
    ) -> Result<Rc<Allocation>> {
        let descriptor = self.descriptor().ok_or("Missing host KV descriptor")?;
        descriptor.owner.require(owner)?;
        if !descriptor.paged {
            return Err("Host state is not a shared KV allocation".into());
        }
        if let Some(allocation) = self.resident(owner)? {
            return Ok(allocation);
        }
        let allocation = upload(descriptor, &self.bytes.borrow())?;
        self.remember(&allocation)?;
        Ok(allocation)
    }
    pub fn download<T: Clone>(
        &self,
        regions: Vec<(T, usize, usize, usize)>,
        transfer: impl FnOnce(&[(T, usize, usize, usize)], &mut [u8]) -> Result<()>,
    ) -> Result<()> {
        let mut valid = self.valid.borrow().clone();
        let mut missing = vec![];
        for (source, source_offset, offset, bytes) in regions {
            MemoryRange { offset, bytes }.within(self.bytes.borrow().len())?;
            source_offset
                .checked_add(bytes)
                .ok_or("Host KV source extent overflow")?;
            let end = offset + bytes;
            let mut cursor = offset;
            let mut append = |start: usize, stop: usize| {
                missing.push((
                    source.clone(),
                    source_offset + (start - offset),
                    start,
                    stop - start,
                ));
            };
            let first = valid
                .range(..=offset)
                .next_back()
                .map_or(offset, |(&start, _)| start);
            for (&start, &stop) in valid.range(first..end) {
                if stop <= cursor {
                    continue;
                }
                if start > cursor {
                    append(cursor, start);
                }
                cursor = stop.min(end);
            }
            if cursor < end {
                append(cursor, end);
            }
            if bytes != 0 {
                cover(&mut valid, offset, end);
            }
        }
        if !missing.is_empty() {
            transfer(&missing, &mut self.bytes.borrow_mut())?;
            *self.valid.borrow_mut() = valid;
        }
        Ok(())
    }
}
fn cover(valid: &mut BTreeMap<usize, usize>, mut start: usize, mut end: usize) {
    if let Some((&previous, &stop)) = valid.range(..=start).next_back()
        && stop >= start
    {
        start = previous;
    }
    while let Some((&next, &stop)) = valid.range(start..).next() {
        if next > end {
            break;
        }
        valid.remove(&next);
        end = end.max(stop);
    }
    valid.insert(start, end);
}
#[derive(Clone)]
pub(crate) struct HostBuffer {
    pub bank: Rc<HostBank>,
    pub offset: usize,
    pub bytes: usize,
}
impl HostBuffer {
    pub fn view(&self, range: MemoryRange) -> Result<Self> {
        range.within(self.bytes)?;
        Ok(Self {
            bank: self.bank.clone(),
            offset: self
                .offset
                .checked_add(range.offset)
                .ok_or("Host KV view overflow")?,
            bytes: range.bytes,
        })
    }
}
#[derive(Default)]
pub(crate) struct HostPool(BTreeMap<u64, Weak<HostBank>>);
impl HostPool {
    pub fn bank(&mut self, id: u64, bytes: usize) -> Result<Rc<HostBank>> {
        self.0.retain(|_, bank| bank.strong_count() != 0);
        if let Some(bank) = self.0.get(&id).and_then(Weak::upgrade) {
            if bank.bytes.borrow().len() != bytes {
                return Err("Host KV bank extent changed".into());
            }
            return Ok(bank);
        }
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(bytes)
            .map_err(|e| e.to_string())?;
        storage.resize(bytes, 0);
        let bank = Rc::new(HostBank {
            bytes: RefCell::new(storage),
            valid: Default::default(),
            descriptor: Default::default(),
            device: Default::default(),
        });
        self.0.insert(id, Rc::downgrade(&bank));
        Ok(bank)
    }
}
#[derive(Clone)]
pub(crate) struct HostPiece {
    pub name: String,
    pub range: MemoryRange,
    pub data: HostBuffer,
}
pub(crate) struct HostSnapshot {
    pub owner: DeviceOwner,
    pub state: Vec<HostPiece>,
    pub kv: Vec<HostPiece>,
    pub bytes: usize,
    pub metadata_id: u64,
    pub metadata_bytes: usize,
}
impl crate::prefix::Resident for HostSnapshot {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        out.insert(self.metadata_id, self.metadata_bytes);
        for p in self.state.iter().chain(&self.kv) {
            let id = p
                .data
                .bank
                .descriptor()
                .expect("Saved host bank has an identity")
                .id;
            out.insert(id, p.data.bank.bytes.borrow().len());
        }
    }
}

pub(crate) type ReadRegion = (Rc<Allocation>, usize, usize, usize);
impl HostPool {
    /// Both devices use identical deduplication, valid-range publication and
    /// weak device residency. The adapter only performs the requested copies.
    pub fn capture(
        &mut self,
        snapshot: &crate::backend::snapshot::Snapshot,
        mut read: impl FnMut(&[ReadRegion], &mut [u8]) -> Result<()>,
    ) -> Result<HostSnapshot> {
        let mut allocations = BTreeMap::new();
        for p in snapshot.state.iter().chain(&snapshot.kv) {
            allocations.insert(p.allocation.id, p.allocation.clone());
        }
        let mut banks = BTreeMap::new();
        for (id, allocation) in allocations {
            let bank = self.bank(id, allocation.used)?;
            let regions = snapshot
                .state
                .iter()
                .chain(&snapshot.kv)
                .filter(|p| p.allocation.id == id)
                .map(|p| (p.allocation.clone(), p.source, p.source, p.range.bytes))
                .collect();
            bank.download(regions, |regions, output| read(regions, output))?;
            bank.remember(&allocation)?;
            banks.insert(id, bank);
        }
        let convert = |pieces: &[crate::backend::snapshot::Piece]| {
            pieces
                .iter()
                .map(|p| HostPiece {
                    name: p.name.clone(),
                    range: p.range,
                    data: HostBuffer {
                        bank: banks[&p.allocation.id].clone(),
                        offset: p.source,
                        bytes: p.range.bytes,
                    },
                })
                .collect()
        };
        Ok(HostSnapshot {
            owner: snapshot.owner,
            state: convert(&snapshot.state),
            kv: convert(&snapshot.kv),
            bytes: snapshot.bytes,
            metadata_id: allocation_id()?,
            metadata_bytes: 0,
        })
    }
    /// Private recurrent state is packed directly to CPU, without an interim
    /// device checkpoint. Sealed KV pages keep their existing immutable IDs.
    pub fn capture_components(
        &mut self,
        owner: DeviceOwner,
        kv: &[crate::backend::snapshot::Piece],
        state: &[crate::backend::snapshot::Piece],
        mut read: impl FnMut(&[ReadRegion], &mut [u8]) -> Result<()>,
    ) -> Result<HostSnapshot> {
        let mut snapshot = self.capture(
            &crate::backend::snapshot::Snapshot {
                owner,
                bytes: kv.iter().map(|p| p.range.bytes).sum(),
                state: vec![],
                kv: kv.to_vec(),
            },
            &mut read,
        )?;
        let mut bytes = 0usize;
        let mut regions = vec![];
        let mut slices = vec![];
        for p in state {
            bytes = bytes
                .checked_add(255)
                .ok_or("Host state alignment overflow")?
                / 256
                * 256;
            regions.push((p.allocation.clone(), p.source, bytes, p.range.bytes));
            slices.push((p.name.clone(), p.range, bytes));
            bytes = bytes
                .checked_add(p.range.bytes)
                .ok_or("Host state size overflow")?;
        }
        if bytes != 0 {
            let id = checkpoint_id(allocation_id()?)?;
            let bank = self.bank(id, bytes)?;
            bank.download(regions, |regions, output| read(regions, output))?;
            let descriptor = Rc::new(Allocation {
                owner,
                id,
                address: 0,
                bytes,
                used: bytes,
                handle: None,
                native: None,
            });
            bank.remember(&descriptor)?;
            snapshot.bytes += state.iter().map(|p| p.range.bytes).sum::<usize>();
            snapshot.state = slices
                .into_iter()
                .map(|(name, range, offset)| HostPiece {
                    name,
                    range,
                    data: HostBuffer {
                        bank: bank.clone(),
                        offset,
                        bytes: range.bytes,
                    },
                })
                .collect();
        }
        Ok(snapshot)
    }
}
impl HostSnapshot {
    pub fn restore_shared(
        &self,
        owner: DeviceOwner,
        mut materialize: impl FnMut(&HostBank) -> Result<Rc<Allocation>>,
    ) -> Result<crate::backend::snapshot::Snapshot> {
        self.owner.require(owner)?;
        let mut kv = vec![];
        for piece in &self.kv {
            if piece.data.bank.descriptor().is_some_and(|d| d.paged) {
                kv.push(crate::backend::snapshot::Piece {
                    allocation: materialize(&piece.data.bank)?,
                    name: piece.name.clone(),
                    range: piece.range,
                    source: piece.data.offset,
                });
            }
        }
        Ok(crate::backend::snapshot::Snapshot {
            owner,
            bytes: kv.iter().map(|p| p.range.bytes).sum(),
            state: vec![],
            kv,
        })
    }
}

/// CPU admission reserves new private state and metadata before starting DMA.
pub(crate) struct Budget(pub BTreeMap<u64, usize>);
impl crate::prefix::Resident for Budget {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        out.extend(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::snapshot::{Piece, Snapshot};
    use crate::prefix::Resident;

    #[test]
    fn overlapping_downloads_copy_only_missing_bytes_and_retry_failed_dma() {
        let mut pool = HostPool::default();
        let bank = pool.bank(2, 32).unwrap();
        let mut copied = vec![];
        let mut read = |regions: &[(u8, usize, usize, usize)], output: &mut [u8]| {
            for &(base, source, dest, bytes) in regions {
                copied.push((source, dest, bytes));
                for i in 0..bytes {
                    output[dest + i] = base + (source + i) as u8;
                }
            }
            Ok(())
        };
        bank.download(vec![(0, 12, 4, 4), (0, 20, 12, 4)], &mut read)
            .unwrap();
        // Different views, including overlap within the same DMA batch, must
        // retain the source offset after splitting at already-valid intervals.
        bank.download(vec![(0, 8, 0, 20), (0, 24, 16, 8)], &mut read)
            .unwrap();
        bank.download(vec![(0, 10, 2, 20)], &mut read).unwrap();
        assert_eq!(
            copied,
            [
                (12, 4, 4),
                (20, 12, 4),
                (8, 0, 4),
                (16, 8, 4),
                (24, 16, 4),
                (28, 20, 4)
            ]
        );
        assert_eq!(&bank.bytes.borrow()[..24], &(8..32).collect::<Vec<u8>>());
        assert_eq!(*bank.valid.borrow(), BTreeMap::from([(0, 24)]));
        assert!(
            bank.download(vec![(0u8, 32, 24, 8)], |_, output| {
                output[24] = 99;
                Err("failed DMA fence".into())
            })
            .is_err()
        );
        assert_eq!(*bank.valid.borrow(), BTreeMap::from([(0, 24)]));
        bank.download(vec![(0u8, 32, 24, 8)], |regions, output| {
            assert_eq!(regions, &[(0, 32, 24, 8)]);
            output[24..].copy_from_slice(&(32..40).collect::<Vec<u8>>());
            Ok(())
        })
        .unwrap();
        assert_eq!(*bank.valid.borrow(), BTreeMap::from([(0, 32)]));
        assert_eq!(&*bank.bytes.borrow(), &(8..40).collect::<Vec<u8>>());
        for region in [(0u8, 0, 32, 1), (0, usize::MAX, 0, 1)] {
            assert!(
                bank.download(vec![region], |_, _| panic!("invalid DMA"))
                    .is_err()
            );
        }
    }

    #[test]
    fn cpu_capture_preserves_roles_offsets_and_weak_replica_identity() {
        for backend in [
            orinfer_device::Backend::Cuda,
            orinfer_device::Backend::Ascend,
        ] {
            capture_lifecycle(backend);
        }
    }
    fn capture_lifecycle(backend: orinfer_device::Backend) {
        let owner = DeviceOwner::new(backend, 0);
        let kv = Rc::new(Allocation {
            owner,
            id: allocation_id().unwrap(),
            address: 0,
            bytes: 128,
            used: 17,
            handle: Some(9),
            native: Some(Rc::new(vec![73u8; 17])),
        });
        let state = Rc::new(Allocation {
            owner,
            id: 0,
            address: 0,
            bytes: 64,
            used: 64,
            handle: None,
            native: Some(Rc::new((0..64u8).collect::<Vec<_>>())),
        });
        let piece = |allocation: Rc<Allocation>, name: &str, offset, bytes| Piece {
            allocation,
            name: name.into(),
            range: MemoryRange { offset, bytes },
            source: offset,
        };
        let kv_piece = piece(kv.clone(), "kv", 0, 17);
        let state_pieces = [
            piece(state.clone(), "ring", 8, 4),
            piece(state, "position", 16, 4),
        ];
        let calls = Cell::new(0);
        let mut read = |regions: &[ReadRegion], output: &mut [u8]| {
            calls.set(calls.get() + 1);
            for (a, source, dest, bytes) in regions {
                let data = a
                    .native
                    .as_ref()
                    .unwrap()
                    .downcast_ref::<Vec<u8>>()
                    .unwrap();
                output[*dest..dest + bytes].copy_from_slice(&data[*source..source + bytes]);
            }
            Ok(())
        };
        let mut pool = HostPool::default();
        let mut host = pool
            .capture_components(
                owner,
                std::slice::from_ref(&kv_piece),
                &state_pieces,
                &mut read,
            )
            .unwrap();
        host.metadata_bytes = 32;
        assert_eq!(host.bytes, 25);
        assert_eq!(host.state[0].range.offset, 8);
        assert_eq!(host.state[1].data.offset, 256);
        assert_eq!(
            &host.state[0].data.bank.bytes.borrow()[..4],
            &[8, 9, 10, 11]
        );
        assert_eq!(
            &host.state[1].data.bank.bytes.borrow()[256..260],
            &[16, 17, 18, 19]
        );
        let mut banks = BTreeMap::new();
        host.allocations(&mut banks);
        assert_eq!(banks.values().sum::<usize>(), 17 + 260 + 32);
        let duplicate = pool
            .capture(
                &Snapshot {
                    owner,
                    bytes: 17,
                    state: vec![],
                    kv: vec![kv_piece],
                },
                &mut read,
            )
            .unwrap();
        assert_eq!(calls.get(), 2, "immutable CPU ranges download once");
        let bank = host.kv[0].data.bank.clone();
        assert!(Rc::ptr_eq(&bank, &duplicate.kv[0].data.bank));
        assert!(bank.resident(owner).unwrap().is_some());
        drop(kv);
        assert!(
            bank.resident(owner).unwrap().is_none(),
            "CPU cache must not pin device backing"
        );
        let materializations = Cell::new(0);
        let materialize = |bank: &HostBank| {
            bank.materialize(owner, |d, bytes| {
                assert_eq!(bytes, &[73; 17]);
                materializations.set(materializations.get() + 1);
                Ok(Rc::new(Allocation {
                    owner,
                    id: d.id,
                    address: 1,
                    bytes: d.bytes,
                    used: d.used,
                    handle: Some(10),
                    native: None,
                }))
            })
        };
        assert!(
            bank.materialize(owner, |_, _| Err("failed upload".into()))
                .is_err()
        );
        assert!(bank.resident(owner).unwrap().is_none());
        assert!(
            bank.materialize(owner, |d, _| Ok(Rc::new(Allocation {
                owner,
                id: d.id + 2,
                address: 1,
                bytes: d.bytes,
                used: d.used,
                handle: Some(10),
                native: None,
            })))
            .is_err()
        );
        assert!(bank.resident(owner).unwrap().is_none());
        let first = host.restore_shared(owner, materialize).unwrap();
        let second = host.restore_shared(owner, materialize).unwrap();
        assert!(first.state.is_empty());
        assert!(Rc::ptr_eq(
            &first.kv[0].allocation,
            &second.kv[0].allocation
        ));
        assert_eq!(materializations.get(), 1);
        let foreign = DeviceOwner::new(backend, 1);
        assert!(host.restore_shared(foreign, materialize).is_err());
        assert!(
            bank.materialize(foreign, |_, _| panic!("foreign DMA"))
                .is_err()
        );
    }
}
