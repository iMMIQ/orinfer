//! Backend-owned backing for shared prefix extents and private continuation state.
pub(crate) use super::kv::Allocation;
use crate::artifact::Result;
pub(crate) use orinfer_device::MemoryRange as Range;
use std::{collections::BTreeMap, rc::Rc};
#[derive(Clone)]
pub(crate) struct Piece {
    pub(super) allocation: Rc<Allocation>,
    pub(super) name: String,
    pub(super) range: Range,
    pub(super) source: usize,
}
#[derive(Clone)]
pub(crate) struct Snapshot {
    pub(super) owner: orinfer_device::DeviceOwner,
    pub bytes: usize,
    pub(super) state: Vec<Piece>,
    pub kv: Vec<Piece>,
}
impl Snapshot {
    pub(super) fn require_owner(&self, owner: orinfer_device::DeviceOwner) -> Result<()> {
        owner.require(self.owner)?;
        for piece in self.state.iter().chain(&self.kv) {
            owner.require(piece.allocation.owner)?;
        }
        Ok(())
    }
}
/// Private, reusable backing for a live shared-workspace continuation. It is
/// never published to the immutable prefix cache or cloned as a Snapshot.
pub(crate) struct Workspace {
    pub(super) snapshot: Snapshot,
    pub(super) resident: bool,
    pub(super) saved_s: f64,
}
impl Workspace {
    pub(crate) fn bytes(&self) -> usize {
        self.snapshot.bytes
    }
}
pub(crate) struct Plan {
    pub(super) owner: orinfer_device::DeviceOwner,
    pub(super) state: BTreeMap<String, Range>,
    pub(super) shared: BTreeMap<String, Range>,
    pub(super) kv: Vec<Piece>,
    pub new_bytes: usize,
}
impl Plan {
    pub(crate) fn host_banks(&self) -> BTreeMap<u64, usize> {
        self.kv
            .iter()
            .map(|p| (p.allocation.id, p.allocation.used))
            .collect()
    }
}
impl Snapshot {
    pub(crate) fn host_banks(&self) -> BTreeMap<u64, usize> {
        self.state
            .iter()
            .chain(&self.kv)
            .map(|p| (p.allocation.id, p.allocation.used))
            .collect()
    }
}
impl crate::prefix::Resident for Plan {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        for p in &self.kv {
            out.insert(p.allocation.id, p.allocation.bytes);
        }
    }
}
impl crate::prefix::Resident for Snapshot {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        for p in self.state.iter().chain(&self.kv) {
            out.insert(p.allocation.id, p.allocation.bytes);
        }
    }
}
pub(super) fn size(ranges: &BTreeMap<String, Range>) -> Result<usize> {
    ranges.values().try_fold(0usize, |offset, r| {
        offset
            .checked_add(r.bytes)
            .and_then(|n| n.checked_add(255))
            .map(|n| n & !255)
            .ok_or("Snapshot size overflow".into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use orinfer_device::{Backend, DeviceOwner};

    #[test]
    fn snapshot_rejects_foreign_session_and_foreign_backing() {
        let owner = DeviceOwner::new(Backend::Cuda, 0);
        let foreign = DeviceOwner::new(Backend::Cuda, 0);
        let mut snapshot = Snapshot {
            owner,
            bytes: 0,
            state: vec![],
            kv: vec![],
        };
        snapshot.require_owner(owner).unwrap();
        assert!(snapshot.require_owner(foreign).is_err());
        snapshot.kv.push(Piece {
            allocation: Rc::new(Allocation {
                id: 2,
                used: 4,
                native: None,
                owner: foreign,
                address: 42,
                bytes: 4,
                handle: None,
            }),
            name: "kv".into(),
            range: Range {
                offset: 0,
                bytes: 4,
            },
            source: 0,
        });
        assert!(snapshot.require_owner(owner).is_err());
    }
}
