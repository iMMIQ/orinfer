//! Backend-owned backing for shared prefix extents and private continuation state.
pub(crate) use super::kv::Allocation;
use crate::artifact::Result;
pub(crate) use orinfer_device::MemoryRange as Range;
use std::{collections::BTreeMap, rc::Rc};
#[derive(Clone)]
pub(crate) struct Piece {
    pub(crate) allocation: Rc<Allocation>,
    pub(crate) name: String,
    pub(crate) range: Range,
    pub(crate) source: usize,
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
#[derive(Clone)]
pub(crate) struct Plan {
    pub(super) owner: orinfer_device::DeviceOwner,
    pub(super) state: BTreeMap<String, Range>,
    pub(super) shared: BTreeMap<String, Range>,
    pub(super) tails: BTreeMap<String, Range>,
    pub(super) host_inherited: Vec<super::kv::host::HostPiece>,
    pub(super) kv: Vec<Piece>,
    pub new_bytes: usize,
}
impl Plan {
    pub(crate) fn for_host_parent(
        &self,
        parent: Option<&super::kv::host::HostSnapshot>,
    ) -> Result<Self> {
        let mut plan = self.clone();
        let Some(parent) = parent else {
            return Ok(plan);
        };
        self.owner.require(parent.owner)?;
        if self.kv.iter().any(|p| p.allocation.handle.is_none()) {
            return Ok(plan);
        }
        for (name, tail) in &self.tails {
            let candidates = parent
                .state
                .iter()
                .chain(&parent.kv)
                .filter(|p| &p.name == name && !p.data.bank.descriptor().is_some_and(|d| d.paged))
                .collect::<Vec<_>>();
            let reuse = super::kv::reuse_tail(*tail, candidates.iter().map(|p| p.range))?;
            for part in reuse.parts {
                let p = candidates[part.index];
                plan.host_inherited.push(super::kv::host::HostPiece {
                    name: p.name.clone(),
                    range: part.range,
                    data: p.data.view(Range {
                        offset: part.source_offset,
                        bytes: part.range.bytes,
                    })?,
                });
            }
            if reuse.missing.bytes > 0 {
                plan.state.insert(name.clone(), reuse.missing);
            } else {
                plan.state.remove(name);
            }
        }
        plan.new_bytes = size(&plan.state)?;
        Ok(plan)
    }
    pub(crate) fn host_banks(&self) -> BTreeMap<u64, usize> {
        let mut banks = self
            .kv
            .iter()
            .map(|p| (p.allocation.id, p.allocation.used))
            .collect::<BTreeMap<_, _>>();
        for p in &self.host_inherited {
            let id = p
                .data
                .bank
                .descriptor()
                .expect("Saved CPU bank identity")
                .id;
            banks.insert(id, p.data.bank.bytes.borrow().len());
        }
        banks
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
    fn cpu_parent_plan_reuses_only_kv_and_accounts_inherited_banks() {
        let owner = DeviceOwner::new(Backend::Cuda, 0);
        let allocation = Rc::new(Allocation {
            owner,
            id: 2,
            address: 0,
            bytes: 20,
            used: 20,
            handle: None,
            native: None,
        });
        let state = vec![
            Piece {
                allocation: allocation.clone(),
                name: "kv".into(),
                range: Range {
                    offset: 0,
                    bytes: 16,
                },
                source: 0,
            },
            Piece {
                allocation,
                name: "position".into(),
                range: Range {
                    offset: 0,
                    bytes: 4,
                },
                source: 16,
            },
        ];
        let mut pool = super::super::kv::host::HostPool::default();
        let parent = pool
            .capture(
                &Snapshot {
                    owner,
                    bytes: 20,
                    state,
                    kv: vec![],
                },
                |regions, output| {
                    for (_, _, dest, bytes) in regions {
                        output[*dest..dest + bytes].fill(73);
                    }
                    Ok(())
                },
            )
            .unwrap();
        let plan = Plan {
            owner,
            state: [
                (
                    "kv".into(),
                    Range {
                        offset: 0,
                        bytes: 24,
                    },
                ),
                (
                    "position".into(),
                    Range {
                        offset: 0,
                        bytes: 4,
                    },
                ),
            ]
            .into(),
            shared: BTreeMap::new(),
            tails: [(
                "kv".into(),
                Range {
                    offset: 0,
                    bytes: 24,
                },
            )]
            .into(),
            host_inherited: vec![],
            kv: vec![],
            new_bytes: 512,
        };
        let inherited = plan.for_host_parent(Some(&parent)).unwrap();
        assert_eq!(inherited.host_inherited.len(), 1);
        assert_eq!(
            inherited.host_inherited[0].range,
            Range {
                offset: 0,
                bytes: 16
            }
        );
        assert_eq!(
            inherited.state["kv"],
            Range {
                offset: 16,
                bytes: 8
            }
        );
        assert_eq!(
            inherited.state["position"],
            Range {
                offset: 0,
                bytes: 4
            }
        );
        assert_eq!(inherited.host_banks().get(&2), Some(&20));
        assert_eq!(plan.state["kv"].bytes, 24);
    }
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
