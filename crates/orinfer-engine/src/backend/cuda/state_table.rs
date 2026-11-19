//! Resolve immutable model bindings once per stable decode cohort.
use super::{Result, sequence::Sequence};
use orinfer_model_sdk::packet::StateBindings;
use std::{
    collections::{BTreeSet, VecDeque},
    rc::Rc,
};

struct Resolved {
    bindings: Rc<StateBindings>,
    owners: Vec<usize>,
    bytes: Rc<[u8]>,
}

#[derive(Default)]
pub(super) struct StateTable {
    recent: VecDeque<Resolved>,
}

impl StateTable {
    pub(super) fn invalidate(&mut self) {
        self.recent.clear();
    }

    pub(super) fn resolve(
        &mut self,
        bindings: Rc<StateBindings>,
        sequences: &[Sequence],
    ) -> Result<Rc<[u8]>> {
        if let Some(index) = self
            .recent
            .iter()
            .position(|entry| Rc::ptr_eq(&entry.bindings, &bindings))
        {
            let cached = &self.recent[index];
            // The retained Rc prevents identity reuse or in-place mutation.
            // Lease validity can change without replacing any allocation.
            for &slot in &cached.owners {
                Self::owner(sequences, slot)?;
            }
            let cached = self.recent.remove(index).unwrap();
            let bytes = Rc::clone(&cached.bytes);
            self.recent.push_back(cached);
            return Ok(bytes);
        }
        let capacity = bindings
            .len()
            .checked_mul(8)
            .ok_or("Address table size overflow")?;
        let mut bytes = Vec::with_capacity(capacity);
        let mut owners = BTreeSet::new();
        for binding in bindings.iter() {
            let address = if let Some((slot, buffer)) = binding {
                let arena = Self::owner(sequences, *slot)?;
                owners.insert(*slot);
                *arena
                    .addresses
                    .get(buffer)
                    .ok_or("Address table requires private state")?
            } else {
                0
            };
            bytes.extend_from_slice(&address.to_le_bytes());
        }
        let bytes: Rc<[u8]> = bytes.into();
        const CAPACITY: usize = 8;
        if self.recent.len() == CAPACITY {
            self.recent.pop_front();
        }
        self.recent.push_back(Resolved {
            bindings,
            owners: owners.into_iter().collect(),
            bytes: Rc::clone(&bytes),
        });
        Ok(bytes)
    }

    fn owner(sequences: &[Sequence], slot: usize) -> Result<&Sequence> {
        sequences
            .get(slot)
            .filter(|s| s.leased && !s.quarantined)
            .ok_or_else(|| "Address table references an unleased arena".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arena(address: u64) -> Sequence {
        let mut arena = Sequence::default();
        arena.addresses.insert("state".into(), address);
        arena.leased = true;
        arena
    }

    #[test]
    fn repeated_bindings_reuse_bytes_and_still_validate_owners() {
        let mut arenas = [arena(42)];
        let mut bindings = Rc::new(vec![Some((0, "state".into())), None]);
        let mut table = StateTable::default();
        let first = table.resolve(Rc::clone(&bindings), &arenas).unwrap();
        let repeated = table.resolve(Rc::clone(&bindings), &arenas).unwrap();
        assert!(Rc::ptr_eq(&first, &repeated));
        assert_eq!(&*first, [42u64.to_le_bytes(), 0u64.to_le_bytes()].concat());
        arenas[0].leased = false;
        assert!(table.resolve(Rc::clone(&bindings), &arenas).is_err());
        arenas[0].leased = true;
        arenas[0].quarantined = true;
        assert!(table.resolve(Rc::clone(&bindings), &arenas).is_err());
        arenas[0].quarantined = false;
        // Mutation makes a new identity because the cache retains the old Rc.
        Rc::make_mut(&mut bindings)[0] = Some((0, "unknown".into()));
        assert!(table.resolve(bindings, &arenas).is_err());
    }

    #[test]
    fn allocation_invalidation_and_changed_membership_resolve_new_addresses() {
        let mut arenas = [arena(42), arena(73)];
        let bindings = Rc::new(vec![Some((0, "state".into()))]);
        let mut table = StateTable::default();
        let old = table.resolve(Rc::clone(&bindings), &arenas).unwrap();
        // Allocation replacement shares the graph invalidation boundary.
        table.invalidate();
        arenas[0].addresses.insert("state".into(), 99);
        let new = table.resolve(bindings, &arenas).unwrap();
        assert!(!Rc::ptr_eq(&old, &new));
        assert_eq!(&*new, 99u64.to_le_bytes());
        let changed = Rc::new(vec![None, Some((1, "state".into()))]);
        let new = table.resolve(changed, &arenas).unwrap();
        assert_eq!(&*new, [0u64.to_le_bytes(), 73u64.to_le_bytes()].concat());
    }

    #[test]
    fn alternating_program_tables_reuse_resolved_bytes() {
        let mut arenas = [arena(42), arena(73)];
        let verify = Rc::new(vec![Some((0, "state".into()))]);
        let commit = Rc::new(vec![Some((1, "state".into()))]);
        let mut table = StateTable::default();
        let first = table.resolve(Rc::clone(&verify), &arenas).unwrap();
        table.resolve(commit, &arenas).unwrap();
        assert!(Rc::ptr_eq(
            &first,
            &table.resolve(Rc::clone(&verify), &arenas).unwrap()
        ));
        arenas[0].leased = false;
        assert!(table.resolve(verify, &arenas).is_err());
    }
}
