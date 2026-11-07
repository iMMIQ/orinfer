//! Shared allocation geometry and pressure policy for owners, snapshots and programs.
use crate::{artifact::Result, backend::arena::ArenaLayout};
use std::collections::BTreeMap;

pub(super) fn growth_layouts(
    contract: &super::batch::Contract,
    buffers: &[super::Definition],
) -> Result<BTreeMap<String, crate::backend::growth::TokenLayout>> {
    let mut growth = BTreeMap::new();
    for state in contract.state.iter().chain(
        contract
            .decoder_replay
            .iter()
            .flat_map(|r| &r.decoder_state),
    ) {
        if !state.append_only {
            continue;
        }
        let definition = buffers
            .iter()
            .find(|b| b.name == state.buffer)
            .ok_or("Missing growth buffer")?;
        if !contract.private.contains(&state.buffer)
            || definition.view.is_some()
            || definition.weight.is_some()
            || definition.data.is_some()
            || definition.prepared_weight.is_some()
            || definition.prepared_tensor.is_some()
            || state.tokens_per_row == 0
            || state.row_bytes == 0
            || state.circular
            || state.window.is_some()
        {
            return Err("Invalid demand-mapped owner state".into());
        }
        growth.insert(
            state.buffer.clone(),
            crate::backend::growth::TokenLayout {
                row_bytes: state.row_bytes,
                tokens_per_row: state.tokens_per_row,
                capacity_bytes: definition.bytes,
            },
        );
    }
    Ok(growth)
}

pub(super) const DEVICE_WORKSPACE: usize = 3 << 30;
pub(super) const SESSION_WORKSPACE: usize = 1 << 30;
const PAGE: usize = 2 << 20;

#[derive(Clone, Copy, Debug)]
pub(super) struct Allocation {
    pub bytes: usize,
    pub resident: usize,
}
pub(super) fn owner_layout<'a>(
    entries: impl Iterator<Item = (&'a str, usize, usize)>,
) -> Result<(ArenaLayout, Allocation)> {
    let layout = ArenaLayout::from_sizes(entries)?;
    let bytes = layout
        .allocation_bytes
        .checked_add(255)
        .ok_or("Owner arena overflow")?;
    Ok((
        layout,
        Allocation {
            bytes,
            resident: round_pages(bytes)?,
        },
    ))
}
pub(super) fn round_pages(bytes: usize) -> Result<usize> {
    Ok(bytes
        .checked_add(PAGE - 1)
        .ok_or("Allocation extent overflow")?
        / PAGE
        * PAGE)
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Requirement {
    pub device: usize,
    pub session: usize,
}
impl Requirement {
    pub fn owners(allocation: Allocation, count: usize, reserve: usize) -> Result<Self> {
        let bytes = allocation
            .bytes
            .checked_mul(count)
            .ok_or("Owner budget overflow")?;
        let resident = allocation
            .resident
            .checked_mul(count)
            .ok_or("Owner extent overflow")?;
        Self::with_extent(bytes, resident, reserve)
    }
    pub fn snapshot(resident: usize) -> Result<Self> {
        Self::with_extent(resident, resident, 0)
    }
    fn with_extent(bytes: usize, resident: usize, reserve: usize) -> Result<Self> {
        Ok(Self {
            device: resident
                .checked_add(reserve)
                .and_then(|n| n.checked_add(DEVICE_WORKSPACE))
                .ok_or("Device requirement overflow")?,
            session: bytes
                .checked_add(reserve.max(SESSION_WORKSPACE))
                .ok_or("Session requirement overflow")?,
        })
    }
    pub fn workspace() -> Self {
        Self {
            device: DEVICE_WORKSPACE,
            session: SESSION_WORKSPACE,
        }
    }
    pub fn fits(self, device_free: usize, session_free: usize) -> bool {
        device_free >= self.device && session_free >= self.session
    }
}

pub(super) trait Pressure {
    fn fits(&mut self, requirement: Requirement) -> Result<bool>;
    fn evict_prefix(&mut self) -> Result<bool>;
    fn evict_program(&mut self) -> Result<bool>;
}
pub(super) fn reclaim_one(cache: &mut impl Pressure) -> Result<bool> {
    Ok(cache.evict_prefix()? || cache.evict_program()?)
}
pub(super) fn ensure_budget(cache: &mut impl Pressure, requirement: Requirement) -> Result<bool> {
    while !cache.fits(requirement)? {
        if !reclaim_one(cache)? {
            // A reclaim attempt can also fence and release completed tickets.
            return cache.fits(requirement);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct CachedPrograms {
        free: usize,
        program: usize,
        reclaimed: usize,
    }
    impl Pressure for CachedPrograms {
        fn fits(&mut self, r: Requirement) -> Result<bool> {
            Ok(r.fits(self.free, self.free))
        }
        fn evict_prefix(&mut self) -> Result<bool> {
            Ok(false)
        }
        fn evict_program(&mut self) -> Result<bool> {
            if self.program == 0 {
                return Ok(false);
            }
            self.free += self.program;
            self.program = 0;
            self.reclaimed += 1;
            Ok(true)
        }
    }
    #[test]
    fn admission_reclaims_programs_even_without_prefix_entries() {
        let mut cache = CachedPrograms {
            free: 0,
            program: 8 << 30,
            reclaimed: 0,
        };
        assert!(ensure_budget(&mut cache, Requirement::workspace()).unwrap());
        assert_eq!(cache.reclaimed, 1);
        assert!(
            !ensure_budget(
                &mut cache,
                Requirement {
                    device: 9 << 30,
                    session: 0
                }
            )
            .unwrap()
        );
    }
    #[test]
    fn physical_and_session_limits_are_checked_independently() {
        let (_, arena) =
            owner_layout([("small", 1, 256), ("large", 2 << 20, 256)].into_iter()).unwrap();
        assert_eq!(arena.resident, 4 << 20);
        let requirement = Requirement::owners(arena, 3, 0).unwrap();
        assert_eq!(requirement.device, 12 * (1 << 20) + DEVICE_WORKSPACE);
        assert!(!requirement.fits(usize::MAX, requirement.session - 1));
        assert!(!requirement.fits(requirement.device - 1, usize::MAX));
        assert!(requirement.fits(requirement.device, requirement.session));
        assert!(Requirement::owners(arena, usize::MAX, 0).is_err());
    }
}
