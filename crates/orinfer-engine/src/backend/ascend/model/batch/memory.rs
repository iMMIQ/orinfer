//! Owner arena allocation, release and reclaimable packet program memory.
use super::super::memory::{self, Allocation, Requirement};
use super::*;
use crate::backend::arena::ArenaLayout;

impl Worker {
    fn fixed_owner_layout(&self) -> Result<(ArenaLayout, Allocation)> {
        let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
        let entries = runtime
            .contract
            .private
            .iter()
            .filter(|n| !self.views.contains_key(*n) && !self.growth.contains_key(*n))
            .map(|n| Ok((n.as_str(), get(&self.buffers, n)?.bytes(), 256)))
            .collect::<Result<Vec<_>>>()?;
        memory::owner_layout(entries.into_iter())
    }
    pub(in crate::backend::ascend::model) fn owner_requirement(
        &self,
        tokens: usize,
        reserve: usize,
    ) -> Result<Requirement> {
        if tokens == 0 || tokens > self.capacity {
            return Err("Invalid owner token reservation".into());
        }
        let (_, allocation) = self.fixed_owner_layout()?;
        let mut needed = Requirement::owners(allocation, 1, reserve)?;
        for (name, layout) in &self.growth {
            let bytes = get(&self.buffers, name)?.physical_extent(layout.extent(tokens)?)?;
            needed.device = needed
                .device
                .checked_add(bytes)
                .ok_or("Owner device budget overflow")?;
            needed.session = needed
                .session
                .checked_add(bytes)
                .ok_or("Owner session budget overflow")?;
        }
        Ok(needed)
    }
    fn future_bytes(&self, owner: &Owner, tokens: usize) -> Result<usize> {
        self.growth.iter().try_fold(0usize, |sum, (name, layout)| {
            sum.checked_add(get(&owner.buffers, name)?.growth_bytes(layout.extent(tokens)?)?)
                .ok_or_else(|| "Owner future KV budget overflow".into())
        })
    }
    /// Capacity promised to live requests is unavailable to new owners, snapshots and Graph scratch.
    pub(in crate::backend::ascend::model) fn available_memory(&self) -> Result<(usize, usize)> {
        let mut held = 0usize;
        if let Some(runtime) = &self.batch {
            for owner in runtime.owners.values() {
                if let Some(tokens) = owner.limit {
                    held = held
                        .checked_add(self.future_bytes(owner, tokens)?)
                        .ok_or("Future KV budget overflow")?;
                }
            }
        }
        Ok((
            self.session.memory_info()?.0.saturating_sub(held),
            self.session.remaining_device_bytes().saturating_sub(held),
        ))
    }
    pub(in crate::backend::ascend::model) fn limit_owner(
        &mut self,
        id: usize,
        tokens: usize,
    ) -> Result<()> {
        if tokens == 0 || tokens > self.capacity {
            return Err("Owner token limit exceeds context".into());
        }
        let owner = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .owners
            .get(&id)
            .ok_or("Unknown owner")?;
        if owner.limit.is_some() {
            return Err("Owner token budget already reserved".into());
        }
        let future = self.future_bytes(owner, tokens)?;
        self.reserve_batch_memory(Requirement::snapshot(future)?, "owner KV reservation")?;
        self.batch
            .as_mut()
            .unwrap()
            .owners
            .get_mut(&id)
            .unwrap()
            .limit = Some(tokens);
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn grow_owner(
        &mut self,
        id: usize,
        tokens: usize,
    ) -> Result<()> {
        let owner = self
            .batch
            .as_ref()
            .ok_or("Missing batch runtime")?
            .owners
            .get(&id)
            .ok_or("Unknown owner")?;
        if owner.limit.is_some_and(|limit| tokens > limit) {
            return Err("KV growth exceeds admitted token budget".into());
        }
        let additional = self.future_bytes(owner, tokens)?;
        if additional == 0 {
            return Ok(());
        }
        // Mapping an admitted request converts its held capacity to resident
        // bytes. Existing Graph replay needs no new scratch reservation.
        if owner.limit.is_none() {
            self.reserve_batch_memory(Requirement::snapshot(additional)?, "KV growth")?;
        }
        let owner = self.batch.as_ref().unwrap().owners.get(&id).unwrap();
        for (name, layout) in &self.growth {
            self.session
                .grow(&get(&owner.buffers, name)?, layout.extent(tokens)?)?;
        }
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn grow_batch(&mut self, layout: &Layout) -> Result<()> {
        for segment in &layout.segments {
            let owner = self
                .batch
                .as_ref()
                .ok_or("Missing batch runtime")?
                .owners
                .get(&segment.slot)
                .ok_or("Unknown growth owner")?;
            let tokens = owner
                .history
                .len()
                .checked_add(segment.tokens)
                .ok_or("Owner token growth overflow")?;
            self.grow_owner(segment.slot, tokens)?;
        }
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn clear_base_growth(&self) -> Result<()> {
        for name in self.growth.keys() {
            self.session.release_pages(&get(&self.buffers, name)?)?;
        }
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn evict_batch_program(&mut self) -> Result<bool> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        let tables_released = runtime.bound_tables.take().is_some();
        while let Some(key) = runtime.lru.pop_front() {
            if runtime.cache.remove(&key).is_some() {
                self.w4.retain(|_, w| w.strong_count() > 0);
                self.session.synchronize()?;
                return Ok(true);
            }
        }
        Ok(tables_released)
    }
    pub(super) fn reserve_batch_memory(&mut self, needed: Requirement, stage: &str) -> Result<()> {
        let (device, session) = self.available_memory()?;
        if needed.fits(device, session) {
            return Ok(());
        }
        self.session.synchronize()?;
        let mut evicted = 0;
        loop {
            let (device, session) = self.available_memory()?;
            if needed.fits(device, session) {
                break;
            }
            if !self.evict_batch_program()? {
                return Err(format!(
                    "Insufficient A2 memory for {stage}: device free {}, session free {}, required {needed:?}",
                    self.session.memory_info()?.0,
                    self.session.remaining_device_bytes()
                ));
            }
            evicted += 1;
        }
        eprintln!(
            "A2 rank {} evicted {evicted} cached packet shapes for {stage}",
            self.rank
        );
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn admit(&mut self, ids: &[usize]) -> Result<()> {
        self.batch
            .as_mut()
            .ok_or("Missing native batch contract")?
            .bound_tables = None;
        let runtime = self.batch.as_ref().ok_or("Missing native batch contract")?;
        let names = runtime.contract.private.clone();
        let count = ids
            .iter()
            .copied()
            .filter(|id| *id != 0 && !runtime.owners.contains_key(id))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let (arena, allocation) = self.fixed_owner_layout()?;
        let arena_bytes = allocation.bytes;
        let needed = Requirement::owners(allocation, count, 0)?;
        self.reserve_batch_memory(needed, "owner admission")?;
        for &id in ids {
            if self.batch.as_ref().unwrap().owners.contains_key(&id) {
                continue;
            }
            if self.batch.as_ref().unwrap().owners.len()
                >= self.batch.as_ref().unwrap().contract.owners
            {
                return Err("Owner capacity exceeded".into());
            }
            let mut buffers = BTreeMap::new();
            let bank = if id == 0 {
                None
            } else {
                Some(self.session.allocate(arena_bytes)?)
            };
            let base = if let Some(bank) = &bank {
                self.session.fill(bank, 0)?.wait()?;
                usize::try_from(arena.base(bank.pointer() as usize as u64)?)
                    .map_err(|e| e.to_string())?
                    - (bank.pointer() as usize)
            } else {
                0
            };
            for name in &names {
                let buffer = if id == 0 {
                    get(&self.buffers, name)?
                } else if let Some((view, bytes)) = self.views.get(name) {
                    get(&buffers, &view.parent)?.slice(view.offset, *bytes)?
                } else if self.growth.contains_key(name) {
                    self.session.reserve(get(&self.buffers, name)?.bytes())?
                } else {
                    bank.as_ref().unwrap().slice(
                        base + arena.offsets[name],
                        get(&self.buffers, name)?.bytes(),
                    )?
                };
                buffers.insert(name.clone(), buffer);
            }
            self.batch.as_mut().unwrap().owners.insert(
                id,
                Owner {
                    buffers,
                    history: vec![],
                    limit: None,
                },
            );
            self.reset_owner(id)?;
        }
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn reset_owner(&mut self, id: usize) -> Result<()> {
        let names = self
            .packet_templates
            .iter()
            .flat_map(|p| p.reset.iter())
            .collect::<std::collections::BTreeSet<_>>();
        let owner = self
            .batch
            .as_mut()
            .ok_or("Missing batch runtime")?
            .owners
            .get_mut(&id)
            .ok_or("Unknown owner")?;
        for name in names {
            self.session.fill(&get(&owner.buffers, name)?, 0)?.wait()?;
        }
        owner.history.clear();
        Ok(())
    }
    pub(in crate::backend::ascend::model) fn release_owners(&mut self) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.clear();
        self.clear_base_growth()?;
        Ok(())
    }
}
