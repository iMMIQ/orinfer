//! Owner arena allocation, release and reclaimable packet program memory.
use super::*;

impl Worker {
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
    pub(super) fn reserve_batch_memory(
        &mut self,
        needed: super::super::memory::Requirement,
        stage: &str,
    ) -> Result<()> {
        if needed.fits(
            self.session.memory_info()?.0,
            self.session.remaining_device_bytes(),
        ) {
            return Ok(());
        }
        self.session.synchronize()?;
        let mut evicted = 0;
        while !needed.fits(
            self.session.memory_info()?.0,
            self.session.remaining_device_bytes(),
        ) {
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
        let entries = names
            .iter()
            .filter(|n| !self.views.contains_key(*n))
            .map(|n| Ok((n.as_str(), get(&self.buffers, n)?.bytes(), 256)))
            .collect::<Result<Vec<_>>>()?;
        let (arena, allocation) = super::super::memory::owner_layout(entries.into_iter())?;
        let arena_bytes = allocation.bytes;
        let needed = super::super::memory::Requirement::owners(allocation, count, 0)?;
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
        Ok(())
    }
}
