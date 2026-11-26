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
    pub(in crate::models::deepseek_v41) fn owner_requirement(
        &self,
        tokens: usize,
        shared_tokens: usize,
        reserve: usize,
    ) -> Result<Requirement> {
        if tokens == 0 || tokens > self.capacity {
            return Err("Invalid owner token reservation".into());
        }
        let (_, allocation) = self.fixed_owner_layout()?;
        let new_banks = usize::from(self.batch.as_ref().is_none_or(|r| r.idle_owners.is_empty()));
        let mut needed = Requirement::owners(allocation, new_banks, reserve)?;
        let shared = shared_tokens.min(tokens);
        const VMM_PAGE: usize = 2 << 20;
        for (name, layout) in &self.growth {
            let total = get(&self.buffers, name)?.physical_extent(layout.extent(tokens)?)?;
            // Whole 2 MiB VMM pages covered by an already-resident device prefix
            // are zero-copy shared; keep any partial tail page budgeted for COW.
            let closed_shared = (shared / layout.tokens_per_row)
                .checked_mul(layout.row_bytes)
                .ok_or("Owner shared prefix budget overflow")?
                / VMM_PAGE
                * VMM_PAGE;
            let bytes = total.saturating_sub(closed_shared);
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
            let buffer = get(&owner.buffers, name)?;
            let end = layout.extent(tokens)?;
            let start = owner.history.len() / layout.tokens_per_row * layout.row_bytes;
            let cow = buffer.kv_writable_bytes(orinfer_model_sdk::device::MemoryRange {
                offset: start,
                bytes: end.saturating_sub(start),
            })?;
            sum.checked_add(buffer.growth_bytes(end)?)
                .and_then(|n| n.checked_add(cow))
                .ok_or_else(|| "Owner future KV budget overflow".into())
        })
    }
    /// Capacity promised to live requests is unavailable to new owners, snapshots and Graph scratch.
    pub(in crate::models::deepseek_v41) fn available_memory(&self) -> Result<(usize, usize)> {
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
    /// Check budget, dropping any excess pooled idle owners beyond one before
    /// reporting pressure so idle owner arenas never trigger false admission drains.
    pub(in crate::models::deepseek_v41) fn check_budget(
        &mut self,
        needed: Requirement,
    ) -> Result<bool> {
        let (device, session) = self.available_memory()?;
        if needed.fits(device, session) {
            return Ok(true);
        }
        let mut dropped = false;
        if let Some(runtime) = self.batch.as_mut() {
            while runtime.idle_owners.len() > 1 {
                runtime.idle_owners.pop();
                dropped = true;
            }
        }
        if !dropped {
            return Ok(false);
        }
        self.session.synchronize()?;
        let (device, session) = self.available_memory()?;
        Ok(needed.fits(device, session))
    }
    pub(in crate::models::deepseek_v41) fn limit_owner(
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
    pub(in crate::models::deepseek_v41) fn grow_owner_protected(
        &mut self,
        id: usize,
        tokens: usize,
        protected: &ProtectedKeys<'_>,
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
            self.reserve_batch_memory_protected(
                Requirement::snapshot(additional)?,
                "KV growth",
                protected,
            )?;
        }
        let owner = self.batch.as_ref().unwrap().owners.get(&id).unwrap();
        let growth = self
            .growth
            .iter()
            .map(|(name, layout)| Ok((get(&owner.buffers, name)?, layout.extent(tokens)?)))
            .collect::<Result<Vec<_>>>()?;
        let writes = self
            .growth
            .iter()
            .map(|(name, layout)| {
                let start = owner.history.len() / layout.tokens_per_row * layout.row_bytes;
                Ok((
                    get(&owner.buffers, name)?,
                    orinfer_model_sdk::device::MemoryRange {
                        offset: start,
                        bytes: layout.extent(tokens)?.saturating_sub(start),
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        self.session.make_kv_writable(&writes)?;
        self.session.grow_many(&growth)
    }
    pub(in crate::models::deepseek_v41) fn grow_owner_readonly(
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
            return Err("KV restore exceeds admitted budget".into());
        }
        let growth = self
            .growth
            .iter()
            .map(|(name, layout)| Ok((get(&owner.buffers, name)?, layout.extent(tokens)?)))
            .collect::<Result<Vec<_>>>()?;
        self.session.grow_many(&growth)
    }
    pub(in crate::models::deepseek_v41) fn grow_batch(&mut self, layout: &Layout) -> Result<()> {
        self.grow_batch_protected(layout, &ProtectedKeys::default())
    }
    pub(in crate::models::deepseek_v41) fn grow_batch_protected(
        &mut self,
        layout: &Layout,
        protected: &ProtectedKeys<'_>,
    ) -> Result<()> {
        if layout.role == super::super::PacketRole::Draft {
            return Ok(());
        }
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
            self.grow_owner_protected(segment.slot, tokens, protected)?;
        }
        Ok(())
    }
    pub(in crate::models::deepseek_v41) fn clear_base_growth(&self) -> Result<()> {
        let buffers = self
            .growth
            .keys()
            .map(|name| get(&self.buffers, name))
            .collect::<Result<Vec<_>>>()?;
        self.session.release_pages_many(&buffers)
    }
    pub(in crate::models::deepseek_v41) fn evict_batch_program(&mut self) -> Result<bool> {
        self.evict_batch_program_protected(&ProtectedKeys::default())
    }
    pub(in crate::models::deepseek_v41) fn evict_batch_program_for_admission(
        &mut self,
    ) -> Result<bool> {
        self.evict_batch_program_protected(&ProtectedKeys {
            keep_idle_owner: true,
            ..ProtectedKeys::default()
        })
    }
    pub(in crate::models::deepseek_v41) fn evict_batch_program_protected(
        &mut self,
        protected: &ProtectedKeys<'_>,
    ) -> Result<bool> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        let tables_released = runtime.bound_tables.take().is_some();
        let min_idle = usize::from(protected.keep_idle_owner);
        if runtime.idle_owners.len() > min_idle {
            runtime.idle_owners.pop();
            return Ok(true);
        }
        while let Some(pos) = runtime.lru.iter().position(|k| Some(k) != protected.mono) {
            let key = runtime.lru.remove(pos).unwrap();
            if runtime.remove(&key) {
                self.w4.retain(|_, w| w.strong_count() > 0);
                self.session.synchronize()?;
                return Ok(true);
            }
        }
        while let Some(pos) = runtime
            .head_lru
            .iter()
            .position(|k| Some(k) != protected.head)
        {
            let key = runtime.head_lru.remove(pos).unwrap();
            if runtime.remove_head(&key) {
                self.w4.retain(|_, w| w.strong_count() > 0);
                self.session.synchronize()?;
                return Ok(true);
            }
        }
        while let Some(pos) = runtime
            .tail_lru
            .iter()
            .position(|k| Some(k) != protected.tail)
        {
            let key = runtime.tail_lru.remove(pos).unwrap();
            if runtime.remove_tail(&key) {
                self.w4.retain(|_, w| w.strong_count() > 0);
                self.session.synchronize()?;
                return Ok(true);
            }
        }
        while let Some(pos) = runtime
            .trunk_lru
            .iter()
            .position(|k| Some(k) != protected.trunk)
        {
            let key = runtime.trunk_lru.remove(pos).unwrap();
            if runtime.remove_trunk(&key) {
                self.w4.retain(|_, w| w.strong_count() > 0);
                self.session.synchronize()?;
                return Ok(true);
            }
        }
        if self.shared.evict() {
            return Ok(true);
        }
        if self
            .vision
            .as_mut()
            .is_some_and(super::super::vision::Encoder::evict_frame)
        {
            self.session.synchronize()?;
            return Ok(true);
        }
        Ok(tables_released)
    }
    pub(in crate::models::deepseek_v41) fn reserve_batch_memory(
        &mut self,
        needed: Requirement,
        stage: &str,
    ) -> Result<()> {
        self.reserve_batch_memory_protected(needed, stage, &ProtectedKeys::default())
    }
    pub(in crate::models::deepseek_v41) fn reserve_batch_memory_protected(
        &mut self,
        needed: Requirement,
        stage: &str,
        protected: &ProtectedKeys<'_>,
    ) -> Result<()> {
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
            if !self.evict_batch_program_protected(protected)? {
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
    pub(in crate::models::deepseek_v41) fn admit(&mut self, ids: &[usize]) -> Result<()> {
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
        let new_banks = count.saturating_sub(runtime.idle_owners.len());
        let (arena, allocation) = self.fixed_owner_layout()?;
        let arena_bytes = allocation.bytes;
        let needed = Requirement::owners(allocation, new_banks, 0)?;
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
            if id != 0
                && let Some(idle) = self.batch.as_mut().unwrap().idle_owners.pop()
            {
                self.batch.as_mut().unwrap().owners.insert(
                    id,
                    Owner {
                        bank: Some(idle.bank),
                        buffers: idle.buffers,
                        history: vec![],
                        limit: None,
                        vision: Default::default(),
                        clean: false,
                    },
                );
                self.reset_owner(id)?;
                continue;
            }
            let mut buffers = BTreeMap::new();
            let bank = if id == 0 {
                None
            } else {
                Some(self.session.allocate(arena_bytes)?)
            };
            let base = if let Some(bank) = &bank {
                // Async compute-stream zero covers every fixed private buffer
                // (including packet.reset), so no second reset memset is needed.
                self.session.fill(bank, 0)?;
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
            let zeroed = bank.is_some();
            self.batch.as_mut().unwrap().owners.insert(
                id,
                Owner {
                    bank,
                    buffers,
                    history: vec![],
                    limit: None,
                    vision: Default::default(),
                    clean: zeroed,
                },
            );
            if zeroed {
                if let Some(inputs) = &mut self.device_inputs {
                    inputs.invalidate(id)?;
                }
            } else {
                self.reset_owner(id)?;
            }
        }
        Ok(())
    }
    pub(in crate::models::deepseek_v41) fn reset_owner(&mut self, id: usize) -> Result<()> {
        if let Some(inputs) = &mut self.device_inputs {
            inputs.invalidate(id)?;
        }
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
        let buffers = names
            .into_iter()
            .map(|name| get(&owner.buffers, name))
            .collect::<Result<Vec<_>>>()?;
        // Compute-stream ordered before any subsequent restore or forward.
        self.session.fill_many(&buffers, 0)?;
        owner.history.clear();
        owner.vision = Default::default();
        owner.clean = true;
        Ok(())
    }
    pub(in crate::models::deepseek_v41) fn release_owners(&mut self) -> Result<()> {
        self.session.synchronize()?;
        if let Some(inputs) = &mut self.device_inputs {
            inputs.invalidate_all()?;
        }
        let runtime = self.batch.as_mut().ok_or("Missing batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.clear();
        runtime.idle_owners.clear();
        self.clear_base_growth()?;
        Ok(())
    }
}
