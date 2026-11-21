//! CPU storage for the same state-span plan as device snapshots.
use super::*;
impl StateBytes for HostBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn view(&self, range: orinfer_model_sdk::device::MemoryRange) -> Result<Self> {
        self.view(range)
    }
    fn paged(&self) -> bool {
        self.bank.descriptor().is_some_and(|d| d.paged)
    }
    fn host_capacity(&self) -> usize {
        self.bank.bytes.borrow().len()
    }
}
impl HostSnapshot {
    /// Device views are copied, including a parent's mutable ranges when it is
    /// evicted after its child. Shared CPU banks remain charged exactly once.
    pub(in crate::models::deepseek_v41) fn offload(
        session: &Session,
        source: &Snapshot,
        pool: &mut HostPool,
    ) -> Result<Self> {
        session.synchronize()?;
        let mut pieces = Vec::with_capacity(source.pieces.len());
        for (id, bytes) in source.host_allocations() {
            let bank = pool.bank(id, bytes)?;
            let regions = source
                .pieces
                .iter()
                .filter(|p| p.bank_id == id)
                .map(|p| Ok((p.buffer.native()?, 0, p.buffer.source, p.buffer.bytes())))
                .collect::<Result<Vec<_>>>()?;
            bank.download(regions, |regions, bytes| {
                session.download_regions(regions, bytes)
            })?;
            bank.remember(
                &source
                    .pieces
                    .iter()
                    .find(|p| p.bank_id == id)
                    .unwrap()
                    .buffer
                    .allocation,
            )?;
            for piece in source.pieces.iter().filter(|p| p.bank_id == id) {
                pieces.push(Piece {
                    name: piece.name.clone(),
                    offset: piece.offset,
                    buffer: HostBuffer {
                        bank: Rc::clone(&bank),
                        offset: piece.buffer.source,
                        bytes: piece.buffer.bytes(),
                    },
                    bank_id: id,
                    bank_bytes: bytes,
                });
            }
        }
        // Keep inherited pieces in logical row order, independent of bank IDs.
        pieces.sort_by(|a, b| (&a.name, a.offset).cmp(&(&b.name, b.offset)));
        Ok(Self {
            history: source.history.clone(),
            pieces,
        })
    }
    fn capture_host<'a>(
        session: &Session,
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        id: u64,
        parent: Option<&Self>,
        pool: &mut HostPool,
    ) -> Result<Self> {
        let (
            CapturePlan {
                mut pieces,
                pending,
                bytes,
            },
            shared_preview,
        ) = Self::plan_checkpoint(layouts, buffers, history, parent)?;
        let gpu_pieces = publish_shared(session, buffers, &shared_preview)?;
        let shared = Self::offload(
            session,
            &Snapshot {
                history: history.to_vec(),
                pieces: gpu_pieces,
            },
            pool,
        )?;
        pieces.extend(shared.pieces);
        if !pending.is_empty() {
            let bank_bytes = bytes.checked_add(255).ok_or("Host snapshot overflow")?;
            let id = kv::checkpoint_id(id)?;
            let bank = pool.bank(id, bank_bytes)?;
            let regions = pending
                .iter()
                .map(|(_, buffer, span, offset)| (buffer.clone(), span.start, *offset, span.len()))
                .collect();
            bank.download(regions, |regions, bytes| {
                session.download_regions(regions, bytes)
            })?;
            for (name, _, span, offset) in pending {
                pieces.push(Piece {
                    name,
                    offset: span.start,
                    buffer: HostBuffer {
                        bank: bank.clone(),
                        offset,
                        bytes: span.len(),
                    },
                    bank_id: id,
                    bank_bytes,
                });
            }
        }
        Ok(Self {
            history: history.to_vec(),
            pieces,
        })
    }
    fn restore_shared_into(
        &self,
        session: &Session,
        buffers: &BTreeMap<String, Buffer>,
    ) -> Result<()> {
        let mut groups = BTreeMap::<&str, Vec<SharedPiece>>::new();
        for p in &self.pieces {
            if p.buffer.bank.descriptor().is_some_and(|d| d.paged) {
                let allocation = session.host_kv(&p.buffer.bank)?;
                groups.entry(&p.name).or_default().push(SharedPiece {
                    allocation,
                    name: p.name.clone(),
                    range: crate::backend::snapshot::Range {
                        offset: p.offset,
                        bytes: p.buffer.bytes,
                    },
                    source: p.buffer.offset,
                });
            }
        }
        session.synchronize()?;
        for (name, pieces) in groups {
            session.restore_kv(&get(buffers, name)?, &pieces)?;
        }
        Ok(())
    }
    #[cfg(test)]
    fn restore_host_into(
        &self,
        session: &Session,
        buffers: &BTreeMap<String, Buffer>,
    ) -> Result<()> {
        self.restore_shared_into(session, buffers)?;
        grow_restored(
            session,
            buffers,
            self.pieces
                .iter()
                .map(|p| (p.name.as_str(), p.offset + p.buffer.bytes)),
        )?;
        self.restore_mutable_host_into(session, buffers)
    }
    fn restore_mutable_host_into(
        &self,
        session: &Session,
        buffers: &BTreeMap<String, Buffer>,
    ) -> Result<()> {
        let pieces = self
            .pieces
            .iter()
            .filter(|p| !p.buffer.bank.descriptor().is_some_and(|d| d.paged))
            .collect::<Vec<_>>();
        let banks = pieces
            .iter()
            .map(|p| p.buffer.bank.bytes.borrow())
            .collect::<Vec<_>>();
        let regions = pieces
            .iter()
            .zip(&banks)
            .map(|(p, bank)| {
                Ok((
                    get(buffers, &p.name)?,
                    p.offset,
                    &bank[p.buffer.offset..p.buffer.offset + p.buffer.bytes],
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if !regions.is_empty() {
            session.upload_regions(&regions)?.wait()?;
        }
        Ok(())
    }
}
impl Worker {
    pub(in crate::models::deepseek_v41) fn plan_host_snapshot(
        &self,
        id: usize,
        complete: bool,
        bank: u64,
        parent: Option<&HostSnapshot>,
    ) -> Result<BTreeMap<u64, usize>> {
        let (owner, layouts) = self.snapshot_state(id, complete)?;
        let (plan, shared) = HostSnapshot::plan_checkpoint(
            layouts.into_iter(),
            &owner.buffers,
            &owner.history,
            parent,
        )?;
        let mut banks = plan
            .pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect::<BTreeMap<_, _>>();
        if !plan.pending.is_empty() {
            banks.insert(
                kv::checkpoint_id(bank)?,
                plan.bytes.checked_add(255).ok_or("Host plan overflow")?,
            );
        }
        for piece in shared {
            banks.insert(piece.allocation.id, piece.allocation.used);
        }
        Ok(banks)
    }
    pub(in crate::models::deepseek_v41) fn save_host_snapshot(
        &self,
        id: usize,
        complete: bool,
        bank: u64,
        parent: Option<&HostSnapshot>,
        pool: &mut HostPool,
    ) -> Result<HostSnapshot> {
        let (owner, layouts) = self.snapshot_state(id, complete)?;
        HostSnapshot::capture_host(
            &self.session,
            layouts.into_iter(),
            &owner.buffers,
            &owner.history,
            bank,
            parent,
            pool,
        )
    }
    pub(in crate::models::deepseek_v41) fn restore_host_snapshot(
        &mut self,
        id: usize,
        snapshot: &HostSnapshot,
    ) -> Result<()> {
        self.reset_owner(id)?;
        {
            let runtime = self.batch.as_ref().ok_or("Missing batch runtime")?;
            snapshot.restore_shared_into(
                &self.session,
                &runtime.owners.get(&id).ok_or("Unknown owner")?.buffers,
            )?;
        }
        self.grow_owner_readonly(id, snapshot.history.len())?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        let owner = runtime.owners.get_mut(&id).ok_or("Unknown A2 owner")?;
        snapshot.restore_mutable_host_into(&self.session, &owner.buffers)?;
        owner.history.clone_from(&snapshot.history);
        runtime.bound_tables = None;
        Ok(())
    }
}

#[cfg(test)]
#[path = "host_tests.rs"]
mod tests;
