//! CPU storage for the same state-span plan as device snapshots.
use super::*;
impl StateBytes for HostBuffer {
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn host_capacity(&self) -> usize {
        self.bank.bytes.borrow().len()
    }
}
impl HostSnapshot {
    /// Device views are copied, including a parent's mutable ranges when it is
    /// evicted after its child. Shared CPU banks remain charged exactly once.
    pub(in crate::backend::ascend::model) fn offload(
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
    pub(in crate::backend::ascend::model) fn plan_host_snapshot(
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
    pub(in crate::backend::ascend::model) fn save_host_snapshot(
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
    pub(in crate::backend::ascend::model) fn restore_host_snapshot(
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
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;
    fn shared_page_lifecycle(session: &Session) {
        let grain = 2 << 20;
        let source = session.reserve(grain * 2).unwrap();
        session.grow_many(&[(source.clone(), 32)]).unwrap();
        session
            .upload(&source, 0, &[73; 32])
            .unwrap()
            .wait()
            .unwrap();
        let layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 1,
            tokens_per_row: 1,
            window: None,
            circular: false,
            append_only: true,
        };
        let buffers = BTreeMap::<String, _>::from([("kv".into(), source.clone())]);
        let before = session.allocated_bytes();
        let (allocation, range, offset) = session.publish_kv(&source, 17).unwrap().remove(0);
        let saved = Snapshot {
            history: vec![1; 17],
            pieces: vec![Piece {
                name: layout.buffer,
                offset: range.offset,
                bank_id: allocation.id,
                bank_bytes: allocation.bytes,
                buffer: DeviceBuffer {
                    allocation,
                    source: offset,
                    bytes: range.bytes,
                },
            }],
        };
        assert_eq!(
            session.allocated_bytes(),
            before,
            "publication allocates no KV backing"
        );
        let original = saved.pieces[0].bank_id;
        let target = session.reserve(grain * 2).unwrap();
        let target_map = BTreeMap::from([("kv".into(), target.clone())]);
        let address = target.pointer();
        saved.restore_into(session, &target_map).unwrap();
        assert_eq!(
            session.allocated_bytes(),
            before,
            "restoration aliases the same physical KV"
        );
        assert_eq!(session.download(&target, 0, 17).unwrap(), [73; 17]);
        session.upload(&target, 17, &[91]).unwrap().wait().unwrap();
        assert_eq!(target.pointer(), address);
        assert_eq!(
            session.allocated_bytes(),
            before + grain,
            "tail writes allocate one private page"
        );
        assert_eq!(session.download(&source, 0, 32).unwrap(), [73; 32]);
        assert_eq!(session.download(&target, 17, 2).unwrap(), [91, 0]);
        let mut pool = HostPool::default();
        let host = HostSnapshot::offload(session, &saved, &mut pool).unwrap();
        assert_eq!(
            host.allocations()[&original],
            17,
            "CPU stores valid bytes, not physical padding"
        );
        drop((saved, buffers, source));
        assert_eq!(
            session.allocated_bytes(),
            grain,
            "CPU cache does not pin device backing"
        );
        let second = session.reserve(grain * 2).unwrap();
        let second_map = BTreeMap::from([("kv".into(), second.clone())]);
        host.restore_host_into(session, &second_map).unwrap();
        assert_eq!(session.allocated_bytes(), grain * 2);
        assert_eq!(second.kv_preview(17).unwrap()[0].0.id, original);
        assert_eq!(session.download(&second, 0, 17).unwrap(), [73; 17]);
        let third = session.reserve(grain * 2).unwrap();
        let third_map = BTreeMap::from([("kv".into(), third.clone())]);
        host.restore_host_into(session, &third_map).unwrap();
        assert_eq!(
            session.allocated_bytes(),
            grain * 2,
            "second CPU hit reuses resident backing"
        );
        session.upload(&second, 0, &[11]).unwrap().wait().unwrap();
        assert_eq!(session.download(&third, 0, 1).unwrap(), [73]);
        assert_eq!(session.download(&second, 0, 1).unwrap(), [11]);
        drop((
            host, pool, target_map, target, second_map, second, third_map, third,
        ));
        assert_eq!(
            session.allocated_bytes(),
            0,
            "all references release physical memory exactly once"
        );
    }
    fn compact_checkpoint_lifecycle(session: &Session) {
        let grain = 2 << 20;
        let source = session.reserve(grain * 2).unwrap();
        let ring = session.allocate(64).unwrap();
        let buffers = BTreeMap::from([
            ("kv".to_owned(), source.clone()),
            ("ring".to_owned(), ring.clone()),
        ]);
        let layouts = [
            StateLayout {
                buffer: "kv".into(),
                row_bytes: 1,
                tokens_per_row: 1,
                window: None,
                circular: false,
                append_only: true,
            },
            StateLayout {
                buffer: "ring".into(),
                row_bytes: 64,
                tokens_per_row: 0,
                window: None,
                circular: false,
                append_only: false,
            },
        ];
        let mut pool = HostPool::default();
        for (i, n) in [2048, 2049, grain + 17].into_iter().enumerate() {
            let value = 71 + i as u8;
            session.grow_many(&[(source.clone(), n + 1)]).unwrap();
            session
                .upload(&source, n - 17, &[value; 17])
                .unwrap()
                .wait()
                .unwrap();
            session.fill(&ring, value).unwrap().wait().unwrap();
            let history = vec![1; n];
            let plan = Snapshot::plan_device(layouts.iter(), &buffers, &history, None).unwrap();
            assert_eq!(plan.pieces.len(), n / grain);
            let expected = plan.allocations(i as u64 + 1).unwrap();
            let private_bytes = plan.bytes + 255;
            drop(plan);
            let before = session.allocated_bytes();
            let saved = Snapshot::capture(
                session,
                layouts.iter(),
                &buffers,
                &history,
                i as u64 + 1,
                None,
            )
            .unwrap();
            assert_eq!(saved.allocations(), expected);
            assert_eq!(
                session.allocated_bytes(),
                before + private_bytes,
                "all private state and partial KV tails share one compact endpoint bank"
            );
            assert_eq!(
                source
                    .kv_writable_bytes(orinfer_device::MemoryRange {
                        offset: n,
                        bytes: 1
                    })
                    .unwrap(),
                0,
                "checkpoint publication must not freeze the writable tail"
            );
            let after = session.allocated_bytes();
            session
                .upload(&source, n, &[value + 1])
                .unwrap()
                .wait()
                .unwrap();
            session
                .upload(&source, n - 1, &[value + 2])
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(
                session.allocated_bytes(),
                after,
                "continued tail writes need no COW allocation"
            );
            let target = session.reserve(grain * 2).unwrap();
            let actual = BTreeMap::from([
                ("kv".to_owned(), target.clone()),
                ("ring".to_owned(), session.allocate(64).unwrap()),
            ]);
            let before = session.allocated_bytes();
            saved.restore_into(session, &actual).unwrap();
            assert_eq!(
                session.allocated_bytes(),
                before + grain,
                "full KV pages alias; only the compact tail gets private backing"
            );
            assert_eq!(session.download(&target, n - 17, 17).unwrap(), [value; 17]);
            assert_eq!(session.download(&target, n, 1).unwrap(), [0]);
            assert_eq!(
                session.download(&actual["ring"], 0, 64).unwrap(),
                [value; 64]
            );
            let host = HostSnapshot::offload(session, &saved, &mut pool).unwrap();
            assert!(
                host.allocations().values().sum::<usize>() <= n + 1024,
                "CPU backing stores valid full pages and packed state, not physical tail padding"
            );
            let before = session.allocated_bytes();
            let direct = HostSnapshot::capture_host(
                session,
                layouts.iter(),
                &buffers,
                &history,
                i as u64 + 100,
                None,
                &mut pool,
            )
            .unwrap();
            assert_eq!(
                session.allocated_bytes(),
                before,
                "direct CPU capture needs no device endpoint bank"
            );
            direct.restore_host_into(session, &actual).unwrap();
            assert_eq!(session.download(&target, n - 1, 1).unwrap(), [value + 2]);
            host.restore_host_into(session, &actual).unwrap();
            assert_eq!(
                session.download(&target, n - 17, 17).unwrap(),
                [value; 17],
                "CPU and device branches retain the same immutable checkpoint values"
            );
        }
        drop((pool, buffers, source, ring));
        assert_eq!(session.allocated_bytes(), 0);
    }
    #[test]
    fn compact_checkpoint_keeps_full_pages_shared_and_tails_writable() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        compact_checkpoint_lifecycle(&session);
        drop(session);
        fixture.assert_clean();
    }
    #[test]
    fn shared_pages_cow_offload_and_resident_reuse() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        shared_page_lifecycle(&session);
        drop(session);
        fixture.assert_clean();
    }
    #[test]
    #[ignore = "Runs shared KV lifecycle on the physical A2 device"]
    fn shared_pages_cow_offload_and_resident_reuse_on_device() {
        use std::{fs::OpenOptions, os::fd::AsRawFd};
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open("artifacts/gpu-experiment.lock")
            .unwrap();
        assert_eq!(
            // SAFETY: The retained file owns the exclusive hardware lock until session destruction.
            unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
            0
        );
        let session = Session::new(
            0,
            crate::backend::ascend::runtime::Limits {
                device_bytes: 64 << 20,
                ..Default::default()
            },
        )
        .unwrap();
        shared_page_lifecycle(&session);
        compact_checkpoint_lifecycle(&session);
    }
    #[test]
    fn host_shared_rows_restore_parent_after_child_offload_and_direct_append() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let layouts = [
            StateLayout {
                buffer: "kv".into(),
                row_bytes: 2,
                tokens_per_row: 2,
                window: None,
                circular: false,
                append_only: true,
            },
            StateLayout {
                buffer: "swa".into(),
                row_bytes: 2,
                tokens_per_row: 1,
                window: Some(3),
                circular: true,
                append_only: false,
            },
            StateLayout {
                buffer: "scale_index_c2_mtp".into(),
                row_bytes: 2,
                tokens_per_row: 0,
                window: None,
                circular: false,
                append_only: false,
            },
        ];
        let buffers = layouts
            .iter()
            .map(|l| (l.buffer.clone(), session.allocate(64).unwrap()))
            .collect::<BTreeMap<_, _>>();
        session
            .upload_inputs(
                &buffers
                    .values()
                    .map(|b| (b.clone(), vec![7; 64]))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .wait()
            .unwrap();
        let parent =
            Snapshot::capture(&session, layouts.iter(), &buffers, &[1; 3], 1, None).unwrap();
        session
            .upload_inputs(
                &buffers
                    .iter()
                    .map(|(name, b)| {
                        (
                            b.clone(),
                            if name == "kv" {
                                vec![7; 64]
                            } else {
                                vec![19; 64]
                            },
                        )
                    })
                    .collect::<Vec<_>>(),
            )
            .unwrap()
            .wait()
            .unwrap();
        let child = Snapshot::capture(
            &session,
            layouts.iter(),
            &buffers,
            &[1; 9],
            2,
            Some(&parent),
        )
        .unwrap();
        let mut pool = HostPool::default();
        let host_child = HostSnapshot::offload(&session, &child, &mut pool).unwrap();
        fixture.failure(4); // Any unnecessary DMA event submission would fail.
        let repeated = HostSnapshot::offload(&session, &child, &mut pool).unwrap();
        fixture.failure(0);
        assert_eq!(repeated.allocations(), host_child.allocations());
        drop(repeated);
        let host_parent = HostSnapshot::offload(&session, &parent, &mut pool).unwrap();
        let cb = &host_child
            .pieces
            .iter()
            .find(|p| p.bank_id == kv::checkpoint_id(1).unwrap())
            .unwrap()
            .buffer
            .bank;
        let pb = &host_parent
            .pieces
            .iter()
            .find(|p| p.bank_id == kv::checkpoint_id(1).unwrap())
            .unwrap()
            .buffer
            .bank;
        assert!(Rc::ptr_eq(cb, pb), "shared KV must have one CPU allocation");
        drop((parent, child));
        let actual = layouts
            .iter()
            .map(|l| (l.buffer.clone(), session.allocate(64).unwrap()))
            .collect::<BTreeMap<_, _>>();
        host_parent.restore_host_into(&session, &actual).unwrap();
        assert_eq!(
            session
                .download(&actual["scale_index_c2_mtp"], 0, 64)
                .unwrap(),
            [7; 64],
            "late parent offload must fill mutable ranges absent from its child"
        );
        host_child.restore_host_into(&session, &actual).unwrap();
        assert_eq!(
            session
                .download(&actual["scale_index_c2_mtp"], 0, 64)
                .unwrap(),
            [19; 64]
        );
        session.fill(&buffers["scale_index_c2_mtp"], 31).unwrap();
        let before = session.allocated_bytes();
        let direct = HostSnapshot::capture_host(
            &session,
            layouts.iter(),
            &buffers,
            &[1; 17],
            3,
            Some(&host_child),
            &mut pool,
        )
        .unwrap();
        assert_eq!(
            session.allocated_bytes(),
            before,
            "direct CPU capture must not allocate a temporary device snapshot"
        );
        let plan =
            HostSnapshot::plan(layouts.iter(), &buffers, &[1; 17], Some(&host_child)).unwrap();
        let mut planned = plan
            .pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect::<BTreeMap<_, _>>();
        planned.insert(kv::checkpoint_id(3).unwrap(), plan.bytes + 255);
        assert_eq!(direct.allocations(), planned);
        drop(plan);
        drop((host_parent, host_child, pool));
        direct.restore_host_into(&session, &actual).unwrap();
        assert_eq!(session.download(&actual["kv"], 0, 16).unwrap(), [7; 16]);
        assert_eq!(
            session
                .download(&actual["scale_index_c2_mtp"], 0, 64)
                .unwrap(),
            [31; 64]
        );
        assert!(HostSnapshot::plan(layouts.iter(), &buffers, &[2; 17], Some(&direct)).is_err());
        drop((direct, actual, buffers, session));
        fixture.assert_clean();
    }
}
