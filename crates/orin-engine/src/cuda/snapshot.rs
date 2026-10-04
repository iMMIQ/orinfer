//! Immutable GPU snapshots; graph-bound sequence addresses never change.
use super::executor::Executor;
use super::{Result, check};
use std::collections::BTreeMap;

pub(crate) struct Snapshot {
    pub bytes: usize,
    address: u64,
    ranges: Vec<(String, usize, usize)>,
}

impl Executor {
    pub(crate) fn snapshot_size(ranges: &BTreeMap<String, usize>) -> Result<usize> {
        ranges.values().try_fold(0usize, |offset, bytes| {
            offset
                .checked_add(*bytes)
                .and_then(|n| n.checked_add(255))
                .map(|n| n & !255)
                .ok_or("Snapshot size overflow".into())
        })
    }

    pub(crate) fn snapshot(&mut self, ranges: BTreeMap<String, usize>) -> Result<Option<Snapshot>> {
        self.sync()?;
        let bytes = Self::snapshot_size(&ranges)?;
        for (name, count) in &ranges {
            if *count > self.sizes.get(name).copied().unwrap_or(0)
                || self
                    .session
                    .virtual_buffers
                    .borrow()
                    .get(name)
                    .is_some_and(|v| *count > v.mapped)
            {
                return Err(format!("{name}: snapshot outside resident allocation"));
            }
        }
        let (mut free, mut total, mut address) = (0, 0, 0);
        // SAFETY: Live context; this thread owns all source buffers. The new
        // allocation immediately joins Session ownership, including error paths.
        unsafe {
            check(
                (self.session.driver.memory_info)(&mut free, &mut total),
                "snapshot memory",
            )?;
            if bytes == 0 || bytes > free.saturating_sub(64 * 1024 * 1024) {
                return Ok(None);
            }
            let status = (self.session.driver.alloc)(&mut address, bytes);
            if status == 2 {
                return Ok(None);
            } // Cache admission must not cause OOM.
            check(status, "allocate prefix snapshot")?;
        }
        self.session.buffers.push(address);
        let mut snapshot = Snapshot {
            address,
            bytes,
            ranges: vec![],
        };
        let mut offset = 0;
        for (name, count) in ranges {
            if count != 0 {
                // SAFETY: Both validated allocations cover count bytes. Copies
                // share the executor stream and complete before publication.
                unsafe {
                    check(
                        (self.session.driver.copy)(
                            address + offset as u64,
                            self.pointers[&name],
                            count,
                            self.session.stream,
                        ),
                        "save prefix state",
                    )?;
                }
                snapshot.ranges.push((name, offset, count));
            }
            offset = (offset + count + 255) & !255;
        }
        self.sync()?;
        Ok(Some(snapshot))
    }

    pub(crate) fn restore_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        self.sync()?;
        for (name, _, count) in &snapshot.ranges {
            if let Some(v) = self.session.virtual_buffers.borrow_mut().get_mut(name) {
                v.grow(&self.session, count.div_ceil(v.stride))?;
            }
        }
        for (name, offset, count) in &snapshot.ranges {
            // SAFETY: Snapshot ranges were checked on creation. Mutable state
            // uses its original graph-bound addresses; mappings grew above.
            unsafe {
                check(
                    (self.session.driver.copy)(
                        self.pointers[name],
                        snapshot.address + *offset as u64,
                        *count,
                        self.session.stream,
                    ),
                    "restore prefix state",
                )?;
            }
        }
        self.sync()
    }

    pub(crate) fn release_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        self.sync()?;
        let index = self
            .session
            .buffers
            .iter()
            .position(|&p| p == snapshot.address)
            .ok_or("Unowned prefix snapshot")?;
        // SAFETY: Completed stream; allocation remains Session-owned until free succeeds.
        unsafe {
            check(
                (self.session.driver.free)(snapshot.address),
                "release prefix snapshot",
            )?;
        }
        self.session.buffers.swap_remove(index);
        Ok(())
    }
}
