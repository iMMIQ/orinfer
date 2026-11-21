//! Stable virtual addresses with separately charged, demand-mapped HBM pages.
use super::*;
use crate::backend::kv::{self, Allocation as SharedAllocation, Mapping as Page, MappingOps};
use std::collections::BTreeMap;

#[repr(C)]
struct Location {
    id: u32,
    kind: i32,
}
#[repr(C)]
struct Properties {
    handle_type: i32,
    allocation_type: i32,
    mem_attr: i32,
    location: Location,
    reserved: u64,
}
#[derive(Clone, Copy)]
pub(super) struct Api {
    reserve: unsafe extern "C" fn(*mut Handle, usize, usize, Handle, u64) -> i32,
    release: unsafe extern "C" fn(Handle) -> i32,
    granularity: unsafe extern "C" fn(*const Properties, i32, *mut usize) -> i32,
    allocate: unsafe extern "C" fn(*mut Handle, usize, *const Properties, u64) -> i32,
    free: unsafe extern "C" fn(Handle) -> i32,
    map: unsafe extern "C" fn(Handle, usize, usize, Handle, u64) -> i32,
    unmap: unsafe extern "C" fn(Handle) -> i32,
}
impl Api {
    fn load(context: &Context) -> Result<Self> {
        if let Some(api) = context.vmm.get() {
            return Ok(api);
        }
        let library = &context.driver._library;
        // SAFETY: Signatures and repr(C) properties follow the installed acl_rt.h.
        let api = unsafe {
            Self {
                reserve: *library
                    .get(b"aclrtReserveMemAddress\0")
                    .map_err(|e| e.to_string())?,
                release: *library
                    .get(b"aclrtReleaseMemAddress\0")
                    .map_err(|e| e.to_string())?,
                granularity: *library
                    .get(b"aclrtMemGetAllocationGranularity\0")
                    .map_err(|e| e.to_string())?,
                allocate: *library
                    .get(b"aclrtMallocPhysical\0")
                    .map_err(|e| e.to_string())?,
                free: *library
                    .get(b"aclrtFreePhysical\0")
                    .map_err(|e| e.to_string())?,
                map: *library.get(b"aclrtMapMem\0").map_err(|e| e.to_string())?,
                unmap: *library.get(b"aclrtUnmapMem\0").map_err(|e| e.to_string())?,
            }
        };
        context.vmm.set(Some(api));
        Ok(api)
    }
}
fn properties(context: &Context) -> Properties {
    Properties {
        handle_type: 0,
        allocation_type: 0,
        mem_attr: 4,
        location: Location {
            id: context.device,
            kind: 1,
        },
        reserved: 0,
    }
}
pub(super) struct Reservation {
    context: Rc<Context>,
    api: Api,
    address: Handle,
    region: kv::Reservation,
}
impl Reservation {
    fn ops<'a>(&self, session: &'a Session) -> Ops<'a> {
        Ops {
            session,
            api: self.api,
            address: self.address,
        }
    }

    fn map_growth(&mut self, session: &Session, bytes: usize) -> Result<Option<(Handle, usize)>> {
        let _current = self.context.enter()?;
        let ops = self.ops(session);
        let result = self.region.grow(bytes, &ops, false);
        if self.region.poisoned {
            self.context.retire();
        }
        Ok(result?.map(|range| {
            // SAFETY: The common core returns a mapped span inside this retained reservation.
            (
                unsafe { self.address.cast::<u8>().add(range.offset).cast() },
                range.bytes,
            )
        }))
    }

    fn clear(&mut self) -> Result<()> {
        let _current = self.context.enter()?;
        self.region.clear(
            |page| {
                check(
                    // SAFETY: Prior accesses completed and this is an owned mapped extent.
                    unsafe { (self.api.unmap)(self.address.cast::<u8>().add(page.offset).cast()) },
                    "unmap ACL KV alias",
                )
            },
            |page| release_physical(&self.context, self.api, page.handle, page.bytes),
        )
    }
}

fn release_physical(context: &Context, api: Api, handle: u64, bytes: usize) -> Result<()> {
    check(
        // SAFETY: The physical handle is owned and its final owned mapping detached.
        unsafe { (api.free)(handle as usize as Handle) },
        "free ACL KV physical allocation",
    )?;
    context.device_bytes.set(context.device_bytes.get() - bytes);
    Ok(())
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if self.context.poisoned.get() {
            // A failed initialization fence may still have device writes in flight.
            std::mem::forget(Rc::clone(&self.context));
            return;
        }
        let result = self.clear().and_then(|()| {
            let _current = self.context.enter()?;
            check(
                // SAFETY: All owned mappings and physical handles were released above.
                unsafe { (self.api.release)(self.address) },
                "release ACL virtual address",
            )
        });
        if let Err(error) = result {
            self.context.retire();
            // A failed transition cannot prove backing is detached. Retain the context.
            std::mem::forget(Rc::clone(&self.context));
            eprintln!("Quarantined ACL virtual memory: {error}");
        }
    }
}
impl Session {
    /// Reserve address space only. Call grow before kernels or Graph capture touch it.
    pub(crate) fn reserve(&self, bytes: usize) -> Result<Buffer> {
        self.context.ready()?;
        if bytes == 0 {
            return Err("Empty ACL virtual reservation".into());
        }
        let api = Api::load(&self.context)?;
        let _current = self.context.enter()?;
        let mut grain = 0;
        check(
            // SAFETY: Properties describe this owned device and grain is a live output.
            unsafe { (api.granularity)(&properties(&self.context), 0, &mut grain) },
            "ACL physical allocation granularity",
        )?;
        let physical_capacity = kv::align(bytes, grain)?;
        let mut address = ptr::null_mut();
        check(
            // SAFETY: Nonzero aligned capacity, huge-page flag and writable address output.
            unsafe { (api.reserve)(&mut address, physical_capacity, 0, ptr::null_mut(), 1) },
            "reserve ACL virtual address",
        )?;
        let reservation = Reservation {
            context: Rc::clone(&self.context),
            api,
            address,
            region: kv::Reservation::new(physical_capacity, grain),
        };
        Ok(Buffer {
            offset: 0,
            bytes,
            allocation: Rc::new(Allocation {
                context: Rc::clone(&self.context),
                pointer: address,
                bytes,
                virtual_memory: Some(RefCell::new(reservation)),
            }),
        })
    }
    /// Prior readers complete before mapping changes; virtual addresses never change.
    #[cfg(test)]
    pub(crate) fn grow(&self, buffer: &Buffer, bytes: usize) -> Result<()> {
        self.make_kv_writable(&[(buffer.clone(), MemoryRange { offset: 0, bytes })])?;
        self.grow_many(&[(buffer.clone(), bytes)])
    }
    /// Map a validated growth set between two fences. All mappings precede
    /// initialization, and every new byte is zero before returning to a caller.
    pub(crate) fn grow_many(&self, buffers: &[(Buffer, usize)]) -> Result<()> {
        // Deduplicate aliases by native identity, but assign logical pages in
        // caller order. CPU allocation addresses differ across TP workers.
        let mut indices = BTreeMap::<usize, usize>::new();
        let mut growth = Vec::<(Buffer, usize)>::new();
        for (buffer, bytes) in buffers {
            self.require(buffer)?;
            MemoryRange {
                offset: 0,
                bytes: *bytes,
            }
            .within(buffer.bytes())?;
            if buffer.growth_bytes(*bytes)? == 0 {
                continue;
            }
            let end = buffer.offset + bytes;
            let index = *indices
                .entry(Rc::as_ptr(&buffer.allocation) as usize)
                .or_insert_with(|| {
                    growth.push((buffer.clone(), end));
                    growth.len() - 1
                });
            growth[index].1 = growth[index].1.max(end);
        }
        let additional = growth.iter().try_fold(0usize, |sum, (buffer, end)| {
            let memory = buffer.allocation.virtual_memory.as_ref().unwrap().borrow();
            sum.checked_add(memory.region.extent(*end)? - memory.region.mapped)
                .ok_or_else(|| "ACL demand growth extent overflow".to_string())
        })?;
        if additional == 0 {
            return Ok(());
        }
        // The prior fence can release completion-owned temporary buffers.
        // Check available capacity after those resources have been retired.
        self.synchronize()?;
        if additional > self.remaining_device_bytes() {
            return Err("ACL demand growth exceeds remaining session budget".into());
        }
        let mut spans = Vec::with_capacity(growth.len());
        for (buffer, end) in &growth {
            let mapped = buffer
                .allocation
                .virtual_memory
                .as_ref()
                .unwrap()
                .borrow_mut()
                .map_growth(self, *end);
            match mapped {
                Ok(Some(span)) => spans.push(span),
                Ok(None) => (),
                Err(error) => {
                    // A partially mapped batch has not been initialized. Retire
                    // it rather than expose those bytes as committed growth.
                    if !spans.is_empty() {
                        self.context.retire();
                    }
                    return Err(error);
                }
            }
        }
        let result = (|| {
            let _current = self.context.enter()?;
            for &(pointer, bytes) in &spans {
                check(
                    // SAFETY: Growth retains every mapped reservation through
                    // this compute-stream initialization and its final fence.
                    unsafe {
                        (self.context.api.memset)(pointer, bytes, 0, bytes, self.context.compute)
                    },
                    "initialize ACL physical pages",
                )?;
            }
            self.context.sync(self.context.compute)
        })();
        if result.is_err() {
            // A failed fence cannot prove initialization completed. The normal
            // reservation quarantine retains backing for any in-flight writes.
            self.context.retire();
        }
        result
    }
    pub(crate) fn release_pages(&self, buffer: &Buffer) -> Result<()> {
        self.require(buffer)?;
        if let Some(memory) = &buffer.allocation.virtual_memory {
            self.synchronize()?;
            if let Err(error) = memory.borrow_mut().clear() {
                self.context.retire();
                return Err(error);
            }
        }
        Ok(())
    }
}
impl Buffer {
    pub(crate) fn physical_extent(&self, bytes: usize) -> Result<usize> {
        MemoryRange { offset: 0, bytes }.within(self.bytes())?;
        self.allocation
            .virtual_memory
            .as_ref()
            .map_or(Ok(bytes), |memory| {
                memory.borrow().region.extent(self.offset + bytes)
            })
    }
    pub(crate) fn growth_bytes(&self, bytes: usize) -> Result<usize> {
        MemoryRange { offset: 0, bytes }.within(self.bytes())?;
        self.allocation
            .virtual_memory
            .as_ref()
            .map_or(Ok(0), |memory| {
                let memory = memory.borrow();
                Ok(memory
                    .region
                    .extent(self.offset + bytes)?
                    .saturating_sub(memory.region.mapped))
            })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod device_tests;

/// CANN resource operations for the shared KV mapping state machine.
struct Ops<'a> {
    session: &'a Session,
    api: Api,
    address: Handle,
}
impl MappingOps for Ops<'_> {
    fn allocate(&self, bytes: usize) -> Result<u64> {
        self.session.context.ready()?;
        if bytes > self.session.remaining_device_bytes() {
            return Err("KV COW exceeds remaining A2 budget".into());
        }
        let mut handle = ptr::null_mut();
        check(
            // SAFETY: Live context, physical extent and writable handle output.
            unsafe {
                (self.api.allocate)(&mut handle, bytes, &properties(&self.session.context), 0)
            },
            "allocate ACL KV COW",
        )?;
        self.session
            .context
            .device_bytes
            .set(self.session.allocated_bytes() + bytes);
        let handle = handle as usize as u64;
        Ok(handle)
    }
    fn release(&self, handle: u64, bytes: usize) -> Result<()> {
        release_physical(&self.session.context, self.api, handle, bytes)
    }
    fn map(&self, offset: usize, bytes: usize, handle: u64, _shared: bool) -> Result<()> {
        check(
            // SAFETY: The shared core validated the owned reservation and complete physical extent.
            unsafe {
                (self.api.map)(
                    self.address.cast::<u8>().add(offset).cast(),
                    bytes,
                    0,
                    handle as usize as Handle,
                    0,
                )
            },
            "map ACL KV allocation",
        )
    }
    fn unmap(&self, offset: usize, _bytes: usize) -> Result<()> {
        check(
            // SAFETY: Caller fenced prior users; offset is an existing mapped extent.
            unsafe { (self.api.unmap)(self.address.cast::<u8>().add(offset).cast()) },
            "unmap ACL KV allocation",
        )
    }
    fn zero(&self, offset: usize, bytes: usize) -> Result<()> {
        check(
            // SAFETY: New writable backing and live compute stream, retained through the batch fence.
            unsafe {
                (self.session.context.api.memset)(
                    self.address.cast::<u8>().add(offset).cast(),
                    bytes,
                    0,
                    bytes,
                    self.session.context.compute,
                )
            },
            "initialize ACL KV COW",
        )
    }
    fn copy(&self, offset: usize, source: &SharedAllocation, bytes: usize) -> Result<()> {
        source.owner.require(self.session.context.owner)?;
        check(
            // SAFETY: Valid immutable source prefix and new destination; common state retains source backing.
            unsafe {
                (self.session.context.api.memcpy)(
                    self.address.cast::<u8>().add(offset).cast(),
                    bytes,
                    source.address as usize as Handle,
                    bytes,
                    3,
                    self.session.context.compute,
                )
            },
            "copy ACL KV private tail",
        )
    }
    fn complete(&self) -> Result<bool> {
        Ok(false)
    }
    fn publish(&self, page: &Page, used: usize) -> Result<Rc<SharedAllocation>> {
        let canonical = self.session.reserve(page.bytes)?;
        {
            let mut memory = canonical
                .allocation
                .virtual_memory
                .as_ref()
                .unwrap()
                .borrow_mut();
            check(
                // SAFETY: New reserved VA, borrowed complete physical allocation. No extra HBM is allocated.
                unsafe {
                    (memory.api.map)(
                        memory.address,
                        page.bytes,
                        0,
                        page.handle as usize as Handle,
                        0,
                    )
                },
                "publish ACL KV alias",
            )?;
            memory.region.mappings.push(Page {
                id: page.id,
                offset: 0,
                bytes: page.bytes,
                handle: page.handle,
                mapped: true,
                shared: None,
            });
            memory.region.mapped = page.bytes;
        }
        Ok(Rc::new(SharedAllocation {
            owner: self.session.context.owner,
            id: page.id,
            address: canonical.pointer() as usize as u64,
            bytes: page.bytes,
            used,
            handle: Some(page.handle),
            native: Some(Rc::new(canonical)),
        }))
    }
}
impl Buffer {
    pub(crate) fn demand_mapped(&self) -> bool {
        self.offset == 0 && self.allocation.virtual_memory.is_some()
    }
    pub(crate) fn kv_preview(&self, bytes: usize) -> Result<Vec<kv::Published>> {
        MemoryRange { offset: 0, bytes }.within(self.bytes)?;
        if self.offset != 0 {
            return Err("Shared KV publication requires a root buffer".into());
        }
        let memory = self
            .allocation
            .virtual_memory
            .as_ref()
            .ok_or("KV buffer is not demand mapped")?
            .borrow();
        kv::preview(
            &memory.region.mappings,
            self.allocation.context.owner,
            self.pointer() as usize as u64,
            MemoryRange { offset: 0, bytes },
        )
    }
    pub(crate) fn kv_writable_bytes(&self, range: MemoryRange) -> Result<usize> {
        range.within(self.bytes)?;
        if let Some(memory) = &self.allocation.virtual_memory {
            kv::writable_bytes(
                &memory.borrow().region.mappings,
                MemoryRange {
                    offset: self.offset + range.offset,
                    bytes: range.bytes,
                },
            )
        } else {
            Ok(0)
        }
    }
}
impl Session {
    pub(crate) fn publish_kv(&self, buffer: &Buffer, bytes: usize) -> Result<Vec<kv::Published>> {
        self.require(buffer)?;
        buffer.kv_preview(bytes)?;
        let _current = self.context.enter()?;
        let mut memory = buffer
            .allocation
            .virtual_memory
            .as_ref()
            .unwrap()
            .borrow_mut();
        let ops = memory.ops(self);
        let result = kv::publish(
            &mut memory.region.mappings,
            MemoryRange { offset: 0, bytes },
            &ops,
        );
        if result.is_err() {
            self.context.retire();
        }
        result
    }
    /// One producer fence and one completion fence cover every private tail copy.
    pub(crate) fn make_kv_writable(&self, writes: &[(Buffer, MemoryRange)]) -> Result<()> {
        let mut needed = false;
        for (buffer, range) in writes {
            self.require(buffer)?;
            needed |= buffer.kv_writable_bytes(*range)? > 0;
        }
        if !needed {
            return Ok(());
        }
        self.synchronize()?;
        let _current = self.context.enter()?;
        let result = (|| {
            for (buffer, range) in writes {
                let Some(memory) = &buffer.allocation.virtual_memory else {
                    continue;
                };
                let mut memory = memory.borrow_mut();
                let ops = memory.ops(self);
                let kv::Reservation {
                    mappings, retained, ..
                } = &mut memory.region;
                kv::make_writable(
                    mappings,
                    retained,
                    MemoryRange {
                        offset: buffer.offset + range.offset,
                        bytes: range.bytes,
                    },
                    &ops,
                )?;
            }
            self.context.sync(self.context.compute)?;
            for (buffer, _) in writes {
                if let Some(memory) = &buffer.allocation.virtual_memory {
                    memory.borrow_mut().region.retained.clear();
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.context.retire();
        }
        result
    }
    pub(crate) fn restore_kv(
        &self,
        buffer: &Buffer,
        pieces: &[crate::backend::snapshot::Piece],
    ) -> Result<()> {
        self.require(buffer)?;
        if buffer.offset != 0 {
            return Err("Shared KV restore requires a root buffer".into());
        }
        let _current = self.context.enter()?;
        let mut memory = buffer
            .allocation
            .virtual_memory
            .as_ref()
            .ok_or("Missing KV reservation")?
            .borrow_mut();
        let mut end = 0;
        for piece in pieces {
            piece.allocation.owner.require(self.context.owner)?;
            let offset = kv::alias_offset(
                &piece.allocation,
                piece.range,
                piece.source,
                memory.region.granularity,
                memory.region.bytes,
            )
            .ok_or("Invalid A2 KV alias")?;
            if offset != end {
                return Err("Noncontiguous shared A2 KV mappings".into());
            }
            end += piece.allocation.bytes;
        }
        memory.clear()?;
        let ops = memory.ops(self);
        for piece in pieces {
            let offset = memory.region.mapped;
            let result = memory.region.map_shared(
                offset,
                piece.allocation.clone(),
                self.context.owner,
                &ops,
            );
            if memory.region.poisoned {
                self.context.retire();
            }
            result?;
        }
        Ok(())
    }
}

impl Session {
    pub(crate) fn snapshot_allocation(
        &self,
        buffer: &Buffer,
        id: u64,
    ) -> Result<Rc<SharedAllocation>> {
        self.require(buffer)?;
        Ok(Rc::new(SharedAllocation {
            owner: self.context.owner,
            id,
            address: buffer.pointer() as usize as u64,
            bytes: kv::align(buffer.bytes(), 2 << 20)?,
            used: buffer.bytes(),
            handle: None,
            native: Some(Rc::new(buffer.clone())),
        }))
    }
}

impl Session {
    /// Re-residency preserves the logical page identity. Native allocations and
    /// aliases are implementation details and do not consume new logical IDs.
    pub(crate) fn host_kv(
        &self,
        bank: &crate::backend::kv::host::HostBank,
    ) -> Result<Rc<SharedAllocation>> {
        bank.materialize(self.context.owner, |descriptor, bytes| {
            self.restore_host_kv(descriptor, bytes)
        })
    }
    fn restore_host_kv(
        &self,
        descriptor: crate::backend::kv::host::Descriptor,
        bytes: &[u8],
    ) -> Result<Rc<SharedAllocation>> {
        descriptor.owner.require(self.context.owner)?;
        if !descriptor.paged
            || bytes.len() != descriptor.used
            || descriptor.bytes > self.remaining_device_bytes()
        {
            return Err("Invalid or unreserved host KV materialization".into());
        }
        let canonical = self.reserve(descriptor.bytes)?;
        {
            let _current = self.context.enter()?;
            let mut memory = canonical
                .allocation
                .virtual_memory
                .as_ref()
                .unwrap()
                .borrow_mut();
            let mut handle = ptr::null_mut();
            check(
                // SAFETY: Valid physical capacity and live handle output.
                unsafe {
                    (memory.api.allocate)(
                        &mut handle,
                        descriptor.bytes,
                        &properties(&self.context),
                        0,
                    )
                },
                "restore CPU KV backing",
            )?;
            self.context
                .device_bytes
                .set(self.allocated_bytes() + descriptor.bytes);
            memory.region.mappings.push(Page {
                id: descriptor.id,
                offset: 0,
                bytes: descriptor.bytes,
                handle: handle as usize as u64,
                mapped: false,
                shared: None,
            });
            // SAFETY: New owned canonical VA and complete retained physical allocation.
            let result = check(
                // SAFETY: This owned context retains the checked native handle and allocation.
                unsafe { (memory.api.map)(memory.address, descriptor.bytes, 0, handle, 0) },
                "map restored CPU KV",
            );
            if result.is_err() {
                self.context.retire();
            }
            result?;
            memory.region.mappings[0].mapped = true;
            memory.region.mapped = descriptor.bytes;
        }
        self.fill(&canonical, 0)?;
        self.upload_regions(&[(canonical.clone(), 0, bytes)])?
            .wait()?;
        let allocation = Rc::new(SharedAllocation {
            owner: self.context.owner,
            id: descriptor.id,
            address: canonical.pointer() as usize as u64,
            bytes: descriptor.bytes,
            used: descriptor.used,
            handle: Some(
                canonical
                    .allocation
                    .virtual_memory
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .region
                    .mappings[0]
                    .handle,
            ),
            native: Some(Rc::new(canonical.clone())),
        });
        Ok(allocation)
    }
}
