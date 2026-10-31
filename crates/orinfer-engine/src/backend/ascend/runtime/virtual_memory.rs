//! Stable virtual addresses with separately charged, demand-mapped HBM pages.
use super::*;

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
fn aligned(bytes: usize, grain: usize) -> Result<usize> {
    if grain == 0 {
        return Err("Empty ACL physical allocation granularity".into());
    }
    Ok(bytes
        .checked_add(grain - 1)
        .ok_or("ACL virtual extent overflow")?
        / grain
        * grain)
}
struct Page {
    handle: Handle,
    offset: usize,
    bytes: usize,
    mapped: bool,
}
pub(super) struct Reservation {
    context: Rc<Context>,
    api: Api,
    address: Handle,
    bytes: usize,
    grain: usize,
    mapped: usize,
    pages: Vec<Page>,
}
impl Reservation {
    fn extent(&self, bytes: usize) -> Result<usize> {
        let extent = aligned(bytes, self.grain)?;
        if extent > self.bytes {
            return Err("ACL demand growth exceeds reservation".into());
        }
        Ok(extent)
    }
    fn grow(&mut self, bytes: usize) -> Result<()> {
        let extent = self.extent(bytes)?;
        if extent <= self.mapped {
            return Ok(());
        }
        self.context.ready()?;
        let additional = extent - self.mapped;
        if additional
            > self
                .context
                .limits
                .device_bytes
                .saturating_sub(self.context.device_bytes.get())
        {
            return Err("ACL demand growth exceeds remaining session budget".into());
        }
        let _current = self.context.enter()?;
        let mut handle = ptr::null_mut();
        check(
            // SAFETY: Current owned device, aligned extent and live handle output.
            unsafe { (self.api.allocate)(&mut handle, additional, &properties(&self.context), 0) },
            "allocate ACL physical pages",
        )?;
        self.context
            .device_bytes
            .set(self.context.device_bytes.get() + additional);
        self.pages.push(Page {
            handle,
            offset: self.mapped,
            bytes: additional,
            mapped: false,
        });
        let page = self.pages.last_mut().unwrap();
        // SAFETY: Offset is inside this retained virtual reservation; allocation has the same extent.
        let pointer = unsafe { self.address.cast::<u8>().add(page.offset).cast() };
        let result = (|| {
            check(
                // SAFETY: Retained aligned reservation and physical handle have identical extents.
                unsafe { (self.api.map)(pointer, additional, 0, handle, 0) },
                "map ACL physical pages",
            )?;
            page.mapped = true;
            check(
                // SAFETY: Newly mapped writable pages remain owned until this stream is fenced.
                unsafe {
                    (self.context.api.memset)(
                        pointer,
                        additional,
                        0,
                        additional,
                        self.context.compute,
                    )
                },
                "initialize ACL physical pages",
            )?;
            self.context.sync(self.context.compute)
        })();
        if result.is_err() {
            self.context.retire();
        }
        result?;
        self.mapped = extent;
        Ok(())
    }
    fn clear(&mut self) -> Result<()> {
        let _current = self.context.enter()?;
        while let Some(page) = self.pages.last_mut() {
            // SAFETY: Every recorded page is wholly inside this retained reservation.
            let pointer = unsafe { self.address.cast::<u8>().add(page.offset).cast() };
            if page.mapped {
                check(
                    // SAFETY: Prior users are fenced, or the final Buffer/Graph owner has dropped.
                    unsafe { (self.api.unmap)(pointer) },
                    "unmap ACL physical pages",
                )?;
                page.mapped = false;
            }
            check(
                // SAFETY: The owned physical handle has no remaining virtual mapping.
                unsafe { (self.api.free)(page.handle) },
                "free ACL physical pages",
            )?;
            self.context
                .device_bytes
                .set(self.context.device_bytes.get() - page.bytes);
            self.pages.pop();
        }
        self.mapped = 0;
        Ok(())
    }
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
        let physical_capacity = aligned(bytes, grain)?;
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
            bytes: physical_capacity,
            grain,
            mapped: 0,
            pages: vec![],
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
    pub(crate) fn grow(&self, buffer: &Buffer, bytes: usize) -> Result<()> {
        self.require(buffer)?;
        MemoryRange { offset: 0, bytes }.within(buffer.bytes())?;
        if let Some(memory) = &buffer.allocation.virtual_memory
            && buffer.growth_bytes(bytes)? != 0
        {
            self.synchronize()?;
            memory.borrow_mut().grow(buffer.offset + bytes)?;
        }
        Ok(())
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
                memory.borrow().extent(self.offset + bytes)
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
                    .extent(self.offset + bytes)?
                    .saturating_sub(memory.mapped))
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::ascend::tests::Fixture;

    #[test]
    fn demand_growth_retains_addresses_contents_and_completion_ownership() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let grain = 2 << 20;
        let buffer = session.reserve(grain * 8 - 19).unwrap();
        let address = buffer.pointer();
        assert_eq!(session.allocated_bytes(), 0);
        assert_eq!(buffer.growth_bytes(17).unwrap(), grain);
        session.grow(&buffer, 17).unwrap();
        assert_eq!(session.allocated_bytes(), grain);
        session
            .upload(&buffer, 0, &[37; 17])
            .unwrap()
            .wait()
            .unwrap();
        session.grow(&buffer, grain + 3).unwrap();
        assert_eq!(session.allocated_bytes(), grain * 2);
        assert_eq!(address, buffer.pointer());
        assert_eq!(session.download(&buffer, 0, 17).unwrap(), vec![37; 17]);
        assert_eq!(session.download(&buffer, grain, 3).unwrap(), vec![0; 3]);
        assert!(session.grow(&buffer, buffer.bytes() + 1).is_err());
        session.release_pages(&buffer).unwrap();
        assert_eq!(session.allocated_bytes(), 0);
        session.grow(&buffer, 19).unwrap();
        assert_eq!(session.download(&buffer, 0, 19).unwrap(), vec![0; 19]);
        let ticket = session.upload(&buffer, 0, &[91; 19]).unwrap();
        drop(buffer);
        assert_eq!(session.allocated_bytes(), grain);
        ticket.wait().unwrap();
        drop(ticket);
        assert_eq!(session.allocated_bytes(), 0);
        drop(session);
        fixture.assert_clean();
    }

    #[test]
    fn demand_growth_failure_preserves_mapped_history_and_budget() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let grain = 2 << 20;
        let buffer = session.reserve(grain * 4).unwrap();
        session.grow(&buffer, 1).unwrap();
        session.upload(&buffer, 0, &[73]).unwrap().wait().unwrap();
        fixture.failure(100);
        assert!(session.grow(&buffer, grain + 1).is_err());
        assert_eq!(session.allocated_bytes(), grain);
        fixture.failure(0);
        assert_eq!(session.download(&buffer, 0, 1).unwrap(), vec![73]);
        session.grow(&buffer, grain + 1).unwrap();
        drop(buffer);
        drop(session);
        fixture.assert_clean();
    }
}
