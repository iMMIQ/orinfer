//! CUDA VMM ownership: graph pointers stay fixed while physical slabs grow.
use super::{Driver, Result, Session, check, sys};
use crate::backend::kv::{self, Mapping as Slab, MappingOps};
use std::{collections::BTreeMap, rc::Rc};

use sys::{CUmemAccessDesc as AccessDesc, CUmemAllocationProp as AllocationProp};

pub(crate) struct Reservation {
    pub address: u64,
    pub stride: usize,
    region: kv::Reservation,
}
impl std::ops::Deref for Reservation {
    type Target = kv::Reservation;

    fn deref(&self) -> &Self::Target {
        &self.region
    }
}
impl Reservation {
    /// Map one sparse weight page. Payload initialization belongs to the uploader.
    pub(crate) fn map_page(&mut self, s: &Session, offset: usize) -> Result<()> {
        self.region.map_sparse(
            offset,
            &Ops {
                session: s,
                address: self.address,
            },
        )
    }
    /// Caller must finish all consumers before releasing a sparse page.
    pub(crate) fn unmap_page(&mut self, s: &Session, offset: usize) -> Result<()> {
        let index = self
            .region
            .mappings
            .iter()
            .position(|p| p.offset == offset)
            .ok_or("Unknown expert page")?;
        let slab = &mut self.region.mappings[index];
        if slab.mapped {
            unsafe_unmap(&s.driver, self.address, slab)?;
            slab.mapped = false;
        }
        unsafe_release(&s.driver, slab.handle)?;
        self.region.mappings.swap_remove(index);
        Ok(())
    }
    pub(crate) fn physical_bytes(&self) -> usize {
        self.region
            .mappings
            .iter()
            .filter(|p| p.shared.is_none())
            .map(|p| p.bytes)
            .sum()
    }
    pub(crate) fn reserve(s: &Session, bytes: usize, stride: usize) -> Result<Self> {
        let prop = properties(s.device);
        let mut granularity = 0;
        let mut address = 0;
        // SAFETY: repr(C) properties match CUDA 12.6 cuda.h; pointers are live.
        unsafe {
            check(
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "VMM granularity",
            )?;
            let bytes = round_up(bytes, granularity)?;
            check(
                sys::cuMemAddressReserve(&mut address, bytes, granularity, 0, 0),
                "reserve KV address space",
            )?;
            Ok(Self {
                address,
                stride,
                region: kv::Reservation::new(bytes, granularity),
            })
        }
    }
    pub(crate) fn extent(&self, tokens: usize) -> Result<usize> {
        let extent = self.required_extent(tokens)?;
        if extent > self.bytes {
            return Err("KV growth exceeds reserved context".into());
        }
        Ok(extent)
    }
    pub(crate) fn required_extent(&self, tokens: usize) -> Result<usize> {
        let raw = crate::backend::growth::TokenLayout {
            row_bytes: self.stride,
            tokens_per_row: 1,
            capacity_bytes: self.bytes,
        }
        .required_bytes(tokens)?;
        round_up(raw, self.granularity)
    }
    pub(crate) fn grow(&mut self, s: &super::Session, tokens: usize) -> Result<()> {
        self.grow_bytes(s, self.extent(tokens)?)
    }
    pub(crate) fn grow_bytes(&mut self, s: &Session, bytes: usize) -> Result<()> {
        self.region.grow(
            bytes,
            &Ops {
                session: s,
                address: self.address,
            },
            true,
        )?;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn allocation_handle(&self) -> Result<u64> {
        match self.region.mappings.as_slice() {
            [slab] if slab.offset == 0 && slab.bytes == self.mapped && slab.shared.is_none() => {
                Ok(slab.handle)
            }
            _ => Err("Shared allocation must have one complete owned mapping".into()),
        }
    }
    pub(crate) fn has_shared_slabs(&self) -> bool {
        self.region.mappings.iter().any(|p| p.shared.is_some())
    }
    /// Append a whole immutable allocation; CUDA currently requires handle offset zero.
    pub(crate) fn map_shared(
        &mut self,
        s: &Session,
        offset: usize,
        allocation: Rc<super::snapshot::Allocation>,
    ) -> Result<()> {
        self.region.map_shared(
            offset,
            allocation,
            s.owner,
            &Ops {
                session: s,
                address: self.address,
            },
        )
    }
    pub(crate) fn writable_bytes(&self, start: usize, bytes: usize) -> Result<usize> {
        kv::writable_bytes(
            &self.region.mappings,
            crate::backend::snapshot::Range {
                offset: start,
                bytes,
            },
        )
    }
    pub(crate) fn make_writable(&mut self, s: &Session, start: usize, bytes: usize) -> Result<()> {
        if self.poisoned {
            return Err("KV reservation is quarantined".into());
        }
        let result = kv::make_writable(
            &mut self.region.mappings,
            &mut self.region.retained,
            crate::backend::snapshot::Range {
                offset: start,
                bytes,
            },
            &Ops {
                session: s,
                address: self.address,
            },
        );
        if result.is_err() {
            self.region.poisoned = true;
        }
        result
    }
    pub(crate) fn preview(
        &self,
        owner: orinfer_model_sdk::device::DeviceOwner,
        range: crate::backend::snapshot::Range,
    ) -> Result<
        Vec<(
            Rc<crate::backend::snapshot::Allocation>,
            crate::backend::snapshot::Range,
            usize,
        )>,
    > {
        kv::preview(&self.region.mappings, owner, self.address, range)
    }
    pub(crate) fn publish(
        &mut self,
        s: &Session,
        range: crate::backend::snapshot::Range,
    ) -> Result<
        Vec<(
            Rc<crate::backend::snapshot::Allocation>,
            crate::backend::snapshot::Range,
            usize,
        )>,
    > {
        let result = kv::publish(
            &mut self.region.mappings,
            range,
            &Ops {
                session: s,
                address: self.address,
            },
        );
        if result.is_err() {
            self.region.poisoned = true;
        }
        result
    }
    pub(crate) fn release_slabs(&mut self, driver: &Driver) -> Result<()> {
        self.region.clear(
            |slab| unsafe_unmap(driver, self.address, slab),
            |slab| unsafe_release(driver, slab.handle),
        )
    }
    pub(crate) fn has_slabs(&self) -> bool {
        !self.region.mappings.is_empty()
    }
}
#[cfg(test)]
fn overlaps(start: usize, end: usize, offset: usize, bytes: usize) -> bool {
    start < end && start < offset.saturating_add(bytes) && offset < end
}
fn unsafe_unmap(_driver: &Driver, address: u64, slab: &Slab) -> Result<()> {
    // SAFETY: The caller owns this mapped extent and synchronized its stream.
    unsafe {
        check(
            sys::cuMemUnmap(address + slab.offset as u64, slab.bytes),
            "unmap KV slab",
        )
    }
}
fn unsafe_release(_driver: &Driver, handle: u64) -> Result<()> {
    // SAFETY: The caller owns the live allocation handle, now unmapped.
    unsafe { check(sys::cuMemRelease(handle), "release KV slab") }
}
#[cfg(test)]
fn release_owned(
    slabs: &mut Vec<Slab>,
    unmap: impl FnMut(&Slab) -> Result<()>,
    mut release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    kv::release_mappings(slabs, &mut vec![], unmap, |page| release(page.handle))
}
fn properties(device: i32) -> AllocationProp {
    AllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType(0),
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: device,
        },
        win32HandleMetaData: std::ptr::null_mut(),
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    }
}
fn round_up(bytes: usize, granularity: usize) -> Result<usize> {
    if !granularity.is_power_of_two() {
        return Err("Invalid CUDA VMM granularity".into());
    }
    kv::align(bytes, granularity)
}
pub(crate) type Reservations = BTreeMap<String, Reservation>;

/// CUDA calls only; allocation identities and COW/publication rules live in kv.
struct Ops<'a> {
    session: &'a Session,
    address: u64,
}
impl MappingOps for Ops<'_> {
    fn allocate(&self, bytes: usize) -> Result<u64> {
        let mut handle = 0;
        // SAFETY: Current session and valid physical allocation output.
        unsafe {
            check(
                sys::cuMemCreate(&mut handle, bytes, &properties(self.session.device), 0),
                "create KV physical allocation",
            )?;
        }
        Ok(handle)
    }
    fn release(&self, handle: u64, _bytes: usize) -> Result<()> {
        unsafe_release(&self.session.driver, handle)
    }
    fn map(&self, offset: usize, bytes: usize, handle: u64, _shared: bool) -> Result<()> {
        let address = self
            .address
            .checked_add(offset as u64)
            .ok_or("KV address overflow")?;
        // SAFETY: Caller validated and owns the virtual extent and physical handle.
        unsafe {
            check(
                sys::cuMemMap(address, bytes, 0, handle, 0),
                "map KV allocation",
            )
        }
    }
    fn access(&self, offset: usize, bytes: usize, shared: bool) -> Result<()> {
        let access = AccessDesc {
            location: properties(self.session.device).location,
            flags: if shared {
                sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ
            } else {
                sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            },
        };
        // SAFETY: This owned mapping has the validated extent.
        unsafe {
            check(
                sys::cuMemSetAccess(self.address + offset as u64, bytes, &access, 1),
                "KV access",
            )
        }
    }
    fn unmap(&self, offset: usize, bytes: usize) -> Result<()> {
        // SAFETY: Caller fenced all previous users of this live mapping.
        unsafe {
            check(
                sys::cuMemUnmap(self.address + offset as u64, bytes),
                "unmap KV allocation",
            )
        }
    }
    fn zero(&self, offset: usize, bytes: usize) -> Result<()> {
        // SAFETY: New writable mapping retained through the completion fence.
        unsafe {
            check(
                sys::cuMemsetD8Async(self.address + offset as u64, 0, bytes, self.session.stream),
                "initialize KV COW",
            )
        }
    }
    fn copy(
        &self,
        offset: usize,
        source: &crate::backend::snapshot::Allocation,
        bytes: usize,
    ) -> Result<()> {
        source.owner.require(self.session.owner)?;
        // SAFETY: Valid source prefix, new destination and retained backing.
        unsafe {
            check(
                sys::cuMemcpyDtoDAsync_v2(
                    self.address + offset as u64,
                    source.address,
                    bytes,
                    self.session.stream,
                ),
                "copy KV private tail",
            )
        }
    }
    fn complete(&self) -> Result<bool> {
        // SAFETY: Live stream; retained COW sources outlive this fence.
        unsafe {
            check(
                sys::cuStreamSynchronize(self.session.stream),
                "complete KV COW",
            )?;
        }
        Ok(true)
    }
    fn publish(
        &self,
        page: &Slab,
        used: usize,
    ) -> Result<Rc<crate::backend::snapshot::Allocation>> {
        let mut canonical = Reservation::reserve(self.session, page.bytes, 1)?;
        let allocation = Rc::new(crate::backend::snapshot::Allocation {
            owner: self.session.owner,
            id: page.id,
            address: canonical.address,
            bytes: page.bytes,
            used,
            handle: Some(page.handle),
            native: None,
        });
        // Until publication succeeds, this alias borrows the original handle.
        canonical.region.mappings.push(Slab {
            id: page.id,
            offset: 0,
            bytes: page.bytes,
            handle: page.handle,
            mapped: false,
            shared: Some(allocation.clone()),
        });
        let ops = Ops {
            session: self.session,
            address: canonical.address,
        };
        let result = ops.map(0, page.bytes, page.handle, true).and_then(|()| {
            canonical.region.mappings[0].mapped = true;
            ops.access(0, page.bytes, true)
        });
        let key = canonical.address.to_string();
        if result.is_ok() {
            canonical.region.mappings[0].shared = None;
            canonical.region.mapped = page.bytes;
        }
        self.session
            .prefix_buffers
            .borrow_mut()
            .insert(key, canonical);
        result?;
        self.session
            .snapshot_allocations
            .borrow_mut()
            .push(allocation.clone());
        Ok(allocation)
    }
}

pub(crate) fn restore_host(
    s: &Session,
    descriptor: crate::backend::kv::host::Descriptor,
    bytes: &[u8],
) -> Result<Rc<crate::backend::snapshot::Allocation>> {
    descriptor.owner.require(s.owner)?;
    if bytes.len() != descriptor.used || !descriptor.paged {
        return Err("Invalid CUDA CPU KV payload".into());
    }
    let mut canonical = Reservation::reserve(s, descriptor.bytes, 1)?;
    let ops = Ops {
        session: s,
        address: canonical.address,
    };
    let handle = ops.allocate(descriptor.bytes)?;
    canonical.region.mappings.push(Slab {
        id: descriptor.id,
        offset: 0,
        bytes: descriptor.bytes,
        handle,
        mapped: false,
        shared: None,
    });
    let allocation = Rc::new(crate::backend::snapshot::Allocation {
        owner: s.owner,
        id: descriptor.id,
        address: canonical.address,
        bytes: descriptor.bytes,
        used: descriptor.used,
        handle: Some(handle),
        native: None,
    });
    let result: Result<()> = (|| {
        ops.map(0, descriptor.bytes, handle, false)?;
        canonical.region.mappings[0].mapped = true;
        ops.access(0, descriptor.bytes, false)?;
        ops.zero(0, descriptor.bytes)?;
        s.transfers
            .borrow_mut()
            .upload(s.stream, canonical.address, bytes)?;
        ops.complete()?;
        ops.access(0, descriptor.bytes, true)?;
        canonical.region.mapped = descriptor.bytes;
        Ok(())
    })();
    s.prefix_buffers
        .borrow_mut()
        .insert(canonical.address.to_string(), canonical);
    s.snapshot_allocations.borrow_mut().push(allocation.clone());
    result?;
    Ok(allocation)
}

#[cfg(test)]
#[path = "virtual_memory_tests.rs"]
mod tests;
