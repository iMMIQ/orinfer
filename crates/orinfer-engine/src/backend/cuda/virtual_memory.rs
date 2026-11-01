//! CUDA VMM ownership: graph pointers stay fixed while physical slabs grow.
use super::{Driver, Result, Session, check, sys};
use crate::backend::kv::{self, Mapping as Slab, MappingOps};
use std::{collections::BTreeMap, rc::Rc};

use sys::{CUmemAccessDesc as AccessDesc, CUmemAllocationProp as AllocationProp};

pub(crate) struct Reservation {
    pub address: u64,
    pub bytes: usize,
    pub stride: usize,
    pub granularity: usize,
    pub mapped: usize,
    slabs: Vec<Slab>,
    poisoned: bool,
    retained: Vec<Rc<crate::backend::snapshot::Allocation>>,
}
impl Reservation {
    /// Map one sparse weight page. Payload initialization belongs to the uploader.
    pub(crate) fn map_page(&mut self, s: &Session, offset: usize) -> Result<()> {
        if self.poisoned
            || !offset.is_multiple_of(self.granularity)
            || offset
                .checked_add(self.granularity)
                .is_none_or(|end| end > self.bytes)
            || self.slabs.iter().any(|p| p.offset == offset)
        {
            return Err("Invalid sparse weight page transition".into());
        }
        let prop = properties(s.device);
        let access = AccessDesc {
            location: prop.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let mut handle = 0;
        // SAFETY: Aligned reserved extent; record ownership before mapping/access can fail.
        unsafe {
            check(
                sys::cuMemCreate(&mut handle, self.granularity, &prop, 0),
                "create expert page",
            )?;
        }
        self.slabs.push(Slab {
            id: kv::allocation_id()?,
            offset,
            bytes: self.granularity,
            handle,
            mapped: false,
            shared: None,
        });
        let slab = self.slabs.last_mut().unwrap();
        let result = (|| {
            // SAFETY: Unique owned page, wholly within the reserved address range.
            unsafe {
                check(
                    sys::cuMemMap(self.address + offset as u64, slab.bytes, 0, handle, 0),
                    "map expert page",
                )?;
                slab.mapped = true;
                check(
                    sys::cuMemSetAccess(self.address + offset as u64, slab.bytes, &access, 1),
                    "expert page access",
                )
            }
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    /// Caller must finish all consumers before releasing a sparse page.
    pub(crate) fn unmap_page(&mut self, s: &Session, offset: usize) -> Result<()> {
        let index = self
            .slabs
            .iter()
            .position(|p| p.offset == offset)
            .ok_or("Unknown expert page")?;
        let slab = &mut self.slabs[index];
        if slab.mapped {
            unsafe_unmap(&s.driver, self.address, slab)?;
            slab.mapped = false;
        }
        unsafe_release(&s.driver, slab.handle)?;
        self.slabs.swap_remove(index);
        Ok(())
    }
    pub(crate) fn physical_bytes(&self) -> usize {
        self.slabs
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
                bytes,
                stride,
                granularity,
                mapped: 0,
                slabs: vec![],
                poisoned: false,
                retained: vec![],
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
        if self.poisoned {
            return Err("KV reservation is quarantined after a failed transition".into());
        }
        let extent = round_up(bytes, self.granularity)?;
        if extent > self.bytes {
            return Err("KV growth exceeds reserved context".into());
        }
        if extent <= self.mapped {
            return Ok(());
        }
        let offset = self.mapped;
        let bytes = extent - offset;
        let address = self
            .address
            .checked_add(offset as u64)
            .ok_or("KV address overflow")?;
        let prop = properties(s.device);
        let access = AccessDesc {
            location: prop.location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let id = kv::allocation_id()?;
        let mut handle = 0;
        // SAFETY: Session owns the context/stream; each successful create is
        // immediately recorded before any fallible map/access/initialization.
        unsafe {
            check(
                sys::cuMemCreate(&mut handle, bytes, &prop, 0),
                "create KV physical slab",
            )?;
        }
        let mut slab = Slab {
            id,
            offset,
            bytes,
            handle,
            mapped: false,
            shared: None,
        };
        let result = (|| {
            // SAFETY: Reserved VA extent and live physical allocation do not overlap prior slabs.
            unsafe {
                check(sys::cuMemMap(address, bytes, 0, handle, 0), "map KV slab")?;
                slab.mapped = true;
                check(
                    sys::cuMemSetAccess(address, bytes, &access, 1),
                    "enable KV slab access",
                )?;
                check(
                    sys::cuMemsetD8Async(address, 0, bytes, s.stream),
                    "initialize KV slab",
                )?;
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.poisoned = true;
            // Keep failed cleanup ownership for Session::cleanup; never expose this extent.
            let mut pending = vec![slab];
            let _ = release_owned(
                &mut pending,
                |slab| unsafe_unmap(&s.driver, self.address, slab),
                |handle| unsafe_release(&s.driver, handle),
            );
            self.slabs.extend(pending);
            return Err(error);
        }
        self.slabs.push(slab);
        self.mapped = extent;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn allocation_handle(&self) -> Result<u64> {
        match self.slabs.as_slice() {
            [slab] if slab.offset == 0 && slab.bytes == self.mapped && slab.shared.is_none() => {
                Ok(slab.handle)
            }
            _ => Err("Shared allocation must have one complete owned mapping".into()),
        }
    }
    pub(crate) fn has_shared_slabs(&self) -> bool {
        self.slabs.iter().any(|p| p.shared.is_some())
    }
    /// Append a whole immutable allocation; CUDA currently requires handle offset zero.
    pub(crate) fn map_shared(
        &mut self,
        s: &Session,
        offset: usize,
        allocation: Rc<super::snapshot::Allocation>,
    ) -> Result<()> {
        s.owner.require(allocation.owner)?;
        let handle = allocation
            .handle
            .ok_or("Prefix allocation is not VMM backed")?;
        let bytes = allocation.bytes;
        if self.poisoned
            || offset != self.mapped
            || !offset.is_multiple_of(self.granularity)
            || !bytes.is_multiple_of(self.granularity)
            || offset.checked_add(bytes).is_none_or(|end| end > self.bytes)
        {
            return Err("Invalid shared prefix mapping".into());
        }
        self.slabs.push(Slab {
            id: allocation.id,
            offset,
            bytes,
            handle,
            mapped: false,
            shared: Some(allocation),
        });
        let slab = self.slabs.last_mut().unwrap();
        let access = AccessDesc {
            location: properties(s.device).location,
            flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ,
        };
        // SAFETY: The retained immutable allocation covers this whole unmapped extent.
        let result = unsafe {
            check(
                sys::cuMemMap(self.address + offset as u64, bytes, 0, handle, 0),
                "map shared prefix",
            )
            .and_then(|()| {
                slab.mapped = true;
                check(
                    sys::cuMemSetAccess(self.address + offset as u64, bytes, &access, 1),
                    "shared prefix read access",
                )
            })
        };
        if result.is_err() {
            self.poisoned = true;
        }
        result?;
        self.mapped += bytes;
        Ok(())
    }
    pub(crate) fn writable_bytes(&self, start: usize, bytes: usize) -> Result<usize> {
        kv::writable_bytes(
            &self.slabs,
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
            &mut self.slabs,
            &mut self.retained,
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
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn preview(
        &self,
        owner: orinfer_device::DeviceOwner,
        range: crate::backend::snapshot::Range,
    ) -> Result<
        Vec<(
            Rc<crate::backend::snapshot::Allocation>,
            crate::backend::snapshot::Range,
            usize,
        )>,
    > {
        kv::preview(&self.slabs, owner, self.address, range)
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
            &mut self.slabs,
            range,
            &Ops {
                session: s,
                address: self.address,
            },
        );
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn release_slabs(&mut self, driver: &Driver) -> Result<()> {
        let result = kv::release_mappings(
            &mut self.slabs,
            &mut self.retained,
            |slab| unsafe_unmap(driver, self.address, slab),
            |handle| unsafe_release(driver, handle),
        );
        if result.is_ok() {
            self.mapped = 0;
            self.poisoned = false;
        } else {
            self.poisoned = true;
        }
        result
    }
    pub(crate) fn has_slabs(&self) -> bool {
        !self.slabs.is_empty()
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
fn release_owned(
    slabs: &mut Vec<Slab>,
    unmap: impl FnMut(&Slab) -> Result<()>,
    release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    kv::release_mappings(slabs, &mut vec![], unmap, release)
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
    if granularity == 0 || !granularity.is_power_of_two() {
        return Err("Invalid CUDA VMM granularity".into());
    }
    bytes
        .checked_add(granularity - 1)
        .map(|n| n & !(granularity - 1))
        .ok_or_else(|| "KV alignment overflow".into())
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
    fn release(&self, handle: u64) -> Result<()> {
        unsafe_release(&self.session.driver, handle)
    }
    fn map(&self, offset: usize, bytes: usize, handle: u64, _shared: bool) -> Result<()> {
        // SAFETY: Caller validated and owns the virtual extent and physical handle.
        unsafe {
            check(
                sys::cuMemMap(self.address + offset as u64, bytes, 0, handle, 0),
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
        canonical.slabs.push(Slab {
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
            canonical.slabs[0].mapped = true;
            ops.access(0, page.bytes, true)
        });
        let key = canonical.address.to_string();
        if result.is_ok() {
            canonical.slabs[0].shared = None;
            canonical.mapped = page.bytes;
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
    canonical.slabs.push(Slab {
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
        canonical.slabs[0].mapped = true;
        ops.access(0, descriptor.bytes, false)?;
        ops.zero(0, descriptor.bytes)?;
        s.transfers
            .borrow_mut()
            .upload(s.stream, canonical.address, bytes)?;
        ops.complete()?;
        ops.access(0, descriptor.bytes, true)?;
        canonical.mapped = descriptor.bytes;
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
mod tests {
    use super::*;
    #[test]
    fn shared_mapping_cleanup_never_releases_snapshot_owned_handle() {
        let allocation = Rc::new(super::super::snapshot::Allocation {
            id: crate::backend::kv::allocation_id().unwrap(),
            used: 8192,
            native: None,
            owner: orinfer_device::DeviceOwner::new(orinfer_device::Backend::Cuda, 0),
            address: 0,
            bytes: 8192,
            handle: Some(7),
        });
        let mut slabs = vec![Slab {
            id: kv::allocation_id().unwrap(),
            offset: 0,
            bytes: 8192,
            handle: 7,
            mapped: true,
            shared: Some(Rc::clone(&allocation)),
        }];
        assert_eq!(Rc::strong_count(&allocation), 2);
        release_owned(
            &mut slabs,
            |_| Ok(()),
            |_| panic!("snapshot registry owns the handle"),
        )
        .unwrap();
        assert_eq!(Rc::strong_count(&allocation), 1);
        assert!(!overlaps(8192, 9000, 0, 8192));
        assert!(overlaps(8191, 8192, 0, 8192));
        assert!(!overlaps(0, 0, 0, 8192));
    }
    #[test]
    #[ignore = "Requires exclusive GPU experiment lock"]
    fn vmm_shared_prefix_cow_and_graph_replay() {
        let mut s = Session::new(Driver::load().unwrap());
        // SAFETY: Test exclusively owns a context reference and its CUDA resources.
        unsafe {
            check(sys::cuInit(0), "init").unwrap();
            check(sys::cuDeviceGet(&mut s.device, 0), "device").unwrap();
            check(sys::cuCtxGetCurrent(&mut s.previous), "previous").unwrap();
            let mut context = std::ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
                "retain",
            )
            .unwrap();
            s.retained = true;
            check(sys::cuCtxSetCurrent(context), "current").unwrap();
            check(sys::cuStreamCreate(&mut s.stream, 1), "stream").unwrap();
        }
        let mut source = Reservation::reserve(&s, 4 << 20, 1).unwrap();
        let bytes = source.granularity * 2;
        source.grow_bytes(&s, bytes).unwrap();
        let source_address = source.address;
        let allocation = Rc::new(super::super::snapshot::Allocation {
            id: crate::backend::kv::allocation_id().unwrap(),
            used: 8192,
            native: None,
            owner: s.owner,
            address: source.address,
            bytes,
            handle: Some(source.allocation_handle().unwrap()),
        });
        s.prefix_buffers.get_mut().insert("source".into(), source);
        let mut left = Reservation::reserve(&s, bytes * 2, 1).unwrap();
        let mut right = Reservation::reserve(&s, bytes * 2, 1).unwrap();
        let left_address = left.address;
        let right_address = right.address;
        // Capture a writer before replacing physical backing at the same VA.
        left.grow_bytes(&s, bytes).unwrap();
        // SAFETY: Captured memset targets a live owned mapping; all source writes finish first.
        unsafe {
            check(
                sys::cuMemsetD8Async(source_address, 37, bytes, s.stream),
                "source initialization",
            )
            .unwrap();
            check(sys::cuStreamSynchronize(s.stream), "init sync").unwrap();
            check(
                sys::cuStreamBeginCapture_v2(
                    s.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "capture",
            )
            .unwrap();
            s.capturing = true;
            check(
                sys::cuMemsetD8Async(left_address, 91, 4, s.stream),
                "record write",
            )
            .unwrap();
            check(
                sys::cuStreamEndCapture(s.stream, &mut s.graph),
                "capture end",
            )
            .unwrap();
            s.capturing = false;
            check(
                sys::cuGraphInstantiateWithFlags(&mut s.exec, s.graph, 0),
                "instantiate",
            )
            .unwrap();
        }
        left.release_slabs(&s.driver).unwrap();
        left.map_shared(&s, 0, Rc::clone(&allocation)).unwrap();
        right.map_shared(&s, 0, Rc::clone(&allocation)).unwrap();
        assert_eq!(left.physical_bytes() + right.physical_bytes(), 0);
        assert_eq!(left.writable_bytes(bytes, 4).unwrap(), 0);
        assert_eq!(left.writable_bytes(0, 4).unwrap(), bytes);
        left.make_writable(&s, 0, 4).unwrap();
        assert_eq!(left.address, left_address);
        assert_eq!(left.physical_bytes(), bytes);
        assert_eq!(right.physical_bytes(), 0);
        // SAFETY: The graph uses unchanged VAs; synchronized downloads cover four bytes.
        unsafe {
            check(sys::cuGraphLaunch(s.exec, s.stream), "COW replay").unwrap();
            check(sys::cuStreamSynchronize(s.stream), "replay sync").unwrap();
            for (address, expected) in [
                (left_address, 91u8),
                (right_address, 37),
                (source_address, 37),
            ] {
                let mut values = [0u8; 4];
                check(
                    sys::cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), address, values.len()),
                    "verify sharing isolation",
                )
                .unwrap();
                assert_eq!(values, [expected; 4]);
            }
        }
        right.grow_bytes(&s, bytes + 1).unwrap();
        assert_eq!(right.physical_bytes(), right.granularity);
        s.virtual_buffers.get_mut().insert("left".into(), left);
        s.virtual_buffers.get_mut().insert("right".into(), right);
        s.cleanup().unwrap();
    }
    #[test]
    fn cleanup_retries_preserve_mapping_and_handle_ownership() {
        let mut slabs = vec![Slab {
            id: kv::allocation_id().unwrap(),
            offset: 0,
            bytes: 4096,
            handle: 7,
            mapped: true,
            shared: None,
        }];
        let mut releases = 0;
        assert!(
            release_owned(
                &mut slabs,
                |_| Err("unmap fault".into()),
                |_| {
                    releases += 1;
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(releases, 0);
        assert!(slabs[0].mapped);
        assert!(release_owned(&mut slabs, |_| Ok(()), |_| Err("release fault".into())).is_err());
        assert!(!slabs[0].mapped);
        assert_eq!(slabs[0].handle, 7);
        release_owned(
            &mut slabs,
            |_| panic!("must not unmap twice"),
            |h| {
                assert_eq!(h, 7);
                Ok(())
            },
        )
        .unwrap();
        assert!(slabs.is_empty());
    }
    #[test]
    #[ignore = "Requires exclusive GPU experiment lock"]
    fn vmm_graph_growth_reset_and_last_token() {
        use super::super::Session;
        let mut s = Session::new(Driver::load().unwrap());
        // SAFETY: Test owns a primary-context reference, stream and graph handles.
        unsafe {
            check(sys::cuInit(0), "init").unwrap();
            check(sys::cuDeviceGet(&mut s.device, 0), "device").unwrap();
            check(sys::cuCtxGetCurrent(&mut s.previous), "previous").unwrap();
            let mut context: sys::CUcontext = std::ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, s.device),
                "retain",
            )
            .unwrap();
            s.retained = true;
            check(sys::cuCtxSetCurrent(context), "current").unwrap();
            check(sys::cuStreamCreate(&mut s.stream, 1), "stream").unwrap();
        }
        let mut r = Reservation::reserve(&s, 262144 * 2048, 2048).unwrap();
        let address = r.address;
        r.grow(&s, 1).unwrap();
        assert_eq!(r.mapped, r.granularity);
        // SAFETY: Capture a write into a mapped buffer; subsequent mappings keep
        // the graph address stable. Every map/unmap follows stream completion.
        unsafe {
            check(sys::cuStreamSynchronize(s.stream), "init sync").unwrap();
            check(
                sys::cuStreamBeginCapture_v2(
                    s.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "capture",
            )
            .unwrap();
            s.capturing = true;
            check(
                sys::cuMemsetD8Async(address, 37, 4, s.stream),
                "record write",
            )
            .unwrap();
            check(
                sys::cuStreamEndCapture(s.stream, &mut s.graph),
                "capture end",
            )
            .unwrap();
            s.capturing = false;
            check(
                sys::cuGraphInstantiateWithFlags(&mut s.exec, s.graph, 0),
                "instantiate",
            )
            .unwrap();
        }
        for tokens in [1024, 1025, 8193, 262144] {
            r.grow(&s, tokens).unwrap();
            assert_eq!(address, r.address);
            assert_eq!(r.mapped, r.extent(tokens).unwrap());
            let last = address + (tokens * 2048 - 4) as u64;
            let mut value = [0u8; 4];
            // SAFETY: First and last writes are within mapped extents.
            unsafe {
                check(
                    sys::cuMemsetD8Async(last, 71, 4, s.stream),
                    "last token write",
                )
                .unwrap();
                check(sys::cuGraphLaunch(s.exec, s.stream), "graph replay").unwrap();
                check(sys::cuStreamSynchronize(s.stream), "sync").unwrap();
                check(
                    sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), last, 4),
                    "last token read",
                )
                .unwrap();
                assert_eq!(value, [71; 4]);
                check(
                    sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
                    "graph read",
                )
                .unwrap();
                assert_eq!(value, [37; 4]);
            }
        }
        r.release_slabs(&s.driver).unwrap();
        assert_eq!(r.mapped, 0);
        r.grow(&s, 1).unwrap();
        let mut value = [1u8; 4];
        // SAFETY: Fresh physical storage is zeroed, graph still uses the same VA.
        unsafe {
            check(sys::cuStreamSynchronize(s.stream), "reset sync").unwrap();
            check(
                sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
                "reset read",
            )
            .unwrap();
            assert_eq!(value, [0; 4]);
            check(sys::cuGraphLaunch(s.exec, s.stream), "reset replay").unwrap();
            check(sys::cuStreamSynchronize(s.stream), "reset replay sync").unwrap();
            check(
                sys::cuMemcpyDtoH_v2(value.as_mut_ptr().cast(), address, 4),
                "replay read",
            )
            .unwrap();
            assert_eq!(value, [37; 4]);
        }
        s.virtual_buffers.borrow_mut().insert("test".into(), r);
        s.cleanup().unwrap();
    }
    #[test]
    fn allocation_abi_and_checked_extents() {
        assert_eq!(std::mem::size_of::<AllocationProp>(), 32);
        assert_eq!(std::mem::size_of::<AccessDesc>(), 12);
        assert_eq!(round_up(2048, 2 << 20).unwrap(), 2 << 20);
        assert!(round_up(usize::MAX, 2 << 20).is_err());
        assert!(round_up(1, 3).is_err());
        let r = Reservation {
            address: 0,
            bytes: 4 << 20,
            stride: 2048,
            granularity: 2 << 20,
            mapped: 0,
            slabs: vec![],
            poisoned: false,
            retained: vec![],
        };
        assert_eq!(r.extent(1024).unwrap(), 2 << 20);
        assert_eq!(r.extent(1025).unwrap(), 4 << 20);
        assert!(r.extent(2049).is_err());
        assert_eq!(r.required_extent(2049).unwrap(), 6 << 20);
        let scale = Reservation {
            address: 0,
            bytes: 8 << 20,
            stride: 32,
            granularity: 2 << 20,
            mapped: 0,
            slabs: vec![],
            poisoned: false,
            retained: vec![],
        };
        assert_eq!(scale.extent(65536).unwrap(), 2 << 20);
        assert_eq!(scale.extent(65537).unwrap(), 4 << 20);
        assert_eq!(scale.extent(262144).unwrap(), 8 << 20);
        assert!(scale.extent(usize::MAX).is_err());
    }
}
