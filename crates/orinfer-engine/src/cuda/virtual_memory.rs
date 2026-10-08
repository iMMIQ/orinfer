//! CUDA VMM ownership: graph pointers stay fixed while physical slabs grow.
use super::{Driver, Result, Session, check, sys};
use std::collections::BTreeMap;

use sys::{CUmemAccessDesc as AccessDesc, CUmemAllocationProp as AllocationProp};

pub(crate) struct Reservation {
    pub address: u64,
    pub bytes: usize,
    pub stride: usize,
    pub granularity: usize,
    pub mapped: usize,
    slabs: Vec<Slab>,
    poisoned: bool,
}
impl Reservation {
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
        let raw = tokens
            .checked_mul(self.stride)
            .ok_or("KV extent overflow")?;
        round_up(raw, self.granularity)
    }
    pub(crate) fn grow(&mut self, s: &super::Session, tokens: usize) -> Result<()> {
        if self.poisoned {
            return Err("KV reservation is quarantined after a failed transition".into());
        }
        let extent = self.extent(tokens)?;
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
            offset,
            bytes,
            handle,
            mapped: false,
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
    pub(crate) fn release_slabs(&mut self, driver: &Driver) -> Result<()> {
        let result = release_owned(
            &mut self.slabs,
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
#[derive(Debug)]
struct Slab {
    offset: usize,
    bytes: usize,
    handle: u64,
    mapped: bool,
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
    mut unmap: impl FnMut(&Slab) -> Result<()>,
    mut release: impl FnMut(u64) -> Result<()>,
) -> Result<()> {
    let mut errors = vec![];
    slabs.retain_mut(|slab| {
        if slab.mapped {
            if let Err(e) = unmap(slab) {
                errors.push(e);
                return true;
            }
            slab.mapped = false;
        }
        if let Err(e) = release(slab.handle) {
            errors.push(e);
            return true;
        }
        false
    });
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cleanup_retries_preserve_mapping_and_handle_ownership() {
        let mut slabs = vec![Slab {
            offset: 0,
            bytes: 4096,
            handle: 7,
            mapped: true,
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
        };
        assert_eq!(scale.extent(65536).unwrap(), 2 << 20);
        assert_eq!(scale.extent(65537).unwrap(), 4 << 20);
        assert_eq!(scale.extent(262144).unwrap(), 8 << 20);
        assert!(scale.extent(usize::MAX).is_err());
    }
}
