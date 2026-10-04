//! CUDA VMM ownership: graph pointers stay fixed while physical slabs grow.
use super::{Driver, Result, Session, check};
use std::collections::BTreeMap;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Location {
    kind: i32,
    id: i32,
}
#[repr(C)]
pub(crate) struct AllocationProp {
    kind: i32,
    handles: i32,
    location: Location,
    windows: *mut std::ffi::c_void,
    flags: [u8; 8],
}
#[repr(C)]
pub(crate) struct AccessDesc {
    location: Location,
    flags: i32,
}

pub(crate) struct Reservation {
    pub address: u64,
    pub bytes: usize,
    pub stride: usize,
    pub granularity: usize,
    pub mapped: usize,
    pub slabs: Vec<(usize, usize, u64)>,
}
impl Reservation {
    pub(crate) fn reserve(s: &Session, bytes: usize, stride: usize) -> Result<Self> {
        let prop = properties(s.device);
        let mut granularity = 0;
        let mut address = 0;
        // SAFETY: repr(C) properties match CUDA 12.6 cuda.h; pointers are live.
        unsafe {
            check(
                (s.driver.vmm_granularity)(&mut granularity, &prop, 0),
                "VMM granularity",
            )?;
            let bytes = round_up(bytes, granularity)?;
            check(
                (s.driver.vmm_reserve)(&mut address, bytes, granularity, 0, 0),
                "reserve KV address space",
            )?;
            Ok(Self {
                address,
                bytes,
                stride,
                granularity,
                mapped: 0,
                slabs: vec![],
            })
        }
    }
    pub(crate) fn extent(&self, tokens: usize) -> Result<usize> {
        let raw = tokens
            .checked_mul(self.stride)
            .ok_or("KV extent overflow")?;
        let extent = round_up(raw, self.granularity)?;
        if extent > self.bytes {
            return Err("KV growth exceeds reserved context".into());
        }
        Ok(extent)
    }
    pub(crate) fn grow(&mut self, s: &Session, tokens: usize) -> Result<()> {
        let extent = self.extent(tokens)?;
        if extent <= self.mapped {
            return Ok(());
        }
        let bytes = extent - self.mapped;
        let offset = self.mapped;
        let address = self
            .address
            .checked_add(offset as u64)
            .ok_or("KV address overflow")?;
        let prop = properties(s.device);
        let access = AccessDesc {
            location: prop.location,
            flags: 3,
        }; // READWRITE
        let mut handle = 0;
        // SAFETY: Caller synchronizes the stream before changing mappings.
        // On partial failure, release any resource not yet owned by this object.
        unsafe {
            check(
                (s.driver.vmm_create)(&mut handle, bytes, &prop, 0),
                "create KV physical slab",
            )?;
            if let Err(e) = check(
                (s.driver.vmm_map)(address, bytes, 0, handle, 0),
                "map KV slab",
            ) {
                (s.driver.vmm_release)(handle);
                return Err(e);
            }
            self.slabs.push((offset, bytes, handle));
            self.mapped = extent;
            check(
                (s.driver.vmm_access)(address, bytes, &access, 1),
                "enable KV slab access",
            )?;
            // Zero once so masked tile accesses within the rounded slab are defined.
            check(
                (s.driver.memset)(address, 0, bytes, s.stream),
                "initialize KV slab",
            )?;
        }
        Ok(())
    }
    pub(crate) fn release_slabs(&mut self, driver: &Driver) -> Result<()> {
        let mut errors = vec![];
        for (offset, bytes, handle) in self.slabs.drain(..) {
            // SAFETY: Work was synchronized; each mapping/handle is owned once.
            unsafe {
                if let Err(e) = check(
                    (driver.vmm_unmap)(self.address + offset as u64, bytes),
                    "unmap KV slab",
                ) {
                    errors.push(e)
                }
                if let Err(e) = check((driver.vmm_release)(handle), "release KV slab") {
                    errors.push(e)
                }
            }
        }
        self.mapped = 0;
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors.join("; "))
        }
    }
}
fn properties(device: i32) -> AllocationProp {
    AllocationProp {
        kind: 1,
        handles: 0,
        location: Location {
            kind: 1,
            id: device,
        },
        windows: std::ptr::null_mut(),
        flags: [0; 8],
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
    #[ignore = "Requires exclusive GPU experiment lock"]
    fn vmm_graph_growth_reset_and_last_token() {
        use super::super::{Handle, Session};
        let mut s = Session::new(Driver::load().unwrap());
        // SAFETY: Test owns a primary-context reference, stream and graph handles.
        unsafe {
            check((s.driver.init)(0), "init").unwrap();
            check((s.driver.device)(&mut s.device, 0), "device").unwrap();
            check((s.driver.current)(&mut s.previous), "previous").unwrap();
            let mut context: Handle = std::ptr::null_mut();
            check((s.driver.retain)(&mut context, s.device), "retain").unwrap();
            s.retained = true;
            check((s.driver.set_current)(context), "current").unwrap();
            check((s.driver.stream_create)(&mut s.stream, 1), "stream").unwrap();
        }
        let mut r = Reservation::reserve(&s, 262144 * 2048, 2048).unwrap();
        let address = r.address;
        r.grow(&s, 1).unwrap();
        assert_eq!(r.mapped, r.granularity);
        // SAFETY: Capture a write into a mapped buffer; subsequent mappings keep
        // the graph address stable. Every map/unmap follows stream completion.
        unsafe {
            check((s.driver.stream_sync)(s.stream), "init sync").unwrap();
            check((s.driver.capture_begin)(s.stream, 0), "capture").unwrap();
            s.capturing = true;
            check((s.driver.memset)(address, 37, 4, s.stream), "record write").unwrap();
            check(
                (s.driver.capture_end)(s.stream, &mut s.graph),
                "capture end",
            )
            .unwrap();
            s.capturing = false;
            check(
                (s.driver.graph_instantiate)(&mut s.exec, s.graph, 0),
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
                check((s.driver.memset)(last, 71, 4, s.stream), "last token write").unwrap();
                check((s.driver.graph_launch)(s.exec, s.stream), "graph replay").unwrap();
                check((s.driver.stream_sync)(s.stream), "sync").unwrap();
                check(
                    (s.driver.download)(value.as_mut_ptr().cast(), last, 4),
                    "last token read",
                )
                .unwrap();
                assert_eq!(value, [71; 4]);
                check(
                    (s.driver.download)(value.as_mut_ptr().cast(), address, 4),
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
            check((s.driver.stream_sync)(s.stream), "reset sync").unwrap();
            check(
                (s.driver.download)(value.as_mut_ptr().cast(), address, 4),
                "reset read",
            )
            .unwrap();
            assert_eq!(value, [0; 4]);
            check((s.driver.graph_launch)(s.exec, s.stream), "reset replay").unwrap();
            check((s.driver.stream_sync)(s.stream), "reset replay sync").unwrap();
            check(
                (s.driver.download)(value.as_mut_ptr().cast(), address, 4),
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
        };
        assert_eq!(r.extent(1024).unwrap(), 2 << 20);
        assert_eq!(r.extent(1025).unwrap(), 4 << 20);
        assert!(r.extent(2049).is_err());
        let scale = Reservation {
            address: 0,
            bytes: 8 << 20,
            stride: 32,
            granularity: 2 << 20,
            mapped: 0,
            slabs: vec![],
        };
        assert_eq!(scale.extent(65536).unwrap(), 2 << 20);
        assert_eq!(scale.extent(65537).unwrap(), 4 << 20);
        assert_eq!(scale.extent(262144).unwrap(), 8 << 20);
        assert!(scale.extent(usize::MAX).is_err());
    }
}
