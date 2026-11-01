//! Bounded host storage for stream-ordered inputs and small control readbacks.
//! Caller-owned slices can be released immediately after an upload is queued.
use super::{Result, check, ptr, sys};
use std::ffi::c_void;

pub(super) const UPLOAD_BYTES: usize = 256 << 10;
const UPLOAD_SLOTS: usize = 16;
pub(super) const READBACK_BYTES: usize = 4096;

#[derive(Default)]
enum Pending {
    #[default]
    None,
    Event,
    Stream,
}
struct Slot {
    pointer: *mut c_void,
    event: sys::CUevent,
    pending: Pending,
    used: usize,
}
#[derive(Default)]
pub(super) struct Transfers {
    uploads: Vec<Slot>,
    next: usize,
    readback: *mut c_void,
    readback_pending: bool,
    readback_capacity: usize,
}
impl Transfers {
    pub(super) fn upload(
        &mut self,
        stream: sys::CUstream,
        destination: u64,
        bytes: &[u8],
    ) -> Result<()> {
        destination
            .checked_add(bytes.len() as u64)
            .ok_or("Input address overflow")?;
        for (i, chunk) in bytes.chunks(UPLOAD_BYTES).enumerate() {
            self.upload_chunk(stream, destination + (i * UPLOAD_BYTES) as u64, chunk)?;
        }
        Ok(())
    }
    fn upload_chunk(
        &mut self,
        stream: sys::CUstream,
        destination: u64,
        bytes: &[u8],
    ) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        if bytes.len() > UPLOAD_BYTES {
            return Err("Input exceeds pinned upload slot".into());
        }
        if let Some(slot) = self.uploads.get_mut(self.next)
            && bytes.len() > UPLOAD_BYTES - slot.used
        {
            // Seal an append-only slab, rather than synchronizing each scalar.
            slot.pending = Pending::Stream;
            // SAFETY: The event covers all queued reads from this live slab.
            unsafe {
                check(sys::cuEventRecord(slot.event, stream), "seal input staging")?;
            }
            slot.pending = Pending::Event;
            self.next = (self.next + 1) % UPLOAD_SLOTS;
        }
        if self.next == self.uploads.len() {
            let mut pointer = ptr::null_mut();
            // SAFETY: Current session owns this allocation until synchronized cleanup.
            unsafe {
                check(
                    sys::cuMemHostAlloc(&mut pointer, UPLOAD_BYTES, 0),
                    "allocate input staging",
                )?;
            }
            // Register the allocation before the next fallible CUDA operation.
            self.uploads.push(Slot {
                pointer,
                event: ptr::null_mut(),
                pending: Pending::None,
                used: 0,
            });
            // SAFETY: The newly registered slot owns the timing-disabled event.
            unsafe {
                check(
                    sys::cuEventCreate(&mut self.uploads[self.next].event, 2),
                    "allocate input completion",
                )?;
            }
        }
        let slot = &mut self.uploads[self.next];
        // SAFETY: A slot is overwritten only after its preceding DMA completed.
        // A failed event record retains the stronger stream dependency.
        unsafe {
            if matches!(slot.pending, Pending::Event) || slot.used + bytes.len() > UPLOAD_BYTES {
                match slot.pending {
                    Pending::None => (),
                    Pending::Event => {
                        check(sys::cuEventSynchronize(slot.event), "reuse input staging")?
                    }
                    Pending::Stream => {
                        check(sys::cuStreamSynchronize(stream), "recover input staging")?
                    }
                }
                slot.used = 0;
                slot.pending = Pending::None;
            }
            let source = slot.pointer.add(slot.used);
            ptr::copy_nonoverlapping(bytes.as_ptr(), source.cast(), bytes.len());
            slot.used += bytes.len();
            slot.pending = Pending::Stream;
            check(
                sys::cuMemcpyHtoDAsync_v2(destination, source, bytes.len(), stream),
                "enqueue model input",
            )?;
        }
        Ok(())
    }

    /// Each region is copied into the next contiguous part of the host output.
    pub(super) fn readback(
        &mut self,
        stream: sys::CUstream,
        regions: &[(u64, usize)],
        output: &mut [u8],
    ) -> Result<()> {
        self.readback_with_limit(stream, regions, output, READBACK_BYTES)
    }
    pub(super) fn bulk_readback(
        &mut self,
        stream: sys::CUstream,
        regions: &[(u64, usize)],
        output: &mut [u8],
    ) -> Result<()> {
        self.readback_with_limit(stream, regions, output, 4 << 20)
    }
    fn readback_with_limit(
        &mut self,
        stream: sys::CUstream,
        regions: &[(u64, usize)],
        output: &mut [u8],
        limit: usize,
    ) -> Result<()> {
        let bytes = regions.iter().try_fold(0usize, |sum, (_, n)| {
            sum.checked_add(*n).ok_or("Control readback size overflow")
        })?;
        if bytes != output.len() || bytes > limit {
            return Err("Control readback exceeds pinned buffer".into());
        }
        if bytes == 0 {
            return Ok(());
        }
        // SAFETY: This buffer remains session-owned even if a queued copy fails.
        // Copies and producer kernels share the explicit stream; only their final
        // dependency must complete before the caller can inspect the controls.
        unsafe {
            if self.readback_pending {
                check(sys::cuStreamSynchronize(stream), "recover control readback")?;
            }
            if self.readback_capacity < bytes && !self.readback.is_null() {
                check(
                    sys::cuMemFreeHost(self.readback),
                    "resize KV readback staging",
                )?;
                self.readback = ptr::null_mut();
                self.readback_capacity = 0;
            }
            if self.readback.is_null() {
                check(
                    sys::cuMemHostAlloc(&mut self.readback, limit, 0),
                    "allocate control readback",
                )?;
                self.readback_capacity = limit;
            }
            self.readback_pending = true;
            let mut offset = 0;
            for &(source, size) in regions {
                check(
                    sys::cuMemcpyDtoHAsync_v2(self.readback.add(offset), source, size, stream),
                    "enqueue model controls",
                )?;
                offset += size;
            }
            check(sys::cuStreamSynchronize(stream), "wait for model controls")?;
            self.readback_pending = false;
            ptr::copy_nonoverlapping(self.readback.cast(), output.as_mut_ptr(), bytes);
        }
        Ok(())
    }

    /// The owning session must first synchronize its stream successfully.
    pub(super) fn release(&mut self) -> Vec<(sys::CUresult, &'static str)> {
        let mut results = Vec::new();
        // SAFETY: Called only by session cleanup after all DMA has completed.
        unsafe {
            for slot in self.uploads.drain(..) {
                if !slot.event.is_null() {
                    results.push((sys::cuEventDestroy_v2(slot.event), "destroy input event"));
                }
                results.push((sys::cuMemFreeHost(slot.pointer), "free input staging"));
            }
            if !self.readback.is_null() {
                results.push((sys::cuMemFreeHost(self.readback), "free control readback"));
                self.readback = ptr::null_mut();
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cuda::{Driver, Session};

    #[test]
    fn invalid_transfer_extents_are_rejected_before_using_cuda() {
        let mut transfers = Transfers::default();
        assert!(
            transfers
                .upload_chunk(ptr::null_mut(), 0, &vec![0; UPLOAD_BYTES + 1])
                .is_err()
        );
        assert!(
            transfers
                .upload(ptr::null_mut(), u64::MAX, &[0; 2])
                .is_err()
        );
        assert!(
            transfers
                .readback(ptr::null_mut(), &[(0, 4)], &mut [0; 8])
                .is_err()
        );
        assert!(
            transfers
                .readback(
                    ptr::null_mut(),
                    &[(0, READBACK_BYTES + 1)],
                    &mut vec![0; READBACK_BYTES + 1]
                )
                .is_err()
        );
        transfers.upload(ptr::null_mut(), 0, &[]).unwrap();
        transfers.readback(ptr::null_mut(), &[], &mut []).unwrap();
        assert!(transfers.uploads.is_empty());
        assert!(transfers.readback.is_null());
    }

    #[test]
    #[ignore = "requires exclusive GPU experiment lock"]
    fn staged_inputs_outlive_callers_and_wrap_without_overwriting_dma() {
        let mut session = Session::new(Driver::load().unwrap());
        // SAFETY: This test owns the current context; every successful handle
        // and allocation immediately belongs to Session for synchronized cleanup.
        unsafe {
            check(sys::cuInit(0), "transfer test init").unwrap();
            check(
                sys::cuDeviceGet(&mut session.device, 0),
                "transfer test device",
            )
            .unwrap();
            check(
                sys::cuCtxGetCurrent(&mut session.previous),
                "transfer test previous",
            )
            .unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, session.device),
                "transfer test context",
            )
            .unwrap();
            session.retained = true;
            check(sys::cuCtxSetCurrent(context), "transfer test current").unwrap();
            check(
                sys::cuStreamCreate(&mut session.stream, 1),
                "transfer test stream",
            )
            .unwrap();
        }
        let iterations = UPLOAD_SLOTS * 3;
        let bytes = iterations * UPLOAD_BYTES;
        let mut allocation = 0;
        // SAFETY: Session retains both source/destination halves until all work completes.
        unsafe {
            check(
                sys::cuMemAlloc_v2(&mut allocation, bytes * 2),
                "transfer test allocation",
            )
            .unwrap();
        }
        session.buffers.push(allocation);
        for i in 0..iterations {
            let size = if i % 3 == 0 { UPLOAD_BYTES } else { 4 };
            let mut data = vec![i as u8; size];
            let offset = i * UPLOAD_BYTES;
            session
                .transfers
                .borrow_mut()
                .upload(session.stream, allocation + offset as u64, &data)
                .unwrap();
            data.fill(255);
            // SAFETY: Both ranges are session-owned and disjoint; the copy is
            // queued after its upload and before the source slot can be reused.
            unsafe {
                check(
                    sys::cuMemcpyDtoDAsync_v2(
                        allocation + (bytes + offset) as u64,
                        allocation + offset as u64,
                        size,
                        session.stream,
                    ),
                    "transfer test dependent copy",
                )
                .unwrap();
            }
        }
        let regions: Vec<_> = (0..iterations)
            .map(|i| (allocation + (bytes + i * UPLOAD_BYTES) as u64, 4))
            .collect();
        let mut result = vec![0; iterations * 4];
        session
            .transfers
            .borrow_mut()
            .readback(session.stream, &regions, &mut result)
            .unwrap();
        for (i, value) in result.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(value, &[i as u8; 4]);
        }
        for i in (0..iterations).step_by(3) {
            let mut tail = [0; 4];
            session
                .transfers
                .borrow_mut()
                .readback(
                    session.stream,
                    &[(allocation + (bytes + (i + 1) * UPLOAD_BYTES - 4) as u64, 4)],
                    &mut tail,
                )
                .unwrap();
            assert_eq!(tail, [i as u8; 4]);
        }
        let mut large = vec![91; UPLOAD_BYTES * 3 + 17];
        session
            .transfers
            .borrow_mut()
            .upload(session.stream, allocation, &large)
            .unwrap();
        large.fill(255);
        let mut ends = [0; 8];
        session
            .transfers
            .borrow_mut()
            .readback(
                session.stream,
                &[(allocation, 4), (allocation + (large.len() - 4) as u64, 4)],
                &mut ends,
            )
            .unwrap();
        assert_eq!(ends, [91; 8]);
        session.cleanup().unwrap();
        assert!(session.transfers.borrow().uploads.is_empty());
        assert!(session.transfers.borrow().readback.is_null());
    }
}
