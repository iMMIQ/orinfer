//! Bounded disk/DMA pipeline. Reader threads never own or call CUDA handles.
use super::{Session, check, sys};
use crate::{artifact::Result, weights::TensorSource};
use std::{
    fs::{File, OpenOptions},
    io::ErrorKind,
    os::unix::fs::{FileExt, OpenOptionsExt},
    ptr,
    sync::{Arc, Mutex, mpsc},
};

const ALIGNMENT: usize = 4096;
const SLOT_BYTES: usize = 8 << 20;
const PAYLOAD_BYTES: usize = SLOT_BYTES - ALIGNMENT;

pub(super) struct Upload {
    pub source: TensorSource,
    pub destination: u64,
}

struct Slot {
    pointer: *mut std::ffi::c_void,
    event: sys::CUevent,
}
struct Staging<'a> {
    session: &'a Session,
    slots: Vec<Slot>,
}
impl Drop for Staging<'_> {
    fn drop(&mut self) {
        // Reader scope has joined before this owner drops. Finish DMA before
        // freeing any host memory, including after a read/upload failure.
        // SAFETY: The session and its explicit stream outlive this pool.
        unsafe {
            let code = sys::cuStreamSynchronize(self.session.stream);
            if code != sys::CUresult::CUDA_SUCCESS {
                // Never release a host range which may still be in use by DMA.
                eprintln!(
                    "CUDA weight staging cleanup: synchronize error {}",
                    code as i32
                );
                return;
            }
            for slot in &self.slots {
                if !slot.event.is_null()
                    && let Err(error) = check(
                        sys::cuEventDestroy_v2(slot.event),
                        "weight staging event cleanup",
                    )
                {
                    eprintln!("CUDA weight staging cleanup: {error}");
                }
                if let Err(error) = check(
                    sys::cuMemFreeHost(slot.pointer),
                    "weight staging memory cleanup",
                ) {
                    eprintln!("CUDA weight staging cleanup: {error}");
                }
            }
        }
    }
}
impl<'a> Staging<'a> {
    fn new(session: &'a Session, count: usize) -> Result<Self> {
        let mut pool = Self {
            session,
            slots: Vec::with_capacity(count),
        };
        for _ in 0..count {
            let mut pointer = ptr::null_mut();
            // SAFETY: Live owner context; allocation is owned before another
            // fallible call and will only be freed by this CUDA thread.
            unsafe {
                check(
                    sys::cuMemHostAlloc(&mut pointer, SLOT_BYTES, 0),
                    "weight staging allocation",
                )?;
            }
            pool.slots.push(Slot {
                pointer,
                event: ptr::null_mut(),
            });
            // SAFETY: Event immediately belongs to the pool; timing is disabled.
            unsafe {
                check(
                    sys::cuEventCreate(&mut pool.slots.last_mut().unwrap().event, 2),
                    "weight staging event",
                )?;
            }
        }
        Ok(pool)
    }
}

#[derive(Clone)]
struct Job {
    slot: usize,
    source: Arc<TensorSource>,
    offset: usize,
    bytes: usize,
    destination: u64,
    // A slot is exclusively borrowed by exactly one job. The coordinator only
    // dispatches it after its previous DMA event completes; all readers join
    // before the CUDA owner frees this allocation. Integers carry no CUDA API.
    pointer: usize,
}

fn unsupported_direct(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

/// Read exactly this payload, accepting an EOF tail on a sector-aligned direct read.
fn read_chunk(
    source: &TensorSource,
    offset: usize,
    bytes: usize,
    target: &mut [u8],
) -> Result<usize> {
    let absolute = source
        .offset
        .checked_add(offset)
        .ok_or("Weight offset overflow")?;
    if bytes == 0
        || offset
            .checked_add(bytes)
            .is_none_or(|end| end > source.bytes)
    {
        return Err("Weight chunk exceeds tensor".into());
    }
    let aligned = absolute / ALIGNMENT * ALIGNMENT;
    let skip = absolute - aligned;
    let extent = (skip + bytes).next_multiple_of(ALIGNMENT);
    if extent > target.len() {
        return Err("Weight chunk exceeds staging slot".into());
    }
    let direct = (target.as_ptr() as usize).is_multiple_of(ALIGNMENT);
    let opened = if direct {
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(&source.path)
    } else {
        File::open(&source.path)
    };
    let (file, direct) = match opened {
        Ok(file) => (file, direct),
        Err(error) if direct && unsupported_direct(&error) => {
            (File::open(&source.path).map_err(|e| e.to_string())?, false)
        }
        Err(error) => return Err(format!("{}: {error}", source.path.display())),
    };
    let size = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len())
        .map_err(|_| "Weight shard length overflow")?;
    if absolute.checked_add(bytes).is_none_or(|end| end > size) {
        return Err(format!(
            "{}: truncated weight payload",
            source.path.display()
        ));
    }
    if direct {
        let read = loop {
            match file.read_at(&mut target[..extent], aligned as u64) {
                Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                result => break result,
            }
        };
        match read {
            Ok(n) if n == extent.min(size - aligned) => return Ok(skip),
            Ok(_) => {
                return Err(format!(
                    "{}: short direct weight read",
                    source.path.display()
                ));
            }
            Err(error) if unsupported_direct(&error) => {}
            Err(error) => return Err(format!("{}: {error}", source.path.display())),
        }
    }
    let file = if direct {
        File::open(&source.path).map_err(|e| e.to_string())?
    } else {
        file
    };
    file.read_exact_at(&mut target[..bytes], absolute as u64)
        .map_err(|e| format!("{}: {e}", source.path.display()))?;
    Ok(0)
}

pub(super) fn upload(session: &Session, uploads: Vec<Upload>, workers: usize) -> Result<()> {
    if uploads.is_empty() {
        return Ok(());
    }
    let pool = Staging::new(session, (workers * 2).min(32))?;
    let mut pending = uploads.into_iter().flat_map(|upload| {
        let source = Arc::new(upload.source);
        let destination = upload.destination;
        (0..source.bytes).step_by(PAYLOAD_BYTES).map(move |offset| {
            let bytes = (source.bytes - offset).min(PAYLOAD_BYTES);
            (
                Arc::clone(&source),
                offset,
                bytes,
                destination + offset as u64,
            )
        })
    });
    std::thread::scope(|scope| {
        let (jobs_tx, jobs_rx) = mpsc::sync_channel::<Job>(pool.slots.len());
        let jobs_rx = Arc::new(Mutex::new(jobs_rx));
        let (done_tx, done_rx) = mpsc::sync_channel(pool.slots.len());
        for _ in 0..workers {
            let jobs = Arc::clone(&jobs_rx);
            let done = done_tx.clone();
            scope.spawn(move || {
                loop {
                    let Ok(job) = jobs.lock().unwrap().recv() else {
                        break;
                    };
                    // SAFETY: A job exclusively borrows its slot until this
                    // completion is received. Only CPU readers touch this range.
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(job.pointer as *mut u8, SLOT_BYTES)
                    };
                    let result = read_chunk(&job.source, job.offset, job.bytes, target);
                    if done.send((job, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(done_tx);
        let enqueue =
            |slot: usize,
             pending: &mut dyn Iterator<Item = (Arc<TensorSource>, usize, usize, u64)>|
             -> Result<bool> {
                let Some((source, offset, bytes, destination)) = pending.next() else {
                    return Ok(false);
                };
                jobs_tx
                    .send(Job {
                        slot,
                        source,
                        offset,
                        bytes,
                        destination,
                        pointer: pool.slots[slot].pointer as usize,
                    })
                    .map_err(|_| "Weight reader queue closed")?;
                Ok(true)
            };
        let result = (|| {
            let mut active = 0;
            for slot in 0..pool.slots.len() {
                active += usize::from(enqueue(slot, &mut pending)?);
            }
            while active != 0 {
                let (job, result) = done_rx.recv().map_err(|_| "Weight readers stopped")?;
                let skip = result?;
                let slot = &pool.slots[job.slot];
                // SAFETY: CPU read completed; GPU allocation covers this chunk.
                // The event completes before the slot is returned to a reader.
                unsafe {
                    check(
                        sys::cuMemcpyHtoDAsync_v2(
                            job.destination,
                            slot.pointer.add(skip),
                            job.bytes,
                            session.stream,
                        ),
                        "stream weight chunk",
                    )?;
                    check(
                        sys::cuEventRecord(slot.event, session.stream),
                        "weight upload event",
                    )?;
                    check(
                        sys::cuEventSynchronize(slot.event),
                        "weight upload completion",
                    )?;
                }
                if !enqueue(job.slot, &mut pending)? {
                    active -= 1;
                }
            }
            Ok(())
        })();
        // Unblock every reader on success and failure before scope joins them.
        drop(jobs_tx);
        drop(done_rx);
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn direct_reads_preserve_header_offsets_chunks_and_eof_tails() {
        let path = std::env::temp_dir().join(format!(
            "orin-stream-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let data: Vec<_> = (0..SLOT_BYTES * 2 + 731).map(|i| (i * 37) as u8).collect();
        fs::write(&path, &data).unwrap();
        let source = TensorSource {
            path: path.clone(),
            offset: 173,
            bytes: data.len() - 173,
        };
        let mut staging = memmap2::MmapMut::map_anon(SLOT_BYTES).unwrap();
        for offset in (0..source.bytes).step_by(PAYLOAD_BYTES) {
            let bytes = (source.bytes - offset).min(PAYLOAD_BYTES);
            let skip = read_chunk(&source, offset, bytes, &mut staging).unwrap();
            assert_eq!(
                &staging[skip..skip + bytes],
                &data[source.offset + offset..source.offset + offset + bytes]
            );
        }
        assert!(read_chunk(&source, source.bytes - 1, 2, &mut staging).is_err());
        fs::write(&path, &data[..100]).unwrap();
        assert!(read_chunk(&source, 0, 100, &mut staging).is_err());
        fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    #[ignore = "requires the serialized SM87 GPU"]
    fn streamed_bytes_and_reader_error_cleanup() {
        let mut session = Session::new(super::super::Driver::load().unwrap());
        // SAFETY: This test owns all CUDA handles on this thread. Session owns
        // each handle immediately after successful creation and cleans it up.
        unsafe {
            check(sys::cuInit(0), "test init").unwrap();
            check(sys::cuDeviceGet(&mut session.device, 0), "test device").unwrap();
            check(
                sys::cuCtxGetCurrent(&mut session.previous),
                "test previous context",
            )
            .unwrap();
            let mut context = ptr::null_mut();
            check(
                sys::cuDevicePrimaryCtxRetain(&mut context, session.device),
                "test context",
            )
            .unwrap();
            session.retained = true;
            check(sys::cuCtxSetCurrent(context), "test current context").unwrap();
            check(sys::cuStreamCreate(&mut session.stream, 1), "test stream").unwrap();
        }
        let path = std::env::temp_dir().join(format!("orin-stream-gpu-{}", std::process::id()));
        let data: Vec<_> = (0..SLOT_BYTES * 2 + 731).map(|i| (i * 37) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let source = TensorSource {
            path: path.clone(),
            offset: 173,
            bytes: data.len() - 173,
        };
        let mut destination = 0;
        // SAFETY: Validated test extent; ownership transfers to Session.
        unsafe {
            check(
                sys::cuMemAlloc_v2(&mut destination, source.bytes),
                "test buffer",
            )
            .unwrap();
        }
        session.buffers.push(destination);
        for workers in [1, 4, 12] {
            let bad = TensorSource {
                path: path.with_extension("missing"),
                offset: 0,
                bytes: 1024,
            };
            assert!(
                upload(
                    &session,
                    vec![
                        Upload {
                            source: source.clone(),
                            destination
                        },
                        Upload {
                            source: bad,
                            destination
                        }
                    ],
                    workers
                )
                .is_err()
            );
            upload(
                &session,
                vec![Upload {
                    source: source.clone(),
                    destination,
                }],
                workers,
            )
            .unwrap();
            let mut readback = vec![0u8; source.bytes];
            // SAFETY: Upload synchronizes every staging event before returning;
            // source and destination cover the complete tensor extent.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(readback.as_mut_ptr().cast(), destination, readback.len()),
                    "test readback",
                )
                .unwrap();
            }
            assert_eq!(readback, data[173..]);
        }
        std::fs::remove_file(path).unwrap();
    }
}
