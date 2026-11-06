//! Bounded disk/DMA pipeline. Reader threads never own or call CUDA handles.
use super::{Session, check, sys};
use crate::{
    artifact::Result,
    weights::{
        TensorSource,
        read::{ALIGNMENT, Reader},
    },
};
use std::{
    collections::VecDeque,
    ptr,
    sync::{Arc, Mutex, mpsc},
    time::Instant,
};

const SLOT_BYTES: usize = 8 << 20;
const PAYLOAD_BYTES: usize = SLOT_BYTES - ALIGNMENT;

pub(super) struct Upload {
    pub source: TensorSource,
    pub destination: u64,
}

#[derive(Default)]
pub(super) struct Statistics {
    pub read_wait_s: f64,
    pub dma_wait_s: f64,
    pub idle_work_s: f64,
    pub chunks: usize,
}

/// Ready reads have priority. While disk readers are busy, the CUDA owner may
/// finish bounded initialization work; no CUDA handle crosses into a reader.
fn receive<T>(
    completed: &mpsc::Receiver<T>,
    idle: &mut dyn FnMut() -> Result<bool>,
) -> Result<(T, f64, f64)> {
    let mut idle_s = 0.;
    loop {
        match completed.try_recv() {
            Ok(value) => return Ok((value, 0., idle_s)),
            Err(mpsc::TryRecvError::Disconnected) => return Err("Weight readers stopped".into()),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        let at = Instant::now();
        let worked = idle()?;
        idle_s += at.elapsed().as_secs_f64();
        if !worked {
            let at = Instant::now();
            let value = completed.recv().map_err(|_| "Weight readers stopped")?;
            return Ok((value, at.elapsed().as_secs_f64(), idle_s));
        }
    }
}

struct Slot {
    memory: Option<memmap2::MmapMut>,
    pointer: *mut std::ffi::c_void,
    event: sys::CUevent,
    registered: bool,
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
                for slot in &mut self.slots {
                    if let Some(memory) = slot.memory.take() {
                        std::mem::forget(memory);
                    }
                }
                return;
            }
            for slot in &mut self.slots {
                if !slot.event.is_null()
                    && let Err(error) = check(
                        sys::cuEventDestroy_v2(slot.event),
                        "weight staging event cleanup",
                    )
                {
                    eprintln!("CUDA weight staging cleanup: {error}");
                }
                if slot.registered
                    && let Err(error) = check(
                        sys::cuMemHostUnregister(slot.pointer),
                        "weight staging unregister",
                    )
                {
                    eprintln!("CUDA weight staging cleanup: {error}");
                    if let Some(memory) = slot.memory.take() {
                        std::mem::forget(memory);
                    }
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
            // Ordinary page-aligned CPU mappings keep direct file I/O on the
            // normal page path. Register once for the existing asynchronous DMA.
            let mut memory = memmap2::MmapMut::map_anon(SLOT_BYTES).map_err(|e| e.to_string())?;
            let pointer = memory.as_mut_ptr().cast();
            pool.slots.push(Slot {
                memory: Some(memory),
                pointer,
                event: ptr::null_mut(),
                registered: false,
            });
            // SAFETY: The pool owns this complete page-aligned mapping and
            // retains it until readers join, DMA completes and registration ends.
            unsafe {
                check(
                    sys::cuMemHostRegister_v2(pointer, SLOT_BYTES, 0),
                    "weight staging register",
                )?;
            }
            pool.slots.last_mut().unwrap().registered = true;
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

pub(super) fn upload(
    session: &Session,
    uploads: Vec<Upload>,
    workers: usize,
) -> Result<Statistics> {
    upload_with_idle(session, uploads, workers, &mut || Ok(false))
}

pub(super) fn upload_with_idle(
    session: &Session,
    mut uploads: Vec<Upload>,
    workers: usize,
    idle: &mut dyn FnMut() -> Result<bool>,
) -> Result<Statistics> {
    if !(1..=32).contains(&workers) {
        return Err("Weight reader count must be 1..32".into());
    }
    if uploads.is_empty() {
        return Ok(Statistics::default());
    }
    for upload in &uploads {
        upload
            .destination
            .checked_add(upload.source.bytes as u64)
            .ok_or("Weight upload address overflow")?;
    }
    uploads
        .sort_by(|a, b| (&a.source.path, a.source.offset).cmp(&(&b.source.path, b.source.offset)));
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
                let mut reader = Reader::default();
                loop {
                    let Ok(job) = jobs.lock().unwrap().recv() else {
                        break;
                    };
                    // SAFETY: A job exclusively borrows its slot until this
                    // completion is received. Only CPU readers touch this range.
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(job.pointer as *mut u8, SLOT_BYTES)
                    };
                    let result = reader.read(&job.source, job.offset, job.bytes, target);
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
            let mut reading = 0;
            let mut in_flight = VecDeque::new();
            let mut statistics = Statistics::default();
            for slot in 0..pool.slots.len() {
                reading += usize::from(enqueue(slot, &mut pending)?);
            }
            active += reading;
            while active != 0 {
                while let Some(&index) = in_flight.front() {
                    let slot: &Slot = &pool.slots[index];
                    // SAFETY: Only the CUDA owner queries events. A completed
                    // event releases this slot for a new CPU read, never sooner.
                    unsafe {
                        let status = sys::cuEventQuery(slot.event);
                        if status == sys::CUresult::CUDA_ERROR_NOT_READY {
                            if reading != 0 {
                                break;
                            }
                            let at = Instant::now();
                            check(
                                sys::cuEventSynchronize(slot.event),
                                "weight upload completion",
                            )?;
                            statistics.dma_wait_s += at.elapsed().as_secs_f64();
                        } else {
                            check(status, "weight upload completion query")?;
                        }
                    }
                    in_flight.pop_front();
                    if enqueue(index, &mut pending)? {
                        reading += 1;
                    } else {
                        active -= 1;
                    }
                }
                if active == 0 {
                    break;
                }
                debug_assert_ne!(reading, 0);
                let ((job, result), wait_s, idle_s) = receive(&done_rx, idle)?;
                statistics.read_wait_s += wait_s;
                statistics.idle_work_s += idle_s;
                reading -= 1;
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
                }
                in_flight.push_back(job.slot);
                statistics.chunks += 1;
            }
            Ok(statistics)
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
    fn ready_reads_take_priority_and_initialization_errors_propagate() {
        let (tx, rx) = mpsc::channel();
        tx.send(7).unwrap();
        assert_eq!(
            receive(&rx, &mut || panic!("ready read must not wait"))
                .unwrap()
                .0,
            7
        );
        let mut calls = 0;
        let (value, _, idle_s) = receive(&rx, &mut || {
            calls += 1;
            if calls == 3 {
                tx.send(9).unwrap();
            }
            Ok(true)
        })
        .unwrap();
        assert_eq!((value, calls), (9, 3));
        assert!(idle_s >= 0.);
        assert_eq!(
            receive(&rx, &mut || Err("module initialization failed".into())).unwrap_err(),
            "module initialization failed"
        );
        drop(tx);
        assert_eq!(
            receive(&rx, &mut || panic!("closed readers must stop")).unwrap_err(),
            "Weight readers stopped"
        );
    }

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
        let mut reader = Reader::default();
        for offset in (0..source.bytes).step_by(PAYLOAD_BYTES) {
            let bytes = (source.bytes - offset).min(PAYLOAD_BYTES);
            let skip = reader.read(&source, offset, bytes, &mut staging).unwrap();
            assert_eq!(
                &staging[skip..skip + bytes],
                &data[source.offset + offset..source.offset + offset + bytes]
            );
        }
        assert!(
            reader
                .read(&source, source.bytes - 1, 2, &mut staging)
                .is_err()
        );
        fs::write(&path, &data[..100]).unwrap();
        assert!(reader.read(&source, 0, 100, &mut staging).is_err());
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
                    workers,
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
            // More jobs than staging slots, intentionally reversed in file
            // order, exercise descriptor reuse and event-gated slot recycling.
            let pieces = (0..97)
                .rev()
                .map(|i| Upload {
                    source: TensorSource {
                        path: path.clone(),
                        offset: 173 + i * 137,
                        bytes: 137,
                    },
                    destination: destination + (i * 137) as u64,
                })
                .collect();
            let stats = upload(&session, pieces, workers).unwrap();
            assert_eq!(stats.chunks, 97);
            let mut compact = vec![0u8; 97 * 137];
            // SAFETY: The complete upload drained all slot events and the
            // registered destination covers this compact readback extent.
            unsafe {
                check(
                    sys::cuMemcpyDtoH_v2(compact.as_mut_ptr().cast(), destination, compact.len()),
                    "test scattered readback",
                )
                .unwrap();
            }
            assert_eq!(compact, data[173..173 + 97 * 137]);
        }
        std::fs::remove_file(path).unwrap();
    }
}
