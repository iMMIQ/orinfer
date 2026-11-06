//! CPU-only readers feed the existing owner-thread ACL upload stream.
use super::*;
use crate::weights::{
    TensorSource,
    read::{ALIGNMENT, Plan, Reader},
};
use std::sync::{Arc, Mutex, mpsc};
const READ_BYTES: usize = 32 << 20;
const READ_SLOTS: usize = 4;
const READERS: usize = 2;
pub(in crate::backend::ascend::model) struct Upload {
    pub source: TensorSource,
    pub buffer: Buffer,
}
#[derive(Debug, Default)]
pub(in crate::backend::ascend::model) struct Stats {
    pub chunks: usize,
    pub bytes: usize,
    pub read_wait_s: f64,
    pub upload_s: f64,
}
pub(in crate::backend::ascend::model) fn run(
    session: &Session,
    uploads: Vec<Upload>,
) -> Result<Stats> {
    let mut plan = Plan::new(
        uploads
            .iter()
            .enumerate()
            .map(|(k, u)| (k, u.source.clone()))
            .collect(),
        READ_BYTES,
    )?;
    for upload in &uploads {
        if upload.source.bytes != upload.buffer.bytes() {
            return Err("Prepared upload extent differs".into());
        }
    }
    if uploads.is_empty() {
        return Ok(Stats::default());
    }
    let mut tickets: VecDeque<Ticket> = VecDeque::new();
    let result = std::thread::scope(|scope| {
        let (jobs_tx, jobs_rx) =
            mpsc::sync_channel::<(crate::weights::read::Batch, memmap2::MmapMut)>(READ_SLOTS);
        let jobs = Arc::new(Mutex::new(jobs_rx));
        let (done_tx, done_rx) = mpsc::sync_channel(READ_SLOTS);
        for _ in 0..READERS {
            let jobs = Arc::clone(&jobs);
            let done = done_tx.clone();
            scope.spawn(move || {
                let mut reader = Reader::default();
                loop {
                    let Ok((batch, mut memory)) = jobs.lock().unwrap().recv() else {
                        break;
                    };
                    let result = reader.read(&batch.source, 0, batch.source.bytes, &mut memory);
                    if done.send((batch, memory, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(done_tx);
        let result = (|| {
            let mut active = 0;
            for _ in 0..READ_SLOTS {
                if let Some(batch) = plan.next() {
                    let memory = memmap2::MmapMut::map_anon(READ_BYTES + ALIGNMENT)
                        .map_err(|e| e.to_string())?;
                    jobs_tx
                        .send((batch, memory))
                        .map_err(|_| "Weight reader queue closed")?;
                    active += 1;
                }
            }
            let mut stats = Stats::default();
            while active != 0 {
                let at = Instant::now();
                let (batch, memory, result) =
                    done_rx.recv().map_err(|_| "Weight readers stopped")?;
                stats.read_wait_s += at.elapsed().as_secs_f64();
                let skip = result?;
                let at = Instant::now();
                for p in batch.pieces {
                    let target = &uploads[p.key].buffer;
                    let data = &memory[skip + p.within..skip + p.within + p.bytes];
                    for (offset, bytes) in data.chunks(UPLOAD_CHUNK).enumerate() {
                        if tickets.len() >= UPLOAD_DEPTH {
                            tickets.pop_front().unwrap().wait()?;
                        }
                        tickets.push_back(session.upload(
                            target,
                            p.offset + offset * UPLOAD_CHUNK,
                            bytes,
                        )?);
                    }
                }
                stats.upload_s += at.elapsed().as_secs_f64();
                stats.chunks += 1;
                stats.bytes += batch.source.bytes;
                if let Some(batch) = plan.next() {
                    jobs_tx
                        .send((batch, memory))
                        .map_err(|_| "Weight reader queue closed")?;
                } else {
                    active -= 1;
                }
            }
            Ok(stats)
        })();
        // Closing both endpoints unblocks all readers before their scope joins,
        // including EOF, allocation and native upload failures. No ACL owner moves.
        drop(jobs_tx);
        drop(done_rx);
        result
    });
    let drained = wait_in_flight(&mut tickets);
    match result {
        Ok(stats) => {
            drained?;
            Ok(stats)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;
    #[test]
    fn merged_reads_fill_aligned_arena_views_and_drain_reader_errors() {
        let mut fixture = Fixture::new();
        let session = fixture.session_with_limits(Limits {
            device_bytes: 128 << 20,
            pinned_bytes: 64 << 20,
            ..Limits::default()
        });
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("a.bin");
        let a = READ_BYTES / 2 + 23;
        let b = READ_BYTES / 2 + 35;
        let source_offset = 173;
        let data: Vec<_> = (0..source_offset + a + b + 97)
            .map(|i| (i * 37) as u8)
            .collect();
        std::fs::write(&path, &data).unwrap();
        let defs: Vec<Definition> = serde_json::from_value(serde_json::json!([
            {"name":"w.a","bytes":a,"prepared_tensor":{"name":"a","dtype":"U8","shape":[a]}},
            {"name":"w.b","bytes":b,"prepared_tensor":{"name":"b","dtype":"U8","shape":[b]}}
        ]))
        .unwrap();
        let layouts = weight_layouts(&defs).unwrap();
        assert_eq!(layouts.len(), 1);
        let buffers = weight_buffers(&session, &defs, &layouts).unwrap();
        assert_eq!(session.allocated_bytes(), layouts[0].allocation_bytes);
        for buffer in buffers.values() {
            assert_eq!(buffer.pointer() as usize % 512, 0);
        }
        let uploads = || {
            vec![
                Upload {
                    source: TensorSource {
                        path: path.clone(),
                        offset: source_offset + a,
                        bytes: b,
                    },
                    buffer: buffers["w.b"].clone(),
                },
                Upload {
                    source: TensorSource {
                        path: path.clone(),
                        offset: source_offset,
                        bytes: a,
                    },
                    buffer: buffers["w.a"].clone(),
                },
            ]
        };
        let stats = run(&session, uploads()).unwrap();
        assert_eq!(stats.chunks, 2);
        assert_eq!(stats.bytes, a + b);
        session.trim_pinned_cache();
        assert_eq!(
            session.download(&buffers["w.a"], 0, a).unwrap(),
            data[source_offset..source_offset + a]
        );
        assert_eq!(
            session.download(&buffers["w.b"], 0, b).unwrap(),
            data[source_offset + a..source_offset + a + b]
        );
        session.trim_pinned_cache();
        let bad = directory.path().join("z.bin");
        std::fs::write(&bad, [0u8]).unwrap();
        let mut failing = uploads();
        failing.push(Upload {
            source: TensorSource {
                path: bad,
                offset: 0,
                bytes: a,
            },
            buffer: buffers["w.a"].clone(),
        });
        assert!(run(&session, failing).unwrap_err().contains("truncated"));
        assert_eq!(fixture.pending_jobs(), 0);
        run(&session, uploads()).unwrap();
        drop(buffers);
        drop(session);
        fixture.assert_clean();
    }
}
