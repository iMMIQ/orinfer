//! Native multi-device collective checks with a coordinated preflight gate.
use super::{Communicator, Counts, Element, RootInfo};
use crate::{
    artifact::Result,
    backend::ascend::runtime::{Limits, Session},
};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    sync::mpsc,
    time::{Duration, Instant},
};

#[derive(Serialize)]
pub struct Report {
    ranks: Vec<Rank>,
    scope: &'static str,
}
#[derive(Serialize)]
struct Rank {
    rank: u32,
    device: u32,
    all_reduce_verified: bool,
    all_gather_verified: bool,
    all_to_all_v_verified: bool,
    reduce_scatter_graph_verified: bool,
    initialize_ms: f64,
}

pub fn run(devices: &[u32]) -> Result<Report> {
    if !(2..=8).contains(&devices.len())
        || devices.iter().copied().collect::<BTreeSet<_>>().len() != devices.len()
    {
        return Err("Select 2..8 distinct Ascend devices for HCCL validation".into());
    }
    let bootstrap = Session::new(devices[0], Limits::default())?;
    let root = RootInfo::new(&bootstrap, devices.len() as u32)?;
    let ranks = std::thread::scope(|scope| {
        let (ready_send, ready_receive) = mpsc::channel();
        let mut workers = vec![];
        for (rank, &device) in devices.iter().enumerate() {
            let (start, gate) = mpsc::channel();
            let ready = ready_send.clone();
            let root = root.clone();
            let worker = scope.spawn(move || {
                let prepared = Session::new(device, Limits::default()).and_then(|session| {
                    if session.memory_info()?.0 < 512 << 20 {
                        return Err("HCCL validation needs 512 MiB free HBM per device".into());
                    }
                    Ok(session)
                });
                ready
                    .send(prepared.as_ref().map(|_| ()).map_err(Clone::clone))
                    .map_err(|e| e.to_string())?;
                let session = prepared?;
                if !gate.recv().map_err(|e| e.to_string())? {
                    return Err("HCCL preflight failed on a peer".into());
                }
                validate_rank(&session, &root, rank as u32, device)
            });
            workers.push((worker, start));
        }
        drop(ready_send);
        let mut preflight = Ok(());
        for _ in devices {
            match ready_receive.recv_timeout(Duration::from_secs(30)) {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    if preflight.is_ok() {
                        preflight = Err(error);
                    }
                }
                Err(error) => {
                    preflight = Err(format!("HCCL preflight: {error}"));
                    break;
                }
            }
        }
        for (_, start) in &workers {
            let _ = start.send(preflight.is_ok());
        }
        let mut results = vec![];
        for (worker, _) in workers {
            results.push(
                worker
                    .join()
                    .map_err(|_| "HCCL validation worker panicked".to_string())
                    .and_then(|r| r),
            );
        }
        preflight?;
        results.into_iter().collect::<Result<Vec<_>>>()
    })?;
    Ok(Report {
        ranks,
        scope: "Native HCCL AllReduce, AllGather, ReduceScatter with changed-input Graph replay and variable-count AlltoAllV across device contexts; no model execution",
    })
}
fn count(source: u32, destination: u32) -> u64 {
    if (source + 2 * destination).is_multiple_of(4) {
        0
    } else {
        u64::from((source + destination) % 3 + 1)
    }
}
fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect()
}
fn validate_rank(session: &Session, root: &RootInfo, rank: u32, device: u32) -> Result<Rank> {
    let start = Instant::now();
    let comm = Communicator::join(session, root, rank)?;
    let initialize_ms = start.elapsed().as_secs_f64() * 1000.;
    let a = session.allocate(128)?;
    let b = session.allocate(128)?;
    let input: Vec<u8> = (0..32)
        .flat_map(|_| (rank as f32 + 1.).to_le_bytes())
        .collect();
    let upload = session.upload(&a, 0, &input)?;
    session.wait_on_compute(&upload)?;
    comm.all_reduce_sum(session, &a, &b, 32, Element::Float32)?
        .wait()?;
    let expected = (root.ranks() * (root.ranks() + 1) / 2) as f32;
    if values(&session.download(&b, 0, 128)?)
        .iter()
        .any(|&v| v != expected)
    {
        return Err(format!("HCCL AllReduce mismatch on rank {rank}"));
    }
    let gathered = session.allocate(128 * root.ranks() as usize)?;
    comm.all_gather(session, &a, &gathered, 32, Element::Float32)?
        .wait()?;
    let actual = values(&session.download(&gathered, 0, gathered.bytes())?);
    if actual
        .iter()
        .enumerate()
        .any(|(i, &v)| v != (i / 32 + 1) as f32)
    {
        return Err(format!("HCCL AllGather mismatch on rank {rank}"));
    }
    let sends: Vec<u64> = (0..root.ranks()).map(|peer| count(rank, peer)).collect();
    let receives: Vec<u64> = (0..root.ranks()).map(|peer| count(peer, rank)).collect();
    let send = Counts::new(sends)?;
    let receive = Counts::new(receives)?;
    let input: Vec<f32> = (0..root.ranks())
        .flat_map(|peer| {
            (0..count(rank, peer)).map(move |i| (rank * 10000 + peer * 100 + i as u32) as f32)
        })
        .collect();
    let expected: Vec<f32> = (0..root.ranks())
        .flat_map(|peer| {
            (0..count(peer, rank)).map(move |i| (peer * 10000 + rank * 100 + i as u32) as f32)
        })
        .collect();
    let a = session.allocate((send.elements() as usize * 4).max(4))?;
    let b = session.allocate((receive.elements() as usize * 4).max(4))?;
    let input: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    if !input.is_empty() {
        session.wait_on_compute(&session.upload(&a, 0, &input)?)?;
    }
    comm.all_to_all_v(session, &a, &b, send, receive, Element::Float32)?
        .wait()?;
    if !expected.is_empty() && values(&session.download(&b, 0, expected.len() * 4)?) != expected {
        return Err(format!("HCCL AlltoAllV mismatch on rank {rank}"));
    }
    comm.check_async_error()?;
    validate_scatter(session, &comm, rank)?;
    Ok(Rank {
        rank,
        device,
        all_reduce_verified: true,
        all_gather_verified: true,
        all_to_all_v_verified: true,
        reduce_scatter_graph_verified: true,
        initialize_ms,
    })
}

fn scatter_input(rank: u32, elements: usize, step: u32, element: Element) -> Vec<u8> {
    use half::bf16;
    let mut output = Vec::with_capacity(elements * element.bytes() as usize);
    let mixed = std::env::var_os("ORINFER_ASCEND_SCATTER_MIXED").is_some();
    for i in 0..elements {
        let bits = (i as u32)
            .wrapping_mul(1664525)
            .wrapping_add(rank.wrapping_mul(1013904223))
            .wrapping_add(step.wrapping_mul(747796405));
        let value = if mixed {
            let value = ((bits & 65535) as i32 - 32768) as f32 / 1024.;
            if bits & 0x40000 == 0 {
                value
            } else {
                value / 1024.
            }
        } else {
            // Every intermediate sum is exactly representable even in BF16.
            ((bits >> 16) % 15) as f32 - 7.
        };
        match element {
            Element::Bfloat16 => {
                output.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
            }
            _ => output.extend_from_slice(&value.to_le_bytes()),
        }
    }
    output
}

fn validate_scatter(session: &Session, comm: &Communicator, rank: u32) -> Result<()> {
    use crate::backend::ascend::graph::Graph;
    // The complete 2K/TP8/hidden5120 geometry used by the model's reductions.
    let count = 256 * 5120;
    let mut differences = Vec::new();
    for element in [Element::Float32, Element::Bfloat16] {
        let bytes = count * element.bytes() as usize;
        let total = bytes * comm.ranks() as usize;
        let source = session.allocate(total)?;
        let all = session.allocate(total)?;
        let partition = session.allocate(bytes)?;
        let direct = session.allocate(bytes)?;
        let gathered = session.allocate(total)?;
        let input = scatter_input(rank, count * comm.ranks() as usize, 0, element);
        upload_large(session, &source, &input)?;
        comm.all_reduce_sum(
            session,
            &source,
            &all,
            (count * comm.ranks() as usize) as u64,
            element,
        )?
        .wait()?;
        comm.reduce_scatter_sum(session, &source, &partition, count as u64, element)?
            .wait()?;

        let check = |step: u32| -> Result<()> {
            let expected = session.download(&all, rank as usize * bytes, bytes)?;
            let actual = session.download(&partition, 0, bytes)?;
            if let Some(directory) = std::env::var_os("ORINFER_ASCEND_SCATTER_DUMP") {
                let directory = std::path::Path::new(&directory);
                std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
                for (kind, data) in [("allreduce", &expected), ("scatter", &actual)] {
                    std::fs::write(
                        directory.join(format!("r{rank}-{element:?}-{step}-{kind}.bin")),
                        data,
                    )
                    .map_err(|e| e.to_string())?;
                }
            }
            if expected != actual {
                let mismatches = expected
                    .chunks_exact(element.bytes() as usize)
                    .zip(actual.chunks_exact(element.bytes() as usize))
                    .filter(|(a, b)| a != b)
                    .count();
                return Err(format!(
                    "ReduceScatter {element:?} differs from AllReduce on rank {rank}: {mismatches}/{count} elements"
                ));
            }
            Ok(())
        };
        if let Err(error) = check(0) {
            differences.push(error);
        }
        comm.validate_scatter(session, &source, &partition, count as u64, element)?;
        // SAFETY: The checked same-context bindings and communicator are held
        // by the captured graph and all its replay completion events.
        let graph = unsafe {
            Graph::capture(
                session,
                Box::new((comm.clone(), source.clone(), partition.clone())),
                |stream| comm.scatter_on_stream(stream, &source, &partition, count as u64, element),
            )
        }?;
        for step in [1, 2] {
            let input = scatter_input(rank, count * comm.ranks() as usize, step, element);
            upload_large(session, &source, &input)?;
            comm.all_reduce_sum(
                session,
                &source,
                &all,
                (count * comm.ranks() as usize) as u64,
                element,
            )?
            .wait()?;
            comm.reduce_scatter_sum(session, &source, &direct, count as u64, element)?
                .wait()?;
            graph.enqueue(session)?.wait()?;
            if session.download(&direct, 0, bytes)? != session.download(&partition, 0, bytes)? {
                differences.push(format!(
                    "ReduceScatter {element:?} Graph differs from direct execution on rank {rank}"
                ));
            }
            if let Err(error) = check(step) {
                differences.push(error);
            }
            comm.all_gather(session, &partition, &gathered, count as u64, element)?
                .wait()?;
            if read_large(session, &gathered)? != read_large(session, &all)? {
                differences.push(format!(
                    "ReduceScatter/AllGather {element:?} mismatch on rank {rank}"
                ));
            }
        }
        comm.check_async_error()?;
    }
    if differences.is_empty() {
        Ok(())
    } else {
        Err(differences.join("; "))
    }
}

fn upload_large(
    session: &Session,
    buffer: &super::super::runtime::Buffer,
    bytes: &[u8],
) -> Result<()> {
    const CHUNK: usize = 8 << 20;
    for (index, chunk) in bytes.chunks(CHUNK).enumerate() {
        session.upload(buffer, index * CHUNK, chunk)?.wait()?;
    }
    Ok(())
}

fn read_large(session: &Session, buffer: &super::super::runtime::Buffer) -> Result<Vec<u8>> {
    const CHUNK: usize = 8 << 20;
    let mut bytes = Vec::with_capacity(buffer.bytes());
    for offset in (0..buffer.bytes()).step_by(CHUNK) {
        bytes.extend(session.download(buffer, offset, (buffer.bytes() - offset).min(CHUNK))?);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "Requires exclusive GPU experiment lock and ORINFER_ASCEND_TEST_DEVICES"]
    fn native_multi_device_collectives() {
        let devices = std::env::var("ORINFER_ASCEND_TEST_DEVICES")
            .expect("select explicit Ascend devices")
            .split(',')
            .map(|d| d.parse().unwrap())
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string(&super::run(&devices).unwrap()).unwrap()
        );
    }
}
