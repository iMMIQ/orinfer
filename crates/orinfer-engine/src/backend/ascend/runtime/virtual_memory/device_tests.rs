//! Explicit device probe for demand-mapped request state; excluded from CPU checks.
use crate::backend::ascend::runtime::{Limits, Session};
use std::{fs::OpenOptions, os::fd::AsRawFd, path::PathBuf, time::Instant};

#[test]
#[ignore = "Measures KV growth with the A2 runtime and device 0"]
fn batched_demand_growth_on_device() {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: The file retains the GPU lock until the session has dropped.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let layouts: Vec<(usize, usize)> = crate::model::read(&PathBuf::from(
        std::env::var_os("ORINFER_A2_GROWTH_LAYOUT").unwrap(),
    ))
    .unwrap();
    assert!(!layouts.is_empty());
    assert!(layouts.iter().all(|&(bytes, ratio)| bytes > 0 && ratio > 0));
    let output = PathBuf::from(std::env::var_os("ORINFER_A2_GROWTH_REPORT").unwrap());
    let session = Session::new(
        0,
        Limits {
            device_bytes: 1 << 30,
            ..Limits::default()
        },
    )
    .unwrap();
    let mut reports = vec![];
    for rows in [2048, 2049, 32768, 128000] {
        let extents = layouts
            .iter()
            .map(|&(bytes, ratio)| rows / ratio * bytes)
            .collect::<Vec<_>>();
        let mut times = [vec![], vec![]];
        for repeat in 0..7 {
            for (mode, samples) in times.iter_mut().enumerate() {
                let buffers = extents
                    .iter()
                    .map(|&bytes| {
                        let capacity = bytes.div_ceil(2 << 20) * (2 << 20) + (2 << 20);
                        session.reserve(capacity).unwrap()
                    })
                    .collect::<Vec<_>>();
                let addresses = buffers.iter().map(|b| b.pointer()).collect::<Vec<_>>();
                let growth = buffers
                    .iter()
                    .cloned()
                    .zip(extents.iter().copied())
                    .collect::<Vec<_>>();
                let started = Instant::now();
                if mode == 0 {
                    for (buffer, bytes) in &growth {
                        session.grow(buffer, *bytes).unwrap();
                    }
                } else {
                    session.grow_many(&growth).unwrap();
                }
                let elapsed = started.elapsed().as_secs_f64();
                if repeat >= 2 {
                    samples.push(elapsed);
                }
                if repeat == 2 {
                    for buffer in &buffers {
                        assert_eq!(session.download(buffer, 0, 8).unwrap(), [0; 8]);
                        session.upload(buffer, 0, &[73; 8]).unwrap().wait().unwrap();
                    }
                    let extensions = buffers
                        .iter()
                        .zip(&extents)
                        .map(|(b, &n)| (b.clone(), b.physical_extent(n).unwrap() + 13))
                        .collect::<Vec<_>>();
                    session.grow_many(&extensions).unwrap();
                    for ((buffer, &address), (_, end)) in
                        buffers.iter().zip(&addresses).zip(&extensions)
                    {
                        assert_eq!(buffer.pointer(), address);
                        assert_eq!(session.download(buffer, 0, 8).unwrap(), [73; 8]);
                        assert_eq!(session.download(buffer, end - 13, 13).unwrap(), [0; 13]);
                    }
                }
                drop((growth, buffers));
                assert_eq!(session.allocated_bytes(), 0);
            }
        }
        for samples in &mut times {
            samples.sort_by(f64::total_cmp);
        }
        let report = serde_json::json!({
            "rows":rows,"buffers":layouts.len(),"bytes":extents.iter().sum::<usize>(),
            "serial_median_ms":times[0][2]*1000.,"batch_median_ms":times[1][2]*1000.,
            "serial_samples_s":times[0],"batch_samples_s":times[1],
            "speedup":times[0][2]/times[1][2],"stable_addresses":true,"zero_growth":true,"preserved_history":true,
        });
        eprintln!("A2_GROWTH {report}");
        reports.push(report);
        std::fs::write(&output, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
}
