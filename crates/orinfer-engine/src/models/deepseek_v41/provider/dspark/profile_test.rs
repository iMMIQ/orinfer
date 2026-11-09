//! Offline native cost surface: real model execution, sampling and Graph replay.
use super::super::super::*;
use super::cost::{Point, Stage, Work};

#[test]
#[ignore = "Requires the native A2 package, fixed token history and eight NPUs"]
#[allow(clippy::assertions_on_constants)]
fn measured_verification_surface() {
    // Reject accidental timing of unoptimized host-side packet preparation.
    assert!(
        !cfg!(debug_assertions),
        "Measure costs with cargo test --release"
    );
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the file retains the shared device lock until all workers drop.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).unwrap());
    let model = path("ORINFER_A2_DSPARK_MODEL");
    let tokens: Vec<u32> = crate::model::read(&path("ORINFER_A2_DSPARK_INPUT")).unwrap();
    let descriptor: Descriptor = crate::model::read(&model.join("cache/model.json")).unwrap();
    let mut runtime = Runtime::load(
        &model,
        crate::execution::LoadOptions {
            mtp_drafts: Some(0),
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let identity = super::Identity {
        schema_version: 2,
        execution_package: descriptor.execution_package,
        target: descriptor.metadata.target,
        cann: "9.2".into(),
        ranks: runtime.team.ranks(),
    };
    eprintln!(
        "DSPARK_COST_IDENTITY {}",
        serde_json::to_string(&identity).unwrap()
    );
    let mut points = vec![];
    for context in [2048, 100256] {
        let input = tokens
            .iter()
            .copied()
            .cycle()
            .take(context)
            .collect::<Vec<_>>();
        let mut requests = vec![];
        for owner in 0..32 {
            let mut request = runtime
                .start_request(
                    GenerationInput {
                        input_tokens: input.clone(),
                        images: vec![],
                        audios: vec![],
                        max_new_tokens: 64,
                        sampling: crate::sampling::Options {
                            temperature: 1.,
                            seed: 20261002 + owner as u64,
                            ..Default::default()
                        },
                        prefix_hints: vec![],
                    },
                    || false,
                )
                .unwrap();
            request.cache_prompt = false;
            request.cache_generated = false;
            if owner == 0 {
                while request.generated == 0 {
                    runtime
                        .advance_requests(&mut [&mut request], &scheduler::Options::default())
                        .unwrap();
                }
                runtime
                    .team
                    .exchange(Command::Save {
                        owner: request.slot,
                        snapshot: 1_000_000,
                        complete: true,
                    })
                    .unwrap();
            } else {
                let first: &Request = &requests[0];
                request.history.clone_from(&first.history);
                request.offset = first.offset;
                request.generated = first.generated;
                request.prefilling = false;
                runtime
                    .team
                    .exchange(Command::Restore {
                        owner: request.slot,
                        snapshot: 1_000_000,
                    })
                    .unwrap();
            }
            requests.push(request);
        }
        for count in [16, 32] {
            for depth in 1..=5 {
                let layout = Layout::verification(
                    requests[..count]
                        .iter()
                        .map(|r| BatchSegment {
                            slot: r.slot,
                            tokens: depth + 1,
                        })
                        .collect(),
                    runtime.team.rows,
                    32,
                )
                .unwrap();
                // Reuse one frozen prefix; vary current tokens among owners so
                // routing is not collapsed onto one identical token row.
                let input = requests[..count]
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let mut row = vec![*r.history.last().unwrap()];
                        row.extend((0..depth).map(|j| tokens[(i * 7 + j) % tokens.len()]));
                        row
                    })
                    .collect::<Vec<_>>();
                for graph in [false, true] {
                    let mut samples = vec![];
                    let mut setup: f64 = 0.;
                    for trial in 0..7 {
                        for r in &requests[..count] {
                            runtime
                                .team
                                .exchange(Command::Restore {
                                    owner: r.slot,
                                    snapshot: 1_000_000,
                                })
                                .unwrap();
                        }
                        runtime
                            .team
                            .exchange(Command::BeginVerification(
                                requests[..count]
                                    .iter()
                                    .map(|r| (r.slot, depth + 1))
                                    .collect(),
                            ))
                            .unwrap();
                        let references = requests[..count].iter_mut().collect::<Vec<_>>();
                        let readout = packet::draft_readout(&references);
                        let (_, timing) = runtime
                            .team
                            .profile_forward(layout.clone(), input.clone(), readout, graph)
                            .unwrap();
                        runtime.team.exchange(Command::EndVerification).unwrap();
                        assert_eq!(timing.replay, graph);
                        setup = setup.max(timing.preparation);
                        if trial > 0 {
                            samples.push(timing.seconds);
                        }
                    }
                    samples.sort_by(f64::total_cmp);
                    let point = Point {
                        stage: Stage::Target,
                        block: 0,
                        full: false,
                        replay: graph,
                        shared: false,
                        shared_setup_s: 0.,
                        setup_s: setup,
                        work: Work {
                            rows: count * (depth + 1),
                            prefill_rows: 0,
                            context: requests[0].history.len(),
                        },
                        seconds: (samples[2] + samples[3]) * 0.5,
                    };
                    eprintln!("DSPARK_COST {}", serde_json::to_string(&point).unwrap());
                    points.push(point);
                }
            }
        }
        runtime.team.exchange(Command::Evict(1_000_000)).unwrap();
        for mut request in requests {
            runtime.finish_request(&mut request, false).unwrap();
        }
    }
    let profile = serde_json::json!({ "identity": identity, "points": points });
    std::fs::write(
        path("ORINFER_A2_DSPARK_REPORT"),
        serde_json::to_vec_pretty(&profile).unwrap(),
    )
    .unwrap();
}
