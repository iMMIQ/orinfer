//! Paired fixed-history blocks, independently of the adaptive depth policy.
use super::*;

#[test]
#[ignore = "Requires the rejection execution package and eight NPUs"]
fn fixed_history_sampling_cost() {
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the file owns the exclusive experiment lock through worker drop.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let model = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_MODEL").unwrap());
    let input: Vec<u32> = crate::model::read(&std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_DSPARK_INPUT").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_REPORT").unwrap());
    assert!(crate::backend::ascend::sampling::rejection_requested().unwrap());
    let mut runtime = Runtime::load(
        &model,
        crate::execution::LoadOptions {
            mtp_drafts: Some(0),
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let owners = runtime.team.model.batch.as_ref().unwrap().owners;
    let stride = runtime
        .team
        .model
        .sampling
        .as_ref()
        .unwrap()
        .rejection
        .as_ref()
        .unwrap()
        .stride;
    let block = runtime.team.draft_blocks[0];
    let mut reports = vec![];
    let trials = std::env::var("ORINFER_A2_REJECTION_TRIALS")
        .map(|v| v.parse::<usize>().unwrap())
        .unwrap_or(20);
    assert!((8..=256).contains(&trials));
    let batches: &[usize] = if std::env::var_os("ORINFER_A2_REJECTION_PROBE_ONLY").is_some() {
        &[1]
    } else {
        &[1, 16]
    };
    for &batch in batches {
        let mut requests = vec![];
        for _ in 0..batch {
            let mut request = runtime
                .start_request(
                    GenerationInput {
                        input_tokens: input.clone(),
                        images: vec![],
                        audios: vec![],
                        max_new_tokens: 128,
                        sampling: crate::sampling::Options {
                            temperature: 1.,
                            seed: 20261002,
                            ..Default::default()
                        },
                        prefix_hints: vec![],
                    },
                    || false,
                )
                .unwrap();
            request.cache_prompt = false;
            request.cache_generated = false;
            while request.generated == 0 {
                runtime
                    .advance_requests(&mut [&mut request], &scheduler::Options::default())
                    .unwrap();
            }
            runtime
                .team
                .exchange(Command::Save {
                    owner: request.slot,
                    snapshot: 1_000_000 + request.slot as u64,
                    complete: true,
                })
                .unwrap();
            requests.push(request);
        }
        // Reverse owner order without changing frozen histories. All physical
        // proposal rows must follow owner identities rather than packet lanes.
        requests.reverse();
        for trial in 0..trials {
            for rejection in if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                for request in &requests {
                    runtime
                        .team
                        .exchange(Command::Restore {
                            owner: request.slot,
                            snapshot: 1_000_000 + request.slot as u64,
                        })
                        .unwrap();
                }
                for request in &mut requests {
                    request.sampling.seed = 20261002 + (trial * owners + request.slot) as u64;
                    request.mtp.rounds = trial;
                }
                let mut refs = requests.iter_mut().collect::<Vec<_>>();
                let round_steps = refs
                    .iter()
                    .map(|r| {
                        (
                            r.slot,
                            crate::mtp::round_step(r.mtp.rounds, r.generated).unwrap(),
                        )
                    })
                    .collect();
                let started = Instant::now();
                let segments = refs
                    .iter()
                    .map(|r| BatchSegment {
                        slot: r.slot,
                        tokens: block,
                    })
                    .collect();
                let mut draft = Layout::new(
                    segments,
                    (0..batch).collect(),
                    false,
                    owners * block,
                    owners,
                )
                .unwrap();
                draft.role = super::super::super::PacketRole::Draft;
                let mut readout = packet::draft_readout(&refs);
                let probe = batch == 1 && trial == 0;
                readout.full = probe;
                if probe {
                    readout.full_owners = None;
                }
                if rejection {
                    readout.store_draft_distributions = true;
                    readout.sampling_steps(&round_steps);
                    for draw in readout.draws.values_mut() {
                        draw.seed ^= crate::mtp::DRAFT_STREAM;
                    }
                }
                let inputs: Vec<Vec<u32>> = refs
                    .iter()
                    .map(|r| {
                        let mut tokens = vec![128799; block];
                        tokens[0] = *r.history.last().unwrap();
                        tokens
                    })
                    .collect();
                let draft_controls =
                    probe.then(|| readout.draws[&refs[0].slot].controls(0, block).unwrap());
                let (proposals, draft_timing) = runtime
                    .team
                    .forward_with(draft, inputs, readout, None, || {})
                    .unwrap();
                let drafts: Vec<Vec<u32>> = refs
                    .iter()
                    .map(|r| {
                        proposals
                            .iter()
                            .filter(|p| p.owner == r.slot)
                            .take(block)
                            .map(|p| p.token)
                            .collect()
                    })
                    .collect();
                assert!(drafts.iter().all(|d| d.len() == block));
                let segments = refs
                    .iter()
                    .map(|r| BatchSegment {
                        slot: r.slot,
                        tokens: block + 1,
                    })
                    .collect();
                let verify = Layout::verification(segments, runtime.team.rows, owners).unwrap();
                let inputs: Vec<Vec<u32>> = refs
                    .iter()
                    .zip(&drafts)
                    .map(|(r, d)| {
                        let mut values = vec![*r.history.last().unwrap()];
                        values.extend(d);
                        values
                    })
                    .collect();
                let proposal_map = drafts.iter().cloned().enumerate().collect();
                let selected = (0..batch).map(|i| (i, block + 1)).collect::<Vec<_>>();
                let mut readout =
                    packet::target_readout(&refs, false, &proposal_map, runtime.vocab(), &selected)
                        .unwrap();
                readout.full = probe;
                readout.capture_sampled_logits = probe;
                if probe {
                    readout.full_owners = None;
                }
                if rejection {
                    readout.sampling_steps(&round_steps);
                    for (r, tokens) in refs.iter().zip(&drafts) {
                        readout.rejection.insert(
                            r.slot,
                            tokens
                                .iter()
                                .enumerate()
                                .map(|(position, &token)| {
                                    crate::backend::ascend::sampling::RejectionProposal {
                                        row: r.slot * stride + position,
                                        token,
                                        log_uniform: crate::sampling::counter_uniform(
                                            r.sampling.seed,
                                            crate::mtp::ACCEPT_STREAM,
                                            (round_steps[&r.slot] + position) as u64,
                                        )
                                        .ln()
                                            as f32,
                                    }
                                })
                                .collect(),
                        );
                    }
                }
                runtime
                    .team
                    .exchange(Command::BeginVerification(
                        refs.iter().map(|r| (r.slot, block + 1)).collect(),
                    ))
                    .unwrap();
                let target_controls = if rejection {
                    readout.rejection_rows(&verify).unwrap().0
                } else {
                    readout.controls(&verify).unwrap()
                };
                let replay = probe.then(|| (verify.clone(), inputs.clone()));
                let (verified, verify_timing) = runtime
                    .team
                    .forward_with(verify, inputs, readout, None, || {})
                    .unwrap();
                let commits = refs
                    .iter_mut()
                    .zip(&drafts)
                    .map(|(r, d)| {
                        let target = verified
                            .iter()
                            .filter(|p| p.owner == r.slot)
                            .map(|p| p.token)
                            .collect::<Vec<_>>();
                        let confirmed = crate::mtp::greedy_commit(d, &target).unwrap();
                        (r.slot, confirmed.len())
                    })
                    .collect::<Vec<_>>();
                runtime
                    .team
                    .exchange(Command::RollbackVerification(commits.clone()))
                    .unwrap();
                runtime.team.exchange(Command::EndVerification).unwrap();
                let seconds = started.elapsed().as_secs_f64();
                let committed: usize = commits.iter().map(|v| v.1).sum();
                if probe {
                    let mode = if rejection { "rejection" } else { "coupled" };
                    for (name, rows) in [("q", &proposals), ("p", &verified)] {
                        assert!(rows.iter().all(|r| r.logits.len() == runtime.vocab()));
                        let bytes = rows
                            .iter()
                            .flat_map(|r| r.logits.iter().flat_map(|v| v.to_le_bytes()))
                            .collect::<Vec<_>>();
                        std::fs::write(output.with_extension(format!("{mode}-{name}.f32")), bytes)
                            .unwrap();
                    }
                    let controls = target_controls
                        .iter()
                        .flat_map(|r| [r[0], r[1], 0, r[2]])
                        .flat_map(i32::to_le_bytes)
                        .collect::<Vec<_>>();
                    std::fs::write(output.with_extension(format!("{mode}-p.i32")), controls)
                        .unwrap();
                    std::fs::write(
                        output.with_extension(format!("{mode}-q.i32")),
                        draft_controls.unwrap(),
                    )
                    .unwrap();
                    let metadata = serde_json::json!({"draft_tokens":drafts[0],
                        "target_tokens":verified.iter().map(|v|v.token).collect::<Vec<_>>(),
                        "accept_log_uniform":(0..block).map(|position|crate::sampling::counter_uniform(
                            refs[0].sampling.seed, crate::mtp::ACCEPT_STREAM,
                            (round_steps[&refs[0].slot]+position) as u64).ln() as f32).collect::<Vec<_>>()});
                    std::fs::write(
                        output.with_extension(format!("{mode}.json")),
                        serde_json::to_vec_pretty(&metadata).unwrap(),
                    )
                    .unwrap();
                    if std::env::var_os("ORINFER_A2_REJECTION_PROBE_ONLY").is_some() {
                        let (view, inputs) = replay.unwrap();
                        for repeat in 0..2 {
                            runtime
                                .team
                                .exchange(Command::Restore {
                                    owner: refs[0].slot,
                                    snapshot: 1_000_000 + refs[0].slot as u64,
                                })
                                .unwrap();
                            runtime
                                .team
                                .exchange(Command::BeginVerification(vec![(
                                    refs[0].slot,
                                    block + 1,
                                )]))
                                .unwrap();
                            let (raw, _, _) = runtime
                                .team
                                .forward(view.clone(), inputs.clone(), true)
                                .unwrap();
                            runtime.team.exchange(Command::EndVerification).unwrap();
                            let bytes = raw[0]
                                .logits
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>();
                            std::fs::write(
                                output.with_extension(format!("{mode}-repeat-{repeat}.f32")),
                                bytes,
                            )
                            .unwrap();
                        }
                    }
                }
                let report = serde_json::json!({"batch":batch,"trial":trial,"warm":trial>=4,
                    "sampler":if rejection {"rejection"} else {"coupled"},"drafts_per_owner":block,
                    "proposed":batch*block,"accepted":committed-batch,"committed":committed,
                    "round_s":seconds,"draft_s":draft_timing.seconds,"verify_s":verify_timing.seconds,
                    "accepted_per_owner":commits.iter().map(|v|v.1-1).collect::<Vec<_>>(),
                    "preparation_s":draft_timing.preparation+verify_timing.preparation,
                    "aggregate_tps":committed as f64/seconds});
                eprintln!("DSPARK_FIXED_SAMPLING {report}");
                reports.push(report);
                std::fs::write(&output, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
            }
        }
        for request in &mut requests {
            runtime
                .team
                .exchange(Command::Restore {
                    owner: request.slot,
                    snapshot: 1_000_000 + request.slot as u64,
                })
                .unwrap();
            runtime
                .team
                .exchange(Command::Evict(1_000_000 + request.slot as u64))
                .unwrap();
            runtime.finish_request(request, false).unwrap();
        }
    }
}
