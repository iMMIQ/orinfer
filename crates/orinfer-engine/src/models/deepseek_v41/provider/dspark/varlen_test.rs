//! Fixed-history checks for retained Graph work envelopes and real owner lengths.
//! Run with HCCL_DETERMINISTIC=strict to isolate arithmetic from fast collective ordering.
use super::*;

#[test]
#[ignore = "Requires the A2 package, eight NPUs and HCCL_DETERMINISTIC=strict"]
fn retained_work_replays_changed_owner_lengths_and_inputs() {
    retained_work(false);
}

#[test]
#[ignore = "Requires the A2 package, eight NPUs and HCCL_DETERMINISTIC=strict"]
fn mixed_work_replays_changed_owner_lengths_and_inputs() {
    retained_work(true);
}

fn retained_work(producer: bool) {
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the descriptor retains the shared device lock through worker drop.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0,
    );
    let strict = std::env::var("HCCL_DETERMINISTIC").is_ok_and(|v| v == "strict");
    assert!(
        strict || std::env::var_os("ORINFER_A2_VARLEN_DIAGNOSTIC").is_some(),
        "Use strict communication for numerical acceptance, or explicitly request a fast-mode diagnostic"
    );
    let model = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_MODEL").unwrap());
    let input: Vec<u32> = crate::model::read(&std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_DSPARK_INPUT").unwrap(),
    ))
    .unwrap();
    let report = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_REPORT").unwrap());
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
    let mut requests = vec![];
    for _ in 0..4 {
        let mut request = runtime
            .start_request(
                GenerationInput {
                    input_tokens: input.clone(),
                    images: vec![],
                    audios: vec![],
                    max_new_tokens: if producer { 512 } else { 64 },
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
    requests.reverse();
    let layout = |lengths: &[usize]| {
        let mut view = Layout::verification(
            requests
                .iter()
                .zip(lengths)
                .map(|(r, &tokens)| BatchSegment {
                    slot: r.slot,
                    tokens,
                })
                .collect(),
            runtime.team.rows,
            owners,
        )
        .unwrap();
        if producer {
            view.outputs.retain(|&owner| owner < 3);
            view.output_positions.truncate(lengths[..3].iter().sum());
            view.decoder_roles[3] = super::super::super::replay::Role::Producer;
        }
        view
    };
    let reference = layout(&[4, 4, 4, if producer { 500 } else { 4 }]);
    // This uses the production guarded warmup and coordinated all-rank capture.
    runtime
        .team
        .exchange(Command::Prepare(reference.clone(), true))
        .unwrap();
    let restore = || {
        for request in &requests {
            runtime
                .team
                .exchange(Command::Restore {
                    owner: request.slot,
                    snapshot: 1_000_000 + request.slot as u64,
                })
                .unwrap();
        }
    };
    let mut reports = vec![];
    let cases = if producer {
        [[6, 2, 4, 500], [2, 6, 4, 500], [4, 4, 4, 500]]
    } else {
        [[6, 2, 6, 2], [2, 6, 2, 6], [4, 4, 4, 4]]
    };
    for (trial, lengths) in cases.into_iter().enumerate() {
        let ordinary = layout(&lengths);
        let mut bounded = ordinary.clone();
        assert!(
            bounded
                .bound_work(
                    &reference,
                    runtime.team.model.batch.as_ref().unwrap(),
                    &runtime.team.work_launches
                )
                .unwrap()
        );
        let tokens = requests
            .iter()
            .zip(lengths)
            .map(|(r, n)| {
                let mut values = vec![*r.history.last().unwrap()];
                values.extend((1..n).map(|p| input[(trial * 7 + p) % input.len()]));
                values
            })
            .collect::<Vec<_>>();
        runtime
            .team
            .exchange(Command::Prepare(ordinary.clone(), true))
            .unwrap();
        let mut results = vec![];
        for view in [ordinary.clone(), ordinary, bounded] {
            restore();
            runtime
                .team
                .exchange(Command::BeginVerification(
                    requests
                        .iter()
                        .zip(lengths)
                        .take(if producer { 3 } else { 4 })
                        .map(|(r, n)| (r.slot, n))
                        .collect(),
                ))
                .unwrap();
            let (output, timing) = runtime
                .team
                .forward_with(view, tokens.clone(), true, None, || {})
                .unwrap();
            runtime.team.exchange(Command::EndVerification).unwrap();
            results.push((output, timing));
        }
        assert!(
            results[2].1.replay,
            "Changed owner lengths missed the retained Graph"
        );
        let mut differences = vec![];
        for right in [1, 2] {
            let mut maximum = 0f64;
            let mut variation = 0f64;
            assert_eq!(results[0].0.len(), if producer { 12 } else { 16 });
            assert_eq!(results[right].0.len(), if producer { 12 } else { 16 });
            for (a, b) in results[0].0.iter().zip(&results[right].0) {
                assert_eq!(a.owner, b.owner);
                assert_eq!(a.logits.len(), runtime.vocab());
                assert_eq!(b.logits.len(), runtime.vocab());
                let probability = |logits: &[f32]| {
                    assert!(logits.iter().all(|v| v.is_finite()));
                    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let mut p = logits
                        .iter()
                        .map(|v| (*v as f64 - top).exp())
                        .collect::<Vec<_>>();
                    let sum = p.iter().sum::<f64>();
                    for v in &mut p {
                        *v /= sum;
                    }
                    p
                };
                let (a, b) = (probability(&a.logits), probability(&b.logits));
                let delta = a
                    .iter()
                    .zip(b)
                    .map(|(a, b)| (a - b).abs())
                    .collect::<Vec<_>>();
                maximum = maximum.max(delta.iter().copied().fold(0., f64::max));
                variation = variation.max(delta.iter().sum::<f64>() * 0.5);
            }
            if strict {
                assert!(
                    maximum < 0.005 && variation < 0.02,
                    "Changed probability {maximum}, TV {variation}"
                );
            }
            differences.push((maximum, variation));
        }
        reports.push(
            serde_json::json!({"trial":trial,"lengths":lengths,"producer":producer,
            "maximum_probability_delta":differences[1].0,"maximum_total_variation":differences[1].1,
            "repeat_probability_delta":differences[0].0,"repeat_total_variation":differences[0].1,
            "deterministic":strict,
            "reference_s":results[0].1.seconds,"retained_s":results[2].1.seconds,
            "replay":results[2].1.replay}),
        );
    }
    for request in &mut requests {
        runtime.finish_request(request, false).unwrap();
    }
    std::fs::write(report, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
}
