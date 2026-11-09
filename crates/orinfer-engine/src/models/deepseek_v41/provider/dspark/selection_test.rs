//! Check device selection against the same forward's logits, including mixed owners.
use super::*;

#[test]
#[ignore = "Requires the A2 package and eight NPUs"]
fn mixed_temperatures_select_from_the_same_forward_logits() {
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: this descriptor retains the shared device lock through worker drop.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let path = |name| std::path::PathBuf::from(std::env::var_os(name).unwrap());
    let input: Vec<u32> = crate::model::read(&path("ORINFER_A2_DSPARK_INPUT")).unwrap();
    let mut runtime = Runtime::load(
        &path("ORINFER_A2_DSPARK_MODEL"),
        crate::execution::LoadOptions {
            mtp_drafts: Some(0),
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut requests = vec![];
    for temperature in [0., 1., 0., 0.7] {
        let mut request = runtime
            .start_request(
                GenerationInput {
                    input_tokens: input.clone(),
                    images: vec![],
                    audios: vec![],
                    max_new_tokens: 64,
                    sampling: crate::sampling::Options {
                        temperature,
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
    let layout = Layout::verification(
        requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.slot,
                tokens: 3,
            })
            .collect(),
        runtime.team.rows,
        runtime.team.model.batch.as_ref().unwrap().owners,
    )
    .unwrap();
    runtime
        .team
        .exchange(Command::Prepare(layout.clone(), true))
        .unwrap();
    let mut checks = vec![];
    for phase in 0..2 {
        if phase == 1 {
            for request in &mut requests {
                request.sampling.temperature = if request.sampling.temperature == 0. {
                    1.
                } else {
                    0.
                };
            }
        }
        for request in &requests {
            runtime
                .team
                .exchange(Command::Restore {
                    owner: request.slot,
                    snapshot: 1_000_000 + request.slot as u64,
                })
                .unwrap();
        }
        runtime
            .team
            .exchange(Command::BeginVerification(
                requests.iter().map(|r| (r.slot, 3)).collect(),
            ))
            .unwrap();
        let tokens = requests
            .iter()
            .enumerate()
            .map(|(i, r)| {
                vec![
                    *r.history.last().unwrap(),
                    input[(phase * 7 + i + 1) % input.len()],
                    input[(phase * 11 + i + 2) % input.len()],
                ]
            })
            .collect();
        let mut readout = packet::draft_readout(&requests.iter_mut().collect::<Vec<_>>());
        assert_eq!(readout.draws.len(), requests.len());
        readout.full = true;
        readout.full_owners = None;
        readout.capture_sampled_logits = true;
        let (outputs, timing) = runtime
            .team
            .forward_with(layout.clone(), tokens, readout, None, || {})
            .unwrap();
        runtime.team.exchange(Command::EndVerification).unwrap();
        assert!(timing.replay);
        let mut positions = BTreeMap::<usize, usize>::new();
        for output in &outputs {
            let request = requests.iter().find(|r| r.slot == output.owner).unwrap();
            let position = positions.entry(output.owner).or_default();
            assert_eq!(output.logits.len(), 129280);
            assert!(output.logits.iter().all(|v| v.is_finite()));
            let control = crate::backend::ascend::sampling::Draw {
                seed: request.sampling.seed,
                step: request.generated,
                temperature: request.sampling.temperature,
            }
            .control(*position)
            .unwrap();
            let inverse = f32::from_bits(control[2] as u32);
            let score = |token: usize| {
                let logit = output.logits[token];
                if inverse == 0. {
                    return logit;
                }
                let bits = crate::sampling::categorical_bits(
                    [token as u32, control[1] as u32],
                    control[0] as u32,
                )[0];
                let uniform = ((bits >> 9) as f32 + 0.5) * 2.0_f32.powi(-23);
                logit * inverse - (-uniform.ln()).ln()
            };
            let expected = (0..output.logits.len())
                .max_by(|&a, &b| score(a).total_cmp(&score(b)).then_with(|| b.cmp(&a)))
                .unwrap();
            let actual = output.token as usize;
            assert!(actual < output.logits.len());
            if inverse == 0. {
                assert_eq!(actual, expected);
            }
            let score_error = (score(expected) - score(actual)).abs();
            assert!(score_error <= 3e-5, "row winner differs: {score_error}");
            assert!((output.value - score(actual)).abs() <= 3e-5);
            checks.push(serde_json::json!({"phase":phase,"owner":output.owner,"position":position,"temperature":request.sampling.temperature,"token":actual,"expected":expected,"score_error":score_error,"graph":timing.replay}));
            *position += 1;
        }
        assert_eq!(outputs.len(), 12);
    }
    std::fs::write(
        path("ORINFER_A2_DSPARK_REPORT"),
        serde_json::to_vec_pretty(&checks).unwrap(),
    )
    .unwrap();
    for mut request in requests {
        runtime
            .team
            .exchange(Command::Evict(1_000_000 + request.slot as u64))
            .unwrap();
        runtime.finish_request(&mut request, false).unwrap();
    }
}
