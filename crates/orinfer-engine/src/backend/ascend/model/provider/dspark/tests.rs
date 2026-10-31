use super::*;

#[test]
#[ignore = "Requires the DSpark A2 package and eight NPUs"]
fn block_draft_matches_target_with_prefix_restore_and_cancellation() {
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the descriptor owns the GPU lock until all native workers drop.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let model = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_MODEL").unwrap());
    let input: Vec<u32> = crate::model::read(&std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_DSPARK_INPUT").unwrap(),
    ))
    .unwrap();
    let mut runtime = Runtime::load(
        &model,
        crate::execution::LoadOptions {
            mtp_drafts: Some(5),
            prefix_cache_bytes: 2 << 30,
            ..Default::default()
        },
    )
    .unwrap();
    let options = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut expected = None;
    let mut reports = vec![];
    for drafts in [0, 5, 0, 5] {
        runtime.draft_limit = drafts;
        let mut tokens = vec![];
        let mut first = None;
        let mut last = Instant::now();
        runtime
            .generate(
                &input,
                64,
                &options,
                || false,
                |token| {
                    let now = Instant::now();
                    first.get_or_insert(now);
                    last = now;
                    tokens.push(token);
                    true
                },
            )
            .unwrap();
        if let Some(reference) = &expected {
            assert_eq!(&tokens, reference);
        } else {
            expected = Some(tokens.clone());
        }
        let seconds = last.duration_since(first.unwrap()).as_secs_f64();
        let report = serde_json::json!({"drafts": drafts, "decode_tokens": tokens.len() - 1,
            "decode_s": seconds, "decode_tps": (tokens.len() - 1) as f64 / seconds,
            "speculation": runtime.speculation});
        eprintln!("DSPARK {:?}", report);
        reports.push(report);
    }
    // A cancelled owner must release its private rings and proposal workspace.
    let mut count = 0;
    runtime
        .generate(
            &input,
            16,
            &options,
            || false,
            |_| {
                count += 1;
                count < 3
            },
        )
        .unwrap();
    let mut replay = vec![];
    runtime
        .generate(
            &input,
            16,
            &options,
            || false,
            |token| {
                replay.push(token);
                true
            },
        )
        .unwrap();
    let expected = expected.unwrap();
    assert_eq!(replay, expected[..16]);
    assert!(runtime.speculation.rounds > 0);
    assert!(runtime.speculation.proposed_tokens > 0);
    // Force every accepted-prefix length, independently of draft quality.
    runtime.draft_limit = 0;
    let mut request = runtime
        .start_request(
            GenerationInput {
                input_tokens: input.clone(),
                images: vec![],
                audios: vec![],
                max_new_tokens: 64,
                sampling: options.clone(),
                prefix_hints: vec![],
            },
            || false,
        )
        .unwrap();
    while request.generated == 0 {
        runtime
            .advance_requests(&mut [&mut request], &scheduler::Options::default())
            .unwrap();
    }
    let slot = request.slot;
    let owners = runtime.team.model.batch.as_ref().unwrap().owners;
    runtime
        .team
        .exchange(Command::Save {
            owner: slot,
            snapshot: u64::MAX,
            complete: true,
        })
        .unwrap();
    for keep in 1..=6 {
        runtime
            .team
            .exchange(Command::Restore {
                owner: slot,
                snapshot: u64::MAX,
            })
            .unwrap();
        runtime
            .team
            .exchange(Command::BeginVerification(vec![slot]))
            .unwrap();
        let verify = Layout::verification(
            vec![BatchSegment { slot, tokens: 6 }],
            runtime.team.rows,
            owners,
        )
        .unwrap();
        runtime
            .team
            .forward(verify, vec![expected[..6].to_vec()], false)
            .unwrap();
        runtime
            .team
            .exchange(Command::RollbackVerification(vec![(slot, keep)]))
            .unwrap();
        runtime.team.exchange(Command::EndVerification).unwrap();
        let next = Layout::new(
            vec![BatchSegment { slot, tokens: 1 }],
            vec![0],
            true,
            runtime.team.rows,
            owners,
        )
        .unwrap();
        let (outputs, _, _) = runtime
            .team
            .forward(next, vec![vec![expected[keep]]], false)
            .unwrap();
        assert_eq!(
            outputs[0].token,
            expected[keep + 1],
            "accepted prefix {keep}"
        );
    }
    runtime.team.exchange(Command::Evict(u64::MAX)).unwrap();
    runtime.finish_request(&mut request, false).unwrap();
    eprintln!("DSPARK_STATE accepted-prefix lengths 1..6 passed");
    let mut short_reference = None;
    for drafts in [0, 5, 0, 5] {
        runtime.draft_limit = drafts;
        let before = runtime.speculation.accepted_draft_tokens;
        let mut tokens = vec![];
        runtime
            .generate(
                &input[..128],
                64,
                &options,
                || false,
                |t| {
                    tokens.push(t);
                    true
                },
            )
            .unwrap();
        if let Some(reference) = &short_reference {
            assert_eq!(&tokens, reference);
        } else {
            short_reference = Some(tokens);
        }
        eprintln!(
            "DSPARK_SHORT drafts={drafts} accepted={}",
            runtime.speculation.accepted_draft_tokens - before
        );
    }
    // Paired mixed lengths exercise owner addressing and the 2048/2049 tail.
    let mut concurrent_reference = None;
    for drafts in [0, 5] {
        runtime.draft_limit = drafts;
        let mut requests = (0..3)
            .map(|owner| {
                let mut prompt = input.clone();
                if owner != 0 {
                    prompt.push(if owner == 1 { 223 } else { 678 });
                }
                runtime
                    .start_request(
                        GenerationInput {
                            input_tokens: prompt,
                            images: vec![],
                            audios: vec![],
                            max_new_tokens: 32,
                            sampling: options.clone(),
                            prefix_hints: vec![],
                        },
                        || false,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut outputs = vec![vec![]; 3];
        while requests.iter().any(|r| !r.is_finished()) {
            let mut refs = requests.iter_mut().collect::<Vec<_>>();
            for step in runtime
                .advance_requests(&mut refs, &scheduler::Options::default())
                .unwrap()
            {
                outputs[step.request].extend(step.tokens);
            }
        }
        for request in &mut requests {
            runtime.finish_request(request, true).unwrap();
        }
        assert!(outputs.iter().all(|tokens| tokens.len() == 32));
        if let Some(reference) = &concurrent_reference {
            assert_eq!(&outputs, reference);
        } else {
            concurrent_reference = Some(outputs);
        }
        eprintln!("DSPARK_BATCH drafts={drafts} B=3 lengths=2048,2049,2049 passed");
    }
    std::fs::write(
        "artifacts/dspark/end-to-end.json",
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .unwrap();
}
