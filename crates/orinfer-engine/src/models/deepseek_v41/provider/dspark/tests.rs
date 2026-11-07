use super::*;

// Compare probability bounds while history is identical. Free-running branches
// and the ordinary repeated control are recorded independently of state checks.
// Outside both top-20 sets each mass is bounded by its set's smallest mass.
fn compare_fixed_prefix(reference: &[Vec<TokenLogprob>], actual: &[Vec<TokenLogprob>]) -> f64 {
    assert_eq!(reference.len(), actual.len());
    let mut maximum = 0f64;
    let mut positions = 0;
    for (owner, (expected, observed)) in reference.iter().zip(actual).enumerate() {
        assert_eq!(expected.len(), observed.len());
        for (position, (a, b)) in expected.iter().zip(observed).enumerate() {
            assert_eq!(a.top.len(), 20);
            assert_eq!(b.top.len(), 20);
            let p = a
                .top
                .iter()
                .map(|&(id, lp)| (id, lp.exp()))
                .collect::<BTreeMap<_, _>>();
            let q = b
                .top
                .iter()
                .map(|&(id, lp)| (id, lp.exp()))
                .collect::<BTreeMap<_, _>>();
            let tail = a
                .top
                .last()
                .unwrap()
                .1
                .exp()
                .max(b.top.last().unwrap().1.exp());
            let error = p
                .keys()
                .chain(q.keys())
                .map(|id| {
                    (p.get(id).copied().unwrap_or(0.) - q.get(id).copied().unwrap_or(0.)).abs()
                })
                .fold(tail, f64::max);
            maximum = maximum.max(error);
            positions += 1;
            if a.token != b.token {
                eprintln!(
                    "DSPARK_GENERATION_BRANCH owner={owner} position={position} tokens={}/{} probability_error_bound={error}",
                    a.token, b.token
                );
                break; // Later free-running histories are different.
            }
        }
    }
    eprintln!("DSPARK_FIXED_HISTORY positions={positions} max_probability_error_bound={maximum}");
    maximum
}
fn logprob_rows(records: &[Vec<TokenLogprob>]) -> serde_json::Value {
    serde_json::json!(
        records
            .iter()
            .map(|owner| owner
                .iter()
                .map(|p| serde_json::json!({"token":p.token,"logprob":p.logprob,"top":p.top}))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}

#[test]
#[ignore = "Measures warmed DSpark concurrency; requires the A2 package and eight NPUs"]
fn concurrent_decode_throughput() {
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open("artifacts/gpu-experiment.lock")
        .unwrap();
    assert_eq!(
        // SAFETY: the descriptor retains the GPU lock through worker teardown.
        unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let model = std::path::PathBuf::from(std::env::var_os("ORINFER_A2_DSPARK_MODEL").unwrap());
    let input: Vec<u32> = crate::model::read(&std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_DSPARK_INPUT").unwrap(),
    ))
    .unwrap();
    let output = std::path::PathBuf::from(
        std::env::var_os("ORINFER_A2_DSPARK_REPORT")
            .unwrap_or_else(|| "artifacts/dspark/concurrency.json".into()),
    );
    let mut runtime = Runtime::load(
        &model,
        crate::execution::LoadOptions {
            mtp_drafts: None,
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let options = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let cases = |name: &str, default: &str| {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|v| v.parse::<usize>().unwrap())
            .collect::<Vec<_>>()
    };
    let batches = cases("ORINFER_A2_DSPARK_BATCHES", "1,2,4,8,16,32");
    let depths = cases("ORINFER_A2_DSPARK_DEPTHS", "0,5,0,5");
    assert!(
        batches
            .iter()
            .all(|&b| (1..=runtime.team.model.batch.as_ref().unwrap().owners).contains(&b))
    );
    assert!(
        depths
            .iter()
            .all(|&d| d <= runtime.team.draft_blocks.last().copied().unwrap_or(0))
    );
    let mut reports = vec![];
    for batch in batches {
        let mut expected = None;
        for (round, &drafts) in depths.iter().enumerate() {
            runtime.draft_limit = drafts;
            let mut requests = vec![];
            let mut outputs = vec![vec![]; batch];
            // Prefill each owner to its first output before timing the cohort.
            // Earlier owners stay idle, so all start decode at the same position.
            for tokens in &mut outputs {
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
                    for step in runtime
                        .advance_requests(&mut [&mut request], &scheduler::Options::default())
                        .unwrap()
                    {
                        tokens.extend(step.tokens);
                    }
                }
                assert_eq!(tokens.len(), 1);
                requests.push(request);
            }
            let proposed = runtime.speculation.proposed_tokens;
            let accepted = runtime.speculation.accepted_draft_tokens;
            let mut finished_s = vec![None; batch];
            let started = Instant::now();
            while requests.iter().any(|r| !r.is_finished()) {
                let mut refs = requests.iter_mut().collect::<Vec<_>>();
                for step in runtime
                    .advance_requests(&mut refs, &scheduler::Options::default())
                    .unwrap()
                {
                    outputs[step.request].extend(step.tokens);
                }
                let elapsed = started.elapsed().as_secs_f64();
                for (request, finished) in requests.iter().zip(&mut finished_s) {
                    if request.is_finished() {
                        finished.get_or_insert(elapsed);
                    }
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            for request in &mut requests {
                runtime.finish_request(request, false).unwrap();
            }
            assert!(outputs.iter().all(|tokens| tokens.len() == 64));
            if let Some(reference) = &expected {
                assert_eq!(&outputs, reference);
            } else {
                expected = Some(outputs);
            }
            let report = serde_json::json!({
                "batch": batch, "drafts": drafts, "warm": round >= depths.len().div_ceil(2),
                "prompt_tokens_per_request": input.len(), "output_tokens_per_request": 64,
                "decode_tokens": batch * 63, "decode_s": seconds,
                "aggregate_decode_tps": (batch * 63) as f64 / seconds,
                "per_request_decode_tps": finished_s.into_iter()
                    .map(|s| 63. / s.unwrap()).collect::<Vec<_>>(),
                "proposed": runtime.speculation.proposed_tokens - proposed,
                "accepted": runtime.speculation.accepted_draft_tokens - accepted,
            });
            eprintln!("DSPARK_CONCURRENCY {report}");
            reports.push(report);
            std::fs::write(&output, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
        }
    }
}

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
            mtp_drafts: None,
            prefix_cache_bytes: 2 << 30,
            ..Default::default()
        },
    )
    .unwrap();
    let maximum = runtime.team.draft_blocks.last().copied().unwrap_or(0);
    let verification_rows = maximum + 1;
    let options = crate::sampling::Options {
        temperature: 0.,
        ..Default::default()
    };
    let mut expected = None;
    let mut reports = vec![];
    for drafts in [0, maximum, 0, maximum] {
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
    for keep in 1..=verification_rows {
        runtime
            .team
            .exchange(Command::Restore {
                owner: slot,
                snapshot: u64::MAX,
            })
            .unwrap();
        runtime
            .team
            .exchange(Command::BeginVerification(vec![(slot, verification_rows)]))
            .unwrap();
        let verify = Layout::verification(
            vec![BatchSegment {
                slot,
                tokens: verification_rows,
            }],
            runtime.team.rows,
            owners,
        )
        .unwrap();
        runtime
            .team
            .forward(verify, vec![expected[..verification_rows].to_vec()], false)
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
    eprintln!("DSPARK_STATE accepted-prefix lengths 1..{verification_rows} passed");
    let mut short_reference = None;
    for drafts in [0, maximum, 0, maximum] {
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
    let mut concurrent_reference: Option<Vec<Vec<TokenLogprob>>> = None;
    let mut batch_options = options.clone();
    batch_options.top_logprobs = Some(20);
    for (repetition, drafts) in [0, 0, maximum].into_iter().enumerate() {
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
                            sampling: batch_options.clone(),
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
        let records = requests
            .iter_mut()
            .map(|r| r.take_logprobs())
            .collect::<Vec<_>>();
        std::fs::write(
            format!("artifacts/dspark/mixed-b3-control-{repetition}-{drafts}.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outputs":outputs,"logprobs":logprob_rows(&records)}),
            )
            .unwrap(),
        )
        .unwrap();
        for request in &mut requests {
            runtime.finish_request(request, true).unwrap();
        }
        assert!(outputs.iter().all(|tokens| tokens.len() == 32));
        if let Some(reference) = &concurrent_reference {
            compare_fixed_prefix(reference, &records);
        } else {
            concurrent_reference = Some(records);
        }
        eprintln!(
            "DSPARK_BATCH drafts={drafts} B=3 lengths=2048,2049,2049 completed; probabilities recorded"
        );
    }
    // Decode verification and newly arriving prompts share the target packet.
    // Prompt writes are not journalled with the speculative decoder owners.
    let mut mixed_reference: Option<Vec<Vec<TokenLogprob>>> = None;
    for drafts in [0, maximum] {
        runtime.draft_limit = drafts;
        if drafts > 0 {
            for batch in 1..=4 {
                for full in [false, true] {
                    for _ in 0..4 {
                        runtime
                            .speculation_policy
                            .baseline(batch, full, true, 0.120);
                    }
                }
            }
        }
        let before = runtime.statistics.mixed_speculative_iterations;
        let mut requests = vec![];
        let mut outputs = vec![vec![]; 4];
        for (owner, tokens) in outputs.iter_mut().enumerate() {
            let mut prompt = if owner == 2 {
                input.iter().cycle().take(8193).copied().collect()
            } else if owner == 3 {
                input[..128].to_vec()
            } else {
                input.clone()
            };
            // Keep this admission independent of the earlier cached fixtures.
            prompt[1] = 42 + owner as u32;
            let mut request = runtime
                .start_request(
                    GenerationInput {
                        input_tokens: prompt,
                        images: vec![],
                        audios: vec![],
                        max_new_tokens: 64,
                        sampling: batch_options.clone(),
                        prefix_hints: vec![],
                    },
                    || false,
                )
                .unwrap();
            request.cache_prompt = false;
            request.cache_generated = false;
            if owner < 2 {
                while request.generated == 0 {
                    for step in runtime
                        .advance_requests(&mut [&mut request], &scheduler::Options::default())
                        .unwrap()
                    {
                        tokens.extend(step.tokens);
                    }
                }
            }
            requests.push(request);
        }
        let started = Instant::now();
        while requests.iter().any(|r| !r.is_finished()) {
            let mut refs = requests.iter_mut().collect::<Vec<_>>();
            for step in runtime
                .advance_requests(&mut refs, &scheduler::Options::default())
                .unwrap()
            {
                outputs[step.request].extend(step.tokens);
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        let records = requests
            .iter_mut()
            .map(|r| r.take_logprobs())
            .collect::<Vec<_>>();
        std::fs::write(
            format!("artifacts/dspark/mixed-cohort-{drafts}.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"outputs":outputs,"logprobs":logprob_rows(&records)}),
            )
            .unwrap(),
        )
        .unwrap();
        for request in &mut requests {
            runtime.finish_request(request, false).unwrap();
        }
        assert!(outputs.iter().all(|tokens| tokens.len() == 64));
        if let Some(reference) = &mixed_reference {
            // New prefill owners must not perturb either continuing decoder.
            assert!(compare_fixed_prefix(&reference[..2], &records[..2]) <= 0.02);
            for (a, b) in reference[..2].iter().zip(&records[..2]) {
                assert_eq!(
                    a.iter().map(|p| p.token).collect::<Vec<_>>(),
                    b.iter().map(|p| p.token).collect::<Vec<_>>()
                );
            }
            compare_fixed_prefix(reference, &records);
        } else {
            mixed_reference = Some(records);
        }
        let rounds = runtime.statistics.mixed_speculative_iterations - before;
        if drafts > 0 {
            assert!(rounds > 0);
        }
        eprintln!("DSPARK_MIXED drafts={drafts} wall_s={seconds} speculative_rounds={rounds}");
        reports.push(serde_json::json!({"kind":"mixed", "drafts":drafts,
            "wall_s":seconds,"mixed_speculative_iterations":rounds}));
    }
    std::fs::write(
        "artifacts/dspark/end-to-end.json",
        serde_json::to_vec_pretty(&reports).unwrap(),
    )
    .unwrap();
}
