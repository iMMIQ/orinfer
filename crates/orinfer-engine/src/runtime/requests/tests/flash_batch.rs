use super::*;

#[test]
#[ignore = "Requires mixed MTP Flash package and exclusive GPU experiment lock"]
fn validate_flash_mixed_speculation() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(7),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        model
            .manifest
            .mtp
            .as_ref()
            .unwrap()
            .mixed_verification_tokens,
        [2]
    );
    let options = scheduler::Options {
        target_tpot_ms: 2000.,
        memory_reserve_bytes: 0,
        ..Default::default()
    };
    let snapshot = |m: &mut ModelRuntime, r: &mut RequestState| {
        let end = r.input.len() + r.generated - 1;
        m.with_request(r, |m, _| {
            m.prefix_ranges(end)?
                .into_iter()
                .map(|(name, range)| {
                    let raw = m
                        .execution
                        .download_bytes(&name, range.offset + range.bytes)?;
                    Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                })
                .collect::<Result<BTreeMap<_, _>>>()
        })
        .unwrap()
    };
    let verification = |m: &mut ModelRuntime, r: &mut RequestState, width: usize| {
        m.with_request(r, |m, _| {
            let spec = m.manifest.mtp.as_ref().unwrap();
            let raw = m
                .execution
                .download_bytes(&spec.verification_logits, width * m.manifest.vocab * 4)?;
            let scores: Vec<_> = floats(&raw, crate::artifact::Dtype::F32)
                .chunks_exact(m.manifest.vocab)
                .map(log_probabilities)
                .collect();
            Ok((scores, m.read_controls(&m.manifest.input, width)?))
        })
        .unwrap()
    };
    let start_decoder = |m: &mut ModelRuntime, lane: usize| {
        let mut source = input(fixture.cases.last().unwrap(), 64);
        source.input_tokens.truncate(128);
        source.input_tokens[0] = 32 + lane as u32;
        source.sampling.seed = 20261002;
        let mut r = m.start_request(source, &|| false).unwrap();
        m.with_request(&mut r, |m, r| m.advance_prefill(r, 128, false))
            .unwrap();
        r
    };
    let start_prefill = |m: &mut ModelRuntime, tokens: usize| {
        let mut source = input(fixture.cases.last().unwrap(), 64);
        source.input_tokens.truncate(tokens);
        source.sampling.seed = 20261002;
        m.start_request(source, &|| false).unwrap()
    };
    let mut rows = vec![];
    let mut shapes = vec![
        (16, 2, 1),
        (128, 2, 1),
        (128, 2, 2),
        (512, 2, 1),
        (128, 2, 4),
        (128, 2, 1),
    ];
    if model
        .manifest
        .batch_layout
        .as_ref()
        .is_some_and(|l| l.segment_tokens > 0)
    {
        shapes.extend(
            [1, 2, 3, 7, 17, 129, 257, 511]
                .into_iter()
                .map(|n| (n, 2, 1)),
        );
    }
    for (tokens, width, count) in shapes {
        let mut p = start_prefill(&mut model, tokens);
        let at = Instant::now();
        model
            .with_request(&mut p, |m, r| m.advance_prefill(r, tokens, false))
            .unwrap();
        let mut serial_s = at.elapsed().as_secs_f64();
        let expected_prefill = distribution(&mut model, &mut p);
        model.finish_request(&mut p, false).unwrap();
        let mut expected = vec![];
        for lane in 0..count {
            let mut d = start_decoder(&mut model, lane);
            let at = Instant::now();
            model
                .with_request(&mut d, |m, r| m.speculative_request_step(r, width))
                .unwrap();
            serial_s += at.elapsed().as_secs_f64();
            expected.push(verification(&mut model, &mut d, width));
            model.finish_request(&mut d, false).unwrap();
        }
        let mut first_states: Option<Vec<BTreeMap<String, String>>> = None;
        let mut first_outputs = None;
        for reverse in [false, true] {
            let mut p = start_prefill(&mut model, tokens);
            let mut decoders: Vec<_> = (0..count)
                .map(|lane| start_decoder(&mut model, lane))
                .collect();
            let mut requests = vec![&mut p];
            requests.extend(decoders.iter_mut());
            let mut selected = vec![(0, tokens)];
            selected.extend((1..=count).map(|i| (i, width)));
            if reverse {
                selected.reverse();
            }
            if tokens == 128 && width == 2 && count == 1 && !reverse {
                model.decode_until = None;
                let selected_work = model
                    .select_work(&mut requests, &options, &BTreeSet::new())
                    .unwrap();
                assert!(
                    matches!(selected_work, Work::MixedSpeculative(_)),
                    "Mixed MTP scheduler selection missing"
                );
            }
            let mut output = vec![];
            let at = Instant::now();
            model
                .speculative_mixed_step(&mut requests, &selected, &mut output)
                .unwrap();
            let seconds = at.elapsed().as_secs_f64();
            let mut metrics = vec![];
            for (i, r) in requests.iter_mut().enumerate() {
                let (actual, reference) = if i == 0 {
                    (
                        vec![distribution(&mut model, r)],
                        vec![expected_prefill.clone()],
                    )
                } else {
                    let (actual, history) = verification(&mut model, r, width);
                    assert_eq!(
                        history,
                        expected[i - 1].1,
                        "Mixed drafter changed teacher-forced history"
                    );
                    (actual, expected[i - 1].0.clone())
                };
                for (row, (actual, reference)) in actual.iter().zip(&reference).enumerate() {
                    let kl: f64 = reference
                        .iter()
                        .zip(actual)
                        .map(|(a, b)| a.exp() * (a - b))
                        .sum();
                    let token = actual
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0;
                    let rank = reference.iter().filter(|&&p| p > reference[token]).count();
                    assert!(
                        kl.is_finite() && kl < 0.05 && rank < 3,
                        "{tokens}/{width}/{count}/{reverse}/{i}/{row}: KL {kl}, rank {rank}"
                    );
                    metrics.push(json!({"request":i,"row":row,"kl":kl,"top3":true}));
                }
            }
            eprintln!(
                "mixed {tokens}/{width}/{count}/{reverse}: {seconds:.6}s vs serial {serial_s:.6}s; metrics {metrics:?}"
            );
            let states: Vec<_> = requests
                .iter_mut()
                .map(|r| snapshot(&mut model, r))
                .collect();
            output.sort_unstable_by_key(|s| s.request);
            let tokens_out: Vec<_> = output
                .iter()
                .map(|s| (s.request, s.tokens.clone()))
                .collect();
            if let Some(reference) = &first_states {
                for (lane, (actual, expected)) in states.iter().zip(reference).enumerate() {
                    let changed: Vec<_> = actual
                        .iter()
                        .filter(|(name, hash)| expected.get(*name) != Some(*hash))
                        .map(|(name, _)| name)
                        .collect();
                    assert!(
                        changed.is_empty(),
                        "Mixed owner/order state differs at lane {lane}: {changed:?}"
                    );
                }
            }
            if let Some(reference) = &first_outputs {
                assert_eq!(&tokens_out, reference, "Mixed outputs differ");
            }
            first_states = Some(states);
            first_outputs = Some(tokens_out);
            drop(requests);
            // Retire one participant before resuming peers through their next
            // speculative round, including compact partial/full commit.
            model.finish_request(&mut p, false).unwrap();
            for decoder in &mut decoders {
                model.advance_requests(&mut [decoder], &options).unwrap();
                assert!(decoder.failure().is_none());
                model.finish_request(decoder, false).unwrap();
            }
            rows.push(json!({"prefill_tokens":tokens,"verification_rows":width,"decoders":count,"reverse":reverse,"serial_round_s":serial_s,"mixed_round_s":seconds,"metrics":metrics,"state_replay_equal":true}));
        }
    }
    assert!(model.reserved_requests.is_empty());
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"seed":20261002,"cases":rows,"scheduler":model.scheduling_statistics()}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires layered Flash package and exclusive GPU experiment lock"]
fn validate_layered_prefill() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: fixture.mtp_drafts,
            ..Default::default()
        },
    )
    .unwrap();
    let chunk: usize = std::env::var("ORINFER_LAYERED_CHUNK").map_or(4096, |s| s.parse().unwrap());
    assert!(
        model
            .manifest
            .prefill_plan(chunk)
            .is_some_and(|p| !p.stages.is_empty())
    );
    let mut source = input(fixture.cases.last().unwrap(), 1);
    assert!(source.input_tokens.len() >= chunk);
    let alternate_token = source
        .input_tokens
        .get(1)
        .copied()
        .unwrap_or((source.input_tokens[0] + 1) % model.manifest.vocab as u32);
    source.input_tokens.truncate(chunk);
    let hash_state = |model: &mut ModelRuntime, req: &mut RequestState| {
        model
            .with_request(req, |m, _| {
                m.prefix_ranges(chunk)?
                    .into_iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(&name, range.offset + range.bytes)?;
                        Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap()
    };
    let mut reference = model.start_request(input(&source, 1), &|| false).unwrap();
    model
        .with_request(&mut reference, |m, r| m.advance_prefill(r, chunk, false))
        .unwrap();
    let expected = distribution(&mut model, &mut reference);
    let expected_state = hash_state(&mut model, &mut reference);
    model.finish_request(&mut reference, false).unwrap();
    let mut rows = vec![];
    for (interruption, budget_s) in [(false, 0.), (true, 0.), (true, 0.2)] {
        let before = model.scheduling_statistics();
        let mut req = model.start_request(input(&source, 1), &|| false).unwrap();
        let mut short = input(&source, 1024);
        let short_chunk = chunk.min(512);
        short.input_tokens.truncate(short_chunk);
        let mut decoder = model.start_request(short, &|| false).unwrap();
        model
            .with_request(&mut decoder, |m, r| {
                m.advance_prefill(r, short_chunk, false)
            })
            .unwrap();
        let mut steps = 0;
        while req.prefilling {
            model
                .with_request(&mut req, |m, r| {
                    m.advance_prefill_step(r, chunk, true, budget_s)
                })
                .unwrap();
            steps += 1;
            assert!(steps < 1000);
            if interruption && req.prefilling {
                model
                    .advance_requests(&mut [&mut decoder], &scheduler::Options::default())
                    .unwrap();
                // Overwrite the shared large arena during the first suspension.
                if steps == 1 {
                    let mut another = input(&source, 1);
                    another.input_tokens.truncate(short_chunk);
                    another.input_tokens[0] = alternate_token;
                    let mut other = model.start_request(another, &|| false).unwrap();
                    model
                        .with_request(&mut other, |m, r| m.advance_prefill(r, short_chunk, false))
                        .unwrap();
                    model.finish_request(&mut other, false).unwrap();
                }
            }
        }
        if budget_s == 0. {
            assert!(steps >= model.manifest.prefill_plan(chunk).unwrap().stages.len());
        }
        let scores = distribution(&mut model, &mut req);
        let kl: f64 = expected
            .iter()
            .zip(&scores)
            .map(|(a, b)| a.exp() * (a - b))
            .sum();
        assert!(kl.abs() < 1e-7, "Layer boundary changed logits: {kl}");
        assert_eq!(hash_state(&mut model, &mut req), expected_state);
        let after = model.scheduling_statistics();
        let save_s = after.prefill_suspend_s - before.prefill_suspend_s;
        let resume_s = after.prefill_resume_s - before.prefill_resume_s;
        if interruption {
            assert!(save_s > 0. && resume_s > 0.);
        } else {
            assert_eq!(save_s, 0.);
            assert_eq!(resume_s, 0.);
        }
        rows.push(json!({"interruption":interruption,"budget_s":budget_s,"steps":steps,"kl":kl,"state_equal":true,"save_s":save_s,"resume_s":resume_s}));
        model.finish_request(&mut req, false).unwrap();
        model.finish_request(&mut decoder, false).unwrap();
    }
    // Two paused forwards must keep independent backing even though both use
    // the same large scratch arena and alternate at every layer boundary.
    let mut peers = [
        model.start_request(input(&source, 1), &|| false).unwrap(),
        model.start_request(input(&source, 1), &|| false).unwrap(),
    ];
    while peers.iter().any(|r| r.prefilling) {
        for req in &mut peers {
            if req.prefilling {
                model
                    .with_request(req, |m, r| m.advance_prefill_step(r, chunk, true, 0.))
                    .unwrap();
            }
        }
    }
    for req in &mut peers {
        assert_eq!(hash_state(&mut model, req), expected_state);
        assert_eq!(distribution(&mut model, req), expected);
        model.finish_request(req, false).unwrap();
    }
    let mut cancelled = model.start_request(input(&source, 1), &|| false).unwrap();
    model
        .with_request(&mut cancelled, |m, r| {
            m.advance_prefill_step(r, chunk, true, 0.)
        })
        .unwrap();
    assert!(cancelled.prefill_frame.is_some());
    model.finish_request(&mut cancelled, false).unwrap();
    assert!(cancelled.prefill_frame.is_none());
    let mut reused = model.start_request(input(&source, 1), &|| false).unwrap();
    model
        .with_request(&mut reused, |m, r| m.advance_prefill(r, chunk, false))
        .unwrap();
    assert_eq!(hash_state(&mut model, &mut reused), expected_state);
    model.finish_request(&mut reused, false).unwrap();
    let mut abandoned = model.start_request(input(&source, 1), &|| false).unwrap();
    model
        .with_request(&mut abandoned, |m, r| {
            m.advance_prefill_step(r, chunk, true, 0.)
        })
        .unwrap();
    drop(abandoned);
    model.reap_abandoned().unwrap();
    assert_eq!(model.execution.snapshot_resident_bytes(), 0);
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({
            "chunk_tokens":chunk,"cases":rows,"cancel_reuse_state_equal":true,"alternating_prefill_state_equal":true,"abandoned_frame_released":true,"scheduler":model.scheduling_statistics()
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires dynamic Flash prefill and exclusive GPU experiment lock"]
fn validate_flash_dynamic_prefill_chunks() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let mut source = input(&fixture.cases[0], 1);
    assert!(source.input_tokens.len() >= 528);
    source.input_tokens.truncate(528);
    assert!(source.images.is_empty());
    let mut captures = vec![];
    // Start with an unaligned extent before the 16-row request can warm
    // shared workspace. Every partition uses the same dynamic template.
    for tail in std::iter::once(14)
        .chain(std::iter::once(16))
        .chain((2..=15).filter(|&n| n != 14))
    {
        let mut req = model.start_request(input(&source, 1), &|| false).unwrap();
        let chunks = if tail == 16 {
            vec![512, 16]
        } else {
            vec![512, tail, 16 - tail]
        };
        for &chunk in &chunks {
            model
                .with_request(&mut req, |m, r| m.advance_prefill(r, chunk, false))
                .unwrap();
        }
        let scores = distribution(&mut model, &mut req);
        let state = model
            .with_request(&mut req, |m, _| {
                m.prefix_ranges(528)?
                    .into_iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(&name, range.offset + range.bytes)?;
                        Ok((name, crate::artifact::sha256(&raw[range.offset..])))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap();
        model.finish_request(&mut req, false).unwrap();
        captures.push((tail, chunks, scores, state));
    }
    let reference = &captures[1];
    let mut rows = vec![];
    for (tail, chunks, scores, state) in &captures {
        let kl: f64 = reference
            .2
            .iter()
            .zip(scores)
            .map(|(a, b)| a.exp() * (a - b))
            .sum();
        let token = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        let nll_delta = reference.2[token] - scores[token];
        let top3 = reference
            .2
            .iter()
            .filter(|&&p| p > reference.2[token])
            .count()
            < 3;
        assert!(
            kl.is_finite() && kl < 0.05 && nll_delta.abs() < 0.1 && top3,
            "Exact tail {tail} changed token quality: KL={kl}, NLL delta={nll_delta}"
        );
        let changed: Vec<_> = reference
            .3
            .iter()
            .filter(|(name, hash)| state.get(*name) != Some(*hash))
            .map(|(name, _)| name.clone())
            .collect();
        rows.push(
            json!({"tail":tail,"chunks":chunks,"kl":kl,"nll_delta":nll_delta,
            "selected_in_reference_top3":top3,"changed_state":changed}),
        );
    }
    std::fs::write(fixture.output, serde_json::to_vec_pretty(&json!({
        "seed":20261002,"scope":"Same quantized weights, real causal tail lengths; not BF16 quantization acceptance", "cases":rows
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires native Flash batch package and exclusive GPU experiment lock"]
fn validate_flash_decode_batches() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!((4..=64).contains(&fixture.numerical_tokens));
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            // Compare identical cold prompt execution. A partial prefix hit
            // changes prompt chunk boundaries and tests a different path.
            prefix_cache_bytes: 0,
            mtp_drafts: fixture.mtp_drafts,

            ..Default::default()
        },
    )
    .unwrap();
    assert!(!model.manifest.batch_profiles.is_empty());
    let spec = model.manifest.mtp.take();
    let options = scheduler::Options {
        // Exercise the fixture's actual batch range, rather than silently
        // deferring every >32-row case at the service's default active limit.
        max_active: fixture.batches.iter().copied().max().unwrap_or(32).min(128),
        memory_reserve_bytes: 256 << 20,
        ..Default::default()
    };
    // After real batched target steps, speculative execution must be able to
    // rebuild the draft from each request's captured hidden/state history.
    model.manifest.mtp = spec;
    let mtp_tested = model.manifest.mtp.is_some();
    if model.manifest.mtp.is_some() {
        let mut requests = vec![];
        for i in 0..8 {
            let item = input(&fixture.cases[i % fixture.cases.len()], 12);
            // Released GPU/host arenas can take a moment to appear in the
            // admission counters. Match the serving worker's retry behavior.
            let began = Instant::now();
            let admitted = loop {
                if model.can_admit_request(&item, &options).unwrap() {
                    break true;
                }
                if began.elapsed() >= std::time::Duration::from_secs(2) {
                    break false;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            };
            if !admitted {
                eprintln!(
                    "MTP admission deferred after {} requests: free {}, fixed {}, KV {}, workspace {} bytes",
                    requests.len(),
                    model.admission_free_bytes().unwrap(),
                    model
                        .execution
                        .additional_state_bytes(&model.request_limits(&item).unwrap())
                        .unwrap(),
                    model
                        .execution
                        .additional_kv_bytes(
                            model.validate_generation(&item).unwrap(),
                            &model.request_limits(&item).unwrap()
                        )
                        .unwrap(),
                    model
                        .pending_workspace(
                            model.validate_generation(&item).unwrap(),
                            item.sampling.temperature > 0.,
                            &["decode", "prefill", "head"]
                        )
                        .unwrap()
                );
                break;
            }
            requests.push(model.start_request(item, &|| false).unwrap());
        }
        assert!(
            requests.len() >= 2,
            "Need two admitted requests for batch-to-MTP regression"
        );
        for request in &mut requests {
            eprintln!(
                "MTP prefill: slot {}, {} tokens",
                request.slot,
                request.input.len()
            );
            while request.prefilling {
                model.advance_requests(&mut [request], &options).unwrap();
            }
        }
        while requests.iter().any(|r| !r.is_finished() && r.generated < 3) {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        assert!(
            requests
                .iter()
                .all(|r| r.generated >= 3 && r.failure().is_none())
        );
        for r in &mut requests[1..] {
            model.finish_request(r, false).unwrap();
        }
        let speculative_before = model.scheduler_statistics.speculative_iterations;
        while !requests[0].is_finished() {
            model
                .advance_requests(&mut [&mut requests[0]], &options)
                .unwrap();
        }
        assert!(requests[0].failure().is_none());
        assert!(model.scheduler_statistics.speculative_iterations > speculative_before);
        model.finish_request(&mut requests[0], true).unwrap();
    }
    let mtp_statistics = serde_json::to_value(model.scheduling_statistics()).unwrap();
    // Draft/verification working memory can reduce admission after its first
    // use. Validate ordinary batches with the real --mtp-drafts 0 deployment,
    // rather than silently skipping their numerical coverage under pressure.
    let mut model = if mtp_tested {
        eprintln!("Batch-to-MTP phase complete; reloading target-only model");
        drop(model);
        ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                mtp_drafts: Some(0),

                ..Default::default()
            },
        )
        .unwrap()
    } else {
        model
    };
    let mut rows = vec![];
    let batches = if fixture.batches.is_empty() {
        vec![2, 3, 4, 5, 7, 8, 9, 11, 16, 32]
    } else {
        fixture.batches.clone()
    };
    assert!(batches.contains(&8) && batches.iter().all(|n| (2..=128).contains(n)));
    for &count in &batches {
        eprintln!("Numerical batch {count}: admitting and prefilling");
        let mut requests = vec![];
        for i in 0..count {
            let item = input(
                &fixture.cases[i % fixture.cases.len()],
                fixture.numerical_tokens,
            );
            if !model.can_admit_request(&item, &options).unwrap() {
                break;
            }
            requests.push(model.start_request(item, &|| false).unwrap());
        }
        if requests.len() != count {
            let row =
                json!({"requested_batch":count,"admitted":requests.len(),"memory_deferred":true});
            eprintln!("{row}");
            rows.push(row);
            for request in &mut requests {
                model.finish_request(request, false).unwrap();
            }
            continue;
        }
        let mut outputs = vec![vec![]; count];
        let mut scores = vec![vec![]; count];
        // Complete every prompt before measuring decode. Otherwise a short
        // request can finish while its peers are still prefilling, and an
        // admitted cohort does not prove that the batch kernel actually ran.
        for i in 0..count {
            while requests[i].prefilling {
                // Freeze prompt chunking as well as token history. Adaptive
                // scheduler feedback must not change the reference's initial state.
                let (_, tokens) = model
                    .with_request(&mut requests[i], |m, r| {
                        let remaining = ModelRuntime::prefill_boundary(r) - r.offset;
                        let chunk = m
                            .manifest
                            .prefill_plans
                            .iter()
                            .filter_map(|p| {
                                let n = p.chunk_tokens.min(remaining);
                                p.accepts(n).then_some(n)
                            })
                            .max()
                            .unwrap_or(1);
                        m.advance_prefill(r, chunk, false)
                    })
                    .unwrap();
                if !tokens.is_empty() {
                    assert_eq!(tokens.len(), 1);
                    outputs[i].extend(tokens);
                    scores[i].push(distribution(&mut model, &mut requests[i]));
                }
            }
            assert!(!requests[i].is_finished());
        }
        let batches_before = model
            .scheduler_statistics
            .batch_histogram
            .get(&count)
            .copied()
            .unwrap_or(0);
        while requests.iter().any(|r| !r.is_finished()) {
            let mut active: Vec<_> = requests.iter_mut().collect();
            for step in model.advance_requests(&mut active, &options).unwrap() {
                assert_eq!(step.tokens.len(), 1);
                outputs[step.request].extend(step.tokens);
                scores[step.request].push(distribution(&mut model, &mut requests[step.request]));
            }
        }
        assert!(
            model.scheduler_statistics.batch_histogram[&count] > batches_before,
            "The admitted cohort must execute a real decode batch of {count}"
        );
        for request in &mut requests {
            model.finish_request(request, true).unwrap();
        }
        for i in 0..count {
            let mut baseline = model
                .start_request(
                    input(
                        &fixture.cases[i % fixture.cases.len()],
                        fixture.numerical_tokens,
                    ),
                    &|| false,
                )
                .unwrap();
            while baseline.prefilling {
                model
                    .with_request(&mut baseline, |m, r| {
                        let remaining = ModelRuntime::prefill_boundary(r) - r.offset;
                        let chunk = m
                            .manifest
                            .prefill_plans
                            .iter()
                            .filter_map(|p| {
                                let n = p.chunk_tokens.min(remaining);
                                p.accepts(n).then_some(n)
                            })
                            .max()
                            .unwrap_or(1);
                        m.advance_prefill(r, chunk, false)
                    })
                    .unwrap();
            }
            let mut kl = 0.;
            let mut nll_delta = 0.;
            let mut top3 = 0;
            let mut max_kl: f64 = 0.;
            let mut max_nll_delta: f64 = 0.;
            for (step, &token) in outputs[i].iter().enumerate() {
                let b = distribution(&mut model, &mut baseline);
                let a = &scores[i][step];
                let step_kl = b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum::<f64>();
                let step_delta = b[token as usize] - a[token as usize];
                kl += step_kl;
                nll_delta += step_delta;
                max_kl = max_kl.max(step_kl);
                max_nll_delta = max_nll_delta.max(step_delta.abs());
                let mut ranked: Vec<_> = b.iter().enumerate().collect();
                ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
                top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
                if step + 1 < outputs[i].len() {
                    *baseline.history.last_mut().unwrap() = token;
                    model
                        .with_request(&mut baseline, |m, _| {
                            m.upload_ids(&m.manifest.token, &[token])
                        })
                        .unwrap();
                    model
                        .advance_requests(&mut [&mut baseline], &options)
                        .unwrap();
                }
            }
            model.finish_request(&mut baseline, false).unwrap();
            let n = outputs[i].len() as f64;
            let row = json!({"batch":count,"lane":i,"case":i % fixture.cases.len(),"same_history_mean_kl":kl/n,"mean_nll_delta":nll_delta/n,"max_kl":max_kl,"max_abs_nll_delta":max_nll_delta,"selected_in_reference_top3":top3 as f64/n,"tokens":outputs[i]});
            eprintln!("{row}");
            assert!(
                max_kl.is_finite() && max_kl < 0.05 && max_nll_delta < 0.1,
                "Batch changed target distribution: {row}"
            );
            rows.push(row);
        }
    }
    assert!(
        rows.iter().any(|r| r["batch"] == 8),
        "A real batch 8 numerical comparison is required"
    );
    if !model.manifest.dynamic_batch_kernels.is_empty() {
        for count in [3, 5, 7, 9] {
            if !batches.contains(&count) {
                continue;
            }
            assert!(
                rows.iter().any(|r| r["batch"] == count),
                "A real dynamic batch {count} numerical comparison is required"
            );
        }
        if rows
            .iter()
            .any(|r| r["batch"].as_u64().is_some_and(|n| !n.is_power_of_two()))
        {
            let statistics = model.scheduling_statistics().batch_execution;
            if fixture.cuda_graph == "off" {
                assert!(statistics.dynamic_direct_iterations > 0);
            } else {
                assert!(statistics.dynamic_graph_captures > 0);
                assert!(statistics.graph_hits > 0);
            }
        }
    }
    let mut survivor = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    while survivor.generated < 2 {
        model
            .advance_requests(&mut [&mut survivor], &options)
            .unwrap();
    }
    let before = state(&mut model, &mut survivor);
    let mut other = model
        .start_request(input(&fixture.cases[1], 8), &|| false)
        .unwrap();
    assert_eq!(before, state(&mut model, &mut survivor));
    model.finish_request(&mut other, false).unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(before, state(&mut model, &mut survivor));
    let limits = model.request_limits(&input(&fixture.cases[1], 8)).unwrap();
    assert_eq!(
        model.execution.additional_state_bytes(&limits).unwrap(),
        0,
        "Choose the retained warm arena, not an earlier reclaimed arena"
    );
    let mut replacement = model
        .start_request(input(&fixture.cases[1], 8), &|| false)
        .unwrap();
    assert_ne!(replacement.slot, survivor.slot);
    assert_eq!(before, state(&mut model, &mut survivor));
    while !survivor.is_finished() || !replacement.is_finished() {
        model
            .advance_requests(&mut [&mut replacement, &mut survivor], &options)
            .unwrap();
    }
    model.finish_request(&mut survivor, true).unwrap();
    model.finish_request(&mut replacement, true).unwrap();
    drop(before);
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.active_kv_bytes(), 0);
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"cases":rows,"cancel_reuse_state_equal":true,"batch_to_mtp":mtp_tested,"mtp_scheduler":mtp_statistics,"scheduler":model.scheduling_statistics()})).unwrap()).unwrap();
}

pub(super) fn distribution(model: &mut ModelRuntime, request: &mut RequestState) -> Vec<f64> {
    let raw = model
        .with_request(request, |m, _| {
            m.execution
                .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
        })
        .unwrap();
    log_probabilities(&floats(&raw, crate::artifact::Dtype::F32))
}

#[test]
#[ignore = "Requires native batched verification package and exclusive GPU lock"]
fn validate_flash_batched_verification() {
    use super::super::speculation::Preparation;
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(
        !model
            .manifest
            .mtp
            .as_ref()
            .unwrap()
            .batch_verification_tokens
            .is_empty()
    );
    let spec = model.manifest.mtp.as_ref().unwrap().clone();
    let options = scheduler::Options::default();
    let mut results = vec![];
    for count in [2, 4, 8, 2, 4, 8] {
        let inputs = |lane| {
            let mut request = input(&fixture.cases[lane % fixture.cases.len()], 32);
            request.input_tokens.push((32 + lane) as u32);
            request
        };
        let mut references = vec![];
        let mut candidates = vec![];
        for lane in 0..count {
            let mut request = model.start_request(inputs(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            let Preparation::Round(round) = model
                .with_request(&mut request, |m, r| m.prepare_speculative_round(r, 4))
                .unwrap()
            else {
                panic!("Expected initialized MTP round")
            };
            candidates.push(round.verification.clone());
            model
                .with_request(&mut request, |m, r| m.verify_single_round(r, &round))
                .unwrap();
            let raw = model
                .with_request(&mut request, |m, _| {
                    m.execution
                        .download_bytes(&spec.verification_logits, 4 * m.manifest.vocab * 4)
                })
                .unwrap();
            let scores: Vec<_> = floats(&raw, crate::artifact::Dtype::F32)
                .chunks_exact(model.manifest.vocab)
                .map(log_probabilities)
                .collect();
            references.push(scores);
            model
                .with_request(&mut request, |m, r| m.commit_verified_round(r, round))
                .unwrap();
            model.finish_request(&mut request, false).unwrap();
        }
        let mut requests = vec![];
        let mut rounds = vec![];
        for (lane, candidate) in candidates.iter().enumerate() {
            let mut request = model.start_request(inputs(lane), &|| false).unwrap();
            while request.prefilling {
                model
                    .advance_requests(&mut [&mut request], &options)
                    .unwrap();
            }
            let Preparation::Round(round) = model
                .with_request(&mut request, |m, r| m.prepare_speculative_round(r, 4))
                .unwrap()
            else {
                panic!("Expected MTP round")
            };
            assert_eq!(&round.verification, candidate);
            model.upload_segment_controls(round.plan.tokens).unwrap();
            model
                .execution
                .ensure_sequence_program(request.slot, &round.plan.program)
                .unwrap();
            model
                .upload_ids(&model.manifest.input, &round.verification)
                .unwrap();
            model
                .prepare_verification_inputs(
                    &round.plan.program,
                    &round.verification,
                    &request.history[..request.history.len() - 1],
                )
                .unwrap();
            model.upload_segment_controls(4).unwrap();
            rounds.push(round);
            requests.push(request);
        }
        let segments: Vec<_> = requests
            .iter()
            .map(|r| BatchSegment {
                slot: r.slot,
                tokens: 4,
            })
            .collect();
        let (plan, bindings) = model
            .model_package
            .batch_operation(
                &segments,
                orinfer_model_sdk::abi::BatchOperation::Verification,
                true,
            )
            .unwrap();
        assert!(bindings.is_empty());
        let at = Instant::now();
        model
            .execution
            .execute_batch_operation(
                orinfer_model_sdk::abi::BatchOperation::Verification,
                segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                &plan,
                true,
                true,
            )
            .unwrap();
        let seconds = at.elapsed().as_secs_f64();
        let mut metrics = vec![];
        for (lane, round) in rounds.into_iter().enumerate() {
            let raw = model
                .with_request(&mut requests[lane], |m, _| {
                    m.execution
                        .download_bytes(&spec.verification_logits, 4 * m.manifest.vocab * 4)
                })
                .unwrap();
            for (row, actual) in floats(&raw, crate::artifact::Dtype::F32)
                .chunks_exact(model.manifest.vocab)
                .map(log_probabilities)
                .enumerate()
            {
                let reference = &references[lane][row];
                let kl: f64 = reference
                    .iter()
                    .zip(&actual)
                    .map(|(p, q)| p.exp() * (p - q))
                    .sum();
                let token = actual
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                let higher = reference.iter().filter(|&&p| p > reference[token]).count();
                let delta = reference[token] - actual[token];
                assert!(
                    kl.is_finite() && kl < 0.05 && higher < 3,
                    "Batch {count}, lane {lane}, row {row}: KL {kl}, selected rank {higher}"
                );
                metrics.push(json!({"lane":lane,"row":row,"kl":kl,"nll_delta":delta,"top3":true}));
            }
            let output = model
                .with_request(&mut requests[lane], |m, r| {
                    m.commit_verified_round(r, round)
                })
                .unwrap();
            assert!(!output.is_empty());
            assert_eq!(
                model
                    .with_request(&mut requests[lane], |m, _| Ok(m
                        .read_control(&m.manifest.position)?
                        as usize))
                    .unwrap(),
                requests[lane].input.len() + requests[lane].generated - 1
            );
            while !requests[lane].is_finished() {
                model
                    .advance_requests(&mut [&mut requests[lane]], &options)
                    .unwrap();
            }
            assert!(requests[lane].failure().is_none());
            model.finish_request(&mut requests[lane], false).unwrap();
        }
        eprintln!("batched verification {count}: {seconds:.6}s");
        results.push(json!({"requests":count,"verification_s":seconds,"metrics":metrics}));
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"seed":20261002,"cases":results})).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash mixed package and exclusive GPU experiment lock"]
fn validate_flash_mixed_decoder_groups() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    let collect = std::env::var_os("ORINFER_COLLECT_MIXED_REFERENCE").is_some();
    let paired: Option<Vec<Vec<Vec<f64>>>> = std::env::var_os("ORINFER_MIXED_REFERENCE")
        .map(|path| crate::model::read(&PathBuf::from(path)).unwrap());
    let mut distributions = vec![];
    let mut cases = vec![
        (2, 128),
        (3, 128),
        (4, 128),
        (8, 128),
        (2, 512),
        (3, 512),
        (4, 512),
        (8, 512),
        (2, 128),
        (3, 128),
        (4, 128),
        (8, 128),
    ];
    let shared_graphs = fixture.cuda_graph == "full";
    if shared_graphs {
        cases.push((2, 128));
    }
    for (count, chunk) in cases {
        // Retain spare leases on repeated geometries so the stateless pieces
        // must survive a real membership change instead of reusing a slot graph.
        let mut spare = vec![];
        if shared_graphs && rows.len() >= 8 {
            for _ in 0..(rows.len() - 7) {
                spare.push(
                    model
                        .start_request(input(&fixture.cases[0], 1), &|| false)
                        .unwrap(),
                );
            }
        }
        let request_input = |lane| {
            let mut request = input(&fixture.cases[0], 4);
            // Distinct histories expose state-table aliasing across lanes.
            request.input_tokens.push(32 + lane as u32);
            request
        };
        let mut reference = vec![];
        for lane in 0..count {
            let mut request = model.start_request(request_input(lane), &|| false).unwrap();
            prefill_reference(&mut model, &mut request);
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
            reference.push(distribution(&mut model, &mut request));
            model.finish_request(&mut request, false).unwrap();
        }
        let mut requests = vec![];
        for lane in 0..count {
            let mut request = model.start_request(request_input(lane), &|| false).unwrap();
            prefill_reference(&mut model, &mut request);
            requests.push(request);
        }
        requests.push(
            model
                .start_request(input(&fixture.cases[1], 4), &|| false)
                .unwrap(),
        );
        let selected: Vec<_> = (0..count)
            .map(|i| (i, 1))
            .chain(std::iter::once((count, chunk)))
            .collect();
        let mut output = vec![];
        let at = Instant::now();
        model
            .execute_selected(
                &mut requests.iter_mut().collect::<Vec<_>>(),
                &selected,
                &mut output,
            )
            .unwrap();
        let elapsed = at.elapsed().as_secs_f64();
        assert_eq!(requests[count].offset, chunk);
        let mut scores = vec![];
        let mut group_distributions = vec![];
        for (lane, baseline) in reference.iter().enumerate() {
            assert!(output.iter().any(|s| s.request == lane));
            let actual = distribution(&mut model, &mut requests[lane]);
            let baseline = paired
                .as_ref()
                .map_or(baseline, |r| &r[rows.len() % r.len()][lane]);
            let kl: f64 = baseline
                .iter()
                .zip(&actual)
                .map(|(b, a)| b.exp() * (b - a))
                .sum();
            let token = *requests[lane].history.last().unwrap() as usize;
            let delta = baseline[token] - actual[token];
            assert!(
                collect || (kl.is_finite() && kl < 0.05 && delta.abs() < 0.1),
                "Group {count}, chunk {chunk}, lane {lane}: KL {kl}, NLL delta {delta}"
            );
            scores.push(json!({"lane":lane,"same_history_kl":kl,"nll_delta":delta}));
            group_distributions.push(actual);
        }
        let row = json!({"decoders":count,"chunk":chunk,"mixed_s":elapsed,"scores":scores});
        eprintln!("mixed decoder group {row}");
        rows.push(row);
        distributions.push(group_distributions);
        model.finish_request(&mut requests[count], false).unwrap();
        for request in &mut requests[..count] {
            while !request.is_finished() {
                model.advance_requests(&mut [request], &options).unwrap();
            }
            assert!(request.failure().is_none());
            model.finish_request(request, false).unwrap();
        }
        for request in &mut spare {
            model.finish_request(request, false).unwrap();
        }
    }
    if shared_graphs {
        let stats = model.execution.batch_statistics();
        assert!(
            stats.shared_graph_captures > 0 && stats.shared_graph_hits > 0,
            "Shared pieces must replay across membership changes: {stats:?}"
        );
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    if collect {
        std::fs::write(
            fixture.output.with_extension("distributions.json"),
            serde_json::to_vec(&distributions).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"seed":20261002,"cases":rows,"cancellation_passed":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash mixed package and exclusive GPU experiment lock"]
fn validate_flash_mixed_segments() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!(fixture.cases.len() >= 2 && fixture.cases[1].input_tokens.len() >= 512);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for (round, chunk) in [16, 128, 512, 16, 128, 512].into_iter().enumerate() {
        let mut baseline = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        while baseline.prefilling {
            model
                .advance_requests(&mut [&mut baseline], &options)
                .unwrap();
        }
        let first = *baseline.history.last().unwrap();
        model
            .advance_requests(&mut [&mut baseline], &options)
            .unwrap();
        let reference = distribution(&mut model, &mut baseline);
        model.finish_request(&mut baseline, false).unwrap();

        let mut anchor = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        while anchor.prefilling {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        assert_eq!(*anchor.history.last().unwrap(), first);
        let mut cold = model
            .start_request(input(&fixture.cases[1], 4), &|| false)
            .unwrap();
        let mut output = vec![];
        let before = model.execution.batch_statistics();
        let inputs_before = model.scheduler_statistics.batch_inputs_s;
        let plan_before = model.scheduler_statistics.batch_plan_s;
        let commit_before = model.scheduler_statistics.batch_commit_s;
        let at = Instant::now();
        model
            .execute_selected(
                &mut [&mut anchor, &mut cold],
                &[(0, 1), (1, chunk)],
                &mut output,
            )
            .unwrap();
        let elapsed = at.elapsed().as_secs_f64();
        assert_eq!(cold.offset, chunk);
        assert_eq!(anchor.generated, 2);
        assert!(output.iter().any(|s| s.request == 0));
        let actual = distribution(&mut model, &mut anchor);
        let kl: f64 = reference
            .iter()
            .zip(&actual)
            .map(|(b, a)| b.exp() * (b - a))
            .sum();
        let token = *anchor.history.last().unwrap();
        let delta = reference[token as usize] - actual[token as usize];
        let after = model.execution.batch_statistics();
        let row = json!({"round":round,"chunk":chunk,"decode_tokens":1,"mixed_s":elapsed,
            "direct_s":after.direct_s-before.direct_s,"kernel_load_s":after.kernel_load_s-before.kernel_load_s,
            "inputs_s":model.scheduler_statistics.batch_inputs_s-inputs_before,
            "plan_s":model.scheduler_statistics.batch_plan_s-plan_before,
            "commit_s":model.scheduler_statistics.batch_commit_s-commit_before,
            "same_history_kl":kl,"nll_delta":delta});
        eprintln!("forced mixed {row}");
        assert!(
            kl.is_finite() && kl < 0.05 && delta.abs() < 0.1,
            "Mixed decoder distribution changed: {row}"
        );
        rows.push(row);
        model.finish_request(&mut cold, false).unwrap();
        // Cancelling a partial prompt cannot release or mutate the decoder's state.
        while !anchor.is_finished() {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        assert!(anchor.failure().is_none());
        model.finish_request(&mut anchor, false).unwrap();
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"seed":20261002,"cases":rows,"cancellation_passed":true}),
        )
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash package and exclusive GPU experiment lock"]
fn validate_flash_bounded_prefill() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),

            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options {
        prefill_budget_ms: 200.,
        ..Default::default()
    };
    let mut rows = vec![];
    for source in &fixture.cases {
        let mut anchor = model
            .start_request(input(&fixture.cases[0], 32), &|| false)
            .unwrap();
        while anchor.prefilling {
            model
                .advance_requests(&mut [&mut anchor], &options)
                .unwrap();
        }
        let mut cold = model.start_request(input(source, 4), &|| false).unwrap();
        let began = Instant::now();
        let mut chunks = vec![];
        let mut scores = vec![];
        let mut output = vec![];
        let bounded_before = model.scheduler_statistics.bounded_prefill_iterations;
        let mixed_before = model.scheduler_statistics.mixed_iterations;
        let mut decode_during_prefill = 0;
        while !cold.is_finished() || !anchor.is_finished() {
            let (offset, generated) = (cold.offset, anchor.generated);
            let waiting = cold.prefilling && !anchor.is_finished();
            for step in model
                .advance_requests(&mut [&mut anchor, &mut cold], &options)
                .unwrap()
            {
                if step.request == 1 {
                    output.extend(step.tokens);
                    scores.push(distribution(&mut model, &mut cold));
                }
            }
            if cold.offset > offset {
                chunks.push((cold.offset - offset, waiting));
            }
            if waiting {
                decode_during_prefill += anchor.generated - generated;
            }
        }
        let elapsed = began.elapsed().as_secs_f64();
        assert!(!output.is_empty() && output.len() <= 4);
        assert!(decode_during_prefill > 0);
        // Both joint and independent deadline-bounded execution are valid;
        // the scheduler chooses by the complete measured iteration cost.
        assert!(
            model.scheduler_statistics.mixed_iterations > mixed_before
                || model.scheduler_statistics.bounded_prefill_iterations > bounded_before
        );
        // A cold-only 2048/4096 tile must not block an already-live decoder.
        assert!(
            chunks
                .iter()
                .filter(|(_, mixed)| *mixed)
                .all(|(n, _)| *n <= 512)
        );
        for request in [&mut cold, &mut anchor] {
            model.finish_request(request, false).unwrap();
        }
        let mut baseline = model.start_request(input(source, 4), &|| false).unwrap();
        while baseline.prefilling {
            model
                .advance_requests(&mut [&mut baseline], &options)
                .unwrap();
        }
        let (mut kl, mut nll, mut top3) = (0., 0., 0);
        for (step, &token) in output.iter().enumerate() {
            let b = distribution(&mut model, &mut baseline);
            let a = &scores[step];
            kl += b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum::<f64>();
            nll += b[token as usize] - a[token as usize];
            let mut ranked: Vec<_> = b.iter().enumerate().collect();
            ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
            top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
            if step + 1 < output.len() {
                *baseline.history.last_mut().unwrap() = token;
                model
                    .with_request(&mut baseline, |m, _| {
                        m.upload_ids(&m.manifest.token, &[token])
                    })
                    .unwrap();
                model
                    .advance_requests(&mut [&mut baseline], &options)
                    .unwrap();
            }
        }
        model.finish_request(&mut baseline, false).unwrap();
        let n = output.len() as f64;
        let row = json!({"prompt_tokens":source.input_tokens.len(),"images":source.images.len(),
            "chunks":chunks,"decoder_tokens_during_prefill":decode_during_prefill,
            "wall_s":elapsed,"same_history_mean_kl":kl/n,"mean_nll_delta":nll/n,
            "selected_in_reference_top3":top3 as f64/n,"tokens":output});
        eprintln!("bounded prefill {row}");
        assert!(
            kl.is_finite() && nll.is_finite() && kl / n < 0.05 && (nll / n).abs() < 0.1,
            "Bounded chunks changed token quality: {row}"
        );
        rows.push(row);
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({
        "seed":20261002,"scope":"Same quantized weights, cold large-chunk reference; not BF16 quantization acceptance",
        "cases":rows,"scheduler":model.scheduling_statistics(),"cleanup_passed":true
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires native Flash package and exclusive GPU experiment lock"]
fn validate_flash_prefix_cache() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 2 << 30,
            mtp_drafts: Some(0),

            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for source in &fixture.cases {
        for snapshot in model.prefix_cache.device.clear() {
            model.execution.release_snapshot(snapshot).unwrap();
        }
        let mut cold = model.start_request(input(source, 4), &|| false).unwrap();
        let mut expected = vec![];
        while !cold.is_finished() {
            for step in model.advance_requests(&mut [&mut cold], &options).unwrap() {
                expected.extend(step.tokens);
            }
        }
        let position = source.input_tokens.len() + cold.generated - 1;
        let ranges = model.prefix_ranges(position).unwrap();
        let committed = |model: &mut ModelRuntime, request: &mut RequestState| {
            model
                .with_request(request, |m, _| {
                    ranges
                        .iter()
                        .map(|(name, range)| {
                            let bytes = m
                                .execution
                                .download_bytes(name, range.offset + range.bytes)?;
                            Ok((name.clone(), bytes[range.offset..].to_vec()))
                        })
                        .collect::<Result<BTreeMap<String, Vec<u8>>>>()
                })
                .unwrap()
        };
        let expected_state = committed(&mut model, &mut cold);
        model.finish_request(&mut cold, true).unwrap();
        let mut warm = model.start_request(input(source, 4), &|| false).unwrap();
        assert_eq!(
            warm.prefix.statistics.cached_tokens,
            source.input_tokens.len()
        );
        let mut actual = vec![];
        while !warm.is_finished() {
            for step in model.advance_requests(&mut [&mut warm], &options).unwrap() {
                actual.extend(step.tokens);
            }
        }
        assert_eq!(
            actual, expected,
            "Full prefix restore changed deterministic continuation"
        );
        assert_eq!(
            committed(&mut model, &mut warm),
            expected_state,
            "Full prefix restore changed committed KV/GDN/conv/PLE/index/pending state"
        );
        rows.push(json!({"prompt_tokens":source.input_tokens.len(),"images":source.images.len(),
            "cached_tokens":warm.prefix.statistics.cached_tokens,"restore_s":warm.prefix.statistics.restore_s,
            "tokens_equal":true,"committed_state_equal":true}));
        model.finish_request(&mut warm, true).unwrap();
    }
    for snapshot in model.prefix_cache.device.clear() {
        model.execution.release_snapshot(snapshot).unwrap();
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({
            "seed":20261002,"cases":rows,"cleanup_passed":true
        }))
        .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash package, a long prompt and exclusive GPU experiment lock"]
fn validate_flash_heterogeneous_scheduling() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!(fixture.cases.len() >= 2 && fixture.cases[0].input_tokens.len() >= 4096);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options {
        prefill_budget_ms: 350.,
        ..Default::default()
    };
    let mut rows = vec![];
    for live_decoder in [false, true] {
        let mut long = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        let mut short = model
            .start_request(input(&fixture.cases[1], 24), &|| false)
            .unwrap();
        if live_decoder {
            while short.prefilling {
                model.advance_requests(&mut [&mut short], &options).unwrap();
            }
        }
        let mut output = vec![];
        let mut scores = vec![];
        let mut chunks = vec![];
        let before = model.scheduler_statistics.clone();
        while !long.is_finished() || !short.is_finished() {
            let offset = long.offset;
            let decoder = !short.prefilling && !short.is_finished();
            for step in model
                .advance_requests(&mut [&mut long, &mut short], &options)
                .unwrap()
            {
                if step.request == 0 {
                    output.extend(step.tokens);
                    scores.push(distribution(&mut model, &mut long));
                }
            }
            if long.offset > offset {
                chunks.push((long.offset - offset, decoder));
            }
        }
        assert!(
            chunks.iter().any(|&(n, decoder)| n >= 2048 && !decoder),
            "Native large path lost: {chunks:?}"
        );
        assert!(
            chunks
                .iter()
                .filter(|(_, decoder)| *decoder)
                .all(|(n, _)| *n <= 512)
        );
        if live_decoder {
            assert!(
                model.scheduler_statistics.decode_burst_iterations > before.decode_burst_iterations
            );
        }
        for r in [&mut long, &mut short] {
            model.finish_request(r, false).unwrap();
        }
        let mut baseline = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        while baseline.prefilling {
            model
                .advance_requests(&mut [&mut baseline], &options)
                .unwrap();
        }
        let native_distribution = distribution(&mut model, &mut baseline);
        let cross_partition_kl: f64 = native_distribution
            .iter()
            .zip(&scores[0])
            .map(|(b, a)| b.exp() * (b - a))
            .sum();
        eprintln!("heterogeneous cross-partition first-token KL {cross_partition_kl}");
        assert!(
            cross_partition_kl.is_finite() && cross_partition_kl < 0.05,
            "Interrupted prefill changed the distribution: {cross_partition_kl}, chunks={chunks:?}"
        );
        model.finish_request(&mut baseline, false).unwrap();
        // Match numerical chunk boundaries to isolate request interleaving.
        // The cross-partition check above and this interleaving check cover
        // different failure modes; neither is a BF16 quantization reference.
        let mut baseline = model
            .start_request(input(&fixture.cases[0], 4), &|| false)
            .unwrap();
        for &(chunk, _) in &chunks {
            model
                .with_request(&mut baseline, |m, r| m.advance_prefill(r, chunk, false))
                .unwrap();
        }
        let (mut max_kl, mut delta, mut top3) = (0.0f64, 0., 0);
        for (step, &token) in output.iter().enumerate() {
            let b = distribution(&mut model, &mut baseline);
            let a = &scores[step];
            let kl: f64 = b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum();
            eprintln!(
                "heterogeneous position: live_decoder={live_decoder}, step={step}, token={token}, KL={kl}, NLL_delta={}",
                b[token as usize] - a[token as usize]
            );
            assert!(
                kl.is_finite() && kl < 0.05,
                "Heterogeneous distribution changed: live_decoder={live_decoder}, step={step}, KL={kl}, chunks={chunks:?}"
            );
            max_kl = max_kl.max(kl);
            delta += b[token as usize] - a[token as usize];
            top3 += usize::from(b.iter().filter(|&&p| p > b[token as usize]).count() < 3);
            if step + 1 < output.len() {
                *baseline.history.last_mut().unwrap() = token;
                model
                    .with_request(&mut baseline, |m, _| {
                        m.upload_ids(&m.manifest.token, &[token])
                    })
                    .unwrap();
                model
                    .advance_requests(&mut [&mut baseline], &options)
                    .unwrap();
            }
        }
        model.finish_request(&mut baseline, false).unwrap();
        assert!(delta / (output.len() as f64) < 0.1 && top3 as f64 / output.len() as f64 >= 0.9);
        let row = json!({"live_decoder":live_decoder,"long_chunks":chunks,"cross_partition_first_token_kl":cross_partition_kl,"max_same_history_kl":max_kl,"mean_nll_delta":delta/output.len() as f64,"selected_in_reference_top3":top3 as f64/output.len() as f64});
        eprintln!("heterogeneous scheduling {row}");
        rows.push(row);
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"cases":rows,"scheduler":model.scheduling_statistics()}))
            .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires native Flash package, 16K decode stress and exclusive GPU lock"]
fn validate_flash_long_generation_reuse() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    assert!(fixture.cases.len() >= 2);
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 2 << 30,
            mtp_drafts: Some(3),
            ..Default::default()
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let endpoint = |m: &mut ModelRuntime, source: &GenerationInput| {
        let mut request = m.start_request(input(source, 16384), &|| false).unwrap();
        let mut chunks = vec![];
        while request.prefilling {
            let offset = request.offset;
            m.advance_requests(&mut [&mut request], &options).unwrap();
            if request.offset > offset {
                chunks.push(request.offset - offset);
            }
        }
        let scores = distribution(m, &mut request);
        let ranges = m.prefix_ranges(request.input.len()).unwrap();
        let state = m
            .with_request(&mut request, |m, _| {
                ranges
                    .iter()
                    .map(|(name, range)| {
                        let raw = m
                            .execution
                            .download_bytes(name, range.offset + range.bytes)?;
                        Ok((name.clone(), raw[range.offset..].to_vec()))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .unwrap();
        (request, chunks, scores, state)
    };
    let (mut reference, expected_chunks, expected_scores, expected_state) =
        endpoint(&mut model, &fixture.cases[1]);
    assert!(expected_chunks.contains(&4096));
    model.finish_request(&mut reference, false).unwrap();
    for snapshot in model.prefix_cache.device.clear() {
        model.execution.release_snapshot(snapshot).unwrap();
    }
    let mut stress = model
        .start_request(input(&fixture.cases[0], 16384), &|| false)
        .unwrap();
    // The engine has no transport EOS policy: continuing through EOS is
    // deliberate here so a successful answer still exercises all 16K steps.
    while !stress.is_finished() {
        model
            .advance_requests(&mut [&mut stress], &options)
            .unwrap();
        if stress.generated / 2048 != (stress.generated.saturating_sub(4)) / 2048 {
            eprintln!("long reuse stress: {} committed tokens", stress.generated);
        }
    }
    assert_eq!(stress.generated, 16384);
    assert!(stress.failure().is_none());
    let rounds = stress.mtp.rounds;
    assert!(rounds > 0);
    model.finish_request(&mut stress, true).unwrap();
    let mut rows = vec![];
    for cached in [false, true] {
        let (mut request, chunks, scores, actual_state) = endpoint(&mut model, &fixture.cases[1]);
        if cached {
            assert_eq!(request.prefix.statistics.cached_tokens, request.input.len());
        } else {
            assert_eq!(request.prefix.statistics.cached_tokens, 0);
            assert_eq!(chunks, expected_chunks);
        }
        let kl: f64 = expected_scores
            .iter()
            .zip(&scores)
            .map(|(a, b)| a.exp() * (a - b))
            .sum();
        let changed: Vec<_> = expected_state
            .iter()
            .filter(|(name, bytes)| actual_state.get(*name) != Some(*bytes))
            .map(|(name, _)| name.clone())
            .collect();
        let row = json!({"cached":cached,"chunks":chunks,"kl":kl,"changed_state":changed});
        eprintln!("long generation reuse {row}");
        assert!(
            kl.is_finite() && kl.abs() < 1e-8,
            "Reuse changed target law: {row}"
        );
        assert!(changed.is_empty(), "Reuse changed committed state: {row}");
        rows.push(row);
        model.finish_request(&mut request, true).unwrap();
    }
    for snapshot in model.prefix_cache.device.clear() {
        model.execution.release_snapshot(snapshot).unwrap();
    }
    model.execution.reclaim_idle_graphs().unwrap();
    model.execution.reclaim_idle_state().unwrap();
    assert!(model.reserved_requests.is_empty());
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"stress_tokens":16384,"mtp_rounds":rounds,
            "cases":rows,"committed_state_equal":true,"cleanup_passed":true}))
        .unwrap(),
    )
    .unwrap();
}
