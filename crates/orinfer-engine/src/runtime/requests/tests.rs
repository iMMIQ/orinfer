use super::*;
use serde::Deserialize;
use serde_json::json;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Fixture {
    model: PathBuf,
    output: PathBuf,
    cases: Vec<GenerationInput>,
    cuda_graph: String,
    #[serde(default = "partial_prefix_boundaries")]
    checkpoint_tokens: Vec<usize>,
}

fn partial_prefix_boundaries() -> Vec<usize> {
    vec![32]
}

fn input(source: &GenerationInput, limit: usize) -> GenerationInput {
    GenerationInput {
        input_tokens: source.input_tokens.clone(),
        images: source.images.clone(),
        max_new_tokens: limit,
        sampling: source.sampling.clone(),
        prefix_hints: vec![],
    }
}

fn state(model: &mut ModelRuntime, request: &mut RequestState) -> BTreeMap<String, Vec<u8>> {
    model
        .with_request(request, |m, _| {
            m.execution
                .private_buffer_sizes()
                .into_iter()
                .map(|(n, bytes)| Ok((n.clone(), m.execution.download_bytes(&n, bytes)?)))
                .collect()
        })
        .unwrap()
}

#[test]
#[ignore = "Requires real model and exclusive GPU experiment lock"]
fn benchmark_prefill_scheduling() {
    #[derive(Deserialize)]
    struct SchedulingFixture {
        #[serde(flatten)]
        fixture: Fixture,
        batches: Vec<usize>,
        repetitions: usize,
        output_tokens: usize,
    }
    let spec: SchedulingFixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let fixture = spec.fixture;
    assert!(!fixture.output.exists());
    assert!(spec.repetitions > 0 && spec.output_tokens > 1);
    assert!(spec.batches.iter().all(|n| (2..=8).contains(n)));
    assert!(fixture.cases.iter().all(|c| c.images.is_empty()));
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
        },
    )
    .unwrap();
    assert!(!model.manifest.prefill_batch_profiles.is_empty());
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for source in &fixture.cases {
        for &count in &spec.batches {
            for repetition in 0..spec.repetitions {
                // Reverse paired order to expose warm-arena and clock drift.
                for joint in if repetition.is_multiple_of(2) {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let began = Instant::now();
                    let mut requests: Vec<_> = (0..count)
                        .map(|_| {
                            model
                                .start_request(input(source, spec.output_tokens), &|| false)
                                .unwrap()
                        })
                        .collect();
                    let admission_s = began.elapsed().as_secs_f64();
                    let prefill_at = Instant::now();
                    let mut first_token_s = vec![None; count];
                    let mut output = vec![vec![]; count];
                    let mut chunks = vec![vec![]; count];
                    while requests.iter().any(|r| r.prefilling) {
                        let offsets: Vec<_> = requests.iter().map(|r| r.offset).collect();
                        let selected: Vec<_> = requests
                            .iter()
                            .enumerate()
                            .filter(|(_, r)| r.prefilling)
                            .map(|(i, _)| i)
                            .take(if joint { count } else { 1 })
                            .collect();
                        let mut active: Vec<_> = requests
                            .iter_mut()
                            .enumerate()
                            .filter(|(i, _)| selected.contains(i))
                            .map(|(_, r)| r)
                            .collect();
                        let steps = model.advance_requests(&mut active, &options).unwrap();
                        drop(active);
                        for (i, request) in requests.iter().enumerate() {
                            if request.offset > offsets[i] {
                                chunks[i].push(request.offset - offsets[i]);
                            }
                        }
                        for step in steps {
                            let i = selected[step.request];
                            first_token_s[i] = Some(began.elapsed().as_secs_f64());
                            output[i].extend(step.tokens);
                        }
                    }
                    let prefill_s = prefill_at.elapsed().as_secs_f64();
                    let decode_at = Instant::now();
                    while requests.iter().any(|r| !r.is_finished()) {
                        let mut active: Vec<_> = requests.iter_mut().collect();
                        for step in model.advance_requests(&mut active, &options).unwrap() {
                            output[step.request].extend(step.tokens);
                        }
                    }
                    let decode_s = decode_at.elapsed().as_secs_f64();
                    let peak_kv_bytes = model.execution.resident_kv_bytes();
                    for (i, request) in requests.iter_mut().enumerate() {
                        assert_eq!(output[i].len(), spec.output_tokens);
                        assert!(first_token_s[i].is_some());
                        model
                            .with_request(request, |m, r| {
                                assert_eq!(
                                    m.read_control(&m.manifest.position)? as usize,
                                    r.input.len() + r.generated - 1
                                );
                                Ok(())
                            })
                            .unwrap();
                        model.finish_request(request, false).unwrap();
                    }
                    assert_eq!(model.execution.resident_kv_bytes(), 0);
                    let tokens = source.input_tokens.len() * count;
                    let row = json!({"mode":if joint {"joint"} else {"serial_large"},
                        "input_tokens_per_request":source.input_tokens.len(),"batch":count,
                        "repetition":repetition,"admission_s":admission_s,
                        "prefill_s":prefill_s,"prefill_tps":tokens as f64/prefill_s,
                        "first_token_s":first_token_s,"chunks":chunks,"decode_s":decode_s,
                        "decode_tps":(count*(spec.output_tokens-1)) as f64/decode_s,
                        "output_tokens":output,"peak_kv_bytes":peak_kv_bytes,
                        "position_and_cleanup_passed":true});
                    eprintln!("prefill scheduling {row}");
                    rows.push(row);
                    std::fs::write(&fixture.output, serde_json::to_vec_pretty(&json!({
                        "seed":20261002,"status":"running","rows":rows,
                        "scope":"No MTP/prefix reuse; all cohort prefills finish before batched decode; native serial 512/2048 plans versus joint 512 chunks. Token differences are diagnostic, not independent quality acceptance."
                    })).unwrap()).unwrap();
                }
            }
        }
    }
    let mut result: serde_json::Value = crate::model::read(&fixture.output).unwrap();
    result["status"] = json!("complete");
    std::fs::write(&fixture.output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real MTP batch model and exclusive GPU experiment lock"]
fn validate_mtp_partial_prefix() {
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
        },
    )
    .unwrap();
    assert!(model.manifest.mtp.is_some());
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for &checkpoint in &fixture.checkpoint_tokens {
        let mut source = input(&fixture.cases[0], 16);
        let continuation = if checkpoint < 512 { 16 } else { 2048 };
        source.input_tokens.truncate(checkpoint + continuation);
        assert!(source.images.is_empty() && source.input_tokens.len() > checkpoint);
        let mut branch = input(&fixture.cases[1], 16);
        assert!(branch.images.is_empty() && branch.input_tokens.len() > checkpoint);
        branch.input_tokens.truncate(source.input_tokens.len());
        branch.input_tokens[..checkpoint].copy_from_slice(&source.input_tokens[..checkpoint]);
        assert_ne!(
            branch.input_tokens[checkpoint],
            source.input_tokens[checkpoint]
        );
        model.prefix_cache_limit = 0;
        model.prefix_cache.budget = 0;
        while let Some(snapshot) = model.prefix_cache.evict_one() {
            model.execution.release_snapshot(snapshot).unwrap();
        }
        let mut cold = model.start_request(input(&branch, 16), &|| false).unwrap();
        let mut reference = vec![];
        while !cold.is_finished() {
            for step in model.advance_requests(&mut [&mut cold], &options).unwrap() {
                reference.extend(step.tokens);
            }
        }
        let expected_state = state(&mut model, &mut cold);
        model.finish_request(&mut cold, false).unwrap();
        model.prefix_cache_limit = 2 << 30;
        model.prefix_cache.budget = 2 << 30;
        let mut populate = model.start_request(input(&source, 16), &|| false).unwrap();
        populate.checkpoints.insert(checkpoint);
        while !populate.is_finished() {
            model
                .advance_requests(&mut [&mut populate], &options)
                .unwrap();
        }
        model.finish_request(&mut populate, true).unwrap();
        let entry = model
            .prefix_cache
            .entries
            .values()
            .find(|e| e.tokens.len() == checkpoint)
            .unwrap();
        assert_eq!(
            entry.warm_tokens,
            checkpoint - 1,
            "Checkpoint consumed its continuation token"
        );
        let mut restored = model.start_request(input(&branch, 16), &|| false).unwrap();
        assert_eq!(restored.prefix.statistics.cached_tokens, checkpoint);
        assert_eq!(restored.warm.tokens, checkpoint);
        let mut actual = vec![];
        while !restored.is_finished() {
            for step in model
                .advance_requests(&mut [&mut restored], &options)
                .unwrap()
            {
                actual.extend(step.tokens);
            }
        }
        assert_eq!(
            actual, reference,
            "Changed continuation after prefix restore diverged"
        );
        let actual_state = state(&mut model, &mut restored);
        for (name, expected) in &expected_state {
            if name.ends_with("_State") || name.ends_with("_History") {
                assert_eq!(
                    &actual_state[name], expected,
                    "Restored causal state differs: {name}"
                );
            }
        }
        let target = &model.manifest.position;
        let draft = &model.manifest.mtp.as_ref().unwrap().position;
        assert_eq!(actual_state[target], expected_state[target]);
        assert_eq!(actual_state[draft], expected_state[draft]);
        let target_position =
            u32::from_le_bytes(expected_state[target][..4].try_into().unwrap()) as usize;
        let draft_position =
            u32::from_le_bytes(expected_state[draft][..4].try_into().unwrap()) as usize;
        for (name, stride) in &model.manifest.kv_cache.as_ref().unwrap().buffers {
            let position = if name.starts_with("Mtp") {
                draft_position
            } else {
                target_position
            };
            let bytes = position * stride;
            assert_eq!(
                actual_state[name][..bytes],
                expected_state[name][..bytes],
                "Restored committed KV differs: {name}"
            );
        }
        let restore_s = restored.prefix.statistics.restore_s;
        model.finish_request(&mut restored, true).unwrap();
        rows.push(json!({"partial_prefix_tokens":checkpoint,
            "checkpoint_mtp_tokens":checkpoint-1,"prompt_tokens":source.input_tokens.len(),
            "changed_continuation_tokens_equal":true,"gdn_and_conv_state_equal":true,
            "target_and_draft_kv_equal":true,"target_position":target_position,
            "draft_position":draft_position,
            "restore_s":restore_s}));
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"rows":rows,"seed":20261002})).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires real joint prefill package and exclusive GPU experiment lock"]
fn validate_joint_prefill_requests() {
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
        },
    )
    .unwrap();
    assert!(!model.manifest.prefill_batch_profiles.is_empty());
    model.manifest.mtp = None;
    let options = scheduler::Options::default();
    let mut rows = vec![];
    for count in [2, 4, 8] {
        let mut requests: Vec<_> = (0..count)
            .map(|i| {
                model
                    .start_request(input(&fixture.cases[i % fixture.cases.len()], 8), &|| false)
                    .unwrap()
            })
            .collect();
        let mut output = vec![vec![]; count];
        let mut logits = vec![vec![]; count];
        let mut chunks = vec![vec![]; count];
        let mut prefill_s = None;
        let at = Instant::now();
        while requests.iter().any(|r| !r.is_finished()) {
            // Reordering the active vector must not change private causal state.
            let reversed = model.scheduler_statistics.iterations.is_multiple_of(2);
            let offsets: Vec<_> = requests.iter().map(|r| r.offset).collect();
            let mut active: Vec<_> = requests.iter_mut().collect();
            if reversed {
                active.reverse();
            }
            let steps = model.advance_requests(&mut active, &options).unwrap();
            drop(active);
            for i in 0..count {
                if requests[i].offset > offsets[i] {
                    chunks[i].push(requests[i].offset - offsets[i]);
                }
            }
            for step in steps {
                let i = if reversed {
                    count - 1 - step.request
                } else {
                    step.request
                };
                let raw = model
                    .with_request(&mut requests[i], |m, _| {
                        m.execution
                            .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
                    })
                    .unwrap();
                assert_eq!(step.tokens.len(), 1);
                logits[i].push(floats(&raw, crate::artifact::Dtype::F32));
                output[i].extend(step.tokens);
            }
            if prefill_s.is_none() && requests.iter().all(|r| !r.prefilling) {
                prefill_s = Some(at.elapsed().as_secs_f64());
            }
        }
        // Admission, cancellation and reuse cannot mutate a live request even
        // when its state was produced by a dense joint prefill.
        let before = state(&mut model, &mut requests[0]);
        let mut cancelled = model
            .start_request(input(&fixture.cases[0], 8), &|| false)
            .unwrap();
        assert_eq!(before, state(&mut model, &mut requests[0]));
        model.finish_request(&mut cancelled, false).unwrap();
        let mut reused = model
            .start_request(input(&fixture.cases[0], 8), &|| false)
            .unwrap();
        assert_eq!(cancelled.slot, reused.slot);
        assert_eq!(before, state(&mut model, &mut requests[0]));
        model.finish_request(&mut reused, false).unwrap();
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
        assert_eq!(model.execution.resident_kv_bytes(), 0);
        let prompt_tokens: usize = (0..count)
            .map(|i| fixture.cases[i % fixture.cases.len()].input_tokens.len())
            .sum();
        let wall_s = at.elapsed().as_secs_f64();
        // Use the same 512-token chunk and LUT4 FFN law as the joint route.
        // A 2048-token serial route intentionally has a different codebook;
        // its quality relative to BF16 is evaluated separately.
        let saved_plans = model.manifest.prefill_plans.clone();
        model
            .manifest
            .prefill_plans
            .retain(|p| p.chunk_tokens <= 512);
        for i in 0..count {
            let source = &fixture.cases[i % fixture.cases.len()];
            model
                .generate_reference(
                    &source.input_tokens,
                    Some(&source.images),
                    1,
                    &source.sampling,
                    || false,
                    |_| true,
                )
                .unwrap();
            let mut kl = 0.;
            let mut nll = 0.;
            let mut top3 = 0;
            for (step, &token) in output[i].iter().enumerate() {
                let reference = floats(
                    &model
                        .execution
                        .download_bytes(&model.manifest.logits, model.manifest.vocab * 4)
                        .unwrap(),
                    crate::artifact::Dtype::F32,
                );
                let a = log_probabilities(&logits[i][step]);
                let b = log_probabilities(&reference);
                kl += a
                    .iter()
                    .zip(&b)
                    .map(|(a, b)| a.exp() * (a - b))
                    .sum::<f64>();
                nll += b[token as usize] - a[token as usize];
                let mut ranked: Vec<_> = reference.iter().enumerate().collect();
                ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
                top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
                if step + 1 < output[i].len() {
                    model.upload_ids(&model.manifest.token, &[token]).unwrap();
                    model
                        .launch_program("decode", ExecutionPhase::Decode)
                        .unwrap();
                }
            }
            let n = output[i].len() as f64;
            let same_recipe = chunks[i].iter().all(|&n| n == 512);
            let row = json!({"batch":count,"case":i,"images":source.images.len(),
                "chunks":chunks[i],"same_chunk_recipe":same_recipe,
                "mean_kl":kl/n,"nll_delta":nll/n,"selected_in_serial_top3":top3 as f64/n});
            eprintln!("joint prefill {row}");
            assert!(
                kl.is_finite()
                    && nll.is_finite()
                    && nll / n < 0.05
                    && top3 as f64 / n >= 0.9
                    && (!same_recipe || (kl / n < 0.005 && (nll / n).abs() < 0.05)),
                "Joint prefill changed token quality: {row}"
            );
            rows.push(row);
        }
        model.manifest.prefill_plans = saved_plans;
        rows.push(
            json!({"batch":count,"prefill_s":prefill_s,"prompt_tokens":prompt_tokens,
            "prefill_tps":prompt_tokens as f64/prefill_s.unwrap(),"wall_s":wall_s,
            "cancellation_reuse_state_equal":true,"reservation_cleanup":true}),
        );
    }
    assert!(
        model
            .scheduler_statistics
            .prefill_batch_histogram
            .iter()
            .any(|(&n, &v)| n > 1 && v > 0)
    );
    std::fs::write(fixture.output, serde_json::to_vec_pretty(&json!({
        "seed":20261002,"scope":"Own-package serial 512-chunk path regression; independent BF16 acceptance is separate",
        "rows":rows,"scheduler":model.scheduler_statistics,
        "execution":model.execution.batch_statistics.get()
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires real batch model and exclusive GPU experiment lock"]
fn validate_continuous_requests() {
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
        },
    )
    .unwrap();
    let options = scheduler::Options::default();
    let orphan = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    let orphan_slot = orphan.slot;
    drop(orphan);
    let mut replacement = model
        .start_request(input(&fixture.cases[0], 8), &|| false)
        .unwrap();
    assert_eq!(
        replacement.slot, orphan_slot,
        "Dropped requests must be reaped before arena reuse"
    );
    assert_eq!(model.reserved_requests.len(), 1);
    model.finish_request(&mut replacement, false).unwrap();
    let mut delivered = vec![];
    let count = model
        .generate(
            &fixture.cases[0].input_tokens,
            None,
            8,
            &Default::default(),
            || false,
            |token| {
                delivered.push(token);
                false
            },
        )
        .unwrap();
    assert_eq!(count, 1);
    assert_eq!(delivered.len(), 1);
    assert!(model.reserved_requests.is_empty());
    let mut rows = vec![
        json!({"abandoned_request_reaped":true,"production_generation_early_stop_released":true}),
    ];
    for count in [2, 3, 4, 5, 7, 8, 9, 15, 31] {
        let mut reference = Vec::new();
        for i in 0..count {
            let source = &fixture.cases[i % fixture.cases.len()];
            let mut tokens = vec![];
            model
                .generate_reference(
                    &source.input_tokens,
                    Some(&source.images),
                    16,
                    &source.sampling,
                    || false,
                    |t| {
                        tokens.push(t);
                        true
                    },
                )
                .unwrap();
            reference.push(tokens);
            if source.images.is_empty() {
                let (cached, bytes) = model
                    .prefix_match_cost(&source.input_tokens, &source.images)
                    .unwrap();
                assert_eq!(cached, source.input_tokens.len());
                let expected: usize = model
                    .prefix_ranges(cached)
                    .unwrap()
                    .values()
                    .map(|range| range.bytes)
                    .sum();
                assert_eq!(bytes, expected, "Cached restore-byte estimate changed");
            }
        }
        let mut requests: Vec<_> = (0..count)
            .map(|i| {
                model
                    .start_request(input(&fixture.cases[i % fixture.cases.len()], 16), &|| {
                        false
                    })
                    .unwrap()
            })
            .collect();
        let mut output = vec![vec![]; count];
        while requests.iter().any(|r| !r.is_finished()) {
            let mut active: Vec<_> = requests.iter_mut().collect();
            for step in model.advance_requests(&mut active, &options).unwrap() {
                output[step.request].extend(step.tokens);
            }
        }
        assert_eq!(
            output, reference,
            "Greedy batch differs for {count} independent lanes"
        );
        for request in &mut requests {
            model.finish_request(request, true).unwrap();
        }
        rows.push(json!({"batch":count,"tokens_equal":true}));
        eprintln!("batch {count}: serial output comparison passed");
    }
    // Starting and cancelling another request must not reset a survivor's
    // FP32 recurrent state, convolution history or position. Reuse its slot.
    let mut survivor = model
        .start_request(input(&fixture.cases[0], 24), &|| false)
        .unwrap();
    while survivor.generated < 4 {
        model
            .advance_requests(&mut [&mut survivor], &options)
            .unwrap();
    }
    let before = state(&mut model, &mut survivor);
    let mut other = model
        .start_request(input(&fixture.cases[1], 4096), &|| false)
        .unwrap();
    assert_eq!(
        before,
        state(&mut model, &mut survivor),
        "Admission changed survivor"
    );
    model.finish_request(&mut other, false).unwrap();
    assert_eq!(
        before,
        state(&mut model, &mut survivor),
        "Cancellation changed survivor"
    );
    let mut reused = model
        .start_request(input(&fixture.cases[1], 16), &|| false)
        .unwrap();
    assert_eq!(other.slot, reused.slot);
    assert_eq!(
        before,
        state(&mut model, &mut survivor),
        "Reuse changed survivor"
    );
    while !survivor.is_finished() || !reused.is_finished() {
        model
            .advance_requests(&mut [&mut reused, &mut survivor], &options)
            .unwrap();
    }
    model.finish_request(&mut reused, true).unwrap();
    model.finish_request(&mut survivor, true).unwrap();
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    rows.push(json!({"cancel_reuse_reorder_state_equal":true}));
    eprintln!("all private buffers: cancellation and reuse isolation passed");
    // Short requests must not each reserve a full 256K virtual address range.
    // On Jetson that exhausts CUDA VMM address reservations near twelve lanes,
    // even though the physical memory admission budget still has ample space.
    let mut arenas: Vec<_> = (0..32)
        .map(|_| {
            let value = input(&fixture.cases[0], 16);
            assert!(model.can_admit_request(&value, &options).unwrap());
            model.start_request(value, &|| false).unwrap()
        })
        .collect();
    for request in &mut arenas {
        model.finish_request(request, false).unwrap();
    }
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    rows.push(json!({"short_request_arenas":32,"reservation_cleanup":true}));
    // Cold mixed prefill must exercise the new recurrent 32/64/128 profiles,
    // rather than only checking requests whose prompt was restored from cache.
    for snapshot in model.prefix_cache.clear() {
        model.execution.release_snapshot(snapshot).unwrap();
    }
    model.prefix_cache_limit = 0;
    model.prefix_cache.budget = 0;
    let mtp = model.manifest.mtp.take();
    let mut reference = vec![];
    for source in &fixture.cases {
        let mut tokens = vec![];
        model
            .generate_reference(
                &source.input_tokens,
                Some(&source.images),
                16,
                &source.sampling,
                || false,
                |t| {
                    tokens.push(t);
                    true
                },
            )
            .unwrap();
        reference.push(tokens);
    }
    let mut requests: Vec<_> = fixture
        .cases
        .iter()
        .map(|c| model.start_request(input(c, 16), &|| false).unwrap())
        .collect();
    let mut output = vec![vec![]; requests.len()];
    let mut logits = vec![vec![]; requests.len()];
    while requests.iter().any(|r| !r.is_finished()) {
        for step in model
            .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
            .unwrap()
        {
            assert_eq!(step.tokens.len(), 1);
            let raw = model
                .with_request(&mut requests[step.request], |m, _| {
                    m.execution
                        .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
                })
                .unwrap();
            logits[step.request].push(floats(&raw, crate::artifact::Dtype::F32));
            output[step.request].extend(step.tokens);
        }
    }
    // Cold joint prefill and serial prefill use different reduction schedules.
    // Report token differences; accept them only after the same-history
    // probability checks below, rather than requiring identical greedy ties.
    for (i, source) in fixture.cases.iter().enumerate() {
        eprintln!(
            "cold prefill case {i}: images={}, exact_output_equal={}",
            source.images.len(),
            output[i] == reference[i]
        );
    }
    for request in &mut requests {
        model.finish_request(request, true).unwrap();
    }
    for (i, source) in fixture.cases.iter().enumerate() {
        model
            .generate_reference(
                &source.input_tokens,
                Some(&source.images),
                1,
                &source.sampling,
                || false,
                |_| true,
            )
            .unwrap();
        let mut kl = 0.;
        let mut nll_delta = 0.;
        let mut top3 = 0;
        for (step, &token) in output[i].iter().enumerate() {
            let target = floats(
                &model
                    .execution
                    .download_bytes(&model.manifest.logits, model.manifest.vocab * 4)
                    .unwrap(),
                crate::artifact::Dtype::F32,
            );
            let actual = &logits[i][step];
            let a = log_probabilities(actual);
            let b = log_probabilities(&target);
            kl += a
                .iter()
                .zip(&b)
                .map(|(a, b)| a.exp() * (a - b))
                .sum::<f64>();
            nll_delta += b[token as usize] - a[token as usize];
            let mut ranked: Vec<_> = target.iter().enumerate().collect();
            ranked.select_nth_unstable_by(3, |a, b| b.1.total_cmp(a.1));
            top3 += usize::from(ranked[..3].iter().any(|(id, _)| *id == token as usize));
            if step + 1 < output[i].len() {
                model.upload_ids(&model.manifest.token, &[token]).unwrap();
                model
                    .launch_program("decode", ExecutionPhase::Decode)
                    .unwrap();
            }
        }
        let count = output[i].len() as f64;
        let row = json!({"case":i,"images":source.images.len(),"same_history_mean_kl":kl/count,
            "mean_nll_delta":nll_delta/count,"selected_in_reference_top3":top3 as f64/count,
            "exact_output_equal":output[i]==reference[i]});
        eprintln!("teacher-forced quality {row}");
        // Distribution drift and token agreement are diagnostics: different
        // valid token choices are allowed by the quality acceptance policy.
        assert!(kl.is_finite(), "Nonfinite batch distribution drift: {row}");
        assert!(
            nll_delta / count < 0.05,
            "Batch degraded target likelihood: {row}"
        );
        rows.push(row);
    }
    rows.push(json!({"cold_mixed_prefill_quality_passed":true,"cases":fixture.cases.len()}));
    let mut sampled_reference = None;
    for reversed in [false, true] {
        let mut requests: Vec<_> = fixture.cases[..2]
            .iter()
            .map(|source| {
                let mut item = input(source, 16);
                item.sampling = crate::sampling::Options {
                    temperature: 0.7,
                    top_k: 20,
                    top_p: 0.9,
                    ..Default::default()
                };
                model.start_request(item, &|| false).unwrap()
            })
            .collect();
        if reversed {
            requests.reverse();
        }
        let mut output = vec![vec![]; requests.len()];
        while requests.iter().any(|r| !r.is_finished()) {
            for step in model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap()
            {
                output[step.request].extend(step.tokens);
            }
        }
        for request in &mut requests {
            model.finish_request(request, false).unwrap();
        }
        if reversed {
            output.reverse();
        }
        if let Some(reference) = &sampled_reference {
            assert_eq!(
                &output, reference,
                "Seeded output changed after lane reordering"
            );
        } else {
            sampled_reference = Some(output);
        }
    }
    model.manifest.mtp = mtp;
    rows.push(json!({"seeded_reorder_equal":true,"seed":20261002}));
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({
            "cases":rows,"scheduler":model.scheduler_statistics,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn log_probabilities(logits: &[f32]) -> Vec<f64> {
    assert!(logits.iter().all(|l| l.is_finite()));
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = logits
        .iter()
        .map(|&l| (l as f64 - maximum).exp())
        .sum::<f64>()
        .ln()
        + maximum;
    logits.iter().map(|&l| l as f64 - z).collect()
}
