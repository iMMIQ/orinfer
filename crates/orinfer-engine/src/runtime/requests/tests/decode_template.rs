//! End-to-end gates for a canonical decode packet and reusable indirect graphs.
use super::*;

#[test]
#[ignore = "Requires Flash mixed rows and exclusive GPU experiment lock"]
fn profile_flash_mixed_layer() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut source = input(fixture.cases.last().unwrap(), 64);
    source.input_tokens.truncate(128);
    let mut decoder = model.start_request(source, &|| false).unwrap();
    model
        .with_request(&mut decoder, |m, r| m.advance_prefill(r, 128, false))
        .unwrap();
    let mut source = input(fixture.cases.last().unwrap(), 64);
    source.input_tokens.truncate(512);
    let mut prefill = model.start_request(source, &|| false).unwrap();
    let segments = model
        .prepare_batch_inputs(&mut [&mut prefill, &mut decoder], &[(0, 512), (1, 1)])
        .unwrap();
    let (calls, _) = model
        .batch_plans
        .cached_batch(
            &segments,
            orinfer_model_sdk::packet::BatchOperation::Ordinary,
        )
        .unwrap();
    let end=calls.iter().position(|call|matches!(&call.operation,crate::model::Operation::Kernel {name} if name.contains("/layer1/"))).unwrap();
    let calls = &calls[..end];
    let trials = model.execution.profile_graph_operations(calls).unwrap();
    let mut records = Vec::new();
    for (i, call) in calls.iter().enumerate() {
        let ms = trials.iter().map(|t| t[i] as f64).sum::<f64>() / trials.len() as f64;
        records.push(json!({"operation":call.operation,"ms":ms}));
    }
    std::fs::write(fixture.output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
    model.finish_request(&mut prefill, false).unwrap();
    model.finish_request(&mut decoder, false).unwrap();
}

#[test]
#[ignore = "Requires the unified Flash package and exclusive GPU experiment lock"]
fn validate_flash_decode_template_reuse() {
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
    assert!(
        model
            .manifest
            .programs
            .contains_key("flash_decode/packet/inputs")
    );
    let mut first = None;
    let mut first_graphs = 0;
    let mut evidence = Vec::new();
    for round in 0..3 {
        let mut blocker = if round == 1 {
            Some(
                model
                    .start_request(input(&fixture.cases[0], 1), &|| false)
                    .unwrap(),
            )
        } else {
            None
        };
        let mut requests = Vec::new();
        for lane in 0..3 {
            let mut source = input(&fixture.cases[lane % fixture.cases.len()], 32);
            source
                .sampling
                .logit_bias
                .extend([(248044, -100.), (248046, -100.)]);
            let mut req = model.start_request(source, &|| false).unwrap();
            while req.prefilling {
                model
                    .with_request(&mut req, |m, r| {
                        m.advance_prefill(r, r.input.len().min(4096), false)
                    })
                    .unwrap();
            }
            requests.push(req);
        }
        let mut output_tokens = vec![Vec::new(); 3];
        for _ in 0..4 {
            let mut refs: Vec<_> = requests.iter_mut().collect();
            let mut output = Vec::new();
            model
                .execute_selected(&mut refs, &[(0, 1), (1, 1), (2, 1)], &mut output)
                .unwrap();
            for step in output {
                output_tokens[step.request].extend(step.tokens);
            }
        }
        let mut state_hashes = Vec::new();
        for req in &mut requests {
            let end = req.input.len() + req.generated - 1;
            let snapshot = model
                .with_request(req, |m, _| {
                    m.prefix_ranges(end)?
                        .into_iter()
                        .map(|(name, range)| {
                            let bytes = m
                                .execution
                                .download_bytes(&name, range.offset + range.bytes)?;
                            Ok((name, crate::artifact::sha256(&bytes[range.offset..])))
                        })
                        .collect::<Result<BTreeMap<_, _>>>()
                })
                .unwrap();
            state_hashes.push(snapshot);
        }
        let value = (output_tokens, state_hashes);
        if let Some(expected) = &first {
            assert_eq!(
                &value, expected,
                "Changing request arenas changed decode state"
            );
        } else {
            first = Some(value);
        }
        let graphs = model
            .scheduling_statistics()
            .batch_execution
            .dynamic_graph_captures;
        if round == 0 {
            first_graphs = graphs;
        } else {
            assert_eq!(
                graphs, first_graphs,
                "Same batch size recaptured after request replacement"
            );
        }
        for req in &mut requests {
            model.finish_request(req, true).unwrap();
        }
        if let Some(r) = &mut blocker {
            model.finish_request(r, true).unwrap();
        }
        evidence.push(json!({"round":round,"graph_captures":graphs,"graph_hits":model.scheduling_statistics().batch_execution.graph_hits}));
    }
    assert!(first_graphs > 0);
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(&json!({"membership_state_equal":true,"cases":evidence}))
            .unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "Requires MTP packages and exclusive GPU experiment lock"]
fn benchmark_flash_shared_speculation() {
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
    let options = scheduler::Options {
        target_tpot_ms: 2000.,
        memory_reserve_bytes: 0,
        ..Default::default()
    };
    let mut results = Vec::new();
    for (case, source) in fixture.cases.iter().enumerate() {
        let mut request = model.start_request(input(source, 64), &|| false).unwrap();
        let at = Instant::now();
        prefill_reference(&mut model, &mut request);
        let prefill_s = at.elapsed().as_secs_f64();
        let mut samples = Vec::new();
        let mut tokens = Vec::new();
        let mut committed = 0;
        let mut round = 0;
        while !request.is_finished() {
            let before = request.generated;
            let at = Instant::now();
            let output = model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
            assert!(request.failure().is_none());
            if round >= 3 {
                committed += request.generated - before;
                samples.push(at.elapsed().as_secs_f64());
            }
            tokens.extend(output.into_iter().flat_map(|s| s.tokens));
            round += 1;
        }
        let seconds = samples.iter().sum::<f64>();
        results.push(
            json!({"case":case,"prompt_tokens":request.input.len(),"prefill_s":prefill_s,
            "committed_tokens":committed,"decode_tps":committed as f64/seconds,
            "samples_s":samples,"tokens":tokens,"mtp":request.speculation_statistics()}),
        );
        eprintln!("MTP case {case}: {} tokens / {seconds:.6}s", committed);
        model.finish_request(&mut request, true).unwrap();
    }
    std::fs::write(fixture.output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires Flash packages and exclusive GPU experiment lock"]
fn benchmark_flash_decode_template() {
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
        memory_reserve_bytes: 0,
        ..Default::default()
    };
    let mut results = Vec::new();
    for &count in &fixture.batches {
        eprintln!("Benchmark decode batch {count}");
        let mut requests = Vec::new();
        for lane in 0..count {
            let source = input(&fixture.cases[lane % fixture.cases.len()], 64);
            let mut request = model.start_request(source, &|| false).unwrap();
            while request.prefilling {
                model
                    .with_request(&mut request, |m, r| {
                        m.advance_prefill(r, r.input.len().min(4096), false)
                    })
                    .unwrap();
            }
            requests.push(request);
        }
        let selected: Vec<_> = (0..count).map(|i| (i, 1)).collect();
        let mut samples = Vec::new();
        let mut device_samples = Vec::new();
        let mut committed_tokens = 0;
        for iteration in 0..20 {
            let generated_before = requests.iter().map(|r| r.generated).sum::<usize>();
            let at = Instant::now();
            let before = model.scheduling_statistics().batch_execution;
            let mut refs: Vec<_> = requests.iter_mut().collect();
            if count == 1 {
                model.advance_requests(&mut refs, &options).unwrap();
            } else {
                let mut output = Vec::new();
                model
                    .execute_selected(&mut refs, &selected, &mut output)
                    .unwrap();
            }
            let committed = requests.iter().map(|r| r.generated).sum::<usize>() - generated_before;
            assert_eq!(committed, count, "Benchmark lost an active decoder");
            assert!(
                requests
                    .iter()
                    .all(|r| !r.is_finished() && r.failure().is_none())
            );
            if iteration >= 4 {
                committed_tokens += committed;
                samples.push(at.elapsed().as_secs_f64());
                let after = model.scheduling_statistics().batch_execution;
                if count > 1 {
                    device_samples
                        .push(after.replay_s + after.direct_s - before.replay_s - before.direct_s);
                }
            }
        }
        let seconds = samples.iter().sum::<f64>();
        results.push(json!({"batch":count,"committed_tokens":committed_tokens,"step_ms":seconds/samples.len() as f64*1000.,"decode_tps":committed_tokens as f64/seconds,"samples_s":samples,"device_samples_s":device_samples}));
        for request in &mut requests {
            model.finish_request(request, true).unwrap();
        }
    }
    std::fs::write(
        fixture.output,
        serde_json::to_vec_pretty(
            &json!({"cases":results,"scheduler":model.scheduling_statistics()}),
        )
        .unwrap(),
    )
    .unwrap();
}
