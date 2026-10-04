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
#[ignore = "Requires real batch model and exclusive GPU experiment lock"]
fn validate_continuous_requests() {
    let fixture: Fixture =
        crate::model::read(&PathBuf::from(std::env::var("ORIN_BATCH_FIXTURE").unwrap())).unwrap();
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
    let mut rows = vec![];
    for count in [2, 3, 4, 5, 8] {
        let mut reference = Vec::new();
        for i in 0..count {
            let source = &fixture.cases[i % fixture.cases.len()];
            let mut tokens = vec![];
            model
                .generate(
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
    assert!(model.reserved_contexts.is_empty());
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
            .generate(
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
    for (i, source) in fixture.cases.iter().enumerate() {
        if source.images.is_empty() {
            assert_eq!(output[i], reference[i], "Cold text prefill changed output");
        }
    }
    for request in &mut requests {
        model.finish_request(request, true).unwrap();
    }
    for (i, source) in fixture.cases.iter().enumerate() {
        model
            .generate(
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
        assert!(kl / count < 0.005, "Batch changed reference law: {row}");
        assert!(
            (nll_delta / count).abs() < 0.05,
            "Batch changed target likelihood: {row}"
        );
        assert!(top3 as f64 / count >= 0.9, "Poor token choices: {row}");
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
