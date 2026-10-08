use super::*;

#[test]
#[ignore = "Requires native Flash batch package and exclusive GPU experiment lock"]
fn validate_flash_decode_batches() {
    let fixture: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!fixture.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &fixture.model,
        LoadOptions {
            cuda_graph: fixture.cuda_graph.parse().unwrap(),
            // Compare identical cold prompt execution. A partial prefix hit
            // changes prompt chunk boundaries and tests a different path.
            prefix_cache_bytes: 0,
            mtp_drafts: None,
        },
    )
    .unwrap();
    assert!(!model.manifest.batch_profiles.is_empty());
    assert!(
        model
            .manifest
            .batch_layout
            .as_ref()
            .unwrap()
            .small_mixed_shapes
            .is_empty()
    );
    let spec = model.manifest.mtp.take();
    let options = scheduler::Options {
        memory_reserve_bytes: 256 << 20,
        ..Default::default()
    };
    let mut rows = vec![];
    for count in [2, 3, 4, 8, 16, 32] {
        let mut requests = vec![];
        for i in 0..count {
            let item = input(&fixture.cases[i % fixture.cases.len()], 4);
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
        while requests.iter().any(|r| !r.is_finished()) {
            let mut active: Vec<_> = requests.iter_mut().collect();
            for step in model.advance_requests(&mut active, &options).unwrap() {
                let raw = model
                    .with_request(&mut requests[step.request], |m, _| {
                        m.execution
                            .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
                    })
                    .unwrap();
                assert_eq!(step.tokens.len(), 1);
                outputs[step.request].extend(step.tokens);
                scores[step.request].push(log_probabilities(&floats(
                    &raw,
                    crate::artifact::Dtype::F32,
                )));
            }
        }
        for request in &mut requests {
            model.finish_request(request, true).unwrap();
        }
        for i in 0..count.min(fixture.cases.len()) {
            let mut baseline = model
                .start_request(input(&fixture.cases[i % fixture.cases.len()], 4), &|| false)
                .unwrap();
            while baseline.prefilling {
                model
                    .advance_requests(&mut [&mut baseline], &options)
                    .unwrap();
            }
            let mut kl = 0.;
            let mut nll_delta = 0.;
            let mut top3 = 0;
            for (step, &token) in outputs[i].iter().enumerate() {
                let raw = model
                    .with_request(&mut baseline, |m, _| {
                        m.execution
                            .download_bytes(&m.manifest.logits, m.manifest.vocab * 4)
                    })
                    .unwrap();
                let b = log_probabilities(&floats(&raw, crate::artifact::Dtype::F32));
                let a = &scores[i][step];
                kl += b.iter().zip(a).map(|(b, a)| b.exp() * (b - a)).sum::<f64>();
                nll_delta += b[token as usize] - a[token as usize];
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
            let row = json!({"batch":count,"case":i,"same_history_mean_kl":kl/n,"mean_nll_delta":nll_delta/n,"selected_in_reference_top3":top3 as f64/n,"tokens":outputs[i]});
            eprintln!("{row}");
            assert!(
                kl / n < 0.05 && (nll_delta / n).abs() < 0.1,
                "Batch changed target distribution: {row}"
            );
            rows.push(row);
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
    // After real batched target steps, speculative execution must be able to
    // rebuild the draft from each request's captured hidden/state history.
    model.manifest.mtp = spec;
    if model.manifest.mtp.is_some() {
        let mut requests = vec![];
        for i in 0..8 {
            let item = input(&fixture.cases[i % fixture.cases.len()], 12);
            if !model.can_admit_request(&item, &options).unwrap() {
                break;
            }
            requests.push(model.start_request(item, &|| false).unwrap());
        }
        assert!(
            requests.len() >= 2,
            "Need two admitted requests for batch-to-MTP regression"
        );
        while requests.iter().any(|r| r.generated < 3) {
            model
                .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
                .unwrap();
        }
        for r in &mut requests[1..] {
            model.finish_request(r, false).unwrap();
        }
        while !requests[0].is_finished() {
            model
                .advance_requests(&mut [&mut requests[0]], &options)
                .unwrap();
        }
        assert!(requests[0].failure().is_none());
        model.finish_request(&mut requests[0], true).unwrap();
    }
    assert!(model.reserved_requests.is_empty());
    assert_eq!(model.execution.active_kv_bytes(), 0);
    model.execution.reclaim_idle_state().unwrap();
    assert_eq!(model.execution.resident_kv_bytes(), 0);
    std::fs::write(fixture.output,serde_json::to_vec_pretty(&json!({"cases":rows,"cancel_reuse_state_equal":true,"batch_to_mtp":true,"scheduler":model.scheduler_statistics})).unwrap()).unwrap();
}
