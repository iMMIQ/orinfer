//! Exercise ordinary/speculative transitions and bounded lazy draft catch-up.
use super::*;

#[test]
#[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
fn validate_dynamic_speculation_transitions() {
    let fixture: serde_json::Value = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let output = PathBuf::from(fixture["output"].as_str().unwrap());
    assert!(!output.exists());
    let mut model = ModelRuntime::load_with_options(
        std::path::Path::new(fixture["model"].as_str().unwrap()),
        LoadOptions {
            mtp_drafts: Some(3),
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let source: GenerationInput = serde_json::from_value(fixture["cases"][0].clone()).unwrap();
    let options = scheduler::Options::default();
    let mut requests = vec![];
    for _ in 0..3 {
        let mut request = model.start_request(input(&source, 128), &|| false).unwrap();
        while request.prefilling {
            model
                .advance_requests(&mut [&mut request], &options)
                .unwrap();
        }
        requests.push(request);
    }
    let feedback_ready = |runtime: &ModelRuntime, states: &mut [RequestState]| {
        let work = Work::SpeculativeBatch(speculation::BatchChoice {
            selected: (0..states.len()).map(|i| (i, 3)).collect(),
        });
        runtime
            .speculation_feedback(&states.iter_mut().collect::<Vec<_>>(), &work)
            .unwrap()
            .warm_ready
    };
    assert!(feedback_ready(&model, &mut requests));
    model.dynamic_speculation.fixed_width = Some(3);
    let joint: Vec<_> = (0..requests.len()).map(|i| (i, 1)).collect();
    let choice = model
        .select_auto_speculation(
            &requests.iter_mut().collect::<Vec<_>>(),
            &joint,
            &options,
            f64::INFINITY,
        )
        .unwrap();
    assert_eq!(choice.selected, vec![(0, 3), (1, 3), (2, 3)]);

    for _ in 0..3 {
        model
            .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
            .unwrap();
    }
    let before: usize = requests.iter().map(|r| r.mtp.rounds).sum();
    assert!(before > 0);
    assert!(feedback_ready(&model, &mut requests));
    // Advertise a small retention window to exercise the rollover scheduler
    // without generating 4096 tokens. Actual capture/storage stay unchanged.
    let ring = model
        .manifest
        .mtp
        .as_ref()
        .unwrap()
        .hidden_ring
        .clone()
        .unwrap();
    let buffer = model
        .manifest
        .buffers
        .iter_mut()
        .find(|b| b.name == ring)
        .unwrap();
    let physical = buffer.shape[0];
    buffer.shape[0] = 8;
    model.dynamic_speculation.fixed_width = Some(1);
    for _ in 0..24 {
        model
            .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
            .unwrap();
    }
    assert_eq!(before, requests.iter().map(|r| r.mtp.rounds).sum::<usize>());
    assert!(model.scheduler_statistics.mtp_catchup_tokens > 0);
    assert!(requests.iter().all(|r| !r.mtp_blocked));
    assert!(!feedback_ready(&model, &mut requests));
    model.dynamic_speculation.fixed_width = Some(3);
    for _ in 0..3 {
        model
            .advance_requests(&mut requests.iter_mut().collect::<Vec<_>>(), &options)
            .unwrap();
    }
    assert!(requests.iter().map(|r| r.mtp.rounds).sum::<usize>() > before);
    assert!(feedback_ready(&model, &mut requests));
    for req in &mut requests {
        model
            .with_request(req, |m, r| {
                assert_eq!(
                    m.read_control(&m.manifest.position)? as usize,
                    r.history.len() - 1
                );
                assert!(r.history.len() - 1 - r.warm.tokens <= 8);
                Ok(())
            })
            .unwrap();
    }
    model
        .manifest
        .buffers
        .iter_mut()
        .find(|b| b.name == ring)
        .unwrap()
        .shape[0] = physical;
    let slot = requests[1].slot;
    model.finish_request(&mut requests[1], false).unwrap();
    let mut replacement = model.start_request(input(&source, 8), &|| false).unwrap();
    assert_eq!(replacement.slot, slot);
    model.finish_request(&mut replacement, false).unwrap();
    for r in &mut requests {
        model.finish_request(r, false).unwrap();
    }
    std::fs::write(output,serde_json::to_vec_pretty(&json!({"passed":true,"ordinary_to_mtp":true,"bounded_catchup":true,"cancel_reuse":true,"scheduler":model.scheduler_statistics})).unwrap()).unwrap();
}
