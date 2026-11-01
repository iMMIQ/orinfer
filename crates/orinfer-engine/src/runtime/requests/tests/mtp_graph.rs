//! Rebind speculative graph owners without recapture or private-state leakage.
use super::*;

fn private_state(
    model: &mut ModelRuntime,
    request: &mut RequestState,
) -> BTreeMap<String, Vec<u8>> {
    model
        .with_request(request, |m, _| {
            m.execution
                .private_buffer_sizes()
                .into_iter()
                .filter(|(_, bytes)| *bytes > 0)
                .map(|(name, bytes)| Ok((name.clone(), m.execution.download_bytes(&name, bytes)?)))
                .collect()
        })
        .unwrap()
}

#[test]
#[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
fn validate_mtp_packet_graph_reuse() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        output: std::path::PathBuf,
        cases: Vec<GenerationInput>,
    }
    let f: Fixture = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: crate::execution::CudaGraphMode::Full,
            prefix_cache_bytes: 0,
            mtp_drafts: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let spec = model.manifest.mtp.clone().unwrap();
    assert!(model.manifest.packet_table().is_some());
    let source = &f.cases[0];
    let mut retained = vec![];
    let mut evidence = vec![];
    let mut packet_plans = BTreeMap::new();
    for cohort in 0..2 {
        let mut requests = vec![];
        let mut rounds = vec![];
        for lane in 0..2 {
            let mut input_tokens = source.input_tokens.clone();
            input_tokens.extend([32, 40 + lane]);
            let mut req = model
                .start_request(
                    GenerationInput {
                        input_tokens,
                        audios: vec![],
                        images: source.images.clone(),
                        max_new_tokens: 32,
                        sampling: source.sampling.clone(),
                        prefix_hints: vec![],
                    },
                    &|| false,
                )
                .unwrap();
            while req.prefilling {
                model
                    .with_request(&mut req, |m, r| {
                        m.advance_prefill(r, r.input.len().min(4096), false)
                    })
                    .unwrap();
            }
            let Preparation::Round(round) = model
                .with_request(&mut req, |m, r| m.prepare_speculative_round(r, 2))
                .unwrap()
            else {
                panic!("Expected a verification round")
            };
            rounds.push(round);
            requests.push(req);
        }
        use orinfer_model_sdk::abi::BatchOperation;
        for phase in [
            BatchOperation::Verification,
            BatchOperation::Commit,
            BatchOperation::Warm,
            BatchOperation::DraftHead,
            BatchOperation::Draft,
        ] {
            eprintln!("Validate MTP graph cohort={cohort} phase={phase:?}");
            let mut segments = vec![];
            for (req, round) in requests.iter_mut().zip(&rounds) {
                let tokens = match phase {
                    BatchOperation::Verification | BatchOperation::Commit => round.plan.tokens,
                    _ => 1,
                };
                model
                    .with_request(req, |m, r| {
                        match phase {
                            BatchOperation::Verification => {
                                m.upload_segment_controls(tokens)?;
                                m.execution
                                    .ensure_sequence_program(r.slot, &round.plan.program)?;
                                m.upload_ids(&m.manifest.input, &round.verification)?;
                                m.prepare_verification_inputs(
                                    &round.plan.program,
                                    &round.verification,
                                    &r.history[..r.history.len() - 1],
                                )?;
                            }
                            BatchOperation::Commit => m.upload_ids(&spec.accepted_inputs, &[1])?,
                            BatchOperation::Warm => {
                                let (warm, _) = spec.warm_plan(1)?;
                                m.upload_ids(&spec.input, &round.verification[..1])?;
                                m.prepare_program_inputs(
                                    &warm.program,
                                    &round.verification[..1],
                                    &[],
                                )?;
                                m.execution.ensure_sequence_program_at(
                                    r.slot,
                                    &warm.program,
                                    round.position,
                                    1,
                                )?;
                            }
                            BatchOperation::Draft => {
                                let token = m.read_control(&spec.token)? as u32;
                                m.prepare_program_inputs(&spec.draft_program, &[token], &[])?;
                                m.execution.ensure_sequence_program_at(
                                    r.slot,
                                    &spec.draft_program,
                                    round.position + 1,
                                    1,
                                )?;
                            }
                            _ => {}
                        }
                        Ok(())
                    })
                    .unwrap();
                segments.push(BatchSegment {
                    slot: req.slot,
                    tokens,
                });
            }
            let (plan, _) = model.model_package.cached_batch(&segments, phase).unwrap();
            if let Some(previous) = packet_plans.get(&phase) {
                assert!(
                    std::rc::Rc::ptr_eq(previous, &plan),
                    "{phase:?} rebuilt the host plan"
                );
            } else {
                packet_plans.insert(phase, plan);
            }
            let mut snapshots = vec![];
            for req in &mut requests {
                snapshots.push(
                    model
                        .with_request(req, |m, _| {
                            let ranges = m
                                .execution
                                .private_buffer_sizes()
                                .into_iter()
                                .filter(|(_, n)| *n > 0)
                                .map(|(name, bytes)| {
                                    (name, crate::backend::snapshot::Range { offset: 0, bytes })
                                })
                                .collect();
                            let plan =
                                m.execution
                                    .plan_snapshot(ranges, &Default::default(), None)?;
                            m.execution
                                .snapshot(&plan)?
                                .ok_or("Insufficient snapshot memory".into())
                        })
                        .unwrap(),
                );
            }
            let inactive: Vec<_> = retained
                .iter_mut()
                .map(|r| private_state(&mut model, r))
                .collect();
            let mode = model
                .execution
                .set_graph_mode_for_test(crate::execution::CudaGraphMode::Off);
            if phase == BatchOperation::Commit {
                model.commit_spec_packet(&segments).unwrap();
            } else {
                model.execute_spec_packet(&segments, phase).unwrap();
            }
            let expected: Vec<_> = requests
                .iter_mut()
                .map(|r| private_state(&mut model, r))
                .collect();
            model.execution.set_graph_mode_for_test(mode);
            let before = model.scheduling_statistics().batch_execution;
            for repeat in 0..3 {
                for (req, snapshot) in requests.iter_mut().zip(&snapshots) {
                    model
                        .with_request(req, |m, _| m.execution.restore_snapshot(snapshot))
                        .unwrap();
                }
                if phase == BatchOperation::Commit {
                    model.commit_spec_packet(&segments).unwrap();
                } else {
                    model.execute_spec_packet(&segments, phase).unwrap();
                }
                let hashes: Vec<_> = requests
                    .iter_mut()
                    .map(|r| private_state(&mut model, r))
                    .collect();
                assert!(hashes == expected, "{phase:?} replay {repeat}");
                let untouched: Vec<_> = retained
                    .iter_mut()
                    .map(|r| private_state(&mut model, r))
                    .collect();
                assert!(
                    untouched == inactive,
                    "{phase:?} changed inactive request state"
                );
            }
            for snapshot in snapshots {
                model.execution.release_snapshot(snapshot).unwrap();
            }
            let after = model.scheduling_statistics().batch_execution;
            assert!(
                after.graph_hits > before.graph_hits,
                "{phase:?} did not replay"
            );
            if cohort == 1 {
                assert_eq!(
                    after.dynamic_graph_captures, before.dynamic_graph_captures,
                    "{phase:?} recaptured for replacement owners"
                );
            }
            evidence.push(serde_json::json!({"cohort":cohort,"phase":format!("{phase:?}"),"owners":segments.iter().map(|s|s.slot).collect::<Vec<_>>(),"new_graphs":after.dynamic_graph_captures-before.dynamic_graph_captures,"hits":after.graph_hits-before.graph_hits,"private_state_equal":true}));
        }
        retained.extend(requests);
    }
    for req in &mut retained {
        model.finish_request(req, false).unwrap();
    }
    std::fs::write(f.output, serde_json::to_vec_pretty(&evidence).unwrap()).unwrap();
}
