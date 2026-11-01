//! Same-history checks for shared target verification and mixed packets.
use super::*;

#[test]
#[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
fn validate_joint_target_verification() {
    #[derive(serde::Deserialize)]
    struct Fixture {
        model: std::path::PathBuf,
        output: std::path::PathBuf,
        cases: Vec<GenerationInput>,
        cuda_graph: String,
        #[serde(default)]
        ordinary_steps: usize,
        #[serde(default)]
        heterogeneous: bool,
    }
    let f: Fixture = crate::model::read(&std::path::PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!f.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            mtp_drafts: Some(7),
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let spec = model.manifest.mtp.clone().unwrap();
    let options = scheduler::Options::default();
    let prepare = |model: &mut ModelRuntime, req: &mut RequestState| {
        while req.prefilling {
            model.advance_requests(&mut [req], &options).unwrap();
        }
        let fixed = model.dynamic_speculation.fixed_width;
        model.dynamic_speculation.fixed_width = Some(1);
        for _ in 0..f.ordinary_steps {
            model.advance_requests(&mut [req], &options).unwrap();
        }
        model.dynamic_speculation.fixed_width = fixed;
    };
    let input = |lane: usize, short: bool| {
        let mut tokens = f.cases[0].input_tokens.clone();
        if short {
            tokens.truncate(17);
        } else {
            tokens.extend([32, 33 + lane as u32]);
        }
        GenerationInput {
            audios: vec![],
            input_tokens: tokens,
            images: vec![],
            max_new_tokens: 16,
            sampling: f.cases[0].sampling.clone(),
            prefix_hints: vec![],
        }
    };
    let scores = |m: &ModelRuntime, buffer: &str, rows: usize| -> Result<Vec<Vec<f64>>> {
        let raw = m
            .execution
            .download_bytes(buffer, rows * m.manifest.vocab * 4)?;
        Ok(floats(&raw, crate::artifact::Dtype::F32)
            .chunks_exact(m.manifest.vocab)
            .map(|v| {
                let maximum = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let norm = maximum
                    + v.iter()
                        .map(|&x| (x as f64 - maximum).exp())
                        .sum::<f64>()
                        .ln();
                v.iter().map(|&x| x as f64 - norm).collect()
            })
            .collect())
    };
    let mut results = vec![];
    for mixed in [false, true] {
        for width in if f.heterogeneous {
            vec![2]
        } else {
            vec![2, 4, 8]
        } {
            let supported = if mixed {
                &spec.mixed_verification_tokens
            } else {
                &spec.batch_verification_tokens
            };
            if !supported.contains(&width) {
                continue;
            }
            for count in [1, 2, 4, 8] {
                let mut expected = vec![];
                let mut requests = vec![];
                let mut rounds = vec![];
                for lane in 0..count {
                    let lane_width = if f.heterogeneous {
                        let widths: Vec<_> = [2, 4, 8]
                            .into_iter()
                            .filter(|n| supported.contains(n))
                            .collect();
                        widths[lane % widths.len()]
                    } else {
                        width
                    };
                    let mut reference = model.start_request(input(lane, false), &|| false).unwrap();
                    prepare(&mut model, &mut reference);
                    let round = model
                        .with_request(&mut reference, |m, r| {
                            m.prepare_speculative_round(r, lane_width)
                        })
                        .unwrap();
                    let Preparation::Round(round) = round else {
                        panic!("Unexpected finished verification reference")
                    };
                    let baseline = model
                        .with_request(&mut reference, |m, r| {
                            // Use the original complete per-request verification body,
                            // independently of the packet path under test.
                            m.upload_segment_controls(round.plan.tokens)?;
                            m.upload_ids(&m.manifest.input, &round.verification)?;
                            m.prepare_verification_inputs(
                                &round.plan.program,
                                &round.verification,
                                &r.history[..r.history.len() - 1],
                            )?;
                            m.launch_rows(
                                &round.plan.program,
                                round.plan.tokens,
                                ExecutionPhase::Decode,
                            )?;
                            scores(m, &spec.verification_logits, round.plan.tokens)
                        })
                        .unwrap();
                    expected.push((round.verification.clone(), baseline));
                    model.finish_request(&mut reference, false).unwrap();
                    let mut req = model.start_request(input(lane, false), &|| false).unwrap();
                    prepare(&mut model, &mut req);
                    let Preparation::Round(round) = model
                        .with_request(&mut req, |m, r| m.prepare_speculative_round(r, lane_width))
                        .unwrap()
                    else {
                        panic!("Unexpected finished verification packet")
                    };
                    assert_eq!(round.verification, expected[lane].0);
                    requests.push(req);
                    rounds.push(round);
                }
                let mut prefix = None;
                let mut segments = vec![];
                let mut expected_prefix = None;
                if mixed {
                    let mut reference = model.start_request(input(0, true), &|| false).unwrap();
                    model
                        .with_request(&mut reference, |m, r| m.advance_prefill(r, 17, false))
                        .unwrap();
                    expected_prefix = Some(
                        model
                            .with_request(&mut reference, |m, _| scores(m, &m.manifest.logits, 1))
                            .unwrap(),
                    );
                    model.finish_request(&mut reference, false).unwrap();
                    let mut req = model.start_request(input(0, true), &|| false).unwrap();
                    segments = model
                        .prepare_batch_inputs(&mut [&mut req], &[(0, 17)])
                        .unwrap();
                    prefix = Some(req);
                }
                // Repeat one fixed-history packet through direct execution,
                // capture, then the hot replay path. Restore all private state,
                // not only positions/KV, before presenting identical inputs.
                let hot_replay = count == 2 && width == 2 && !mixed;
                let mut snapshots = Vec::new();
                if hot_replay {
                    for req in &mut requests {
                        snapshots.push(
                            model
                                .with_request(req, |m, _| {
                                    let ranges = m
                                        .execution
                                        .private_buffer_sizes()
                                        .into_iter()
                                        .filter(|(_, bytes)| *bytes > 0)
                                        .map(|(name, bytes)| {
                                            (
                                                name,
                                                crate::backend::snapshot::Range {
                                                    offset: 0,
                                                    bytes,
                                                },
                                            )
                                        })
                                        .collect();
                                    let plan =
                                        m.execution.plan_snapshot(ranges, &Default::default())?;
                                    m.execution
                                        .snapshot(&plan)?
                                        .ok_or("No room for replay state".into())
                                })
                                .unwrap(),
                        );
                    }
                }
                let hits = model.scheduling_statistics().batch_execution.graph_hits;
                for _ in 0..if hot_replay { 3 } else { 1 } {
                    if hot_replay {
                        segments.clear();
                        for (req, snapshot) in requests.iter_mut().zip(&snapshots) {
                            model
                                .with_request(req, |m, _| m.execution.restore_snapshot(snapshot))
                                .unwrap();
                        }
                    }
                    for (req, round) in requests.iter_mut().zip(&rounds) {
                        model.execution.activate_sequence(req.slot).unwrap();
                        model.upload_segment_controls(round.plan.tokens).unwrap();
                        model
                            .execution
                            .ensure_sequence_program(req.slot, &round.plan.program)
                            .unwrap();
                        model
                            .upload_ids(&model.manifest.input, &round.verification)
                            .unwrap();
                        model
                            .prepare_verification_inputs(
                                &round.plan.program,
                                &round.verification,
                                &req.history[..req.history.len() - 1],
                            )
                            .unwrap();
                        segments.push(BatchSegment {
                            slot: req.slot,
                            tokens: round.plan.tokens,
                        });
                    }
                    model
                        .execute_spec_packet(
                            &segments,
                            if mixed {
                                orinfer_model_sdk::abi::BatchOperation::MixedRows { prefills: 1 }
                            } else {
                                orinfer_model_sdk::abi::BatchOperation::Verification
                            },
                        )
                        .unwrap();
                }
                if hot_replay && f.cuda_graph != "off" {
                    assert!(
                        model.scheduling_statistics().batch_execution.graph_hits > hits,
                        "Repeated verification did not replay a graph"
                    );
                }
                for snapshot in snapshots {
                    model.execution.release_snapshot(snapshot).unwrap();
                }
                let mut max_kl = 0f64;
                let mut max_nll = 0f64;
                let mut compare = |baseline: &[Vec<f64>], actual: Vec<Vec<f64>>| {
                    assert_eq!(baseline.len(), actual.len());
                    for (b, a) in baseline.iter().zip(actual) {
                        let kl: f64 = b.iter().zip(&a).map(|(b, a)| b.exp() * (b - a)).sum();
                        let token = b
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.total_cmp(b))
                            .unwrap()
                            .0;
                        let delta = (a[token] - b[token]).abs();
                        assert!(
                            kl.abs() < 0.05 && delta < 0.1,
                            "verification KL={kl}, NLL={delta}, batch={count}, width={width}, mixed={mixed}"
                        );
                        max_kl = max_kl.max(kl.abs());
                        max_nll = max_nll.max(delta);
                    }
                };
                for ((req, round), (_, baseline)) in requests.iter_mut().zip(&rounds).zip(&expected)
                {
                    compare(
                        baseline,
                        model
                            .with_request(req, |m, _| {
                                scores(m, &spec.verification_logits, round.plan.tokens)
                            })
                            .unwrap(),
                    );
                    model.finish_request(req, false).unwrap();
                }
                if let Some(mut req) = prefix {
                    compare(
                        expected_prefix.as_ref().unwrap(),
                        model
                            .with_request(&mut req, |m, _| scores(m, &m.manifest.logits, 1))
                            .unwrap(),
                    );
                    model.finish_request(&mut req, false).unwrap();
                }
                model.execution.reclaim_idle_state().unwrap();
                assert_eq!(model.execution.resident_kv_bytes(), 0);
                results.push(serde_json::json!({"mixed":mixed,"width":width,"batch":count,"hot_replay":hot_replay,"max_kl":max_kl,"max_selected_nll_delta":max_nll}));
                std::fs::write(&f.output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
            }
        }
    }
}
