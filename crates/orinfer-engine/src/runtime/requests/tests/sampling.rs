//! Compare complete sampled requests and committed state using the same weights.
use super::*;

#[derive(Clone)]
struct RotatingMask {
    vocab: usize,
    step: usize,
}
impl crate::sampling::Constraint for RotatingMask {
    fn fork(&self) -> Box<dyn crate::sampling::Constraint> {
        Box::new(self.clone())
    }
    fn needs_mask(&self) -> bool {
        self.step < 5
    }
    fn mask(&mut self) -> Result<Vec<u32>> {
        let mut bits = vec![0; self.vocab.div_ceil(32)];
        for token in 0..self.vocab {
            if token % 4 != self.step % 4 {
                bits[token / 32] |= 1 << (token % 32);
            }
        }
        Ok(bits)
    }
    fn consume(&mut self, token: u32) -> Result<()> {
        if !self.needs_mask() {
            return Err("Released grammar consumed another token".into());
        }
        if token as usize >= self.vocab || token as usize % 4 == self.step % 4 {
            return Err("Masked token committed".into());
        }
        self.step += 1;
        Ok(())
    }
    fn finished(&self) -> bool {
        false
    }
}

#[test]
#[ignore = "Requires batched GPU sampler package and the exclusive GPU lock"]
fn validate_batched_gpu_sampling_requests() {
    let f: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!f.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let batch = model
        .manifest
        .sampling
        .as_ref()
        .unwrap()
        .batch
        .clone()
        .expect("Batched GPU sampler contract");
    let mut records = Vec::new();
    for count in if f.batches.is_empty() {
        vec![1, 3, 8]
    } else {
        f.batches.clone()
    } {
        let mut expected = None;
        for (pass, batched) in [false, true, true, false].into_iter().enumerate() {
            model.manifest.sampling.as_mut().unwrap().batch = batched.then_some(batch.clone());
            let mut create = |owner: usize, replacement: bool| {
                let mut source = input(&f.cases[owner % f.cases.len()], 32);
                source.sampling = if owner.is_multiple_of(5) && count > 1 {
                    crate::sampling::Options {
                        temperature: 0.,
                        ..Default::default()
                    }
                } else {
                    crate::sampling::Options {
                        temperature: 0.8,
                        top_k: [0, 7, 40][owner % 3],
                        top_p: [1., 0.9, 0.8][owner % 3],
                        presence_penalty: 0.2,
                        frequency_penalty: 0.1,
                        repetition_penalty: 1.1,
                        logit_bias: BTreeMap::from([(10, 0.25)]),
                        seed: crate::sampling::EVALUATION_SEED
                            + owner as u64
                            + u64::from(replacement) * 100,
                        ..Default::default()
                    }
                };
                let mut req = model.start_request(source, &|| false).unwrap();
                if owner % 3 == 2 {
                    req.set_constraint(Box::new(RotatingMask {
                        vocab: model.manifest.vocab,
                        step: 0,
                    }))
                    .unwrap();
                }
                req
            };
            let mut requests: Vec<_> = (0..count).map(|i| create(i, false)).collect();
            let options = scheduler::Options {
                policy: scheduler::Policy::Throughput,
                ..Default::default()
            };
            let before = model.execution.batch_statistics();
            let at = Instant::now();
            let mut tokens = vec![Vec::new(); count];
            let mut iteration = 0;
            let mut replaced = false;
            while requests.iter().any(|r| !r.is_finished()) {
                let mut owners: Vec<_> =
                    (0..count).filter(|&i| !requests[i].is_finished()).collect();
                let mut refs: Vec<_> = requests
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(i, r)| owners.contains(&i).then_some(r))
                    .collect();
                if iteration % 2 == 1 {
                    owners.reverse();
                    refs.reverse();
                }
                for step in model.advance_requests(&mut refs, &options).unwrap() {
                    tokens[owners[step.request]].extend(step.tokens);
                }
                assert!(requests.iter().all(|r| r.failure().is_none()));
                iteration += 1;
                if count > 1 && !replaced && requests[count - 1].generated >= 8 {
                    model
                        .finish_request(&mut requests[count - 1], false)
                        .unwrap();
                    let mut source = input(&f.cases[(count - 1) % f.cases.len()], 32);
                    source.sampling = requests[count - 1].sampling.clone();
                    source.sampling.seed += 100;
                    let mut req = model.start_request(source, &|| false).unwrap();
                    if (count - 1) % 3 == 2 {
                        req.set_constraint(Box::new(RotatingMask {
                            vocab: model.manifest.vocab,
                            step: 0,
                        }))
                        .unwrap();
                    }
                    requests[count - 1] = req;
                    replaced = true;
                }
            }
            let seconds = at.elapsed().as_secs_f64();
            let after = model.execution.batch_statistics();
            // Row 0 is greedy. Other owners exercise distinct parameters and
            // masks, reverse row order, then replace a cancelled arena mid-run.
            if batched && count > 1 {
                assert!(after.sampling_rows > before.sampling_rows);
            }
            if !batched {
                assert_eq!(after.sampling_rows, before.sampling_rows);
            }
            let states: Vec<_> = requests
                .iter_mut()
                .map(|r| single_decode::state(&mut model, r))
                .collect();
            if let Some((old_tokens, old_states)) = &expected {
                assert_eq!(&tokens, old_tokens, "Sampling packet changed seeded tokens");
                assert_eq!(
                    &states, old_states,
                    "Sampling packet changed committed state"
                );
            } else {
                expected = Some((tokens.clone(), states));
            }
            for req in &mut requests {
                model.finish_request(req, false).unwrap();
            }
            records.push(json!({"requests":count,"pass":pass,"batched":batched,"seconds":seconds,
                "tokens":tokens.iter().map(Vec::len).sum::<usize>(), "sampling_packets":after.sampling_packets-before.sampling_packets,
                "sampling_rows":after.sampling_rows-before.sampling_rows,"tokens_exact":true,"states_exact":true,"cancel_reuse":replaced}));
        }
    }
    std::fs::write(f.output, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires GPU sampler package and the exclusive GPU experiment lock"]
fn validate_gpu_sampling_requests() {
    let f: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    assert!(!f.output.exists());
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            prefix_cache_bytes: 0,
            mtp_drafts: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    let sampler = model
        .manifest
        .sampling
        .clone()
        .expect("GPU sampling contract");
    let options = scheduler::Options::default();
    let mut results = vec![];
    for (k, p) in [(0, 1.), (0, 0.9), (40, 0.8)] {
        let mut expected = None;
        let mut timings = vec![];
        for gpu in [false, true] {
            model.manifest.sampling = gpu.then_some(sampler.clone());
            let mut source = input(&f.cases[0], 64);
            source.sampling = crate::sampling::Options {
                temperature: 0.8,
                top_k: k,
                top_p: p,
                presence_penalty: 0.2,
                frequency_penalty: 0.1,
                repetition_penalty: 1.1,
                seed: crate::sampling::EVALUATION_SEED,
                ..Default::default()
            };
            let mut req = model.start_request(source, &|| false).unwrap();
            let start = Instant::now();
            let mut output = vec![];
            let mut maintenance_calls = 0;
            while !req.is_finished() {
                for step in model
                    .advance_requests_with(&mut [&mut req], &options, || {
                        maintenance_calls += 1;
                    })
                    .unwrap()
                {
                    output.extend(step.tokens);
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            assert_eq!(output.len(), 64);
            assert!(maintenance_calls > 0);
            let final_state = single_decode::state(&mut model, &mut req);
            if let Some((tokens, state)) = &expected {
                assert_eq!(
                    &output, tokens,
                    "GPU sampling changed seeded request output"
                );
                assert_eq!(
                    &final_state, state,
                    "GPU sampling changed committed causal state"
                );
            } else {
                expected = Some((output.clone(), final_state));
            }
            model.finish_request(&mut req, false).unwrap();
            timings.push(json!({"gpu":gpu,"seconds":elapsed,"tokens":output.len(),"request_tps":output.len() as f64 / elapsed,"maintenance_calls":maintenance_calls}));
        }
        results.push(json!({"top_k":k,"top_p":p,"seeded_tokens_equal":true,"committed_state_equal":true,"timings":timings}));
    }
    std::fs::write(f.output, serde_json::to_vec_pretty(&results).unwrap()).unwrap();
}

#[test]
#[ignore = "Requires Flash Next sampled verification and the exclusive GPU lock"]
fn validate_sampled_lookup_requests() {
    use crate::runtime::requests::speculation::LookupChoice;

    let f: Fixture = crate::model::read(&PathBuf::from(
        std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
    ))
    .unwrap();
    let mut model = ModelRuntime::load_with_options(
        &f.model,
        LoadOptions {
            cuda_graph: f.cuda_graph.parse().unwrap(),
            prompt_lookup: true,
            ..Default::default()
        },
    )
    .unwrap();
    let start = |model: &mut ModelRuntime| {
        (0..4)
            .map(|lane| {
                let mut source = input(&f.cases[lane], 44);
                source.sampling = crate::sampling::Options {
                    temperature: 0.8,
                    top_k: 20,
                    top_p: 0.9,
                    seed: crate::sampling::EVALUATION_SEED + lane as u64,
                    ..Default::default()
                };
                let mut request = model.start_request(source, &|| false).unwrap();
                prefill_reference(model, &mut request);
                request
            })
            .collect::<Vec<_>>()
    };
    let ordinary = [(0, 1), (1, 1), (2, 1), (3, 1)];
    let mut baseline = start(&mut model);
    let mut baseline_times = vec![];
    for _ in 0..34 {
        let at = Instant::now();
        model
            .execute_selected(
                &mut baseline.iter_mut().collect::<Vec<_>>(),
                &ordinary,
                &mut vec![],
            )
            .unwrap();
        baseline_times.push(at.elapsed().as_secs_f64());
    }
    let expected: Vec<_> = baseline
        .iter()
        .map(|req| req.history[req.input.len()..].to_vec())
        .collect();
    for request in &mut baseline {
        model.finish_request(request, false).unwrap();
    }

    let mut requests = start(&mut model);
    let mut packet_times = vec![];
    for (iteration, lookup) in [3, 1, 2, 0, 3, 1].into_iter().enumerate() {
        let mut draft = expected[lookup][requests[lookup].generated..][..7].to_vec();
        let accepted = if iteration.is_multiple_of(2) { 7 } else { 3 };
        if accepted < draft.len() {
            draft[accepted] = (draft[accepted] + 1) % model.manifest.vocab as u32;
        }
        let proposals = vec![(lookup, draft)];
        let other = ordinary.into_iter().filter(|&(i, _)| i != lookup).collect();
        let positions: Vec<_> = requests.iter().map(|r| r.warm.tokens).collect();
        let mut steps = vec![];
        let graph_hits = model.execution.batch_statistics().graph_hits;
        let at = Instant::now();
        model
            .execute_sampled_lookup(
                &mut requests.iter_mut().collect::<Vec<_>>(),
                LookupChoice {
                    ordinary: other,
                    proposals,
                },
                &mut steps,
                &mut || {},
            )
            .unwrap();
        packet_times.push(at.elapsed().as_secs_f64());
        if iteration >= 2 && f.cuda_graph == "decode_only" {
            assert!(model.execution.batch_statistics().graph_hits >= graph_hits + 2);
        }
        assert_eq!(steps.len(), 4);
        for step in steps {
            let req = &requests[step.request];
            assert_eq!(req.warm.tokens, positions[step.request]);
            assert_eq!(
                step.tokens.len(),
                if step.request == lookup {
                    accepted + 1
                } else {
                    1
                }
            );
            let range = req.generated - step.tokens.len()..req.generated;
            assert_eq!(step.tokens, expected[step.request][range]);
        }
    }
    let mut steps = vec![];
    model
        .execute_selected(
            &mut requests.iter_mut().collect::<Vec<_>>(),
            &ordinary,
            &mut steps,
        )
        .unwrap();
    for step in steps {
        let request = &requests[step.request];
        assert_eq!(step.tokens, [expected[step.request][request.generated - 1]]);
    }
    for request in &mut requests {
        model
            .with_request(request, |m, r| {
                assert!(m.refresh_request_mtp(r, true)?);
                assert_eq!(r.warm.tokens, r.history.len() - 1);
                Ok(())
            })
            .unwrap();
        model.finish_request(request, false).unwrap();
    }
    std::fs::write(
        &f.output,
        json!({
            "ordinary_seconds": baseline_times,
            "lookup_seconds": packet_times,
            "graph": model.execution.batch_statistics(),
        })
        .to_string(),
    )
    .unwrap();
}
