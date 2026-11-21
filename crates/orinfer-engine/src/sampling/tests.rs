mod sampling_tests {
    use super::super::*;

    fn naive_law(vocab: usize, mut ordered: Vec<Score>, options: &Options) -> Distribution {
        ordered.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if options.top_k > 0 {
            ordered.truncate(options.top_k);
        }
        let maximum = ordered[0].1;
        for (_, mass) in &mut ordered {
            *mass = ((*mass - maximum) / options.temperature).exp();
        }
        let threshold = ordered.iter().map(|x| x.1).sum::<f64>() * options.top_p;
        let mut total = 0.0;
        let mut count = 0;
        for &(_, mass) in &ordered {
            total += mass;
            count += 1;
            if total >= threshold {
                break;
            }
        }
        ordered.truncate(count);
        for (_, mass) in &mut ordered {
            *mass /= total;
        }
        let probabilities = (count > 128).then(|| {
            let mut probabilities = vec![0.0; vocab];
            for &(id, p) in &ordered {
                probabilities[id as usize] = p;
            }
            probabilities
        });
        Distribution {
            probabilities,
            vocab,
            ordered,
            fallback: None,
        }
    }

    fn naive_draw(ordered: &[Score], uniform: f64) -> u32 {
        let mut mass = 0.0;
        let mut last = None;
        for &(id, p) in ordered {
            if p > 0.0 {
                last = Some(id);
            }
            mass += p;
            if uniform < mass {
                break;
            }
        }
        last.unwrap()
    }

    #[test]
    fn shortlist_matches_full_sort_with_ties_processors_and_masks() {
        let mut peaked: Vec<_> = (0..1025)
            .map(|i| -22.0 - (i * 17 % 101) as f32 / 7.0)
            .collect();
        peaked[..8].copy_from_slice(&[-0.0, 0.0, -0.0, 0.0, -0.25, -1.0, -3.0, -18.125]);
        let flat = vec![2.0; SHORTLIST_LIMIT + 1];
        let history = [0, 0, 1, 4, 4, 16, u32::MAX];
        for logits in [&peaked, &flat] {
            let mask = vec![0x5555_5555; logits.len().div_ceil(32)];
            for processors in [
                Options::default(),
                Options {
                    repetition_penalty: 1.3,
                    presence_penalty: -0.2,
                    frequency_penalty: 0.5,
                    logit_bias: [(1, -0.5), (4, 1.0), (256, 8.0)].into(),
                    ..Options::default()
                },
            ] {
                for temperature in [0.2, 1.0, 2.0] {
                    for top_p in [1.0, 0.9, 0.5, 0.1] {
                        for top_k in [0, 20] {
                            let options = Options {
                                temperature,
                                top_p,
                                top_k,
                                ..processors.clone()
                            };
                            for mask in [None, Some(mask.as_slice())] {
                                let scores =
                                    processed_scores(logits, &history, &options, mask).unwrap();
                                let expected = naive_law(logits.len(), scores.clone(), &options);
                                let law = Distribution::from_scores(logits.len(), scores, &options);
                                for id in 0..logits.len() as u32 {
                                    assert!(
                                        (law.probability(id) - expected.probability(id)).abs()
                                            < 1e-12
                                    );
                                }
                                for uniform in [0.0, 1e-15, 0.5, 0.999999999, 1.0 - f64::EPSILON] {
                                    assert_eq!(
                                        law.draw(uniform).unwrap(),
                                        naive_draw(&expected.ordered, uniform)
                                    );
                                }
                                if mask.is_none() {
                                    for step in 0..8 {
                                        let uniform = counter_uniform(options.seed, 0, step);
                                        assert_eq!(
                                            sample(logits, &history, &options, step as usize)
                                                .unwrap(),
                                            naive_draw(&expected.ordered, uniform)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let options = Options::default();
        let mut flat_shortlist =
            Shortlist::new(&Scores::Logits(Cow::Borrowed(&flat)), 1.0, |_, _| {});
        assert!(!flat_shortlist.truncate(0.9));
        assert_eq!(
            Distribution::from_logits(&peaked, &[], &options)
                .unwrap()
                .draw(0.0),
            Ok(0)
        );
        assert_eq!(
            sample(&[], &[], &options, 0),
            Err("Invalid sampling logits".into())
        );
        assert_eq!(
            sample(&[1.0, f32::INFINITY], &[], &options, 0),
            Err("Invalid sampling logits".into())
        );
        assert_eq!(
            Distribution::from_masked(&peaked, &[], &options, Some(&[0])).unwrap_err(),
            "Grammar permits no tokens"
        );
    }

    #[test]
    fn untruncated_draws_preserve_tail_order_and_cdf_boundaries() {
        let mut logits = vec![-1000.0; 257];
        logits[..5].copy_from_slice(&[0.0, -0.0, -18.125, -18.25, -42.0]);
        let options = Options::default();
        let expected = naive_law(
            logits.len(),
            processed_scores(&logits, &[], &options, None).unwrap(),
            &options,
        );
        let law = Distribution::from_logits(&logits, &[], &options).unwrap();
        assert_eq!(law.ordered.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        assert!(law.fallback.is_some());
        assert!(0.999999999 >= law.ordered.iter().map(|x| x.1).sum());
        let boundary = expected.probability(0);
        for uniform in [
            0.0,
            boundary.next_down(),
            boundary,
            0.999999999,
            1.0_f64.next_down(),
        ] {
            let token = naive_draw(&expected.ordered, uniform);
            assert_eq!(law.draw(uniform), Ok(token));
            assert_eq!(
                draw_stochastic(Scores::Logits(Cow::Borrowed(&logits)), &options, uniform),
                Ok(token)
            );
        }

        // A broad tail can hold enough mass to move a nucleus beyond 18T.
        let mut broad = vec![-18.125; 1025];
        broad[0] = 0.0;
        let options = Options {
            top_p: 0.999999,
            ..options
        };
        let reference = naive_law(
            broad.len(),
            processed_scores(&broad, &[], &options, None).unwrap(),
            &options,
        );
        let law = Distribution::from_logits(&broad, &[], &options).unwrap();
        assert!(law.ordered.len() > 128);
        assert_eq!(law.ordered, reference.ordered);
    }

    #[test]
    fn shortlist_residual_and_verification_match_full_sort() {
        let mut first = vec![-24.0; 257];
        first[..4].copy_from_slice(&[1.25, 0.0, 0.25, 1.0]);
        let mut second = first.clone();
        second.swap(0, 2);
        let mut draft_logits = first.clone();
        draft_logits.swap(0, 3);
        let mut logits = first.clone();
        logits.extend(&second);
        for top_p in [1.0, 0.9, 0.5, 0.1] {
            let options = Options {
                top_p,
                ..Options::default()
            };
            let reference = |row| {
                naive_law(
                    first.len(),
                    processed_scores(row, &[], &options, None).unwrap(),
                    &options,
                )
            };
            let p = reference(&first);
            let q = reference(&draft_logits);
            let next = reference(&second);
            let target = Distribution::from_logits(&first, &[], &options).unwrap();
            let draft = Distribution::from_logits(&draft_logits, &[], &options).unwrap();
            let residual: Vec<_> = (0..first.len() as u32)
                .map(|id| (id, (p.probability(id) - q.probability(id)).max(0.0)))
                .filter(|x| x.1 > 0.0)
                .collect();
            let total: f64 = residual.iter().map(|x| x.1).sum();
            let residual: Vec<_> = residual
                .into_iter()
                .map(|(id, w)| (id, w / total))
                .collect();
            let actual = target.residual(&draft).unwrap();
            for &(id, probability) in &residual {
                assert!((actual.probability(id) - probability).abs() < 1e-12);
            }
            let token = naive_draw(&q.ordered, 0.0);
            let proposal = crate::mtp::Proposal {
                token,
                distribution: draft,
            };
            for step in 0..32 {
                let accept = counter_uniform(options.seed, 0x4d54_5002, step as u64)
                    < (p.probability(token) / q.probability(token)).min(1.0);
                let expected = if accept {
                    vec![
                        token,
                        naive_draw(
                            &next.ordered,
                            counter_uniform(options.seed, 0, step as u64 + 1),
                        ),
                    ]
                } else {
                    vec![naive_draw(
                        &residual,
                        counter_uniform(options.seed, 0x4d54_5003, step as u64),
                    )]
                };
                assert_eq!(
                    verify(
                        std::slice::from_ref(&proposal),
                        &logits,
                        &[],
                        &options,
                        &mut Decoder::default(),
                        step,
                        first.len()
                    )
                    .unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    #[ignore = "release timing diagnostic"]
    fn sampling_release_timing() {
        use std::{hint::black_box, time::Instant};

        let logits: Vec<_> = (0usize..248_320)
            .map(|i| {
                let score = ((i * 12_307 % 248_321) as f32) / 248_321.0;
                if i % 1021 == 0 {
                    -6.0 * score
                } else {
                    -20.0 - 24.0 * score
                }
            })
            .collect();
        let options = Options::default();
        let measure = |f: &mut dyn FnMut(u64)| {
            for i in 0..8 {
                f(i);
            }
            let start = Instant::now();
            for i in 0..64 {
                f(i);
            }
            start.elapsed().as_secs_f64() * 1e3 / 64.0
        };
        let reference = || {
            let scores = processed_scores(black_box(&logits), &[], &options, None).unwrap();
            naive_law(logits.len(), scores, &options)
        };
        let reference_sample_ms = measure(&mut |i| {
            black_box(naive_draw(
                &reference().ordered,
                counter_uniform(options.seed, 0, i),
            ));
        });
        let reference_distribution_ms = measure(&mut |_| {
            black_box(reference());
        });
        let sample_ms = measure(&mut |i| {
            black_box(sample(black_box(&logits), &[], &options, i as usize).unwrap());
        });
        let distribution_ms = measure(&mut |_| {
            black_box(Distribution::from_logits(black_box(&logits), &[], &options).unwrap());
        });
        eprintln!(
            "vocab={} sample_ms={reference_sample_ms:.3}->{sample_ms:.3} distribution_ms={reference_distribution_ms:.3}->{distribution_ms:.3}",
            logits.len()
        );
    }

    #[test]
    fn unprocessed_scores_preserve_generic_bits_masks_and_seeded_sampling() {
        let mut logits = vec![
            -0.0,
            0.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        logits.extend((0..1024).map(|i| ((i * 17 % 101) as f32 - 50.0) / 7.0));
        let mask = vec![0x5555_5555; logits.len().div_ceil(32)];
        let history = [0, 0, 2, u32::MAX];
        for frequency in [0.0, -0.0] {
            let options = Options {
                frequency_penalty: frequency,
                ..Options::default()
            };
            // An irrelevant bias preserves values while exercising the generic
            // processor as a reference, including IEEE signed-zero behavior.
            let generic = Options {
                logit_bias: [(u32::MAX, 0.0)].into(),
                ..options.clone()
            };
            for mask in [None, Some(mask.as_slice())] {
                let fast = processed_scores(&logits, &history, &options, mask).unwrap();
                let reference = processed_scores(&logits, &history, &generic, mask).unwrap();
                assert_eq!(
                    fast.iter()
                        .map(|x| (x.0, x.1.to_bits()))
                        .collect::<Vec<_>>(),
                    reference
                        .iter()
                        .map(|x| (x.0, x.1.to_bits()))
                        .collect::<Vec<_>>()
                );
                for temperature in [0.0, 0.7, 2.0] {
                    for top_k in [0, 1, 20] {
                        let options = Options {
                            temperature,
                            top_k,
                            top_p: 0.8,
                            ..options.clone()
                        };
                        let generic = Options {
                            logit_bias: generic.logit_bias.clone(),
                            ..options.clone()
                        };
                        // Remove extreme logits here to exercise a broad CDF.
                        let fast =
                            Distribution::from_masked(&logits[7..], &history, &options, mask)
                                .unwrap();
                        let reference =
                            Distribution::from_masked(&logits[7..], &history, &generic, mask)
                                .unwrap();
                        assert_eq!(fast.ordered, reference.ordered);
                        for step in 0..100 {
                            let u = counter_uniform(EVALUATION_SEED, 0, step);
                            assert_eq!(fast.draw(u), reference.draw(u));
                        }
                    }
                }
            }
        }
        assert!(processed_scores(&[0.0, f32::NAN], &[], &Options::default(), Some(&[1])).is_err());
    }
    #[test]
    fn sparse_top_k_and_residual_match_dense_probabilities_and_draws() {
        let logits: Vec<_> = (0..257)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let draft_logits: Vec<_> = logits.iter().rev().copied().collect();
        for k in [1, 20, 128, 200, 0] {
            let options = Options {
                top_k: k,
                top_p: 0.95,
                ..Default::default()
            };
            let target = Distribution::from_logits(&logits, &[], &options).unwrap();
            let draft = Distribution::from_logits(&draft_logits, &[], &options).unwrap();
            if (1..=128).contains(&k) {
                assert!(target.probabilities.is_none());
            }
            let dense = |law: &Distribution| Distribution {
                probabilities: Some(law.dense_probabilities().into_owned()),
                vocab: law.vocab,
                ordered: law.ordered.clone(),
                fallback: None,
            };
            let a = target.residual(&draft).unwrap();
            let b = dense(&target).residual(&dense(&draft)).unwrap();
            assert_eq!(a.ordered, b.ordered);
            for id in 0..257 {
                assert_eq!(a.probability(id), b.probability(id));
            }
            for step in 0..100 {
                let u = counter_uniform(EVALUATION_SEED, 0, step);
                assert_eq!(a.draw(u), b.draw(u));
            }
            assert!(target.residual(&target).is_err());
        }
    }
    #[test]
    fn greedy_matches_distribution_processors_and_total_order() {
        for repeat in [0.5, 1.0, 1.05, 2.0, f64::MAX] {
            for presence in [-2.0, 0.0, 2.0] {
                for frequency in [-2.0, 0.0, 2.0] {
                    let options = Options {
                        temperature: 0.0,
                        repetition_penalty: repeat,
                        presence_penalty: presence,
                        frequency_penalty: frequency,
                        ..Options::default()
                    };
                    for logits in [vec![0.0, -0.0, 1.0, -2.0], vec![-0.0, 0.0], vec![1.0; 7]] {
                        let history = [0, 0, 2, 6, 100];
                        let reference = Distribution::from_logits(&logits, &history, &options)
                            .and_then(|d| d.draw(0.5));
                        assert_eq!(greedy(&logits, &history, &options), reference);
                    }
                }
            }
        }
        assert!(greedy(&[1., f32::NAN], &[], &Options::default()).is_err());
    }
    #[test]
    fn rounded_mass_does_not_select_an_underflowed_token() {
        let law = Distribution {
            probabilities: Some(vec![1.0 - 1e-12, 0.0]),
            vocab: 2,
            ordered: vec![(0, 1.0 - 1e-12), (1, 0.0)],
            fallback: None,
        };
        assert_eq!(law.draw(1.0 - f64::EPSILON).unwrap(), 0);
    }
    #[test]
    fn repetition_and_top_k_ties() {
        let mut options = Options {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Options::default()
        };
        assert_eq!(sample(&[3., 2.], &[0], &options, 0).unwrap(), 1);
        assert_eq!(sample(&[-1., -1.5], &[0], &options, 0).unwrap(), 1);
        options.temperature = 1.0;
        options.top_k = 2;
        let p = Distribution::from_logits(&[1., 1., 1., 1.], &[], &options).unwrap();
        assert_eq!(p.probability(0), 0.5);
        assert_eq!(p.probability(1), 0.5);
        assert_eq!(p.probability(2), 0.0);
        options.repetition_penalty = f64::MAX;
        assert!(Distribution::from_logits(&[-2.], &[0], &options).is_err());
    }
    #[test]
    fn greedy_penalty_nucleus_and_seed() {
        let mut options = Options {
            temperature: 0.0,
            ..Options::default()
        };
        assert_eq!(sample(&[1., 2., 3.], &[], &options, 0).unwrap(), 2);
        options.frequency_penalty = 2.0;
        assert_eq!(sample(&[1., 2., 3.], &[2, 2], &options, 0).unwrap(), 1);
        options = Options {
            top_p: 0.01,
            ..Options::default()
        };
        assert_eq!(sample(&[1., 2., 3.], &[], &options, 3).unwrap(), 2);
        options.top_p = 1.0;
        let first = sample(&[1., 2., 3.], &[], &options, 37).unwrap();
        assert_eq!(first, sample(&[1., 2., 3.], &[], &options, 37).unwrap());
        assert!(sample(&[f32::NAN], &[], &options, 0).is_err());
    }
}

mod philox_tests {
    use super::super::{EVALUATION_SEED, categorical_bits, categorical_key, counter_uniform};

    #[test]
    fn philox_matches_random123_known_answers() {
        // DEShawResearch/random123 tests/kat_vectors, Philox2x32-10.
        for (counter, key, expected) in [
            ([0, 0], 0, [0xff1d_ae59, 0x6cd1_0df2]),
            ([u32::MAX, u32::MAX], u32::MAX, [0x2c3f_628b, 0xab4f_d7ad]),
            (
                [0x243f_6a88, 0x85a3_08d3],
                0x1319_8a2e,
                [0xdd7c_e038, 0xf62a_4c12],
            ),
        ] {
            assert_eq!(categorical_bits(counter, key), expected);
        }
        assert_ne!(
            categorical_key(EVALUATION_SEED, 1),
            categorical_key(EVALUATION_SEED, 2)
        );
    }

    #[test]
    fn matches_frozen_python_counter_contract() {
        // Generated by the independently specified op25 Python contract.
        let vectors = [
            (0, 0, 6_720_141_781_353_792_u64),
            (1, 0, 4_787_244_564_846_236),
            (1, 1, 7_011_218_493_468_269),
            (2, 7, 1_098_476_309_306_172),
            (u64::MAX, u64::MAX, 7_253_473_543_011_370),
            (42, 8192, 6_879_436_830_168_841),
        ];
        for (request, step, numerator) in vectors {
            let actual = counter_uniform(EVALUATION_SEED, request, step);
            let expected = numerator as f64 / 9_007_199_254_740_992.0;
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert!((0.0..1.0).contains(&actual));
        }
    }
}

mod point_mass_tests {
    use super::super::*;
    #[test]
    fn shared_gumbel_draft_preserves_serial_autoregressive_samples() {
        fn law(history: &[u32], draft: bool) -> [f64; 3] {
            let last = history.last().copied().unwrap_or(0) as usize;
            let weights = if draft {
                [0.25, 0.4, 0.35]
            } else {
                [0.2, 0.3, 0.5]
            };
            std::array::from_fn(|i| weights[(i + last + history.len()) % 3])
        }
        fn sample(history: &[u32], seed: u64, draft: bool) -> u32 {
            let key = categorical_key(seed, history.len() as u64);
            law(history, draft)
                .into_iter()
                .enumerate()
                .map(|(token, p)| {
                    let bits = categorical_bits([token as u32, key[1]], key[0])[0];
                    let u = (f64::from(bits >> 9) + 0.5) / 8388608.0;
                    (token as u32, p.ln() - (-u.ln()).ln())
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0
        }
        for seed in 0..128 {
            let mut serial = vec![];
            for _ in 0..24 {
                serial.push(sample(&serial, seed, false));
            }
            for block in [1, 2, 5, 8] {
                let mut output = vec![];
                while output.len() < serial.len() {
                    let mut history = output.clone();
                    let mut drafts = vec![];
                    let mut targets = vec![];
                    for _ in 0..block {
                        targets.push(sample(&history, seed, false));
                        let token = sample(&history, seed, true);
                        drafts.push(token);
                        history.push(token);
                    }
                    targets.push(sample(&history, seed, false));
                    let committed = crate::mtp::greedy_commit(&drafts, &targets).unwrap();
                    output.extend(committed.into_iter().take(serial.len() - output.len()));
                }
                assert_eq!(output, serial, "seed={seed}, block={block}");
            }
        }
    }
    #[test]
    fn sampled_target_prefix_commit_retains_categorical_law() {
        let laws = [[0.2_f64, 0.3, 0.5], [0.6, 0.1, 0.3], [0.3, 0.4, 0.3]];
        let mut first = [0usize; 3];
        let mut accepted = [0usize; 2];
        let n = 32768;
        for seed in 0..n {
            let targets = laws
                .iter()
                .enumerate()
                .map(|(step, law)| {
                    let key = categorical_key(seed as u64, step as u64);
                    law.iter()
                        .enumerate()
                        .map(|(token, p)| {
                            let bits = categorical_bits([token as u32, key[1]], key[0])[0];
                            let u = (f64::from(bits >> 9) + 0.5) / 8388608.0;
                            (token as u32, p.ln() - (-u.ln()).ln())
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap()
                        .0
                })
                .collect::<Vec<_>>();
            let committed = crate::mtp::greedy_commit(&[1, 0], &targets).unwrap();
            first[committed[0] as usize] += 1;
            accepted[0] += usize::from(committed.len() > 1);
            accepted[1] += usize::from(committed.len() > 2);
        }
        for (observed, p) in first.into_iter().zip(laws[0]) {
            assert!((observed as f64 / n as f64 - p).abs() < 0.02);
        }
        assert!((accepted[0] as f64 / n as f64 - 0.3).abs() < 0.02);
        assert!((accepted[1] as f64 / n as f64 - 0.18).abs() < 0.02);
    }
    #[test]
    fn deterministic_proposal_retains_exact_residual_sampling() {
        let q = Distribution::point_mass(1, 3).unwrap();
        assert_eq!(q.probability(1), 1.0);
        assert_eq!(q.probability(0), 0.0);
        assert!(Distribution::point_mass(3, 3).is_err());
        let p = Distribution::from_logits(
            &[0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
            &[],
            &Options::default(),
        )
        .unwrap();
        let residual = p.residual(&q).unwrap();
        assert_eq!(residual.probability(1), 0.0);
        assert!((residual.probability(0) - 2.0 / 7.0).abs() < 1e-6);
        assert!((residual.probability(2) - 5.0 / 7.0).abs() < 1e-6);
    }
}
