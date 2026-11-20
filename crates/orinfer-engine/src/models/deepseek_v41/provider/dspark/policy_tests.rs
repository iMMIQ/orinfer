use super::*;
#[test]
fn opportunity_amortizes_only_displaced_producer_tokens() {
    let b = Budget {
        work: Work {
            rows: 800,
            prefill_rows: 780,
            context: 100000,
        },
        prefill_s: 0.156,
        ..Budget::default()
    };
    assert!(
        (b.opportunity(Work {
            prefill_rows: 700,
            ..b.work
        }) - 0.016)
            .abs()
            < 1e-9
    );
    assert_eq!(b.opportunity(b.work), 0.);
    assert_eq!(
        b.opportunity(Work {
            prefill_rows: 790,
            ..b.work
        }),
        0.
    );
    assert_eq!(Budget::default().opportunity(Work::default()), 0.);
}
fn budget(rows: usize) -> Budget {
    Budget {
        rows,
        remaining: 1024,
        uniform: false,
        ..Budget::default()
    }
}
#[test]
fn owner_allocation_keeps_budget_block_tails_and_sampling_evidence() {
    let mut high = Feedback::default();
    let mut low = Feedback::default();
    for _ in 0..32 {
        high.observe(&[0.1; 5], 5, 5, 5);
        low.observe(&[0.99; 5], 5, 0, 5);
    }
    for latency in [false, true] {
        let original = [3, 3, 3, 3, 0];
        let maxima = [5, 5, 5, 5, 0];
        let feedback = [&high, &low, &high, &low, &high];
        let selected = redistribute(&original, &maxima, &feedback, 5, latency);
        assert_eq!(selected, [5, 1, 5, 1, 0]);
        assert_eq!(
            selected.iter().sum::<usize>(),
            original.iter().sum::<usize>()
        );
        let progress = |depths: &[usize]| {
            adaptive::progress(
                depths.iter().zip(feedback).map(|(&d, f)| f.progress(d, 5)),
                depths.len(),
                latency,
            )
        };
        assert!(progress(&selected) > progress(&original));
        let tail = redistribute(
            &[3, 1, 3, 0],
            &[5, 1, 5, 0],
            &[&high, &low, &low, &high],
            5,
            latency,
        );
        assert_eq!(tail.iter().sum::<usize>(), 7);
        assert_eq!((tail[1], tail[3]), (1, 0));
        let unknown = Feedback::default();
        assert_eq!(
            redistribute(&[3; 4], &[5; 4], &[&unknown; 4], 5, latency),
            [3; 4]
        );
    }
}
#[test]
fn uncalibrated_mixed_selection_keeps_feedback_cohort_uninitialized() {
    let mut p = Policy::default();
    let feedback = Feedback::default();
    let limits = Budget {
        work: Work {
            rows: 800,
            prefill_rows: 798,
            context: 131072,
        },
        ..budget(128)
    };
    assert_eq!(
        p.select_requests(false, true, limits, &[5; 2], &[&feedback; 2], &[5])
            .unwrap(),
        vec![0; 2]
    );
    assert!(!p.curves.contains_key(&(2, false, true)));
}
#[test]
fn shrinking_batch_samples_ordinary_decode_before_borrowing_its_price() {
    let mut p = Policy::default();
    for _ in 0..6 {
        p.baseline(4, false, false, 0.12);
        for (stage, block, rows, seconds) in [
            (Stage::Target, 0, 12, 0.10),
            (Stage::Draft, 5, 2, 0.006),
            (Stage::Restore, 5, 2, 0.0002),
        ] {
            p.costs.observe(Point {
                shared: false,
                shared_setup_s: 0.,
                stage,
                block,
                full: false,
                replay: false,
                work: Work {
                    rows,
                    ..Work::default()
                },
                seconds,
                setup_s: 0.,
            });
        }
    }
    let feedback = Feedback::default();
    // Four-owner direct execution overprices the unseen two-owner decode.
    // Reusing target/draft evidence must leave an ordinary sample possible.
    assert_eq!(
        p.select_requests(false, false, budget(12), &[5; 2], &[&feedback; 2], &[5])
            .unwrap(),
        vec![0; 2]
    );
    for _ in 0..6 {
        p.baseline(2, false, false, 0.02);
    }
    assert_eq!(
        p.select_requests(false, false, budget(12), &[5; 2], &[&feedback; 2], &[5])
            .unwrap(),
        vec![0; 2]
    );
}
#[test]
fn single_owner_timings_do_not_prevent_bounded_c32_calibration() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(32, false, false, 0.113);
    }
    for (stage, block, rows, seconds) in [
        (Stage::Target, 0, 6, 0.110),
        (Stage::Draft, 5, 1, 0.0125),
        (Stage::Restore, 5, 1, 0.0003),
    ] {
        p.costs.observe(Point {
            shared: false,
            shared_setup_s: 0.,
            stage,
            block,
            full: false,
            replay: false,
            work: Work {
                rows,
                ..Work::default()
            },
            seconds,
            setup_s: 0.,
        });
    }
    let states = (0..32).map(|_| Feedback::default()).collect::<Vec<_>>();
    let feedback = states.iter().collect::<Vec<_>>();
    let selected = p
        .select_requests(
            false,
            false,
            Budget {
                uniform: true,
                ..budget(192)
            },
            &[5; 32],
            &feedback,
            &[5, 8, 16],
        )
        .unwrap();
    assert_eq!(selected, vec![5; 32]);
    // The same unknown shape cannot bypass the actual loss account.
    p.ledger.balance = -1.;
    assert_eq!(
        p.select_requests(
            false,
            false,
            Budget {
                uniform: true,
                ..budget(192)
            },
            &[5; 32],
            &feedback,
            &[5, 8, 16],
        )
        .unwrap(),
        vec![0; 32]
    );
}
#[test]
fn ordinary_prices_include_host_commit_and_record_one_sample_per_round() {
    let mut p = Policy::default();
    let work = Work {
        rows: 16,
        ..Work::default()
    };
    for round in 0..4 {
        p.start_iteration(0.4);
        p.begin_round((0..16).collect(), true, work, true, 1);
        let setup = if round == 0 { 0.4 } else { 0. };
        p.baseline_execution(
            16,
            true,
            work,
            &super::super::super::super::team::Timing {
                shared: false,
                shared_setup: 0.,
                seconds: 0.069,
                preparation: setup,
                setup,
                replay: true,
                ..Default::default()
            },
        );
        if round == 0 {
            assert!(p.costs.predict(Stage::Decode, 0, true, work).is_none());
        }
        p.finish_iteration(
            0.096 + setup,
            &(0..16).map(|owner| (owner, 1)).collect::<Vec<_>>(),
        );
    }
    let priced = p.costs.predict(Stage::Decode, 0, true, work).unwrap();
    assert_eq!(priced.observations, 4);
    assert_eq!(priced.samples, adaptive::HOT_SAMPLES);
    assert!((priced.seconds - 0.096).abs() < 1e-12);
    assert_eq!(p.curves[&(16, true, false)].calls, 4);
}
#[test]
fn new_owners_reuse_cohort_priors_without_resetting_their_own_rejections() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(16, false, false, 0.070);
    }
    for _ in 0..6 {
        p.observe(
            16,
            (false, false, false),
            [0.014, 0.380, 0.001, 0.],
            &[(5, 5); 16],
            5,
        );
    }
    let mut f = Feedback::default();
    p.prepare_feedback(16, false, false, false, &mut f);
    assert_eq!(
        f.blocks[&5].acceptance.sample_counts(),
        &[0; crate::execution::MAX_MTP_DRAFTS]
    );
    assert_eq!(
        p.select_requests(false, false, budget(96), &[5; 16], &[&f; 16], &[5])
            .unwrap(),
        [5; 16]
    );
    for _ in 0..32 {
        f.observe(&[1.; 5], 5, 0, 5);
    }
    p.prepare_feedback(16, false, false, false, &mut f);
    assert!(f.blocks[&5].acceptance.rate(0, 0.7) < 0.001);
    assert_eq!(
        p.select_requests(false, false, budget(96), &[5; 16], &[&f; 16], &[5])
            .unwrap(),
        [0; 16]
    );
}
#[test]
fn a_prefill_only_packet_has_no_speculative_plan_or_budget_effect() {
    let mut p = Policy::default();
    assert!(
        p.select_requests(false, true, budget(127), &[], &[], &[5, 8, 16])
            .unwrap()
            .is_empty()
    );
    assert!(p.curves.is_empty());
    assert_eq!(p.ledger.startup_spent, 0.);
}
#[test]
fn trained_acceptance_prediction_learns_a_request_local_bias() {
    let mut h = History::default();
    for _ in 0..32 {
        h.observe(&[0.05; 5], 5, 5);
    }
    assert!(h.rates(5, &[])[4] > 0.8);
    let mut rejected = History::default();
    for _ in 0..32 {
        rejected.observe(&[1.; 5], 5, 0);
    }
    assert!(rejected.rates(5, &[])[0] < 0.1);
}
#[test]
fn acceptance_head_changes_expected_progress_with_the_same_history() {
    let high = History {
        previous: vec![0.9; 5],
        errors: [ForecastError {
            head: 0.,
            history: 1.,
            observed: true,
        }; crate::execution::MAX_MTP_DRAFTS],
        ..Default::default()
    };
    let low = History {
        previous: vec![0.1; 5],
        errors: [ForecastError {
            head: 0.,
            history: 1.,
            observed: true,
        }; crate::execution::MAX_MTP_DRAFTS],
        ..Default::default()
    };
    assert!(high.rates(5, &[]).iter().sum::<f64>() > low.rates(5, &[]).iter().sum::<f64>());
    let invalid = History {
        previous: vec![f64::NAN, f64::INFINITY, -1., 2., f64::NAN],
        ..Default::default()
    };
    assert_eq!(invalid.rates(5, &[]), History::default().rates(5, &[]));
}
#[test]
fn rejected_prefix_does_not_calibrate_its_untested_suffix() {
    let mut history = History {
        previous: vec![0.9; 5],
        ..Default::default()
    };
    history.observe(&[0.9; 5], 5, 1);
    assert!(history.bias[0] > 0.);
    assert!(history.bias[1] < 0.);
    assert_eq!(
        &history.bias[2..],
        &[0.; crate::execution::MAX_MTP_DRAFTS - 2]
    );
    let before = history.bias;
    history.previous.fill(f64::NAN);
    history.observe(&[0.5; 5], 5, 0);
    assert_eq!(history.bias, before);
}
#[test]
fn confidence_calibration_scores_the_lagged_forecast_without_lookahead() {
    let mut history = History {
        previous: vec![0.9],
        ..Default::default()
    };
    history.observe(&[0.1], 1, 0);
    assert!((history.bias[0] + 0.18).abs() < 1e-12);
    assert!(!history.errors[0].useful());
    assert_eq!(history.rates(1, &[])[0], history.acceptance.rate(0, 0.7));
}
#[test]
fn a_predictive_head_can_beat_the_request_history() {
    let mut history = History {
        previous: vec![0.9],
        ..Default::default()
    };
    history.observe(&[0.9], 1, 1);
    assert!(history.errors[0].useful());
    assert!(history.rates(1, &[])[0] > history.acceptance.rate(0, 0.7));
}
#[test]
fn actual_earnings_fund_a_new_cohort_without_recreating_startup_credit() {
    let mut p = Policy::default();
    p.start_iteration(0.4);
    p.begin_round(
        vec![1],
        true,
        Work {
            rows: 1,
            ..Work::default()
        },
        false,
        1,
    );
    p.finish_iteration(5., &[(1, 1)]);
    assert!((p.ledger.balance - 0.1).abs() < 1e-12);
    assert_eq!(p.ledger.startup_spent, 0.);
    p.start_iteration(0.4);
    p.begin_round(
        vec![2, 3, 4, 5],
        true,
        Work {
            rows: 4,
            ..Work::default()
        },
        true,
        6,
    );
    p.finish_iteration(0.15, &[(2, 0), (3, 0), (4, 0), (5, 0)]);
    assert!((p.ledger.startup_spent - 0.05).abs() < 1e-12);
    assert_eq!(p.ledger.balance, 0.);
}
#[test]
fn actual_losses_survive_context_and_curve_changes_and_settle_once() {
    let mut p = Policy::default();
    p.baseline(1, false, false, 0.02);
    p.start_iteration(0.4);
    p.begin_round(
        vec![7],
        false,
        Work {
            rows: 1,
            context: 2048,
            ..Work::default()
        },
        true,
        6,
    );
    p.finish_iteration(0.8, &[(7, 0), (9, 100)]);
    assert_eq!(p.ledger.startup_spent, 0.4);
    assert!((p.ledger.balance + 0.4).abs() < 1e-12);
    p.finish_iteration(0.8, &[(7, 0)]);
    assert!((p.ledger.balance + 0.4).abs() < 1e-12);
    p.curves.clear();
    p.start_iteration(0.4);
    p.begin_round(
        vec![11],
        false,
        Work {
            rows: 1,
            context: 131072,
            ..Work::default()
        },
        true,
        1,
    );
    p.finish_iteration(0.02, &[(11, 1)]);
    assert!((p.ledger.balance + 0.3996).abs() < 1e-12);
    assert_eq!(p.ledger.startup_spent, 0.4);
}
#[test]
fn slow_first_replay_finishes_hot_calibration_instead_of_disabling_the_graph() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(16, true, false, 0.067);
    }
    for (replay, target, setup) in [
        (false, 0.45, [0.01, 0.02]),
        (false, 0.44, [0., 0.]),
        (true, 0.4, [0.06, 0.3]),
    ] {
        p.observe(
            16,
            (true, false, false),
            Costs {
                shared: [false; 2],
                shared_setup: [0.; 2],
                stages: [0.01, target, 0.001, 0.],
                draft_replay: replay,
                verify_replay: replay,
                work: Work {
                    rows: 96,
                    context: 2048,
                    ..Work::default()
                },
                setup,
                wall_s: None,
            },
            &[(5, 5); 16],
            5,
        );
    }
    let fs = (0..16)
        .map(|_| {
            let mut f = Feedback::default();
            for _ in 0..2 {
                f.observe(&[0.99; 5], 5, 5, 5);
            }
            f
        })
        .collect::<Vec<_>>();
    let refs = fs.iter().collect::<Vec<_>>();
    let limits = Budget {
        work: Work {
            rows: 16,
            context: 2048,
            ..Work::default()
        },
        rows: 96,
        remaining: 512,
        latency: true,
        slack_s: 1.2,
        ..Budget::default()
    };
    assert_eq!(
        p.select_requests(true, false, limits, &[5; 16], &refs, &[5])
            .unwrap(),
        [5; 16]
    );
}
#[test]
fn initial_direct_baseline_cannot_mint_speculation_savings() {
    let mut p = Policy::default();
    let f = Feedback::default();
    p.baseline(1, false, false, 0.12);
    assert_eq!(
        p.select_requests(false, false, budget(17), &[16], &[&f], &[5, 8, 16])
            .unwrap(),
        [0]
    );
    for _ in 0..4 {
        p.baseline(1, false, false, 0.02);
    }
    assert!(
        p.select_requests(false, false, budget(17), &[16], &[&f], &[5, 8, 16])
            .unwrap()[0]
            > 0
    );
}
#[test]
fn cold_prefill_preparation_cannot_block_a_parallel_draft_probe() {
    let mut p = Policy::default();
    p.costs.observe(Point {
        shared: false,
        shared_setup_s: 0.,
        stage: Stage::Target,
        block: 0,
        full: true,
        replay: false,
        work: Work {
            rows: 1,
            prefill_rows: 1,
            context: 0,
        },
        seconds: 0.34,
        setup_s: 1.8,
    });
    for _ in 0..4 {
        p.baseline_execution(
            1,
            true,
            Work {
                rows: 1,
                context: 318,
                ..Work::default()
            },
            &super::super::super::super::team::Timing {
                shared: false,
                shared_setup: 0.,
                seconds: 0.11,
                preparation: 0.,
                setup: 0.,
                replay: false,
                ..Default::default()
            },
        );
    }
    let f = Feedback::default();
    let mut limits = budget(17);
    limits.work = Work {
        rows: 1,
        context: 318,
        ..Work::default()
    };
    limits.latency = true;
    limits.slack_s = 0.4;
    assert_eq!(
        p.select_requests(true, false, limits, &[16], &[&f], &[5, 8, 16])
            .unwrap(),
        [5]
    );
}
#[test]
fn selecting_without_execution_never_mints_or_spends_credit() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(1, false, false, 0.02);
    }
    let f = Feedback::default();
    let expected = p
        .select_requests(false, false, budget(16), &[5], &[&f], &[5])
        .unwrap();
    for _ in 0..100 {
        assert_eq!(
            p.select_requests(false, false, budget(16), &[5], &[&f], &[5])
                .unwrap(),
            expected
        );
    }
    assert_eq!(p.ledger.startup_spent, 0.);
    assert_eq!(p.ledger.balance, 0.);
    assert_eq!(p.curves[&(1, false, false)].calls, 4);
}
#[test]
fn a_cheaper_larger_block_cannot_price_a_shorter_physical_plan() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(2, false, false, 0.02);
    }
    for _ in 0..16 {
        p.observe(
            2,
            (false, false, false),
            [0.25, 0.03, 0.001, 0.],
            &[(3, 3); 2],
            5,
        );
        p.observe(
            2,
            (false, false, false),
            [0.001, 0.03, 0.001, 0.],
            &[(8, 8); 2],
            8,
        );
    }
    let f = Feedback::default();
    let mut limits = budget(8);
    limits.uniform = true;
    assert_eq!(
        p.select_requests(false, false, limits, &[8; 2], &[&f, &f], &[5, 8, 16])
            .unwrap(),
        [0; 2]
    );
}
#[test]
fn latency_uses_mean_owner_tpot_while_throughput_uses_total_yield() {
    let mut p = Policy::default();
    let mut high = Feedback::default();
    let mut low = Feedback::default();
    for _ in 0..4 {
        p.baseline(2, false, false, 0.02);
    }
    for _ in 0..16 {
        p.observe(
            2,
            (false, false, false),
            [0.003, 0.046, 0.001, 0.],
            &[(5, 5), (5, 0)],
            5,
        );
        high.observe(&[1.; 5], 5, 5, 5);
        low.observe(&[0.; 5], 5, 0, 5);
    }
    let mut limits = budget(12);
    limits.latency = true;
    assert_eq!(
        p.select_requests(false, false, limits, &[5; 2], &[&high, &low], &[5])
            .unwrap(),
        [0; 2]
    );
    let selected = p
        .select_requests(false, false, budget(12), &[5; 2], &[&high, &low], &[5])
        .unwrap();
    assert!(selected[0] > 0);
}
#[test]
fn stable_depth_uses_the_shared_cost_curve_and_keeps_cheap_prefixes() {
    for (accepted, expected) in [(5, 5), (1, 1)] {
        let mut policy = Policy::default();
        for _ in 0..4 {
            policy.baseline(2, false, false, 0.020);
        }
        for _ in 0..16 {
            policy.observe(
                2,
                (false, false, true),
                [0.003, 0.030, 0.001, 0.],
                &[(5, accepted); 2],
                5,
            );
        }
        let a = Feedback::default();
        let b = Feedback::default();
        // Ordinary decode and one-draft verification have different
        // execution geometry. Supply the measured short verification cost.
        for _ in 0..4 {
            policy.costs.observe(Point {
                shared: false,
                shared_setup_s: 0.,
                stage: Stage::Target,
                block: 0,
                full: false,
                replay: false,
                work: Work {
                    rows: 4,
                    ..Work::default()
                },
                seconds: 0.022,
                setup_s: 0.,
            });
        }
        let mut limits = budget(0);
        limits.uniform = true;
        assert_eq!(
            policy
                .select_requests(false, false, limits, &[5, 5], &[&a, &b], &[5, 8, 16])
                .unwrap(),
            [expected; 2]
        );
    }
}
#[test]
fn noisy_large_blocks_do_not_disable_the_trained_short_block() {
    let mut policy = Policy::default();
    let mut state = Feedback::default();
    for _ in 0..16 {
        policy.baseline(1, false, false, 0.020);
        policy.observe(
            1,
            (false, false, false),
            [0.005, 0.04, 0.001, 0.],
            &[(5, 5)],
            5,
        );
        state.observe(&[1.; 5], 5, 5, 5);
    }
    let expected = state.rates(5, &[], 5);
    for _ in 0..4 {
        policy.observe(
            1,
            (false, false, false),
            [0.02, 0.1, 0.002, 0.],
            &[(16, 0)],
            16,
        );
        state.observe(&[0.01; 16], 16, 0, 16);
    }
    assert_eq!(state.rates(5, &[], 5), expected);
    assert!(
        expected
            .iter()
            .enumerate()
            .all(|(i, &p)| (p - (1. - 0.15 * 0.8_f64.powi(16)).powi(i as i32 + 1)).abs() < 1e-12)
    );
    assert_eq!(
        policy
            .select_requests(false, false, budget(0), &[16], &[&state], &[5, 16])
            .unwrap(),
        [5]
    );
}
#[test]
fn cold_setup_is_amortized_and_short_tails_reuse_the_warm_block() {
    let (mut policy, state) = measured(5, 0.99, 5, [0.003, 0.030, 0.001]);
    policy.observe(
        1,
        (false, false, false),
        [0.003, 0.030, 0.001, 0.5],
        &[(5, 5)],
        5,
    );
    assert_eq!(
        policy
            .select_requests(
                false,
                false,
                Budget {
                    rows: 0,
                    remaining: 16,
                    uniform: false,
                    ..Budget::default()
                },
                &[16],
                &[&state],
                &[5, 16]
            )
            .unwrap(),
        [5]
    );
    let wider = policy
        .select_requests(false, false, budget(0), &[16], &[&state], &[5, 16])
        .unwrap();
    assert!(wider[0] > 5 && wider[0] <= 16);
    let prior = [0.9, 0.5, 0.2];
    assert_eq!(
        Feedback::default().rates(3, &prior, 5),
        vec![0.9, 0.9 * 0.5, 0.9 * 0.5 * 0.2]
    );
}
fn measured(
    maximum: usize,
    confidence: f64,
    accepted: usize,
    timing: [f64; 3],
) -> (Policy, Feedback) {
    let mut p = Policy::default();
    let mut f = Feedback::default();
    for _ in 0..4 {
        p.baseline(1, false, false, 0.020);
    }
    for _ in 0..16 {
        p.observe(
            1,
            (false, false, false),
            [timing[0], timing[1], timing[2], 0.],
            &[(maximum, accepted)],
            maximum,
        );
        f.observe(&vec![confidence; maximum], maximum, accepted, maximum);
    }
    (p, f)
}
#[test]
fn profitable_large_blocks_and_unprofitable_sampling_use_separate_costs() {
    let (mut fast, f) = measured(16, 0.99, 16, [0.003, 0.035, 0.001]);
    assert_eq!(
        fast.select_requests(false, false, budget(0), &[16], &[&f], &[16])
            .unwrap(),
        [16]
    );
    assert_eq!(
        fast.select_requests(true, false, budget(0), &[16], &[&f], &[16])
            .unwrap(),
        [0]
    );
    assert_eq!(
        fast.select_requests(
            false,
            true,
            Budget {
                work: Work {
                    rows: 129,
                    prefill_rows: 128,
                    context: 2048
                },
                ..budget(0)
            },
            &[16],
            &[&f],
            &[16]
        )
        .unwrap(),
        [0]
    );
    let (mut slow, f) = measured(16, 0.01, 0, [0.010, 0.040, 0.003]);
    assert_eq!(
        slow.select_requests(false, false, budget(0), &[16], &[&f], &[16])
            .unwrap(),
        [0]
    );
}
#[test]
fn expensive_cold_probes_respect_loss_budget_and_round_rows() {
    let (mut p, f) = measured(5, 0.8, 3, [0.003, 1.0, 0.001]);
    let limits = || Budget {
        rows: 4,
        remaining: 1024,
        uniform: true,
        ..Budget::default()
    };
    // A new width needs promotion even though a larger direct-only width
    // was too expensive. Observe exactly the rows that were actually run.
    for _ in 0..3 {
        let selected = p
            .select_requests(false, false, limits(), &[5], &[&f], &[5])
            .unwrap();
        assert_eq!(selected, [0]);
        assert_eq!(p.ledger.startup_spent, 0.);
    }
    assert_eq!(
        p.select_requests(false, false, limits(), &[5], &[&f], &[5])
            .unwrap(),
        [0]
    );
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(2, false, false, 0.020);
    }
    let limits = Budget {
        rows: 7,
        uniform: true,
        ..budget(7)
    };
    let a = Feedback::default();
    let b = Feedback::default();
    assert_eq!(
        p.select_requests(false, false, limits, &[5; 2], &[&a, &b], &[5])
            .unwrap(),
        [2; 2]
    );
}
#[test]
fn direct_mixed_baseline_and_partial_replay_feedback_do_not_deadlock() {
    let mut policy = Policy::default();
    let work = Work {
        rows: 128,
        prefill_rows: 112,
        context: 8192,
    };
    let feedback = (0..16).map(|_| Feedback::default()).collect::<Vec<_>>();
    let feedback = feedback.iter().collect::<Vec<_>>();
    let limits = || Budget {
        work,
        rows: 96,
        remaining: 1024,
        uniform: true,
        prefill_s: 112. / 4000.,
        remaining_by_owner: vec![],
        ..Budget::default()
    };
    for _ in 0..4 {
        policy.baseline_execution(
            16,
            false,
            work,
            &super::super::super::super::team::Timing {
                shared: false,
                shared_setup: 0.,
                seconds: 0.1,
                preparation: 0.01,
                setup: 0.,
                replay: false,
                ..Default::default()
            },
        );
    }
    let selected = policy
        .select_requests(false, true, limits(), &[5; 16], &feedback, &[5])
        .unwrap();
    assert_eq!(selected, [5; 16]);
    policy.observe(
        16,
        (false, true, true),
        Costs {
            shared: [false; 2],
            shared_setup: [0.; 2],
            stages: [0.01, 0.1, 0.001, 0.],
            draft_replay: true,
            verify_replay: false,
            work: work.extend(selected.iter().sum()),
            setup: [0.; 2],
            wall_s: None,
        },
        &selected.iter().map(|&n| (n, n)).collect::<Vec<_>>(),
        5,
    );
    // Draft replay has no direct sample yet: a bounded subsequent probe
    // can seed it, while target feedback is already usable independently.
    assert!(
        policy
            .costs
            .predict(Stage::Target, 0, false, work.extend(selected.iter().sum()))
            .is_some()
    );
    assert!(
        policy
            .costs
            .predict(
                Stage::Draft,
                5,
                false,
                Work {
                    rows: 16,
                    ..Work::default()
                }
            )
            .is_some()
    );
    assert_eq!(
        policy.curves[&(16, false, true)].blocks[&(5, true)].samples,
        1
    );
}
#[test]
fn graph_cost_replaces_cold_direct_cost_and_budget_bounds_prefixes() {
    let (mut p, f) = measured(16, 0.99, 16, [0.250, 2.0, 0.01]);
    assert_eq!(
        p.select_requests(false, false, budget(0), &[16], &[&f], &[16])
            .unwrap(),
        [0]
    );
    for _ in 0..16 {
        p.observe(
            1,
            (false, false, false),
            [0.007, 0.036, 0.001, 0.],
            &[(16, 16)],
            16,
        );
    }
    let chosen = p
        .select_requests(false, false, budget(8), &[16], &[&f], &[16])
        .unwrap();
    assert!(chosen[0] > 0 && chosen[0] <= 7);
    assert_eq!(
        p.select_requests(false, false, budget(1), &[16], &[&f], &[16])
            .unwrap(),
        [0]
    );
}
#[test]
fn newer_confidence_allocates_a_fixed_acceptance_budget_without_cross_owner_leaks() {
    let mut p = Policy::default();
    for _ in 0..4 {
        p.baseline(2, false, false, 0.020);
    }
    for _ in 0..8 {
        p.observe(
            2,
            (false, false, false),
            [0.002, 0.025, 0.001, 0.],
            &[(8, 8); 2],
            8,
        );
    }
    let high = Feedback {
        blocks: [(
            8,
            History {
                previous: vec![0.99; 8],
                errors: [ForecastError {
                    head: 0.,
                    history: 1.,
                    observed: true,
                }; crate::execution::MAX_MTP_DRAFTS],
                ..Default::default()
            },
        )]
        .into(),
    };
    let low = Feedback {
        blocks: [(
            8,
            History {
                previous: vec![0.01; 8],
                errors: [ForecastError {
                    head: 0.,
                    history: 1.,
                    observed: true,
                }; crate::execution::MAX_MTP_DRAFTS],
                ..Default::default()
            },
        )]
        .into(),
    };
    let a = p
        .select_requests(false, false, budget(8), &[8, 8], &[&high, &low], &[8])
        .unwrap();
    let b = p
        .select_requests(false, false, budget(8), &[8, 8], &[&low, &high], &[8])
        .unwrap();
    assert_eq!(a.iter().sum::<usize>(), b.iter().sum::<usize>());
    assert!(a[0] > a[1]);
    assert_eq!(a, [b[1], b[0]]);
    let reset = Feedback::default();
    assert!(reset.blocks.is_empty());
}
