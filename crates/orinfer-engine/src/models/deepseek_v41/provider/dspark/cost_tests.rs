use super::*;
#[test]
fn cheap_profiled_prefix_waits_for_full_acceptance_calibration() {
    use super::super::{Budget, Feedback, History, Policy};
    let mut policy = Policy::default();
    for _ in 0..4 {
        policy.baseline(16, false, false, 0.08);
    }
    let state = Feedback {
        blocks: [(
            5,
            History {
                previous: vec![0.01; 5],
                ..Default::default()
            },
        )]
        .into(),
    };
    for (rows, seconds) in [(32, 0.005), (96, 0.1)] {
        for replay in [false, true] {
            policy.costs.insert(
                Point {
                    work: Work {
                        context: 0,
                        ..point(rows, replay, seconds).work
                    },
                    ..point(rows, replay, seconds)
                },
                false,
            );
        }
    }
    for (stage, block) in [(Stage::Draft, 5), (Stage::Restore, 5)] {
        policy.costs.insert(
            Point {
                stage,
                block,
                work: Work {
                    rows: 16,
                    ..Default::default()
                },
                ..point(16, false, 0.001)
            },
            false,
        );
    }
    let select = |p: &mut Policy| {
        p.select_requests(
            false,
            false,
            Budget {
                rows: 96,
                remaining: 256,
                uniform: true,
                work: Work {
                    rows: 16,
                    ..Default::default()
                },
                ..Default::default()
            },
            &[5; 16],
            &[&state; 16],
            &[5],
        )
        .unwrap()
    };
    assert_eq!(select(&mut policy), [5; 16]);
    for _ in 0..crate::mtp::adaptive::CALIBRATION_ROUNDS {
        policy.observe(
            16,
            (false, false, true),
            [0.001, 0.1, 0.001, 0.],
            &[(5, 0); 16],
            5,
        );
    }
    let selected = select(&mut policy);
    assert!(selected.iter().any(|&depth| depth > 0 && depth < 5));
}
#[test]
fn imported_timings_do_not_consume_the_runtime_primer() {
    let mut model = Model::default();
    let work = Work {
        rows: 96,
        context: 2048,
        ..Work::default()
    };
    model.insert(point(96, true, 0.29), false);
    assert!(model.profiled(Stage::Target, 0, false, work));
    assert!(!model.profiled(Stage::Target, 0, false, Work { rows: 80, ..work }));
    assert!(!model.profiled(
        Stage::Target,
        0,
        false,
        Work {
            context: 100256,
            ..work
        }
    ));
    assert!(
        !model
            .predict(Stage::Target, 0, false, work)
            .is_some_and(|p| p.replay)
    );
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 0));
    assert!(
        (model
            .predict_mode(Stage::Target, 0, false, work, true)
            .unwrap()
            .seconds
            - 0.29)
            .abs()
            < 1e-12
    );
    model.observe(point(96, true, 0.4));
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 1));
    model.observe(point(96, true, 0.29));
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (1, 2));
    assert!(
        (model
            .predict_mode(Stage::Target, 0, false, work, true)
            .unwrap()
            .seconds
            - 0.29)
            .abs()
            < 1e-12
    );
}
#[test]
fn capture_primer_is_not_a_stable_replay_cost() {
    let mut model = Model::default();
    for (replay, seconds) in [(false, 0.45), (false, 0.44), (true, 0.4)] {
        model.observe(point(96, replay, seconds));
    }
    let work = Work {
        rows: 96,
        context: 2048,
        ..Work::default()
    };
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (1, 3));
    model.observe(point(96, true, 0.29));
    assert!(
        (model
            .predict_mode(Stage::Target, 0, false, work, true)
            .unwrap()
            .seconds
            - 0.29)
            .abs()
            < 1e-12
    );
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (2, 4));
}
#[test]
fn full_cost_surface_learns_new_shapes_via_lru_and_finishes_calibration() {
    use super::super::Policy;
    let work = point(8, false, 0.06).work;
    for direct in [false, true] {
        let mut policy = Policy::default();
        if direct {
            policy.costs.observe(Point {
                stage: Stage::Decode,
                ..point(8, false, 0.06)
            });
        } else {
            policy.costs.insert(
                Point {
                    stage: Stage::Decode,
                    ..point(8, true, 0.03)
                },
                false,
            );
        }
        for rows in 1..=2047 {
            policy.costs.observe(point(rows, false, 0.1));
        }
        assert_eq!(policy.costs.cells.len(), 2048);
        assert_eq!(
            policy.baseline_budget(false, true, work, 128., 2.).max_uses,
            if direct { Some(3.) } else { Some(4.) }
        );
        let rounds = if direct { 3 } else { 4 };
        for _ in 0..rounds {
            policy.costs.observe(Point {
                stage: Stage::Decode,
                ..point(8, false, 0.06)
            });
        }
        assert_eq!(policy.costs.cells.len(), 2048);
        assert_eq!(
            policy.baseline_budget(false, true, work, 128., 2.).max_uses,
            None
        );
    }
    // Modes evict oldest entries instead of wiping the entire table at 2048.
    let mut model = Model::default();
    for rows in 1..=2048 {
        model.observe(point(rows, true, 0.05));
    }
    model.observe(point(3000, true, 0.05));
    assert_eq!(model.modes.len(), 2048);
    assert!(
        model
            .predict(Stage::Target, 0, false, point(2048, true, 0.05).work)
            .unwrap()
            .replay
    );
    assert!(
        model
            .predict(Stage::Target, 0, false, point(3000, true, 0.05).work)
            .unwrap()
            .replay
    );
}
#[test]
fn ordinary_calibration_does_not_buy_graph_reuse_after_auto_replans() {
    use super::super::Policy;
    let mut policy = Policy::default();
    let work = point(8, false, 0.06).work;
    policy.costs.insert(
        Point {
            stage: Stage::Decode,
            setup_s: 0.2,
            ..point(8, true, 0.03)
        },
        false,
    );
    for seen in 0..4 {
        let budget = policy.baseline_budget(false, true, work, 128., 2.);
        assert_eq!(budget.max_uses, Some((4 - seen) as f64));
        assert!(!budget.profitable(0));
        assert_eq!(
            policy.target_budget(false, true, work, 128., 2.).max_uses,
            None
        );
        policy.costs.observe(Point {
            stage: Stage::Decode,
            ..point(8, false, 0.06)
        });
    }
    let steady = policy.baseline_budget(false, true, work, 128., 2.);
    assert_eq!(steady.max_uses, None);
    assert!(steady.profitable(0));
    let mixed = policy.baseline_budget(
        false,
        false,
        Work {
            prefill_rows: 4,
            ..work
        },
        128.,
        2.,
    );
    assert_eq!(mixed.max_uses, None);
}
#[test]
fn distant_large_packets_are_priors_not_small_packet_calibration() {
    let mut model = Model::default();
    for _ in 0..5 {
        model.observe(point(96, false, 0.15));
    }
    let work = Work {
        rows: 6,
        context: 2048,
        ..Work::default()
    };
    let prior = model.predict(Stage::Target, 0, false, work).unwrap();
    assert!((prior.seconds - 0.15).abs() < 1e-12);
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (0, 0));
    assert_eq!(
        model.evidence(Stage::Target, 0, false, Work { rows: 48, ..work }),
        (4, 5)
    );
    assert_eq!(
        model.evidence(Stage::Target, 0, false, Work { rows: 47, ..work }),
        (0, 0)
    );
    for _ in 0..5 {
        model.observe(point(6, false, 0.02));
    }
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (4, 5));
}
#[test]
fn old_round_samples_do_not_block_calibration_in_a_new_context() {
    use super::super::{Budget, Feedback, Policy};
    let mut policy = Policy::default();
    for _ in 0..5 {
        policy.baseline(1, false, false, 0.02);
        policy.observe(
            1,
            (false, false, false),
            [0.008, 0.02, 0.001, 0.],
            &[(5, 4)],
            5,
        );
        policy.costs.observe(Point {
            stage: Stage::Decode,
            ..point(1, false, 0.02)
        });
        policy.costs.observe(point(96, false, 0.15));
    }
    let work = Work {
        rows: 1,
        context: 2048,
        ..Work::default()
    };
    let state = Feedback::default();
    assert!(
        policy.curves[&(1, false, false)].rounds[&(5, 6, 1, false)]
            .time
            .samples()
            >= 3
    );
    assert_eq!(
        policy
            .costs
            .evidence(Stage::Target, 0, false, Work { rows: 6, ..work }),
        (0, 0)
    );
    let allocation = policy
        .select_requests_with_forecast(
            (false, false),
            Budget {
                rows: 6,
                remaining: 256,
                work,
                ..Budget::default()
            },
            &[5],
            &[&state],
            &[5],
            |depths, _| {
                Ok(Some(Work {
                    rows: 1 + depths[0],
                    ..work
                }))
            },
        )
        .unwrap();
    assert_eq!(allocation.depths, [5]);
}
#[test]
fn calibration_combines_direct_and_replay_without_claiming_graph_residency() {
    let mut model = Model::default();
    for replay in [false, false, true, true] {
        model.observe(point(6, replay, if replay { 0.03 } else { 0.1 }));
    }
    let work = Work {
        rows: 5,
        context: 2048,
        ..Work::default()
    };
    assert_eq!(model.evidence(Stage::Target, 0, false, work), (2, 4));
    assert_eq!(
        model
            .evidence(Stage::Target, 0, false, Work { rows: 24, ..work })
            .0,
        0
    );
    let prediction = model.predict(Stage::Target, 0, false, work).unwrap();
    assert!(!prediction.replay);
    assert!((prediction.seconds - 0.1).abs() < 1e-12);
}
fn point(rows: usize, replay: bool, seconds: f64) -> Point {
    Point {
        shared: false,
        shared_setup_s: 0.,
        stage: Stage::Target,
        block: 0,
        full: false,
        replay,
        work: Work {
            rows,
            context: 2048,
            prefill_rows: 0,
        },
        seconds,
        setup_s: 0.,
    }
}
#[test]
fn shared_execution_and_capture_cost_do_not_pollute_direct_or_full_modes() {
    let mut model = Model::default();
    let work = point(800, false, 0.).work;
    model.observe(point(800, false, 0.10));
    model.observe(Point {
        shared: true,
        shared_setup_s: 0.03,
        setup_s: 0.05,
        ..point(800, false, 0.07)
    });
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.07
    );
    assert!((model.cold_estimate(Stage::Target, 0, work, false).unwrap() - 0.02).abs() < 1e-12);
    model.observe(point(800, true, 0.05));
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.05
    );
    model.observe(point(800, false, 0.10));
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.10
    );
}
#[test]
fn captures_survive_zero_setup_replays_and_price_direct_transition() {
    let mut model = Model::default();
    model.observe(Point {
        setup_s: 0.020,
        ..point(71, false, 0.105)
    });
    model.observe(Point {
        setup_s: 0.294,
        ..point(71, true, 0.070)
    });
    for _ in 0..32 {
        model.observe(point(71, true, 0.070));
    }
    let work = point(71, false, 0.).work;
    let short = model.budget(Stage::Target, 0, false, work, 5., 1.);
    assert_eq!(short.capture_s, Some(0.294));
    assert!(!short.profitable(0));
    assert!(
        model
            .budget(Stage::Target, 0, false, work, 64., 1.)
            .profitable(0)
    );
    assert_eq!(
        model.cold_estimate(Stage::Target, 0, work, false),
        Some(0.020)
    );
}
#[test]
fn distant_replay_priors_do_not_block_local_capture_probes() {
    for distant in [
        Work {
            rows: 20,
            context: 131072,
            ..Work::default()
        },
        Work {
            rows: 2048,
            context: 1024,
            ..Work::default()
        },
    ] {
        let mut model = Model::default();
        let work = Work {
            rows: 20,
            context: 1024,
            ..Work::default()
        };
        model.observe(Point {
            work,
            ..point(20, false, 0.11)
        });
        model.insert(
            Point {
                work: distant,
                setup_s: 0.25,
                ..point(distant.rows, true, 0.51)
            },
            false,
        );
        let prior = model.predict_mode(Stage::Target, 0, false, work, true);
        if distant.rows >= 128 {
            assert!(prior.is_none());
        } else {
            assert_eq!(prior.unwrap().seconds, 0.51);
        }
        let budget = model.budget(Stage::Target, 0, false, work, 64., 1.);
        assert!(budget.replay_s.is_none());
        assert!(budget.profitable(0));
        assert!(
            !model
                .budget(Stage::Target, 0, false, work, 5., 1.)
                .profitable(0)
        );

        // Nearby rows still price execution, without denying a local probe.
        model.insert(
            Point {
                work: Work {
                    rows: 32,
                    context: 2048,
                    ..Work::default()
                },
                ..point(32, true, 0.06)
            },
            false,
        );
        let budget = model.budget(Stage::Target, 0, false, work, 64., 1.);
        assert_eq!(budget.replay_s, None);
        assert!(budget.profitable(0));
    }
}
#[test]
fn larger_draft_graph_does_not_block_a_smaller_local_capture() {
    let mut model = Model::default();
    let work = Work {
        rows: 16,
        ..Work::default()
    };
    for (rows, replay, seconds, setup_s) in [(16, false, 0.014, 0.), (32, true, 0.022, 0.04)] {
        model.observe(Point {
            shared: false,
            shared_setup_s: 0.,
            stage: Stage::Draft,
            block: 5,
            full: false,
            replay,
            work: Work { rows, ..work },
            seconds,
            setup_s,
        });
    }
    assert_eq!(
        model
            .predict_mode(Stage::Draft, 5, false, work, true)
            .unwrap()
            .seconds,
        0.022
    );
    let budget = model.budget(Stage::Draft, 5, false, work, 40., 0.4);
    assert_eq!(budget.replay_s, None);
    assert!(budget.profitable(0));
    assert!(
        !model
            .budget(Stage::Draft, 5, false, work, 4., 0.4)
            .profitable(0)
    );
    model.observe(Point {
        shared: false,
        shared_setup_s: 0.,
        stage: Stage::Draft,
        block: 5,
        full: false,
        replay: true,
        work,
        seconds: 0.018,
        setup_s: 0.04,
    });
    assert!(
        !model
            .budget(Stage::Draft, 5, false, work, 40., 0.4)
            .profitable(0)
    );
}
#[test]
fn prompt_capture_does_not_price_a_producer_from_a_larger_final_window() {
    let mut model = Model::default();
    let work = Work {
        rows: 1921,
        prefill_rows: 1921,
        context: 0,
    };
    model.observe(Point {
        work,
        ..point(1921, false, 0.16)
    });
    model.insert(
        Point {
            work: Work {
                rows: 2048,
                prefill_rows: 2048,
                context: 0,
            },
            setup_s: 0.25,
            ..point(2048, true, 0.20)
        },
        false,
    );
    assert!(
        model
            .budget(Stage::Target, 0, false, work, 16., 1.)
            .replay_s
            .is_none()
    );
}
#[test]
fn verification_priors_stay_within_the_native_row_strategy() {
    let mut model = Model::default();
    model.observe(Point {
        stage: Stage::Decode,
        ..point(32, true, 0.113)
    });
    model.observe(point(24, true, 0.050));
    for (rows, seconds) in [(78, 0.065), (96, 0.067)] {
        model.observe(Point {
            work: Work {
                context: 8192,
                ..point(rows, true, seconds).work
            },
            setup_s: 0.295,
            ..point(rows, true, seconds)
        });
    }
    assert!(
        model
            .predict_mode(
                Stage::Target,
                0,
                false,
                Work {
                    rows: 192,
                    context: 2048,
                    prefill_rows: 0,
                },
                true
            )
            .is_none()
    );
    for (rows, seconds) in [(128, 0.10), (160, 0.11)] {
        model.observe(point(rows, true, seconds));
    }
    let prediction = model
        .predict_mode(
            Stage::Target,
            0,
            false,
            Work {
                rows: 192,
                context: 2048,
                prefill_rows: 0,
            },
            true,
        )
        .unwrap();
    assert!((prediction.seconds - 0.12).abs() < 1e-12);
}
#[test]
fn priors_do_not_imply_graph_residency_and_modes_do_not_mix() {
    let mut model = Model::default();
    model.insert(point(8, false, 0.04), false);
    model.insert(point(8, true, 0.02), false);
    let work = point(8, false, 0.).work;
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.04
    );
    model.observe(point(8, true, 0.02));
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.02
    );
    model.observe(point(8, false, 0.20));
    model.observe(point(8, true, 0.02));
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.02
    );
}
#[test]
fn unseen_batch_uses_shared_prior_and_replay_preserves_tiers() {
    let mut model = Model::default();
    for replay in [false, true] {
        model.insert(point(8, replay, 0.02), false);
        model.insert(point(16, replay, 0.04), false);
    }
    let work = point(12, false, 0.).work;
    assert!(
        (model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds
            - 0.03)
            .abs()
            < 1e-12
    );
    model
        .modes
        .insert((family(Stage::Target, 0, false, work), 12), (true, 1));
    assert_eq!(
        model
            .predict(Stage::Target, 0, false, work)
            .unwrap()
            .seconds,
        0.04
    );
}
