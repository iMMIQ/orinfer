use super::*;
#[test]
fn eager_mode_never_promotes_hot_packets_and_retires_any_captured_rank() {
    let hot = CacheState::Direct {
        uses: 255,
        consecutive: 255,
    };
    let plan = direct_plan(&[hot, hot]).unwrap();
    assert!(!plan.capture());
    assert!(!plan.invalidate);
    assert_eq!(plan.reason, "graph_disabled");
    for states in [[hot, CacheState::Graph], [hot, CacheState::Missing]] {
        let plan = direct_plan(&states).unwrap();
        assert!(plan.invalidate);
        assert!(!plan.capture());
    }
    assert!(direct_plan(&[]).is_err());
}
fn budget(uses: f64) -> CaptureBudget {
    CaptureBudget {
        uses,
        max_uses: None,
        direct_s: Some(0.105),
        replay_s: Some(0.070),
        capture_s: Some(0.294),
        slack_s: 1.,
        trace: false,
    }
}
#[test]
fn latency_credit_is_bounded_and_repays_capture_overrun() {
    assert!((capture_credit(0., 0.4, 0.18, 1) - 0.22).abs() < 1e-12);
    assert!((capture_credit(0.22, 0.4, 0.42, 1) - 0.20).abs() < 1e-12);
    assert_eq!(capture_credit(0.4, 0.4, 0.01, 1), 0.4);
    assert_eq!(capture_credit(0.2, 0.4, 1.0, 1), 0.);
    assert_eq!(capture_credit(0., 0.4, 0.54, 6), 0.4);
    assert!((capture_credit(0.2, 0.4, 0.1, 0) - 0.1).abs() < 1e-12);
}
#[test]
fn speculative_progress_can_pay_for_capture_without_unbounded_burst_slack() {
    let budget = CaptureBudget {
        uses: 64.,
        max_uses: None,
        direct_s: Some(0.54),
        replay_s: None,
        capture_s: Some(0.30),
        slack_s: capture_interval(0.4, 6.) - 0.30 + capture_credit(0., 0.4, 0.54, 6),
        trace: false,
    };
    assert!(budget.profitable(0));
    assert!(
        !CaptureBudget {
            slack_s: capture_interval(0.4, 1.) - 0.30,
            ..budget
        }
        .profitable(0)
    );
    assert_eq!(capture_interval(0.4, 6.), 0.8);
    assert_eq!(capture_interval(0.4, 1.), 0.4);
    assert_eq!(capture_interval(0.4, 0.), 0.4);
}
#[test]
fn short_tail_keeps_direct_and_long_packet_can_capture() {
    let states = [CacheState::Direct {
        uses: 2,
        consecutive: 2,
    }; 2];
    assert_eq!(
        plan(&states, true, Some(budget(5.))).unwrap().mode,
        Mode::Direct
    );
    assert_eq!(
        plan(&states, true, Some(budget(64.))).unwrap().mode,
        Mode::CaptureReplay
    );
    let no_slack = CaptureBudget {
        slack_s: 0.1,
        ..budget(64.)
    };
    assert_eq!(
        plan(&states, true, Some(no_slack)).unwrap().mode,
        Mode::Direct
    );
    assert_eq!(
        plan(&[CacheState::Graph; 2], true, Some(no_slack))
            .unwrap()
            .mode,
        Mode::Replay
    );
}
#[test]
fn first_replay_trial_uses_measured_cost_instead_of_a_fixed_reuse_count() {
    let measured = CaptureBudget {
        uses: 8.9409,
        max_uses: None,
        direct_s: Some(0.21065),
        replay_s: None,
        capture_s: Some(0.35755),
        slack_s: 1.1494,
        trace: false,
    };
    let ready = [CacheState::Direct {
        uses: 2,
        consecutive: 2,
    }; 8];
    assert!(plan(&ready, true, Some(measured)).unwrap().capture());
    assert!(
        !CaptureBudget {
            uses: 4.,
            max_uses: None,
            ..measured
        }
        .profitable(0)
    );
    assert!(
        !CaptureBudget {
            slack_s: 0.4,
            ..measured
        }
        .profitable(0)
    );
    assert!(
        !CaptureBudget {
            replay_s: Some(0.22),
            ..measured
        }
        .profitable(0)
    );
    assert!(!measured.profitable(3));
    assert!(
        !plan(&[CacheState::Missing; 8], true, Some(measured))
            .unwrap()
            .capture()
    );
}
#[test]
fn temporary_calibration_limit_survives_empirical_reuse_and_keeps_graph_hits() {
    let bounded = CaptureBudget {
        max_uses: Some(2.),
        ..budget(128.)
    };
    let hot = [CacheState::Direct {
        uses: 32,
        consecutive: 32,
    }; 8];
    assert!(!plan(&hot, true, Some(bounded)).unwrap().capture());
    assert_eq!(
        plan(&[CacheState::Graph; 8], true, Some(bounded))
            .unwrap()
            .mode,
        Mode::Replay
    );
    assert!(
        CaptureBudget {
            capture_s: Some(0.001),
            ..bounded
        }
        .profitable(0)
    );
}
#[test]
fn consecutive_verification_can_measure_replay_before_calibration_stops() {
    let measured = CaptureBudget {
        uses: 6.76517457959058,
        max_uses: None,
        direct_s: Some(0.182719728),
        replay_s: None,
        capture_s: Some(0.313406262),
        slack_s: 1.154249368,
        trace: false,
    };
    let ready = CacheState::Direct {
        uses: 4,
        consecutive: 4,
    };
    assert!(plan(&[ready; 8], true, Some(measured)).unwrap().capture());
    let scattered = CacheState::Direct {
        uses: 4,
        consecutive: 1,
    };
    assert!(
        !plan(&[scattered; 8], true, Some(measured))
            .unwrap()
            .capture()
    );
    assert!(!plan(&[ready; 8], false, Some(measured)).unwrap().capture());
    let mut lagging = [ready; 8];
    lagging[7] = CacheState::Direct {
        uses: 3,
        consecutive: 3,
    };
    assert!(!plan(&lagging, true, Some(measured)).unwrap().capture());
    for budget in [
        CaptureBudget {
            slack_s: 0.4,
            ..measured
        },
        CaptureBudget {
            replay_s: Some(0.19),
            ..measured
        },
    ] {
        assert!(!plan(&[ready; 8], true, Some(budget)).unwrap().capture());
    }
}
#[test]
fn retained_prompt_tail_can_learn_reuse_across_requests() {
    let unknown = CaptureBudget {
        uses: 1.,
        max_uses: None,
        direct_s: Some(0.14),
        replay_s: None,
        capture_s: Some(0.30),
        slack_s: f64::INFINITY,
        trace: false,
    };
    let state = |uses| CacheState::Direct {
        uses,
        consecutive: 1,
    };
    assert_eq!(
        plan(&[state(7); 8], false, Some(unknown)).unwrap().mode,
        Mode::Direct
    );
    assert_eq!(
        plan(&[state(8); 8], false, Some(unknown)).unwrap().mode,
        Mode::CaptureReplay
    );
    for budget in [
        CaptureBudget {
            replay_s: Some(0.20),
            ..unknown
        },
        CaptureBudget {
            slack_s: 0.10,
            ..unknown
        },
    ] {
        assert_eq!(
            plan(&[state(16); 8], false, Some(budget)).unwrap().mode,
            Mode::Direct
        );
    }
    assert_eq!(
        plan(&[CacheState::Missing, state(16)], false, Some(unknown))
            .unwrap()
            .mode,
        Mode::BuildDirect
    );
}
#[test]
fn retained_draft_shape_can_learn_reuse_across_short_cohorts() {
    let budget = CaptureBudget {
        uses: 1.,
        max_uses: None,
        direct_s: Some(0.014),
        replay_s: None,
        capture_s: Some(0.04),
        slack_s: 0.4,
        trace: false,
    };
    let state = |uses| CacheState::Direct {
        uses,
        consecutive: 1,
    };
    assert_eq!(
        plan(&[state(7); 8], true, Some(budget)).unwrap().mode,
        Mode::Direct
    );
    assert_eq!(
        plan(&[state(8); 8], true, Some(budget)).unwrap().mode,
        Mode::CaptureReplay
    );
    assert_eq!(
        plan(
            &[state(16); 8],
            true,
            Some(CaptureBudget {
                slack_s: 0.01,
                ..budget
            })
        )
        .unwrap()
        .mode,
        Mode::Direct
    );
}
#[test]
fn asymmetric_cache_defer_invalidates_every_rank() {
    let states = [
        CacheState::Graph,
        CacheState::Direct {
            uses: 3,
            consecutive: 3,
        },
    ];
    let p = plan(&states, true, Some(budget(2.))).unwrap();
    assert!(p.invalidate);
    assert_eq!(p.mode, Mode::BuildDirect);
    assert_eq!(
        plan(
            &[CacheState::Missing, CacheState::Graph],
            true,
            Some(budget(64.))
        )
        .unwrap()
        .mode,
        Mode::BuildDirect
    );
    // All-Missing across every rank builds direct without an empty Invalidate barrier.
    let cold = plan(&[CacheState::Missing; 8], true, Some(budget(64.))).unwrap();
    assert!(!cold.invalidate);
    assert_eq!(cold.mode, Mode::BuildDirect);
    let cold_direct = direct_plan(&[CacheState::Missing; 8]).unwrap();
    assert!(!cold_direct.invalidate);
    assert_eq!(cold_direct.mode, Mode::BuildDirect);
}
