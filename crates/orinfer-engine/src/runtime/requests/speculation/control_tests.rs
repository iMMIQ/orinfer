use super::*;

fn key(batch: usize) -> Key {
    Key {
        batch,
        context: 8,
        sampling: false,
        prefill: 0,
        shared_work: false,
    }
}
fn candidate(width: usize, expected_tokens: f64, prior_seconds: f64) -> Candidate {
    Candidate {
        width,
        expected_tokens,
        prior_seconds,
        remaining_tokens: 4096,
        warm_ready: true,
        review_acceptance: false,
    }
}
fn hot(c: &mut Controller, k: Key, width: usize, seconds: f64, tokens: usize) {
    c.observe(
        k,
        width,
        Measurement {
            seconds,
            tokens,
            preparation_s: 0.,
            warm_ready: true,
        },
    );
}
fn normal(c: &mut Controller, k: Key) {
    for _ in 0..4 {
        hot(c, k, 1, 0.03, k.batch);
    }
}

#[test]
fn acceptance_preserves_conditional_sampling_and_request_identity() {
    let mut a = Acceptance::default();
    a.observe(3, 1);
    assert_eq!(a.samples, [1, 1, 0, 0, 0, 0, 0]);
    assert!(a.expected(2) > 1. && a.expected(2) < 3.);
    a.observe(1, 1);
    assert_eq!(a.samples[1], 1);
    assert_eq!(Acceptance::default().samples, [0; 7]);
}

#[test]
fn successful_acceptance_observations_never_lower_their_own_estimate() {
    let mut a = Acceptance::default();
    a.observe(1, 0);
    for _ in 0..32 {
        let before = a.expected(1);
        a.observe(1, 1);
        assert!(a.expected(1) >= before);
    }
    let before = a.expected(1);
    a.observe(1, 0);
    assert!(a.expected(1) < before);
    assert_eq!(a.samples[1..], [0; 6]);
}

#[test]
fn cold_shapes_finish_in_width_order_without_mode_readiness_states() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let plans = [candidate(2, 7., 0.04), candidate(3, 9., 0.05)];
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 8);
    }
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.4), Some((3, true)));
        hot(&mut c, k, 3, 0.05, 12);
    }
    assert_eq!(c.choose(k, &plans, 0.4), Some((3, false)));
}

#[test]
fn global_minimum_is_not_hidden_by_compounding_per_depth_margins() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 8);
        hot(&mut c, k, 3, 0.05, 12);
    }
    let plans = [candidate(2, 6., 0.04), candidate(3, 7.6, 0.05)];
    // Both beat ordinary decoding by more than 2%. The deeper plan is the
    // global minimum even though its incremental improvement is below 2%.
    assert_eq!(c.choose(k, &plans, 0.4), Some((3, false)));
    assert_eq!(c.choose(k, &[plans[1], plans[0]], 0.4), Some((3, false)));
}

#[test]
fn profitable_calibration_does_not_require_a_full_round_of_credit() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    // Priming paid its actual loss from startup. Exhaust the startup pool to
    // exercise continuation using only this workload's marginal savings.
    hot(&mut c, k, 2, 0.04, 4);
    c.budget.startup_spent = c.calibration_limit;
    let plans = [candidate(2, 7., 0.04)];
    assert_eq!(c.budget.available(c.calibration_limit), 0.);
    for _ in 0..3 {
        assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 8);
    }
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, false)));
    assert_eq!(c.budget.startup_spent, c.calibration_limit);
    // Predicted profit is no grant: one unexpected loss blocks further probes.
    hot(&mut c, k, 3, 0.1, 0);
    assert!(c.budget.balance < 0.);
    assert_eq!(c.choose(k, &[candidate(3, 10., 0.05)], 0.4), None);
}

#[test]
fn draft_lag_needs_full_deadline_and_credit_but_can_pay_back_over_hot_rounds() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 8);
    }
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.1,
            tokens: 8,
            preparation_s: 0.02,
            warm_ready: false,
        },
    );
    let plan = candidate(2, 7., 0.04);
    let cold = Candidate {
        warm_ready: false,
        ..plan
    };
    assert_eq!(c.choose(k, &[plan], 0.05), Some((2, false)));
    assert_eq!(c.choose(k, &[cold], 0.05), None);
    assert_eq!(c.choose(k, &[cold], 0.4), Some((2, false)));
    c.budget.startup_spent = c.calibration_limit;
    hot(&mut c, k, 3, 0.1, 0);
    assert!(c.budget.balance < 0.);
    assert_eq!(c.choose(k, &[cold], 0.4), None);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 3);
    assert!((c.costs.profiles[&k].arms[&2].time.mean() - 0.04).abs() < 1e-12);
    // Clearing the physical lag permits the measured hot path immediately;
    // no controller-owned cooldown or readiness flag needs resetting.
    assert_eq!(c.choose(k, &[plan], 0.4), Some((2, false)));
}

#[test]
fn first_cold_sample_prices_retry_but_allows_consecutive_hot_calibration() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.1,
            tokens: 8,
            preparation_s: 0.02,
            warm_ready: false,
        },
    );
    c.budget.startup_spent = c.calibration_limit;
    let plan = candidate(2, 7., 0.04);
    assert_eq!(c.choose(k, &[plan], 0.4), Some((2, true)));
    assert_eq!(
        c.choose(
            k,
            &[Candidate {
                warm_ready: false,
                ..plan
            }],
            0.4
        ),
        None
    );
}

#[test]
fn ordinary_graph_recovery_cannot_mint_exploration_credit() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let before = c.budget.balance - c.budget.startup_spent;
    c.observe(
        k,
        1,
        Measurement {
            seconds: 0.13,
            tokens: 4,
            preparation_s: 0.2, // Nested timers may overlap; wall time is paid once.
            warm_ready: true,
        },
    );
    let after = c.budget.balance - c.budget.startup_spent;
    assert!((after - before - 0.13 * 0.02 + 0.1).abs() < 1e-12);
    let mut cold = Controller::default();
    cold.observe(
        k,
        1,
        Measurement {
            seconds: 1.,
            tokens: 4,
            preparation_s: 0.9,
            warm_ready: true,
        },
    );
    assert_eq!(cold.budget.startup_spent, 0.);
    assert_eq!(cold.budget.balance, 0.02);
}

#[test]
fn ordinary_latency_allowance_cannot_make_a_lossy_shape_look_profitable() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    hot(&mut c, k, 1, 0.05, 4);
    for _ in 0..3 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.05, 6);
    }
    let base = c.costs.normal(k).unwrap();
    assert!(base.upper() > 0.05);
    assert!(base.mean() * 6. / 4. < 0.05);
    // The old decision used the larger deadline allowance as its savings
    // baseline, even though actual settlement charged a loss against the mean.
    assert_eq!(c.choose(k, &[candidate(2, 6., 0.05)], 0.4), None);
}

#[test]
fn ordinary_recovery_cannot_inflate_the_speculation_savings_baseline() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.045, 6);
    }
    c.observe(
        k,
        1,
        Measurement {
            seconds: 0.13,
            tokens: 4,
            preparation_s: 0.1,
            warm_ready: true,
        },
    );
    assert!(c.costs.normal(k).unwrap().mean() > 0.04);
    assert_eq!(c.choose(k, &[candidate(2, 6., 0.045)], 0.4), None);
}

#[test]
fn stale_acceptance_gets_a_funded_shallow_review_without_invented_profit() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 4);
        hot(&mut c, k, 3, 0.06, 4);
    }
    let mut shallow = candidate(2, 4.4, 0.04);
    shallow.warm_ready = false;
    let mut deep = candidate(3, 5., 0.06);
    deep.warm_ready = false;
    assert_eq!(c.choose(k, &[shallow, deep], 0.4), None);
    shallow.review_acceptance = true;
    deep.review_acceptance = true;
    // Unused startup subsidy is reserved for cost calibration, not reviews.
    assert_eq!(c.choose(k, &[deep, shallow], 0.4), None);
    for _ in 0..32 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    assert_eq!(c.choose(k, &[deep, shallow], 0.4), Some((2, true)));
    hot(&mut c, k, 4, 0.1, 0);
    c.budget.startup_spent = c.calibration_limit;
    // Even an optimistic yield cannot fund a review without enough cash for
    // the excess cost of a single guaranteed ordinary step.
    shallow.expected_tokens = 8.;
    assert_eq!(c.budget.balance, 0.);
    assert_eq!(c.choose(k, &[shallow], 0.4), None);
}

#[test]
fn acceptance_review_skips_shapes_that_cannot_win_at_perfect_acceptance() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.08, 4);
    }
    for _ in 0..100 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    let mut plan = candidate(2, 4.4, 0.08);
    plan.warm_ready = false;
    plan.review_acceptance = true;
    assert!(c.budget.balance > 0.05);
    assert_eq!(c.choose(k, &[plan], 0.4), None);
}

#[test]
fn calibrated_losses_preserve_debt_without_spending_unused_startup_subsidy() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 4);
    }
    let spent = c.budget.startup_spent;
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.1,
            tokens: 4,
            preparation_s: 0.06,
            warm_ready: false,
        },
    );
    assert!(c.budget.balance < -0.06);
    assert_eq!(c.budget.startup_spent, spent);
    assert!(c.budget.available(c.calibration_limit) > 0.);
    let cold = Candidate {
        warm_ready: false,
        ..candidate(2, 8., 0.04)
    };
    assert_eq!(c.choose(k, &[cold], 0.4), None);
    // Unknown arithmetic may still use the bounded calibration pool.
    assert_eq!(
        c.choose(k, &[candidate(3, 10., 0.05)], 0.4),
        Some((3, true))
    );
    assert_eq!(c.budget.startup_spent, spent);
}
#[test]
fn preparation_is_charged_once_and_never_counted_as_hot_arithmetic() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let plans = [candidate(2, 1.8, 0.04)];
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
    let before = c.budget.balance - c.budget.startup_spent;
    assert_eq!(
        c.observe(
            k,
            2,
            Measurement {
                seconds: 0.08,
                preparation_s: 0.04,
                tokens: 2,
                warm_ready: false
            }
        ),
        None
    );
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 0);
    let after = c.budget.balance - c.budget.startup_spent;
    assert!((after - before - 0.06 + 0.08).abs() < 1e-12);
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 2);
    }
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, false)));
}

#[test]
fn zero_yield_is_real_loss_without_a_timing_or_acceptance_sample() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let before = c.budget.balance - c.budget.startup_spent;
    assert_eq!(
        c.observe(
            k,
            2,
            Measurement {
                seconds: 0.05,
                tokens: 0,
                preparation_s: 0.,
                warm_ready: true
            }
        ),
        None
    );
    let after = c.budget.balance - c.budget.startup_spent;
    assert!((after - before + 0.05).abs() < 1e-12);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 0);
}

#[test]
fn context_and_membership_changes_cannot_reset_budget_or_debt() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let plans = [candidate(2, 7., 0.04)];
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.6,
            preparation_s: 0.2,
            tokens: 0,
            warm_ready: false,
        },
    );
    let spent = c.budget.startup_spent;
    assert_eq!(spent, 0.4);
    for context in 9..15 {
        let other = Key { context, ..k };
        normal(&mut c, other);
        assert_eq!(c.choose(other, &plans, 0.4), None);
        assert_eq!(c.budget.startup_spent, spent);
    }
}

#[test]
fn earned_model_credit_funds_new_shapes_without_copying_timings() {
    let mut c = Controller::default();
    let one = key(1);
    normal(&mut c, one);
    for _ in 0..100 {
        hot(&mut c, one, 2, 0.01, 2);
    }
    c.budget.startup_spent = c.calibration_limit;
    let many = key(4);
    normal(&mut c, many);
    assert!(!c.costs.profiles[&many].arms.contains_key(&2));
    assert_eq!(
        c.choose(many, &[candidate(2, 7., 0.09)], 0.4),
        Some((2, true))
    );
    hot(&mut c, many, 2, 0.6, 0);
    assert!(c.budget.balance < 0.);
    let shared = Key {
        shared_work: true,
        ..one
    };
    normal(&mut c, shared);
    assert!(!c.costs.profiles[&shared].arms.contains_key(&2));
    assert_eq!(c.choose(shared, &[candidate(2, 1.8, 0.04)], 0.4), None);
}

#[test]
fn startup_is_one_bounded_model_lifetime_pool() {
    let mut c = Controller::default();
    for batch in 1..20 {
        let k = key(batch);
        normal(&mut c, k);
        // Selection itself does not allocate or consume money.
        let before = c.budget.startup_spent;
        let _ = c.choose(k, &[candidate(2, batch as f64 * 1.8, 0.04)], 0.4);
        assert_eq!(c.budget.startup_spent, before);
        hot(&mut c, k, 2, 0.05, 0);
        assert!(c.budget.startup_spent <= c.calibration_limit);
    }
    assert_eq!(c.budget.startup_spent, 0.4);
}

#[test]
fn profile_eviction_preserves_workload_debt_and_startup_history() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    hot(&mut c, k, 2, 0.8, 0);
    let balance = c.budget.balance;
    for context in 10..270 {
        hot(&mut c, Key { context, ..k }, 1, 0.03, 4);
    }
    assert_eq!(c.costs.profiles.len(), 256);
    assert!(!c.costs.profiles.contains_key(&k));
    assert!(c.budget.balance < 0.);
    assert!((c.budget.balance - balance - 260. * 0.03 * 0.02).abs() < 1e-12);
    normal(&mut c, k);
    assert_eq!(c.choose(k, &[candidate(2, 7., 0.04)], 0.4), None);
    assert_eq!(c.budget.startup_spent, 0.4);
}

#[test]
fn timing_samples_survive_transient_execution_cost_changes() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 8);
    }
    let before = c.costs.profiles[&k].arms[&2].time.samples();
    hot(&mut c, k, 2, 0.045, 7);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), before + 1);
    assert_eq!(
        c.choose(k, &[candidate(2, 7., 0.04)], 0.4),
        Some((2, false))
    );
}
#[test]
fn measured_goodput_uses_current_request_yield_instead_of_training_yield() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 4);
        hot(&mut c, k, 3, 0.06, 4);
    }
    assert_eq!(
        c.choose(k, &[candidate(2, 7., 0.04), candidate(3, 8., 0.06)], 0.4),
        Some((2, false))
    );
    assert_eq!(
        c.choose(k, &[candidate(2, 4.1, 0.04), candidate(3, 11., 0.06)], 0.4),
        Some((3, false))
    );
    assert_eq!(
        c.choose(k, &[candidate(2, 4.1, 0.04), candidate(3, 4.2, 0.06)], 0.4),
        None
    );
}
#[test]
fn stale_probe_refreshes_once_without_resetting_samples() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 8);
    }
    for _ in 0..REFRESH_AFTER {
        hot(&mut c, k, 1, 0.03, 4);
    }
    let plans = [candidate(2, 7., 0.04)];
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
    hot(&mut c, k, 2, 0.04, 8);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 4);
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, false)));
}
#[test]
fn latency_and_short_tails_gate_probes_and_measured_choices() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let mut plan = candidate(2, 1.8, 0.04);
    assert_eq!(c.choose(k, &[plan], 0.01), None);
    assert_eq!(c.choose(k, &[plan], f64::NAN), None);
    plan.remaining_tokens = 2;
    assert_eq!(c.choose(k, &[plan], 0.4), None);
    plan.remaining_tokens = 100;
    assert_eq!(c.choose(k, &[plan], 0.4), Some((2, true)));
}
#[test]
fn invalid_feedback_cannot_poison_costs_or_budget() {
    let mut c = Controller::default();
    let k = key(1);
    for seconds in [0., -0.1, f64::NAN, f64::INFINITY] {
        c.observe(
            k,
            2,
            Measurement {
                seconds,
                preparation_s: 0.,
                tokens: 2,
                warm_ready: true,
            },
        );
    }
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.03,
            preparation_s: f64::NAN,
            tokens: 2,
            warm_ready: false,
        },
    );
    assert!(c.costs.profiles.is_empty());
    assert_eq!(c.budget.balance, 0.);
}
#[test]
fn graph_metadata_is_not_a_timing_reset_or_a_preparation_charge() {
    use crate::scheduler::BatchExecutionStatistics;
    let before = BatchExecutionStatistics::default();
    let mut after = BatchExecutionStatistics {
        graph_capture_deferrals: 1,
        graph_evictions: 1,
        graph_invalidations: 2,
        direct_iterations: 1,
        graph_hits: 1,
        ..Default::default()
    };
    assert_eq!(after.preparation_s_since(before), 0.);
    after.capture_s = 0.01;
    after.kernel_load_s = 0.02;
    after.workspace_mapping_s = 0.03;
    assert!((after.preparation_s_since(before) - 0.06).abs() < 1e-12);
    assert_eq!(after.preparation_s_since(after), 0.);
}

#[test]
fn default_runtime_prior_can_bootstrap_without_cold_debt_state() {
    let mut c = Controller::default();
    let k = key(1);
    for _ in 0..4 {
        hot(&mut c, k, 1, 0.012, 1);
    }
    let plans = [
        candidate(2, 1.7, 0.012 + 0.055),
        candidate(3, 2.19, 0.012 + 2. * 0.055),
    ];
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
    assert_eq!(c.budget.startup_spent, 0.);
    // Unexpected materialization cost is paid from the common pool as it
    // occurs; unused startup money remains available to other workloads.
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.22,
            preparation_s: 0.20,
            tokens: 2,
            warm_ready: false,
        },
    );
    let spent = c.budget.startup_spent;
    assert!(spent > 0.19 && spent < 0.2);
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.015, 2);
        assert_eq!(c.budget.startup_spent, spent);
    }
}

#[test]
fn many_new_shapes_cannot_reset_global_credit_or_debt() {
    let mut c = Controller::default();
    let one = key(1);
    normal(&mut c, one);
    hot(&mut c, one, 2, 2., 0);
    assert_eq!(c.budget.startup_spent, c.calibration_limit);
    for batch in 1..=300 {
        let k = key(batch);
        normal(&mut c, k);
        assert_eq!(
            c.choose(k, &[candidate(2, batch as f64 * 1.8, 0.04)], 0.4),
            None
        );
    }
    assert!(c.budget.balance < 0.);
    assert_eq!(c.budget.startup_spent, c.calibration_limit);
    assert_eq!(c.costs.profiles.len(), 256);
}
#[test]
fn fixed_diagnostic_policy_obeys_legal_shapes_and_slack() {
    let mut c = Controller {
        single_policy: SinglePolicy::Fixed(4),
        ..Default::default()
    };
    let plans = [candidate(2, 1.5, 0.03), candidate(3, 2., 0.04)];
    assert_eq!(c.choose(key(1), &plans, 0.4), Some((3, false)));
    assert_eq!(c.choose(key(1), &plans, 0.01), None);
    assert_eq!(c.choose(key(2), &plans, 0.4), None);
}

#[test]
fn unprepared_execution_is_billed_without_poisoning_steady_costs() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..4 {
        c.observe(
            k,
            2,
            Measurement {
                seconds: 0.08,
                tokens: 7,
                preparation_s: 0.,
                warm_ready: false,
            },
        );
    }
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 0);
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.04, 8);
    }
    let before = c.costs.profiles[&k].arms[&2].time.mean();
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.09,
            tokens: 6,
            preparation_s: 0.,
            warm_ready: false,
        },
    );
    let cost = &c.costs.profiles[&k].arms[&2];
    assert_eq!(cost.time.samples(), 3);
    assert_eq!(cost.time.mean(), before);
    assert_eq!(cost.mean(), cost.time.mean());
    assert!(cost.transition_s > 0.);
    assert_eq!(
        c.choose(k, &[candidate(2, 7., 0.04)], 0.4),
        Some((2, false))
    );
}

#[test]
fn catchup_at_a_context_boundary_does_not_poison_the_cheaper_width() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    c.observe(
        k,
        2,
        Measurement {
            seconds: 0.062,
            tokens: 8,
            preparation_s: 0.,
            warm_ready: false,
        },
    );
    for _ in 0..4 {
        hot(&mut c, k, 2, 0.039, 8);
        hot(&mut c, k, 3, 0.057, 12);
    }
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 3);
    assert!((c.costs.profiles[&k].arms[&2].time.mean() - 0.039).abs() < 1e-12);
    assert_eq!(
        c.choose(k, &[candidate(2, 7.6, 0.05), candidate(3, 9.6, 0.06)], 0.4),
        Some((2, false))
    );
}

#[test]
fn one_cold_execution_cannot_inflate_the_ordinary_baseline() {
    let mut c = Controller::default();
    let k = key(4);
    hot(&mut c, k, 1, 0.105, 4);
    assert_eq!(c.costs.normal(k).unwrap().time.samples(), 0);
    for _ in 0..3 {
        hot(&mut c, k, 1, 0.042, 4);
    }
    assert_eq!(c.costs.normal(k).unwrap().time.samples(), 3);
    assert!((c.costs.normal(k).unwrap().mean() - 0.042).abs() < 1e-12);
}
#[test]
fn ordinary_baseline_is_refreshed_without_discarding_speculative_samples() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..REFRESH_AFTER {
        hot(&mut c, k, 2, 0.04, 8);
    }
    let plans = [candidate(2, 7., 0.04)];
    assert_eq!(c.choose(k, &plans, 0.4), None);
    hot(&mut c, k, 1, 0.03, 4);
    assert_eq!(c.choose(k, &plans, 0.4), Some((2, false)));
    assert_eq!(
        c.costs.profiles[&k].arms[&2].time.samples(),
        REFRESH_AFTER - 1
    );
}

#[test]
fn aggregate_progress_is_optimized_within_the_decode_deadline() {
    let mut c = Controller::default();
    let k = key(4);
    for _ in 0..4 {
        hot(&mut c, k, 1, 0.04, 4);
        hot(&mut c, k, 2, 0.05, 8);
    }
    assert_eq!(
        // One owner's 1.1-token expectation does not replace the whole
        // cohort's total 7.1-token progress. Its actual deadline still binds.
        c.choose(k, &[candidate(2, 7.1, 0.05)], 0.4),
        Some((2, false))
    );
    assert_eq!(c.choose(k, &[candidate(2, 7.1, 0.05)], 0.04), None);
}

#[test]
fn latency_utility_averages_request_tpot_without_a_worst_owner_veto() {
    let near_uniform = [1.546, 1.969, 1.943, 1.969];
    assert!(progress(near_uniform.into_iter(), 4, true) > 7.3);
    assert_eq!(progress([2.; 4].into_iter(), 4, true), 8.);
    assert_eq!(progress([2.].into_iter(), 1, true), 2.);
    let skewed = [1.1, 3., 3., 3.];
    let latency = progress(skewed.into_iter(), 4, true);
    let throughput = progress(skewed.into_iter(), 4, false);
    let mut c = Controller::default();
    let k = key(4);
    for _ in 0..4 {
        hot(&mut c, k, 1, 0.04, 4);
        hot(&mut c, k, 4, 0.09, 10);
    }
    assert_eq!(
        c.choose(k, &[candidate(4, throughput, 0.09)], 0.4),
        Some((4, false))
    );
    assert_eq!(c.choose(k, &[candidate(4, latency, 0.09)], 0.4), None);
}
