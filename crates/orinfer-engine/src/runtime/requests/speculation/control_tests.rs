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
    assert_eq!(&a.sample_counts()[..7], &[1, 1, 0, 0, 0, 0, 0]);
    assert!(a.expected(2) > 1. && a.expected(2) < 3.);
    a.observe(1, 1);
    assert_eq!(a.sample_counts()[1], 1);
    assert_eq!(Acceptance::default().sample_counts(), &[0; 31]);
}

#[test]
fn cold_shapes_finish_in_width_order_without_mode_readiness_states() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let plans = [candidate(2, 7., 0.04), candidate(3, 9., 0.05)];
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 8);
    }
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((3, true)));
        hot(&mut c, k, 3, 0.05, 12);
    }
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((3, false)));
}
#[test]
fn preparation_is_charged_once_and_never_counted_as_hot_arithmetic() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let plans = [candidate(2, 1.8, 0.04)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    let before = c.budget.balance(k.family()) - c.budget.startup_spent;
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
    let after = c.budget.balance(k.family()) - c.budget.startup_spent;
    assert!((after - before - 0.06 + 0.08).abs() < 1e-12);
    for _ in 0..4 {
        assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 2);
    }
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, false)));
}

#[test]
fn zero_yield_is_real_loss_without_a_timing_or_acceptance_sample() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let before = c.budget.balance(k.family()) - c.budget.startup_spent;
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
    let after = c.budget.balance(k.family()) - c.budget.startup_spent;
    assert!((after - before + 0.05).abs() < 1e-12);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 0);
}

#[test]
fn context_and_membership_changes_cannot_reset_budget_or_debt() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let plans = [candidate(2, 7., 0.04)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
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
        assert_eq!(c.choose(other, &plans, 0.03, 0.4), None);
        assert_eq!(c.budget.startup_spent, spent);
    }
}

#[test]
fn independent_workloads_cannot_spend_each_others_profits() {
    let mut c = Controller::default();
    let one = key(1);
    normal(&mut c, one);
    for _ in 0..100 {
        hot(&mut c, one, 2, 0.01, 2);
    }
    let many = key(4);
    normal(&mut c, many);
    hot(&mut c, many, 2, 0.6, 0);
    assert!(c.budget.balance(one.family()) > 0.3);
    assert!(c.budget.balance(many.family()) < 0.);
    assert_eq!(c.choose(many, &[candidate(2, 7., 0.04)], 0.03, 0.4), None);
    let shared = Key {
        shared_work: true,
        ..one
    };
    normal(&mut c, shared);
    assert!(!c.costs.profiles[&shared].arms.contains_key(&2));
}

#[test]
fn startup_is_one_bounded_model_lifetime_pool() {
    let mut c = Controller::default();
    for batch in 1..20 {
        let k = key(batch);
        normal(&mut c, k);
        // Selection itself does not allocate or consume money.
        let before = c.budget.startup_spent;
        let _ = c.choose(k, &[candidate(2, batch as f64 * 1.8, 0.04)], 0.03, 0.4);
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
    let balance = c.budget.balance(k.family());
    for context in 10..270 {
        hot(&mut c, Key { context, ..k }, 1, 0.03, 4);
    }
    assert_eq!(c.costs.profiles.len(), 256);
    assert!(!c.costs.profiles.contains_key(&k));
    assert!(c.budget.balance(k.family()) < 0.);
    assert!((c.budget.balance(k.family()) - balance - 260. * 0.03 * 0.02).abs() < 1e-12);
    normal(&mut c, k);
    assert_eq!(c.choose(k, &[candidate(2, 7., 0.04)], 0.03, 0.4), None);
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
        c.choose(k, &[candidate(2, 7., 0.04)], 0.03, 0.4),
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
        c.choose(
            k,
            &[candidate(2, 7., 0.04), candidate(3, 8., 0.06)],
            0.03,
            0.4
        ),
        Some((2, false))
    );
    assert_eq!(
        c.choose(
            k,
            &[candidate(2, 4.1, 0.04), candidate(3, 11., 0.06)],
            0.03,
            0.4
        ),
        Some((3, false))
    );
    assert_eq!(
        c.choose(
            k,
            &[candidate(2, 4.1, 0.04), candidate(3, 4.2, 0.06)],
            0.03,
            0.4
        ),
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
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    hot(&mut c, k, 2, 0.04, 8);
    assert_eq!(c.costs.profiles[&k].arms[&2].time.samples(), 4);
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, false)));
}
#[test]
fn latency_and_short_tails_gate_probes_and_measured_choices() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let mut plan = candidate(2, 1.8, 0.04);
    assert_eq!(c.choose(k, &[plan], 0.03, 0.01), None);
    assert_eq!(c.choose(k, &[plan], 0.03, f64::NAN), None);
    plan.remaining_tokens = 2;
    assert_eq!(c.choose(k, &[plan], 0.03, 0.4), None);
    plan.remaining_tokens = 100;
    assert_eq!(c.choose(k, &[plan], 0.03, 0.4), Some((2, true)));
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
    assert_eq!(c.budget.balance(k.family()), 0.);
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
    assert_eq!(c.choose(k, &plans, 0.012, 0.4), Some((2, true)));
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
        assert_eq!(c.choose(k, &plans, 0.012, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.015, 2);
        assert_eq!(c.budget.startup_spent, spent);
    }
}

#[test]
fn full_account_table_keeps_existing_debt_and_rejects_new_exploration() {
    let mut c = Controller::default();
    for batch in 1..=256 {
        normal(&mut c, key(batch));
    }
    let before = c.budget.startup_spent;
    let k = key(257);
    normal(&mut c, k);
    assert_eq!(c.choose(k, &[candidate(2, 514., 0.04)], 0.03, 0.4), None);
    assert_eq!(c.budget.startup_spent, before);
}
#[test]
fn fixed_diagnostic_policy_obeys_legal_shapes_and_slack() {
    let mut c = Controller {
        single_policy: SinglePolicy::Fixed(4),
        ..Default::default()
    };
    let plans = [candidate(2, 1.5, 0.03), candidate(3, 2., 0.04)];
    assert_eq!(c.choose(key(1), &plans, 0.02, 0.4), Some((3, false)));
    assert_eq!(c.choose(key(1), &plans, 0.02, 0.01), None);
    assert_eq!(c.choose(key(2), &plans, 0.02, 0.4), None);
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
    assert!(cost.mean() > cost.time.mean());
    assert_eq!(
        c.choose(k, &[candidate(2, 7., 0.04)], 0.03, 0.4),
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
        c.choose(
            k,
            &[candidate(2, 7.6, 0.05), candidate(3, 9.6, 0.06)],
            0.03,
            0.4
        ),
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
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
    hot(&mut c, k, 1, 0.03, 4);
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, false)));
    assert_eq!(
        c.costs.profiles[&k].arms[&2].time.samples(),
        REFRESH_AFTER - 1
    );
}

#[test]
fn latency_and_throughput_use_distinct_progress_objectives() {
    let yields = [1.1, 2., 2., 2.];
    let latency = progress(yields.into_iter(), 4, true);
    let throughput = progress(yields.into_iter(), 4, false);
    assert!((latency - 4.4).abs() < 1e-12);
    assert!((throughput - 7.1).abs() < 1e-12);
    assert_eq!(
        progress([2.; 4].into_iter(), 4, true),
        progress([2.; 4].into_iter(), 4, false)
    );
    let mut c = Controller::default();
    let k = key(4);
    for _ in 0..4 {
        hot(&mut c, k, 1, 0.04, 4);
        hot(&mut c, k, 2, 0.05, 8);
    }
    assert_eq!(
        c.choose(k, &[candidate(2, throughput, 0.05)], 0.04, 0.4),
        Some((2, false))
    );
    assert_eq!(c.choose(k, &[candidate(2, latency, 0.05)], 0.04, 0.4), None);
}
