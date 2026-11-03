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

fn candidate(width: usize, tokens: f64, seconds: f64) -> Candidate {
    Candidate {
        width,
        expected_tokens: tokens,
        prior_seconds: seconds,
        remaining_tokens: 4096,
    }
}

fn hot(c: &mut Controller, k: Key, width: usize, seconds: f64, tokens: usize) {
    c.observe(k, width, seconds, tokens, ExecutionObservation::default());
}

fn normal(c: &mut Controller, k: Key) {
    for _ in 0..3 {
        hot(c, k, 1, 0.03, k.batch);
    }
}

#[test]
fn row_budget_favors_accepting_prefixes_and_respects_caps_and_holes() {
    let choices = vec![
        vec![(2, 1.9), (3, 2.7), (4, 3.4)],
        vec![(2, 1.2), (3, 1.22), (4, 1.221)],
    ];
    assert_eq!(allocate(&choices, 6), Some(vec![4, 2]));
    assert_eq!(allocate(&choices, 8), Some(vec![4, 4]));
    assert_eq!(allocate(&choices, 9), None);
    assert_eq!(allocate(&choices, 3), None);
    assert_eq!(allocate(&[vec![(2, 1.8), (4, 2.9)]], 3), None);
}

#[test]
fn acceptance_tracks_tested_prefix_and_never_invents_suffix_samples() {
    let mut a = Acceptance::default();
    a.observe(3, 1);
    assert_eq!(a.samples, [1, 1, 0, 0, 0, 0, 0]);
    assert!(a.expected(2) > 1. && a.expected(2) < 3.);
    a.observe(1, 1);
    assert_eq!(a.samples[1], 1);
}

#[test]
fn cold_loss_does_not_starve_funded_single_calibration() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let plans = [candidate(2, 1.8, 0.06)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    c.observe(
        k,
        2,
        0.2,
        1,
        ExecutionObservation {
            preparing: true,
            ..Default::default()
        },
    );
    assert!(c.credit < 0.);
    assert_eq!(c.cohorts[&k].arms[&2].hot(0).time.samples, 0);
    for _ in 0..3 {
        assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
        hot(&mut c, k, 2, 0.04, 2);
    }
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, false)));
    assert!(c.loan.is_none());
    assert!(c.bootstrap_used);
}

#[test]
fn batch_probes_need_funded_calibration_and_no_context_reset() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    let plans = [candidate(2, 6., 0.06)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
    for _ in 0..200 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    c.observe(
        k,
        2,
        0.4,
        4,
        ExecutionObservation {
            preparing: true,
            ..Default::default()
        },
    );
    assert!(c.balances[&k.family()] < 0.);
    let other_context = Key { context: 9, ..k };
    normal(&mut c, other_context);
    assert_eq!(c.choose(other_context, &plans, 0.03, 0.4), None);
    let other_batch = key(8);
    normal(&mut c, other_batch);
    assert_eq!(
        c.choose(other_batch, &[candidate(2, 12., 0.08)], 0.03, 0.4),
        None
    );
}

#[test]
fn profitable_single_cannot_fund_a_cold_batch_by_itself() {
    let mut c = Controller::default();
    normal(&mut c, key(1));
    for _ in 0..30 {
        hot(&mut c, key(1), 2, 0.03, 2);
    }
    assert!(c.credit > 0.3);
    normal(&mut c, key(4));
    assert_eq!(c.choose(key(4), &[candidate(2, 6., 0.06)], 0.03, 0.4), None);
}

#[test]
fn shared_cohort_cannot_reuse_exclusive_readiness_or_bootstrap() {
    let mut c = Controller::default();
    let exclusive = key(1);
    normal(&mut c, exclusive);
    for _ in 0..30 {
        hot(&mut c, exclusive, 2, 0.03, 2);
    }
    let partial = Key {
        shared_work: true,
        ..exclusive
    };
    let plans = [candidate(2, 1.8, 0.06)];
    assert_eq!(c.choose(partial, &plans, 0.03, 0.4), None);
    normal(&mut c, partial);
    assert_eq!(c.choose(partial, &plans, 0.03, 0.4), None);
    assert!(!c.bootstrap_used);
    assert!(!c.cohorts[&partial].arms.contains_key(&2));
    for _ in 0..200 {
        hot(&mut c, partial, 1, 0.03, 1);
    }
    assert_eq!(c.choose(partial, &plans, 0.03, 0.4), Some((2, true)));
}

#[test]
fn zero_yield_attempt_is_charged_without_training_cost_or_acceptance() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    let before = c.credit;
    hot(&mut c, k, 2, 0.05, 0);
    assert!((c.credit - before + 0.05).abs() < 1e-9);
    assert_eq!(c.cohorts[&k].arms[&2].hot(0).time.samples, 0);
    let before = c.credit;
    hot(&mut c, k, 1, 0.05, 0);
    assert!((c.credit - before + 0.05).abs() < 1e-9);
}

#[test]
fn exhausted_calibration_pauses_and_does_not_mint_another_loan() {
    let mut c = Controller::default();
    let k = key(1);
    c.set_calibration_limit(0.05);
    normal(&mut c, k);
    let plans = [candidate(2, 1.8, 0.04)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    c.observe(
        k,
        2,
        0.2,
        1,
        ExecutionObservation {
            preparing: true,
            ..Default::default()
        },
    );
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
    let neighbor = Key { context: 9, ..k };
    normal(&mut c, neighbor);
    assert_eq!(c.choose(neighbor, &plans, 0.03, 0.4), None);
    assert!(c.bootstrap_used);
    assert!(c.cohorts[&k].calibrating == Some(2));
}

#[test]
fn deferred_capture_needs_funding_even_after_a_probe_was_admitted() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..200 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    let plans = [candidate(2, 6., 0.06)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, true)));
    c.observe(
        k,
        2,
        0.2,
        8,
        ExecutionObservation {
            preparing: true,
            ..Default::default()
        },
    );
    assert!(c.cohorts[&k].calibrating == Some(2));
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
}

#[test]
fn calibration_requires_remaining_work_and_possible_batch_payback() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..600 {
        hot(&mut c, k, 1, 0.03, 4);
    }
    let mut plan = candidate(2, 4.01, 0.06);
    assert_eq!(c.choose(k, &[plan], 0.03, 0.4), None);
    plan.remaining_tokens = 32768;
    assert_eq!(c.choose(k, &[plan], 0.03, 0.4), Some((2, true)));
    let mut c = Controller::default();
    normal(&mut c, key(1));
    let mut tail = candidate(2, 1.8, 0.06);
    tail.remaining_tokens = 3;
    assert_eq!(c.choose(key(1), &[tail], 0.03, 0.4), None);
    tail.remaining_tokens = 32;
    assert_eq!(c.choose(key(1), &[tail], 0.03, 0.4), Some((2, true)));
}

#[test]
fn recapture_requires_fresh_hot_samples_before_reusing_a_winner() {
    let mut c = Controller::default();
    let k = key(1);
    normal(&mut c, k);
    for _ in 0..3 {
        hot(&mut c, k, 2, 0.04, 2);
    }
    let plans = [candidate(2, 2., 0.06)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), Some((2, false)));
    c.bootstrap_used = true;
    c.observe(
        k,
        2,
        0.4,
        2,
        ExecutionObservation {
            mode: ExecutionMode::Mixed,
            preparing: true,
            ..Default::default()
        },
    );
    assert_eq!(c.cohorts[&k].arms[&2].hot(0).time.samples, 0);
    assert_eq!(
        c.cohorts[&k].arms[&2].modes[ExecutionMode::Direct as usize]
            .time
            .samples,
        0
    );
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
}

#[test]
fn mixed_roles_earn_budget_without_treating_priors_as_samples() {
    let mut c = Controller::default();
    let k = Key {
        prefill: 32,
        ..key(4)
    };
    normal(&mut c, k);
    let other_shape = Key { prefill: 64, ..k };
    for _ in 0..200 {
        hot(&mut c, other_shape, 1, 0.03, 4);
    }
    assert_eq!(k.family(), other_shape.family());
    assert_eq!(
        c.choose(k, &[candidate(2, 6., 0.06)], 0.03, 0.4),
        Some((2, true))
    );
    assert_eq!(c.cohorts[&k].arms[&2].hot(0).time.samples, 0);
}

#[test]
fn direct_and_replay_costs_and_generations_remain_independent() {
    let mut c = Controller::default();
    let k = key(1);
    for _ in 0..3 {
        hot(&mut c, k, 2, 0.08, 2);
    }
    c.observe(
        k,
        2,
        0.03,
        2,
        ExecutionObservation {
            mode: ExecutionMode::Replay,
            ..Default::default()
        },
    );
    let arm = &c.cohorts[&k].arms[&2];
    assert_eq!(arm.hot(0).time.samples, 1);
    assert_eq!(arm.modes[ExecutionMode::Direct as usize].time.samples, 3);
    c.observe(
        k,
        2,
        0.12,
        1,
        ExecutionObservation {
            mode: ExecutionMode::Replay,
            preparing: true,
            generation: 1,
            ..Default::default()
        },
    );
    assert_eq!(c.cohorts[&k].arms[&2].hot(1).time.samples, 0);
    assert!(c.cohorts[&k].arms[&2].prior > 0.);
}

#[test]
fn neighbor_prior_does_not_fabricate_hot_samples() {
    let mut c = Controller::default();
    let k = key(1);
    for _ in 0..3 {
        hot(&mut c, k, 3, 0.05, 3);
    }
    let neighbor = Key { context: 9, ..k };
    assert!((c.prior(neighbor, 3, 0.3) - 0.05).abs() < 1e-9);
    assert!(!c.cohorts.contains_key(&neighbor));
    assert_eq!(c.prior(Key { context: 15, ..k }, 3, 0.3), 0.3);
}

#[test]
fn steady_low_acceptance_selects_ordinary_and_respects_deadline() {
    let mut c = Controller::default();
    let k = key(4);
    normal(&mut c, k);
    for _ in 0..3 {
        hot(&mut c, k, 2, 0.05, 4);
    }
    let plans = [candidate(2, 4.1, 0.05)];
    assert_eq!(c.choose(k, &plans, 0.03, 0.4), None);
    assert!(c.cohorts[&k].arms[&2].state == State::Cooldown);
    assert_eq!(c.choose(k, &plans, 0.03, 0.01), None);
}

#[test]
fn single_fixed_respects_legal_widths_and_leaves_batch_adaptive() {
    let mut c = Controller {
        single_policy: SinglePolicy::Fixed(4),
        ..Default::default()
    };
    let plans = [candidate(2, 1.5, 0.03), candidate(3, 2., 0.04)];
    assert_eq!(c.choose(key(1), &plans, 0.02, 0.4), Some((3, false)));
    assert_eq!(c.choose(key(2), &plans, 0.02, 0.4), None);
    assert_eq!(c.choose(key(1), &plans, 0.02, 0.01), None);
    c.single_policy = SinglePolicy::Fixed(1);
    assert_eq!(c.choose(key(1), &plans, 0.02, 0.4), None);
}

#[test]
fn local_acceptance_does_not_inherit_another_prompts_rejection_rate() {
    let mut c = Controller {
        single_policy: SinglePolicy::ExploreLocal,
        ..Default::default()
    };
    for _ in 0..4 {
        hot(&mut c, key(1), 1, 0.03, 1);
        hot(&mut c, key(1), 3, 0.05, 3);
        hot(&mut c, key(1), 4, 0.055, 2);
    }
    let plans = [candidate(3, 2.8, 0.05), candidate(4, 3.8, 0.055)];
    assert_eq!(c.choose(key(1), &plans, 0.03, 0.4), Some((4, false)));
    c.single_policy = SinglePolicy::Explore;
    assert_eq!(c.choose(key(1), &plans, 0.03, 0.4), Some((3, false)));
}

#[test]
fn execution_feedback_identifies_preparation_and_partial_replay() {
    use crate::scheduler::BatchExecutionStatistics;
    let before = BatchExecutionStatistics::default();
    let mut after = before;
    after.graph_capture_deferrals = 1;
    after.direct_iterations = 1;
    after.graph_hits = 1;
    let observed = after.observation_since(before);
    assert!(observed.preparing);
    assert_eq!(observed.mode, ExecutionMode::Mixed);
    let hot = after.observation_since(after);
    assert!(!hot.preparing);
    after.graph_evictions = 1;
    assert_eq!(after.observation_since(before).generation, 1);
}
