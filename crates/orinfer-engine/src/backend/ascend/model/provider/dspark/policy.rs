//! Shared batch cost and request-local, causal confidence budgets.
use crate::artifact::Result;
use std::collections::BTreeMap;

#[derive(Default)]
pub(in crate::backend::ascend::model::provider) struct Feedback {
    blocks: BTreeMap<usize, History>,
}
#[derive(Default)]
struct History {
    previous: Vec<f64>,
    older: Vec<f64>,
    acceptance: Vec<f64>,
    attempts: Vec<usize>,
}
impl Feedback {
    pub fn observe(&mut self, confidence: &[f64], proposed: usize, accepted: usize, block: usize) {
        self.blocks
            .entry(block)
            .or_default()
            .observe(confidence, proposed, accepted);
    }
    fn rates(&self, maximum: usize, older: bool, prior: &[f64], block: usize) -> Vec<f64> {
        self.blocks
            .get(&block)
            .unwrap_or(&History::default())
            .rates(maximum, older, prior)
    }
}
impl History {
    fn observe(&mut self, confidence: &[f64], proposed: usize, accepted: usize) {
        self.older = std::mem::replace(&mut self.previous, confidence.to_vec());
        if self.acceptance.len() < confidence.len() {
            self.acceptance.resize(confidence.len(), 0.7);
            self.attempts.resize(confidence.len(), 0);
        }
        for i in 0..proposed.min(confidence.len()) {
            if accepted < i {
                break;
            }
            let value = if accepted > i { 1. } else { 0. };
            update(&mut self.acceptance[i], value, self.attempts[i] > 0);
            self.attempts[i] += 1;
        }
    }
    fn rates(&self, maximum: usize, older: bool, prior: &[f64]) -> Vec<f64> {
        let confidence = if older { &self.older } else { &self.previous };
        let mut survival = 1.;
        (0..maximum)
            .map(|i| {
                let actual = if self.attempts.get(i).copied().unwrap_or(0) > 0 {
                    self.acceptance[i]
                } else {
                    prior.get(i).copied().unwrap_or(0.7)
                };
                let predicted = confidence.get(i).copied().unwrap_or(actual);
                survival *= ((actual + predicted) * 0.5).clamp(0., 1.);
                survival
            })
            .collect()
    }
}

#[derive(Default)]
pub(in crate::backend::ascend::model::provider) struct Budget {
    pub rows: usize,
    pub remaining: usize,
    pub uniform: bool,
}

#[derive(Default)]
pub(in crate::backend::ascend::model::provider) struct Policy {
    curves: BTreeMap<(usize, bool, bool), Curve>,
}
#[derive(Default)]
struct Curve {
    baseline: f64,
    baseline_samples: usize,
    ticks: usize,
    blocks: BTreeMap<(usize, bool), BlockCurve>,
}
#[derive(Default)]
struct BlockCurve {
    samples: usize,
    draft: f64,
    extra: f64,
    restore: f64,
    setup: f64,
    seen: BTreeMap<usize, usize>,
    acceptance: Vec<f64>,
    previous: Vec<f64>,
    older: Vec<f64>,
}
fn update(old: &mut f64, sample: f64, initialized: bool) {
    *old = if initialized {
        *old * 0.8 + sample * 0.2
    } else {
        sample
    };
}
fn update_cost(old: &mut f64, sample: f64, initialized: bool) {
    if !initialized || sample < *old {
        // The first steady Graph replay replaces direct/capture startup cost.
        *old = sample;
    } else {
        update(old, sample, true);
    }
}
fn queue(
    maxima: &[usize],
    feedback: &[&Feedback],
    prior: &[f64],
    older: bool,
    block: usize,
) -> Vec<(f64, usize, usize)> {
    let mut rows = maxima
        .iter()
        .zip(feedback)
        .enumerate()
        .flat_map(|(owner, (&maximum, state))| {
            state
                .rates(maximum, older, prior, block)
                .into_iter()
                .enumerate()
                .map(move |(position, rate)| (rate, owner, position))
        })
        .collect::<Vec<_>>();
    // Cumulative rates are nonincreasing, so this also preserves each prefix.
    rows.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    rows
}
impl Policy {
    pub fn baseline(&mut self, batch: usize, full: bool, mixed: bool, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        update_cost(&mut curve.baseline, seconds, curve.baseline_samples > 0);
        curve.baseline_samples += 1;
    }
    pub fn select_requests(
        &mut self,
        full: bool,
        mixed: bool,
        budget: Budget,
        maxima: &[usize],
        feedback: &[&Feedback],
        blocks: &[usize],
    ) -> Result<Vec<usize>> {
        let batch = maxima.len();
        if batch == 0
            || feedback.len() != batch
            || maxima.iter().any(|&m| m > crate::execution::MAX_MTP_DRAFTS)
        {
            return Err("Invalid DSpark request-depth contract".into());
        }
        let mut depths = vec![0; batch];
        let available = batch + maxima.iter().sum::<usize>();
        let limit = if budget.rows == 0 {
            available
        } else {
            budget.rows.min(available)
        };
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        curve.ticks += 1;
        if limit <= batch || curve.baseline_samples < 4 {
            return Ok(depths);
        }
        if blocks.is_empty()
            || blocks.windows(2).any(|pair| pair[0] >= pair[1])
            || blocks
                .iter()
                .any(|&block| block == 0 || block > crate::execution::MAX_MTP_DRAFTS)
        {
            return Err("Invalid DSpark physical blocks".into());
        }
        let mut best = batch;
        let mut best_maxima = maxima.to_vec();
        let mut best_block = blocks[0];
        let mut throughput = batch as f64 / (curve.baseline * 0.97);
        let explore = curve.ticks.is_multiple_of(16);
        for &block in blocks {
            let width = maxima.iter().copied().max().unwrap_or(0).min(block);
            let local = maxima
                .iter()
                .map(|&m| {
                    if budget.uniform {
                        if m >= width { width } else { 0 }
                    } else {
                        m.min(block)
                    }
                })
                .collect::<Vec<_>>();
            let positive = local.iter().filter(|&&m| m > 0).count();
            let local_limit = limit.min(batch + local.iter().sum::<usize>());
            // A row budget too small to reach this block belongs to a shorter template.
            if blocks
                .iter()
                .copied()
                .filter(|&b| b < block)
                .max()
                .is_some_and(|previous| {
                    local_limit - batch <= previous
                        || local.iter().copied().max().unwrap_or(0) <= previous
                })
            {
                continue;
            }
            let cost = curve.blocks.entry((block, budget.uniform)).or_default();
            if cost.acceptance.len() < block {
                cost.acceptance.resize(block, 0.7);
            }
            let minimum_samples = if block == blocks[0] { 8 } else { 3 };
            if cost.samples < minimum_samples
                && block != blocks[0]
                && budget.remaining < (minimum_samples - cost.samples) * (block + 1)
            {
                continue;
            }
            if cost.samples < minimum_samples
                || explore
                    && !budget.uniform
                    && !curve.ticks.is_multiple_of(32)
                    && (curve.ticks / 16) % blocks.len()
                        == blocks.iter().position(|&b| b == block).unwrap()
            {
                best = local_limit;
                best_maxima = local;
                best_block = block;
                break;
            }
            if curve.ticks.is_multiple_of(32) {
                continue;
            }
            let mean = Feedback {
                blocks: [(
                    block,
                    History {
                        previous: cost.previous.clone(),
                        older: cost.older.clone(),
                        ..Default::default()
                    },
                )]
                .into(),
            };
            let shared = vec![&mean; batch];
            let scoring = if budget.uniform {
                shared.as_slice()
            } else {
                feedback
            };
            let mut expected = batch as f64;
            for (index, &(rate, _, _)) in queue(&local, scoring, &cost.acceptance, true, block)
                .iter()
                .take(local_limit - batch)
                .enumerate()
            {
                let rows = batch + index + 1;
                expected += rate;
                if (budget.uniform && (rows - batch).is_multiple_of(positive))
                    || (!budget.uniform && (rows.is_power_of_two() || rows == local_limit))
                {
                    let remaining_rounds = (budget.remaining * batch) as f64 / expected;
                    let pending_setups =
                        3usize.saturating_sub(cost.seen.get(&rows).copied().unwrap_or(0));
                    let setup = if budget.uniform {
                        0.
                    } else {
                        cost.setup * pending_setups as f64 / remaining_rounds.max(1.)
                    };
                    let time = setup
                        + cost.draft
                        + curve.baseline
                        + cost.restore
                        + cost.extra * (rows - batch) as f64 / batch as f64;
                    if expected / time > throughput {
                        throughput = expected / time;
                        best = rows;
                        best_maxima.clone_from(&local);
                        best_block = block;
                    }
                }
            }
        }
        if budget.uniform {
            if best == batch && curve.ticks.is_multiple_of(16) && !curve.ticks.is_multiple_of(32) {
                let width = maxima.iter().copied().max().unwrap_or(0);
                best_maxima = maxima
                    .iter()
                    .map(|&m| if m >= width { width } else { 0 })
                    .collect();
                best = batch + best_maxima.iter().sum::<usize>();
            }
            let positive = best_maxima.iter().filter(|&&m| m > 0).count();
            if let Some(count) = (best - batch).checked_div(positive) {
                for (depth, &maximum) in depths.iter_mut().zip(&best_maxima) {
                    if maximum > 0 {
                        *depth = count;
                    }
                }
            }
            return Ok(depths);
        }
        // N-2 confidence chooses capacity; N-1 allocates that capacity by owner.
        for &(_, owner, _) in queue(&best_maxima, feedback, &[], false, best_block)
            .iter()
            .take(best - batch)
        {
            depths[owner] += 1;
        }
        Ok(depths)
    }
    pub fn observe(
        &mut self,
        batch: usize,
        mode: (bool, bool, bool),
        timing: [f64; 4],
        counts: &[(usize, usize)],
        block: usize,
        confidence: &[f64],
    ) {
        let (full, mixed, uniform) = mode;
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        let positive = counts.iter().filter(|p| p.0 > 0).count();
        let proposed = counts.iter().map(|p| p.0).sum::<usize>() as f64 / batch as f64;
        if positive == 0 {
            return;
        }
        let baseline = curve.baseline;
        let curve = curve.blocks.entry((block, uniform)).or_default();
        curve.older = std::mem::replace(&mut curve.previous, confidence.to_vec());
        curve.setup = curve.setup.max(timing[3]);
        *curve
            .seen
            .entry(batch + counts.iter().map(|p| p.0).sum::<usize>())
            .or_default() += 1;
        // Retain a conservative full-cohort draft cost when only some owners draft.
        let scale = if uniform {
            1.
        } else {
            batch as f64 / positive as f64
        };
        update_cost(&mut curve.draft, timing[0] * scale, curve.samples > 0);
        update_cost(
            &mut curve.extra,
            ((timing[1] - baseline) / proposed).max(0.),
            curve.samples > 0,
        );
        update_cost(&mut curve.restore, timing[2] * scale, curve.samples > 0);
        let width = counts.iter().map(|p| p.0).max().unwrap_or(0);
        if curve.acceptance.len() < width {
            curve.acceptance.resize(width, 0.7);
        }
        for i in 0..curve.acceptance.len() {
            let attempts = counts.iter().filter(|&&(k, a)| k > i && a >= i).count();
            let passed = counts.iter().filter(|&&(k, a)| k > i && a > i).count();
            if attempts > 0 {
                update(
                    &mut curve.acceptance[i],
                    passed as f64 / attempts as f64,
                    curve.samples > 0,
                );
            }
        }
        curve.samples += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget(rows: usize) -> Budget {
        Budget {
            rows,
            remaining: 1024,
            uniform: false,
        }
    }
    #[test]
    fn stable_depth_uses_the_shared_cost_curve_and_keeps_cheap_prefixes() {
        for (accepted, confidence, expected) in
            [(5, [0.9; 5], 5), (1, [0.9, 0.01, 0.01, 0.01, 0.01], 1)]
        {
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
                    &confidence,
                );
            }
            let a = Feedback::default();
            let b = Feedback::default();
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
                &[],
            );
            state.observe(&[1.; 5], 5, 5, 5);
        }
        let expected = state.rates(5, true, &[], 5);
        for _ in 0..4 {
            policy.observe(
                1,
                (false, false, false),
                [0.02, 0.1, 0.002, 0.],
                &[(16, 0)],
                16,
                &[],
            );
            state.observe(&[0.01; 16], 16, 0, 16);
        }
        assert_eq!(state.rates(5, true, &[], 5), expected);
        assert_eq!(expected, vec![1.; 5]);
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
            &[],
        );
        assert_eq!(
            policy
                .select_requests(
                    false,
                    false,
                    Budget {
                        rows: 0,
                        remaining: 16,
                        uniform: false
                    },
                    &[16],
                    &[&state],
                    &[5, 16]
                )
                .unwrap(),
            [5]
        );
        assert_eq!(
            policy
                .select_requests(false, false, budget(0), &[16], &[&state], &[5, 16])
                .unwrap(),
            [16]
        );
        let prior = [0.9, 0.5, 0.2];
        assert_eq!(
            Feedback::default().rates(3, true, &prior, 5),
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
                &vec![confidence; maximum],
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
            fast.select_requests(false, true, budget(0), &[16], &[&f], &[16])
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
    fn graph_cost_replaces_cold_direct_cost_and_budget_bounds_prefixes() {
        let (mut p, f) = measured(16, 0.99, 16, [0.250, 2.0, 0.01]);
        assert_eq!(
            p.select_requests(false, false, budget(0), &[16], &[&f], &[16])
                .unwrap(),
            [0]
        );
        p.observe(
            1,
            (false, false, false),
            [0.007, 0.036, 0.001, 0.],
            &[(16, 16)],
            16,
            &[],
        );
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
    fn newer_confidence_allocates_a_fixed_older_budget_without_cross_owner_leaks() {
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
                &[],
            );
        }
        let high = Feedback {
            blocks: [(
                8,
                History {
                    previous: vec![0.99; 8],
                    older: vec![0.8; 8],
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
                    older: vec![0.8; 8],
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
}
