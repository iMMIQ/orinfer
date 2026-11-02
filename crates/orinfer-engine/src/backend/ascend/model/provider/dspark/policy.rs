//! Shared batch cost and request-local, causal confidence budgets.
use crate::artifact::Result;
use std::collections::BTreeMap;
#[path = "cost.rs"]
mod cost;
pub(in crate::backend::ascend::model::provider) use cost::{Identity, Work};
use cost::{Model, Point, Stage};

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
    pub fn progress(&self, depth: usize, block: usize) -> f64 {
        1. + self.rates(depth, true, &[], block).iter().sum::<f64>()
    }

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
    pub remaining_by_owner: Vec<usize>,
    pub uniform: bool,
    pub work: Work,
    pub prefill_tokens_per_s: f64,
}

pub(in crate::backend::ascend::model::provider) struct Policy {
    curves: BTreeMap<(usize, bool, bool), Curve>,
    trace: bool,
    costs: Model,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            curves: BTreeMap::new(),
            trace: std::env::var_os("ORINFER_A2_DSPARK_POLICY_TRACE").is_some(),
            costs: Model::default(),
        }
    }
}
#[derive(Default)]
struct Curve {
    baseline: f64,
    baseline_samples: usize,
    ticks: usize,
    probe_credit: f64,
    blocks: BTreeMap<(usize, bool), BlockCurve>,
}
#[derive(Default)]
struct BlockCurve {
    samples: usize,
    acceptance_samples: usize,
    seen: BTreeMap<usize, usize>,
    acceptance: Vec<f64>,
    previous: Vec<f64>,
    older: Vec<f64>,
}

pub(in crate::backend::ascend::model::provider) struct Costs {
    pub stages: [f64; 4],
    pub draft_replay: bool,
    pub verify_replay: bool,
    pub work: Work,
    pub setup: [f64; 2],
}

impl From<[f64; 4]> for Costs {
    fn from(stages: [f64; 4]) -> Self {
        Self {
            stages,
            draft_replay: false,
            verify_replay: false,
            work: Work::default(),
            setup: [stages[3] * 0.5; 2],
        }
    }
}
fn update(old: &mut f64, sample: f64, initialized: bool) {
    *old = if initialized {
        *old * 0.8 + sample * 0.2
    } else {
        sample
    };
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
fn reuse_steps(
    budget: &Budget,
    depths: &[usize],
    maxima: &[usize],
    feedback: &[&Feedback],
    block: usize,
) -> f64 {
    depths
        .iter()
        .enumerate()
        .map(|(i, &depth)| {
            let remaining = budget
                .remaining_by_owner
                .get(i)
                .copied()
                .unwrap_or(budget.remaining);
            let guard = if depth == 0 {
                0
            } else if budget.uniform {
                maxima[i]
            } else {
                depth
            };
            remaining.saturating_sub(guard).max(1) as f64 / feedback[i].progress(depth, block)
        })
        .fold(f64::INFINITY, f64::min)
        .max(1.)
}
impl Policy {
    pub fn load(identity: Identity) -> Result<Self> {
        let mut policy = Self::default();
        policy.costs = Model::load(identity, policy.trace)?;
        Ok(policy)
    }
    pub fn target_budget(
        &self,
        full: bool,
        decode: bool,
        work: Work,
        uses: f64,
        slack: f64,
    ) -> super::super::super::team::CaptureBudget {
        self.costs.budget(
            if decode { Stage::Decode } else { Stage::Target },
            0,
            full,
            work,
            uses,
            slack,
        )
    }
    pub fn draft_budget(
        &self,
        block: usize,
        owners: usize,
        uses: f64,
        slack: f64,
    ) -> super::super::super::team::CaptureBudget {
        self.costs.budget(
            Stage::Draft,
            block,
            false,
            Work {
                rows: owners,
                ..Work::default()
            },
            uses,
            slack,
        )
    }
    pub fn baseline_execution(
        &mut self,
        batch: usize,
        full: bool,
        work: Work,
        timing: &super::super::super::team::Timing,
    ) {
        if self.trace {
            eprintln!(
                "DSPARK_BASELINE {}",
                serde_json::json!({
                    "batch": batch, "full": full, "mixed": work.prefill_rows > 0,
                    "seconds": timing.seconds, "replay": timing.replay,
                })
            );
        }
        self.costs.observe(Point {
            stage: if batch > 0 && work.prefill_rows == 0 {
                Stage::Decode
            } else {
                Stage::Target
            },
            block: 0,
            full,
            replay: timing.replay,
            work,
            seconds: timing.seconds + timing.preparation - timing.setup,
            setup_s: timing.setup,
        });
        let curve = self
            .curves
            .entry((batch, full, work.prefill_rows > 0))
            .or_default();
        update(
            &mut curve.baseline,
            timing.seconds,
            curve.baseline_samples > 0,
        );
        curve.baseline_samples += 1;
    }
    #[cfg(test)]
    pub fn baseline(&mut self, batch: usize, full: bool, mixed: bool, seconds: f64) {
        let work = Work {
            rows: batch,
            ..Work::default()
        };
        self.costs.observe(Point {
            stage: if batch > 0 && work.prefill_rows == 0 {
                Stage::Decode
            } else {
                Stage::Target
            },
            block: 0,
            full,
            replay: false,
            work,
            seconds,
            setup_s: 0.,
        });
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        curve.baseline = seconds;
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
        let work = if budget.work.rows == 0 {
            Work {
                rows: batch,
                ..Work::default()
            }
        } else {
            budget.work
        };
        let ordinary = if mixed { Stage::Target } else { Stage::Decode };
        let normal = self.costs.predict(ordinary, 0, full, work);
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        curve.ticks += 1;
        if limit <= batch || normal.is_none() {
            if self.trace {
                trace_selection(
                    curve,
                    full,
                    mixed,
                    limit,
                    maxima,
                    &depths,
                    "baseline_or_budget",
                );
            }
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
        let normal = normal.unwrap();
        curve.probe_credit = (curve.probe_credit + normal.seconds * 0.025).min(normal.seconds * 3.);
        let normal_horizon = reuse_steps(&budget, &vec![0; batch], maxima, feedback, blocks[0]);
        let normal_time = self
            .costs
            .price(ordinary, 0, full, work, normal_horizon, 0)
            .unwrap_or(normal.seconds);
        let mut throughput = batch as f64 / (normal_time * 0.97);
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
            let mut local_limit = limit.min(batch + local.iter().sum::<usize>());
            if positive == 0 {
                continue;
            }
            if budget.uniform {
                local_limit = batch + (local_limit - batch) / positive * positive;
            }
            if local_limit == batch {
                continue;
            }
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
            let draft_work = Work {
                rows: positive,
                ..Work::default()
            };
            let draft = self.costs.predict(Stage::Draft, block, false, draft_work);
            let restore = self.costs.predict(Stage::Restore, block, full, draft_work);
            let horizon = reuse_steps(&budget, &local, maxima, feedback, block);
            let graph_budget = self.costs.budget(
                Stage::Target,
                0,
                full,
                work.extend(local_limit - batch),
                horizon,
                f64::INFINITY,
            );
            let unknown = draft.is_none() || restore.is_none();
            let missing_graph_prior =
                graph_budget.replay_s.is_none() || graph_budget.capture_s.is_none();
            let cold_probe = missing_graph_prior
                && horizon >= 16.
                && cost.seen.get(&local_limit).copied().unwrap_or(0) < 3;
            let probe_cost = draft.map_or(normal_time, |p| p.seconds)
                + graph_budget.direct_s.unwrap_or(normal_time)
                - normal_time;
            let periodic_probe = explore
                && horizon >= 16.
                && !curve.ticks.is_multiple_of(32)
                && (curve.ticks / 16) % blocks.len()
                    == blocks.iter().position(|&b| b == block).unwrap()
                && curve.probe_credit >= probe_cost.max(0.);
            if (unknown && cost.samples == 0 && (block == blocks[0] || horizon >= 16.))
                || cold_probe
                || periodic_probe
            {
                if periodic_probe {
                    curve.probe_credit -= probe_cost.max(0.);
                }
                best = local_limit;
                best_maxima = local;
                best_block = block;
                break;
            }
            if unknown || curve.ticks.is_multiple_of(32) {
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
            let mut scored_depths = vec![0; batch];
            for (index, &(rate, owner, _)) in queue(&local, scoring, &cost.acceptance, true, block)
                .iter()
                .take(local_limit - batch)
                .enumerate()
            {
                let rows = batch + index + 1;
                expected += rate;
                scored_depths[owner] += 1;
                if (budget.uniform && (rows - batch).is_multiple_of(positive))
                    || (!budget.uniform && (rows.is_power_of_two() || rows == local_limit))
                {
                    let horizon = reuse_steps(&budget, &scored_depths, maxima, feedback, block);
                    let warmup = 2usize.saturating_sub(cost.seen.get(&rows).copied().unwrap_or(0));
                    let target_work = work.extend(rows - batch);
                    let Some(target_time) =
                        self.costs
                            .price(Stage::Target, 0, full, target_work, horizon, warmup)
                    else {
                        continue;
                    };
                    let Some(draft_time) =
                        self.costs
                            .price(Stage::Draft, block, false, draft_work, horizon, warmup)
                    else {
                        continue;
                    };
                    let restore_time = restore.unwrap().seconds;
                    let displaced = work.prefill_rows - target_work.prefill_rows;
                    let opportunity = if budget.prefill_tokens_per_s > 0. {
                        displaced as f64 / budget.prefill_tokens_per_s
                    } else {
                        0.
                    };
                    let time = draft_time + target_time + restore_time + opportunity;
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
            let positive = best_maxima.iter().filter(|&&m| m > 0).count();
            if let Some(count) = (best - batch).checked_div(positive) {
                for (depth, &maximum) in depths.iter_mut().zip(&best_maxima) {
                    if maximum > 0 {
                        *depth = count;
                    }
                }
            }
            if self.trace {
                trace_selection(curve, full, mixed, limit, maxima, &depths, "uniform_score");
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
        if self.trace {
            trace_selection(curve, full, mixed, limit, maxima, &depths, "request_score");
        }
        Ok(depths)
    }
    pub fn observe(
        &mut self,
        batch: usize,
        mode: (bool, bool, bool),
        timing: impl Into<Costs>,
        counts: &[(usize, usize)],
        block: usize,
        confidence: &[f64],
    ) {
        let costs = timing.into();
        let timing = costs.stages;
        let (full, mixed, uniform) = mode;
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        let positive = counts.iter().filter(|p| p.0 > 0).count();
        if positive == 0 {
            return;
        }
        let curve = curve.blocks.entry((block, uniform)).or_default();
        curve.older = std::mem::replace(&mut curve.previous, confidence.to_vec());
        *curve
            .seen
            .entry(batch + counts.iter().map(|p| p.0).sum::<usize>())
            .or_default() += 1;
        let draft_work = Work {
            rows: positive,
            ..Work::default()
        };
        let work = if costs.work.rows == 0 {
            Work {
                rows: batch + counts.iter().map(|p| p.0).sum::<usize>(),
                ..Work::default()
            }
        } else {
            costs.work
        };
        for point in [
            Point {
                stage: Stage::Draft,
                block,
                full: false,
                replay: costs.draft_replay,
                work: draft_work,
                seconds: timing[0],
                setup_s: costs.setup[0],
            },
            Point {
                stage: Stage::Target,
                block: 0,
                full,
                replay: costs.verify_replay,
                work,
                seconds: timing[1],
                setup_s: costs.setup[1],
            },
            Point {
                stage: Stage::Restore,
                block,
                full,
                replay: false,
                work: draft_work,
                seconds: timing[2],
                setup_s: 0.,
            },
        ] {
            self.costs.observe(point);
        }
        curve.samples += 1;
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
                    curve.acceptance_samples > 0,
                );
            }
        }
        curve.acceptance_samples += 1;
    }
}

fn trace_selection(
    curve: &Curve,
    full: bool,
    mixed: bool,
    limit: usize,
    maxima: &[usize],
    depths: &[usize],
    reason: &str,
) {
    eprintln!(
        "DSPARK_POLICY {}",
        serde_json::json!({
            "batch": maxima.len(), "full": full, "mixed": mixed, "limit": limit,
            "maxima": maxima, "depths": depths, "reason": reason,
            "baseline_samples": curve.baseline_samples, "baseline_s": curve.baseline,
            "ticks": curve.ticks,
            "blocks": curve.blocks.iter().map(|(&(block, uniform), c)| serde_json::json!({
                "block":block,"uniform":uniform,"samples":c.samples,
                "acceptance":c.acceptance,
            })).collect::<Vec<_>>(),
        })
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    fn budget(rows: usize) -> Budget {
        Budget {
            rows,
            remaining: 1024,
            uniform: false,
            ..Budget::default()
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
            // Ordinary decode and one-draft verification have different
            // execution geometry. Supply the measured short verification cost.
            policy.costs.observe(Point {
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
    fn costly_direct_shapes_get_bounded_graph_probes_with_rounded_budgets() {
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
            assert_eq!(selected, [3]);
            p.observe(
                1,
                (false, false, true),
                [0.003, 1.0, 0.001, 0.],
                &[(3, 0)],
                5,
                &[],
            );
        }
        assert_eq!(
            p.select_requests(false, false, limits(), &[5], &[&f], &[5])
                .unwrap(),
            [0]
        );
        let mut p = Policy::default();
        p.baseline(2, false, false, 0.020);
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
            prefill_tokens_per_s: 4000.,
            remaining_by_owner: vec![],
        };
        policy.baseline_execution(
            16,
            false,
            work,
            &super::super::super::super::team::Timing {
                seconds: 0.1,
                preparation: 0.01,
                setup: 0.,
                replay: false,
            },
        );
        let selected = policy
            .select_requests(false, true, limits(), &[5; 16], &feedback, &[5])
            .unwrap();
        assert_eq!(selected, [5; 16]);
        policy.observe(
            16,
            (false, true, true),
            Costs {
                stages: [0.01, 0.1, 0.001, 0.],
                draft_replay: true,
                verify_replay: false,
                work: work.extend(80),
                setup: [0.; 2],
            },
            &[(5, 5); 16],
            5,
            &[0.99; 5],
        );
        // Draft replay has no direct sample yet: a bounded subsequent probe
        // can seed it, while target feedback is already usable independently.
        assert!(
            policy
                .costs
                .predict(Stage::Target, 0, false, work.extend(80))
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
                &[],
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
