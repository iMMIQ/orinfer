//! Shared batch cost and request-local, causal confidence budgets.
use crate::{artifact::Result, mtp::adaptive};
use std::collections::BTreeMap;
#[path = "cost.rs"]
mod cost;
#[cfg(test)]
#[path = "profile_test.rs"]
mod profile_test;
pub(in crate::models::deepseek_v41::provider) use cost::{Identity, Work};
use cost::{Model, Point, Stage};

#[derive(Default)]
pub(in crate::models::deepseek_v41::provider) struct Feedback {
    blocks: BTreeMap<usize, History>,
}
#[derive(Default)]
struct History {
    previous: Vec<f64>,
    bias: [f64; crate::execution::MAX_MTP_DRAFTS],
    errors: [ForecastError; crate::execution::MAX_MTP_DRAFTS],
    acceptance: adaptive::Acceptance,
}
#[derive(Clone, Copy, Default)]
struct ForecastError {
    head: f64,
    history: f64,
    observed: bool,
}
impl ForecastError {
    fn observe(&mut self, head: f64, history: f64, label: f64) {
        update(&mut self.head, (head - label).powi(2), self.observed);
        update(&mut self.history, (history - label).powi(2), self.observed);
        self.observed = true;
    }
    fn useful(self) -> bool {
        self.observed && self.head < self.history
    }
}
impl Feedback {
    pub fn progress(&self, depth: usize, block: usize) -> f64 {
        1. + self.rates(depth, &[], block).iter().sum::<f64>()
    }

    pub fn observe(&mut self, confidence: &[f64], proposed: usize, accepted: usize, block: usize) {
        self.blocks
            .entry(block)
            .or_default()
            .observe(confidence, proposed, accepted);
    }
    fn rates(&self, maximum: usize, prior: &[f64], block: usize) -> Vec<f64> {
        self.blocks
            .get(&block)
            .unwrap_or(&History::default())
            .rates(maximum, prior)
    }
}
impl History {
    fn observe(&mut self, confidence: &[f64], proposed: usize, accepted: usize) {
        // Only the accepted prefix and first rejection were actually tested.
        // Keep calibration private to this request and physical draft block.
        // The planner used the preceding round's head. Assess and calibrate
        // that forecast before seeing this round's head or updating history.
        for (i, &prediction) in self
            .previous
            .iter()
            .take(
                proposed
                    .min(accepted.saturating_add(1))
                    .min(self.bias.len()),
            )
            .enumerate()
        {
            if prediction.is_finite() && (0. ..=1.).contains(&prediction) {
                let label = f64::from(i < accepted);
                let actual = self.acceptance.estimate(i, 0.7);
                let predicted = (actual + (prediction + self.bias[i]).clamp(0., 1.)) * 0.5;
                self.errors[i].observe(predicted, actual, label);
                self.bias[i] = self.bias[i] * 0.8 + (label - prediction) * 0.2;
            }
        }
        self.previous = confidence.to_vec();
        self.acceptance.observe(proposed, accepted);
    }
    fn rates(&self, maximum: usize, prior: &[f64]) -> Vec<f64> {
        let mut survival = 1.;
        (0..maximum)
            .map(|i| {
                let actual = self
                    .acceptance
                    .estimate(i, prior.get(i).copied().unwrap_or(0.7));
                // This is the checkpoint's trained AcceptRatePredictor, not
                // the draft token's softmax probability. Use one forecast for
                // ranking and useful progress, corrected by tested outcomes.
                let predicted = self.previous.get(i).copied().filter(|p| {
                    p.is_finite()
                        && (0. ..=1.).contains(p)
                        && self.errors.get(i).is_some_and(|error| error.useful())
                });
                let rate = predicted.map_or(actual, |p| {
                    (actual + (p + self.bias.get(i).copied().unwrap_or(0.)).clamp(0., 1.)) * 0.5
                });
                survival *= rate.clamp(0., 1.);
                survival
            })
            .collect()
    }
}

pub(in crate::models::deepseek_v41::provider) struct Budget {
    pub rows: usize,
    pub remaining: usize,
    pub remaining_by_owner: Vec<usize>,
    pub uniform: bool,
    pub work: Work,
    pub prefill_s: f64,
    pub latency: bool,
    pub slack_s: f64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            rows: 0,
            remaining: 0,
            remaining_by_owner: vec![],
            uniform: false,
            work: Work::default(),
            prefill_s: 0.,
            latency: false,
            slack_s: f64::INFINITY,
        }
    }
}
impl Budget {
    fn opportunity(&self, retained: Work) -> f64 {
        self.prefill_s * self.work.prefill_rows.saturating_sub(retained.prefill_rows) as f64
            / self.work.prefill_rows.max(1) as f64
    }
}

pub(in crate::models::deepseek_v41::provider) struct Allocation {
    pub depths: Vec<usize>,
    pub reference: Option<Vec<usize>>,
}
impl From<Vec<usize>> for Allocation {
    fn from(depths: Vec<usize>) -> Self {
        Self {
            depths,
            reference: None,
        }
    }
}

pub(in crate::models::deepseek_v41::provider) struct Policy {
    curves: BTreeMap<(usize, bool, bool), Curve>,
    trace: bool,
    costs: Model,
    ledger: adaptive::Budget,
    calibration_limit: f64,
    pending: Option<Receipt>,
    next_calibrating: bool,
}
struct Receipt {
    family: (usize, bool, usize),
    slots: Vec<usize>,
    width: usize,
    normal_s: f64,
    preparation_s: f64,
    calibrating: bool,
    normal: Option<Point>,
}
impl Default for Policy {
    fn default() -> Self {
        Self {
            curves: BTreeMap::new(),
            trace: std::env::var_os("ORINFER_A2_DSPARK_POLICY_TRACE").is_some(),
            costs: Model::default(),
            ledger: adaptive::Budget::default(),
            calibration_limit: 0.4,
            pending: None,
            next_calibrating: true,
        }
    }
}
#[derive(Default)]
struct Curve {
    baseline: f64,
    baseline_samples: usize,
    calls: usize,
    last_normal: Option<usize>,
    rounds: BTreeMap<(usize, usize, usize, bool), adaptive::Cost>,
    blocks: BTreeMap<(usize, bool), BlockCurve>,
}
#[derive(Default)]
struct BlockCurve {
    samples: usize,
    last: usize,
    seen: BTreeMap<usize, usize>,
    acceptance: adaptive::Acceptance,
}

#[derive(Default)]
pub(in crate::models::deepseek_v41::provider) struct Costs {
    pub stages: [f64; 4],
    pub draft_replay: bool,
    pub verify_replay: bool,
    pub shared: [bool; 2],
    pub shared_setup: [f64; 2],
    pub work: Work,
    pub setup: [f64; 2],
    pub wall_s: Option<f64>,
}

impl From<[f64; 4]> for Costs {
    fn from(stages: [f64; 4]) -> Self {
        Self {
            stages,
            draft_replay: false,
            verify_replay: false,
            shared: [false; 2],
            shared_setup: [0.; 2],
            work: Work::default(),
            setup: [stages[3] * 0.5; 2],
            wall_s: None,
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
fn retain_profitable_prefixes(candidates: &mut Vec<adaptive::Candidate>) {
    let mut incumbents = BTreeMap::<usize, f64>::new();
    for candidate in candidates.iter().filter(|c| c.samples > 0) {
        let rate = candidate.mean_s / candidate.expected_tokens;
        incumbents
            .entry(candidate.width)
            .and_modify(|old| *old = old.min(rate))
            .or_insert(rate);
    }
    // A physical block is already calibrated. A new logical prefix must pay
    // back against its measured incumbent, not merely beat ordinary decode. Prices
    // already amortize its own build, direct primer and Graph capture debt.
    candidates.retain(|c| {
        c.samples > 0
            || incumbents
                .get(&c.width)
                .is_none_or(|&rate| c.mean_s / c.expected_tokens < rate * (1. - 1e-6))
    });
}
#[cfg(test)]
#[test]
fn logical_prefix_probes_compete_with_the_hot_physical_block() {
    let candidate = |id, width, seconds, progress, samples| adaptive::Candidate {
        id,
        width,
        mean_s: seconds,
        upper_s: seconds,
        expected_tokens: progress,
        maximum_tokens: progress,
        remaining_tokens: 1024,
        transition_s: 0.,
        warm_ready: true,
        review_acceptance: false,
        samples,
        last: None,
    };
    let mut choices = vec![
        candidate(0, 6, 0.09, 60., 1),
        candidate(1, 6, 0.08, 40., 0),
        candidate(2, 6, 0.06, 50., 0),
        candidate(3, 9, 1., 20., 0),
    ];
    retain_profitable_prefixes(&mut choices);
    assert_eq!(choices.iter().map(|c| c.id).collect::<Vec<_>>(), [0, 2, 3]);
    assert_eq!(choices[1].samples, 0);
}
fn queue(
    maxima: &[usize],
    feedback: &[&Feedback],
    prior: &[f64],
    block: usize,
) -> Vec<(f64, usize, usize)> {
    let mut rows = maxima
        .iter()
        .zip(feedback)
        .enumerate()
        .flat_map(|(owner, (&maximum, state))| {
            state
                .rates(maximum, prior, block)
                .into_iter()
                .enumerate()
                .map(move |(position, rate)| (rate, owner, position))
        })
        .collect::<Vec<_>>();
    // Cumulative rates are nonincreasing, so this also preserves each prefix.
    rows.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    rows
}
/// Spend the selected verification budget on useful owner prefixes without
/// changing the draft owner set or physical block. Launch compatibility is
/// checked by the caller before adopting this allocation.
pub(super) fn redistribute(
    depths: &[usize],
    maxima: &[usize],
    feedback: &[&Feedback],
    block: usize,
    latency: bool,
) -> Vec<usize> {
    if depths.iter().filter(|&&d| d > 0).count() <= 1
        || depths
            .iter()
            .zip(maxima)
            .all(|(&d, &m)| d == 0 || d == m.min(block))
    {
        return depths.to_vec();
    }
    let rates = feedback
        .iter()
        .map(|f| f.rates(block, &[], block))
        .collect::<Vec<_>>();
    let mut selected = depths
        .iter()
        .map(|&d| usize::from(d > 0))
        .collect::<Vec<_>>();
    let mut yields = selected
        .iter()
        .zip(&rates)
        .map(|(&d, r)| 1. + r.iter().take(d).sum::<f64>())
        .collect::<Vec<_>>();
    let extra = depths.iter().sum::<usize>() - selected.iter().sum::<usize>();
    for _ in 0..extra {
        let owner = (0..selected.len())
            .filter(|&i| selected[i] > 0 && selected[i] < maxima[i].min(block))
            .max_by(|&a, &b| {
                let gain = |i: usize| {
                    let p = rates[i][selected[i]];
                    if latency {
                        p / (yields[i] * (yields[i] + p))
                    } else {
                        p
                    }
                };
                gain(a).total_cmp(&gain(b)).then_with(|| b.cmp(&a))
            });
        let Some(owner) = owner else {
            return depths.to_vec();
        };
        yields[owner] += rates[owner][selected[owner]];
        selected[owner] += 1;
    }
    let previous = adaptive::progress(
        depths
            .iter()
            .zip(&rates)
            .map(|(&d, r)| 1. + r.iter().take(d).sum::<f64>()),
        depths.len(),
        latency,
    );
    let expected = adaptive::progress(yields.into_iter(), depths.len(), latency);
    if expected > previous + 1e-6 {
        selected
    } else {
        depths.to_vec()
    }
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
    pub fn observe_feedback(
        &self,
        owner: usize,
        feedback: &mut Feedback,
        confidence: &[f64],
        counts: (usize, usize),
        block: usize,
    ) {
        feedback.observe(confidence, counts.0, counts.1, block);
        if self.trace {
            let errors = &feedback.blocks[&block].errors[..block];
            let compared = errors.iter().filter(|e| e.observed).count();
            eprintln!(
                "DSPARK_FEEDBACK {}",
                serde_json::json!({
                    "owner":owner,"block":block,"proposed":counts.0,"accepted":counts.1,
                    "trusted_positions":errors.iter().filter(|e| e.useful()).count(),
                    "compared_positions":compared,
                    "forecast_brier":errors.iter().filter(|e| e.observed).map(|e| e.head).sum::<f64>() / compared.max(1) as f64,
                    "history_brier":errors.iter().filter(|e| e.observed).map(|e| e.history).sum::<f64>() / compared.max(1) as f64,
                })
            );
        }
    }
    pub fn prepare_feedback(
        &self,
        batch: usize,
        full: bool,
        mixed: bool,
        uniform: bool,
        feedback: &mut Feedback,
    ) {
        if let Some(curve) = self.curves.get(&(batch, full, mixed)) {
            for (&(block, allocation), observed) in &curve.blocks {
                if allocation == uniform {
                    feedback.blocks.entry(block).or_insert_with(|| History {
                        acceptance: observed.acceptance.prior(),
                        ..History::default()
                    });
                }
            }
        }
    }
    pub fn start_iteration(&mut self, limit: f64) {
        self.pending = None;
        if limit.is_finite() && limit > 0. {
            self.calibration_limit = limit;
        }
    }
    pub fn begin_round(
        &mut self,
        slots: Vec<usize>,
        full: bool,
        work: Work,
        shared: bool,
        width: usize,
    ) {
        if slots.is_empty() {
            return;
        }
        let stage = if work.prefill_rows > 0 {
            Stage::Target
        } else {
            Stage::Decode
        };
        self.pending = Some(Receipt {
            family: (
                slots.len(),
                full,
                usize::from(work.prefill_rows > 0) + 2 * usize::from(shared),
            ),
            slots,
            width,
            preparation_s: 0.,
            calibrating: width > 1 && self.next_calibrating,
            normal: None,
            normal_s: self
                .costs
                .predict(stage, 0, full, work)
                .filter(|p| p.samples >= adaptive::HOT_SAMPLES)
                .map_or(0., |p| p.seconds),
        });
    }
    pub fn finish_iteration(&mut self, seconds: f64, deltas: &[(usize, usize)]) {
        let Some(receipt) = self.pending.take() else {
            return;
        };
        let tokens = deltas
            .iter()
            .filter(|(slot, _)| receipt.slots.contains(slot))
            .map(|(_, n)| n)
            .sum();
        let sample = adaptive::Measurement {
            seconds,
            tokens,
            warm_ready: true,
            preparation_s: receipt.preparation_s,
        };
        if !sample.valid() {
            return;
        }
        if tokens > 0
            && let Some(mut point) = receipt.normal
        {
            // Price ordinary decode over the same completed iteration as
            // verification: CPU sampling, constraints and commit belong here.
            // Record one sample, excluding only the separately billed setup.
            point.seconds = (seconds - receipt.preparation_s).max(0.);
            self.costs.observe(point);
        }
        self.ledger.settle(
            receipt.width,
            sample,
            receipt.normal_s,
            receipt.slots.len(),
            self.calibration_limit,
            receipt.calibrating,
        );
        if self.trace {
            eprintln!(
                "DSPARK_SETTLE {}",
                serde_json::json!({"family":receipt.family,"width":receipt.width,"seconds":seconds,"committed":tokens,"normal_s":receipt.normal_s,"balance":self.ledger.balance,"startup_spent":self.ledger.startup_spent})
            );
        }
    }
    pub fn load(identity: Identity, package_root: &std::path::Path) -> Result<Self> {
        let mut policy = Self::default();
        policy.costs = Model::load(identity, package_root, policy.trace)?;
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
        if let Some(r) = &mut self.pending {
            r.preparation_s += timing.setup;
        }
        if self.trace {
            eprintln!(
                "DSPARK_BASELINE {}",
                serde_json::json!({
                    "batch": batch, "full": full, "mixed": work.prefill_rows > 0,
                    "seconds": timing.seconds, "replay": timing.replay,
                })
            );
        }
        let point = Point {
            stage: if batch > 0 && work.prefill_rows == 0 {
                Stage::Decode
            } else {
                Stage::Target
            },
            block: 0,
            full,
            replay: timing.replay,
            shared: timing.shared,
            shared_setup_s: timing.shared_setup,
            work,
            seconds: timing.seconds + timing.preparation - timing.setup,
            setup_s: timing.setup,
        };
        if let Some(receipt) = &mut self.pending {
            receipt.normal = Some(point);
        } else {
            // Standalone prefill has no decoder settlement receipt.
            self.costs.observe(point);
        }
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
        curve.calls += 1;
        curve.last_normal = Some(curve.calls);
    }
    #[cfg(test)]
    pub fn baseline(&mut self, batch: usize, full: bool, mixed: bool, seconds: f64) {
        let work = Work {
            rows: batch,
            ..Work::default()
        };
        self.costs.observe(Point {
            shared: false,
            shared_setup_s: 0.,
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
        curve.calls += 1;
        curve.last_normal = Some(curve.calls);
    }
    #[cfg(test)]
    pub fn select_requests(
        &mut self,
        full: bool,
        mixed: bool,
        budget: Budget,
        maxima: &[usize],
        feedback: &[&Feedback],
        blocks: &[usize],
    ) -> Result<Vec<usize>> {
        let work = if budget.work.rows == 0 {
            Work {
                rows: maxima.len(),
                ..Work::default()
            }
        } else {
            budget.work
        };
        self.select_requests_with_forecast(
            (full, mixed),
            budget,
            maxima,
            feedback,
            blocks,
            |depths, _| Ok(Some(work.extend(depths.iter().sum()))),
        )
        .map(|a| a.depths)
    }
    pub fn select_requests_with_forecast(
        &mut self,
        mode: (bool, bool),
        budget: Budget,
        maxima: &[usize],
        feedback: &[&Feedback],
        blocks: &[usize],
        mut forecast: impl FnMut(&[usize], Option<&[usize]>) -> Result<Option<Work>>,
    ) -> Result<Allocation> {
        let (full, mixed) = mode;
        let batch = maxima.len();
        if feedback.len() != batch || maxima.iter().any(|&m| m > crate::execution::MAX_MTP_DRAFTS) {
            return Err("Invalid DSpark request-depth contract".into());
        }
        // Bounded packet selection can schedule only prefill; let the shared
        // ordinary path execute it without creating a speculation receipt.
        if batch == 0 {
            return Ok(vec![].into());
        }
        let ordinary = vec![0; batch];
        let limit = if budget.rows == 0 {
            batch + maxima.iter().sum::<usize>()
        } else {
            budget.rows.min(batch + maxima.iter().sum::<usize>())
        };
        let work = if budget.work.rows == 0 {
            Work {
                rows: batch,
                ..Work::default()
            }
        } else {
            budget.work
        };
        let stage = if mixed { Stage::Target } else { Stage::Decode };
        // Neighboring direct batches can overprice a smaller ordinary decode.
        // Sample its real execution mode before speculation can monopolize it;
        // owner/context changes still reuse the cohort, costs and loss ledger.
        if !mixed
            && self
                .curves
                .get(&(batch, full, mixed))
                .is_none_or(|curve| curve.baseline_samples == 0)
        {
            return Ok(ordinary.into());
        }
        let Some(normal) = self.costs.predict(stage, 0, full, work) else {
            return Ok(ordinary.into());
        };
        // Establish a hot ordinary baseline before pricing savings. The first
        // direct call may be far slower than the eventual Graph replay.
        if limit <= batch || normal.samples < adaptive::HOT_SAMPLES {
            return Ok(ordinary.into());
        }
        if blocks.is_empty()
            || blocks.windows(2).any(|b| b[0] >= b[1])
            || blocks
                .iter()
                .any(|&b| b == 0 || b > crate::execution::MAX_MTP_DRAFTS)
        {
            return Err("Invalid DSpark physical blocks".into());
        }
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        let available = self.ledger.available(self.calibration_limit);
        let horizon = reuse_steps(&budget, &ordinary, maxima, feedback, blocks[0]);
        let normal_s = self
            .costs
            .price(stage, 0, full, work, horizon, 0)
            .unwrap_or(normal.seconds);
        let mut estimates = vec![];
        let mut plans = vec![];
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
            if positive == 0 {
                continue;
            }
            let local_limit = limit.min(batch + local.iter().sum::<usize>());
            let local_limit = if budget.uniform {
                batch + (local_limit - batch) / positive * positive
            } else {
                local_limit
            };
            let prior = curve.blocks.get(&(block, budget.uniform));
            let priors = (0..block)
                .map(|i| prior.map_or(0.7, |p| p.acceptance.estimate(i, 0.7)))
                .collect::<Vec<_>>();
            let allocation = queue(&local, feedback, &priors, block);
            let mut depths = vec![0; batch];
            for (index, &(_, owner, _)) in allocation.iter().take(local_limit - batch).enumerate() {
                depths[owner] += 1;
                let rows = batch + index + 1;
                if budget.uniform && !(rows - batch).is_multiple_of(positive) {
                    continue;
                }
                if !budget.uniform
                    && !rows.is_power_of_two()
                    && rows != local_limit
                    && depths[owner] != local[owner]
                    && !(rows - batch).is_multiple_of(positive)
                    && !prior.is_some_and(|p| p.seen.contains_key(&rows))
                {
                    continue;
                }
                // A parallel block is a physical cost key, distinct from the
                // adopted prefix length. Runtime chooses this exact smallest block.
                let mut selected = if budget.uniform {
                    let depth = (rows - batch) / positive;
                    local
                        .iter()
                        .map(|&m| if m > 0 { depth } else { 0 })
                        .collect::<Vec<_>>()
                } else {
                    depths.clone()
                };
                let maximum = *selected.iter().max().unwrap();
                if blocks.iter().copied().find(|&b| b >= maximum) != Some(block) {
                    continue;
                }
                let mut reference = None;
                let mut target_work = None;
                if budget.uniform {
                    let candidate =
                        redistribute(&selected, &local, feedback, block, budget.latency);
                    if candidate != selected
                        && let Some(work) = forecast(&candidate, Some(&selected))?
                    {
                        reference = Some(std::mem::replace(&mut selected, candidate));
                        target_work = Some(work);
                    }
                }
                let target_work = match target_work {
                    Some(work) => work,
                    None => forecast(&selected, None)?.ok_or("Missing DSpark packet forecast")?,
                };
                let yields = feedback
                    .iter()
                    .zip(&selected)
                    .map(|(f, &depth)| 1. + f.rates(depth, &priors, block).iter().sum::<f64>());
                let expected = adaptive::progress(yields, batch, budget.latency);
                let owners = selected.iter().filter(|&&d| d > 0).count();
                let draft_work = Work {
                    rows: owners,
                    ..Work::default()
                };
                let uses = reuse_steps(&budget, &selected, maxima, feedback, block);
                let seen = prior.and_then(|p| p.seen.get(&rows)).copied().unwrap_or(0);
                let warmup = 2usize.saturating_sub(seen);
                let target = self.costs.predict(Stage::Target, 0, full, target_work);
                let draft = self.costs.predict(Stage::Draft, block, false, draft_work);
                let restore = self.costs.predict(Stage::Restore, block, full, draft_work);
                let target_s = self
                    .costs
                    .price(Stage::Target, 0, full, target_work, uses, warmup)
                    .unwrap_or(normal.seconds);
                let draft_s = self
                    .costs
                    .price(Stage::Draft, block, false, draft_work, uses, warmup)
                    .unwrap_or(normal.seconds);
                let restore_s = restore.map_or(normal.seconds * 0.1, |p| p.seconds);
                let opportunity = budget.opportunity(target_work);
                let mean = target_s + draft_s + restore_s + opportunity;
                let observed = curve.rounds.get(&(block, rows, owners, budget.uniform));
                let evidence = [
                    self.costs.evidence(Stage::Target, 0, full, target_work),
                    self.costs.evidence(Stage::Draft, block, false, draft_work),
                    self.costs.evidence(Stage::Restore, block, full, draft_work),
                ];
                let inferred = evidence.iter().map(|p| p.0).min().unwrap();
                let samples = observed.map_or(0, |c| c.time.samples()).max(inferred);
                // One representative maximum packet calibrates a physical
                // family. Shorter prefixes reuse its conservative measured
                // cost; they must not each trigger a fresh Graph calibration.
                // Exact offline costs may admit shorter packets after the
                // full-prefix acceptance primer, never replace that primer.
                if samples < adaptive::HOT_SAMPLES
                    && rows != local_limit
                    && !(prior.is_some_and(|p| {
                        p.seen.get(&local_limit).copied().unwrap_or(0)
                            >= adaptive::CALIBRATION_ROUNDS
                    }) && self.costs.profiled(Stage::Target, 0, full, target_work))
                {
                    continue;
                }
                let mut upper = mean.max(
                    target.map_or(normal.seconds, |p| p.seconds)
                        + draft.map_or(normal.seconds, |p| p.seconds)
                        + restore_s
                        + opportunity,
                );
                if samples < adaptive::HOT_SAMPLES {
                    for (stage, block, work, current) in [
                        (Stage::Target, 0, target_work, target),
                        (Stage::Draft, block, draft_work, draft),
                    ] {
                        // A replay observation prices steady execution. The
                        // executor still reconfirms real cache residency and
                        // bills any unexpected preparation to the same ledger.
                        if current.is_some_and(|p| p.replay) {
                            continue;
                        }
                        let preparing = self.costs.budget(
                            stage,
                            block,
                            stage == Stage::Target && full,
                            work,
                            uses,
                            f64::INFINITY,
                        );
                        if seen >= 2 && preparing.profitable(0) {
                            upper += preparing.capture_s.unwrap_or(0.);
                        } else if seen == 0 {
                            upper += self
                                .costs
                                .probe_preparation(stage, block, work, false)
                                .unwrap_or(0.);
                        }
                    }
                }
                estimates.push(adaptive::Candidate {
                    id: plans.len(),
                    // DSpark generates the whole parallel block even when only
                    // a short prefix is adopted. Calibration orders physical
                    // blocks; equal-cost prefixes compete by useful progress.
                    width: block + 1,
                    expected_tokens: expected,
                    remaining_tokens: if budget.remaining_by_owner.is_empty() {
                        budget.remaining.saturating_mul(batch)
                    } else {
                        budget.remaining_by_owner.iter().sum()
                    },
                    mean_s: mean,
                    upper_s: upper,
                    maximum_tokens: adaptive::progress(
                        selected.iter().map(|&d| d as f64 + 1.),
                        batch,
                        budget.latency,
                    ),
                    transition_s: 0.,
                    warm_ready: true,
                    review_acceptance: feedback.iter().any(|f| !f.blocks.contains_key(&block))
                        || curve
                            .last_normal
                            .is_some_and(|n| prior.is_some_and(|p| n > p.last)),
                    samples,
                    last: observed.map(|c| c.last).or_else(|| prior.map(|p| p.last)),
                });
                plans.push(Allocation {
                    depths: selected,
                    reference,
                });
            }
        }
        retain_profitable_prefixes(&mut estimates);
        estimates.sort_by(|a, b| {
            (a.mean_s / a.expected_tokens).total_cmp(&(b.mean_s / b.expected_tokens))
        });
        let decision = adaptive::select(
            adaptive::Selection {
                normal_s,
                fixed_s: 0.,
                batch,
                slack_s: budget.slack_s,
                available_s: self.ledger.balance,
                calibration_s: available,
                allow_exploration: true,
                calls: curve.calls,
                normal_last: curve.last_normal,
            },
            &estimates,
        );
        self.next_calibrating = decision.as_ref().is_some_and(|d| {
            estimates
                .iter()
                .find(|c| c.id == d.id)
                .is_some_and(|c| c.samples < adaptive::HOT_SAMPLES)
        });
        let reason = if decision.as_ref().is_some_and(|d| d.exploring) {
            "shared_probe"
        } else {
            "shared_score"
        };
        let allocation = decision.map_or_else(|| ordinary.into(), |d| plans.swap_remove(d.id));
        let depths = &allocation.depths;
        if self.trace {
            trace_selection(curve, (full, mixed), &budget, limit, maxima, depths, reason);
        }
        Ok(allocation)
    }
    pub fn observe(
        &mut self,
        batch: usize,
        mode: (bool, bool, bool),
        timing: impl Into<Costs>,
        counts: &[(usize, usize)],
        block: usize,
    ) {
        let costs = timing.into();
        if let Some(r) = &mut self.pending {
            r.preparation_s += costs.setup.iter().sum::<f64>();
        }
        let timing = costs.stages;
        let (full, mixed, uniform) = mode;
        let curve = self.curves.entry((batch, full, mixed)).or_default();
        curve.calls += 1;
        let call = curve.calls;
        let positive = counts.iter().filter(|p| p.0 > 0).count();
        if positive == 0 {
            return;
        }
        let rows = batch + counts.iter().map(|p| p.0).sum::<usize>();
        let key = (block, rows, positive, uniform);
        if curve.rounds.len() == 256 && !curve.rounds.contains_key(&key) {
            let victim = *curve.rounds.iter().min_by_key(|(_, c)| c.last).unwrap().0;
            curve.rounds.remove(&victim);
        }
        curve.rounds.entry(key).or_default().observe(
            adaptive::Measurement {
                seconds: costs.wall_s.unwrap_or_else(|| {
                    timing.iter().sum::<f64>() + costs.setup.iter().sum::<f64>()
                }),
                preparation_s: costs.setup.iter().sum(),
                tokens: counts.iter().map(|p| p.1 + 1).sum(),
                warm_ready: true,
            },
            call,
        );
        let curve = curve.blocks.entry((block, uniform)).or_default();
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
                shared: costs.shared[0],
                shared_setup_s: costs.shared_setup[0],
                work: draft_work,
                seconds: timing[0],
                setup_s: costs.setup[0],
            },
            Point {
                stage: Stage::Target,
                block: 0,
                full,
                replay: costs.verify_replay,
                shared: costs.shared[1],
                shared_setup_s: costs.shared_setup[1],
                work,
                seconds: timing[1],
                setup_s: costs.setup[1],
            },
            Point {
                shared: false,
                shared_setup_s: 0.,
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
        curve.last = call;
        curve.acceptance.observe_batch(counts);
    }
}

fn trace_selection(
    curve: &Curve,
    mode: (bool, bool),
    budget: &Budget,
    limit: usize,
    maxima: &[usize],
    depths: &[usize],
    reason: &str,
) {
    let (full, mixed) = mode;
    eprintln!(
        "DSPARK_POLICY {}",
        serde_json::json!({
            "batch": maxima.len(), "full": full, "mixed": mixed, "limit": limit,
            "maxima": maxima, "depths": depths, "reason": reason,
            "baseline_samples": curve.baseline_samples, "baseline_s": curve.baseline,
            "prefill_rows":budget.work.prefill_rows,"producer_s":budget.prefill_s,
            "ticks": curve.calls,
            "blocks": curve.blocks.iter().map(|(&(block, uniform), c)| serde_json::json!({
                "block":block,"uniform":uniform,"samples":c.samples,
                "acceptance":(0..block).map(|i|c.acceptance.rate(i,0.7)).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        })
    );
}

#[cfg(test)]
mod tests {
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
        assert_eq!(
            history.rates(1, &[])[0],
            history.acceptance.estimate(0, 0.7)
        );
    }
    #[test]
    fn a_predictive_head_can_beat_the_request_history() {
        let mut history = History {
            previous: vec![0.9],
            ..Default::default()
        };
        history.observe(&[0.9], 1, 1);
        assert!(history.errors[0].useful());
        assert!(history.rates(1, &[])[0] > history.acceptance.estimate(0, 0.7));
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
                .all(
                    |(i, &p)| (p - (1. - 0.15 * 0.8_f64.powi(16)).powi(i as i32 + 1)).abs() < 1e-12
                )
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
}
