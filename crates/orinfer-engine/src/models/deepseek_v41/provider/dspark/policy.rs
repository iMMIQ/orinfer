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
    pub(super) fn rates(&self, maximum: usize, prior: &[f64], block: usize) -> Vec<f64> {
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
            if (0. ..=1.).contains(&prediction) {
                let label = f64::from(i < accepted);
                let actual = self.acceptance.rate(i, 0.7);
                let predicted = (actual + (prediction + self.bias[i]).clamp(0., 1.)) * 0.5;
                self.errors[i].observe(predicted, actual, label);
                self.bias[i] = adaptive::ema(self.bias[i], label - prediction);
            }
        }
        self.previous = confidence.to_vec();
        self.acceptance.observe(proposed, accepted);
    }
    fn rates(&self, maximum: usize, prior: &[f64]) -> Vec<f64> {
        adaptive::prefix_survival((0..maximum).map(|i| {
            let actual = self
                .acceptance
                .rate(i, prior.get(i).copied().unwrap_or(0.7));
            // This is the checkpoint's trained AcceptRatePredictor, not
            // the draft token's softmax probability. Use one forecast for
            // ranking and useful progress, corrected by tested outcomes.
            let predicted = self.previous.get(i).copied().filter(|p| {
                (0. ..=1.).contains(p) && self.errors.get(i).is_some_and(|error| error.useful())
            });
            let rate = predicted.map_or(actual, |p| {
                (actual + (p + self.bias.get(i).copied().unwrap_or(0.)).clamp(0., 1.)) * 0.5
            });
            rate.clamp(0., 1.)
        }))
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
            trace: crate::models::deepseek_v41::diag().dspark_policy_trace,
            costs: Model::default(),
            ledger: adaptive::Budget::default(),
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
        adaptive::ema(*old, sample)
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
            crate::models::deepseek_v41::log_async(format!(
                "DSPARK_FEEDBACK {}",
                serde_json::json!({
                    "owner":owner,"block":block,"proposed":counts.0,"accepted":counts.1,
                    "trusted_positions":errors.iter().filter(|e| e.useful()).count(),
                    "compared_positions":compared,
                    "forecast_brier":errors.iter().filter(|e| e.observed).map(|e| e.head).sum::<f64>() / compared.max(1) as f64,
                    "history_brier":errors.iter().filter(|e| e.observed).map(|e| e.history).sum::<f64>() / compared.max(1) as f64,
                })
            ));
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
        self.ledger.set_limit(limit);
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
                .filter(|p| {
                    p.samples >= adaptive::HOT_SAMPLES
                        || (p.observations >= 1
                            && (self.costs.has_calibrated_neighbor(stage, full, work)
                                || self.costs.profiled(stage, 0, full, work)))
                })
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
            receipt.calibrating,
        );
        if self.trace {
            crate::models::deepseek_v41::log_async(format!(
                "DSPARK_SETTLE {}",
                serde_json::json!({"family":receipt.family,"width":receipt.width,"seconds":seconds,"committed":tokens,"normal_s":receipt.normal_s,"balance":self.ledger.balance,"startup_spent":self.ledger.startup_spent})
            ));
        }
    }
    pub fn preparation_s(&self) -> f64 {
        self.pending.as_ref().map_or(0., |r| r.preparation_s)
    }
    pub fn diagnostics(&self) -> (usize, f64) {
        (self.costs.cell_count(), self.ledger.balance)
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
    pub fn baseline_budget(
        &self,
        full: bool,
        decode: bool,
        work: Work,
        uses: f64,
        slack: f64,
    ) -> super::super::super::team::CaptureBudget {
        let mut budget = self.target_budget(full, decode, work, uses, slack);
        if decode
            && let Some((hot, seen)) = self.costs.local_evidence(Stage::Decode, 0, full, work)
            && hot < adaptive::HOT_SAMPLES
        {
            // Auto reconsiders speculation after these ordinary samples.
            // The remaining output is not a forecast of ordinary calls.
            budget.max_uses = Some(adaptive::CALIBRATION_ROUNDS.saturating_sub(seen).max(1) as f64);
        }
        budget
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
            crate::models::deepseek_v41::log_async(format!(
                "DSPARK_BASELINE {}",
                serde_json::json!({
                    "batch": batch, "full": full, "mixed": work.prefill_rows > 0,
                    "seconds": timing.seconds, "replay": timing.replay,
                })
            ));
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
        // Once a nearby cohort in the same stage is already hot-calibrated (or
        // an offline prior matches this work), a single local observation
        // confirms this batch's baseline without repeating 4 direct rounds.
        let baseline_ready = normal.samples >= adaptive::HOT_SAMPLES
            || (normal.observations >= 1
                && (self.costs.has_calibrated_neighbor(stage, full, work)
                    || self.costs.profiled(stage, 0, full, work)));
        if limit <= batch || !baseline_ready {
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
        let available = self.ledger.available();
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
                .map(|i| prior.map_or(0.7, |p| p.acceptance.rate(i, 0.7)))
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
                // Round history spans contexts. Only nearby stage samples
                // establish calibration for this packet's actual workload.
                let samples = evidence.iter().map(|p| p.0).min().unwrap();
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
        self.next_calibrating = decision.as_ref().is_some_and(|d| d.calibrating);
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
        adaptive::evict_oldest(&mut curve.rounds, key, 256, |cost| cost.last);
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
        *curve.seen.entry(rows).or_default() += 1;
        let draft_work = Work {
            rows: positive,
            ..Work::default()
        };
        let work = if costs.work.rows == 0 {
            Work {
                rows,
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
    crate::models::deepseek_v41::log_async(format!(
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
    ));
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
