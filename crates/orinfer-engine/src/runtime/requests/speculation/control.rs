//! Runtime decisions only: kernels and legal row geometries remain package-owned.
mod cost;
// Reprofiling is a real execution and may displace captured graphs. Acceptance
// changes are evaluated every round; only unused timing estimates age here.
use cost::{Cost, Costs};

#[cfg(test)]
use crate::mtp::adaptive::REFRESH_AFTER;
use crate::mtp::adaptive::{self, Budget};
pub(in crate::runtime) use crate::mtp::adaptive::{Acceptance, Measurement, progress};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::runtime) struct Key {
    pub batch: usize,
    pub context: u32,
    pub sampling: bool,
    pub prefill: usize,
    pub shared_work: bool,
}
impl Key {
    fn family(self) -> (usize, bool, usize) {
        (
            self.batch,
            self.sampling,
            usize::from(self.prefill > 0) + 2 * usize::from(self.shared_work),
        )
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
enum SinglePolicy {
    #[default]
    Auto,
    Fixed(usize),
}

pub(in crate::runtime) struct Controller {
    single_policy: SinglePolicy,
    costs: Costs,
    budget: Budget,
    calibration_limit: f64,
    prefill_calibration: Option<(super::super::cost::Key, usize)>,
    trace: bool,
    #[cfg(test)]
    pub(in crate::runtime) fixed_width: Option<usize>,
}
impl Default for Controller {
    fn default() -> Self {
        let single_policy = match std::env::var("ORINFER_SINGLE_MTP_POLICY").as_deref() {
            Ok("auto" | "explore" | "explore_local") | Err(_) => SinglePolicy::Auto,
            Ok("fixed0") => SinglePolicy::Fixed(1),
            Ok("fixed1") => SinglePolicy::Fixed(2),
            Ok("fixed2") => SinglePolicy::Fixed(3),
            Ok("fixed3") => SinglePolicy::Fixed(4),
            Ok(other) => panic!("Unsupported ORINFER_SINGLE_MTP_POLICY: {other}"),
        };
        Self {
            single_policy,
            costs: Costs::default(),
            budget: Budget::default(),
            calibration_limit: 0.4,
            prefill_calibration: None,
            trace: std::env::var("ORINFER_MTP_TRACE").as_deref() == Ok("1"),
            #[cfg(test)]
            fixed_width: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::runtime) struct Candidate {
    pub width: usize,
    pub expected_tokens: f64,
    pub prior_seconds: f64,
    pub remaining_tokens: usize,
    /// Read from the selected owners' actual draft cursors for this round.
    pub warm_ready: bool,
    /// The selected owners advanced without fresh draft acceptance observations.
    pub review_acceptance: bool,
}

/// Freeze CUDA prices; the common selector owns scoring and funding.
fn select(
    costs: &cost::Profile,
    candidates: &[Candidate],
    input: adaptive::Selection,
    prior: impl Fn(&Candidate) -> f64,
) -> Option<adaptive::Decision> {
    let choices = candidates
        .iter()
        .map(|c| {
            let shape = costs.arms.get(&c.width);
            let hot = shape
                .filter(|s| s.time.samples() > 0)
                .map_or_else(|| prior(c), Cost::mean);
            adaptive::Candidate {
                id: c.width,
                width: c.width,
                expected_tokens: c.expected_tokens,
                maximum_tokens: (input.batch * c.width) as f64,
                remaining_tokens: c.remaining_tokens,
                mean_s: hot,
                upper_s: shape
                    .filter(|s| s.time.samples() > 0)
                    .map_or(hot, Cost::upper),
                transition_s: shape
                    .filter(|_| !c.warm_ready)
                    .map_or(0., |s| s.transition_s),
                warm_ready: c.warm_ready,
                review_acceptance: c.review_acceptance,
                samples: shape.map_or(0, |s| s.time.samples()),
                last: shape.map(|s| s.last),
            }
        })
        .collect::<Vec<_>>();
    adaptive::select(input, &choices)
}

impl Controller {
    pub(in crate::runtime) fn tracing(&self) -> bool {
        self.trace
    }
    pub(in crate::runtime) fn set_calibration_limit(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            self.calibration_limit = seconds;
        }
    }
    /// Mixed packets perform their prompt work once. Extra accepted tokens
    /// displace only standalone decoder work, not another copy of the prompt.
    fn fixed_work(&self, key: Key, normal: f64) -> Option<f64> {
        if key.prefill == 0 {
            return Some(0.);
        }
        if normal <= 0. {
            return None;
        }
        self.costs
            .normal(Key { prefill: 0, ..key })
            .filter(|cost| cost.time.samples() >= adaptive::HOT_SAMPLES)
            .map(|cost| (normal - cost.time.mean()).max(0.))
    }
    /// A waiting prompt gets the next calibration opportunity after existing
    /// debt is repaid. The lease expires when that workload stops recurring.
    pub(in crate::runtime) fn calibrate_prefill(
        &mut self,
        key: super::super::cost::Key,
        excess_s: Option<f64>,
    ) {
        match excess_s {
            None if self
                .prefill_calibration
                .is_some_and(|(pending, _)| pending == key) =>
            {
                self.prefill_calibration = None;
            }
            Some(excess)
                if excess
                    <= self.calibration_limit
                        + (self.calibration_limit - self.budget.startup_spent).max(0.)
                    && self.prefill_calibration.is_none_or(|(pending, until)| {
                        pending == key || self.costs.clock >= until
                    }) =>
            {
                self.prefill_calibration = Some((
                    key,
                    self.costs.clock.saturating_add(adaptive::REFRESH_AFTER),
                ));
            }
            _ => {}
        }
    }
    pub(in crate::runtime) fn exploration_credit(&self, calibrating: bool) -> f64 {
        if calibrating {
            self.budget.available(self.calibration_limit)
        } else {
            self.budget.balance
        }
    }
    pub(in crate::runtime) fn settle_action(
        &mut self,
        seconds: f64,
        baseline: f64,
        measured: bool,
        calibrating: bool,
    ) {
        if !seconds.is_finite() || seconds <= 0. || !baseline.is_finite() {
            return;
        }
        let delta = baseline - seconds;
        self.budget.settle_delta(
            if measured { delta } else { delta.min(0.) },
            self.calibration_limit,
            calibrating,
        );
    }
    pub(in crate::runtime) fn earn_ordinary(&mut self, seconds: f64, preparation_s: f64) {
        if seconds.is_finite() && seconds > 0. && preparation_s.is_finite() && preparation_s >= 0. {
            self.budget.settle_delta(
                (seconds - preparation_s).max(0.) * 0.02,
                self.calibration_limit,
                false,
            );
        }
    }
    /// The single settlement path charges preparation, rejection and zero yield.
    /// Returned hot marginal cost trains the runtime's unmeasured-shape prior.
    #[cfg(test)]
    pub(in crate::runtime) fn observe(
        &mut self,
        key: Key,
        width: usize,
        measurement: Measurement,
    ) -> Option<f64> {
        self.observe_with_alternative(key, width, measurement, None)
    }
    pub(in crate::runtime) fn observe_with_alternative(
        &mut self,
        key: Key,
        width: usize,
        measurement: Measurement,
        alternative: Option<super::super::cost::Alternative>,
    ) -> Option<f64> {
        if !measurement.valid() || key.batch == 0 {
            return None;
        }
        let normal = self
            .costs
            .normal(key)
            .filter(|s| s.time.samples() >= 3)
            .map_or(0., |s| s.time.mean());
        let calibrating = width > 1
            && self
                .costs
                .profiles
                .get(&key)
                .and_then(|p| p.arms.get(&width))
                .is_none_or(|s| s.time.samples() < 3);
        if let Some(alternative) = alternative {
            self.settle_action(
                measurement.seconds,
                alternative.seconds(measurement.tokens),
                alternative.measured,
                alternative.calibrating || calibrating,
            );
        } else if key.prefill > 0 && width > 1 {
            let baseline = self.fixed_work(key, normal).map(|fixed| {
                fixed + (normal - fixed) * measurement.tokens as f64 / key.batch as f64
            });
            self.settle_action(
                measurement.seconds,
                baseline.unwrap_or(0.),
                baseline.is_some(),
                calibrating,
            );
        } else {
            self.budget.settle(
                width,
                measurement,
                normal,
                key.batch,
                self.calibration_limit,
                calibrating,
            );
        }
        self.costs.observe(key, width, measurement);
        if self.trace {
            eprintln!(
                "MTP_OBSERVE {key:?} width={width} {measurement:?} normal={normal:.6} balance={:.6} startup_spent={:.6}",
                self.budget.balance, self.budget.startup_spent
            );
        }
        (width > 1 && normal > 0. && measurement.is_steady() && measurement.tokens > 0)
            .then(|| (measurement.seconds - normal).max(0.) / (width - 1) as f64)
    }
    pub(in crate::runtime) fn choose(
        &mut self,
        key: Key,
        candidates: &[Candidate],
        slack: f64,
    ) -> Option<(usize, bool)> {
        #[cfg(test)]
        if let Some(width) = self.fixed_width {
            return candidates
                .iter()
                .any(|c| c.width == width)
                .then_some((width, false));
        }
        if key.batch == 1
            && key.prefill == 0
            && let SinglePolicy::Fixed(width) = self.single_policy
        {
            return candidates
                .iter()
                .filter(|c| c.width <= width && c.prior_seconds <= slack)
                .max_by_key(|c| c.width)
                .map(|c| (c.width, false));
        }
        let profile = self.costs.profiles.get(&key)?;
        let base = profile.arms.get(&1)?;
        if base.time.samples() < 3 || key.batch == 0 {
            return None;
        }
        let normal = base.time.mean();
        let fixed_s = self.fixed_work(key, normal)?;
        let available = self.budget.available(self.calibration_limit);
        let decision = select(
            profile,
            candidates,
            adaptive::Selection {
                normal_s: normal,
                fixed_s,
                batch: key.batch,
                slack_s: slack,
                available_s: self.budget.balance,
                calibration_s: available,
                allow_exploration: self
                    .prefill_calibration
                    .is_none_or(|(_, until)| self.costs.clock >= until),
                calls: profile.calls,
                normal_last: Some(base.last),
            },
            |c| self.costs.prior(key, c.width, c.prior_seconds),
        );
        let decision = decision.map(|choice| (choice.id, choice.exploring));
        if self.trace {
            let prices: Vec<_> = candidates
                .iter()
                .map(|c| {
                    let cost = profile.arms.get(&c.width);
                    (
                        c.width,
                        c.expected_tokens,
                        c.warm_ready,
                        c.review_acceptance,
                        cost.map(|s| (s.time.samples(), s.time.mean(), s.mean(), s.transition_s)),
                    )
                })
                .collect();
            eprintln!(
                "MTP_CHOOSE {key:?} decision={decision:?} normal={normal:.6} calibration_credit={available:.6} earned_credit={:.6} prices={prices:?}",
                self.budget.balance
            );
        }
        decision
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
