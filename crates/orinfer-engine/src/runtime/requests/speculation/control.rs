//! Runtime decisions only: kernels and legal row geometries remain package-owned.
mod cost;
// Reprofiling is a real execution and may displace captured graphs. Acceptance
// changes are evaluated every round; only unused timing estimates age here.
#[cfg(test)]
use crate::mtp::adaptive::REFRESH_AFTER;
use crate::mtp::adaptive::{self, Budget};
pub(in crate::runtime) use crate::mtp::adaptive::{Acceptance, Measurement, progress};
use cost::{Cost, Costs};

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
}

/// Convert CUDA timing arms into the shared, backend-neutral selector input.
fn select(
    costs: &cost::Profile,
    candidates: &[Candidate],
    normal: f64,
    slack: f64,
    available: f64,
    prior: impl Fn(&Candidate) -> f64,
    batch: usize,
) -> Option<adaptive::Decision> {
    let candidates = candidates
        .iter()
        .map(|candidate| {
            let shape = costs.arms.get(&candidate.width);
            let samples = shape.map_or(0, |s| s.time.samples());
            let mean = shape
                .filter(|s| s.time.samples() > 0)
                .map_or_else(|| prior(candidate), Cost::mean);
            let upper = shape
                .filter(|s| s.time.samples() > 0)
                .map_or(mean, Cost::upper);
            adaptive::Candidate {
                id: candidate.width,
                width: candidate.width,
                expected_tokens: candidate.expected_tokens,
                remaining_tokens: candidate.remaining_tokens,
                mean_s: mean,
                upper_s: upper,
                probe_reserve_s: None,
                samples,
                last: shape.map(|s| s.last),
            }
        })
        .collect::<Vec<_>>();
    adaptive::select(
        adaptive::Selection {
            normal_s: normal,
            batch,
            slack_s: slack,
            available_s: available,
            calls: costs.calls,
            normal_last: costs.arms.get(&1).map(|s| s.last),
        },
        &candidates,
    )
}

impl Controller {
    pub(super) fn tracing(&self) -> bool {
        self.trace
    }
    pub(in crate::runtime) fn set_calibration_limit(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            self.calibration_limit = seconds;
        }
    }
    /// The single settlement path charges preparation, rejection and zero yield.
    /// Returned hot marginal cost trains the runtime's unmeasured-shape prior.
    pub(in crate::runtime) fn observe(
        &mut self,
        key: Key,
        width: usize,
        measurement: Measurement,
    ) -> Option<f64> {
        if !measurement.valid() || key.batch == 0 {
            return None;
        }
        let normal = self
            .costs
            .normal(key)
            .filter(|s| s.time.samples() >= adaptive::HOT_SAMPLES)
            .map_or(0., Cost::mean);
        self.budget.settle(
            key.family(),
            width,
            measurement,
            normal,
            key.batch,
            self.calibration_limit,
        );
        self.costs.observe(key, width, measurement);
        if self.trace {
            eprintln!(
                "MTP_OBSERVE {key:?} width={width} {measurement:?} normal={normal:.6} balance={:.6} startup_spent={:.6}",
                self.budget.balance(key.family()),
                self.budget.startup_spent
            );
        }
        (width > 1 && normal > 0. && measurement.is_steady() && measurement.tokens > 0)
            .then(|| (measurement.seconds - normal).max(0.) / (width - 1) as f64)
    }
    pub(in crate::runtime) fn choose(
        &mut self,
        key: Key,
        candidates: &[Candidate],
        normal_seconds: f64,
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
        let normal = base.mean().max(normal_seconds.min(base.upper()));
        let available = self.budget.available(key.family(), self.calibration_limit);
        if available == f64::NEG_INFINITY {
            return None;
        }
        let decision = select(
            profile,
            candidates,
            normal,
            slack,
            available,
            |c| self.costs.prior(key, c.width, c.prior_seconds),
            key.batch,
        );
        let decision = decision.map(|choice| (choice.id, choice.exploring));
        if self.trace {
            eprintln!(
                "MTP_CHOOSE {key:?} decision={decision:?} normal={normal:.6} available={available:.6}"
            );
        }
        decision
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
