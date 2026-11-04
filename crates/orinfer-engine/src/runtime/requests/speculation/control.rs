//! Runtime decisions only: kernels and legal row geometries remain package-owned.
mod budget;
mod cost;
// Reprofiling is a real execution and may displace captured graphs. Acceptance
// changes are evaluated every round; only unused timing estimates age here.
const REFRESH_AFTER: usize = 512;
use budget::Budget;
use cost::{Cost, Costs};

#[derive(Clone, Copy, Debug, Default)]
pub(in crate::runtime) struct Acceptance {
    // Conditional acceptance at each actually tested prefix position. A rejected
    // prefix does not turn its untested suffix into independent rejections.
    probability: [f64; 7],
    samples: [usize; 7],
}
impl Acceptance {
    pub(in crate::runtime) fn observe(&mut self, proposed: usize, accepted: usize) {
        for j in 0..proposed.min(accepted.saturating_add(1)).min(7) {
            let value = f64::from(j < accepted);
            let prior = if self.samples[j] == 0 {
                0.7
            } else {
                self.probability[j]
            };
            self.probability[j] = prior * 0.8 + value * 0.2;
            self.samples[j] += 1;
        }
    }
    pub(in crate::runtime) fn expected(&self, drafts: usize) -> f64 {
        let mut prefix = 1.;
        let mut total = 1.;
        for j in 0..drafts.min(7) {
            // The neutral prior participates once, at EMA initialization.
            // Reapplying a sample-count weight can make a success lower the
            // estimate after an early rejection.
            prefix *= if self.samples[j] == 0 {
                0.7
            } else {
                self.probability[j]
            };
            total += prefix;
        }
        total
    }
}

/// Mean request TPOT is cost * mean(1 / expected yield). Express its inverse
/// as token-equivalent progress; throughput instead uses total expected yield.
pub(super) fn progress(yields: impl Iterator<Item = f64>, batch: usize, latency: bool) -> f64 {
    if latency {
        let inverse: f64 = yields.map(f64::recip).sum();
        let batch = batch as f64;
        batch * batch / inverse
    } else {
        yields.sum()
    }
}

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

/// A complete execution sample. Preparation is charged in full but excluded
/// from hot arithmetic; direct and replay execution share one timing estimate.
#[derive(Clone, Copy, Debug, Default)]
pub(in crate::runtime) struct Measurement {
    pub seconds: f64,
    pub preparation_s: f64,
    pub tokens: usize,
    /// The request draft state was caught up before this invocation.
    pub warm_ready: bool,
}
impl Measurement {
    fn is_steady(self) -> bool {
        self.warm_ready && self.preparation_s == 0.
    }
    fn valid(self) -> bool {
        self.seconds.is_finite()
            && self.seconds > 0.
            && self.preparation_s.is_finite()
            && self.preparation_s >= 0.
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
    /// Read from the selected owners' actual draft cursors for this round.
    pub warm_ready: bool,
    /// The selected owners advanced without fresh draft acceptance observations.
    pub review_acceptance: bool,
}

struct Decision {
    width: usize,
    exploring: bool,
}

/// Choose from legal plans using frozen costs and request-local expected yield.
/// The only policy states are measured samples and time since last measurement.
fn select(
    costs: &cost::Profile,
    candidates: &[Candidate],
    normal: f64,
    slack: f64,
    funds: (f64, f64),
    prior: impl Fn(&Candidate) -> f64,
    batch: usize,
) -> Option<Decision> {
    if slack.is_nan() || slack <= 0. {
        return None;
    }
    // Ordinary execution is the zero-speculation candidate. Keep its baseline
    // current as well; no special calibration or recovery state is needed.
    if costs
        .arms
        .get(&1)
        .is_some_and(|s| costs.calls.saturating_sub(s.last) >= REFRESH_AFTER)
    {
        return None;
    }
    let baseline = normal / batch.max(1) as f64;
    let mut best = (1, baseline * 0.98);
    let mut probe: Option<(Decision, (usize, usize))> = None;
    for candidate in candidates {
        if candidate.width < 2
            || !candidate.expected_tokens.is_finite()
            || candidate.expected_tokens < batch as f64
        {
            continue;
        }
        let shape = costs.arms.get(&candidate.width);
        let samples = shape.map_or(0, |s| s.time.samples());
        let available = if samples < 3 { funds.1 } else { funds.0 };
        let transition = shape
            .filter(|_| !candidate.warm_ready)
            .map_or(0., |s| s.transition_s);
        let hot = shape
            .filter(|s| s.time.samples() > 0)
            .map_or_else(|| prior(candidate), Cost::mean);
        // Re-entry may pay back over the same three-round window used for hot
        // calibration. The full transition still needs deadline and cash cover.
        let horizon = (candidate.remaining_tokens as f64 / candidate.expected_tokens).clamp(1., 3.);
        let mean = hot + transition / horizon;
        let upper = shape
            .filter(|s| s.time.samples() > 0)
            .map_or(hot, Cost::upper)
            + transition;
        if !upper.is_finite() || upper <= 0. || upper > slack {
            continue;
        }
        // Budget remaining output for one priming round as well as the hot
        // samples. The extra round is conservative after partial profiling.
        let missing = if samples < 3 { 4 - samples } else { 0 };
        if candidate.remaining_tokens
            < batch.saturating_mul(candidate.width.saturating_add(missing).saturating_sub(1))
        {
            continue;
        }
        let score = mean / candidate.expected_tokens;
        let review = candidate.review_acceptance
            && hot / (batch as f64 * candidate.width as f64) < baseline * 0.98;
        // Acceptance cannot refresh itself while ordinary decoding advances.
        // A review reserves the loss of only one guaranteed target per owner,
        // rather than financing a probe with the stale acceptance estimate.
        let reserved_tokens = if review {
            batch as f64
        } else {
            candidate.expected_tokens
        };
        let loss = (upper - baseline * reserved_tokens).max(0.);
        if samples >= 3 && score < best.1 && (candidate.warm_ready || loss <= available) {
            best = (candidate.width, score);
        }
        let stale = shape.is_some_and(|s| costs.calls.saturating_sub(s.last) >= REFRESH_AFTER);
        // Exploration spends marginal time relative to ordinary execution,
        // including while collecting hot samples. Reserving the full round
        // can strand a profitable shape between catch-up and its hot samples.
        // Unexpected rejection or preparation is settled as actual debt.
        let recoverable = if review {
            // A review can change acceptance, but not the measured arithmetic
            // cost. Bound its possible benefit by perfect acceptance.
            (baseline - hot / (batch as f64 * candidate.width as f64))
                * candidate.remaining_tokens as f64
                - transition
        } else {
            baseline
                * candidate.remaining_tokens as f64
                * (1. - batch as f64 / candidate.expected_tokens)
        };
        if (missing > 0 || stale || review) && loss <= available && loss <= recoverable {
            // Finish the smallest cold shape before trying a deeper shape.
            // Afterwards refresh oldest costs, without resetting their samples.
            let rank = (
                usize::from(missing == 0 && !review),
                if missing > 0 || review {
                    candidate.width
                } else {
                    shape.unwrap().last
                },
            );
            if probe.as_ref().is_none_or(|(_, priority)| rank < *priority) {
                probe = Some((
                    Decision {
                        width: candidate.width,
                        exploring: true,
                    },
                    rank,
                ));
            }
        }
    }
    probe.map(|(decision, _)| decision).or_else(|| {
        (best.0 > 1).then_some(Decision {
            width: best.0,
            exploring: false,
        })
    })
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
            .filter(|s| s.time.samples() >= 3)
            .map_or(0., |s| s.time.mean());
        let calibrating = width > 1
            && self
                .costs
                .profiles
                .get(&key)
                .and_then(|p| p.arms.get(&width))
                .is_none_or(|s| s.time.samples() < 3);
        self.budget.settle(
            width,
            measurement,
            normal,
            key.batch,
            self.calibration_limit,
            calibrating,
        );
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
        let available = self.budget.available(self.calibration_limit);
        let decision = select(
            profile,
            candidates,
            normal,
            slack,
            (self.budget.balance, available),
            |c| self.costs.prior(key, c.width, c.prior_seconds),
            key.batch,
        );
        let decision = decision.map(|choice| (choice.width, choice.exploring));
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
