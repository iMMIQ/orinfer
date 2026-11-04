//! Shared speculation decisions and actual-loss accounting. Backends own legal plans and execution resources.
mod budget;
use crate::{execution::MAX_MTP_DRAFTS, scheduler};
pub(crate) use budget::Budget;
pub(crate) const REFRESH_AFTER: usize = 512;
pub(crate) const HOT_SAMPLES: usize = 3;
pub(crate) const CALIBRATION_ROUNDS: usize = HOT_SAMPLES + 1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Acceptance {
    // Conditional acceptance at each actually tested prefix position. A rejected
    // prefix does not turn its untested suffix into independent rejections.
    probability: [f64; MAX_MTP_DRAFTS],
    samples: [usize; MAX_MTP_DRAFTS],
}
impl Default for Acceptance {
    fn default() -> Self {
        Self {
            probability: [0.7; MAX_MTP_DRAFTS],
            samples: [0; MAX_MTP_DRAFTS],
        }
    }
}
impl Acceptance {
    /// Reuse learned conditional priors without inventing request observations.
    pub(crate) fn prior(&self) -> Self {
        Self {
            probability: std::array::from_fn(|i| self.rate(i, 0.7)),
            samples: [0; MAX_MTP_DRAFTS],
        }
    }
    pub(crate) fn observe(&mut self, proposed: usize, accepted: usize) {
        self.observe_batch(&[(proposed, accepted)]);
    }
    pub(crate) fn observe_batch(&mut self, counts: &[(usize, usize)]) {
        for j in 0..MAX_MTP_DRAFTS {
            let attempts = counts.iter().filter(|&&(k, a)| k > j && a >= j).count();
            if attempts == 0 {
                continue;
            }
            let passed = counts.iter().filter(|&&(k, a)| k > j && a > j).count();
            let value = passed as f64 / attempts as f64;
            // A packet with N tested owners contains N real observations.
            // Aggregate without making the result depend on owner order.
            let decay = 0.8_f64.powf(attempts as f64);
            self.probability[j] = self.probability[j] * decay + value * (1. - decay);
            self.samples[j] += attempts;
        }
    }
    pub(crate) fn rate(&self, position: usize, prior: f64) -> f64 {
        if position < MAX_MTP_DRAFTS && self.samples[position] > 0 {
            self.probability[position]
        } else {
            prior
        }
    }
    #[cfg(test)]
    pub(crate) fn sample_counts(&self) -> &[usize] {
        &self.samples
    }
    pub(crate) fn estimate(&self, position: usize, prior: f64) -> f64 {
        self.rate(position, prior)
    }
    pub(crate) fn expected(&self, drafts: usize) -> f64 {
        let mut prefix = 1.;
        let mut total = 1.;
        for j in 0..drafts.min(MAX_MTP_DRAFTS) {
            // The neutral prior enters once, at conditional EMA initialization.
            prefix *= self.estimate(j, 0.7);
            total += prefix;
        }
        total
    }
}

/// Harmonic progress is the inverse of mean request TPOT. Throughput uses the total.
pub(crate) fn progress(yields: impl Iterator<Item = f64>, batch: usize, latency: bool) -> f64 {
    if latency {
        let inverse: f64 = yields.map(f64::recip).sum();
        (batch as f64) * (batch as f64) / inverse
    } else {
        yields.sum()
    }
}

/// A complete execution sample. Preparation is charged in full but excluded
/// from hot arithmetic; direct and replay execution share one timing estimate.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Measurement {
    pub seconds: f64,
    pub preparation_s: f64,
    pub tokens: usize,
    /// The request draft state was caught up before this invocation.
    pub warm_ready: bool,
}
impl Measurement {
    pub(crate) fn is_steady(self) -> bool {
        self.warm_ready && self.preparation_s == 0.
    }
    pub(crate) fn valid(self) -> bool {
        self.seconds.is_finite()
            && self.seconds > 0.
            && self.preparation_s.is_finite()
            && self.preparation_s >= 0.
    }
}

#[derive(Default)]
pub(crate) struct Timing {
    time: scheduler::Estimate,
    trials: usize,
}
impl Timing {
    pub fn observations(&self) -> usize {
        self.trials
    }
    pub fn samples(&self) -> usize {
        if self.trials <= 1 {
            0
        } else {
            self.time.samples
        }
    }
    pub fn mean(&self) -> f64 {
        self.time.mean
    }
    pub fn upper(&self) -> f64 {
        self.time.upper()
    }
    pub fn observe(&mut self, seconds: f64) {
        if self.trials == 1 {
            // The first use may capture a graph or load a lazy CUDA module.
            self.time = Default::default();
        }
        self.time.observe(seconds);
        self.trials += 1;
    }
}

#[derive(Default)]
pub(crate) struct Cost {
    pub time: Timing,
    overhead: f64,
    /// Extra cost when ordinary decode left the owner's draft state behind.
    pub transition_s: f64,
    pub last: usize,
}
impl Cost {
    pub fn mean(&self) -> f64 {
        self.time.mean() + self.overhead
    }
    pub fn upper(&self) -> f64 {
        self.time.upper() + self.overhead
    }
    pub(crate) fn observe(&mut self, measurement: Measurement, call: usize) {
        self.last = call;
        let extra = if measurement.warm_ready
            && measurement.preparation_s > 0.
            && self.time.samples() > 0
        {
            (measurement.seconds - self.time.mean()).max(0.)
        } else {
            0.
        };
        // Draft catch-up is already priced once by transition_s. Only resource
        // preparation of an otherwise warm invocation belongs in steady cost.
        self.overhead = self.overhead * 0.8 + extra * 0.2;
        if !measurement.warm_ready {
            // Before any hot sample, the observed round is a conservative
            // upper bound on its transition. It must still price the next
            // cold attempt; consecutive warm calibration pays no such cost.
            let hot = if self.time.samples() > 0 {
                self.time.mean()
            } else {
                0.
            };
            self.transition_s = self.transition_s * 0.8 + (measurement.seconds - hot).max(0.) * 0.2;
        }
        if measurement.is_steady() && measurement.tokens > 0 {
            self.time.observe(measurement.seconds);
        }
    }
}

/// Backend supplies legal physical plans and measured prices; readiness is not inferred from cost.
pub(crate) struct Candidate {
    pub id: usize,
    pub width: usize,
    pub expected_tokens: f64,
    pub maximum_tokens: f64,
    pub remaining_tokens: usize,
    pub mean_s: f64,
    pub upper_s: f64,
    pub transition_s: f64,
    pub warm_ready: bool,
    pub review_acceptance: bool,
    pub samples: usize,
    pub last: Option<usize>,
}
pub(crate) struct Selection {
    pub normal_s: f64,
    pub batch: usize,
    pub slack_s: f64,
    pub available_s: f64,
    pub calibration_s: f64,
    pub calls: usize,
    pub normal_last: Option<usize>,
}
pub(crate) struct Decision {
    pub id: usize,
    pub exploring: bool,
}
/// Pure scoring. A decision neither earns credit nor spends the startup pool.
pub(crate) fn select(input: Selection, candidates: &[Candidate]) -> Option<Decision> {
    let Selection {
        normal_s: normal,
        batch,
        slack_s: slack,
        available_s: earned,
        calibration_s: calibration,
        calls,
        normal_last,
    } = input;
    if batch == 0
        || !normal.is_finite()
        || normal <= 0.
        || slack.is_nan()
        || slack <= 0.
        || earned.is_nan()
        || calibration.is_nan()
        || calibration == f64::NEG_INFINITY
        || normal_last.is_some_and(|last| calls.saturating_sub(last) >= REFRESH_AFTER)
    {
        return None;
    }
    let baseline = normal / batch as f64;
    let mut best = (usize::MAX, baseline * 0.98);
    let mut probe: Option<(Decision, (usize, usize))> = None;
    for c in candidates {
        if c.width < 2
            || !c.expected_tokens.is_finite()
            || c.expected_tokens < batch as f64
            || !c.maximum_tokens.is_finite()
            || c.maximum_tokens < batch as f64
        {
            continue;
        }
        let available = if c.samples < HOT_SAMPLES {
            calibration
        } else {
            earned
        };
        let horizon = (c.remaining_tokens as f64 / c.expected_tokens).clamp(1., 3.);
        let hot = c.mean_s;
        let mean = hot + c.transition_s / horizon;
        let upper = c.upper_s + c.transition_s;
        if !mean.is_finite() || mean <= 0. || !upper.is_finite() || upper <= 0. || upper > slack {
            continue;
        }
        let missing = if c.samples < HOT_SAMPLES {
            CALIBRATION_ROUNDS - c.samples
        } else {
            0
        };
        if c.remaining_tokens
            < batch.saturating_mul(c.width.saturating_add(missing).saturating_sub(1))
        {
            continue;
        }
        let score = mean / c.expected_tokens;
        let review = c.review_acceptance && hot / c.maximum_tokens < baseline * 0.98;
        let reserved_tokens = if review {
            batch as f64
        } else {
            c.expected_tokens
        };
        let loss = (upper - baseline * reserved_tokens).max(0.);
        if c.samples >= HOT_SAMPLES && score < best.1 && (c.warm_ready || loss <= available) {
            best = (c.id, score);
        }
        let stale = c
            .last
            .is_some_and(|last| calls.saturating_sub(last) >= REFRESH_AFTER);
        let recoverable = if review {
            (baseline - hot / c.maximum_tokens) * c.remaining_tokens as f64 - c.transition_s
        } else {
            baseline * c.remaining_tokens as f64 * (1. - batch as f64 / c.expected_tokens)
        };
        if (missing > 0 || stale || review) && loss <= available && loss <= recoverable {
            let rank = (
                usize::from(missing == 0 && !review),
                if missing > 0 || review {
                    c.width
                } else {
                    c.last.unwrap()
                },
            );
            if probe.as_ref().is_none_or(|(_, old)| rank < *old) {
                probe = Some((
                    Decision {
                        id: c.id,
                        exploring: true,
                    },
                    rank,
                ));
            }
        }
    }
    probe.map(|(d, _)| d).or_else(|| {
        (best.0 != usize::MAX).then_some(Decision {
            id: best.0,
            exploring: false,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_acceptance_weights_real_attempts_and_keeps_owner_order_independent() {
        let mut a = Acceptance::default();
        a.observe_batch(&[(5, 5); 16]);
        assert!((a.rate(0, 0.7) - (1. - 0.3 * 0.8_f64.powi(16))).abs() < 1e-12);
        assert_eq!(a.sample_counts()[0], 16);
        let mut b = Acceptance::default();
        b.observe_batch(&[(5, 0), (5, 5)]);
        let mut c = Acceptance::default();
        c.observe_batch(&[(5, 5), (5, 0)]);
        assert_eq!(b.probability, c.probability);
        let prior = a.prior();
        assert_eq!(prior.sample_counts(), &[0; MAX_MTP_DRAFTS]);
        let mut fresh = prior;
        fresh.observe(5, 5);
        assert!(fresh.rate(0, 0.7) > a.rate(0, 0.7));
    }
    #[test]
    fn parallel_capacity_does_not_inflate_an_adopted_prefix_review() {
        let input = || Selection {
            normal_s: 0.02,
            batch: 1,
            slack_s: 0.4,
            available_s: 0.1,
            calibration_s: 0.1,
            calls: 1,
            normal_last: Some(1),
        };
        let mut c = Candidate {
            id: 0,
            width: 6,
            expected_tokens: 1.,
            maximum_tokens: 2.,
            remaining_tokens: 512,
            mean_s: 0.05,
            upper_s: 0.05,
            transition_s: 0.,
            warm_ready: true,
            review_acceptance: true,
            samples: 3,
            last: Some(0),
        };
        assert!(select(input(), std::slice::from_ref(&c)).is_none());
        c.maximum_tokens = 6.;
        assert!(select(input(), std::slice::from_ref(&c)).unwrap().exploring);
    }
    #[test]
    fn deep_acceptance_keeps_untested_suffixes_out_of_statistics() {
        let mut a = Acceptance::default();
        a.observe(16, 9);
        assert_eq!(&a.sample_counts()[..10], &[1; 10]);
        assert_eq!(&a.sample_counts()[10..], &[0; 21]);
        assert!((a.rate(9, 0.7) - 0.56).abs() < 1e-12);
        assert_eq!(a.rate(10, 0.7), 0.7);
        a.observe_batch(&[(16, 9), (16, 16)]);
        assert!((a.rate(10, 0.7) - 0.76).abs() < 1e-12);
        assert_eq!(a.sample_counts()[10], 1);
        assert!(a.expected(16) > a.expected(7));
    }
}
