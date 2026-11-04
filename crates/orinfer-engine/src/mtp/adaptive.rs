//! Shared speculation decisions and actual-loss accounting. Backends own legal plans and execution resources.
mod budget;
use crate::{execution::MAX_MTP_DRAFTS, scheduler};
pub(crate) use budget::Budget;
pub(crate) const REFRESH_AFTER: usize = 512;
pub(crate) const HOT_SAMPLES: usize = 3;
pub(crate) const CALIBRATION_ROUNDS: usize = HOT_SAMPLES + 1;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Acceptance {
    // Conditional acceptance at each actually tested prefix position. A rejected
    // prefix does not turn its untested suffix into independent rejections.
    probability: [f64; MAX_MTP_DRAFTS],
    samples: [usize; MAX_MTP_DRAFTS],
}
impl Acceptance {
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
            self.probability[j] = if self.samples[j] == 0 {
                value
            } else {
                self.probability[j] * 0.8 + value * 0.2
            };
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
        if position >= MAX_MTP_DRAFTS {
            return prior;
        }
        let weight = self.samples[position] as f64 / (self.samples[position] as f64 + 4.);
        self.probability[position] * weight + prior * (1. - weight)
    }
    pub(crate) fn expected(&self, drafts: usize) -> f64 {
        let mut prefix = 1.;
        let mut total = 1.;
        for j in 0..drafts.min(MAX_MTP_DRAFTS) {
            // Shrink sparse request observations toward a neutral model prior.
            prefix *= self.estimate(j, 0.7);
            total += prefix;
        }
        total
    }
}

/// Latency protects the slowest owner's expected token rate. Throughput uses
/// total expected commits. Both return token-equivalent progress for the packet;
/// the latency value is a conservative lower bound on its total expected yield.
pub(crate) fn progress(yields: impl Iterator<Item = f64>, batch: usize, latency: bool) -> f64 {
    let (total, minimum) = yields.fold((0., f64::INFINITY), |(total, minimum), value| {
        (total + value, minimum.min(value))
    });
    if latency {
        minimum * batch as f64
    } else {
        total
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
        let extra = if !measurement.is_steady() && self.time.samples() > 0 {
            (measurement.seconds - self.time.mean()).max(0.)
        } else {
            0.
        };
        self.overhead = (31. * self.overhead + extra) / 32.;
        if measurement.is_steady() && measurement.tokens > 0 {
            self.time.observe(measurement.seconds);
        }
    }
}

/// Backend-supplied estimate of one legal physical plan.
pub(crate) struct Candidate {
    pub id: usize,
    pub width: usize,
    pub expected_tokens: f64,
    pub remaining_tokens: usize,
    pub mean_s: f64,
    pub upper_s: f64,
    /// A backend with separately measured arithmetic can reserve its expected
    /// hot loss plus the full pending preparation, without inventing samples.
    pub probe_reserve_s: Option<f64>,
    pub samples: usize,
    pub last: Option<usize>,
}
pub(crate) struct Selection {
    pub normal_s: f64,
    pub batch: usize,
    pub slack_s: f64,
    pub available_s: f64,
    pub calls: usize,
    pub normal_last: Option<usize>,
}
pub(crate) struct Decision {
    pub id: usize,
    pub exploring: bool,
}
/// Score frozen legal plans; choosing never spends or creates budget.
pub(crate) fn select(input: Selection, candidates: &[Candidate]) -> Option<Decision> {
    let Selection {
        normal_s: normal,
        batch,
        slack_s: slack,
        available_s: available,
        calls,
        normal_last,
    } = input;
    if batch == 0
        || !normal.is_finite()
        || normal <= 0.
        || slack.is_nan()
        || slack <= 0.
        || available.is_nan()
        || available == f64::NEG_INFINITY
        || normal_last.is_some_and(|last| calls.saturating_sub(last) >= REFRESH_AFTER)
    {
        return None;
    }
    let baseline = normal / batch as f64;
    let mut best = (usize::MAX, baseline * 0.98);
    let mut probe: Option<(Decision, (usize, usize))> = None;
    for candidate in candidates {
        if candidate.width < 2
            || !candidate.expected_tokens.is_finite()
            || candidate.expected_tokens <= batch as f64
        {
            continue;
        }
        let (samples, mean, upper) = (candidate.samples, candidate.mean_s, candidate.upper_s);
        if !mean.is_finite() || mean <= 0. || !upper.is_finite() || upper <= 0. || upper > slack {
            continue;
        }
        let missing = if samples < HOT_SAMPLES {
            CALIBRATION_ROUNDS - samples
        } else {
            0
        };
        if candidate.remaining_tokens
            < batch.saturating_mul(candidate.width.saturating_add(missing).saturating_sub(1))
        {
            continue;
        }
        let score = mean / candidate.expected_tokens;
        if samples >= HOT_SAMPLES && score < best.1 {
            best = (candidate.id, score);
        }
        let stale = candidate
            .last
            .is_some_and(|last| calls.saturating_sub(last) >= REFRESH_AFTER);
        let loss = (upper - baseline * candidate.expected_tokens).max(0.);
        let required = if missing > 0 {
            candidate.probe_reserve_s.unwrap_or(upper)
        } else {
            loss
        };
        if !required.is_finite() || required < 0. || required > upper {
            continue;
        }
        let recoverable = baseline
            * candidate.remaining_tokens as f64
            * (1. - batch as f64 / candidate.expected_tokens);
        if (missing > 0 || stale) && required <= available.max(0.) && required <= recoverable {
            let rank = (
                usize::from(missing == 0),
                if missing > 0 {
                    candidate.width
                } else {
                    candidate.last.unwrap()
                },
            );
            if probe.as_ref().is_none_or(|(_, priority)| rank < *priority) {
                probe = Some((
                    Decision {
                        id: candidate.id,
                        exploring: true,
                    },
                    rank,
                ));
            }
        }
    }
    probe.map(|(decision, _)| decision).or_else(|| {
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
    fn full_account_table_does_not_allow_unmetered_trained_plans() {
        let selected = select(
            Selection {
                normal_s: 0.1,
                batch: 1,
                slack_s: 1.,
                available_s: f64::NEG_INFINITY,
                calls: 10,
                normal_last: Some(10),
            },
            &[Candidate {
                id: 0,
                width: 6,
                expected_tokens: 6.,
                remaining_tokens: 1024,
                mean_s: 0.1,
                upper_s: 0.1,
                probe_reserve_s: None,
                samples: 3,
                last: Some(10),
            }],
        );
        assert!(selected.is_none());
    }
    #[test]
    fn measured_zero_loss_calibration_can_repay_debt_without_funding_risky_shapes() {
        let input = || Selection {
            normal_s: 0.02,
            batch: 1,
            slack_s: 0.4,
            available_s: -0.07,
            calls: 4,
            normal_last: Some(1),
        };
        let mut candidate = Candidate {
            id: 0,
            width: 6,
            expected_tokens: 6.,
            remaining_tokens: 512,
            mean_s: 0.06,
            upper_s: 0.08,
            probe_reserve_s: Some(0.),
            samples: 1,
            last: Some(4),
        };
        assert!(
            select(input(), std::slice::from_ref(&candidate))
                .unwrap()
                .exploring
        );
        candidate.probe_reserve_s = Some(0.001);
        assert!(select(input(), std::slice::from_ref(&candidate)).is_none());
        candidate.probe_reserve_s = None;
        assert!(select(input(), std::slice::from_ref(&candidate)).is_none());
    }
    #[test]
    fn separately_measured_arithmetic_reserves_full_preparation_without_promoting_samples() {
        let input = || Selection {
            normal_s: 0.07,
            batch: 16,
            slack_s: 1.2,
            available_s: 0.41,
            calls: 3,
            normal_last: Some(1),
        };
        let mut candidate = Candidate {
            id: 0,
            width: 6,
            expected_tokens: 96.,
            remaining_tokens: 8192,
            mean_s: 0.47,
            upper_s: 0.81,
            probe_reserve_s: None,
            samples: 2,
            last: Some(3),
        };
        assert!(select(input(), std::slice::from_ref(&candidate)).is_none());
        candidate.probe_reserve_s = Some(0.4);
        assert!(
            select(input(), std::slice::from_ref(&candidate))
                .unwrap()
                .exploring
        );
        assert_eq!(candidate.samples, 2);
        candidate.probe_reserve_s = Some(f64::NAN);
        assert!(select(input(), std::slice::from_ref(&candidate)).is_none());
    }
    #[test]
    fn deep_acceptance_keeps_untested_suffixes_out_of_statistics() {
        let mut a = Acceptance::default();
        a.observe(16, 9);
        assert_eq!(&a.sample_counts()[..10], &[1; 10]);
        assert_eq!(&a.sample_counts()[10..], &[0; 21]);
        assert_eq!(a.rate(9, 0.7), 0.);
        assert_eq!(a.rate(10, 0.7), 0.7);
        a.observe_batch(&[(16, 9), (16, 16)]);
        assert_eq!(a.rate(10, 0.7), 1.);
        assert_eq!(a.sample_counts()[10], 1);
        assert!(a.expected(16) > a.expected(7));
    }
}
