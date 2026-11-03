//! Runtime decisions only: kernels and legal row geometries remain package-owned.
use crate::scheduler::{Estimate, ExecutionMode, ExecutionObservation};
use std::collections::BTreeMap;

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
            self.probability[j] = if self.samples[j] == 0 {
                value
            } else {
                self.probability[j] * 0.8 + value * 0.2
            };
            self.samples[j] += 1;
        }
    }
    pub(in crate::runtime) fn expected(&self, drafts: usize) -> f64 {
        let mut prefix = 1.;
        let mut total = 1.;
        for j in 0..drafts.min(7) {
            // Shrink sparse request observations toward a neutral model prior.
            let confidence = self.samples[j] as f64 / (self.samples[j] as f64 + 4.);
            prefix *= self.probability[j] * confidence + 0.7 * (1. - confidence);
            total += prefix;
        }
        total
    }
}

/// Spend a fixed verification-row budget on request prefixes. The estimates
/// come exclusively from earlier committed rounds, never the current drafts.
/// Every increment is a legal package width, including packages with holes.
pub(in crate::runtime) fn allocate(
    choices: &[Vec<(usize, f64)>],
    budget: usize,
) -> Option<Vec<usize>> {
    let mut indices = vec![0; choices.len()];
    let mut used = choices
        .iter()
        .try_fold(0usize, |n, c| n.checked_add(c.first()?.0))?;
    if used > budget {
        return None;
    }
    while used < budget {
        let next = choices
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let current = c[indices[i]];
                let next = *c.get(indices[i] + 1)?;
                let rows = next.0.checked_sub(current.0)?;
                (rows > 0 && rows <= budget - used).then_some((
                    i,
                    rows,
                    (next.1 - current.1) / rows as f64,
                ))
            })
            // Lower request index wins ties; comparisons use frozen estimates.
            .max_by(|a, b| a.2.total_cmp(&b.2).then_with(|| b.0.cmp(&a.0)))?;
        used += next.1;
        indices[next.0] += 1;
    }
    Some(choices.iter().zip(indices).map(|(c, i)| c[i].0).collect())
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

#[derive(Clone, Copy, Default)]
struct Performance {
    time: Estimate,
    tokens: f64,
    generation: usize,
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
enum State {
    #[default]
    Unseen,
    Calibrating,
    Ready,
    Cooldown,
}

#[derive(Default)]
struct Arm {
    modes: [Performance; 3],
    mode: ExecutionMode,
    preparation: Estimate,
    prior: f64,
    state: State,
    last_trial: usize,
}

impl Arm {
    fn hot(&self, generation: usize) -> Performance {
        let p = self.modes[self.mode as usize];
        if p.generation == generation {
            p
        } else {
            Performance::default()
        }
    }
}

#[derive(Default)]
struct Cohort {
    arms: BTreeMap<usize, Arm>,
    calls: usize,
    active: usize,
    calibrating: Option<usize>,
}
#[derive(Clone, Copy, Default, PartialEq)]
enum SinglePolicy {
    #[default]
    Auto,
    Explore,
    ExploreLocal,
    Fixed(usize),
}

pub(in crate::runtime) struct Controller {
    single_policy: SinglePolicy,
    cohorts: BTreeMap<Key, Cohort>,
    // Funding survives context changes. Separate family balances prevent a
    // profitable singleton from subsidizing repeated unprofitable batch probes.
    balances: BTreeMap<(usize, bool, usize), f64>,
    credit: f64,
    credit_cap: f64,
    calibration_limit: f64,
    bootstrap_used: bool,
    loan: Option<(Key, usize)>,
    generation: usize,
    #[cfg(test)]
    pub(in crate::runtime) fixed_width: Option<usize>,
}
impl Default for Controller {
    fn default() -> Self {
        let single_policy = match std::env::var("ORINFER_SINGLE_MTP_POLICY").as_deref() {
            Ok("explore") => SinglePolicy::Explore,
            Ok("explore_local") => SinglePolicy::ExploreLocal,
            Ok("fixed0") => SinglePolicy::Fixed(1),
            Ok("fixed1") => SinglePolicy::Fixed(2),
            Ok("fixed2") => SinglePolicy::Fixed(3),
            Ok("fixed3") => SinglePolicy::Fixed(4),
            Ok("auto") | Err(_) => SinglePolicy::Auto,
            Ok(other) => panic!("Unsupported ORINFER_SINGLE_MTP_POLICY: {other}"),
        };
        Self {
            cohorts: Default::default(),
            balances: Default::default(),
            credit: 0.,
            credit_cap: 0.4,
            calibration_limit: 0.4,
            bootstrap_used: false,
            loan: None,
            generation: 0,
            single_policy,
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
impl Controller {
    pub(in crate::runtime) fn set_calibration_limit(&mut self, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            self.calibration_limit = seconds;
        }
    }

    fn retain_key(&mut self, key: Key) {
        if self.cohorts.len() >= 256 && !self.cohorts.contains_key(&key) {
            let oldest = self.cohorts.keys().next().copied().unwrap();
            self.cohorts.remove(&oldest);
        }
    }
    pub(in crate::runtime) fn observe(
        &mut self,
        key: Key,
        width: usize,
        seconds: f64,
        tokens: usize,
        observation: ExecutionObservation,
    ) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        self.retain_key(key);
        self.generation = observation.generation;
        let normal = self
            .cohorts
            .get(&key)
            .and_then(|c| c.arms.get(&1))
            .map(|a| a.hot(self.generation).time.mean / key.batch.max(1) as f64)
            .unwrap_or(0.);
        let change = if width == 1 && tokens > 0 {
            seconds * 0.02
        } else {
            normal * tokens as f64 - seconds
        };
        // Preparation and zero-yield attempts are real costs even though they
        // cannot train a steady throughput estimate.
        self.credit = (self.credit + change).min(self.credit_cap);
        if self.balances.len() >= 256 && !self.balances.contains_key(&key.family()) {
            let victim = *self.balances.keys().next().unwrap();
            self.balances.remove(&victim);
        }
        let balance = self.balances.entry(key.family()).or_default();
        *balance = (*balance + change).min(self.credit_cap);
        let cohort = self.cohorts.entry(key).or_default();
        cohort.calls += 1;
        let arm = cohort.arms.entry(width).or_default();
        arm.mode = observation.mode;
        arm.last_trial = cohort.calls;
        if observation.preparing || observation.evicted {
            arm.modes = [Performance {
                generation: self.generation,
                ..Default::default()
            }; 3];
        }
        let p = &mut arm.modes[arm.mode as usize];
        if p.generation != self.generation {
            *p = Performance {
                generation: self.generation,
                ..Default::default()
            };
        }
        if observation.preparing || observation.evicted {
            arm.preparation.observe((seconds - arm.prior).max(1e-6));
            return;
        }
        if tokens == 0 {
            return;
        }
        p.time.observe(seconds);
        p.tokens = if p.time.samples == 1 {
            tokens as f64
        } else {
            p.tokens * 0.8 + tokens as f64 * 0.2
        };
        arm.prior = p.time.upper();
        arm.state = if p.time.samples >= 3 {
            State::Ready
        } else {
            State::Calibrating
        };
        if width > 1 && p.time.samples >= 3 {
            if cohort.calibrating == Some(width) {
                cohort.calibrating = None;
            }
            if self.loan == Some((key, width)) {
                self.loan = None;
            }
        }
    }

    fn prior(&self, key: Key, width: usize, fallback: f64) -> f64 {
        self.cohorts
            .iter()
            .filter(|(k, _)| k.family() == key.family() && k.context.abs_diff(key.context) <= 1)
            .filter_map(|(k, c)| {
                let arm = c.arms.get(&width)?;
                (arm.prior > 0.).then_some((k.context.abs_diff(key.context), arm.prior))
            })
            .min_by_key(|(distance, _)| *distance)
            .map_or(fallback, |(_, seconds)| seconds)
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
        let single = key.batch == 1 && key.prefill == 0;
        if single && let SinglePolicy::Fixed(width) = self.single_policy {
            return candidates
                .iter()
                .filter(|c| c.width <= width && c.prior_seconds <= slack)
                .max_by_key(|c| c.width)
                .map(|c| (c.width, false));
        }
        let local_acceptance = single && self.single_policy == SinglePolicy::ExploreLocal;
        let priors: BTreeMap<_, _> = candidates
            .iter()
            .map(|c| (c.width, self.prior(key, c.width, c.prior_seconds)))
            .collect();
        self.credit_cap = candidates
            .iter()
            .map(|c| c.prior_seconds * 4.)
            .filter(|s| s.is_finite())
            .fold(self.calibration_limit, f64::max);
        let funds = self
            .credit
            .min(*self.balances.get(&key.family()).unwrap_or(&0.));
        self.retain_key(key);
        let cohort = self.cohorts.entry(key).or_default();
        let normal = cohort.arms.get(&1)?.hot(self.generation);
        if normal.time.samples < 3 {
            return None;
        }
        let normal_seconds = normal
            .time
            .mean
            .max(normal_seconds.min(normal.time.upper()));
        let baseline = normal_seconds / key.batch.max(1) as f64;
        if cohort
            .calibrating
            .is_some_and(|w| !candidates.iter().any(|c| c.width == w))
        {
            cohort.calibrating = None;
            if self.loan.is_some_and(|(k, _)| k == key) {
                self.loan = None;
            }
        }
        let mut best = None;
        let mut probe = None;
        for candidate in candidates {
            if candidate.width < 2
                || !candidate.expected_tokens.is_finite()
                || candidate.expected_tokens <= 0.
            {
                continue;
            }
            let arm = cohort.arms.get(&candidate.width);
            let p = arm.map(|a| a.hot(self.generation)).unwrap_or_default();
            let seconds = if p.time.samples > 0 {
                p.time.upper()
            } else {
                priors[&candidate.width]
            };
            if !seconds.is_finite() || seconds <= 0. || seconds > slack {
                continue;
            }
            let samples = p.time.samples;
            let score = if samples > 0 {
                // Request-local acceptance already includes sparse-sample shrinkage.
                // A previous prompt's acceptance must not cap this prompt's yield.
                let expected = if local_acceptance {
                    candidate.expected_tokens
                } else {
                    candidate.expected_tokens.min(p.tokens)
                };
                p.time.mean / expected.max(1.)
            } else {
                f64::INFINITY
            };
            let stale = arm.is_some_and(|a| cohort.calls.saturating_sub(a.last_trial) >= 32);
            let pending = cohort.calibrating == Some(candidate.width);
            let calibrate = samples < 3 || (stale && score >= baseline);
            // A deferred graph can still need capture on its next invocation.
            // Keep funding preparation until an actual hot sample arrives.
            let setup = if samples == 0 {
                arm.filter(|a| a.preparation.samples > 0)
                    .map_or(seconds, |a| a.preparation.upper())
            } else {
                0.
            };
            let missing = if samples >= 3 { 3 } else { 3 - samples };
            let required =
                setup + missing as f64 * (seconds - baseline * candidate.expected_tokens).max(0.);
            // Calibration needs enough legal invocations before the shortest
            // owner finishes. Even a zero-overhead draft cannot repay setup
            // beyond this optimistic bound on the remaining batch's savings.
            let enough_work = candidate.remaining_tokens
                >= key.batch.saturating_mul(candidate.width + missing - 1);
            let recoverable = baseline
                * candidate.remaining_tokens as f64
                * (1. - key.batch as f64 / candidate.expected_tokens).max(0.);
            let loan = self
                .loan
                .is_some_and(|(k, w)| k.family() == key.family() && w == candidate.width);
            let initial = single && !key.shared_work && !self.bootstrap_used;
            let available = funds
                + if loan || initial {
                    self.calibration_limit
                } else {
                    0.
                };
            if calibrate
                && enough_work
                && (single || required <= recoverable)
                && (pending || cohort.calibrating.is_none())
                && available >= required
                && probe.is_none()
            {
                probe = Some((candidate.width, initial));
            }
            let margin = if samples > 0 {
                ((p.time.upper() - p.time.mean)
                    / p.time.mean.max(1e-9)
                    / (samples.max(1) as f64).sqrt())
                .clamp(0.01, 0.1)
            } else {
                0.1
            };
            let threshold = if cohort.active == candidate.width {
                1.
            } else {
                1. - margin
            };
            if samples >= 3
                && score < baseline * threshold
                && best.is_none_or(|(_, cost)| score < cost)
            {
                best = Some((candidate.width, score));
            }
        }
        if let Some((width, initial)) = probe {
            let arm = cohort.arms.entry(width).or_default();
            if arm.state == State::Cooldown {
                arm.modes[arm.mode as usize] = Performance {
                    generation: self.generation,
                    ..Default::default()
                };
            }
            arm.state = State::Calibrating;
            cohort.calibrating = Some(width);
            if initial {
                self.bootstrap_used = true;
                self.loan = Some((key, width));
            } else if self
                .loan
                .is_some_and(|(k, w)| k.family() == key.family() && w == width)
            {
                self.loan = Some((key, width));
            }
            return Some((width, true));
        }
        for (&width, arm) in &mut cohort.arms {
            if width > 1
                && arm.hot(self.generation).time.samples >= 3
                && best.is_none_or(|(w, _)| w != width)
            {
                arm.state = State::Cooldown;
            }
        }
        cohort.active = best.map_or(0, |(width, _)| width);
        best.map(|(width, _)| (width, false))
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
