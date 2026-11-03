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
    needs_setup: bool,
    recapture_cost: f64,
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
    recovering: bool,
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
    batch_allocated: f64,
    loan: Option<(Key, usize)>,
    generation: usize,
    trace: bool,
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
            batch_allocated: 0.,
            loan: None,
            generation: 0,
            trace: std::env::var("ORINFER_MTP_TRACE").as_deref() == Ok("1"),
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

    /// Train the common draft-cost fallback from actual hot packet execution.
    /// Acceptance changes committed yield, not the cost per added draft row.
    pub(in crate::runtime) fn extra_seconds(
        &self,
        key: Key,
        width: usize,
        seconds: f64,
        observation: ExecutionObservation,
    ) -> Option<f64> {
        if width <= 1 || observation.preparing || !seconds.is_finite() || seconds <= 0. {
            return None;
        }
        let normal = self
            .cohorts
            .get(&key)?
            .arms
            .get(&1)?
            .hot(observation.generation);
        (normal.time.samples >= 3)
            .then(|| (seconds - normal.time.mean).max(0.) / (width - 1) as f64)
    }

    fn retain_key(&mut self, key: Key) {
        if self.cohorts.len() >= 256 && !self.cohorts.contains_key(&key) {
            let oldest = self.cohorts.keys().next().copied().unwrap();
            self.cohorts.remove(&oldest);
        }
        if self.balances.len() >= 256 && !self.balances.contains_key(&key.family()) {
            let victim = *self.balances.keys().next().unwrap();
            self.balances.remove(&victim);
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
        let previous_loan_funds = self.loan.map(|(k, _)| {
            self.credit
                .min(*self.balances.get(&k.family()).unwrap_or(&0.))
        });
        // Preparation and zero-yield attempts are real costs even though they
        // cannot train a steady throughput estimate.
        self.credit = (self.credit + change).min(self.credit_cap);
        let balance = self.balances.entry(key.family()).or_default();
        *balance = (*balance + change).min(self.credit_cap);
        let family_funds = *balance;
        if previous_loan_funds.is_some_and(|funds| funds < 0.)
            && self.loan.is_some_and(|(k, _)| {
                self.credit
                    .min(*self.balances.get(&k.family()).unwrap_or(&0.))
                    >= 0.
            })
        {
            self.loan = None;
        }
        if self.trace {
            if observation.evicted {
                eprintln!("MTP_CACHE_EVICT {key:?} width={width} victim=unknown");
            }
            eprintln!(
                "MTP_OBSERVE {key:?} width={width} seconds={seconds:.6} tokens={tokens} {observation:?} normal={normal:.6} credit={:.6} family={family_funds:.6} loan={:?}",
                self.credit, self.loan
            );
        }
        let cohort = self.cohorts.entry(key).or_default();
        cohort.calls += 1;
        let arm = cohort.arms.entry(width).or_default();
        arm.recapture_cost = arm
            .modes
            .iter()
            .filter(|p| p.generation == self.generation && p.time.samples >= 3)
            .map(|p| p.time.mean)
            .min_by(f64::total_cmp)
            .map_or(0., |hot| {
                arm.recapture_cost * (31. / 32.)
                    + if observation.preparing {
                        (seconds - hot).max(0.) / 32.
                    } else {
                        0.
                    }
            });
        arm.mode = observation.mode;
        arm.last_trial = cohort.calls;
        arm.needs_setup = observation.deferred;
        // Capture and unrelated eviction do not change the hot arithmetic of
        // this shape. Keep its measured modes; an explicit invalidation below
        // still discards samples from an obsolete generation.
        let p = &mut arm.modes[arm.mode as usize];
        if p.generation != self.generation {
            *p = Performance {
                generation: self.generation,
                ..Default::default()
            };
        }
        if observation.preparing {
            arm.preparation.observe((seconds - arm.prior).max(1e-6));
            // An eviction counter does not identify the victim. Another arm's
            // actual recapture is charged when it executes, without inventing
            // a cohort-wide delay from an unrelated graph's eviction.
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
            // A profitable arm may cross a context bucket before its cold
            // preparation is repaid. Keep the same bounded loan for its next
            // calibration; readiness alone does not repay that debt.
            if self.loan == Some((key, width)) && self.credit.min(family_funds) >= 0. {
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
        let priors: BTreeMap<_, _> = candidates
            .iter()
            .map(|c| (c.width, self.prior(key, c.width, c.prior_seconds)))
            .collect();
        self.credit_cap = candidates
            .iter()
            .map(|c| c.prior_seconds * 4.)
            .filter(|s| s.is_finite())
            .fold(self.calibration_limit, f64::max);
        let family_funds = *self.balances.get(&key.family()).unwrap_or(&0.);
        let funds = self.credit.min(family_funds);
        // Allocate only already-earned model credit, with one lifetime cap
        // across every batch/context family. An indebted family cannot ask for
        // another allocation. Execution still debits both ledgers in full.
        self.retain_key(key);
        let cohort = self.cohorts.entry(key).or_default();
        let normal = cohort.arms.get(&1)?.hot(self.generation);
        if normal.time.samples < 3 {
            return None;
        }
        let transferable = if key.batch > 1 && family_funds >= 0. {
            // A setup plus three hot samples should not put hundreds of
            // milliseconds into a packet that normally takes only a few ms.
            (self.calibration_limit.min(normal.time.mean * 4.) - self.batch_allocated)
                .max(0.)
                .min((self.credit - funds).max(0.))
        } else {
            0.
        };
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
            let recapture_cost = arm.map_or(0., |a| a.recapture_cost);
            let seconds = (if p.time.samples > 0 {
                p.time.upper()
            } else {
                priors[&candidate.width]
            }) + recapture_cost;
            if !seconds.is_finite() || seconds <= 0. || seconds > slack {
                continue;
            }
            let samples = p.time.samples;
            // Some row geometries cost more than a wider packet. A measured
            // slower shape cannot win when that wider shape also has at least
            // its expected yield. Keep its samples instead of periodically
            // resetting and recapturing an already dominated implementation.
            // A new mode/generation or an unmeasured alternative still probes.
            let dominated = samples >= 3
                && candidates.iter().any(|other| {
                    if other.width == candidate.width
                        || other.expected_tokens < candidate.expected_tokens
                        || !other.expected_tokens.is_finite()
                    {
                        return false;
                    }
                    let Some(other) = cohort.arms.get(&other.width) else {
                        return false;
                    };
                    let cost = other.recapture_cost;
                    let other = other.hot(self.generation);
                    other.time.samples >= 3
                        && other.time.upper() + cost < p.time.mean + recapture_cost
                });
            let score = if samples > 0 {
                // Timing samples describe a shape, not the acceptance of the
                // current request. Its conditional estimator already shrinks
                // sparse observations toward the model prior.
                (p.time.mean + recapture_cost) / candidate.expected_tokens.max(1.)
            } else {
                f64::INFINITY
            };
            if self.loan == Some((key, candidate.width)) && samples >= 3 && score >= baseline {
                self.loan = None;
            }
            let stale = arm.is_some_and(|a| cohort.calls.saturating_sub(a.last_trial) >= 32);
            let pending = cohort.calibrating == Some(candidate.width);
            let recovering = samples < 3
                && arm.is_some_and(|a| {
                    (a.state == State::Ready || pending && cohort.recovering)
                        && (a.prior + recapture_cost) / candidate.expected_tokens.max(1.) < baseline
                });
            let cooling = arm.is_some_and(|a| a.state == State::Cooldown) && !stale;
            let calibrate = samples < 3 || (stale && score >= baseline);
            // Completed preparation is already charged. Reserve setup again
            // only for a deferred capture or a changed graph generation.
            let needs_setup = arm.is_none_or(|a| {
                a.needs_setup || a.modes[a.mode as usize].generation != self.generation
            });
            let setup = if samples == 0 && (!pending || needs_setup) {
                arm.filter(|a| a.preparation.samples > 0)
                    .map_or(seconds, |a| a.preparation.upper().min(seconds))
            } else {
                0.
            };
            let missing = if samples >= 3 { 1 } else { 3 - samples };
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
            let loan = self.loan.is_some_and(|(k, w)| {
                k.family() == key.family()
                    && k.context.abs_diff(key.context) <= 1
                    && w == candidate.width
                    && samples < 3
            });
            let initial = single && !key.shared_work && !self.bootstrap_used;
            // Admission was funded by this family's earned allowance. Finish
            // that calibration using actual model-wide savings as well; a
            // cold debit must not strand a profitable batch after one sample.
            // This grants no new loan and cannot admit another cold arm.
            let calibration_funds = if (pending || recovering) && samples < 3 {
                self.credit
            } else {
                funds
            };
            let allocation = if !pending && !recovering {
                transferable.min((self.credit - calibration_funds).max(0.))
            } else {
                0.
            };
            let available = calibration_funds
                + allocation
                + if loan || initial {
                    self.calibration_limit
                } else {
                    0.
                };
            if self.trace {
                eprintln!(
                    "MTP_CANDIDATE {key:?} width={} samples={samples} baseline={baseline:.6} seconds={seconds:.6} score={score:.6} pending={pending} setup={setup:.6} required={required:.6} available={available:.6} loan={loan} allocation={allocation:.6} cooling={cooling} dominated={dominated} recapture={recapture_cost:.6}",
                    candidate.width
                );
            }
            if calibrate
                && !dominated
                && !cooling
                && enough_work
                && (single || required <= recoverable)
                && (pending || recovering || cohort.calibrating.is_none())
                && available >= required
                && probe.as_ref().is_none_or(|&(_, _, _, priority, cost)| {
                    let rank = usize::from(pending) + 2 * usize::from(recovering);
                    rank > priority
                        || rank == priority
                            && recovering
                            && seconds / candidate.expected_tokens < cost
                })
            {
                probe = Some((
                    candidate.width,
                    initial,
                    (required - calibration_funds).max(0.).min(allocation),
                    usize::from(pending) + 2 * usize::from(recovering),
                    seconds / candidate.expected_tokens,
                ));
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
        if let Some((width, initial, allocation, priority, _)) = probe {
            *self.balances.entry(key.family()).or_default() += allocation;
            self.batch_allocated += allocation;
            if let Some(previous) = cohort.calibrating.filter(|&previous| previous != width)
                && let Some(arm) = cohort.arms.get_mut(&previous)
            {
                arm.state = State::Cooldown;
                arm.last_trial = cohort.calls;
            }
            let arm = cohort.arms.entry(width).or_default();
            // A stale retry refreshes measured costs. Only a new mode or
            // invalidation needs three fresh hot samples; cooldown is not a
            // change to the arithmetic or launch geometry.
            arm.state = State::Calibrating;
            cohort.calibrating = Some(width);
            // Keep restoration ahead of other stale/cold probes until its
            // actual execution mode has three hot samples, including after
            // the first sample changes the arm from Ready to Calibrating.
            cohort.recovering = priority >= 2;
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
        if let Some((width, _)) = best {
            cohort.arms.get_mut(&width).unwrap().state = State::Ready;
        }
        best.map(|(width, _)| (width, false))
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
