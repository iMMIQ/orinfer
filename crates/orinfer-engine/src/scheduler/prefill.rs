//! Bounded CPU cost curves for real packets, independent of device and model.
use super::{Estimate, Timing};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct Work {
    pub tokens: usize,
    pub context: usize,
    pub decode_rows: usize,
    pub decode_context: usize,
    pub final_rows: usize,
    pub full: bool,
    /// Backend-declared execution family; curves never cross this boundary.
    pub domain: usize,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(crate) enum Path {
    Direct,
    Replay,
    Shared,
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Family(usize, bool, usize, usize, usize, usize);
fn bucket(n: usize) -> usize {
    n.max(1).next_power_of_two()
}
fn optional_bucket(n: usize) -> usize {
    if n == 0 { 0 } else { bucket(n) }
}
fn family(w: Work) -> Family {
    Family(
        w.domain,
        w.full && (w.decode_rows > 0 || w.final_rows > 0),
        bucket(w.context),
        optional_bucket(w.decode_rows),
        bucket(w.decode_context),
        optional_bucket(w.final_rows),
    )
}
struct Sample {
    time: Timing,
    context: Estimate,
    last: u64,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Prediction {
    pub seconds: f64,
    pub upper_s: f64,
    pub path: Path,
    pub samples: usize,
}
#[derive(Default)]
struct Curves {
    cells: BTreeMap<(Family, usize, Path), Sample>,
    modes: BTreeMap<(Family, usize), Path>,
    clock: u64,
}
impl Curves {
    fn sample(&mut self, work: Work, path: Path) -> &mut Sample {
        self.clock += 1;
        let group = family(work);
        let key = (group, work.tokens, path);
        if !self.cells.contains_key(&key) && self.cells.len() == 256 {
            let old = *self.cells.iter().min_by_key(|(_, s)| s.last).unwrap().0;
            self.cells.remove(&old);
            if !self.cells.keys().any(|k| (k.0, k.1) == (old.0, old.1)) {
                self.modes.remove(&(old.0, old.1));
            }
        }
        self.modes.insert((group, work.tokens), path);
        let sample = self.cells.entry(key).or_insert_with(|| Sample {
            time: Timing::default(),
            context: Estimate::default(),
            last: 0,
        });
        sample.last = self.clock;
        sample
    }

    fn exact_upper(&self, work: Work) -> Option<f64> {
        self.cells
            .get(&(family(work), work.tokens, Path::Direct))
            .map(|sample| sample.time.upper())
    }
}

pub(crate) struct Costs {
    curves: Curves,
    restores: BTreeMap<(bool, usize), Estimate>,
    cold_token_s: f64,
}
impl Costs {
    pub fn new(cold_tokens_per_s: f64) -> Self {
        assert!(cold_tokens_per_s.is_finite() && cold_tokens_per_s > 0.);
        Self {
            curves: Default::default(),
            restores: BTreeMap::new(),
            cold_token_s: 1. / cold_tokens_per_s,
        }
    }
    pub fn observe(&mut self, work: Work, path: Path, seconds: f64, setup_s: f64) {
        if !seconds.is_finite()
            || seconds <= 0.
            || !setup_s.is_finite()
            || setup_s < 0.
            || work.tokens + work.decode_rows == 0
        {
            return;
        }
        let sample = self.curves.sample(work, path);
        // Materialization/capture stays outside recurring execution prices.
        if setup_s == 0. || sample.time.observations() == 0 {
            sample.time.observe(seconds);
            sample.context.observe(work.context.max(1) as f64);
        }
    }
    pub fn estimate(&self, work: Work) -> Prediction {
        let path = self
            .curves
            .modes
            .get(&(family(work), work.tokens))
            .copied()
            .unwrap_or(Path::Direct);
        self.estimate_path(work, path)
    }
    pub fn estimate_path(&self, work: Work, path: Path) -> Prediction {
        if work.tokens + work.decode_rows == 0 {
            return Prediction {
                seconds: 0.,
                upper_s: 0.,
                path,
                samples: 0,
            };
        }
        let local = self.measured(work, path);
        if let Some(p) = local.filter(|p| p.samples > 0) {
            return p;
        }
        // A cold direct primer must not permanently block a larger window
        // after replay has established real hot execution costs. Transfer hot
        // execution as a prior with its measured residual, never residency.
        for other in [Path::Replay, Path::Shared, Path::Direct] {
            if other != path
                && let Some(p) = self.measured(work, other).filter(|p| p.samples > 0)
            {
                return Prediction {
                    path,
                    samples: 0,
                    ..p
                };
            }
        }
        if let Some(p) = local {
            return p;
        }
        // Unprimed modes retain a conservative cold prior.
        for other in [Path::Replay, Path::Shared, Path::Direct] {
            if other != path
                && let Some(p) = self.measured(work, other)
            {
                return Prediction {
                    seconds: p.seconds * 1.25,
                    upper_s: p.upper_s * 1.25,
                    path,
                    samples: 0,
                };
            }
        }
        if work.tokens > 0 && work.decode_rows > 0 {
            let prompt = self.estimate_path(
                Work {
                    decode_rows: 0,
                    decode_context: 0,
                    ..work
                },
                path,
            );
            let decode = self
                .curves
                .cells
                .iter()
                .filter(|((f, q, _), _)| *q == 0 && f.1 == work.full)
                .min_by_key(|((f, _, _), _)| {
                    f.3.abs_diff(bucket(work.decode_rows))
                        + f.4.ilog2().abs_diff(bucket(work.decode_context).ilog2()) as usize
                })
                .map(|((f, _, _), s)| {
                    let scale = (work.decode_rows as f64 / f.3.max(1) as f64).max(1.);
                    (s.time.mean() * scale, s.time.upper() * scale)
                })
                .unwrap_or((
                    work.decode_rows as f64 * self.cold_token_s,
                    work.decode_rows as f64 * self.cold_token_s,
                ));
            return Prediction {
                seconds: prompt.seconds + decode.0,
                upper_s: prompt.upper_s + decode.1,
                path,
                samples: 0,
            };
        }
        let seconds = (work.tokens + work.decode_rows) as f64 * self.cold_token_s;
        Prediction {
            seconds,
            upper_s: seconds,
            path,
            samples: 0,
        }
    }
    fn measured(&self, work: Work, path: Path) -> Option<Prediction> {
        let wanted = family(work);
        let geometry = self
            .curves
            .cells
            .keys()
            .filter(|(f, q, p)| {
                *p == path
                    && (*q == 0) == (work.tokens == 0)
                    && f.4.ilog2().abs_diff(wanted.4.ilog2()) <= 1
                    && f.0 == wanted.0
                    && f.1 == wanted.1
                    && (f.3 == 0) == (wanted.3 == 0)
                    && (f.5 == 0) == (wanted.5 == 0)
            })
            .min_by_key(|(f, _, _)| f.3.abs_diff(wanted.3) + f.5.abs_diff(wanted.5))?
            .0;
        let mut profiles = BTreeMap::<usize, Vec<(usize, &Sample)>>::new();
        for ((f, q, p), sample) in &self.curves.cells {
            if *p == path
                && f.0 == geometry.0
                && f.1 == geometry.1
                && f.3 == geometry.3
                && f.5 == geometry.5
                && f.4.ilog2().abs_diff(wanted.4.ilog2()) <= 1
            {
                profiles.entry(f.2).or_default().push((*q, sample));
            }
        }
        let profiles = profiles
            .into_values()
            .filter_map(|mut points| {
                let hot = points.iter().any(|(_, s)| s.time.samples() > 0);
                points.retain(|(_, s)| !hot || s.time.samples() > 0);
                if points.is_empty() {
                    return None;
                }
                if !points.iter().any(|(q, _)| (*q == 0) == (work.tokens == 0)) {
                    return None;
                }
                points.sort_unstable_by_key(|p| p.0);
                let context =
                    points.iter().map(|(_, s)| s.context.mean).sum::<f64>() / points.len() as f64;
                Some((context, points))
            })
            .collect::<Vec<_>>();
        let lower = profiles
            .iter()
            .filter(|p| p.0 <= work.context.max(1) as f64)
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let upper = profiles
            .iter()
            .filter(|p| p.0 >= work.context.max(1) as f64)
            .min_by(|a, b| a.0.total_cmp(&b.0));
        let price = |points: &[(usize, &Sample)]| {
            let (mean, upper) = if (work.decode_rows > 0 || work.final_rows > 0)
                && work.tokens > 0
                && points.len() == 1
                && points[0].0 != work.tokens
            {
                // Decode and final replay/head include fixed costs. With
                // one prompt size, transfer the measured pure-prompt delta;
                // scaling the entire packet would multiply that fixed cost
                // and permanently trap the scheduler at its minimum window.
                let prompt = |tokens| {
                    self.estimate_path(
                        Work {
                            tokens,
                            decode_rows: 0,
                            decode_context: 0,
                            final_rows: 0,
                            full: false,
                            ..work
                        },
                        path,
                    )
                };
                let anchor = prompt(points[0].0);
                let query = prompt(work.tokens);
                (
                    (points[0].1.time.mean() + query.seconds - anchor.seconds).max(0.),
                    (points[0].1.time.upper() + query.upper_s - anchor.seconds).max(0.),
                )
            } else {
                (
                    interpolate(points, work.tokens, |s| s.time.mean()),
                    interpolate(points, work.tokens, |s| s.time.upper()),
                )
            };
            let samples = points
                .iter()
                .map(|(_, s)| s.time.samples())
                .min()
                .unwrap_or(0);
            (mean, upper.max(mean), samples)
        };
        let (seconds, upper_s, samples) = match (lower, upper) {
            (Some(a), Some(b)) if a.0 < b.0 => {
                let fraction = ((work.context as f64 - a.0) / (b.0 - a.0)).clamp(0., 1.);
                let x = price(&a.1);
                let y = price(&b.1);
                (
                    x.0 + (y.0 - x.0) * fraction,
                    x.1 + (y.1 - x.1) * fraction,
                    x.2.min(y.2),
                )
            }
            (Some(a), _) | (_, Some(a)) => price(&a.1),
            _ => return None,
        };
        Some(Prediction {
            seconds,
            upper_s,
            samples,
            path,
        })
    }
    pub fn observe_restore(&mut self, host: bool, tokens: usize, seconds: f64) {
        if tokens > 0 && seconds.is_finite() && seconds > 0. {
            self.restores
                .entry((host, bucket(tokens)))
                .or_default()
                .observe(seconds);
        }
    }
    pub fn restore(&self, host: bool, tokens: usize) -> f64 {
        if tokens == 0 {
            return 0.;
        }
        self.restores
            .iter()
            .filter(|((h, _), _)| *h == host)
            .min_by_key(|((_, n), _)| n.ilog2().abs_diff(bucket(tokens).ilog2()))
            .map_or(0., |(_, s)| s.mean)
    }
}
fn interpolate(points: &[(usize, &Sample)], q: usize, value: impl Fn(&Sample) -> f64) -> f64 {
    if points.len() == 1 {
        return value(points[0].1) * q.max(1) as f64 / points[0].0.max(1) as f64;
    }
    // Fit the whole token curve, with nonnegative slope and intercept. A
    // nearest-neighbour secant amplifies timing noise between adjacent row
    // counts into arbitrarily large extrapolation; clipping its intercept
    // afterwards does not constrain that slope.
    let count = points.len() as f64;
    let x = points.iter().map(|(n, _)| *n as f64).sum::<f64>() / count;
    let y = points.iter().map(|(_, s)| value(s)).sum::<f64>() / count;
    let variance = points
        .iter()
        .map(|(n, _)| (*n as f64 - x).powi(2))
        .sum::<f64>();
    if variance == 0. {
        return y * q.max(1) as f64 / x.max(1.);
    }
    let covariance = points
        .iter()
        .map(|(n, s)| (*n as f64 - x) * (value(s) - y))
        .sum::<f64>();
    let slope = (covariance / variance).max(0.);
    let intercept = y - slope * x;
    if intercept >= 0. {
        intercept + slope * q as f64
    } else {
        // Constrained least squares on the intercept=0 boundary.
        let xy = points
            .iter()
            .map(|(n, s)| *n as f64 * value(s))
            .sum::<f64>();
        let xx = points.iter().map(|(n, _)| (*n as f64).powi(2)).sum::<f64>();
        xy / xx * q as f64
    }
}

/// Greedy registered prefill profiles and checkpoint policy.
pub(crate) struct Profiles {
    shapes: BTreeMap<usize, Estimate>,
    weight_pass: Option<f64>,
    timings: Curves,
    stages: BTreeMap<(usize, usize, bool), f64>,
    restore_s_per_byte: f64,
    dynamic: Option<DynamicPrefill>,
}

struct DynamicPrefill {
    capacity: usize,
    seconds_per_row: Option<f64>,
}

impl Profiles {
    pub fn new(shapes: impl IntoIterator<Item = usize>) -> Self {
        Self {
            shapes: shapes
                .into_iter()
                .chain([1])
                .map(|n| (n, Estimate::default()))
                .collect(),
            weight_pass: None,
            timings: Default::default(),
            stages: BTreeMap::new(),
            restore_s_per_byte: 1. / 20e9,
            dynamic: None,
        }
    }

    pub fn with_dynamic_capacity(mut self, capacity: usize) -> Self {
        self.dynamic = Some(DynamicPrefill {
            capacity,
            seconds_per_row: None,
        });
        self
    }

    pub fn observe(&mut self, chunk: usize, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            if (1..=8).contains(&chunk) {
                self.weight_pass = Some(
                    self.weight_pass
                        .map_or(seconds, |old| old * 0.8 + seconds * 0.2),
                );
            }
            self.shapes.entry(chunk).or_default().observe(seconds);
        }
    }

    pub fn observe_at(&mut self, chunk: usize, context: usize, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        self.observe(chunk, seconds);
        let weight_pass_s = self.weight_pass_s();
        if let Some(dynamic) = &mut self.dynamic
            && (128..=dynamic.capacity).contains(&chunk)
            && seconds > weight_pass_s
        {
            // Fit the variable work after one weight pass. Small tails are
            // dominated by launch cost and give unstable throughput samples.
            let sample = (seconds - weight_pass_s) / chunk as f64;
            let weight = 0.2 * (chunk as f64 / 1024.).min(1.);
            dynamic.seconds_per_row = Some(
                dynamic
                    .seconds_per_row
                    .map_or(sample, |old| old * (1. - weight) + sample * weight),
            );
        }
        self.timings
            .sample(Self::work(chunk, context), Path::Direct)
            .time
            .observe_recurring(seconds);
    }

    fn work(chunk: usize, context: usize) -> Work {
        Work {
            tokens: chunk,
            context,
            ..Work::default()
        }
    }

    pub fn chunk_upper(&self, chunk: usize, context: usize) -> f64 {
        self.timings
            .exact_upper(Self::work(chunk, context))
            .unwrap_or_else(|| self.chunk_cost(chunk) * 1.2)
    }

    pub fn observe_stage(&mut self, chunk: usize, context: usize, warm: bool, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            // Charge the slowest observed layer in this geometry. Draft warm
            // runs in a separate stage with a different amount of work.
            let peak = self
                .stages
                .entry((bucket(chunk), bucket(context), warm))
                .or_default();
            *peak = peak.max(seconds);
        }
    }

    pub fn stage_upper(&self, chunk: usize, context: usize, count: usize, warm: bool) -> f64 {
        self.stages
            .get(&(bucket(chunk), bucket(context), warm))
            .copied()
            .unwrap_or_else(|| (self.chunk_upper(chunk, context) / count.max(1) as f64).max(0.001))
    }
    pub fn bounded_chunk_at(&self, remaining: usize, context: usize, budget_ms: f64) -> usize {
        let fitting = self
            .shapes
            .keys()
            .copied()
            .filter(|&n| {
                n <= remaining
                    && self.chunk_upper(n, context.saturating_add(n)) * 1000. <= budget_ms
            })
            .max()
            .unwrap_or(1);
        // A cold measurement or indivisible tail/head cost can make every
        // compiled tile exceed the soft budget. Repeated one-token weight
        // passes are then much worse than one minimum useful tile. Explore
        // that tile again so later measurements can correct the estimate.
        let minimum = self
            .shapes
            .keys()
            .rev()
            .copied()
            .find(|&n| n > 1 && n <= 16 && n <= remaining);
        minimum
            .filter(|&n| {
                fitting == 1
                    && self.chunk_upper(n, context.saturating_add(n)) / (n as f64)
                        < self.chunk_upper(1, context.saturating_add(1)) * 0.5
            })
            .unwrap_or(fitting)
    }
    pub fn observe_restore(&mut self, bytes: usize, seconds: f64) {
        if bytes > 0 && seconds.is_finite() && seconds > 0. {
            self.restore_s_per_byte = self.restore_s_per_byte * 0.8 + seconds / bytes as f64 * 0.2;
        }
    }
    pub fn restore_cost(&self, bytes: usize) -> f64 {
        bytes as f64 * self.restore_s_per_byte
    }
    fn weight_pass_s(&self) -> f64 {
        self.weight_pass.unwrap_or(0.1)
    }
    pub fn chunk_cost(&self, tokens: usize) -> f64 {
        self.shapes
            .get(&tokens)
            .filter(|estimate| estimate.samples > 0)
            .map(|estimate| estimate.mean)
            .unwrap_or_else(|| {
                if self.dynamic.is_some() {
                    self.span(tokens)
                } else {
                    self.weight_pass_s() * (tokens as f64 / 64.).max(1.)
                }
            })
    }
    /// Largest real profile within the estimated budget; one token guarantees
    /// progress even when no profile fits. This is a soft scheduling target.
    #[cfg(test)]
    pub fn bounded_chunk(&self, remaining: usize, budget_ms: f64) -> usize {
        self.shapes
            .keys()
            .copied()
            .filter(|&n| n <= remaining && self.chunk_cost(n) * 1000. <= budget_ms)
            .max()
            .unwrap_or(1)
    }
    fn span(&self, mut tokens: usize) -> f64 {
        if let Some(dynamic) = &self.dynamic {
            // The native plan executes one exact-sized forward pass per
            // chunk. Measurements of other lengths are not chunk boundaries.
            let passes = tokens.div_ceil(dynamic.capacity);
            let weight_pass_s = self.weight_pass_s();
            let per_row = dynamic.seconds_per_row.unwrap_or(weight_pass_s / 64.);
            return passes as f64 * weight_pass_s + tokens as f64 * per_row;
        }
        let mut total = 0.;
        for &chunk in self.shapes.keys().rev() {
            let count = tokens / chunk;
            tokens %= chunk;
            total += count as f64 * self.chunk_cost(chunk);
        }
        total
    }
    pub fn remaining(&self, mut offset: usize, end: usize) -> f64 {
        let mut total = 0.;
        while offset < end {
            let boundary = ((offset / 8192 + 1) * 8192).min(end);
            total += self.span(boundary - offset);
            offset = boundary;
        }
        total
    }
    pub fn score(&self, prefix: usize, end: usize, bytes: usize) -> Option<f64> {
        let cost = self.remaining(prefix, end) + self.restore_cost(bytes);
        // A complete head-bearing checkpoint always avoids model execution.
        (prefix == end || cost < self.remaining(0, end) * 0.95).then_some(cost)
    }
    #[cfg(test)]
    pub fn admit(
        &self,
        prefix: usize,
        end: usize,
        bytes: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
    ) -> bool {
        self.admit_with_reuse(prefix, end, bytes, start, checkpoints, false)
    }
    pub fn admit_with_reuse(
        &self,
        prefix: usize,
        end: usize,
        bytes: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
        reused: bool,
    ) -> bool {
        if prefix <= start || self.score(prefix, end, bytes).is_none() {
            return false;
        }
        if prefix >= 64 && (2..=8).contains(&end.saturating_sub(prefix)) {
            // The boundary before <think> can be reused after template rendering.
            return true;
        }
        let mut trial = checkpoints.clone();
        trial.insert(prefix);
        let (mut previous, mut split) = (start, 0.);
        for &point in trial.range((
            std::ops::Bound::Excluded(start),
            std::ops::Bound::Excluded(end),
        )) {
            split += self.remaining(previous, point);
            previous = point;
        }
        split += self.remaining(previous, end);
        let baseline = self.remaining(start, end);
        // Do not create a speculative boundary by destroying an efficient
        // large chunk. Periodic checkpoints are already chunk boundaries.
        // Account from the restored endpoint and include already admitted
        // boundaries, rather than evaluating each split against a cold run.
        let extra = (split - baseline).max(0.);
        extra <= (baseline * 0.1).max(0.05)
            || (reused
                && baseline - self.remaining(prefix, end) > extra + self.restore_cost(bytes) * 2.)
    }
}

#[cfg(test)]
#[path = "prefill_tests.rs"]
mod tests;
