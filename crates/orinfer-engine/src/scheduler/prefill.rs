//! Bounded CPU cost curves for real packets, independent of device and model.
use super::{Estimate, Timing};
use std::collections::BTreeMap;

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
pub(crate) struct Costs {
    cells: BTreeMap<(Family, usize, Path), Sample>,
    modes: BTreeMap<(Family, usize), Path>,
    restores: BTreeMap<(bool, usize), Estimate>,
    clock: u64,
    cold_token_s: f64,
}
impl Costs {
    pub fn new(cold_tokens_per_s: f64) -> Self {
        assert!(cold_tokens_per_s.is_finite() && cold_tokens_per_s > 0.);
        Self {
            cells: BTreeMap::new(),
            modes: BTreeMap::new(),
            restores: BTreeMap::new(),
            clock: 0,
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
        // Materialization/capture stays outside recurring execution prices.
        if setup_s == 0. || sample.time.observations() == 0 {
            sample.time.observe(seconds);
            sample.context.observe(work.context.max(1) as f64);
        }
    }
    pub fn estimate(&self, work: Work) -> Prediction {
        let path = self
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
            .cells
            .keys()
            .filter(|(f, _, p)| {
                *p == path
                    && f.0 == wanted.0
                    && f.1 == wanted.1
                    && (f.3 == 0) == (wanted.3 == 0)
                    && (f.5 == 0) == (wanted.5 == 0)
            })
            .min_by_key(|(f, _, _)| f.3.abs_diff(wanted.3) + f.5.abs_diff(wanted.5))?
            .0;
        let mut profiles = BTreeMap::<usize, Vec<(usize, &Sample)>>::new();
        for ((f, q, p), sample) in &self.cells {
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

#[cfg(test)]
mod tests {
    use super::*;
    fn observe(c: &mut Costs, w: Work, path: Path, seconds: f64) {
        for _ in 0..4 {
            c.observe(w, path, seconds, 0.);
        }
    }
    #[test]
    fn learns_token_context_and_execution_cost_without_mixing_modes_or_final_windows() {
        let mut c = Costs::new(4000.);
        for context in [8192, 32768] {
            for tokens in [512, 2048] {
                let w = Work {
                    tokens,
                    context,
                    domain: 128,
                    ..Default::default()
                };
                observe(
                    &mut c,
                    w,
                    Path::Direct,
                    0.04 + tokens as f64 * 0.00005 * (context as f64 / 8192.),
                );
                observe(
                    &mut c,
                    w,
                    Path::Replay,
                    0.02 + tokens as f64 * 0.00005 * (context as f64 / 8192.),
                );
            }
        }
        let w = Work {
            tokens: 1024,
            context: 16384,
            domain: 128,
            ..Default::default()
        };
        assert!((c.estimate_path(w, Path::Replay).seconds - 0.1224).abs() < 1e-8);
        assert!((c.estimate_path(w, Path::Direct).seconds - 0.1424).abs() < 1e-8);
        observe(
            &mut c,
            Work {
                final_rows: 128,
                ..w
            },
            Path::Replay,
            1.,
        );
        assert!((c.estimate_path(w, Path::Replay).seconds - 0.1224).abs() < 1e-8);
        assert!(
            c.estimate_path(
                Work {
                    final_rows: 128,
                    ..w
                },
                Path::Replay
            )
            .seconds
                > 0.9
        );
    }
    #[test]
    fn final_decoder_window_is_not_multiplied_when_prompt_size_changes() {
        let mut c = Costs::new(4000.);
        let w = Work {
            context: 8192,
            domain: 128,
            ..Default::default()
        };
        for tokens in [512, 2048] {
            observe(
                &mut c,
                Work { tokens, ..w },
                Path::Direct,
                0.02 + tokens as f64 * 0.00005,
            );
        }
        let final_prompt = Work {
            tokens: 128,
            final_rows: 128,
            full: true,
            ..w
        };
        observe(&mut c, final_prompt, Path::Direct, 0.1);
        let p = c.estimate(Work {
            tokens: 1024,
            ..final_prompt
        });
        assert!((p.seconds - 0.1448).abs() < 1e-9);
        let mixed = Work {
            decode_rows: 8,
            ..final_prompt
        };
        observe(&mut c, mixed, Path::Direct, 0.15);
        let p = c.estimate(Work {
            tokens: 1024,
            ..mixed
        });
        assert!((p.seconds - 0.1948).abs() < 1e-9);
    }
    #[test]
    fn cold_direct_primer_does_not_block_hot_mode_transfer_to_larger_windows() {
        let mut c = Costs::new(4000.);
        let w = Work {
            domain: 128,
            ..Default::default()
        };
        for tokens in [512, 2048] {
            observe(
                &mut c,
                Work { tokens, ..w },
                Path::Replay,
                tokens as f64 / 10240.,
            );
        }
        let mixed = Work {
            tokens: 512,
            decode_rows: 8,
            ..w
        };
        c.observe(mixed, Path::Direct, 0.3, 0.2);
        observe(&mut c, mixed, Path::Replay, 0.1);
        let p = c.estimate(Work {
            tokens: 1024,
            ..mixed
        });
        assert_eq!(p.path, Path::Direct);
        assert_eq!(p.samples, 0);
        assert!((p.seconds - 0.15).abs() < 1e-9);
        assert!(p.upper_s < 0.2);
    }
    #[test]
    fn adjacent_row_timing_noise_cannot_create_unbounded_extrapolation() {
        let mut c = Costs::new(4000.);
        let w = Work {
            domain: 128,
            ..Default::default()
        };
        observe(&mut c, Work { tokens: 1024, ..w }, Path::Direct, 0.12);
        observe(&mut c, Work { tokens: 1025, ..w }, Path::Direct, 0.14);
        let p = c.estimate(Work { tokens: 2048, ..w });
        assert!((0.24..0.29).contains(&p.seconds));
        assert!(p.upper_s < 0.3);
    }
    #[test]
    fn sparse_mixed_observations_preserve_decoder_intercept() {
        let mut c = Costs::new(4000.);
        let w = Work {
            context: 8192,
            domain: 128,
            ..Default::default()
        };
        for tokens in [512, 2048] {
            observe(
                &mut c,
                Work { tokens, ..w },
                Path::Direct,
                0.02 + tokens as f64 * 0.00005,
            );
        }
        observe(
            &mut c,
            Work {
                tokens: 128,
                decode_rows: 8,
                ..w
            },
            Path::Direct,
            0.125,
        );
        let p = c.estimate(Work {
            tokens: 1024,
            decode_rows: 8,
            ..w
        });
        assert!((p.seconds - 0.1698).abs() < 1e-8);
        assert!(p.upper_s < 0.2);
    }
    #[test]
    fn preparation_and_first_observation_do_not_poison_hot_cost_or_claim_replay_residency() {
        let mut c = Costs::new(4000.);
        let w = Work {
            tokens: 1024,
            ..Default::default()
        };
        c.observe(w, Path::Direct, 2., 1.);
        observe(&mut c, w, Path::Direct, 0.1);
        assert!((c.estimate(w).seconds - 0.1).abs() < 1e-9);
        assert_eq!(c.estimate(w).path, Path::Direct);
        c.observe(w, Path::Replay, 0.08, 0.4);
        observe(&mut c, w, Path::Replay, 0.06);
        assert!((c.estimate(w).seconds - 0.06).abs() < 1e-9);
        assert_eq!(c.estimate(Work { tokens: 1025, ..w }).path, Path::Direct);
    }
    #[test]
    fn cache_restore_paths_and_observation_memory_are_bounded() {
        let mut c = Costs::new(4000.);
        c.observe_restore(false, 100000, 0.01);
        c.observe_restore(true, 100000, 0.2);
        assert_eq!(c.restore(false, 100001), 0.01);
        assert_eq!(c.restore(true, 100001), 0.2);
        assert_eq!(c.restore(false, 0), 0.);
        for tokens in 1..=1000 {
            c.observe(
                Work {
                    tokens,
                    ..Default::default()
                },
                Path::Direct,
                0.1,
                0.,
            );
        }
        assert_eq!(c.cells.len(), 256);
        assert!(c.modes.len() <= 256);
        c.observe(
            Work {
                tokens: 1,
                ..Default::default()
            },
            Path::Direct,
            f64::NAN,
            0.,
        );
        assert_eq!(c.cells.len(), 256);
    }
}
