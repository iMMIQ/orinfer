//! Complete execution prices. Layer-only estimates remain in prefill_cost.
use super::*;
use crate::scheduler::{BatchExecutionStatistics, Timing};

pub(in crate::runtime) type Key = (usize, usize, usize, usize);

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Measurement {
    pub seconds: f64,
    pub preparation_s: f64,
}
impl Measurement {
    pub fn add(&mut self, other: Self) {
        self.seconds += other.seconds;
        self.preparation_s += other.preparation_s;
    }
}

pub(super) struct Timer {
    began: Instant,
    before: BatchExecutionStatistics,
}
impl Timer {
    pub fn start(before: BatchExecutionStatistics) -> Self {
        Self {
            began: Instant::now(),
            before,
        }
    }
    pub fn finish(self, after: BatchExecutionStatistics) -> Measurement {
        let seconds = self.began.elapsed().as_secs_f64();
        Measurement {
            seconds,
            preparation_s: after.preparation_s_since(self.before).min(seconds),
        }
    }
}

#[derive(Default)]
pub(in crate::runtime) struct Sample {
    pub time: Timing,
    overhead_s: f64,
    pub last: usize,
}
impl Sample {
    pub fn mean(&self) -> f64 {
        self.time.mean() + self.overhead_s
    }
    pub fn upper(&self) -> f64 {
        self.time.upper() + self.overhead_s
    }
    fn observe(&mut self, sample: Measurement, clock: usize) {
        if !sample.seconds.is_finite()
            || sample.seconds <= 0.
            || !sample.preparation_s.is_finite()
            || sample.preparation_s < 0.
        {
            return;
        }
        self.last = clock;
        let extra = if sample.preparation_s > 0. && self.time.samples() > 0 {
            (sample.seconds - self.time.mean()).max(0.)
        } else {
            0.
        };
        self.overhead_s = self.overhead_s * 0.8 + extra * 0.2;
        // A new request's head, shifted warm and commit are recurring work.
        // Only executor-reported materialization excludes a later hot sample.
        if self.time.observations() == 0 || sample.preparation_s == 0. {
            self.time.observe(sample.seconds);
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::runtime) struct Alternative {
    pub prefill_s: f64,
    pub decode_s: f64,
    pub decoders: usize,
    pub measured: bool,
    pub calibrating: bool,
    pub probing: bool,
}
impl Alternative {
    pub fn seconds(self, tokens: usize) -> f64 {
        self.prefill_s + self.decode_s * tokens as f64 / self.decoders.max(1) as f64
    }
}

impl ModelRuntime {
    pub(super) fn observe_iteration(&mut self, key: Key, sample: Measurement) {
        if self.iteration_costs.len() == 256
            && !self.iteration_costs.contains_key(&key)
            && let Some(old) = self
                .iteration_costs
                .iter()
                .min_by_key(|(_, s)| s.last)
                .map(|(&k, _)| k)
        {
            self.iteration_costs.remove(&old);
        }
        self.iteration_costs
            .entry(key)
            .or_default()
            .observe(sample, self.scheduler_statistics.iterations);
    }
    pub(super) fn iteration_upper(&self, key: Key) -> Option<f64> {
        self.iteration_costs
            .range((key.0, key.1, 0, 0)..=(key.0, key.1, key.2, usize::MAX))
            .filter(|(k, s)| k.3 == key.3 && s.time.samples() > 0)
            .map(|(_, s)| s.upper())
            .max_by(f64::total_cmp)
    }
    pub(super) fn iteration_sample(&self, key: Key) -> Option<&Sample> {
        self.iteration_costs
            .range((key.0, key.1, 0, 0)..=(key.0, key.1, key.2, usize::MAX))
            .filter(|(k, s)| k.3 == key.3 && s.time.samples() > 0)
            .max_by(|(_, a), (_, b)| a.mean().total_cmp(&b.mean()))
            .map(|(_, s)| s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_expensive_observation_does_not_lock_out_hot_measurements() {
        let mut sample = Sample::default();
        for (i, seconds) in [1.53, 1.30, 1.31, 1.29].into_iter().enumerate() {
            sample.observe(
                Measurement {
                    seconds,
                    preparation_s: 0.,
                },
                i,
            );
        }
        assert_eq!(sample.time.samples(), 3);
        assert!(sample.mean() < 1.54 * 0.95);
        assert!(sample.upper() >= sample.mean());
    }
    #[test]
    fn complete_frame_adds_active_slices_and_recurring_warm() {
        let mut frame = Measurement::default();
        frame.add(Measurement {
            seconds: 0.2,
            preparation_s: 0.01,
        });
        frame.add(Measurement {
            seconds: 0.15,
            preparation_s: 0.,
        });
        frame.add(Measurement {
            seconds: 0.06,
            preparation_s: 0.,
        });
        assert!((frame.seconds - 0.41).abs() < 1e-12);
        assert_eq!(frame.preparation_s, 0.01);
        let mut sample = Sample::default();
        sample.observe(frame, 0);
        frame.preparation_s = 0.;
        for i in 1..4 {
            sample.observe(frame, i);
        }
        assert!((sample.mean() - 0.41).abs() < 1e-12);
    }
    #[test]
    fn mixed_baseline_charges_prompt_work_once_at_real_decode_progress() {
        let alternative = Alternative {
            prefill_s: 1.,
            decode_s: 0.04,
            decoders: 4,
            measured: true,
            calibrating: false,
            probing: false,
        };
        assert!((alternative.seconds(12) - 1.12).abs() < 1e-12);
        assert_eq!(alternative.seconds(0), 1.);
    }
}
