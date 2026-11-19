//! One model-lifetime time account survives timing-profile eviction. Actual
//! earnings fund new shapes; only the bounded startup pool creates credit.
use super::{Measurement, valid_duration};

pub(crate) struct Budget {
    pub balance: f64,
    limit: f64,
    pub startup_spent: f64,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            balance: 0.,
            startup_spent: 0.,
            limit: 0.4,
        }
    }
}

impl Budget {
    pub fn set_limit(&mut self, limit: f64) {
        if limit.is_finite() && limit > 0. {
            self.limit = limit;
        }
    }

    #[cfg(test)]
    pub fn limit(&self) -> f64 {
        self.limit
    }

    fn remaining(&self) -> f64 {
        (self.limit - self.startup_spent).max(0.)
    }

    pub fn available(&self) -> f64 {
        self.balance + self.remaining()
    }

    pub fn can_reserve(&self, excess: f64) -> bool {
        excess <= self.limit + self.remaining()
    }

    pub fn settle_action(
        &mut self,
        seconds: f64,
        baseline: f64,
        measured: bool,
        calibrating: bool,
    ) {
        if seconds.is_finite() && seconds > 0. && baseline.is_finite() {
            let delta = baseline - seconds;
            self.settle_delta(if measured { delta } else { delta.min(0.) }, calibrating);
        }
    }

    pub fn earn_ordinary(&mut self, seconds: f64, preparation_s: f64) {
        if valid_duration(seconds, preparation_s) {
            self.settle_delta((seconds - preparation_s).max(0.) * 0.02, false);
        }
    }
    pub fn settle(
        &mut self,
        width: usize,
        sample: Measurement,
        normal: f64,
        batch: usize,
        calibrating: bool,
    ) {
        let delta = if width == 1 && sample.tokens > 0 {
            // A speculative shape can evict the ordinary graph. Once its
            // hot baseline is known, recovery is a real workload cost too.
            // Use wall-time excess, so nested preparation timers never add
            // duplicate charges. Initial baseline discovery stays exempt.
            let recovery = if normal > 0. && sample.preparation_s > 0. {
                (sample.seconds - normal).max(0.)
            } else {
                0.
            };
            sample.seconds * 0.02 - recovery
        } else {
            normal * sample.tokens as f64 / batch.max(1) as f64 - sample.seconds
        };
        self.settle_delta(delta, calibrating);
    }
    fn settle_delta(&mut self, delta: f64, calibrating: bool) {
        if !delta.is_finite() {
            return;
        }
        let remaining = self.remaining();
        self.balance = (self.balance + delta).min(self.limit);
        let subsidy = if calibrating {
            (-self.balance).max(0.).min(remaining)
        } else {
            0.
        };
        self.balance += subsidy;
        self.startup_spent += subsidy;
    }
}
