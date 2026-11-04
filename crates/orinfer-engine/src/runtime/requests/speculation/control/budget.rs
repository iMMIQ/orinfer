//! One model-lifetime time account survives timing-profile eviction. Actual
//! earnings fund new shapes; only the bounded startup pool creates credit.
use super::Measurement;

#[derive(Default)]
pub(super) struct Budget {
    pub balance: f64,
    pub startup_spent: f64,
}
impl Budget {
    pub fn available(&self, limit: f64) -> f64 {
        self.balance + (limit - self.startup_spent).max(0.)
    }
    pub fn settle(
        &mut self,
        width: usize,
        sample: Measurement,
        normal: f64,
        batch: usize,
        limit: f64,
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
        if !delta.is_finite() {
            return;
        }
        let remaining = (limit - self.startup_spent).max(0.);
        self.balance = (self.balance + delta).min(limit);
        let subsidy = if calibrating {
            (-self.balance).max(0.).min(remaining)
        } else {
            0.
        };
        self.balance += subsidy;
        self.startup_spent += subsidy;
    }
}
