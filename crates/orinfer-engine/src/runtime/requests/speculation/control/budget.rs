//! Exploration accounts survive timing-profile eviction. One model-lifetime
//! startup pool pays actual losses; profits stay in the workload that earned them.
use super::Measurement;
use std::collections::BTreeMap;
type Family = (usize, bool, usize);

#[derive(Default)]
pub(super) struct Budget {
    accounts: BTreeMap<Family, f64>,
    pub startup_spent: f64,
}
impl Budget {
    fn account(&mut self, family: Family) -> Option<&mut f64> {
        if self.accounts.len() == 256 && !self.accounts.contains_key(&family) {
            return None;
        }
        Some(self.accounts.entry(family).or_default())
    }
    pub fn balance(&self, family: Family) -> f64 {
        self.accounts.get(&family).copied().unwrap_or(0.)
    }
    pub fn available(&self, family: Family, limit: f64) -> f64 {
        if self.accounts.len() == 256 && !self.accounts.contains_key(&family) {
            return f64::NEG_INFINITY;
        }
        self.balance(family) + (limit - self.startup_spent).max(0.)
    }
    pub fn settle(
        &mut self,
        family: Family,
        width: usize,
        sample: Measurement,
        normal: f64,
        batch: usize,
        limit: f64,
    ) {
        let delta = if width == 1 && sample.tokens > 0 {
            sample.seconds * 0.02
        } else {
            normal * sample.tokens as f64 / batch.max(1) as f64 - sample.seconds
        };
        if !delta.is_finite() {
            return;
        }
        let remaining = (limit - self.startup_spent).max(0.);
        if let Some(balance) = self.account(family) {
            *balance = (*balance + delta).min(limit);
            let subsidy = (-*balance).max(0.).min(remaining);
            *balance += subsidy;
            self.startup_spent += subsidy;
        }
    }
}
