//! Exploration accounts survive timing-profile eviction. One model-lifetime
//! startup pool pays actual losses; profits stay in the workload that earned them.
use super::Measurement;
use std::collections::BTreeMap;
type Family = (usize, bool, usize);

#[derive(Default)]
struct Account {
    balance: f64,
    startup_spent: f64,
}
#[derive(Default)]
pub(crate) struct Budget {
    accounts: BTreeMap<Family, Account>,
    pub startup_spent: f64,
    family_limit: Option<f64>,
}
impl Budget {
    fn account(&mut self, family: Family) -> Option<&mut Account> {
        if self.accounts.len() == 256 && !self.accounts.contains_key(&family) {
            return None;
        }
        Some(self.accounts.entry(family).or_default())
    }
    pub fn balance(&self, family: Family) -> f64 {
        self.accounts.get(&family).map_or(0., |a| a.balance)
    }
    pub fn available(&self, family: Family, limit: f64) -> f64 {
        if self.accounts.len() == 256 && !self.accounts.contains_key(&family) {
            return f64::NEG_INFINITY;
        }
        let used = self.accounts.get(&family).map_or(0., |a| a.startup_spent);
        self.balance(family)
            + (limit - self.startup_spent)
                .max(0.)
                .min((self.family_limit.unwrap_or(limit) - used).max(0.))
    }
    pub fn restrict_family_startup(&mut self, limit: f64) {
        self.family_limit = Some(limit);
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
        let family_limit = self.family_limit.unwrap_or(limit);
        if let Some(account) = self.account(family) {
            account.balance = (account.balance + delta).min(limit);
            let subsidy = (-account.balance)
                .max(0.)
                .min(remaining)
                .min((family_limit - account.startup_spent).max(0.));
            account.balance += subsidy;
            account.startup_spent += subsidy;
            self.startup_spent += subsidy;
        }
    }
}
