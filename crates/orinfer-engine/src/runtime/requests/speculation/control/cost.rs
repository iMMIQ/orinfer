//! CUDA timing identities; shared samples do not own execution resources.
use super::{Key, Measurement};
pub(super) use crate::mtp::adaptive::Cost;
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct Profile {
    pub arms: BTreeMap<usize, Cost>,
    pub calls: usize,
    last: usize,
}
#[derive(Default)]
pub(super) struct Costs {
    pub profiles: BTreeMap<Key, Profile>,
    clock: usize,
}
impl Costs {
    pub fn normal(&self, key: Key) -> Option<&Cost> {
        self.profiles.get(&key)?.arms.get(&1)
    }
    pub fn observe(&mut self, key: Key, width: usize, measurement: Measurement) {
        if self.profiles.len() == 256 && !self.profiles.contains_key(&key) {
            let victim = *self.profiles.iter().min_by_key(|(_, p)| p.last).unwrap().0;
            self.profiles.remove(&victim);
        }
        self.clock += 1;
        let profile = self.profiles.entry(key).or_default();
        profile.calls += 1;
        profile.last = self.clock;
        profile
            .arms
            .entry(width)
            .or_default()
            .observe(measurement, profile.calls);
    }
    pub fn prior(&self, key: Key, width: usize, fallback: f64) -> f64 {
        self.profiles
            .iter()
            .filter(|(other, _)| {
                other.family() == key.family()
                    && other.prefill == key.prefill
                    && other.context.abs_diff(key.context) <= 1
            })
            .filter_map(|(other, profile)| {
                let cost = profile.arms.get(&width)?;
                (cost.time.samples() >= 3)
                    .then_some((other.context.abs_diff(key.context), cost.upper()))
            })
            .min_by_key(|(distance, _)| *distance)
            .map_or(fallback, |(_, seconds)| seconds)
    }
}
