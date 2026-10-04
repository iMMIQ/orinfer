//! CPU policy for bounded continuous batching. Capacity is independent of the
//! number of queued clients; physical memory and iteration cost govern admission.
use crate::artifact::Result;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Options {
    pub max_active: usize,
    pub max_batch_tokens: usize,
    pub prefill_budget_ms: f64,
    pub memory_reserve_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_active: 32,
            max_batch_tokens: 128,
            prefill_budget_ms: 200.,
            memory_reserve_bytes: 1 << 30,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<()> {
        if !(1..=128).contains(&self.max_active)
            || !(1..=128).contains(&self.max_batch_tokens)
            || !self.prefill_budget_ms.is_finite()
            || self.prefill_budget_ms <= 0.
        {
            return Err(
                "Scheduler requires active/batch limits in 1..128 and positive prefill budget"
                    .into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Waiting {
    pub age_s: f64,
    pub remaining_s: f64,
    pub restore_s: f64,
}
/// Aging takes precedence after two seconds; fresh work uses estimated cost,
/// including expensive small-shape tails and prefix restoration.
pub fn select_waiting(requests: &[Waiting]) -> Option<usize> {
    requests
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let old_a = a.age_s >= 2.;
            let old_b = b.age_s >= 2.;
            old_b.cmp(&old_a).then_with(|| {
                if old_a {
                    b.age_s.total_cmp(&a.age_s)
                } else {
                    (a.remaining_s + a.restore_s)
                        .total_cmp(&(b.remaining_s + b.restore_s))
                        .then(b.age_s.total_cmp(&a.age_s))
                }
            })
        })
        .map(|(index, _)| index)
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct Statistics {
    pub iterations: usize,
    pub batch_histogram: std::collections::BTreeMap<usize, usize>,
    pub prefill_tokens: usize,
    pub decode_tokens: usize,
    pub mixed_iterations: usize,
    pub speculative_iterations: usize,
    pub peak_active: usize,
    pub compute_s: f64,
    pub admission_deferrals: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aging_overrides_locality_and_restore_cost_is_charged() {
        let req = |age_s, remaining_s, restore_s| Waiting {
            age_s,
            remaining_s,
            restore_s,
        };
        assert_eq!(select_waiting(&[]), None);
        assert_eq!(
            select_waiting(&[req(0.1, 0., 0.5), req(0.2, 0.2, 0.)]),
            Some(1)
        );
        assert_eq!(
            select_waiting(&[req(3., 10., 1.), req(1., 0., 0.)]),
            Some(0)
        );
        assert_eq!(
            select_waiting(&[req(3., 0., 0.), req(4., 10., 0.)]),
            Some(1)
        );
    }
}
