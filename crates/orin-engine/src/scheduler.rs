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

/// Cached text admissions can form a cohort before the first decode. During
/// ongoing generation, bound restoration work to protect inter-token latency.
pub fn admission_should_yield(
    has_decoders: bool,
    admitted: usize,
    elapsed_s: f64,
    cached_text: bool,
) -> bool {
    if !cached_text {
        return elapsed_s >= 0.010;
    }
    // Initial CUDA arena allocation is slower than reusing warm arenas. Allow
    // a bounded cold cohort while no decoder is waiting for its next token.
    let (count, seconds) = if has_decoders { (8, 0.100) } else { (32, 3.0) };
    admitted >= count || elapsed_s >= seconds
}

/// Host timings; replay includes stream synchronization, not GPU event timing.
#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct BatchExecutionStatistics {
    pub graph_hits: usize,
    pub graph_misses: usize,
    pub graph_evictions: usize,
    pub graph_invalidations: usize,
    pub captured_operations: usize,
    pub capture_s: f64,
    pub eviction_s: f64,
    pub replay_s: f64,
    pub direct_s: f64,
    pub sequence_captures: usize,
    pub sequence_capture_s: f64,
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
    pub admissions: usize,
    pub request_start_s: f64,
    pub prefix_restore_s: f64,
    pub prefill_completion_s: f64,
    pub batch_inputs_s: f64,
    pub batch_plan_s: f64,
    pub batch_commit_s: f64,
    pub batch_execution: BatchExecutionStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_cohorts_are_bounded_without_delaying_decoders_for_cold_work() {
        assert!(!admission_should_yield(false, 1, 0.020, true));
        assert!(!admission_should_yield(false, 31, 0.900, true));
        assert!(admission_should_yield(false, 32, 0.900, true));
        assert!(!admission_should_yield(false, 20, 1.500, true));
        assert!(admission_should_yield(false, 2, 3.001, true));
        assert!(!admission_should_yield(true, 4, 0.080, true));
        assert!(admission_should_yield(true, 8, 0.080, true));
        assert!(admission_should_yield(true, 4, 0.101, true));
        for active in [false, true] {
            assert!(admission_should_yield(active, 1, 0.020, false));
            assert!(!admission_should_yield(active, 1, 0.005, false));
        }
    }
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
