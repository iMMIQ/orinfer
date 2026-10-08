//! CPU policy for bounded continuous batching. Capacity is independent of the
//! number of queued clients; physical memory and iteration cost govern admission.
use crate::artifact::Result;
use serde::Serialize;

/// Orin CUDA allocations consume system RAM. CUDA's free-memory counter alone
/// does not protect the host or account for other processes' resident memory.
/// Keep one sixteenth of physical RAM for the host, in addition to the caller's
/// configurable admission reserve. Read MemAvailable, which includes RAM the
/// kernel can reclaim, rather than MemFree or the amount of swap.
pub(crate) fn usable_host_bytes() -> Result<usize> {
    usable_host_bytes_from_meminfo(
        &std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("Read host memory budget: {e}"))?,
    )
}

fn usable_host_bytes_from_meminfo(text: &str) -> Result<usize> {
    let (mut total, mut available) = (None, None);
    for line in text.lines() {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let target = match name {
            "MemTotal" => &mut total,
            "MemAvailable" => &mut available,
            _ => continue,
        };
        let mut fields = value.split_whitespace();
        let bytes = fields
            .next()
            .and_then(|v| v.parse::<usize>().ok())
            .and_then(|n| n.checked_mul(1024))
            .ok_or("Invalid host memory budget")?;
        if fields.next() != Some("kB") || fields.next().is_some() || target.replace(bytes).is_some()
        {
            return Err("Invalid host memory budget".into());
        }
    }
    let total = total.filter(|&n| n > 0).ok_or("Missing MemTotal")?;
    let available = available
        .filter(|&n| n <= total)
        .ok_or("Invalid or missing MemAvailable")?;
    Ok(available.saturating_sub(total / 16))
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Options {
    pub max_active: usize,
    pub max_batch_tokens: usize,
    pub prefill_budget_ms: f64,
    pub target_tpot_ms: f64,
    pub memory_reserve_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            max_active: 32,
            max_batch_tokens: 128,
            prefill_budget_ms: 200.,
            target_tpot_ms: 400.,
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
            || !self.target_tpot_ms.is_finite()
            || self.target_tpot_ms <= 0.
        {
            return Err(
                "Scheduler requires active/batch limits in 1..128 and positive scheduling time budgets"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Immutable queue descriptor. Media hashes are calculated once, while cache
/// matches and timing estimates remain live as checkpoints arrive or disappear.
pub struct RequestHint {
    pub(crate) owner: u64,
    pub(crate) tokens: Vec<u32>,
    pub(crate) media: crate::prefix::Media,
    pub(crate) image_work: usize,
}
impl RequestHint {
    pub fn common_tokens(&self, other: &Self) -> usize {
        if self.owner != other.owner {
            return 0;
        }
        let common = self
            .tokens
            .iter()
            .zip(&other.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        self.media.common_tokens(&other.media, common)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Waiting {
    pub age_s: f64,
    pub remaining_s: f64,
    pub restore_s: f64,
    pub startup_s: f64,
    pub remaining_tokens: usize,
}
impl Waiting {
    pub fn cost_s(self) -> f64 {
        (self.remaining_s + self.restore_s + self.startup_s).max(0.001)
    }
}

/// Smooth aging preserves cost ordering under sustained load. Only the hard
/// promotion horizon changes to oldest-first; unlike the old two-second rule,
/// a burst does not immediately collapse the entire queue to FCFS.
pub fn select_waiting(requests: &[Waiting]) -> Option<usize> {
    requests
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let old_a = a.age_s >= 30.;
            let old_b = b.age_s >= 30.;
            old_b.cmp(&old_a).then_with(|| {
                if old_a {
                    b.age_s.total_cmp(&a.age_s)
                } else {
                    (a.cost_s() / (1. + a.age_s / 2.))
                        .total_cmp(&(b.cost_s() / (1. + b.age_s / 2.)))
                        .then(b.age_s.total_cmp(&a.age_s))
                }
            })
        })
        .map(|(index, _)| index)
}

/// EWMA plus an error allowance, rather than treating a mean as a deadline
/// guarantee. Invalid measurements never poison subsequent scheduling.
#[derive(Clone, Copy, Default)]
pub(crate) struct Estimate {
    pub mean: f64,
    error: f64,
    pub samples: usize,
}
impl Estimate {
    pub fn observe(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        if self.samples == 0 {
            self.mean = seconds;
        } else {
            self.error = self.error * 0.8 + (seconds - self.mean).abs() * 0.2;
            self.mean = self.mean * 0.8 + seconds * 0.2;
        }
        self.samples += 1;
    }
    pub fn upper(self) -> f64 {
        self.mean + 2. * self.error
    }
}

/// Prefill requests spend real wall time, not iteration counts. Deficits may
/// become negative after a large indivisible chunk; refill enough rounds for
/// some request to become eligible without spinning or leaving the GPU idle.
pub(crate) fn replenish(credits: &mut [f64], quantum_s: f64) {
    let highest = credits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if highest <= 0. && highest.is_finite() {
        let refill = ((-highest / quantum_s).floor() + 1.) * quantum_s;
        for credit in credits {
            *credit += refill;
        }
    }
}

/// Largest useful prefill window after reserving the next decoder step. A
/// minimum execution quantum bounds starvation when the target is infeasible.
pub(crate) fn prefill_window(slack_s: f64, decode_s: f64, ceiling_ms: f64) -> f64 {
    (slack_s - decode_s).max(0.).min(ceiling_ms / 1000.)
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
        // Build a small prompt cohort from work that has already arrived;
        // ongoing decoders retain the existing admission time bound.
        return if has_decoders {
            elapsed_s >= 0.010
        } else {
            admitted >= 8 || elapsed_s >= 0.200
        };
    }
    // Initial CUDA arena allocation is slower than reusing warm arenas. Allow
    // a bounded cold cohort while no decoder is waiting for its next token.
    let (count, seconds) = if has_decoders { (8, 0.100) } else { (32, 3.0) };
    admitted >= count || elapsed_s >= seconds
}

/// Host timings; replay includes stream synchronization, not GPU event timing.
#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct BatchExecutionStatistics {
    pub dynamic_graph_captures: usize,
    pub dynamic_direct_iterations: usize,
    pub graph_capture_deferrals: usize,
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
    pub prefill_batch_histogram: std::collections::BTreeMap<usize, usize>,
    pub prefill_tokens: usize,
    pub prefill_chunk_histogram: std::collections::BTreeMap<usize, usize>,
    pub bounded_prefill_iterations: usize,
    pub max_bounded_prefill_s: f64,
    pub decode_tokens: usize,
    pub mixed_iterations: usize,
    pub speculative_iterations: usize,
    pub peak_active: usize,
    pub compute_s: f64,
    pub admission_deferrals: usize,
    pub admissions: usize,
    pub deadline_decode_iterations: usize,
    pub prefill_progress_overrides: usize,
    pub max_decode_gap_s: f64,
    pub decode_budget_overruns: usize,
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
    fn unified_memory_admission_preserves_host_headroom() {
        let total_kib = 64 * 1024 * 1024;
        let usable = |available_kib| {
            usable_host_bytes_from_meminfo(&format!(
                "MemTotal: {total_kib} kB\nMemFree: 0 kB\nMemAvailable: {available_kib} kB\nSwapFree: 999999999 kB\n"
            ))
            .unwrap()
        };
        // A large CUDA free-memory reading must not admit work when Linux is
        // near its low-memory threshold. Swap cannot fund device allocations.
        assert_eq!((16usize << 30).min(usable(3 * 1024 * 1024)), 0);
        let free = (16usize << 30).min(usable(6 * 1024 * 1024));
        assert_eq!(free.saturating_sub(1 << 30), 1 << 30);
        assert_eq!(usable(20 * 1024 * 1024), 16 << 30);
        for bad in [
            "MemTotal: 1 kB",
            "MemTotal: 1 kB\nMemAvailable: 2 kB",
            "MemTotal: 10 MB\nMemAvailable: 1 kB",
            "MemTotal: 10 kB\nMemAvailable: 1 kB\nMemAvailable: 2 kB",
            "MemTotal: 0 kB\nMemAvailable: 0 kB",
        ] {
            assert!(usable_host_bytes_from_meminfo(bad).is_err());
        }
    }
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
        assert!(admission_should_yield(true, 1, 0.020, false));
        assert!(!admission_should_yield(true, 1, 0.005, false));
        assert!(!admission_should_yield(false, 7, 0.190, false));
        assert!(admission_should_yield(false, 8, 0.190, false));
        assert!(admission_should_yield(false, 1, 0.201, false));
    }
    #[test]
    fn aging_preserves_cost_order_and_eventually_promotes_long_work() {
        let req = |age_s, remaining_s, restore_s| Waiting {
            age_s,
            remaining_s,
            restore_s,
            startup_s: 0.,
            remaining_tokens: 0,
        };
        assert_eq!(select_waiting(&[]), None);
        assert_eq!(
            select_waiting(&[req(3., 10., 1.), req(2., 0.2, 0.)]),
            Some(1)
        );
        assert_eq!(
            select_waiting(&[req(31., 10., 1.), req(2., 0.2, 0.)]),
            Some(0)
        );
        assert_eq!(
            select_waiting(&[req(31., 0., 0.), req(32., 10., 0.)]),
            Some(1)
        );
        assert_eq!(
            select_waiting(&[req(0.1, 0., 0.5), req(0.2, 0.2, 0.)]),
            Some(1)
        );
        let expensive_start = Waiting {
            startup_s: 2.,
            ..req(0., 0., 0.)
        };
        assert_eq!(
            select_waiting(&[expensive_start, req(0., 0.1, 0.)]),
            Some(1)
        );
    }
    #[test]
    fn real_service_deficits_charge_large_chunks_without_idling() {
        let mut credits = [0., 0.];
        replenish(&mut credits, 0.2);
        credits[0] -= 2.;
        credits[1] -= 0.3;
        replenish(&mut credits, 0.2);
        assert!(credits[0] < 0. && credits[1] > 0.);
        credits[1] -= 2.;
        replenish(&mut credits, 0.2);
        assert!(credits.iter().any(|&c| c > 0.));
        let old = credits;
        replenish(&mut credits, 0.2);
        assert_eq!(old, credits);
    }
    #[test]
    fn timing_reserves_decoder_time_and_accounts_for_prediction_error() {
        assert_eq!(prefill_window(0.3, 0.1, 200.), 0.19999999999999998);
        assert_eq!(prefill_window(0.05, 0.1, 200.), 0.);
        let mut estimate = Estimate::default();
        estimate.observe(0.1);
        estimate.observe(0.4);
        assert!(estimate.upper() > estimate.mean);
        let before = estimate.upper();
        estimate.observe(f64::NAN);
        assert_eq!(estimate.upper(), before);
        let options = Options {
            target_tpot_ms: f64::NAN,
            ..Default::default()
        };
        assert!(options.validate().is_err());
    }
    #[test]
    fn queue_locality_never_crosses_image_identity_or_model_owners() {
        let hint = |owner, image| RequestHint {
            owner,
            tokens: vec![1, 2, 3, 4],
            media: crate::prefix::Media(vec![(2, [image; 32])]),
            image_work: 16,
        };
        assert_eq!(hint(1, 0).common_tokens(&hint(1, 0)), 4);
        assert_eq!(hint(1, 0).common_tokens(&hint(1, 1)), 2);
        assert_eq!(hint(1, 0).common_tokens(&hint(2, 0)), 0);
    }
}
