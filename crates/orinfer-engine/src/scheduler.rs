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
    pub remaining_tokens: usize,
}

/// Compare a cold request's predicted first-token time under mixed execution
/// with letting the current decoder cohort finish first. Cached admissions do
/// not join this queue; aging bounds deferral even under continuous hot traffic.
#[derive(Default)]
pub struct AdmissionCosts {
    decode_rate_per_request: std::collections::BTreeMap<usize, f64>,
    mixed_prefill_rate: std::collections::BTreeMap<usize, f64>,
}
impl AdmissionCosts {
    pub fn observe(
        &mut self,
        decoders: usize,
        decode_tokens: usize,
        prefill_tokens: usize,
        seconds: f64,
    ) {
        if decoders == 0 || !seconds.is_finite() || seconds <= 0. {
            return;
        }
        let key = decoders.next_power_of_two();
        let (rates, rate) = if prefill_tokens > 0 {
            (
                &mut self.mixed_prefill_rate,
                prefill_tokens as f64 / seconds,
            )
        } else if decode_tokens > 0 {
            (
                &mut self.decode_rate_per_request,
                decode_tokens as f64 / seconds / decoders as f64,
            )
        } else {
            return;
        };
        rates
            .entry(key)
            .and_modify(|old| *old = *old * 0.8 + rate * 0.2)
            .or_insert(rate);
    }

    pub fn defer_cold(
        &self,
        waiting: Waiting,
        decoders: usize,
        remaining_decode_tokens: usize,
        has_prefills: bool,
    ) -> bool {
        if decoders == 0 || has_prefills || waiting.remaining_tokens == 0 || waiting.age_s >= 30. {
            return false;
        }
        let key = decoders.next_power_of_two();
        let Some(&decode_rate) = self.decode_rate_per_request.get(&key) else {
            return false;
        };
        // Cold-start prior: a few prompt tokens fit into one decoder interval.
        // Measured mixed iterations replace it as soon as they are available.
        let mixed_rate = self
            .mixed_prefill_rate
            .get(&key)
            .copied()
            .unwrap_or(decode_rate * 4.);
        let drain_s = remaining_decode_tokens as f64 / (decode_rate * decoders as f64);
        let queued_s = drain_s + waiting.remaining_s + waiting.restore_s;
        let mixed_s = waiting.remaining_tokens as f64 / mixed_rate + waiting.restore_s;
        queued_s.is_finite() && mixed_s.is_finite() && queued_s * 1.1 < mixed_s
    }
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
    fn aging_overrides_locality_and_restore_cost_is_charged() {
        let req = |age_s, remaining_s, restore_s| Waiting {
            age_s,
            remaining_s,
            restore_s,
            remaining_tokens: 0,
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

    #[test]
    fn cold_admission_uses_measured_cost_and_preserves_cache_hits_and_progress() {
        let mut costs = AdmissionCosts::default();
        let cold = Waiting {
            age_s: 0.,
            remaining_s: 2.5,
            restore_s: 0.,
            remaining_tokens: 2048,
        };
        assert!(!costs.defer_cold(cold, 1, 200, false));
        costs.observe(1, 15, 0, 0.6);
        assert!(costs.defer_cold(cold, 1, 200, false));
        assert!(!costs.defer_cold(cold, 1, 4000, false));
        assert!(!costs.defer_cold(cold, 1, 200, true));
        assert!(!costs.defer_cold(cold, 0, 0, false));
        assert!(!costs.defer_cold(Waiting { age_s: 30., ..cold }, 1, 200, false));
        assert!(!costs.defer_cold(
            Waiting {
                remaining_tokens: 0,
                remaining_s: 0.,
                ..cold
            },
            1,
            200,
            false
        ));
        // An efficient mixed backend must admit immediately instead of using
        // the conservative cold-start estimate forever.
        costs.observe(1, 1, 256, 0.2);
        assert!(!costs.defer_cold(cold, 1, 200, false));
        costs.observe(1, 0, 0, f64::NAN);
        assert!(!costs.defer_cold(cold, 1, 200, false));
    }
}
