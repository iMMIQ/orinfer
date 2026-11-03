//! CPU policy for bounded continuous batching. Capacity is independent of the
//! number of queued clients; physical memory and iteration cost govern admission.
use crate::artifact::Result;
use serde::Serialize;
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Policy {
    #[default]
    Latency,
    Throughput,
}
impl fmt::Display for Policy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Latency => "latency",
            Self::Throughput => "throughput",
        })
    }
}
impl FromStr for Policy {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "latency" => Ok(Self::Latency),
            "throughput" => Ok(Self::Throughput),
            _ => Err("Scheduler policy must be latency or throughput".into()),
        }
    }
}

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
    pub policy: Policy,
    pub max_active: usize,
    pub max_batch_tokens: usize,
    /// Total verification rows; zero uses the model package's legal capacity.
    pub max_speculative_tokens: usize,
    pub prefill_budget_ms: f64,
    pub target_tpot_ms: f64,
    pub memory_reserve_bytes: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            policy: Policy::Latency,
            max_active: 32,
            max_batch_tokens: 128,
            max_speculative_tokens: 0,
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
            || self.max_speculative_tokens > 4096
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
        .min_by(|(_, a), (_, b)| waiting_order(a, b))
        .map(|(index, _)| index)
}

pub(crate) fn waiting_order(a: &Waiting, b: &Waiting) -> std::cmp::Ordering {
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
}

/// Bounded, online output-length estimates. Completion limits are ceilings,
/// never training labels. Callers only observe naturally completed requests.
pub struct OutputLengths {
    bins: [(u64, f64); usize::BITS as usize],
}
impl Default for OutputLengths {
    fn default() -> Self {
        Self {
            bins: [(0, 0.); usize::BITS as usize],
        }
    }
}
impl OutputLengths {
    pub fn observe(&mut self, input: usize, output: usize) {
        if input == 0 || output == 0 {
            return;
        }
        let bin = &mut self.bins[input.ilog2() as usize];
        // Exponential forgetting handles language/task changes in a long file.
        bin.1 = if bin.0 == 0 {
            output as f64 / input as f64
        } else {
            bin.1 * 0.95 + output as f64 / input as f64 * 0.05
        };
        bin.0 = bin.0.saturating_add(1);
    }
    pub fn estimate(&self, input: usize, limit: usize) -> f64 {
        let key = input.max(1).ilog2() as usize;
        let ratio = self
            .bins
            .iter()
            .enumerate()
            .filter(|(_, (samples, _))| *samples > 0)
            .min_by_key(|(i, _)| i.abs_diff(key))
            .map_or(1., |(_, (_, ratio))| *ratio);
        (input as f64 * ratio).max(1.).min(limit as f64)
    }
}

/// Offline admission groups similar remaining output lengths, while starting
/// long work first when there is no existing cohort. Hard aging still wins.
pub fn rank_throughput_waiting(
    requests: &[Waiting],
    expected: &[f64],
    cohort_remaining: Option<f64>,
) -> Vec<usize> {
    assert_eq!(requests.len(), expected.len());
    let mut order: Vec<_> = (0..requests.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        let old_a = requests[a].age_s >= 30.;
        let old_b = requests[b].age_s >= 30.;
        old_b
            .cmp(&old_a)
            .then_with(|| {
                if old_a {
                    requests[b].age_s.total_cmp(&requests[a].age_s)
                } else {
                    let mismatch = |i: usize| {
                        cohort_remaining.map_or(0., |cohort| {
                            (expected[i].max(1.) / cohort.max(1.)).ln().abs()
                        })
                    };
                    mismatch(a)
                        .total_cmp(&mismatch(b))
                        .then_with(|| expected[b].total_cmp(&expected[a]))
                        .then_with(|| requests[a].cost_s().total_cmp(&requests[b].cost_s()))
                }
            })
            .then(a.cmp(&b))
    });
    order
}

/// EWMA plus a rolling positive-residual P95 allowance. This is a soft deadline
/// estimate, not a guarantee; invalid measurements never poison later decisions.
#[derive(Clone, Copy, Default)]
pub(crate) struct Estimate {
    pub mean: f64,
    error: f64,
    pub samples: usize,
    residuals: [f64; 32],
    residual_cursor: usize,
}
impl Estimate {
    pub fn observe(&mut self, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        if self.samples == 0 {
            self.mean = seconds;
        } else {
            self.residuals[self.residual_cursor % self.residuals.len()] =
                (seconds - self.mean).max(0.);
            self.residual_cursor += 1;
            self.error = self.error * 0.8 + (seconds - self.mean).abs() * 0.2;
            self.mean = self.mean * 0.8 + seconds * 0.2;
        }
        self.samples += 1;
    }
    pub fn upper(self) -> f64 {
        let count = self.residual_cursor.min(self.residuals.len());
        let percentile = if count == 0 {
            0.
        } else {
            let mut values = self.residuals;
            values[..count].sort_unstable_by(f64::total_cmp);
            values[(count * 95).div_ceil(100).saturating_sub(1)]
        };
        self.mean + (2. * self.error).max(percentile)
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

/// A cold joint-path prior must not fragment a native large tile. Once
/// measured, compare prompt progress against the native opportunity cost.
pub(crate) fn joint_prefill_worthwhile(
    loses_large_tile: bool,
    measured: bool,
    joint_s: f64,
    native_s: f64,
) -> bool {
    (!loses_large_tile || measured) && joint_s < native_s * 0.95
}

/// Wall time, rather than a fixed round count, also bounds MTP bursts.
pub(crate) fn decode_quantum_s(prefill_s: f64, options: &Options) -> f64 {
    prefill_s
        .min(options.prefill_budget_ms / 1000.)
        .min(options.target_tpot_ms / 1000.)
}

/// Cached text admissions can form a cohort before the first decode. During
/// ongoing generation, bound restoration work to protect inter-token latency.
pub fn admission_should_yield(
    has_decoders: bool,
    admitted: usize,
    elapsed_s: f64,
    batchable_start: bool,
) -> bool {
    if !batchable_start {
        // Build a small prompt cohort from work that has already arrived;
        // ongoing decoders retain the existing admission time bound.
        return if has_decoders {
            elapsed_s >= 0.010
        } else {
            admitted >= 8 || elapsed_s >= 0.200
        };
    }
    // Cached text and deferred image encoding can amortize startup in a cohort.
    // Callers also enforce the live decoder's remaining slack between owners.
    let (count, seconds) = if has_decoders { (8, 0.100) } else { (32, 3.0) };
    admitted >= count || elapsed_s >= seconds
}

/// Host timings; replay includes stream synchronization, not GPU event timing.
#[derive(Default, Clone, Copy, Debug, Serialize)]
pub struct BatchExecutionStatistics {
    pub sampling_packets: usize,
    pub sampling_rows: usize,
    pub workspace_mappings: usize,
    pub workspace_mapped_bytes: usize,
    pub workspace_mapping_s: f64,
    pub dynamic_graph_captures: usize,
    pub dynamic_direct_iterations: usize,
    pub direct_iterations: usize,
    pub graph_capture_deferrals: usize,
    pub graph_hits: usize,
    pub graph_misses: usize,
    pub graph_evictions: usize,
    pub graph_invalidations: usize,
    pub shared_graph_hits: usize,
    pub shared_graph_captures: usize,
    pub captured_operations: usize,
    pub capture_s: f64,
    pub eviction_s: f64,
    pub replay_s: f64,
    pub direct_s: f64,
    pub kernel_load_s: f64,
    pub sequence_captures: usize,
    pub sequence_capture_s: f64,
}

impl BatchExecutionStatistics {
    /// One-time materialization must not inflate the hot scheduling estimate.
    pub(crate) fn initialization_s(self) -> f64 {
        self.kernel_load_s + self.workspace_mapping_s
    }

    pub(crate) fn observation_since(self, before: Self) -> ExecutionObservation {
        let replay = self.graph_hits > before.graph_hits || self.replay_s > before.replay_s;
        let direct = self.direct_iterations > before.direct_iterations;
        ExecutionObservation {
            mode: match (direct, replay) {
                (true, true) => ExecutionMode::Mixed,
                (false, true) => ExecutionMode::Replay,
                _ => ExecutionMode::Direct,
            },
            preparing: self.initialization_s() > before.initialization_s()
                || self.capture_s > before.capture_s
                || self.sequence_capture_s > before.sequence_capture_s
                || self.graph_capture_deferrals > before.graph_capture_deferrals,
            evicted: self.graph_evictions > before.graph_evictions,
            // An eviction can remove another arm's graph. Invalidate its old
            // readiness too; its timing remains usable only as a cost prior.
            generation: self
                .graph_invalidations
                .saturating_add(self.graph_evictions),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ExecutionMode {
    #[default]
    Direct,
    Replay,
    Mixed,
}

/// Describes work that actually ran; a prior estimate cannot make a cold graph ready.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ExecutionObservation {
    pub mode: ExecutionMode,
    pub preparing: bool,
    pub evicted: bool,
    pub generation: usize,
}

#[derive(Default, Clone, Debug, Serialize)]
pub struct Statistics {
    pub mtp_probe_iterations: usize,
    pub mtp_request_depth_histogram: std::collections::BTreeMap<usize, usize>,
    pub maintenance_calls: usize,
    pub maintenance_s: f64,
    pub mtp_depth_histogram: std::collections::BTreeMap<usize, usize>,
    pub mtp_catchup_tokens: usize,
    pub mtp_catchup_s: f64,
    pub layered_prefill_iterations: usize,
    pub prefill_suspend_s: f64,
    pub prefill_resume_s: f64,
    pub peak_prefill_frame_bytes: usize,
    pub iterations: usize,
    pub batch_histogram: std::collections::BTreeMap<usize, usize>,
    pub prefill_batch_histogram: std::collections::BTreeMap<usize, usize>,
    pub prefill_tokens: usize,
    pub prefill_chunk_histogram: std::collections::BTreeMap<usize, usize>,
    pub bounded_prefill_iterations: usize,
    pub max_bounded_prefill_s: f64,
    pub decode_tokens: usize,
    pub mixed_iterations: usize,
    pub mixed_speculative_iterations: usize,
    pub speculative_iterations: usize,
    pub peak_active: usize,
    pub compute_s: f64,
    pub admission_deferrals: usize,
    pub admissions: usize,
    pub deadline_decode_iterations: usize,
    pub prefill_progress_overrides: usize,
    pub independent_prefill_choices: usize,
    pub joint_prefill_choices: usize,
    pub decode_burst_iterations: usize,
    pub max_decode_gap_s: f64,
    pub decode_budget_overruns: usize,
    pub request_start_s: f64,
    pub prefix_restore_s: f64,
    pub prefill_completion_s: f64,
    pub batch_inputs_s: f64,
    pub batch_plan_s: f64,
    pub batch_commit_s: f64,
    /// Model-wide speculative counters, published with this scheduler snapshot.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speculation: Option<crate::mtp::Statistics>,
    pub batch_execution: BatchExecutionStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lengths_learn_completions_and_cohorts_keep_aging() {
        let mut lengths = OutputLengths::default();
        assert_eq!(lengths.estimate(50, 128), lengths.estimate(50, 4096));
        for _ in 0..10 {
            lengths.observe(50, 10);
            lengths.observe(500, 200);
        }
        assert!((lengths.estimate(50, 4096) - 10.).abs() < 1e-6);
        assert!((lengths.estimate(500, 4096) - 200.).abs() < 1e-6);
        assert_eq!(lengths.estimate(500, 100), 100.);
        let mut requests = vec![
            Waiting {
                age_s: 0.,
                remaining_s: 1.,
                restore_s: 0.,
                startup_s: 0.,
                remaining_tokens: 50
            };
            3
        ];
        let expected = [10., 100., 40.];
        assert_eq!(
            rank_throughput_waiting(&requests, &expected, None),
            vec![1, 2, 0]
        );
        assert_eq!(
            rank_throughput_waiting(&requests, &expected, Some(35.)),
            vec![2, 1, 0]
        );
        requests[0].age_s = 31.;
        assert_eq!(
            rank_throughput_waiting(&requests, &expected, Some(35.))[0],
            0
        );
    }
    #[test]
    fn joint_prefill_must_pay_for_fragmenting_a_large_native_tile() {
        assert!(!joint_prefill_worthwhile(true, false, 0.2, 0.8));
        assert!(joint_prefill_worthwhile(true, true, 0.2, 0.8));
        assert!(!joint_prefill_worthwhile(true, true, 1.0, 0.8));
        assert!(joint_prefill_worthwhile(false, false, 0.08, 0.2));
        assert!(!joint_prefill_worthwhile(false, true, 0.2, 0.2));
    }
    #[test]
    fn decode_bursts_are_bounded_by_both_latency_and_prompt_budgets() {
        let mut options = Options::default();
        assert_eq!(decode_quantum_s(0.05, &options), 0.05);
        assert_eq!(decode_quantum_s(3.0, &options), 0.2);
        options.target_tpot_ms = 50.;
        assert_eq!(decode_quantum_s(3.0, &options), 0.05);
    }
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
    fn tail_margin_remembers_stalls_and_expires_after_a_bounded_window() {
        let mut estimate = Estimate::default();
        for _ in 0..40 {
            estimate.observe(0.1);
        }
        estimate.observe(0.9);
        estimate.observe(0.9);
        for _ in 0..12 {
            estimate.observe(0.1);
        }
        assert!(estimate.upper() > 0.65);
        for _ in 0..40 {
            estimate.observe(0.1);
        }
        assert!(estimate.upper() < 0.11);
        assert_eq!(estimate.residuals.len(), 32);
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
