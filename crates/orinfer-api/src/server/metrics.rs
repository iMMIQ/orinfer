//! Request wall phases and runtime counters for bounded serving diagnostics.
use super::*;
use std::fmt::Write;
const LIMITS: [f64; 16] = [
    0.001,
    0.01,
    0.1,
    0.5,
    1.,
    2.,
    5.,
    10.,
    20.,
    30.,
    60.,
    120.,
    300.,
    600.,
    1800.,
    f64::INFINITY,
];
#[derive(Clone, Default)]
struct Histogram {
    count: u64,
    sum: f64,
    buckets: [u64; 16],
}
impl Histogram {
    fn observe(&mut self, value: f64) {
        debug_assert!(value.is_finite() && value >= 0.);
        self.count += 1;
        self.sum += value;
        for (limit, bucket) in LIMITS.iter().zip(&mut self.buckets) {
            if value <= *limit {
                *bucket += 1;
            }
        }
    }
    fn render(&self, out: &mut String, name: &str, label: &str) {
        let _ = writeln!(out, "# TYPE {name} histogram");
        for (limit, count) in LIMITS.iter().zip(self.buckets) {
            let bound = if limit.is_infinite() {
                "+Inf".into()
            } else {
                limit.to_string()
            };
            let _ = writeln!(out, "{name}_bucket{{{label},le=\"{bound}\"}} {count}");
        }
        let _ = writeln!(
            out,
            "{name}_sum{{{label}}} {}\n{name}_count{{{label}}} {}",
            self.sum, self.count
        );
    }
}
#[derive(Clone, Default)]
pub(super) struct Measurements {
    histograms: [Histogram; 8],
    prompt_tokens: u64,
    cached_tokens: u64,
    output_tokens: u64,
}
pub(super) struct Completed {
    pub queue_s: f64,
    pub first_s: f64,
    pub last_s: f64,
    pub complete_s: f64,
    pub prompt_tokens: usize,
    pub output_tokens: usize,
}
impl Measurements {
    pub fn observe(&mut self, timing: Completed, prefix: &orinfer_engine::prefix::Statistics) {
        let values = [
            timing.queue_s,
            (timing.first_s - timing.queue_s).max(0.),
            (timing.last_s - timing.first_s).max(0.),
            (timing.complete_s - timing.last_s).max(0.),
            timing.first_s,
            timing.complete_s,
            prefix.restore_s,
            prefix.store_s,
        ];
        for (histogram, value) in self.histograms.iter_mut().zip(values) {
            histogram.observe(value);
        }
        self.prompt_tokens += timing.prompt_tokens as u64;
        self.cached_tokens += prefix.cached_tokens as u64;
        self.output_tokens += timing.output_tokens as u64;
    }
    fn render(
        &self,
        model: &str,
        active: usize,
        queued: usize,
        statistics: &orinfer_engine::scheduler::Statistics,
    ) -> String {
        let escaped = model
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n");
        let label = format!("model=\"{escaped}\"");
        let mut out = String::new();
        for (histogram, name) in self.histograms.iter().zip([
            "queue",
            "prefill",
            "decode",
            "finalize",
            "time_to_first_token",
            "latency",
            "prefix_restore",
            "prefix_store",
        ]) {
            histogram.render(&mut out, &format!("orinfer_request_{name}_seconds"), &label);
        }
        for (name, value) in [
            ("orinfer_prompt_tokens_total", self.prompt_tokens),
            ("orinfer_cached_prompt_tokens_total", self.cached_tokens),
            ("orinfer_generated_tokens_total", self.output_tokens),
        ] {
            let _ = writeln!(out, "# TYPE {name} counter\n{name}{{{label}}} {value}");
        }
        for (name, value) in [
            ("orinfer_requests_running", active),
            ("orinfer_requests_waiting", queued),
        ] {
            let _ = writeln!(out, "# TYPE {name} gauge\n{name}{{{label}}} {value}");
        }
        for (name, value) in [
            ("orinfer_compute_seconds_total", statistics.compute_s),
            ("orinfer_batch_plan_seconds_total", statistics.batch_plan_s),
            (
                "orinfer_admission_seconds_total",
                statistics.request_start_s,
            ),
            (
                "orinfer_prefix_restore_seconds_total",
                statistics.prefix_restore_s,
            ),
        ] {
            let _ = writeln!(out, "# TYPE {name} counter\n{name}{{{label}}} {value}");
        }
        if let Some(health) = &statistics.backend_health {
            for (name, value) in [
                (
                    "orinfer_device_rank_free_bytes_min",
                    health.min_rank_free_bytes,
                ),
                (
                    "orinfer_device_rank_free_bytes_max",
                    health.max_rank_free_bytes,
                ),
                (
                    "orinfer_prefix_cache_device_entries",
                    health.device_prefix_entries,
                ),
                (
                    "orinfer_prefix_cache_device_bytes",
                    health.device_prefix_bytes,
                ),
                (
                    "orinfer_prefix_cache_host_entries",
                    health.host_prefix_entries,
                ),
                ("orinfer_prefix_cache_host_bytes", health.host_prefix_bytes),
                ("orinfer_dspark_cost_cells", health.dspark_cost_cells),
            ] {
                let _ = writeln!(out, "# TYPE {name} gauge\n{name}{{{label}}} {value}");
            }
            for (name, value) in [
                (
                    "orinfer_prefix_cache_host_evict_max_seconds",
                    health.host_evict_max_s,
                ),
                (
                    "orinfer_dspark_budget_balance_seconds",
                    health.dspark_budget_balance_s,
                ),
            ] {
                let _ = writeln!(out, "# TYPE {name} gauge\n{name}{{{label}}} {value}");
            }
            for (name, value) in [
                (
                    "orinfer_prefix_cache_evictions_total",
                    health.prefix_evictions,
                ),
                (
                    "orinfer_prefix_cache_host_evictions_total",
                    health.host_prefix_evictions,
                ),
                (
                    "orinfer_prefix_cache_save_skips_total",
                    health.prefix_save_skips,
                ),
                ("orinfer_program_reclaims_total", health.program_reclaims),
                (
                    "orinfer_sampling_request_failures_total",
                    health.sampling_request_failures,
                ),
            ] {
                let _ = writeln!(out, "# TYPE {name} counter\n{name}{{{label}}} {value}");
            }
        }
        out
    }
}
pub(super) async fn endpoint(State(state): State<Service>, headers: HeaderMap) -> Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    let measurements = match state.activity.measurements.lock() {
        Ok(value) => value.clone(),
        Err(_) => return error(StatusCode::SERVICE_UNAVAILABLE, "Metrics unavailable"),
    };
    let statistics = state
        .activity
        .statistics
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default();
    let body = measurements.render(
        &state.model,
        state.activity.active.load(Ordering::Relaxed),
        state.activity.queued.load(Ordering::Relaxed),
        &statistics,
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cumulative_buckets_and_nonoverlapping_wall_phases_keep_their_scope() {
        let mut metrics = Measurements::default();
        metrics.observe(
            Completed {
                queue_s: 0.25,
                first_s: 1.,
                last_s: 3.,
                complete_s: 3.5,
                prompt_tokens: 100,
                output_tokens: 8,
            },
            &orinfer_engine::prefix::Statistics {
                cached_tokens: 75,
                restore_s: 0.1,
                store_s: 0.2,
                ..Default::default()
            },
        );
        metrics.observe(
            Completed {
                queue_s: 0.1,
                first_s: 0.2,
                last_s: 0.2,
                complete_s: 0.5,
                prompt_tokens: 50,
                output_tokens: 1,
            },
            &Default::default(),
        );
        assert!((metrics.histograms[..4].iter().map(|h| h.sum).sum::<f64>() - 4.).abs() < 1e-12);
        let text = metrics.render("test\"model", 2, 3, &Default::default());
        assert!(text.contains("orinfer_request_latency_seconds_count{model=\"test\\\"model\"} 2"));
        assert!(text.contains(
            "orinfer_request_latency_seconds_bucket{model=\"test\\\"model\",le=\"+Inf\"} 2"
        ));
        assert!(text.contains("orinfer_cached_prompt_tokens_total{model=\"test\\\"model\"} 75"));
        let stats = orinfer_engine::scheduler::Statistics {
            backend_health: Some(orinfer_engine::scheduler::BackendHealthStatistics {
                min_rank_free_bytes: 10 << 30,
                max_rank_free_bytes: 12 << 30,
                device_prefix_entries: 4,
                device_prefix_bytes: 1 << 30,
                host_prefix_entries: 100,
                host_prefix_bytes: 32 << 30,
                prefix_evictions: 7,
                host_prefix_evictions: 11,
                host_evict_max_s: 0.015,
                prefix_save_skips: 2,
                program_reclaims: 3,
                sampling_request_failures: 1,
                dspark_cost_cells: 512,
                dspark_budget_balance_s: -0.25,
            }),
            ..Default::default()
        };
        let rendered = metrics.render("dsv41", 4, 2, &stats);
        for metric in [
            "orinfer_device_rank_free_bytes_min{model=\"dsv41\"}",
            "orinfer_prefix_cache_host_evict_max_seconds{model=\"dsv41\"} 0.015",
            "orinfer_prefix_cache_save_skips_total{model=\"dsv41\"} 2",
            "orinfer_sampling_request_failures_total{model=\"dsv41\"} 1",
            "orinfer_dspark_budget_balance_seconds{model=\"dsv41\"} -0.25",
        ] {
            assert!(rendered.contains(metric), "missing {metric}");
        }
    }
}
