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
    }
}
