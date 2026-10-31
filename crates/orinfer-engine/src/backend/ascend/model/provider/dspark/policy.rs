//! Runtime cost feedback uses completed rounds, never future proposal draws.
use std::collections::BTreeMap;

#[derive(Default)]
pub(in crate::backend::ascend::model::provider) struct Policy {
    curves: BTreeMap<(usize, bool), Curve>,
}
struct Curve {
    baseline: f64,
    baseline_samples: usize,
    samples: usize,
    ticks: usize,
    draft: f64,
    extra: f64,
    restore: f64,
    acceptance: [f64; 5],
    previous_confidence: [f64; 5],
    older_confidence: [f64; 5],
}
impl Default for Curve {
    fn default() -> Self {
        Self {
            baseline: 0.,
            baseline_samples: 0,
            samples: 0,
            ticks: 0,
            draft: 0.,
            extra: 0.,
            restore: 0.,
            acceptance: [0.7; 5],
            previous_confidence: [0.7; 5],
            older_confidence: [0.7; 5],
        }
    }
}
fn update(old: &mut f64, sample: f64, initialized: bool) {
    *old = if initialized {
        *old * 0.8 + sample * 0.2
    } else {
        sample
    };
}
fn update_cost(old: &mut f64, sample: f64, initialized: bool) {
    if !initialized || sample < *old {
        // Once Graph replaces direct execution, discard its startup cost.
        *old = sample;
    } else {
        update(old, sample, true);
    }
}
impl Policy {
    pub fn baseline(&mut self, batch: usize, full: bool, seconds: f64) {
        if !seconds.is_finite() || seconds <= 0. {
            return;
        }
        let curve = self.curves.entry((batch, full)).or_default();
        update_cost(&mut curve.baseline, seconds, curve.baseline_samples > 0);
        curve.baseline_samples += 1;
    }
    pub fn select(&mut self, batch: usize, full: bool, maximum: usize) -> usize {
        let curve = self.curves.entry((batch, full)).or_default();
        curve.ticks += 1;
        if curve.baseline_samples < 4 {
            return 0;
        }
        // Amortize the common graph's initial direct runs before judging it.
        if curve.samples < 8 {
            return maximum;
        }
        if curve.ticks.is_multiple_of(32) {
            return 0;
        }
        let mut best = 0;
        let mut cost = curve.baseline * 0.97;
        let mut survival = 1.;
        let mut outputs = 1.;
        for k in 1..=maximum.min(5) {
            survival *=
                ((curve.acceptance[k - 1] + curve.older_confidence[k - 1]) * 0.5).clamp(0., 1.);
            outputs += survival;
            let estimate =
                (curve.draft + curve.baseline + curve.extra * k as f64 + curve.restore) / outputs;
            if estimate < cost {
                best = k;
                cost = estimate;
            }
        }
        // Bounded probes let a changed workload recover from a prior fallback.
        if best == 0 && curve.ticks.is_multiple_of(16) {
            maximum
        } else {
            best
        }
    }
    pub fn observe(
        &mut self,
        batch: usize,
        full: bool,
        timing: [f64; 3],
        counts: &[(usize, usize)],
        confidence: [f64; 5],
    ) {
        let curve = self.curves.entry((batch, full)).or_default();
        let proposed = counts.iter().map(|p| p.0).sum::<usize>() as f64 / batch as f64;
        if proposed <= 0. {
            return;
        }
        update_cost(&mut curve.draft, timing[0], curve.samples > 0);
        update_cost(
            &mut curve.extra,
            ((timing[1] - curve.baseline) / proposed).max(0.),
            curve.samples > 0,
        );
        update_cost(&mut curve.restore, timing[2], curve.samples > 0);
        for i in 0..5 {
            let attempts = counts
                .iter()
                .filter(|&&(k, accepted)| k > i && accepted >= i)
                .count();
            let passed = counts
                .iter()
                .filter(|&&(k, accepted)| k > i && accepted > i)
                .count();
            if attempts > 0 {
                update(
                    &mut curve.acceptance[i],
                    passed as f64 / attempts as f64,
                    curve.samples > 0,
                );
            }
        }
        curve.older_confidence = curve.previous_confidence;
        curve.previous_confidence = confidence.map(|p| p.clamp(0., 1.));
        curve.samples += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn measured(confidence: [f64; 5], counts: &[(usize, usize)], timing: [f64; 3]) -> Policy {
        let mut p = Policy::default();
        for _ in 0..4 {
            p.baseline(1, false, 0.020);
        }
        for _ in 0..16 {
            p.observe(1, false, timing, counts, confidence);
        }
        p
    }
    #[test]
    fn profitable_blocks_stay_enabled_and_unprofitable_sampling_falls_back() {
        let mut fast = measured([0.9; 5], &[(5, 5)], [0.003, 0.025, 0.001]);
        assert_eq!(fast.select(1, false, 5), 5);
        let mut slow = measured([0.01; 5], &[(5, 0)], [0.010, 0.040, 0.003]);
        assert_eq!(slow.select(1, false, 5), 0);
        assert_eq!(fast.select(1, true, 5), 0); // Independent full-logits cost.
        assert_eq!(fast.select(32, false, 5), 0); // Independent concurrency.
    }
    #[test]
    fn cheap_prefix_wins_when_later_proposals_rarely_accept() {
        let mut p = measured(
            [0.9, 0.01, 0.01, 0.01, 0.01],
            &[(5, 1)],
            [0.003, 0.030, 0.001],
        );
        assert_eq!(p.select(1, false, 5), 1);
    }
    #[test]
    fn graph_cost_replaces_cold_direct_cost() {
        let mut p = measured([0.99; 5], &[(5, 5)], [0.025, 0.200, 0.001]);
        assert_eq!(p.select(1, false, 5), 0);
        p.observe(1, false, [0.007, 0.036, 0.001], &[(5, 5)], [0.99; 5]);
        assert_eq!(p.select(1, false, 5), 5);
    }
}
