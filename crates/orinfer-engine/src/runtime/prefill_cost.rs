//! Estimate the actual greedy prefill partition, including its expensive tails.
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Costs {
    // None means this shape has not yet been measured on this model instance.
    shapes: BTreeMap<usize, Option<f64>>,
    weight_pass_s: f64,
    restore_s_per_byte: f64,
}
impl Costs {
    pub fn new(shapes: impl IntoIterator<Item = usize>) -> Self {
        let mut shapes: BTreeMap<_, _> = shapes.into_iter().map(|n| (n, None)).collect();
        shapes.insert(1, None);
        Self {
            shapes,
            // Conservative cold-start prior; online timings replace it.
            weight_pass_s: 0.1,
            restore_s_per_byte: 1.0 / 20e9,
        }
    }
    pub fn observe(&mut self, chunk: usize, seconds: f64) {
        if seconds.is_finite() && seconds > 0. {
            if chunk <= 8 {
                self.weight_pass_s = self.weight_pass_s * 0.8 + seconds * 0.2;
            }
            let cost = self.shapes.entry(chunk).or_default();
            *cost = Some(cost.map_or(seconds, |old| old * 0.8 + seconds * 0.2));
        }
    }
    pub fn observe_restore(&mut self, bytes: usize, seconds: f64) {
        if bytes > 0 && seconds.is_finite() && seconds > 0. {
            self.restore_s_per_byte = self.restore_s_per_byte * 0.8 + seconds / bytes as f64 * 0.2;
        }
    }
    pub fn restore_cost(&self, bytes: usize) -> f64 {
        bytes as f64 * self.restore_s_per_byte
    }
    pub fn chunk_cost(&self, tokens: usize) -> f64 {
        self.shapes
            .get(&tokens)
            .copied()
            .flatten()
            .unwrap_or(self.weight_pass_s * (tokens as f64 / 64.).max(1.))
    }
    /// Largest real profile within the estimated budget; one token guarantees
    /// progress even when no profile fits. This is a soft scheduling target.
    pub fn bounded_chunk(&self, remaining: usize, budget_ms: f64) -> usize {
        self.shapes
            .keys()
            .copied()
            .filter(|&n| n <= remaining && self.chunk_cost(n) * 1000. <= budget_ms)
            .max()
            .unwrap_or(1)
    }
    fn span(&self, mut tokens: usize) -> f64 {
        let mut total = 0.;
        for (&chunk, &seconds) in self.shapes.iter().rev() {
            let count = tokens / chunk;
            tokens %= chunk;
            let seconds = seconds.unwrap_or(self.weight_pass_s * (chunk as f64 / 64.).max(1.));
            total += count as f64 * seconds;
        }
        total
    }
    pub fn remaining(&self, mut offset: usize, end: usize) -> f64 {
        let mut total = 0.;
        while offset < end {
            let boundary = ((offset / 8192 + 1) * 8192).min(end);
            total += self.span(boundary - offset);
            offset = boundary;
        }
        total
    }
    pub fn score(&self, prefix: usize, end: usize, bytes: usize) -> Option<f64> {
        let cost = self.remaining(prefix, end) + bytes as f64 * self.restore_s_per_byte;
        // A complete head-bearing checkpoint always avoids model execution.
        (prefix == end || cost < self.remaining(0, end) * 0.95).then_some(cost)
    }
    pub fn admit(
        &self,
        prefix: usize,
        end: usize,
        bytes: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
    ) -> bool {
        if prefix <= start || self.score(prefix, end, bytes).is_none() {
            return false;
        }
        let mut trial = checkpoints.clone();
        trial.insert(prefix);
        let (mut previous, mut split) = (start, 0.);
        for &point in trial.range((
            std::ops::Bound::Excluded(start),
            std::ops::Bound::Excluded(end),
        )) {
            split += self.remaining(previous, point);
            previous = point;
        }
        split += self.remaining(previous, end);
        let baseline = self.remaining(start, end);
        // Do not create a speculative boundary by destroying an efficient
        // large chunk. Periodic checkpoints are already chunk boundaries.
        // Account from the restored endpoint and include already admitted
        // boundaries, rather than evaluating each split against a cold run.
        split - baseline <= (baseline * 0.1).max(0.05)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_small_prefixes_that_destroy_large_chunks() {
        let costs = Costs::new([2, 4, 8, 512, 2048]);
        let checkpoints = BTreeSet::new();
        for end in [512, 2048, 8192, 32768] {
            for prefix in [1, 26] {
                assert!(costs.score(prefix, end, 160 << 20).is_none());
                assert!(!costs.admit(prefix, end, 160 << 20, 0, &checkpoints));
            }
            assert!(costs.score(end, end, 2 << 30).is_some());
        }
        assert!(costs.admit(512, 520, 180 << 20, 0, &checkpoints));
        assert!(costs.score(527, 536, 180 << 20).is_some());
        assert!(costs.admit(8192, 32768, 450 << 20, 0, &checkpoints));
        assert!(costs.admit(26, 46, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(62, 71, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(62, 71, 160 << 20, 47, &checkpoints));
        assert!(costs.admit(44, 200, 160 << 20, 0, &checkpoints));
        assert!(costs.admit(82, 200, 160 << 20, 0, &checkpoints));
        assert!(!costs.admit(82, 200, 160 << 20, 0, &BTreeSet::from([44])));
    }
    #[test]
    fn measurements_change_selection_and_account_for_restore() {
        let mut costs = Costs::new([8, 512]);
        assert!(costs.score(8, 512, 160 << 20).is_none());
        costs.observe(512, 5.);
        costs.observe(8, 0.02);
        assert!(costs.score(8, 512, 160 << 20).is_some());
        assert!(costs.admit(8, 512, 160 << 20, 0, &BTreeSet::new()));
        costs.observe_restore(160 << 20, 100.);
        assert!(costs.score(8, 512, 160 << 20).is_none());
    }
    #[test]
    fn mixed_chunks_follow_measured_cost_and_never_pad_a_tail() {
        let mut costs = Costs::new([16, 128, 512, 2048, 4096]);
        assert_eq!(costs.bounded_chunk(8192, 200.), 128);
        costs.observe(16, 0.04);
        costs.observe(128, 0.25);
        costs.observe(512, 0.6);
        assert_eq!(costs.bounded_chunk(8192, 200.), 16);
        assert_eq!(costs.bounded_chunk(8192, 700.), 512);
        assert_eq!(costs.bounded_chunk(15, 200.), 1);
        assert_eq!(costs.bounded_chunk(8192, 1.), 1);
        assert_eq!(costs.bounded_chunk(8192, f64::INFINITY), 4096);
    }
}
