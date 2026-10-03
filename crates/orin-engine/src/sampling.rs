//! Counter uniforms for op25. Request identity and committed step are stable
//! across queueing, batch permutations and replay. This does not allocate state.

/// Seed used by the fixed local evaluation protocol.
pub const EVALUATION_SEED: u64 = 20_261_002;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Return an IEEE f64 draw in [0,1) for the explicit-uniform sampling ABI.
///
/// `absolute_step` is the committed sampling position, not a resident row,
/// graph replay number or number of speculative proposals. A branch that
/// retains identity and position reuses its draw; an independent branch must
/// receive a distinct request identity from the scheduler.
pub fn counter_uniform(seed: u64, request_id: u64, absolute_step: u64) -> f64 {
    let bits = splitmix64(splitmix64(seed ^ request_id) ^ absolute_step) >> 11;
    // Integers below 2^53 and multiplication by a power of two are exact.
    bits as f64 * (1.0 / 9_007_199_254_740_992.0)
}

/// Host sampling of the resident model's logits. Greedy keeps the GPU fast path.
#[derive(Clone, Debug)]
pub struct Options {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub seed: u64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: EVALUATION_SEED,
        }
    }
}
impl Options {
    pub fn validate(&self) -> Result<(), String> {
        if !self.temperature.is_finite()
            || !(0.0..=2.0).contains(&self.temperature)
            || !self.top_p.is_finite()
            || !(0.0..=1.0).contains(&self.top_p)
            || self.top_p == 0.0
            || !self.presence_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.presence_penalty)
            || !self.frequency_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.frequency_penalty)
        {
            return Err("Invalid sampling parameters".into());
        }
        Ok(())
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 && self.presence_penalty == 0.0 && self.frequency_penalty == 0.0
    }
}
pub fn sample(
    logits: &[f32],
    history: &[u32],
    options: &Options,
    step: usize,
) -> Result<u32, String> {
    options.validate()?;
    if logits.is_empty()
        || logits.len() > u32::MAX as usize
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err("Invalid sampling logits".into());
    }
    let mut counts = vec![0usize; logits.len()];
    for &id in history {
        if let Some(n) = counts.get_mut(id as usize) {
            *n += 1;
        }
    }
    let mut scores: Vec<(u32, f64)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let score = v as f64
                - options.frequency_penalty * counts[i] as f64
                - if counts[i] > 0 {
                    options.presence_penalty
                } else {
                    0.0
                };
            (i as u32, score)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if options.temperature == 0.0 {
        return Ok(scores[0].0);
    }
    if options.top_k > 0 {
        scores.truncate(options.top_k);
    }
    let maximum = scores[0].1;
    for (_, score) in &mut scores {
        *score = ((*score - maximum) / options.temperature).exp();
    }
    let total: f64 = scores.iter().map(|x| x.1).sum();
    let mut mass = 0.0;
    let cutoff = scores
        .iter()
        .position(|x| {
            mass += x.1;
            mass >= total * options.top_p
        })
        .unwrap_or(scores.len() - 1);
    scores.truncate(cutoff + 1);
    let total: f64 = scores.iter().map(|x| x.1).sum();
    let draw = counter_uniform(options.seed, 0, step as u64) * total;
    let mut cumulative = 0.0;
    for &(id, p) in &scores {
        cumulative += p;
        if draw < cumulative {
            return Ok(id);
        }
    }
    Ok(scores.last().expect("Nonempty nucleus").0)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn greedy_penalty_nucleus_and_seed() {
        let mut options = Options {
            temperature: 0.0,
            ..Options::default()
        };
        assert_eq!(sample(&[1., 2., 3.], &[], &options, 0).unwrap(), 2);
        options.frequency_penalty = 2.0;
        assert_eq!(sample(&[1., 2., 3.], &[2, 2], &options, 0).unwrap(), 1);
        options = Options {
            top_p: 0.01,
            ..Options::default()
        };
        assert_eq!(sample(&[1., 2., 3.], &[], &options, 3).unwrap(), 2);
        options.top_p = 1.0;
        let first = sample(&[1., 2., 3.], &[], &options, 37).unwrap();
        assert_eq!(first, sample(&[1., 2., 3.], &[], &options, 37).unwrap());
        assert!(sample(&[f32::NAN], &[], &options, 0).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::{EVALUATION_SEED, counter_uniform};

    #[test]
    fn matches_frozen_python_counter_contract() {
        // Generated by the independently specified op25 Python contract.
        let vectors = [
            (0, 0, 6_720_141_781_353_792_u64),
            (1, 0, 4_787_244_564_846_236),
            (1, 1, 7_011_218_493_468_269),
            (2, 7, 1_098_476_309_306_172),
            (u64::MAX, u64::MAX, 7_253_473_543_011_370),
            (42, 8192, 6_879_436_830_168_841),
        ];
        for (request, step, numerator) in vectors {
            let actual = counter_uniform(EVALUATION_SEED, request, step);
            let expected = numerator as f64 / 9_007_199_254_740_992.0;
            assert_eq!(actual.to_bits(), expected.to_bits());
            assert!((0.0..1.0).contains(&actual));
        }
    }
}
