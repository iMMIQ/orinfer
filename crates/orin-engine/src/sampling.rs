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
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub repetition_penalty: f64,
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
            repetition_penalty: 1.0,
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
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
            || !self.frequency_penalty.is_finite()
            || !(-2.0..=2.0).contains(&self.frequency_penalty)
        {
            return Err("Invalid sampling parameters".into());
        }
        Ok(())
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.repetition_penalty == 1.0
    }
}
/// Normalized target or draft law after applying the same history processors.
/// Dense probabilities support exact p/q acceptance and residual sampling.
#[derive(Clone, Debug)]
pub struct Distribution {
    probabilities: Vec<f64>,
    ordered: Vec<(u32, f64)>,
}
impl Distribution {
    pub fn from_logits(logits: &[f32], history: &[u32], options: &Options) -> Result<Self, String> {
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
            .map(|(i, &value)| {
                let mut score = f64::from(value);
                if counts[i] > 0 {
                    score = if score < 0.0 {
                        score * options.repetition_penalty
                    } else {
                        score / options.repetition_penalty
                    };
                    score -= options.presence_penalty;
                }
                score -= options.frequency_penalty * counts[i] as f64;
                (i as u32, score)
            })
            .collect();
        if scores.iter().any(|(_, score)| !score.is_finite()) {
            return Err("Nonfinite processed sampling logits".into());
        }
        let compare =
            |a: &(u32, f64), b: &(u32, f64)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0));
        if options.temperature == 0.0 {
            let best = *scores
                .iter()
                .min_by(|a, b| compare(a, b))
                .expect("Nonempty logits");
            scores = vec![(best.0, 1.0)];
        } else {
            if options.top_k > 0 && options.top_k < scores.len() {
                scores.select_nth_unstable_by(options.top_k, compare);
                scores.truncate(options.top_k);
            }
            scores.sort_unstable_by(compare);
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
            for (_, mass) in &mut scores {
                *mass /= total;
            }
        }
        let mut probabilities = vec![0.0; logits.len()];
        for &(id, mass) in &scores {
            probabilities[id as usize] = mass;
        }
        Ok(Self {
            probabilities,
            ordered: scores,
        })
    }
    pub fn probability(&self, token: u32) -> f64 {
        self.probabilities
            .get(token as usize)
            .copied()
            .unwrap_or(0.0)
    }
    pub fn draw(&self, uniform: f64) -> Result<u32, String> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("Sampling uniform outside [0,1)".into());
        }
        let mut mass = 0.0;
        for &(id, p) in &self.ordered {
            mass += p;
            if uniform < mass {
                return Ok(id);
            }
        }
        // Floating-point cumulative mass can finish slightly below one. Never
        // fall back to an underflowed zero-probability token.
        Ok(self
            .ordered
            .iter()
            .rev()
            .find(|(_, p)| *p > 0.0)
            .ok_or("Empty sampling distribution")?
            .0)
    }
    /// On a rejected proposal the correction law is proportional to (p-q)+.
    pub fn residual(&self, draft: &Self) -> Result<Self, String> {
        if self.probabilities.len() != draft.probabilities.len() {
            return Err("Target/draft vocabulary differs".into());
        }
        let mut probabilities: Vec<f64> = self
            .probabilities
            .iter()
            .zip(&draft.probabilities)
            .map(|(p, q)| (p - q).max(0.0))
            .collect();
        let total: f64 = probabilities.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err("Rejected proposal has no residual mass".into());
        }
        for p in &mut probabilities {
            *p /= total;
        }
        let ordered = probabilities
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > 0.0)
            .map(|(id, &p)| (id as u32, p))
            .collect();
        Ok(Self {
            probabilities,
            ordered,
        })
    }
}

pub fn sample(
    logits: &[f32],
    history: &[u32],
    options: &Options,
    step: usize,
) -> Result<u32, String> {
    Distribution::from_logits(logits, history, options)?.draw(counter_uniform(
        options.seed,
        0,
        step as u64,
    ))
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn rounded_mass_does_not_select_an_underflowed_token() {
        let law = Distribution {
            probabilities: vec![1.0 - 1e-12, 0.0],
            ordered: vec![(0, 1.0 - 1e-12), (1, 0.0)],
        };
        assert_eq!(law.draw(1.0 - f64::EPSILON).unwrap(), 0);
    }
    #[test]
    fn repetition_and_top_k_ties() {
        let mut options = Options {
            temperature: 0.0,
            repetition_penalty: 2.0,
            ..Options::default()
        };
        assert_eq!(sample(&[3., 2.], &[0], &options, 0).unwrap(), 1);
        assert_eq!(sample(&[-1., -1.5], &[0], &options, 0).unwrap(), 1);
        options.temperature = 1.0;
        options.top_k = 2;
        let p = Distribution::from_logits(&[1., 1., 1., 1.], &[], &options).unwrap();
        assert_eq!(p.probability(0), 0.5);
        assert_eq!(p.probability(1), 0.5);
        assert_eq!(p.probability(2), 0.0);
        options.repetition_penalty = f64::MAX;
        assert!(Distribution::from_logits(&[-2.], &[0], &options).is_err());
    }
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
