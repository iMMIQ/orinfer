//! Counter uniforms for op25. Request identity and committed step are stable
//! across queueing, batch permutations and replay. This does not allocate state.

/// Seed used by the fixed local evaluation protocol.
pub const EVALUATION_SEED: u64 = 20_261_002;

mod constrained;
pub use constrained::{Constraint, Decoder, TokenLogprob, verify};

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
    pub logit_bias: std::collections::BTreeMap<u32, f64>,
    pub top_logprobs: Option<usize>,
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
            logit_bias: Default::default(),
            top_logprobs: None,
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
            || self
                .logit_bias
                .values()
                .any(|b| !b.is_finite() || !(-100.0..=100.0).contains(b))
            || self.top_logprobs.is_some_and(|n| n > 20)
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
            && self.logit_bias.is_empty()
    }
}
/// Normalized target or draft law after applying the same history processors.
/// Small truncated laws store only their support; large laws retain dense lookup.
#[derive(Clone, Debug)]
pub struct Distribution {
    probabilities: Option<Vec<f64>>,
    vocab: usize,
    ordered: Vec<(u32, f64)>,
}
impl Distribution {
    pub fn from_logits(logits: &[f32], history: &[u32], options: &Options) -> Result<Self, String> {
        Self::from_masked(logits, history, options, None)
    }
    pub fn from_masked(
        logits: &[f32],
        history: &[u32],
        options: &Options,
        mask: Option<&[u32]>,
    ) -> Result<Self, String> {
        options.validate()?;
        if logits.is_empty()
            || logits.len() > u32::MAX as usize
            || logits.iter().any(|x| !x.is_finite())
        {
            return Err("Invalid sampling logits".into());
        }
        let mut counts = if options.repetition_penalty != 1.0
            || options.presence_penalty != 0.0
            || options.frequency_penalty != 0.0
        {
            vec![0usize; logits.len()]
        } else {
            vec![]
        };
        if !counts.is_empty() {
            for &id in history {
                if let Some(n) = counts.get_mut(id as usize) {
                    *n += 1;
                }
            }
        }
        let mut scores: Vec<(u32, f64)> = logits
            .iter()
            .enumerate()
            .map(|(i, &value)| {
                let mut score = f64::from(value);
                let count = counts.get(i).copied().unwrap_or(0);
                if count > 0 {
                    score = if score < 0.0 {
                        score * options.repetition_penalty
                    } else {
                        score / options.repetition_penalty
                    };
                    score -= options.presence_penalty;
                }
                score -= options.frequency_penalty * count as f64;
                score += options.logit_bias.get(&(i as u32)).copied().unwrap_or(0.0);
                (i as u32, score)
            })
            .collect();
        if scores.iter().any(|(_, score)| !score.is_finite()) {
            return Err("Nonfinite processed sampling logits".into());
        }
        if let Some(mask) = mask {
            scores.retain(|(id, _)| {
                mask.get(*id as usize / 32)
                    .is_some_and(|bits| bits & (1 << (*id % 32)) != 0)
            });
            if scores.is_empty() {
                return Err("Grammar permits no tokens".into());
            }
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
        let probabilities = (scores.len() > 128).then(|| {
            let mut dense = vec![0.0; logits.len()];
            for &(id, mass) in &scores {
                dense[id as usize] = mass;
            }
            dense
        });
        Ok(Self {
            probabilities,
            vocab: logits.len(),
            ordered: scores,
        })
    }
    pub fn probability(&self, token: u32) -> f64 {
        if let Some(dense) = &self.probabilities {
            dense.get(token as usize).copied().unwrap_or(0.0)
        } else {
            self.ordered
                .iter()
                .find(|(id, _)| *id == token)
                .map_or(0.0, |(_, mass)| *mass)
        }
    }
    fn dense_probabilities(&self) -> std::borrow::Cow<'_, [f64]> {
        if let Some(values) = &self.probabilities {
            std::borrow::Cow::Borrowed(values)
        } else {
            let mut values = vec![0.0; self.vocab];
            for &(id, p) in &self.ordered {
                values[id as usize] = p;
            }
            std::borrow::Cow::Owned(values)
        }
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
        if self.vocab != draft.vocab {
            return Err("Target/draft vocabulary differs".into());
        }
        if self.probabilities.is_none() && draft.probabilities.is_none() {
            let mut ordered: Vec<_> = self
                .ordered
                .iter()
                .map(|&(id, p)| (id, (p - draft.probability(id)).max(0.0)))
                .filter(|(_, p)| *p > 0.0)
                .collect();
            ordered.sort_unstable_by_key(|&(id, _)| id);
            let total: f64 = ordered.iter().map(|(_, p)| p).sum();
            if !total.is_finite() || total <= 0.0 {
                return Err("Rejected proposal has no residual mass".into());
            }
            for (_, p) in &mut ordered {
                *p /= total;
            }
            return Ok(Self {
                probabilities: None,
                vocab: self.vocab,
                ordered,
            });
        }
        let mut probabilities: Vec<f64> = self
            .dense_probabilities()
            .iter()
            .zip(draft.dense_probabilities().iter())
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
            probabilities: Some(probabilities),
            vocab: self.vocab,
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
    if options.temperature == 0.0 {
        return greedy(logits, history, options);
    }
    Distribution::from_logits(logits, history, options)?.draw(counter_uniform(
        options.seed,
        0,
        step as u64,
    ))
}

/// Greedy needs neither sorting nor a dense probability distribution. Retain
/// f64 processors, full-vocabulary validation and total-order tie semantics.
fn greedy(logits: &[f32], history: &[u32], options: &Options) -> Result<u32, String> {
    options.validate()?;
    if logits.is_empty() || logits.len() > u32::MAX as usize {
        return Err("Invalid sampling logits".into());
    }
    let mut counts = vec![0usize; logits.len()];
    for &id in history {
        if let Some(count) = counts.get_mut(id as usize) {
            *count += 1;
        }
    }
    let mut best = (0, f64::NEG_INFINITY);
    for (id, (&value, &count)) in logits.iter().zip(&counts).enumerate() {
        if !value.is_finite() {
            return Err("Invalid sampling logits".into());
        }
        let mut score = f64::from(value);
        if count > 0 {
            score = if score < 0.0 {
                score * options.repetition_penalty
            } else {
                score / options.repetition_penalty
            };
            score -= options.presence_penalty;
        }
        score -= options.frequency_penalty * count as f64;
        score += options.logit_bias.get(&(id as u32)).copied().unwrap_or(0.0);
        if !score.is_finite() {
            return Err("Nonfinite processed sampling logits".into());
        }
        if score.total_cmp(&best.1).is_gt() {
            best = (id as u32, score);
        }
    }
    Ok(best.0)
}

#[cfg(test)]
mod sampling_tests {
    use super::*;
    #[test]
    fn sparse_top_k_and_residual_match_dense_probabilities_and_draws() {
        let logits: Vec<_> = (0..257)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 7.0)
            .collect();
        let draft_logits: Vec<_> = logits.iter().rev().copied().collect();
        for k in [1, 20, 128, 200, 0] {
            let options = Options {
                top_k: k,
                top_p: 0.95,
                ..Default::default()
            };
            let target = Distribution::from_logits(&logits, &[], &options).unwrap();
            let draft = Distribution::from_logits(&draft_logits, &[], &options).unwrap();
            if (1..=128).contains(&k) {
                assert!(target.probabilities.is_none());
            }
            let dense = |law: &Distribution| Distribution {
                probabilities: Some(law.dense_probabilities().into_owned()),
                vocab: law.vocab,
                ordered: law.ordered.clone(),
            };
            let a = target.residual(&draft).unwrap();
            let b = dense(&target).residual(&dense(&draft)).unwrap();
            assert_eq!(a.ordered, b.ordered);
            for id in 0..257 {
                assert_eq!(a.probability(id), b.probability(id));
            }
            for step in 0..100 {
                let u = counter_uniform(EVALUATION_SEED, 0, step);
                assert_eq!(a.draw(u), b.draw(u));
            }
            assert!(target.residual(&target).is_err());
        }
    }
    #[test]
    fn greedy_matches_distribution_processors_and_total_order() {
        for repeat in [0.5, 1.0, 1.05, 2.0, f64::MAX] {
            for presence in [-2.0, 0.0, 2.0] {
                for frequency in [-2.0, 0.0, 2.0] {
                    let options = Options {
                        temperature: 0.0,
                        repetition_penalty: repeat,
                        presence_penalty: presence,
                        frequency_penalty: frequency,
                        ..Options::default()
                    };
                    for logits in [vec![0.0, -0.0, 1.0, -2.0], vec![-0.0, 0.0], vec![1.0; 7]] {
                        let history = [0, 0, 2, 6, 100];
                        let reference = Distribution::from_logits(&logits, &history, &options)
                            .and_then(|d| d.draw(0.5));
                        assert_eq!(greedy(&logits, &history, &options), reference);
                    }
                }
            }
        }
        assert!(greedy(&[1., f32::NAN], &[], &Options::default()).is_err());
    }
    #[test]
    fn rounded_mass_does_not_select_an_underflowed_token() {
        let law = Distribution {
            probabilities: Some(vec![1.0 - 1e-12, 0.0]),
            vocab: 2,
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
