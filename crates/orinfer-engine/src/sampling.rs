//! Counter uniforms for op25. Request identity and committed step are stable
//! across queueing, batch permutations and replay. This does not allocate state.

use std::borrow::Cow;
use std::cmp::Ordering;

/// Seed used by the fixed local evaluation protocol.
pub const EVALUATION_SEED: u64 = 20_261_002;

mod constrained;
mod prepared;
mod reporting;
pub use constrained::{Constraint, Decoder, TokenLogprob, verify};
pub(crate) use constrained::{RowParameters, verify_validated};
pub(crate) use prepared::Prepared;

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

/// Row identity for the device categorical RNG. Independent of owner slots,
/// packet shape, vocabulary partition and speculative verification depth.
pub fn categorical_key(seed: u64, absolute_step: u64) -> [u32; 2] {
    let key = splitmix64(splitmix64(seed) ^ absolute_step);
    [key as u32, (key >> 32) as u32]
}

/// Random123 Philox2x32-10 reference for device validation and shared backends.
pub fn categorical_bits(mut counter: [u32; 2], mut key: u32) -> [u32; 2] {
    for _ in 0..10 {
        let product = u64::from(counter[0]) * 0xd256_d193;
        counter = [(product >> 32) as u32 ^ key ^ counter[1], product as u32];
        key = key.wrapping_add(0x9e37_79b9);
    }
    counter
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
    /// Full categorical law without history processors or probability reports.
    /// Grammar eligibility is checked separately against the request decoder.
    pub fn device_categorical(&self) -> bool {
        self.temperature > 0.0 && self.device_untruncated() && self.top_logprobs.is_none()
    }
    pub(crate) fn device_untruncated(&self) -> bool {
        (self.temperature == 0.0 || (1e-5..=2.0).contains(&self.temperature))
            && self.top_p == 1.0
            && self.top_k == 0
            && self.presence_penalty == 0.0
            && self.frequency_penalty == 0.0
            && self.repetition_penalty == 1.0
            && self.logit_bias.is_empty()
    }
    pub fn validate(&self) -> Result<(), String> {
        for (name, value, min, max) in [
            ("temperature", self.temperature, 0.0, 2.0),
            ("top_p", self.top_p, 0.0, 1.0),
            ("presence_penalty", self.presence_penalty, -2.0, 2.0),
            ("frequency_penalty", self.frequency_penalty, -2.0, 2.0),
        ] {
            if !(min..=max).contains(&value) || name == "top_p" && value == 0.0 {
                return Err(format!("{name}: Invalid sampling value"));
            }
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty: Expected positive finite value".into());
        }
        for (&id, bias) in &self.logit_bias {
            if !(-100.0..=100.0).contains(bias) {
                return Err(format!("logit_bias.{id}: Bias outside -100..100"));
            }
        }
        if self.top_logprobs.is_some_and(|n| n > 20) {
            return Err("top_logprobs: Expected 0–20".into());
        }
        Ok(())
    }
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0 && !self.processes_scores()
    }
    fn processes_scores(&self) -> bool {
        self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.repetition_penalty != 1.0
            || !self.logit_bias.is_empty()
    }
}

fn validate_logits(logits: &[f32]) -> Result<(), String> {
    if logits.is_empty()
        || logits.len() > u32::MAX as usize
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err("Invalid sampling logits".into());
    }
    Ok(())
}
fn processed_scores(
    logits: &[f32],
    history: &[u32],
    options: &Options,
    mask: Option<&[u32]>,
) -> Result<Vec<(u32, f64)>, String> {
    validate_logits(logits)?;
    let mut scores = if !options.processes_scores() {
        // Adding zero preserves the generic processors' signed-zero result.
        // Input validation already guarantees these f64 scores are finite.
        logits
            .iter()
            .enumerate()
            .map(|(i, &value)| (i as u32, f64::from(value) + 0.0))
            .collect()
    } else {
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
        let scores: Vec<(u32, f64)> = logits
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
        scores
    };
    if let Some(mask) = mask {
        scores.retain(|(id, _)| {
            mask.get(*id as usize / 32)
                .is_some_and(|bits| bits & (1 << (*id % 32)) != 0)
        });
        if scores.is_empty() {
            return Err("Grammar permits no tokens".into());
        }
    }
    Ok(scores)
}

type Score = (u32, f64);
const SPARSE_SUPPORT: usize = 128;
const SHORTLIST_LIMIT: usize = 4096;

fn compare(a: &Score, b: &Score) -> Ordering {
    b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
}

#[derive(Clone, Debug)]
enum Scores<'a> {
    Logits(Cow<'a, [f32]>),
    Processed(Vec<Score>),
}

impl Scores<'_> {
    fn iter(&self) -> impl Iterator<Item = Score> {
        let (logits, processed) = match self {
            Self::Logits(logits) => (logits.as_ref(), &[][..]),
            Self::Processed(scores) => (&[][..], scores.as_slice()),
        };
        logits
            .iter()
            .enumerate()
            // Match the signed-zero behavior of the history processors.
            .map(|(id, &score)| (id as u32, f64::from(score) + 0.0))
            .chain(processed.iter().copied())
    }

    fn len(&self) -> usize {
        match self {
            Self::Logits(logits) => logits.len(),
            Self::Processed(scores) => scores.len(),
        }
    }

    fn for_fallback(self) -> Scores<'static> {
        // Rounded probabilities can collapse distinct scores into a tie.
        // Keep their original ordering keys for an exact tail draw.
        match self {
            Self::Logits(logits) => Scores::Logits(Cow::Owned(logits.into_owned())),
            Self::Processed(scores) => Scores::Processed(scores),
        }
    }

    fn full_order(self, options: &Options) -> Vec<Score> {
        let mut ordered = match self {
            Self::Logits(_) => self.iter().collect(),
            Self::Processed(scores) => scores,
        };
        if options.temperature == 0.0 {
            let best = ordered.iter().min_by(|a, b| compare(a, b)).unwrap().0;
            return vec![(best, 1.0)];
        }
        if options.top_k > 0 && options.top_k < ordered.len() {
            ordered.select_nth_unstable_by(options.top_k, compare);
            ordered.truncate(options.top_k);
        }
        ordered.sort_unstable_by(compare);
        let maximum = ordered[0].1;
        for (_, score) in &mut ordered {
            *score = weight(*score, maximum, options.temperature);
        }
        let total = ordered.iter().map(|x| x.1).sum::<f64>();
        let (cutoff, _, mass) = nucleus(&ordered, total * options.top_p).unwrap();
        ordered.truncate(cutoff);
        normalize(&mut ordered, mass);
        ordered
    }
}

fn weight(score: f64, maximum: f64, temperature: f64) -> f64 {
    ((score - maximum) / temperature).exp()
}

fn normalize(ordered: &mut [Score], total: f64) {
    for (_, p) in ordered {
        *p /= total;
    }
}

// Return the retained length and cumulative weights on both sides of the cut.
fn nucleus(ordered: &[Score], threshold: f64) -> Option<(usize, f64, f64)> {
    let mut mass = 0.0;
    for (i, &(_, w)) in ordered.iter().enumerate() {
        let previous = mass;
        mass += w;
        if mass >= threshold {
            return Some((i + 1, previous, mass));
        }
    }
    None
}

struct Shortlist {
    ordered: Vec<Score>,
    total: f64,
    // Bounds the effects of a different summation order on the normalized CDF.
    error: f64,
}

impl Shortlist {
    fn new(scores: &Scores<'_>, temperature: f64, mut store_weight: impl FnMut(u32, f64)) -> Self {
        let maximum = scores.iter().map(|(_, s)| s).reduce(f64::max).unwrap();
        let mut threshold = (u32::MAX, maximum - 18.0 * temperature);
        // Such weights round away in the legacy descending sum, which
        // starts at 1. Even top_p=1 cuts off before these tokens.
        let negligible = maximum - 38.0 * temperature;
        let mut ordered = Vec::new();
        let mut total = 0.0;
        for (id, score) in scores.iter() {
            if score < negligible {
                continue;
            }
            let entry = (id, score);
            if score > threshold.1 || (score == threshold.1 && compare(&entry, &threshold).is_le())
            {
                ordered.push(entry);
                if ordered.len() == SHORTLIST_LIMIT {
                    // Bound storage on broad rows while retaining a prefix
                    // of the full law, including ties at its edge.
                    let middle = SHORTLIST_LIMIT / 2;
                    let (_, last, _) = ordered.select_nth_unstable_by(middle - 1, compare);
                    threshold = *last;
                    ordered.truncate(middle);
                }
            }
            let w = weight(score, maximum, temperature);
            total += w;
            store_weight(id, w);
        }
        ordered.sort_unstable_by(compare);
        for (_, score) in &mut ordered {
            *score = weight(*score, maximum, temperature);
        }
        Self {
            ordered,
            total,
            error: 4.0 * scores.len() as f64 * f64::EPSILON,
        }
    }

    fn truncate(&mut self, top_p: f64) -> bool {
        let threshold = self.total * top_p;
        let margin = self.error * self.total;
        let Some((cutoff, before, after)) = nucleus(&self.ordered, threshold) else {
            return false;
        };
        // A rounded normalizer can move the nucleus by one or more tokens.
        // Recompute in descending score order if the cutoff is ambiguous.
        if (before > 0.0 && threshold - before <= margin) || after - threshold <= margin {
            return false;
        }
        self.ordered.truncate(cutoff);
        normalize(&mut self.ordered, after);
        true
    }
}

fn draw_ordered(ordered: &[Score], uniform: f64, margin: f64) -> Option<u32> {
    let mut mass = 0.0;
    for &(id, p) in ordered {
        let previous = mass;
        mass += p;
        if uniform < mass {
            return (margin == 0.0
                || (uniform == 0.0 || uniform - previous > margin) && mass - uniform > margin)
                .then_some(id);
        }
    }
    None
}

fn draw_complete(ordered: &[Score], uniform: f64) -> Result<u32, String> {
    // The CDF can finish slightly below one. Find the last live token.
    draw_ordered(ordered, uniform, 0.0)
        .or_else(|| ordered.iter().rev().find(|(_, p)| *p > 0.0).map(|x| x.0))
        .ok_or_else(|| "Empty sampling distribution".into())
}

#[derive(Clone, Debug)]
struct Tail {
    scores: Scores<'static>,
    temperature: f64,
    margin: f64,
}

/// Normalized target or draft law after applying the same history processors.
/// Small truncated laws store only their support; large laws retain dense lookup.
#[derive(Clone, Debug)]
pub struct Distribution {
    probabilities: Option<Vec<f64>>,
    vocab: usize,
    ordered: Vec<Score>,
    fallback: Option<Tail>,
}
impl Distribution {
    /// A deterministic proposer is an exact point mass, including when its
    /// verifier uses stochastic sampling and positive-residual correction.
    pub fn point_mass(token: u32, vocab: usize) -> Result<Self, String> {
        if vocab == 0 || token as usize >= vocab {
            return Err("Proposal token exceeds vocabulary".into());
        }
        Ok(Self {
            probabilities: None,
            vocab,
            ordered: vec![(token, 1.0)],
            fallback: None,
        })
    }
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
        Self::from_validated(logits, history, options, mask)
    }
    pub(crate) fn from_validated(
        logits: &[f32],
        history: &[u32],
        options: &Options,
        mask: Option<&[u32]>,
    ) -> Result<Self, String> {
        if mask.is_none() && !options.processes_scores() && options.temperature > 0.0 {
            validate_logits(logits)?;
            Ok(Self::from_source(
                logits.len(),
                Scores::Logits(Cow::Borrowed(logits)),
                options,
            ))
        } else {
            let scores = processed_scores(logits, history, options, mask)?;
            Ok(Self::from_scores(logits.len(), scores, options))
        }
    }

    // Reporting may have partitioned the processed scores.
    fn from_scores(vocab: usize, scores: Vec<Score>, options: &Options) -> Self {
        Self::from_source(vocab, Scores::Processed(scores), options)
    }

    fn from_source(vocab: usize, scores: Scores<'_>, options: &Options) -> Self {
        if options.temperature > 0.0 && options.top_k == 0 && scores.len() > SPARSE_SUPPORT {
            if options.top_p == 1.0 {
                let mut dense = vec![0.0; vocab];
                let mut shortlist = Shortlist::new(&scores, options.temperature, |id, w| {
                    dense[id as usize] = w;
                });
                if shortlist.ordered.len() < scores.len() {
                    normalize(&mut shortlist.ordered, shortlist.total);
                    for p in &mut dense {
                        *p /= shortlist.total;
                    }
                    return Self {
                        probabilities: Some(dense),
                        vocab,
                        ordered: shortlist.ordered,
                        fallback: Some(Tail {
                            scores: scores.for_fallback(),
                            temperature: options.temperature,
                            margin: shortlist.error,
                        }),
                    };
                }
            } else {
                let mut shortlist = Shortlist::new(&scores, options.temperature, |_, _| {});
                if shortlist.truncate(options.top_p) {
                    return Self::from_ordered(vocab, shortlist.ordered);
                }
            }
        }
        Self::from_ordered(vocab, scores.full_order(options))
    }

    fn from_ordered(vocab: usize, ordered: Vec<Score>) -> Self {
        let probabilities = (ordered.len() > SPARSE_SUPPORT).then(|| {
            let mut dense = vec![0.0; vocab];
            for &(id, mass) in &ordered {
                dense[id as usize] = mass;
            }
            dense
        });
        Self {
            probabilities,
            vocab,
            ordered,
            fallback: None,
        }
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
        if let Some(tail) = &self.fallback {
            if let Some(id) = draw_ordered(&self.ordered, uniform, tail.margin) {
                return Ok(id);
            }
            let ordered = Scores::Processed(tail.scores.iter().collect()).full_order(&Options {
                temperature: tail.temperature,
                ..Options::default()
            });
            return draw_complete(&ordered, uniform);
        }
        draw_complete(&self.ordered, uniform)
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
                fallback: None,
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
            fallback: None,
        })
    }
}

pub fn sample(
    logits: &[f32],
    history: &[u32],
    options: &Options,
    step: usize,
) -> Result<u32, String> {
    options.validate()?;
    if options.temperature == 0.0 {
        return greedy(logits, history, options);
    }
    let scores = if options.processes_scores() {
        Scores::Processed(processed_scores(logits, history, options, None)?)
    } else {
        validate_logits(logits)?;
        Scores::Logits(Cow::Borrowed(logits))
    };
    draw_stochastic(
        scores,
        options,
        counter_uniform(options.seed, 0, step as u64),
    )
}

fn draw_stochastic(scores: Scores<'_>, options: &Options, uniform: f64) -> Result<u32, String> {
    if options.top_k == 0 && scores.len() > SPARSE_SUPPORT {
        let mut shortlist = Shortlist::new(&scores, options.temperature, |_, _| {});
        if options.top_p == 1.0 {
            normalize(&mut shortlist.ordered, shortlist.total);
            if let Some(id) = draw_ordered(&shortlist.ordered, uniform, shortlist.error) {
                return Ok(id);
            }
        } else if shortlist.truncate(options.top_p) {
            return draw_complete(&shortlist.ordered, uniform);
        }
    }
    draw_complete(&scores.full_order(options), uniform)
}

/// Greedy needs neither sorting nor a dense probability distribution. Retain
/// f64 processors, full-vocabulary validation and total-order tie semantics.
fn greedy(logits: &[f32], history: &[u32], options: &Options) -> Result<u32, String> {
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

    fn naive_law(vocab: usize, mut ordered: Vec<Score>, options: &Options) -> Distribution {
        ordered.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        if options.top_k > 0 {
            ordered.truncate(options.top_k);
        }
        let maximum = ordered[0].1;
        for (_, mass) in &mut ordered {
            *mass = ((*mass - maximum) / options.temperature).exp();
        }
        let threshold = ordered.iter().map(|x| x.1).sum::<f64>() * options.top_p;
        let mut total = 0.0;
        let mut count = 0;
        for &(_, mass) in &ordered {
            total += mass;
            count += 1;
            if total >= threshold {
                break;
            }
        }
        ordered.truncate(count);
        for (_, mass) in &mut ordered {
            *mass /= total;
        }
        let probabilities = (count > 128).then(|| {
            let mut probabilities = vec![0.0; vocab];
            for &(id, p) in &ordered {
                probabilities[id as usize] = p;
            }
            probabilities
        });
        Distribution {
            probabilities,
            vocab,
            ordered,
            fallback: None,
        }
    }

    fn naive_draw(ordered: &[Score], uniform: f64) -> u32 {
        let mut mass = 0.0;
        let mut last = None;
        for &(id, p) in ordered {
            if p > 0.0 {
                last = Some(id);
            }
            mass += p;
            if uniform < mass {
                break;
            }
        }
        last.unwrap()
    }

    #[test]
    fn shortlist_matches_full_sort_with_ties_processors_and_masks() {
        let mut peaked: Vec<_> = (0..1025)
            .map(|i| -22.0 - (i * 17 % 101) as f32 / 7.0)
            .collect();
        peaked[..8].copy_from_slice(&[-0.0, 0.0, -0.0, 0.0, -0.25, -1.0, -3.0, -18.125]);
        let flat = vec![2.0; SHORTLIST_LIMIT + 1];
        let history = [0, 0, 1, 4, 4, 16, u32::MAX];
        for logits in [&peaked, &flat] {
            let mask = vec![0x5555_5555; logits.len().div_ceil(32)];
            for processors in [
                Options::default(),
                Options {
                    repetition_penalty: 1.3,
                    presence_penalty: -0.2,
                    frequency_penalty: 0.5,
                    logit_bias: [(1, -0.5), (4, 1.0), (256, 8.0)].into(),
                    ..Options::default()
                },
            ] {
                for temperature in [0.2, 1.0, 2.0] {
                    for top_p in [1.0, 0.9, 0.5, 0.1] {
                        for top_k in [0, 20] {
                            let options = Options {
                                temperature,
                                top_p,
                                top_k,
                                ..processors.clone()
                            };
                            for mask in [None, Some(mask.as_slice())] {
                                let scores =
                                    processed_scores(logits, &history, &options, mask).unwrap();
                                let expected = naive_law(logits.len(), scores.clone(), &options);
                                let law = Distribution::from_scores(logits.len(), scores, &options);
                                for id in 0..logits.len() as u32 {
                                    assert!(
                                        (law.probability(id) - expected.probability(id)).abs()
                                            < 1e-12
                                    );
                                }
                                for uniform in [0.0, 1e-15, 0.5, 0.999999999, 1.0 - f64::EPSILON] {
                                    assert_eq!(
                                        law.draw(uniform).unwrap(),
                                        naive_draw(&expected.ordered, uniform)
                                    );
                                }
                                if mask.is_none() {
                                    for step in 0..8 {
                                        let uniform = counter_uniform(options.seed, 0, step);
                                        assert_eq!(
                                            sample(logits, &history, &options, step as usize)
                                                .unwrap(),
                                            naive_draw(&expected.ordered, uniform)
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let options = Options::default();
        let mut flat_shortlist =
            Shortlist::new(&Scores::Logits(Cow::Borrowed(&flat)), 1.0, |_, _| {});
        assert!(!flat_shortlist.truncate(0.9));
        assert_eq!(
            Distribution::from_logits(&peaked, &[], &options)
                .unwrap()
                .draw(0.0),
            Ok(0)
        );
        assert_eq!(
            sample(&[], &[], &options, 0),
            Err("Invalid sampling logits".into())
        );
        assert_eq!(
            sample(&[1.0, f32::INFINITY], &[], &options, 0),
            Err("Invalid sampling logits".into())
        );
        assert_eq!(
            Distribution::from_masked(&peaked, &[], &options, Some(&[0])).unwrap_err(),
            "Grammar permits no tokens"
        );
    }

    #[test]
    fn untruncated_draws_preserve_tail_order_and_cdf_boundaries() {
        let mut logits = vec![-1000.0; 257];
        logits[..5].copy_from_slice(&[0.0, -0.0, -18.125, -18.25, -42.0]);
        let options = Options::default();
        let expected = naive_law(
            logits.len(),
            processed_scores(&logits, &[], &options, None).unwrap(),
            &options,
        );
        let law = Distribution::from_logits(&logits, &[], &options).unwrap();
        assert_eq!(law.ordered.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        assert!(law.fallback.is_some());
        assert!(0.999999999 >= law.ordered.iter().map(|x| x.1).sum());
        let boundary = expected.probability(0);
        for uniform in [
            0.0,
            boundary.next_down(),
            boundary,
            0.999999999,
            1.0_f64.next_down(),
        ] {
            let token = naive_draw(&expected.ordered, uniform);
            assert_eq!(law.draw(uniform), Ok(token));
            assert_eq!(
                draw_stochastic(Scores::Logits(Cow::Borrowed(&logits)), &options, uniform),
                Ok(token)
            );
        }

        // A broad tail can hold enough mass to move a nucleus beyond 18T.
        let mut broad = vec![-18.125; 1025];
        broad[0] = 0.0;
        let options = Options {
            top_p: 0.999999,
            ..options
        };
        let reference = naive_law(
            broad.len(),
            processed_scores(&broad, &[], &options, None).unwrap(),
            &options,
        );
        let law = Distribution::from_logits(&broad, &[], &options).unwrap();
        assert!(law.ordered.len() > 128);
        assert_eq!(law.ordered, reference.ordered);
    }

    #[test]
    fn shortlist_residual_and_verification_match_full_sort() {
        let mut first = vec![-24.0; 257];
        first[..4].copy_from_slice(&[1.25, 0.0, 0.25, 1.0]);
        let mut second = first.clone();
        second.swap(0, 2);
        let mut draft_logits = first.clone();
        draft_logits.swap(0, 3);
        let mut logits = first.clone();
        logits.extend(&second);
        for top_p in [1.0, 0.9, 0.5, 0.1] {
            let options = Options {
                top_p,
                ..Options::default()
            };
            let reference = |row| {
                naive_law(
                    first.len(),
                    processed_scores(row, &[], &options, None).unwrap(),
                    &options,
                )
            };
            let p = reference(&first);
            let q = reference(&draft_logits);
            let next = reference(&second);
            let target = Distribution::from_logits(&first, &[], &options).unwrap();
            let draft = Distribution::from_logits(&draft_logits, &[], &options).unwrap();
            let residual: Vec<_> = (0..first.len() as u32)
                .map(|id| (id, (p.probability(id) - q.probability(id)).max(0.0)))
                .filter(|x| x.1 > 0.0)
                .collect();
            let total: f64 = residual.iter().map(|x| x.1).sum();
            let residual: Vec<_> = residual
                .into_iter()
                .map(|(id, w)| (id, w / total))
                .collect();
            let actual = target.residual(&draft).unwrap();
            for &(id, probability) in &residual {
                assert!((actual.probability(id) - probability).abs() < 1e-12);
            }
            let token = naive_draw(&q.ordered, 0.0);
            let proposal = crate::mtp::Proposal {
                token,
                distribution: draft,
            };
            for step in 0..32 {
                let accept = counter_uniform(options.seed, 0x4d54_5002, step as u64)
                    < (p.probability(token) / q.probability(token)).min(1.0);
                let expected = if accept {
                    vec![
                        token,
                        naive_draw(
                            &next.ordered,
                            counter_uniform(options.seed, 0, step as u64 + 1),
                        ),
                    ]
                } else {
                    vec![naive_draw(
                        &residual,
                        counter_uniform(options.seed, 0x4d54_5003, step as u64),
                    )]
                };
                assert_eq!(
                    verify(
                        std::slice::from_ref(&proposal),
                        &logits,
                        &[],
                        &options,
                        &mut Decoder::default(),
                        step,
                        first.len()
                    )
                    .unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    #[ignore = "release timing diagnostic"]
    fn sampling_release_timing() {
        use std::{hint::black_box, time::Instant};

        let logits: Vec<_> = (0usize..248_320)
            .map(|i| {
                let score = ((i * 12_307 % 248_321) as f32) / 248_321.0;
                if i % 1021 == 0 {
                    -6.0 * score
                } else {
                    -20.0 - 24.0 * score
                }
            })
            .collect();
        let options = Options::default();
        let measure = |f: &mut dyn FnMut(u64)| {
            for i in 0..8 {
                f(i);
            }
            let start = Instant::now();
            for i in 0..64 {
                f(i);
            }
            start.elapsed().as_secs_f64() * 1e3 / 64.0
        };
        let reference = || {
            let scores = processed_scores(black_box(&logits), &[], &options, None).unwrap();
            naive_law(logits.len(), scores, &options)
        };
        let reference_sample_ms = measure(&mut |i| {
            black_box(naive_draw(
                &reference().ordered,
                counter_uniform(options.seed, 0, i),
            ));
        });
        let reference_distribution_ms = measure(&mut |_| {
            black_box(reference());
        });
        let sample_ms = measure(&mut |i| {
            black_box(sample(black_box(&logits), &[], &options, i as usize).unwrap());
        });
        let distribution_ms = measure(&mut |_| {
            black_box(Distribution::from_logits(black_box(&logits), &[], &options).unwrap());
        });
        eprintln!(
            "vocab={} sample_ms={reference_sample_ms:.3}->{sample_ms:.3} distribution_ms={reference_distribution_ms:.3}->{distribution_ms:.3}",
            logits.len()
        );
    }

    #[test]
    fn unprocessed_scores_preserve_generic_bits_masks_and_seeded_sampling() {
        let mut logits = vec![
            -0.0,
            0.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            -f32::from_bits(1),
        ];
        logits.extend((0..1024).map(|i| ((i * 17 % 101) as f32 - 50.0) / 7.0));
        let mask = vec![0x5555_5555; logits.len().div_ceil(32)];
        let history = [0, 0, 2, u32::MAX];
        for frequency in [0.0, -0.0] {
            let options = Options {
                frequency_penalty: frequency,
                ..Options::default()
            };
            // An irrelevant bias preserves values while exercising the generic
            // processor as a reference, including IEEE signed-zero behavior.
            let generic = Options {
                logit_bias: [(u32::MAX, 0.0)].into(),
                ..options.clone()
            };
            for mask in [None, Some(mask.as_slice())] {
                let fast = processed_scores(&logits, &history, &options, mask).unwrap();
                let reference = processed_scores(&logits, &history, &generic, mask).unwrap();
                assert_eq!(
                    fast.iter()
                        .map(|x| (x.0, x.1.to_bits()))
                        .collect::<Vec<_>>(),
                    reference
                        .iter()
                        .map(|x| (x.0, x.1.to_bits()))
                        .collect::<Vec<_>>()
                );
                for temperature in [0.0, 0.7, 2.0] {
                    for top_k in [0, 1, 20] {
                        let options = Options {
                            temperature,
                            top_k,
                            top_p: 0.8,
                            ..options.clone()
                        };
                        let generic = Options {
                            logit_bias: generic.logit_bias.clone(),
                            ..options.clone()
                        };
                        // Remove extreme logits here to exercise a broad CDF.
                        let fast =
                            Distribution::from_masked(&logits[7..], &history, &options, mask)
                                .unwrap();
                        let reference =
                            Distribution::from_masked(&logits[7..], &history, &generic, mask)
                                .unwrap();
                        assert_eq!(fast.ordered, reference.ordered);
                        for step in 0..100 {
                            let u = counter_uniform(EVALUATION_SEED, 0, step);
                            assert_eq!(fast.draw(u), reference.draw(u));
                        }
                    }
                }
            }
        }
        assert!(processed_scores(&[0.0, f32::NAN], &[], &Options::default(), Some(&[1])).is_err());
    }
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
                fallback: None,
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
            fallback: None,
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
    use super::{EVALUATION_SEED, categorical_bits, categorical_key, counter_uniform};

    #[test]
    fn philox_matches_random123_known_answers() {
        // DEShawResearch/random123 tests/kat_vectors, Philox2x32-10.
        for (counter, key, expected) in [
            ([0, 0], 0, [0xff1d_ae59, 0x6cd1_0df2]),
            ([u32::MAX, u32::MAX], u32::MAX, [0x2c3f_628b, 0xab4f_d7ad]),
            (
                [0x243f_6a88, 0x85a3_08d3],
                0x1319_8a2e,
                [0xdd7c_e038, 0xf62a_4c12],
            ),
        ] {
            assert_eq!(categorical_bits(counter, key), expected);
        }
        assert_ne!(
            categorical_key(EVALUATION_SEED, 1),
            categorical_key(EVALUATION_SEED, 2)
        );
    }

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

#[cfg(test)]
mod point_mass_tests {
    use super::*;
    #[test]
    fn shared_gumbel_draft_preserves_serial_autoregressive_samples() {
        fn law(history: &[u32], draft: bool) -> [f64; 3] {
            let last = history.last().copied().unwrap_or(0) as usize;
            let weights = if draft {
                [0.25, 0.4, 0.35]
            } else {
                [0.2, 0.3, 0.5]
            };
            std::array::from_fn(|i| weights[(i + last + history.len()) % 3])
        }
        fn sample(history: &[u32], seed: u64, draft: bool) -> u32 {
            let key = categorical_key(seed, history.len() as u64);
            law(history, draft)
                .into_iter()
                .enumerate()
                .map(|(token, p)| {
                    let bits = categorical_bits([token as u32, key[1]], key[0])[0];
                    let u = (f64::from(bits >> 9) + 0.5) / 8388608.0;
                    (token as u32, p.ln() - (-u.ln()).ln())
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap()
                .0
        }
        for seed in 0..128 {
            let mut serial = vec![];
            for _ in 0..24 {
                serial.push(sample(&serial, seed, false));
            }
            for block in [1, 2, 5, 8] {
                let mut output = vec![];
                while output.len() < serial.len() {
                    let mut history = output.clone();
                    let mut drafts = vec![];
                    let mut targets = vec![];
                    for _ in 0..block {
                        targets.push(sample(&history, seed, false));
                        let token = sample(&history, seed, true);
                        drafts.push(token);
                        history.push(token);
                    }
                    targets.push(sample(&history, seed, false));
                    let committed = crate::mtp::greedy_commit(&drafts, &targets).unwrap();
                    output.extend(committed.into_iter().take(serial.len() - output.len()));
                }
                assert_eq!(output, serial, "seed={seed}, block={block}");
            }
        }
    }
    #[test]
    fn sampled_target_prefix_commit_retains_categorical_law() {
        let laws = [[0.2_f64, 0.3, 0.5], [0.6, 0.1, 0.3], [0.3, 0.4, 0.3]];
        let mut first = [0usize; 3];
        let mut accepted = [0usize; 2];
        let n = 32768;
        for seed in 0..n {
            let targets = laws
                .iter()
                .enumerate()
                .map(|(step, law)| {
                    let key = categorical_key(seed as u64, step as u64);
                    law.iter()
                        .enumerate()
                        .map(|(token, p)| {
                            let bits = categorical_bits([token as u32, key[1]], key[0])[0];
                            let u = (f64::from(bits >> 9) + 0.5) / 8388608.0;
                            (token as u32, p.ln() - (-u.ln()).ln())
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap()
                        .0
                })
                .collect::<Vec<_>>();
            let committed = crate::mtp::greedy_commit(&[1, 0], &targets).unwrap();
            first[committed[0] as usize] += 1;
            accepted[0] += usize::from(committed.len() > 1);
            accepted[1] += usize::from(committed.len() > 2);
        }
        for (observed, p) in first.into_iter().zip(laws[0]) {
            assert!((observed as f64 / n as f64 - p).abs() < 0.02);
        }
        assert!((accepted[0] as f64 / n as f64 - 0.3).abs() < 0.02);
        assert!((accepted[1] as f64 / n as f64 - 0.18).abs() < 0.02);
    }
    #[test]
    fn deterministic_proposal_retains_exact_residual_sampling() {
        let q = Distribution::point_mass(1, 3).unwrap();
        assert_eq!(q.probability(1), 1.0);
        assert_eq!(q.probability(0), 0.0);
        assert!(Distribution::point_mass(3, 3).is_err());
        let p = Distribution::from_logits(
            &[0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
            &[],
            &Options::default(),
        )
        .unwrap();
        let residual = p.residual(&q).unwrap();
        assert_eq!(residual.probability(1), 0.0);
        assert!((residual.probability(0) - 2.0 / 7.0).abs() < 1e-6);
        assert!((residual.probability(2) - 5.0 / 7.0).abs() < 1e-6);
    }
}
