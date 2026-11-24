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
pub(crate) use prepared::{BackgroundLane, Prepared, prepare_lane};

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
mod tests;
