//! Request-local constraints and target probability reporting. Speculative
//! branches fork the parser; only committed target tokens advance live state.
use super::{Distribution, Options, counter_uniform};

type Result<T> = std::result::Result<T, String>;

pub trait Constraint: Send {
    fn fork(&self) -> Box<dyn Constraint>;
    /// A completed control phase may relinquish masking and resume GPU sampling.
    fn needs_mask(&self) -> bool {
        true
    }
    /// Packed little-endian token bits, matching the model vocabulary.
    fn mask(&mut self) -> Result<Vec<u32>>;
    fn consume(&mut self, token: u32) -> Result<()>;
    fn finished(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct TokenLogprob {
    pub token: u32,
    pub logprob: f64,
    pub top: Vec<(u32, f64)>,
}

pub struct Law {
    pub distribution: Distribution,
    reporting: Option<Distribution>,
    top: usize,
}
#[derive(Default)]
pub struct Decoder {
    constraint: Option<Box<dyn Constraint>>,
    records: Vec<TokenLogprob>,
    failure: Option<String>,
}
impl Decoder {
    pub fn set_constraint(&mut self, constraint: Box<dyn Constraint>) {
        self.constraint = Some(constraint);
    }
    pub fn constrained(&self) -> bool {
        self.constraint.as_ref().is_some_and(|c| c.needs_mask())
    }
    pub fn finished(&self) -> bool {
        self.failure.is_some() || self.constraint.as_ref().is_some_and(|c| c.finished())
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
    pub fn fail(&mut self, error: String) {
        self.failure = Some(error);
    }
    pub fn fork(&self) -> Self {
        Self {
            constraint: self.constraint.as_ref().map(|c| c.fork()),
            records: vec![],
            failure: None,
        }
    }
    pub fn take_records(&mut self) -> Vec<TokenLogprob> {
        std::mem::take(&mut self.records)
    }
    /// Reuse a validated GPU argmax when it is allowed, or select a singleton
    /// mask without downloading logits. The caller must use unprocessed greedy
    /// sampling without probability reporting and validate finite GPU logits.
    pub fn greedy_candidate(&mut self, selected: u32, vocab: usize) -> Result<Option<u32>> {
        if selected as usize >= vocab {
            return Err("Selected token outside vocabulary".into());
        }
        if !self.constrained() {
            return Ok(Some(selected));
        }
        let result = (|| {
            let mask = self
                .constraint
                .as_mut()
                .expect("Active constraint")
                .mask()?;
            if mask
                .get(selected as usize / 32)
                .is_some_and(|bits| bits & (1 << (selected % 32)) != 0)
            {
                return Ok(Some(selected));
            }
            let mut only = None;
            for (word, &bits) in mask.iter().take(vocab.div_ceil(32)).enumerate() {
                let valid = (vocab - word * 32).min(32);
                let bits = if valid == 32 {
                    bits
                } else {
                    bits & ((1u32 << valid) - 1)
                };
                if bits == 0 {
                    continue;
                }
                if only.is_some() || bits.count_ones() > 1 {
                    return Ok(None);
                }
                only = Some((word * 32 + bits.trailing_zeros() as usize) as u32);
            }
            only.map(Some)
                .ok_or_else(|| "Grammar permits no tokens".to_owned())
        })();
        if let Err(error) = &result {
            self.fail(error.clone());
        }
        result
    }
    pub fn law(&mut self, logits: &[f32], history: &[u32], options: &Options) -> Result<Law> {
        let mask = match self.constraint.as_mut().map(|c| c.mask()).transpose() {
            Ok(mask) => mask,
            Err(error) => {
                self.fail(error.clone());
                return Err(error);
            }
        };
        let distribution = Distribution::from_masked(logits, history, options, mask.as_deref());
        if let Err(error) = &distribution
            && error == "Grammar permits no tokens"
        {
            self.fail(error.clone());
        }
        let distribution = distribution?;
        // Report the target law before top-k/nucleus truncation. Greedy selects
        // an argmax but still reports real softmax probabilities at T=1.
        let reporting = if options.top_logprobs.is_some() {
            let report = Options {
                temperature: if options.temperature == 0.0 {
                    1.0
                } else {
                    options.temperature
                },
                top_p: 1.0,
                top_k: 0,
                ..options.clone()
            };
            Some(Distribution::from_masked(
                logits,
                history,
                &report,
                mask.as_deref(),
            )?)
        } else {
            None
        };
        Ok(Law {
            distribution,
            reporting,
            top: options.top_logprobs.unwrap_or(0),
        })
    }
    pub fn consume(&mut self, token: u32) -> Result<()> {
        if let Some(c) = self.constraint.as_mut()
            && let Err(error) = c.consume(token)
        {
            self.fail(error.clone());
            return Err(error);
        }
        Ok(())
    }
    pub fn commit(&mut self, token: u32, law: Law) -> Result<()> {
        self.consume(token)?;
        if let Some(reporting) = law.reporting {
            let log = |p: f64| p.ln().max(-9999.0);
            self.records.push(TokenLogprob {
                token,
                logprob: log(reporting.probability(token)),
                top: reporting
                    .ordered
                    .iter()
                    .take(law.top)
                    .map(|&(t, p)| (t, log(p)))
                    .collect(),
            });
        }
        Ok(())
    }
}

/// Both p and q use the same grammar at the accepted prefix. Later verification
/// rows are discarded on rejection or grammar termination, including their
/// parser changes and probability records.
pub fn verify(
    proposals: &[crate::mtp::Proposal],
    logits: &[f32],
    history: &[u32],
    options: &Options,
    decoder: &mut Decoder,
    step: usize,
    vocab: usize,
) -> Result<Vec<u32>> {
    if vocab == 0 || logits.len() < (proposals.len() + 1) * vocab {
        return Err("Invalid constrained verification logits".into());
    }
    let mut history = history.to_vec();
    let mut committed = vec![];
    for i in 0..=proposals.len() {
        let law = decoder.law(&logits[i * vocab..(i + 1) * vocab], &history, options)?;
        let (token, rejected) = if let Some(q) = proposals.get(i) {
            let qx = q.distribution.probability(q.token);
            if qx <= 0.0 {
                return Err("Proposal outside constrained draft distribution".into());
            }
            let p = &law.distribution;
            if counter_uniform(options.seed, 0x4d54_5002, (step + i) as u64)
                < (p.probability(q.token) / qx).min(1.0)
            {
                (q.token, false)
            } else {
                (
                    p.residual(&q.distribution)?.draw(counter_uniform(
                        options.seed,
                        0x4d54_5003,
                        (step + i) as u64,
                    ))?,
                    true,
                )
            }
        } else {
            (
                law.distribution
                    .draw(counter_uniform(options.seed, 0, (step + i) as u64))?,
                false,
            )
        };
        decoder.commit(token, law)?;
        committed.push(token);
        history.push(token);
        if rejected || decoder.finished() {
            break;
        }
    }
    Ok(committed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpu_greedy_respects_masks_singletons_and_vocabulary_tail() {
        #[derive(Clone)]
        struct Fixed(Vec<u32>);
        impl Constraint for Fixed {
            fn fork(&self) -> Box<dyn Constraint> {
                Box::new(self.clone())
            }
            fn mask(&mut self) -> Result<Vec<u32>> {
                Ok(self.0.clone())
            }
            fn consume(&mut self, _: u32) -> Result<()> {
                Ok(())
            }
            fn finished(&self) -> bool {
                false
            }
        }
        for (mask, selected, expected) in [
            (vec![3], 0, Some(0)),
            (vec![3], 2, None),
            (vec![2], 0, Some(1)),
            (vec![0, u32::MAX], 0, Some(32)),
        ] {
            let mut decoder = Decoder::default();
            decoder.set_constraint(Box::new(Fixed(mask)));
            assert_eq!(decoder.greedy_candidate(selected, 33).unwrap(), expected);
        }
        let mut decoder = Decoder::default();
        decoder.set_constraint(Box::new(Fixed(vec![0])));
        assert!(decoder.greedy_candidate(0, 3).is_err());
        assert!(decoder.finished() && decoder.failure().is_some());
        let mut decoder = Decoder::default();
        assert_eq!(decoder.greedy_candidate(1, 3).unwrap(), Some(1));
        assert!(decoder.greedy_candidate(3, 3).is_err());
        assert!(decoder.failure().is_none());
    }
    #[derive(Clone)]
    struct Alternating(usize);
    impl Constraint for Alternating {
        fn fork(&self) -> Box<dyn Constraint> {
            Box::new(self.clone())
        }
        fn mask(&mut self) -> Result<Vec<u32>> {
            Ok(vec![1 << (self.0 % 2)])
        }
        fn consume(&mut self, t: u32) -> Result<()> {
            if t as usize != self.0 % 2 {
                return Err("Invalid grammar transition".into());
            }
            self.0 += 1;
            Ok(())
        }
        fn finished(&self) -> bool {
            self.0 == 3
        }
    }
    #[test]
    fn target_and_draft_masks_rejection_and_records_are_private() {
        let mut live = Decoder::default();
        live.set_constraint(Box::new(Alternating(0)));
        let mut draft = live.fork();
        let opt = Options {
            temperature: 0.0,
            top_logprobs: Some(2),
            ..Default::default()
        };
        let mut proposals = vec![];
        for t in [0, 1] {
            let law = draft.law(&[1., 9.], &[], &opt).unwrap();
            draft.consume(t).unwrap();
            proposals.push(crate::mtp::Proposal {
                token: t,
                distribution: law.distribution,
            });
        }
        let logits = [1., 9., 9., 1., 1., 9.];
        assert_eq!(
            verify(&proposals, &logits, &[], &opt, &mut live, 0, 2).unwrap(),
            [0, 1, 0]
        );
        assert!(live.finished());
        let records = live.take_records();
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|r| r.top.len() == 1 && r.logprob == 0.0));
        assert!(!draft.finished());
        let opt = Options {
            temperature: 0.0,
            top_logprobs: Some(2),
            ..Default::default()
        };
        let q = Distribution::from_logits(&[9., 1.], &[], &opt).unwrap();
        let mut live = Decoder::default();
        assert_eq!(
            verify(
                &[crate::mtp::Proposal {
                    token: 0,
                    distribution: q
                }],
                &[1., 9., f32::NAN, f32::NAN],
                &[],
                &opt,
                &mut live,
                0,
                2
            )
            .unwrap(),
            [1]
        );
        assert_eq!(live.take_records().len(), 1);
    }
    #[test]
    fn greedy_reports_target_softmax_and_bias_changes_selection() {
        let mut decoder = Decoder::default();
        let opt = Options {
            temperature: 0.0,
            top_logprobs: Some(2),
            logit_bias: [(0, 10.)].into(),
            ..Default::default()
        };
        let law = decoder.law(&[1., 2.], &[], &opt).unwrap();
        assert_eq!(law.distribution.draw(0.5).unwrap(), 0);
        decoder.commit(0, law).unwrap();
        let report = decoder.take_records().remove(0);
        assert!((report.logprob - (1.0 / (1.0 + (-9f64).exp())).ln()).abs() < 1e-12);
        assert_eq!(report.top.len(), 2);
    }
    #[test]
    fn impossible_grammar_is_request_failure_but_invalid_logits_remain_engine_failure() {
        struct Empty;
        impl Constraint for Empty {
            fn fork(&self) -> Box<dyn Constraint> {
                Box::new(Self)
            }
            fn mask(&mut self) -> Result<Vec<u32>> {
                Ok(vec![0])
            }
            fn consume(&mut self, _: u32) -> Result<()> {
                unreachable!()
            }
            fn finished(&self) -> bool {
                false
            }
        }
        let mut decoder = Decoder::default();
        decoder.set_constraint(Box::new(Empty));
        assert!(decoder.law(&[1., 2.], &[], &Options::default()).is_err());
        assert!(decoder.failure().is_some() && decoder.finished());
        let mut decoder = Decoder::default();
        assert!(decoder.law(&[f32::NAN], &[], &Options::default()).is_err());
        assert!(decoder.failure().is_none());
    }
}
