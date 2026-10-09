//! A request-private reasoning cap, forked together with the grammar for MTP.
use orinfer_engine::sampling::Constraint;
use std::sync::Arc;

pub(super) struct Budget {
    inner: Option<Box<dyn Constraint>>,
    remaining: usize,
    close: u32,
    vocab: usize,
    thinking: bool,
    tokens: Arc<Vec<Vec<u8>>>,
    suffix: Vec<u8>,
    eos: Vec<u32>,
}
impl Budget {
    pub fn new(
        inner: Option<Box<dyn Constraint>>,
        limit: usize,
        close: u32,
        tokens: Arc<Vec<Vec<u8>>>,
        eos: Vec<u32>,
    ) -> Self {
        Self {
            inner,
            remaining: limit,
            close,
            vocab: tokens.len(),
            thinking: true,
            tokens,
            suffix: vec![],
            eos,
        }
    }
}
impl Constraint for Budget {
    fn needs_mask(&self) -> bool {
        self.thinking || self.inner.as_ref().is_some_and(|c| c.needs_mask())
    }
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(Self {
            inner: self.inner.as_ref().map(|c| c.fork()),
            remaining: self.remaining,
            close: self.close,
            vocab: self.vocab,
            thinking: self.thinking,
            tokens: Arc::clone(&self.tokens),
            suffix: self.suffix.clone(),
            eos: self.eos.clone(),
        })
    }
    fn mask(&mut self) -> Result<Vec<u32>, String> {
        let mut mask = match &mut self.inner {
            Some(inner) => inner.mask()?,
            None => vec![u32::MAX; self.vocab.div_ceil(32)],
        };
        if self.thinking && self.remaining == 0 {
            let word = self.close as usize / 32;
            let bit = 1 << (self.close % 32);
            if mask.get(word).is_none_or(|v| v & bit == 0) {
                return Err("Thinking boundary conflicts with output grammar".into());
            }
            mask.fill(0);
            mask[word] = bit;
        } else if self.thinking {
            // Ending the completion inside reasoning leaves no answer. The
            // close delimiter remains available before and at the budget cap.
            for &eos in &self.eos {
                if let Some(word) = mask.get_mut(eos as usize / 32) {
                    *word &= !(1 << (eos % 32));
                }
            }
        }
        Ok(mask)
    }
    fn consume(&mut self, token: u32) -> Result<(), String> {
        if self.thinking {
            if token == self.close {
                self.thinking = false;
            } else {
                self.remaining = self
                    .remaining
                    .checked_sub(1)
                    .ok_or("Token exceeded thinking budget")?;
                let bytes = self
                    .tokens
                    .get(token as usize)
                    .ok_or("Token outside tokenizer")?;
                self.suffix.extend_from_slice(
                    bytes
                        .strip_prefix(&[llguidance::toktrie::TokTrie::SPECIAL_TOKEN_MARKER])
                        .unwrap_or(bytes),
                );
                const END: &[u8] = b"</think>";
                if self.suffix.windows(END.len()).any(|v| v == END) {
                    self.thinking = false;
                }
                let keep = self.suffix.len().saturating_sub(END.len() - 1);
                self.suffix.drain(..keep);
            }
        }
        if let Some(inner) = &mut self.inner {
            inner.consume(token)?;
        }
        Ok(())
    }
    fn finished(&self) -> bool {
        self.inner.as_ref().is_some_and(|c| c.finished())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orinfer_engine::sampling::{Decoder, Options, verify};
    #[test]
    fn target_and_mtp_draft_cross_the_budget_without_sharing_state() {
        let mut live = Decoder::default();
        let tokens = Arc::new(vec![b"a".to_vec(), b"b".to_vec(), b"</think>".to_vec()]);
        live.set_constraint(Box::new(Budget::new(
            None,
            1,
            2,
            Arc::clone(&tokens),
            vec![],
        )));
        let mut draft = live.fork();
        let options = Options {
            temperature: 0.,
            ..Default::default()
        };
        let mut proposals = vec![];
        for expected in [0, 2, 0] {
            let law = draft.law(&[9., 1., 0.], &[], &options).unwrap();
            let token = law.distribution.draw(0.5).unwrap();
            assert_eq!(token, expected);
            draft.consume(token).unwrap();
            proposals.push(orinfer_engine::mtp::Proposal {
                token,
                distribution: law.distribution,
            });
        }
        let logits = [9., 1., 0., 9., 1., 0., 9., 1., 0., 9., 1., 0.];
        assert_eq!(
            verify(&proposals, &logits, &[], &options, &mut live, 0, 3).unwrap(),
            [0, 2, 0, 0]
        );
        assert!(!live.finished());
        assert!(!live.constrained());
        let mut zero = Budget::new(None, 0, 2, tokens, vec![]);
        assert_eq!(zero.mask().unwrap(), [4]);
        zero.consume(2).unwrap();
        assert_eq!(zero.mask().unwrap(), [u32::MAX]);
    }
    #[test]
    fn naturally_closed_split_delimiters_do_not_cap_the_answer() {
        let tokens = Arc::new(vec![
            b"</thi".to_vec(),
            b"nk>".to_vec(),
            b"</think>".to_vec(),
        ]);
        let mut budget = Budget::new(None, 2, 2, tokens, vec![]);
        budget.consume(0).unwrap();
        budget.consume(1).unwrap();
        assert!(!budget.needs_mask());
        assert_eq!(budget.mask().unwrap(), [u32::MAX]);
        budget.consume(0).unwrap();
    }
    #[test]
    fn eos_waits_for_the_reasoning_boundary() {
        let tokens = Arc::new(vec![b"a".to_vec(), b"eos".to_vec(), b"</think>".to_vec()]);
        let mut budget = Budget::new(None, 1, 2, tokens, vec![1]);
        assert_eq!(budget.mask().unwrap()[0] & 2, 0);
        budget.consume(2).unwrap();
        assert_ne!(budget.mask().unwrap()[0] & 2, 0);
    }
    #[test]
    fn greedy_gpu_candidates_cross_the_cap_on_private_draft_state() {
        let tokens = Arc::new(vec![b"a".to_vec(), b"eos".to_vec(), b"</think>".to_vec()]);
        let mut target = Decoder::default();
        target.set_constraint(Box::new(Budget::new(None, 1, 2, tokens, vec![1])));
        let mut draft = target.fork();
        assert_eq!(draft.greedy_candidate(0, 3).unwrap(), Some(0));
        draft.consume(0).unwrap();
        assert_eq!(draft.greedy_candidate(0, 3).unwrap(), Some(2));
        draft.consume(2).unwrap();
        assert!(!draft.constrained());
        assert!(target.constrained());
        assert_eq!(target.greedy_candidate(1, 3).unwrap(), None);
        assert_eq!(target.greedy_candidate(0, 3).unwrap(), Some(0));
        target.consume(0).unwrap();
        assert_eq!(target.greedy_candidate(1, 3).unwrap(), Some(2));
        target.consume(2).unwrap();
        assert_eq!(target.greedy_candidate(1, 3).unwrap(), Some(1));
    }
}
