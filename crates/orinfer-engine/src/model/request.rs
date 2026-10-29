//! Backend-neutral request handle consumed by the shared continuous worker.
use crate::backend::ascend::model::provider::Request as Ascend;
use crate::runtime::requests::RequestState as Cuda;
enum Implementation {
    Cuda(Box<Cuda>),
    Ascend(Box<Ascend>),
}
pub struct RequestState(Implementation);
impl RequestState {
    pub(super) fn cuda(inner: Cuda) -> Self {
        Self(Implementation::Cuda(Box::new(inner)))
    }
    pub(super) fn ascend(inner: Ascend) -> Self {
        Self(Implementation::Ascend(Box::new(inner)))
    }
    pub(super) fn cuda_mut(&mut self) -> crate::artifact::Result<&mut Cuda> {
        match &mut self.0 {
            Implementation::Cuda(inner) => Ok(inner),
            _ => Err("Request belongs to another backend".into()),
        }
    }
    pub(super) fn ascend_mut(&mut self) -> crate::artifact::Result<&mut Ascend> {
        match &mut self.0 {
            Implementation::Ascend(inner) => Ok(inner),
            _ => Err("Request belongs to another backend".into()),
        }
    }
    pub(super) fn cuda_ref(&self) -> Option<&Cuda> {
        match &self.0 {
            Implementation::Cuda(inner) => Some(inner),
            _ => None,
        }
    }
    pub(super) fn ascend_ref(&self) -> Option<&Ascend> {
        match &self.0 {
            Implementation::Ascend(inner) => Some(inner),
            _ => None,
        }
    }
    pub fn retain_shared_prefixes_only(&mut self, full_prompt_shared: bool) {
        match &mut self.0 {
            Implementation::Cuda(inner) => inner.retain_shared_prefixes_only(full_prompt_shared),
            Implementation::Ascend(inner) => inner.retain_shared_prefixes_only(full_prompt_shared),
        }
    }
    pub fn set_constraint(
        &mut self,
        constraint: Box<dyn crate::sampling::Constraint>,
    ) -> crate::artifact::Result<()> {
        match &mut self.0 {
            Implementation::Cuda(inner) => inner.set_constraint(constraint),
            Implementation::Ascend(inner) => inner.set_constraint(constraint),
        }
    }
    pub fn take_logprobs(&mut self) -> Vec<crate::sampling::TokenLogprob> {
        match &mut self.0 {
            Implementation::Cuda(inner) => inner.take_logprobs(),
            Implementation::Ascend(inner) => inner.take_logprobs(),
        }
    }
    pub fn generated_tokens(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner.generated_tokens(),
            Implementation::Ascend(inner) => inner.generated_tokens(),
        }
    }
    pub fn computed_prompt_tokens(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner.computed_prompt_tokens(),
            Implementation::Ascend(inner) => inner.computed_prompt_tokens(),
        }
    }
    pub fn decode_slack_s(&self, target_tpot_ms: f64) -> Option<f64> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.decode_slack_s(target_tpot_ms),
            Implementation::Ascend(inner) => inner.decode_slack_s(target_tpot_ms),
        }
    }
    pub fn is_prefilling(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => inner.is_prefilling(),
            Implementation::Ascend(inner) => inner.is_prefilling(),
        }
    }
    pub fn remaining_prefill_tokens(&self) -> usize {
        match &self.0 {
            Implementation::Cuda(inner) => inner.remaining_prefill_tokens(),
            Implementation::Ascend(inner) => inner.remaining_prefill_tokens(),
        }
    }
    pub fn is_finished(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => inner.is_finished(),
            Implementation::Ascend(inner) => inner.is_finished(),
        }
    }
    pub fn constraint_finished(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(inner) => inner.constraint_finished(),
            Implementation::Ascend(inner) => inner.constraint_finished(),
        }
    }
    pub fn failure(&self) -> Option<&str> {
        match &self.0 {
            Implementation::Cuda(inner) => inner.failure(),
            Implementation::Ascend(inner) => inner.failure(),
        }
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        match &self.0 {
            Implementation::Cuda(inner) => inner.prefix_statistics(),
            Implementation::Ascend(inner) => inner.prefix_statistics(),
        }
    }
    pub fn speculation_statistics(&self) -> &crate::mtp::Statistics {
        match &self.0 {
            Implementation::Cuda(inner) => inner.speculation_statistics(),
            Implementation::Ascend(inner) => inner.speculation_statistics(),
        }
    }
}
