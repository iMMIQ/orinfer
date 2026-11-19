//! Backend-neutral request handle consumed by the shared continuous worker.
use crate::artifact::Result;
use crate::models::deepseek_v41::provider::Request as Ascend;
use crate::runtime::requests::RequestState as Cuda;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

pub(crate) struct RequestCore {
    pub(crate) owner: u64,
    abandoned: Arc<Mutex<Vec<usize>>>,
    pub(crate) slot: usize,
    pub(crate) input: Vec<u32>,
    pub(crate) history: Vec<u32>,
    pub(crate) limit: usize,
    pub(crate) sampling: crate::sampling::Options,
    pub(crate) offset: usize,
    pub(crate) prefilling: bool,
    pub(crate) released: bool,
    pub(crate) mtp: crate::mtp::Statistics,
    pub(crate) generated: usize,
    pub(crate) last_output: Instant,
    pub(crate) last_prefill: Instant,
    pub(crate) shared_work: bool,
    pub(crate) decoder: crate::sampling::Decoder,
    pub(crate) cache_prompt: bool,
    pub(crate) cache_generated: bool,
}
impl Drop for RequestCore {
    fn drop(&mut self) {
        if !self.released
            && let Ok(mut abandoned) = self.abandoned.lock()
        {
            abandoned.push(self.slot);
        }
    }
}
impl RequestCore {
    pub(crate) fn new(
        owner: u64,
        slot: usize,
        abandoned: Arc<Mutex<Vec<usize>>>,
        input: Vec<u32>,
        limit: usize,
        sampling: crate::sampling::Options,
        offset: usize,
    ) -> Self {
        let now = Instant::now();
        Self {
            owner,
            slot,
            abandoned,
            history: input.clone(),
            input,
            limit,
            sampling,
            offset,
            prefilling: true,
            released: false,
            mtp: Default::default(),
            generated: 0,
            last_output: now,
            last_prefill: now,
            shared_work: false,
            decoder: Default::default(),
            cache_prompt: true,
            cache_generated: true,
        }
    }
    /// Offline lookahead can retain shared branches without copying one-off
    /// prompts and generated continuations into the cache.
    pub fn retain_shared_prefixes_only(&mut self, full_prompt_shared: bool) {
        self.cache_prompt = full_prompt_shared;
        self.cache_generated = false;
    }
    pub fn set_constraint(
        &mut self,
        constraint: Box<dyn crate::sampling::Constraint>,
    ) -> Result<()> {
        if self.generated != 0 {
            return Err("Attach constraint before generation".into());
        }
        self.decoder.set_constraint(constraint);
        Ok(())
    }
    pub fn take_logprobs(&mut self) -> Vec<crate::sampling::TokenLogprob> {
        self.decoder.take_records()
    }
    pub fn generated_tokens(&self) -> usize {
        self.generated
    }
    pub fn decode_slack_s(&self, target_tpot_ms: f64) -> Option<f64> {
        (!self.prefilling && !self.is_finished())
            .then(|| target_tpot_ms / 1000. - self.last_output.elapsed().as_secs_f64())
    }
    pub fn is_prefilling(&self) -> bool {
        self.prefilling
    }
    pub fn remaining_prefill_tokens(&self) -> usize {
        if self.prefilling {
            self.input.len().saturating_sub(self.offset)
        } else {
            0
        }
    }
    pub fn is_finished(&self) -> bool {
        self.generated == self.limit || self.released || self.decoder.finished()
    }
    pub fn speculation_statistics(&self) -> &crate::mtp::Statistics {
        &self.mtp
    }
    pub fn constraint_finished(&self) -> bool {
        self.decoder.finished()
    }
    pub fn failure(&self) -> Option<&str> {
        self.decoder.failure()
    }
}

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
    fn core(&self) -> &RequestCore {
        match &self.0 {
            Implementation::Cuda(inner) => &inner.core,
            Implementation::Ascend(inner) => &inner.core,
        }
    }
    fn core_mut(&mut self) -> &mut RequestCore {
        match &mut self.0 {
            Implementation::Cuda(inner) => &mut inner.core,
            Implementation::Ascend(inner) => &mut inner.core,
        }
    }
    pub fn retain_shared_prefixes_only(&mut self, full_prompt_shared: bool) {
        self.core_mut()
            .retain_shared_prefixes_only(full_prompt_shared)
    }
    pub fn set_constraint(
        &mut self,
        constraint: Box<dyn crate::sampling::Constraint>,
    ) -> crate::artifact::Result<()> {
        self.core_mut().set_constraint(constraint)
    }
    pub fn take_logprobs(&mut self) -> Vec<crate::sampling::TokenLogprob> {
        self.core_mut().take_logprobs()
    }
    pub fn generated_tokens(&self) -> usize {
        self.core().generated_tokens()
    }
    pub fn computed_prompt_tokens(&self) -> usize {
        self.core()
            .offset
            .saturating_sub(self.prefix_statistics().cached_tokens)
    }
    pub fn decode_slack_s(&self, target_tpot_ms: f64) -> Option<f64> {
        self.core().decode_slack_s(target_tpot_ms)
    }
    pub fn is_prefilling(&self) -> bool {
        self.core().is_prefilling()
    }
    /// Native asynchronous encoders retain charged host media until consumed.
    pub fn has_pending_host_images(&self) -> bool {
        match &self.0 {
            Implementation::Cuda(_) => false,
            Implementation::Ascend(inner) => inner.has_pending_host_images(),
        }
    }
    pub fn remaining_prefill_tokens(&self) -> usize {
        self.core().remaining_prefill_tokens()
    }
    pub fn is_finished(&self) -> bool {
        self.core().is_finished()
    }
    pub fn constraint_finished(&self) -> bool {
        self.core().constraint_finished()
    }
    pub fn failure(&self) -> Option<&str> {
        self.core().failure()
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        match &self.0 {
            Implementation::Cuda(inner) => inner.prefix_statistics(),
            Implementation::Ascend(inner) => inner.prefix_statistics(),
        }
    }
    pub fn speculation_statistics(&self) -> &crate::mtp::Statistics {
        self.core().speculation_statistics()
    }
}
