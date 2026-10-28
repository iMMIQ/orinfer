//! Persistent requests and iteration-level execution. API transport owns output
//! parsing; this module owns GPU state, sampling counters and prefix endpoints.
use super::*;
use crate::{architecture::BatchSegment, prefix::Media, scheduler};
use std::collections::{BTreeMap, BTreeSet};
mod prefill;
#[cfg(test)]
mod profile;
mod prompt_lookup;
mod sampling;
pub(super) mod speculation;
#[cfg(test)]
mod tests;

#[derive(serde::Deserialize)]
pub struct GenerationInput {
    pub input_tokens: Vec<u32>,
    #[serde(default)]
    pub images: Vec<crate::vision::ImageInput>,
    #[serde(default)]
    pub audios: Vec<crate::audio::AudioInput>,
    pub max_new_tokens: usize,
    pub sampling: crate::sampling::Options,
    #[serde(default)]
    pub prefix_hints: Vec<usize>,
}

pub struct RequestState {
    owner: u64,
    abandoned: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    slot: usize,
    input: Vec<u32>,
    history: Vec<u32>,
    limit: usize,
    sampling: crate::sampling::Options,
    offset: usize,
    prefilling: bool,
    released: bool,
    media: Media,
    warm: super::speculation::PrefillWarm,
    checkpoints: BTreeSet<usize>,
    prefill_frame: Option<prefill::Frame>,
    prefix: super::prefix::Context,
    mtp: crate::mtp::Statistics,
    generated: usize,
    served: usize,
    last_output: Instant,
    last_prefill: Instant,
    credit_s: f64,
    lookup_costs: BTreeMap<usize, speculation::Profile>,
    prompt_lookup: prompt_lookup::PromptLookup,
    mtp_blocked: bool,
    draft_acceptance: speculation::control::Acceptance,
    decoder: crate::sampling::Decoder,
    cache_prompt: bool,
    cache_generated: bool,
    pending_audio: Vec<crate::audio::AudioInput>,
}
impl Drop for RequestState {
    fn drop(&mut self) {
        if !self.released
            && let Ok(mut abandoned) = self.abandoned.lock()
        {
            abandoned.push(self.slot);
        }
    }
}
impl RequestState {
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
    pub fn computed_prompt_tokens(&self) -> usize {
        self.offset
            .saturating_sub(self.prefix.statistics.cached_tokens)
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
    pub fn constraint_finished(&self) -> bool {
        self.decoder.finished()
    }
    pub fn failure(&self) -> Option<&str> {
        self.decoder.failure()
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix.statistics
    }
    pub fn speculation_statistics(&self) -> &crate::mtp::Statistics {
        &self.mtp
    }
}
#[derive(Debug)]
pub struct StepOutput {
    /// Index in the slice supplied to advance_requests, independent of slot ID.
    pub request: usize,
    pub tokens: Vec<u32>,
}

enum Work {
    Speculative(usize, usize),
    SpeculativeBatch(speculation::BatchChoice),
    MixedSpeculative(Vec<(usize, usize)>),
    Prefill(usize, usize, bool, f64),
    Batch(Vec<(usize, usize)>),
    Idle,
}

pub(super) struct Reservation {
    kv_bytes: usize,
    context: usize,
    sampling: bool,
    vision: bool,
    audio: bool,
}

impl ModelRuntime {
    pub(super) fn reap_abandoned(&mut self) -> Result<()> {
        let slots = std::mem::take(
            &mut *self
                .abandoned
                .lock()
                .map_err(|_| "Abandoned arena queue poisoned")?,
        );
        for (index, &slot) in slots.iter().enumerate() {
            if let Err(error) = self
                .execution
                .release_sequence(slot, &self.manifest.reset_buffers)
            {
                self.abandoned
                    .lock()
                    .map_err(|_| "Abandoned arena queue poisoned")?
                    .extend_from_slice(&slots[index..]);
                return Err(error);
            }
            self.reserved_requests.remove(&slot);
        }
        self.execution.collect_snapshots()
    }
    fn validate_generation(&self, input: &GenerationInput) -> Result<usize> {
        let context = input
            .input_tokens
            .len()
            .checked_add(input.max_new_tokens)
            .ok_or("Context budget overflow")?;
        if input.input_tokens.is_empty()
            || input.max_new_tokens == 0
            || context > self.manifest.max_context
            || input
                .input_tokens
                .iter()
                .any(|&id| id as usize >= self.manifest.vocab)
        {
            return Err("Invalid request input or context budget".into());
        }
        input.sampling.validate()?;
        Ok(context)
    }
    fn request_limits(&self, input: &GenerationInput) -> Result<BTreeMap<String, usize>> {
        let context = self.validate_generation(input)?;
        let mut limits = BTreeMap::new();
        if let Some(kv) = &self.manifest.kv_cache
            && kv.demand_mapping
        {
            for (name, stride) in &kv.buffers {
                limits.insert(
                    name.clone(),
                    context.checked_mul(*stride).ok_or("KV budget overflow")?,
                );
            }
        }
        if let Some(name) = &self.request_hidden_ring {
            let buffer = self
                .manifest
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing request hidden ring")?;
            let rows = buffer.shape[0];
            // Kernels retain their full-ring modulo. A request shorter than
            // that ring can only address rows below its total token budget.
            let allocated_rows = context.next_power_of_two().min(rows);
            limits.insert(name.clone(), allocated_rows * (buffer.bytes()? / rows));
        }
        if let Some(v) = &self.manifest.vision {
            let features = input.images.iter().try_fold(0usize, |sum, image| {
                sum.checked_add(v.feature_count(image)?)
                    .ok_or("Image feature count overflow".to_string())
            })?;
            if features > v.max_features {
                return Err("Too many image features".into());
            }
            limits.insert(v.features.clone(), features.max(1) * v.hidden * 2);
            limits.insert(v.feature_index.clone(), context * 4);
            limits.insert(v.mrope_positions.clone(), context * 12);
            if let Some(name) = self
                .manifest
                .mtp
                .as_ref()
                .and_then(|s| s.feature_index.as_ref())
            {
                limits.insert(name.clone(), context * 4);
            }
        } else if !input.images.is_empty() {
            return Err("Model has no image adapter".into());
        }
        if let Some(a) = &self.manifest.audio {
            let features = input.audios.iter().try_fold(0usize, |n, v| {
                n.checked_add(v.features()?)
                    .ok_or("Audio feature overflow".to_string())
            })?;
            if features > a.max_features {
                return Err("Audio exceeds encoder capacity".into());
            }
            limits.insert(a.features.clone(), features.max(1) * a.hidden * 2);
            limits.insert(a.feature_index.clone(), context * 4);
            limits.insert(a.positions.clone(), context * 12);
        } else if !input.audios.is_empty() {
            return Err("Model has no audio adapter".into());
        }
        Ok(limits)
    }
    pub(crate) fn request_hint(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<scheduler::RequestHint> {
        let image_work = images.iter().try_fold(0usize, |n, image| {
            let patches = image
                .grid_height
                .checked_mul(image.grid_width)
                .ok_or("Image work overflow")?;
            n.checked_add(patches).ok_or("Image work overflow")
        })?;
        let audio_work = audios.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.features()?)
                .ok_or("Audio work overflow".to_string())
        })?;
        let image_work = image_work
            .checked_add(audio_work)
            .ok_or("Media work overflow")?;
        Ok(scheduler::RequestHint {
            owner: self.owner_id,
            tokens: input.to_vec(),
            media: self.prefix_media(input, images, audios)?,
            image_work,
        })
    }
    pub(crate) fn waiting_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<scheduler::Waiting> {
        self.hint_cost(&self.request_hint(input, images, &[])?)
    }
    fn startup_key(image_work: usize, context: usize, cached: bool) -> (usize, usize, bool) {
        (
            image_work.next_power_of_two(),
            context.next_power_of_two(),
            cached,
        )
    }
    pub(crate) fn hint_cost(&self, hint: &scheduler::RequestHint) -> Result<scheduler::Waiting> {
        if hint.owner != self.owner_id {
            return Err("Queue descriptor belongs to another model".into());
        }
        let (cached, bytes) = self.prefix_match_media(&hint.tokens, &hint.media);
        let startup_s = self
            .startup_costs
            .get(&Self::startup_key(
                hint.image_work,
                hint.tokens.len(),
                cached > 0,
            ))
            .map(|e| e.upper())
            .unwrap_or(0.01 + hint.image_work as f64 * 0.001);
        Ok(scheduler::Waiting {
            age_s: 0.,
            remaining_s: self.prefill_costs.remaining(cached, hint.tokens.len()),
            restore_s: self.prefill_costs.restore_cost(bytes),
            startup_s,
            remaining_tokens: hint.tokens.len().saturating_sub(cached),
        })
    }
    pub(crate) fn share_prefill_checkpoint(
        &self,
        req: &mut RequestState,
        hint: &scheduler::RequestHint,
    ) -> Result<bool> {
        if req.owner != self.owner_id
            || hint.owner != self.owner_id
            || !req.prefilling
            || self.prefix_cache.budget == 0
        {
            return Ok(false);
        }
        let common = req
            .input
            .iter()
            .zip(&hint.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let common = req.media.common_tokens(&hint.media, common);
        if common < 32 {
            return Ok(false);
        }
        let bytes: usize = self.prefix_ranges(common)?.values().map(|r| r.bytes).sum();
        if bytes > self.prefix_cache.budget
            || self
                .prefill_costs
                .score(common, hint.tokens.len(), bytes)
                .is_none()
        {
            return Ok(false);
        }
        if common == req.input.len() {
            req.cache_prompt = true;
            return Ok(true);
        }
        if req.checkpoints.contains(&common) {
            return Ok(true);
        }
        if common <= req.offset || req.prefill_frame.is_some() {
            return Ok(false);
        }
        if self.admit_prefix_checkpoint(
            common,
            req.input.len(),
            req.offset,
            &req.checkpoints,
            true,
        )? {
            req.checkpoints.insert(common);
            return Ok(true);
        }
        Ok(false)
    }
    pub(crate) fn decode_admission_allowed(
        &self,
        active: &[&RequestState],
        prompt_tokens: usize,
        options: &scheduler::Options,
    ) -> bool {
        if options.policy == scheduler::Policy::Throughput
            || active.len() < 2
            || active.iter().any(|r| r.prefilling)
        {
            return true;
        }
        let context: usize = active.iter().map(|r| r.history.len()).sum();
        let key = (
            active.len() + 1,
            active.len() + 1,
            (context + prompt_tokens).next_power_of_two(),
            0,
        );
        let Some(next) = self.iteration_costs.get(&key).filter(|e| e.samples >= 4) else {
            return true;
        };
        if next.upper() * 1000. > options.target_tpot_ms {
            return false;
        }
        let previous = (active.len(), active.len(), context.next_power_of_two(), 0);
        self.iteration_costs
            .get(&previous)
            .filter(|e| e.samples >= 4)
            .is_none_or(|old| {
                (active.len() + 1) as f64 / next.mean >= active.len() as f64 / old.mean * 0.99
            })
    }
    pub(crate) fn can_admit_request(
        &mut self,
        input: &GenerationInput,
        options: &scheduler::Options,
    ) -> Result<bool> {
        self.reap_abandoned()?;
        options.validate()?;
        if self.reserved_requests.len() >= options.max_active {
            return Ok(false);
        }
        let context = self.validate_generation(input)?;
        let limits = self.request_limits(input)?;
        let mut fixed = self.execution.additional_state_bytes(&limits)?;
        let mut prospective = self.execution.additional_kv_bytes(context, &limits)?;
        let sampling = input.sampling.temperature > 0. && input.sampling.top_logprobs.is_none();
        let mut programs = vec!["decode", "prefill", "head"];
        if !input.images.is_empty()
            && let Some(vision) = &self.manifest.vision
        {
            programs.extend(vision.plans.iter().map(|p| p.program.as_str()));
        }
        if !input.audios.is_empty() {
            programs.extend(["audio_stem", "audio_encoder"]);
        }
        let workspace = self.pending_workspace(context, sampling, &programs)?;
        let future = self
            .reserved_requests
            .values()
            .map(|r| r.kv_bytes)
            .sum::<usize>()
            .saturating_sub(self.execution.active_kv_bytes());
        let mut needed = fixed
            .checked_add(prospective)
            .and_then(|n| n.checked_add(future))
            .and_then(|n| n.checked_add(workspace))
            .and_then(|n| n.checked_add(options.memory_reserve_bytes))
            .ok_or("Admission budget overflow")?;
        // Reclaim idle arenas before evicting useful prefixes. Snapshot bytes
        // are only prospective capacity; they must not hide idle allocations
        // or graphs that need releasing under current memory pressure.
        if self.admission_free_bytes()? < needed {
            self.execution.reclaim_idle_graphs()?;
        }
        if self.admission_free_bytes()? < needed {
            self.execution.reclaim_idle_state()?;
            let additional = self.execution.additional_kv_bytes(context, &limits)?;
            let state = self.execution.additional_state_bytes(&limits)?;
            needed = needed
                .checked_sub(prospective)
                .and_then(|n| n.checked_sub(fixed))
                .and_then(|n| n.checked_add(additional))
                .and_then(|n| n.checked_add(state))
                .ok_or("Reclaimed KV budget overflow")?;
            fixed = state;
            prospective = additional;
        }
        if self
            .admission_free_bytes()?
            .saturating_add(self.prefix_cache.bytes)
            < needed
        {
            // Releasing every cache entry still cannot satisfy the outstanding
            // reservations. Keep useful prefixes until active work completes.
            return self.defer_memory_admission(needed, fixed, prospective, future, workspace);
        }
        while self.admission_free_bytes()? < needed {
            let Some(snapshot) = self.prefix_cache.evict_one() else {
                return self.defer_memory_admission(needed, fixed, prospective, future, workspace);
            };
            self.execution.release_snapshot(snapshot)?;
        }
        self.trim_request_cache(options, fixed + prospective, context, sampling)?;
        self.idle_admission_since = None;
        Ok(true)
    }
    fn defer_memory_admission(
        &mut self,
        needed: usize,
        fixed: usize,
        kv: usize,
        future: usize,
        workspace: usize,
    ) -> Result<bool> {
        self.scheduler_statistics.admission_deferrals += 1;
        // Driver/host accounting can recover after releasing the last arena.
        // Let the worker queue and recheck instead of failing in that transient
        // window. A persistently oversized request still fails after two seconds.
        if self.reserved_requests.is_empty() {
            let since = self.idle_admission_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= std::time::Duration::from_secs(2) {
                let message = format!(
                    "Request context/output budget exceeds available inference memory (free {} MiB, CUDA free {}, host budget {}, required {} MiB: state {}, KV {}, future {}, workspace {})",
                    self.admission_free_bytes()? >> 20,
                    self.execution.free_bytes()? >> 20,
                    scheduler::usable_host_bytes()? >> 20,
                    needed >> 20,
                    fixed >> 20,
                    kv >> 20,
                    future >> 20,
                    workspace >> 20
                );
                crate::error::record_capacity();
                return Err(message);
            }
        } else {
            self.idle_admission_since = None;
        }
        Ok(false)
    }
    fn admission_free_bytes(&self) -> Result<usize> {
        Ok(self
            .execution
            .free_bytes()?
            .min(scheduler::usable_host_bytes()?))
    }
    fn pending_workspace(
        &self,
        context: usize,
        additional_sampler: bool,
        programs: &[&str],
    ) -> Result<usize> {
        // Media encoding can wait behind text packets. Retain its shared
        // workspace budget across later admissions until first materialization.
        let mut programs = programs.to_vec();
        if self.reserved_requests.values().any(|r| r.vision)
            && let Some(vision) = &self.manifest.vision
        {
            programs.extend(vision.plans.iter().map(|p| p.program.as_str()));
        }
        if self.reserved_requests.values().any(|r| r.audio) {
            programs.extend(["audio_stem", "audio_encoder"]);
        }
        let mut bytes = self
            .execution
            .pending_prefill_workspace_bytes(
                context.max(
                    self.reserved_requests
                        .values()
                        .map(|r| r.context)
                        .max()
                        .unwrap_or(0),
                ),
            )?
            .checked_add(self.execution.pending_text_workspace_bytes()?)
            .and_then(|n| n.checked_add(self.execution.pending_workspace_bytes(&programs).ok()?))
            .ok_or("Pending shared workspace overflow")?;
        let samplers = self
            .reserved_requests
            .values()
            .filter(|r| r.sampling)
            .count()
            + usize::from(additional_sampler);
        if samplers > 0
            && let Some(sampling) = &self.manifest.sampling
        {
            bytes = bytes
                .checked_add(
                    self.execution
                        .pending_workspace_bytes(&[&sampling.program])?,
                )
                .ok_or("Pending sampler workspace overflow")?;
            if samplers > 1
                && let Some(batch) = &sampling.batch
            {
                bytes = bytes
                    .checked_add(self.execution.pending_workspace_range(
                        &batch.scratch,
                        samplers.min(batch.capacity) * batch.stride,
                    )?)
                    .ok_or("Pending sampler rows overflow")?;
            }
        }
        Ok(bytes)
    }
    fn trim_request_cache(
        &mut self,
        options: &scheduler::Options,
        additional: usize,
        context: usize,
        additional_sampler: bool,
    ) -> Result<()> {
        let future = self
            .reserved_requests
            .values()
            .map(|r| r.kv_bytes)
            .sum::<usize>()
            .saturating_sub(self.execution.active_kv_bytes())
            .checked_add(self.pending_workspace(
                context,
                additional_sampler,
                &["decode", "prefill", "head"],
            )?)
            .ok_or("Future workspace budget overflow")?;
        let free = self.admission_free_bytes()?;
        let available = free
            .saturating_sub(future)
            .saturating_sub(additional)
            .saturating_sub(options.memory_reserve_bytes);
        self.prefix_cache.budget = self
            .prefix_cache_limit
            .min(self.prefix_cache.bytes.saturating_add(available));
        while self.prefix_cache.bytes > self.prefix_cache.budget {
            if let Some(snapshot) = self.prefix_cache.evict_one() {
                self.execution.release_snapshot(snapshot)?;
            } else {
                break;
            }
        }
        Ok(())
    }
    pub(crate) fn start_request(
        &mut self,
        mut input: GenerationInput,
        cancelled: &impl Fn() -> bool,
    ) -> Result<RequestState> {
        self.reap_abandoned()?;
        let started = Instant::now();

        let context = self.validate_generation(&input)?;
        let image_work: usize = input
            .images
            .iter()
            .map(|image| image.grid_height * image.grid_width)
            .sum();
        let image_work = image_work
            .checked_add(input.audios.iter().try_fold(0usize, |n, a| {
                n.checked_add(a.features()?)
                    .ok_or("Audio work overflow".to_string())
            })?)
            .ok_or("Media work overflow")?;
        let prompt_len = input.input_tokens.len();
        let limits = self.request_limits(&input)?;
        let slot = self.execution.lease_sequence(&limits)?;
        let result = (|| {
            self.execution
                .reset_sequence_reusing_pages(&self.manifest.reset_buffers)?;
            let media = self.prefix_media(&input.input_tokens, &input.images, &input.audios)?;
            self.prepare_owned_visual_capacity(
                &input.input_tokens,
                &mut input.images,
                context,
                cancelled,
            )?;
            self.prepare_audio_metadata(&input.input_tokens, &input.audios, context)?;
            let mut prefix = super::prefix::Context::default();
            let (offset, mut warm) =
                self.restore_prefix(&input.input_tokens, &media, &mut prefix)?;
            if offset > 0
                && offset < input.input_tokens.len()
                && let Some(spec) = self.manifest.mtp.clone()
            {
                if warm.tokens != offset - 1 {
                    return Err("Restored prefix contains an inconsistent MTP cursor".into());
                }
                let at = Instant::now();
                self.mtp_warm_state(
                    &spec,
                    &input.input_tokens[offset..offset + 1],
                    cancelled,
                    ExecutionPhase::Prefill,
                    false,
                )?;
                warm.tokens = offset;
                warm.seconds += at.elapsed().as_secs_f64();
            }
            let mut checkpoints = BTreeSet::new();
            if self.prefix_cache.budget != 0 {
                checkpoints.extend((8192..input.input_tokens.len()).step_by(8192));
                for hint in input
                    .prefix_hints
                    .into_iter()
                    .chain(std::iter::once(prefix.statistics.matched_tokens))
                {
                    if hint > offset
                        && hint < input.input_tokens.len()
                        && self.admit_prefix_checkpoint(
                            hint,
                            input.input_tokens.len(),
                            offset,
                            &checkpoints,
                            hint == prefix.statistics.matched_tokens,
                        )?
                    {
                        checkpoints.insert(hint);
                    }
                }
            }
            self.reserved_requests.insert(
                slot,
                Reservation {
                    kv_bytes: self.execution.reserved_kv_bytes(context)?,
                    context,
                    sampling: input.sampling.temperature > 0.
                        && input.sampling.top_logprobs.is_none(),
                    vision: !input.images.is_empty(),
                    audio: !input.audios.is_empty(),
                },
            );
            self.scheduler_statistics.peak_active = self
                .scheduler_statistics
                .peak_active
                .max(self.reserved_requests.len());
            Ok(RequestState {
                owner: self.owner_id,
                abandoned: std::sync::Arc::clone(&self.abandoned),
                slot,
                history: input.input_tokens.clone(),
                input: input.input_tokens,
                limit: input.max_new_tokens,
                sampling: input.sampling,
                offset,
                prefilling: true,
                released: false,
                media,
                warm,
                checkpoints,
                prefill_frame: None,
                prefix,
                mtp: Default::default(),
                generated: 0,
                served: 0,
                last_output: Instant::now(),
                last_prefill: Instant::now(),
                credit_s: 0.,
                lookup_costs: Default::default(),
                prompt_lookup: Default::default(),
                mtp_blocked: false,
                draft_acceptance: Default::default(),
                decoder: Default::default(),
                cache_prompt: true,
                cache_generated: true,
                pending_audio: if offset < prompt_len {
                    input.audios
                } else {
                    vec![]
                },
            })
        })();
        if let Ok(req) = &result
            && req.pending_audio.is_empty()
        {
            self.startup_costs
                .entry(Self::startup_key(
                    image_work,
                    prompt_len,
                    req.prefix.statistics.cached_tokens > 0,
                ))
                .or_default()
                .observe(
                    (started.elapsed().as_secs_f64() - req.prefix.statistics.restore_s)
                        .max(0.000001),
                );
        }
        if result.is_err() {
            self.reserved_requests.remove(&slot);
            self.execution
                .release_sequence(slot, &self.manifest.reset_buffers)?;
        }
        self.scheduler_statistics.request_start_s += started.elapsed().as_secs_f64();
        if let Ok(request) = &result {
            self.scheduler_statistics.admissions += 1;
            self.scheduler_statistics.prefix_restore_s += request.prefix.statistics.restore_s;
        }
        result
    }
    fn prepare_pending_audio(&mut self, req: &mut RequestState) -> Result<()> {
        if req.pending_audio.is_empty() {
            return Ok(());
        }
        let work = req.pending_audio.iter().try_fold(0usize, |n, a| {
            n.checked_add(a.features()?)
                .ok_or("Audio work overflow".to_string())
        })?;
        let began = Instant::now();
        self.execution.activate_sequence(req.slot)?;
        self.prepare_audio_capacity(
            &req.input,
            &req.pending_audio,
            self.reserved_requests[&req.slot].context,
            &|| false,
        )?;
        let seconds = began.elapsed().as_secs_f64();
        self.startup_costs
            .entry(Self::startup_key(
                work,
                req.input.len(),
                req.prefix.statistics.cached_tokens > 0,
            ))
            .or_default()
            .observe(seconds);
        self.scheduler_statistics.request_start_s += seconds;
        req.pending_audio.clear();
        Ok(())
    }
    fn prepare_audio_cohorts(&mut self, requests: &mut [&mut RequestState]) -> Result<()> {
        let Some(a) = self.manifest.audio.as_ref().filter(|a| a.window_size > 0) else {
            for req in requests {
                self.prepare_pending_audio(req)?;
            }
            return Ok(());
        };
        let (window, capacity) = (a.window_size, a.max_features);
        let mut cohorts = vec![];
        let mut cohort = vec![];
        let mut rows = 0usize;
        for (i, req) in requests.iter().enumerate() {
            if let Some(input) = req.pending_audio.first() {
                let features = input.features()?;
                if rows.div_ceil(window) * window + features > capacity {
                    cohorts.push(std::mem::take(&mut cohort));
                    rows = 0;
                }
                rows = rows.div_ceil(window) * window + features;
                cohort.push((i, features));
            }
        }
        if !cohort.is_empty() {
            cohorts.push(cohort);
        }
        for cohort in cohorts {
            let began = Instant::now();
            self.encode_audio_cohort(
                &cohort
                    .iter()
                    .map(|&(i, _)| (requests[i].slot, &requests[i].pending_audio[0]))
                    .collect::<Vec<_>>(),
            )?;
            let seconds = began.elapsed().as_secs_f64();
            self.scheduler_statistics.request_start_s += seconds;
            for (i, features) in cohort {
                let req = &mut requests[i];
                // Keep a conservative bound for admission alongside decoders;
                // a packed encoder is indivisible at this scheduling point.
                self.startup_costs
                    .entry(Self::startup_key(
                        features,
                        req.input.len(),
                        req.prefix.statistics.cached_tokens > 0,
                    ))
                    .or_default()
                    .observe(seconds);
                req.pending_audio.clear();
            }
        }
        Ok(())
    }
    fn prepare_selected_audio(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
    ) -> Result<()> {
        if selected
            .iter()
            .all(|&(i, _)| requests[i].pending_audio.is_empty())
        {
            return Ok(());
        }
        let owners: BTreeSet<_> = selected.iter().map(|&(i, _)| i).collect();
        let mut active: Vec<_> = requests
            .iter_mut()
            .enumerate()
            .filter(|(i, req)| owners.contains(i) && !req.pending_audio.is_empty())
            .map(|(_, req)| &mut **req)
            .collect();
        self.prepare_audio_cohorts(&mut active)
    }
    fn with_request<T>(
        &mut self,
        req: &mut RequestState,
        run: impl FnOnce(&mut Self, &mut RequestState) -> Result<T>,
    ) -> Result<T> {
        if req.owner != self.owner_id || req.released {
            return Err("Request does not own a live arena in this model".into());
        }
        self.execution.activate_sequence(req.slot)?;
        run(self, req)
    }
    fn prefill_boundary(req: &RequestState) -> usize {
        req.checkpoints
            .range((
                std::ops::Bound::Excluded(req.offset),
                std::ops::Bound::Unbounded,
            ))
            .next()
            .copied()
            .unwrap_or(req.input.len())
    }
    fn after_prefill(&mut self, req: &mut RequestState, chunk: usize) -> Result<Vec<u32>> {
        req.offset += chunk;
        if self.read_control(&self.manifest.position)? as usize != req.offset {
            return Err("Request prefill position mismatch".into());
        }
        let checkpoint = req.checkpoints.contains(&req.offset);
        if let Some(spec) = self.manifest.mtp.clone() {
            let at = Instant::now();
            // A checkpoint must exclude the next prompt token from the draft
            // state. A different continuation can restore this same prefix.
            let end = if !checkpoint && req.offset < req.input.len() {
                req.offset + 1
            } else {
                req.offset
            };
            let shifted = &req.input[req.warm.tokens + 1..end];
            if !shifted.is_empty() {
                self.mtp_warm_state(&spec, shifted, &|| false, ExecutionPhase::Prefill, false)?;
            }
            req.warm.tokens = end - 1;
            req.warm.seconds += at.elapsed().as_secs_f64();
        }
        if checkpoint {
            self.store_prefix(
                &req.input[..req.offset],
                &req.media,
                req.warm.tokens,
                true,
                &mut req.prefix,
            )?;
            // Bridge only after saving the prefix-consistent P-1 draft state.
            // This consumes h[P-1] before the next target block can overwrite
            // its slot in the bounded hidden ring.
            if req.offset < req.input.len()
                && let Some(spec) = self.manifest.mtp.clone()
            {
                let at = Instant::now();
                self.mtp_warm_state(
                    &spec,
                    &req.input[req.offset..req.offset + 1],
                    &|| false,
                    ExecutionPhase::Prefill,
                    false,
                )?;
                req.warm.tokens = req.offset;
                req.warm.seconds += at.elapsed().as_secs_f64();
            }
        }
        if req.offset == req.input.len() {
            self.complete_prefill(req)
        } else {
            Ok(vec![])
        }
    }
    fn complete_prefill(&mut self, req: &mut RequestState) -> Result<Vec<u32>> {
        if let Some(spec) = self.manifest.mtp.clone() {
            let shifted = &req.input[req.warm.tokens + 1..];
            if !shifted.is_empty() {
                let at = Instant::now();
                self.mtp_warm_state(&spec, shifted, &|| false, ExecutionPhase::Prefill, false)?;
                req.warm.seconds += at.elapsed().as_secs_f64();
                req.warm.tokens = req.input.len() - 1;
            }
        }
        if req.cache_prompt {
            self.store_prefix(
                &req.input,
                &req.media,
                req.warm.tokens,
                true,
                &mut req.prefix,
            )?;
        }
        let pending = match self.select_request_token(req) {
            Ok(token) => token,
            Err(_) if req.failure().is_some() => return Ok(vec![]),
            Err(error) => return Err(error),
        };
        self.upload_ids(&self.manifest.token, &[pending])?;
        req.history.push(pending);
        req.generated += 1;
        req.prefilling = false;
        if let Some(spec) = self.manifest.mtp.clone() {
            let at = Instant::now();
            self.mtp_warm(&spec, &[pending], &|| false, ExecutionPhase::Prefill)?;
            req.warm.tokens = req.input.len();
            req.mtp.initial_warm_s = req.warm.seconds + at.elapsed().as_secs_f64();
        }
        req.mtp.committed_tokens = req.generated;
        Ok(vec![pending])
    }
    fn advance_prefill(
        &mut self,
        req: &mut RequestState,
        chunk: usize,
        bounded: bool,
    ) -> Result<(usize, Vec<u32>)> {
        let remaining = Self::prefill_boundary(req) - req.offset;
        if chunk == 0 || chunk > remaining {
            return Err("Invalid selected prefill extent".into());
        }
        self.prepare_pending_audio(req)?;
        let plan = self.manifest.prefill_plan(chunk);
        if chunk != 1 && plan.is_none() {
            return Err("Missing selected prefill shape".into());
        }
        let (chunk, program, head) = plan
            .map(|p| {
                (
                    chunk,
                    p.prefill_program.clone(),
                    Some(p.head_program.clone()),
                )
            })
            .unwrap_or((1, "decode".into(), None));
        let at = Instant::now();
        let load_before = self.execution.batch_statistics().initialization_s();
        self.upload_ids(
            if head.is_none() {
                &self.manifest.token
            } else {
                &self.manifest.input
            },
            &req.input[req.offset..req.offset + chunk],
        )?;
        self.prepare_prefill_inputs(
            &program,
            &req.input[req.offset..req.offset + chunk],
            &req.input[..req.offset],
        )?;
        self.upload_segment_controls(chunk)?;
        self.launch_rows(&program, chunk, ExecutionPhase::Prefill)?;
        if let Some(spec) = &self.manifest.mtp {
            self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
        }
        let regular_s = at.elapsed().as_secs_f64()
            - (self.execution.batch_statistics().initialization_s() - load_before);
        if (req.offset + chunk == req.input.len()
            || req.checkpoints.contains(&(req.offset + chunk)))
            && let Some(head) = head
        {
            self.launch_rows(&head, chunk, ExecutionPhase::Prefill)?;
        }
        let tokens = self.after_prefill(req, chunk)?;
        let seconds = at.elapsed().as_secs_f64();
        // Head/checkpoint/MTP warm-up is terminal work, and module loading is
        // a one-time cost. Neither predicts the next ordinary prompt tile.
        self.prefill_costs
            .observe_at(chunk, req.offset, regular_s.max(0.000001));
        self.scheduler_statistics.prefill_tokens += chunk;
        *self
            .scheduler_statistics
            .prefill_chunk_histogram
            .entry(chunk)
            .or_default() += 1;
        if bounded {
            self.scheduler_statistics.bounded_prefill_iterations += 1;
            self.scheduler_statistics.max_bounded_prefill_s =
                self.scheduler_statistics.max_bounded_prefill_s.max(seconds);
        }
        Ok((chunk, tokens))
    }
    fn iteration_key(
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> (usize, usize, usize, usize) {
        let rows = selected.iter().map(|(_, n)| n).sum::<usize>();
        let context = selected
            .iter()
            .map(|(i, n)| requests[*i].history.len().saturating_mul(*n))
            .sum::<usize>()
            .checked_next_power_of_two()
            .unwrap_or(1);
        (
            selected.len(),
            rows,
            context,
            selected
                .iter()
                .filter(|(i, _)| requests[*i].prefilling)
                .count()
                | (selected
                    .iter()
                    .filter(|(i, n)| {
                        requests[*i].prefilling
                            && requests[*i].offset + n == requests[*i].input.len()
                    })
                    .count()
                    << 8)
                | (selected
                    .iter()
                    .filter(|(i, n)| {
                        requests[*i].prefilling
                            && requests[*i]
                                .checkpoints
                                .contains(&(requests[*i].offset + n))
                    })
                    .count()
                    << 16),
        )
    }
    fn predict_iteration(
        &self,
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> f64 {
        let key = Self::iteration_key(requests, selected);
        if let Some(upper) = self
            .iteration_costs
            .range((key.0, key.1, 0, 0)..=(key.0, key.1, key.2, usize::MAX))
            .filter(|(k, _)| k.3 == key.3)
            .map(|(_, e)| e.upper())
            .max_by(f64::total_cmp)
        {
            return upper;
        }
        if selected
            .iter()
            .any(|(i, n)| *n > 1 && requests[*i].prefilling)
        {
            // Joined projections/expert tiles pay one weight pass. A cold
            // linear rows*context prior greatly overestimates prompt work and
            // can permanently prevent the scheduler from trying useful tiles.
            let prompt_s: f64 = selected
                .iter()
                .filter(|(i, n)| *n > 1 && requests[*i].prefilling)
                .map(|(i, n)| self.prefill_costs.chunk_upper(*n, requests[*i].offset + n))
                .sum();
            // Decoding requests wider than one row are verification segments of
            // the same joined packet: they extend the decode row count instead
            // of each paying a separate prefill weight pass. Summing their
            // chunk costs froze batched speculation out of every probe.
            let decode_rows = selected
                .iter()
                .filter(|(i, n)| *n == 1 || !requests[*i].prefilling)
                .map(|(_, n)| *n)
                .sum::<usize>();
            // The window covers complete execution, including shifted MTP warm
            // and endpoint/cache work; these costs do not vanish in a joint body.
            let endpoints = (key.3 >> 8) & 255;
            let checkpoints = key.3 >> 16;
            let warm = if self.manifest.mtp.is_some() {
                (key.3 & 255) as f64 * 0.015
            } else {
                0.
            };
            return prompt_s * if selected.len() > 1 { 0.75 } else { 1. }
                + decode_rows as f64 * 0.005
                + warm
                + (endpoints + checkpoints) as f64 * 0.02;
        }
        let kv: usize = selected
            .iter()
            .map(|(i, n)| requests[*i].history.len() * n)
            .sum();
        0.09 + key.1 as f64 * 0.001 + kv as f64 * 0.0000001
    }
    fn pack_prefill(
        &self,
        requests: &[&mut RequestState],
        candidates: &[usize],
        selected: &mut Vec<(usize, usize)>,
        cap: usize,
        budget_ms: f64,
    ) -> Result<()> {
        let mut shapes = vec![1];
        if let Some(layout) = &self.manifest.batch_layout {
            shapes.extend(
                layout
                    .profiles
                    .keys()
                    .copied()
                    .filter(|n| cap > 128 || layout.small_mixed_shapes.contains(n)),
            );
        } else if cap != 1 {
            return Err("Missing batch layout".into());
        }
        shapes.sort_unstable();
        // Split cold prompt cohorts into real chunks before sharing the dense
        // projections. Leave room for up to four requests in a 2048-row plan.
        let per_request_cap = if cap > 128 {
            cap / candidates.len().clamp(1, 4)
        } else {
            cap
        };
        let mut used: usize = selected.iter().map(|(_, n)| n).sum();
        for &i in candidates {
            if requests[i].prefill_frame.is_some() {
                continue;
            }
            if used == cap {
                break;
            }
            let remaining = Self::prefill_boundary(requests[i]) - requests[i].offset;
            let dynamic_cap = self
                .manifest
                .batch_layout
                .as_ref()
                .map_or(0, |l| l.segment_tokens);
            let chunk = if dynamic_cap > 0 {
                let limit = dynamic_cap
                    .min(cap - used)
                    .min(remaining)
                    .min(per_request_cap);
                let mut low = 1;
                let mut high = limit;
                let mut best = None;
                while low <= high {
                    let n = low + (high - low) / 2;
                    let mut trial = selected.clone();
                    trial.push((i, n));
                    if self.predict_iteration(requests, &trial) * 1000. <= budget_ms {
                        best = Some(n);
                        low = n + 1;
                    } else {
                        high = n - 1;
                    }
                }
                best
            } else {
                shapes
                    .iter()
                    .copied()
                    .filter(|&n| n <= cap - used && n <= remaining && n <= per_request_cap)
                    .filter(|&n| {
                        let mut trial = selected.clone();
                        trial.push((i, n));
                        self.predict_iteration(requests, &trial) * 1000. <= budget_ms
                    })
                    .max()
            };
            if let Some(chunk) = chunk {
                selected.push((i, chunk));
                used += chunk;
            }
        }
        Ok(())
    }

    fn measured_iteration(
        &self,
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> bool {
        let key = Self::iteration_key(requests, selected);
        self.iteration_costs
            .range((key.0, key.1, 0, 0)..=(key.0, key.1, key.2, usize::MAX))
            .any(|(k, e)| k.3 == key.3 && e.samples >= 2)
    }

    fn native_prefill_chunk(
        &self,
        requests: &[&mut RequestState],
        i: usize,
        budget_s: f64,
        progress_due: bool,
    ) -> Option<usize> {
        if let Some(frame) = &requests[i].prefill_frame {
            return (self.prefill_stage_cost(requests[i], frame.tokens) <= budget_s
                || progress_due)
                .then_some(frame.tokens);
        }
        let remaining = Self::prefill_boundary(requests[i]) - requests[i].offset;
        let shapes = std::iter::once(1).chain(
            self.manifest
                .prefill_plans
                .iter()
                .filter_map(|p| p.chunk(remaining)),
        );
        let fitting = shapes
            .filter(|&n| n <= remaining)
            .filter(|&n| {
                let staged = self
                    .manifest
                    .prefill_plans
                    .iter()
                    .any(|p| p.accepts(n) && !p.stages.is_empty());
                (if staged {
                    self.prefill_stage_cost(requests[i], n)
                } else {
                    self.predict_iteration(requests, &[(i, n)])
                }) <= budget_s
            })
            .max();
        // A soft deadline must not turn an overloaded prefill into endless
        // one-token weight passes. Force only a minimum useful native tile.
        if progress_due && fitting.is_none_or(|n| n == 1) {
            return Some(self.prefill_costs.bounded_chunk_at(
                remaining,
                requests[i].offset,
                budget_s.max(0.) * 1000.,
            ));
        }
        fitting
    }
    pub(crate) fn advance_requests(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
        self.advance_requests_with(requests, options, || {})
    }
    pub(crate) fn advance_requests_with(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &scheduler::Options,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        let mut maintenance = Some(maintenance);
        let mut maintenance_s = 0.;
        let mut overlap = || {
            if let Some(f) = maintenance.take() {
                let start = Instant::now();
                f();
                maintenance_s += start.elapsed().as_secs_f64();
            }
        };
        #[cfg(test)]
        profile::mark("entry");
        self.reap_abandoned()?;
        options.validate()?;
        for r in requests.iter() {
            if r.owner != self.owner_id || r.released {
                return Err("Invalid active request arena".into());
            }
        }
        if requests.is_empty() {
            return Ok(vec![]);
        }
        // Encoder work follows the selected text packet. Encoding every admitted
        // recording here would delay existing decoders and the first prompt
        // behind media that cannot fit in this iteration's token budget.
        self.trim_request_cache(options, 0, 0, false)?;
        self.scheduler_statistics.iterations += 1;
        let at = Instant::now();
        let mut output = vec![];
        let mut initialized = BTreeSet::new();
        let completion_at = Instant::now();
        for (i, req) in requests.iter_mut().enumerate() {
            if req.prefilling && req.offset == req.input.len() {
                let tokens = self.with_request(req, |model, req| model.complete_prefill(req))?;
                initialized.insert(i);
                output.push(StepOutput { request: i, tokens });
            }
        }
        self.scheduler_statistics.prefill_completion_s += completion_at.elapsed().as_secs_f64();
        let has_decoders = requests
            .iter()
            .enumerate()
            .any(|(i, r)| !r.prefilling && !r.is_finished() && !initialized.contains(&i));
        if !has_decoders {
            self.unmerged_prefill_s = 0.;
            self.decode_until = None;
        }
        let feedback_at = Instant::now();
        let feedback_stats = self.execution.batch_statistics();
        self.maintain_lazy_drafts(requests)?;
        let prefill_work = self.select_work(requests, options, &initialized)?;
        let media_at = Instant::now();
        match &prefill_work {
            Work::Prefill(i, tokens, ..) => {
                self.prepare_selected_audio(requests, &[(*i, *tokens)])?;
            }
            Work::Batch(selected) | Work::MixedSpeculative(selected) => {
                self.prepare_selected_audio(requests, selected)?;
            }
            _ => {}
        }
        // Media startup is accounted independently. Text-packet observations
        // must compare the same work for native and joint execution.
        let media_time = media_at.elapsed();
        let at = at + media_time;
        let feedback_at = feedback_at + media_time;
        let feedback = self.speculation_feedback(requests, &prefill_work);
        let does_prefill = match &prefill_work {
            Work::Prefill(..) => true,
            Work::MixedSpeculative(_) => true,
            Work::Batch(selected) => selected.iter().any(|&(i, _)| requests[i].prefilling),
            _ => false,
        };
        match prefill_work {
            Work::MixedSpeculative(selected) => {
                let prefills: Vec<_> = selected
                    .iter()
                    .copied()
                    .filter(|(i, _)| requests[*i].prefilling)
                    .collect();
                let began = Instant::now();
                self.speculative_mixed_step_with(requests, &selected, &mut output, &mut overlap)?;
                let rows: usize = selected.iter().map(|(_, n)| n).sum();
                for (i, n) in prefills {
                    requests[i].served = self.scheduler_statistics.iterations;
                    requests[i].credit_s -= began.elapsed().as_secs_f64() * n as f64 / rows as f64;
                    requests[i].last_prefill = Instant::now();
                }
                self.unmerged_prefill_s = 0.;
                self.scheduler_statistics.speculative_iterations += 1;
                *self
                    .scheduler_statistics
                    .batch_histogram
                    .entry(selected.len())
                    .or_default() += 1;
            }
            Work::SpeculativeBatch(choice) => {
                let selected = choice.selected;
                self.speculative_packet_step(requests, &selected, &[], &mut output, &mut overlap)?;
                self.scheduler_statistics.speculative_iterations += 1;
                *self
                    .scheduler_statistics
                    .batch_histogram
                    .entry(selected.len())
                    .or_default() += 1;
                self.unmerged_prefill_s = 0.;
            }
            Work::Speculative(i, rows) => {
                self.unmerged_prefill_s = 0.;
                let began = Instant::now();
                let lookup_rounds = requests[i].mtp.lookup_rounds;
                let result = self.with_request(requests[i], |model, req| {
                    model.speculative_request_step(req, rows)
                });
                let tokens = match result {
                    Ok(tokens) => tokens,
                    Err(_) if requests[i].failure().is_some() => vec![],
                    Err(error) => return Err(error),
                };
                let seconds = began.elapsed().as_secs_f64();
                let per_token = seconds / tokens.len().max(1) as f64;
                // Copies and model drafts have different costs and acceptance.
                // A successful copy must not make a later model draft look
                // artificially profitable at the same verification width.
                let request = &mut requests[i];
                if request.mtp.lookup_rounds != lookup_rounds {
                    request
                        .lookup_costs
                        .entry(rows)
                        .or_default()
                        .observe(seconds);
                }
                self.mtp_seconds_per_token = self.mtp_seconds_per_token * 0.8 + per_token * 0.2;
                self.scheduler_statistics.speculative_iterations += 1;
                self.scheduler_statistics.decode_tokens += tokens.len();
                output.push(StepOutput { request: i, tokens });
                *self
                    .scheduler_statistics
                    .batch_histogram
                    .entry(1)
                    .or_default() += 1;
            }
            Work::Prefill(i, chunk, bounded, budget_s) => {
                let key = Self::iteration_key(requests, &[(i, chunk)]);
                let before = self.execution.batch_statistics();
                let (_, tokens) = self.with_request(requests[i], |model, req| {
                    model.advance_prefill_step(req, chunk, bounded, budget_s)
                })?;
                let after = self.execution.batch_statistics();
                let cold_s = after.initialization_s() - before.initialization_s()
                    + after.sequence_capture_s
                    - before.sequence_capture_s;
                if !self
                    .manifest
                    .prefill_plans
                    .iter()
                    .any(|p| p.accepts(chunk) && !p.stages.is_empty())
                {
                    self.iteration_costs
                        .entry(key)
                        .or_default()
                        .observe((at.elapsed().as_secs_f64() - cold_s).max(0.000001));
                }
                requests[i].served = self.scheduler_statistics.iterations;
                requests[i].credit_s -= at.elapsed().as_secs_f64();
                if has_decoders {
                    self.unmerged_prefill_s += at.elapsed().as_secs_f64();
                }
                requests[i].last_prefill = Instant::now();
                if !tokens.is_empty() {
                    output.push(StepOutput { request: i, tokens });
                }
            }
            Work::Batch(selected) => {
                if selected.iter().any(|&(i, _)| !requests[i].prefilling) {
                    self.unmerged_prefill_s = 0.;
                }
                self.execute_selected_with(requests, &selected, &mut output, &mut overlap)?;
            }
            Work::Idle => {}
        }
        if options.policy == scheduler::Policy::Latency
            && does_prefill
            && requests.iter().any(|r| !r.prefilling && !r.is_finished())
        {
            self.decode_until = Some(
                Instant::now()
                    + std::time::Duration::from_secs_f64(scheduler::decode_quantum_s(
                        at.elapsed().as_secs_f64(),
                        options,
                    )),
            );
        }
        for step in &output {
            if !step.tokens.is_empty() {
                let req = &mut requests[step.request];
                if req.generated > step.tokens.len() {
                    let gap = req.last_output.elapsed().as_secs_f64();
                    self.scheduler_statistics.max_decode_gap_s =
                        self.scheduler_statistics.max_decode_gap_s.max(gap);
                    self.scheduler_statistics.decode_budget_overruns +=
                        usize::from(gap * 1000. > options.target_tpot_ms);
                }
                req.last_output = Instant::now();
            }
        }
        if let Some(feedback) = feedback {
            let stats = self.execution.batch_statistics();
            let cold = stats.initialization_s() - feedback_stats.initialization_s()
                + stats.capture_s
                - feedback_stats.capture_s
                + stats.sequence_capture_s
                - feedback_stats.sequence_capture_s;
            if requests.iter().map(|r| r.mtp.lookup_rounds).sum::<usize>() == feedback.lookups {
                self.dynamic_speculation.observe(
                    feedback.key,
                    feedback.width,
                    (feedback_at.elapsed().as_secs_f64() - cold).max(1e-6),
                    feedback
                        .indices
                        .iter()
                        .map(|&i| requests[i].generated)
                        .sum::<usize>()
                        - feedback.generated,
                );
            }
        }
        overlap();
        self.scheduler_statistics.maintenance_calls += 1;
        self.scheduler_statistics.maintenance_s += maintenance_s;
        self.scheduler_statistics.compute_s += at.elapsed().as_secs_f64();
        Ok(output)
    }
    fn select_work(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &scheduler::Options,
        initialized: &BTreeSet<usize>,
    ) -> Result<Work> {
        let mut prefills: Vec<_> = requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.prefilling)
            .map(|(i, _)| i)
            .collect();
        let mut credits: Vec<_> = prefills.iter().map(|&i| requests[i].credit_s).collect();
        scheduler::replenish(&mut credits, options.prefill_budget_ms / 1000.);
        for (&i, credit) in prefills.iter().zip(credits) {
            requests[i].credit_s = credit;
        }
        // Freeze clock-dependent scores for the entire sort. Reading elapsed()
        // in a comparator can reverse its ordering as time advances and panic
        // with a large cold cohort.
        let ordering_at = Instant::now();
        let age = |i: usize| {
            ordering_at
                .saturating_duration_since(requests[i].last_prefill)
                .as_secs_f64()
        };
        prefills.sort_unstable_by(|&a, &b| {
            let urgent = |i: usize| age(i) >= 30.;
            // Finish short tails before switching to another request's large
            // native tile. Hard aging still wins over this bounded preference.
            let near_end =
                |i: usize| requests[i].input.len() - requests[i].offset <= options.max_batch_tokens;
            let score = |i: usize| {
                self.prefill_costs
                    .remaining(requests[i].offset, requests[i].input.len())
                    / (1. + age(i) / 2.)
            };
            urgent(b)
                .cmp(&urgent(a))
                .then_with(|| near_end(b).cmp(&near_end(a)))
                .then_with(|| (requests[b].credit_s > 0.).cmp(&(requests[a].credit_s > 0.)))
                .then_with(|| score(a).total_cmp(&score(b)))
                .then_with(|| requests[a].slot.cmp(&requests[b].slot))
        });
        let prefill = prefills.first().copied();
        let throughput = options.policy == scheduler::Policy::Throughput;
        let mut decode: Vec<_> = requests
            .iter()
            .enumerate()
            .filter(|(i, r)| !r.prefilling && !r.is_finished() && !initialized.contains(i))
            .map(|(i, _)| i)
            .collect();
        let target: Vec<_> = decode.iter().map(|&i| (i, 1)).collect();
        let progress_due = prefill.is_some_and(|i| {
            requests[i].last_prefill.elapsed().as_secs_f64()
                >= (options.target_tpot_ms / 1000. * 4.).max(2.)
        });
        let decode_round = prefill.is_none()
            || (!throughput
                && !decode.is_empty()
                && !progress_due
                && self
                    .decode_until
                    .is_some_and(|until| Instant::now() < until));
        if decode_round && prefill.is_some() {
            self.scheduler_statistics.decode_burst_iterations += 1;
        }
        // Prompt lookup remains an explicitly enabled proposal source. Its
        // timing/acceptance never trains the model-draft controller.
        if decode_round && decode.len() == 1 && self.prompt_lookup {
            let i = decode[0];
            if let Some(spec) = &self.manifest.mtp {
                let req = &requests[i];
                let slack = if throughput {
                    f64::INFINITY
                } else {
                    req.decode_slack_s(options.target_tpot_ms)
                        .unwrap_or(f64::INFINITY)
                };
                if req.sampling.is_greedy()
                    && req.sampling.top_logprobs.is_none()
                    && let Some(width) = spec
                        .verification_widths()
                        .into_iter()
                        .filter(|&n| {
                            n <= spec.default_verification_tokens
                                && n <= req.limit - req.generated
                                && (options.max_speculative_tokens == 0
                                    || n <= options.max_speculative_tokens)
                                && n <= self
                                    .manifest
                                    .max_context
                                    .saturating_sub(req.history.len().saturating_sub(1))
                                && req.prompt_lookup.available(&req.history, n - 1)
                                && req.lookup_costs.get(&n).map_or(
                                    self.mtp_seconds_per_token * n as f64,
                                    speculation::Profile::upper,
                                ) <= slack
                        })
                        .max()
                {
                    return Ok(Work::Speculative(i, width));
                }
            }
        }
        if decode_round && !decode.is_empty() {
            let slack = if throughput {
                f64::INFINITY
            } else {
                decode
                    .iter()
                    .filter_map(|&i| requests[i].decode_slack_s(options.target_tpot_ms))
                    .fold(f64::INFINITY, f64::min)
            };
            if let Some(choice) = self.select_auto_speculation(requests, &target, options, slack) {
                if choice.selected.len() == 1
                    && self.manifest.mtp.as_ref().is_some_and(|spec| {
                        !spec
                            .batch_verification_tokens
                            .contains(&choice.selected[0].1)
                    })
                {
                    return Ok(Work::Speculative(
                        choice.selected[0].0,
                        choice.selected[0].1,
                    ));
                }
                return Ok(Work::SpeculativeBatch(choice));
            }
        }
        let mixed = self.manifest.batch_layout.as_ref().is_none_or(|layout| {
            layout.segment_tokens > 0
                || !layout.small_mixed_shapes.is_empty()
                || self.manifest.batch_profiles.is_empty()
        });
        let mut cap = options
            .max_batch_tokens
            .min(*self.manifest.batch_profiles.iter().max().unwrap_or(&1));
        if mixed && prefill.is_some() {
            cap = self
                .manifest
                .prefill_batch_profiles
                .iter()
                .map(|p| p.tokens)
                .max()
                .unwrap_or(cap)
                .max(cap);
        }
        if throughput && let Some(i) = prefill {
            // The existing packet template and native prefill execute identical
            // state transitions. Offline scheduling compares total work cost,
            // without buying a decode burst after every prompt fragment.
            let native = self
                .native_prefill_chunk(requests, i, f64::INFINITY, false)
                .ok_or("Missing native prefill progress")?;
            if mixed {
                let mut joint: Vec<_> = decode
                    .iter()
                    .take(options.max_batch_tokens.min(cap))
                    .map(|&i| (i, 1))
                    .collect();
                self.pack_prefill(requests, &prefills, &mut joint, cap, f64::INFINITY)?;
                if joint.len() > 1
                    && joint.iter().any(|(i, _)| requests[*i].prefilling)
                    && self.prefer_joint_prefill(requests, &joint, f64::INFINITY, true)
                {
                    self.scheduler_statistics.joint_prefill_choices += 1;
                    joint.sort_unstable_by_key(|(i, _)| requests[*i].slot);
                    return Ok(Work::Batch(joint));
                }
            }
            self.scheduler_statistics.independent_prefill_choices += 1;
            return Ok(Work::Prefill(i, native, false, f64::INFINITY));
        }
        if decode.is_empty() {
            let Some(i) = prefill else {
                return Ok(Work::Idle);
            };
            let native = self
                .native_prefill_chunk(requests, i, f64::INFINITY, false)
                .ok_or("Missing native prefill progress")?;
            if mixed && prefills.len() > 1 {
                let mut joint = vec![];
                self.pack_prefill(requests, &prefills, &mut joint, cap, f64::INFINITY)?;
                if joint.len() > 1
                    && self.prefer_joint_prefill(requests, &joint, f64::INFINITY, false)
                {
                    self.scheduler_statistics.joint_prefill_choices += 1;
                    joint.sort_unstable_by_key(|(i, _)| requests[*i].slot);
                    return Ok(Work::Batch(joint));
                }
            }
            self.scheduler_statistics.independent_prefill_choices += 1;
            return Ok(Work::Prefill(
                i,
                native,
                false,
                options.prefill_budget_ms / 1000.,
            ));
        }
        let rotate = self.scheduler_cursor % decode.len();
        decode.rotate_left(rotate);
        // Prompt arenas never raise the caller's decoder row limit.
        decode.truncate(options.max_batch_tokens.min(cap));
        self.scheduler_cursor = self.scheduler_cursor.wrapping_add(decode.len());
        let mut selected: Vec<_> = decode.iter().map(|&i| (i, 1)).collect();
        if decode_round {
            selected.sort_unstable_by_key(|(i, _)| requests[*i].slot);
            return Ok(Work::Batch(selected));
        }
        let decode_s = self.predict_iteration(requests, &selected);
        let slack_s = decode
            .iter()
            .filter_map(|&i| requests[i].decode_slack_s(options.target_tpot_ms))
            .fold(f64::INFINITY, f64::min);
        let mut window_s = scheduler::prefill_window(slack_s, decode_s, options.prefill_budget_ms);
        if self.unmerged_prefill_s > 0. {
            window_s = window_s.min((decode_s - self.unmerged_prefill_s).max(0.));
        }
        let native = prefill.and_then(|i| {
            self.native_prefill_chunk(requests, i, window_s, progress_due)
                .map(|n| (i, n))
        });
        if mixed && !prefills.is_empty() {
            let mut joint = selected.clone();
            self.pack_prefill(
                requests,
                &prefills,
                &mut joint,
                cap,
                (decode_s + window_s).min(slack_s).max(0.) * 1000.,
            )?;
            if joint.len() > selected.len()
                && self.prefer_joint_prefill(requests, &joint, window_s, false)
            {
                self.scheduler_statistics.joint_prefill_choices += 1;
                for &(i, _) in &joint {
                    if requests[i].prefilling {
                        requests[i].served = self.scheduler_statistics.iterations;
                    }
                }
                joint.sort_unstable_by_key(|(i, _)| requests[*i].slot);
                if let Some(speculative) =
                    self.select_mixed_speculation(requests, &joint, options, slack_s)
                {
                    return Ok(Work::MixedSpeculative(speculative));
                }
                return Ok(Work::Batch(joint));
            }
        }
        if let Some((i, n)) = native {
            if progress_due && self.predict_iteration(requests, &[(i, n)]) > window_s {
                self.scheduler_statistics.prefill_progress_overrides += 1;
            }
            self.scheduler_statistics.independent_prefill_choices += 1;
            return Ok(Work::Prefill(i, n, true, window_s.max(0.)));
        }
        if prefill.is_some() {
            self.scheduler_statistics.deadline_decode_iterations += 1;
        }
        selected.sort_unstable_by_key(|(i, _)| requests[*i].slot);
        Ok(Work::Batch(selected))
    }

    fn prefer_joint_prefill(
        &self,
        requests: &[&mut RequestState],
        joint: &[(usize, usize)],
        native_budget_s: f64,
        probe: bool,
    ) -> bool {
        // Reuse measured packet timings. An offline job can pay the first two
        // real executions to discover useful shapes instead of permanently
        // rejecting an unmeasured mixed path with a latency-oriented prior.
        if probe && !self.measured_iteration(requests, joint) {
            return true;
        }
        let mut loses_large_tile = false;
        let decoders: Vec<_> = joint
            .iter()
            .copied()
            .filter(|&(i, _)| !requests[i].prefilling)
            .collect();
        let mut native_s = if decoders.is_empty() {
            0.
        } else {
            self.predict_iteration(requests, &decoders)
        };
        let mut has_native_prefill = false;
        for &(i, n) in joint {
            if requests[i].prefilling {
                let Some(native) = self.native_prefill_chunk(requests, i, native_budget_s, false)
                else {
                    // A joint step can still fit when a separate weight pass
                    // would miss the deadline. Evaluate its complete cost.
                    continue;
                };
                loses_large_tile |= native > n;
                has_native_prefill = true;
                native_s +=
                    self.predict_iteration(requests, &[(i, native)]) * n as f64 / native as f64;
            }
        }
        let joint_s = self.predict_iteration(requests, joint);
        // There is no independent deadline-safe candidate in this case.
        if !has_native_prefill {
            return true;
        }
        scheduler::joint_prefill_worthwhile(
            loses_large_tile,
            self.measured_iteration(requests, joint),
            joint_s,
            native_s,
        )
    }

    fn execute_selected(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        self.execute_selected_with(requests, selected, output, &mut || {})
    }
    fn execute_selected_with(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        if selected.len() == 1
            && selected[0].1 == 1
            && (requests[selected[0].0].prefilling || !self.manifest.batch_profiles.contains(&1))
        {
            self.prepare_selected_audio(requests, selected)?;
            let at = Instant::now();
            let key = Self::iteration_key(requests, selected);
            let capture_before = self.execution.batch_statistics().sequence_capture_s;
            let load_before = self.execution.batch_statistics().initialization_s();
            let i = selected[0].0;
            let tokens = self.with_request(requests[i], |model, req| {
                if req.prefilling && model.manifest.prefill_plan(1).is_some() {
                    return model
                        .advance_prefill(req, 1, false)
                        .map(|(_, tokens)| tokens);
                }
                #[cfg(test)]
                profile::mark("single_activated");
                if req.prefilling {
                    model.upload_ids(
                        &model.manifest.token,
                        &req.input[req.offset..req.offset + 1],
                    )?;
                }
                let phase = if req.prefilling {
                    ExecutionPhase::Prefill
                } else {
                    ExecutionPhase::Decode
                };
                let (tokens, history) = if req.prefilling {
                    (
                        &req.input[req.offset..req.offset + 1],
                        &req.input[..req.offset],
                    )
                } else {
                    let last = req.history.len() - 1;
                    (&req.history[last..], &req.history[..last])
                };
                model.prepare_inputs(tokens, history)?;
                #[cfg(test)]
                profile::mark("single_inputs");
                model.upload_segment_controls(1)?;
                #[cfg(test)]
                profile::mark("single_controls");
                model.execution.submit_program("decode", phase)?;
                overlap();
                model.execution.sync()?;
                #[cfg(test)]
                profile::mark("single_execute");
                if let Some(spec) = &model.manifest.mtp {
                    model.mtp_capture(spec, 1, phase)?;
                }
                if req.prefilling {
                    model.scheduler_statistics.prefill_tokens += 1;
                    model.after_prefill(req, 1)
                } else {
                    model.scheduler_statistics.decode_tokens += 1;
                    model.commit_ordinary_token(req)
                }
            })?;
            if !tokens.is_empty() {
                output.push(StepOutput { request: i, tokens });
            }
            *self
                .scheduler_statistics
                .batch_histogram
                .entry(1)
                .or_default() += 1;
            let capture_s = self.execution.batch_statistics().sequence_capture_s - capture_before;
            let load_s = self.execution.batch_statistics().initialization_s() - load_before;
            self.iteration_costs
                .entry(key)
                .or_default()
                .observe((at.elapsed().as_secs_f64() - capture_s - load_s).max(0.000001));
            return Ok(());
        }
        self.execute_batch_selected(requests, selected, output, overlap)
    }

    fn execute_batch_selected(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        #[cfg(test)]
        profile::mark("policy");
        let inputs_at = Instant::now();
        let load_before = self.execution.batch_statistics().initialization_s();
        let segments = self.prepare_batch_inputs(requests, selected)?;
        #[cfg(test)]
        profile::mark("inputs");
        self.scheduler_statistics.batch_inputs_s += inputs_at.elapsed().as_secs_f64();
        let plan_at = Instant::now();
        let decode_only = selected.iter().all(|(i, _)| !requests[*i].prefilling);
        let graph_key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        let plan = if self.execution.has_batch_graph(&graph_key, decode_only) {
            std::rc::Rc::new(Vec::new())
        } else {
            self.model_package
                .cached_batch(&segments, orinfer_model_sdk::abi::BatchOperation::Ordinary)?
                .0
        };
        #[cfg(test)]
        profile::mark("plan");
        self.scheduler_statistics.batch_plan_s += plan_at.elapsed().as_secs_f64();
        let key = Self::iteration_key(requests, selected);
        let capture_before = self.execution.batch_statistics().capture_s;
        self.execution.execute_batch_operation_with(
            orinfer_model_sdk::abi::BatchOperation::Ordinary,
            graph_key,
            &plan,
            decode_only,
            selected
                .iter()
                .all(|&(i, _)| requests[i].limit - requests[i].generated >= 2),
            overlap,
        )?;
        #[cfg(test)]
        profile::mark("execute");
        let commit_at = Instant::now();
        *self
            .scheduler_statistics
            .batch_histogram
            .entry(selected.len())
            .or_default() += 1;
        if !decode_only {
            self.scheduler_statistics.mixed_iterations += 1;
            let count = selected
                .iter()
                .filter(|(i, _)| requests[*i].prefilling)
                .count();
            *self
                .scheduler_statistics
                .prefill_batch_histogram
                .entry(count)
                .or_default() += 1;
        }
        self.commit_batch(requests, selected, output)?;
        let commit_s = commit_at.elapsed().as_secs_f64();
        self.scheduler_statistics.batch_commit_s += commit_s;
        let seconds = inputs_at.elapsed().as_secs_f64();
        let load_s = self.execution.batch_statistics().initialization_s() - load_before;
        let capture_s = self.execution.batch_statistics().capture_s - capture_before;
        self.iteration_costs
            .entry(key)
            .or_default()
            .observe((seconds - capture_s - load_s).max(0.000001));
        let rows: usize = selected.iter().map(|(_, n)| n).sum();
        for &(i, chunk) in selected {
            requests[i].credit_s -= seconds * chunk as f64 / rows as f64;
            requests[i].last_prefill = Instant::now();
        }
        #[cfg(test)]
        profile::mark("commit");
        Ok(())
    }
    pub(super) fn upload_segment_controls(&self, tokens: usize) -> Result<()> {
        if let Some(controls) = &self.manifest.segment_controls {
            let tokens = u32::try_from(tokens).map_err(|_| "Segment length exceeds u32")?;
            self.upload_ids(&controls.length, &[tokens])?;
            self.upload_ids(
                &controls.last_index,
                &[tokens.checked_sub(1).ok_or("Empty segment")?],
            )?;
        }
        Ok(())
    }
    pub(super) fn upload_sequence_segment_controls(
        &mut self,
        slot: usize,
        tokens: usize,
    ) -> Result<()> {
        if let Some(controls) = &self.manifest.segment_controls {
            let tokens = u32::try_from(tokens).map_err(|_| "Segment length exceeds u32")?;
            self.execution
                .upload_sequence_ids(slot, &controls.length, &[tokens])?;
            self.execution.upload_sequence_ids(
                slot,
                &controls.last_index,
                &[tokens.checked_sub(1).ok_or("Empty segment")?],
            )?;
        }
        Ok(())
    }
    fn prepare_batch_inputs(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
    ) -> Result<Vec<BatchSegment>> {
        self.prepare_selected_audio(requests, selected)?;
        self.execution.prepare_batch_workspace()?;
        let segments: Vec<_> = selected
            .iter()
            .map(|&(i, tokens)| BatchSegment {
                slot: requests[i].slot,
                tokens,
            })
            .collect();
        for &(i, chunk) in selected {
            let req = &mut requests[i];
            if req.prefilling {
                self.execution.upload_sequence_ids(
                    req.slot,
                    if chunk == 1 {
                        &self.manifest.token
                    } else {
                        &self.manifest.input
                    },
                    &req.input[req.offset..req.offset + chunk],
                )?;
            }
            if let Some(controls) = &self.manifest.segment_controls {
                let tokens = u32::try_from(chunk).map_err(|_| "Segment length exceeds u32")?;
                self.execution.upload_sequence_bytes(
                    req.slot,
                    &controls.length,
                    &tokens.to_le_bytes(),
                )?;
                self.execution.upload_sequence_bytes(
                    req.slot,
                    &controls.last_index,
                    &tokens.checked_sub(1).ok_or("Empty segment")?.to_le_bytes(),
                )?;
            }
            let program = if let Some(program) =
                self.manifest.batch_layout.as_ref().and_then(|layout| {
                    if layout.segment_tokens > 0 && chunk > 1 {
                        layout
                            .preparation_programs
                            .range(chunk..)
                            .next()
                            .map(|(_, p)| p)
                    } else {
                        layout.preparation_programs.get(&chunk)
                    }
                }) {
                program.clone()
            } else if chunk == 1 {
                "decode".into()
            } else {
                self.manifest
                    .prefill_plan(chunk)
                    .ok_or("Missing prefill program shape")?
                    .prefill_program
                    .clone()
            };
            let position = if req.prefilling {
                req.offset
            } else {
                req.input.len() + req.generated - 1
            };
            self.execution
                .ensure_sequence_program_at(req.slot, &program, position, chunk)?;
            let (tokens, history) = if req.prefilling {
                (
                    &req.input[req.offset..req.offset + chunk],
                    &req.input[..req.offset],
                )
            } else {
                let last = req.history.len() - 1;
                (&req.history[last..], &req.history[..last])
            };
            self.prepare_batch_target_inputs(req.slot, tokens, history)?;
        }
        let table = self.manifest.state_pointer_table.as_ref().filter(|table| {
            segments.len() <= table.max_rows && segments.iter().any(|s| s.tokens == 1)
        });
        let packet = self.manifest.packet_table();
        if let (Some(table), None) = (table, packet) {
            self.execution.upload_sequence_addresses(
                &table.buffer,
                self.model_package.state_bindings(&segments)?,
            )?;
        } else if packet.is_some() {
            let bindings = self.model_package.state_bindings(&segments)?;
            let suffix = packet
                .map_or(Some(0), |p| segments.len().checked_mul(p.columns.len()))
                .ok_or("Packet operand table size overflow")?;
            let prefix = bindings
                .len()
                .checked_sub(suffix)
                .ok_or("Truncated packet operand table")?;
            let mut ranges = vec![];
            if prefix != 0 {
                ranges.push((
                    self.manifest
                        .state_pointer_table
                        .as_ref()
                        .ok_or("Missing ordinary state table")?
                        .buffer
                        .as_str(),
                    0..prefix,
                ));
            }
            if let Some(packet) = packet {
                ranges.push((packet.buffer.as_str(), prefix..bindings.len()));
            }
            self.execution
                .upload_sequence_address_ranges(bindings, &ranges)?;
        }
        Ok(segments)
    }
    fn commit_batch(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        let sampled = self.gpu_sample_batch(requests, selected)?;
        let fast: Vec<_> = selected
            .iter()
            .filter_map(|&(i, _)| {
                let req = &requests[i];
                (!req.prefilling
                    && req.sampling.is_greedy()
                    && !req.decoder.constrained()
                    && req.sampling.top_logprobs.is_none())
                .then_some((i, req.slot))
            })
            .collect();
        for &(i, _) in &fast {
            if requests[i].owner != self.owner_id || requests[i].released {
                return Err("Request does not own a live arena in this model".into());
            }
        }
        let controls = self.execution.read_sequence_controls(
            &fast.iter().map(|&(_, slot)| slot).collect::<Vec<_>>(),
            [
                &self.manifest.position,
                &self.manifest.status,
                &self.manifest.token,
            ],
        )?;
        let mut controls = fast.iter().zip(controls).peekable();
        for &(i, chunk) in selected {
            if requests[i].failure().is_some() {
                continue;
            }
            if let Some(&token) = sampled.get(&i) {
                let req = &mut requests[i];
                self.execution
                    .upload_sequence_ids(req.slot, &self.manifest.token, &[token])?;
                req.history.push(token);
                req.generated += 1;
                req.mtp.committed_tokens = req.generated;
                self.scheduler_statistics.decode_tokens += 1;
                output.push(StepOutput {
                    request: i,
                    tokens: vec![token],
                });
                continue;
            }
            if controls.peek().is_some_and(|((index, _), _)| *index == i) {
                let (_, [position, status, token]) = controls.next().unwrap();
                let req = &mut requests[i];
                if position < 0 || position as usize != req.input.len() + req.generated {
                    return Err("Batched decode position mismatch".into());
                }
                if status != 0 {
                    return Err("Model token status failure".into());
                }
                if token < 0 || token as usize >= self.manifest.vocab {
                    return Err("Selected token outside vocabulary".into());
                }
                let token = token as u32;
                // The GPU already wrote this token into the owner's private
                // input. Greedy selection needs neither arena rebinding nor
                // a redundant host upload before the next packet.
                req.history.push(token);
                req.generated += 1;
                req.mtp.committed_tokens = req.generated;
                self.scheduler_statistics.decode_tokens += 1;
                output.push(StepOutput {
                    request: i,
                    tokens: vec![token],
                });
                continue;
            }
            let tokens = self.with_request(requests[i], |model, req| {
                if req.prefilling {
                    model.scheduler_statistics.prefill_tokens += chunk;
                    *model
                        .scheduler_statistics
                        .prefill_chunk_histogram
                        .entry(chunk)
                        .or_default() += 1;
                    model.after_prefill(req, chunk)
                } else {
                    model.scheduler_statistics.decode_tokens += 1;
                    model.commit_ordinary_token(req)
                }
            })?;
            if !tokens.is_empty() {
                output.push(StepOutput { request: i, tokens });
            }
        }
        Ok(())
    }
    fn commit_ordinary_token(&mut self, req: &mut RequestState) -> Result<Vec<u32>> {
        let position = self.read_control(&self.manifest.position)? as usize;
        if position != req.input.len() + req.generated {
            return Err("Batched decode position mismatch".into());
        }
        let token = match self.select_request_token(req) {
            Ok(token) => token,
            Err(_) if req.failure().is_some() => return Ok(vec![]),
            Err(error) => return Err(error),
        };
        self.upload_ids(&self.manifest.token, &[token])?;
        req.history.push(token);
        req.generated += 1;
        req.mtp.committed_tokens = req.generated;
        Ok(vec![token])
    }
    fn select_request_token(&self, req: &mut RequestState) -> Result<u32> {
        if !req.decoder.constrained() && req.sampling.top_logprobs.is_none() {
            return self.select_target(&req.history, &req.sampling, req.generated);
        }
        if self.read_control(&self.manifest.status)? != 0 {
            return Err("Model token status failure".into());
        }
        if self.manifest.sampling.is_some()
            && req.sampling.temperature > 0.
            && req.sampling.top_logprobs.is_none()
        {
            let mask = req.decoder.gpu_mask()?;
            if mask.as_ref().is_some_and(|words| {
                words.iter().enumerate().all(|(i, &bits)| {
                    let live = self.manifest.vocab.saturating_sub(i * 32).min(32);
                    live == 0 || bits & (u32::MAX >> (32 - live)) == 0
                })
            }) {
                req.decoder.fail("Grammar permits no tokens".into());
                return Err("Grammar permits no tokens".into());
            }
            if let Some(token) =
                self.gpu_sample(&req.history, &req.sampling, req.generated, mask.as_deref())?
            {
                req.decoder.consume(token)?;
                return Ok(token);
            }
        }
        if req.sampling.is_greedy() && req.sampling.top_logprobs.is_none() {
            let selected = self.read_control(&self.manifest.token)? as u32;
            if let Some(token) = req
                .decoder
                .greedy_candidate(selected, self.manifest.vocab)?
            {
                req.decoder.consume(token)?;
                return Ok(token);
            }
        }
        let spec = self
            .manifest
            .buffers
            .iter()
            .find(|b| b.name == self.manifest.logits)
            .ok_or("Missing logits")?;
        let raw = self
            .execution
            .download_bytes(&self.manifest.logits, spec.bytes()?)?;
        let law = req
            .decoder
            .law(&floats(&raw, spec.dtype), &req.history, &req.sampling)?;
        let token = law.distribution.draw(crate::sampling::counter_uniform(
            req.sampling.seed,
            0,
            req.generated as u64,
        ))?;
        req.decoder.commit(token, law)?;
        Ok(token)
    }
    pub(crate) fn finish_request(&mut self, req: &mut RequestState, cache: bool) -> Result<()> {
        if req.released {
            return Ok(());
        }
        let stored = self.with_request(req, |model, req| {
            if cache && req.cache_generated && !req.prefilling && model.prefix_cache.budget != 0 {
                let consistent = model.refresh_request_mtp(req, false)?;
                if consistent {
                    model.store_decoded_prefix(
                        &req.history,
                        &req.media,
                        &|| false,
                        &mut req.prefix,
                    )?;
                }
            }
            Ok(())
        });
        self.release_prefill_frame(req)?;
        self.execution
            .release_sequence(req.slot, &self.manifest.reset_buffers)?;
        req.prefix.kv.clear();
        self.reserved_requests.remove(&req.slot);
        req.released = true;
        self.execution.collect_snapshots()?;
        stored
    }
}
