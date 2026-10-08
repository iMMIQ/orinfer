//! Persistent requests and iteration-level execution. API transport owns output
//! parsing; this module owns GPU state, sampling counters and prefix endpoints.
use super::*;
use crate::{architecture::BatchSegment, prefix::Media, scheduler};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
mod profile;
mod speculation;
#[cfg(test)]
mod tests;

#[derive(serde::Deserialize)]
pub struct GenerationInput {
    pub input_tokens: Vec<u32>,
    #[serde(default)]
    pub images: Vec<crate::vision::ImageInput>,
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
    prefix: super::prefix::Context,
    mtp: crate::mtp::Statistics,
    generated: usize,
    served: usize,
    mtp_blocked: bool,
    decoder: crate::sampling::Decoder,
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
    pub fn is_prefilling(&self) -> bool {
        self.prefilling
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
    Speculative(usize),
    Prefill(usize, Option<f64>),
    Batch(Vec<(usize, usize)>),
    Idle,
}

pub(super) struct Reservation {
    kv_bytes: usize,
    context: usize,
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
        Ok(limits)
    }
    pub(crate) fn waiting_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<scheduler::Waiting> {
        // The checkpoint already records the actual restored payload bytes.
        // Rebuilding every tensor range for each queued request is unnecessary.
        let (cached, bytes) = self.prefix_match_cost(input, images)?;
        Ok(scheduler::Waiting {
            age_s: 0.,
            remaining_s: self.prefill_costs.remaining(cached, input.len()),
            restore_s: self.prefill_costs.restore_cost(bytes),
            remaining_tokens: input.len().saturating_sub(cached),
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
        let workspace = self.pending_workspace(context)?;
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
        self.trim_request_cache(options, fixed + prospective, context)?;
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
                return Err(format!(
                    "Request context/output budget exceeds available GPU memory (free {} MiB, required {} MiB: state {}, KV {}, future {}, workspace {})",
                    self.admission_free_bytes()? >> 20,
                    needed >> 20,
                    fixed >> 20,
                    kv >> 20,
                    future >> 20,
                    workspace >> 20
                ));
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
    fn pending_workspace(&self, context: usize) -> Result<usize> {
        self.execution.pending_prefill_workspace_bytes(
            context.max(
                self.reserved_requests
                    .values()
                    .map(|r| r.context)
                    .max()
                    .unwrap_or(0),
            ),
        )
    }
    fn trim_request_cache(
        &mut self,
        options: &scheduler::Options,
        additional: usize,
        context: usize,
    ) -> Result<()> {
        let future = self
            .reserved_requests
            .values()
            .map(|r| r.kv_bytes)
            .sum::<usize>()
            .saturating_sub(self.execution.active_kv_bytes())
            .checked_add(self.pending_workspace(context)?)
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
        input: GenerationInput,
        cancelled: &impl Fn() -> bool,
    ) -> Result<RequestState> {
        self.reap_abandoned()?;
        let started = Instant::now();

        let context = self.validate_generation(&input)?;
        let limits = self.request_limits(&input)?;
        let slot = self.execution.lease_sequence(&limits)?;
        let result = (|| {
            self.execution
                .reset_sequence_reusing_pages(&self.manifest.reset_buffers)?;
            self.prepare_visual_capacity(&input.input_tokens, &input.images, context, cancelled)?;
            let media = self.prefix_media(&input.input_tokens, &input.images)?;
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
                prefix,
                mtp: Default::default(),
                generated: 0,
                served: 0,
                mtp_blocked: false,
                decoder: Default::default(),
            })
        })();
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
        self.store_prefix(
            &req.input,
            &req.media,
            req.warm.tokens,
            true,
            &mut req.prefix,
        )?;
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
        budget_ms: Option<f64>,
    ) -> Result<(usize, Vec<u32>)> {
        let remaining = Self::prefill_boundary(req) - req.offset;
        let maximum = budget_ms.map_or(remaining, |ms| {
            self.prefill_costs.bounded_chunk(remaining, ms)
        });
        let plan = self
            .manifest
            .prefill_plans
            .iter()
            .filter(|p| p.chunk_tokens <= maximum)
            .max_by_key(|p| p.chunk_tokens);
        let (chunk, program, head) = plan
            .map(|p| {
                (
                    p.chunk_tokens,
                    p.prefill_program.clone(),
                    Some(p.head_program.clone()),
                )
            })
            .unwrap_or((1, "decode".into(), None));
        let at = Instant::now();
        self.upload_ids(
            if chunk == 1 {
                &self.manifest.token
            } else {
                &self.manifest.input
            },
            &req.input[req.offset..req.offset + chunk],
        )?;
        self.prepare_inputs(
            &req.input[req.offset..req.offset + chunk],
            &req.input[..req.offset],
        )?;
        self.upload_segment_controls(chunk)?;
        self.launch_program(&program, ExecutionPhase::Prefill)?;
        if let Some(spec) = &self.manifest.mtp {
            self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
        }
        if (req.offset + chunk == req.input.len()
            || req.checkpoints.contains(&(req.offset + chunk)))
            && let Some(head) = head
        {
            self.launch_program(&head, ExecutionPhase::Prefill)?;
        }
        let tokens = self.after_prefill(req, chunk)?;
        let seconds = at.elapsed().as_secs_f64();
        self.prefill_costs.observe(chunk, seconds);
        self.scheduler_statistics.prefill_tokens += chunk;
        *self
            .scheduler_statistics
            .prefill_chunk_histogram
            .entry(chunk)
            .or_default() += 1;
        if budget_ms.is_some() {
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
            .map(|(i, _)| requests[*i].history.len())
            .max()
            .unwrap_or(1);
        (
            selected.len().next_power_of_two(),
            rows.next_power_of_two(),
            context.next_power_of_two(),
            selected
                .iter()
                .filter(|(i, _)| requests[*i].prefilling)
                .count(),
        )
    }
    fn predict_iteration(
        &self,
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> f64 {
        let key = Self::iteration_key(requests, selected);
        if let Some(&seconds) = self.iteration_costs.get(&key) {
            return seconds;
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
        let mut has_prefill = false;
        for &i in candidates {
            if used == cap {
                break;
            }
            let remaining = Self::prefill_boundary(requests[i]) - requests[i].offset;
            let chunk = shapes
                .iter()
                .copied()
                .filter(|&n| n <= cap - used && n <= remaining && n <= per_request_cap)
                .filter(|&n| {
                    let mut trial = selected.clone();
                    trial.push((i, n));
                    (n == 1 && !has_prefill)
                        || self.predict_iteration(requests, &trial) * 1000. <= budget_ms
                })
                .max();
            if let Some(chunk) = chunk {
                selected.push((i, chunk));
                used += chunk;
                has_prefill = true;
            }
        }
        Ok(())
    }
    pub(crate) fn advance_requests(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
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
        self.trim_request_cache(options, 0, 0)?;
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
        match self.select_work(requests, options, &initialized)? {
            Work::Speculative(i) => {
                let began = Instant::now();
                let result = self.with_request(requests[i], |model, req| {
                    model.speculative_request_step(req, options.max_batch_tokens)
                });
                let tokens = match result {
                    Ok(tokens) => tokens,
                    Err(_) if requests[i].failure().is_some() => vec![],
                    Err(error) => return Err(error),
                };
                let per_token = began.elapsed().as_secs_f64() / tokens.len().max(1) as f64;
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
            Work::Prefill(i, budget_ms) => {
                let (_, tokens) = self.with_request(requests[i], |model, req| {
                    model.advance_prefill(req, budget_ms)
                })?;
                requests[i].served = self.scheduler_statistics.iterations;
                if !tokens.is_empty() {
                    output.push(StepOutput { request: i, tokens });
                }
            }
            Work::Batch(selected) => self.execute_selected(requests, &selected, &mut output)?,
            Work::Idle => {}
        }
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
        prefills.sort_unstable_by_key(|&i| (requests[i].served, requests[i].slot));
        let prefill = prefills.first().copied();
        let mut decode: Vec<_> = requests
            .iter()
            .enumerate()
            .filter(|(i, r)| !r.prefilling && !r.is_finished() && !initialized.contains(i))
            .map(|(i, _)| i)
            .collect();
        let target: Vec<_> = decode.iter().map(|&i| (i, 1)).collect();
        let prefer_mtp = prefill.is_none()
            && options.max_batch_tokens >= 2
            && self.manifest.mtp.is_some()
            && !decode.is_empty()
            && (decode.len() == 1
                || (decode.len() <= 4
                    && self.mtp_seconds_per_token
                        < self.predict_iteration(requests, &target) / decode.len() as f64));
        if prefer_mtp {
            let i = decode[self.scheduler_cursor % decode.len()];
            self.scheduler_cursor = self.scheduler_cursor.wrapping_add(1);
            return Ok(Work::Speculative(i));
        }
        let mut cap = options
            .max_batch_tokens
            .min(*self.manifest.batch_profiles.iter().max().unwrap_or(&1));
        // Preserve the dense large-chunk path for long cold prompts. Short
        // tails can share projection weights without reading across histories.
        let long_joint = decode.is_empty()
            && prefills.len() > 1
            && !self.manifest.prefill_batch_profiles.is_empty()
            && prefills
                .iter()
                .any(|&i| Self::prefill_boundary(requests[i]) - requests[i].offset > cap);
        if long_joint {
            cap = self
                .manifest
                .prefill_batch_profiles
                .iter()
                .map(|p| p.tokens)
                .max()
                .unwrap();
        }
        let mixed = self.manifest.batch_layout.as_ref().is_none_or(|layout| {
            !layout.small_mixed_shapes.is_empty() || self.manifest.batch_profiles.is_empty()
        });
        let joint_prefill = long_joint
            || (mixed
                && decode.is_empty()
                && prefills.len() > 1
                && prefills
                    .iter()
                    .all(|&i| Self::prefill_boundary(requests[i]) - requests[i].offset <= cap));
        if decode.is_empty() && !joint_prefill {
            return Ok(prefill
                .map(|i| Work::Prefill(i, None))
                .unwrap_or(Work::Idle));
        }
        if !decode.is_empty() {
            let rotate = self.scheduler_cursor % decode.len();
            decode.rotate_left(rotate);
        }
        // Packages without joint mixed profiles run a bounded real prompt
        // chunk between decode iterations; cold-only prompts keep large tiles.
        if let Some(i) = prefill
            && !mixed
            && self.scheduler_statistics.iterations.is_multiple_of(2)
        {
            return Ok(Work::Prefill(i, Some(options.prefill_budget_ms)));
        }
        // Even a one-row configuration must advance waiting prompt work.
        let reserve = usize::from(
            prefill.is_some()
                && mixed
                && (cap > 1 || self.scheduler_statistics.iterations.is_multiple_of(2)),
        );
        decode.truncate(cap - reserve);
        self.scheduler_cursor = self.scheduler_cursor.wrapping_add(decode.len());
        let mut selected: Vec<_> = decode.iter().map(|&i| (i, 1)).collect();
        self.pack_prefill(
            requests,
            if mixed { &prefills } else { &[] },
            &mut selected,
            cap,
            if joint_prefill {
                f64::INFINITY
            } else {
                options.prefill_budget_ms
            },
        )?;
        for &(i, _) in &selected {
            if requests[i].prefilling {
                requests[i].served = self.scheduler_statistics.iterations;
            }
        }
        if selected.is_empty() {
            return Ok(Work::Idle);
        }
        // Membership selection above provides fairness. Canonical execution
        // order reuses the same graph when the API's active vector is reordered.
        selected.sort_unstable_by_key(|(i, _)| requests[*i].slot);
        Ok(Work::Batch(selected))
    }
    fn execute_selected(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        if selected.len() == 1 && selected[0].1 == 1 {
            let i = selected[0].0;
            let tokens = self.with_request(requests[i], |model, req| {
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
                model.upload_segment_controls(1)?;
                model.launch_program("decode", phase)?;
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
            return Ok(());
        }
        #[cfg(test)]
        profile::mark("policy");
        let inputs_at = Instant::now();
        let segments = self.prepare_batch_inputs(requests, selected)?;
        #[cfg(test)]
        profile::mark("inputs");
        self.scheduler_statistics.batch_inputs_s += inputs_at.elapsed().as_secs_f64();
        let plan_at = Instant::now();
        let decode_only = selected.iter().all(|(i, _)| !requests[*i].prefilling);
        let graph_key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        let plan = if self.execution.has_batch_graph(&graph_key, decode_only) {
            Vec::new()
        } else {
            self.model_package.batch_plan(&segments)?
        };
        #[cfg(test)]
        profile::mark("plan");
        self.scheduler_statistics.batch_plan_s += plan_at.elapsed().as_secs_f64();
        let key = Self::iteration_key(requests, selected);
        let compute_at = Instant::now();
        self.execution
            .execute_batch(graph_key, &plan, decode_only)?;
        #[cfg(test)]
        profile::mark("execute");
        let commit_at = Instant::now();
        let seconds = compute_at.elapsed().as_secs_f64();
        self.iteration_costs
            .entry(key)
            .and_modify(|old| *old = *old * 0.8 + seconds * 0.2)
            .or_insert(seconds);
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
        self.scheduler_statistics.batch_commit_s += commit_at.elapsed().as_secs_f64();
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
    fn prepare_batch_inputs(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
    ) -> Result<Vec<BatchSegment>> {
        let segments: Vec<_> = selected
            .iter()
            .map(|&(i, tokens)| BatchSegment {
                slot: requests[i].slot,
                tokens,
            })
            .collect();
        for &(i, chunk) in selected {
            let req = &mut requests[i];
            self.execution.activate_sequence(req.slot)?;
            if req.prefilling {
                self.upload_ids(
                    if chunk == 1 {
                        &self.manifest.token
                    } else {
                        &self.manifest.input
                    },
                    &req.input[req.offset..req.offset + chunk],
                )?;
            }
            self.upload_segment_controls(chunk)?;
            let program = if chunk == 1 {
                "decode".into()
            } else {
                self.manifest
                    .prefill_plans
                    .iter()
                    .find(|p| p.chunk_tokens == chunk)
                    .ok_or("Missing prefill program shape")?
                    .prefill_program
                    .clone()
            };
            self.execution.ensure_sequence_program(req.slot, &program)?;
            let (tokens, history) = if req.prefilling {
                (
                    &req.input[req.offset..req.offset + chunk],
                    &req.input[..req.offset],
                )
            } else {
                let last = req.history.len() - 1;
                (&req.history[last..], &req.history[..last])
            };
            self.prepare_inputs(tokens, history)?;
        }
        if let Some(table) = self.manifest.state_pointer_table.as_ref().filter(|table| {
            segments.iter().map(|s| s.tokens).sum::<usize>() <= table.max_rows
                && segments.iter().any(|s| s.tokens == 1)
        }) {
            self.execution.upload_sequence_addresses(
                &table.buffer,
                &self.model_package.state_bindings(&segments)?,
            )?;
        }
        Ok(segments)
    }
    fn commit_batch(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        for &(i, chunk) in selected {
            let tokens = self.with_request(requests[i], |model, req| {
                if req.prefilling {
                    model.scheduler_statistics.prefill_tokens += chunk;
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
            if cache && !req.prefilling && model.prefix_cache.budget != 0 {
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
        self.execution
            .release_sequence(req.slot, &self.manifest.reset_buffers)?;
        req.prefix.kv.clear();
        self.reserved_requests.remove(&req.slot);
        req.released = true;
        self.execution.collect_snapshots()?;
        stored
    }
}
