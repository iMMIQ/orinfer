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
    prefix_kv: Vec<crate::cuda::snapshot::Piece>,
    prefix: crate::prefix::Statistics,
    mtp: crate::mtp::Statistics,
    generated: usize,
    served: usize,
    mtp_blocked: bool,
}
impl RequestState {
    pub fn generated_tokens(&self) -> usize {
        self.generated
    }
    pub fn computed_prompt_tokens(&self) -> usize {
        self.offset.saturating_sub(self.prefix.cached_tokens)
    }
    pub fn is_prefilling(&self) -> bool {
        self.prefilling
    }
    pub fn is_finished(&self) -> bool {
        self.generated == self.limit || self.released
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix
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

impl ModelRuntime {
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
        })
    }
    pub(crate) fn can_admit_request(
        &mut self,
        input: &GenerationInput,
        options: &scheduler::Options,
    ) -> Result<bool> {
        options.validate()?;
        if self.reserved_requests.len() >= options.max_active {
            return Ok(false);
        }
        let context = self.validate_generation(input)?;
        let limits = self.request_limits(input)?;
        let fixed = self.execution.additional_state_bytes(&limits)?;
        let prospective = self.execution.reserved_kv_bytes(context)?;
        let workspace = self.pending_workspace(context)?;
        let future = self
            .reserved_requests
            .values()
            .sum::<usize>()
            .saturating_sub(self.execution.resident_kv_bytes());
        let needed = fixed
            .checked_add(prospective)
            .and_then(|n| n.checked_add(future))
            .and_then(|n| n.checked_add(workspace))
            .and_then(|n| n.checked_add(options.memory_reserve_bytes))
            .ok_or("Admission budget overflow")?;
        if self
            .execution
            .free_bytes()?
            .saturating_add(self.prefix_cache.bytes)
            < needed
        {
            // Releasing every cache entry still cannot satisfy the outstanding
            // reservations. Keep useful prefixes until active work completes.
            self.scheduler_statistics.admission_deferrals += 1;
            if self.reserved_requests.is_empty() {
                return Err("Request context/output budget exceeds available GPU memory".into());
            }
            return Ok(false);
        }
        while self.execution.free_bytes()? < needed {
            let Some(snapshot) = self.prefix_cache.evict_one() else {
                self.scheduler_statistics.admission_deferrals += 1;
                if self.reserved_requests.is_empty() {
                    return Err("Request context/output budget exceeds available GPU memory".into());
                }
                return Ok(false);
            };
            self.execution.release_snapshot(snapshot)?;
        }
        self.trim_request_cache(options, fixed + prospective, context)?;
        Ok(true)
    }
    fn pending_workspace(&self, context: usize) -> Result<usize> {
        self.execution.pending_prefill_workspace_bytes(
            context.max(self.reserved_contexts.values().copied().max().unwrap_or(0)),
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
            .sum::<usize>()
            .saturating_sub(self.execution.resident_kv_bytes())
            .checked_add(self.pending_workspace(context)?)
            .ok_or("Future workspace budget overflow")?;
        let free = self.execution.free_bytes()?;
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
        let started = Instant::now();
        if self.manifest.batch_profiles.is_empty() {
            return Err(
                "Model operator package lacks continuous batching; upgrade_batching.py is required"
                    .into(),
            );
        }
        let context = self.validate_generation(&input)?;
        let limits = self.request_limits(&input)?;
        let slot = self.execution.lease_sequence(&limits)?;
        let result = (|| {
            self.execution
                .reset_sequence(&self.manifest.reset_buffers)?;
            self.prepare_visual_capacity(&input.input_tokens, &input.images, context, cancelled)?;
            let media = self.prefix_media(&input.input_tokens, &input.images)?;
            let (offset, mut warm) = self.restore_prefix(&input.input_tokens, &media)?;
            if offset > 0
                && offset < input.input_tokens.len()
                && let Some(spec) = self.manifest.mtp.clone()
            {
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
                    .chain(std::iter::once(self.prefix_statistics.matched_tokens))
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
            self.reserved_requests
                .insert(slot, self.execution.reserved_kv_bytes(context)?);
            self.reserved_contexts.insert(slot, context);
            self.scheduler_statistics.peak_active = self
                .scheduler_statistics
                .peak_active
                .max(self.reserved_requests.len());
            Ok(RequestState {
                owner: self.owner_id,
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
                prefix_kv: std::mem::take(&mut self.prefix_kv),
                prefix: std::mem::take(&mut self.prefix_statistics),
                mtp: Default::default(),
                generated: 0,
                served: 0,
                mtp_blocked: false,
            })
        })();
        if result.is_err() {
            self.reserved_requests.remove(&slot);
            self.reserved_contexts.remove(&slot);
            self.execution
                .release_sequence(slot, &self.manifest.reset_buffers)?;
            self.prefix_kv.clear();
        }
        self.scheduler_statistics.request_start_s += started.elapsed().as_secs_f64();
        if let Ok(request) = &result {
            self.scheduler_statistics.admissions += 1;
            self.scheduler_statistics.prefix_restore_s += request.prefix.restore_s;
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
        std::mem::swap(&mut self.prefix_kv, &mut req.prefix_kv);
        std::mem::swap(&mut self.prefix_statistics, &mut req.prefix);
        let result = run(self, req);
        std::mem::swap(&mut self.prefix_kv, &mut req.prefix_kv);
        std::mem::swap(&mut self.prefix_statistics, &mut req.prefix);
        result
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
            let end = if req.offset < req.input.len() {
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
            self.store_prefix(&req.input[..req.offset], &req.media, req.warm.tokens, true)?;
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
        self.store_prefix(&req.input, &req.media, req.warm.tokens, true)?;
        let pending = self.select_target(&req.history, &req.sampling, req.generated)?;
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
    fn advance_prefill(&mut self, req: &mut RequestState) -> Result<(usize, Vec<u32>)> {
        let remaining = Self::prefill_boundary(req) - req.offset;
        let plan = self
            .manifest
            .prefill_plans
            .iter()
            .filter(|p| p.chunk_tokens <= remaining)
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
        self.upload_ids(
            if chunk == 1 {
                &self.manifest.token
            } else {
                &self.manifest.input
            },
            &req.input[req.offset..req.offset + chunk],
        )?;
        self.upload_ids("BatchSegmentLength", &[chunk as u32])?;
        self.upload_ids("BatchLastIndex", &[(chunk - 1) as u32])?;
        let at = Instant::now();
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
        let seconds = at.elapsed().as_secs_f64();
        self.prefill_costs.observe(chunk, seconds);
        self.scheduler_statistics.prefill_tokens += chunk;
        Ok((chunk, self.after_prefill(req, chunk)?))
    }
    fn iteration_key(
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> (usize, usize, usize) {
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
    pub(crate) fn advance_requests(
        &mut self,
        requests: &mut [&mut RequestState],
        options: &scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
        #[cfg(test)]
        profile::mark("entry");
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
        let prefill = requests
            .iter()
            .enumerate()
            .filter(|(_, r)| r.prefilling)
            .min_by_key(|(_, r)| r.served)
            .map(|(i, _)| i);
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
            let began = Instant::now();
            let tokens = self.with_request(requests[i], |model, req| {
                model.speculative_request_step(req, options.max_batch_tokens)
            })?;
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
            self.scheduler_statistics.compute_s += at.elapsed().as_secs_f64();
            return Ok(output);
        }
        if decode.is_empty() {
            if let Some(i) = prefill {
                let (_, tokens) =
                    self.with_request(requests[i], |model, req| model.advance_prefill(req))?;
                requests[i].served = self.scheduler_statistics.iterations;
                if !tokens.is_empty() {
                    output.push(StepOutput { request: i, tokens });
                }
            }
            self.scheduler_statistics.compute_s += at.elapsed().as_secs_f64();
            return Ok(output);
        }
        let rotate = self.scheduler_cursor % decode.len();
        decode.rotate_left(rotate);
        let cap = options.max_batch_tokens.min(
            *self
                .manifest
                .batch_profiles
                .iter()
                .max()
                .ok_or("Missing batch profiles")?,
        );
        // Even a one-row configuration must advance waiting prompt work.
        let reserve = usize::from(
            prefill.is_some()
                && (cap > 1 || self.scheduler_statistics.iterations.is_multiple_of(2)),
        );
        decode.truncate(cap - reserve);
        self.scheduler_cursor = self.scheduler_cursor.wrapping_add(decode.len());
        let mut selected: Vec<_> = decode.iter().map(|&i| (i, 1)).collect();
        if let Some(i) = prefill {
            let remaining = Self::prefill_boundary(requests[i]) - requests[i].offset;
            let available = cap - selected.len();
            let mut shapes = vec![1];
            shapes.extend(
                self.manifest
                    .batch_layout
                    .as_ref()
                    .ok_or("Missing batch layout")?
                    .profiles
                    .iter()
                    .filter(|(_, kind)| {
                        matches!(
                            kind,
                            crate::architecture::PrefillKind::Sequence
                                | crate::architecture::PrefillKind::Recurrent
                        )
                    })
                    .map(|(&n, _)| n),
            );
            shapes.sort_unstable();
            let chunk = shapes
                .into_iter()
                .filter(|&n| n <= available && n <= remaining)
                .filter(|&n| {
                    let mut trial = selected.clone();
                    trial.push((i, n));
                    n == 1
                        || self.predict_iteration(requests, &trial) * 1000.
                            <= options.prefill_budget_ms
                })
                .max();
            if let Some(chunk) = chunk {
                selected.push((i, chunk));
                requests[i].served = self.scheduler_statistics.iterations;
            }
        }
        if selected.is_empty() {
            return Ok(output);
        }
        // Membership selection above provides fairness. Canonical execution
        // order reuses the same graph when the API's active vector is reordered.
        selected.sort_unstable_by_key(|(i, _)| requests[*i].slot);
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
            self.scheduler_statistics.compute_s += at.elapsed().as_secs_f64();
            return Ok(output);
        }
        let segments: Vec<_> = selected
            .iter()
            .map(|&(i, tokens)| BatchSegment {
                slot: requests[i].slot,
                tokens,
            })
            .collect();
        #[cfg(test)]
        profile::mark("policy");
        let inputs_at = Instant::now();
        for &(i, chunk) in &selected {
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
            self.upload_ids("BatchSegmentLength", &[chunk as u32])?;
            self.upload_ids("BatchLastIndex", &[(chunk - 1) as u32])?;
            let program = if chunk == 1 {
                "decode".into()
            } else {
                format!("prefill_m{chunk}")
            };
            self.execution.ensure_sequence_program(req.slot, &program)?;
        }
        #[cfg(test)]
        profile::mark("inputs");
        self.scheduler_statistics.batch_inputs_s += inputs_at.elapsed().as_secs_f64();
        let plan_at = Instant::now();
        let decode_only = selected.iter().all(|(i, _)| !requests[*i].prefilling);
        let graph_key: Vec<_> = segments.iter().map(|s| (s.slot, s.tokens)).collect();
        let plan = if self.execution.has_batch_graph(&graph_key, decode_only) {
            Vec::new()
        } else {
            crate::architecture::batch_plan(&self.manifest, &segments)?
        };
        #[cfg(test)]
        profile::mark("plan");
        self.scheduler_statistics.batch_plan_s += plan_at.elapsed().as_secs_f64();
        let key = Self::iteration_key(requests, &selected);
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
        }
        for &(i, chunk) in &selected {
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
        self.scheduler_statistics.compute_s += at.elapsed().as_secs_f64();
        self.scheduler_statistics.batch_commit_s += commit_at.elapsed().as_secs_f64();
        #[cfg(test)]
        profile::mark("commit");
        Ok(output)
    }
    fn commit_ordinary_token(&mut self, req: &mut RequestState) -> Result<Vec<u32>> {
        let position = self.read_control(&self.manifest.position)? as usize;
        if position != req.input.len() + req.generated {
            return Err("Batched decode position mismatch".into());
        }
        let token = self.select_target(&req.history, &req.sampling, req.generated)?;
        self.upload_ids(&self.manifest.token, &[token])?;
        req.history.push(token);
        req.generated += 1;
        req.mtp.committed_tokens = req.generated;
        Ok(vec![token])
    }
    pub(crate) fn finish_request(&mut self, req: &mut RequestState, cache: bool) -> Result<()> {
        if req.released {
            return Ok(());
        }
        let stored = self.with_request(req, |model, req| {
            if cache && !req.prefilling && model.prefix_cache.budget != 0 {
                let consistent = model.refresh_request_mtp(req, false)?;
                if consistent {
                    model.store_decoded_prefix(&req.history, &req.media, &|| false)?;
                }
            }
            Ok(())
        });
        req.prefix_kv.clear();
        self.reserved_requests.remove(&req.slot);
        self.reserved_contexts.remove(&req.slot);
        let released = self
            .execution
            .release_sequence(req.slot, &self.manifest.reset_buffers);
        req.released = true;
        self.execution.collect_snapshots()?;
        stored.and(released)
    }
}
