//! Native A2 generation provider behind the common continuous worker API.
use super::{
    batch::Layout,
    team::{Command, Team},
};
use crate::{
    artifact::{FileIdentity, Result, read_identity, resolve_file, sha256},
    model::{GenerationInput, StepOutput},
    prefix::{Cache, Entry, Media, Resident},
    sampling::{Constraint, Decoder, TokenLogprob},
    scheduler,
};
use orinfer_model_sdk::architecture::BatchSegment;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    schema_version: u32,
    architecture: String,
    compute_policy: String,
    execution_package: String,
    frontend_assets: BTreeMap<String, String>,
    metadata: Metadata,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Metadata {
    target: String,
    max_context: usize,
    vocab: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Package {
    schema_version: u32,
    backend: String,
    plan: FileIdentity,
}
struct Snapshot {
    id: u64,
    bytes: usize,
}
impl Resident for Snapshot {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        out.insert(self.id, self.bytes);
    }
}
pub(crate) struct Runtime {
    team: Team,
    pub frontend_assets: BTreeMap<String, String>,
    cache: Cache<Snapshot>,
    prefix: crate::prefix::Statistics,
    statistics: scheduler::Statistics,
    active: BTreeSet<usize>,
    abandoned: Arc<Mutex<Vec<usize>>>,
    owner: u64,
    next_snapshot: u64,
    hints: Vec<usize>,
    prefill_tokens_per_s: f64,
}
pub(crate) struct Request {
    owner: u64,
    slot: usize,
    abandoned: Arc<Mutex<Vec<usize>>>,
    input: Vec<u32>,
    history: Vec<u32>,
    offset: usize,
    generated: usize,
    limit: usize,
    prefilling: bool,
    released: bool,
    sampling: crate::sampling::Options,
    decoder: Decoder,
    prefix: crate::prefix::Statistics,
    mtp: crate::mtp::Statistics,
    last_output: Instant,
    cache_prompt: bool,
    cache_generated: bool,
}
impl Drop for Request {
    fn drop(&mut self) {
        if !self.released
            && let Ok(mut queue) = self.abandoned.lock()
        {
            queue.push(self.slot);
        }
    }
}
impl Request {
    pub fn retain_shared_prefixes_only(&mut self, shared: bool) {
        self.cache_prompt = shared;
        self.cache_generated = false;
    }
    pub fn set_constraint(&mut self, constraint: Box<dyn Constraint>) -> Result<()> {
        if self.generated != 0 {
            return Err("Attach constraint before generation".into());
        }
        self.decoder.set_constraint(constraint);
        Ok(())
    }
    pub fn take_logprobs(&mut self) -> Vec<TokenLogprob> {
        self.decoder.take_records()
    }
    pub fn generated_tokens(&self) -> usize {
        self.generated
    }
    pub fn computed_prompt_tokens(&self) -> usize {
        self.offset.saturating_sub(self.prefix.cached_tokens)
    }
    pub fn decode_slack_s(&self, target_ms: f64) -> Option<f64> {
        (!self.prefilling && !self.is_finished())
            .then(|| target_ms / 1000. - self.last_output.elapsed().as_secs_f64())
    }
    pub fn is_prefilling(&self) -> bool {
        self.prefilling
    }
    pub fn remaining_prefill_tokens(&self) -> usize {
        if self.prefilling {
            self.input.len() - self.offset
        } else {
            0
        }
    }
    pub fn is_finished(&self) -> bool {
        self.released || self.generated == self.limit || self.decoder.finished()
    }
    pub fn constraint_finished(&self) -> bool {
        self.decoder.finished()
    }
    pub fn failure(&self) -> Option<&str> {
        self.decoder.failure()
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix
    }
    pub fn speculation_statistics(&self) -> &crate::mtp::Statistics {
        &self.mtp
    }
}
impl Runtime {
    pub fn recognizes(raw: &serde_json::Value) -> bool {
        raw["metadata"]["target"] == "ascend910b"
    }
    pub fn load(path: &Path, options: crate::execution::LoadOptions) -> Result<Self> {
        if options.verify_weights {
            return Err("Native A2 serving currently supports structural weight validation; --verify-weights is not implemented for this experimental package".into());
        }
        if options.resident_experts != crate::execution::ResidentExperts::Full {
            return Err("Native A2 serving requires full resident experts".into());
        }
        if options.mtp_drafts.is_some_and(|n| n > 0) || options.prompt_lookup {
            return Err("A2 native service currently supports target generation with MTP and prompt lookup disabled".into());
        }
        let descriptor = crate::model::resolve_manifest(path)?;
        let cache_root = descriptor.parent().ok_or("Missing A2 cache root")?;
        let root = cache_root.parent().ok_or("Missing A2 model root")?;
        let model: Descriptor = crate::model::read(&descriptor)?;
        if model.schema_version != 1
            || model.architecture != "deepseek_v41"
            || model.compute_policy != "w4a8"
            || model.metadata.target != "ascend910b"
            || model.metadata.vocab != 129280
        {
            return Err("Unsupported native A2 model descriptor".into());
        }
        crate::loader::validate_frontend_assets(root, &model.frontend_assets)?;
        let package_root = crate::operators::resolve(cache_root, &model.execution_package)?;
        let raw = std::fs::read(resolve_file(&package_root, "package.json")?)
            .map_err(|e| e.to_string())?;
        if sha256(&raw) != model.execution_package {
            return Err("A2 execution package digest differs".into());
        }
        let package: Package = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if package.schema_version != 1 || package.backend != "ascend" {
            return Err("Invalid native A2 execution package".into());
        }
        read_identity(&package_root, &package.plan)?;
        let team = Team::load(&resolve_file(&package_root, &package.plan.file)?, root)?;
        if team.model.capacity != model.metadata.max_context
            || team.model.batch.as_ref().unwrap().state.is_empty()
        {
            return Err("A2 service context/state contract differs".into());
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        eprintln!(
            "A2 MODEL READY after {:.3}s; owners={}, context={}, rows={}",
            team.load_s,
            team.model.batch.as_ref().unwrap().owners,
            team.model.capacity,
            team.rows
        );
        Ok(Self {
            team,
            frontend_assets: model.frontend_assets,
            cache: Cache::new(options.prefix_cache_bytes),
            prefix: Default::default(),
            statistics: Default::default(),
            active: Default::default(),
            abandoned: Default::default(),
            owner: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_snapshot: 0,
            hints: vec![],
            prefill_tokens_per_s: 4000.,
        })
    }
    pub fn expert_cache_statistics(&self) -> Option<crate::execution::ExpertCacheStatistics> {
        None
    }
    pub fn frontend_assets(&self) -> &BTreeMap<String, String> {
        &self.frontend_assets
    }
    pub fn batching_supported(&self) -> bool {
        true
    }
    pub fn prefill_packet_capacity(&self) -> usize {
        self.team.rows
    }
    pub fn scheduler_statistics(&self) -> scheduler::Statistics {
        self.statistics.clone()
    }
    pub fn max_context(&self) -> usize {
        self.team.model.capacity
    }
    pub fn mtp_drafts(&self) -> usize {
        0
    }
    pub fn vocab(&self) -> usize {
        129280
    }
    pub fn vision(&self) -> Option<&crate::vision::VisionSpec> {
        None
    }
    pub fn speculation_statistics(&self) -> Option<&crate::mtp::Statistics> {
        None
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix
    }
    pub fn prefix_cache_enabled(&self) -> bool {
        self.cache.budget > 0
    }
    pub fn set_prefix_cache_hints(&mut self, hints: Vec<usize>) {
        self.hints = hints;
    }
    fn matched(&self, input: &[u32]) -> Option<u64> {
        self.cache
            .match_prefix_by(input, &Media::default(), |entry| {
                Some(-(entry.tokens.len() as f64))
            })
            .checkpoint
    }
    pub fn cached_prefix_tokens(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<usize> {
        self.require_text(images, &[])?;
        Ok(self
            .matched(input)
            .map_or(0, |id| self.cache.entries[&id].tokens.len()))
    }
    fn require_text(
        &self,
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<()> {
        if !images.is_empty() || !audios.is_empty() {
            Err("A2 native service supports text inputs".into())
        } else {
            Ok(())
        }
    }
    pub fn request_hint(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<scheduler::RequestHint> {
        self.require_text(images, audios)?;
        Ok(scheduler::RequestHint {
            owner: self.owner,
            tokens: input.to_vec(),
            media: Media::default(),
            image_work: 0,
        })
    }
    pub fn estimated_request_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<scheduler::Waiting> {
        self.estimated_hint_cost(&self.request_hint(input, images, &[])?)
    }
    pub fn estimated_hint_cost(&self, hint: &scheduler::RequestHint) -> Result<scheduler::Waiting> {
        if hint.owner != self.owner {
            return Err("Foreign A2 request hint".into());
        }
        let remaining = hint
            .tokens
            .len()
            .saturating_sub(self.cached_prefix_tokens(&hint.tokens, &[])?);
        Ok(scheduler::Waiting {
            age_s: 0.,
            remaining_s: remaining as f64 / self.prefill_tokens_per_s,
            restore_s: 0.,
            startup_s: 0.,
            remaining_tokens: remaining,
        })
    }
    pub fn decode_admission_allowed(
        &self,
        _active: &[&Request],
        _tokens: usize,
        _options: &scheduler::Options,
    ) -> bool {
        true
    }
    fn reap(&mut self) -> Result<()> {
        let abandoned = std::mem::take(
            &mut *self
                .abandoned
                .lock()
                .map_err(|_| "A2 abandoned queue poisoned")?,
        );
        for slot in abandoned {
            if self.active.remove(&slot) {
                self.team.exchange(Command::Release(Some(slot)))?;
            }
        }
        Ok(())
    }
    fn evict(&mut self) -> Result<bool> {
        if let Some(snapshot) = self.cache.evict_one() {
            self.team.exchange(Command::Evict(snapshot.id))?;
            self.prefix.evictions += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn can_admit(
        &mut self,
        input: &GenerationInput,
        options: &scheduler::Options,
    ) -> Result<bool> {
        self.reap()?;
        self.require_text(&input.images, &input.audios)?;
        input.sampling.validate()?;
        if input.input_tokens.is_empty()
            || input.max_new_tokens == 0
            || input
                .input_tokens
                .len()
                .checked_add(input.max_new_tokens)
                .is_none_or(|n| n > self.max_context())
            || input
                .input_tokens
                .iter()
                .any(|&id| id as usize >= self.vocab())
        {
            return Err("A2 request exceeds token/context contract".into());
        }
        if self.active.len()
            >= options
                .max_active
                .min(self.team.model.batch.as_ref().unwrap().owners)
        {
            return Ok(false);
        }
        let needed = self
            .team
            .owner_bytes
            .checked_add(options.memory_reserve_bytes)
            .and_then(|n| n.checked_add(3 << 30))
            .ok_or("A2 admission size overflow")?;
        while self.team.minimum_bytes(Command::Memory)? < needed {
            if !self.evict()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub fn start_request(
        &mut self,
        input: GenerationInput,
        cancelled: impl Fn() -> bool,
    ) -> Result<Request> {
        let started = Instant::now();
        if cancelled() {
            return Err("A2 request cancelled before admission".into());
        }
        self.reap()?;
        self.require_text(&input.images, &input.audios)?;
        let slot = (0..self.team.model.batch.as_ref().unwrap().owners)
            .find(|slot| !self.active.contains(slot))
            .ok_or("A2 owner capacity exhausted")?;
        self.team.exchange(Command::Admit(vec![slot]))?;
        self.active.insert(slot);
        let mut prefix = crate::prefix::Statistics::default();
        let mut offset = 0;
        if let Some(id) = self.matched(&input.input_tokens) {
            let entry = self.cache.touch(id);
            let at = Instant::now();
            let restored = self.team.exchange(Command::Restore {
                owner: slot,
                snapshot: entry.snapshot.id,
            });
            if let Err(error) = restored {
                self.active.remove(&slot);
                let _ = self.team.exchange(Command::Release(Some(slot)));
                return Err(error);
            }
            prefix.cached_tokens = entry.tokens.len();
            prefix.matched_tokens = entry.tokens.len();
            prefix.restore_s = at.elapsed().as_secs_f64();
            offset = entry.tokens.len();
        }
        self.statistics.admissions += 1;
        self.statistics.request_start_s += started.elapsed().as_secs_f64();
        self.statistics.prefix_restore_s += prefix.restore_s;
        self.statistics.peak_active = self.statistics.peak_active.max(self.active.len());
        Ok(Request {
            owner: self.owner,
            slot,
            abandoned: Arc::clone(&self.abandoned),
            history: input.input_tokens.clone(),
            input: input.input_tokens,
            offset,
            generated: 0,
            limit: input.max_new_tokens,
            prefilling: true,
            released: false,
            sampling: input.sampling,
            decoder: Default::default(),
            prefix,
            mtp: Default::default(),
            last_output: Instant::now(),
            cache_prompt: true,
            cache_generated: true,
        })
    }
    fn store(&mut self, request: &mut Request) -> Result<()> {
        if self.cache.budget == 0 {
            return Ok(());
        }
        let count = if request.prefilling {
            request.offset
        } else {
            request.history.len().saturating_sub(1)
        };
        if count == 0
            || self
                .cache
                .entries
                .values()
                .any(|e| e.tokens == request.history[..count])
        {
            return Ok(());
        }
        let at = Instant::now();
        // State is owned by the workers; cache handles never cross device buffers.
        while self.team.minimum_bytes(Command::Memory)? < self.team.owner_bytes + (3 << 30) {
            if !self.evict()? {
                return Ok(());
            }
        }
        self.next_snapshot = self
            .next_snapshot
            .checked_add(1)
            .ok_or("A2 snapshot identity overflow")?;
        let id = self.next_snapshot;
        let bytes = self.team.maximum_bytes(Command::Save {
            owner: request.slot,
            snapshot: id,
        })?;
        if bytes > self.cache.budget {
            self.team.exchange(Command::Evict(id))?;
            return Ok(());
        }
        while self.cache.bytes + bytes > self.cache.budget {
            if !self.evict()? {
                self.team.exchange(Command::Evict(id))?;
                return Ok(());
            }
        }
        self.cache.insert(Entry::new(
            request.history[..count].to_vec(),
            Media::default(),
            Snapshot { id, bytes },
            bytes,
            0,
            false,
        ));
        request.prefix.store_s += at.elapsed().as_secs_f64();
        request.prefix.checkpoints_stored += 1;
        request.prefix.resident_bytes = self.cache.bytes;
        request.prefix.entries = self.cache.entries.len();
        self.prefix.resident_bytes = self.cache.bytes;
        self.prefix.entries = self.cache.entries.len();
        Ok(())
    }
    pub fn share_prefill_checkpoint(
        &mut self,
        request: &mut Request,
        hint: &scheduler::RequestHint,
    ) -> Result<bool> {
        if request.owner != self.owner || hint.owner != self.owner {
            return Err("Foreign A2 request".into());
        }
        if request.prefilling
            && request.offset > 0
            && request.input[..request.offset]
                == hint.tokens[..hint.tokens.len().min(request.offset)]
        {
            self.store(request)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub fn advance_requests(
        &mut self,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
    ) -> Result<Vec<StepOutput>> {
        self.advance_requests_with(requests, options, || {})
    }
    pub fn advance_requests_with(
        &mut self,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        self.reap()?;
        if requests.iter().any(|r| r.owner != self.owner || r.released) {
            return Err("Foreign or released A2 request".into());
        }
        let has_prefill = requests.iter().any(|r| !r.is_finished() && r.prefilling);
        let has_decode = requests.iter().any(|r| !r.is_finished() && !r.prefilling);
        let mut budget = if has_prefill {
            if has_decode {
                ((options.prefill_budget_ms / 1000. * self.prefill_tokens_per_s) as usize)
                    .clamp(1, self.team.rows)
            } else {
                // With no decoder waiting for its next token, fill the same
                // native packet instead of buying another weight/collective
                // pass from a latency-oriented throughput prior. Per-request
                // checkpoint boundaries below still bound each segment.
                self.team.rows
            }
        } else {
            self.team
                .model
                .batch
                .as_ref()
                .unwrap()
                .owners
                .min(options.max_batch_tokens)
        };
        let mut segments = vec![];
        let mut tokens = vec![];
        let mut outputs = vec![];
        let mut selected = vec![];
        // Existing decode rows precede bounded prefill work in a shared packet.
        for prefill in [false, true] {
            for (i, request) in requests.iter().enumerate() {
                if budget == 0 || request.is_finished() || request.prefilling != prefill {
                    continue;
                }
                let count = if prefill {
                    let next = 4096 - request.offset % 4096;
                    (request.input.len() - request.offset).min(budget).min(next)
                } else {
                    1
                };
                let values = if prefill {
                    request.input[request.offset..request.offset + count].to_vec()
                } else {
                    vec![*request.history.last().ok_or("Missing A2 decode input")?]
                };
                if !prefill || request.offset + count == request.input.len() {
                    outputs.push(segments.len());
                }
                segments.push(BatchSegment {
                    slot: request.slot,
                    tokens: count,
                });
                tokens.push(values);
                selected.push((i, count));
                budget -= count;
            }
        }
        if segments.is_empty() {
            maintenance();
            return Ok(vec![]);
        }
        let full = selected.iter().any(|&(i, _)| {
            !requests[i].sampling.is_greedy()
                || requests[i].sampling.top_logprobs.is_some()
                || requests[i].decoder.constrained()
        });
        let batch = segments.len();
        let prefill_rows = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .map(|&(_, count)| count)
            .sum::<usize>();
        let prefill_requests = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .count();
        let layout = Layout::new(
            segments,
            outputs,
            !has_prefill,
            if has_prefill {
                self.team.rows
            } else {
                self.team.model.batch.as_ref().unwrap().owners
            },
            self.team.model.batch.as_ref().unwrap().owners,
        )?;
        let (values, seconds, preparation) =
            self.team.forward_with(layout, tokens, full, maintenance)?;
        self.statistics.iterations += 1;
        self.statistics.compute_s += seconds;
        self.statistics.batch_plan_s += preparation;
        *self.statistics.batch_histogram.entry(batch).or_default() += 1;
        if prefill_rows > 0 {
            *self
                .statistics
                .prefill_chunk_histogram
                .entry(prefill_rows)
                .or_default() += 1;
            *self
                .statistics
                .prefill_batch_histogram
                .entry(prefill_requests)
                .or_default() += 1;
            self.statistics.bounded_prefill_iterations += 1;
            self.statistics.max_bounded_prefill_s =
                self.statistics.max_bounded_prefill_s.max(seconds);
            if prefill_requests < batch {
                self.statistics.mixed_iterations += 1;
            }
        }
        let mut steps = vec![];
        for (i, count) in selected {
            let request = &mut requests[i];
            if request.prefilling {
                request.offset += count;
                self.statistics.prefill_tokens += count;
            } else {
                self.statistics.decode_tokens += 1;
            }
            if let Some(output) = values.iter().find(|output| output.owner == request.slot) {
                if request.prefilling && request.cache_prompt {
                    self.store(request)?;
                }
                let token = if full {
                    let law =
                        request
                            .decoder
                            .law(&output.logits, &request.history, &request.sampling)?;
                    let token = law.distribution.draw(crate::sampling::counter_uniform(
                        request.sampling.seed,
                        0,
                        request.generated as u64,
                    ))?;
                    request.decoder.commit(token, law)?;
                    token
                } else {
                    request.decoder.consume(output.token)?;
                    output.token
                };
                request.history.push(token);
                request.generated += 1;
                request.prefilling = false;
                request.last_output = Instant::now();
                steps.push(StepOutput {
                    request: i,
                    tokens: vec![token],
                });
            }
            if request.prefilling && request.cache_prompt && request.offset.is_multiple_of(4096) {
                self.store(request)?;
            }
        }
        Ok(steps)
    }
    pub fn finish_request(&mut self, request: &mut Request, cache: bool) -> Result<()> {
        if request.owner != self.owner {
            return Err("Foreign A2 request".into());
        }
        if request.released {
            return Ok(());
        }
        if cache && request.cache_generated {
            self.store(request)?;
        }
        self.team.exchange(Command::Release(Some(request.slot)))?;
        self.active.remove(&request.slot);
        request.released = true;
        request.prefix.resident_bytes = self.cache.bytes;
        request.prefix.entries = self.cache.entries.len();
        self.prefix = request.prefix.clone();
        self.prefix.resident_bytes = self.cache.bytes;
        self.prefix.entries = self.cache.entries.len();
        Ok(())
    }
    pub fn generate(
        &mut self,
        input: &[u32],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        let hints = std::mem::take(&mut self.hints);
        let mut request = self.start_request(
            GenerationInput {
                input_tokens: input.to_vec(),
                images: vec![],
                audios: vec![],
                max_new_tokens: limit,
                sampling: options.clone(),
                prefix_hints: hints,
            },
            &cancelled,
        )?;
        let mut delivered = 0;
        let mut stopped = false;
        let result = (|| {
            while !request.is_finished() && !stopped {
                if cancelled() {
                    return Err("Request cancelled".into());
                }
                for step in
                    self.advance_requests(&mut [&mut request], &scheduler::Options::default())?
                {
                    for token in step.tokens {
                        delivered += 1;
                        if !emit(token) {
                            stopped = true;
                            break;
                        }
                    }
                }
            }
            Ok(delivered)
        })();
        let cleanup = self.finish_request(&mut request, result.is_ok() && !cancelled());
        match (result, cleanup) {
            (Ok(n), Ok(())) => Ok(n),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    }
    pub fn generate_visual(
        &mut self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.require_text(images, &[])?;
        self.generate(input, limit, options, cancelled, emit)
    }
}
