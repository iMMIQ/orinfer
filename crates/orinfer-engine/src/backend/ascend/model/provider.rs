//! Native A2 generation provider behind the common continuous worker API.
use super::{
    batch::Layout,
    memory,
    team::{Command, Output, Team},
};
mod budget;
mod dspark;
mod offload;
mod packet;
#[cfg(test)]
use crate::prefix::Cache;
use crate::{
    artifact::{FileIdentity, Result, resolve_file},
    model::{GenerationInput, StepOutput},
    prefix::{Admission, Entry, Media, Resident, Tiered},
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
pub(super) struct Descriptor {
    pub(super) schema_version: u32,
    pub(super) architecture: String,
    pub(super) compute_policy: String,
    pub(super) execution_package: String,
    pub(super) frontend_assets: BTreeMap<String, String>,
    pub(super) metadata: Metadata,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Metadata {
    pub(super) target: String,
    pub(super) max_context: usize,
    pub(super) vocab: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Package {
    pub(crate) schema_version: u32,
    pub(crate) backend: String,
    pub(crate) plan: FileIdentity,
}
struct Snapshot {
    id: u64,
    banks: BTreeMap<u64, usize>,
    output: Option<Arc<Output>>,
    host_overhead: usize,
}
impl Resident for Snapshot {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
        // Even identities name device banks, odd identities name CPU heads.
        // Every endpoint reports a shared bank with the same physical extent.
        for (&id, &bytes) in &self.banks {
            out.insert(id * 2, bytes);
        }
        let overhead = self.host_overhead
            + self
                .output
                .as_ref()
                .map_or(0, |o| o.logits.len() * 4 + std::mem::size_of::<Output>());
        if overhead != 0 {
            out.insert(self.id * 2 + 1, overhead);
        }
    }
}
#[cfg(test)]
fn matched(cache: &Cache<Snapshot>, input: &[u32], reserve: usize) -> Option<u64> {
    cache
        .match_prefix_by(input, &Media::default(), |entry| {
            (entry.tokens.len() <= input.len().saturating_sub(reserve)
                || entry.tokens.len() == input.len() && entry.snapshot.output.is_some())
            .then_some(-(entry.tokens.len() as f64))
        })
        .checkpoint
}
pub(crate) struct Runtime {
    team: Team,
    pub frontend_assets: BTreeMap<String, String>,
    cache: Tiered<Snapshot, Snapshot>,
    prefix: crate::prefix::Statistics,
    statistics: scheduler::Statistics,
    active: BTreeSet<usize>,
    resident: BTreeSet<usize>,
    abandoned: Arc<Mutex<Vec<usize>>>,
    owner: u64,
    next_snapshot: u64,
    hints: Vec<usize>,
    prefill_tokens_per_s: f64,
    draft_limit: usize,
    speculation_policy: dspark::Policy,
    speculation: crate::mtp::Statistics,
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
    draft_feedback: dspark::Feedback,
    last_output: Instant,
    last_prefill: Instant,
    ready_output: Option<Arc<Output>>,
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
    fn needs_logits(&self) -> bool {
        !self.sampling.is_greedy()
            || self.sampling.top_logprobs.is_some()
            || self.decoder.constrained()
    }
    fn accept_output(&mut self, output: &Output) -> Result<u32> {
        let token = if self.needs_logits() {
            let law = self
                .decoder
                .law(&output.logits, &self.history, &self.sampling)?;
            let token = law.distribution.draw(crate::sampling::counter_uniform(
                self.sampling.seed,
                0,
                self.generated as u64,
            ))?;
            self.decoder.commit(token, law)?;
            token
        } else {
            self.decoder.consume(output.token)?;
            output.token
        };
        self.history.push(token);
        self.generated += 1;
        self.prefilling = false;
        self.last_output = Instant::now();
        Ok(token)
    }
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

fn cached_outputs(requests: &mut [&mut Request]) -> Result<Vec<StepOutput>> {
    let mut steps = vec![];
    for (i, request) in requests.iter_mut().enumerate() {
        if !request.is_finished()
            && let Some(output) = request.ready_output.take()
        {
            let token = request.accept_output(&output)?;
            steps.push(StepOutput {
                request: i,
                tokens: vec![token],
            });
        }
    }
    Ok(steps)
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
        if options.prompt_lookup {
            return Err("A2 native service does not enable prompt lookup".into());
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
        let package_root = crate::operators::resolve(cache_root, &model.execution_package)?;
        let raw = std::fs::read(resolve_file(&package_root, "package.json")?)
            .map_err(|e| e.to_string())?;
        let package: Package = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
        if package.schema_version != 1 || package.backend != "ascend" {
            return Err("Invalid native A2 execution package".into());
        }
        let team = Team::load_with_drafts(
            &resolve_file(&package_root, &package.plan.file)?,
            root,
            options.mtp_drafts,
        )?;
        let draft_limit = options
            .mtp_drafts
            .unwrap_or_else(|| team.draft_blocks.first().copied().unwrap_or(0));
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
        let host_budget = options.kv_offload_bytes / team.ranks();
        Ok(Self {
            team,
            frontend_assets: model.frontend_assets,
            cache: Tiered::new(options.prefix_cache_bytes, host_budget),
            prefix: Default::default(),
            statistics: Default::default(),
            active: Default::default(),
            resident: Default::default(),
            abandoned: Default::default(),
            owner: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_snapshot: 0,
            hints: vec![],
            prefill_tokens_per_s: 4000.,
            draft_limit,
            speculation_policy: Default::default(),
            speculation: Default::default(),
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
        let mut statistics = self.statistics.clone();
        statistics.speculation = self.speculation_statistics().cloned();
        statistics
    }
    pub fn max_context(&self) -> usize {
        self.team.model.capacity
    }
    pub fn mtp_drafts(&self) -> usize {
        self.draft_limit
    }
    pub fn vocab(&self) -> usize {
        129280
    }
    pub fn vision(&self) -> Option<&crate::vision::VisionSpec> {
        None
    }
    pub fn speculation_statistics(&self) -> Option<&crate::mtp::Statistics> {
        (self.draft_limit > 0).then_some(&self.speculation)
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix
    }
    pub fn prefix_cache_enabled(&self) -> bool {
        self.cache.enabled()
    }
    pub fn set_prefix_cache_hints(&mut self, hints: Vec<usize>) {
        self.hints = hints;
    }
    fn final_decoder_window(&self) -> usize {
        self.team
            .model
            .batch
            .as_ref()
            .and_then(|c| c.decoder_replay.as_ref())
            .filter(|r| r.final_prompt_only)
            .map_or(0, |r| r.keep)
    }
    fn complete_prompt_cache(&self) -> bool {
        self.prefix_cache_enabled()
            && self
                .team
                .model
                .batch
                .as_ref()
                .and_then(|c| c.decoder_replay.as_ref())
                .is_some_and(|r| r.final_prompt_only && !r.decoder_state.is_empty())
    }
    pub fn cached_prefix_tokens(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<usize> {
        self.require_text(images, &[])?;
        Ok(self
            .matched(input)
            .map_or(0, |hit| self.entry(hit).tokens.len()))
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
                self.release_resident(slot)?;
            }
        }
        Ok(())
    }
    fn release_resident(&mut self, slot: usize) -> Result<()> {
        if self.resident.remove(&slot) {
            self.team.exchange(Command::Release(Some(slot)))?;
        }
        Ok(())
    }
    fn head_only(&self, input: &GenerationInput) -> bool {
        input.max_new_tokens == 1
            && self.matched(&input.input_tokens).is_some_and(|id| {
                self.entry(id).tokens.len() == input.input_tokens.len()
                    && self.entry(id).snapshot.output.is_some()
            })
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
        // A complete cached head with one requested output never needs KV.
        // Allocating an owner here can evict the very checkpoints being hit.
        if self.head_only(input) {
            return Ok(true);
        }
        let needed = self.team.owner_requirement(
            input
                .input_tokens
                .len()
                .checked_add(input.max_new_tokens)
                .ok_or("Owner token budget overflow")?,
            options.memory_reserve_bytes,
        )?;
        memory::ensure_budget(self, needed)
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
        let reap_s = started.elapsed().as_secs_f64();
        self.require_text(&input.images, &input.audios)?;
        let slot = (0..self.team.model.batch.as_ref().unwrap().owners)
            .find(|slot| !self.active.contains(slot))
            .ok_or("A2 owner capacity exhausted")?;
        let resident = !self.head_only(&input);
        let setup_s = started.elapsed().as_secs_f64() - reap_s;
        let mut admit_s = 0.;
        let mut limit_s = 0.;
        if resident {
            let at = Instant::now();
            self.team.exchange(Command::Admit(vec![slot]))?;
            admit_s = at.elapsed().as_secs_f64();
            let at = Instant::now();
            self.team.exchange(Command::Limit(
                slot,
                input
                    .input_tokens
                    .len()
                    .checked_add(input.max_new_tokens)
                    .ok_or("Owner token budget overflow")?,
            ))?;
            limit_s = at.elapsed().as_secs_f64();
            self.resident.insert(slot);
        }
        self.active.insert(slot);
        let mut prefix = crate::prefix::Statistics::default();
        let mut offset = 0;
        let mut ready_output = None;
        let at = Instant::now();
        let matched = self.matched(&input.input_tokens);
        let lookup_s = at.elapsed().as_secs_f64();
        if let Some(id) = matched {
            let entry = if id.0 {
                self.cache.host.touch(id.1)
            } else {
                self.cache.device.touch(id.1)
            };
            if id.0 {
                prefix.host_hits += 1;
            }
            let at = Instant::now();
            if resident {
                let restored = self.team.exchange(Command::Restore {
                    owner: slot,
                    snapshot: entry.snapshot.id,
                });
                if let Err(error) = restored {
                    self.active.remove(&slot);
                    self.resident.remove(&slot);
                    let _ = self.team.exchange(Command::Release(Some(slot)));
                    return Err(error);
                }
                prefix.restore_s = at.elapsed().as_secs_f64();
            }
            prefix.cached_tokens = entry.tokens.len();
            prefix.matched_tokens = entry.tokens.len();
            offset = entry.tokens.len();
            if offset == input.input_tokens.len() {
                ready_output.clone_from(&entry.snapshot.output);
            }
        }
        self.statistics.admissions += 1;
        self.statistics.request_start_s += started.elapsed().as_secs_f64();
        self.statistics.prefix_restore_s += prefix.restore_s;
        self.statistics.peak_active = self.statistics.peak_active.max(self.active.len());
        if std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some() {
            eprintln!(
                "A2 request start {}",
                serde_json::json!({
                    "owner": slot, "tokens": input.input_tokens.len(),
                    "cached_tokens": offset, "resident": resident,
                    "wall_s": started.elapsed().as_secs_f64(), "reap_s": reap_s,
                    "setup_s": setup_s, "admit_s": admit_s, "limit_s": limit_s,
                    "lookup_s": lookup_s, "restore_s": prefix.restore_s,
                })
            );
        }
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
            draft_feedback: Default::default(),
            last_output: Instant::now(),
            last_prefill: Instant::now(),
            ready_output,
            cache_prompt: true,
            cache_generated: true,
        })
    }
    fn store(&mut self, request: &mut Request, output: Option<&Output>) -> Result<()> {
        if !self.prefix_cache_enabled() || !self.resident.contains(&request.slot) {
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
                .device
                .entries
                .values()
                .chain(self.cache.host.entries.values())
                .any(|e| {
                    e.tokens == request.history[..count] && (output.is_none() || e.logits_valid)
                })
        {
            return Ok(());
        }
        if output.is_some_and(|o| o.logits.len() != self.vocab()) {
            return Err("Complete A2 prompt cache requires the unprocessed target logits".into());
        }
        let at = Instant::now();
        self.next_snapshot = self
            .next_snapshot
            .checked_add(1)
            .filter(|&id| id <= u64::MAX / 4)
            .ok_or("A2 snapshot identity overflow")?;
        let id = self.next_snapshot;
        let head = output.cloned().map(Arc::new);
        let mut host = self.cache.device.budget == 0;
        let (snapshot, bytes) = loop {
            // Eviction can remove the owner's parent lineage. Re-plan after
            // every reclamation, before allocating or committing any new bank.
            let banks = if host {
                self.team
                    .host_allocations(request.slot, id, output.is_some(), true)?
            } else {
                self.team
                    .plan_snapshot_allocations(request.slot, id, output.is_some())?
            };
            let incoming = banks
                .get(&crate::backend::kv::checkpoint_id(id)?)
                .copied()
                .unwrap_or(0);
            let device_bytes = banks.values().try_fold(0usize, |sum, bytes| {
                sum.checked_add(*bytes)
                    .ok_or("Snapshot accounting overflow")
            })?;
            let host_overhead = if host {
                self.host_overhead(count, banks.len())?
            } else {
                0
            };
            let bytes = device_bytes
                .checked_add(host_overhead)
                .ok_or("Host snapshot accounting overflow")?
                .checked_add(
                    output.map_or(0, |o| o.logits.len() * 4 + std::mem::size_of::<Output>()),
                )
                .ok_or("Snapshot size overflow")?;
            let budget = if host {
                self.cache.host.budget
            } else {
                self.cache.device.budget
            };
            if bytes > budget {
                if !host && self.cache.host.budget > 0 {
                    host = true;
                    continue;
                }
                return Ok(());
            }
            let snapshot = Snapshot {
                id,
                banks,
                output: head.clone(),
                host_overhead,
            };
            let cache = if host {
                &self.cache.host
            } else {
                &self.cache.device
            };
            if cache.admission(&snapshot, 0) != Admission::Ready {
                if !(if host {
                    self.evict_host()?
                } else {
                    self.evict()?
                }) {
                    return Ok(());
                }
                continue;
            }
            if !host
                && incoming != 0
                && !self
                    .team
                    .has_budget(memory::Requirement::snapshot(incoming)?)?
            {
                if !memory::reclaim_one(self)? {
                    if self.cache.host.budget > 0 {
                        host = true;
                        continue;
                    }
                    return Ok(());
                }
                continue;
            }
            let actual = if host {
                self.team
                    .host_allocations(request.slot, id, output.is_some(), false)?
            } else {
                self.team
                    .snapshot_allocations(request.slot, id, output.is_some())?
            };
            if actual != snapshot.banks {
                return Err("A2 snapshot allocation differs from its reserved plan".into());
            }
            break (snapshot, bytes);
        };
        // Upgrade endpoints across both tiers after the new snapshot is complete.
        for cache in [&mut self.cache.device, &mut self.cache.host] {
            let obsolete = cache
                .entries
                .iter()
                .filter(|(_, e)| e.tokens == request.history[..count])
                .map(|(&id, _)| id)
                .collect::<Vec<_>>();
            for old in obsolete {
                let entry = cache.remove(old).ok_or("Missing upgraded A2 prefix")?;
                self.team.exchange(Command::Evict(entry.snapshot.id))?;
            }
        }
        let cache = if host {
            &mut self.cache.host
        } else {
            &mut self.cache.device
        };
        cache.insert(Entry::new(
            request.history[..count].to_vec(),
            Media::default(),
            snapshot,
            bytes,
            0,
            output.is_some(),
        ));
        if host {
            request.prefix.host_stores += 1;
        }
        request.prefix.store_s += at.elapsed().as_secs_f64();
        request.prefix.checkpoints_stored += 1;
        self.cache
            .update_statistics(self.team.ranks(), &mut request.prefix);
        self.cache
            .update_statistics(self.team.ranks(), &mut self.prefix);
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
            && self.resident.contains(&request.slot)
            && request.offset > 0
            && request.input[..request.offset]
                == hint.tokens[..hint.tokens.len().min(request.offset)]
        {
            self.store(request, None)?;
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
        if self.draft_limit > 0
            && requests
                .iter()
                .any(|r| !r.is_finished() && !r.prefilling && r.ready_output.is_none())
        {
            return self.advance_dspark(requests, options, maintenance);
        }
        let Some(packet::Selection {
            layout,
            tokens,
            selected,
            full,
            batch,
            prefill_rows,
            prefill_requests,
        }) = self.select_packet(requests, options, None)?
        else {
            maintenance();
            return cached_outputs(requests);
        };
        if self.team.needs_preparation(&layout)?
            && !memory::ensure_budget(self, memory::Requirement::workspace())?
        {
            return Err("Insufficient A2 workspace after reclaiming cached state".into());
        }
        let (values, timing) = self.team.forward_with(layout, tokens, full, maintenance)?;
        let seconds = timing.seconds;
        let preparation = timing.preparation;
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
        let decoders = batch - prefill_requests;
        if decoders > 0 {
            self.speculation_policy.baseline_execution(
                decoders,
                full,
                prefill_rows > 0,
                seconds,
                timing.replay,
            );
        }
        let mut steps = self.commit_packet(requests, selected, &values)?;
        steps.extend(cached_outputs(requests)?);
        Ok(steps)
    }
    fn commit_packet(
        &mut self,
        requests: &mut [&mut Request],
        selected: Vec<(usize, usize)>,
        values: &[Output],
    ) -> Result<Vec<StepOutput>> {
        let complete_cache = self.complete_prompt_cache();
        let keep = self.final_decoder_window();
        let mut steps = vec![];
        for (i, count) in selected {
            let request = &mut requests[i];
            if request.prefilling {
                request.offset += count;
                request.last_prefill = Instant::now();
                self.statistics.prefill_tokens += count;
            } else {
                self.statistics.decode_tokens += 1;
            }
            if let Some(output) = values.iter().find(|output| output.owner == request.slot) {
                if request.prefilling && request.cache_prompt {
                    self.store(request, complete_cache.then_some(output))?;
                }
                let token = request.accept_output(output)?;
                steps.push(StepOutput {
                    request: i,
                    tokens: vec![token],
                });
            }
            if request.prefilling
                && request.cache_prompt
                && (request.offset.is_multiple_of(4096)
                    || keep > 0 && request.input.len() - request.offset == keep)
            {
                self.store(request, None)?;
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
            self.store(request, None)?;
        }
        self.release_resident(request.slot)?;
        self.active.remove(&request.slot);
        request.released = true;
        self.cache
            .update_statistics(self.team.ranks(), &mut request.prefix);
        self.prefix = request.prefix.clone();
        self.cache
            .update_statistics(self.team.ranks(), &mut self.prefix);
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

#[cfg(test)]
mod tests;
