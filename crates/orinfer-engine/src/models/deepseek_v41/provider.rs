//! Native A2 generation provider behind the common continuous worker API.
use super::{
    batch::Layout,
    memory,
    team::{Command, Output, Team},
};
mod dspark;
mod offload;
mod packet;
mod prefill;
mod vision;

impl memory::Pressure for Runtime {
    fn fits(&mut self, requirement: memory::Requirement) -> Result<bool> {
        self.team.has_budget(requirement)
    }
    fn evict_prefix(&mut self) -> Result<bool> {
        self.evict_fast_except(None, 16)
    }
    fn evict_program(&mut self) -> Result<bool> {
        let reclaimed = self.team.reclaim_programs()?;
        if reclaimed {
            self.health.program_reclaims += 1;
        }
        Ok(reclaimed)
    }
}
/// Admission pressure protects the incoming request's matched device prefix
/// against self-eviction and keeps one idle owner arena so `owner_requirement`
/// computed with `new_banks = 0` stays valid through `StartOwner`.
struct AdmissionPressure<'a> {
    runtime: &'a mut Runtime,
    protected: Option<u64>,
}
impl memory::Pressure for AdmissionPressure<'_> {
    fn fits(&mut self, requirement: memory::Requirement) -> Result<bool> {
        self.runtime.team.has_budget(requirement)
    }
    fn evict_prefix(&mut self) -> Result<bool> {
        self.runtime.evict_fast_except(self.protected, 16)
    }
    fn evict_program(&mut self) -> Result<bool> {
        let reclaimed = self.runtime.team.reclaim_programs_for_admission()?;
        if reclaimed {
            self.runtime.health.program_reclaims += 1;
        }
        Ok(reclaimed)
    }
}
#[cfg(test)]
use crate::prefix::Cache;
use crate::{
    artifact::{FileIdentity, Result, resolve_file},
    model::request::RequestCore,
    model::{GenerationInput, StepOutput},
    prefix::{Admission, Entry, Media, Resident, Tiered},
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
    #[serde(default)]
    pub(super) chat_format: Option<String>,
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
    #[serde(default)]
    pub(crate) generated_assets: Vec<crate::operators::generated::Asset>,
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
    // Native A2 overlays consume only the current packet's image rows. The
    // encoder's physical capacity does not bound the complete image history.
    vision_spec: Option<crate::vision::VisionSpec>,
    vision_cursor: usize,
    vision_seconds_per_patch: f64,
    vision_cache: crate::vision::FeatureCache<Vec<u8>>,
    image_wave: Option<vision::Wave>,
    pub frontend_assets: BTreeMap<String, String>,
    cache: Tiered<Snapshot, Snapshot>,
    prefix: crate::prefix::Statistics,
    cache_reclamation: CacheReclamation,
    statistics: scheduler::Statistics,
    active: BTreeSet<usize>,
    resident: BTreeSet<usize>,
    pad_slot: Option<usize>,
    abandoned: Arc<Mutex<Vec<usize>>>,
    owner: u64,
    next_snapshot: u64,
    hints: Vec<usize>,
    prefill_costs: scheduler::prefill::Costs,
    input_window: Option<Arc<super::host_inputs::Window>>,
    decode_burst: scheduler::DecodeBurst,
    capture_credit_s: f64,
    draft_limit: usize,
    rejection_sampling: bool,
    speculation_policy: dspark::Policy,
    speculation: crate::mtp::Statistics,
    next_slot: usize,
    health: HealthCounters,
}
#[derive(Default)]
struct HealthCounters {
    host_evict_max_s: f64,
    prefix_save_skips: usize,
    program_reclaims: usize,
    sampling_request_failures: usize,
}
/// Shared-cache events belong to the runtime. Charge them once to the next
/// completion so request-log aggregation survives overlapping request lifetimes.
#[derive(Default)]
struct CacheReclamation {
    device: usize,
    demotions: usize,
    host: usize,
}
impl CacheReclamation {
    fn drain_into(&mut self, statistics: &mut crate::prefix::Statistics) {
        statistics.evictions += std::mem::take(&mut self.device);
        statistics.host_stores += std::mem::take(&mut self.demotions);
        statistics.host_evictions += std::mem::take(&mut self.host);
    }
}
pub(crate) struct Request {
    pub(crate) core: RequestCore,
    media: Media,
    pending_images: std::collections::VecDeque<vision::Pending>,
    image_features: Arc<Vec<super::vision::Span>>,
    device_sampling: bool,
    processed_sampling: bool,
    prefix: crate::prefix::Statistics,
    draft_feedback: dspark::Feedback,
    ready_output: Option<Arc<Output>>,
}
impl std::ops::Deref for Request {
    type Target = RequestCore;

    fn deref(&self) -> &RequestCore {
        &self.core
    }
}
impl std::ops::DerefMut for Request {
    fn deref_mut(&mut self) -> &mut RequestCore {
        &mut self.core
    }
}
impl Request {
    fn host_sampling(&self) -> bool {
        !self.sampling.is_greedy()
            || self.sampling.top_logprobs.is_some()
            || self.decoder.constrained()
    }
    fn device_sample(&self) -> bool {
        self.device_sampling
            && !self.prefilling
            && (((self.sampling.is_greedy() && self.sampling.top_logprobs.is_none()
                || self.sampling.device_categorical())
                && !self.decoder.constrained())
                || (self.processed_sampling
                    && self.host_sampling()
                    && self.sampling.device_untruncated()))
    }
    fn needs_logits(&self) -> bool {
        self.host_sampling() && !self.device_sample()
    }
    fn accept_device(&mut self, output: &Output) -> Result<u32> {
        if let Some(error) = &output.error {
            self.decoder.fail(error.clone());
            return Err(error.clone());
        }
        if self.sampling.top_logprobs.is_some() != output.report.is_some() {
            return Err("Device sampling probability record differs from request".into());
        }
        self.decoder
            .commit_device(output.token, output.report.clone())?;
        Ok(output.token)
    }
    fn verify_device(&mut self, draft: &[u32], target: &[&Output]) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(target.len());
        for (position, output) in target.iter().enumerate() {
            let token = self.accept_device(output)?;
            tokens.push(token);
            if draft.get(position) != Some(&token) || self.decoder.finished() {
                break;
            }
        }
        Ok(tokens)
    }
    fn accept_output(&mut self, output: &Output) -> Result<u32> {
        if let Some(error) = &output.error {
            self.decoder.fail(error.clone());
            return Err(error.clone());
        }
        let token = if self.host_sampling() && !(self.device_sample() && output.logits.is_empty()) {
            let core = &mut self.core;
            let law = core
                .decoder
                .law_validated(&output.logits, &core.history, &core.sampling)?;
            let token = law.distribution.draw(crate::sampling::counter_uniform(
                core.sampling.seed,
                0,
                core.generated as u64,
            ))?;
            core.decoder.commit(token, law)?;
            token
        } else {
            self.accept_device(output)?
        };
        self.history.push(token);
        self.generated += 1;
        self.prefilling = false;
        self.last_output = Instant::now();
        Ok(token)
    }
    pub fn has_pending_host_images(&self) -> bool {
        !self.pending_images.is_empty() || !self.image_features.is_empty()
    }
    pub fn prefix_statistics(&self) -> &crate::prefix::Statistics {
        &self.prefix
    }
}

fn cached_outputs(requests: &mut [&mut Request]) -> Result<Vec<StepOutput>> {
    let mut steps = vec![];
    for (i, request) in requests.iter_mut().enumerate() {
        if !request.is_finished()
            && let Some(output) = request.ready_output.take()
        {
            match request.accept_output(&output) {
                Ok(token) => steps.push(StepOutput {
                    request: i,
                    tokens: vec![token],
                }),
                Err(_) if request.decoder.failure().is_some() => {}
                Err(error) => return Err(error),
            }
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
        let vision_cache_bytes = super::load::host_memory_budget("ORINFER_VISION_CACHE_MIB", 128)?;
        let descriptor = crate::model::resolve_manifest(path)?;
        let cache_root = descriptor.parent().ok_or("Missing A2 cache root")?;
        let root = cache_root.parent().ok_or("Missing A2 model root")?;
        let model: Descriptor = crate::model::read(&descriptor)?;
        if model.schema_version != 1
            || model.architecture != "deepseek_v41"
            || model.compute_policy != "w4a8"
            || model.metadata.target != "ascend910b"
            || model.metadata.vocab != 129280
            || model
                .metadata
                .chat_format
                .as_deref()
                .is_some_and(|f| f != "deepseek_v41")
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
            options.cuda_graph != crate::execution::CudaGraphMode::Off,
        )?;
        let draft_limit = options
            .mtp_drafts
            .unwrap_or_else(|| team.draft_blocks.first().copied().unwrap_or(0));
        if team.model.capacity != model.metadata.max_context
            || team.model.batch.as_ref().unwrap().state.is_empty()
        {
            return Err("A2 service context/state contract differs".into());
        }
        let speculation_policy = dspark::Policy::load(
            dspark::Identity {
                schema_version: 2,
                execution_package: model.execution_package,
                target: model.metadata.target,
                cann: "9.2".into(),
                ranks: team.ranks(),
            },
            &package_root,
        )?;
        let rejection_sampling = crate::backend::ascend::sampling::rejection_requested()?;
        if rejection_sampling {
            let contract = team
                .model
                .sampling
                .as_ref()
                .and_then(|s| s.rejection.as_ref())
                .ok_or("Execution package has no probability rejection sampler")?;
            if contract.ranks != team.ranks()
                || contract.pool_rows != team.model.batch.as_ref().unwrap().owners * contract.stride
            {
                return Err("Rejection pool differs from the model owner/rank geometry".into());
            }
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
        let vision_spec = team.model.vision.as_ref().map(|program| {
            let mut spec = program.spec.clone();
            spec.max_features = team.model.capacity;
            spec
        });
        Ok(Self {
            team,
            vision_spec,
            vision_cursor: 0,
            vision_seconds_per_patch: 60e-6,
            vision_cache: crate::vision::FeatureCache::new(vision_cache_bytes),
            image_wave: None,
            frontend_assets: model.frontend_assets,
            cache: Tiered::new(options.prefix_cache_bytes, host_budget),
            prefix: Default::default(),
            cache_reclamation: Default::default(),
            statistics: Default::default(),
            active: Default::default(),
            resident: Default::default(),
            pad_slot: None,
            abandoned: Default::default(),
            owner: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            next_snapshot: 0,
            hints: vec![],
            // Bootstrap only; actual packet and restore observations replace it.
            prefill_costs: scheduler::prefill::Costs::new(4000.),
            input_window: None,
            decode_burst: Default::default(),
            capture_credit_s: 0.,
            draft_limit,
            rejection_sampling,
            speculation_policy,
            speculation: Default::default(),
            next_slot: 1,
            health: Default::default(),
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
    pub fn vision_parallelism(&self) -> usize {
        self.team.ranks()
    }
    pub fn scheduler_statistics(&self) -> scheduler::Statistics {
        let mut statistics = self.statistics.clone();
        statistics.speculation = self.speculation_statistics().cloned();
        let (min_rank_free_bytes, max_rank_free_bytes) = self.team.rank_free_range();
        let (dspark_cost_cells, dspark_budget_balance_s) = self.speculation_policy.diagnostics();
        statistics.backend_health = Some(scheduler::BackendHealthStatistics {
            min_rank_free_bytes,
            max_rank_free_bytes,
            device_prefix_entries: self.cache.device.entries.len(),
            device_prefix_bytes: self.cache.device.bytes,
            host_prefix_entries: self.cache.host.entries.len(),
            host_prefix_bytes: self.cache.host.bytes.saturating_mul(self.team.ranks()),
            prefix_evictions: self.prefix.evictions + self.cache_reclamation.device,
            host_prefix_evictions: self.prefix.host_evictions + self.cache_reclamation.host,
            host_evict_max_s: self.health.host_evict_max_s,
            prefix_save_skips: self.health.prefix_save_skips,
            program_reclaims: self.health.program_reclaims,
            sampling_request_failures: self.health.sampling_request_failures,
            dspark_cost_cells,
            dspark_budget_balance_s,
        });
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
        self.vision_spec.as_ref()
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
        let media = Media::images(input, images, self.vision())?;
        Ok(self
            .matched(input, &media)
            .map_or(0, |hit| self.entry(hit).tokens.len()))
    }
    fn require_text(
        &self,
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<()> {
        if !audios.is_empty() || !images.is_empty() && self.vision().is_none() {
            Err("A2 package has no adapter for the supplied media".into())
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
        let media = Media::images(input, images, self.vision())?;
        let cached = self
            .matched(input, &media)
            .map_or(0, |hit| self.entry(hit).tokens.len());
        let mut image_work = 0usize;
        if let Some(spec) = self.vision() {
            for (image, (start, identity)) in images.iter().zip(&media.0) {
                if start + spec.feature_count(image)? > cached
                    && !self.vision_cache.contains(identity)
                {
                    image_work += image.grid_height * image.grid_width;
                }
            }
        }
        Ok(scheduler::RequestHint {
            owner: self.owner,
            tokens: input.to_vec(),
            media,
            image_work,
            checkpoint: Default::default(),
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
        let hit = self.matched_hint(hint);
        let cached = hit.map_or(0, |h| self.entry(h).tokens.len());
        let remaining = hint.tokens.len().saturating_sub(cached);
        let image_s = hint.image_work as f64 * self.vision_seconds_per_patch
            / hint.media.0.len().max(1).min(self.team.ranks()) as f64;
        let startup_s = image_s
            * self
                .team
                .model
                .vision
                .as_ref()
                .map_or(1., super::vision::Program::startup_fraction);
        Ok(scheduler::Waiting {
            age_s: 0.,
            remaining_s: self.remaining_prefill_cost(cached, hint.tokens.len()) + image_s
                - startup_s,
            restore_s: hit
                .filter(|_| remaining > 0)
                .map_or(0., |h| self.prefill_costs.restore(h.0, cached)),
            startup_s,
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
    pub(super) fn ensure_pad_slot(&mut self) -> Result<Option<usize>> {
        const MAX_PAD_TOKENS: usize = 64;
        if let Some(slot) = self.pad_slot {
            if !self.active.contains(&slot) {
                return Ok(Some(slot));
            }
            self.pad_slot = None;
        }
        let owners = self.team.model.batch.as_ref().map_or(0, |b| b.owners);
        if owners < 2 || self.active.contains(&0) {
            return Ok(None);
        }
        let needed = self.team.owner_requirement(MAX_PAD_TOKENS, 0, 0)?;
        if !self.team.has_budget(needed)? {
            return Ok(None);
        }
        self.team.exchange(Command::StartOwner {
            slot: 0,
            limit: MAX_PAD_TOKENS,
            snapshot: None,
        })?;
        self.pad_slot = Some(0);
        Ok(Some(0))
    }
    fn release_pad_slot(&mut self) -> Result<()> {
        if let Some(slot) = self.pad_slot.take() {
            self.team.exchange(Command::Release(Some(slot)))?;
        }
        Ok(())
    }
    fn head_only(&self, input: &GenerationInput) -> bool {
        input.max_new_tokens == 1
            && Media::images(&input.input_tokens, &input.images, self.vision())
                .ok()
                .and_then(|media| self.matched(&input.input_tokens, &media))
                .is_some_and(|id| {
                    self.entry(id).tokens.len() == input.input_tokens.len()
                        && self.entry(id).snapshot.output.is_some()
                })
    }
    pub fn can_admit_borrowed(
        &mut self,
        input: &crate::model::AdmissionInput<'_>,
        options: &scheduler::Options,
    ) -> Result<bool> {
        self.reap()?;
        self.require_text(input.images, input.audios)?;
        if let Some(vision) = self.vision() {
            vision.image_spans(input.input_tokens, input.images)?;
        }
        input.sampling.validate()?;
        let total_tokens = input
            .input_tokens
            .len()
            .checked_add(input.max_new_tokens)
            .ok_or("Owner token budget overflow")?;
        if input.input_tokens.is_empty()
            || input.max_new_tokens == 0
            || total_tokens > self.max_context()
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
        let matched = if let Some(hint) = input.hint.filter(|h| h.owner == self.owner) {
            self.matched_hint(hint)
        } else {
            let media = Media::images(input.input_tokens, input.images, self.vision())?;
            self.matched(input.input_tokens, &media)
        };
        // A complete cached head with one requested output never needs KV.
        // Allocating an owner here can evict the very checkpoints being hit.
        if input.max_new_tokens == 1
            && matched.is_some_and(|id| {
                let entry = self.entry(id);
                entry.tokens.len() == input.input_tokens.len() && entry.snapshot.output.is_some()
            })
        {
            return Ok(true);
        }
        let (shared_tokens, protected) = match matched {
            Some((false, key)) => (self.cache.device.entries[&key].tokens.len(), Some(key)),
            _ => (0, None),
        };
        let needed = self.team.owner_requirement(
            total_tokens,
            shared_tokens,
            options.memory_reserve_bytes,
        )?;
        let capacity = self.prefill_packet_capacity();
        if options.policy != scheduler::Policy::Latency
            || self.active.is_empty()
            || input.input_tokens.len() <= capacity
        {
            let mut pressure = AdmissionPressure {
                runtime: self,
                protected,
            };
            return memory::ensure_budget(&mut pressure, needed);
        }
        if self.team.has_budget(needed)? {
            return Ok(true);
        }
        let cached = matched.map_or(0, |hit| self.entry(hit).tokens.len());
        if input.input_tokens.len().saturating_sub(cached) > capacity {
            // Reclaim stale device prefixes directly on device (without
            // synchronous PCIe offload) so cold prompts do not stall behind
            // full device caches or trigger prolonged memory_guard drains.
            if self.cache.device.bytes > 0 {
                let spill_start = Instant::now();
                for _ in 0..16 {
                    if spill_start.elapsed().as_secs_f64() >= 0.02
                        || !self.evict_fast_except(protected, 16)?
                    {
                        break;
                    }
                    if self.team.has_budget(needed)? {
                        return Ok(true);
                    }
                }
            }
            // Let live owners progress while the scheduler waits for memory
            // to admit a cold prompt. Its aged memory guard prevents starvation.
            return Ok(false);
        }
        let mut pressure = AdmissionPressure {
            runtime: self,
            protected,
        };
        if memory::reclaim_one(&mut pressure)? {
            memory::ensure_budget(&mut pressure, needed)
        } else {
            self.team.has_budget(needed)
        }
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
        let media = Media::images(&input.input_tokens, &input.images, self.vision())?;
        // Prefer pooled non-zero slots in round-robin order so short requests
        // reuse idle owner arenas without slot-0 full-session syncs or hotspots.
        let owners = self.team.model.batch.as_ref().unwrap().owners;
        let slot = if owners > 1 {
            let start = self.next_slot.clamp(1, owners - 1);
            if let Some(slot) = (start..owners)
                .chain(1..start)
                .find(|slot| !self.active.contains(slot))
            {
                self.next_slot = if slot + 1 < owners { slot + 1 } else { 1 };
                slot
            } else if !self.active.contains(&0) {
                0
            } else {
                return Err("A2 owner capacity exhausted".into());
            }
        } else if !self.active.contains(&0) {
            0
        } else {
            return Err("A2 owner capacity exhausted".into());
        };
        if self.pad_slot == Some(slot) {
            self.release_pad_slot()?;
        }
        let resident = !self.head_only(&input);
        let setup_s = started.elapsed().as_secs_f64() - reap_s;
        let mut prefix = crate::prefix::Statistics::default();
        let mut offset = 0;
        let mut ready_output = None;
        let at = Instant::now();
        let matched = self.matched(&input.input_tokens, &media);
        let lookup_s = at.elapsed().as_secs_f64();
        let snapshot_id = if let Some(id) = matched {
            let entry = if id.0 {
                self.cache.host.touch(id.1)
            } else {
                self.cache.device.touch(id.1)
            };
            if id.0 {
                prefix.host_hits += 1;
            }
            prefix.cached_tokens = entry.tokens.len();
            prefix.matched_tokens = entry.tokens.len();
            offset = entry.tokens.len();
            if offset == input.input_tokens.len() {
                ready_output.clone_from(&entry.snapshot.output);
            }
            Some((id.0, entry.snapshot.id))
        } else {
            None
        };
        let mut admit_s = 0.;
        if resident {
            let limit = input
                .input_tokens
                .len()
                .checked_add(input.max_new_tokens)
                .ok_or("Owner token budget overflow")?;
            let at = Instant::now();
            let start_cmd = Command::StartOwner {
                slot,
                limit,
                snapshot: snapshot_id.map(|(_, sid)| sid),
            };
            if snapshot_id.is_some_and(|(host_tier, _)| host_tier) {
                self.team.exchange_heavy(start_cmd)?;
            } else {
                self.team.exchange(start_cmd)?;
            }
            let elapsed = at.elapsed().as_secs_f64();
            if let Some((host_tier, _)) = snapshot_id {
                prefix.restore_s = elapsed;
                self.prefill_costs
                    .observe_restore(host_tier, offset, prefix.restore_s);
            } else {
                admit_s = elapsed;
            }
            self.resident.insert(slot);
        }
        self.active.insert(slot);
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
                    "setup_s": setup_s, "admit_s": admit_s, "limit_s": 0.0,
                    "lookup_s": lookup_s, "restore_s": prefix.restore_s,
                })
            );
        }
        let pending_images =
            self.pending_images(&input.input_tokens, input.images, offset, &media)?;
        Ok(Request {
            core: RequestCore::new(
                self.owner,
                slot,
                Arc::clone(&self.abandoned),
                input.input_tokens,
                input.max_new_tokens,
                input.sampling,
                offset,
            ),
            media,
            pending_images,
            image_features: Default::default(),
            device_sampling: self.team.model.sampling.is_some(),
            processed_sampling: self
                .team
                .model
                .sampling
                .as_ref()
                .is_some_and(|s| s.processed_local.is_some()),
            prefix,
            draft_feedback: Default::default(),
            ready_output,
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
        if count == 0 {
            return Ok(());
        }
        let tokens = &request.history[..count];
        let exists = self
            .cache
            .device
            .exact_matches(tokens, &request.media)
            .into_iter()
            .any(|id| output.is_none() || self.cache.device.entries[&id].logits_valid)
            || self
                .cache
                .host
                .exact_matches(tokens, &request.media)
                .into_iter()
                .any(|id| output.is_none() || self.cache.host.entries[&id].logits_valid);
        if exists {
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
        let head_bytes = output.map_or(0, |o| o.logits.len() * 4 + std::mem::size_of::<Output>());
        let mut host = self.cache.device.budget == 0;
        // Fast path: when device cache has headroom, plan + budget check + save
        // in one 8-rank IPC instead of two round-trips.
        let fast = if !host && self.cache.device.budget > self.cache.device.bytes + head_bytes {
            let max_incoming = self
                .cache
                .device
                .budget
                .saturating_sub(self.cache.device.bytes)
                .saturating_sub(head_bytes);
            if let Some(banks) =
                self.team
                    .try_plan_and_save(request.slot, id, output.is_some(), max_incoming)?
            {
                let device_bytes = banks.values().try_fold(0usize, |sum, bytes| {
                    sum.checked_add(*bytes)
                        .ok_or("Snapshot accounting overflow")
                })?;
                let bytes = device_bytes
                    .checked_add(head_bytes)
                    .ok_or("Snapshot size overflow")?;
                let snapshot = Snapshot {
                    id,
                    banks,
                    output: head.clone(),
                    host_overhead: 0,
                };
                if bytes <= self.cache.device.budget
                    && self.cache.device.admission(&snapshot, 0) == Admission::Ready
                {
                    Some((snapshot, bytes))
                } else {
                    self.team.exchange(Command::RollbackSave {
                        owner: request.slot,
                        snapshot: id,
                    })?;
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        let (snapshot, bytes) = if let Some(saved) = fast {
            saved
        } else {
            loop {
                // Eviction can remove the owner's parent lineage. Re-plan after
                // every reclamation, before allocating or committing any new bank.
                let (banks, min_device_free, min_session_free) = if host {
                    (
                        self.team
                            .host_allocations(request.slot, id, output.is_some(), true)?,
                        usize::MAX,
                        usize::MAX,
                    )
                } else {
                    self.team
                        .plan_snapshot_with_budget(request.slot, id, output.is_some())?
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
                    .checked_add(head_bytes)
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
                        self.evict_host_for(&snapshot)?
                    } else {
                        self.evict()?
                    }) {
                        return Ok(());
                    }
                    continue;
                }
                if !host
                    && incoming != 0
                    && !memory::Requirement::snapshot(incoming)?
                        .fits(min_device_free, min_session_free)
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
                        .try_host_allocations(request.slot, id, output.is_some(), false)?
                } else {
                    self.team
                        .try_snapshot_allocations(request.slot, id, output.is_some())?
                };
                let Some(actual) = actual else {
                    self.health.prefix_save_skips += 1;
                    if !host && self.cache.host.budget > 0 {
                        host = true;
                        continue;
                    }
                    return Ok(());
                };
                if actual != snapshot.banks {
                    return Err("A2 snapshot allocation differs from its reserved plan".into());
                }
                break (snapshot, bytes);
            }
        };
        // Upgrade endpoints across both tiers after the new snapshot is complete.
        let mut obsolete_ids = Vec::new();
        for cache in [&mut self.cache.device, &mut self.cache.host] {
            let obsolete = cache.exact_matches(&request.history[..count], &request.media);
            for old in obsolete {
                let entry = cache.remove(old).ok_or("Missing upgraded A2 prefix")?;
                obsolete_ids.push(entry.snapshot.id);
            }
        }
        if !obsolete_ids.is_empty() {
            self.team.exchange(Command::EvictMany(obsolete_ids))?;
        }
        let cache = if host {
            &mut self.cache.host
        } else {
            &mut self.cache.device
        };
        cache.insert(Entry::new(
            request.history[..count].to_vec(),
            request.media.clone(),
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
        if !request.prefilling || !self.resident.contains(&request.slot) {
            return Ok(false);
        }
        let cached = self
            .matched_hint(hint)
            .map_or(0, |hit| self.entry(hit).tokens.len());
        if request.offset > cached
            && request.media.common_tokens(&hint.media, request.offset) == request.offset
            && request.input[..request.offset]
                == hint.tokens[..hint.tokens.len().min(request.offset)]
        {
            self.store(request, None)?;
            Ok(self
                .matched_hint(hint)
                .is_some_and(|hit| self.entry(hit).tokens.len() > cached))
        } else {
            let common = request
                .input
                .iter()
                .zip(&hint.tokens)
                .take_while(|(a, b)| a == b)
                .count();
            let common = request.media.common_tokens(&hint.media, common);
            Ok(request.cache_prompt
                && (request.pending_images.is_empty()
                    || common == request.input.len() && hint.tokens.len() == request.input.len())
                && offload::pending_endpoint(
                    request.offset,
                    request.input.len(),
                    hint.tokens.len(),
                    common,
                    cached,
                    self.final_decoder_window(),
                    self.complete_prompt_cache(),
                ))
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
        self.prepare_images(requests, options)?;
        let decoding = requests
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.prefilling && !r.is_finished())
            .map(|(i, r)| (i, r.generated))
            .collect::<Vec<_>>();
        // Parallel CANN preparation uses the shared calibration horizon;
        // one model-lifetime balance funds all batches and contexts.
        self.speculation_policy.start_iteration(
            options.target_tpot_ms / 1000. * crate::mtp::adaptive::CALIBRATION_ROUNDS as f64,
        );
        let started = Instant::now();
        let before_prefill = self.statistics.prefill_tokens;
        let before_iterations = self.statistics.iterations;
        let before_mixed = self.statistics.mixed_iterations;
        let burst = options.policy == scheduler::Policy::Latency
            && self.decode_burst.allows(started, decoding.len());
        let result = self.advance_packet(requests, options, maintenance);
        if result.is_ok() && self.statistics.iterations > before_iterations {
            let seconds = started.elapsed().as_secs_f64();
            if self.statistics.prefill_tokens > before_prefill {
                // Keep DecodeBurst active for healthy decoder cohorts (>= 4) or
                // when decoders were held back by a pure-prefill step or an
                // over-budget mixed step, while avoiding wasteful decode-only
                // weight passes for tiny 1..3 decoder tails.
                if self.statistics.mixed_iterations == before_mixed
                    || decoding.len() >= 3
                    || seconds > options.target_tpot_ms / 1000.
                {
                    self.decode_burst.after_prefill(
                        Instant::now(),
                        seconds,
                        decoding.len(),
                        options,
                    );
                } else {
                    self.decode_burst.clear();
                }
            } else {
                self.decode_burst.observe_decode(
                    seconds,
                    self.speculation_policy.preparation_s(),
                    decoding.len(),
                );
                self.statistics.decode_burst_iterations += usize::from(burst);
            }
        }
        let has_role = |prefill| {
            requests
                .iter()
                .any(|r| !r.is_finished() && r.prefilling == prefill)
        };
        if options.policy != scheduler::Policy::Latency || !has_role(true) || !has_role(false) {
            self.decode_burst.clear();
        }
        let deltas = decoding
            .iter()
            .map(|&(i, before)| {
                (
                    requests[i].slot,
                    requests[i].generated.saturating_sub(before),
                )
            })
            .collect::<Vec<_>>();
        self.speculation_policy
            .finish_iteration(started.elapsed().as_secs_f64(), &deltas);
        self.capture_credit_s = if !decoding.is_empty() && result.is_ok() {
            let committed = decoding
                .iter()
                .map(|&(i, before)| requests[i].generated.saturating_sub(before))
                .min()
                .unwrap_or(0);
            super::team::capture_credit(
                self.capture_credit_s,
                options.target_tpot_ms / 1000.,
                started.elapsed().as_secs_f64(),
                committed,
            )
        } else {
            0.
        };
        result
    }
    fn advance_packet(
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
        let Some(selection) = self.select_packet(requests, options, None)? else {
            maintenance();
            return cached_outputs(requests);
        };
        self.advance_selection(selection, requests, options, false, maintenance)
    }
    pub(super) fn begin_forward_with_workspace(
        &mut self,
        layout: &Layout,
        error: &str,
    ) -> Result<super::team::PreparedForward> {
        let prepared = self.team.begin_forward(layout)?;
        if !prepared.all_graphs() && !prepared.workspace_fits {
            if !memory::ensure_budget(self, memory::Requirement::workspace())? {
                return Err(error.into());
            }
            return self.team.begin_forward(layout);
        }
        Ok(prepared)
    }
    pub(super) fn begin_verification_forward_with_workspace(
        &mut self,
        layout: &Layout,
        journal: Vec<(usize, usize)>,
        error: &str,
    ) -> Result<Option<super::team::PreparedForward>> {
        let (prepared, journal_ready) = self
            .team
            .begin_verification_forward(layout, journal.clone())?;
        let need_workspace = !prepared.all_graphs() && !prepared.workspace_fits;
        if !need_workspace && journal_ready {
            return Ok(Some(prepared));
        }
        if !memory::ensure_budget(self, memory::Requirement::workspace())? {
            if need_workspace {
                return Err(error.into());
            }
            return Ok(None);
        }
        let (prepared, journal_ready) = self.team.begin_verification_forward(layout, journal)?;
        if !prepared.all_graphs() && !prepared.workspace_fits {
            return Err(error.into());
        }
        Ok(journal_ready.then_some(prepared))
    }
    fn advance_selection(
        &mut self,
        mut selection: packet::Selection,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
        speculative_baseline: bool,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        self.bucket_selection(&mut selection, requests)?;
        let packet::Selection {
            layout,
            tokens,
            selected,
            full,
            batch,
            prefill_rows,
            prefill_requests,
            ..
        } = selection;
        let prepared = self.begin_forward_with_workspace(
            &layout,
            "Insufficient A2 workspace after reclaiming cached state",
        )?;
        let shared = packet::mark_shared(&selected, requests);
        let work = packet::work(&selected, requests, prefill_rows);
        let prefill_work = self.prefill_work(&selected, requests, full, layout.decode);
        self.speculation_policy.begin_round(
            selected
                .iter()
                .filter(|&&(i, _)| !requests[i].prefilling)
                .map(|&(i, _)| requests[i].slot)
                .collect(),
            full,
            work,
            shared,
            1,
        );
        let (uses, slack) = self.capture_context(
            &selected,
            requests,
            None,
            self.team.draft_blocks.first().copied().unwrap_or(0),
            options,
        );
        let capture = if speculative_baseline {
            self.speculation_policy
                .baseline_budget(full, layout.decode, work, uses, slack)
        } else {
            self.speculation_policy
                .target_budget(full, layout.decode, work, uses, slack)
        };
        let mut readout =
            packet::target_readout(requests, full, &BTreeMap::new(), self.vocab(), &selected)?;
        self.prepare_host_inputs(&layout, &selected, requests, &mut readout);
        let (values, timing) = self.team.finish_forward(
            prepared,
            layout,
            tokens,
            readout,
            Some(capture),
            maintenance,
        )?;
        timing.record(&mut self.statistics.batch_execution);
        self.observe_prefill(prefill_work, &timing);
        let seconds = timing.seconds;
        self.statistics.sampling_mask_rows += timing.mask_rows;
        self.statistics.sampling_report_rows += timing.report_rows;
        self.statistics.sampling_prepare_s += timing.mask_prepare_s;
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
        self.speculation_policy
            .baseline_execution(decoders, full, work, &timing);
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
                if request.prefilling && request.cache_prompt && output.error.is_none() {
                    self.store(request, complete_cache.then_some(output))?;
                }
                match request.accept_output(output) {
                    Ok(token) => steps.push(StepOutput {
                        request: i,
                        tokens: vec![token],
                    }),
                    Err(_) if request.decoder.failure().is_some() => {
                        self.health.sampling_request_failures += 1;
                    }
                    Err(error) => return Err(error),
                }
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
        if self
            .input_window
            .as_ref()
            .is_some_and(|w| w.owner == request.slot)
        {
            self.input_window = None;
        }
        if self.active.is_empty() {
            self.capture_credit_s = 0.;
            self.decode_burst.clear();
        }
        request.released = true;
        self.cache_reclamation.drain_into(&mut request.prefix);
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
