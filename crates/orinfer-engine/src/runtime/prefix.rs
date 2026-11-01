use super::*;
use crate::backend::snapshot::Range;
use crate::prefix::{Entry, Media, Resident};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(super) struct Context {
    parent: Option<(bool, u64)>,
    pub statistics: crate::prefix::Statistics,
}

fn hash_floats(hash: &mut Sha256, values: &[f32]) {
    let (chunks, tail) = values.as_chunks::<256>();
    for chunk in chunks {
        hash.update(chunk.map(f32::to_le_bytes).as_flattened());
    }
    for value in tail {
        hash.update(value.to_le_bytes());
    }
}

impl ModelRuntime {
    pub(crate) fn prefix_media(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        audios: &[crate::audio::AudioInput],
    ) -> Result<Media> {
        let mut media = vec![];
        let mut cursor = 0;
        for image in images {
            let v = self
                .manifest
                .vision
                .as_ref()
                .ok_or("Model has no image adapter")?;
            let start = input
                .get(cursor..)
                .ok_or("Image span outside prompt")?
                .iter()
                .position(|&id| id == v.image_token_id)
                .map(|p| p + cursor)
                .ok_or("Image marker missing")?;
            let mut hash = Sha256::new();
            hash.update((image.grid_height as u64).to_le_bytes());
            hash.update((image.grid_width as u64).to_le_bytes());
            if image.rgb.is_empty() {
                hash_floats(&mut hash, &image.pixels);
            } else {
                hash.update(b"rgb8");
                hash.update(&image.rgb);
            }
            media.push((start, hash.finalize().into()));
            cursor = start
                .checked_add(v.feature_count(image)?)
                .ok_or("Image extent overflow")?;
            if cursor > input.len() {
                return Err("Image span outside prompt".into());
            }
        }
        cursor = 0;
        for audio in audios {
            let a = self
                .manifest
                .audio
                .as_ref()
                .ok_or("Model has no audio adapter")?;
            let features = audio.features()?;
            let start = input
                .iter()
                .skip(cursor)
                .position(|&t| t == a.token_id)
                .map(|i| i + cursor)
                .ok_or("Missing audio marker")?;
            let mut hash = Sha256::new();
            hash.update((audio.frames as u64).to_le_bytes());
            hash_floats(&mut hash, &audio.mel);
            media.push((start, hash.finalize().into()));
            cursor = start
                .checked_add(features)
                .filter(|&n| n <= input.len())
                .ok_or("Audio span outside prompt")?;
        }
        media.sort_by_key(|item| item.0);
        Ok(Media(media))
    }
    pub(crate) fn prefix_match_tokens(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<usize> {
        self.prefix_match_cost(input, images)
            .map(|(tokens, _)| tokens)
    }
    pub(crate) fn prefix_match_cost(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<(usize, usize)> {
        if !self.prefix_enabled() {
            return Ok((0, 0));
        }
        let media = self.prefix_media(input, images, &[])?;
        Ok(self.prefix_match_media(input, &media))
    }
    pub(super) fn prefix_match_media(&self, input: &[u32], media: &Media) -> (usize, usize) {
        if !self.prefix_enabled() {
            return (0, 0);
        }
        self.prefix_hit(input, media)
            .map(|(host, id)| {
                if host {
                    let e = &self.host_prefix_cache.entries[&id];
                    (e.tokens.len(), e.bytes)
                } else {
                    let e = &self.prefix_cache.entries[&id];
                    (e.tokens.len(), e.bytes)
                }
            })
            .unwrap_or((0, 0))
    }
    pub(crate) fn prefix_enabled(&self) -> bool {
        self.prefix_cache.budget > 0 || self.host_prefix_cache.budget > 0
    }
    fn prefix_hit(&self, input: &[u32], media: &Media) -> Option<(bool, u64)> {
        crate::prefix::match_tiers(
            &self.prefix_cache,
            &self.host_prefix_cache,
            input,
            media,
            |e| {
                self.prefill_costs
                    .score(e.tokens.len(), input.len(), e.bytes)
            },
            |e| {
                self.prefill_costs
                    .score(e.tokens.len(), input.len(), e.bytes)
            },
        )
    }
    pub(super) fn restore_prefix(
        &mut self,
        input: &[u32],
        media: &Media,
        context: &mut Context,
    ) -> Result<(usize, super::speculation::PrefillWarm)> {
        context.statistics = crate::prefix::Statistics::default();
        context.parent = None;
        self.execution.collect_snapshots()?;
        let at = Instant::now();
        let hit = self.prefix_hit(input, media);
        let device_match = self
            .prefix_cache
            .match_prefix_by(input, media, |_| None)
            .matched_tokens;
        let host_match = self
            .host_prefix_cache
            .match_prefix_by(input, media, |_| None)
            .matched_tokens;
        context.statistics.matched_tokens = device_match.max(host_match);
        context.statistics.lookup_s = at.elapsed().as_secs_f64();
        let Some((host, id)) = hit else {
            return Ok((0, super::speculation::PrefillWarm::default()));
        };
        let at = Instant::now();
        let (tokens, warm, bytes) = if host {
            let e = self.host_prefix_cache.touch(id);
            self.execution.restore_host_snapshot(&e.snapshot)?;
            context.statistics.host_hits += 1;
            (e.tokens.len(), e.warm_tokens, e.bytes)
        } else {
            let e = self.prefix_cache.touch(id);
            self.execution.restore_snapshot(&e.snapshot)?;
            (e.tokens.len(), e.warm_tokens, e.bytes)
        };
        context.parent = Some((host, id));
        context.statistics.cached_tokens = tokens;
        context.statistics.restore_s = at.elapsed().as_secs_f64();
        self.prefill_costs
            .observe_restore(bytes, context.statistics.restore_s);
        Ok((
            tokens,
            super::speculation::PrefillWarm {
                tokens: warm,
                seconds: 0.0,
            },
        ))
    }
    pub(super) fn admit_prefix_checkpoint(
        &self,
        prefix: usize,
        end: usize,
        start: usize,
        checkpoints: &BTreeSet<usize>,
        reused: bool,
    ) -> Result<bool> {
        let bytes = self.prefix_ranges(prefix)?.values().map(|r| r.bytes).sum();
        Ok(
            bytes <= self.prefix_cache.budget.max(self.host_prefix_cache.budget)
                && self.prefill_costs.admit_with_reuse(
                    prefix,
                    end,
                    bytes,
                    start,
                    checkpoints,
                    reused,
                ),
        )
    }
    pub(super) fn prefix_ranges(&self, tokens: usize) -> Result<BTreeMap<String, Range>> {
        let m = &self.manifest;
        let mut names: BTreeSet<String> = m.reset_buffers.iter().cloned().collect();
        names.extend([
            m.position.clone(),
            m.token.clone(),
            m.status.clone(),
            m.logits.clone(),
        ]);
        let mut mtp_kv = BTreeSet::new();
        if let Some(kv) = &m.kv_cache {
            for g in kv.growth.values().filter(|g| g.position != m.position) {
                mtp_kv.extend(g.buffers.iter().cloned());
            }
        }
        if let Some(spec) = &m.mtp {
            names.extend([spec.position.clone(), spec.target_length.clone()]);
            if let Some(ring) = &spec.hidden_ring {
                names.insert(ring.clone());
            }
            if let Some(kv) = &m.kv_cache {
                for growth in kv.growth.values().filter(|g| g.position == spec.position) {
                    mtp_kv.extend(growth.buffers.iter().cloned());
                }
            }
        }
        let mut ranges = BTreeMap::new();
        for name in names {
            if m.mtp.is_none() && mtp_kv.contains(&name) {
                continue;
            }
            let b = m
                .buffers
                .iter()
                .find(|b| b.name == name)
                .ok_or("Missing prefix state buffer")?;
            let range = if let Some(stride) = m.kv_cache.as_ref().and_then(|k| k.buffers.get(&name))
            {
                let count = if mtp_kv.contains(&name) {
                    tokens - 1
                } else {
                    tokens
                };
                let divisor = m
                    .kv_cache
                    .as_ref()
                    .and_then(|k| k.prefix_divisors.get(&name))
                    .copied()
                    .unwrap_or(1);
                Range {
                    offset: 0,
                    bytes: (count / divisor * divisor)
                        .checked_mul(*stride)
                        .ok_or("Prefix KV extent overflow")?,
                }
            } else if m.mtp.as_ref().and_then(|s| s.hidden_ring.as_ref()) == Some(&name) {
                // A warmed checkpoint only needs h[P-1] for the next bridge.
                let rows = b.shape[0];
                let row = b.bytes()? / rows;
                Range {
                    offset: (tokens - 1) % rows * row,
                    bytes: row,
                }
            } else {
                Range {
                    offset: 0,
                    bytes: b.bytes()?,
                }
            };
            ranges.insert(name, range);
        }
        Ok(ranges)
    }
    pub(super) fn store_prefix(
        &mut self,
        input: &[u32],
        media: &Media,
        warm_tokens: usize,
        logits_valid: bool,
        context: &mut Context,
    ) -> Result<()> {
        let at = Instant::now();
        if self.prefix_enabled()
            && self.manifest.mtp.is_some()
            && (input.is_empty() || warm_tokens != input.len() - 1)
        {
            return Err("Prefix checkpoint must retain exactly P-1 MTP inputs".into());
        }
        let same = self.prefix_cache.entries.values().any(|e| {
            e.tokens == input
                && e.media.common_tokens(media, input.len()) == input.len()
                && (!logits_valid || e.logits_valid)
        }) || self.host_prefix_cache.entries.values().any(|e| {
            e.tokens == input
                && e.media.common_tokens(media, input.len()) == input.len()
                && (!logits_valid || e.logits_valid)
        });
        if self.prefix_enabled()
            && !same
            && self
                .manifest
                .kv_cache
                .as_ref()
                .is_some_and(|k| k.direct_prefill)
            && self
                .manifest
                .mtp
                .as_ref()
                .is_none_or(|s| s.hidden_ring.is_some())
        {
            let ranges = self.prefix_ranges(input.len())?;
            let minimum = ranges.values().try_fold(0usize, |n, r| {
                n.checked_add(r.bytes).ok_or("Prefix size overflow")
            })?;
            if minimum > self.prefix_cache.budget.max(self.host_prefix_cache.budget) {
                context.statistics.store_s += at.elapsed().as_secs_f64();
                self.update_prefix_usage(context);
                return Ok(());
            }
            let kv_names = self
                .manifest
                .kv_cache
                .as_ref()
                .map(|k| k.buffers.keys().cloned().collect())
                .unwrap_or_default();
            let parent = context
                .parent
                .filter(|(host, _)| !host)
                .and_then(|(_, id)| self.prefix_cache.entries.get(&id))
                .filter(|e| {
                    input.starts_with(&e.tokens)
                        && e.media.common_tokens(media, e.tokens.len()) == e.tokens.len()
                })
                .map(|e| &e.snapshot);
            let plan = self.execution.plan_snapshot(ranges, &kv_names, parent)?;
            let protected = (!logits_valid)
                .then(|| {
                    self.prefix_cache
                        .match_prefix_by(input, media, |e| {
                            e.logits_valid.then_some(-(e.tokens.len() as f64))
                        })
                        .checkpoint
                })
                .flatten();
            let mut retained = BTreeMap::new();
            plan.allocations(&mut retained);
            if let Some(entry) = protected.and_then(|id| self.prefix_cache.entries.get(&id)) {
                entry.snapshot.allocations(&mut retained);
            }
            if retained.values().sum::<usize>() + plan.new_bytes > self.prefix_cache.budget {
                self.store_host_prefix(input, media, warm_tokens, logits_valid, &plan, context)?;
                context.statistics.store_s += at.elapsed().as_secs_f64();
                self.update_prefix_usage(context);
                return Ok(());
            }
            // Reclaim before allocating so capture itself respects the budget,
            // including shared blocks pinned by the pending checkpoint.
            while self.prefix_cache.resident_with(Some(&plan)) + plan.new_bytes
                > self.prefix_cache.budget
            {
                if !self.evict_prefix(protected, Some(context))? {
                    return Err("Prefix budget planning failure".into());
                }
            }
            let snapshot = loop {
                if let Some(s) = self.execution.snapshot(&plan)? {
                    break Some(s);
                }
                if !self.evict_prefix(protected, Some(context))? {
                    break None;
                }
            };
            if let Some(snapshot) = snapshot {
                let mut own = BTreeMap::new();
                snapshot.allocations(&mut own);
                if own.values().sum::<usize>() <= self.prefix_cache.budget {
                    while self.prefix_cache.resident_with(Some(&snapshot))
                        > self.prefix_cache.budget
                    {
                        if !self.evict_prefix(protected, Some(context))? {
                            return Err("Prefix budget accounting failure".into());
                        }
                    }
                    // Upgrade a head-less endpoint, rather than keeping two states.
                    let obsolete: Vec<_> = self
                        .prefix_cache
                        .entries
                        .iter()
                        .filter(|(_, e)| {
                            e.tokens == input
                                && e.media.common_tokens(media, input.len()) == input.len()
                        })
                        .map(|(&id, _)| id)
                        .collect();
                    for id in obsolete {
                        let e = self.prefix_cache.remove(id).expect("known endpoint");
                        self.execution.release_snapshot(e.snapshot)?;
                    }
                    let bytes = snapshot.bytes;
                    self.remove_host_endpoint(input, media);
                    let id = self.prefix_cache.insert(Entry::new(
                        input.to_vec(),
                        media.clone(),
                        snapshot,
                        bytes,
                        warm_tokens,
                        logits_valid,
                    ));
                    context.parent = Some((false, id));
                    context.statistics.checkpoints_stored += 1;
                } else {
                    self.execution.release_snapshot(snapshot)?;
                }
            } else {
                self.store_host_prefix(input, media, warm_tokens, logits_valid, &plan, context)?;
            }
        }
        context.statistics.store_s += at.elapsed().as_secs_f64();
        self.update_prefix_usage(context);
        Ok(())
    }
    fn reserve_host_prefix(
        &mut self,
        banks: BTreeMap<u64, usize>,
        extra: usize,
        metadata: usize,
        context: &mut Context,
    ) -> bool {
        let plan = crate::backend::kv::host::Budget(banks);
        let total = |base: usize| {
            base.checked_add(extra)
                .and_then(|n| n.checked_add(metadata))
        };
        let mut own = BTreeMap::new();
        plan.allocations(&mut own);
        if total(own.values().sum()).is_none_or(|n| n > self.host_prefix_cache.budget) {
            return false;
        }
        while total(self.host_prefix_cache.resident_with(Some(&plan)))
            .is_none_or(|n| n > self.host_prefix_cache.budget)
        {
            if self.host_prefix_cache.evict_one().is_none() {
                return false;
            }
            context.statistics.host_evictions += 1;
        }
        true
    }
    fn host_metadata(tokens: usize, banks: usize) -> Result<usize> {
        tokens
            .checked_mul(16)
            .and_then(|n| banks.checked_mul(1024).and_then(|b| n.checked_add(b)))
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| "Host cache metadata overflow".into())
    }
    fn store_host_prefix(
        &mut self,
        input: &[u32],
        media: &Media,
        warm: usize,
        logits: bool,
        plan: &crate::backend::snapshot::Plan,
        context: &mut Context,
    ) -> Result<()> {
        if self.host_prefix_cache.budget == 0 {
            return Ok(());
        }
        let parent = context
            .parent
            .filter(|(host, _)| *host)
            .and_then(|(_, id)| self.host_prefix_cache.entries.get(&id))
            .filter(|e| {
                input.starts_with(&e.tokens)
                    && e.media.common_tokens(media, e.tokens.len()) == e.tokens.len()
            })
            .map(|e| &e.snapshot);
        let plan = plan.for_host_parent(parent)?;
        let banks = plan.host_banks();
        let metadata =
            Self::host_metadata(input.len(), banks.len().max(self.manifest.buffers.len()))?;
        if !self.reserve_host_prefix(banks, plan.new_bytes, metadata, context) {
            return Ok(());
        }
        let mut snapshot = self
            .execution
            .host_snapshot(&plan, &mut self.host_kv_pool)?;
        snapshot.metadata_bytes = metadata;
        let bytes = snapshot.bytes;
        self.remove_host_endpoint(input, media);
        let id = self.host_prefix_cache.insert(Entry::new(
            input.to_vec(),
            media.clone(),
            snapshot,
            bytes,
            warm,
            logits,
        ));
        context.parent = Some((true, id));
        context.statistics.host_stores += 1;
        context.statistics.checkpoints_stored += 1;
        Ok(())
    }
    fn remove_host_endpoint(&mut self, input: &[u32], media: &Media) {
        let ids: Vec<_> = self
            .host_prefix_cache
            .entries
            .iter()
            .filter(|(_, e)| {
                e.tokens == input && e.media.common_tokens(media, input.len()) == input.len()
            })
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            self.host_prefix_cache.remove(id);
        }
    }
    pub(super) fn evict_prefix(
        &mut self,
        protected: Option<u64>,
        context: Option<&mut Context>,
    ) -> Result<bool> {
        let Some(entry) = self.prefix_cache.evict_entry_one_except(protected) else {
            return Ok(false);
        };
        let mut local = Context::default();
        let context = context.unwrap_or(&mut local);
        if self.host_prefix_cache.budget > 0 {
            let banks = entry.snapshot.host_banks();
            let metadata = Self::host_metadata(
                entry.tokens.len(),
                banks.len().max(self.manifest.buffers.len()),
            )?;
            if self.reserve_host_prefix(banks, 0, metadata, context) {
                let mut snapshot = self
                    .execution
                    .offload_snapshot(&entry.snapshot, &mut self.host_kv_pool)?;
                snapshot.metadata_bytes = metadata;
                self.remove_host_endpoint(&entry.tokens, &entry.media);
                self.host_prefix_cache.insert(Entry::new(
                    entry.tokens,
                    entry.media,
                    snapshot,
                    entry.bytes,
                    entry.warm_tokens,
                    entry.logits_valid,
                ));
                context.statistics.host_stores += 1;
            }
        }
        self.execution.release_snapshot(entry.snapshot)?;
        context.statistics.evictions += 1;
        Ok(true)
    }
    fn update_prefix_usage(&self, context: &mut Context) {
        self.prefix_cache.update_statistics(&mut context.statistics);
        context.statistics.host_resident_bytes = self.host_prefix_cache.bytes;
        context.statistics.host_entries = self.host_prefix_cache.entries.len();
    }
    pub(super) fn store_decoded_prefix(
        &mut self,
        history: &[u32],
        media: &Media,
        cancelled: &impl Fn() -> bool,
        context: &mut Context,
    ) -> Result<()> {
        let position = self.read_control(&self.manifest.position)? as usize;
        if self.prefix_enabled() && !cancelled() && position != 0 && position <= history.len() {
            let spec = self.manifest.mtp.clone();
            let warm = if let Some(s) = &spec {
                // The draft has already consumed the pending token's embedding.
                // Cache only the prefix-consistent P-1 range; preserve the live cursor.
                let old = self.read_control(&s.position)? as usize;
                if old != position {
                    return Err("Decoded MTP cache position mismatch".into());
                }
                self.upload_ids(&s.position, &[(position - 1) as u32])?;
                position - 1
            } else {
                0
            };
            let result = self.store_prefix(&history[..position], media, warm, false, context);
            if let Some(s) = &spec {
                self.upload_ids(&s.position, &[position as u32])?;
            }
            result?;
        }
        self.execution.collect_snapshots()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_hash_preserves_little_endian_float_bits_and_tails() {
        let values: Vec<_> = (0..1025u32)
            .map(|i| f32::from_bits(i.wrapping_mul(1664525)))
            .chain([0., -0., f32::INFINITY, f32::from_bits(0x7fc00001)])
            .collect();
        for n in [0, 1, 255, 256, 257, 1025, values.len()] {
            let mut expected = Sha256::new();
            for value in &values[..n] {
                expected.update(value.to_bits().to_le_bytes());
            }
            let mut actual = Sha256::new();
            hash_floats(&mut actual, &values[..n]);
            assert_eq!(actual.finalize(), expected.finalize());
        }
    }
}
