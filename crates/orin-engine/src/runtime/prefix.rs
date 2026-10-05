use super::*;
use crate::cuda::snapshot::Range;
use crate::prefix::{Entry, Media, Resident};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

impl ModelRuntime {
    pub(crate) fn prefix_media(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
    ) -> Result<Media> {
        let Some(v) = &self.manifest.vision else {
            return if images.is_empty() {
                Ok(Media::default())
            } else {
                Err("Model has no vision adapter".into())
            };
        };
        let mut media = vec![];
        let mut cursor = 0;
        for image in images {
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
            for pixel in &image.pixels {
                hash.update(pixel.to_bits().to_le_bytes());
            }
            media.push((start, hash.finalize().into()));
            cursor = start
                .checked_add(v.feature_count(image)?)
                .ok_or("Image extent overflow")?;
            if cursor > input.len() {
                return Err("Image span outside prompt".into());
            }
        }
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
        if self.prefix_cache.budget == 0 {
            return Ok((0, 0));
        }
        let media = self.prefix_media(input, images)?;
        Ok(self
            .prefix_cache
            .match_prefix_by(input, &media, |e| {
                self.prefill_costs
                    .score(e.tokens.len(), input.len(), e.bytes)
            })
            .checkpoint
            .map(|id| {
                let entry = &self.prefix_cache.entries[&id];
                (entry.tokens.len(), entry.bytes)
            })
            .unwrap_or((0, 0)))
    }
    pub(super) fn restore_prefix(
        &mut self,
        input: &[u32],
        media: &Media,
    ) -> Result<(usize, super::speculation::PrefillWarm)> {
        self.prefix_statistics = crate::prefix::Statistics::default();
        self.prefix_kv.clear();
        self.execution.collect_snapshots()?;
        if self.prefix_cache.budget == 0 {
            return Ok((0, super::speculation::PrefillWarm::default()));
        }
        let at = Instant::now();
        let matched = self.prefix_cache.match_prefix_by(input, media, |e| {
            self.prefill_costs
                .score(e.tokens.len(), input.len(), e.bytes)
        });
        self.prefix_statistics.lookup_s = at.elapsed().as_secs_f64();
        self.prefix_statistics.matched_tokens = matched.matched_tokens;
        let Some(id) = matched.checkpoint else {
            return Ok((0, super::speculation::PrefillWarm::default()));
        };
        let entry = self.prefix_cache.touch(id);
        let at = Instant::now();
        self.execution.restore_snapshot(&entry.snapshot)?;
        self.prefix_kv.clone_from(&entry.snapshot.kv);
        self.prefix_statistics.cached_tokens = entry.tokens.len();
        self.prefix_statistics.restore_s = at.elapsed().as_secs_f64();
        self.prefill_costs
            .observe_restore(entry.bytes, self.prefix_statistics.restore_s);
        Ok((
            entry.tokens.len(),
            super::speculation::PrefillWarm {
                tokens: entry.warm_tokens,
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
    ) -> Result<bool> {
        let bytes = self.prefix_ranges(prefix)?.values().map(|r| r.bytes).sum();
        Ok(self
            .prefill_costs
            .admit(prefix, end, bytes, start, checkpoints))
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
                Range {
                    offset: 0,
                    bytes: count
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
    ) -> Result<()> {
        let at = Instant::now();
        if self.prefix_cache.budget != 0
            && self.manifest.mtp.is_some()
            && (input.is_empty() || warm_tokens != input.len() - 1)
        {
            return Err("Prefix checkpoint must retain exactly P-1 MTP inputs".into());
        }
        let same = self.prefix_cache.entries.values().any(|e| {
            e.tokens == input
                && e.media.common_tokens(media, input.len()) == input.len()
                && (!logits_valid || e.logits_valid)
        });
        if self.prefix_cache.budget != 0
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
            if minimum > self.prefix_cache.budget {
                self.prefix_statistics.store_s += at.elapsed().as_secs_f64();
                self.update_prefix_usage();
                return Ok(());
            }
            let kv_names = self
                .manifest
                .kv_cache
                .as_ref()
                .map(|k| k.buffers.keys().cloned().collect())
                .unwrap_or_default();
            let plan = self
                .execution
                .plan_snapshot(ranges, &kv_names, &self.prefix_kv)?;
            let mut retained = BTreeMap::new();
            plan.allocations(&mut retained);
            if retained.values().sum::<usize>() + plan.new_bytes > self.prefix_cache.budget {
                self.prefix_statistics.store_s += at.elapsed().as_secs_f64();
                self.update_prefix_usage();
                return Ok(());
            }
            // Reclaim before allocating so capture itself respects the budget,
            // including shared blocks pinned by the pending checkpoint.
            while self.prefix_cache.resident_with(Some(&plan)) + plan.new_bytes
                > self.prefix_cache.budget
            {
                let old = self
                    .prefix_cache
                    .evict_one()
                    .ok_or("Prefix budget planning failure")?;
                self.prefix_statistics.evictions += 1;
                self.execution.release_snapshot(old)?;
            }
            let snapshot = loop {
                if let Some(s) = self.execution.snapshot(&plan)? {
                    break Some(s);
                }
                let Some(old) = self.prefix_cache.evict_one() else {
                    break None;
                };
                self.prefix_statistics.evictions += 1;
                self.execution.release_snapshot(old)?;
            };
            if let Some(snapshot) = snapshot {
                let mut own = BTreeMap::new();
                snapshot.allocations(&mut own);
                if own.values().sum::<usize>() <= self.prefix_cache.budget {
                    while self.prefix_cache.resident_with(Some(&snapshot))
                        > self.prefix_cache.budget
                    {
                        let old = self
                            .prefix_cache
                            .evict_one()
                            .ok_or("Prefix budget accounting failure")?;
                        self.prefix_statistics.evictions += 1;
                        self.execution.release_snapshot(old)?;
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
                    self.prefix_kv.clone_from(&snapshot.kv);
                    let bytes = snapshot.bytes;
                    self.prefix_cache.insert(Entry::new(
                        input.to_vec(),
                        media.clone(),
                        snapshot,
                        bytes,
                        warm_tokens,
                        logits_valid,
                    ));
                    self.prefix_statistics.checkpoints_stored += 1;
                } else {
                    self.execution.release_snapshot(snapshot)?;
                }
            }
        }
        self.prefix_statistics.store_s += at.elapsed().as_secs_f64();
        self.update_prefix_usage();
        Ok(())
    }
    fn update_prefix_usage(&mut self) {
        self.prefix_statistics.resident_bytes = self.prefix_cache.bytes;
        self.prefix_statistics.logical_bytes =
            self.prefix_cache.entries.values().map(|e| e.bytes).sum();
        self.prefix_statistics.shared_bytes = self
            .prefix_statistics
            .logical_bytes
            .saturating_sub(self.prefix_cache.bytes);
        self.prefix_statistics.entries = self.prefix_cache.entries.len();
    }
    pub(super) fn store_decoded_prefix(
        &mut self,
        history: &[u32],
        media: &Media,
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let position = self.read_control(&self.manifest.position)? as usize;
        if self.prefix_cache.budget != 0
            && !cancelled()
            && position != 0
            && position <= history.len()
        {
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
            let result = self.store_prefix(&history[..position], media, warm, false);
            if let Some(s) = &spec {
                self.upload_ids(&s.position, &[position as u32])?;
            }
            result?;
        }
        self.prefix_kv.clear();
        self.execution.collect_snapshots()
    }
}
