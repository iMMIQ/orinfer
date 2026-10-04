use super::*;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn media_identity(images: &[crate::vision::ImageInput]) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update((images.len() as u64).to_le_bytes());
    for image in images {
        hash.update((image.grid_height as u64).to_le_bytes());
        hash.update((image.grid_width as u64).to_le_bytes());
        for pixel in &image.pixels {
            hash.update(pixel.to_bits().to_le_bytes());
        }
    }
    hash.finalize().into()
}

impl ModelRuntime {
    pub(super) fn restore_prefix(
        &mut self,
        input: &[u32],
        media: &[u8; 32],
    ) -> Result<(usize, super::speculation::PrefillWarm)> {
        self.prefix_statistics = crate::prefix::Statistics::default();
        if self.prefix_cache.budget == 0 {
            return Ok((0, super::speculation::PrefillWarm::default()));
        }
        let at = Instant::now();
        let index = self.prefix_cache.find(input, media);
        self.prefix_statistics.lookup_s = at.elapsed().as_secs_f64();
        let Some(index) = index else {
            return Ok((0, super::speculation::PrefillWarm::default()));
        };
        let entry = self.prefix_cache.touch(index);
        let at = Instant::now();
        self.execution.restore_snapshot(&entry.snapshot)?;
        self.prefix_statistics.cached_tokens = entry.tokens.len();
        self.prefix_statistics.restore_s = at.elapsed().as_secs_f64();
        Ok((
            entry.tokens.len(),
            super::speculation::PrefillWarm {
                tokens: entry.warm_tokens,
                seconds: 0.0,
            },
        ))
    }

    pub(super) fn prefix_ranges(&self, tokens: usize) -> Result<BTreeMap<String, usize>> {
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
            let bytes = if let Some(stride) = m.kv_cache.as_ref().and_then(|k| k.buffers.get(&name))
            {
                let count = if mtp_kv.contains(&name) {
                    tokens - 1
                } else {
                    tokens
                };
                count
                    .checked_mul(*stride)
                    .ok_or("Prefix KV extent overflow")?
            } else {
                b.bytes()?
            };
            ranges.insert(name, bytes);
        }
        Ok(ranges)
    }

    pub(super) fn store_prefix(
        &mut self,
        input: &[u32],
        media: [u8; 32],
        warm_tokens: usize,
    ) -> Result<()> {
        let at = Instant::now();
        let same = self
            .prefix_cache
            .find(input, &media)
            .is_some_and(|i| self.prefix_cache.entries[i].tokens.len() == input.len());
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
            let bytes = Executor::snapshot_size(&ranges)?;
            if bytes <= self.prefix_cache.budget {
                for old in self.prefix_cache.evict_for(bytes) {
                    self.execution.release_snapshot(old)?;
                }
                if let Some(snapshot) = self.execution.snapshot(ranges)? {
                    self.prefix_cache.insert(crate::prefix::Entry {
                        tokens: input.to_vec(),
                        media,
                        bytes: snapshot.bytes,
                        snapshot,
                        warm_tokens,
                    });
                }
            }
        }
        self.prefix_statistics.store_s += at.elapsed().as_secs_f64();
        self.prefix_statistics.resident_bytes = self.prefix_cache.bytes;
        self.prefix_statistics.entries = self.prefix_cache.entries.len();
        Ok(())
    }
}
