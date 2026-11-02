//! A second radix cache for complete CPU state endpoints, under a global budget.
use super::*;
type Hit = (bool, u64);
impl Runtime {
    pub(super) fn matched(&self, input: &[u32]) -> Option<Hit> {
        matched_tier(&self.cache, input, self.final_decoder_window())
    }
    pub(super) fn entry(&self, hit: Hit) -> &Entry<Snapshot> {
        if hit.0 {
            &self.cache.host.entries[&hit.1]
        } else {
            &self.cache.device.entries[&hit.1]
        }
    }
    /// Conservative per-rank reservation covers histories, names, pieces, radix
    /// metadata and CPU heads. Payload capacity is accounted separately by bank.
    pub(super) fn host_overhead(&self, tokens: usize, banks: usize) -> Result<usize> {
        let contract = self
            .team
            .model
            .batch
            .as_ref()
            .ok_or("Missing A2 cache contract")?;
        let layouts = contract.state.iter().chain(
            contract
                .decoder_replay
                .iter()
                .flat_map(|r| &r.decoder_state),
        );
        let mut piece_bytes = 0usize;
        for layout in layouts {
            piece_bytes = piece_bytes
                .checked_add(512 + layout.buffer.len() * 2)
                .ok_or("Host metadata size overflow")?;
        }
        piece_bytes
            .checked_mul(banks.checked_add(1).ok_or("Host bank count overflow")?)
            .and_then(|n| tokens.checked_mul(16).and_then(|t| n.checked_add(t)))
            .and_then(|n| n.checked_add(8192))
            .ok_or_else(|| "Host metadata size overflow".into())
    }
    pub(super) fn evict_host(&mut self) -> Result<bool> {
        if let Some(snapshot) = self.cache.host.evict_one() {
            self.team.exchange(Command::Evict(snapshot.id))?;
            self.cache_reclamation.host += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub(super) fn evict(&mut self) -> Result<bool> {
        let Some(key) = self.cache.device.victim_except(None) else {
            return Ok(false);
        };
        let entry = &self.cache.device.entries[&key];
        let id = entry.snapshot.id;
        let mut host_snapshot = None;
        if self.cache.host.budget > 0 {
            let banks = self.team.offload_allocations(id)?;
            let snapshot = Snapshot {
                id,
                host_overhead: self.host_overhead(entry.tokens.len(), banks.len())?,
                banks,
                output: entry.snapshot.output.clone(),
            };
            if self.cache.host.admission(&snapshot, 0) != Admission::TooLarge {
                while self.cache.host.admission(&snapshot, 0) == Admission::Reclaim {
                    if !self.evict_host()? {
                        break;
                    }
                }
                if self.cache.host.admission(&snapshot, 0) == Admission::Ready {
                    self.team.exchange(Command::Offload(id))?;
                    host_snapshot = Some(snapshot);
                }
            }
        }
        if host_snapshot.is_none() {
            self.team.exchange(Command::Evict(id))?;
        }
        let entry = self
            .cache
            .device
            .remove(key)
            .ok_or("Missing device prefix")?;
        if let Some(snapshot) = host_snapshot {
            let (mut entry, _) = entry.replace_snapshot(snapshot);
            let mut allocations = BTreeMap::new();
            entry.snapshot.allocations(&mut allocations);
            entry.bytes = allocations.values().sum();
            self.cache.host.insert(entry);
            self.cache_reclamation.demotions += 1;
        }
        self.cache_reclamation.device += 1;
        self.cache
            .update_statistics(self.team.ranks(), &mut self.prefix);
        Ok(true)
    }
}
fn matched_tier(cache: &Tiered<Snapshot, Snapshot>, input: &[u32], reserve: usize) -> Option<Hit> {
    let score = |entry: &Entry<Snapshot>| {
        (entry.tokens.len() <= input.len().saturating_sub(reserve)
            || entry.tokens.len() == input.len() && entry.snapshot.output.is_some())
        .then_some(-(entry.tokens.len() as f64))
    };
    cache.matched(input, &Media::default(), score, score)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn entry(tokens: &[u32], id: u64) -> Entry<Snapshot> {
        Entry::new(
            tokens.to_vec(),
            Media::default(),
            Snapshot {
                id,
                banks: [(id, 64)].into(),
                output: None,
                host_overhead: 16,
            },
            80,
            0,
            false,
        )
    }
    #[test]
    fn tier_lookup_preserves_decoder_boundary_and_device_ties() {
        let mut cache = Tiered::new(256, 256);
        cache.device.insert(entry(&[1, 2], 1));
        cache.host.insert(entry(&[1, 2, 3], 2));
        assert_eq!(matched_tier(&cache, &[1, 2, 3, 4, 5], 2), Some((true, 1)));
        assert_eq!(matched_tier(&cache, &[1, 2, 3, 4], 2), Some((false, 1)));
        cache.device.insert(entry(&[1, 2, 3], 3));
        assert_eq!(matched_tier(&cache, &[1, 2, 3, 4, 5], 2), Some((false, 2)));
        let moved = cache.device.evict_entry_one_except(Some(1)).unwrap();
        cache.host.insert(moved);
        let mut stats = crate::prefix::Statistics::default();
        cache.update_statistics(8, &mut stats);
        assert_eq!(stats.resident_bytes, 80);
        assert_eq!(stats.host_resident_bytes, 160 * 8);
        assert_eq!(stats.host_entries, 2);
        assert!(cache.host.bytes * 8 <= cache.host.budget * 8);
    }
}
