//! A second radix cache for complete CPU state endpoints, under a global budget.
use super::*;
type Hit = (bool, u64);
impl Runtime {
    pub(super) fn matched(&self, input: &[u32]) -> Option<Hit> {
        matched_tier(
            &self.cache,
            &self.host_cache,
            input,
            self.final_decoder_window(),
        )
    }
    pub(super) fn entry(&self, hit: Hit) -> &Entry<Snapshot> {
        if hit.0 {
            &self.host_cache.entries[&hit.1]
        } else {
            &self.cache.entries[&hit.1]
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
        if let Some(snapshot) = self.host_cache.evict_one() {
            self.team.exchange(Command::Evict(snapshot.id))?;
            self.prefix.host_evictions += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    pub(super) fn evict(&mut self) -> Result<bool> {
        let Some(mut entry) = self.cache.evict_entry_one_except(None) else {
            return Ok(false);
        };
        let id = entry.snapshot.id;
        let mut offload = false;
        if self.host_cache.budget > 0 {
            entry.snapshot.banks = self.team.offload_allocations(id)?;
            entry.snapshot.host_overhead =
                self.host_overhead(entry.tokens.len(), entry.snapshot.banks.len())?;
            let mut allocations = BTreeMap::new();
            entry.snapshot.allocations(&mut allocations);
            entry.bytes = allocations.values().sum();
            if entry.bytes <= self.host_cache.budget {
                while self.host_cache.resident_with(Some(&entry.snapshot)) > self.host_cache.budget
                {
                    if !self.evict_host()? {
                        break;
                    }
                }
                if self.host_cache.resident_with(Some(&entry.snapshot)) <= self.host_cache.budget {
                    self.team.exchange(Command::Offload(id))?;
                    self.host_cache.insert(entry);
                    self.prefix.host_stores += 1;
                    offload = true;
                }
            }
        }
        if !offload {
            self.team.exchange(Command::Evict(id))?;
        }
        self.prefix.evictions += 1;
        update_statistics(
            &self.cache,
            &self.host_cache,
            self.team.ranks(),
            &mut self.prefix,
        );
        Ok(true)
    }
}
fn matched_tier(
    device: &Cache<Snapshot>,
    host: &Cache<Snapshot>,
    input: &[u32],
    reserve: usize,
) -> Option<Hit> {
    let score = |entry: &Entry<Snapshot>| {
        (entry.tokens.len() <= input.len().saturating_sub(reserve)
            || entry.tokens.len() == input.len() && entry.snapshot.output.is_some())
        .then_some(-(entry.tokens.len() as f64))
    };
    crate::prefix::match_tiers(device, host, input, &Media::default(), score, score)
}
pub(super) fn update_statistics(
    device: &Cache<Snapshot>,
    host: &Cache<Snapshot>,
    ranks: usize,
    stats: &mut crate::prefix::Statistics,
) {
    device.update_statistics(stats);
    stats.host_resident_bytes = host.bytes.saturating_mul(ranks);
    stats.host_entries = host.entries.len();
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
        let mut device = Cache::new(256);
        let mut host = Cache::new(256);
        device.insert(entry(&[1, 2], 1));
        host.insert(entry(&[1, 2, 3], 2));
        assert_eq!(
            matched_tier(&device, &host, &[1, 2, 3, 4, 5], 2),
            Some((true, 1))
        );
        assert_eq!(
            matched_tier(&device, &host, &[1, 2, 3, 4], 2),
            Some((false, 1))
        );
        device.insert(entry(&[1, 2, 3], 3));
        assert_eq!(
            matched_tier(&device, &host, &[1, 2, 3, 4, 5], 2),
            Some((false, 2))
        );
        let moved = device.evict_entry_one_except(Some(1)).unwrap();
        host.insert(moved);
        let mut stats = crate::prefix::Statistics::default();
        update_statistics(&device, &host, 8, &mut stats);
        assert_eq!(stats.resident_bytes, 80);
        assert_eq!(stats.host_resident_bytes, 160 * 8);
        assert_eq!(stats.host_entries, 2);
        assert!(host.bytes * 8 <= host.budget * 8);
    }
}
