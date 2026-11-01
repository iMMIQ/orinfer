//! Device-independent tier ownership and admission. Native adapters move bytes.
use super::*;

pub(crate) struct Tiered<D, H> {
    pub device: Cache<D>,
    pub host: Cache<H>,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Admission {
    Ready,
    Reclaim,
    TooLarge,
}

impl<D: Resident, H: Resident> Tiered<D, H> {
    pub fn new(device_bytes: usize, host_bytes: usize) -> Self {
        Self {
            device: Cache::new(device_bytes),
            host: Cache::new(host_bytes),
        }
    }
    pub fn enabled(&self) -> bool {
        self.device.budget > 0 || self.host.budget > 0
    }
    pub fn matched(
        &self,
        tokens: &[u32],
        media: &Media,
        device_score: impl Fn(&Entry<D>) -> Option<f64>,
        host_score: impl Fn(&Entry<H>) -> Option<f64>,
    ) -> Option<(bool, u64)> {
        match_tiers(
            &self.device,
            &self.host,
            tokens,
            media,
            device_score,
            host_score,
        )
    }
    /// Device bytes are per worker; host bytes include all tensor-parallel ranks.
    pub fn update_statistics(&self, host_ranks: usize, stats: &mut Statistics) {
        self.device.update_statistics(stats);
        stats.host_resident_bytes = self.host.bytes.saturating_mul(host_ranks);
        stats.host_entries = self.host.entries.len();
    }
}

impl<T: Resident> Cache<T> {
    /// Assess a pending allocation plan before native allocation or transfer.
    /// `extra` covers unpublished payload and metadata. Shared physical banks
    /// are charged once; overflow and an individually oversized plan cannot
    /// trigger destructive eviction of the entire tier.
    pub fn admission(&self, incoming: &dyn Resident, extra: usize) -> Admission {
        let mut own = BTreeMap::new();
        incoming.allocations(&mut own);
        let fits = |banks: &BTreeMap<u64, usize>| {
            banks
                .values()
                .try_fold(extra, |n, bytes| n.checked_add(*bytes))
                .is_some_and(|n| n <= self.budget)
        };
        if !fits(&own) {
            return Admission::TooLarge;
        }
        for e in self.entries.values() {
            e.snapshot.allocations(&mut own);
        }
        if fits(&own) {
            Admission::Ready
        } else {
            Admission::Reclaim
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Banks(Vec<(u64, usize)>);
    impl Resident for Banks {
        fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
            out.extend(self.0.iter().copied());
        }
    }
    fn entry(tokens: &[u32], banks: &[(u64, usize)]) -> Entry<Banks> {
        Entry::new(
            tokens.to_vec(),
            Media::default(),
            Banks(banks.to_vec()),
            banks.iter().map(|(_, bytes)| bytes).sum(),
            0,
            true,
        )
    }
    #[test]
    fn tier_admission_counts_shared_banks_metadata_and_physical_granularity() {
        // CUDA and A2 use different physical grains, with identical policy.
        for grain in [64 << 10, 2 << 20] {
            let mut tiers = Tiered::new(2 * grain, 2 * grain);
            assert!(tiers.enabled());
            tiers.device.insert(entry(&[1, 2], &[(2, grain)]));
            let parent = tiers.host.insert(entry(&[1, 2], &[(2, grain)]));
            let incoming = Banks(vec![(2, grain), (4, grain - 1)]);
            assert_eq!(tiers.host.admission(&incoming, 1), Admission::Ready);
            assert_eq!(tiers.host.admission(&incoming, 2), Admission::TooLarge);
            assert!(tiers.host.entries.contains_key(&parent));
            let separate = Banks(vec![(6, grain)]);
            assert_eq!(tiers.host.admission(&separate, grain), Admission::Reclaim);
            assert_eq!(tiers.host.bytes, grain, "assessment does not evict");
            tiers
                .host
                .insert(entry(&[1, 2, 3], &[(2, grain), (4, grain)]));
            let score = |e: &Entry<Banks>| Some(-(e.tokens.len() as f64));
            assert_eq!(
                tiers.matched(&[1, 2, 3, 4], &Media::default(), score, score),
                Some((true, 2))
            );
            tiers
                .device
                .insert(entry(&[1, 2, 3], &[(2, grain), (4, grain)]));
            assert_eq!(
                tiers.matched(&[1, 2, 3, 4], &Media::default(), score, score),
                Some((false, 2))
            );
            let mut stats = Statistics::default();
            tiers.update_statistics(8, &mut stats);
            assert_eq!(stats.resident_bytes, 2 * grain);
            assert_eq!(stats.host_resident_bytes, 16 * grain);
        }
    }
    #[test]
    fn oversized_and_overflowing_plans_leave_existing_endpoints_resident() {
        let mut cache = Cache::new(usize::MAX);
        let id = cache.insert(entry(&[1], &[(2, 64)]));
        for (incoming, extra) in [
            (Banks(vec![(4, usize::MAX)]), 1),
            (Banks(vec![(4, usize::MAX), (6, 1)]), 0),
        ] {
            assert_eq!(cache.admission(&incoming, extra), Admission::TooLarge);
            assert_eq!(cache.entries.len(), 1);
            assert!(cache.entries.contains_key(&id));
        }
        // The incoming plan fits by itself, but the union requires reclamation.
        assert_eq!(
            cache.admission(&Banks(vec![(4, usize::MAX)]), 0),
            Admission::Reclaim
        );
    }
    #[test]
    fn demotion_preserves_endpoint_history_and_keeps_source_until_commit() {
        let mut tiers = Tiered::new(128, 128);
        let id = tiers.device.insert(entry(&[1, 2], &[(2, 64)]));
        tiers.device.touch(id);
        tiers.device.touch(id);
        let victim = tiers.device.victim_except(None).unwrap();
        assert_eq!(tiers.device.bytes, 64);
        assert_eq!(tiers.device.find(&[1, 2], &Media::default()), Some(id));
        assert!(tiers.device.victim_except(Some(id)).is_none());
        let source = tiers.device.remove(victim).unwrap();
        let (host, old) = source.replace_snapshot(Banks(vec![(2, 32)]));
        assert_eq!(old.0, vec![(2, 64)]);
        assert_eq!(host.hits, 2);
        assert!(host.logits_valid);
        let host_id = tiers.host.insert(host);
        assert_eq!(tiers.host.bytes, 32);
        assert_eq!(tiers.host.entries[&host_id].hits, 2);
        assert_eq!(tiers.device.bytes, 0);
    }
}
