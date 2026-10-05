//! Compressed token radix index with exact state endpoints and byte accounting.
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Default, Debug, Clone, Serialize)]
pub struct Statistics {
    pub cached_tokens: usize,
    pub matched_tokens: usize,
    pub lookup_s: f64,
    pub restore_s: f64,
    pub store_s: f64,
    pub resident_bytes: usize,
    pub logical_bytes: usize,
    pub shared_bytes: usize,
    pub entries: usize,
    pub evictions: usize,
    pub checkpoints_stored: usize,
}

/// Image identities are scoped to the first affected token, not the request.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Media(pub Vec<(usize, [u8; 32])>);
impl Media {
    pub fn common_tokens(&self, other: &Self, limit: usize) -> usize {
        let mut left = self.0.iter().filter(|(p, _)| *p < limit);
        let mut right = other.0.iter().filter(|(p, _)| *p < limit);
        loop {
            match (left.next(), right.next()) {
                (None, None) => return limit,
                (Some(a), Some(b)) if a == b => (),
                (Some(a), Some(b)) => return a.0.min(b.0),
                (Some(a), None) | (None, Some(a)) => return a.0,
            }
        }
    }
}

/// Account physical allocations once even when many endpoints share KV ranges.
pub(crate) trait Resident {
    fn allocations(&self, out: &mut BTreeMap<u64, usize>);
}
pub(crate) struct Entry<T> {
    pub tokens: Vec<u32>,
    pub media: Media,
    pub snapshot: T,
    pub bytes: usize,
    pub warm_tokens: usize,
    /// A decoded endpoint may have valid state but no matching target head.
    pub logits_valid: bool,
    used: u64,
    hits: u64,
}
impl<T> Entry<T> {
    pub fn new(
        tokens: Vec<u32>,
        media: Media,
        snapshot: T,
        bytes: usize,
        warm_tokens: usize,
        logits_valid: bool,
    ) -> Self {
        Self {
            tokens,
            media,
            snapshot,
            bytes,
            warm_tokens,
            logits_valid,
            used: 0,
            hits: 0,
        }
    }
}

#[derive(Default)]
struct Node {
    label: Vec<u32>,
    endpoints: Vec<u64>,
    children: BTreeMap<u32, Node>,
}
impl Node {
    fn insert(&mut self, tokens: &[u32], id: u64) {
        if tokens.is_empty() {
            self.endpoints.push(id);
            return;
        }
        let child = self.children.entry(tokens[0]).or_default();
        if child.label.is_empty() {
            child.label = tokens.to_vec();
            child.endpoints.push(id);
            return;
        }
        let common = child
            .label
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        if common < child.label.len() {
            let mut tail = std::mem::take(child);
            child.label = tail.label.drain(..common).collect();
            child.children.insert(tail.label[0], tail);
        }
        child.insert(&tokens[common..], id);
    }
    fn remove(&mut self, tokens: &[u32], id: u64) {
        if tokens.is_empty() {
            self.endpoints.retain(|&e| e != id);
            return;
        }
        let key = tokens[0];
        if let Some(child) = self.children.get_mut(&key) {
            child.remove(&tokens[child.label.len()..], id);
            if child.endpoints.is_empty() && child.children.len() == 1 {
                let (_, tail) = child.children.pop_first().expect("one child");
                child.label.extend(tail.label);
                child.endpoints = tail.endpoints;
                child.children = tail.children;
            }
            if child.endpoints.is_empty() && child.children.is_empty() {
                self.children.remove(&key);
            }
        }
    }
    fn descendants(&self, ids: &mut Vec<u64>) {
        ids.extend(&self.endpoints);
        for child in self.children.values() {
            child.descendants(ids);
        }
    }
}
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Match {
    pub checkpoint: Option<u64>,
    pub matched_tokens: usize,
}
pub(crate) struct Cache<T> {
    pub budget: usize,
    pub bytes: usize,
    pub entries: BTreeMap<u64, Entry<T>>,
    root: Node,
    clock: u64,
    next_id: u64,
}
impl<T: Resident> Cache<T> {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            entries: BTreeMap::new(),
            root: Node::default(),
            clock: 0,
            next_id: 0,
        }
    }
    pub fn match_prefix_by(
        &self,
        tokens: &[u32],
        media: &Media,
        score: impl Fn(&Entry<T>) -> Option<f64>,
    ) -> Match {
        let mut result = Match::default();
        let mut best = f64::INFINITY;
        let (mut node, mut depth) = (&self.root, 0);
        loop {
            for id in &node.endpoints {
                let e = &self.entries[id];
                let common = e.media.common_tokens(media, depth);
                result.matched_tokens = result.matched_tokens.max(common);
                if common == depth
                    && (depth < tokens.len() || e.logits_valid)
                    && let Some(cost) = score(e)
                    && cost <= best
                {
                    result.checkpoint = Some(*id);
                    best = cost;
                }
            }
            let child = tokens.get(depth).and_then(|id| node.children.get(id));
            // A deeper token match can fail on an image identity. Off-path
            // siblings may still provide a longer media-compatible overlap.
            for (key, sibling) in &node.children {
                if Some(key) == tokens.get(depth) {
                    continue;
                }
                let mut ids = vec![];
                sibling.descendants(&mut ids);
                for id in ids {
                    result.matched_tokens = result
                        .matched_tokens
                        .max(self.entries[&id].media.common_tokens(media, depth));
                }
            }
            if let Some(child) = child {
                let common = child
                    .label
                    .iter()
                    .zip(&tokens[depth..])
                    .take_while(|(a, b)| a == b)
                    .count();
                depth += common;
                if common == child.label.len() {
                    node = child;
                    continue;
                }
                node = child;
            }
            let mut descendants = vec![];
            node.descendants(&mut descendants);
            for id in descendants {
                result.matched_tokens = result
                    .matched_tokens
                    .max(self.entries[&id].media.common_tokens(media, depth));
            }
            return result;
        }
    }
    #[cfg(test)]
    pub fn match_prefix(&self, tokens: &[u32], media: &Media) -> Match {
        self.match_prefix_by(tokens, media, |e| Some(-(e.tokens.len() as f64)))
    }
    #[cfg(test)]
    pub fn find(&self, tokens: &[u32], media: &Media) -> Option<u64> {
        self.match_prefix(tokens, media).checkpoint
    }
    pub fn touch(&mut self, id: u64) -> &Entry<T> {
        self.clock += 1;
        let e = self.entries.get_mut(&id).expect("checked prefix id");
        e.used = self.clock;
        e.hits += 1;
        e
    }
    pub fn resident_with(&self, extra: Option<&dyn Resident>) -> usize {
        let mut allocations = BTreeMap::new();
        for e in self.entries.values() {
            e.snapshot.allocations(&mut allocations);
        }
        if let Some(extra) = extra {
            extra.allocations(&mut allocations);
        }
        allocations.values().sum()
    }
    pub fn remove(&mut self, id: u64) -> Option<Entry<T>> {
        let entry = self.entries.remove(&id)?;
        self.root.remove(&entry.tokens, id);
        self.bytes = self.resident_with(None);
        Some(entry)
    }
    fn ancestor_depth(&self, entry: &Entry<T>) -> usize {
        let (mut node, mut depth, mut best) = (&self.root, 0, 0);
        while let Some(child) = entry.tokens.get(depth).and_then(|id| node.children.get(id)) {
            depth += child.label.len();
            if depth >= entry.tokens.len() {
                break;
            }
            for id in &child.endpoints {
                if self.entries[id].media.common_tokens(&entry.media, depth) == depth {
                    best = depth;
                }
            }
            node = child;
        }
        best
    }
    /// SegLen-style utility: saved replay distance per reclaimable byte,
    /// discounted by age. Tombstoned parents are skipped when computing distance.
    pub fn evict_one(&mut self) -> Option<T> {
        let mut references: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        for e in self.entries.values() {
            let mut own = BTreeMap::new();
            e.snapshot.allocations(&mut own);
            for (id, bytes) in own {
                let r = references.entry(id).or_insert((bytes, 0));
                r.1 += 1;
            }
        }
        let victim = self
            .entries
            .iter()
            .min_by(|(_, a), (_, b)| {
                let utility = |e: &Entry<T>| {
                    let parent = self.ancestor_depth(e);
                    let mut own = BTreeMap::new();
                    e.snapshot.allocations(&mut own);
                    let freed: usize = own
                        .iter()
                        .filter(|(id, _)| references[id].1 == 1)
                        .map(|(_, n)| *n)
                        .sum();
                    (e.tokens.len() - parent).max(1) as f64 * (1.0 + e.hits as f64)
                        / (freed.max(1) as f64 * (self.clock - e.used + 1) as f64)
                };
                utility(a).total_cmp(&utility(b)).then(a.used.cmp(&b.used))
            })
            .map(|(&id, _)| id)?;
        Some(self.remove(victim).expect("selected victim").snapshot)
    }
    pub fn insert(&mut self, mut entry: Entry<T>) -> u64 {
        self.clock += 1;
        self.next_id += 1;
        entry.used = self.clock;
        self.root.insert(&entry.tokens, self.next_id);
        self.entries.insert(self.next_id, entry);
        self.bytes = self.resident_with(None);
        debug_assert!(self.bytes <= self.budget);
        self.next_id
    }
    #[cfg(test)]
    pub fn clear(&mut self) -> Vec<T> {
        self.root = Node::default();
        self.bytes = 0;
        std::mem::take(&mut self.entries)
            .into_values()
            .map(|e| e.snapshot)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Clone)]
    struct Blocks(Vec<(u64, usize)>);
    impl Resident for Blocks {
        fn allocations(&self, out: &mut BTreeMap<u64, usize>) {
            out.extend(self.0.iter().copied());
        }
    }
    fn entry(tokens: &[u32], blocks: &[(u64, usize)], valid: bool) -> Entry<Blocks> {
        Entry::new(
            tokens.to_vec(),
            Media::default(),
            Blocks(blocks.to_vec()),
            blocks.iter().map(|b| b.1).sum(),
            tokens.len() - 1,
            valid,
        )
    }
    #[test]
    fn selects_cheapest_compatible_endpoint_and_preserves_raw_overlap() {
        let mut c = Cache::new(100);
        let a = c.insert(entry(&[1, 2], &[(1, 10)], true));
        let b = c.insert(entry(&[1, 2, 3], &[(2, 10)], true));
        let query = [1, 2, 3, 4];
        assert_eq!(
            c.match_prefix_by(&query, &Media::default(), |e| {
                Some(if e.tokens.len() == 2 { 1. } else { 2. })
            }),
            Match {
                checkpoint: Some(a),
                matched_tokens: 3
            }
        );
        assert_eq!(
            c.match_prefix_by(&query, &Media::default(), |_| None),
            Match {
                checkpoint: None,
                matched_tokens: 3
            }
        );
        assert_eq!(c.find(&query, &Media::default()), Some(b));
    }
    #[test]
    fn splits_branches_and_never_manufactures_an_intermediate_state() {
        let mut c = Cache::new(1000);
        let a = c.insert(entry(&[1, 2, 3, 4], &[(1, 10)], true));
        assert_eq!(
            c.match_prefix(&[1, 2, 3, 9], &Media::default()),
            Match {
                checkpoint: None,
                matched_tokens: 3
            }
        );
        let b = c.insert(entry(&[1, 2], &[(2, 10)], true));
        let d = c.insert(entry(&[1, 2, 3, 9], &[(3, 10)], true));
        assert_eq!(c.find(&[1, 2, 3, 4, 5], &Media::default()), Some(a));
        assert_eq!(c.find(&[1, 2, 3, 8], &Media::default()), Some(b));
        c.remove(b);
        assert_eq!(c.find(&[1, 2, 3, 8], &Media::default()), None);
        assert_eq!(c.find(&[1, 2, 3, 9], &Media::default()), Some(d));
        c.remove(a);
        c.remove(d);
        assert_eq!(c.bytes, 0);
        assert!(c.root.children.is_empty());
    }
    #[test]
    fn image_changes_only_invalidate_the_affected_suffix() {
        let mut c = Cache::new(1000);
        let plain = c.insert(entry(&[1, 2], &[(1, 10)], true));
        let mut a = entry(&[1, 2, 3, 4, 5], &[(2, 10)], true);
        a.media = Media(vec![(2, [1; 32]), (4, [2; 32])]);
        c.insert(a);
        let changed = Media(vec![(2, [1; 32]), (4, [3; 32])]);
        assert_eq!(
            c.match_prefix(&[1, 2, 3, 4, 5, 6], &changed),
            Match {
                checkpoint: Some(plain),
                matched_tokens: 4
            }
        );
        assert_eq!(
            c.match_prefix(&[1, 2, 3, 4, 5, 6], &Media(vec![(2, [9; 32])]))
                .matched_tokens,
            2
        );
    }
    #[test]
    fn byte_budget_counts_shared_blocks_and_has_no_eight_entry_cap() {
        let mut c = Cache::new(1000);
        for i in 1..=20 {
            c.insert(entry(&vec![1; i], &[(1, 100), (i as u64 + 1, 10)], true));
        }
        assert_eq!(c.entries.len(), 20);
        assert_eq!(c.bytes, 300);
        for _ in 0..19 {
            c.evict_one().unwrap();
        }
        assert_eq!(c.bytes, 110);
        c.evict_one();
        assert_eq!(c.bytes, 0);
    }
    #[test]
    fn output_state_requires_an_extension_when_its_head_is_stale() {
        let mut c = Cache::new(100);
        let p = c.insert(entry(&[1, 2], &[(1, 10)], true));
        let g = c.insert(entry(&[1, 2, 3], &[(2, 10)], false));
        assert_eq!(c.find(&[1, 2, 3], &Media::default()), Some(p));
        assert_eq!(c.find(&[1, 2, 3, 4], &Media::default()), Some(g));
        let mut upgraded = entry(&[1, 2, 3], &[(3, 10)], true);
        upgraded.media = Media::default();
        c.insert(upgraded);
        assert_ne!(c.find(&[1, 2, 3], &Media::default()), Some(p));
    }
    #[test]
    fn radix_matches_a_linear_oracle_after_arbitrary_splits_and_evictions() {
        let mut c = Cache::new(1_000_000);
        let mut random = 20261002u64;
        for step in 0..400 {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            let len = 1 + ((random >> 32) as usize % 12);
            let tokens: Vec<_> = (0..len)
                .map(|i| ((random >> (i * 3 % 48)) % 5) as u32)
                .collect();
            let mut e = entry(&tokens, &[(step + 1, 10)], step % 3 != 0);
            e.media = Media(vec![(2, [(step % 3) as u8; 32])]);
            c.insert(e);
            if step % 4 == 0 {
                c.evict_one();
            }
            let query: Vec<_> = tokens.iter().copied().chain([7]).collect();
            let media = Media(vec![(2, [(step % 2) as u8; 32])]);
            let found = c.match_prefix(&query, &media);
            let mut expected_match = 0;
            let mut expected_checkpoint = 0;
            for e in c.entries.values() {
                let raw = e
                    .tokens
                    .iter()
                    .zip(&query)
                    .take_while(|(a, b)| a == b)
                    .count();
                let common = e.media.common_tokens(&media, raw);
                expected_match = expected_match.max(common);
                if common == e.tokens.len() && (common < query.len() || e.logits_valid) {
                    expected_checkpoint = expected_checkpoint.max(common);
                }
            }
            assert_eq!(
                found.matched_tokens, expected_match,
                "step {step}, query {query:?}, media {media:?}"
            );
            assert_eq!(
                found
                    .checkpoint
                    .map(|id| c.entries[&id].tokens.len())
                    .unwrap_or(0),
                expected_checkpoint
            );
        }
    }
}
