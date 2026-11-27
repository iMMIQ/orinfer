//! Compressed token radix index with exact state endpoints and byte accounting.
mod tiered;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
pub(crate) use tiered::{Admission, CheckpointMemo, Tiered};

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
    /// CPU tier, summed across all tensor-parallel workers.
    pub host_resident_bytes: usize,
    pub host_entries: usize,
    pub host_hits: usize,
    pub host_stores: usize,
    pub host_evictions: usize,
}

/// Image identities are scoped to the first affected token, not the request.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Media(pub Vec<(usize, [u8; 32])>);
pub(crate) fn hash_floats(hash: &mut Sha256, values: &[f32]) {
    let (chunks, tail) = values.as_chunks::<256>();
    for chunk in chunks {
        hash.update(chunk.map(f32::to_le_bytes).as_flattened());
    }
    for value in tail {
        hash.update(value.to_le_bytes());
    }
}
impl Media {
    pub fn images(
        input: &[u32],
        images: &[crate::vision::ImageInput],
        vision: Option<&crate::vision::VisionSpec>,
    ) -> crate::artifact::Result<Self> {
        let Some(vision) = vision else {
            return if images.is_empty() {
                Ok(Self::default())
            } else {
                Err("Model has no image adapter".into())
            };
        };
        let spans = vision.image_spans(input, images)?;
        let mut media = Vec::with_capacity(images.len());
        for (image, span) in images.iter().zip(spans) {
            let mut hash = Sha256::new();
            hash.update((image.grid_height as u64).to_le_bytes());
            hash.update((image.grid_width as u64).to_le_bytes());
            if image.rgb.is_empty() {
                hash_floats(&mut hash, &image.pixels);
            } else {
                hash.update(b"rgb8");
                hash.update(&image.rgb);
            }
            media.push((span.start, hash.finalize().into()));
        }
        Ok(Self(media))
    }
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
    /// False for target-only checkpoints captured without warming the draft model.
    pub mtp_warmed: bool,
    used: u64,
    hits: u64,
}
impl<T> Entry<T> {
    /// Preserve reuse history and endpoint identity when changing storage tier.
    pub fn replace_snapshot<U>(self, snapshot: U) -> (Entry<U>, T) {
        (
            Entry {
                tokens: self.tokens,
                media: self.media,
                snapshot,
                bytes: self.bytes,
                warm_tokens: self.warm_tokens,
                logits_valid: self.logits_valid,
                mtp_warmed: self.mtp_warmed,
                used: self.used,
                hits: self.hits,
            },
            self.snapshot,
        )
    }
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
            mtp_warmed: true,
            used: 0,
            hits: 0,
        }
    }
    pub fn with_mtp_warmed(mut self, mtp_warmed: bool) -> Self {
        self.mtp_warmed = mtp_warmed;
        self
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
    logical_bytes: usize,
    bank_refs: HashMap<u64, (usize, usize)>,
    entry_banks: HashMap<u64, Vec<(u64, usize)>>,
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
            logical_bytes: 0,
            bank_refs: HashMap::new(),
            entry_banks: HashMap::new(),
        }
    }
    pub fn match_prefix_by(
        &self,
        tokens: &[u32],
        media: &Media,
        score: impl Fn(&Entry<T>) -> Option<f64>,
    ) -> Match {
        self.probe::<true>(tokens, media, score)
    }
    /// O(depth) radix walk returning all endpoint IDs whose token prefix and
    /// media identities match `tokens` exactly.
    pub fn exact_matches(&self, tokens: &[u32], media: &Media) -> Vec<u64> {
        let (mut node, mut depth) = (&self.root, 0);
        while depth < tokens.len() {
            let Some(child) = node.children.get(&tokens[depth]) else {
                return vec![];
            };
            if !tokens[depth..].starts_with(&child.label) {
                return vec![];
            }
            depth += child.label.len();
            node = child;
        }
        node.endpoints
            .iter()
            .copied()
            .filter(|id| self.entries[id].media.common_tokens(media, depth) == depth)
            .collect()
    }
    fn probe<const TRACK_OVERLAP: bool>(
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
                if TRACK_OVERLAP {
                    result.matched_tokens = result.matched_tokens.max(common);
                }
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
            if TRACK_OVERLAP {
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
            }
            if let Some(child) = child {
                let remaining = &tokens[depth..];
                if remaining.starts_with(&child.label) {
                    depth += child.label.len();
                    node = child;
                    continue;
                }
                if TRACK_OVERLAP {
                    depth += child
                        .label
                        .iter()
                        .zip(remaining)
                        .take_while(|(a, b)| a == b)
                        .count();
                    node = child;
                }
            }
            if TRACK_OVERLAP {
                let mut descendants = vec![];
                node.descendants(&mut descendants);
                for id in descendants {
                    result.matched_tokens = result
                        .matched_tokens
                        .max(self.entries[&id].media.common_tokens(media, depth));
                }
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
        let e = self.mark_recent(id).expect("checked prefix id");
        e.hits += 1;
        e
    }
    pub fn mark_recent(&mut self, id: u64) -> Option<&mut Entry<T>> {
        let e = self.entries.get_mut(&id)?;
        self.clock += 1;
        e.used = self.clock;
        Some(e)
    }
    pub fn resident_with(&self, extra: Option<&dyn Resident>) -> usize {
        let Some(extra) = extra else {
            return self.bytes;
        };
        let mut own = BTreeMap::new();
        extra.allocations(&mut own);
        let added: usize = own
            .iter()
            .filter(|(id, _)| !self.bank_refs.contains_key(id))
            .map(|(_, bytes)| *bytes)
            .sum();
        self.bytes + added
    }
    pub fn unique_bytes(&self, id: u64) -> usize {
        self.entry_banks.get(&id).map_or(0, |banks| {
            banks
                .iter()
                .filter(|(bank_id, _)| self.bank_refs.get(bank_id).is_some_and(|r| r.1 == 1))
                .map(|(_, bytes)| *bytes)
                .sum()
        })
    }
    pub fn contains_bank(&self, bank_id: u64) -> bool {
        self.bank_refs.contains_key(&bank_id)
    }
    pub fn update_statistics(&self, statistics: &mut Statistics) {
        statistics.resident_bytes = self.bytes;
        statistics.logical_bytes = self.logical_bytes;
        statistics.shared_bytes = self.logical_bytes.saturating_sub(self.bytes);
        statistics.entries = self.entries.len();
    }
    pub fn remove(&mut self, id: u64) -> Option<Entry<T>> {
        let entry = self.entries.remove(&id)?;
        self.root.remove(&entry.tokens, id);
        self.logical_bytes = self.logical_bytes.saturating_sub(entry.bytes);
        if let Some(banks) = self.entry_banks.remove(&id) {
            for (bank_id, _) in banks {
                if let std::collections::hash_map::Entry::Occupied(mut slot) =
                    self.bank_refs.entry(bank_id)
                {
                    let (bytes, refs) = slot.get_mut();
                    if *refs <= 1 {
                        self.bytes = self.bytes.saturating_sub(*bytes);
                        slot.remove();
                    } else {
                        *refs -= 1;
                    }
                }
            }
        }
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
        self.evict_one_except(None)
    }
    /// A generated endpoint must not evict the prompt that makes repeating
    /// that request useful. General memory admission may still evict any entry.
    pub fn evict_one_except(&mut self, protected: Option<u64>) -> Option<T> {
        self.evict_entry_one_except(protected).map(|e| e.snapshot)
    }
    /// Move an endpoint between storage tiers without losing its token identity.
    pub fn evict_entry_one_except(&mut self, protected: Option<u64>) -> Option<Entry<T>> {
        let victim = self.victim_except(protected)?;
        self.remove(victim)
    }
    /// Select without unpublishing the source. Tier transfers commit removal
    /// only after the adapter reports successful completion.
    pub fn victim_except(&self, protected: Option<u64>) -> Option<u64> {
        self.victim_with_priority(protected, false, |_| false)
    }

    /// GPU capture preserves live prompts first, then favors prompt
    /// checkpoints and proven continuations over unproven generated tails.
    pub fn victim_for_capture(&self, active_prompt: impl Fn(u64) -> bool) -> Option<u64> {
        self.victim_with_priority(None, true, active_prompt)
    }

    fn victim_with_priority(
        &self,
        protected: Option<u64>,
        prefer_unproven: bool,
        active_prompt: impl Fn(u64) -> bool,
    ) -> Option<u64> {
        let mut best: Option<(u64, bool, bool, f64, u64)> = None;
        for (&id, entry) in &self.entries {
            if Some(id) == protected {
                continue;
            }
            let active = active_prompt(id);
            let prio = prefer_unproven && (entry.logits_valid || entry.hits != 0);
            if let Some((_, best_active, best_prio, _, _)) = best
                && (active, prio) > (best_active, best_prio)
            {
                continue;
            }
            let parent = self.ancestor_depth(entry);
            let freed = self.unique_bytes(id);
            let util = (entry.tokens.len() - parent).max(1) as f64 * (1.0 + entry.hits as f64)
                / (freed.max(1) as f64 * (self.clock - entry.used + 1) as f64);
            let is_better = match best {
                None => true,
                Some((_, best_active, best_prio, best_util, best_used)) => (active, prio)
                    .cmp(&(best_active, best_prio))
                    .then_with(|| util.total_cmp(&best_util))
                    .then_with(|| entry.used.cmp(&best_used))
                    .is_lt(),
            };
            if is_better {
                best = Some((id, active, prio, util, entry.used));
            }
        }
        best.map(|(id, ..)| id)
    }
    pub fn insert(&mut self, mut entry: Entry<T>) -> u64 {
        self.clock += 1;
        self.next_id += 1;
        entry.used = self.clock;
        let id = self.next_id;
        let mut own = BTreeMap::new();
        entry.snapshot.allocations(&mut own);
        let banks: Vec<(u64, usize)> = own.into_iter().collect();
        for &(bank_id, bytes) in &banks {
            self.bank_refs
                .entry(bank_id)
                .and_modify(|(_, refs)| *refs += 1)
                .or_insert_with(|| {
                    self.bytes += bytes;
                    (bytes, 1)
                });
        }
        self.logical_bytes = self.logical_bytes.saturating_add(entry.bytes);
        self.entry_banks.insert(id, banks);
        self.root.insert(&entry.tokens, id);
        self.entries.insert(id, entry);
        debug_assert!(self.bytes <= self.budget);
        id
    }
    #[cfg(test)]
    pub fn clear(&mut self) -> Vec<T> {
        self.root = Node::default();
        self.bytes = 0;
        self.logical_bytes = 0;
        self.bank_refs.clear();
        self.entry_banks.clear();
        std::mem::take(&mut self.entries)
            .into_values()
            .map(|e| e.snapshot)
            .collect()
    }
}

/// Device wins an equal token prefix. Model adapters supply state-validity and
/// compute/transfer cost scores; neither tree needs device-specific policies.
pub(crate) fn match_tiers<D: Resident, H: Resident>(
    device: &Cache<D>,
    host: &Cache<H>,
    tokens: &[u32],
    media: &Media,
    device_score: impl Fn(&Entry<D>) -> Option<f64>,
    host_score: impl Fn(&Entry<H>) -> Option<f64>,
) -> Option<(bool, u64)> {
    // Tier selection consumes only restorable endpoints. Other branches can
    // contribute partial-overlap statistics, while endpoints must be on path.
    let d = device
        .probe::<false>(tokens, media, device_score)
        .checkpoint;
    let h = host.probe::<false>(tokens, media, host_score).checkpoint;
    match (d, h) {
        (Some(d), Some(h)) if host.entries[&h].tokens.len() > device.entries[&d].tokens.len() => {
            Some((true, h))
        }
        (Some(d), _) => Some((false, d)),
        (_, h) => h.map(|id| (true, id)),
    }
}

#[cfg(test)]
#[path = "prefix_tests.rs"]
mod tests;
