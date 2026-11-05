//! Service-scoped immutable media. URL contents are fetched before lookup;
//! explicit UUID references may reuse retained bytes without fetching them.
use super::{ImageInput, Result, VisionSpec};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex, Weak},
};

type Id = [u8; 32];
struct Entry {
    data: Arc<Vec<u8>>,
    image: Arc<ImageInput>,
    policy: Id,
    aliases: BTreeSet<Id>,
    bytes: usize,
    used: u64,
}
#[derive(Default)]
struct State {
    entries: BTreeMap<Id, Entry>,
    aliases: BTreeMap<Id, Id>,
    flights: BTreeMap<Id, Weak<Mutex<()>>>,
    bytes: usize,
    clock: u64,
    hits: usize,
    misses: usize,
    source_hits: usize,
    evictions: usize,
}
pub(in crate::server) struct Cache {
    budget: usize,
    state: Mutex<State>,
}
fn digest(data: &[u8]) -> Id {
    Sha256::digest(data).into()
}
impl State {
    fn remove(&mut self, id: &Id) -> Option<Entry> {
        let entry = self.entries.remove(id)?;
        self.bytes -= entry.bytes;
        for alias in &entry.aliases {
            self.aliases.remove(alias);
        }
        Some(entry)
    }
    fn evict(&mut self) {
        let id = *self.entries.iter().min_by_key(|(_, e)| e.used).unwrap().0;
        self.remove(&id);
        self.evictions += 1;
    }
}
impl Cache {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            state: Mutex::new(State::default()),
        }
    }
    pub fn policy(spec: &VisionSpec, height: usize, width: usize) -> Result<Id> {
        let mut hash = Sha256::new();
        hash.update(b"orinfer-image-preprocess-v1");
        hash.update(serde_json::to_vec(spec).map_err(|e| e.to_string())?);
        hash.update((height as u64).to_le_bytes());
        hash.update((width as u64).to_le_bytes());
        Ok(hash.finalize().into())
    }
    pub fn identity(data: &[u8]) -> Id {
        digest(data)
    }
    /// The weak gate coalesces duplicate work without retaining source bytes or
    /// holding the cache lock while decoding. Other images progress in parallel.
    pub fn gate(&self, id: Id) -> Arc<Mutex<()>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(gate) = state.flights.get(&id).and_then(Weak::upgrade) {
            return gate;
        }
        state.flights.retain(|_, gate| gate.strong_count() > 0);
        let gate = Arc::new(Mutex::new(()));
        state.flights.insert(id, Arc::downgrade(&gate));
        gate
    }
    pub fn get(&self, id: &Id, policy: Id) -> Option<Arc<ImageInput>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.clock = state.clock.saturating_add(1);
        let clock = state.clock;
        if !state.entries.get(id).is_some_and(|e| e.policy == policy) {
            state.misses += 1;
            return None;
        }
        state.hits += 1;
        let entry = state.entries.get_mut(id).unwrap();
        entry.used = clock;
        Some(Arc::clone(&entry.image))
    }
    pub fn insert(&self, id: Id, policy: Id, data: &[u8], image: &ImageInput) {
        let bytes = data
            .len()
            .saturating_add(image.host_bytes())
            .saturating_add(512);
        if bytes > self.budget {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let aliases = state.remove(&id).map_or_else(BTreeSet::new, |e| e.aliases);
        let bytes = bytes.saturating_add(aliases.len().saturating_mul(128));
        if bytes > self.budget {
            return;
        }
        while state.bytes > self.budget - bytes {
            state.evict();
        }
        // Allocate only after making room. Active requests retain their own
        // charged image copies if a cached artifact is evicted.
        let entry = Entry {
            data: Arc::new(data.to_vec()),
            image: Arc::new(image.clone()),
            policy,
            aliases,
            bytes,
            used: state.clock,
        };
        for alias in &entry.aliases {
            state.aliases.insert(*alias, id);
        }
        state.bytes += bytes;
        state.entries.insert(id, entry);
    }
    pub fn bind(&self, uuid: &str, data: &[u8]) -> Result<()> {
        let alias = digest(uuid.as_bytes());
        let id = digest(data);
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(previous) = state.aliases.get(&alias) {
            return if *previous == id {
                Ok(())
            } else {
                Err("Image uuid is already bound to different contents; use a new uuid".into())
            };
        }
        let Some(mut entry) = state.remove(&id) else {
            // An uncached/oversized input remains usable in this request.
            return Ok(());
        };
        entry.bytes = entry.bytes.saturating_add(128);
        if entry.bytes > self.budget {
            return Ok(());
        }
        while state.bytes > self.budget - entry.bytes {
            state.evict();
        }
        entry.aliases.insert(alias);
        for alias in &entry.aliases {
            state.aliases.insert(*alias, id);
        }
        state.bytes += entry.bytes;
        state.entries.insert(id, entry);
        Ok(())
    }
    pub fn source(&self, uuid: &str) -> Result<Arc<Vec<u8>>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let id = state.aliases.get(&digest(uuid.as_bytes())).copied();
        state.clock = state.clock.saturating_add(1);
        let clock = state.clock;
        let entry = id.and_then(|id| state.entries.get_mut(&id)).ok_or(
            "Image uuid cache miss; resend image_url.url with the image payload and the same uuid",
        )?;
        entry.used = clock;
        let data = Arc::clone(&entry.data);
        state.source_hits += 1;
        Ok(data)
    }
    pub fn statistics(&self) -> serde_json::Value {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        serde_json::json!({"budget_bytes":self.budget,"resident_bytes":state.bytes,
            "entries":state.entries.len(),"references":state.aliases.len(),
            "hits":state.hits,"misses":state.misses,"reference_hits":state.source_hits,
            "evictions":state.evictions})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image(value: u8) -> ImageInput {
        ImageInput {
            rgb: vec![value; 12],
            ..Default::default()
        }
    }
    #[test]
    fn references_content_changes_policy_and_eviction_are_independent() {
        let cache = Cache::new(800);
        let data = [1u8, 2, 3];
        let id = Cache::identity(&data);
        cache.insert(id, [1; 32], &data, &image(7));
        cache.bind("first", &data).unwrap();
        let retained = cache.get(&id, [1; 32]).unwrap();
        assert_eq!(cache.source("first").unwrap().as_slice(), data);
        assert!(cache.get(&id, [2; 32]).is_none());
        // Reprocessing preserves the source identity and its UUID aliases.
        cache.insert(id, [2; 32], &data, &image(9));
        assert_eq!(cache.source("first").unwrap().as_slice(), data);
        assert_eq!(retained.rgb, vec![7; 12]);
        assert!(
            cache
                .bind("first", &[5])
                .unwrap_err()
                .contains("different contents")
        );
        let next = [4u8];
        cache.insert(Cache::identity(&next), [1; 32], &next, &image(8));
        assert!(cache.source("first").unwrap_err().contains("resend"));
        assert!(cache.statistics()["resident_bytes"].as_u64().unwrap() <= 800);
        assert_eq!(retained.rgb, vec![7; 12]);
        let disabled = Cache::new(0);
        disabled.insert(id, [1; 32], &data, &image(0));
        disabled.bind("first", &data).unwrap();
        assert!(disabled.source("first").is_err());
    }
}
