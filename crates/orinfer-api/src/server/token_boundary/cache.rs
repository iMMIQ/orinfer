//! Codec-local encodings of complete plain regions. Text equality is the key;
//! controls and tokenizer identity remain owned by the enclosing Boundary.
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

const MIN_BYTES: usize = 1024;
const OVERHEAD: usize = 64 << 10;
const MAX_ENTRIES: usize = 256;

struct Entry {
    tokens: Arc<[u32]>,
    bytes: usize,
    used: u64,
}

#[derive(Clone, Default, serde::Serialize)]
pub(super) struct Statistics {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub reused_tokens: u64,
    pub evictions: u64,
}

#[derive(Default)]
struct State {
    entries: HashMap<String, Entry>,
    clock: u64,
    statistics: Statistics,
}

pub(super) struct Cache {
    budget: usize,
    state: Mutex<State>,
}

impl Cache {
    pub fn new(budget: usize) -> Self {
        let mut state = State::default();
        state.statistics.resident_bytes = budget.min(OVERHEAD);
        Self {
            budget,
            state: Mutex::new(state),
        }
    }

    pub fn get(&self, text: &str) -> Option<Arc<[u32]>> {
        if text.len() < MIN_BYTES || text.len() > self.budget {
            return None;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.clock = state.clock.saturating_add(1);
        let clock = state.clock;
        if let Some(entry) = state.entries.get_mut(text) {
            entry.used = clock;
            let tokens = Arc::clone(&entry.tokens);
            state.statistics.hits += 1;
            state.statistics.reused_tokens += tokens.len() as u64;
            Some(tokens)
        } else {
            state.statistics.misses += 1;
            None
        }
    }

    pub fn insert(&self, text: String, tokens: Arc<[u32]>) {
        // Include retained source capacity, token storage and map/Arc overhead.
        let bytes = text
            .capacity()
            .saturating_add(tokens.len().saturating_mul(4))
            .saturating_add(256);
        if text.len() < MIN_BYTES || bytes > self.budget.saturating_sub(OVERHEAD) {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        // Concurrent misses may finish together. Retain one immutable encoding.
        if state.entries.contains_key(&text) {
            return;
        }
        while state.statistics.resident_bytes > self.budget - bytes
            || state.entries.len() >= MAX_ENTRIES
        {
            let key = state
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.used)
                .unwrap()
                .0
                .clone();
            let removed = state.entries.remove(&key).unwrap();
            state.statistics.resident_bytes -= removed.bytes;
            state.statistics.evictions += 1;
        }
        let used = state.clock;
        state.entries.insert(
            text,
            Entry {
                tokens,
                bytes,
                used,
            },
        );
        state.statistics.resident_bytes += bytes;
    }

    pub fn statistics(&self) -> Statistics {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        Statistics {
            budget_bytes: self.budget,
            entries: state.entries.len(),
            ..state.statistics.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn changed_text_eviction_and_inflight_tokens_keep_their_identity() {
        let cache = Cache::new(OVERHEAD + 3000);
        let first = "a".repeat(1024);
        let second = "b".repeat(1024);
        cache.insert(first.clone(), Arc::from([1, 2, 3]));
        cache.insert(second.clone(), Arc::from([4, 5]));
        let retained = cache.get(&first).unwrap();
        cache.insert("c".repeat(1024), Arc::from([6]));
        assert!(cache.get(&second).is_none());
        assert_eq!(&*cache.get(&first).unwrap(), &[1, 2, 3]);
        assert!(cache.get(&(first.clone() + "!")).is_none());
        cache.insert("large".repeat(1024), Arc::from([7]));
        assert!(cache.statistics().resident_bytes <= OVERHEAD + 3000);
        assert_eq!(&*retained, &[1, 2, 3]);
        assert_eq!(cache.statistics().evictions, 1);
        let disabled = Cache::new(0);
        disabled.insert(first.clone(), retained);
        assert!(disabled.get(&first).is_none());
        assert_eq!(disabled.statistics().resident_bytes, 0);
    }
}
