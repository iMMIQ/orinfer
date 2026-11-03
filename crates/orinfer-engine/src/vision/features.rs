//! Model-scoped immutable encoder outputs, independent of token-prefix positions.
use std::{collections::BTreeMap, sync::Arc};

struct Entry<T> {
    value: Arc<T>,
    bytes: usize,
    used: u64,
}
pub(crate) struct FeatureCache<T> {
    entries: BTreeMap<[u8; 32], Entry<T>>,
    budget: usize,
    bytes: usize,
    clock: u64,
}
impl<T> FeatureCache<T> {
    pub fn new(budget: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            budget,
            bytes: 0,
            clock: 0,
        }
    }
    pub fn contains(&self, key: &[u8; 32]) -> bool {
        self.entries.contains_key(key)
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn get(&mut self, key: &[u8; 32]) -> Option<Arc<T>> {
        let entry = self.entries.get_mut(key)?;
        self.clock = self.clock.saturating_add(1);
        entry.used = self.clock;
        Some(Arc::clone(&entry.value))
    }
    /// Include the retained allocation and conservative map/Arc metadata.
    /// Active requests keep their own Arc when a cached output is evicted.
    pub fn insert(&mut self, key: [u8; 32], value: Arc<T>, allocation_bytes: usize) {
        let Some(bytes) = allocation_bytes.checked_add(192) else {
            return;
        };
        if bytes > self.budget {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.bytes -= old.bytes;
        }
        while self.bytes > self.budget - bytes {
            let key = *self.entries.iter().min_by_key(|(_, v)| v.used).unwrap().0;
            self.bytes -= self.entries.remove(&key).unwrap().bytes;
        }
        self.clock = self.clock.saturating_add(1);
        self.bytes += bytes;
        self.entries.insert(
            key,
            Entry {
                value,
                bytes,
                used: self.clock,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_features_preserve_active_outputs_and_distinct_media() {
        let mut cache = FeatureCache::new(2 * (192 + 4));
        let active = Arc::new(vec![1, 2, 3, 4]);
        cache.insert([1; 32], Arc::clone(&active), 4);
        cache.insert([2; 32], Arc::new(vec![9; 4]), 4);
        assert!(Arc::ptr_eq(&active, &cache.get(&[1; 32]).unwrap()));
        cache.insert([3; 32], Arc::new(vec![7; 4]), 4);
        assert!(!cache.contains(&[2; 32]));
        assert!(cache.contains(&[1; 32]) && cache.contains(&[3; 32]));
        cache.insert([4; 32], Arc::new(vec![0; 1024]), 1024);
        assert_eq!(cache.bytes(), 392);
        cache.insert([1; 32], Arc::new(vec![8; 4]), 4);
        assert_eq!(*active, [1, 2, 3, 4]);
        assert_eq!(**cache.get(&[1; 32]).as_ref().unwrap(), [8; 4]);
        let mut disabled = FeatureCache::new(0);
        disabled.insert([1; 32], active, 4);
        assert_eq!(disabled.bytes(), 0);
    }
}
