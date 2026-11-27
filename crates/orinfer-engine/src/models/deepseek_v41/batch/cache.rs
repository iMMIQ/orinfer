//! Generic LRU program cache with per-family consecutive-use tracking.

use std::{
    collections::{BTreeMap, VecDeque},
    rc::Rc,
};

pub(in crate::models::deepseek_v41) type Family = (super::super::PacketRole, bool, usize);

pub(in crate::models::deepseek_v41) trait FamilyKey:
    Ord + Clone
{
    fn family(&self) -> Family;
}

pub(in crate::models::deepseek_v41) struct ProgramCache<K, V> {
    entries: BTreeMap<K, Rc<V>>,
    recent: BTreeMap<Family, (K, u8)>,
    lru: VecDeque<K>,
}

impl<K: FamilyKey, V> Default for ProgramCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: FamilyKey, V> ProgramCache<K, V> {
    pub(super) fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            recent: BTreeMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub(super) fn get(&self, key: &K) -> Option<&Rc<V>> {
        self.entries.get(key)
    }

    pub(super) fn insert(&mut self, key: K, value: Rc<V>) {
        self.entries.insert(key, value);
    }

    pub(super) fn touch(&mut self, key: K) {
        self.lru.retain(|k| k != &key);
        self.lru.push_back(key);
    }

    pub(super) fn consecutive_uses(&self, key: &K) -> u8 {
        self.recent
            .get(&key.family())
            .filter(|(previous, _)| previous == key)
            .map_or(0, |(_, uses)| *uses)
    }

    pub(super) fn record_use(&mut self, key: K) {
        let uses = self.consecutive_uses(&key).saturating_add(1);
        self.recent.insert(key.family(), (key, uses));
    }

    pub(super) fn remove(&mut self, key: &K) -> bool {
        let removed = self.entries.remove(key).is_some();
        let family = key.family();
        if self
            .recent
            .get(&family)
            .is_some_and(|(last, _)| last == key)
        {
            self.recent.remove(&family);
        }
        removed
    }

    pub(super) fn remove_with_lru(&mut self, key: &K) -> bool {
        let removed = self.remove(key);
        self.lru.retain(|k| k != key);
        removed
    }

    pub(super) fn trim_to(&mut self, capacity: usize) -> bool {
        let mut evicted = false;
        while self.entries.len() > capacity {
            let Some(old) = self.lru.pop_front() else {
                break;
            };
            evicted |= self.remove(&old);
        }
        evicted
    }

    pub(super) fn evict_oldest_unprotected(&mut self, protected: Option<&K>) -> bool {
        while let Some(pos) = self.lru.iter().position(|k| Some(k) != protected) {
            let Some(key) = self.lru.remove(pos) else {
                break;
            };
            if self.remove(&key) {
                return true;
            }
        }
        false
    }

    pub(super) fn count_where(&self, mut pred: impl FnMut(&V) -> bool) -> usize {
        self.entries.values().filter(|v| pred(v)).count()
    }

    #[cfg(test)]
    pub(super) fn contains_key(&self, key: &K) -> bool {
        self.entries.contains_key(key)
    }

    #[cfg(test)]
    pub(super) fn recent_len(&self) -> usize {
        self.recent.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deepseek_v41::PacketRole;

    #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct TestKey {
        family: Family,
        id: usize,
    }
    impl FamilyKey for TestKey {
        fn family(&self) -> Family {
            self.family
        }
    }

    #[test]
    fn program_cache_tracks_lru_consecutive_uses_and_protected_eviction() {
        let fam = (PacketRole::Target, false, 1);
        let k1 = TestKey { family: fam, id: 1 };
        let k2 = TestKey { family: fam, id: 2 };
        let k3 = TestKey { family: fam, id: 3 };
        let mut cache = ProgramCache::<TestKey, bool>::new();
        cache.insert(k1.clone(), Rc::new(true));
        cache.touch(k1.clone());
        cache.record_use(k1.clone());
        cache.record_use(k1.clone());
        assert_eq!(cache.consecutive_uses(&k1), 2);

        cache.insert(k2.clone(), Rc::new(false));
        cache.touch(k2.clone());
        assert_eq!(cache.count_where(|&v| v), 1);

        // Protecting oldest key k1 forces eviction of k2 instead.
        assert!(cache.evict_oldest_unprotected(Some(&k1)));
        assert!(cache.get(&k1).is_some());
        assert!(cache.get(&k2).is_none());
        assert_eq!(cache.consecutive_uses(&k1), 2);

        // Evicting k1 clears its consecutive-use streak.
        cache.insert(k3.clone(), Rc::new(true));
        cache.touch(k3.clone());
        assert!(cache.trim_to(1));
        assert!(cache.get(&k1).is_none());
        assert_eq!(cache.consecutive_uses(&k1), 0);
        assert!(cache.get(&k3).is_some());
    }
}
