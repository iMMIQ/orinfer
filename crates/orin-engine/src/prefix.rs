//! Prefix identity, bounded endpoint checkpoints and reuse accounting.
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Default, Debug, Clone, Serialize)]
pub struct Statistics {
    pub cached_tokens: usize,
    pub lookup_s: f64,
    pub restore_s: f64,
    pub store_s: f64,
    pub resident_bytes: usize,
    pub entries: usize,
}

pub(crate) struct Entry<T> {
    pub tokens: Vec<u32>,
    pub media: [u8; 32],
    pub snapshot: T,
    pub bytes: usize,
    pub warm_tokens: usize,
}

pub(crate) struct Cache<T> {
    pub budget: usize,
    pub bytes: usize,
    pub entries: VecDeque<Entry<T>>,
}

impl<T> Cache<T> {
    pub fn new(budget: usize) -> Self {
        Self {
            budget,
            bytes: 0,
            entries: VecDeque::new(),
        }
    }
    pub fn find(&self, tokens: &[u32], media: &[u8; 32]) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.media == *media && tokens.starts_with(&e.tokens))
            .max_by_key(|(_, e)| e.tokens.len())
            .map(|(i, _)| i)
    }
    pub fn touch(&mut self, index: usize) -> &Entry<T> {
        let entry = self.entries.remove(index).expect("checked prefix index");
        self.entries.push_back(entry);
        self.entries.back().expect("retained prefix entry")
    }
    pub fn evict_for(&mut self, bytes: usize) -> Vec<T> {
        let mut removed = vec![];
        while self.bytes > self.budget.saturating_sub(bytes) || self.entries.len() >= 8 {
            let Some(entry) = self.entries.pop_front() else {
                break;
            };
            self.bytes -= entry.bytes;
            removed.push(entry.snapshot);
        }
        removed
    }
    pub fn insert(&mut self, entry: Entry<T>) {
        debug_assert!(entry.bytes <= self.budget.saturating_sub(self.bytes));
        self.bytes += entry.bytes;
        self.entries.push_back(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(tokens: &[u32], media: u8, bytes: usize) -> Entry<usize> {
        Entry {
            tokens: tokens.to_vec(),
            media: [media; 32],
            snapshot: tokens.len(),
            bytes,
            warm_tokens: tokens.len() - 1,
        }
    }
    #[test]
    fn only_complete_state_boundaries_and_identical_media_are_reused() {
        let mut c = Cache::new(100);
        c.insert(entry(&[1, 2], 0, 20));
        c.insert(entry(&[1, 2, 3, 4], 0, 20));
        assert_eq!(c.find(&[1, 2, 3], &[0; 32]), Some(0));
        assert_eq!(c.find(&[1, 2, 3, 4, 5], &[0; 32]), Some(1));
        assert_eq!(c.find(&[1, 9], &[0; 32]), None);
        assert_eq!(c.find(&[1, 2], &[1; 32]), None);
        assert_eq!(c.find(&[1], &[0; 32]), None);
    }
    #[test]
    fn lru_eviction_obeys_bytes_and_endpoint_count() {
        let mut c = Cache::new(100);
        c.insert(entry(&[1], 0, 40));
        c.insert(entry(&[2], 0, 40));
        c.touch(0);
        assert_eq!(c.evict_for(50), vec![1]);
        assert_eq!(c.entries[0].tokens, vec![1]);
        c.insert(entry(&[3], 0, 50));
        assert_eq!(c.bytes, 90);
        let mut c = Cache::new(1000);
        for i in 0..8 {
            c.insert(entry(&[i], 0, 1));
        }
        assert_eq!(c.evict_for(1), vec![1]);
        assert_eq!(c.entries.len(), 7);
    }
}
