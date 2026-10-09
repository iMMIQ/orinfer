//! Bounded request-local proposals copied from an earlier token suffix.
//! The target model still verifies every token; this never commits a lookup.
use std::collections::{HashMap, VecDeque};

const KEY: usize = 4;
const CONTEXT: usize = 8;
const WINDOW: usize = 8192;

#[derive(Default)]
pub(super) struct PromptLookup {
    index: HashMap<[u32; KEY], VecDeque<usize>>,
    order: VecDeque<([u32; KEY], usize)>,
    indexed: usize,
    misses: usize,
    cooldown: usize,
}

impl PromptLookup {
    pub fn propose(&mut self, history: &[u32], count: usize) -> Option<Vec<u32>> {
        if history.len() < KEY || count == 0 {
            return None;
        }
        // Only append committed history. Neither rejected drafts nor another
        // request's prefix/cache entries enter the index.
        let first = self.indexed.max(history.len().saturating_sub(WINDOW));
        for start in first..=history.len() - KEY {
            let key = history[start..start + KEY].try_into().unwrap();
            self.index.entry(key).or_default().push_back(start);
            self.order.push_back((key, start));
            while self.order.len() > WINDOW {
                let (old, position) = self.order.pop_front().unwrap();
                let entries = self.index.get_mut(&old).unwrap();
                let removed = entries.pop_front();
                debug_assert_eq!(removed, Some(position));
                if entries.is_empty() {
                    self.index.remove(&old);
                }
            }
        }
        self.indexed = history.len() - KEY + 1;
        self.find(history, count, KEY)
            .map(|end| history[end..end + count].to_vec())
    }

    pub fn available(&self, history: &[u32], count: usize) -> bool {
        self.find(history, count, CONTEXT).is_some()
    }

    fn find(&self, history: &[u32], count: usize, minimum: usize) -> Option<usize> {
        if history.len() < KEY || count == 0 {
            return None;
        }
        if history.len() < self.cooldown {
            return None;
        }
        let key: [u32; KEY] = history[history.len() - KEY..].try_into().unwrap();
        let positions = self.index.get(&key)?;
        let mut best = None;
        let mut longest = KEY - 1;
        for &start in positions.iter().rev().take(64) {
            let end = start + KEY;
            if end + count > history.len() {
                continue;
            }
            let mut matched = KEY;
            while matched < CONTEXT
                && matched < end
                && history[end - matched - 1] == history[history.len() - matched - 1]
            {
                matched += 1;
            }
            if matched > longest {
                longest = matched;
                best = Some(end);
            }
            if longest == CONTEXT {
                break;
            }
        }
        best.filter(|_| longest >= minimum)
    }

    pub fn observe(&mut self, accepted: usize, proposed: usize, progress: usize) {
        if accepted * 2 >= proposed {
            self.misses = 0;
        } else {
            self.misses += 1;
            if self.misses >= 2 {
                self.cooldown = progress.saturating_add(64);
                self.misses = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn longest_context_wins_and_only_existing_continuations_are_proposed() {
        let mut lookup = PromptLookup::default();
        let history = [
            1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92, 0, 5, 6, 7, 8, 80, 81, 82, 0, 1, 2, 3, 4, 5, 6, 7,
            8,
        ];
        assert_eq!(lookup.propose(&history, 3), Some(vec![90, 91, 92]));
        assert_eq!(PromptLookup::default().propose(&[1, 2, 3, 4], 3), None);
        assert_eq!(PromptLookup::default().propose(&history, 0), None);
    }

    #[test]
    fn index_is_incremental_bounded_and_request_local() {
        let mut lookup = PromptLookup::default();
        let mut history: Vec<_> = (0..WINDOW as u32 * 2).collect();
        assert_eq!(lookup.propose(&history, 3), None);
        assert_eq!(lookup.order.len(), WINDOW - KEY + 1);
        let size = lookup.order.len();
        lookup.propose(&history, 3);
        assert_eq!(lookup.order.len(), size);
        history.extend_from_slice(&[10_000, 10_001, 10_002, 10_003]);
        assert_eq!(
            lookup.propose(&history, 3),
            Some(vec![10_004, 10_005, 10_006])
        );
        assert!(lookup.order.len() <= WINDOW);
        assert_eq!(
            PromptLookup::default().propose(&[10_000, 10_001, 10_002, 10_003], 3),
            None
        );
    }

    #[test]
    fn repeated_rejections_temporarily_disable_lookup() {
        let history = [1, 2, 3, 4, 8, 9, 1, 2, 3, 4];
        let mut lookup = PromptLookup::default();
        assert_eq!(lookup.propose(&history, 2), Some(vec![8, 9]));
        lookup.observe(0, 2, history.len());
        lookup.observe(0, 2, history.len());
        assert_eq!(lookup.propose(&history, 2), None);
        let mut resumed = history.to_vec();
        resumed.extend_from_slice(&[0; 64]);
        resumed.extend_from_slice(&[1, 2, 3, 4]);
        assert_eq!(lookup.propose(&resumed, 2), Some(vec![0, 0]));
        lookup.observe(1, 2, resumed.len());
        assert_eq!(lookup.misses, 0);
    }

    #[test]
    fn only_long_contexts_offer_a_wider_verification() {
        let history = [
            1, 2, 3, 4, 5, 6, 7, 8, 90, 91, 92, 0, 1, 2, 3, 4, 5, 6, 7, 8,
        ];
        let mut lookup = PromptLookup::default();
        lookup.propose(&history, 3);
        assert!(lookup.available(&history, 3));
        assert!(!PromptLookup::default().available(&history, 3));
        assert!(!lookup.available(&[5, 6, 7, 8], 3));
        lookup.observe(1, 7, history.len());
        lookup.observe(1, 7, history.len());
        assert!(!lookup.available(&history, 3));
    }
}
