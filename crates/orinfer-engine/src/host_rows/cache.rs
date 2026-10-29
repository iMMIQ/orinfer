//! Bounded four-way row storage. Immutable table bytes are the only payload.
use crate::artifact::Result;

const WAYS: usize = 4;
const OVERHEAD: usize = 4096;

pub(super) struct RowCache {
    width: usize,
    tags: Vec<u64>,
    ages: Vec<u64>,
    data: Vec<u8>,
    clock: u64,
    entries: usize,
}
impl RowCache {
    pub(super) fn new(width: usize, budget: usize) -> Result<Option<Self>> {
        let overhead = OVERHEAD + std::mem::size_of::<Self>();
        let cost = width
            .checked_add(16)
            .ok_or("Host row cache size overflow")?;
        if width == 0 {
            return Err("Empty host cache row".into());
        }
        let slots = budget.saturating_sub(overhead) / cost / WAYS * WAYS;
        if slots == 0 {
            return Ok(None);
        }
        fn allocate<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
            let mut result = Vec::new();
            result.try_reserve_exact(count).map_err(|e| e.to_string())?;
            result.resize(count, value);
            Ok(result)
        }
        let cache = Self {
            width,
            tags: allocate(slots, u64::MAX)?,
            ages: allocate(slots, 0)?,
            data: allocate(
                slots
                    .checked_mul(width)
                    .ok_or("Host cache extent overflow")?,
                0,
            )?,
            clock: 0,
            entries: 0,
        };
        if cache.bytes() > budget {
            return Err("Host cache allocation exceeds its budget".into());
        }
        Ok(Some(cache))
    }
    fn start(&self, mut row: u64) -> usize {
        row = (row ^ (row >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        row = (row ^ (row >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        row ^= row >> 31;
        (row % (self.tags.len() / WAYS) as u64) as usize * WAYS
    }
    pub(super) fn get(&mut self, row: u64) -> Option<&[u8]> {
        let first = self.start(row);
        let slot = (first..first + WAYS).find(|&slot| self.tags[slot] == row)?;
        self.clock = self.clock.saturating_add(1);
        self.ages[slot] = self.clock;
        Some(&self.data[slot * self.width..(slot + 1) * self.width])
    }
    pub(super) fn insert(&mut self, row: u64, data: &[u8]) {
        assert_eq!(data.len(), self.width);
        assert_ne!(row, u64::MAX);
        let first = self.start(row);
        let slot = (first..first + WAYS)
            .find(|&slot| self.tags[slot] == row || self.tags[slot] == u64::MAX)
            .unwrap_or_else(|| {
                (first..first + WAYS)
                    .min_by_key(|&slot| self.ages[slot])
                    .unwrap()
            });
        self.entries += usize::from(self.tags[slot] == u64::MAX);
        self.data[slot * self.width..(slot + 1) * self.width].copy_from_slice(data);
        self.clock = self.clock.saturating_add(1);
        self.ages[slot] = self.clock;
        self.tags[slot] = row;
    }
    pub(super) fn bytes(&self) -> usize {
        OVERHEAD
            + std::mem::size_of::<Self>()
            + self.tags.capacity() * 8
            + self.ages.capacity() * 8
            + self.data.capacity()
    }
    pub(super) fn entries(&self) -> usize {
        self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn collisions_keep_bytes_and_recent_rows_within_budget() {
        let budget = OVERHEAD + std::mem::size_of::<RowCache>() + WAYS * (36 + 16);
        let mut cache = RowCache::new(36, budget).unwrap().unwrap();
        assert_eq!(cache.tags.len(), 4);
        assert!(cache.bytes() <= budget);
        for id in 0..4 {
            cache.insert(id, &[id as u8; 36]);
        }
        assert_eq!(cache.get(0).unwrap(), &[0; 36]);
        cache.insert(4, &[4; 36]);
        assert!(cache.get(1).is_none());
        assert_eq!(cache.get(0).unwrap(), &[0; 36]);
        assert_eq!(cache.get(4).unwrap(), &[4; 36]);
        assert_eq!(cache.entries(), 4);
        assert!(RowCache::new(36, 0).unwrap().is_none());
        assert!(RowCache::new(36, budget - 1).unwrap().is_none());
    }
}
