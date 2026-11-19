//! Whole compressed-expert SLRU ownership over shared CUDA VMM pages.
use super::{Session, Value, check, sys, weight_upload};
use crate::{
    artifact::{Argument, Dtype, Kernel, Result},
    execution::ResidentExperts,
    weights::TensorSource,
};
use orinfer_model_sdk::architecture::ExpertBank;
use std::collections::{BTreeMap, BTreeSet};

type Key = (usize, usize);
#[derive(Default)]
struct Queue {
    keys: BTreeMap<Key, u64>,
    order: BTreeSet<(u64, Key)>,
}
impl Queue {
    fn remove(&mut self, key: Key) -> bool {
        if let Some(tick) = self.keys.remove(&key) {
            self.order.remove(&(tick, key));
            true
        } else {
            false
        }
    }
    fn insert(&mut self, key: Key, tick: u64) {
        self.remove(key);
        self.keys.insert(key, tick);
        self.order.insert((tick, key));
    }
    fn victim(&self, pinned: &BTreeSet<Key>) -> Option<Key> {
        self.order
            .iter()
            .map(|(_, k)| *k)
            .find(|k| !pinned.contains(k))
    }
}

pub use crate::backend::report::ExpertCacheStatistics as Statistics;

pub(super) struct Plan {
    pub banks: Vec<ExpertBank>,
    pub names: BTreeSet<String>,
    pub capacity: usize,
}
impl Plan {
    pub fn new(
        prepared: &crate::loader::PreparedModel,
        option: ResidentExperts,
    ) -> Result<Option<Self>> {
        let ResidentExperts::Count(count) = option else {
            return Ok(None);
        };
        if count == 0 || prepared.expert_banks.is_empty() {
            return Err(
                "Finite residency requires a model execution package with expert-cache contracts"
                    .into(),
            );
        }
        if prepared.expert_banks.iter().any(|b| count > b.experts) {
            return Err("Resident expert count exceeds the model's experts per bank".into());
        }
        if prepared.expert_banks.iter().all(|b| count == b.experts) {
            return Ok(None);
        }
        let mut names = BTreeSet::new();
        let buffers: BTreeMap<_, _> = prepared
            .plan
            .buffers
            .iter()
            .map(|b| (b.name.as_str(), b))
            .collect();
        let kernels: BTreeMap<_, _> = prepared
            .plan
            .kernels
            .iter()
            .map(|k| (k.name.as_str(), k))
            .collect();
        let mut consumers = BTreeSet::new();
        let mut banks = prepared.expert_banks.clone();
        // MTP=0 removes verification consumers from the active manifest.
        for bank in &mut banks {
            bank.consumers
                .retain(|name, _| kernels.contains_key(name.as_str()));
        }
        for bank in &banks {
            if bank.experts == 0 || bank.buffers.is_empty() || bank.consumers.is_empty() {
                return Err("Empty expert residency contract".into());
            }
            for name in &bank.buffers {
                let b = buffers.get(name.as_str()).ok_or("Missing expert tensor")?;
                if b.data.is_none()
                    || b.access != crate::artifact::Access::Read
                    || b.shape.first() != Some(&bank.experts)
                    || !names.insert(name.clone())
                    || !kernels.values().any(|k| {
                        k.args
                            .iter()
                            .any(|a| matches!(a, Argument::Buffer { name: used } if used == name))
                    })
                {
                    return Err("Invalid or overlapping expert tensor bank".into());
                }
            }
            for (name, index) in &bank.consumers {
                let k = kernels
                    .get(name.as_str())
                    .ok_or("Unknown expert consumer")?;
                let Some(Argument::Buffer { name: counts }) = k.args.get(*index) else {
                    return Err("Expert histogram must be a buffer argument".into());
                };
                let b = buffers
                    .get(counts.as_str())
                    .ok_or("Missing expert counts")?;
                if b.dtype != Dtype::I32
                    || b.shape != [bank.experts]
                    || b.data.is_some()
                    || !consumers.insert(name.clone())
                {
                    return Err("Invalid expert histogram contract".into());
                }
                if !k
                    .args
                    .iter()
                    .any(|a| matches!(a, Argument::Buffer { name } if bank.buffers.contains(name)))
                {
                    return Err("Expert consumer does not use its bank".into());
                }
            }
        }
        let capacity = count
            .checked_mul(prepared.expert_banks.len())
            .ok_or("Expert capacity overflow")?;
        Ok(Some(Self {
            banks,
            names,
            capacity,
        }))
    }
}

pub(super) struct Pager {
    banks: Vec<ExpertBank>,
    sources: BTreeMap<String, TensorSource>,
    consumers: BTreeMap<String, (usize, usize)>,
    refs: BTreeMap<(String, usize), usize>,
    prob: Queue,
    hot: Queue,
    transient: BTreeSet<Key>,
    clock: u64,
    workers: usize,
    pub statistics: Statistics,
    poisoned: bool,
}
impl Pager {
    pub fn new(plan: Plan, sources: BTreeMap<String, TensorSource>, workers: usize) -> Self {
        let consumers = plan
            .banks
            .iter()
            .enumerate()
            .flat_map(|(i, b)| b.consumers.iter().map(move |(k, n)| (k.clone(), (i, *n))))
            .collect();
        Self {
            banks: plan.banks,
            sources,
            consumers,
            refs: BTreeMap::new(),
            prob: Queue::default(),
            hot: Queue::default(),
            transient: BTreeSet::new(),
            clock: 0,
            workers,
            statistics: Statistics {
                capacity_experts: plan.capacity,
                ..Default::default()
            },
            poisoned: false,
        }
    }
    fn pages(&self, s: &Session, key: Key) -> Result<Vec<(String, usize)>> {
        let reservations = s.expert_buffers.borrow();
        let mut pages = vec![];
        for name in &self.banks[key.0].buffers {
            let r = &reservations[name];
            let stride = self.sources[name].bytes / self.banks[key.0].experts;
            let start = key.1 * stride;
            let end = start + stride;
            for offset in (start / r.granularity * r.granularity..end).step_by(r.granularity) {
                pages.push((name.clone(), offset));
            }
        }
        Ok(pages)
    }
    fn release(&mut self, s: &Session, key: Key) -> Result<()> {
        for page in self.pages(s, key)? {
            let n = self
                .refs
                .get_mut(&page)
                .ok_or("Lost expert page ownership")?;
            if *n == 1 {
                s.expert_buffers
                    .borrow_mut()
                    .get_mut(&page.0)
                    .unwrap()
                    .unmap_page(s, page.1)?;
                self.refs.remove(&page);
            } else {
                *n -= 1;
            }
        }
        Ok(())
    }
    pub fn ensure(&mut self, s: &Session, kernel: &Kernel, values: &[Value]) -> Result<()> {
        let Some(&(bank, argument)) = self.consumers.get(&kernel.name) else {
            return Ok(());
        };
        if self.poisoned {
            return Err("Expert cache is quarantined after a failed transition".into());
        }
        let Some(Value::Pointer(address)) = values.get(argument) else {
            return Err("Unbound expert histogram".into());
        };
        let mut counts = vec![0i32; self.banks[bank].experts];
        // SAFETY: Producer and all previous consumers finish before page eviction;
        // the model-declared histogram has this exact validated byte extent.
        unsafe {
            check(sys::cuStreamSynchronize(s.stream), "expert route barrier")?;
            check(
                sys::cuMemcpyDtoH_v2(counts.as_mut_ptr().cast(), *address, counts.len() * 4),
                "expert counts",
            )?;
        }
        if counts.iter().any(|&n| n < 0) {
            return Err("Negative expert route count".into());
        }
        let wanted: BTreeSet<_> = counts
            .iter()
            .enumerate()
            .filter(|(_, n)| **n > 0)
            .map(|(e, _)| (bank, e))
            .collect();
        self.poisoned = true;
        for key in std::mem::take(&mut self.transient) {
            self.release(s, key)?;
        }
        let mut uploads = vec![];
        for &key in &wanted {
            self.clock = self
                .clock
                .checked_add(1)
                .ok_or("Expert cache clock overflow")?;
            self.statistics.lookups += 1;
            if self.hot.keys.contains_key(&key) || self.prob.keys.contains_key(&key) {
                self.statistics.hits += 1;
                self.prob.remove(key);
                self.hot.insert(key, self.clock);
                if self.hot.keys.len() > self.statistics.capacity_experts * 3 / 4 {
                    let old = self.hot.order.first().unwrap().1;
                    self.hot.remove(old);
                    self.prob.insert(old, self.clock);
                }
                continue;
            }
            let capacity = self.statistics.capacity_experts;
            if self.prob.keys.len() + self.hot.keys.len() >= capacity
                && let Some(victim) = self
                    .prob
                    .victim(&wanted)
                    .or_else(|| self.hot.victim(&wanted))
            {
                self.release(s, victim)?;
                self.prob.remove(victim);
                self.hot.remove(victim);
                self.statistics.evictions += 1;
            }
            for page in self.pages(s, key)? {
                if !self.refs.contains_key(&page) {
                    let mut reservations = s.expert_buffers.borrow_mut();
                    let r = reservations.get_mut(&page.0).unwrap();
                    r.map_page(s, page.1)?;
                    let source = &self.sources[&page.0];
                    let bytes = r.granularity.min(source.bytes - page.1);
                    uploads.push(weight_upload::Upload {
                        source: TensorSource {
                            path: source.path.clone(),
                            offset: source.offset + page.1,
                            bytes,
                        },
                        destination: r.address + page.1 as u64,
                    });
                    self.statistics.loaded_bytes += bytes as u64;
                }
                *self.refs.entry(page).or_default() += 1;
            }
            if self.prob.keys.len() + self.hot.keys.len() < capacity {
                self.prob.insert(key, self.clock);
            } else {
                self.transient.insert(key);
            }
        }
        weight_upload::upload(s, uploads, self.workers)?;
        self.statistics.resident_experts = self.prob.keys.len() + self.hot.keys.len();
        self.statistics.transient_experts = self.transient.len();
        self.statistics.peak_transient_experts = self
            .statistics
            .peak_transient_experts
            .max(self.transient.len());
        self.statistics.mapped_bytes = s
            .expert_buffers
            .borrow()
            .values()
            .map(|r| r.physical_bytes())
            .sum();
        self.statistics.peak_mapped_bytes = self
            .statistics
            .peak_mapped_bytes
            .max(self.statistics.mapped_bytes);
        self.poisoned = false;
        Ok(())
    }
}

impl super::executor::Executor {
    pub(crate) fn expert_statistics(&self) -> Option<Statistics> {
        self.expert_cache.borrow().as_ref().map(|p| p.statistics)
    }
    pub(super) fn ensure_experts(&self, kernel: &Kernel, values: &[Value]) -> Result<()> {
        if let Some(pager) = self.expert_cache.borrow_mut().as_mut() {
            pager.ensure(&self.session, kernel, values)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn victim_selection_preserves_the_entire_current_union() {
        let mut q = Queue::default();
        for i in 0..4 {
            q.insert((0, i), i as u64);
        }
        let pinned = BTreeSet::from([(0, 0), (0, 1), (0, 3)]);
        assert_eq!(q.victim(&pinned), Some((0, 2)));
        q.insert((0, 2), 7);
        assert_eq!(q.keys.len(), 4);
        assert_eq!(q.order.len(), 4);
        assert_eq!(q.victim(&q.keys.keys().copied().collect()), None);
        assert!(q.remove((0, 2)));
        assert!(!q.remove((0, 2)));
    }
}
