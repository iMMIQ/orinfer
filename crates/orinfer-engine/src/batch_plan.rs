//! Bounded cache of model-owned packet commands and owner-specific state bindings.
use crate::{
    artifact::Result,
    execution::{BufferView, Invocation},
    model::Operation,
};
use orinfer_model_sdk::{
    architecture::{BatchSegment, ImageGrid},
    packet::{BatchOperation, BatchPlan, StateBindings},
};
use orinfer_models::Adapter;
use std::{
    cell::RefCell,
    collections::{BTreeSet, VecDeque},
    rc::Rc,
};

type CachedBatch = (Rc<Vec<Invocation>>, Rc<StateBindings>);

struct Entry {
    operation: BatchOperation,
    segments: Vec<BatchSegment>,
    plan: Option<Commands>,
    bindings: Rc<StateBindings>,
    bytes: usize,
}

#[derive(Clone)]
struct Commands {
    plan: Rc<Vec<Invocation>>,
    bytes: usize,
    owner_independent: bool,
}

pub(crate) struct BatchPlanCache {
    adapter: Box<dyn Adapter>,
    indirect_packets: bool,
    entries: RefCell<VecDeque<Entry>>,
}

impl BatchPlanCache {
    pub(crate) fn new(adapter: Box<dyn Adapter>, indirect_packets: bool) -> Self {
        Self {
            adapter,
            indirect_packets,
            entries: RefCell::new(VecDeque::new()),
        }
    }

    #[cfg(test)]
    pub(crate) fn batch_plan(&self, segments: &[BatchSegment]) -> Result<Vec<Invocation>> {
        self.cached_batch(segments, BatchOperation::Ordinary)
            .map(|(plan, _)| (*plan).clone())
    }

    pub(crate) fn supports_batch(
        &self,
        segments: &[BatchSegment],
        operation: BatchOperation,
    ) -> bool {
        self.batch_operation(segments, operation, false).is_ok()
    }

    pub(crate) fn cached_batch(
        &self,
        segments: &[BatchSegment],
        operation: BatchOperation,
    ) -> Result<CachedBatch> {
        let mut entries = self.entries.borrow_mut();
        if let Some(index) = entries.iter().position(|entry| {
            entry.operation == operation && entry.segments == segments && entry.plan.is_some()
        }) {
            let entry = entries.remove(index).expect("cached batch");
            let result = (
                entry.plan.as_ref().unwrap().plan.clone(),
                entry.bindings.clone(),
            );
            entries.push_back(entry);
            return Ok(result);
        }
        let shaped = self
            .indirect_packets
            .then(|| {
                entries.iter().find_map(|entry| {
                    if entry.operation == operation
                        && entry
                            .segments
                            .iter()
                            .map(|s| s.tokens)
                            .eq(segments.iter().map(|s| s.tokens))
                    {
                        entry
                            .plan
                            .as_ref()
                            .filter(|commands| commands.owner_independent)
                            .cloned()
                    } else {
                        None
                    }
                })
            })
            .flatten();
        let (plan, bindings) = if let Some(plan) = shaped {
            let (_, bindings) = self.batch_operation(segments, operation, false)?;
            (plan, bindings)
        } else {
            let (plan, bindings) = self.batch_operation(segments, operation, true)?;
            let commands = Commands {
                bytes: plan.iter().map(invocation_bytes).sum(),
                owner_independent: plan.iter().all(|call| call.sequence.is_none()),
                plan: Rc::new(plan),
            };
            (commands, bindings)
        };
        let bytes = plan.bytes;
        let result = plan.plan.clone();
        let bindings = Rc::new(bindings);
        Self::retain(
            &mut entries,
            Entry {
                operation,
                segments: segments.to_vec(),
                plan: Some(plan),
                bindings: bindings.clone(),
                bytes,
            },
        );
        Ok((result, bindings))
    }

    pub(crate) fn state_bindings(&self, segments: &[BatchSegment]) -> Result<Rc<StateBindings>> {
        let mut entries = self.entries.borrow_mut();
        let operation = BatchOperation::Ordinary;
        if let Some(index) = entries
            .iter()
            .position(|entry| entry.operation == operation && entry.segments == segments)
        {
            let entry = entries.remove(index).expect("cached state bindings");
            let bindings = entry.bindings.clone();
            entries.push_back(entry);
            return Ok(bindings);
        }
        let (_, bindings) = self.batch_operation(segments, operation, false)?;
        let bindings = Rc::new(bindings);
        Self::retain(
            &mut entries,
            Entry {
                operation,
                segments: segments.to_vec(),
                plan: None,
                bindings: bindings.clone(),
                bytes: 0,
            },
        );
        Ok(bindings)
    }

    fn retain(entries: &mut VecDeque<Entry>, mut entry: Entry) {
        const BUDGET: usize = 16 << 20;
        entry.bytes += entry.bindings.len() * std::mem::size_of::<Option<(usize, String)>>()
            + entry
                .bindings
                .iter()
                .flatten()
                .map(|(_, name)| name.len())
                .sum::<usize>();
        if entry.bytes <= BUDGET {
            entries
                .retain(|old| old.operation != entry.operation || old.segments != entry.segments);
            while entries.len() >= 32
                || entries.iter().map(|old| old.bytes).sum::<usize>() + entry.bytes > BUDGET
            {
                entries.pop_front();
            }
            entries.push_back(entry);
        }
    }

    pub(crate) fn batch_operation(
        &self,
        segments: &[BatchSegment],
        operation: BatchOperation,
        include_plan: bool,
    ) -> Result<BatchPlan> {
        if segments.is_empty()
            || segments.len() > 128
            || segments
                .iter()
                .any(|s| !(1..=262144).contains(&s.tokens) || s.slot == usize::MAX)
        {
            return Err("Invalid batch segment geometry".into());
        }
        let result = self
            .adapter
            .batch_operation(segments, operation, include_plan)?;
        if !include_plan && !result.0.is_empty() {
            return Err("Bindings-only packet returned execution commands".into());
        }
        Ok(result)
    }

    pub(crate) fn prefetch_inputs(&self, tokens: &[u32], history: &[u32]) {
        if (1..=4096).contains(&tokens.len()) {
            self.adapter
                .prefetch_inputs(tokens, &history[history.len().saturating_sub(2)..]);
        }
    }

    pub(crate) fn prepare_inputs(
        &self,
        program: &str,
        tokens: &[u32],
        history: &[u32],
        mut upload: impl FnMut(&str, &[u8]) -> Result<()>,
    ) -> Result<()> {
        if !(1..=4096).contains(&tokens.len()) || history.len() > 262144 || program.len() > 256 {
            return Err("Invalid model input preparation extent".into());
        }
        let uploads = self.adapter.prepare(program, tokens, history)?;
        let mut seen = BTreeSet::new();
        if uploads.len() > 16
            || uploads
                .iter()
                .any(|(name, data)| data.len() > 128 * 1024 * 1024 || !seen.insert(name))
        {
            return Err("Duplicate or oversized input upload".into());
        }
        for (name, data) in uploads {
            upload(&name, &data)?;
        }
        Ok(())
    }

    pub(crate) fn visual_layout(
        &self,
        tokens: &[u32],
        images: &[crate::vision::ImageInput],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        visual_layout(self.adapter.as_ref(), tokens, images, capacity)
    }
}

fn invocation_bytes(call: &Invocation) -> usize {
    let names = match &call.operation {
        Operation::Kernel { name } => name.len(),
        Operation::Copy {
            source,
            destination,
            ..
        } => source.len() + destination.len(),
        Operation::Zero { destination, .. } => destination.len(),
    };
    std::mem::size_of::<Invocation>()
        + names
        + call
            .views
            .iter()
            .map(|(name, view)| {
                std::mem::size_of::<(String, BufferView)>() + 32 + name.len() + view.buffer.len()
            })
            .sum::<usize>()
        + call.launch.as_ref().map_or(0, |launch| {
            launch.arguments.len() * std::mem::size_of::<(usize, i32)>()
        })
}

pub(crate) fn visual_layout(
    adapter: &dyn Adapter,
    tokens: &[u32],
    images: &[crate::vision::ImageInput],
    capacity: usize,
) -> Result<(Vec<i32>, Vec<u32>)> {
    if tokens.len() > capacity || capacity > 262144 || images.len() > 4096 {
        return Err("Invalid visual layout extent".into());
    }
    let grids: Vec<_> = images
        .iter()
        .map(|image| ImageGrid {
            grid_height: image.grid_height,
            grid_width: image.grid_width,
        })
        .collect();
    let result = adapter.visual(tokens, &grids, capacity)?;
    if result.0.len() != capacity
        || result.1.len() != capacity.checked_mul(3).ok_or("Visual size overflow")?
    {
        return Err("Invalid visual position geometry".into());
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PacketAdapter;

    impl Adapter for PacketAdapter {
        fn batch(&self, segments: &[BatchSegment], include_plan: bool) -> Result<BatchPlan> {
            self.batch_operation(segments, BatchOperation::Ordinary, include_plan)
        }

        fn batch_operation(
            &self,
            segments: &[BatchSegment],
            operation: BatchOperation,
            include_plan: bool,
        ) -> Result<BatchPlan> {
            let plan = if include_plan {
                vec![Invocation {
                    operation: Operation::Kernel {
                        name: format!("{operation:?}"),
                    },
                    sequence: None,
                    views: Default::default(),
                    launch: None,
                }]
            } else {
                vec![]
            };
            let bindings = segments
                .iter()
                .map(|s| Some((s.slot, "state".into())))
                .collect();
            Ok((plan, bindings))
        }
    }

    #[test]
    fn shape_reuse_preserves_commands_and_rebinds_each_owner_in_order() {
        let cache = BatchPlanCache::new(Box::new(PacketAdapter), true);
        let owners = [2, 7].map(|slot| BatchSegment { slot, tokens: 1 });
        let reversed = [owners[1], owners[0]];
        let (plan, bindings) = cache
            .cached_batch(&owners, BatchOperation::Ordinary)
            .unwrap();
        let (same_shape, swapped) = cache
            .cached_batch(&reversed, BatchOperation::Ordinary)
            .unwrap();
        assert!(Rc::ptr_eq(&plan, &same_shape));
        assert_eq!(bindings[0], Some((2, "state".into())));
        assert_eq!(swapped[0], Some((7, "state".into())));
        assert!(Rc::ptr_eq(
            &swapped,
            &cache.state_bindings(&reversed).unwrap()
        ));
        let (verification, _) = cache
            .cached_batch(&reversed, BatchOperation::Verification)
            .unwrap();
        assert!(!Rc::ptr_eq(&plan, &verification));
        assert!(cache.cached_batch(&[], BatchOperation::Ordinary).is_err());
    }
}
