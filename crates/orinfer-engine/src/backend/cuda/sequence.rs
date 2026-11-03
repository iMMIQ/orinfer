//! Request arenas share immutable weights and workspace. Switching an arena
//! changes host bindings, never copies recurrent state or KV payloads.
use super::{Argument, Handle, Result, Value, check, executor::Executor, ptr, sys};
use crate::{execution::Invocation, model::Operation};
use std::collections::BTreeMap;
use std::time::Instant;

// Count nodes as well as entries: a large batch graph has thousands of nodes.
// Oversized plans retain direct execution rather than exceeding this budget.
const BATCH_GRAPH_NODES: usize = 12288;

#[derive(Clone, Copy)]
struct BatchOperations<'a> {
    calls: &'a [Invocation],
    retained: Option<&'a std::rc::Rc<Vec<Invocation>>>,
}
impl BatchOperations<'_> {
    fn matches(&self, cached: &std::rc::Rc<Vec<Invocation>>) -> bool {
        self.retained
            .is_some_and(|plan| std::rc::Rc::ptr_eq(plan, cached))
            || cached.as_slice() == self.calls
    }
    fn retain(&self) -> std::rc::Rc<Vec<Invocation>> {
        self.retained
            .cloned()
            .unwrap_or_else(|| std::rc::Rc::new(self.calls.to_vec()))
    }
}

fn packet_graph_rows(
    operation: orinfer_model_sdk::abi::BatchOperation,
    segments: &[(usize, usize)],
) -> usize {
    if operation == orinfer_model_sdk::abi::BatchOperation::Commit {
        segments.len()
    } else {
        segments.iter().map(|(_, tokens)| tokens).sum()
    }
}

fn fixed_resize_cost(current: usize, needed: usize, limited: bool) -> usize {
    if current < needed {
        needed - current
    } else if limited && current - needed >= 65536 {
        // A shrink replaces storage and invalidates captures too. Prefer an
        // already fitting idle arena before resizing another recording's slot.
        needed
    } else {
        0
    }
}

/// A shared piece can cut through a certified branch. Only complete regions
/// may fork. Contiguous helpers replacing one row slot share its boundaries.
fn parallel_regions(
    operations: &[Invocation],
    configured: &[[String; 4]],
    slots: &BTreeMap<String, String>,
) -> Result<Vec<[usize; 4]>> {
    let mut positions = BTreeMap::new();
    let mut ambiguous = std::collections::BTreeSet::new();
    for (i, operation) in operations.iter().enumerate() {
        if let Operation::Kernel { name } = &operation.operation {
            let slot = slots.get(name).unwrap_or(name).as_str();
            if let Some((_, last)) = positions.get_mut(slot) {
                if *last + 1 != i {
                    // Private stages may repeat per owner without making
                    // unrelated shared branch boundaries ambiguous.
                    ambiguous.insert(slot);
                }
                *last = i;
            } else {
                positions.insert(slot, (i, i));
            }
        }
    }
    let mut regions = Vec::new();
    for region in configured {
        if region.iter().any(|name| ambiguous.contains(name.as_str())) {
            continue;
        }
        let [Some(fork), Some(start), Some(last), Some(join)] =
            region.each_ref().map(|name| positions.get(name.as_str()))
        else {
            continue;
        };
        let indices = [fork.0, start.0, last.1, join.0];
        let [fork, start, last, join] = indices;
        if !(fork < start && start <= last && last + 1 == join) {
            return Err("Invalid fork/join region".into());
        }
        regions.push(indices);
    }
    regions.sort();
    if regions.windows(2).any(|p| p[0][3] >= p[1][0]) {
        return Err("Overlapping fork/join regions".into());
    }
    Ok(regions)
}

fn batch_graph_victim(
    cache: &super::executor::BatchGraphCache,
    incoming: usize,
) -> Option<super::executor::BatchGraphKey> {
    if cache.len() < 16
        && cache.values().map(|(_, _, nodes)| nodes).sum::<usize>() + incoming <= BATCH_GRAPH_NODES
    {
        return None;
    }
    cache
        .iter()
        .min_by_key(|(_, (_, last, _))| *last)
        .map(|(key, _)| key.clone())
}

#[derive(Default)]
pub(super) struct Sequence {
    pub addresses: BTreeMap<String, u64>,
    pub sizes: BTreeMap<String, usize>,
    pub reservations: super::virtual_memory::Reservations,
    pub graphs: BTreeMap<String, Handle>,
    pub leased: bool,
    pub quarantined: bool,
    fixed_allocation: Option<u64>,
    fixed_slices: std::collections::BTreeSet<String>,
}
impl Sequence {
    /// Mark a load-time buffer as an interior slice of one shared allocation
    /// so lease growth and idle reclaim never free its address.
    pub(crate) fn mark_fixed_slice(&mut self, name: &str) {
        self.fixed_slices.insert(name.to_string());
    }
}

pub(super) struct IdleSelection {
    limits: BTreeMap<String, usize>,
    slot: Option<usize>,
}

impl Executor {
    pub(crate) fn active_request_slot(&self) -> Option<usize> {
        self.sequences
            .get(self.active_sequence)
            .filter(|s| s.leased && !s.quarantined)
            .map(|_| self.active_sequence)
    }
    #[cfg(test)]
    pub(crate) fn set_graph_mode_for_test(
        &mut self,
        mode: crate::execution::CudaGraphMode,
    ) -> crate::execution::CudaGraphMode {
        std::mem::replace(&mut self.cuda_graph, mode)
    }
    /// Bind each source arena and a disjoint transient scratch row. No request
    /// address is supplied by a caller outside this retained executor.
    pub(crate) fn upload_scratch_rows(
        &self,
        table: &str,
        scratch: &str,
        stride: usize,
        source: &str,
        source_bytes: usize,
        slots: &[usize],
    ) -> Result<()> {
        let bytes = stride
            .checked_mul(slots.len())
            .ok_or("Scratch rows overflow")?;
        if slots.is_empty()
            || slots.len() > 128
            || stride == 0
            || !self.pointers.is_private(source)
            || slots
                .len()
                .checked_mul(16)
                .is_none_or(|n| n > self.sizes.get(table).copied().unwrap_or(0))
        {
            return Err("Invalid scratch row table".into());
        }
        self.ensure_workspace(scratch, bytes)?;
        let mut addresses = Vec::with_capacity(slots.len() * 16);
        for (row, &slot) in slots.iter().enumerate() {
            let arena = self
                .sequences
                .get(slot)
                .filter(|s| s.leased && !s.quarantined)
                .ok_or("Scratch rows require a live arena")?;
            if arena.sizes.get(source).is_none_or(|&n| n < source_bytes) {
                return Err("Scratch row source exceeds arena".into());
            }
            addresses.extend_from_slice(
                &arena
                    .addresses
                    .get(source)
                    .ok_or("Missing scratch row source")?
                    .to_le_bytes(),
            );
            let address = self.pointers[scratch]
                .checked_add((row * stride) as u64)
                .ok_or("Scratch row address overflow")?;
            addresses.extend_from_slice(&address.to_le_bytes());
        }
        self.upload_bytes(table, &addresses)
    }

    /// Bind actual rows to an AOT template. Unseen lengths only specialize
    /// CPU launch parameters; they do not introduce modules or online compilation.
    pub(crate) fn submit_program_rows(
        &self,
        name: &str,
        rows: usize,
        phase: crate::execution::ExecutionPhase,
    ) -> Result<()> {
        let direct = self.direct.as_ref().ok_or("Missing row program bindings")?;
        let Some(template) = direct
            .row_execution
            .as_ref()
            .filter(|s| s.programs.contains(name))
        else {
            return self.submit_program(name, phase);
        };
        self.ensure_kv(name)?;
        let operations = {
            let mut cache = direct.row_cache.borrow_mut();
            if let Some(index) = cache.iter().position(|(n, m, _)| n == name && *m == rows) {
                let entry = cache.remove(index).expect("cached row program");
                let operations = entry.2.clone();
                cache.push_back(entry);
                operations
            } else {
                let program = direct.programs.get(name).ok_or("Missing row template")?;
                let operations = std::rc::Rc::new(template.specialize(
                    name,
                    program,
                    &direct.row_kernels,
                    rows,
                )?);
                while cache.len() >= 128
                    || cache.iter().map(|(_, _, ops)| ops.len()).sum::<usize>() + operations.len()
                        > 65536
                {
                    if cache.pop_front().is_none() {
                        break;
                    }
                }
                if operations.len() <= 65536 {
                    cache.push_back((name.into(), rows, operations.clone()));
                }
                operations
            }
        };
        self.ensure_invocation_kernels(&operations)?;
        for invocation in operations.iter() {
            match &invocation.operation {
                Operation::Copy {
                    destination, bytes, ..
                }
                | Operation::Zero { destination, bytes } => {
                    self.ensure_writable_range(destination, 0, *bytes)?;
                }
                _ => {}
            }
        }
        if self.cuda_graph.uses_graph(phase) {
            self.invoke_shared_pieces(&operations, true)
        } else {
            for invocation in operations.iter() {
                self.invoke(invocation)?;
            }
            Ok(())
        }
    }

    /// Upload an architecture-supplied pointer table. Addresses are derived only
    /// from retained, leased arenas; null entries are inactive lanes. The upload
    /// finishes before capture/replay and cannot change arena ownership.
    pub(crate) fn upload_sequence_addresses(
        &self,
        name: &str,
        bindings: std::rc::Rc<orinfer_model_sdk::abi::StateBindings>,
    ) -> Result<()> {
        let bytes = self
            .state_table
            .borrow_mut()
            .resolve(bindings, &self.sequences)?;
        self.upload_bytes(name, &bytes)
    }
    pub(crate) fn upload_sequence_address_ranges(
        &self,
        bindings: std::rc::Rc<orinfer_model_sdk::abi::StateBindings>,
        tables: &[(&str, std::ops::Range<usize>)],
    ) -> Result<()> {
        let bytes = self
            .state_table
            .borrow_mut()
            .resolve(bindings, &self.sequences)?;
        for (name, range) in tables {
            let start = range
                .start
                .checked_mul(8)
                .ok_or("Address table range overflow")?;
            let end = range
                .end
                .checked_mul(8)
                .ok_or("Address table range overflow")?;
            self.upload_bytes(
                name,
                bytes
                    .get(start..end)
                    .ok_or("Address table range exceeds bindings")?,
            )?;
        }
        Ok(())
    }
    /// Read controls from leased arenas without rebinding the active sequence.
    /// All producers and copies share the worker stream; a cohort needs only
    /// one completion wait, bounded by the existing pinned readback allocation.
    pub(crate) fn read_sequence_controls<const N: usize>(
        &self,
        slots: &[usize],
        names: [&str; N],
    ) -> Result<Vec<[i32; N]>> {
        let values = self.read_sequence_vectors(slots, &names, 1)?;
        Ok(values
            .as_chunks::<N>()
            .0
            .iter()
            .map(|chunk| std::array::from_fn(|j| chunk[j] as i32))
            .collect())
    }
    /// Read equally sized control vectors for all owners with one synchronization
    /// per bounded staging slab, preserving owner/name/vector order.
    pub(crate) fn read_sequence_vectors(
        &self,
        slots: &[usize],
        names: &[&str],
        count: usize,
    ) -> Result<Vec<u32>> {
        let words = names
            .len()
            .checked_mul(count)
            .ok_or("Control size overflow")?;
        let bytes = words.checked_mul(4).ok_or("Control size overflow")?;
        if bytes == 0 || bytes > super::transfers::READBACK_BYTES {
            return Err("Invalid sequence control extent".into());
        }
        let total = slots
            .len()
            .checked_mul(words)
            .ok_or("Control vector overflow")?;
        let mut regions = Vec::with_capacity(slots.len() * names.len());
        for &slot in slots {
            let arena = self
                .sequences
                .get(slot)
                .filter(|s| s.leased && !s.quarantined)
                .ok_or("Controls require a live leased arena")?;
            for &name in names {
                if arena.sizes.get(name).is_none_or(|&n| n < count * 4) {
                    return Err("Sequence control exceeds allocation".into());
                }
                regions.push((
                    *arena
                        .addresses
                        .get(name)
                        .ok_or("Missing sequence control")?,
                    count * 4,
                ));
            }
        }
        let mut values = vec![0u32; total];
        let capacity = super::transfers::READBACK_BYTES / bytes;
        for (i, chunk) in values.chunks_mut(capacity * words).enumerate() {
            let start = i * capacity * names.len();
            let owners = chunk.len() / words;
            // SAFETY: Initialized u32 storage covers every checked source region;
            // readback waits for DMA completion before this storage is inspected.
            let raw = unsafe {
                std::slice::from_raw_parts_mut(chunk.as_mut_ptr().cast(), chunk.len() * 4)
            };
            self.session.transfers.borrow_mut().readback(
                self.session.stream,
                &regions[start..start + owners * names.len()],
                raw,
            )?;
        }
        Ok(values)
    }
    #[cfg(test)]
    pub(crate) fn private_buffer_sizes(&self) -> BTreeMap<String, usize> {
        let reservations = self.session.virtual_buffers.borrow();
        self.sequences[self.active_sequence]
            .sizes
            .iter()
            .map(|(n, &bytes)| {
                (
                    n.clone(),
                    reservations.get(n).map_or(bytes, |r| r.mapped.min(bytes)),
                )
            })
            .collect()
    }
    pub(crate) fn free_bytes(&self) -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: The worker owns the current context and both output scalars.
        unsafe {
            check(
                sys::cuMemGetInfo_v2(&mut free, &mut total),
                "sequence free memory",
            )?;
        }
        Ok(free)
    }
    pub(crate) fn request_fixed_bytes(&self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        let payload = self
            .sequence_specs
            .iter()
            .filter(|b| !self.sequence_strides.contains_key(&b.name))
            .try_fold(0usize, |sum, b| {
                sum.checked_add(limits.get(&b.name).copied().unwrap_or(b.bytes()?))
                    .ok_or_else(|| "Sequence size overflow".to_string())
            })?;
        payload
            .checked_add(self.fixed_sequence_layout(limits)?.padding_bytes)
            .ok_or("Sequence arena size overflow".into())
    }
    fn fixed_sequence_layout(
        &self,
        limits: &BTreeMap<String, usize>,
    ) -> Result<super::weight_arena::ArenaLayout> {
        super::weight_arena::ArenaLayout::new(self.sequence_specs.iter().filter(|b| {
            !self.sequence_strides.contains_key(&b.name) && !limits.contains_key(&b.name)
        }))
    }
    pub(crate) fn lease_sequence(&mut self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        let reused = self.idle_sequence(limits)?;
        let result = self.lease_sequence_inner(limits);
        self.idle_selection.get_mut().take();
        if result.is_err()
            && let Some(slot) = reused
        {
            self.sequences[slot].quarantined = true;
        }
        result
    }
    fn idle_sequence(&self, limits: &BTreeMap<String, usize>) -> Result<Option<usize>> {
        if let Some(cached) = self.idle_selection.borrow().as_ref()
            && cached.limits == *limits
        {
            return Ok(cached.slot);
        }
        let current = self.session.virtual_buffers.borrow();
        let mut best = None;
        for (slot, sequence) in self.sequences.iter().enumerate() {
            if sequence.leased || sequence.quarantined {
                continue;
            }
            let reservations = if slot == self.active_sequence {
                &*current
            } else {
                &sequence.reservations
            };
            let cost = self.sequence_specs.iter().try_fold(0usize, |sum, b| {
                let bytes = limits.get(&b.name).copied().unwrap_or(b.bytes()?);
                let additional = if let Some(&stride) = self.sequence_strides.get(&b.name) {
                    let r = &reservations[&b.name];
                    let extent = r.required_extent(bytes.div_ceil(stride))?;
                    if extent > r.bytes {
                        extent
                    } else {
                        extent.saturating_sub(r.mapped)
                    }
                } else {
                    fixed_resize_cost(sequence.sizes[&b.name], bytes, limits.contains_key(&b.name))
                };
                sum.checked_add(additional)
                    .ok_or("Idle sequence budget overflow".to_string())
            })?;
            if best.is_none_or(|(_, old)| cost < old) {
                best = Some((slot, cost));
            }
            if cost == 0 {
                break;
            }
        }
        let slot = best.map(|(slot, _)| slot);
        *self.idle_selection.borrow_mut() = Some(IdleSelection {
            limits: limits.clone(),
            slot,
        });
        Ok(slot)
    }
    fn lease_sequence_inner(&mut self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        for (name, &bytes) in limits {
            let capacity = self
                .sequence_specs
                .iter()
                .find(|b| b.name == *name)
                .ok_or("Unknown private buffer limit")?
                .bytes()?;
            if bytes == 0 || bytes > capacity {
                return Err("Invalid private buffer limit".into());
            }
        }
        if let Some(slot) = self.idle_sequence(limits)? {
            self.activate_sequence(slot)?;
            let needed: Vec<_> = self
                .sequence_specs
                .iter()
                .map(|b| {
                    Ok((
                        b.name.clone(),
                        limits.get(&b.name).copied().unwrap_or(b.bytes()?),
                    ))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .filter(|(n, b)| {
                    let old = self.sequences[slot].sizes[n];
                    fixed_resize_cost(old, *b, limits.contains_key(n)) != 0
                })
                .collect();
            if !needed.is_empty() {
                self.sync()?;
                let replace = needed.iter().any(|(name, bytes)| {
                    !self.sequence_strides.contains_key(name)
                        || *bytes > self.session.virtual_buffers.borrow()[name].bytes
                });
                if replace {
                    self.invalidate_current_graphs()?;
                }
                for (name, bytes) in needed {
                    if let Some(&stride) = self.sequence_strides.get(&name) {
                        if bytes <= self.session.virtual_buffers.get_mut()[&name].bytes {
                            self.sequences[slot].sizes.insert(name.clone(), bytes);
                            self.sizes.insert(name, bytes);
                            continue;
                        }
                        let replacement = super::virtual_memory::Reservation::reserve(
                            &self.session,
                            bytes,
                            stride,
                        )?;
                        let address = replacement.address;
                        let retired = format!("@retired/{slot}/{name}");
                        let old = self
                            .session
                            .virtual_buffers
                            .get_mut()
                            .insert(name.clone(), replacement)
                            .ok_or("Missing idle KV reservation")?;
                        self.session
                            .virtual_buffers
                            .get_mut()
                            .insert(retired.clone(), old);
                        self.sequences[slot].addresses.insert(name.clone(), address);
                        self.sequences[slot].sizes.insert(name.clone(), bytes);
                        self.pointers.bind_private(&name, address)?;
                        self.sizes.insert(name, bytes);
                        let old = self
                            .session
                            .virtual_buffers
                            .get_mut()
                            .get_mut(&retired)
                            .unwrap();
                        old.release_slabs(&self.session.driver)?;
                        // SAFETY: Idle arena, synchronized stream and invalidated
                        // graphs; retired ownership remains registered on failure.
                        unsafe {
                            check(
                                sys::cuMemAddressFree(old.address, old.bytes),
                                "free idle KV reservation",
                            )?;
                        }
                        self.session.virtual_buffers.get_mut().remove(&retired);
                        continue;
                    }
                    let mut address = 0;
                    // SAFETY: Idle arena; no captured graph can retain the old
                    // allocation. Transfer new ownership before initialization.
                    unsafe {
                        check(
                            sys::cuMemAlloc_v2(&mut address, bytes),
                            "grow request metadata",
                        )?;
                        self.session.buffers.push(address);
                        check(
                            sys::cuMemsetD8Async(address, 0, bytes, self.session.stream),
                            "initialize request metadata",
                        )?;
                        let old = self.sequences[slot].addresses[&name];
                        if old != 0 && !self.sequences[slot].fixed_slices.remove(&name) {
                            check(sys::cuMemFree_v2(old), "free idle metadata")?;
                            self.session.buffers.retain(|&p| p != old);
                        }
                    }
                    self.sequences[slot].addresses.insert(name.clone(), address);
                    self.sequences[slot].sizes.insert(name.clone(), bytes);
                    self.pointers.bind_private(&name, address)?;
                    self.sizes.insert(name, bytes);
                }
            }
            self.sequences[slot].leased = true;
            return Ok(slot);
        }
        self.sync()?;
        let mut sequence = Sequence::default();
        let mut allocated = Vec::new();
        let result = (|| -> Result<()> {
            let layout = self.fixed_sequence_layout(limits)?;
            let base = if layout.allocation_bytes > 0 {
                let mut owner = 0;
                // SAFETY: The session owns the allocation before initialization;
                // each aligned slice remains private to this request arena.
                unsafe {
                    check(
                        sys::cuMemAlloc_v2(&mut owner, layout.allocation_bytes),
                        "allocate request arena",
                    )?;
                    self.session.buffers.push(owner);
                    allocated.push(owner);
                    sequence.fixed_allocation = Some(owner);
                    check(
                        sys::cuMemsetD8Async(
                            owner,
                            0,
                            layout.allocation_bytes,
                            self.session.stream,
                        ),
                        "initialize request arena",
                    )?;
                }
                layout.base(owner)?
            } else {
                0
            };
            for b in &self.sequence_specs {
                let bytes = limits.get(&b.name).copied().unwrap_or(b.bytes()?);
                let address = if let Some(&offset) = layout.offsets.get(&b.name) {
                    sequence.fixed_slices.insert(b.name.clone());
                    base.checked_add(offset as u64)
                        .ok_or("Request arena address overflow")?
                } else if let Some(&stride) = self.sequence_strides.get(&b.name) {
                    let r =
                        super::virtual_memory::Reservation::reserve(&self.session, bytes, stride)?;
                    let address = r.address;
                    sequence.reservations.insert(b.name.clone(), r);
                    address
                } else {
                    let mut address = 0;
                    // SAFETY: Validated buffer extent; ownership is transferred before
                    // initialization or any subsequent fallible operation.
                    unsafe {
                        check(
                            sys::cuMemAlloc_v2(&mut address, bytes),
                            "allocate request state",
                        )?;
                        self.session.buffers.push(address);
                        allocated.push(address);
                        check(
                            sys::cuMemsetD8Async(address, 0, bytes, self.session.stream),
                            "initialize request state",
                        )?;
                    }
                    address
                };
                sequence.addresses.insert(b.name.clone(), address);
                sequence.sizes.insert(b.name.clone(), bytes);
            }
            Ok(())
        })();
        if let Err(error) = result {
            let failed: Vec<_> = sequence
                .reservations
                .into_iter()
                .map(|(name, r)| {
                    let key = format!("@failed/{name}");
                    self.session
                        .virtual_buffers
                        .get_mut()
                        .insert(key.clone(), r);
                    key
                })
                .collect();
            self.sync()?;
            // SAFETY: No kernel or graph has observed this incomplete arena.
            // Unmapped VMM reservations and ordinary allocations are owned here.
            unsafe {
                for key in failed {
                    let r = &self.session.virtual_buffers.get_mut()[&key];
                    check(
                        sys::cuMemAddressFree(r.address, r.bytes),
                        "rollback request reservation",
                    )?;
                    self.session.virtual_buffers.get_mut().remove(&key);
                }
                for address in allocated {
                    check(sys::cuMemFree_v2(address), "rollback request state")?;
                    self.session.buffers.retain(|&p| p != address);
                }
            }
            return Err(error);
        }
        sequence.leased = true;
        let slot = self.sequences.len();
        self.sequences.push(sequence);
        if let Err(error) = self.activate_sequence(slot) {
            self.sequences[slot].quarantined = true;
            return Err(error);
        }
        Ok(slot)
    }
    pub(crate) fn activate_sequence(&mut self, slot: usize) -> Result<()> {
        if slot >= self.sequences.len() {
            return Err("Unknown request arena".into());
        }
        if slot == self.active_sequence {
            return Ok(());
        }
        self.save_live_workspace()?;
        self.sync()?;
        let buffers = self.session.virtual_buffers.get_mut();
        for name in self.sequence_strides.keys() {
            if let Some(r) = buffers.remove(name) {
                self.sequences[self.active_sequence]
                    .reservations
                    .insert(name.clone(), r);
            }
        }
        buffers.append(&mut self.sequences[slot].reservations);
        self.sequences[self.active_sequence].graphs = std::mem::take(self.graphs.get_mut());
        *self.graphs.get_mut() = std::mem::take(&mut self.sequences[slot].graphs);
        self.active_sequence = slot;
        for (name, &address) in &self.sequences[slot].addresses {
            self.pointers.bind_private(name, address)?;
            self.sizes
                .insert(name.clone(), self.sequences[slot].sizes[name]);
        }
        Ok(())
    }
    fn invalidate_current_graphs(&mut self) -> Result<()> {
        // Arena address replacement must retire resolved tables as well as
        // graph arguments. KV page mapping preserves the reserved addresses.
        self.state_table.get_mut().invalidate();
        let keys: Vec<_> = self
            .batch_graphs
            .get_mut()
            .keys()
            .filter(|key| key.iter().any(|(slot, _)| *slot == self.active_sequence))
            .cloned()
            .collect();
        let mut statistics = self.batch_statistics.get();
        statistics.graph_invalidations += keys.len();
        self.batch_statistics.set(statistics);
        let mut execs: Vec<_> = std::mem::take(self.graphs.get_mut())
            .into_values()
            .collect();
        for key in keys {
            execs.push(self.batch_graphs.get_mut().remove(&key).unwrap().0);
        }
        self.destroy_graphs(&execs)
    }
    fn destroy_graphs(&self, execs: &[Handle]) -> Result<()> {
        let mut owned = self.session.graphs.borrow_mut();
        let mut index = 0;
        while index < owned.len() {
            if execs.contains(&owned[index].1) {
                let (graph, exec) = owned[index];
                // SAFETY: Idle stream, graphs removed from every execution map.
                unsafe {
                    check(sys::cuGraphExecDestroy(exec), "invalidate metadata graph")?;
                    owned[index].1 = std::ptr::null_mut();
                    check(sys::cuGraphDestroy(graph), "invalidate metadata capture")?;
                }
                owned.remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }
    /// Retire captures before destroying reusable buffers. Graphs belonging to
    /// active sequences remain valid; idle sequence state is still warm.
    pub(crate) fn reclaim_idle_graphs(&mut self) -> Result<()> {
        let has_idle_graphs = self.batch_graphs.borrow().keys().any(|key| {
            if key.shared.is_some() {
                self.sequences.iter().all(|s| !s.leased)
            } else {
                key.iter().all(|(slot, _)| !self.sequences[*slot].leased)
            }
        }) || self.sequences.iter().enumerate().any(|(slot, sequence)| {
            !sequence.leased
                && if slot == self.active_sequence {
                    !self.graphs.borrow().is_empty()
                } else {
                    !sequence.graphs.is_empty()
                }
        });
        if !has_idle_graphs {
            return Ok(());
        }
        self.sync()?;
        let keys: Vec<_> = self
            .batch_graphs
            .get_mut()
            .keys()
            .filter(|key| {
                if key.shared.is_some() {
                    self.sequences.iter().all(|s| !s.leased)
                } else {
                    key.iter().all(|(slot, _)| !self.sequences[*slot].leased)
                }
            })
            .cloned()
            .collect();
        let mut execs = Vec::new();
        for key in keys {
            execs.push(self.batch_graphs.get_mut().remove(&key).unwrap().0);
        }
        for (slot, sequence) in self.sequences.iter_mut().enumerate() {
            if !sequence.leased {
                let graphs = if slot == self.active_sequence {
                    self.graphs.get_mut()
                } else {
                    &mut sequence.graphs
                };
                execs.extend(std::mem::take(graphs).into_values());
            }
        }
        let mut statistics = self.batch_statistics.get();
        statistics.graph_evictions += execs.len();
        self.batch_statistics.set(statistics);
        self.destroy_graphs(&execs)
    }
    pub(crate) fn reserved_kv_bytes(&self, tokens: usize) -> Result<usize> {
        self.sequence_strides.keys().try_fold(0usize, |sum, name| {
            let r = self.session.virtual_buffers.borrow();
            sum.checked_add(
                r.get(name)
                    .ok_or("Missing live KV reservation")?
                    .required_extent(tokens)?,
            )
            .ok_or("KV reservation sum overflow".into())
        })
    }
    #[cfg(test)]
    pub(crate) fn resident_kv_bytes(&self) -> usize {
        self.session
            .virtual_buffers
            .borrow()
            .iter()
            .filter(|(n, _)| self.sequence_strides.contains_key(*n))
            .map(|(_, r)| r.mapped)
            .sum::<usize>()
            + self
                .sequences
                .iter()
                .flat_map(|s| s.reservations.values())
                .map(|r| r.mapped)
                .sum::<usize>()
    }
    #[cfg(test)]
    pub(crate) fn shared_kv_bytes(&self) -> usize {
        self.session
            .virtual_buffers
            .borrow()
            .iter()
            .filter(|(n, _)| self.sequence_strides.contains_key(*n))
            .map(|(_, v)| v.mapped.saturating_sub(v.physical_bytes()))
            .sum::<usize>()
            + self
                .sequences
                .iter()
                .flat_map(|s| s.reservations.values())
                .map(|v| v.mapped.saturating_sub(v.physical_bytes()))
                .sum::<usize>()
    }
    pub(crate) fn active_kv_bytes(&self) -> usize {
        // Shared pages are cache-owned. Retain the private replacement budget
        // until the request finishes, including worst-case COW writes.
        let current = if self.sequences[self.active_sequence].leased {
            self.session
                .virtual_buffers
                .borrow()
                .iter()
                .filter(|(name, _)| self.sequence_strides.contains_key(*name))
                .map(|(_, r)| r.physical_bytes())
                .sum()
        } else {
            0
        };
        current
            + self
                .sequences
                .iter()
                .filter(|s| s.leased)
                .flat_map(|s| s.reservations.values())
                .map(|r| r.physical_bytes())
                .sum::<usize>()
    }
    pub(crate) fn additional_kv_bytes(
        &self,
        tokens: usize,
        limits: &BTreeMap<String, usize>,
    ) -> Result<usize> {
        let Some(slot) = self.idle_sequence(limits)? else {
            return self.reserved_kv_bytes(tokens);
        };
        let sequence = &self.sequences[slot];
        let current = self.session.virtual_buffers.borrow();
        let buffers = if slot == self.active_sequence {
            &*current
        } else {
            &sequence.reservations
        };
        self.sequence_strides.keys().try_fold(0usize, |sum, name| {
            let r = buffers.get(name).ok_or("Missing idle KV reservation")?;
            let extent = r.required_extent(tokens)?;
            let additional = if extent > r.bytes {
                extent
            } else {
                extent.saturating_sub(r.mapped)
            };
            sum.checked_add(additional)
                .ok_or("Additional KV budget overflow".into())
        })
    }
    pub(crate) fn pending_prefill_workspace_bytes(&self, tokens: usize) -> Result<usize> {
        let reservations = self.session.virtual_buffers.borrow();
        self.prefill_workspace
            .iter()
            .filter_map(|n| reservations.get(n))
            .try_fold(0usize, |sum, r| {
                sum.checked_add(r.extent(tokens)?.saturating_sub(r.mapped))
                    .ok_or("Prefill workspace budget overflow".into())
            })
    }
    pub(crate) fn additional_state_bytes(&self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        if let Some(slot) = self.idle_sequence(limits)? {
            let s = &self.sequences[slot];
            self.sequence_specs
                .iter()
                .filter(|b| !self.sequence_strides.contains_key(&b.name))
                .try_fold(0usize, |sum, b| {
                    sum.checked_add(
                        limits
                            .get(&b.name)
                            .copied()
                            .unwrap_or(b.bytes()?)
                            .saturating_sub(s.sizes[&b.name]),
                    )
                    .ok_or("Additional state budget overflow".into())
                })
        } else {
            self.request_fixed_bytes(limits)
        }
    }
    /// Keep one warm idle arena; reclaim the others only under admission
    /// pressure. Empty ordinary buffers are rebuilt before their next lease.
    pub(crate) fn reclaim_idle_state(&mut self) -> Result<()> {
        self.idle_selection.get_mut().take();
        let idle: Vec<_> = self
            .sequences
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.leased && !s.quarantined)
            .map(|(i, _)| i)
            .collect();
        let keep = idle.iter().copied().max_by_key(|&slot| {
            self.sequence_specs
                .iter()
                .filter(|b| !self.sequence_strides.contains_key(&b.name))
                .map(|b| self.sequences[slot].sizes[&b.name])
                .sum::<usize>()
        });
        for &slot in &idle {
            // Admission retries must not synchronize and switch arenas that
            // were already reclaimed. Keep the warm fixed buffers unchanged.
            let release_fixed = Some(slot) != keep
                && self.sequence_specs.iter().any(|b| {
                    !self.sequence_strides.contains_key(&b.name)
                        && self.sequences[slot].sizes[&b.name] != 0
                });
            let release_kv = {
                let current = self.session.virtual_buffers.borrow();
                let buffers = if slot == self.active_sequence {
                    &*current
                } else {
                    &self.sequences[slot].reservations
                };
                self.sequence_strides
                    .keys()
                    .any(|name| buffers[name].mapped != 0)
            };
            if !release_fixed && !release_kv {
                continue;
            }
            self.activate_sequence(slot)?;
            self.sync()?;
            for (name, buffer) in self.session.virtual_buffers.get_mut().iter_mut() {
                if self.sequence_strides.contains_key(name) {
                    buffer.release_slabs(&self.session.driver)?;
                }
            }
            if Some(slot) == keep {
                continue;
            }
            let names: Vec<_> = self
                .sequence_specs
                .iter()
                .filter(|b| {
                    !self.sequence_strides.contains_key(&b.name)
                        && self.sequences[slot].sizes[&b.name] != 0
                })
                .map(|b| b.name.clone())
                .collect();
            if names.is_empty() {
                continue;
            }
            self.activate_sequence(slot)?;
            self.sync()?;
            self.invalidate_current_graphs()?;
            self.sequences[slot].quarantined = true;
            for name in names {
                let address = self.sequences[slot].addresses[&name];
                if !self.sequences[slot].fixed_slices.contains(&name) {
                    // SAFETY: Unleased arena, synchronized stream, invalidated
                    // graphs. Ownership is removed immediately after successful free.
                    unsafe {
                        check(sys::cuMemFree_v2(address), "reclaim idle request state")?;
                    }
                    self.session.buffers.retain(|&p| p != address);
                }
                self.sequences[slot].addresses.insert(name.clone(), 0);
                self.sequences[slot].sizes.insert(name.clone(), 0);
                self.pointers.bind_private(&name, 0)?;
                self.sizes.insert(name, 0);
            }
            if let Some(owner) = self.sequences[slot].fixed_allocation {
                // SAFETY: Every slice in this unleased arena has been retired;
                // no graph or pending stream operation can retain its owner.
                unsafe {
                    check(sys::cuMemFree_v2(owner), "reclaim idle request arena")?;
                }
                self.session.buffers.retain(|&p| p != owner);
                self.sequences[slot].fixed_allocation = None;
                self.sequences[slot].fixed_slices.clear();
            }
            self.sequences[slot].quarantined = false;
        }
        Ok(())
    }
    pub(crate) fn release_sequence(&mut self, slot: usize, resets: &[String]) -> Result<()> {
        self.idle_selection.get_mut().take();
        let result = self
            .activate_sequence(slot)
            .and_then(|()| self.reset_sequence_reusing_pages(resets));
        if let Err(error) = result {
            self.sequences[slot].quarantined = true;
            return Err(error);
        }
        self.sequences[slot].quarantined = false;
        self.sequences[slot].leased = false;
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn prepare_reference_sequence(&mut self) -> Result<()> {
        if self.sequences.iter().any(|s| s.leased) {
            return Err("Complete active requests before using whole-request generation".into());
        }
        let slot = self.lease_sequence(&BTreeMap::new())?;
        self.sequences[slot].leased = false;
        Ok(())
    }
    pub(crate) fn ensure_sequence_program(&mut self, slot: usize, program: &str) -> Result<()> {
        self.activate_sequence(slot)?;
        self.ensure_kv(program)
    }
    pub(crate) fn ensure_sequence_program_at(
        &mut self,
        slot: usize,
        program: &str,
        position: usize,
        tokens: usize,
    ) -> Result<()> {
        // Mapped, private KV pages can be checked without rebinding the arena.
        // Only growth or COW needs the slower reservation-switching path.
        if let Some(growth) = self.growth.get(program) {
            if tokens == 0
                || tokens > growth.tokens
                || (growth.length.is_none() && tokens != growth.tokens)
            {
                return Err("Invalid logical KV write length".into());
            }
            let end = position.checked_add(tokens).ok_or("KV position overflow")?;
            let current = self.session.virtual_buffers.borrow();
            if current.is_empty() {
                return Ok(());
            }
            let arena = self
                .sequences
                .get(slot)
                .filter(|s| s.leased && !s.quarantined)
                .ok_or("KV preparation requires a live arena")?;
            let reservations = if slot == self.active_sequence {
                &*current
            } else {
                &arena.reservations
            };
            let mut ready = true;
            for name in &growth.buffers {
                // Growth can include shared, context-sized workspace as well
                // as private KV. Switching owners only swaps private mappings.
                let owner = if self.sequence_strides.contains_key(name) {
                    reservations
                } else {
                    &*current
                };
                let b = owner
                    .get(name)
                    .ok_or_else(|| format!("Missing demand KV buffer {name} for {program}"))?;
                let start = position.checked_mul(b.stride).ok_or("KV write overflow")?;
                let bytes = tokens.checked_mul(b.stride).ok_or("KV write overflow")?;
                ready &= b.extent(end)? <= b.mapped && b.writable_bytes(start, bytes)? == 0;
            }
            if ready {
                return Ok(());
            }
        }
        self.activate_sequence(slot)?;
        self.ensure_kv_extent(program, position, tokens)
    }
    pub(crate) fn prepare_batch_workspace(&mut self) -> Result<()> {
        self.save_live_workspace()
    }
    /// Upload to stable private storage without changing the current bindings.
    /// Virtual storage keeps the regular COW/mapping and staging safeguards.
    pub(crate) fn upload_sequence_bytes(
        &mut self,
        slot: usize,
        name: &str,
        bytes: &[u8],
    ) -> Result<()> {
        let arena = self
            .sequences
            .get(slot)
            .filter(|s| s.leased && !s.quarantined)
            .ok_or("Input requires a live leased arena")?;
        if let Some(&address) = arena.addresses.get(name) {
            if bytes.len() > arena.sizes[name] {
                return Err("Sequence input exceeds allocation".into());
            }
            if !self.sequence_strides.contains_key(name) {
                return self.session.transfers.borrow_mut().upload(
                    self.session.stream,
                    address,
                    bytes,
                );
            }
            self.activate_sequence(slot)?;
        }
        self.upload_bytes(name, bytes)
    }
    pub(crate) fn upload_sequence_ids(
        &mut self,
        slot: usize,
        name: &str,
        ids: &[u32],
    ) -> Result<()> {
        let bytes = ids.len().checked_mul(4).ok_or("Sequence input overflow")?;
        // SAFETY: Initialized u32 storage is copied into owned pinned staging
        // before upload_sequence_bytes returns; no host pointer is retained.
        let raw = unsafe { std::slice::from_raw_parts(ids.as_ptr().cast(), bytes) };
        self.upload_sequence_bytes(slot, name, raw)
    }
    /// Copy into a live owner's private buffer on the existing stream.
    /// Fixed buffers keep their stable addresses without rebinding the arena;
    /// demand-mapped destinations retain the regular mapping/COW safeguards.
    pub(crate) fn copy_to_sequence_async(
        &mut self,
        source: &str,
        source_offset: usize,
        slot: usize,
        destination: &str,
        bytes: usize,
    ) -> Result<()> {
        let arena = self
            .sequences
            .get(slot)
            .filter(|s| s.leased && !s.quarantined)
            .ok_or("Copy requires a live leased arena")?;
        if !arena.addresses.contains_key(destination) {
            return Err("Copy destination requires private storage".into());
        }
        if self.sequence_strides.contains_key(destination) {
            self.activate_sequence(slot)?;
            self.ensure_writable_range(destination, 0, bytes)?;
        }
        self.invoke(&Invocation {
            operation: crate::model::Operation::Copy {
                source: source.into(),
                destination: destination.into(),
                bytes,
            },
            sequence: Some(slot),
            launch: None,
            views: BTreeMap::from([(
                source.into(),
                crate::execution::BufferView {
                    buffer: source.into(),
                    offset: source_offset,
                },
            )]),
        })
    }
    fn address(
        &self,
        name: &str,
        sequence: Option<usize>,
        views: &BTreeMap<String, crate::execution::BufferView>,
    ) -> Result<u64> {
        let view = views.get(name);
        let target = view.map_or(name, |v| v.buffer.as_str());
        let offset = view.map_or(0, |v| v.offset);
        let size = sequence
            .and_then(|slot| self.sequences.get(slot)?.sizes.get(target))
            .or_else(|| self.sizes.get(target))
            .copied()
            .ok_or("Unknown invocation buffer")?;
        if offset >= size {
            return Err(format!("{target}: invocation offset exceeds buffer"));
        }
        let address = sequence
            .and_then(|slot| self.sequences.get(slot)?.addresses.get(target))
            .or_else(|| self.pointers.get(target))
            .copied()
            .ok_or("Unknown invocation address")?;
        address
            .checked_add(offset as u64)
            .ok_or("Invocation address overflow".into())
    }
    fn invoke(&self, invocation: &Invocation) -> Result<()> {
        self.invoke_on(invocation, self.session.stream)
    }
    fn invoke_on(&self, invocation: &Invocation, stream: sys::CUstream) -> Result<()> {
        let address = |name: &str| self.address(name, invocation.sequence, &invocation.views);
        match &invocation.operation {
            Operation::Kernel { name } => {
                let direct = self.direct.as_ref().ok_or("Missing direct bindings")?;
                let kernels = direct.kernels.borrow();
                let k = kernels
                    .get(name)
                    .ok_or_else(|| format!("Missing batch kernel {name}"))?;
                let mut values = k
                    .spec
                    .args
                    .iter()
                    .zip(&k.values)
                    .map(|(arg, value)| {
                        Ok(match arg {
                            Argument::Buffer { name } => Value::Pointer(address(name)?),
                            Argument::BufferSlice { name, offset } => {
                                Value::Pointer(address(name)? + *offset as u64)
                            }
                            _ => *value,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let dynamic_spec;
                let spec = if let Some(launch) = &invocation.launch {
                    dynamic_spec = {
                        let mut spec = k.spec.clone();
                        spec.grid = launch.grid;
                        spec
                    };
                    for &(index, value) in &launch.arguments {
                        if !matches!(k.spec.args.get(index), Some(Argument::I32 { .. })) {
                            return Err("Dynamic launch changed a non-i32 argument".into());
                        }
                        values[index] = Value::I32(value);
                    }
                    self.kernels.borrow().validate_launch(&dynamic_spec)?;
                    &dynamic_spec
                } else {
                    &k.spec
                };
                self.ensure_experts(spec, &values)?;
                super::launch_kernel(
                    spec,
                    k.function
                        .ok_or("Batch kernel was not resolved before submission")?,
                    &mut values,
                    &self.session.driver,
                    stream,
                )
            }
            Operation::Copy {
                source,
                destination,
                bytes,
            } => {
                self.check_view_range(source, *bytes, invocation)?;
                self.check_view_range(destination, *bytes, invocation)?;
                // SAFETY: Architecture views refer to owned buffers, checked byte
                // ranges, and are ordered with their producers on this stream.
                unsafe {
                    check(
                        sys::cuMemcpyDtoDAsync_v2(
                            address(destination)?,
                            address(source)?,
                            *bytes,
                            stream,
                        ),
                        "batch copy",
                    )
                }
            }
            Operation::Zero { destination, bytes } => {
                self.check_view_range(destination, *bytes, invocation)?;
                // SAFETY: Checked writable range, ordered on the owner stream.
                unsafe {
                    check(
                        sys::cuMemsetD8Async(address(destination)?, 0, *bytes, stream),
                        "batch zero",
                    )
                }
            }
        }
    }
    fn check_view_range(&self, name: &str, bytes: usize, invocation: &Invocation) -> Result<()> {
        let v = invocation.views.get(name);
        let target = v.map_or(name, |v| v.buffer.as_str());
        let writing = match &invocation.operation {
            Operation::Zero { destination, .. } | Operation::Copy { destination, .. } => {
                destination == name
            }
            _ => false,
        };
        if self.allocations.weights.addresses.contains_key(target) && writing {
            return Err("Invocation writes immutable weights".into());
        }
        let start = v.map_or(0, |v| v.offset);
        let size = invocation
            .sequence
            .and_then(|slot| self.sequences.get(slot)?.sizes.get(target))
            .or_else(|| self.sizes.get(target))
            .copied()
            .unwrap_or(0);
        if start.checked_add(bytes).is_none_or(|end| end > size) {
            return Err("Invocation range exceeds buffer".into());
        }
        Ok(())
    }
    fn capture(&self, operations: &[Invocation]) -> Result<Handle> {
        self.ensure_invocation_kernels(operations)?;
        if !self.parallel_regions.borrow().is_empty() {
            self.ensure_parallel_handles()?;
        }
        // SAFETY: All bindings and backing allocations outlive graph execution;
        // capture records work, and the worker is the stream's sole submitter.
        unsafe {
            check(
                sys::cuStreamBeginCapture_v2(
                    self.session.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "begin request capture",
            )?;
        }
        let result = self.capture_parallel_operations(operations);
        let mut graph = ptr::null_mut();
        // SAFETY: Always end capture, including a failed recorded operation.
        let end = unsafe {
            check(
                sys::cuStreamEndCapture(self.session.stream, &mut graph),
                "end request capture",
            )
        };
        if let Err(error) = result.and(end) {
            if !graph.is_null() {
                // SAFETY: This local failed capture is not owned elsewhere.
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(error);
        }
        let flags = match self.prioritize_parallel_graph(graph, operations) {
            Ok(flags) => flags,
            Err(error) => {
                // SAFETY: Local graph has not been instantiated or published.
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
                return Err(error);
            }
        };
        let mut exec = ptr::null_mut();
        // SAFETY: Graph is locally owned until transferred into Session.
        unsafe {
            if let Err(error) = check(
                sys::cuGraphInstantiateWithFlags(&mut exec, graph, flags),
                "instantiate request graph",
            ) {
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
            // SM87 has no device-side graph launch: every replay would otherwise
            // pay per-node command encoding. Upload once; replays then only
            // trigger execution. Upload failure frees the local pair.
            if let Err(error) = check(
                sys::cuGraphUpload(exec, self.session.stream),
                "upload request graph",
            ) {
                sys::cuGraphExecDestroy(exec);
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
        }
        self.session.graphs.borrow_mut().push((graph, exec));
        Ok(exec)
    }
    fn prioritize_parallel_graph(
        &self,
        graph: sys::CUgraph,
        operations: &[Invocation],
    ) -> Result<u64> {
        #[cfg(test)]
        let test_policy = self.parallel_priority.borrow();
        #[cfg(test)]
        let policy = test_policy.as_str();
        #[cfg(not(test))]
        let policy = "small_side";
        if policy.is_empty() || policy == "off" {
            return Ok(0);
        }
        let regions = parallel_regions(
            operations,
            &self.parallel_regions.borrow(),
            &self.row_kernel_slots,
        )?;
        if regions.is_empty() {
            return Ok(0);
        }
        let kernels = self
            .direct
            .as_ref()
            .ok_or("Missing direct kernels")?
            .kernels
            .borrow();
        let mut preferred = std::collections::BTreeSet::new();
        for [lo, side, last, _] in regions {
            let end = last + 1;
            let range = if policy == "main" {
                lo..side
            } else {
                side..end
            };
            for o in &operations[range] {
                if let Operation::Kernel { name } = &o.operation {
                    let k = &kernels[name];
                    if policy == "main"
                        || (policy == "small_side"
                            && o.launch
                                .as_ref()
                                .map_or(&k.spec.grid, |launch| &launch.grid)
                                .iter()
                                .product::<u32>()
                                <= 48)
                    {
                        preferred.insert(k.function.ok_or("Unresolved priority kernel")? as usize);
                    }
                }
            }
        }
        if !matches!(policy, "main" | "small_side") {
            return Err("Unknown priority policy".into());
        }
        let mut count = 0;
        // SAFETY: Query node handles owned by this completed, uninstantiated capture.
        unsafe {
            check(
                sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut count),
                "count graph nodes",
            )?;
        }
        let mut nodes = vec![ptr::null_mut(); count];
        // SAFETY: Output storage has the queried node capacity.
        unsafe {
            check(
                sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut count),
                "read graph nodes",
            )?;
        }
        for node in nodes.into_iter().take(count) {
            let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL;
            // SAFETY: Node belongs to this graph; output enum initialized.
            unsafe {
                check(sys::cuGraphNodeGetType(node, &mut kind), "graph node type")?;
            }
            if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
                continue;
            }
            // SAFETY: Valid zero storage for the driver output structure/attribute union.
            let (mut params, mut attr): (
                sys::CUDA_KERNEL_NODE_PARAMS_v2,
                sys::CUkernelNodeAttrValue,
            ) = unsafe { std::mem::zeroed() };
            // SAFETY: A kernel node accepts this output parameter structure.
            unsafe {
                check(
                    sys::cuGraphKernelNodeGetParams_v2(node, &mut params),
                    "graph kernel params",
                )?;
            }
            attr.priority = if preferred.contains(&(params.func as usize)) {
                -1
            } else {
                0
            };
            // SAFETY: Priority is a supported scalar node attribute; other union bytes are zero.
            unsafe {
                check(
                    sys::cuGraphKernelNodeSetAttribute(
                        node,
                        sys::CUkernelNodeAttrID::CU_LAUNCH_ATTRIBUTE_PRIORITY,
                        &attr,
                    ),
                    "graph priority",
                )?;
            }
        }
        // CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY, from the installed CUDA header.
        Ok(8)
    }
    fn ensure_parallel_handles(&self) -> Result<(sys::CUstream, sys::CUevent, sys::CUevent)> {
        let existing = *self.session.parallel_handles.borrow();
        let handles = if let Some(handles) = existing {
            handles
        } else {
            let mut stream = ptr::null_mut();
            let mut fork = ptr::null_mut();
            let mut join = ptr::null_mut();
            // SAFETY: Current owned context; retain every new handle before capture uses it.
            let result = unsafe {
                check(sys::cuStreamCreate(&mut stream, 1), "create side stream")
                    .and_then(|()| check(sys::cuEventCreate(&mut fork, 2), "create fork"))
                    .and_then(|()| check(sys::cuEventCreate(&mut join, 2), "create join"))
            };
            if let Err(error) = result {
                // SAFETY: Only successfully created local handles are destroyed.
                unsafe {
                    if !fork.is_null() {
                        sys::cuEventDestroy_v2(fork);
                    }
                    if !join.is_null() {
                        sys::cuEventDestroy_v2(join);
                    }
                    if !stream.is_null() {
                        sys::cuStreamDestroy_v2(stream);
                    }
                }
                return Err(error);
            }
            let handles = (stream, fork, join);
            *self.session.parallel_handles.borrow_mut() = Some(handles);
            handles
        };
        Ok(handles)
    }
    fn capture_parallel_operations(&self, operations: &[Invocation]) -> Result<()> {
        let regions = parallel_regions(
            operations,
            &self.parallel_regions.borrow(),
            &self.row_kernel_slots,
        )?;
        if regions.is_empty() {
            return operations.iter().try_for_each(|op| self.invoke(op));
        }
        let (side, fork_event, join_event) = self.ensure_parallel_handles()?;
        for (i, op) in operations.iter().enumerate() {
            if let Some(&[_, start, last, _]) = regions.iter().find(|r| r[0] == i) {
                // SAFETY: Capture forks from the main stream's completed prefix;
                // adapter-certified branches have disjoint writable buffers.
                unsafe {
                    check(
                        sys::cuEventRecord(fork_event, self.session.stream),
                        "capture fork",
                    )?;
                    check(
                        sys::cuStreamWaitEvent(side, fork_event, 0),
                        "capture side dependency",
                    )?;
                }
                #[cfg(test)]
                let gate_first = self.parallel_gate_first.get();
                #[cfg(not(test))]
                let gate_first = false;
                if gate_first {
                    self.invoke_on(&operations[last], side)?;
                    for operation in &operations[start..last] {
                        self.invoke_on(operation, side)?;
                    }
                } else {
                    for operation in &operations[start..=last] {
                        self.invoke_on(operation, side)?;
                    }
                }
                // SAFETY: The side event follows every producer in the branch.
                unsafe {
                    check(
                        sys::cuEventRecord(join_event, side),
                        "capture branch completion",
                    )?;
                }
            }
            if regions.iter().any(|r| r[1] <= i && i <= r[2]) {
                continue;
            }
            if regions.iter().any(|r| r[3] == i) {
                // SAFETY: Join before consumers and before the next layer reuses scratch.
                unsafe {
                    check(
                        sys::cuStreamWaitEvent(self.session.stream, join_event, 0),
                        "capture join",
                    )?;
                }
            }
            self.invoke(op)?;
        }
        Ok(())
    }
    pub(super) fn program_graph(&self, name: &str) -> Result<Handle> {
        if let Some(&graph) = self.graphs.borrow().get(name) {
            return Ok(graph);
        }
        if self.cuda_graph == crate::execution::CudaGraphMode::Off {
            return Err("CUDA graphs are disabled".into());
        }
        let ops = self
            .direct
            .as_ref()
            .ok_or("Missing capture bindings")?
            .programs
            .get(name)
            .ok_or("Unknown program")?;
        let invocations: Vec<_> = ops
            .iter()
            .cloned()
            .map(|operation| Invocation {
                operation,
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect();
        let at = Instant::now();
        let graph = self.capture(&invocations)?;
        let mut statistics = self.batch_statistics.get();
        statistics.sequence_captures += 1;
        statistics.sequence_capture_s += at.elapsed().as_secs_f64();
        self.batch_statistics.set(statistics);
        self.graphs.borrow_mut().insert(name.into(), graph);
        Ok(graph)
    }
    /// Query after ensuring every segment's KV mapping: arena growth invalidates
    /// captures. The sole worker cannot change the cache between this and replay.
    pub(crate) fn has_batch_graph(&self, key: &[(usize, usize)], decode: bool) -> bool {
        (self.cuda_graph == crate::execution::CudaGraphMode::Full
            || (decode && self.cuda_graph == crate::execution::CudaGraphMode::DecodeOnly))
            && self.batch_graphs.borrow().keys().any(|cached| {
                cached == &key.to_vec().into()
                    || cached.shared.is_some()
                        && cached.operation == orinfer_model_sdk::abi::BatchOperation::Ordinary
                        && cached.rows == key.len()
                        && key.iter().all(|(_, tokens)| *tokens == 1)
            })
    }
    pub(crate) fn execute_batch(
        &mut self,
        key: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        self.execute_batch_operation(
            orinfer_model_sdk::abi::BatchOperation::Ordinary,
            key,
            operations,
            decode,
            allow_capture,
        )
    }
    fn reserve_graph_nodes(&self, incoming: usize, shared: bool) -> Result<()> {
        use super::executor::{BatchGraphKey, SharedGraphKey};
        enum Victim {
            Batch(BatchGraphKey),
            Shared(SharedGraphKey),
        }
        loop {
            let victim = {
                let batches = self.batch_graphs.borrow();
                let pieces = self.shared_graphs.borrow();
                let oldest_piece = || {
                    pieces
                        .iter()
                        .min_by_key(|(_, (_, last))| *last)
                        .map(|(key, _)| Victim::Shared(key.clone()))
                };
                if !shared && let Some(key) = batch_graph_victim(&batches, incoming) {
                    Some(Victim::Batch(key))
                } else if shared && pieces.len() >= 256 {
                    oldest_piece()
                } else if batches.values().map(|(_, _, n)| n).sum::<usize>()
                    + pieces.keys().map(|k| k.operations.len()).sum::<usize>()
                    + incoming
                    <= BATCH_GRAPH_NODES
                {
                    None
                } else {
                    let oldest_batch = batches.iter().min_by_key(|(_, (_, last, _))| *last);
                    let oldest_shared = pieces.iter().min_by_key(|(_, (_, last))| *last);
                    match (oldest_batch, oldest_shared) {
                        (Some((key, (_, last, _))), Some((_, (_, plast)))) if last <= plast => {
                            Some(Victim::Batch(key.clone()))
                        }
                        (_, Some((key, _))) => Some(Victim::Shared(key.clone())),
                        (Some((key, _)), None) => Some(Victim::Batch(key.clone())),
                        (None, None) => return Err("Oversized graph node reservation".into()),
                    }
                }
            };
            let Some(victim) = victim else { return Ok(()) };
            let at = Instant::now();
            self.sync()?;
            let graph = match victim {
                Victim::Batch(key) => self.batch_graphs.borrow_mut().remove(&key).unwrap().0,
                Victim::Shared(key) => self.shared_graphs.borrow_mut().remove(&key).unwrap().0,
            };
            self.destroy_graphs(&[graph])?;
            let mut stats = self.batch_statistics.get();
            stats.graph_evictions += 1;
            stats.eviction_s += at.elapsed().as_secs_f64();
            self.batch_statistics.set(stats);
        }
    }
    fn shared_addresses(&self, op: &Invocation) -> Result<Option<Vec<u64>>> {
        if op.sequence.is_some() {
            return Ok(None);
        }
        let names: Vec<&str> = match &op.operation {
            Operation::Kernel { name } => {
                let direct = self.direct.as_ref().ok_or("Missing direct bindings")?;
                let kernels = direct.kernels.borrow();
                let kernel = kernels.get(name).ok_or("Missing shared kernel")?;
                let mut addresses = vec![];
                for argument in &kernel.spec.args {
                    if let Argument::Buffer { name } | Argument::BufferSlice { name, .. } = argument
                    {
                        let target = op
                            .views
                            .get(name)
                            .map_or(name.as_str(), |v| v.buffer.as_str());
                        if self.pointers.is_private(target) {
                            return Ok(None);
                        }
                        addresses.push(self.address(name, None, &op.views)?);
                    }
                }
                return Ok(Some(addresses));
            }
            Operation::Copy {
                source,
                destination,
                ..
            } => vec![source, destination],
            Operation::Zero { destination, .. } => vec![destination],
        };
        if names.iter().any(|name| {
            let target = op.views.get(*name).map_or(*name, |v| v.buffer.as_str());
            self.pointers.is_private(target)
        }) {
            return Ok(None);
        }
        names
            .into_iter()
            .map(|name| self.address(name, None, &op.views))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
    /// Share stateless pieces across changing request membership. Private stages
    /// remain direct; pointer-table contents are uploaded before any replay.
    fn invoke_shared_pieces(&self, operations: &[Invocation], allow_capture: bool) -> Result<()> {
        use super::executor::SharedGraphKey;
        let mut cursor = 0;
        while cursor < operations.len() {
            let start = cursor;
            let mut addresses = vec![];
            while cursor < operations.len() {
                let Some(mut pointers) = self.shared_addresses(&operations[cursor])? else {
                    break;
                };
                addresses.append(&mut pointers);
                cursor += 1;
            }
            if cursor == start {
                self.invoke(&operations[cursor])?;
                cursor += 1;
                continue;
            }
            let piece = &operations[start..cursor];
            // Tiny pieces do not amortize instantiation or an extra graph launch.
            if !(8..=BATCH_GRAPH_NODES).contains(&piece.len()) {
                for op in piece {
                    self.invoke(op)?;
                }
                continue;
            }
            let key = SharedGraphKey {
                operations: std::rc::Rc::new(piece.to_vec()),
                addresses,
            };
            let tick = self.batch_graph_clock.get().wrapping_add(1);
            self.batch_graph_clock.set(tick);
            let cached = self
                .shared_graphs
                .borrow_mut()
                .get_mut(&key)
                .map(|(graph, last)| {
                    *last = tick;
                    *graph
                });
            let graph = if let Some(graph) = cached {
                Some(graph)
            } else {
                let count = {
                    let mut trials = self.shared_graph_trials.borrow_mut();
                    if trials.len() >= 512 && !trials.contains_key(&key) {
                        trials.clear();
                    }
                    let count = trials.entry(key.clone()).or_default();
                    *count += 1;
                    *count
                };
                if !allow_capture || count < 2 {
                    None
                } else {
                    self.reserve_graph_nodes(piece.len(), true)?;
                    let at = Instant::now();
                    let graph = self.capture(piece)?;
                    let mut stats = self.batch_statistics.get();
                    stats.capture_s += at.elapsed().as_secs_f64();
                    stats.captured_operations += piece.len();
                    stats.graph_misses += 1;
                    stats.shared_graph_captures += 1;
                    self.batch_statistics.set(stats);
                    self.shared_graphs.borrow_mut().insert(key, (graph, tick));
                    Some(graph)
                }
            };
            if let Some(graph) = graph {
                // SAFETY: All piece addresses are weights/workspace, retained by
                // the session. Current resolved addresses are part of the key.
                unsafe {
                    check(
                        sys::cuGraphLaunch(graph, self.session.stream),
                        "shared batch piece",
                    )?;
                }
                let mut stats = self.batch_statistics.get();
                stats.graph_hits += usize::from(cached.is_some());
                stats.shared_graph_hits += usize::from(cached.is_some());
                self.batch_statistics.set(stats);
            } else {
                for op in piece {
                    self.invoke(op)?;
                }
            }
        }
        Ok(())
    }
    /// COW and live workspace preservation must run before this lookup. Exact
    /// graphs retain owner-specific addresses and are invalidated on arena
    /// replacement. Indirect graphs use only fixed session addresses plus the
    /// pointer table; its request-owned contents are refreshed before replay.
    fn replay_cached_batch(
        &self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: &[(usize, usize)],
        operations: BatchOperations<'_>,
        decode: bool,
        overlap: &mut dyn FnMut(),
    ) -> Result<bool> {
        if !(self.cuda_graph == crate::execution::CudaGraphMode::Full
            || decode && self.cuda_graph == crate::execution::CudaGraphMode::DecodeOnly)
        {
            return Ok(false);
        }
        let rows = packet_graph_rows(operation, segments);
        let graph = self
            .batch_graphs
            .borrow_mut()
            .iter_mut()
            .find(|(key, _)| {
                key.operation == operation
                    && key.rows == rows
                    && match &key.shared {
                        None => key.segments == segments,
                        Some(shared) => operations.matches(&shared.operations),
                    }
            })
            .map(|(_, (graph, last, _))| {
                let tick = self.batch_graph_clock.get().wrapping_add(1);
                self.batch_graph_clock.set(tick);
                *last = tick;
                *graph
            });
        let Some(graph) = graph else { return Ok(false) };
        let at = Instant::now();
        // SAFETY: Same plan, retained addresses and prepared writable state.
        // Captured modules remain owned by the session until graph destruction.
        unsafe {
            check(
                sys::cuGraphLaunch(graph, self.session.stream),
                "hot batch graph",
            )?;
        }
        overlap();
        self.sync()?;
        let mut stats = self.batch_statistics.get();
        stats.graph_hits += 1;
        stats.replay_s += at.elapsed().as_secs_f64();
        self.batch_statistics.set(stats);
        Ok(true)
    }
    pub(crate) fn execute_batch_operation(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        self.execute_batch_operation_with(
            operation,
            segments,
            operations,
            decode,
            allow_capture,
            &mut || {},
        )
    }
    pub(crate) fn execute_batch_operation_with(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
        allow_capture: bool,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        self.execute_batch_operations(
            operation,
            segments,
            BatchOperations {
                calls: operations,
                retained: None,
            },
            decode,
            allow_capture,
            overlap,
        )
    }
    pub(crate) fn execute_cached_batch_operation(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &std::rc::Rc<Vec<Invocation>>,
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        self.execute_cached_batch_operation_with(
            operation,
            segments,
            operations,
            decode,
            allow_capture,
            &mut || {},
        )
    }
    pub(crate) fn execute_cached_batch_operation_with(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &std::rc::Rc<Vec<Invocation>>,
        decode: bool,
        allow_capture: bool,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        self.execute_batch_operations(
            operation,
            segments,
            BatchOperations {
                calls: operations,
                retained: Some(operations),
            },
            decode,
            allow_capture,
            overlap,
        )
    }
    fn execute_batch_operations(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        prepared: BatchOperations<'_>,
        decode: bool,
        allow_capture: bool,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        let operations = prepared.calls;
        if operations.is_empty() && decode && segments.iter().all(|(_, n)| *n == 1) {
            let cached = {
                let mut cache = self.batch_graphs.borrow_mut();
                cache
                    .iter_mut()
                    .find(|(key, _)| {
                        key.shared.is_some()
                            && key.operation == operation
                            && key.rows == segments.len()
                    })
                    .map(|(_, (graph, last, _))| {
                        let tick = self.batch_graph_clock.get().wrapping_add(1);
                        self.batch_graph_clock.set(tick);
                        *last = tick;
                        *graph
                    })
            };
            if let Some(graph) = cached {
                self.save_live_workspace()?;
                let at = Instant::now();
                // SAFETY: Captured session addresses cannot change; pointer-table owners were
                // validated and uploaded by the caller before every replay.
                unsafe {
                    check(
                        sys::cuGraphLaunch(graph, self.session.stream),
                        "hot indirect batch graph",
                    )?;
                }
                overlap();
                self.sync()?;
                let mut stats = self.batch_statistics.get();
                stats.graph_hits += 1;
                stats.replay_s += at.elapsed().as_secs_f64();
                self.batch_statistics.set(stats);
                return Ok(());
            }
        }
        // Batch kernels can address inactive sequences without activating them.
        self.save_live_workspace()?;
        // Copy/zero destinations can also write KV (e.g. state restoration).
        // Prepare COW before graph capture/replay, including inactive arenas.
        for invocation in operations {
            let (destination, bytes) = match &invocation.operation {
                Operation::Copy {
                    destination, bytes, ..
                }
                | Operation::Zero { destination, bytes } => (destination, *bytes),
                _ => continue,
            };
            let view = invocation.views.get(destination);
            let name = view.map_or(destination.as_str(), |v| v.buffer.as_str());
            let offset = view.map_or(0, |v| v.offset);
            if let Some(slot) = invocation.sequence
                && slot != self.active_sequence
            {
                if self
                    .sequences
                    .get(slot)
                    .and_then(|s| s.reservations.get(name))
                    .is_some_and(|v| v.has_shared_slabs())
                {
                    self.sync()?;
                    self.sequences[slot]
                        .reservations
                        .get_mut(name)
                        .unwrap()
                        .make_writable(&self.session, offset, bytes)?;
                }
            } else {
                self.ensure_writable_range(name, offset, bytes)?;
            }
        }
        if self.replay_cached_batch(operation, &segments, prepared, decode, overlap)? {
            return Ok(());
        }
        // A fully indirect plan captures only session-owned table/workspace
        // addresses. Changing request membership cannot invalidate those pointers.
        let shared = if !operations.is_empty() {
            let mut addresses = Vec::new();
            let mut indirect = true;
            for call in operations {
                if let Some(mut resolved) = self.shared_addresses(call)? {
                    addresses.append(&mut resolved);
                } else {
                    indirect = false;
                    break;
                }
            }
            if indirect {
                Some(super::executor::SharedGraphKey {
                    operations: prepared.retain(),
                    addresses,
                })
            } else {
                None
            }
        } else {
            None
        };
        let rows = packet_graph_rows(operation, &segments);
        let segments = if shared.is_some() {
            Vec::new()
        } else {
            segments
        };
        let key = super::executor::BatchGraphKey {
            operation,
            segments,
            rows,
            shared,
        };
        self.ensure_invocation_kernels(operations)?;
        let dynamic = operations.iter().any(|op| op.launch.is_some());
        let configured = (self.cuda_graph == crate::execution::CudaGraphMode::Full
            || (decode && self.cuda_graph == crate::execution::CudaGraphMode::DecodeOnly))
            && operations.len() <= BATCH_GRAPH_NODES;
        // A one-off membership executes directly. Capture only after a repeat,
        // and only when the remaining output budget can amortize the capture.
        // Existing graphs remain reusable even on the final token.
        let graph_enabled = configured
            && (self.batch_graphs.borrow().contains_key(&key) || {
                let mut trials = self.batch_graph_trials.borrow_mut();
                if trials.len() >= 32 && !trials.contains_key(&key) {
                    trials.clear();
                }
                let count = trials.entry(key.clone()).or_default();
                *count += 1;
                allow_capture && *count >= 2
            });
        if configured && !graph_enabled {
            let mut statistics = self.batch_statistics.get();
            statistics.graph_capture_deferrals += 1;
            self.batch_statistics.set(statistics);
        }
        if graph_enabled {
            let tick = self.batch_graph_clock.get().wrapping_add(1);
            self.batch_graph_clock.set(tick);
            let cached = self
                .batch_graphs
                .borrow_mut()
                .get_mut(&key)
                .map(|(g, last, _)| {
                    *last = tick;
                    *g
                });
            let graph = if let Some(graph) = cached {
                let mut statistics = self.batch_statistics.get();
                statistics.graph_hits += 1;
                self.batch_statistics.set(statistics);
                graph
            } else {
                let mut statistics = self.batch_statistics.get();
                statistics.graph_misses += 1;
                self.batch_statistics.set(statistics);
                // Bound graph cache size; varying slot memberships cannot retain
                // unbounded captures. Stream is synchronized before destruction.
                self.reserve_graph_nodes(operations.len(), false)?;
                let at = Instant::now();
                let graph = self.capture(operations)?;
                let mut statistics = self.batch_statistics.get();
                statistics.capture_s += at.elapsed().as_secs_f64();
                statistics.dynamic_graph_captures += usize::from(dynamic);
                statistics.captured_operations += operations.len();
                self.batch_statistics.set(statistics);
                self.batch_graphs
                    .borrow_mut()
                    .insert(key, (graph, tick, operations.len()));
                graph
            };
            let at = Instant::now();
            // SAFETY: Captured pointers refer to stable, retained request arenas.
            unsafe {
                check(
                    sys::cuGraphLaunch(graph, self.session.stream),
                    "batch graph",
                )?;
            }
            overlap();
            self.sync()?;
            let mut statistics = self.batch_statistics.get();
            statistics.replay_s += at.elapsed().as_secs_f64();
            self.batch_statistics.set(statistics);
        } else {
            let at = Instant::now();
            if configured {
                self.invoke_shared_pieces(operations, allow_capture)?;
            } else {
                for op in operations {
                    self.invoke(op)?;
                }
            }
            overlap();
            self.sync()?;
            let mut statistics = self.batch_statistics.get();
            statistics.direct_s += at.elapsed().as_secs_f64();
            statistics.direct_iterations += 1;
            statistics.dynamic_direct_iterations += usize::from(dynamic);
            self.batch_statistics.set(statistics);
        }
        Ok(())
    }
}

#[cfg(test)]
impl Executor {
    pub(crate) fn profile_events(&mut self, count: usize) -> Result<Vec<sys::CUevent>> {
        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            let mut event = ptr::null_mut();
            // SAFETY: Live context; transfer ownership before any further error.
            unsafe {
                check(sys::cuEventCreate(&mut event, 0), "profile event")?;
            }
            self.session.events.push(event);
            events.push(event);
        }
        Ok(events)
    }
    pub(crate) fn profile_hot_graph(
        &mut self,
        key: &[(usize, usize)],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        let batch = {
            self.batch_graphs
                .borrow()
                .iter()
                .find(|(cached, _)| {
                    **cached == super::executor::BatchGraphKey::from(key.to_vec())
                        || cached.shared.is_some()
                            && cached.operation == orinfer_model_sdk::abi::BatchOperation::Ordinary
                            && cached.rows == key.len()
                            && key.iter().all(|(_, tokens)| *tokens == 1)
                })
                .map(|(_, graph)| graph.0)
        };
        let graph = if let Some(graph) = batch {
            graph
        } else if key.len() == 1 {
            self.program_graph("decode")?
        } else {
            return Err("Missing hot batch graph".into());
        };
        self.profile_graph_time(graph, repetitions)
    }
    pub(crate) fn profile_invocations_hot(
        &mut self,
        operations: &[Invocation],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        self.sync()?;
        let graph = self.capture(operations)?;
        self.profile_graph_time(graph, repetitions)
    }
    fn profile_graph_time(&mut self, graph: Handle, repetitions: usize) -> Result<Vec<f32>> {
        let events = self.profile_events(2)?;
        let mut times = vec![];
        for _ in 0..repetitions {
            let mut ms = 0.;
            // SAFETY: Stable graph bindings, owned events and worker stream.
            unsafe {
                check(
                    sys::cuEventRecord(events[0], self.session.stream),
                    "profile start",
                )?;
                check(
                    sys::cuGraphLaunch(graph, self.session.stream),
                    "profile replay",
                )?;
                check(
                    sys::cuEventRecord(events[1], self.session.stream),
                    "profile end",
                )?;
                check(sys::cuEventSynchronize(events[1]), "profile completion")?;
                check(
                    sys::cuEventElapsedTime(&mut ms, events[0], events[1]),
                    "profile elapsed",
                )?;
            }
            times.push(ms);
        }
        Ok(times)
    }
    pub(crate) fn profile_graph_operations(
        &mut self,
        operations: &[Invocation],
    ) -> Result<Vec<Vec<f32>>> {
        self.profile_graph_operations_at_position(operations, None)
    }
    pub(crate) fn profile_stream_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: (&str, u32),
    ) -> Result<Vec<Vec<f32>>> {
        self.sync()?;
        self.ensure_invocation_kernels(operations)?;
        let events = self.profile_events(operations.len() + 1)?;
        let mut trials = vec![];
        for trial in 0..4 {
            self.upload_ids(position.0, &[position.1])?;
            for (i, operation) in operations.iter().enumerate() {
                // SAFETY: Owned events and live worker stream; ordinary launches.
                unsafe {
                    check(
                        sys::cuEventRecord(events[i], self.session.stream),
                        "profile stream boundary",
                    )?;
                }
                self.invoke(operation)?;
            }
            // SAFETY: Records completion after all ordinary launches.
            unsafe {
                check(
                    sys::cuEventRecord(*events.last().unwrap(), self.session.stream),
                    "profile stream end",
                )?;
            }
            self.sync()?;
            if trial == 0 {
                continue;
            }
            let mut times = vec![];
            for pair in events.windows(2) {
                let mut ms = 0.;
                // SAFETY: Completed events on the same stream, in launch order.
                unsafe {
                    check(
                        sys::cuEventElapsedTime(&mut ms, pair[0], pair[1]),
                        "profile stream elapsed",
                    )?;
                }
                times.push(ms);
            }
            trials.push(times);
        }
        Ok(trials)
    }
    pub(crate) fn profile_graph_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: Option<(&str, u32)>,
    ) -> Result<Vec<Vec<f32>>> {
        self.sync()?;
        self.ensure_invocation_kernels(operations)?;
        let events = self.profile_events(operations.len() + 1)?;
        // SAFETY: Sole stream owner; external event nodes retain timestamps in replay.
        unsafe {
            check(
                sys::cuStreamBeginCapture_v2(
                    self.session.stream,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL,
                ),
                "profile capture",
            )?;
        }
        let result = (|| -> Result<()> {
            for (i, operation) in operations.iter().enumerate() {
                // SAFETY: Owned event, during capture; CU_EVENT_RECORD_EXTERNAL=1.
                unsafe {
                    check(
                        sys::cuEventRecordWithFlags(events[i], self.session.stream, 1),
                        "profile boundary",
                    )?;
                }
                self.invoke(operation)?;
            }
            // SAFETY: Last event records completion of all captured operations.
            unsafe {
                check(
                    sys::cuEventRecordWithFlags(*events.last().unwrap(), self.session.stream, 1),
                    "profile last boundary",
                )?;
            }
            Ok(())
        })();
        let mut graph = ptr::null_mut();
        // SAFETY: Always terminate capture after either success or failure.
        let end = unsafe {
            check(
                sys::cuStreamEndCapture(self.session.stream, &mut graph),
                "profile capture end",
            )
        };
        if let Err(error) = result.and(end) {
            if !graph.is_null() {
                // SAFETY: Failed graph has no other owner.
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(error);
        }
        let mut exec = ptr::null_mut();
        // SAFETY: Locally owned graph transferred to session on success.
        unsafe {
            if let Err(error) = check(
                sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
                "profile instantiate",
            ) {
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
            if let Err(error) = check(
                sys::cuGraphUpload(exec, self.session.stream),
                "profile graph upload",
            ) {
                sys::cuGraphExecDestroy(exec);
                sys::cuGraphDestroy(graph);
                return Err(error);
            }
        }
        self.session.graphs.borrow_mut().push((graph, exec));
        let mut trials = vec![];
        for trial in 0..4 {
            if let Some((name, value)) = position {
                self.upload_ids(name, &[value])?;
            }
            // SAFETY: All request buffers are retained; bounded extra steps fit reservations.
            unsafe {
                check(
                    sys::cuGraphLaunch(exec, self.session.stream),
                    "profile operation replay",
                )?;
            }
            self.sync()?;
            if trial == 0 {
                continue;
            }
            let mut times = vec![];
            for pair in events.windows(2) {
                let mut ms = 0.;
                // SAFETY: Events completed on the same stream in replay order.
                unsafe {
                    check(
                        sys::cuEventElapsedTime(&mut ms, pair[0], pair[1]),
                        "profile operation elapsed",
                    )?;
                }
                times.push(ms);
            }
            trials.push(times);
        }
        Ok(trials)
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Session must see every inactive reservation before its cleanup.
        let buffers = self.session.virtual_buffers.get_mut();
        for (slot, sequence) in self.sequences.iter_mut().enumerate() {
            for (name, reservation) in std::mem::take(&mut sequence.reservations) {
                buffers.insert(format!("@{slot}/{name}"), reservation);
            }
        }
    }
}

#[cfg(test)]
mod graph_cache_tests {
    use super::*;

    fn calls(names: &[&str]) -> Vec<Invocation> {
        names
            .iter()
            .map(|name| Invocation {
                operation: Operation::Kernel {
                    name: (*name).into(),
                },
                sequence: None,
                launch: None,
                views: BTreeMap::new(),
            })
            .collect()
    }

    #[test]
    fn partial_shared_pieces_fork_only_complete_branches() {
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        let ops = calls(&["root", "main", "side", "join", "tail"]);
        assert_eq!(
            parallel_regions(&ops, std::slice::from_ref(&region), &BTreeMap::new()).unwrap(),
            vec![[0, 2, 2, 3]]
        );
        for piece in [&ops[..3], &ops[1..], &ops[2..3]] {
            assert!(
                parallel_regions(piece, std::slice::from_ref(&region), &BTreeMap::new())
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[test]
    fn branch_boundaries_include_all_helpers_of_a_row_slot() {
        let ops = calls(&[
            "root/pack",
            "root/gemm",
            "main",
            "side/pack",
            "side/gemm",
            "join",
        ]);
        let slots = [
            ("root/pack", "root"),
            ("root/gemm", "root"),
            ("side/pack", "side"),
            ("side/gemm", "side"),
        ]
        .map(|(name, slot)| (name.into(), slot.into()))
        .into();
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        assert_eq!(
            parallel_regions(&ops, std::slice::from_ref(&region), &slots).unwrap(),
            vec![[0, 3, 4, 5]]
        );
        // A repeated slot separated by another operation is ambiguous.
        let ops = calls(&["root/pack", "main", "root/gemm", "side/gemm", "join"]);
        assert!(
            parallel_regions(&ops, &[region], &slots)
                .unwrap()
                .is_empty()
        );
        let ops = calls(&["root", "private", "other", "private", "side", "join"]);
        let region = ["root", "side", "side", "join"].map(str::to_owned);
        assert_eq!(
            parallel_regions(&ops, &[region], &slots).unwrap(),
            vec![[0, 4, 4, 5]]
        );
    }

    #[test]
    fn malformed_and_overlapping_branches_are_rejected() {
        let ops = calls(&["a", "b", "c", "d", "e"]);
        assert!(
            parallel_regions(
                &ops,
                &[["a", "b", "b", "d"].map(str::to_owned)],
                &BTreeMap::new(),
            )
            .is_err()
        );
        assert!(
            parallel_regions(
                &ops,
                &[
                    ["a", "b", "b", "c"].map(str::to_owned),
                    ["c", "d", "d", "e"].map(str::to_owned),
                ],
                &BTreeMap::new(),
            )
            .is_err()
        );
    }

    #[test]
    fn indirect_graph_identity_tracks_shape_and_workspace_not_request_slots() {
        use super::super::executor::{BatchGraphKey, SharedGraphKey};
        let graph = SharedGraphKey {
            operations: std::rc::Rc::new(vec![]),
            addresses: vec![42],
        };
        let first = BatchGraphKey {
            operation: orinfer_model_sdk::abi::BatchOperation::Ordinary,
            segments: vec![],
            rows: 2,
            shared: Some(graph.clone()),
        };
        let mut next = first.clone();
        assert_eq!(first, next);
        next.shared.as_mut().unwrap().addresses[0] = 43;
        assert_ne!(first, next);
        next = first.clone();
        next.rows = 3;
        assert_ne!(first, next);
        next = first.clone();
        next.operation = orinfer_model_sdk::abi::BatchOperation::Verification;
        assert_ne!(first, next);
    }

    #[test]
    fn cached_graph_plans_retain_storage_and_recheck_changed_launches_and_views() {
        let original = std::rc::Rc::new(vec![Invocation {
            operation: Operation::Kernel {
                name: "indirect".into(),
            },
            sequence: None,
            launch: Some(orinfer_model_sdk::operators::dynamic::RowLaunch {
                grid: [2, 1, 1],
                arguments: vec![(0, 2)],
            }),
            views: BTreeMap::from([(
                "Pointers".into(),
                crate::execution::BufferView {
                    buffer: "Pointers".into(),
                    offset: 16,
                },
            )]),
        }]);
        let prepared = BatchOperations {
            calls: &original,
            retained: Some(&original),
        };
        let retained = prepared.retain();
        assert!(std::rc::Rc::ptr_eq(&original, &retained));
        assert!(prepared.matches(&retained));
        let key = super::super::executor::SharedGraphKey {
            operations: retained.clone(),
            addresses: vec![42],
        };
        let mut equal = key.clone();
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Equal);
        equal.operations = std::rc::Rc::new(original.as_ref().clone());
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Equal);
        equal.addresses[0] = 43;
        assert_eq!(key.cmp(&equal), std::cmp::Ordering::Less);
        let borrowed = BatchOperations {
            calls: &original,
            retained: None,
        };
        assert!(borrowed.matches(&retained));
        assert!(!std::rc::Rc::ptr_eq(&borrowed.retain(), &retained));
        for field in 0..4 {
            let mut changed = original.as_ref().clone();
            match field {
                0 => changed[0].launch.as_mut().unwrap().grid[0] = 3,
                1 => changed[0].launch.as_mut().unwrap().arguments[0].1 = 3,
                2 => changed[0].views.get_mut("Pointers").unwrap().offset = 24,
                _ => changed[0].sequence = Some(7),
            }
            let changed = std::rc::Rc::new(changed);
            assert!(!prepared.matches(&changed));
            let changed_key = super::super::executor::SharedGraphKey {
                operations: changed,
                addresses: key.addresses.clone(),
            };
            assert_ne!(key.cmp(&changed_key), std::cmp::Ordering::Equal);
        }
    }

    #[test]
    fn verification_graphs_cannot_alias_ordinary_batches() {
        use super::super::executor::BatchGraphKey;
        use orinfer_model_sdk::abi::BatchOperation;
        let ordinary = BatchGraphKey::from(vec![(2, 4), (3, 4)]);
        let verification = BatchGraphKey {
            operation: BatchOperation::Verification,
            segments: ordinary.segments.clone(),
            rows: ordinary.rows,
            shared: None,
        };
        let mut cache = super::super::executor::BatchGraphCache::new();
        cache.insert(ordinary.clone(), (ptr::null_mut(), 1, 1));
        assert!(!cache.contains_key(&verification));
        cache.insert(verification, (ptr::null_mut(), 2, 1));
        assert_eq!(cache.len(), 2);
        assert!(cache.contains_key(&ordinary));
    }
    #[test]
    fn node_budget_evicts_least_recently_used_before_entry_limit() {
        let mut cache = super::super::executor::BatchGraphCache::new();
        let large = BATCH_GRAPH_NODES * 2 / 3;
        let small = BATCH_GRAPH_NODES / 4;
        let remaining = BATCH_GRAPH_NODES - large - small;
        cache.insert(vec![(2, 1)].into(), (ptr::null_mut(), 8, large));
        cache.insert(vec![(0, 1)].into(), (ptr::null_mut(), 3, small));
        assert_eq!(batch_graph_victim(&cache, remaining), None);
        assert_eq!(
            batch_graph_victim(&cache, remaining + 1),
            Some(vec![(0, 1)].into())
        );
        cache.get_mut(&vec![(0, 1)].into()).unwrap().1 = 10;
        assert_eq!(
            batch_graph_victim(&cache, remaining + 1),
            Some(vec![(2, 1)].into())
        );
        cache.remove(&vec![(0, 1)].into());
        assert_eq!(batch_graph_victim(&cache, BATCH_GRAPH_NODES - large), None);
    }
    #[test]
    fn entry_limit_bounds_small_graphs() {
        let cache = (0..16)
            .map(|slot| (vec![(slot, 1)].into(), (ptr::null_mut(), slot as u64, 1)))
            .collect();
        assert_eq!(batch_graph_victim(&cache, 1), Some(vec![(0, 1)].into()));
    }
    #[test]
    fn commit_graph_shape_counts_owners_while_target_counts_token_rows() {
        use orinfer_model_sdk::abi::BatchOperation;
        assert_eq!(
            packet_graph_rows(BatchOperation::Commit, &[(7, 3), (31, 8)]),
            2
        );
        assert_eq!(
            packet_graph_rows(BatchOperation::Commit, &[(2, 2), (19, 4)]),
            2
        );
        assert_eq!(
            packet_graph_rows(BatchOperation::Verification, &[(7, 3), (31, 8)]),
            11
        );
    }
}
