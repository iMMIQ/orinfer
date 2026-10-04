//! Request arenas share immutable weights and workspace. Switching an arena
//! changes host bindings, never copies recurrent state or KV payloads.
use super::{Argument, Handle, Result, Value, check, executor::Executor, ptr};
use crate::{execution::Invocation, model::Operation};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct Sequence {
    pub addresses: BTreeMap<String, u64>,
    pub sizes: BTreeMap<String, usize>,
    pub reservations: super::virtual_memory::Reservations,
    pub graphs: BTreeMap<String, Handle>,
    pub leased: bool,
}

impl Executor {
    #[cfg(test)]
    pub(crate) fn private_buffer_sizes(&self) -> BTreeMap<String, usize> {
        let reservations = self.session.virtual_buffers.borrow();
        self.sequences[self.active_sequence]
            .sizes
            .iter()
            .map(|(n, &bytes)| (n.clone(), reservations.get(n).map_or(bytes, |r| r.mapped)))
            .collect()
    }
    pub(crate) fn free_bytes(&self) -> Result<usize> {
        let (mut free, mut total) = (0, 0);
        // SAFETY: The worker owns the current context and both output scalars.
        unsafe {
            check(
                (self.session.driver.memory_info)(&mut free, &mut total),
                "sequence free memory",
            )?;
        }
        Ok(free)
    }
    pub(crate) fn request_fixed_bytes(&self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        self.sequence_specs
            .iter()
            .filter(|b| !self.sequence_strides.contains_key(&b.name))
            .try_fold(0usize, |sum, b| {
                sum.checked_add(limits.get(&b.name).copied().unwrap_or(b.bytes()?))
                    .ok_or("Sequence size overflow".into())
            })
    }
    pub(crate) fn lease_sequence(&mut self, limits: &BTreeMap<String, usize>) -> Result<usize> {
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
        if let Some(slot) = self.sequences.iter().position(|s| !s.leased) {
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
                .filter(|(n, b)| self.sequences[slot].sizes[n] < *b)
                .collect();
            if !needed.is_empty() {
                self.sync()?;
                self.invalidate_current_graphs()?;
                for (name, bytes) in needed {
                    if let Some(&stride) = self.sequence_strides.get(&name) {
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
                        self.pointers.insert(name.clone(), address);
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
                                (self.session.driver.vmm_address_free)(old.address, old.bytes),
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
                            (self.session.driver.alloc)(&mut address, bytes),
                            "grow request metadata",
                        )?;
                        self.session.buffers.push(address);
                        check(
                            (self.session.driver.memset)(address, 0, bytes, self.session.stream),
                            "initialize request metadata",
                        )?;
                        let old = self.sequences[slot].addresses[&name];
                        check((self.session.driver.free)(old), "free idle metadata")?;
                        self.session.buffers.retain(|&p| p != old);
                    }
                    self.sequences[slot].addresses.insert(name.clone(), address);
                    self.sequences[slot].sizes.insert(name.clone(), bytes);
                    self.pointers.insert(name.clone(), address);
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
            for b in &self.sequence_specs {
                let bytes = limits.get(&b.name).copied().unwrap_or(b.bytes()?);
                let address = if let Some(&stride) = self.sequence_strides.get(&b.name) {
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
                            (self.session.driver.alloc)(&mut address, bytes),
                            "allocate request state",
                        )?;
                        self.session.buffers.push(address);
                        allocated.push(address);
                        check(
                            (self.session.driver.memset)(address, 0, bytes, self.session.stream),
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
                        (self.session.driver.vmm_address_free)(r.address, r.bytes),
                        "rollback request reservation",
                    )?;
                    self.session.virtual_buffers.get_mut().remove(&key);
                }
                for address in allocated {
                    check(
                        (self.session.driver.free)(address),
                        "rollback request state",
                    )?;
                    self.session.buffers.retain(|&p| p != address);
                }
            }
            return Err(error);
        }
        sequence.leased = true;
        let slot = self.sequences.len();
        self.sequences.push(sequence);
        self.activate_sequence(slot)?;
        Ok(slot)
    }
    pub(crate) fn activate_sequence(&mut self, slot: usize) -> Result<()> {
        if slot >= self.sequences.len() {
            return Err("Unknown request arena".into());
        }
        if slot == self.active_sequence {
            return Ok(());
        }
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
            self.pointers.insert(name.clone(), address);
            self.sizes
                .insert(name.clone(), self.sequences[slot].sizes[name]);
        }
        Ok(())
    }
    fn invalidate_current_graphs(&mut self) -> Result<()> {
        let mut execs: Vec<_> = std::mem::take(self.graphs.get_mut())
            .into_values()
            .collect();
        execs.extend(
            std::mem::take(self.batch_graphs.get_mut())
                .into_values()
                .map(|(g, _)| g),
        );
        let mut owned = self.session.graphs.borrow_mut();
        let mut index = 0;
        while index < owned.len() {
            if execs.contains(&owned[index].1) {
                let (graph, exec) = owned.remove(index);
                // SAFETY: Idle stream, graphs removed from every execution map.
                unsafe {
                    check(
                        (self.session.driver.graph_exec_destroy)(exec),
                        "invalidate metadata graph",
                    )?;
                    check(
                        (self.session.driver.graph_destroy)(graph),
                        "invalidate metadata capture",
                    )?;
                }
            } else {
                index += 1;
            }
        }
        Ok(())
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
        if let Some(s) = self.sequences.iter().find(|s| !s.leased) {
            Ok(limits
                .iter()
                .filter(|(name, _)| !self.sequence_strides.contains_key(*name))
                .map(|(n, b)| b.saturating_sub(s.sizes[n]))
                .sum())
        } else {
            self.request_fixed_bytes(limits)
        }
    }
    pub(crate) fn release_sequence(&mut self, slot: usize, resets: &[String]) -> Result<()> {
        self.activate_sequence(slot)?;
        self.reset_sequence(resets)?;
        self.sequences[slot].leased = false;
        Ok(())
    }
    pub(crate) fn prepare_legacy_sequence(&mut self) -> Result<()> {
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
                            _ => *value,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                super::launch_kernel(
                    &k.spec,
                    k.function,
                    &mut values,
                    &self.session.driver,
                    self.session.stream,
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
                        (self.session.driver.copy)(
                            address(destination)?,
                            address(source)?,
                            *bytes,
                            self.session.stream,
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
                        (self.session.driver.memset)(
                            address(destination)?,
                            0,
                            *bytes,
                            self.session.stream,
                        ),
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
        // SAFETY: All bindings and backing allocations outlive graph execution;
        // capture records work, and the worker is the stream's sole submitter.
        unsafe {
            check(
                (self.session.driver.capture_begin)(self.session.stream, 0),
                "begin request capture",
            )?;
        }
        let result = operations.iter().try_for_each(|op| self.invoke(op));
        let mut graph = ptr::null_mut();
        // SAFETY: Always end capture, including a failed recorded operation.
        let end = unsafe {
            check(
                (self.session.driver.capture_end)(self.session.stream, &mut graph),
                "end request capture",
            )
        };
        if let Err(error) = result.and(end) {
            if !graph.is_null() {
                // SAFETY: This local failed capture is not owned elsewhere.
                unsafe {
                    (self.session.driver.graph_destroy)(graph);
                }
            }
            return Err(error);
        }
        let mut exec = ptr::null_mut();
        // SAFETY: Graph is locally owned until transferred into Session.
        unsafe {
            if let Err(error) = check(
                (self.session.driver.graph_instantiate)(&mut exec, graph, 0),
                "instantiate request graph",
            ) {
                (self.session.driver.graph_destroy)(graph);
                return Err(error);
            }
        }
        self.session.graphs.borrow_mut().push((graph, exec));
        Ok(exec)
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
                views: BTreeMap::new(),
            })
            .collect();
        let graph = self.capture(&invocations)?;
        self.graphs.borrow_mut().insert(name.into(), graph);
        Ok(graph)
    }
    pub(crate) fn execute_batch(
        &self,
        key: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
    ) -> Result<()> {
        let graph_enabled = self.cuda_graph == crate::execution::CudaGraphMode::Full
            || (decode && self.cuda_graph == crate::execution::CudaGraphMode::DecodeOnly);
        if graph_enabled {
            let tick = self.batch_graph_clock.get().wrapping_add(1);
            self.batch_graph_clock.set(tick);
            let cached = self
                .batch_graphs
                .borrow_mut()
                .get_mut(&key)
                .map(|(g, last)| {
                    *last = tick;
                    *g
                });
            let graph = if let Some(graph) = cached {
                graph
            } else {
                // Bound graph cache size; varying slot memberships cannot retain
                // unbounded captures. Stream is synchronized before destruction.
                if self.batch_graphs.borrow().len() >= 16 {
                    self.sync()?;
                    let victim = self
                        .batch_graphs
                        .borrow()
                        .iter()
                        .min_by_key(|(_, (_, last))| *last)
                        .map(|(k, (g, _))| (k.clone(), *g))
                        .expect("nonempty graphs");
                    self.batch_graphs.borrow_mut().remove(&victim.0);
                    let mut owned = self.session.graphs.borrow_mut();
                    if let Some(index) = owned.iter().position(|(_, e)| *e == victim.1) {
                        let (g, e) = owned.remove(index);
                        // SAFETY: Completed graph is removed from all owners.
                        unsafe {
                            check(
                                (self.session.driver.graph_exec_destroy)(e),
                                "evict batch graph",
                            )?;
                            check(
                                (self.session.driver.graph_destroy)(g),
                                "evict batch capture",
                            )?;
                        }
                    }
                }
                let graph = self.capture(operations)?;
                self.batch_graphs.borrow_mut().insert(key, (graph, tick));
                graph
            };
            // SAFETY: Captured pointers refer to stable, retained request arenas.
            unsafe {
                check(
                    (self.session.driver.graph_launch)(graph, self.session.stream),
                    "batch graph",
                )?;
            }
        } else {
            for op in operations {
                self.invoke(op)?;
            }
        }
        self.sync()
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
