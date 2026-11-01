//! Prepared model-execution boundary. Device handles remain inside implementations.
#[cfg(test)]
use super::cuda::sys;
use super::report::ExpertCacheStatistics as Statistics;
pub(crate) use super::report::LoadStats;
use super::snapshot::{Plan, Range, Snapshot, Workspace};
use crate::{
    artifact::Result,
    execution::{ExecutionPhase, Invocation, LoadOptions},
};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

enum Implementation {
    Cuda(Box<super::cuda::executor::Executor>),
}
pub(crate) struct Executor {
    implementation: Implementation,
}
impl Executor {
    pub(crate) fn load(
        prepared: &crate::loader::PreparedModel,
        options: LoadOptions,
        preload: &BTreeSet<String>,
    ) -> Result<(Self, LoadStats)> {
        let target = orinfer_device::Target::from_package(&prepared.plan.target)?;
        target.require_host(std::env::consts::ARCH)?;
        match target.backend {
            orinfer_device::Backend::Cuda => {
                let (inner,stats)=super::cuda::executor::Executor::load(prepared,options,preload)?;
                Ok((Self { implementation: Implementation::Cuda(Box::new(inner)) },stats))
            }
            orinfer_device::Backend::Ascend => Err("This package requires Ascend model execution; no Ascend model provider is registered yet".into()),
        }
    }
    pub(crate) fn batch_statistics(&self) -> crate::scheduler::BatchExecutionStatistics {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.batch_statistics.get(),
        }
    }
    pub(crate) fn set_batch_statistics(
        &self,
        statistics: crate::scheduler::BatchExecutionStatistics,
    ) {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.batch_statistics.set(statistics),
        }
    }
    pub(crate) fn peak_kv_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.peak_kv_bytes.get(),
        }
    }
    pub(crate) fn peak_prefill_workspace_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.peak_prefill_workspace_bytes.get(),
        }
    }
    pub(crate) fn workspace_save_s(&self) -> f64 {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.workspace_save_s,
        }
    }
    pub(crate) fn workspace_restore_s(&self) -> f64 {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.workspace_restore_s,
        }
    }
    pub(crate) fn buffer_size(&self, name: &str) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.sizes[name],
        }
    }
    pub(crate) fn cleanup(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.session.cleanup(),
        }
    }
    pub(crate) fn read_ready_bytes(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_ready_bytes(name, bytes),
        }
    }
    pub(crate) fn read_ready_control(&self, name: &str) -> Result<i32> {
        let raw = self.read_ready_bytes(name, 4)?;
        Ok(i32::from_le_bytes(
            raw.try_into().map_err(|_| "Invalid control width")?,
        ))
    }
    pub(crate) fn sync(&self) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.sync(),
        }
    }
    pub(crate) fn reset_sequence(&self, names: &[String]) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.reset_sequence(names),
        }
    }
    pub(crate) fn reset_sequence_reusing_pages(&self, names: &[String]) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.reset_sequence_reusing_pages(names),
        }
    }
    pub(crate) fn download_bytes(&self, name: &str, bytes: usize) -> Result<Vec<u8>> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.download_bytes(name, bytes),
        }
    }
    pub(crate) fn copy_range(
        &self,
        source: &str,
        destination: &str,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.copy_range(source, destination, offset, bytes),
        }
    }
    pub(crate) fn copy_range_async(
        &self,
        source: &str,
        source_offset: usize,
        destination: &str,
        offset: usize,
        bytes: usize,
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => {
                inner.copy_range_async(source, source_offset, destination, offset, bytes)
            }
        }
    }
    pub(crate) fn launch_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.launch_program(name, phase),
        }
    }
    pub(crate) fn submit_program(&self, name: &str, phase: ExecutionPhase) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.submit_program(name, phase),
        }
    }
    #[cfg(test)]
    pub(crate) fn ensure_workspace(&self, name: &str, bytes: usize) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.ensure_workspace(name, bytes),
        }
    }
    pub(crate) fn pending_workspace_bytes(&self, programs: &[&str]) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.pending_workspace_bytes(programs),
        }
    }
    pub(crate) fn pending_workspace_range(&self, name: &str, bytes: usize) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.pending_workspace_range(name, bytes),
        }
    }
    pub(crate) fn pending_text_workspace_bytes(&self) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.pending_text_workspace_bytes(),
        }
    }
    pub(crate) fn upload_ids(&self, name: &str, ids: &[u32]) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.upload_ids(name, ids),
        }
    }
    pub(crate) fn read_control(&self, name: &str) -> Result<i32> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_control(name),
        }
    }
    pub(crate) fn read_control_group<const N: usize>(&self, names: [&str; N]) -> Result<[i32; N]> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_control_group::<N>(names),
        }
    }
    pub(crate) fn upload_bytes(&self, name: &str, bytes: &[u8]) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.upload_bytes(name, bytes),
        }
    }
    pub(crate) fn upload_bytes_at(&self, name: &str, offset: usize, bytes: &[u8]) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.upload_bytes_at(name, offset, bytes),
        }
    }
    pub(crate) fn read_controls(&self, name: &str, count: usize) -> Result<Vec<u32>> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_controls(name, count),
        }
    }
    pub(crate) fn active_request_slot(&self) -> Option<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.active_request_slot(),
        }
    }
    #[cfg(test)]
    pub(crate) fn set_graph_mode_for_test(
        &mut self,
        mode: crate::execution::CudaGraphMode,
    ) -> crate::execution::CudaGraphMode {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.set_graph_mode_for_test(mode),
        }
    }
    pub(crate) fn upload_scratch_rows(
        &self,
        table: &str,
        scratch: &str,
        stride: usize,
        source: &str,
        source_bytes: usize,
        slots: &[usize],
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => {
                inner.upload_scratch_rows(table, scratch, stride, source, source_bytes, slots)
            }
        }
    }
    pub(crate) fn submit_program_rows(
        &self,
        name: &str,
        rows: usize,
        phase: crate::execution::ExecutionPhase,
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.submit_program_rows(name, rows, phase),
        }
    }
    pub(crate) fn upload_sequence_addresses(
        &self,
        name: &str,
        bindings: std::rc::Rc<orinfer_model_sdk::abi::StateBindings>,
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.upload_sequence_addresses(name, bindings),
        }
    }
    pub(crate) fn upload_sequence_address_ranges(
        &self,
        bindings: std::rc::Rc<orinfer_model_sdk::abi::StateBindings>,
        tables: &[(&str, std::ops::Range<usize>)],
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.upload_sequence_address_ranges(bindings, tables),
        }
    }
    pub(crate) fn read_sequence_controls<const N: usize>(
        &self,
        slots: &[usize],
        names: [&str; N],
    ) -> Result<Vec<[i32; N]>> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_sequence_controls::<N>(slots, names),
        }
    }
    pub(crate) fn read_sequence_vectors(
        &self,
        slots: &[usize],
        names: &[&str],
        count: usize,
    ) -> Result<Vec<u32>> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.read_sequence_vectors(slots, names, count),
        }
    }
    #[cfg(test)]
    pub(crate) fn private_buffer_sizes(&self) -> BTreeMap<String, usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.private_buffer_sizes(),
        }
    }
    pub(crate) fn free_bytes(&self) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.free_bytes(),
        }
    }
    pub(crate) fn lease_sequence(&mut self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.lease_sequence(limits),
        }
    }
    pub(crate) fn activate_sequence(&mut self, slot: usize) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.activate_sequence(slot),
        }
    }
    pub(crate) fn reclaim_idle_graphs(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.reclaim_idle_graphs(),
        }
    }
    pub(crate) fn reserved_kv_bytes(&self, tokens: usize) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.reserved_kv_bytes(tokens),
        }
    }
    #[cfg(test)]
    pub(crate) fn resident_kv_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.resident_kv_bytes(),
        }
    }
    #[cfg(test)]
    pub(crate) fn shared_kv_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.shared_kv_bytes(),
        }
    }
    pub(crate) fn active_kv_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.active_kv_bytes(),
        }
    }
    pub(crate) fn additional_kv_bytes(
        &self,
        tokens: usize,
        limits: &BTreeMap<String, usize>,
    ) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.additional_kv_bytes(tokens, limits),
        }
    }
    pub(crate) fn pending_prefill_workspace_bytes(&self, tokens: usize) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.pending_prefill_workspace_bytes(tokens),
        }
    }
    pub(crate) fn additional_state_bytes(&self, limits: &BTreeMap<String, usize>) -> Result<usize> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.additional_state_bytes(limits),
        }
    }
    pub(crate) fn reclaim_idle_state(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.reclaim_idle_state(),
        }
    }
    pub(crate) fn release_sequence(&mut self, slot: usize, resets: &[String]) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.release_sequence(slot, resets),
        }
    }
    #[cfg(test)]
    pub(crate) fn prepare_reference_sequence(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.prepare_reference_sequence(),
        }
    }
    pub(crate) fn ensure_sequence_program(&mut self, slot: usize, program: &str) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.ensure_sequence_program(slot, program),
        }
    }
    pub(crate) fn ensure_sequence_program_at(
        &mut self,
        slot: usize,
        program: &str,
        position: usize,
        tokens: usize,
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => {
                inner.ensure_sequence_program_at(slot, program, position, tokens)
            }
        }
    }
    pub(crate) fn prepare_batch_workspace(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.prepare_batch_workspace(),
        }
    }
    pub(crate) fn upload_sequence_bytes(
        &mut self,
        slot: usize,
        name: &str,
        bytes: &[u8],
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.upload_sequence_bytes(slot, name, bytes),
        }
    }
    pub(crate) fn upload_sequence_ids(
        &mut self,
        slot: usize,
        name: &str,
        ids: &[u32],
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.upload_sequence_ids(slot, name, ids),
        }
    }
    pub(crate) fn copy_to_sequence_async(
        &mut self,
        source: &str,
        source_offset: usize,
        slot: usize,
        destination: &str,
        bytes: usize,
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => {
                inner.copy_to_sequence_async(source, source_offset, slot, destination, bytes)
            }
        }
    }
    pub(crate) fn has_batch_graph(&self, key: &[(usize, usize)], decode: bool) -> bool {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.has_batch_graph(key, decode),
        }
    }
    pub(crate) fn execute_batch(
        &mut self,
        key: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => {
                inner.execute_batch(key, operations, decode, allow_capture)
            }
        }
    }
    #[cfg(test)]
    pub(crate) fn execute_batch_operation(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &[Invocation],
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.execute_batch_operation(
                operation,
                segments,
                operations,
                decode,
                allow_capture,
            ),
        }
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
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.execute_batch_operation_with(
                operation,
                segments,
                operations,
                decode,
                allow_capture,
                overlap,
            ),
        }
    }
    pub(crate) fn execute_cached_batch_operation(
        &mut self,
        operation: orinfer_model_sdk::abi::BatchOperation,
        segments: Vec<(usize, usize)>,
        operations: &std::rc::Rc<Vec<Invocation>>,
        decode: bool,
        allow_capture: bool,
    ) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.execute_cached_batch_operation(
                operation,
                segments,
                operations,
                decode,
                allow_capture,
            ),
        }
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
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.execute_cached_batch_operation_with(
                operation,
                segments,
                operations,
                decode,
                allow_capture,
                overlap,
            ),
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_events(&mut self, count: usize) -> Result<Vec<sys::CUevent>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.profile_events(count),
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_hot_graph(
        &mut self,
        key: &[(usize, usize)],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.profile_hot_graph(key, repetitions),
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_invocations_hot(
        &mut self,
        operations: &[Invocation],
        repetitions: usize,
    ) -> Result<Vec<f32>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.profile_invocations_hot(operations, repetitions),
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_graph_operations(
        &mut self,
        operations: &[Invocation],
    ) -> Result<Vec<Vec<f32>>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.profile_graph_operations(operations),
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_stream_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: (&str, u32),
    ) -> Result<Vec<Vec<f32>>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => {
                inner.profile_stream_operations_at_position(operations, position)
            }
        }
    }
    #[cfg(test)]
    pub(crate) fn profile_graph_operations_at_position(
        &mut self,
        operations: &[Invocation],
        position: Option<(&str, u32)>,
    ) -> Result<Vec<Vec<f32>>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => {
                inner.profile_graph_operations_at_position(operations, position)
            }
        }
    }
    #[cfg(test)]
    pub(crate) fn snapshot_resident_bytes(&self) -> usize {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.snapshot_resident_bytes(),
        }
    }
    pub(crate) fn plan_snapshot(
        &self,
        ranges: BTreeMap<String, Range>,
        kv_names: &BTreeSet<String>,
    ) -> Result<Plan> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.plan_snapshot(ranges, kv_names),
        }
    }
    pub(crate) fn snapshot(&mut self, plan: &Plan) -> Result<Option<Snapshot>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.snapshot(plan),
        }
    }
    pub(crate) fn reserve_workspace(
        &mut self,
        ranges: BTreeMap<String, Range>,
    ) -> Result<Option<Rc<RefCell<Workspace>>>> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.reserve_workspace(ranges),
        }
    }
    pub(crate) fn retain_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.retain_workspace(workspace),
        }
    }
    pub(crate) fn resume_workspace(&mut self, workspace: &Rc<RefCell<Workspace>>) -> Result<f64> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.resume_workspace(workspace),
        }
    }
    pub(crate) fn restore_snapshot(&self, snapshot: &Snapshot) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.restore_snapshot(snapshot),
        }
    }
    pub(crate) fn offload_snapshot(
        &self,
        snapshot: &Snapshot,
        pool: &mut super::kv::host::HostPool,
    ) -> Result<super::kv::host::HostSnapshot> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.offload_snapshot(snapshot, pool),
        }
    }
    pub(crate) fn host_snapshot(
        &mut self,
        plan: &Plan,
        pool: &mut super::kv::host::HostPool,
    ) -> Result<super::kv::host::HostSnapshot> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.host_snapshot(plan, pool),
        }
    }
    pub(crate) fn restore_host_snapshot(
        &self,
        snapshot: &super::kv::host::HostSnapshot,
    ) -> Result<()> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.restore_host_snapshot(snapshot),
        }
    }
    pub(crate) fn release_snapshot(&mut self, snapshot: Snapshot) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.release_snapshot(snapshot),
        }
    }
    pub(crate) fn collect_snapshots(&mut self) -> Result<()> {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner.collect_snapshots(),
        }
    }
    pub(crate) fn expert_statistics(&self) -> Option<Statistics> {
        match &self.implementation {
            Implementation::Cuda(inner) => inner.expert_statistics(),
        }
    }
}
// Device-specific diagnostic tests can inspect native state. Production callers
// use only the execution boundary above.
#[cfg(test)]
impl std::ops::Deref for Executor {
    type Target = super::cuda::executor::Executor;
    fn deref(&self) -> &Self::Target {
        match &self.implementation {
            Implementation::Cuda(inner) => inner,
        }
    }
}
#[cfg(test)]
impl std::ops::DerefMut for Executor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match &mut self.implementation {
            Implementation::Cuda(inner) => inner,
        }
    }
}
