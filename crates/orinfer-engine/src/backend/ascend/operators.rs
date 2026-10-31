//! Native ACLNN grouped W4A8. Fixed-shape preparation is separate from execution.
use super::{
    api::Handle,
    check,
    runtime::{Buffer, Context, Session, Ticket},
};
use crate::artifact::Result;
use libloading::Library;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    path::{Path, PathBuf},
    ptr,
    rc::Rc,
};

type CreateTensor = unsafe extern "C" fn(
    *const i64,
    u64,
    i32,
    *const i64,
    i64,
    i32,
    *const i64,
    u64,
    Handle,
) -> Handle;
type CreateList = unsafe extern "C" fn(*const Handle, u64) -> Handle;
type CreateIntArray = unsafe extern "C" fn(*const i64, u64) -> Handle;
type Destroy = unsafe extern "C" fn(Handle) -> i32;
type Prepare = unsafe extern "C" fn(
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    Handle,
    i64,
    i64,
    i64,
    i64,
    Handle,
    Handle,
    Handle,
    Handle,
    *mut u64,
    *mut Handle,
) -> i32;
type Execute = unsafe extern "C" fn(Handle, u64, Handle, Handle) -> i32;

struct Api {
    _operator_library: Library,
    _meta_library: Library,
    create_tensor: CreateTensor,
    create_list: CreateList,
    create_int_array: CreateIntArray,
    destroy_tensor: Destroy,
    destroy_list: Destroy,
    destroy_int_array: Destroy,
    repeatable: Destroy,
    destroy_executor: Destroy,
    prepare: Prepare,
    execute: Execute,
}
impl Api {
    fn load() -> Result<Rc<Self>> {
        let root = std::env::var_os("ASCEND_HOME_PATH")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| "/usr/local/Ascend/ascend-toolkit/latest".into());
        let op_path = std::env::var_os("ORINFER_ASCEND_OPAPI")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| root.join("lib64/libopapi.so"));
        let meta_path = root.join("lib64/libnnopbase.so");
        // CANN operator libraries are process-wide; load each pair once per
        // thread. Sessions are thread-affine, so no cross-thread sharing.
        thread_local! {
            static CACHE: RefCell<BTreeMap<(PathBuf, PathBuf), Rc<Api>>> =
                const { RefCell::new(BTreeMap::new()) };
        }
        CACHE.with(|cache| {
            let mut cache = cache.borrow_mut();
            if let Some(api) = cache.get(&(op_path.clone(), meta_path.clone())) {
                return Ok(Rc::clone(api));
            }
            let api = Self::from_paths(&op_path, &meta_path)?;
            cache.insert((op_path, meta_path), Rc::clone(&api));
            Ok(api)
        })
    }
    fn from_paths(op: &Path, meta: &Path) -> Result<Rc<Self>> {
        // SAFETY: Explicitly selected CANN libraries; the owner retains them for
        // every tensor, executor and in-flight invocation using copied symbols.
        unsafe {
            let operator = Library::new(op).map_err(|e| format!("{}: {e}", op.display()))?;
            let metadata = Library::new(meta).map_err(|e| format!("{}: {e}", meta.display()))?;
            Ok(Rc::new(Self {
                create_tensor: *metadata
                    .get(b"aclCreateTensor\0")
                    .map_err(|e| e.to_string())?,
                create_list: *metadata
                    .get(b"aclCreateTensorList\0")
                    .map_err(|e| e.to_string())?,
                create_int_array: *metadata
                    .get(b"aclCreateIntArray\0")
                    .map_err(|e| e.to_string())?,
                destroy_tensor: *metadata
                    .get(b"aclDestroyTensor\0")
                    .map_err(|e| e.to_string())?,
                destroy_list: *metadata
                    .get(b"aclDestroyTensorList\0")
                    .map_err(|e| e.to_string())?,
                destroy_int_array: *metadata
                    .get(b"aclDestroyIntArray\0")
                    .map_err(|e| e.to_string())?,
                repeatable: *metadata
                    .get(b"aclSetAclOpExecutorRepeatable\0")
                    .map_err(|e| e.to_string())?,
                destroy_executor: *metadata
                    .get(b"aclDestroyAclOpExecutor\0")
                    .map_err(|e| e.to_string())?,
                prepare: *operator
                    .get(b"aclnnGroupedMatmulV5GetWorkspaceSize\0")
                    .map_err(|e| e.to_string())?,
                execute: *operator
                    .get(b"aclnnGroupedMatmulV5\0")
                    .map_err(|e| e.to_string())?,
                _operator_library: operator,
                _meta_library: metadata,
            }))
        }
    }
}

struct Tensor {
    api: Rc<Api>,
    handle: Handle,
    buffer: Buffer,
}
impl Tensor {
    fn new(
        api: &Rc<Api>,
        buffer: &Buffer,
        shape: &[usize],
        dtype: i32,
        bits: usize,
    ) -> Result<Self> {
        let elements = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d).filter(|_| d > 0))
            .ok_or("Empty/overflowing ACL tensor shape")?;
        let bytes = elements
            .checked_mul(bits)
            .and_then(|n| n.checked_add(7))
            .map(|n| n / 8)
            .ok_or("ACL tensor bytes overflow")?;
        if bytes != buffer.bytes() {
            return Err("ACL tensor shape/dtype differs from buffer extent".into());
        }
        let dims: Vec<i64> = shape
            .iter()
            .map(|&d| i64::try_from(d).map_err(|_| "ACL tensor dimension overflow"))
            .collect::<std::result::Result<_, _>>()?;
        let mut stride = vec![1i64; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1]
                .checked_mul(dims[i + 1])
                .ok_or("ACL tensor stride overflow")?;
        }
        // SAFETY: Checked contiguous ND layout spans this retained allocation.
        // ACL copies the metadata arrays; device backing remains owned by Tensor.
        let handle = unsafe {
            (api.create_tensor)(
                dims.as_ptr(),
                dims.len() as u64,
                dtype,
                stride.as_ptr(),
                0,
                2,
                dims.as_ptr(),
                dims.len() as u64,
                buffer.pointer(),
            )
        };
        if handle.is_null() {
            return Err("aclCreateTensor returned null".into());
        }
        Ok(Self {
            api: Rc::clone(api),
            handle,
            buffer: buffer.clone(),
        })
    }
}
impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: This tensor has not transferred metadata ownership to a list.
            unsafe {
                (self.api.destroy_tensor)(self.handle);
            }
        }
    }
}
struct TensorList {
    api: Rc<Api>,
    handle: Handle,
    _buffers: Vec<Buffer>,
}
impl TensorList {
    fn new(api: &Rc<Api>, mut tensors: Vec<Tensor>) -> Result<Self> {
        let handles: Vec<_> = tensors.iter().map(|t| t.handle).collect();
        // SAFETY: Every tensor is live and comes from this same metadata library.
        let handle = unsafe { (api.create_list)(handles.as_ptr(), handles.len() as u64) };
        if handle.is_null() {
            return Err("aclCreateTensorList returned null".into());
        }
        let buffers = tensors
            .iter_mut()
            .map(|tensor| {
                // aclCreateTensorList takes ownership of its tensors. Destroy only
                // the list, which destroys its children; retain device storage here.
                tensor.handle = ptr::null_mut();
                tensor.buffer.clone()
            })
            .collect();
        Ok(Self {
            api: Rc::clone(api),
            handle,
            _buffers: buffers,
        })
    }
}
impl Drop for TensorList {
    fn drop(&mut self) {
        // SAFETY: The list solely owns its child tensor metadata; no job remains.
        unsafe {
            (self.api.destroy_list)(self.handle);
        }
    }
}

struct IntArray {
    api: Rc<Api>,
    handle: Handle,
    _values: Box<[i64; 2]>,
}
impl IntArray {
    fn new(api: &Rc<Api>, expected_rows: usize) -> Result<Self> {
        let expected = i64::try_from(expected_rows)
            .ok()
            .filter(|&n| n > 0)
            .ok_or("Invalid grouped matmul expert row estimate")?;
        let values = Box::new([expected, 0]);
        // SAFETY: Stable owned values and the metadata library outlive the
        // array, executor and every captured invocation that retains Prepared.
        let handle = unsafe { (api.create_int_array)(values.as_ptr(), 2) };
        if handle.is_null() {
            return Err("aclCreateIntArray returned null".into());
        }
        Ok(Self {
            api: Rc::clone(api),
            handle,
            _values: values,
        })
    }
}
impl Drop for IntArray {
    fn drop(&mut self) {
        // SAFETY: Sole owner; Prepared is released after all completion fences.
        unsafe {
            (self.api.destroy_int_array)(self.handle);
        }
    }
}

/// ND packed signed INT4 weights [experts,K,N], per-channel UINT64 dequant
/// scales [experts,1,N], FP32 assist [experts,N], token scales [M], BF16 output.
#[derive(Clone, Copy, Debug)]
pub struct GroupedShape {
    pub rows: usize,
    pub input: usize,
    pub output: usize,
    pub experts: usize,
}
impl GroupedShape {
    fn validate(self, ends: &[i64]) -> Result<()> {
        if self.rows == 0
            || self.input == 0
            || self.output == 0
            || self.experts == 0
            || !self.input.is_multiple_of(256)
            || !self.output.is_multiple_of(32)
            || ends.len() != self.experts
            || ends.windows(2).any(|w| w[0] > w[1])
            || ends.first().is_none_or(|&n| n < 0)
            || ends.last().copied() != i64::try_from(self.rows).ok()
        {
            return Err("Invalid grouped W4A8 geometry or cumulative expert row ends".into());
        }
        Ok(())
    }
}
pub struct GroupedInputs<'a> {
    pub activation: &'a Buffer,
    pub weight: &'a Buffer,
    pub weight_scale: &'a Buffer,
    pub assist: &'a Buffer,
    pub token_scale: &'a Buffer,
    pub output: &'a Buffer,
}
fn group_counts(ends: &[i64]) -> Vec<u8> {
    let mut previous = 0;
    ends.iter()
        .flat_map(|&end| {
            let count = end - previous;
            previous = end;
            count.to_le_bytes()
        })
        .collect()
}
struct Prepared {
    shape: GroupedShape,
    api: Rc<Api>,
    context: Rc<Context>,
    executor: Handle,
    workspace: Option<Buffer>,
    workspace_bytes: usize,
    _lists: Vec<TensorList>,
    _groups: Tensor,
    _tuning: Option<IntArray>,
}
impl Drop for Prepared {
    fn drop(&mut self) {
        if !self.executor.is_null() {
            // SAFETY: Reusable executor is owned here, all submitted jobs retain
            // Prepared until completion, and its tensor metadata is still live.
            unsafe {
                (self.api.destroy_executor)(self.executor);
            }
        }
    }
}
pub struct GroupedW4A8 {
    prepared: Rc<Prepared>,
    last: RefCell<Option<Ticket>>,
}
impl GroupedW4A8 {
    pub(super) fn branch_accesses(&self) -> Vec<(Buffer, bool)> {
        self.prepared
            ._lists
            .iter()
            .flat_map(|list| list._buffers.iter())
            .chain(std::iter::once(&self.prepared._groups.buffer))
            .chain(self.prepared.workspace.iter())
            // ACLNN transforms A8 in place. Conservatively treat every retained
            // span as writable when checking independence between branches.
            .cloned()
            .map(|buffer| (buffer, true))
            .collect()
    }
    pub fn prepare(
        session: &Session,
        shape: GroupedShape,
        ends: &[i64],
        input: GroupedInputs<'_>,
    ) -> Result<Self> {
        Self::prepare_with_workspace(session, shape, ends, input, None, None)
    }
    /// Plans on this session's single compute stream may share one stable
    /// workspace allocation. Parallel branches require disjoint workspaces.
    /// An external `groups` buffer lets a device kernel own the per-expert
    /// row counts instead of host uploads through update_groups.
    pub fn prepare_with_workspace(
        session: &Session,
        shape: GroupedShape,
        ends: &[i64],
        input: GroupedInputs<'_>,
        workspace: Option<&Buffer>,
        groups: Option<&Buffer>,
    ) -> Result<Self> {
        Self::prepare_with_workspace_hint(session, shape, ends, input, workspace, groups, None)
    }
    /// A positive expected row count per expert selects CANN's tiling without
    /// changing tensor capacity or the device-owned live routing partitions.
    pub fn prepare_with_workspace_hint(
        session: &Session,
        shape: GroupedShape,
        ends: &[i64],
        input: GroupedInputs<'_>,
        workspace: Option<&Buffer>,
        groups: Option<&Buffer>,
        expected_rows: Option<usize>,
    ) -> Result<Self> {
        shape.validate(ends)?;
        let buffers = [
            input.activation,
            input.weight,
            input.weight_scale,
            input.assist,
            input.token_scale,
            input.output,
        ];
        for buffer in buffers {
            session.require(buffer)?;
        }
        if buffers[..5]
            .iter()
            .any(|buffer| buffer.overlaps(input.output))
        {
            return Err("Grouped W4A8 output must not alias an input".into());
        }
        if let Some(workspace) = workspace {
            session.require(workspace)?;
            if buffers.iter().any(|buffer| buffer.overlaps(workspace)) {
                return Err("ACLNN workspace must not alias an operand".into());
            }
        }
        if let Some(groups) = groups {
            session.require(groups)?;
            if groups.bytes() != shape.experts * 8 {
                return Err("External W4A8 group counts extent differs".into());
            }
            if buffers.iter().any(|buffer| buffer.overlaps(groups)) {
                return Err("External W4A8 groups must not alias an operand".into());
            }
        }
        session.synchronize()?;
        let _current = session.context.enter()?;
        let api = Api::load()?;
        let GroupedShape {
            rows: m,
            input: k,
            output: n,
            experts: e,
        } = shape;
        let descriptors = [
            Tensor::new(&api, input.activation, &[m, k], 2, 8)?,
            Tensor::new(&api, input.weight, &[e, k, n], 29, 4)?,
            Tensor::new(&api, input.weight_scale, &[e, 1, n], 10, 64)?,
            Tensor::new(&api, input.assist, &[e, n], 0, 32)?,
            Tensor::new(&api, input.token_scale, &[m], 0, 32)?,
            Tensor::new(&api, input.output, &[m, n], 27, 16)?,
        ];
        let lists = descriptors
            .into_iter()
            .map(|tensor| TensorList::new(&api, vec![tensor]))
            .collect::<Result<Vec<_>>>()?;
        let group_data = group_counts(ends);
        let owned;
        let group_buffer = match groups {
            Some(external) => {
                session.upload(external, 0, &group_data)?.wait()?;
                external.clone()
            }
            None => {
                owned = session.allocate(group_data.len())?;
                session.upload(&owned, 0, &group_data)?.wait()?;
                owned
            }
        };
        let groups = Tensor::new(&api, &group_buffer, &[e], 9, 64)?;
        let tuning = expected_rows.map(|n| IntArray::new(&api, n)).transpose()?;
        let (mut workspace_bytes, mut executor) = (0u64, ptr::null_mut());
        check(
            // SAFETY: Public aclnnGroupedMatmulV5 ABI. ND layouts and same-session
            // backing were validated. Group rows are per-expert counts.
            unsafe {
                (api.prepare)(
                    lists[0].handle,
                    lists[1].handle,
                    lists[3].handle,
                    lists[2].handle,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    lists[4].handle,
                    groups.handle,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    2,
                    0,
                    1,
                    0,
                    tuning.as_ref().map_or(ptr::null_mut(), |a| a.handle),
                    lists[5].handle,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    &mut workspace_bytes,
                    &mut executor,
                )
            },
            "prepare ACLNN grouped W4A8",
        )?;
        if executor.is_null() {
            return Err("ACLNN returned a null executor".into());
        }
        // SAFETY: Called immediately after successful first-phase preparation.
        // Repeatable executors are explicitly destroyed after their last fence.
        let status = unsafe { (api.repeatable)(executor) };
        if status != 0 {
            // The API does not grant destroy ownership when reuse fails. No work
            // was submitted; retain the native plan and libraries until exit.
            std::mem::forget((
                lists,
                groups,
                tuning,
                api,
                executor,
                Rc::clone(&session.context),
            ));
            return Err(format!(
                "ACLNN executor reuse unavailable: ACL error {status}"
            ));
        }
        let mut prepared = Prepared {
            shape,
            api,
            context: Rc::clone(&session.context),
            executor,
            workspace: None,
            workspace_bytes: 0,
            _lists: lists,
            _groups: groups,
            _tuning: tuning,
        };
        prepared.workspace_bytes =
            usize::try_from(workspace_bytes).map_err(|_| "ACLNN workspace overflow")?;
        if prepared.workspace_bytes > 0 {
            if let Some(workspace) = workspace {
                if workspace.bytes() < prepared.workspace_bytes {
                    return Err("Shared ACLNN workspace is too small".into());
                }
                prepared.workspace = Some(workspace.clone());
            } else {
                let (free, _) = session.memory_info()?;
                if prepared.workspace_bytes > free.saturating_sub(256 << 20) {
                    return Err(format!(
                        "ACLNN workspace {} bytes would exhaust available HBM {free} bytes",
                        prepared.workspace_bytes
                    ));
                }
                prepared.workspace = Some(session.allocate(prepared.workspace_bytes)?);
            }
        }
        Ok(Self {
            prepared: Rc::new(prepared),
            last: RefCell::new(None),
        })
    }
    pub fn workspace_bytes(&self) -> usize {
        self.prepared.workspace_bytes
    }
    /// Update routing partitions without changing fixed tensor addresses/shapes.
    pub fn update_groups(&self, session: &Session, ends: &[i64]) -> Result<()> {
        session.same_context(&self.prepared.context)?;
        self.prepared.shape.validate(ends)?;
        if let Some(ticket) = self.last.borrow_mut().take() {
            ticket.wait()?;
        }
        // Captured graphs may also reference the group tensor. Fence their
        // readers before the transfer stream changes the routing partition.
        session.synchronize()?;
        let data = group_counts(ends);
        session
            .upload(&self.prepared._groups.buffer, 0, &data)?
            .wait()
    }
    pub(super) fn workspace(&self) -> Option<Buffer> {
        self.prepared.workspace.clone()
    }
    pub fn capture(&self, session: &Session) -> Result<super::graph::Graph> {
        session.same_context(&self.prepared.context)?;
        let prepared = Rc::clone(&self.prepared);
        let workspace = prepared
            .workspace
            .as_ref()
            .map_or(ptr::null_mut(), Buffer::pointer);
        // SAFETY: Prepared owns all tensors, workspace and native libraries at
        // stable addresses, and the graph retains it for capture and every replay.
        unsafe {
            super::graph::Graph::capture(session, Box::new(Rc::clone(&prepared)), |stream| {
                check(
                    (prepared.api.execute)(
                        workspace,
                        prepared.workspace_bytes as u64,
                        prepared.executor,
                        stream,
                    ),
                    "capture ACLNN grouped W4A8",
                )
            })
        }
    }
    pub fn enqueue(&self, session: &Session) -> Result<Ticket> {
        if !Rc::ptr_eq(&session.context, &self.prepared.context) {
            return Err("ACLNN plan belongs to another session".into());
        }
        if let Some(ticket) = self.last.borrow().as_ref()
            && !ticket.is_ready()?
        {
            return Err("ACLNN executor already has an in-flight invocation".into());
        }
        let prepared = Rc::clone(&self.prepared);
        let workspace = prepared
            .workspace
            .as_ref()
            .map_or(ptr::null_mut(), Buffer::pointer);
        // SAFETY: Prepared owns every input/output tensor, workspace, executor,
        // context and native library; the completion retains an Rc through DMA.
        let ticket = unsafe {
            session.submit_compute(vec![], Box::new(Rc::clone(&prepared)), |stream| {
                check(
                    (prepared.api.execute)(
                        workspace,
                        prepared.workspace_bytes as u64,
                        prepared.executor,
                        stream,
                    ),
                    "execute ACLNN grouped W4A8",
                )
            })
        }?;
        *self.last.borrow_mut() = Some(ticket.clone());
        Ok(ticket)
    }
    /// Submits the repeatable executor on an explicit stream without a
    /// completion fence. Used by captured segments that replay the grouped
    /// matmul inside an ACL Graph.
    pub(super) fn execute_on_stream(&self, stream: super::api::Handle) -> Result<()> {
        let prepared = &self.prepared;
        let workspace = prepared
            .workspace
            .as_ref()
            .map_or(ptr::null_mut(), Buffer::pointer);
        // SAFETY: Prepared owns every operand, workspace and executor; the
        // captured graph retains it through capture and every replay.
        check(
            unsafe {
                (prepared.api.execute)(
                    workspace,
                    prepared.workspace_bytes as u64,
                    prepared.executor,
                    stream,
                )
            },
            "execute ACLNN grouped W4A8",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouped_rows_allow_empty_experts_and_reject_ambiguous_partitions() {
        let shape = GroupedShape {
            rows: 4,
            input: 256,
            output: 32,
            experts: 3,
        };
        shape.validate(&[1, 1, 4]).unwrap();
        for bad in [&[-1, 1, 4][..], &[1, 4, 3], &[1, 2, 3], &[4]] {
            assert!(shape.validate(bad).is_err());
        }
    }
}
