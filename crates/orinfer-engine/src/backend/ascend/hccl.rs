//! Stream-ordered native collectives. Communicators retain their device context.
use super::{
    DriverLease, acquire,
    api::Handle,
    runtime::{Buffer, Context, Session, Ticket},
};
use crate::artifact::Result;
use libloading::Library;
use std::{ffi::c_void, path::Path, ptr, rc::Rc, sync::Arc};
pub mod validation;

pub const ROOT_BYTES: usize = 4108;
#[repr(C)]
#[derive(Clone)]
struct NativeRoot {
    bytes: [u8; ROOT_BYTES],
}
type ReduceFn = unsafe extern "C" fn(Handle, Handle, u64, i32, i32, Handle, Handle) -> i32;
struct Api {
    _library: Library,
    root: unsafe extern "C" fn(*mut NativeRoot) -> i32,
    init: unsafe extern "C" fn(u32, *const NativeRoot, u32, *mut Handle) -> i32,
    destroy: unsafe extern "C" fn(Handle) -> i32,
    async_error: unsafe extern "C" fn(Handle, *mut i32) -> i32,
    reduce: ReduceFn,
    scatter: Option<ReduceFn>,
    gather: unsafe extern "C" fn(Handle, Handle, u64, i32, Handle, Handle) -> i32,
    exchange: unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *const c_void,
        i32,
        *const c_void,
        *const c_void,
        *const c_void,
        i32,
        Handle,
        Handle,
    ) -> i32,
}
impl Api {
    fn load() -> Result<Arc<Self>> {
        let path = std::env::var_os("ORINFER_HCCL_LIBRARY")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("ASCEND_HOME_PATH")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| "/usr/local/Ascend/ascend-toolkit/latest".into())
                    .join("lib64/libhccl.so")
            });
        Self::from_path(&path)
    }
    fn from_path(path: &Path) -> Result<Arc<Self>> {
        // SAFETY: Public C ABI from hccl.h and hccl_comm.h. The owner is retained
        // by root descriptors, communicators, and every in-flight collective.
        unsafe {
            let library = Library::new(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Ok(Arc::new(Self {
                root: *library
                    .get(b"HcclGetRootInfo\0")
                    .map_err(|e| e.to_string())?,
                init: *library
                    .get(b"HcclCommInitRootInfo\0")
                    .map_err(|e| e.to_string())?,
                destroy: *library
                    .get(b"HcclCommDestroy\0")
                    .map_err(|e| e.to_string())?,
                async_error: *library
                    .get(b"HcclGetCommAsyncError\0")
                    .map_err(|e| e.to_string())?,
                reduce: *library.get(b"HcclAllReduce\0").map_err(|e| e.to_string())?,
                scatter: library
                    .get::<ReduceFn>(b"HcclReduceScatter\0")
                    .ok()
                    .map(|s| *s),
                gather: *library.get(b"HcclAllGather\0").map_err(|e| e.to_string())?,
                exchange: *library.get(b"HcclAlltoAllV\0").map_err(|e| e.to_string())?,
                _library: library,
            }))
        }
    }
}
fn check(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        crate::error::record_ascend(status);
        Err(format!("{operation}: HCCL error {status}"))
    }
}

/// Opaque bootstrap bytes may be transported between rank processes. Do not log
/// their contents. The descriptor retains process ACL initialization while used.
#[derive(Clone)]
pub struct RootInfo {
    native: NativeRoot,
    ranks: u32,
    api: Arc<Api>,
    _driver: Arc<DriverLease>,
}
impl RootInfo {
    pub fn new(session: &Session, ranks: u32) -> Result<Self> {
        Self::with_api(session, ranks, Api::load()?)
    }
    /// Keep the rank reduction order independent of collective message length.
    /// HCCL reads its process environment at initialization, before rank joins.
    /// Ordinary deterministic mode can change BF16 order with message length.
    pub fn new_ordered(session: &Session, ranks: u32) -> Result<Self> {
        if !std::env::var("HCCL_DETERMINISTIC")
            .is_ok_and(|value| value.eq_ignore_ascii_case("strict"))
        {
            return Err("Variable-row BF16 reductions require HCCL_DETERMINISTIC=strict".into());
        }
        Self::new(session, ranks)
    }
    fn with_api(session: &Session, ranks: u32, api: Arc<Api>) -> Result<Self> {
        if ranks == 0 || ranks > 1024 {
            return Err("Invalid HCCL rank count".into());
        }
        session.same_context(&session.context)?;
        let _current = session.context.enter()?;
        let driver = Arc::clone(&session.context.driver);
        let mut native = NativeRoot {
            bytes: [0; ROOT_BYTES],
        };
        // SAFETY: Live retained library and correctly sized C HcclRootInfo output.
        check(unsafe { (api.root)(&mut native) }, "create HCCL root info")?;
        Ok(Self {
            native,
            ranks,
            api,
            _driver: driver,
        })
    }
    pub fn from_bytes(ranks: u32, bytes: &[u8]) -> Result<Self> {
        if ranks == 0 || ranks > 1024 {
            return Err("Invalid HCCL rank count".into());
        }
        let bytes = bytes
            .try_into()
            .map_err(|_| "Invalid HCCL root info byte length")?;
        Ok(Self {
            native: NativeRoot { bytes },
            ranks,
            api: Api::load()?,
            _driver: acquire()?,
        })
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.native.bytes
    }
    pub fn ranks(&self) -> u32 {
        self.ranks
    }
    #[cfg(test)]
    pub(super) fn fixture(session: &Session, path: &Path) -> Result<Self> {
        Self::fixture_ranks(session, path, 1)
    }
    #[cfg(test)]
    pub(super) fn fixture_ranks(session: &Session, path: &Path, ranks: u32) -> Result<Self> {
        Self::with_api(session, ranks, Api::from_path(path)?)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Element {
    Int8,
    Int32,
    Int64,
    Float32,
    Bfloat16,
}
impl Element {
    fn native(self) -> i32 {
        match self {
            Self::Int8 => 0,
            Self::Int32 => 2,
            Self::Float32 => 4,
            Self::Int64 => 5,
            Self::Bfloat16 => 11,
        }
    }
    pub(super) fn bytes(self) -> u64 {
        match self {
            Self::Int8 => 1,
            Self::Bfloat16 => 2,
            Self::Int32 | Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}
pub struct Counts {
    counts: Vec<u64>,
    offsets: Vec<u64>,
    elements: u64,
}
impl Counts {
    pub fn new(counts: Vec<u64>) -> Result<Self> {
        if counts.is_empty() || counts.len() > 1024 {
            return Err("Invalid HCCL count vector length".into());
        }
        let mut elements = 0u64;
        let offsets = counts
            .iter()
            .map(|&count| {
                let offset = elements;
                elements = elements
                    .checked_add(count)
                    .ok_or("HCCL count sum overflow")?;
                Ok(offset)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            counts,
            offsets,
            elements,
        })
    }
    fn validate(&self, peers: u32, element: Element, bytes: usize) -> Result<()> {
        if self.counts.len() != peers as usize
            || self
                .elements
                .checked_mul(element.bytes())
                .is_none_or(|n| n > bytes as u64)
        {
            return Err("HCCL counts exceed collective buffer or rank count".into());
        }
        Ok(())
    }
    pub fn elements(&self) -> u64 {
        self.elements
    }
}
struct NativeComm {
    api: Arc<Api>,
    context: Rc<Context>,
    handle: Handle,
}
impl NativeComm {
    fn status(&self, status: i32, operation: &str) -> Result<()> {
        let result = check(status, operation);
        if result.is_err() {
            self.context.retire();
        }
        result
    }
}
impl Drop for NativeComm {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        if let Ok(_current) = self.context.enter() {
            // SAFETY: Completions retain the communicator, so no submitted
            // operation remains when its last owner is destroyed.
            let status = unsafe { (self.api.destroy)(self.handle) };
            if status == 0 {
                return;
            }
            let _ = self.status(status, "destroy HCCL communicator");
        }
        self.context.retire();
        // Failed teardown cannot release the runtime/library under live HCCL state.
        std::mem::forget(Rc::clone(&self.context));
        std::mem::forget(Arc::clone(&self.api));
    }
}
#[derive(Clone)]
pub struct Communicator {
    native: Rc<NativeComm>,
    rank: u32,
    ranks: u32,
}
impl Communicator {
    pub fn join(session: &Session, root: &RootInfo, rank: u32) -> Result<Self> {
        session.same_context(&session.context)?;
        if rank >= root.ranks {
            return Err("HCCL rank outside bootstrap group".into());
        }
        let _current = session.context.enter()?;
        let mut native = NativeComm {
            api: Arc::clone(&root.api),
            context: Rc::clone(&session.context),
            handle: ptr::null_mut(),
        };
        // SAFETY: Context is current, rank is valid, root retains the native
        // bootstrap bytes. Peers join this same group concurrently.
        let status =
            unsafe { (native.api.init)(root.ranks, &root.native, rank, &mut native.handle) };
        // Failed initialization does not transfer output ownership. HCCL can
        // write a handle and roll it back internally before returning an error.
        if status != 0 {
            native.handle = ptr::null_mut();
        }
        native.status(status, "join HCCL group")?;
        if native.handle.is_null() {
            return Err("HCCL returned null communicator".into());
        }
        Ok(Self {
            native: Rc::new(native),
            rank,
            ranks: root.ranks,
        })
    }
    pub fn rank(&self) -> u32 {
        self.rank
    }
    pub fn ranks(&self) -> u32 {
        self.ranks
    }
    pub fn check_async_error(&self) -> Result<()> {
        let _current = self.native.context.enter()?;
        let mut status = 0;
        self.native.status(
            // SAFETY: Communicator and status output are live in this context.
            unsafe { (self.native.api.async_error)(self.native.handle, &mut status) },
            "query HCCL async error",
        )?;
        self.native.status(status, "HCCL asynchronous collective")
    }
    fn validate(&self, session: &Session, source: &Buffer, destination: &Buffer) -> Result<()> {
        session.same_context(&self.native.context)?;
        session.require(source)?;
        session.require(destination)?;
        Ok(())
    }
    pub fn all_reduce_sum(
        &self,
        session: &Session,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<Ticket> {
        self.validate(session, source, destination)?;
        if source.overlaps(destination) && source.pointer() != destination.pointer() {
            return Err("HCCL AllReduce requires identical or disjoint spans".into());
        }
        let bytes = count
            .checked_mul(element.bytes())
            .ok_or("HCCL reduction byte size overflow")?;
        if count == 0 || bytes > source.bytes() as u64 || bytes > destination.bytes() as u64 {
            return Err("HCCL reduction outside buffer".into());
        }
        let native = Rc::clone(&self.native);
        // SAFETY: Checked same-context extents and datatype. Completion retains
        // both buffers and the communicator through the SUM collective fence.
        unsafe {
            session.submit_compute(
                vec![source.clone(), destination.clone()],
                Box::new(Rc::clone(&native)),
                |stream| {
                    native.status(
                        (native.api.reduce)(
                            source.pointer(),
                            destination.pointer(),
                            count,
                            element.native(),
                            0,
                            native.handle,
                            stream,
                        ),
                        "HCCL AllReduce",
                    )
                },
            )
        }
    }
    /// Reduce a rank-major input to this rank's contiguous output partition.
    /// `count` is the number of output elements on each rank.
    pub fn reduce_scatter_sum(
        &self,
        session: &Session,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<Ticket> {
        self.validate_scatter(session, source, destination, count, element)?;
        let comm = self.clone();
        // SAFETY: Same-context, disjoint ranges and both element extents are
        // checked. The completion retains the communicator and allocations.
        unsafe {
            session.submit_compute(
                vec![],
                Box::new((self.clone(), source.clone(), destination.clone())),
                move |stream| comm.scatter_on_stream(stream, source, destination, count, element),
            )
        }
    }

    pub(super) fn validate_scatter(
        &self,
        session: &Session,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<()> {
        self.validate(session, source, destination)?;
        if self.native.api.scatter.is_none() {
            return Err("HCCL library does not provide ReduceScatter".into());
        }
        let output = count
            .checked_mul(element.bytes())
            .ok_or("HCCL ReduceScatter byte size overflow")?;
        let input = output
            .checked_mul(u64::from(self.ranks))
            .ok_or("HCCL ReduceScatter input size overflow")?;
        if source.overlaps(destination)
            || count == 0
            || input > source.bytes() as u64
            || output > destination.bytes() as u64
        {
            return Err("HCCL ReduceScatter requires disjoint spans with valid extents".into());
        }
        Ok(())
    }

    pub(super) fn scatter_on_stream(
        &self,
        stream: Handle,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<()> {
        let scatter = self
            .native
            .api
            .scatter
            .ok_or("HCCL ReduceScatter unavailable")?;
        // SAFETY: The enclosing completion or captured segment validated both
        // extents and retains the same-context buffers and communicator.
        self.native.status(
            unsafe {
                scatter(
                    source.pointer(),
                    destination.pointer(),
                    count,
                    element.native(),
                    0,
                    self.native.handle,
                    stream,
                )
            },
            "HCCL ReduceScatter",
        )
    }

    pub fn all_gather(
        &self,
        session: &Session,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<Ticket> {
        self.validate(session, source, destination)?;
        let bytes = count
            .checked_mul(element.bytes())
            .ok_or("HCCL gather byte size overflow")?;
        let output = bytes
            .checked_mul(u64::from(self.ranks))
            .ok_or("HCCL gather output overflow")?;
        if source.overlaps(destination)
            || count == 0
            || bytes > source.bytes() as u64
            || output > destination.bytes() as u64
        {
            return Err("HCCL AllGather requires disjoint spans with valid extents".into());
        }
        let comm = self.clone();
        // SAFETY: Same-context, disjoint spans and element extents are checked;
        // the completion retains the communicator and both device allocations.
        unsafe {
            session.submit_compute(
                vec![],
                Box::new((self.clone(), source.clone(), destination.clone())),
                move |stream| comm.gather_on_stream(stream, source, destination, count, element),
            )
        }
    }

    pub(super) fn gather_on_stream(
        &self,
        stream: Handle,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<()> {
        // SAFETY: The caller checks spans and retains all owners through the
        // enclosing completion; a captured graph uses the same stable bindings.
        self.native.status(
            unsafe {
                (self.native.api.gather)(
                    source.pointer(),
                    destination.pointer(),
                    count,
                    element.native(),
                    self.native.handle,
                    stream,
                )
            },
            "HCCL AllGather",
        )
    }
    /// Submits one validated SUM collective on an explicit stream. Used by
    /// captured segments that replay the collective inside an ACL Graph.
    pub(super) fn reduce_on_stream(
        &self,
        stream: super::api::Handle,
        source: &Buffer,
        destination: &Buffer,
        count: u64,
        element: Element,
    ) -> Result<()> {
        let native = &self.native;
        // SAFETY: The caller validated extents and datatype when the segment
        // was compiled; buffers are retained by the captured graph.
        native.status(
            unsafe {
                (native.api.reduce)(
                    source.pointer(),
                    destination.pointer(),
                    count,
                    element.native(),
                    0,
                    native.handle,
                    stream,
                )
            },
            "HCCL AllReduce",
        )
    }
    pub fn all_to_all_v(
        &self,
        session: &Session,
        source: &Buffer,
        destination: &Buffer,
        send: Counts,
        receive: Counts,
        element: Element,
    ) -> Result<Ticket> {
        self.validate(session, source, destination)?;
        send.validate(self.ranks, element, source.bytes())?;
        receive.validate(self.ranks, element, destination.bytes())?;
        if source.overlaps(destination) {
            return Err("HCCL AlltoAllV requires disjoint buffers".into());
        }
        let native = Rc::clone(&self.native);
        let counts = Rc::new((send, receive));
        // SAFETY: Exact HcclAlltoAllV ABI and checked byte extents. Both host count
        // arrays, communicator and device buffers are retained through completion.
        unsafe {
            session.submit_compute(
                vec![source.clone(), destination.clone()],
                Box::new((Rc::clone(&native), Rc::clone(&counts))),
                |stream| {
                    native.status(
                        (native.api.exchange)(
                            source.pointer(),
                            counts.0.counts.as_ptr().cast(),
                            counts.0.offsets.as_ptr().cast(),
                            element.native(),
                            destination.pointer(),
                            counts.1.counts.as_ptr().cast(),
                            counts.1.offsets.as_ptr().cast(),
                            element.native(),
                            native.handle,
                            stream,
                        ),
                        "HCCL AlltoAllV",
                    )
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_include_empty_peers_and_check_units_and_overflow() {
        let counts = Counts::new(vec![0, 2, 1]).unwrap();
        assert_eq!(counts.offsets, [0, 0, 2]);
        counts.validate(3, Element::Float32, 12).unwrap();
        assert!(counts.validate(3, Element::Float32, 11).is_err());
        assert!(counts.validate(2, Element::Float32, 12).is_err());
        assert!(Counts::new(vec![u64::MAX, 1]).is_err());
        assert!(
            Counts::new(vec![u64::MAX])
                .unwrap()
                .validate(1, Element::Int64, usize::MAX)
                .is_err()
        );
    }
}
