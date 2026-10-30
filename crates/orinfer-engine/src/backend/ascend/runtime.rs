//! Owned CANN resources. Every asynchronous submission retains its storage.
use super::{
    DriverLease, acquire,
    api::{Api, Handle},
    check,
};
use crate::artifact::Result;
use orinfer_device::{Backend, DeviceOwner, MemoryRange};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    ptr,
    rc::Rc,
    sync::Arc,
};

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub device_bytes: usize,
    pub pinned_bytes: usize,
    pub timeout_ms: i32,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            device_bytes: 256 << 20,
            pinned_bytes: 16 << 20,
            timeout_ms: 10_000,
        }
    }
}

/// One thread owns a context and two ordered streams. Buffers and completions
/// retain the context, and Rc intentionally prevents moving it to another thread.
pub struct Session {
    pub(super) context: Rc<Context>,
    pending: RefCell<Vec<Rc<Completion>>>,
    // Session ownership avoids a Context -> Pinned -> Context Rc cycle.
    // Physical idle capacity remains charged to the pinned-memory budget.
    pinned_cache: RefCell<Vec<Rc<Pinned>>>,
}
pub(super) struct Context {
    pub driver: Arc<DriverLease>,
    pub api: Api,
    pub handle: Handle,
    compute: Handle,
    transfer: Handle,
    owner: DeviceOwner,
    limits: Limits,
    device_bytes: Cell<usize>,
    pinned_bytes: Cell<usize>,
    poisoned: Cell<bool>,
    // Only retired completion generations enter this context-local cache.
    // Raw handles avoid an Rc cycle back into their owning context.
    completion_events: RefCell<Vec<Handle>>,
}

pub(super) struct Current<'a> {
    context: &'a Context,
    previous: Handle,
}
impl Context {
    pub(super) fn retire(&self) {
        self.poisoned.set(true);
    }
    pub(super) fn compute_stream(&self) -> Handle {
        self.compute
    }
    pub(super) fn timeout_ms(&self) -> i32 {
        self.limits.timeout_ms
    }
    pub fn enter(&self) -> Result<Current<'_>> {
        let previous = self.api.current()?;
        if previous != self.handle {
            // SAFETY: Retained ACL library and owned context.
            unsafe {
                check(
                    (self.api.set_context)(self.handle),
                    "set current ACL context",
                )?;
            }
        }
        Ok(Current {
            context: self,
            previous,
        })
    }
    fn sync(&self, stream: Handle) -> Result<()> {
        let _current = self.enter()?;
        // SAFETY: This live stream belongs to this retained context.
        let result = check(
            unsafe { (self.api.sync_stream)(stream, self.limits.timeout_ms) },
            "synchronize ACL stream",
        );
        if result.is_err() {
            self.poisoned.set(true);
        }
        result
    }
    fn ready(&self) -> Result<()> {
        if self.poisoned.get() {
            return Err("Ascend session retired after an execution failure".into());
        }
        Ok(())
    }
}
impl Drop for Current<'_> {
    fn drop(&mut self) {
        if !self.previous.is_null() && self.previous != self.context.handle {
            // SAFETY: This guard restores the previous live thread-local context.
            unsafe {
                (self.context.api.set_context)(self.previous);
            }
        }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // No allocation or completion can outlive the last Rc<Context>. Never
        // reset a device: other processes and independently owned contexts exist.
        let mut previous = ptr::null_mut();
        // SAFETY: This sole owner retains the library through all destruction.
        unsafe {
            (self.api.get_context)(&mut previous);
            if !self.handle.is_null() {
                (self.api.set_context)(self.handle);
            }
            for event in self.completion_events.get_mut().drain(..) {
                // Every cached event was fenced and lost its last Ticket/waiter.
                (self.api.destroy_event)(event);
            }
            for stream in [self.transfer, self.compute] {
                if !stream.is_null() {
                    (self.api.sync_stream)(stream, self.limits.timeout_ms);
                    (self.api.destroy_stream)(stream);
                }
            }
            if !self.handle.is_null() {
                (self.api.destroy_context)(self.handle);
            }
            if !previous.is_null() && previous != self.handle {
                (self.api.set_context)(previous);
            }
        }
    }
}

#[derive(Clone)]
pub struct Buffer {
    allocation: Rc<Allocation>,
    offset: usize,
    bytes: usize,
}
struct Allocation {
    context: Rc<Context>,
    pointer: Handle,
    bytes: usize,
}
impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn pointer(&self) -> Handle {
        // SAFETY: Every view is checked against its retained allocation.
        unsafe { self.allocation.pointer.cast::<u8>().add(self.offset).cast() }
    }
    /// A bounded tensor view retains the original allocation and its context.
    pub fn slice(&self, offset: usize, bytes: usize) -> Result<Self> {
        MemoryRange { offset, bytes }.within(self.bytes())?;
        if bytes == 0 {
            return Err("Ascend tensor view must be nonempty".into());
        }
        Ok(Self {
            allocation: Rc::clone(&self.allocation),
            offset: self.offset + offset,
            bytes,
        })
    }
    pub(super) fn overlaps(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.allocation, &other.allocation)
            && self.offset < other.offset + other.bytes
            && other.offset < self.offset + self.bytes
    }
    fn range(&self, offset: usize, bytes: usize) -> Result<Handle> {
        MemoryRange { offset, bytes }.within(self.bytes())?;
        // SAFETY: Checked byte offset within this owned device allocation.
        Ok(unsafe { self.pointer().cast::<u8>().add(offset).cast() })
    }
}
impl Drop for Allocation {
    fn drop(&mut self) {
        if let Ok(_current) = self.context.enter() {
            // SAFETY: Async users retain Buffer until their completion. This is
            // the last allocation reference; the context remains live.
            if unsafe { (self.context.api.free)(self.pointer) } == 0 {
                self.context
                    .device_bytes
                    .set(self.context.device_bytes.get() - self.bytes);
            }
        }
    }
}

struct Pinned {
    context: Rc<Context>,
    pointer: Handle,
    bytes: usize,
}
impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: Completion retains this pinned storage through the DMA fence.
        if unsafe { (self.context.api.free_host)(self.pointer) } == 0 {
            self.context
                .pinned_bytes
                .set(self.context.pinned_bytes.get() - self.bytes);
        }
    }
}
struct Retained {
    _buffers: Vec<Buffer>,
    _extra: Option<Box<dyn Any>>,
    _host: Option<Rc<Pinned>>,
}
struct Completion {
    context: Rc<Context>,
    event: Handle,
    complete: Cell<bool>,
    retained: RefCell<Option<Retained>>,
}
impl Completion {
    fn done(&self) -> Result<bool> {
        if self.complete.get() {
            return Ok(true);
        }
        let _current = self.context.enter()?;
        // aclrtEventRecordedStatus: NOT_READY=0, COMPLETE=1. Wait-status is
        // different: no pending stream waiters does not complete the record.
        let mut state = 0;
        check(
            // SAFETY: Recorded event belongs to the retained context; state is live.
            unsafe { (self.context.api.query_event)(self.event, &mut state) },
            "query ACL completion",
        )?;
        match state {
            1 => {
                // Keep retirement through wait()/Drop even after a completed
                // record query, so retained DMA storage has one hard-fence
                // release path shared with explicit waits and teardown.
                Ok(true)
            }
            0 => Ok(false),
            _ => Err("Unknown ACL recorded-event status".into()),
        }
    }
    fn wait(&self) -> Result<()> {
        if !self.complete.get() {
            let _current = self.context.enter()?;
            // SAFETY: Recorded event remains live; timeout is validated at creation.
            let result = check(
                unsafe {
                    (self.context.api.sync_event)(self.event, self.context.limits.timeout_ms)
                },
                "wait for ACL completion",
            );
            if result.is_err() {
                self.context.poisoned.set(true);
            }
            result?;
            self.complete.set(true);
        }
        // Release retained resources only after the event really completed;
        // a failed fence leaves them quarantined through Drop instead.
        self.retained.borrow_mut().take();
        Ok(())
    }
}
impl Drop for Completion {
    fn drop(&mut self) {
        if self.wait().is_err() {
            // A failed fence cannot prove DMA stopped. Quarantine its context,
            // event and backing until process exit instead of freeing live data.
            std::mem::forget(self.retained.get_mut().take());
            std::mem::forget(Rc::clone(&self.context));
            return;
        }
        // Drop runs only after all Ticket clones, pending references and queued
        // compute waiters release this generation. Keep its hard-fenced event
        // for a later record; capture events have separate ownership and flags.
        const MAX_IDLE_COMPLETIONS: usize = 256;
        if !self.context.poisoned.get() {
            let mut idle = self.context.completion_events.borrow_mut();
            if idle.len() < MAX_IDLE_COMPLETIONS {
                idle.push(self.event);
                return;
            }
        }
        if let Ok(_current) = self.context.enter() {
            // SAFETY: Completed event has no outstanding storage dependencies.
            unsafe {
                (self.context.api.destroy_event)(self.event);
            }
        }
    }
}

#[derive(Clone)]
pub struct Ticket {
    completion: Rc<Completion>,
}
impl Ticket {
    pub fn wait(&self) -> Result<()> {
        self.completion.wait()
    }
    pub fn is_ready(&self) -> Result<bool> {
        self.completion.done()
    }
}

impl Session {
    pub fn new(device: u32, limits: Limits) -> Result<Self> {
        Self::with_driver(acquire()?, device, limits)
    }
    pub(super) fn with_driver(
        driver: Arc<DriverLease>,
        device: u32,
        limits: Limits,
    ) -> Result<Self> {
        if limits.device_bytes == 0 || limits.pinned_bytes == 0 || limits.timeout_ms <= 0 {
            return Err("Ascend memory budgets and timeout must be positive".into());
        }
        let mut count = 0;
        // SAFETY: ACL initialized and count is a live writable scalar.
        check(unsafe { (driver.count)(&mut count) }, "ACL device count")?;
        if device >= count {
            return Err(format!(
                "Ascend device {device} outside device count {count}"
            ));
        }
        let ordinal = i32::try_from(device).map_err(|_| "Ascend device ordinal overflow")?;
        let api = Api::load(&driver)?;
        let mut context = Context {
            driver,
            api,
            handle: ptr::null_mut(),
            compute: ptr::null_mut(),
            transfer: ptr::null_mut(),
            owner: DeviceOwner::new(Backend::Ascend, device),
            limits,
            device_bytes: Cell::new(0),
            pinned_bytes: Cell::new(0),
            poisoned: Cell::new(false),
            completion_events: RefCell::new(vec![]),
        };
        let previous = api.current()?;
        // SAFETY: Valid device index and live output slots. Only successful
        // creation transfers ownership; failed outputs may already be rolled back.
        // Context rolls back earlier successful resources on every later error.
        unsafe {
            let result = (|| -> Result<()> {
                let mut handle = ptr::null_mut();
                check(
                    (api.create_context)(&mut handle, ordinal),
                    "create ACL context",
                )?;
                context.handle = handle;
                let mut stream = ptr::null_mut();
                check(
                    (api.create_stream)(&mut stream),
                    "create ACL compute stream",
                )?;
                context.compute = stream;
                let mut stream = ptr::null_mut();
                check(
                    (api.create_stream)(&mut stream),
                    "create ACL transfer stream",
                )?;
                context.transfer = stream;
                Ok(())
            })();
            if !previous.is_null() {
                (api.set_context)(previous);
            }
            result?;
        }
        Ok(Self {
            context: Rc::new(context),
            pending: RefCell::new(vec![]),
            pinned_cache: RefCell::new(vec![]),
        })
    }
    pub fn memory_info(&self) -> Result<(usize, usize)> {
        let _current = self.context.enter()?;
        let (mut free, mut total) = (0, 0);
        check(
            // SAFETY: Owned current context, live scalar outputs; ACL_HBM_MEM=1.
            unsafe { (self.context.api.memory_info)(1, &mut free, &mut total) },
            "ACL HBM memory info",
        )?;
        Ok((free, total))
    }
    pub(super) fn compute_stream(&self) -> Handle {
        self.context.compute
    }
    pub(super) fn same_context(&self, context: &Rc<Context>) -> Result<()> {
        self.context.ready()?;
        if !Rc::ptr_eq(&self.context, context) {
            return Err("Ascend operation belongs to another session".into());
        }
        Ok(())
    }
    pub fn allocated_bytes(&self) -> usize {
        self.context.device_bytes.get()
    }
    /// Physical pinned storage, including idle reusable buffers.
    pub fn pinned_bytes(&self) -> usize {
        self.context.pinned_bytes.get()
    }
    /// Release cached idle capacity. In-flight backing stays completion-owned.
    pub fn trim_pinned_cache(&self) {
        self.pinned_cache
            .borrow_mut()
            .retain(|host| Rc::strong_count(host) != 1);
    }
    pub(super) fn require(&self, buffer: &Buffer) -> Result<()> {
        self.context.ready()?;
        self.context.owner.require(buffer.allocation.context.owner)
    }
    pub fn allocate(&self, bytes: usize) -> Result<Buffer> {
        self.context.ready()?;
        self.reap()?;
        if bytes == 0
            || bytes
                > self
                    .context
                    .limits
                    .device_bytes
                    .saturating_sub(self.allocated_bytes())
        {
            return Err(format!(
                "Ascend allocation {bytes} bytes exceeds remaining session budget {} bytes or is empty",
                self.context
                    .limits
                    .device_bytes
                    .saturating_sub(self.allocated_bytes())
            ));
        }
        let _current = self.context.enter()?;
        let mut pointer = ptr::null_mut();
        check(
            // SAFETY: Valid byte size and output slot. HUGE_FIRST=0 uses the
            // device's large-page pool with normal-page fallback; NORMAL_ONLY
            // exhausts the smaller normal-page pool before HBM is full.
            unsafe { (self.context.api.malloc)(&mut pointer, bytes, 0) },
            &format!(
                "allocate ACL device memory ({bytes} bytes, {} already allocated)",
                self.allocated_bytes()
            ),
        )?;
        if pointer.is_null() {
            return Err("ACL device allocation returned null".into());
        }
        self.context
            .device_bytes
            .set(self.allocated_bytes() + bytes);
        Ok(Buffer {
            offset: 0,
            bytes,
            allocation: Rc::new(Allocation {
                context: Rc::clone(&self.context),
                pointer,
                bytes,
            }),
        })
    }
    fn pinned(&self, bytes: usize) -> Result<Rc<Pinned>> {
        if bytes == 0 || bytes > self.context.limits.pinned_bytes {
            return Err("Ascend pinned allocation exceeds session budget or is empty".into());
        }
        let mut cache = self.pinned_cache.borrow_mut();
        // Completion::wait releases the retained host reference only after a
        // hard DMA fence. A cache-only reference therefore has no DMA reader
        // or writer, even if its completed public Ticket still exists.
        if let Some(host) = cache
            .iter()
            .filter(|host| Rc::strong_count(host) == 1 && host.bytes >= bytes)
            .min_by_key(|host| host.bytes)
        {
            return Ok(Rc::clone(host));
        }
        while bytes
            > self
                .context
                .limits
                .pinned_bytes
                .saturating_sub(self.pinned_bytes())
        {
            let index = cache
                .iter()
                .enumerate()
                .filter(|(_, host)| Rc::strong_count(host) == 1)
                .max_by_key(|(_, host)| host.bytes)
                .map(|(index, _)| index)
                .ok_or("Ascend pinned allocation exceeds session budget or is empty")?;
            cache.swap_remove(index);
        }
        // Bound host bookkeeping as well as physical bytes. In-flight entries
        // are never evicted; excess transient allocations retain the original
        // completion-owned lifetime rather than growing the cache.
        const MAX_CACHED: usize = 64;
        if cache.len() == MAX_CACHED
            && let Some(index) = cache.iter().position(|host| Rc::strong_count(host) == 1)
        {
            cache.swap_remove(index);
        }
        let mut pointer = ptr::null_mut();
        check(
            // SAFETY: Live ACL runtime, positive bounded byte size and writable output.
            unsafe { (self.context.api.malloc_host)(&mut pointer, bytes) },
            "allocate ACL pinned memory",
        )?;
        if pointer.is_null() {
            return Err("ACL pinned allocation returned null".into());
        }
        self.context.pinned_bytes.set(self.pinned_bytes() + bytes);
        let host = Rc::new(Pinned {
            context: Rc::clone(&self.context),
            pointer,
            bytes,
        });
        if cache.len() < MAX_CACHED {
            cache.push(Rc::clone(&host));
        }
        Ok(host)
    }
    pub fn reap(&self) -> Result<()> {
        let mut pending = self.pending.borrow_mut();
        let mut index = 0;
        while index < pending.len() {
            if pending[index].done()? {
                // A ready query does not replace the hard completion fence
                // required before releasing retained DMA backing.
                pending[index].wait()?;
                pending.swap_remove(index);
            } else {
                index += 1;
            }
        }
        Ok(())
    }
    pub fn synchronize(&self) -> Result<()> {
        self.context.sync(self.context.transfer)?;
        self.context.sync(self.context.compute)?;
        // Both owned streams completed. Every pending fence was recorded on
        // one of them; mark all fences before releasing nested ticket owners.
        // Surviving public Tickets observe the same completed state.
        for pending in self.pending.borrow().iter() {
            pending.complete.set(true);
        }
        for pending in self.pending.borrow().iter() {
            pending.wait()?;
        }
        self.pending.borrow_mut().clear();
        Ok(())
    }
    pub fn wait_on_compute(&self, ticket: &Ticket) -> Result<Ticket> {
        self.context.ready()?;
        self.context
            .owner
            .require(ticket.completion.context.owner)?;
        // SAFETY: Both handles belong to this context. The compute completion
        // retains the source Ticket until the queued wait has itself completed.
        unsafe {
            self.submit_compute(vec![], Box::new(ticket.clone()), |stream| {
                check(
                    (self.context.api.wait_event)(
                        stream,
                        ticket.completion.event,
                        self.context.limits.timeout_ms,
                    ),
                    "ACL compute dependency",
                )
            })
        }
    }
    fn submit(
        &self,
        stream: Handle,
        retained: Retained,
        enqueue: impl FnOnce(Handle) -> Result<()>,
    ) -> Result<Ticket> {
        self.context.ready()?;
        let _current = self.context.enter()?;
        let cached = self.context.completion_events.borrow_mut().pop();
        let mut event = cached.unwrap_or(ptr::null_mut());
        if event.is_null() {
            check(
                // SAFETY: ACL_EVENT_SYNC=1 supports host synchronization; output is live.
                unsafe { (self.context.api.create_event)(&mut event, 1) },
                "create ACL completion",
            )?;
        }
        let result = enqueue(stream).and_then(|()| {
            check(
                // SAFETY: Event and stream belong to this context, and retained
                // owns all backing through the recorded fence's completion.
                unsafe { (self.context.api.record_event)(event, stream) },
                "record ACL completion",
            )
        });
        if let Err(error) = result {
            if self.context.sync(stream).is_err() || self.context.poisoned.get() {
                std::mem::forget(retained);
                std::mem::forget(Rc::clone(&self.context));
            }
            // SAFETY: Event was created here. Its backing was fenced/quarantined.
            unsafe {
                (self.context.api.destroy_event)(event);
            }
            return Err(error);
        }
        let completion = Rc::new(Completion {
            context: Rc::clone(&self.context),
            event,
            complete: Cell::new(false),
            retained: RefCell::new(Some(retained)),
        });
        self.pending.borrow_mut().push(Rc::clone(&completion));
        Ok(Ticket { completion })
    }
    pub fn upload(&self, buffer: &Buffer, offset: usize, data: &[u8]) -> Result<Ticket> {
        self.require(buffer)?;
        let destination = buffer.range(offset, data.len())?;
        self.reap()?;
        let host = self.pinned(data.len())?;
        // SAFETY: The pinned lease is exclusive and spans data.len().
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), host.pointer.cast(), data.len());
        }
        self.submit(
            self.context.transfer,
            Retained {
                _buffers: vec![buffer.clone()],
                _host: Some(Rc::clone(&host)),
                _extra: None,
            },
            |stream| {
                // SAFETY: Validated device extent and retained pinned source; H2D=1.
                check(
                    unsafe {
                        (self.context.api.memcpy)(
                            destination,
                            data.len(),
                            host.pointer,
                            data.len(),
                            1,
                            stream,
                        )
                    },
                    "ACL upload",
                )
            },
        )
    }
    fn input_pinned(&self, bytes: usize) -> Result<Rc<Pinned>> {
        loop {
            self.reap()?;
            match self.pinned(bytes) {
                Ok(host) => return Ok(host),
                Err(error)
                    if error == "Ascend pinned allocation exceeds session budget or is empty" =>
                {
                    // Only wait for already-recorded DMA backing. The enclosing
                    // packet completion is not recorded until submission ends.
                    let oldest = self
                        .pending
                        .borrow()
                        .iter()
                        .find(|c| {
                            c.retained
                                .borrow()
                                .as_ref()
                                .is_some_and(|r| r._host.is_some())
                        })
                        .cloned();
                    oldest.ok_or(error)?.wait()?;
                }
                Err(error) => return Err(error),
            }
        }
    }
    /// Copy complete inputs into bounded pinned slabs. Each slab's completion
    /// owns its DMA backing; pressure waits for the oldest submitted upload.
    /// Compute-stream order protects earlier readers of reused device inputs.
    pub(super) fn upload_inputs(&self, inputs: &[(Buffer, Vec<u8>)]) -> Result<Ticket> {
        let mut bytes = 0usize;
        for (buffer, data) in inputs {
            self.require(buffer)?;
            if data.len() != buffer.bytes() {
                return Err("Model input upload byte extent differs".into());
            }
            bytes = bytes
                .checked_add(data.len())
                .ok_or("Input upload overflow")?;
        }
        if bytes == 0 {
            return Err("Model input upload is empty".into());
        }
        let capacity = (4 << 20).min(self.context.limits.pinned_bytes);
        // Weight loading can leave idle 8 MiB leases. Do not retain that
        // physical capacity for each small input slab queued behind compute.
        self.pinned_cache
            .borrow_mut()
            .retain(|host| host.bytes <= capacity || Rc::strong_count(host) != 1);
        let (mut index, mut offset) = (0, 0);
        let mut last = None;
        while bytes > 0 {
            let size = bytes.min(capacity);
            let host = self.input_pinned(size)?;
            let mut pieces = vec![];
            let mut written = 0;
            while written < size {
                let (buffer, data) = &inputs[index];
                let count = (data.len() - offset).min(size - written);
                // SAFETY: The exclusive slab and source span were checked;
                // every caller-owned byte is copied before returning.
                unsafe {
                    ptr::copy_nonoverlapping(
                        data.as_ptr().add(offset),
                        host.pointer.cast::<u8>().add(written),
                        count,
                    );
                }
                pieces.push((buffer.clone(), offset, written, count));
                written += count;
                offset += count;
                if offset == data.len() {
                    index += 1;
                    offset = 0;
                }
            }
            last = Some(self.submit(
                self.context.compute,
                Retained {
                    _buffers: pieces.iter().map(|(b, ..)| b.clone()).collect(),
                    _host: Some(Rc::clone(&host)),
                    _extra: None,
                },
                |stream| {
                    for (buffer, offset, start, count) in &pieces {
                        let destination = buffer.range(*offset, *count)?;
                        check(
                            // SAFETY: Checked destination and slab extents,
                            // retained source and destination; H2D=1.
                            unsafe {
                                (self.context.api.memcpy)(
                                    destination,
                                    *count,
                                    host.pointer.cast::<u8>().add(*start).cast(),
                                    *count,
                                    1,
                                    stream,
                                )
                            },
                            "ACL input upload",
                        )?;
                    }
                    Ok(())
                },
            )?);
            bytes -= size;
        }
        last.ok_or_else(|| "Model input upload is empty".into())
    }
    pub fn download(&self, buffer: &Buffer, offset: usize, bytes: usize) -> Result<Vec<u8>> {
        self.require(buffer)?;
        let source = buffer.range(offset, bytes)?;
        // Download observes all producers on both owned streams before copying.
        self.synchronize()?;
        let host = self.pinned(bytes)?;
        let ticket = self.submit(
            self.context.transfer,
            Retained {
                _buffers: vec![buffer.clone()],
                _host: Some(Rc::clone(&host)),
                _extra: None,
            },
            |stream| {
                // SAFETY: Retained device source and pinned destination span bytes; D2H=2.
                check(
                    unsafe {
                        (self.context.api.memcpy)(host.pointer, bytes, source, bytes, 2, stream)
                    },
                    "ACL download",
                )
            },
        )?;
        ticket.wait()?;
        let mut out = vec![0; bytes];
        // SAFETY: DMA completed, pinned source and host destination span bytes.
        unsafe {
            ptr::copy_nonoverlapping(host.pointer.cast(), out.as_mut_ptr(), bytes);
        }
        self.reap()?;
        Ok(out)
    }
    pub fn fill(&self, buffer: &Buffer, value: u8) -> Result<Ticket> {
        self.require(buffer)?;
        self.submit(
            self.context.compute,
            Retained {
                _buffers: vec![buffer.clone()],
                _host: None,
                _extra: None,
            },
            |stream| {
                // SAFETY: Entire destination is retained until completion.
                check(
                    unsafe {
                        (self.context.api.memset)(
                            buffer.pointer(),
                            buffer.bytes(),
                            value.into(),
                            buffer.bytes(),
                            stream,
                        )
                    },
                    "ACL fill",
                )
            },
        )
    }
    /// Copy on the compute stream; cross-stream producers require a dependency.
    pub fn copy(
        &self,
        source: &Buffer,
        source_offset: usize,
        destination: &Buffer,
        destination_offset: usize,
        bytes: usize,
    ) -> Result<Ticket> {
        self.copy_many(&[(
            source.clone(),
            source_offset,
            destination.clone(),
            destination_offset,
            bytes,
        )])
    }
    /// Validate every span before submission and retain the whole ordered copy
    /// batch through one completion. Cross-stream producers need a dependency.
    pub(super) fn copy_many(
        &self,
        copies: &[(Buffer, usize, Buffer, usize, usize)],
    ) -> Result<Ticket> {
        if copies.is_empty() {
            return Err("ACL device copy batch is empty".into());
        }
        let mut spans = Vec::with_capacity(copies.len());
        let mut buffers = Vec::with_capacity(copies.len() * 2);
        for (source, source_offset, destination, destination_offset, bytes) in copies {
            self.require(source)?;
            self.require(destination)?;
            let src = source.range(*source_offset, *bytes)?;
            let dst = destination.range(*destination_offset, *bytes)?;
            if *bytes == 0
                || (Rc::ptr_eq(&source.allocation, &destination.allocation)
                    && source.offset + source_offset
                        < destination.offset + destination_offset + bytes
                    && destination.offset + destination_offset
                        < source.offset + source_offset + bytes)
            {
                return Err("ACL device copy requires nonempty, nonoverlapping extents".into());
            }
            spans.push((src, dst, *bytes));
            buffers.extend([source.clone(), destination.clone()]);
        }
        self.submit(
            self.context.compute,
            Retained {
                _buffers: buffers,
                _host: None,
                _extra: None,
            },
            |stream| {
                for &(src, dst, bytes) in &spans {
                    check(
                        // SAFETY: All checked spans are retained through the batch fence; D2D=3.
                        unsafe { (self.context.api.memcpy)(dst, bytes, src, bytes, 3, stream) },
                        "ACL device copy",
                    )?;
                }
                Ok(())
            },
        )
    }
    /// Submit a backend-owned operation. The caller proves all pointers passed
    /// to enqueue are covered by buffers and extra, which are retained to a fence.
    pub(super) unsafe fn submit_compute(
        &self,
        buffers: Vec<Buffer>,
        extra: Box<dyn Any>,
        enqueue: impl FnOnce(Handle) -> Result<()>,
    ) -> Result<Ticket> {
        for buffer in &buffers {
            self.require(buffer)?;
        }
        self.submit(
            self.context.compute,
            Retained {
                _buffers: buffers,
                _host: None,
                _extra: Some(extra),
            },
            enqueue,
        )
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        // A failed completion quarantines its resources through Completion::drop.
        let _ = self.synchronize();
    }
}
