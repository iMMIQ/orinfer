//! Stable-address ACL Graph resources with retained native operation owners.
mod branches;
use super::{
    api::Handle,
    check,
    runtime::{Context, Session, Ticket},
};
use crate::artifact::Result;
pub(super) use branches::Branches;
use std::{any::Any, cell::RefCell, ptr, rc::Rc};
struct Api {
    begin: unsafe extern "C" fn(Handle, i32) -> i32,
    end: unsafe extern "C" fn(Handle, *mut Handle) -> i32,
    execute: unsafe extern "C" fn(Handle, Handle) -> i32,
    destroy: unsafe extern "C" fn(Handle) -> i32,
}
struct Captured {
    context: Rc<Context>,
    api: Api,
    handle: Handle,
    active: bool,
    keepalive: Option<Box<dyn Any>>,
}
impl Drop for Captured {
    fn drop(&mut self) {
        let Ok(_current) = self.context.enter() else {
            std::mem::forget(self.keepalive.take());
            std::mem::forget(Rc::clone(&self.context));
            return;
        };
        if self.active {
            // SAFETY: Roll back a capture begun on this owned stream. This also
            // runs during unwinding before the captured operands can be dropped.
            if unsafe { (self.api.end)(self.context.compute_stream(), &mut self.handle) } != 0 {
                std::mem::forget(self.keepalive.take());
                std::mem::forget(Rc::clone(&self.context));
                return;
            }
        }
        if !self.handle.is_null() {
            // SAFETY: Every graph submission retains Captured through its fence.
            unsafe {
                (self.api.destroy)(self.handle);
            }
        }
    }
}
pub struct Graph {
    captured: Rc<Captured>,
    last: RefCell<Option<Ticket>>,
}
impl Graph {
    /// The caller retains this graph through an enclosing compute completion
    /// and serializes invocations on the owning session's compute stream.
    pub(super) unsafe fn execute_on_stream(&self, stream: Handle) -> Result<()> {
        check(
            // SAFETY: The caller supplies this graph's owned stream and retains
            // its stable operands and native resources through completion.
            unsafe { (self.captured.api.execute)(self.captured.handle, stream) },
            "execute ACL Graph segment",
        )
    }
    /// Caller guarantees stable addresses and retains every native resource used
    /// by enqueue in keepalive. Allocation and preparation happen before capture.
    pub(super) unsafe fn capture(
        session: &Session,
        keepalive: Box<dyn Any>,
        enqueue: impl FnOnce(Handle) -> Result<()>,
    ) -> Result<Self> {
        if !session.graphs_enabled() {
            return Err("ACL Graph capture is disabled for this session".into());
        }
        session.same_context(&session.context)?;
        session.synchronize()?;
        let _current = session.context.enter()?;
        let library = &session.context.driver._library;
        // SAFETY: These public CANN signatures are from acl_rt.h. Captured keeps
        // the context's driver/library and every operand alive through destruction.
        let api = unsafe {
            Api {
                begin: *library
                    .get(b"aclmdlRICaptureBegin\0")
                    .map_err(|e| e.to_string())?,
                end: *library
                    .get(b"aclmdlRICaptureEnd\0")
                    .map_err(|e| e.to_string())?,
                execute: *library
                    .get(b"aclmdlRIExecuteAsync\0")
                    .map_err(|e| e.to_string())?,
                destroy: *library
                    .get(b"aclmdlRIDestroy\0")
                    .map_err(|e| e.to_string())?,
            }
        };
        let mut captured = Captured {
            context: Rc::clone(&session.context),
            api,
            handle: ptr::null_mut(),
            active: false,
            keepalive: Some(keepalive),
        };
        check(
            // SAFETY: GLOBAL=0; stream is idle and owned by this context/thread.
            unsafe { (captured.api.begin)(session.compute_stream(), 0) },
            "begin ACL Graph capture",
        )?;
        captured.active = true;
        enqueue(session.compute_stream())?;
        check(
            // SAFETY: Exactly one matching end for the live owned stream capture.
            unsafe { (captured.api.end)(session.compute_stream(), &mut captured.handle) },
            "end ACL Graph capture",
        )?;
        captured.active = false;
        if captured.handle.is_null() {
            return Err("ACL capture returned null model instance".into());
        }
        Ok(Self {
            captured: Rc::new(captured),
            last: RefCell::new(None),
        })
    }
    pub fn enqueue(&self, session: &Session) -> Result<Ticket> {
        session.same_context(&self.captured.context)?;
        if let Some(last) = self.last.borrow().as_ref()
            && !last.is_ready()?
        {
            return Err("ACL Graph already has an in-flight invocation".into());
        }
        let captured = Rc::clone(&self.captured);
        // SAFETY: The recorded graph's stable operands and native resources are
        // retained by Captured, which this completion owns until execution ends.
        let ticket = unsafe {
            session.submit_compute(vec![], Box::new(Rc::clone(&captured)), |stream| {
                check(
                    (captured.api.execute)(captured.handle, stream),
                    "execute ACL Graph",
                )
            })
        }?;
        *self.last.borrow_mut() = Some(ticket.clone());
        Ok(ticket)
    }
}
