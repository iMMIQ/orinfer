use super::{CudaGraphMode, ExecutionPhase, Result, check, sys};
use std::cell::{Cell, RefCell};

pub(crate) struct Profile {
    pub gate_first: Cell<bool>,
    pub priority: RefCell<String>,
    pub events: Cell<Option<[sys::CUevent; 2]>>,
    pub millis: RefCell<Vec<f32>>,
}
impl Default for Profile {
    fn default() -> Self {
        Self {
            gate_first: Cell::new(false),
            priority: RefCell::new("small_side".into()),
            events: Cell::new(None),
            millis: RefCell::new(vec![]),
        }
    }
}
impl Profile {
    fn decode_events(&self, name: &str, phase: ExecutionPhase) -> Option<[sys::CUevent; 2]> {
        if name == "decode" && phase == ExecutionPhase::Decode {
            self.events.get()
        } else {
            None
        }
    }
    pub(super) fn replay(
        &self,
        name: &str,
        phase: ExecutionPhase,
        stream: sys::CUstream,
        graph: sys::CUgraphExec,
    ) -> Option<Result<()>> {
        let [begin, end] = self.decode_events(name, phase)?;
        Some((|| {
            // SAFETY: Graph and events are retained by the session and recorded on its stream.
            unsafe {
                check(sys::cuEventRecord(begin, stream), "live graph begin")?;
                check(sys::cuGraphLaunch(graph, stream), "model graph")?;
                check(sys::cuEventRecord(end, stream), "live graph end")
            }
        })())
    }
    pub(super) fn finish(
        &self,
        name: &str,
        phase: ExecutionPhase,
        mode: CudaGraphMode,
    ) -> Result<()> {
        if mode.uses_graph(phase)
            && let Some([begin, end]) = self.decode_events(name, phase)
        {
            let mut millis = 0.;
            // SAFETY: Both events were recorded on the synchronized stream and are retained by Session.
            unsafe {
                check(
                    sys::cuEventElapsedTime(&mut millis, begin, end),
                    "live graph elapsed",
                )?;
            }
            self.millis.borrow_mut().push(millis);
        }
        Ok(())
    }
}
