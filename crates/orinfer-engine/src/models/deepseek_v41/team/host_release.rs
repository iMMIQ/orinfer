//! Reclaim immutable CPU tables alongside native rank/context destruction.
use crate::artifact::Result;
use std::{
    sync::{Arc, OnceLock},
    thread::{self, JoinHandle},
    time::Duration,
};

pub(super) fn start<T: Send + Sync + 'static>(
    shared: Arc<OnceLock<Result<Arc<T>>>>,
) -> Option<JoinHandle<()>> {
    match thread::Builder::new()
        .name("orinfer-host-release".into())
        .spawn(move || {
            loop {
                // Loading workers release the outer handle before reporting
                // Ready. Their table readers retain the inner Arc until done.
                let readers_done = Arc::strong_count(&shared) == 1
                    && match shared.get() {
                        Some(Ok(tables)) => Arc::strong_count(tables) == 1,
                        _ => true,
                    };
                if readers_done {
                    break;
                }
                // Shutdown only: no polling is added to inference commands.
                thread::sleep(Duration::from_millis(1));
            }
            drop(shared);
        }) {
        Ok(handle) => Some(handle),
        Err(error) => {
            // Existing worker readers still own their allocations on failure.
            eprintln!("CPU table release thread could not start: {error}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[test]
    fn cpu_release_waits_for_readers_and_overlaps_native_destruction() {
        struct Payload(mpsc::Sender<()>);
        impl Drop for Payload {
            fn drop(&mut self) {
                self.0.send(()).unwrap();
            }
        }
        let (dropped, notification) = mpsc::channel();
        let shared = Arc::new(OnceLock::new());
        let payload = Arc::new(Payload(dropped));
        assert!(shared.set(Ok(Arc::clone(&payload))).is_ok());
        let load_handle = Arc::clone(&shared);
        let cleanup = start(shared).unwrap();
        assert!(notification.try_recv().is_err());
        drop(load_handle);
        assert!(notification.try_recv().is_err());
        let (native_started, native_start) = mpsc::channel();
        let (native_release, native_gate) = mpsc::channel();
        let worker = thread::spawn(move || {
            drop(payload);
            native_started.send(()).unwrap();
            native_gate.recv().unwrap();
        });
        native_start.recv_timeout(Duration::from_secs(5)).unwrap();
        let released = notification.recv_timeout(Duration::from_secs(5));
        native_release.send(()).unwrap();
        worker.join().unwrap();
        cleanup.join().unwrap();
        released.unwrap();
    }

    #[test]
    fn partial_or_failed_initialization_has_no_readers_to_wait_for() {
        start::<()>(Arc::new(OnceLock::new()))
            .unwrap()
            .join()
            .unwrap();
        let shared = Arc::new(OnceLock::new());
        shared.set(Err("load failed".into())).unwrap();
        start::<()>(shared).unwrap().join().unwrap();
    }
}
