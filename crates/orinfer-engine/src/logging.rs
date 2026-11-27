//! Shared non-blocking bounded stderr logger for hot-path runtime telemetry.

use std::sync::{OnceLock, mpsc};

/// Queue one diagnostic line for asynchronous stderr output.
/// A stalled stderr pipe drops excess lines rather than stalling device workers.
pub fn log_async(line: String) {
    static LOGGER: OnceLock<mpsc::SyncSender<String>> = OnceLock::new();
    let tx = LOGGER.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel::<String>(8192);
        let _ = std::thread::Builder::new()
            .name("orinfer-log".into())
            .spawn(move || {
                while let Ok(msg) = rx.recv() {
                    eprintln!("{msg}");
                }
            });
        tx
    });
    let _ = tx.try_send(line);
}
