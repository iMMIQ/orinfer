mod args;
mod client;
mod diag;
mod serve;

use args::{Cli, Command};
use std::process::ExitCode;

fn print_json(value: impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn exit_with_unfinished_workers(pending: bool) {
    if pending {
        // SAFETY: Native teardown is still live or resources were quarantined.
        // Running CANN process-exit hooks concurrently can double-free native
        // state. _exit terminates this standalone process without those hooks;
        // the OS reclaims its threads, accelerator handles and GPU lease.
        unsafe { libc::_exit(1) };
    }
}
fn main() -> ExitCode {
    let cli = Cli::parse_command();
    if matches!(&cli.command, Some(Command::Service(_)))
        && std::env::var_os("ASCEND_COREDUMP_SIGNAL").is_none()
    {
        // SAFETY: main has not started runtime, preprocessing or accelerator
        // threads. CANN trace handlers otherwise preempt graceful shutdown.
        unsafe { std::env::set_var("ASCEND_COREDUMP_SIGNAL", "none") };
    }
    let result = match cli
        .command
        .unwrap_or(Command::Diagnostics(diag::Command::Info))
    {
        Command::Client(command) => client::run(command, &cli.data_dir),
        Command::Diagnostics(command) => diag::run(command),
        Command::Service(command) => serve::run(command, &cli.data_dir),
    };
    match result {
        Ok(()) => {
            exit_with_unfinished_workers(
                orinfer_engine::backend::ascend::runtime::has_quarantined_resources(),
            );
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            exit_with_unfinished_workers(
                orinfer_api::has_live_model_workers()
                    || orinfer_engine::backend::ascend::runtime::has_quarantined_resources(),
            );
            ExitCode::FAILURE
        }
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn unfinished_native_worker_exit_bypasses_process_destructors() {
        const CHILD: &str = "ORINFER_TEST_UNFINISHED_NATIVE_WORKER";
        if std::env::var_os(CHILD).is_some() {
            extern "C" fn conflicting_destructor() {
                // SAFETY: Test-only process exit callback; no resources escape.
                unsafe { libc::_exit(77) };
            }
            // SAFETY: Static C callback remains valid through process exit.
            assert_eq!(unsafe { libc::atexit(conflicting_destructor) }, 0);
            super::exit_with_unfinished_workers(true);
            panic!("Emergency exit returned");
        }
        super::exit_with_unfinished_workers(false);
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::unfinished_native_worker_exit_bypasses_process_destructors",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(1));
    }
}
