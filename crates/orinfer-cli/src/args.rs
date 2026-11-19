use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use orinfer_engine::execution::{CudaGraphMode, LoadOptions, ResidentExperts, parse_load_workers};
use std::path::PathBuf;

use crate::{client, diag, serve};

#[derive(Parser)]
#[command(
    name = "orinfer",
    version,
    about = "Native model inference and accelerator execution packages"
)]
pub(crate) struct Cli {
    #[arg(long, global = true, default_value_os_t = orinfer_api::manager::default_data_dir())]
    pub data_dir: PathBuf,
    #[command(subcommand)]
    pub command: Option<Command>,
}
impl Cli {
    pub fn parse_command() -> Self {
        let matches = Self::command().get_matches();
        let mut cli = Self::from_arg_matches(&matches).unwrap_or_else(|error| error.exit());
        if let Some(Command::Service(serve::Command::Batch(args))) = &mut cli.command {
            args.set_overrides(matches.subcommand_matches("batch").unwrap());
        }
        cli
    }
}

#[derive(Subcommand)]
pub(crate) enum Command {
    #[command(flatten)]
    Client(client::Command),
    #[command(flatten)]
    Diagnostics(diag::Command),
    #[command(flatten)]
    Service(serve::Command),
}

#[derive(Args, Clone)]
pub(crate) struct LoadArgs {
    /// Routed experts per layer equivalent, shared globally; full keeps all. MTP stays resident.
    #[arg(long, value_name = "full|N", default_value_t = ResidentExperts::default())]
    resident_experts: ResidentExperts,
    /// Graph capture: decode_only, full or off. --graph is the backend-neutral alias.
    #[arg(long, visible_alias = "graph", value_name = "MODE", default_value_t = CudaGraphMode::default())]
    cuda_graph: CudaGraphMode,
    /// Hash loaded GPU weights and CPU row tables at startup (slower).
    #[arg(long)]
    verify_weights: bool,
    /// Parallel weight readers; default selects a bounded number of CPU cores.
    #[arg(long, value_name = "N", value_parser = parse_load_workers)]
    load_workers: Option<usize>,
    /// Opt in to target-verified proposals from this request's token history.
    #[arg(long)]
    prompt_lookup: bool,
}
impl LoadArgs {
    pub fn options(self) -> LoadOptions {
        LoadOptions {
            cuda_graph: self.cuda_graph,
            verify_weights: self.verify_weights,
            load_workers: self.load_workers,
            resident_experts: self.resident_experts,
            prompt_lookup: self.prompt_lookup,
            ..Default::default()
        }
    }

    pub fn encoder_options(self) -> LoadOptions {
        LoadOptions {
            cuda_graph: self.cuda_graph,
            verify_weights: self.verify_weights,
            load_workers: self.load_workers,
            ..Default::default()
        }
    }
}

pub(crate) fn parse_preprocess_workers(value: &str) -> Result<usize, String> {
    if value == "auto" || value == "0" {
        return Ok(0);
    }
    value
        .parse::<usize>()
        .ok()
        .filter(|n| (1..=32).contains(n))
        .ok_or_else(|| "Preprocess workers must be auto or 1..32".into())
}

pub(crate) fn parse_request_defaults(
    value: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    serde_json::from_str(value).map_err(|e| format!("Expected a JSON object: {e}"))
}
pub(crate) fn parse_keep_alive(value: &str) -> Result<u64, String> {
    if value == "forever" {
        return Ok(u64::MAX);
    }
    let (number, factor) = match value.as_bytes().last() {
        Some(b's') => (&value[..value.len() - 1], 1),
        Some(b'm') => (&value[..value.len() - 1], 60),
        Some(b'h') => (&value[..value.len() - 1], 3600),
        _ => (value, 1),
    };
    number
        .parse::<u64>()
        .ok()
        .and_then(|n| n.checked_mul(factor))
        .ok_or_else(|| "Expected seconds, 5m, 1h or forever".into())
}
