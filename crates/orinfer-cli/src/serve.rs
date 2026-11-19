use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use orinfer_api::{
    BatchConfig, PreprocessingLimits, ServerConfig,
    manager::{self, client},
};
use orinfer_engine::{
    execution::{LoadOptions, parse_cache_mib, parse_mtp_drafts},
    scheduler::Policy,
};
use serde_json::json;

use crate::args::{LoadArgs, parse_keep_alive, parse_preprocess_workers, parse_request_defaults};
use crate::print_json;

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Start the model service; an optional directory serves one model directly.
    Serve(Box<ServeArgs>),
    #[command(hide = true)]
    Worker(Box<ServeArgs>),
    /// Process Chat requests from JSONL without an HTTP server.
    Batch(Box<BatchArgs>),
}

#[derive(Clone)]
struct ModelLimit(Option<usize>);

fn parse_max_loaded_models(value: &str) -> Result<ModelLimit, std::num::ParseIntError> {
    if value == "auto" {
        return Ok(ModelLimit(None));
    }
    value.parse().map(|count| ModelLimit(Some(count)))
}

#[derive(Clone, Debug)]
struct MtpDrafts(Option<usize>);

impl std::str::FromStr for MtpDrafts {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        parse_mtp_drafts(value).map(Self)
    }
}

#[derive(Args, Clone)]
struct InferenceArgs {
    #[arg(long, default_value_t = ServerConfig::default().model)]
    model: String,
    /// latency (serve default) or throughput (batch default).
    #[arg(long, value_name = "POLICY")]
    scheduler_policy: Option<orinfer_engine::scheduler::Policy>,
    /// JSON object of Chat request defaults; explicit request fields take priority.
    #[arg(long, value_name = "JSON", value_parser = parse_request_defaults)]
    default_request_params: Option<serde_json::Map<String, serde_json::Value>>,
    #[arg(long, default_value_os_t = ServerConfig::default().gpu_lock)]
    gpu_lock: PathBuf,
    #[command(flatten)]
    load: LoadArgs,
    /// auto adapts up to the package default; 0 disables MTP; 1..31 caps drafts within package capacity.
    #[arg(long, value_name = "N", default_value = "auto")]
    mtp_drafts: MtpDrafts,
    /// Device prefix-cache budget per device in MiB; 0 disables the device tier.
    #[arg(long = "prefix-cache-mib", value_name = "MIB", default_value_t = ServerConfig::default().prefix_cache_bytes >> 20, value_parser = parse_cache_mib)]
    prefix_cache_bytes: usize,
    /// Global CPU prefix-state cache in MiB, shared across TP ranks; 0 disables the CPU tier.
    #[arg(long = "kv-offload-mib", value_name = "MIB", default_value_t = 0, value_parser = parse_cache_mib)]
    kv_offload_bytes: usize,
    #[arg(long, default_value_t = ServerConfig::default().scheduler.max_active)]
    max_active_requests: usize,
    /// Ordinary decode row budget; mixed prefill may use the package capacity.
    #[arg(long, default_value_t = ServerConfig::default().scheduler.max_batch_tokens)]
    max_batch_tokens: usize,
    /// Total MTP verification row budget; 0 uses the package's legal capacity.
    #[arg(long, default_value_t = 0)]
    max_speculative_tokens: usize,
    /// Predicted mixed prefill budget in milliseconds (latency policy).
    #[arg(long, default_value_t = ServerConfig::default().scheduler.prefill_budget_ms)]
    prefill_budget_ms: f64,
    /// Soft decode token interval target in milliseconds (latency policy).
    #[arg(long, default_value_t = ServerConfig::default().scheduler.target_tpot_ms)]
    target_tpot_ms: f64,
    #[arg(long = "memory-reserve-mib", value_name = "MIB", default_value_t = ServerConfig::default().scheduler.memory_reserve_bytes >> 20, value_parser = parse_cache_mib)]
    memory_reserve_bytes: usize,
    /// auto selects 1..8 preprocessing tasks from CPU and memory budgets; explicit 1..32 overrides it.
    #[arg(long, default_value = "auto", value_parser = parse_preprocess_workers)]
    preprocess_workers: usize,
    #[arg(long, default_value_t = PreprocessingLimits::default().memory_mib)]
    preprocess_memory_mib: usize,
    /// Retained image payload/preprocessing cache in MiB; 0 disables reuse and UUID references.
    #[arg(long, default_value_t = PreprocessingLimits::default().image_cache_mib)]
    image_cache_mib: usize,
    /// Maximum HTTP request body in MiB, bounded by the preprocessing memory budget.
    #[arg(long, default_value_t = PreprocessingLimits::default().request_body_mib)]
    request_body_mib: usize,
}
#[derive(Args)]
pub(crate) struct ServeArgs {
    /// Prepared model directory. Omit for managed serving.
    model_dir: Option<PathBuf>,
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long, default_value_t = ServerConfig::default().listen)]
    listen: String,
    /// Queue deadline in milliseconds; 0 means unlimited.
    #[arg(long, default_value_t = PreprocessingLimits::default().queue_ms)]
    queue_timeout_ms: u64,
    #[arg(long, default_value_t = PreprocessingLimits::default().output_ms)]
    output_timeout_ms: u64,
    #[arg(long, default_value_t = PreprocessingLimits::default().drain_ms)]
    drain_timeout_ms: u64,
    /// Idle model lifetime; accepts seconds, 5m, 1h or forever.
    #[arg(long, default_value = "5m", value_parser = parse_keep_alive)]
    keep_alive: u64,
    /// Loaded-model limit; auto admits models by unified-memory availability.
    #[arg(long, default_value = "auto", value_parser = parse_max_loaded_models)]
    max_loaded_models: ModelLimit,
    #[arg(long, default_value_t = 120)]
    load_timeout_s: u64,
}
#[derive(Args)]
pub(crate) struct BatchArgs {
    #[arg(skip)]
    overrides: serde_json::Map<String, serde_json::Value>,
    /// Registered model name or prepared model directory.
    model_dir: PathBuf,
    #[command(flatten)]
    inference: InferenceArgs,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    output: PathBuf,
    /// Skip committed responses and retry failed rows in the same job.
    #[arg(long)]
    resume: bool,
    /// auto plans file-wide prefix/length order; input preserves file traversal.
    #[arg(long, default_value_t = orinfer_api::BatchOrder::Auto)]
    batch_order: orinfer_api::BatchOrder,
}
impl From<InferenceArgs> for ServerConfig {
    fn from(args: InferenceArgs) -> Self {
        let load = args.load.options();
        Self {
            model: args.model,
            default_request_params: args.default_request_params.unwrap_or_default(),
            gpu_lock: args.gpu_lock,
            cuda_graph: load.cuda_graph,
            verify_weights: load.verify_weights,
            load_workers: load.load_workers,
            resident_experts: load.resident_experts,
            mtp_drafts: args.mtp_drafts.0,
            prompt_lookup: load.prompt_lookup,
            prefix_cache_bytes: args.prefix_cache_bytes,
            kv_offload_bytes: args.kv_offload_bytes,
            scheduler: orinfer_engine::scheduler::Options {
                policy: args.scheduler_policy.unwrap_or_default(),
                max_active: args.max_active_requests,
                max_batch_tokens: args.max_batch_tokens,
                max_speculative_tokens: args.max_speculative_tokens,
                prefill_budget_ms: args.prefill_budget_ms,
                target_tpot_ms: args.target_tpot_ms,
                memory_reserve_bytes: args.memory_reserve_bytes,
            },
            limits: PreprocessingLimits {
                preprocess_workers: args.preprocess_workers,
                memory_mib: args.preprocess_memory_mib,
                image_cache_mib: args.image_cache_mib,
                request_body_mib: args.request_body_mib,
                ..Default::default()
            },
            ..Default::default()
        }
    }
}
impl From<ServeArgs> for ServerConfig {
    fn from(args: ServeArgs) -> Self {
        let mut config: Self = args.inference.into();
        config.model_dir = args.model_dir.unwrap_or_default();
        config.listen = args.listen;
        config.limits.queue_ms = args.queue_timeout_ms;
        config.limits.output_ms = args.output_timeout_ms;
        config.limits.drain_ms = args.drain_timeout_ms;
        config
    }
}
impl From<BatchArgs> for BatchConfig {
    fn from(args: BatchArgs) -> Self {
        let policy = args
            .inference
            .scheduler_policy
            .unwrap_or(orinfer_engine::scheduler::Policy::Throughput);
        let mut config: ServerConfig = args.inference.into();
        config.scheduler.policy = policy;
        Self {
            input: args.input,
            output: args.output,
            resume: args.resume,
            order: args.batch_order,
            model_dir: args.model_dir,
            model: config.model,
            gpu_lock: config.gpu_lock,
            scheduler: config.scheduler,
            limits: config.limits,
            default_request_params: config.default_request_params,
            load: LoadOptions {
                cuda_graph: config.cuda_graph,
                verify_weights: config.verify_weights,
                load_workers: config.load_workers,
                resident_experts: config.resident_experts,
                mtp_drafts: config.mtp_drafts,
                prompt_lookup: config.prompt_lookup,
                prefix_cache_bytes: config.prefix_cache_bytes,
                kv_offload_bytes: config.kv_offload_bytes,
            },
        }
    }
}
impl BatchArgs {
    pub fn set_overrides(&mut self, matches: &clap::ArgMatches) {
        let settings: ServerConfig = self.inference.clone().into();
        let serde_json::Value::Object(mut overrides) =
            serde_json::to_value(manager::WorkerOptions::from_settings(&settings)).unwrap()
        else {
            unreachable!("Worker options serialize as a JSON object");
        };
        overrides.retain(|key, _| {
            let id = match key.as_str() {
                "prefix_cache_mib" => "prefix_cache_bytes",
                "kv_offload_mib" => "kv_offload_bytes",
                "memory_reserve_mib" => "memory_reserve_bytes",
                id => id,
            };
            matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine)
        });
        self.overrides = overrides;
    }
}

pub(crate) fn run(command: Command, data_dir: &Path) -> Result<(), String> {
    match command {
        Command::Serve(args) if args.model_dir.is_none() => {
            let settings: ServerConfig = args.inference.into();
            manager::run(manager::ManagerConfig {
                worker_options: manager::WorkerOptions::from_settings(&settings),
                data_dir: data_dir.into(),
                listen: args.listen,
                keep_alive_s: args.keep_alive,
                max_loaded_models: args.max_loaded_models.0,
                load_timeout_s: args.load_timeout_s,
                gpu_lock: settings.gpu_lock,
                memory_reserve_mib: settings.scheduler.memory_reserve_bytes >> 20,
            })
        }

        Command::Serve(args) | Command::Worker(args) => orinfer_api::run((*args).into()),
        Command::Batch(args) => run_batch(*args, data_dir),
    }
}

fn run_batch(args: BatchArgs, data_dir: &Path) -> Result<(), String> {
    let failed = if !args.model_dir.is_dir() {
        let result = client::request(
            data_dir,
            "POST",
            "/batch",
            Some(
                &json!({"name":args.model_dir.to_string_lossy(),"options":args.overrides,
                    "scheduler_policy":args.inference.scheduler_policy.unwrap_or(Policy::Throughput).to_string(),
                    "input":std::path::absolute(&args.input).map_err(|e| e.to_string())?,
                    "output":std::path::absolute(&args.output).map_err(|e| e.to_string())?,
                    "resume":args.resume,"order":args.batch_order.to_string()}),
            ),
        )?;
        let failed = result["success"] != true;
        print_json(result["report"].clone())?;
        failed
    } else {
        let report = orinfer_api::run_batch(args.into())?;
        let failed = report.failed != 0 || report.truncated != 0 || report.interrupted;
        print_json(report)?;
        failed
    };
    if failed {
        return Err(
            "Batch has errors, truncated responses or was interrupted; inspect the output JSONL"
                .into(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{Cli, Command as TopCommand};
    use clap::{CommandFactory, FromArgMatches, Parser};
    use orinfer_engine::execution::CudaGraphMode;
    use orinfer_engine::execution::ResidentExperts;

    #[test]
    fn managed_batch_only_overrides_explicit_model_options() {
        let matches = Cli::command()
            .try_get_matches_from([
                "orinfer",
                "batch",
                "translator",
                "--input",
                "in.jsonl",
                "--output",
                "out.jsonl",
                "--mtp-drafts",
                "0",
                "--cuda-graph",
                "off",
                "--scheduler-policy",
                "latency",
            ])
            .unwrap();
        let cli = Cli::from_arg_matches(&matches).unwrap();
        let Some(TopCommand::Service(Command::Batch(mut args))) = cli.command else {
            panic!("batch command")
        };
        args.set_overrides(matches.subcommand_matches("batch").unwrap());
        let options = args.overrides;
        assert_eq!(options.len(), 2);
        assert_eq!(options["mtp_drafts"], "0");
        assert_eq!(options["cuda_graph"], "off");
        assert_eq!(
            args.inference.scheduler_policy,
            Some(orinfer_engine::scheduler::Policy::Latency)
        );
    }
    fn serve(options: &[&str]) -> Result<ServerConfig, clap::Error> {
        let mut args = vec!["orinfer", "serve", "/tmp"];
        args.extend_from_slice(options);
        let Some(TopCommand::Service(Command::Serve(args))) = Cli::try_parse_from(args)?.command
        else {
            unreachable!()
        };
        Ok((*args).into())
    }
    #[test]
    fn defaults_and_server_options_are_preserved() {
        let config = serve(&[]).unwrap();
        config.validate().unwrap();
        assert_eq!(config.listen, "[::]:8088");
        assert_eq!(config.cuda_graph, CudaGraphMode::DecodeOnly);
        assert_eq!(config.prefix_cache_bytes, 4usize << 30);
        assert_eq!(config.scheduler.max_active, 32);
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Latency
        );
        assert_eq!(config.mtp_drafts, None);
        assert_eq!(config.resident_experts, ResidentExperts::Full);
        assert!(!config.prompt_lookup);
        assert!(!config.verify_weights);
        assert_eq!(config.load_workers, None);
        assert!(config.default_request_params.is_empty());
        let config = serve(&[
            "--verify-weights",
            "--prompt-lookup",
            "--mtp-drafts=7",
            "--resident-experts=256",
            "--cuda-graph",
            "off",
            "--max-active-requests",
            "8",
            "--max-batch-tokens",
            "64",
            "--prefill-budget-ms",
            "150",
            "--target-tpot-ms",
            "350",
            "--memory-reserve-mib",
            "2048",
            "--prefix-cache-mib",
            "0",
        ])
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.cuda_graph, CudaGraphMode::Off);
        assert_eq!(config.mtp_drafts, Some(7));
        assert_eq!(config.resident_experts, ResidentExperts::Count(256));
        assert!(config.prompt_lookup);
        assert!(config.verify_weights);
        assert_eq!(config.prefix_cache_bytes, 0);
        assert_eq!(config.scheduler.memory_reserve_bytes, 2usize << 30);
    }
    #[test]
    fn batch_reuses_inference_options_without_http_settings() {
        let Some(TopCommand::Service(Command::Batch(args))) = Cli::try_parse_from([
            "orinfer",
            "batch",
            "/tmp",
            "--input",
            "requests.jsonl",
            "--output",
            "results.jsonl",
            "--resume",
            "--model",
            "translation",
            "--max-active-requests",
            "64",
            "--mtp-drafts",
            "0",
            "--default-request-params",
            "{\"seed\":20261002,\"temperature\":0}",
        ])
        .unwrap()
        .command
        else {
            panic!("Expected batch command")
        };
        let config: BatchConfig = (*args).into();
        assert_eq!(config.input, PathBuf::from("requests.jsonl"));
        assert_eq!(config.output, PathBuf::from("results.jsonl"));
        assert!(config.resume);
        assert_eq!(config.model, "translation");
        assert_eq!(config.scheduler.max_active, 64);
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Throughput
        );
        assert_eq!(config.load.mtp_drafts, Some(0));
        assert_eq!(config.default_request_params["seed"], 20261002);
        assert!(
            Cli::try_parse_from(["orinfer", "batch", "/tmp", "--input", "requests.jsonl"]).is_err()
        );
    }
    #[test]
    fn scheduler_policy_is_explicit_and_batch_defaults_do_not_change_serve() {
        let config = serve(&["--scheduler-policy", "throughput"]).unwrap();
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Throughput
        );
        assert!(serve(&["--scheduler-policy", "unknown"]).is_err());
        let Some(TopCommand::Service(Command::Batch(args))) = Cli::try_parse_from([
            "orinfer",
            "batch",
            "/tmp",
            "--input",
            "input.jsonl",
            "--output",
            "output.jsonl",
            "--scheduler-policy",
            "latency",
        ])
        .unwrap()
        .command
        else {
            unreachable!()
        };
        let config: BatchConfig = (*args).into();
        assert_eq!(
            config.scheduler.policy,
            orinfer_engine::scheduler::Policy::Latency
        );
    }
    #[test]
    fn syntax_help_and_limits_are_checked_before_loading() {
        assert_eq!(
            serve(&["--load-workers", "12"]).unwrap().load_workers,
            Some(12)
        );
        assert!(serve(&["--load-workers", "0"]).is_err());
        assert!(serve(&["--load-workers", "33"]).is_err());
        for args in [
            vec!["orinfer", "--help"],
            vec!["orinfer", "--version"],
            vec!["orinfer", "serve", "--help"],
            vec!["orinfer", "batch", "--help"],
            vec!["orinfer", "package", "--help"],
            vec!["orinfer", "package", "install", "--help"],
        ] {
            assert_eq!(Cli::try_parse_from(args).err().unwrap().exit_code(), 0);
        }
        for options in [
            &["--cuda-graph"][..],
            &["--cuda-graph", "on"],
            &["--verify-weights=false"],
            &["--mtp-drafts", "32"],
            &["--unknown", "off"],
            &["--prefix-cache-mib", "1.5"],
            &["--listen", "a", "--listen", "b"],
            &["--default-request-params", "[]"],
            &["--default-request-params", "{broken"],
        ] {
            assert!(serve(options).is_err());
        }
        for (name, value) in [
            ("--max-active-requests", "129"),
            ("--max-batch-tokens", "0"),
            ("--target-tpot-ms", "NaN"),
            ("--output-timeout-ms", "0"),
        ] {
            assert!(serve(&[name, value]).unwrap().validate().is_err());
        }
    }
    #[test]
    fn deployment_defaults_are_typed_and_preserved() {
        let config = serve(&[
            "--default-request-params",
            r#"{"enable_thinking":true,"temperature":0.7}"#,
        ])
        .unwrap();
        config.validate().unwrap();
        assert_eq!(config.default_request_params["enable_thinking"], true);
        assert_eq!(config.default_request_params["temperature"], 0.7);
        for defaults in [
            r#"{"enable_thinking":"true"}"#,
            r#"{"unknown":1}"#,
            r#"{"model":"x"}"#,
            r#"{"messages":[]}"#,
        ] {
            assert!(
                serve(&["--default-request-params", defaults])
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
    }
}
