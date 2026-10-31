use clap::{Args, CommandFactory, FromArgMatches, Parser, Subcommand};
use orinfer_api::{BatchConfig, PreprocessingLimits, ServerConfig};
use orinfer_engine::execution::{
    CudaGraphMode, LoadOptions, ResidentExperts, parse_cache_mib, parse_load_workers,
    parse_mtp_drafts,
};
use std::{path::PathBuf, process::ExitCode};

#[derive(Clone, Debug)]
struct MtpDrafts(Option<usize>);
impl std::str::FromStr for MtpDrafts {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, String> {
        parse_mtp_drafts(value).map(Self)
    }
}

#[derive(Parser)]
#[command(
    name = "orinfer",
    version,
    about = "Native model inference and accelerator execution packages"
)]
struct Cli {
    #[arg(long, global = true, default_value_os_t = orinfer_api::manager::default_data_dir())]
    data_dir: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}
#[derive(Subcommand)]
enum Command {
    /// Install a native hf://owner/model or ms://owner/model snapshot.
    Pull {
        source: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        revision: Option<String>,
        /// Matching execution archive URL/path when it is not already installed.
        #[arg(long)]
        package: Option<String>,
        #[arg(long)]
        options: Option<String>,
    },
    /// Register an existing native model without copying its weights.
    Import {
        name: String,
        path: PathBuf,
        #[arg(long)]
        options: Option<String>,
    },
    /// List registered models on disk.
    List,
    /// List loaded model workers.
    Ps,
    /// Show a registered model and its execution profile.
    Show { name: String },
    /// Load a model before the first request.
    Load {
        name: String,
        #[arg(long, value_parser = parse_keep_alive)]
        keep_alive: Option<u64>,
    },
    /// Drain and unload a model, keeping its files.
    Stop {
        name: String,
        #[arg(long)]
        force: bool,
    },
    /// Remove a model registration, keeping externally owned files.
    Rm { name: String },
    /// Chat interactively or translate a prompt/stdin using a registered model.
    Run {
        name: String,
        prompt: Option<String>,
        #[arg(long, default_value = "{}")]
        params: String,
    },
    /// Transcribe an audio file with a registered ASR model.
    Transcribe {
        name: String,
        input: PathBuf,
        #[arg(long)]
        language: Option<String>,
        #[arg(long)]
        prompt: Option<String>,
    },
    /// Show recent worker log output.
    Logs {
        name: String,
        #[arg(long, default_value_t = 100)]
        tail: usize,
    },
    /// Manage precompiled model execution packages without CUDA or Python.
    Package {
        #[command(subcommand)]
        command: PackageCommand,
    },
    /// Show target and implementation status (default command).
    Info,
    /// Inspect accelerator runtime availability without loading a model.
    Devices {
        #[arg(long)]
        backend: orinfer_engine::backend::Backend,
    },
    /// Validate bounded native Ascend transfers and W4A8 arithmetic on one device.
    ValidateAscend {
        #[arg(long, default_value_t = 0)]
        device: u32,
    },
    /// Validate TileLang A2 AOT operators and changed-input native Graph replay.
    ValidateAscendKernels {
        #[arg(long, default_value_t = 7)]
        device: u32,
        #[arg(long)]
        fixtures: PathBuf,
    },
    /// Benchmark complete single-sequence native A2 model execution.
    BenchAscend {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
        #[arg(long, default_value_t = 128)]
        output_tokens: usize,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long)]
        trace: Option<PathBuf>,
    },
    /// Benchmark shared-row native A2 requests with owner-private state.
    BenchAscendBatch {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        inputs: PathBuf,
    },
    /// Validate native HCCL AllReduce and variable-count AlltoAllV on 2..8 devices.
    ValidateHccl {
        #[arg(long, value_delimiter = ',', required = true)]
        devices: Vec<u32>,
    },
    /// Validate conditional-memory metadata and immutable host row staging on CPU.
    ValidateEngram {
        #[arg(long)]
        metadata: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Print the benchmark specification as JSON.
    Plan,
    /// Check AOT fixture and file hashes without CUDA.
    ValidateArtifact { manifest: PathBuf },
    /// Execute an SM87 AOT fixture and validate graph replay.
    RunArtifact { manifest: PathBuf },
    /// Inspect a native execution library identity and capabilities without CUDA.
    InspectLibrary { library: PathBuf },
    /// Inspect the native model package plan without CUDA.
    PlanModel { model_dir: PathBuf },
    /// Verify model, execution package and tensor hashes without CUDA.
    ValidateModel { model_dir: PathBuf },
    /// Run requests through the native model package.
    RunModel(ModelArgs),
    /// Compute normalized text or image document embeddings with a native package.
    EmbedModel(ModelArgs),
    /// Score and rank documents for a query with a native reranker package.
    RerankModel(ModelArgs),
    /// Score frozen token histories through real prefill/decode kernels.
    ScoreModel(ModelArgs),
    /// Start the model service; an optional directory serves one model directly.
    Serve(Box<ServeArgs>),
    #[command(hide = true)]
    Worker(Box<ServeArgs>),
    /// Process Chat requests from JSONL without an HTTP server.
    Batch(Box<BatchArgs>),
}
#[derive(Subcommand)]
enum PackageCommand {
    /// Verify and atomically install a .tar.xz or .tar.gz execution package.
    Install {
        archive: PathBuf,
        /// Installation cache; defaults to ORINFER_EXECUTION_CACHE or the user cache.
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}
#[derive(Args, Clone)]
struct LoadArgs {
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
    /// Opt in to copied greedy drafts from this request's token history.
    #[arg(long)]
    prompt_lookup: bool,
}
#[derive(Args)]
struct ModelArgs {
    model_dir: PathBuf,
    requests: PathBuf,
    #[command(flatten)]
    load: LoadArgs,
}
#[derive(Args, Clone)]
struct InferenceArgs {
    /// Model name for batch, or prepared model directory; omit for managed serving.
    model_dir: Option<PathBuf>,
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
    /// auto adapts up to the package default; 0 disables MTP; 1..7 caps drafts.
    #[arg(long, value_name = "N", default_value = "auto")]
    mtp_drafts: MtpDrafts,
    /// Snapshot cache budget in MiB; 0 disables reuse.
    #[arg(long = "prefix-cache-mib", value_name = "MIB", default_value_t = ServerConfig::default().prefix_cache_bytes >> 20, value_parser = parse_cache_mib)]
    prefix_cache_bytes: usize,
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
    #[arg(long, default_value_t = PreprocessingLimits::default().preprocess_workers)]
    preprocess_workers: usize,
    #[arg(long, default_value_t = PreprocessingLimits::default().memory_mib)]
    preprocess_memory_mib: usize,
}
#[derive(Args)]
struct ServeArgs {
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
    #[arg(long, default_value = "auto")]
    max_loaded_models: String,
    #[arg(long, default_value_t = 120)]
    load_timeout_s: u64,
}
#[derive(Args)]
struct BatchArgs {
    #[arg(skip)]
    overrides: serde_json::Map<String, serde_json::Value>,
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
        Self {
            model_dir: args.model_dir.unwrap_or_default(),
            model: args.model,
            default_request_params: args.default_request_params.unwrap_or_default(),
            listen: ServerConfig::default().listen,
            gpu_lock: args.gpu_lock,
            cuda_graph: args.load.cuda_graph,
            verify_weights: args.load.verify_weights,
            load_workers: args.load.load_workers,
            resident_experts: args.load.resident_experts,
            mtp_drafts: args.mtp_drafts.0,
            prompt_lookup: args.load.prompt_lookup,
            prefix_cache_bytes: args.prefix_cache_bytes,
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
                ..Default::default()
            },
        }
    }
}
impl From<ServeArgs> for ServerConfig {
    fn from(args: ServeArgs) -> Self {
        let mut config: Self = args.inference.into();
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
            model_dir: config.model_dir,
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
            },
        }
    }
}
fn parse_request_defaults(
    value: &str,
) -> Result<serde_json::Map<String, serde_json::Value>, String> {
    serde_json::from_str(value).map_err(|e| format!("Expected a JSON object: {e}"))
}
fn print_json(value: impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn parse_keep_alive(value: &str) -> Result<u64, String> {
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

fn run_chat(
    data: &std::path::Path,
    name: &str,
    prompt: Option<String>,
    params: &str,
) -> Result<(), String> {
    use std::io::{IsTerminal, Read, Write};
    let parameters = parse_request_defaults(params)?;
    if parameters.contains_key("model") || parameters.contains_key("messages") {
        return Err("Use the model argument and prompt for model/messages".into());
    }
    let mut messages = Vec::new();
    let mut send = |prompt: String| -> Result<(), String> {
        messages.push(serde_json::json!({"role":"user","content":prompt}));
        let mut body = serde_json::Value::Object(parameters.clone());
        body["model"] = serde_json::json!(name);
        body["messages"] = serde_json::json!(messages);
        messages.push(orinfer_api::manager::client::chat(data, body)?);
        Ok(())
    };
    if let Some(prompt) = prompt {
        return send(prompt);
    }
    if !std::io::stdin().is_terminal() {
        let mut text = String::new();
        std::io::stdin()
            .take((32 << 20) + 1)
            .read_to_string(&mut text)
            .map_err(|e| e.to_string())?;
        if text.len() > 32 << 20 {
            return Err("Prompt exceeds 32 MiB".into());
        }
        return send(text);
    }
    loop {
        print!(">>> ");
        std::io::stdout().flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if std::io::stdin()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?
            == 0
            || line.trim() == "/bye"
        {
            break;
        }
        send(line)?;
    }
    Ok(())
}

fn execute(command: Command, data_dir: &std::path::Path) -> Result<(), String> {
    use orinfer_api::manager::client;
    use serde_json::json;
    match command {
        Command::Pull {
            source,
            name,
            revision,
            package,
            options,
        } => {
            let options = options
                .map(|value| serde_json::from_str::<orinfer_api::manager::WorkerOptions>(&value))
                .transpose()
                .map_err(|e| e.to_string())?;
            print_json(client::request(
                data_dir,
                "POST",
                "/pull",
                Some(
                    &json!({"source":source,"name":name,"revision":revision,"package":package,"options":options}),
                ),
            )?)
        }
        Command::Import {
            name,
            path,
            options,
        } => {
            let options = options
                .map(|value| serde_json::from_str::<orinfer_api::manager::WorkerOptions>(&value))
                .transpose()
                .map_err(|e| e.to_string())?;
            print_json(client::request(
                data_dir,
                "POST",
                "/import",
                Some(
                    &json!({"name":name,"path":path.canonicalize().map_err(|e| e.to_string())?,"options":options}),
                ),
            )?)
        }
        Command::Transcribe {
            name,
            input,
            language,
            prompt,
        } => {
            let value = client::transcribe(
                data_dir,
                &name,
                &input,
                language.as_deref(),
                prompt.as_deref(),
            )?;
            println!(
                "{}",
                value["text"].as_str().ok_or("Missing transcription text")?
            );
            Ok(())
        }
        Command::List => print_json(client::request(data_dir, "GET", "/models", None)?),
        Command::Ps => print_json(client::request(data_dir, "GET", "/ps", None)?),
        Command::Show { name } => print_json(client::show(data_dir, &name)?),
        Command::Load { name, keep_alive } => print_json(client::request(
            data_dir,
            "POST",
            "/load",
            Some(&json!({"name":name,"keep_alive_s":keep_alive})),
        )?),
        Command::Stop { name, force } => print_json(client::request(
            data_dir,
            "POST",
            "/stop",
            Some(&json!({"name":name,"force":force})),
        )?),
        Command::Rm { name } => print_json(client::request(
            data_dir,
            "POST",
            "/remove",
            Some(&json!({"name":name})),
        )?),
        Command::Run {
            name,
            prompt,
            params,
        } => run_chat(data_dir, &name, prompt, &params),
        Command::Logs { name, tail } => {
            let value = client::request(
                data_dir,
                "POST",
                "/logs",
                Some(&json!({"name":name,"tail":tail})),
            )?;
            print!("{}", value["text"].as_str().unwrap_or(""));
            Ok(())
        }
        Command::Package {
            command: PackageCommand::Install { archive, cache },
        } => print_json(orinfer_engine::operators::install::install(
            &archive,
            &cache.unwrap_or_else(orinfer_engine::operators::default_cache),
        )?),
        Command::Info => {
            println!("Orinfer {}", env!("CARGO_PKG_VERSION"));
            println!(
                "Target: {} / {} / {}",
                orinfer_engine::TARGET_DEVICE,
                orinfer_engine::TARGET_ARCH,
                orinfer_engine::CUDA_ARCH
            );
            println!("Model: {}", orinfer_engine::FIRST_MODEL);
            println!("Status: {}", orinfer_engine::STATUS);
            Ok(())
        }
        Command::Devices { backend } => print_json(orinfer_engine::backend::inspect(backend)?),
        Command::ValidateAscend { device } => {
            print_json(orinfer_engine::backend::ascend::validation::run(device)?)
        }
        Command::ValidateAscendKernels { device, fixtures } => print_json(
            orinfer_engine::backend::ascend::kernels::validation::run(device, &fixtures)?,
        ),
        Command::BenchAscend {
            plan,
            model_dir,
            inputs,
            output_tokens,
            repeats,
            trace,
        } => print_json(&orinfer_engine::backend::ascend::model::run(
            &plan,
            &model_dir,
            &inputs,
            output_tokens,
            repeats,
            trace.as_deref(),
        )?),
        Command::BenchAscendBatch {
            plan,
            model_dir,
            inputs,
        } => print_json(&orinfer_engine::backend::ascend::model::run_batch(
            &plan, &model_dir, &inputs,
        )?),
        Command::ValidateHccl { devices } => print_json(
            orinfer_engine::backend::ascend::hccl::validation::run(&devices)?,
        ),
        Command::ValidateEngram {
            metadata,
            model_dir,
        } => print_json(orinfer_engine::host_rows::validation::run(
            &metadata, &model_dir,
        )?),
        Command::Plan => {
            println!("{}", orinfer_engine::BENCHMARK_PLAN);
            Ok(())
        }
        Command::Serve(args) if args.inference.model_dir.is_none() => {
            let max_loaded_models = if args.max_loaded_models == "auto" {
                None
            } else {
                Some(
                    args.max_loaded_models
                        .parse::<usize>()
                        .map_err(|e| e.to_string())?,
                )
            };
            let settings: ServerConfig = args.inference.into();
            orinfer_api::manager::run(orinfer_api::manager::ManagerConfig {
                worker_options: orinfer_api::manager::WorkerOptions::from_settings(&settings),
                data_dir: data_dir.into(),
                listen: args.listen,
                keep_alive_s: args.keep_alive,
                max_loaded_models,
                load_timeout_s: args.load_timeout_s,
                gpu_lock: settings.gpu_lock,
                memory_reserve_mib: settings.scheduler.memory_reserve_bytes >> 20,
            })
        }
        Command::Serve(args) | Command::Worker(args) => orinfer_api::run((*args).into()),
        Command::Batch(args) => {
            let model = args
                .inference
                .model_dir
                .as_ref()
                .ok_or("Model name or directory is required")?;
            if !model.is_dir() {
                let result = client::request(
                    data_dir,
                    "POST",
                    "/batch",
                    Some(
                        &json!({"name":model.to_string_lossy(),"options":args.overrides,
                    "scheduler_policy":args.inference.scheduler_policy.unwrap_or(orinfer_engine::scheduler::Policy::Throughput).to_string(),
                    "input":std::path::absolute(&args.input).map_err(|e| e.to_string())?,
                    "output":std::path::absolute(&args.output).map_err(|e| e.to_string())?,
                    "resume":args.resume,"order":args.batch_order.to_string()}),
                    ),
                )?;
                let success = result["success"] == true;
                print_json(result["report"].clone())?;
                return if success {
                    Ok(())
                } else {
                    Err("Batch has errors, truncated responses or was interrupted; inspect the output JSONL".into())
                };
            }
            let report = orinfer_api::run_batch((*args).into())?;
            let failed = report.failed != 0 || report.truncated != 0 || report.interrupted;
            print_json(report)?;
            if failed {
                Err("Batch has errors, truncated responses or was interrupted; inspect the output JSONL".into())
            } else {
                Ok(())
            }
        }
        Command::ValidateArtifact { manifest } => {
            print_json(orinfer_engine::artifact::validate_artifact(&manifest)?)
        }
        Command::RunArtifact { manifest } => {
            print_json(orinfer_engine::artifact::run_artifact(&manifest)?)
        }
        Command::InspectLibrary { library } => {
            print_json(orinfer_engine::model::inspect_execution_library(&library)?)
        }
        Command::PlanModel { model_dir } => {
            print_json(orinfer_engine::model::inspect_plan(&model_dir)?)
        }
        Command::ValidateModel { model_dir } => {
            print_json(orinfer_engine::model::validate_model(&model_dir)?)
        }
        Command::RerankModel(args) => print_json(orinfer_api::reranking::run_model(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                ..Default::default()
            },
        )?),
        Command::EmbedModel(args) => print_json(orinfer_api::embeddings::run_model(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                ..Default::default()
            },
        )?),
        Command::RunModel(args) => print_json(orinfer_engine::model::run_with_options(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                prompt_lookup: args.load.prompt_lookup,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                resident_experts: args.load.resident_experts,
                ..Default::default()
            },
        )?),
        Command::ScoreModel(args) => print_json(orinfer_engine::model::score_with_options(
            &args.model_dir,
            &args.requests,
            LoadOptions {
                cuda_graph: args.load.cuda_graph,
                prompt_lookup: args.load.prompt_lookup,
                verify_weights: args.load.verify_weights,
                load_workers: args.load.load_workers,
                resident_experts: args.load.resident_experts,
                ..Default::default()
            },
        )?),
    }
}
fn batch_overrides(
    matches: &clap::ArgMatches,
    inference: &InferenceArgs,
) -> serde_json::Map<String, serde_json::Value> {
    let settings: ServerConfig = inference.clone().into();
    let mut values = serde_json::to_value(orinfer_api::manager::WorkerOptions::from_settings(
        &settings,
    ))
    .unwrap()
    .as_object()
    .unwrap()
    .clone();
    let fields = [
        ("prefix_cache_bytes", "prefix_cache_mib"),
        ("memory_reserve_bytes", "memory_reserve_mib"),
    ];
    values.retain(|key, _| {
        let id = fields
            .iter()
            .find(|(_, field)| *field == key)
            .map(|(id, _)| *id)
            .unwrap_or(key.as_str());
        matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine)
    });
    values
}
fn exit_with_unfinished_workers(pending: bool) {
    if pending {
        // SAFETY: An API error returned while native teardown is still live.
        // Running CANN process-exit hooks concurrently can double-free native
        // state. _exit terminates this standalone process without those hooks;
        // the OS reclaims its threads, accelerator handles and GPU lease.
        unsafe { libc::_exit(1) };
    }
}
fn main() -> ExitCode {
    let matches = Cli::command().get_matches();
    let mut cli = Cli::from_arg_matches(&matches).unwrap_or_else(|e| e.exit());
    if let Some(Command::Batch(args)) = &mut cli.command {
        args.overrides = batch_overrides(
            matches.subcommand_matches("batch").unwrap(),
            &args.inference,
        );
    }
    if matches!(
        &cli.command,
        Some(Command::Serve(_) | Command::Worker(_) | Command::Batch(_))
    ) && std::env::var_os("ASCEND_COREDUMP_SIGNAL").is_none()
    {
        // SAFETY: main has not started runtime, preprocessing or accelerator
        // threads. CANN trace handlers otherwise preempt graceful shutdown.
        unsafe { std::env::set_var("ASCEND_COREDUMP_SIGNAL", "none") };
    }
    match execute(cli.command.unwrap_or(Command::Info), &cli.data_dir) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            exit_with_unfinished_workers(orinfer_api::has_live_model_workers());
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
    #[test]
    fn backend_inventory_and_graph_alias_are_typed() {
        let Some(Command::Devices { backend }) =
            Cli::try_parse_from(["orinfer", "devices", "--backend", "ascend"])
                .unwrap()
                .command
        else {
            unreachable!()
        };
        assert_eq!(backend, orinfer_engine::backend::Backend::Ascend);
        assert!(Cli::try_parse_from(["orinfer", "devices", "--backend", "unknown"]).is_err());
        let Some(Command::RunModel(args)) = Cli::try_parse_from([
            "orinfer",
            "run-model",
            "model",
            "requests.json",
            "--graph",
            "full",
        ])
        .unwrap()
        .command
        else {
            unreachable!()
        };
        assert_eq!(args.load.cuda_graph, CudaGraphMode::Full);
    }
    use super::*;
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
        let Some(Command::Batch(args)) = cli.command else {
            panic!("batch command")
        };
        let options = batch_overrides(
            matches.subcommand_matches("batch").unwrap(),
            &args.inference,
        );
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
        let Some(Command::Serve(args)) = Cli::try_parse_from(args)?.command else {
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
        assert_eq!(config.prefix_cache_bytes, 12usize << 30);
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
        let Some(Command::Batch(args)) = Cli::try_parse_from([
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
        let Some(Command::Batch(args)) = Cli::try_parse_from([
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
            &["--mtp-drafts", "8"],
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
    fn package_install_accepts_an_archive_and_optional_cache() {
        for cache in [None, Some("/tmp/packages")] {
            let mut args = vec!["orinfer", "package", "install", "model.tar.xz"];
            if let Some(path) = cache {
                args.extend(["--cache", path]);
            }
            let Some(Command::Package {
                command:
                    PackageCommand::Install {
                        archive,
                        cache: parsed,
                    },
            }) = Cli::try_parse_from(args).unwrap().command
            else {
                unreachable!()
            };
            assert_eq!(archive, PathBuf::from("model.tar.xz"));
            assert_eq!(parsed, cache.map(PathBuf::from));
        }
        for args in [
            vec!["orinfer", "package"],
            vec!["orinfer", "package", "install"],
            vec!["orinfer", "package", "install", "model.tar.xz", "extra"],
        ] {
            assert!(Cli::try_parse_from(args).is_err());
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
    #[test]
    fn model_subcommands_support_equals_and_reject_extra_positionals() {
        let cli = Cli::try_parse_from([
            "orinfer",
            "run-model",
            "model",
            "requests.json",
            "--cuda-graph=full",
            "--verify-weights",
        ])
        .unwrap();
        let Some(Command::RunModel(args)) = cli.command else {
            unreachable!()
        };
        assert_eq!(args.load.cuda_graph, CudaGraphMode::Full);
        assert!(args.load.verify_weights);
        assert_eq!(
            serve(&["--load-workers", "12"]).unwrap().load_workers,
            Some(12)
        );
        assert!(serve(&["--load-workers", "0"]).is_err());
        assert!(serve(&["--load-workers", "33"]).is_err());
        for verify in [false, true] {
            let mut argv = vec!["orinfer", "score-model", "model", "requests.json"];
            if verify {
                argv.push("--verify-weights");
            }
            let Some(Command::ScoreModel(args)) = Cli::try_parse_from(argv).unwrap().command else {
                unreachable!()
            };
            assert_eq!(args.load.verify_weights, verify);
        }
        assert!(Cli::try_parse_from(["orinfer", "run-model", "model"]).is_err());
        assert!(Cli::try_parse_from(["orinfer", "plan-model", "model", "extra"]).is_err());
    }
}
