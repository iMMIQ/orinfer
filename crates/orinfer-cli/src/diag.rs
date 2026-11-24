use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::{args::LoadArgs, print_json};

#[derive(Subcommand)]
pub(crate) enum Command {
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
    /// Compare the complete native A2 vision encoder with checkpoint reference fixtures.
    ValidateAscendVision {
        #[arg(long, default_value_t = 7)]
        device: u32,
        #[arg(long)]
        program: PathBuf,
        #[arg(long)]
        model_dir: PathBuf,
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
    /// Execute a CUDA AOT fixture and validate graph replay.
    RunArtifact { manifest: PathBuf },
    /// Inspect the native model package plan without CUDA.
    PlanModel {
        model_dir: PathBuf,
        /// Inspect private operands for the listed packet shapes before address-table wrapping.
        #[arg(long)]
        packet_probes: Option<PathBuf>,
    },
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
}
#[derive(Subcommand)]
pub(crate) enum PackageCommand {
    /// Offline: encode exact FP32 RoPE assets for compact execution packages.
    EncodeRope {
        #[arg(long)]
        cos: PathBuf,
        #[arg(long)]
        sin: PathBuf,
        #[arg(long)]
        columns: usize,
        #[arg(long)]
        output: PathBuf,
    },
    /// Verify and atomically install a .tar.xz or .tar.gz execution package.
    Install {
        archive: PathBuf,
        /// Installation cache; defaults to ORINFER_EXECUTION_CACHE or the user cache.
        #[arg(long)]
        cache: Option<PathBuf>,
    },
}
#[derive(Args)]
pub(crate) struct ModelArgs {
    model_dir: PathBuf,
    requests: PathBuf,
    #[command(flatten)]
    load: LoadArgs,
}

pub(crate) fn run(command: Command) -> Result<(), String> {
    match command {
        Command::Package {
            command: PackageCommand::Install { archive, cache },
        } => print_json(orinfer_engine::operators::install::install(
            &archive,
            &cache.unwrap_or_else(orinfer_engine::operators::default_cache),
        )?),
        Command::Package {
            command:
                PackageCommand::EncodeRope {
                    cos,
                    sin,
                    columns,
                    output,
                },
        } => print_json(orinfer_engine::operators::generated::encode_rope(
            &cos, &sin, columns, &output,
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
        Command::ValidateAscendVision {
            device,
            program,
            model_dir,
            fixtures,
        } => print_json(orinfer_engine::backend::ascend::model::vision::validate(
            device, &program, &model_dir, &fixtures,
        )?),
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
        Command::ValidateArtifact { manifest } => {
            print_json(orinfer_engine::artifact::validate_artifact(&manifest)?)
        }
        Command::RunArtifact { manifest } => {
            print_json(orinfer_engine::artifact::run_artifact(&manifest)?)
        }
        Command::PlanModel {
            model_dir,
            packet_probes,
        } => {
            if let Some(probes) = packet_probes {
                print_json(orinfer_engine::model::inspect_packet_layout(
                    &model_dir, &probes,
                )?)
            } else {
                print_json(orinfer_engine::model::inspect_plan(&model_dir)?)
            }
        }
        Command::ValidateModel { model_dir } => {
            print_json(orinfer_engine::model::validate_model(&model_dir)?)
        }
        Command::RerankModel(args) => print_json(orinfer_api::reranking::run_model(
            &args.model_dir,
            &args.requests,
            args.load.encoder_options(),
        )?),
        Command::EmbedModel(args) => print_json(orinfer_api::embeddings::run_model(
            &args.model_dir,
            &args.requests,
            args.load.encoder_options(),
        )?),
        Command::RunModel(args) => print_json(orinfer_engine::model::run_with_options(
            &args.model_dir,
            &args.requests,
            args.load.options(),
        )?),
        Command::ScoreModel(args) => print_json(orinfer_engine::model::score_with_options(
            &args.model_dir,
            &args.requests,
            args.load.options(),
        )?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{Cli, Command as TopCommand};
    use clap::Parser;
    use orinfer_engine::execution::CudaGraphMode;

    #[test]
    fn backend_inventory_and_graph_alias_are_typed() {
        let Some(TopCommand::Diagnostics(Command::Devices { backend })) =
            Cli::try_parse_from(["orinfer", "devices", "--backend", "ascend"])
                .unwrap()
                .command
        else {
            unreachable!()
        };
        assert_eq!(backend, orinfer_engine::backend::Backend::Ascend);
        assert!(Cli::try_parse_from(["orinfer", "devices", "--backend", "unknown"]).is_err());
        let Some(TopCommand::Diagnostics(Command::RunModel(args))) = Cli::try_parse_from([
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
        assert_eq!(args.load.options().cuda_graph, CudaGraphMode::Full);
    }
    #[test]
    fn package_install_accepts_an_archive_and_optional_cache() {
        for cache in [None, Some("/tmp/packages")] {
            let mut args = vec!["orinfer", "package", "install", "model.tar.xz"];
            if let Some(path) = cache {
                args.extend(["--cache", path]);
            }
            let Some(TopCommand::Diagnostics(Command::Package {
                command:
                    PackageCommand::Install {
                        archive,
                        cache: parsed,
                    },
            })) = Cli::try_parse_from(args).unwrap().command
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
        let Some(TopCommand::Diagnostics(Command::RunModel(args))) = cli.command else {
            unreachable!()
        };
        let load = args.load.options();
        assert_eq!(load.cuda_graph, CudaGraphMode::Full);
        assert!(load.verify_weights);
        for verify in [false, true] {
            let mut argv = vec!["orinfer", "score-model", "model", "requests.json"];
            if verify {
                argv.push("--verify-weights");
            }
            let Some(TopCommand::Diagnostics(Command::ScoreModel(args))) =
                Cli::try_parse_from(argv).unwrap().command
            else {
                unreachable!()
            };
            assert_eq!(args.load.options().verify_weights, verify);
        }
        assert!(Cli::try_parse_from(["orinfer", "run-model", "model"]).is_err());
        assert!(Cli::try_parse_from(["orinfer", "plan-model", "model", "extra"]).is_err());
    }
}
