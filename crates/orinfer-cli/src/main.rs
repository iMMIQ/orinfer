use orinfer_engine::execution::LoadOptions;
use std::process::ExitCode;

fn model_options(args: &[String]) -> Result<LoadOptions, String> {
    if args.len() < 2 {
        return Err(
            "Usage: orinfer <run-model | score-model> MODEL_DIR REQUESTS.json [--cuda-graph decode_only|full|off]"
                .into(),
        );
    }
    let mut options = LoadOptions::default();
    for pair in args[2..].chunks(2) {
        if pair.len() != 2 {
            return Err("Model option needs a value".into());
        }
        match pair[0].as_str() {
            "--cuda-graph" => options.cuda_graph = pair[1].parse()?,
            _ => return Err(format!("Unknown model option {}", pair[0])),
        }
    }
    Ok(options)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|c| c == "serve") {
        return match orinfer_api::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("serve: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args
        .first()
        .is_some_and(|c| c == "validate-model" || c == "plan-model")
    {
        if args.len() != 2 {
            eprintln!("Usage: orinfer <validate-model | plan-model> MODEL_DIR");
            return ExitCode::from(2);
        }
        let path = std::path::Path::new(&args[1]);
        let result = if args[0] == "plan-model" {
            orinfer_engine::model::inspect_plan(path)
                .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        } else {
            orinfer_engine::model::validate_model(path)
                .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        };
        return match result {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{}: {error}", args[0]);
                ExitCode::FAILURE
            }
        };
    }
    if args
        .first()
        .is_some_and(|c| c == "run-model" || c == "score-model")
    {
        let options = match model_options(&args[1..]) {
            Ok(options) => options,
            Err(error) => {
                eprintln!("{error}");
                return ExitCode::from(2);
            }
        };
        let result = if args[0] == "score-model" {
            orinfer_engine::model::score_with_options(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
                options,
            )
            .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        } else {
            orinfer_engine::model::run_with_options(
                std::path::Path::new(&args[1]),
                std::path::Path::new(&args[2]),
                options,
            )
            .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        };
        return match result {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{}: {error}", args[0]);
                ExitCode::FAILURE
            }
        };
    }
    if let Some(command @ ("validate-artifact" | "run-artifact")) = args.first().map(String::as_str)
    {
        if args.len() != 2 {
            eprintln!("Usage: orinfer {command} MANIFEST.json");
            return ExitCode::from(2);
        }
        let path = std::path::Path::new(&args[1]);
        let result = if command == "validate-artifact" {
            orinfer_engine::artifact::validate_artifact(path)
                .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        } else {
            orinfer_engine::artifact::run_artifact(path)
                .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        };
        return match result {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("{command}: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.len() > 1 {
        eprintln!("Usage: orinfer [info | plan | --version | --help]");
        return ExitCode::from(2);
    }
    match args.first().map(String::as_str).unwrap_or("info") {
        "info" => {
            println!("Orinfer {}", env!("CARGO_PKG_VERSION"));
            println!(
                "Target: {} / {} / {}",
                orinfer_engine::TARGET_DEVICE,
                orinfer_engine::TARGET_ARCH,
                orinfer_engine::CUDA_ARCH
            );
            println!("Model: {}", orinfer_engine::FIRST_MODEL);
            println!("Status: {}", orinfer_engine::STATUS);
        }
        "plan" => println!("{}", orinfer_engine::BENCHMARK_PLAN),
        "--version" | "-V" => println!("orinfer {}", env!("CARGO_PKG_VERSION")),
        "--help" | "-h" => println!(
            "Usage: orinfer [info | plan | --version | --help]\n       orinfer <validate-artifact | run-artifact> MANIFEST.json\n       orinfer <run-model | score-model> MODEL_DIR REQUESTS.json [--cuda-graph MODE]\n       orinfer serve MODEL_DIR [--listen HOST:PORT] [--model MODEL_ID] [--cuda-graph MODE]\n\n--cuda-graph MODE     decode_only (default), full or off\n--listen HOST:PORT    Default: 0.0.0.0:8088\n--prefix-cache-mib N  Server snapshot budget: 12288 MiB; 0 disables reuse\n--max-active-requests N  Active request limit: 32 (1..128)\n--max-batch-tokens N     Iteration token budget: 128 (1..128)\n--prefill-budget-ms N    Mixed prefill predicted budget: 200 ms\n--memory-reserve-mib N   Admission memory reserve: 1024 MiB\n--preprocess-workers N  Blocking CPU preprocessing limit: 2\n--preprocess-memory-mib N  Preprocessing/queued image memory budget: 2048 MiB\n--queue-timeout-ms N    Queue wait deadline: 0 (unlimited)\n--output-timeout-ms N   Stalled output deadline: 60000 ms\n--drain-timeout-ms N    SIGINT/SIGTERM HTTP drain deadline: 30000 ms\n\ninfo    Show target and implementation status\nplan    Print the benchmark specification as JSON\nvalidate-artifact    Check AOT fixture and file hashes without CUDA\nrun-artifact         Execute an SM87 AOT projection fixture and validate replay\nplan-model           Inspect the native model package plan without CUDA\nvalidate-model       Verify model, execution package and tensor hashes without CUDA\nrun-model            Load custom model data and native package plans with private state\nscore-model          Score frozen token histories through real prefill/decode kernels"
        ),
        other => {
            eprintln!(
                "Unknown command: {other}\nUsage: orinfer [info | plan | --version | --help]"
            );
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use orinfer_engine::execution::CudaGraphMode;

    #[test]
    fn run_model_graph_modes_are_validated_before_loading() {
        let required = vec!["model-dir".into(), "requests.json".into()];
        assert_eq!(
            model_options(&required).unwrap().cuda_graph,
            CudaGraphMode::DecodeOnly
        );
        assert!(model_options(&[]).is_err());
        assert!(model_options(&required[..1]).is_err());
        for mode in ["decode_only", "full", "off"] {
            let args = [required.clone(), vec!["--cuda-graph".into(), mode.into()]].concat();
            assert_eq!(model_options(&args).unwrap().cuda_graph.to_string(), mode);
        }
        for options in [
            vec!["--cuda-graph"],
            vec!["--cuda-graph", "on"],
            vec!["--unknown", "off"],
        ] {
            let args = [
                required.clone(),
                options.into_iter().map(String::from).collect(),
            ]
            .concat();
            assert!(model_options(&args).is_err());
        }
    }
}
