mod server;

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|c| c == "serve") {
        return match server::run(&args[1..]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("serve: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if args.first().is_some_and(|c| c == "run-model") {
        if args.len() != 3 {
            eprintln!("Usage: orin-llm run-model MODEL.json REQUESTS.json");
            return ExitCode::from(2);
        }
        return match orin_engine::model::run(
            std::path::Path::new(&args[1]),
            std::path::Path::new(&args[2]),
        )
        .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("run-model: {error}");
                ExitCode::FAILURE
            }
        };
    }
    if let Some(command @ ("validate-artifact" | "run-artifact")) = args.first().map(String::as_str)
    {
        if args.len() != 2 {
            eprintln!("Usage: orin-llm {command} MANIFEST.json");
            return ExitCode::from(2);
        }
        let path = std::path::Path::new(&args[1]);
        let result = if command == "validate-artifact" {
            orin_engine::artifact::validate_artifact(path)
                .and_then(|r| serde_json::to_string_pretty(&r).map_err(|e| e.to_string()))
        } else {
            orin_engine::artifact::run_artifact(path)
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
        eprintln!("Usage: orin-llm [info | plan | --version | --help]");
        return ExitCode::from(2);
    }
    match args.first().map(String::as_str).unwrap_or("info") {
        "info" => {
            println!("Orin LLM {}", env!("CARGO_PKG_VERSION"));
            println!(
                "Target: {} / {} / {}",
                orin_engine::TARGET_DEVICE,
                orin_engine::TARGET_ARCH,
                orin_engine::CUDA_ARCH
            );
            println!("Model: {}", orin_engine::FIRST_MODEL);
            println!("Status: {}", orin_engine::STATUS);
        }
        "plan" => println!("{}", orin_engine::BENCHMARK_PLAN),
        "--version" | "-V" => println!("orin-llm {}", env!("CARGO_PKG_VERSION")),
        "--help" | "-h" => println!(
            "Usage: orin-llm [info | plan | --version | --help]\n       orin-llm <validate-artifact | run-artifact> MANIFEST.json\n       orin-llm run-model MODEL.json REQUESTS.json\n       orin-llm serve MODEL.json TOKENIZER_DIR [--listen HOST:PORT] [--model MODEL_ID]\n\ninfo    Show target and implementation status\nplan    Print the benchmark specification as JSON\nvalidate-artifact    Check AOT fixture and file hashes without CUDA\nrun-artifact         Execute an SM87 AOT projection fixture and validate replay\nrun-model            Load and execute AOT text-model programs with private state"
        ),
        other => {
            eprintln!(
                "Unknown command: {other}\nUsage: orin-llm [info | plan | --version | --help]"
            );
            return ExitCode::from(2);
        }
    }
    ExitCode::SUCCESS
}
