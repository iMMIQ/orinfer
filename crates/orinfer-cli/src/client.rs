use std::path::{Path, PathBuf};

use clap::Subcommand;
use orinfer_api::manager::{WorkerOptions, client};
use serde_json::{Map, Value, json};

use crate::args::{parse_keep_alive, parse_request_defaults};
use crate::print_json;

#[derive(Subcommand)]
pub(crate) enum Command {
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
        #[arg(long, value_parser = parse_worker_options)]
        options: Option<Box<WorkerOptions>>,
    },
    /// Register an existing native model without copying its weights.
    Import {
        name: String,
        path: PathBuf,
        #[arg(long, value_parser = parse_worker_options)]
        options: Option<Box<WorkerOptions>>,
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
        #[arg(long, default_value = "{}", value_parser = parse_request_defaults)]
        params: Map<String, Value>,
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
}

fn parse_worker_options(value: &str) -> Result<Box<WorkerOptions>, serde_json::Error> {
    serde_json::from_str(value)
}

pub(crate) fn run(command: Command, data_dir: &Path) -> Result<(), String> {
    match command {
        Command::Pull {
            source,
            name,
            revision,
            package,
            options,
        } => print_json(client::request(
            data_dir,
            "POST",
            "/pull",
            Some(
                &json!({"source":source,"name":name,"revision":revision,"package":package,"options":options}),
            ),
        )?),
        Command::Import {
            name,
            path,
            options,
        } => print_json(client::request(
            data_dir,
            "POST",
            "/import",
            Some(
                &json!({"name":name,"path":path.canonicalize().map_err(|e| e.to_string())?,"options":options}),
            ),
        )?),
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
        } => run_chat(data_dir, &name, prompt, params),
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
    }
}

fn run_chat(
    data: &Path,
    name: &str,
    prompt: Option<String>,
    parameters: Map<String, Value>,
) -> Result<(), String> {
    use std::io::{IsTerminal, Read, Write};
    if parameters.contains_key("model") || parameters.contains_key("messages") {
        return Err("Use the model argument and prompt for model/messages".into());
    }
    let mut messages = Vec::new();
    let mut send = |prompt: String| -> Result<(), String> {
        messages.push(json!({"role":"user","content":prompt}));
        let mut body = serde_json::Value::Object(parameters.clone());
        body["model"] = json!(name);
        body["messages"] = json!(messages);
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
