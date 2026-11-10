//! Model defaults follow official model cards and clients. Request/deployment
//! values are applied afterwards; model identity never comes from the API alias.
use super::{Options, Result, Value, normalize_top_k};
use std::path::Path;

pub(super) struct Defaults {
    sampling: Options,
    non_thinking: Option<Options>,
    pub thinking: bool,
    max_tokens: usize,
    deepseek: bool,
}

impl Defaults {
    pub fn load(directory: &Path, descriptor: &Value, generation: &Value) -> Result<Self> {
        let config: Value = match std::fs::read(directory.join("config.json")) {
            Ok(raw) => serde_json::from_slice(&raw).map_err(|e| e.to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(e.to_string()),
        };
        let source = config["quantization_config"]["source"]
            .as_str()
            .or_else(|| descriptor["metadata"]["model"].as_str())
            .or_else(|| config["_name_or_path"].as_str())
            .unwrap_or("")
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        let qwen38 = source.starts_with("qwen3.8-");
        let qwen35 = source == "qwen3.5-4b" || source.starts_with("qwen3.5-4b-");
        let index = source.starts_with("index-translate-");
        let qwen = qwen38 || qwen35;
        let deepseek = descriptor["metadata"]["chat_format"] == "deepseek_v41";
        let mut sampling = Options::default();
        if qwen {
            // Qwen3.8 / Qwen3.5-4B model cards: general-task thinking defaults.
            sampling.top_p = 0.95;
            sampling.top_k = 20;
            sampling.presence_penalty = if qwen35 { 1.5 } else { 0.0 };
        }
        sampling.temperature = generation["temperature"]
            .as_f64()
            .unwrap_or(sampling.temperature);
        sampling.top_p = generation["top_p"].as_f64().unwrap_or(sampling.top_p);
        sampling.top_k = generation["top_k"]
            .as_i64()
            .map(normalize_top_k)
            .transpose()?
            .unwrap_or(sampling.top_k);
        sampling.repetition_penalty = generation["repetition_penalty"]
            .as_f64()
            .unwrap_or(sampling.repetition_penalty);
        sampling.presence_penalty = generation["presence_penalty"]
            .as_f64()
            .unwrap_or(sampling.presence_penalty);
        sampling.frequency_penalty = generation["frequency_penalty"]
            .as_f64()
            .unwrap_or(sampling.frequency_penalty);
        if index {
            // IndexTeam's translate.py defaults to greedy for all three sizes.
            sampling.temperature = 0.0;
        }
        if deepseek {
            // Match DeepInfer, including checkpoints exported with legacy greedy defaults.
            sampling = Options::default();
        }
        let mut non_thinking = qwen.then(|| Options {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 20,
            presence_penalty: 1.5,
            ..sampling.clone()
        });
        if generation["do_sample"] == false && !deepseek {
            sampling.temperature = 0.0;
            if let Some(instruct) = &mut non_thinking {
                instruct.temperature = 0.0;
            }
        }
        sampling.validate()?;
        if let Some(instruct) = &non_thinking {
            instruct.validate()?;
        }
        Ok(Self {
            sampling,
            non_thinking,
            thinking: qwen,
            deepseek,
            // Index follows its client; DeepSeek thinking uses the API policy below.
            max_tokens: if index { 1024 } else { 8192 },
        })
    }

    pub fn max_tokens(&self, thinking: bool, effort: Option<&str>) -> usize {
        if self.deepseek && thinking {
            if effort == Some("max") { 131072 } else { 65536 }
        } else {
            self.max_tokens
        }
    }

    pub fn sampling(&self, thinking: bool) -> &Options {
        if !thinking && let Some(instruct) = &self.non_thinking {
            instruct
        } else {
            &self.sampling
        }
    }
}
