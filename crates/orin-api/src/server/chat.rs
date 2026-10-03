use minijinja::{Environment, context};
use orin_engine::sampling::Options;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
use tokenizers::Tokenizer;

type Result<T> = std::result::Result<T, String>;

#[cfg(test)]
mod mtp_fixture_export {
    use super::*;
    #[test]
    #[ignore = "Offline export of native chat-template code fixtures for the GPU MTP test"]
    fn export_mtp_chat_fixture() {
        let source = std::env::var("ORIN_MTP_FIXTURE_SPEC").unwrap();
        let raw: Value = serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
        let codec = ChatCodec::load(Path::new(raw["tokenizer"].as_str().unwrap())).unwrap();
        let mut cases = vec![];
        for case in raw["cases"].as_array().unwrap() {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let prepared = codec.prepare(request, "qwen3.8-27b", 8704, None).unwrap();
            assert!(prepared.images.is_empty() && prepared.sampling.is_greedy());
            cases.push(json!({"id":case["id"],"input_tokens":prepared.input,
                "max_new_tokens":prepared.max_tokens}));
        }
        let fixture = json!({"model":raw["model"],"output":raw["result"],
            "cases":cases,"eos":codec.eos,"repetitions":raw["repetitions"]});
        let output = Path::new(raw["fixture"].as_str().unwrap());
        assert!(!output.exists());
        std::fs::write(output, serde_json::to_vec_pretty(&fixture).unwrap()).unwrap();
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub seed: Option<u64>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<Value>,
    pub stop: Option<Value>,
    pub n: Option<usize>,
    pub logprobs: Option<bool>,
    pub response_format: Option<Value>,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub user: Option<String>,
}

pub struct Prepared {
    pub input: Vec<u32>,
    pub images: Vec<orin_engine::vision::ImageInput>,
    pub max_tokens: usize,
    pub sampling: Options,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel: bool,
    pub stops: Vec<String>,
    pub thinking: bool,
    pub stream: bool,
    pub include_usage: bool,
}

pub struct ChatCodec {
    pub tokenizer: Tokenizer,
    template: Environment<'static>,
    pub eos: BTreeSet<u32>,
}
impl ChatCodec {
    pub fn load(directory: &Path) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(directory.join("tokenizer.json")).map_err(|e| e.to_string())?;
        let template = std::fs::read_to_string(directory.join("chat_template.jinja"))
            .map_err(|e| e.to_string())?;
        let generation: Value = serde_json::from_slice(
            &std::fs::read(directory.join("generation_config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let ids = &generation["eos_token_id"];
        let eos: BTreeSet<u32> = if let Some(id) = ids.as_u64() {
            [u32::try_from(id).map_err(|e| e.to_string())?]
                .into_iter()
                .collect()
        } else {
            ids.as_array()
                .ok_or("Missing eos_token_id")?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| "Invalid EOS ID".into())
                })
                .collect::<Result<_>>()?
        };
        if eos.is_empty() || eos.iter().any(|id| tokenizer.id_to_token(*id).is_none()) {
            return Err("EOS outside tokenizer vocabulary".into());
        }
        let mut environment = Environment::new();
        // The checkpoint uses Python's startswith/endswith methods on strings.
        environment.set_unknown_method_callback(|_, value, method, args| {
            if let (Some(text), [arg]) = (value.as_str(), args)
                && let Some(arg) = arg.as_str()
            {
                match method {
                    "startswith" => return Ok(minijinja::Value::from(text.starts_with(arg))),
                    "endswith" => return Ok(minijinja::Value::from(text.ends_with(arg))),
                    _ => {}
                }
            }
            Err(minijinja::Error::new(
                minijinja::ErrorKind::UnknownMethod,
                "Unsupported template method",
            ))
        });
        // Match Transformers' Unicode-preserving JSON filter and separators.
        environment.add_filter(
            "tojson",
            |value: minijinja::Value| -> std::result::Result<minijinja::Value, minijinja::Error> {
                let raw = serde_json::to_string(&value).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })?;
                let mut spaced = String::new();
                let (mut quoted, mut escaped) = (false, false);
                for c in raw.chars() {
                    spaced.push(c);
                    if escaped {
                        escaped = false;
                        continue;
                    }
                    if quoted && c == '\\' {
                        escaped = true;
                    } else if c == '"' {
                        quoted = !quoted;
                    } else if !quoted && (c == ',' || c == ':') {
                        spaced.push(' ');
                    }
                }
                Ok(minijinja::Value::from_safe_string(spaced))
            },
        );
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        environment
            .add_template_owned("chat", template)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            tokenizer,
            template: environment,
            eos,
        })
    }
    pub fn prepare(
        &self,
        request: ChatRequest,
        model: &str,
        context_limit: usize,
        vision: Option<&orin_engine::vision::VisionSpec>,
    ) -> Result<Prepared> {
        if request.model != model {
            return Err(format!("Unknown model {}; expected {model}", request.model));
        }
        if request.n.is_some_and(|n| n != 1)
            || request.logprobs == Some(true)
            || request
                .response_format
                .as_ref()
                .is_some_and(|v| v["type"] != "text")
        {
            return Err("Supported: n=1, logprobs=false, response_format.type=text".into());
        }
        if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
            return Err("Specify only one output-token limit".into());
        }
        let max_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(512);
        if max_tokens == 0 {
            return Err("Output-token limit must be positive".into());
        }
        let sampling = Options {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            seed: request
                .seed
                .unwrap_or(orin_engine::sampling::EVALUATION_SEED),
        };
        sampling.validate()?;
        let thinking = request.enable_thinking.unwrap_or(false);
        if request
            .reasoning_effort
            .as_deref()
            .is_some_and(|s| !["xhigh", "medium", "low"].contains(&s))
        {
            return Err("reasoning_effort: xhigh, medium or low".into());
        }
        let stops = match request.stop {
            None | Some(Value::Null) => vec![],
            Some(Value::String(s)) => vec![s],
            Some(Value::Array(v)) => v
                .into_iter()
                .map(|v| {
                    v.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "stop must contain strings".into())
                })
                .collect::<Result<_>>()?,
            _ => return Err("stop must be a string or array".into()),
        };
        if stops.len() > 4 || stops.iter().any(String::is_empty) {
            return Err("stop allows at most four nonempty strings".into());
        }
        let mut names = BTreeSet::new();
        for tool in &request.tools {
            let function = &tool["function"];
            let name = function["name"].as_str().ok_or("Tool name missing")?;
            if tool["type"] != "function" || !valid_name(name) || !names.insert(name.to_owned()) {
                return Err("Tools need unique, valid function names".into());
            }
            if !function["parameters"].is_null() && !function["parameters"].is_object() {
                return Err("Tool parameters must be a JSON schema object".into());
            }
        }
        let choice = request.tool_choice.unwrap_or(json!("auto"));
        let tools = match choice.as_str() {
            Some("none") => vec![],
            Some("auto") => request.tools,
            Some("required") if !request.tools.is_empty() => request.tools,
            None if choice["type"] == "function" => {
                let name = choice["function"]["name"]
                    .as_str()
                    .ok_or("Forced tool name missing")?;
                vec![
                    request
                        .tools
                        .into_iter()
                        .find(|v| v["function"]["name"] == name)
                        .ok_or("Forced tool was not declared")?,
                ]
            }
            _ => return Err("Invalid tool_choice".into()),
        };
        let (raw_messages, images) = super::image::messages(request.messages, vision)?;
        let mut messages = normalize_messages(raw_messages)?;
        if choice == "required" || choice.is_object() {
            let instruction = if let Some(name) = choice["function"]["name"].as_str() {
                format!("For this response you must call the function {name}.")
            } else {
                "For this response you must call at least one of the provided functions.".into()
            };
            if messages[0]["role"] == "system" {
                let content = messages[0]["content"].as_str().unwrap_or("");
                messages[0]["content"] = json!(format!("{content}\n\n{instruction}"));
            } else {
                messages.insert(0, json!({"role":"system", "content":instruction}));
            }
        }
        let rendered = self.template.get_template("chat").map_err(|e| e.to_string())?.render(context! {
            messages => messages, tools => tools, add_generation_prompt => true,
            enable_thinking => thinking, reasoning_effort => request.reasoning_effort.as_deref().unwrap_or("xhigh"),
            preserve_thinking => true,
        }).map_err(|e| format!("Chat template: {e}"))?;
        let input = self
            .tokenizer
            .encode(rendered, false)
            .map_err(|e| e.to_string())?
            .get_ids()
            .to_vec();
        let input = if let Some(v) = vision {
            v.expand(&input, &images, context_limit)?
        } else {
            input
        };
        if input.is_empty()
            || input
                .len()
                .checked_add(max_tokens)
                .is_none_or(|n| n > context_limit)
        {
            return Err(format!(
                "Context limit {context_limit}: {} prompt tokens + {max_tokens} output tokens",
                input.len()
            ));
        }
        let include_usage = request
            .stream_options
            .as_ref()
            .is_some_and(|v| v["include_usage"] == true);
        let _ = request.user; // Attribution only; it does not alter sampling identity.
        Ok(Prepared {
            input,
            images,
            max_tokens,
            sampling,
            tools,
            tool_choice: choice,
            parallel: request.parallel_tool_calls.unwrap_or(true),
            stops,
            thinking,
            stream: request.stream,
            include_usage,
        })
    }
}
fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}
fn text_content(v: &Value) -> Result<String> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|v| {
                if v["type"] == "text" {
                    v["text"]
                        .as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "Text part needs text".into())
                } else {
                    Err("Only text message parts are supported".into())
                }
            })
            .collect::<Result<Vec<_>>>()
            .map(|v| v.concat()),
        _ => Err("Invalid message content".into()),
    }
}
fn normalize_messages(mut messages: Vec<Value>) -> Result<Vec<Value>> {
    if messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut pending = BTreeSet::new();
    let mut all_ids = BTreeSet::new();
    let mut user = false;
    let mut dialogue = false;
    for message in &mut messages {
        let role = message["role"]
            .as_str()
            .ok_or("Message role missing")?
            .to_owned();
        if !["system", "developer", "user", "assistant", "tool"].contains(&role.as_str()) {
            return Err("Unsupported message role".into());
        }
        if role == "system" || role == "developer" {
            if dialogue {
                return Err("System/developer messages must precede the dialogue".into());
            }
            message["role"] = json!("system");
        } else {
            dialogue = true;
        }
        if role == "user" {
            user = true;
        }
        if role != "tool" && !pending.is_empty() {
            return Err("Missing response for preceding tool call".into());
        }
        message["content"] = json!(text_content(&message["content"])?);
        if role == "tool" {
            let id = message["tool_call_id"]
                .as_str()
                .ok_or("Tool response needs tool_call_id")?;
            if !pending.remove(id) {
                return Err("Unknown or duplicate tool_call_id".into());
            }
        }
        if let Some(calls) = message.get_mut("tool_calls") {
            if role != "assistant" {
                return Err("Only assistant messages can contain tool_calls".into());
            }
            for call in calls.as_array_mut().ok_or("tool_calls must be an array")? {
                let id = call["id"].as_str().ok_or("Tool call needs id")?.to_owned();
                if id.is_empty() || !all_ids.insert(id.clone()) {
                    return Err("Duplicate/empty tool call id".into());
                }
                if call["type"] != "function"
                    || !call["function"]["name"].as_str().is_some_and(valid_name)
                {
                    return Err("Invalid assistant tool call".into());
                }
                let arguments = call["function"]["arguments"]
                    .as_str()
                    .ok_or("Tool arguments must be a JSON string")?;
                let parsed: Value = serde_json::from_str(arguments)
                    .map_err(|e| format!("Invalid tool arguments: {e}"))?;
                if !parsed.is_object() {
                    return Err("Tool arguments must encode an object".into());
                }
                call["function"]["arguments"] = parsed;
                pending.insert(id);
            }
        }
    }
    if !pending.is_empty() || !user {
        return Err("Need a user query and responses to all tool calls".into());
    }
    let mut normalized: Vec<Value> = vec![];
    for message in messages {
        if message["role"] == "system" && normalized.first().is_some_and(|m| m["role"] == "system")
        {
            let current = normalized[0]["content"].as_str().unwrap_or("");
            normalized[0]["content"] = json!(format!(
                "{current}\n\n{}",
                message["content"].as_str().unwrap_or("")
            ));
        } else {
            normalized.push(message);
        }
    }
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn history_and_multimodal_validation() {
        let history = vec![
            json!({"role":"user","content":"read"}),
            json!({"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"x\"}"}}]}),
            json!({"role":"tool","tool_call_id":"a","content":"data"}),
        ];
        let normalized = normalize_messages(history.clone()).unwrap();
        assert_eq!(
            normalized[1]["tool_calls"][0]["function"]["arguments"]["path"],
            "x"
        );
        assert!(normalize_messages(history[..2].to_vec()).is_err());
        assert!(
            normalize_messages(vec![
                json!({"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]})
            ])
            .is_err()
        );
    }
    #[test]
    fn leading_instruction_messages_are_combined_in_order() {
        let normalized = normalize_messages(vec![
            json!({"role":"system","content":"one"}),
            json!({"role":"developer","content":"two"}),
            json!({"role":"user","content":"query"}),
        ])
        .unwrap();
        assert_eq!(normalized.len(), 2);
        assert_eq!(normalized[0]["content"], "one\n\ntwo");
        assert!(
            normalize_messages(vec![
                json!({"role":"user","content":"q"}),
                json!({"role":"system","content":"late"})
            ])
            .is_err()
        );
    }

    #[test]
    #[ignore = "Needs local checkpoint and Transformers-generated reference cases"]
    fn checkpoint_matches_reference_tokens() {
        let directory = std::env::var("ORIN_TOKENIZER_DIR").expect("ORIN_TOKENIZER_DIR");
        let cases = std::env::var("ORIN_CHAT_REFERENCE").expect("ORIN_CHAT_REFERENCE");
        let codec = ChatCodec::load(Path::new(&directory)).unwrap();
        let cases: Value = serde_json::from_slice(&std::fs::read(cases).unwrap()).unwrap();
        for case in cases.as_array().unwrap() {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let vision: Option<orin_engine::vision::VisionSpec> = case
                .get("vision")
                .map(|v| serde_json::from_value(v.clone()).unwrap());
            let prepared = codec
                .prepare(request, "qwen3.8-27b", 8704, vision.as_ref())
                .unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(
                prepared.input, expected,
                "Checkpoint template mismatch: {}",
                case["request"]["messages"]
            );
            if let Some(expected) = case.get("mrope_positions") {
                let expected: Vec<u32> = serde_json::from_value(expected.clone()).unwrap();
                let (_, positions) = vision
                    .as_ref()
                    .unwrap()
                    .layout(&prepared.input, &prepared.images, prepared.input.len() + 16)
                    .unwrap();
                assert_eq!(
                    positions, expected,
                    "Official multimodal positions mismatch"
                );
            }
        }
    }
}
