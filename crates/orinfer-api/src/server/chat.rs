use minijinja::{Environment, context};
use orinfer_engine::sampling::Options;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
use tokenizers::Tokenizer;

fn nullable_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

type Result<T> = std::result::Result<T, String>;

#[cfg(test)]
mod mtp_fixture_export {
    use super::*;
    #[test]
    #[ignore = "Offline export of native chat-template code fixtures for the GPU MTP test"]
    fn export_mtp_chat_fixture() {
        let source = std::env::var("ORINFER_MTP_FIXTURE_SPEC").unwrap();
        let raw: Value = serde_json::from_slice(&std::fs::read(source).unwrap()).unwrap();
        let codec = ChatCodec::load(Path::new(raw["tokenizer"].as_str().unwrap())).unwrap();
        let descriptor: Value = serde_json::from_slice(
            &std::fs::read(Path::new(raw["model"].as_str().unwrap()).join("cache/model.json"))
                .unwrap(),
        )
        .unwrap();
        let vision: Option<orinfer_engine::vision::VisionSpec> =
            serde_json::from_value(descriptor["metadata"]["vision"].clone()).unwrap();
        let mut cases = vec![];
        for case in raw["cases"].as_array().unwrap() {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let prepared = codec
                .prepare(
                    request,
                    "qwen3.8-27b",
                    descriptor["metadata"]["max_context"].as_u64().unwrap() as usize,
                    vision.as_ref(),
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .unwrap();
            cases.push(json!({"id":case["id"],"input_tokens":prepared.input,
                "max_new_tokens":prepared.max_tokens,"sampling":prepared.sampling,
                "images":prepared.images,"stop_after":case["stop_after"]}));
        }
        let fixture = json!({"model":raw["model"],"output":raw["result"],
            "cases":cases,"eos":codec.eos,"repetitions":raw["repetitions"],
            "cuda_graph":raw.get("cuda_graph").cloned().unwrap_or(json!("decode_only"))});
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
    #[serde(default, deserialize_with = "nullable_default")]
    pub tools: Vec<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default, deserialize_with = "nullable_default")]
    pub stream: bool,
    pub stream_options: Option<Value>,
    pub stop: Option<Value>,
    pub n: Option<usize>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub logit_bias: Option<std::collections::BTreeMap<u32, f64>>,
    pub response_format: Option<Value>,
    pub enable_thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub user: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<std::collections::BTreeMap<String, String>>,
    pub safety_identifier: Option<String>,
    pub service_tier: Option<String>,
}

pub struct Prepared {
    pub input: Vec<u32>,
    pub prefix_hints: Vec<usize>,
    pub images: Vec<orinfer_engine::vision::ImageInput>,
    pub max_tokens: usize,
    pub sampling: Options,
    pub tools: Vec<Value>,
    pub tool_choice: Value,
    pub parallel: bool,
    pub stops: Vec<String>,
    pub thinking: bool,
    pub structured: bool,
    pub constrained_tools: bool,
    pub stream: bool,
    pub include_usage: bool,
    pub constraint: Option<Box<dyn orinfer_engine::sampling::Constraint>>,
}

pub struct ChatCodec {
    pub tokenizer: Tokenizer,
    pub asset_hashes: std::collections::BTreeMap<String, String>,
    template: Environment<'static>,
    pub eos: BTreeSet<u32>,
    sampling_defaults: Options,
    pub grammar: super::grammar::Factory,
}
impl ChatCodec {
    pub fn load(directory: &Path) -> Result<Self> {
        let mut assets = std::collections::BTreeMap::new();
        let mut asset_hashes = std::collections::BTreeMap::new();
        for name in [
            "tokenizer.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            let bytes = std::fs::read(directory.join(name)).map_err(|e| e.to_string())?;
            asset_hashes.insert(name.to_owned(), orinfer_engine::artifact::sha256(&bytes));
            assets.insert(name, bytes);
        }
        let mut tokenizer =
            Tokenizer::from_bytes(&assets["tokenizer.json"]).map_err(|e| e.to_string())?;
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        let template = String::from_utf8(assets.remove("chat_template.jinja").unwrap())
            .map_err(|e| e.to_string())?;
        let generation: Value =
            serde_json::from_slice(&assets["generation_config.json"]).map_err(|e| e.to_string())?;
        let ids = &generation["eos_token_id"];
        let sampling_defaults = Options {
            temperature: generation["temperature"].as_f64().unwrap_or(1.0),
            top_p: generation["top_p"].as_f64().unwrap_or(1.0),
            top_k: generation["top_k"]
                .as_u64()
                .map(|n| usize::try_from(n).map_err(|e| e.to_string()))
                .transpose()?
                .unwrap_or(0),
            repetition_penalty: generation["repetition_penalty"].as_f64().unwrap_or(1.0),
            ..Options::default()
        };
        sampling_defaults.validate()?;
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
                let mut value = serde_json::to_value(&value).map_err(|e| {
                    minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
                })?;
                canonical(&mut value);
                let raw = serde_json::to_string(&value).unwrap();
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
        let grammar = super::grammar::Factory::new(
            assets.remove("tokenizer.json").unwrap(),
            eos.iter().copied().collect(),
        );
        Ok(Self {
            sampling_defaults,
            tokenizer,
            asset_hashes,
            template: environment,
            eos,
            grammar,
        })
    }
    pub fn prepare(
        &self,
        request: ChatRequest,
        model: &str,
        context_limit: usize,
        vision: Option<&orinfer_engine::vision::VisionSpec>,
        preparation: &mut super::preparation::Context,
    ) -> Result<Prepared> {
        preparation.checkpoint(0)?;
        if request.model != model {
            return Err(format!(
                "model: Unknown model {}; expected {model}",
                request.model
            ));
        }
        if request.n.is_some_and(|n| n != 1) {
            return Err("n: Only n=1 is supported".into());
        }
        if request.top_logprobs.is_some_and(|n| n > 20) {
            return Err("top_logprobs: Expected 0–20".into());
        }
        if request.top_logprobs.is_some() && request.logprobs != Some(true) {
            return Err("top_logprobs: Requires logprobs=true".into());
        }
        if request.store == Some(true) {
            return Err("store: Stored completions are unsupported; use false".into());
        }
        if request
            .service_tier
            .as_deref()
            .is_some_and(|t| !["auto", "default"].contains(&t))
        {
            return Err("service_tier: Only auto/default are available locally".into());
        }
        if request.metadata.as_ref().is_some_and(|m| {
            m.len() > 16
                || m.iter()
                    .any(|(k, v)| k.chars().count() > 64 || v.chars().count() > 512)
        }) {
            return Err(
                "metadata: At most 16 entries; keys <=64 and values <=512 characters".into(),
            );
        }
        if request
            .safety_identifier
            .as_ref()
            .is_some_and(|s| s.chars().count() > 128)
        {
            return Err("safety_identifier: At most 128 characters".into());
        }
        if let Some(options) = &request.stream_options {
            let options = options
                .as_object()
                .ok_or("stream_options: Expected object")?;
            if !request.stream {
                return Err("stream_options: Requires stream=true".into());
            }
            for (name, value) in options {
                if !["include_usage", "include_obfuscation"].contains(&name.as_str()) {
                    return Err(format!("stream_options.{name}: Unsupported option"));
                }
                if !value.is_boolean() {
                    return Err(format!("stream_options.{name}: Expected boolean"));
                }
                if name == "include_obfuscation" && value == true {
                    return Err(
                        "stream_options.include_obfuscation: Only false is supported".into(),
                    );
                }
            }
        }
        let schema = super::grammar::output_schema(request.response_format.as_ref())?;
        if request.max_tokens.is_some() && request.max_completion_tokens.is_some() {
            return Err("max_completion_tokens: Specify only one output-token limit".into());
        }
        let explicit_limit =
            request.max_completion_tokens.is_some() || request.max_tokens.is_some();
        let mut max_tokens = request
            .max_completion_tokens
            .or(request.max_tokens)
            .unwrap_or(8192);
        if max_tokens == 0 {
            return Err("max_completion_tokens: Output-token limit must be positive".into());
        }
        let sampling = Options {
            temperature: request
                .temperature
                .unwrap_or(self.sampling_defaults.temperature),
            top_p: request.top_p.unwrap_or(self.sampling_defaults.top_p),
            top_k: request.top_k.unwrap_or(self.sampling_defaults.top_k),
            repetition_penalty: request
                .repetition_penalty
                .unwrap_or(self.sampling_defaults.repetition_penalty),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            seed: request
                .seed
                .map(|s| s as u64)
                .unwrap_or(orinfer_engine::sampling::EVALUATION_SEED),
            logit_bias: request.logit_bias.unwrap_or_default(),
            top_logprobs: request
                .logprobs
                .unwrap_or(false)
                .then_some(request.top_logprobs.unwrap_or(0)),
        };
        for (&id, bias) in &sampling.logit_bias {
            if self.tokenizer.id_to_token(id).is_none()
                || !bias.is_finite()
                || !(-100.0..=100.0).contains(bias)
            {
                return Err(format!(
                    "logit_bias.{id}: Invalid token or bias outside -100..100"
                ));
            }
        }
        for (name, value, min, max) in [
            ("temperature", sampling.temperature, 0.0, 2.0),
            ("top_p", sampling.top_p, 0.0, 1.0),
            ("presence_penalty", sampling.presence_penalty, -2.0, 2.0),
            ("frequency_penalty", sampling.frequency_penalty, -2.0, 2.0),
        ] {
            if !value.is_finite() || value < min || value > max || name == "top_p" && value == 0.0 {
                return Err(format!("{name}: Invalid sampling value"));
            }
        }
        if !sampling.repetition_penalty.is_finite() || sampling.repetition_penalty <= 0.0 {
            return Err("repetition_penalty: Expected positive finite value".into());
        }
        sampling.validate()?;
        if sampling.top_logprobs.is_some() {
            self.grammar
                .token_bytes(0)
                .map_err(|e| format!("logprobs: {e}"))?;
        }
        let effort = match request.reasoning_effort.as_deref() {
            None | Some("high" | "xhigh" | "max") => "xhigh",
            Some("minimal" | "low") => "low",
            Some("medium") => "medium",
            Some("none") => "xhigh",
            _ => {
                return Err(
                    "reasoning_effort: Expected none/minimal/low/medium/high/xhigh/max".into(),
                );
            }
        };
        if request.reasoning_effort.as_deref() == Some("none")
            && request.enable_thinking == Some(true)
        {
            return Err("enable_thinking: Conflicts with reasoning_effort=none".into());
        }
        let thinking = request.enable_thinking.unwrap_or(
            request
                .reasoning_effort
                .as_ref()
                .is_some_and(|s| s != "none"),
        );
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
        if request.tools.len() > 128 {
            return Err("tools: At most 128 functions".into());
        }
        for (index, tool) in request.tools.iter().enumerate() {
            let function = &tool["function"];
            let name = function["name"].as_str().ok_or("Tool name missing")?;
            if tool["type"] != "function" || !valid_name(name) || !names.insert(name.to_owned()) {
                return Err("Tools need unique, valid function names".into());
            }
            if !function["parameters"].is_null() && !function["parameters"].is_object() {
                return Err(format!(
                    "tools[{index}].function.parameters: Expected JSON schema object"
                ));
            }
            if function
                .get("strict")
                .is_some_and(|s| !s.is_null() && !s.is_boolean())
            {
                return Err(format!("tools[{index}].function.strict: Expected boolean"));
            }
        }
        let choice = request.tool_choice.unwrap_or(if request.tools.is_empty() {
            json!("none")
        } else {
            json!("auto")
        });
        let mut tools = match choice.as_str() {
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
        let (raw_messages, images) = super::image::messages(request.messages, vision, preparation)?;
        let mut messages = normalize_messages(raw_messages)?;
        let parallel = request.parallel_tool_calls.unwrap_or(true);
        let grammar = super::grammar::recipe(schema.as_ref(), &tools, &choice, parallel, thinking)?;
        if grammar.is_some() && !stops.is_empty() {
            return Err("stop: Cannot interrupt constrained JSON or tool output".into());
        }
        let constraint = grammar
            .map(|g| {
                self.grammar.compile(
                    g,
                    if schema.is_some() {
                        "response_format"
                    } else {
                        "tools"
                    },
                )
            })
            .transpose()?;
        if constraint.is_some() {
            let instruction = if tools.is_empty() {
                format!(
                    "Return only JSON conforming to this schema: {}",
                    schema.as_ref().unwrap()
                )
            } else {
                "When calling a tool, output <tool_call>{\"name\":\"FUNCTION_NAME\",\"arguments\":{...}}</tool_call>. Use valid JSON inside the tags.".into()
            };
            if messages[0]["role"] == "system" {
                let existing = messages[0]["content"].as_str().unwrap_or("");
                messages[0]["content"] = json!(format!("{instruction}\n\n{existing}"));
            } else {
                messages.insert(0, json!({"role":"system","content":instruction}));
            }
        }
        for tool in &mut tools {
            canonical(tool);
        }
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
        preparation.checkpoint(0)?;
        let rendered = self
            .template
            .get_template("chat")
            .map_err(|e| e.to_string())?
            .render(context! {
                messages => messages, tools => tools, add_generation_prompt => true,
                enable_thinking => thinking, reasoning_effort => effort,
                preserve_thinking => true,
            })
            .map_err(|e| format!("Chat template: {e}"))?;
        preparation.checkpoint(0)?;
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
        if !explicit_limit {
            max_tokens = max_tokens.min(context_limit.saturating_sub(input.len()));
        }
        if input.is_empty()
            || max_tokens == 0
            || input
                .len()
                .checked_add(max_tokens)
                .is_none_or(|n| n > context_limit)
        {
            return Err(format!(
                "max_completion_tokens: Context limit {context_limit}: {} prompt tokens + {max_tokens} output tokens",
                input.len()
            ));
        }
        let include_usage = request
            .stream_options
            .as_ref()
            .is_some_and(|v| v["include_usage"] == true);
        // Hint at actual template token boundaries; never alter or pad the prompt.
        let ends: Vec<_> = input
            .iter()
            .enumerate()
            .filter(|(_, id)| self.eos.contains(id))
            .map(|(i, _)| i + 1)
            .collect();
        let mut prefix_hints = vec![];
        if messages.first().is_some_and(|m| m["role"] == "system")
            && let Some(&p) = ends.first()
        {
            prefix_hints.push(p);
        }
        if ends.len() >= 2 {
            prefix_hints.push(ends[ends.len() - 2]);
        }
        prefix_hints.sort_unstable();
        prefix_hints.dedup();
        let _ = request.user; // Attribution only; it does not alter sampling identity.
        Ok(Prepared {
            input,
            prefix_hints,
            images,
            max_tokens,
            sampling,
            tools,
            tool_choice: choice,
            parallel,
            stops,
            thinking,
            structured: schema.is_some(),
            constrained_tools: constraint.is_some(),
            stream: request.stream,
            include_usage,
            constraint,
        })
    }
}
fn canonical(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.sort_keys();
            for value in map.values_mut() {
                canonical(value);
            }
        }
        Value::Array(values) => {
            for value in values {
                canonical(value);
            }
        }
        _ => {}
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
    fn nullable_optional_fields_and_signed_seed_match_the_wire_contract() {
        let request: ChatRequest=serde_json::from_value(json!({"model":"test","messages":[],"tools":null,"stream":null,"seed":-1,"store":false,"metadata":{"case":"test"},"logprobs":true,"top_logprobs":3,"logit_bias":{"42":-100}})).unwrap();
        assert!(!request.stream && request.tools.is_empty());
        assert_eq!(request.seed, Some(-1));
        assert_eq!(request.logit_bias.unwrap()[&42], -100.0);
        assert!(
            serde_json::from_value::<ChatRequest>(
                json!({"model":"test","messages":[],"stream":"true"})
            )
            .is_err()
        );
    }
    #[test]
    fn saved_calibration_truncation_does_not_discard_chat_history() {
        use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
        let directory = std::env::temp_dir().join(format!(
            "orin-chat-truncation-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let model = WordLevel::builder()
            .vocab(
                [
                    ("[UNK]".into(), 0),
                    ("one".into(), 1),
                    ("two".into(), 2),
                    ("three".into(), 3),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 2,
                ..Default::default()
            }))
            .unwrap();
        assert_eq!(tokenizer.encode("one two three", false).unwrap().len(), 2);
        tokenizer
            .save(directory.join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id":0,"temperature":0.7,"top_p":0.95,"top_k":20,"repetition_penalty":1.05}"#,
        )
        .unwrap();
        std::fs::write(
            directory.join("chat_template.jinja"),
            "{{ messages[0].content }}",
        )
        .unwrap();
        let codec = ChatCodec::load(&directory).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        let request = || {
            serde_json::from_value(json!({
                "model":"test", "messages":[{"role":"user","content":"one two three"}],
                "max_tokens":1
            }))
            .unwrap()
        };
        assert_eq!(
            codec
                .prepare(
                    request(),
                    "test",
                    4,
                    None,
                    &mut crate::server::preparation::Context::unbounded()
                )
                .unwrap()
                .input,
            [1, 2, 3]
        );
        assert!(
            codec
                .prepare(
                    request(),
                    "test",
                    3,
                    None,
                    &mut crate::server::preparation::Context::unbounded()
                )
                .is_err()
        );
        let prepared = codec
            .prepare(
                request(),
                "test",
                4,
                None,
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
        assert_eq!(prepared.sampling.temperature, 0.7);
        assert_eq!(prepared.sampling.top_p, 0.95);
        assert_eq!(prepared.sampling.top_k, 20);
        assert_eq!(prepared.sampling.repetition_penalty, 1.05);
        let override_request = serde_json::from_value(json!({
            "model":"test", "messages":[{"role":"user","content":"one"}],
            "max_tokens":1,"temperature":0,"top_k":0,"top_p":1,
            "repetition_penalty":1
        }))
        .unwrap();
        assert!(
            codec
                .prepare(
                    override_request,
                    "test",
                    4,
                    None,
                    &mut crate::server::preparation::Context::unbounded()
                )
                .unwrap()
                .sampling
                .is_greedy()
        );
    }

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
        let directory = std::env::var("ORINFER_TOKENIZER_DIR").expect("ORINFER_TOKENIZER_DIR");
        let cases = std::env::var("ORINFER_CHAT_REFERENCE").expect("ORINFER_CHAT_REFERENCE");
        let codec = ChatCodec::load(Path::new(&directory)).unwrap();
        let cases: Value = serde_json::from_slice(&std::fs::read(cases).unwrap()).unwrap();
        for case in cases.as_array().unwrap() {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let vision: Option<orinfer_engine::vision::VisionSpec> = case
                .get("vision")
                .map(|v| serde_json::from_value(v.clone()).unwrap());
            let prepared = codec
                .prepare(
                    request,
                    "qwen3.8-27b",
                    8704,
                    vision.as_ref(),
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(
                prepared.input, expected,
                "Checkpoint template mismatch: {}",
                case["request"]["messages"]
            );
            if let Some(expected) = case.get("mrope_positions") {
                let expected: Vec<u32> = serde_json::from_value(expected.clone()).unwrap();
                let (_, positions) = orinfer_engine::model::media_layout(
                    Path::new(&directory),
                    &prepared.input,
                    &prepared.images,
                    prepared.input.len() + 16,
                )
                .unwrap();
                assert_eq!(
                    positions, expected,
                    "Official multimodal positions mismatch"
                );
            }
        }
    }
}
