use minijinja::{Environment, context, value::Serde};
use orinfer_engine::sampling::Options;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::Path};
use tokenizers::Tokenizer;
mod deepseek;
#[cfg(test)]
mod defaults_test;
mod generation_defaults;
mod template_options;
use super::output::ToolFormat;
pub(super) use template_options::{BUDGET_FIELDS, THINKING_FIELDS};
#[cfg(test)]
mod tool_generation_test;

fn nullable_default<'de, D, T>(deserializer: D) -> std::result::Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

type Result<T> = std::result::Result<T, String>;

fn normalize_top_k(value: i64) -> Result<usize> {
    match value {
        -1 | 0 => Ok(0),
        1..=2_147_483_647 => Ok(value as usize),
        _ => Err("top_k: Expected -1 or 0 (disabled), or 1..=2147483647".into()),
    }
}

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
            let model = request.model.clone();
            let prepared = codec
                .prepare(
                    request,
                    &model,
                    descriptor["metadata"]["max_context"].as_u64().unwrap() as usize,
                    vision.as_ref(),
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .unwrap();
            cases.push(json!({"id":case["id"],"input_tokens":prepared.input,
                "max_new_tokens":prepared.max_tokens,"sampling":prepared.sampling,
                "images":prepared.images,"audios":prepared.audios,"stop_after":case["stop_after"]}));
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
    pub top_k: Option<i64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub ignore_eos: bool,
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
    #[serde(default, deserialize_with = "nullable_default")]
    pub chat_template_kwargs: serde_json::Map<String, Value>,
    pub asr_language: Option<String>,
    #[serde(default)]
    pub image_processing: super::image::ImageProcessing,
    pub reasoning_effort: Option<String>,
    pub thinking_token_budget: Option<usize>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<String>,
    pub user: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<std::collections::BTreeMap<String, String>>,
    pub safety_identifier: Option<String>,
    pub service_tier: Option<String>,
}

pub struct Prepared {
    pub tool_format: ToolFormat,
    pub input: Vec<u32>,
    pub prefix_hints: Vec<usize>,
    pub images: Vec<orinfer_engine::vision::ImageInput>,
    pub audios: Vec<orinfer_engine::audio::AudioInput>,
    pub max_tokens: usize,
    pub ignore_eos: bool,
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
    pub boundary: Option<super::token_boundary::Boundary>,
    pub tool_format: ToolFormat,
    pub tools_supported: bool,
    pub audio: Option<orinfer_engine::audio::AudioSpec>,
    pub tokenizer: Tokenizer,
    pub asset_hashes: std::collections::BTreeMap<String, String>,
    template: Environment<'static>,
    pub eos: BTreeSet<u32>,
    defaults: generation_defaults::Defaults,
    pub grammar: super::grammar::Factory,
}
impl ChatCodec {
    pub fn load(directory: &Path) -> Result<Self> {
        let mut assets = std::collections::BTreeMap::new();
        // These are deployment identities, verified by validate-model before serving.
        // Standalone tokenizer fixtures need no deployment descriptor.
        let asset_hashes = match std::fs::read(directory.join("cache/model.json")) {
            Ok(raw) => {
                let descriptor: Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
                serde_json::from_value(descriptor["frontend_assets"].clone())
                    .map_err(|e| e.to_string())?
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Default::default(),
            Err(e) => return Err(e.to_string()),
        };
        for name in [
            "tokenizer.json",
            "chat_template.jinja",
            "generation_config.json",
        ] {
            let bytes = std::fs::read(directory.join(name)).map_err(|e| e.to_string())?;
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
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Transformers uses Unicode-preserving JSON with Python separators, not HTML escaping.
        environment.add_filter("tojson", template_json);
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        let grammar = super::grammar::Factory::new(
            assets.remove("tokenizer.json").unwrap(),
            eos.iter().copied().collect(),
        );
        let descriptor: Value = match std::fs::read(directory.join("cache/model.json")) {
            Ok(raw) => serde_json::from_slice(&raw).map_err(|e| e.to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(e) => return Err(e.to_string()),
        };
        let audio: Option<orinfer_engine::audio::AudioSpec> = descriptor["metadata"]
            .get("audio")
            .filter(|v| !v.is_null())
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| e.to_string())?;
        let tool_format = match descriptor["metadata"]["chat_format"].as_str() {
            Some("deepseek_v41") => ToolFormat::DeepSeekV41,
            Some(other) => return Err(format!("Unsupported chat format: {other}")),
            None => ToolFormat::Qwen,
        };
        let boundary = (tool_format == ToolFormat::DeepSeekV41)
            .then(|| super::token_boundary::Boundary::new(&tokenizer))
            .transpose()?;
        let template = template
            .replace("{% generation %}", "")
            .replace("{% endgeneration %}", "");
        environment
            .add_template_owned(
                "chat",
                boundary
                    .as_ref()
                    .map_or_else(|| template.clone(), |b| b.template(&template)),
            )
            .map_err(|e| e.to_string())?;
        if tool_format == ToolFormat::DeepSeekV41 {
            // Serving prepares the frontend alongside model loading. Share one
            // trie instead of building it on the first constrained request.
            grammar.prepare()?;
        }
        let defaults = generation_defaults::Defaults::load(directory, &descriptor, &generation)?;
        Ok(Self {
            boundary,
            tool_format,
            tools_supported: tool_format == ToolFormat::DeepSeekV41
                || descriptor["metadata"]["target"] != "ascend910b",
            audio,
            defaults,
            tokenizer,
            asset_hashes,
            template: environment,
            eos,
            grammar,
        })
    }
    pub fn default_max_tokens(&self) -> usize {
        self.defaults.max_tokens
    }

    pub fn prepare(
        &self,
        mut request: ChatRequest,
        model: &str,
        context_limit: usize,
        vision: Option<&orinfer_engine::vision::VisionSpec>,
        preparation: &mut super::preparation::Context,
    ) -> Result<Prepared> {
        let normalize_started = std::time::Instant::now();
        preparation.checkpoint(0)?;
        request.resolve_template_options()?;
        let preserve_thinking_override = request.template_bool("preserve_thinking")?;
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
            .unwrap_or(self.default_max_tokens());
        if max_tokens == 0 {
            return Err("max_completion_tokens: Output-token limit must be positive".into());
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
                .map_or(self.defaults.thinking, |s| s != "none"),
        );
        let sampling_defaults = self.defaults.sampling(thinking);
        let sampling = Options {
            temperature: request.temperature.unwrap_or(sampling_defaults.temperature),
            top_p: request.top_p.unwrap_or(sampling_defaults.top_p),
            top_k: request
                .top_k
                .map(normalize_top_k)
                .transpose()?
                .unwrap_or(sampling_defaults.top_k),
            repetition_penalty: request
                .repetition_penalty
                .unwrap_or(sampling_defaults.repetition_penalty),
            presence_penalty: request
                .presence_penalty
                .unwrap_or(sampling_defaults.presence_penalty),
            frequency_penalty: request
                .frequency_penalty
                .unwrap_or(sampling_defaults.frequency_penalty),
            // Resolve once into Prepared; target, draft, retry and prefix
            // restoration retain this request's sampling identity.
            seed: match request.seed {
                Some(seed) => seed as u64,
                None => {
                    let mut bytes = [0; 8];
                    getrandom::fill(&mut bytes)
                        .map_err(|e| format!("Generate sampling seed: {e}"))?;
                    u64::from_ne_bytes(bytes)
                }
            },
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
            if !function["parameters"].is_null() {
                super::grammar::validate_schema(
                    &function["parameters"],
                    &format!("tools[{index}].function.parameters"),
                )?;
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
                if self.tool_format == ToolFormat::DeepSeekV41 {
                    if !request.tools.iter().any(|v| v["function"]["name"] == name) {
                        return Err("Forced tool was not declared".into());
                    }
                    request.tools
                } else {
                    vec![
                        request
                            .tools
                            .into_iter()
                            .find(|v| v["function"]["name"] == name)
                            .ok_or("Forced tool was not declared")?,
                    ]
                }
            }
            _ => return Err("Invalid tool_choice".into()),
        };
        if self.tool_format == ToolFormat::DeepSeekV41 {
            deepseek::text_parts(&mut request.messages)?;
        }
        let (messages_with_audio, audios) =
            super::audio::messages(request.messages, self.audio.as_ref(), preparation)?;
        let (raw_messages, images) = super::image::messages_with_options(
            messages_with_audio,
            vision,
            &request.image_processing,
            preparation,
        )?;
        let mut messages = normalize_messages_for(raw_messages, self.tool_format)?;
        if self.audio.is_some() {
            for message in &mut messages {
                if message["role"] == "user"
                    && message["content"]
                        .as_str()
                        .is_some_and(|s| s.contains("<|audio_pad|>"))
                {
                    message["content"] = json!([{"type":"audio"}]);
                }
            }
        }
        let parallel = request.parallel_tool_calls.unwrap_or(true);
        let (grammar, native_tools) = super::grammar::recipe_for(
            self.tool_format,
            schema.as_ref(),
            &tools,
            &choice,
            parallel,
            thinking,
        )?;
        if grammar.is_some() && !stops.is_empty() {
            return Err("stop: Cannot interrupt constrained JSON or tool output".into());
        }
        let mut constraint = grammar
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
        if constraint.is_some() && !native_tools {
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
        if self.tool_format == ToolFormat::Qwen {
            for tool in &mut tools {
                canonical(tool);
            }
        }
        if self.tool_format != ToolFormat::DeepSeekV41
            && (choice == "required" || choice.is_object())
        {
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
        let mut render_tools = tools.clone();
        if let Some(boundary) = &self.boundary {
            for message in &mut messages {
                boundary.escape_value(message);
            }
            for tool in &mut render_tools {
                boundary.escape_value(tool);
            }
            for value in request.chat_template_kwargs.values_mut() {
                boundary.escape_value(value);
            }
            if !images.is_empty() {
                let marker = boundary.control("<｜deepseek_image｜>")?;
                for message in &mut messages {
                    if let Some(text) = message["content"].as_str() {
                        message["content"] = json!(text.replace("<｜deepseek_image｜>", &marker));
                    }
                }
            }
        }
        let (messages, tool_schemas) = if self.tool_format == ToolFormat::DeepSeekV41 {
            deepseek::prepare(messages, &render_tools, self.boundary.as_ref())?
        } else {
            (messages, Vec::new())
        };
        let deepseek_effort = match request.reasoning_effort.as_deref() {
            Some("minimal" | "low") => 50,
            Some("max" | "xhigh") => 100,
            _ => 75,
        };
        let preserve_thinking = preserve_thinking_override
            .unwrap_or(self.tool_format != ToolFormat::DeepSeekV41 || !tools.is_empty());
        let controlled = context! {
            messages => Serde(&messages), tools => Serde(&render_tools), add_generation_prompt => true,
            enable_thinking => thinking, thinking => thinking, reasoning_effort => effort,
            preserve_thinking => preserve_thinking,
            tool_schemas => Serde(&tool_schemas), deepseek_effort => deepseek_effort,
            canonical_tools => !native_tools && constraint.is_some() && !tools.is_empty(),
        };
        preparation.timing.normalize_s += normalize_started.elapsed().as_secs_f64();
        let render_started = std::time::Instant::now();
        let mut rendered = self
            .template
            .get_template("chat")
            .map_err(|e| e.to_string())?
            .render(context! { ..controlled, ..Serde(&request.chat_template_kwargs) })
            .map_err(|e| format!("Chat template: {e}"))?;
        preparation.checkpoint(0)?;
        if let Some(language) = request.asr_language {
            if self.audio.is_none() {
                return Err("asr_language requires an ASR model".into());
            }
            rendered.push_str(&format!(
                "language {}<asr_text>",
                super::audio::language(&language)?
            ));
        }
        preparation.timing.render_s += render_started.elapsed().as_secs_f64();
        let tokenize_started = std::time::Instant::now();
        let input = if let Some(boundary) = &self.boundary {
            boundary.encode(&rendered)?
        } else {
            self.tokenizer
                .encode(rendered, false)
                .map_err(|e| e.to_string())?
                .get_ids()
                .to_vec()
        };
        preparation.timing.tokenize_s += tokenize_started.elapsed().as_secs_f64();
        let input = if let Some(v) = vision {
            v.expand(&input, &images, context_limit)?
        } else {
            input
        };
        let input = if let Some(audio) = &self.audio {
            audio.expand(&input, &audios, context_limit)?
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
                "max_completion_tokens: Request exceeds the context window of {context_limit} tokens: {} prompt tokens + {max_tokens} output tokens",
                input.len()
            ));
        }
        if let Some(key) = &request.prompt_cache_key
            && (key.is_empty() || key.chars().count() > 64)
        {
            return Err("prompt_cache_key: Expected 1–64 characters".into());
        }
        if request
            .prompt_cache_retention
            .as_deref()
            .is_some_and(|v| !["in-memory", "24h"].contains(&v))
        {
            return Err("prompt_cache_retention: Expected in-memory or 24h".into());
        }
        // One local worker needs no affinity routing. Exact token/media identity
        // still governs reuse; retention hints cannot reserve memory or prevent
        // eviction. Neither field changes the prompt or sampling identity.
        let constrained_tools = constraint.is_some();
        if thinking && let Some(budget) = request.thinking_token_budget {
            if max_tokens < 2 {
                return Err("max_completion_tokens: Thinking budget needs room for </think> and an answer token".into());
            }
            let close = self
                .tokenizer
                .token_to_id("</think>")
                .ok_or("thinking_token_budget: Tokenizer has no </think> token")?;
            constraint = Some(Box::new(
                super::thinking::Budget::new(
                    constraint,
                    budget.min(max_tokens - 2),
                    close,
                    self.grammar.token_table()?,
                    self.eos.iter().copied().collect(),
                )
                .with_token_boundaries(self.boundary.is_some()),
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
            tool_format: if self.tool_format == ToolFormat::DeepSeekV41
                && !native_tools
                && constraint.is_some()
                && !tools.is_empty()
            {
                ToolFormat::Qwen
            } else {
                self.tool_format
            },
            input,
            prefix_hints,
            images,
            audios,
            max_tokens,
            ignore_eos: request.ignore_eos,
            sampling,
            tools,
            tool_choice: choice,
            parallel,
            stops,
            thinking,
            structured: schema.is_some(),
            constrained_tools,
            stream: request.stream,
            include_usage,
            constraint,
        })
    }
}
struct PythonJsonFormatter;
impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}
fn template_json(
    value: minijinja::Value,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let serialize = || -> std::result::Result<String, serde_json::Error> {
        let mut value = serde_json::to_value(&value)?;
        canonical(&mut value);
        let mut bytes = Vec::new();
        let mut serializer =
            serde_json::Serializer::with_formatter(&mut bytes, PythonJsonFormatter);
        serde::Serialize::serialize(&value, &mut serializer)?;
        // serde_json always writes UTF-8, including unescaped Unicode.
        Ok(String::from_utf8(bytes).expect("JSON serializer produces UTF-8"))
    };
    serialize()
        .map(minijinja::Value::from_safe_string)
        .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string()))
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
fn text_content(v: &Value, separator: &str) -> Result<String> {
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
            .map(|v| v.join(separator)),
        _ => Err("Invalid message content".into()),
    }
}
#[cfg(test)]
fn normalize_messages(messages: Vec<Value>) -> Result<Vec<Value>> {
    normalize_messages_for(messages, ToolFormat::Qwen)
}
fn normalize_messages_for(mut messages: Vec<Value>, format: ToolFormat) -> Result<Vec<Value>> {
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
            if dialogue && format != ToolFormat::DeepSeekV41 {
                return Err("System/developer messages must precede the dialogue".into());
            }
            if format != ToolFormat::DeepSeekV41 {
                message["role"] = json!("system");
            }
        } else {
            dialogue = true;
        }
        if role == "user" {
            user = true;
        }
        if role != "tool" && !pending.is_empty() {
            return Err("Missing response for preceding tool call".into());
        }
        message["content"] = json!(text_content(
            &message["content"],
            if format == ToolFormat::DeepSeekV41 {
                "\n\n"
            } else {
                ""
            }
        )?);
        if format == ToolFormat::DeepSeekV41
            && message.get("reasoning_content").is_none()
            && let Some(reasoning) = message.get("reasoning").cloned()
        {
            message["reasoning_content"] = reasoning;
        }
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
                // Preserve failed/truncated calls for client error recovery. The
                // native template requires a parameter map, so retain raw input
                // under an explicit marker instead of guessing missing values.
                let raw = serde_json::from_str::<Value>(arguments);
                let raw = if format == ToolFormat::DeepSeekV41 {
                    raw.and_then(|v| {
                        if let Some(s) = v.as_str() {
                            serde_json::from_str(s)
                        } else {
                            Ok(v)
                        }
                    })
                } else {
                    raw
                };
                let parsed = match raw {
                    Ok(value) if value.is_object() => value,
                    _ if format == ToolFormat::DeepSeekV41 => json!({"arguments":arguments}),
                    _ => json!({"__orinfer_raw_arguments":arguments}),
                };
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
        if format != ToolFormat::DeepSeekV41
            && message["role"] == "system"
            && normalized.first().is_some_and(|m| m["role"] == "system")
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
    fn ollama_tool_history_vectors_preserve_text_and_call_identity() {
        for case in crate::test_fixtures::ollama()["requests"]
            .as_array()
            .unwrap()
        {
            let request: ChatRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let messages = normalize_messages(request.messages).unwrap();
            assert_eq!(json!(messages), case["expected_messages"], "{}", case["id"]);
        }
    }
    #[test]
    fn template_json_matches_python_separators_unicode_and_string_escaping() {
        let value = json!({"z":["世界", "a,b:c", "\"\\", "<>&'"],"a":{"b":true},"empty":[]});
        let result = template_json(minijinja::Value::from(Serde(&value))).unwrap();
        assert_eq!(
            result.as_str().unwrap(),
            r#"{"a": {"b": true}, "empty": [], "z": ["世界", "a,b:c", "\"\\", "<>&'"]}"#
        );
    }
    #[test]
    fn template_python_methods_support_real_python_argument_shapes() {
        let mut environment = Environment::new();
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        assert_eq!(environment.render_str("{{ 'ABC'.lower() }} {{ 'abcdef'.startswith(['xy', 'ab']) }} {{ 'abc'.endswith('bc') }} {{ ' a '.strip() }}", minijinja::context!()).unwrap(), "abc True True a");
    }
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
                "max_tokens":1,"seed":20261002
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
        for retention in ["in-memory", "24h"] {
            let mut cached = request();
            cached.prompt_cache_key = Some("session-key".into());
            cached.prompt_cache_retention = Some(retention.into());
            let result = codec
                .prepare(
                    cached,
                    "test",
                    4,
                    None,
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .unwrap();
            assert_eq!(result.input, prepared.input);
            assert_eq!(result.sampling.seed, prepared.sampling.seed);
        }
        for (key, retention, param) in [
            ("", "24h", "prompt_cache_key"),
            ("key", "forever", "prompt_cache_retention"),
        ] {
            let mut invalid = request();
            invalid.prompt_cache_key = Some(key.into());
            invalid.prompt_cache_retention = Some(retention.into());
            let error = codec
                .prepare(
                    invalid,
                    "test",
                    4,
                    None,
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .err()
                .unwrap();
            assert!(error.starts_with(param), "{error}");
        }
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
        for raw in ["{\"path\":", "[]", "null"] {
            let mut failed = history.clone();
            failed[1]["tool_calls"][0]["function"]["arguments"] = json!(raw);
            failed[2]["content"] = json!("Invalid arguments; retry with path.");
            let result = normalize_messages(failed).unwrap();
            assert_eq!(
                result[1]["tool_calls"][0]["function"]["arguments"]["__orinfer_raw_arguments"],
                raw
            );
        }
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
            let model = request.model.clone();
            let prepared = codec
                .prepare(
                    request,
                    &model,
                    case["context_limit"].as_u64().unwrap_or(8704) as usize,
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
