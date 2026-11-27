//! Incremental model tool/reasoning formats into shared Chat Completions deltas.
mod dsml;
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolFormat {
    #[default]
    Qwen,
    DeepSeekV41,
}

#[derive(Default)]
struct Schemas {
    compiled: OnceLock<Result<BTreeMap<String, jsonschema::ValidatorMap>>>,
}
impl Schemas {
    fn get(
        &self,
        tools: &[Value],
        name: &str,
        pointer: &str,
    ) -> Result<Option<&jsonschema::Validator>> {
        let compiled = self.compiled.get_or_init(|| {
            tools
                .iter()
                .map(|tool| {
                    let name = tool["function"]["name"]
                        .as_str()
                        .ok_or("Tool name missing")?;
                    let schema = &tool["function"]["parameters"];
                    let schema = if schema.is_null() {
                        &Value::Bool(true)
                    } else {
                        schema
                    };
                    let compiled = jsonschema::options()
                        .offline()
                        .build_map(schema)
                        .map_err(|e| e.to_string())?;
                    Ok((name.to_owned(), compiled))
                })
                .collect()
        });
        let compiled = compiled.as_ref().map_err(Clone::clone)?;
        Ok(compiled.get(name).and_then(|map| map.get(pointer)))
    }
    fn validate(&self, tools: &[Value], name: &str, arguments: &Value) -> Result<()> {
        self.get(tools, name, "#")?
            .ok_or("Missing tool schema")?
            .validate(arguments)
            .map_err(|e| e.to_string())
    }
}

type Result<T> = std::result::Result<T, String>;
type Controls<'a> = Option<&'a [std::ops::Range<usize>]>;

fn marker_at(controls: Controls<'_>, at: usize, marker: &str) -> bool {
    let Some(controls) = controls else {
        return true;
    };
    let range = if let Some(offset) = marker.find("｜DSML｜") {
        at + offset..at + offset + "｜DSML｜".len()
    } else if matches!(marker, "<think>" | "</think>") {
        at..at + marker.len()
    } else {
        return true;
    };
    controls
        .binary_search_by_key(&range.start, |r| r.start)
        .is_ok_and(|i| controls[i] == range)
}
fn find_marker(input: &str, controls: Controls<'_>, marker: &str) -> Option<usize> {
    input
        .match_indices(marker)
        .map(|(at, _)| at)
        .find(|&at| marker_at(controls, at, marker))
}

pub struct Output {
    format: ToolFormat,
    dsml_calls: bool,
    dsml_call: Option<dsml::Call>,
    schemas: Schemas,
    pending: String,
    controls: Option<Vec<std::ops::Range<usize>>>,
    reasoning: bool,
    in_tool: bool,
    pub stopped: bool,
    pub incomplete_call: bool,
    pub content: String,
    pub reasoning_content: String,
    pub calls: Vec<Value>,
    tools: Vec<Value>,
    stops: Vec<String>,
    call_prefix: String,
    streamed_call: Option<(String, String)>,
    cursor: usize,
    content_spans: Vec<std::ops::Range<usize>>,
    structured: bool,
    json_started: bool,
    constrained_tools: bool,
}
impl Output {
    pub fn new(thinking: bool, tools: Vec<Value>, stops: Vec<String>, id: &str) -> Self {
        Self {
            format: ToolFormat::Qwen,
            dsml_calls: false,
            dsml_call: None,
            schemas: Schemas::default(),
            pending: String::new(),
            controls: None,
            reasoning: thinking,
            in_tool: false,
            stopped: false,
            incomplete_call: false,
            content: String::new(),
            reasoning_content: String::new(),
            calls: vec![],
            tools,
            stops,
            call_prefix: id.into(),
            streamed_call: None,
            cursor: 0,
            content_spans: vec![],
            structured: false,
            json_started: false,
            constrained_tools: false,
        }
    }
    pub fn structured_json(&mut self, enabled: bool) {
        self.structured = enabled;
    }
    pub fn tool_format(&mut self, format: ToolFormat) {
        self.format = format;
    }
    pub fn constrained_tools(&mut self, enabled: bool) {
        self.constrained_tools = enabled;
    }
    pub fn token_boundaries(&mut self) {
        self.controls = Some(Vec::new());
    }
    pub fn push_token(&mut self, text: &str, special: Option<&str>) -> Result<Vec<Value>> {
        if let (Some(controls), Some(spelling)) = (&mut self.controls, special) {
            let start = text
                .strip_suffix(spelling)
                .ok_or("Decoded control token does not end at its token boundary")?
                .len();
            controls.push(self.pending.len() + start..self.pending.len() + text.len());
        }
        self.push(text, false)
    }
    pub fn push(&mut self, text: &str, final_chunk: bool) -> Result<Vec<Value>> {
        self.pending.push_str(text);
        let mut deltas = vec![];
        loop {
            if self.stopped {
                self.pending.clear();
                break;
            }
            // Once a constrained JSON body starts, all tags inside its strings
            // are data. Thinking and tool envelopes are parsed before that body.
            if self.structured
                && !self.reasoning
                && !self.in_tool
                && (self.json_started || self.pending.trim_start().starts_with(|c: char| c != '<'))
            {
                self.json_started = true;
                let text = self.pending.clone();
                self.emit_text(&text, &mut deltas);
                self.drain(text.len());
                break;
            }
            let stop = self
                .stops
                .iter()
                .filter_map(|s| self.pending.find(s).map(|at| (at, s.len())))
                .min_by_key(|x| x.0);
            if let Some(dsml_call) = self.dsml_call.as_mut() {
                let limit = stop.map_or(self.pending.len(), |(at, _)| at);
                let (consumed, event) =
                    dsml_call.advance(&self.pending[..limit], self.controls.as_deref())?;
                let waiting = event.is_none();
                self.drain(consumed);
                match event {
                    Some(dsml::Event::Begin(name)) => self.stream_call(&name, "", &mut deltas)?,
                    Some(dsml::Event::Arguments(fragment)) => {
                        self.argument_delta(&fragment, &mut deltas)?
                    }
                    Some(dsml::Event::End) => {
                        let (name, arguments) = self
                            .streamed_call
                            .as_ref()
                            .ok_or("DSML tool call ended before function name")?;
                        if self.constrained_tools {
                            let value: Value =
                                serde_json::from_str(arguments).map_err(|e| e.to_string())?;
                            self.schemas.validate(&self.tools, name, &value)?;
                        }
                        self.calls.push(json!({"id":format!("call_{}_{}",self.call_prefix,self.calls.len()),"type":"function","function":{"name":name,"arguments":arguments}}));
                        self.streamed_call = None;
                        self.dsml_call = None;
                        self.in_tool = false;
                    }
                    None => {}
                }
                if consumed == 0 && waiting {
                    if stop.is_some() {
                        self.pending.clear();
                        self.stopped = true;
                    }
                    break;
                }
                continue;
            }
            if self.in_tool {
                let end = call_end(&self.pending);
                if stop.is_some_and(|(at, _)| end.is_none_or(|end| at < end)) {
                    self.pending.clear();
                    self.stopped = true;
                    break;
                }
                if let Some((name, arguments)) =
                    partial_call(&self.pending, &self.tools, &self.schemas)?
                {
                    self.stream_call(&name, &arguments, &mut deltas)?;
                }
                if let Some(end) = end {
                    let call = parse_call(
                        &self.pending[..end],
                        &self.tools,
                        &format!("call_{}_{}", self.call_prefix, self.calls.len()),
                        self.constrained_tools,
                        &self.schemas,
                    )?;
                    self.stream_call(&call.name, &call.arguments, &mut deltas)?;
                    self.calls.push(call.into_value());
                    self.drain(end + "</tool_call>".len());
                    self.in_tool = false;
                    self.streamed_call = None;
                    continue;
                }
                break;
            }
            let markers: &[&str] = if self.format == ToolFormat::DeepSeekV41
                && !self.tools.is_empty()
                && !self.reasoning
            {
                if self.dsml_calls {
                    &[dsml::INVOKE, dsml::CLOSE]
                } else if self.constrained_tools {
                    &[
                        "\n\n<｜DSML｜ calls>",
                        dsml::OPEN,
                        "<tool_call>",
                        "<think>",
                        "</think>",
                    ]
                } else {
                    &["\n\n<｜DSML｜ calls>", dsml::OPEN, "<think>", "</think>"]
                }
            } else if self.tools.is_empty() || self.format == ToolFormat::DeepSeekV41 {
                &["<think>", "</think>"]
            } else {
                &["<tool_call>", "<think>", "</think>"]
            };
            let marker = markers
                .iter()
                .filter_map(|tag| {
                    find_marker(&self.pending, self.controls.as_deref(), tag).map(|at| (at, *tag))
                })
                .min_by_key(|x| x.0);
            if let Some((at, _)) = stop.filter(|(at, _)| marker.is_none_or(|(m, _)| *at <= m)) {
                let before = self.pending[..at].to_owned();
                self.emit_text(&before, &mut deltas);
                self.pending.clear();
                self.stopped = true;
                break;
            }
            if let Some((at, tag)) = marker {
                let before = self.pending[..at].to_owned();
                if self.dsml_calls {
                    if !before.trim().is_empty() {
                        return Err("Unexpected text inside DSML calls".into());
                    }
                } else {
                    self.emit_text(&before, &mut deltas);
                }
                self.drain(at + tag.len());
                match tag {
                    dsml::OPEN | "\n\n<｜DSML｜ calls>" => self.dsml_calls = true,
                    dsml::CLOSE => self.dsml_calls = false,
                    dsml::INVOKE => {
                        self.dsml_call = Some(dsml::Call::default());
                        self.in_tool = true;
                    }
                    "<tool_call>" if self.reasoning => {
                        return Err("Tool call inside reasoning".into());
                    }
                    "<tool_call>" => self.in_tool = true,
                    "<think>" => self.reasoning = true,
                    "</think>" => self.reasoning = false,
                    _ => unreachable!(),
                }
                continue;
            }
            // Withhold the longest suffix which might become a marker or stop.
            let hold = if final_chunk {
                0
            } else {
                markers
                    .iter()
                    .copied()
                    .chain(self.stops.iter().map(String::as_str))
                    .flat_map(|s| {
                        (1..s.len())
                            .filter(|&n| s.is_char_boundary(n))
                            .map(move |n| &s[..n])
                    })
                    .filter(|prefix| self.pending.ends_with(prefix))
                    .map(str::len)
                    .max()
                    .unwrap_or(0)
            };
            let n = self.pending.len() - hold;
            let text = self.pending[..n].to_owned();
            if self.dsml_calls {
                if !text.trim().is_empty() {
                    return Err("Unexpected text inside DSML calls".into());
                }
            } else {
                self.emit_text(&text, &mut deltas);
            }
            self.drain(n);
            break;
        }
        Ok(deltas)
    }
    fn emit_text(&mut self, text: &str, deltas: &mut Vec<Value>) {
        if text.is_empty() {
            return;
        }
        if self.reasoning {
            self.reasoning_content.push_str(text);
            deltas.push(json!({"reasoning_content":text}));
        } else {
            self.content_spans
                .push(self.cursor..self.cursor + text.len());
            self.content.push_str(text);
            deltas.push(json!({"content":text}));
        }
    }
    fn drain(&mut self, bytes: usize) {
        if let Some(controls) = &mut self.controls {
            controls.retain(|r| r.end > bytes);
            for range in controls {
                range.start = range.start.saturating_sub(bytes);
                range.end -= bytes;
            }
        }
        self.pending.drain(..bytes);
        self.cursor += bytes;
    }
    pub fn take_content_span(&mut self) -> Option<std::ops::Range<usize>> {
        if self.content_spans.is_empty() {
            None
        } else {
            Some(self.content_spans.remove(0))
        }
    }
    pub fn is_reasoning(&self) -> bool {
        self.reasoning
    }
    pub fn processed_bytes(&self) -> usize {
        self.cursor
    }
    fn stream_call(&mut self, name: &str, arguments: &str, deltas: &mut Vec<Value>) -> Result<()> {
        let index = self.calls.len();
        let previous = if let Some((previous_name, previous)) = &self.streamed_call {
            if previous_name != name || !arguments.starts_with(previous) {
                return Err("Tool argument prefix changed during streaming".into());
            }
            previous.len()
        } else {
            deltas.push(json!({"tool_calls":[{"index":index,"id":format!("call_{}_{}",self.call_prefix,index),"type":"function","function":{"name":name,"arguments":""}}]}));
            self.streamed_call = Some((name.into(), String::new()));
            0
        };
        if arguments.len() > previous {
            self.argument_delta(&arguments[previous..], deltas)?;
        }
        Ok(())
    }
    fn argument_delta(&mut self, fragment: &str, deltas: &mut Vec<Value>) -> Result<()> {
        let (_, arguments) = self
            .streamed_call
            .as_mut()
            .ok_or("Tool argument before name")?;
        arguments.push_str(fragment);
        deltas.push(
            json!({"tool_calls":[{"index":self.calls.len(),"function":{"arguments":fragment}}]}),
        );
        Ok(())
    }
    pub fn finish(
        &mut self,
        exhausted: bool,
        choice: &Value,
        parallel: bool,
    ) -> Result<Vec<Value>> {
        let deltas = self.push("", true)?;
        if self.in_tool {
            if let Some((name, arguments)) = &self.streamed_call {
                // Preserve exactly the prefix already delivered by SSE. A length
                // stop is still reported as length; clients must not execute it
                // blindly, but can replay the call with a tool error to recover.
                if exhausted || !self.constrained_tools {
                    self.calls.push(
                        json!({"id":format!("call_{}_{}",self.call_prefix,self.calls.len()),
                        "type":"function","function":{"name":name,"arguments":arguments}}),
                    );
                    self.in_tool = false;
                    self.incomplete_call = true;
                    self.streamed_call = None;
                } else {
                    return Err("Model emitted an incomplete tool call".into());
                }
            } else if !exhausted {
                return Err("Model emitted an incomplete tool call".into());
            }
        }
        if self.dsml_calls {
            if self.constrained_tools && !exhausted {
                return Err("Incomplete DSML calls envelope".into());
            }
            self.incomplete_call = true;
        }
        if !parallel && self.calls.len() > 1 {
            return Err("Model emitted multiple calls with parallel_tool_calls=false".into());
        }
        if !exhausted && !self.stopped {
            if (choice == "required" || choice.is_object()) && self.calls.is_empty() {
                return Err("Model did not satisfy required tool_choice".into());
            }
            if choice == "none" && !self.calls.is_empty() {
                return Err("Model emitted a tool call with tool_choice=none".into());
            }
        }
        Ok(deltas)
    }
    pub fn message(&self) -> Value {
        let mut message = json!({"role":"assistant", "content":if self.content.is_empty() { Value::Null } else { json!(self.content) }});
        if !self.reasoning_content.is_empty() {
            message["reasoning_content"] = json!(self.reasoning_content);
        }
        if !self.calls.is_empty() {
            message["tool_calls"] = json!(self.calls);
        }
        message
    }
    pub fn finish_reason(&self, exhausted: bool) -> &'static str {
        if exhausted || self.incomplete_call {
            "length"
        } else if !self.calls.is_empty() {
            "tool_calls"
        } else {
            "stop"
        }
    }
}
fn json_value_end(text: &str) -> Option<usize> {
    let mut parser = serde_json::Deserializer::from_str(text).into_iter::<Value>();
    parser.next()?.ok()?;
    Some(parser.byte_offset())
}
fn call_end(body: &str) -> Option<usize> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('{') {
        let prefix = body.len() - trimmed.len();
        let end = json_value_end(trimmed)? + prefix;
        let tail = body[end..].trim_start();
        tail.starts_with("</tool_call>")
            .then_some(body.len() - tail.len())
    } else {
        body.find("</tool_call>")
    }
}
fn json_fields(body: &str) -> Option<(Option<String>, Option<&str>)> {
    let mut rest = body.trim_start().strip_prefix('{')?.trim_start();
    let mut name = None;
    let mut arguments = None;
    loop {
        let end = json_value_end(rest)?;
        let key: String = serde_json::from_str(&rest[..end]).ok()?;
        rest = rest[end..].trim_start().strip_prefix(':')?.trim_start();
        let end = json_value_end(rest);
        if key == "name" {
            name = end.and_then(|n| serde_json::from_str(&rest[..n]).ok());
        }
        if key == "arguments" {
            arguments = Some(&rest[..end.unwrap_or(rest.len())]);
        }
        let Some(end) = end else { break };
        rest = rest[end..].trim_start();
        let Some(next) = rest.strip_prefix(',') else {
            break;
        };
        rest = next.trim_start();
        if rest.is_empty() {
            break;
        }
    }
    Some((name, arguments))
}
fn parameter_value(
    value: &str,
    schema: &Value,
    validator: Option<&jsonschema::Validator>,
) -> Result<Value> {
    let parsed = if schema["type"] == "string" {
        json!(value)
    } else {
        match serde_json::from_str::<Value>(value) {
            Ok(v) if validator.is_none_or(|validator| validator.is_valid(&v)) => v,
            _ => json!(value),
        }
    };
    Ok(parsed)
}
fn partial_call(
    body: &str,
    tools: &[Value],
    schemas: &Schemas,
) -> Result<Option<(String, String)>> {
    let body = body.trim_start();
    if body.starts_with('{') {
        if let Some((Some(name), arguments)) = json_fields(body) {
            return Ok(Some((name, arguments.unwrap_or("").into())));
        }
        return Ok(None);
    }
    let Some(rest) = body.strip_prefix("<function=") else {
        return Ok(None);
    };
    let Some((name, mut rest)) = rest.split_once('>') else {
        return Ok(None);
    };
    let tool = tools
        .iter()
        .find(|t| t["function"]["name"] == name)
        .unwrap_or(&Value::Null);
    let mut arguments = String::from("{");
    let mut count = 0;
    rest = rest.trim_start();
    while let Some(rest_value) = rest.strip_prefix("<parameter=") {
        let Some((key, raw_value)) = rest_value.split_once('>') else {
            break;
        };
        let schema = &tool["function"]["parameters"]["properties"][key];
        let complete = raw_value.find("</parameter>");
        if complete.is_none() && schema["type"] != "string" {
            break;
        }
        let value = if let Some(end) = complete {
            &raw_value[..end]
        } else {
            let tag = "</parameter>";
            let hold = (1..tag.len())
                .filter(|&n| raw_value.ends_with(&tag[..n]))
                .max()
                .unwrap_or(0);
            &raw_value[..raw_value.len() - hold]
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        if count > 0 {
            arguments.push(',');
        }
        arguments.push_str(&serde_json::to_string(key).map_err(|e| e.to_string())?);
        arguments.push(':');
        if let Some(end) = complete {
            let pointer = format!("#/properties/{}", key.replace('~', "~0").replace('/', "~1"));
            arguments.push_str(
                &parameter_value(value, schema, schemas.get(tools, name, &pointer)?)?.to_string(),
            );
            rest = raw_value[end + "</parameter>".len()..].trim_start();
            count += 1;
        } else {
            let encoded = serde_json::to_string(value).map_err(|e| e.to_string())?;
            arguments.push_str(&encoded[..encoded.len() - 1]);
            return Ok(Some((name.into(), arguments)));
        }
    }
    if rest.starts_with("</function>") {
        arguments.push('}');
    }
    Ok(Some((name.into(), arguments)))
}
struct ParsedCall {
    id: String,
    name: String,
    arguments: String,
}
impl ParsedCall {
    fn into_value(self) -> Value {
        json!({"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments}})
    }
}
fn parse_call(
    body: &str,
    tools: &[Value],
    id: &str,
    constrained: bool,
    schemas: &Schemas,
) -> Result<ParsedCall> {
    let body = body.trim();
    if body.starts_with('{') {
        let raw = body;
        let (name, raw_arguments) = json_fields(raw).ok_or("Invalid JSON tool fields")?;
        let name = name.ok_or("Tool name missing")?;
        let raw_arguments = raw_arguments.ok_or("Missing JSON arguments")?;
        if constrained {
            let arguments: Value = serde_json::from_str(raw_arguments)
                .map_err(|e| format!("Invalid JSON tool arguments: {e}"))?;
            tools
                .iter()
                .find(|t| t["function"]["name"] == name)
                .ok_or("Undeclared tool")?;
            schemas.validate(tools, &name, &arguments)?;
        }
        return Ok(ParsedCall {
            id: id.into(),
            name,
            arguments: raw_arguments.into(),
        });
    }
    let rest = body
        .strip_prefix("<function=")
        .ok_or("Tool call needs <function=...>")?;
    let (name, rest) = rest
        .split_once('>')
        .ok_or("Missing function name delimiter")?;
    let tool = tools
        .iter()
        .find(|t| t["function"]["name"] == name)
        .unwrap_or(&Value::Null);
    if constrained && tool.is_null() {
        return Err(format!("Undeclared tool {name}"));
    }
    let schema = &tool["function"]["parameters"];
    let mut rest = rest.trim_start();
    let mut arguments = Map::new();
    while let Some(parameters) = rest.strip_prefix("<parameter=") {
        let (key, value) = parameters
            .split_once('>')
            .ok_or("Missing parameter delimiter")?;
        let (value, tail) = value
            .split_once("</parameter>")
            .ok_or("Missing closing parameter tag")?;
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        if key.is_empty() || arguments.contains_key(key) {
            return Err("Empty or duplicate tool parameter".into());
        }
        let parameter_schema = &schema["properties"][key];
        let pointer = format!("#/properties/{}", key.replace('~', "~0").replace('/', "~1"));
        let parsed = parameter_value(value, parameter_schema, schemas.get(tools, name, &pointer)?)
            .map_err(|e| format!("Parameter {key}: {e}"))?;
        arguments.insert(key.to_owned(), parsed);
        rest = tail.trim_start();
    }
    if rest != "</function>" {
        return Err("Unexpected text inside tool call".into());
    }
    let arguments = Value::Object(arguments);
    if constrained {
        schemas.validate(tools, name, &arguments)?;
    }
    Ok(ParsedCall {
        id: id.into(),
        name: name.into(),
        arguments: serde_json::to_string(&arguments).map_err(|e| e.to_string())?,
    })
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
