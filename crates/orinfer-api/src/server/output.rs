//! Incremental Qwen XML tool/reasoning parsing into Chat Completions deltas.
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, sync::OnceLock};

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

pub struct Output {
    schemas: Schemas,
    pending: String,
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
            schemas: Schemas::default(),
            pending: String::new(),
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
    pub fn constrained_tools(&mut self, enabled: bool) {
        self.constrained_tools = enabled;
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
                    self.stream_call(
                        call["function"]["name"].as_str().unwrap(),
                        call["function"]["arguments"].as_str().unwrap(),
                        &mut deltas,
                    )?;
                    self.calls.push(call);
                    self.drain(end + "</tool_call>".len());
                    self.in_tool = false;
                    self.streamed_call = None;
                    continue;
                }
                break;
            }
            let markers: &[&str] = if self.tools.is_empty() {
                &["<think>", "</think>"]
            } else {
                &["<tool_call>", "<think>", "</think>"]
            };
            let marker = markers
                .iter()
                .filter_map(|tag| self.pending.find(tag).map(|at| (at, *tag)))
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
                self.emit_text(&before, &mut deltas);
                self.drain(at + tag.len());
                match tag {
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
            self.emit_text(&text, &mut deltas);
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
            0
        };
        if arguments.len() > previous {
            deltas.push(json!({"tool_calls":[{"index":index,"function":{"arguments":&arguments[previous..]}}]}));
        }
        self.streamed_call = Some((name.into(), arguments.into()));
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
        let Some((key, value)) = rest_value.split_once('>') else {
            break;
        };
        let schema = &tool["function"]["parameters"]["properties"][key];
        let complete = value.find("</parameter>");
        if complete.is_none() && schema["type"] != "string" {
            break;
        }
        let value = if let Some(end) = complete {
            &value[..end]
        } else {
            let tag = "</parameter>";
            let hold = (1..tag.len())
                .filter(|&n| value.ends_with(&tag[..n]))
                .max()
                .unwrap_or(0);
            &value[..value.len() - hold]
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        if count > 0 {
            arguments.push(',');
        }
        arguments.push_str(&serde_json::to_string(key).unwrap());
        arguments.push(':');
        if let Some(end) = complete {
            let pointer = format!("#/properties/{}", key.replace('~', "~0").replace('/', "~1"));
            arguments.push_str(
                &parameter_value(value, schema, schemas.get(tools, name, &pointer)?)?.to_string(),
            );
            // Complete offsets refer to the original parameter suffix.
            rest = rest_value.split_once('>').unwrap().1[end + "</parameter>".len()..].trim_start();
            count += 1;
        } else {
            let encoded = serde_json::to_string(value).unwrap();
            arguments.push_str(&encoded[..encoded.len() - 1]);
            return Ok(Some((name.into(), arguments)));
        }
    }
    if rest.starts_with("</function>") {
        arguments.push('}');
    }
    Ok(Some((name.into(), arguments)))
}
fn parse_call(
    body: &str,
    tools: &[Value],
    id: &str,
    constrained: bool,
    schemas: &Schemas,
) -> Result<Value> {
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
        return Ok(
            json!({"id":id,"type":"function","function":{"name":name,"arguments":raw_arguments}}),
        );
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
    Ok(
        json!({"id":id, "type":"function", "function":{"name":name,"arguments":serde_json::to_string(&arguments).map_err(|e| e.to_string())?}}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_calls_validate_ranges_patterns_array_limits_and_local_references() {
        let tools = vec![
            json!({"type":"function","function":{"name":"check","parameters":{
                "type":"object", "$defs":{"count":{"type":"integer","minimum":0,"maximum":10}},
                "properties":{"count":{"$ref":"#/$defs/count"},"label":{"type":"string","pattern":"^[a-z]+$","minLength":2,"maxLength":4},"items":{"type":"array","items":{"type":"boolean"},"minItems":1,"maxItems":2}},
                "required":["count","label","items"], "additionalProperties":false
            }}}),
        ];
        let schemas = Schemas::default();
        let valid = json!({"count":3,"label":"good","items":[true]});
        schemas.validate(&tools, "check", &valid).unwrap();
        for (key, invalid) in [
            ("count", json!(-1)),
            ("count", json!(11)),
            ("label", json!("BAD")),
            ("label", json!("x")),
            ("label", json!("abcde")),
            ("items", json!([])),
            ("items", json!([true, false, true])),
        ] {
            let mut value = valid.clone();
            value[key] = invalid;
            let body = json!({"name":"check","arguments":value}).to_string();
            assert!(parse_call(&body, &tools, "x", true, &schemas).is_err());
            assert!(parse_call(&body, &tools, "x", false, &schemas).is_ok());
        }
        let xml = "<tool_call><function=check><parameter=count>3</parameter><parameter=label>good</parameter><parameter=items>[true]</parameter></function></tool_call>";
        for at in (0..=xml.len()).filter(|&at| xml.is_char_boundary(at)) {
            let mut output = Output::new(false, tools.clone(), vec![], "x");
            output.constrained_tools(true);
            output.push(&xml[..at], false).unwrap();
            output.push(&xml[at..], true).unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(
                    output.calls[0]["function"]["arguments"].as_str().unwrap()
                )
                .unwrap(),
                valid
            );
        }
    }
    #[test]
    fn property_validation_keeps_root_references_and_escaped_names() {
        let tools = vec![
            json!({"type":"function","function":{"name":"check","parameters":{
                "type":"object","$defs":{"int":{"type":"integer"}},
                "properties":{"key/~":{"$ref":"#/$defs/int"}}, "required":["key/~"]
            }}}),
        ];
        let schemas = Schemas::default();
        let body = "<function=check><parameter=key/~>7</parameter></function>";
        let call = parse_call(body, &tools, "x", true, &schemas).unwrap();
        assert_eq!(call["function"]["arguments"], "{\"key/~\":7}");
        assert!(
            schemas
                .get(&tools, "check", "#/properties/key~1~0")
                .unwrap()
                .is_some()
        );
    }
    #[test]
    fn json_and_xml_arguments_stream_before_completion_and_preserve_fragments() {
        let bodies = [
            "<tool_call><function=read><parameter=path>hello世界\\x</parameter><parameter=limit>3</parameter></function></tool_call>",
            r#"<tool_call>{"name":"read","arguments":{"path":"hello世界</tool_call>\\x","limit":3}}</tool_call>"#,
            r#"<tool_call>{"arguments": { "path": "世界", "limit": 3 }, "name":"read"}</tool_call>"#,
        ];
        for body in bodies {
            for split in (0..=body.len()).filter(|&i| body.is_char_boundary(i)) {
                let mut p = Output::new(false, tools(), vec![], "test");
                let mut deltas = p.push(&body[..split], false).unwrap();
                deltas.extend(p.push(&body[split..], false).unwrap());
                deltas.extend(p.finish(false, &json!("required"), false).unwrap());
                let arguments = deltas
                    .iter()
                    .flat_map(|d| d["tool_calls"].as_array().into_iter().flatten())
                    .filter_map(|c| c["function"]["arguments"].as_str())
                    .collect::<String>();
                assert_eq!(
                    arguments,
                    p.calls[0]["function"]["arguments"].as_str().unwrap()
                );
            }
            let mut p = Output::new(false, tools(), vec![], "test");
            let prefix = body.find("世界").unwrap();
            if body.starts_with("<tool_call>{\"arguments\"") {
                continue;
            }
            let deltas = p.push(&body[..prefix], false).unwrap();
            assert!(deltas.iter().any(|d| d["tool_calls"].is_array()));
            assert!(p.calls.is_empty());
        }
    }
    #[test]
    fn content_spans_exclude_reasoning_tools_and_stop_suffixes() {
        let mut p = Output::new(true, tools(), vec!["STOP".into()], "test");
        p.push("秘密</think>回答ST", false).unwrap();
        let span = p.take_content_span().unwrap();
        assert_eq!(span, "秘密</think>".len().."秘密</think>回答".len());
        assert!(p.take_content_span().is_none());
        p.push("OP trailing", false).unwrap();
        assert!(p.stopped);
        assert_eq!(p.content, "回答");
        assert!(p.take_content_span().is_none());
    }
    fn tools() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"],"additionalProperties":false}}}),
        ]
    }
    #[test]
    fn every_split_preserves_xml_and_typed_arguments() {
        let text = "OK\n<tool_call>\n<function=read>\n<parameter=path>\nx\n</parameter>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>";
        for at in 0..=text.len() {
            let mut parser = Output::new(false, tools(), vec![], "test");
            parser.push(&text[..at], false).unwrap();
            parser.push(&text[at..], false).unwrap();
            parser.finish(false, &json!("auto"), true).unwrap();
            assert_eq!(parser.content, "OK\n");
            let args: Value =
                serde_json::from_str(parser.calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap();
            assert_eq!(args, json!({"path":"x","limit":3}));
        }
    }
    #[test]
    fn stop_reasoning_and_incomplete_calls() {
        let mut p = Output::new(true, tools(), vec!["END".into()], "t");
        p.push("考える</thi", false).unwrap();
        p.push("nk>答えEN", false).unwrap();
        assert_eq!(p.content, "答え");
        p.push("Dhidden", false).unwrap();
        assert!(p.stopped);
        assert_eq!(p.reasoning_content, "考える");
        let mut p = Output::new(false, tools(), vec![], "t");
        p.push("<tool_call><function=read>", false).unwrap();
        assert!(p.finish(true, &json!("auto"), true).is_ok());
        assert_eq!(p.calls[0]["function"]["arguments"], "{");
        assert!(p.incomplete_call);
        assert!(p.content.is_empty());
        let mut p = Output::new(false, tools(), vec![], "t");
        p.push("<tool_call><function=read>", false).unwrap();
        p.finish(false, &json!("auto"), true).unwrap();
        assert!(p.incomplete_call);
    }
    #[test]
    fn tool_markup_is_literal_without_tools() {
        let mut p = Output::new(false, vec![], vec![], "t");
        p.push("Example: <tool_call>...</tool_call>", false)
            .unwrap();
        p.finish(false, &json!("none"), true).unwrap();
        assert_eq!(p.content, "Example: <tool_call>...</tool_call>");
        assert!(p.calls.is_empty());
    }
    #[test]
    fn structured_json_keeps_protocol_tags_inside_strings_at_every_split() {
        let body = r#"{"text":"<think>中文</think><tool_call>literal</tool_call>"}"#;
        for thinking in [false, true] {
            let raw = if thinking {
                format!("Reasoning</think>{body}")
            } else {
                body.into()
            };
            for split in (0..=raw.len()).filter(|&i| raw.is_char_boundary(i)) {
                let mut p = Output::new(thinking, tools(), vec![], "test");
                p.structured_json(true);
                p.push(&raw[..split], false).unwrap();
                p.push(&raw[split..], false).unwrap();
                p.finish(false, &json!("auto"), true).unwrap();
                assert_eq!(p.content, body);
                assert!(p.calls.is_empty());
            }
        }
    }
    #[test]
    fn ordinary_call_errors_reach_clients_while_constrained_calls_stay_strict() {
        for body in [
            "<function=READ><parameter=path>x</parameter></function>",
            "<function=read><parameter=limit>bad</parameter></function>",
            "<function=read></function>",
            r#"{"name":"read","arguments":{"limit":"bad"}}"#,
            r#"{"name":"read","arguments":[]}"#,
        ] {
            let call = parse_call(body, &tools(), "call_x_0", false, &Schemas::default()).unwrap();
            assert!(call["function"]["arguments"].is_string());
            assert!(parse_call(body, &tools(), "x", true, &Schemas::default()).is_err());
            let raw = format!("<tool_call>{body}</tool_call>");
            for at in (0..=raw.len()).filter(|&i| raw.is_char_boundary(i)) {
                let mut parser = Output::new(false, tools(), vec![], "x");
                let mut deltas = parser.push(&raw[..at], false).unwrap();
                deltas.extend(parser.push(&raw[at..], false).unwrap());
                deltas.extend(parser.finish(false, &json!("auto"), true).unwrap());
                let arguments = deltas
                    .iter()
                    .flat_map(|d| d["tool_calls"].as_array().into_iter().flatten())
                    .filter_map(|c| c["function"]["arguments"].as_str())
                    .collect::<String>();
                assert_eq!(arguments, call["function"]["arguments"]);
                assert_eq!(parser.calls[0], call);
            }
        }
    }
}
