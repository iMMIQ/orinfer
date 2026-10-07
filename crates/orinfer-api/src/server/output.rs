//! Incremental Qwen XML tool/reasoning parsing into Chat Completions deltas.
use serde_json::{Map, Value, json};

type Result<T> = std::result::Result<T, String>;

pub struct Output {
    pending: String,
    reasoning: bool,
    in_tool: bool,
    pub stopped: bool,
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
            pending: String::new(),
            reasoning: thinking,
            in_tool: false,
            stopped: false,
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
                if let Some((name, arguments)) = partial_call(&self.pending, &self.tools)? {
                    self.stream_call(&name, &arguments, &mut deltas)?;
                }
                if let Some(end) = end {
                    let call = parse_call(
                        &self.pending[..end],
                        &self.tools,
                        &format!("call_{}_{}", self.call_prefix, self.calls.len()),
                        self.constrained_tools,
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
                // The incremental prefix may already be streamed; incomplete
                // calls never become completed calls in the final response.
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
        if self.in_tool && !exhausted && !self.stopped {
            return Err("Model emitted an incomplete tool call".into());
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
        if key == "arguments" && rest.starts_with('{') {
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
fn parameter_value(value: &str, schema: &Value) -> Result<Value> {
    let parsed = if schema["type"] == "string" {
        json!(value)
    } else {
        match serde_json::from_str::<Value>(value) {
            Ok(v) if validate_schema(&v, schema).is_ok() => v,
            _ => json!(value),
        }
    };
    validate_schema(&parsed, schema)?;
    Ok(parsed)
}
fn partial_call(body: &str, tools: &[Value]) -> Result<Option<(String, String)>> {
    let body = body.trim_start();
    if body.starts_with('{') {
        if let Some((Some(name), arguments)) = json_fields(body) {
            if !tools.iter().any(|t| t["function"]["name"] == name) {
                return Err("Undeclared tool".into());
            }
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
        .ok_or("Undeclared tool")?;
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
            arguments.push_str(&parameter_value(value, schema)?.to_string());
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
fn parse_call(body: &str, tools: &[Value], id: &str, constrained: bool) -> Result<Value> {
    let body = body.trim();
    if body.starts_with('{') {
        let raw = body;
        let body: Value =
            serde_json::from_str(raw).map_err(|e| format!("Invalid JSON tool call: {e}"))?;
        let name = body["name"].as_str().ok_or("Tool name missing")?;
        let tool = tools
            .iter()
            .find(|t| t["function"]["name"] == name)
            .ok_or("Undeclared tool")?;
        let arguments = body.get("arguments").ok_or("Tool arguments missing")?;
        if !arguments.is_object() {
            return Err("Tool arguments must be an object".into());
        }
        if !constrained {
            validate_schema(arguments, &tool["function"]["parameters"])?;
        }
        let (_, raw_arguments) = json_fields(raw).ok_or("Invalid JSON tool fields")?;
        return Ok(
            json!({"id":id,"type":"function","function":{"name":name,"arguments":raw_arguments.ok_or("Missing JSON arguments")?}}),
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
        .ok_or_else(|| format!("Undeclared tool {name}"))?;
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
        let parsed = parameter_value(value, parameter_schema)
            .map_err(|e| format!("Parameter {key}: {e}"))?;
        arguments.insert(key.to_owned(), parsed);
        rest = tail.trim_start();
    }
    if rest != "</function>" {
        return Err("Unexpected text inside tool call".into());
    }
    let arguments = Value::Object(arguments);
    validate_schema(&arguments, schema)?;
    Ok(
        json!({"id":id, "type":"function", "function":{"name":name,"arguments":serde_json::to_string(&arguments).map_err(|e| e.to_string())?}}),
    )
}
// Structural checks catch malformed calls. This is not a constrained decoder or
// a complete JSON Schema implementation; clients still validate tool arguments.
fn validate_schema(value: &Value, schema: &Value) -> Result<()> {
    if schema == &Value::Bool(false) {
        return Err("Forbidden by schema".into());
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(choices) = schema[keyword].as_array() {
            let valid = choices
                .iter()
                .filter(|s| validate_schema(value, s).is_ok())
                .count();
            if valid == 0 || (keyword == "oneOf" && valid != 1) {
                return Err(format!("Does not match {keyword}"));
            }
        }
    }
    if let Some(choices) = schema["allOf"].as_array() {
        for s in choices {
            validate_schema(value, s)?;
        }
    }
    if let Some(types) = schema.get("type") {
        let matches = |name: &str| match name {
            "string" => value.is_string(),
            "object" => value.is_object(),
            "array" => value.is_array(),
            "number" => value.is_number(),
            "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
            "boolean" => value.is_boolean(),
            "null" => value.is_null(),
            _ => false,
        };
        let valid = types.as_str().is_some_and(matches)
            || types
                .as_array()
                .is_some_and(|a| a.iter().any(|t| t.as_str().is_some_and(matches)));
        if !valid {
            return Err("Incorrect parameter type".into());
        }
    }
    if schema["enum"]
        .as_array()
        .is_some_and(|a| !a.contains(value))
    {
        return Err("Value outside enum".into());
    }
    if schema.get("const").is_some_and(|c| c != value) {
        return Err("Incorrect const".into());
    }
    if let Some(object) = value.as_object() {
        if let Some(required) = schema["required"].as_array() {
            for key in required {
                if !key.as_str().is_some_and(|s| object.contains_key(s)) {
                    return Err("Required parameter missing".into());
                }
            }
        }
        for (key, value) in object {
            if let Some(property) = schema["properties"].get(key) {
                validate_schema(value, property)?;
            } else if schema["additionalProperties"] == false {
                return Err("Unexpected parameter".into());
            } else if schema["additionalProperties"].is_object() {
                validate_schema(value, &schema["additionalProperties"])?;
            }
        }
    }
    if let Some(items) = value.as_array()
        && let Some(schema) = schema.get("items")
    {
        for v in items {
            validate_schema(v, schema)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(p.finish(false, &json!("auto"), true).is_err());
        assert!(p.finish(true, &json!("auto"), true).is_ok());
        assert!(p.calls.is_empty());
        assert!(p.content.is_empty());
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
    fn invalid_and_undeclared_calls_rejected() {
        assert!(parse_call("<function=nope></function>", &tools(), "x", false).is_err());
        assert!(
            parse_call(
                "<function=read><parameter=limit>bad</parameter></function>",
                &tools(),
                "x",
                false
            )
            .is_err()
        );
        assert!(parse_call("<function=read></function>", &tools(), "x", false).is_err());
    }
}
