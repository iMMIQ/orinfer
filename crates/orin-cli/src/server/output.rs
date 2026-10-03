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
        }
    }
    pub fn push(&mut self, text: &str, final_chunk: bool) -> Result<Vec<Value>> {
        self.pending.push_str(text);
        let mut deltas = vec![];
        loop {
            if self.stopped {
                self.pending.clear();
                break;
            }
            let stop = self
                .stops
                .iter()
                .filter_map(|s| self.pending.find(s).map(|at| (at, s.len())))
                .min_by_key(|x| x.0);
            if self.in_tool {
                let end = self.pending.find("</tool_call>");
                if stop.is_some_and(|(at, _)| end.is_none_or(|end| at < end)) {
                    self.pending.clear();
                    self.stopped = true;
                    break;
                }
                if let Some(end) = end {
                    let call = parse_call(
                        &self.pending[..end],
                        &self.tools,
                        &format!("call_{}_{}", self.call_prefix, self.calls.len()),
                    )?;
                    let mut delta_call = call.clone();
                    delta_call["index"] = json!(self.calls.len());
                    deltas.push(json!({"tool_calls":[delta_call]}));
                    self.calls.push(call);
                    self.pending.drain(..end + "</tool_call>".len());
                    self.in_tool = false;
                    continue;
                }
                // Incomplete calls stay private, including on output-token exhaustion.
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
                self.pending.drain(..at + tag.len());
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
            self.pending.drain(..n);
            self.emit_text(&text, &mut deltas);
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
            self.content.push_str(text);
            deltas.push(json!({"content":text}));
        }
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
fn parse_call(body: &str, tools: &[Value], id: &str) -> Result<Value> {
    let body = body.trim();
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
        let parsed = if parameter_schema["type"] == "string" {
            json!(value)
        } else {
            match serde_json::from_str::<Value>(value) {
                Ok(v) if validate_schema(&v, parameter_schema).is_ok() => v,
                _ => json!(value),
            }
        };
        validate_schema(&parsed, parameter_schema).map_err(|e| format!("Parameter {key}: {e}"))?;
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
    if let Some(items) = value.as_array() {
        if let Some(schema) = schema.get("items") {
            for v in items {
                validate_schema(v, schema)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn invalid_and_undeclared_calls_rejected() {
        assert!(parse_call("<function=nope></function>", &tools(), "x").is_err());
        assert!(
            parse_call(
                "<function=read><parameter=limit>bad</parameter></function>",
                &tools(),
                "x"
            )
            .is_err()
        );
        assert!(parse_call("<function=read></function>", &tools(), "x").is_err());
    }
}
