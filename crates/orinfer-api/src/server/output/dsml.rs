//! DSv4.1 wire syntax. Values are consumed once; only a closing-tag suffix waits.
use super::{Controls, Result, find_marker, marker_at};
use std::collections::BTreeSet;

pub(super) const OPEN: &str = "<｜DSML｜ calls>";
pub(super) const CLOSE: &str = "</｜DSML｜ calls>";
pub(super) const INVOKE: &str = "<｜DSML｜ invoke ";
const END: &str = "</｜DSML｜ invoke>";
const PARAMETER: &str = "<｜DSML｜ parameter ";
const PARAMETER_END: &str = "</｜DSML｜ parameter>";

pub(super) enum Event {
    Begin(String),
    Arguments(String),
    End,
}
#[derive(Default)]
enum Stage {
    #[default]
    Name,
    Object,
    Parameter,
    Value(bool),
    CloseValue(bool),
    End,
}
#[derive(Default)]
pub(super) struct Call {
    stage: Stage,
    parameters: BTreeSet<String>,
    header: Header,
}
impl Call {
    pub fn advance(
        &mut self,
        input: &str,
        controls: Controls<'_>,
    ) -> Result<(usize, Option<Event>)> {
        match self.stage {
            Stage::Name => {
                let Some((length, attributes)) = self.header.advance(input)? else {
                    return Ok((0, None));
                };
                let name = attributes
                    .get("name")
                    .ok_or("Missing DSML function name")?
                    .clone();
                if name.is_empty() {
                    return Err("Empty DSML function name".into());
                }
                self.stage = Stage::Object;
                Ok((length, Some(Event::Begin(name))))
            }
            Stage::Object => {
                self.stage = Stage::Parameter;
                Ok((0, Some(Event::Arguments("{".into()))))
            }
            Stage::Parameter => {
                let rest = input.trim_start();
                let space = input.len() - rest.len();
                if rest.starts_with(END) && marker_at(controls, space, END) {
                    self.stage = Stage::End;
                    return Ok((space + END.len(), Some(Event::Arguments("}".into()))));
                }
                if END.starts_with(rest) || PARAMETER.starts_with(rest) {
                    return Ok((space, None));
                }
                let Some(body) = rest.strip_prefix(PARAMETER) else {
                    return Err("Expected DSML parameter or closing invoke".into());
                };
                if !marker_at(controls, space, PARAMETER) {
                    return Err("Expected a DSML control token".into());
                }
                let Some((length, attributes)) = self.header.advance(body)? else {
                    return Ok((space, None));
                };
                let key = attributes
                    .get("name")
                    .ok_or("Missing DSML parameter name")?;
                let string = match attributes.get("string").map(String::as_str) {
                    Some("true") => true,
                    Some("false") => false,
                    _ => return Err("DSML parameter requires string=true or false".into()),
                };
                let first = self.parameters.is_empty();
                if key.is_empty() || !self.parameters.insert(key.clone()) {
                    return Err("Empty or duplicate DSML parameter".into());
                }
                self.stage = Stage::Value(string);
                Ok((
                    space + PARAMETER.len() + length,
                    Some(Event::Arguments(format!(
                        "{}{}:{}",
                        if first { "" } else { "," },
                        serde_json::to_string(key).unwrap(),
                        if string { "\"" } else { "" }
                    ))),
                ))
            }
            Stage::Value(string) => {
                let end = find_marker(input, controls, PARAMETER_END);
                let count = end.unwrap_or_else(|| input.len() - suffix(input, PARAMETER_END));
                let value = &input[..count];
                let fragment = if string {
                    let encoded = serde_json::to_string(value).unwrap();
                    encoded[1..encoded.len() - 1].to_owned()
                } else {
                    value.to_owned()
                };
                if end.is_some() {
                    self.stage = Stage::CloseValue(string);
                }
                if count == 0 && end.is_none() {
                    return Ok((0, None));
                }
                Ok((count, Some(Event::Arguments(fragment))))
            }
            Stage::CloseValue(string) => {
                if !input.starts_with(PARAMETER_END) || !marker_at(controls, 0, PARAMETER_END) {
                    return Err("Missing DSML parameter terminator".into());
                }
                self.stage = Stage::Parameter;
                Ok((
                    PARAMETER_END.len(),
                    Some(Event::Arguments(if string { "\"" } else { "" }.into())),
                ))
            }
            Stage::End => Ok((0, Some(Event::End))),
        }
    }
}
fn suffix(input: &str, marker: &str) -> usize {
    (1..marker.len())
        .filter(|&n| marker.is_char_boundary(n) && input.ends_with(&marker[..n]))
        .max()
        .unwrap_or(0)
}
#[derive(Default)]
struct Header {
    scanned: usize,
    quoted: bool,
}
impl Header {
    fn advance(
        &mut self,
        input: &str,
    ) -> Result<Option<(usize, std::collections::BTreeMap<String, String>)>> {
        // A user stop can complete across chunks inside a pending header.
        if self.scanned > input.len() {
            return Ok(None);
        }
        for (offset, ch) in input[self.scanned..].char_indices() {
            if ch == '"' {
                self.quoted = !self.quoted;
            }
            if ch == '>' && !self.quoted {
                let end = self.scanned + offset + 1;
                *self = Self::default();
                return header(&input[..end])?
                    .map(Some)
                    .ok_or_else(|| "Invalid DSML attributes".into());
            }
        }
        self.scanned = input.len();
        Ok(None)
    }
}
fn header(input: &str) -> Result<Option<(usize, std::collections::BTreeMap<String, String>)>> {
    let mut attributes = std::collections::BTreeMap::new();
    let mut rest = input.trim_start();
    loop {
        if let Some(tail) = rest.strip_prefix('>') {
            return Ok(Some((input.len() - tail.len(), attributes)));
        }
        let Some((key, value)) = rest.split_once("=\"") else {
            return Ok(None);
        };
        if key.trim().contains(['<', '>', '"']) {
            return Err("Invalid DSML attribute".into());
        }
        let Some((value, tail)) = value.split_once('"') else {
            return Ok(None);
        };
        if attributes
            .insert(key.trim().to_owned(), value.to_owned())
            .is_some()
        {
            return Err("Duplicate DSML attribute".into());
        }
        rest = tail.trim_start();
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    fn tools() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"edit","parameters":{"type":"object","properties":{"text":{"type":"string"},"flag":{"type":"boolean"},"count":{"type":"integer"}}}}}),
        ]
    }
    #[test]
    fn every_split_preserves_wire_types_escaping_and_raw_content_spans() {
        let raw = "思考</think>说明\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n<｜DSML｜ parameter name=\"text\" string=\"true\">true\n\"a\\b\" <think>中文</think></｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"flag\" string=\"false\">false</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"count\" string=\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        for split in (0..=raw.len()).filter(|&n| raw.is_char_boundary(n)) {
            let mut p = Output::new(true, tools(), vec![], "t");
            p.tool_format(ToolFormat::DeepSeekV41);
            let mut deltas = p.push(&raw[..split], false).unwrap();
            deltas.extend(p.push(&raw[split..], false).unwrap());
            deltas.extend(p.finish(false, &json!("auto"), true).unwrap());
            assert_eq!(p.reasoning_content, "思考");
            assert_eq!(p.content, "说明");
            assert_eq!(p.finish_reason(false), "tool_calls");
            assert_eq!(p.calls.len(), 1);
            let args: Value =
                serde_json::from_str(p.calls[0]["function"]["arguments"].as_str().unwrap())
                    .unwrap();
            assert_eq!(
                args,
                json!({"text":"true\n\"a\\b\" <think>中文</think>","flag":false,"count":42})
            );
            let streamed = deltas
                .iter()
                .filter_map(|d| d["tool_calls"][0]["function"]["arguments"].as_str())
                .collect::<String>();
            assert_eq!(streamed, p.calls[0]["function"]["arguments"]);
            assert_eq!(p.processed_bytes(), raw.len());
            let mut content = String::new();
            while let Some(span) = p.take_content_span() {
                content.push_str(&raw[span]);
            }
            assert_eq!(content, "说明");
        }
    }
    #[test]
    fn string_chunks_stream_before_parameter_end_and_truncation_keeps_exact_prefix() {
        let mut p = Output::new(false, tools(), vec![], "t");
        p.tool_format(ToolFormat::DeepSeekV41);
        let header = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"edit\">\n<｜DSML｜ parameter name=\"text\" string=\"true\">";
        let mut deltas = p.push(header, false).unwrap();
        for chunk in ["中文\\\"", "<think>", "代码\n"] {
            let ds = p.push(chunk, false).unwrap();
            assert!(!ds.is_empty());
            deltas.extend(ds);
        }
        deltas.extend(p.finish(true, &json!("auto"), true).unwrap());
        assert_eq!(p.finish_reason(true), "length");
        assert!(p.incomplete_call);
        let streamed = deltas
            .iter()
            .filter_map(|d| d["tool_calls"][0]["function"]["arguments"].as_str())
            .collect::<String>();
        assert_eq!(streamed, p.calls[0]["function"]["arguments"]);
    }
    #[test]
    fn parallel_calls_and_markers_in_thinking_do_not_leak_into_text() {
        let mut p = Output::new(true, tools(), vec![], "t");
        p.tool_format(ToolFormat::DeepSeekV41);
        let invoke = "<｜DSML｜ invoke name=\"edit\">\n</｜DSML｜ invoke>\n";
        p.push(&format!("Literal <｜DSML｜ calls> in thought</think>\n\n<｜DSML｜ calls>\n{invoke}{invoke}</｜DSML｜ calls>"),false).unwrap();
        p.finish(false, &json!("required"), true).unwrap();
        assert_eq!(p.calls.len(), 2);
        assert!(p.content.is_empty());
        assert!(p.reasoning_content.contains("<｜DSML｜ calls>"));
    }
}
