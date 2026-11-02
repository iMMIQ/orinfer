//! DSv4.1 data preparation; the model template owns role/turn rendering.
use super::*;

/// Preserve the checkpoint's text-part boundaries before shared media packing
/// turns content arrays into strings. Non-text parts keep its validation path.
pub(super) fn text_parts(messages: &mut [Value]) -> Result<()> {
    for message in messages {
        if message["content"]
            .as_array()
            .is_some_and(|parts| parts.iter().all(|p| p["type"] == "text"))
        {
            message["content"] = json!(text_content(&message["content"], "\n\n")?);
        }
    }
    Ok(())
}

pub(super) fn prepare(
    mut messages: Vec<Value>,
    tools: &[Value],
) -> Result<(Vec<Value>, Vec<String>)> {
    let schemas = tools
        .iter()
        .map(|tool| {
            let function = tool["function"]
                .as_object()
                .ok_or("Missing function definition")?;
            let function: serde_json::Map<_, _> = function
                .iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            python_json(&Value::Object(function))
        })
        .collect::<Result<Vec<_>>>()?;
    if !tools.is_empty() && messages.first().is_none_or(|m| m["role"] != "system") {
        messages.insert(0, json!({"role":"system","content":""}));
    }
    let mut output: Vec<Value> = vec![];
    let mut order = std::collections::BTreeMap::<String, usize>::new();
    let mut blocks: Vec<(Option<String>, String)> = vec![];
    let flush = |output: &mut Vec<Value>,
                 blocks: &mut Vec<(Option<String>, String)>,
                 order: &std::collections::BTreeMap<String, usize>| {
        if blocks.is_empty() {
            return;
        }
        let only_tools = blocks.iter().all(|(id, _)| id.is_some());
        let mut results = blocks
            .iter_mut()
            .filter(|(id, _)| id.is_some())
            .map(|(id, text)| (id.clone(), std::mem::take(text)))
            .collect::<Vec<_>>();
        results.sort_by_key(|(id, _)| order.get(id.as_ref().unwrap()).copied().unwrap_or(0));
        let mut results = results.into_iter();
        let content = blocks
            .drain(..)
            .map(|(id, text)| {
                if id.is_some() {
                    results.next().unwrap().1
                } else {
                    text
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        output.push(json!({"role":"user","content":content,"tool_results_only":only_tools}));
    };
    for mut message in messages {
        let role = message["role"].as_str().unwrap().to_owned();
        if role == "user" || role == "tool" {
            let Value::String(content) = message["content"].take() else {
                return Err("Expected normalized message text".into());
            };
            blocks.push(if role == "tool" {
                (
                    Some(message["tool_call_id"].as_str().unwrap().to_owned()),
                    format!("<tool_result>{content}</tool_result>"),
                )
            } else {
                (None, content)
            });
            continue;
        }
        flush(&mut output, &mut blocks, &order);
        if role == "assistant"
            && let Some(calls) = message["tool_calls"].as_array()
        {
            order = calls
                .iter()
                .enumerate()
                .map(|(i, c)| (c["id"].as_str().unwrap().to_owned(), i))
                .collect();
            let mut encoded = String::from("\n\n<｜DSML｜ calls>\n");
            for (index, call) in calls.iter().enumerate() {
                if index > 0 {
                    encoded.push('\n');
                }
                let function = &call["function"];
                encoded.push_str(&format!(
                    "<｜DSML｜ invoke name=\"{}\">\n",
                    function["name"].as_str().unwrap()
                ));
                let arguments = function["arguments"]
                    .as_object()
                    .ok_or("Expected normalized tool argument map")?;
                for (index, (key, value)) in arguments.iter().enumerate() {
                    if key.contains('"') {
                        return Err("DSML parameter names cannot contain quotes".into());
                    }
                    if index > 0 {
                        encoded.push('\n');
                    }
                    let string = value.is_string();
                    let value = if let Some(s) = value.as_str() {
                        s.to_owned()
                    } else {
                        python_json(value)?
                    };
                    encoded.push_str(&format!("<｜DSML｜ parameter name=\"{key}\" string=\"{string}\">{value}</｜DSML｜ parameter>"));
                }
                encoded.push_str("\n</｜DSML｜ invoke>");
            }
            encoded.push_str("\n</｜DSML｜ calls>");
            if !calls.is_empty() {
                message["dsml_calls"] = json!(encoded);
            }
        }
        output.push(message);
    }
    flush(&mut output, &mut blocks, &order);
    Ok((output, schemas))
}
fn python_json(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonJsonFormatter);
    serde::Serialize::serialize(value, &mut serializer).map_err(|e| e.to_string())?;
    Ok(String::from_utf8(bytes).expect("JSON is UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_template_matches_official_tool_and_thinking_vectors() {
        let fixtures: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/chat/deepseek-v41.json"
        ))
        .unwrap();
        let mut env = Environment::new();
        env.add_template(
            "chat",
            include_str!("../../../../../tools/model/deepseek_v41/templates/chat.jinja"),
        )
        .unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let request = &case["request"];
            let mut raw = request["messages"].as_array().unwrap().clone();
            text_parts(&mut raw).unwrap();
            let (raw, images) = super::super::super::image::messages(
                raw,
                None,
                &mut super::super::super::preparation::Context::unbounded(),
            )
            .unwrap();
            assert!(images.is_empty());
            let messages = normalize_messages_for(raw, ToolFormat::DeepSeekV41).unwrap();
            let tools = request["tools"].as_array().cloned().unwrap_or_default();
            let (messages, schemas) = prepare(messages, &tools).unwrap();
            let effort = match request["reasoning_effort"].as_str() {
                Some("low") => 50,
                Some("max") => 100,
                _ => 75,
            };
            let rendered=env.get_template("chat").unwrap().render(context!{messages=>Serde(&messages),tools=>Serde(&tools),tool_schemas=>Serde(&schemas),enable_thinking=>request["enable_thinking"]==true,preserve_thinking=>!tools.is_empty(),deepseek_effort=>effort}).unwrap();
            assert_eq!(rendered, case["prompt"], "{}", case["name"]);
        }
    }
}
