use super::*;
#[test]
fn ollama_output_vectors_preserve_semantics_at_every_character_split() {
    let tools = vec![
        json!({"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}),
        json!({"type":"function","function":{"name":"get_time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}}),
    ];
    for case in crate::test_fixtures::ollama()["output"].as_array().unwrap() {
        let raw = case["raw"].as_str().unwrap();
        for at in (0..=raw.len()).filter(|&at| raw.is_char_boundary(at)) {
            let mut output = Output::new(false, tools.clone(), vec![], "test");
            let mut deltas = output.push(&raw[..at], false).unwrap();
            deltas.extend(output.push(&raw[at..], false).unwrap());
            deltas.extend(output.finish(false, &json!("auto"), true).unwrap());
            assert_eq!(
                json!(output.content),
                case["content"],
                "{} split={at}",
                case["id"]
            );
            assert_eq!(
                json!(output.reasoning_content),
                case["reasoning"],
                "{} split={at}",
                case["id"]
            );
            for field in ["content", "reasoning_content"] {
                let streamed: String = deltas
                    .iter()
                    .filter_map(|delta| delta[field].as_str())
                    .collect();
                let expected = if field == "content" {
                    &case["content"]
                } else {
                    &case["reasoning"]
                };
                assert_eq!(
                    json!(streamed),
                    *expected,
                    "{} {field} split={at}",
                    case["id"]
                );
            }
            assert!(
                deltas
                    .iter()
                    .all(|delta| delta.as_object().unwrap().len() == 1)
            );
            let expected = case["calls"].as_array().unwrap();
            assert_eq!(output.calls.len(), expected.len());
            for (index, call) in output.calls.iter().enumerate() {
                assert_eq!(call["function"]["name"], expected[index]["name"]);
                let arguments: Value =
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
                assert_eq!(arguments, expected[index]["arguments"]);
                let fragments: Vec<_> = deltas
                    .iter()
                    .flat_map(|delta| delta["tool_calls"].as_array().into_iter().flatten())
                    .filter(|call| call["index"] == index)
                    .collect();
                assert_eq!(
                    fragments
                        .iter()
                        .filter(|call| call.get("id").is_some())
                        .count(),
                    1
                );
                let joined: String = fragments
                    .iter()
                    .filter_map(|call| call["function"]["arguments"].as_str())
                    .collect();
                assert_eq!(joined, call["function"]["arguments"].as_str().unwrap());
            }
        }
    }
}

#[test]
fn ollama_finish_reason_vectors_keep_length_precedence_over_tools() {
    for case in crate::test_fixtures::ollama()["finish"]["cases"]
        .as_array()
        .unwrap()
    {
        let mut output = Output::new(false, vec![], vec![], "test");
        if case["has_tool"] == true {
            output.calls.push(json!({"type":"function"}));
        }
        assert_eq!(
            output.finish_reason(case["exhausted"].as_bool().unwrap()),
            case["expected"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
    let mut output = Output::new(false, vec![], vec![], "test");
    output.incomplete_call = true;
    assert_eq!(output.finish_reason(false), "length");
}
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
    let call = parse_call(body, &tools, "x", true, &schemas)
        .unwrap()
        .into_value();
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
        let call = parse_call(body, &tools(), "call_x_0", false, &Schemas::default())
            .unwrap()
            .into_value();
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
#[test]
fn token_identity_controls_thinking_and_dsml_including_string_arguments() {
    let literal = "中文 </think> <think> <｜DSML｜ calls>";
    let mut parser = Output::new(true, tools(), vec![], "boundary");
    parser.tool_format(ToolFormat::DeepSeekV41);
    parser.token_boundaries();
    for ch in literal.chars() {
        parser.push_token(&ch.to_string(), None).unwrap();
    }
    assert!(parser.is_reasoning());
    parser.push_token("</think>", Some("</think>")).unwrap();
    for ch in literal.chars() {
        parser.push_token(&ch.to_string(), None).unwrap();
    }
    assert!(parser.calls.is_empty());
    assert!(!parser.is_reasoning());
    // Structural DSML markers arrive as one special ID, with ordinary
    // punctuation/attributes split arbitrarily around them.
    let mut feed = |plain: &str, special: bool| {
        if special {
            parser.push_token(plain, Some(plain)).unwrap();
        } else {
            for ch in plain.chars() {
                parser.push_token(&ch.to_string(), None).unwrap();
            }
        }
    };
    feed("\n\n<", false);
    feed("｜DSML｜", true);
    feed(" calls>\n<", false);
    feed("｜DSML｜", true);
    feed(" invoke name=\"read\">\n<", false);
    feed("｜DSML｜", true);
    feed(" parameter name=\"path\" string=\"true\">", false);
    let argument = "literal </｜DSML｜ parameter> 中文 </think>";
    feed(argument, false);
    feed("</", false);
    feed("｜DSML｜", true);
    feed(" parameter>\n</", false);
    feed("｜DSML｜", true);
    feed(" invoke>\n</", false);
    feed("｜DSML｜", true);
    feed(" calls>", false);
    parser.finish(false, &json!("auto"), true).unwrap();
    assert_eq!(parser.reasoning_content, literal);
    assert_eq!(parser.content, literal);
    assert_eq!(parser.calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(parser.calls[0]["function"]["arguments"].as_str().unwrap())
            .unwrap(),
        json!({"path":argument})
    );
}
