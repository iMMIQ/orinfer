//! Converged native tool constraints. Complex schemas use the shared JSON path.
use super::*;
const END: &str = "</｜DSML｜ parameter>";
fn literal(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}
fn parameter(
    schema: &Value,
    defs: &Value,
    stem: &str,
    key: &str,
    depth: usize,
) -> Option<(String, String)> {
    if depth > 64 {
        return None;
    }
    if key.contains('"') {
        return None;
    }
    let mut schema = schema.clone();
    if let Some(reference) = schema["$ref"].as_str() {
        let name = reference.strip_prefix("#/$defs/")?;
        let mut target = defs.get(name)?.as_object()?.clone();
        for (k, v) in schema.as_object()? {
            if k != "$ref" {
                target.insert(k.clone(), v.clone());
            }
        }
        schema = Value::Object(target);
    }
    if let Some(branches) = schema["anyOf"].as_array() {
        // Sibling assertions also constrain every branch. Keep their conjunction
        // in the shared JSON compiler rather than dropping it during lowering.
        if schema.as_object()?.keys().any(|k| {
            ![
                "anyOf",
                "title",
                "description",
                "default",
                "examples",
                "$defs",
            ]
            .contains(&k.as_str())
        }) {
            return None;
        }
        let mut rules = String::new();
        let mut alternatives = vec![];
        for (i, branch) in branches.iter().enumerate() {
            let (rule, name) = parameter(branch, defs, &format!("{stem}a{i}"), key, depth + 1)?;
            rules.push_str(&rule);
            alternatives.push(name);
        }
        rules.push_str(&format!("{stem}: {}\n", alternatives.join(" | ")));
        return Some((rules, stem.into()));
    }
    if let Some(types) = schema["type"].as_array() {
        let branches = types
            .iter()
            .map(|t| {
                let mut s = schema.clone();
                s["type"] = t.clone();
                s
            })
            .collect::<Vec<_>>();
        return parameter(&json!({"anyOf":branches}), defs, stem, key, depth + 1);
    }
    let string = schema["type"] == "string";
    schema.get("type")?;
    let begin = literal(&format!(
        "<｜DSML｜ parameter name=\"{key}\" string=\"{string}\">"
    ));
    let end = literal(END);
    if !string {
        schema["$defs"] = defs.clone();
        return Some((
            format!("{stem}: {begin} {stem}v {end} \"\\n\"\n{stem}v: %json {schema}\n"),
            stem.into(),
        ));
    }
    let allowed = [
        "type",
        "title",
        "description",
        "default",
        "examples",
        "enum",
        "const",
        "minLength",
        "maxLength",
        "pattern",
    ];
    if schema
        .as_object()?
        .keys()
        .any(|k| !allowed.contains(&k.as_str()))
    {
        return None;
    }
    let values = schema
        .get("const")
        .map(|v| vec![v.clone()])
        .or_else(|| schema["enum"].as_array().cloned());
    if let Some(values) = values {
        if schema.get("pattern").is_some()
            || schema.get("minLength").is_some()
            || schema.get("maxLength").is_some()
        {
            return None;
        }
        let values = values
            .iter()
            .map(|v| v.as_str().filter(|v| !v.contains(END)).map(literal))
            .collect::<Option<Vec<_>>>()?;
        if values.is_empty() {
            return None;
        }
        return Some((
            format!("{stem}: {begin} ({}) {end} \"\\n\"\n", values.join(" | ")),
            stem.into(),
        ));
    }
    let regex = if let Some(pattern) = schema["pattern"].as_str() {
        if schema.get("minLength").is_some() || schema.get("maxLength").is_some() {
            return None;
        }
        let anchored_start = pattern.starts_with('^');
        let anchored_end = pattern.ends_with('$');
        let pattern = pattern.strip_prefix('^').unwrap_or(pattern);
        let pattern = pattern.strip_suffix('$').unwrap_or(pattern);
        if pattern.contains(['^', '$']) {
            return None;
        }
        format!(
            "{}(?:{}){}",
            if anchored_start { "" } else { "(?s:.*)" },
            pattern.replace('/', "\\/"),
            if anchored_end { "" } else { "(?s:.*)" }
        )
    } else {
        let min = schema["minLength"].as_u64().unwrap_or(0);
        let max = schema["maxLength"]
            .as_u64()
            .map(|v| v.to_string())
            .unwrap_or_default();
        format!("(?s:.{{{min},{max}}})")
    };
    Some((
        format!("{stem}: {begin} {stem}v \"\\n\"\n{stem}v[suffix={end}]: /{regex}/\n"),
        stem.into(),
    ))
}
pub(super) fn recipe(
    tools: &[Value],
    choice: &Value,
    parallel: bool,
    thinking: bool,
) -> Result<Option<String>> {
    let mut grammar = String::from("ws: /[ \\t\\r\\n]*/\n");
    let mut calls = vec![];
    for (i, tool) in tools.iter().enumerate() {
        let f = &tool["function"];
        if choice["function"]["name"]
            .as_str()
            .is_some_and(|name| f["name"] != name)
        {
            continue;
        }
        let empty = json!({"type":"object","properties":{}});
        let schema = f
            .get("parameters")
            .filter(|x| !x.is_null())
            .unwrap_or(&empty);
        let allowed = [
            "type",
            "properties",
            "required",
            "additionalProperties",
            "$defs",
            "title",
            "description",
            "$schema",
        ];
        if !schema
            .as_object()
            .is_some_and(|o| o.keys().all(|k| allowed.contains(&k.as_str())))
            || schema.get("type").is_some_and(|t| t != "object")
            || schema
                .get("additionalProperties")
                .is_some_and(|v| v != false)
        {
            return Ok(None);
        }
        let defs = schema.get("$defs").cloned().unwrap_or(json!({}));
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        let mut parameters = vec![];
        if required.iter().any(|key| {
            key.as_str()
                .is_none_or(|s| schema["properties"].get(s).is_none())
        }) {
            return Ok(None);
        }
        if let Some(properties) = schema["properties"].as_object() {
            for (j, (key, value)) in properties.iter().enumerate() {
                let Some((rules, name)) = parameter(value, &defs, &format!("p{i}x{j}"), key, 0)
                else {
                    return Ok(None);
                };
                grammar.push_str(&rules);
                parameters.push(format!(
                    "{name}{}",
                    if required.iter().any(|v| v == key) {
                        ""
                    } else {
                        "?"
                    }
                ));
            }
        }
        let start = literal(&format!(
            "<｜DSML｜ invoke name=\"{}\">\n",
            f["name"].as_str().ok_or("Tool name missing")?
        ));
        let end = literal("</｜DSML｜ invoke>\n");
        grammar.push_str(&format!(
            "call{i}: {start} {} {end}\n",
            if parameters.is_empty() {
                "\"\\n\"".into()
            } else {
                parameters.join(" ")
            }
        ));
        calls.push(format!("call{i}"));
    }
    if calls.is_empty() {
        return Ok(None);
    }
    grammar.push_str(&format!(
        "call: {}\ncalls: {} call{} {}\n",
        calls.join(" | "),
        literal("\n\n<｜DSML｜ calls>\n"),
        if parallel { "+" } else { "" },
        literal("</｜DSML｜ calls>")
    ));
    let forced = choice == "required" || choice.is_object();
    grammar.push_str(&format!(
        "plain: /{}/\n",
        text_without("\n\n<｜DSML｜ calls>")
    ));
    let content = if forced { "calls" } else { "plain calls?" };
    if thinking {
        grammar.push_str(&format!(
            "thought[suffix=\"</think>\"]: /(?s:.*)/\nstart: thought ws {content}\n"
        ));
    } else {
        grammar.push_str(&format!("start: {content}\n"));
    }
    Ok(Some(grammar))
}
