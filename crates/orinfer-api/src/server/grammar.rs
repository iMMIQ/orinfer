//! Shared tokenizer trie and request-private llguidance grammars. No tokenizer,
//! grammar state or sampled history is stored in the GPU prefix cache.
use llguidance::{
    Matcher, ParserFactory,
    api::TopLevelGrammar,
    toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie},
};
use orinfer_engine::sampling::Constraint;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};

type Result<T> = std::result::Result<T, String>;
pub struct Factory {
    raw: Vec<u8>,
    eos: Vec<u32>,
    bytes: OnceLock<Result<Arc<Vec<Vec<u8>>>>>,
    parser: OnceLock<Result<ParserFactory>>,
}
impl Factory {
    pub fn new(raw: Vec<u8>, eos: Vec<u32>) -> Self {
        Self {
            raw,
            eos,
            bytes: OnceLock::new(),
            parser: OnceLock::new(),
        }
    }
    pub fn token_table(&self) -> Result<Arc<Vec<Vec<u8>>>> {
        self.bytes
            .get_or_init(|| {
                let raw = serde_json::from_slice(&self.raw).map_err(|e| e.to_string())?;
                let tokens =
                    llguidance::token_bytes_from_tokenizer_json(&raw).map_err(|e| e.to_string())?;
                Ok(Arc::new(tokens))
            })
            .as_ref()
            .cloned()
            .map_err(Clone::clone)
    }
    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        let tokens = self.token_table()?;
        let bytes = tokens.get(id as usize).ok_or("Token outside tokenizer")?;
        Ok(bytes
            .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
            .unwrap_or(bytes)
            .to_vec())
    }
    fn factory(&self) -> Result<&ParserFactory> {
        self.parser
            .get_or_init(|| {
                let bytes = self.token_table()?;
                let info = TokRxInfo::new(bytes.len() as u32, self.eos[0]);
                let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &bytes)));
                let mut factory = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
                factory.quiet();
                factory.limits_mut().verbose_errors = false;
                factory.limits_mut().max_lexer_states = 64_000;
                factory.limits_mut().max_grammar_size = 100_000;
                Ok(factory)
            })
            .as_ref()
            .map_err(Clone::clone)
    }
    pub fn compile(&self, grammar: String, param: &str) -> Result<Box<dyn Constraint>> {
        compile(self.factory()?, grammar, &self.eos, param)
    }
}
fn compile(
    factory: &ParserFactory,
    grammar: String,
    eos: &[u32],
    param: &str,
) -> Result<Box<dyn Constraint>> {
    let mut parser = factory
        .create_parser(TopLevelGrammar::from_lark(grammar))
        .map_err(|e| format!("{param}: {e}"))?;
    let warnings = parser.grammar_warnings();
    if !warnings.is_empty() {
        return Err(format!(
            "{param}: Unsupported schema constraints: {}",
            warnings.join("; ")
        ));
    }
    let mut matcher = Matcher::new(Ok(parser));
    matcher
        .compute_mask()
        .map_err(|e| format!("{param}: {e}"))?;
    Ok(Box::new(Grammar {
        matcher,
        eos: eos.to_vec(),
    }))
}
struct Grammar {
    matcher: Matcher,
    eos: Vec<u32>,
}
impl Constraint for Grammar {
    fn fork(&self) -> Box<dyn Constraint> {
        Box::new(Self {
            matcher: self.matcher.deep_clone(),
            eos: self.eos.clone(),
        })
    }
    fn mask(&mut self) -> Result<Vec<u32>> {
        let mut mask = self
            .matcher
            .compute_mask_or_eos()
            .map_err(|e| e.to_string())?;
        if mask.is_allowed(self.eos[0]) {
            for &eos in &self.eos {
                mask.allow_token(eos);
            }
        }
        Ok(mask.as_slice().to_vec())
    }
    fn consume(&mut self, token: u32) -> Result<()> {
        let token = if self.eos.contains(&token) {
            self.eos[0]
        } else {
            token
        };
        self.matcher.consume_token(token).map_err(|e| e.to_string())
    }
    fn finished(&self) -> bool {
        self.matcher.is_stopped()
    }
}

/// Regular language excluding a delimiter, including text containing partial
/// delimiter prefixes. No lookaround or regex backtracking is required.
fn text_without(marker: &str) -> String {
    let mut branches = vec![];
    let mut tails = vec![];
    let mut prefix = String::new();
    for ch in marker.chars() {
        if !prefix.is_empty() {
            tails.push(prefix.clone());
        }
        let escaped = match ch {
            '/' => "\\/".into(),
            '\\' => "\\\\".into(),
            c => c.to_string(),
        };
        branches.push(format!("{prefix}[^{escaped}]"));
        prefix.push_str(&escaped);
    }
    format!("({})*({})?", branches.join("|"), tails.join("|"))
}
fn check_schema(schema: &Value, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err("Schema nesting exceeds 64".into());
    }
    match schema {
        Value::Bool(_) => Ok(()),
        Value::Object(obj) => {
            if let Some(reference) = obj.get("$ref").and_then(Value::as_str)
                && !reference.starts_with('#')
            {
                return Err("Only local schema references are supported".into());
            }
            if obj.contains_key("x-guidance") {
                return Err("x-guidance schema extensions are unsupported".into());
            }
            for key in ["properties", "$defs", "definitions", "patternProperties"] {
                if let Some(children) = obj.get(key) {
                    for child in children
                        .as_object()
                        .ok_or("Schema properties/definitions must be objects")?
                        .values()
                    {
                        check_schema(child, depth + 1)?;
                    }
                }
            }
            for key in ["items", "additionalProperties", "not", "if", "then", "else"] {
                if let Some(child) = obj.get(key) {
                    check_schema(child, depth + 1)?;
                }
            }
            for key in ["anyOf", "allOf", "oneOf", "prefixItems"] {
                if let Some(children) = obj.get(key) {
                    for child in children
                        .as_array()
                        .ok_or("Schema alternatives must be arrays")?
                    {
                        check_schema(child, depth + 1)?;
                    }
                }
            }
            Ok(())
        }
        _ => Err("Schema must be an object or boolean".into()),
    }
}
pub fn output_schema(format: Option<&Value>) -> Result<Option<Value>> {
    let Some(format) = format else {
        return Ok(None);
    };
    match format["type"].as_str() {
        Some("text") => Ok(None),
        Some("json_object") => Ok(Some(json!({"type":"object"}))),
        Some("json_schema") => {
            let spec = &format["json_schema"];
            let name = spec["name"]
                .as_str()
                .ok_or("response_format.json_schema.name: Missing schema name")?;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            {
                return Err("response_format.json_schema.name: Invalid schema name".into());
            }
            let schema = spec
                .get("schema")
                .ok_or("response_format.json_schema.schema: Missing schema")?;
            validate_schema(schema, "response_format.json_schema.schema")?;
            if spec
                .get("strict")
                .is_some_and(|s| !s.is_null() && !s.is_boolean())
            {
                return Err("response_format.json_schema.strict: Expected boolean".into());
            }
            Ok(Some(schema.clone()))
        }
        _ => Err("response_format.type: Expected text, json_object or json_schema".into()),
    }
}
pub fn validate_schema(schema: &Value, param: &str) -> Result<()> {
    if schema.to_string().len() > 128 * 1024 {
        return Err(format!("{param}: Schema exceeds 128 KiB"));
    }
    check_schema(schema, 0).map_err(|e| format!("{param}: {e}"))
}

pub fn recipe(
    schema: Option<&Value>,
    tools: &[Value],
    choice: &Value,
    parallel: bool,
    thinking: bool,
) -> Result<Option<String>> {
    let forced = choice == "required" || choice.is_object();
    let strict = tools.iter().any(|t| t["function"]["strict"] == true);
    if schema.is_none() && !(forced || strict || !parallel && !tools.is_empty()) {
        return Ok(None);
    }
    let mut grammar = String::from("ws: /[ \\t\\r\\n]*/\n");
    let mut content = String::new();
    if let Some(schema) = schema {
        grammar.push_str(&format!("json: %json {}\n", schema));
        content.push_str("json");
    } else {
        grammar.push_str(&format!("plain: /{}/\n", text_without("<tool_call>")));
        content.push_str("plain");
    }
    if !tools.is_empty() {
        let mut calls = vec![];
        for (i, t) in tools.iter().enumerate() {
            let function = &t["function"];
            let mut params = function
                .get("parameters")
                .filter(|p| !p.is_null())
                .cloned()
                .unwrap_or(json!({"type":"object","properties":{},"additionalProperties":false}));
            validate_schema(&params, &format!("tools[{i}].function.parameters"))?;
            if params.get("type").is_some_and(|t| {
                t != "object"
                    && !t
                        .as_array()
                        .is_some_and(|a| a.iter().any(|t| t == "object"))
            }) {
                return Err(format!(
                    "tools[{i}].function.parameters: Arguments must be an object"
                ));
            }
            params["type"] = json!("object");
            // Keep each parameter schema at its own root, so local $ref/$defs
            // retain their meaning instead of resolving against the envelope.
            let name = serde_json::to_string(&function["name"].to_string()).unwrap();
            grammar.push_str(&format!(
                r#"call{i}: head ws "{{" ws "\"name\"" ws ":" ws {name} ws "," ws "\"arguments\"" ws ":" ws args{i} ws "}}" ws "</tool_call>"
args{i}: %json {params}
"#
            ));
            calls.push(format!("call{i}"));
        }
        grammar.push_str(if schema.is_some() {
            "head: \"<tool_call>\"\n"
        } else {
            "head[lazy]: /(?s:.*)/ \"<tool_call>\"\n"
        });
        grammar.push_str(&format!(
            "call: {}\ncalls: call{}\n",
            calls.join(" | "),
            if parallel { " call*" } else { "" }
        ));
        content = if schema.is_some() {
            if forced {
                "calls".into()
            } else {
                "(json | calls)".into()
            }
        } else if forced {
            format!("call{} plain", if parallel { "+" } else { "" })
        } else {
            format!("call{} plain", if parallel { "*" } else { "?" })
        };
    }
    if thinking {
        grammar.push_str("thought[suffix=\"</think>\"]: /(?s:.*)/\n");
        grammar.push_str(&format!("start: thought ws {content}\n"));
    } else {
        grammar.push_str(&format!("start: {content}\n"));
    }
    Ok(Some(grammar))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check(grammar: String, text: &str) -> Result<()> {
        let factory = ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env()).unwrap();
        let mut c = compile(&factory, grammar, &[261], "test")?;
        for b in text.bytes() {
            let mask = c.mask()?;
            if mask[b as usize / 32] & (1 << (b % 32)) == 0 {
                return Err(format!("Forbidden byte {b}"));
            }
            c.consume(b as u32)?;
        }
        let mask = c.mask()?;
        if mask[261 / 32] & (1 << (261 % 32)) == 0 {
            return Err("Not accepting EOS".into());
        }
        Ok(())
    }
    #[test]
    fn json_schema_nested_values_and_thinking() {
        let schema = json!({"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false});
        let g = recipe(Some(&schema), &[], &json!("auto"), true, false)
            .unwrap()
            .unwrap();
        assert!(check(g.clone(), "{\"x\":3}").is_ok());
        assert!(check(g.clone(), "{\"x\":\"bad\"}").is_err());
        assert!(check(g, "{\"y\":3}").is_err());
        let g = recipe(Some(&schema), &[], &json!("auto"), true, true)
            .unwrap()
            .unwrap();
        let result = check(g, "I should reason <about> this. </think>{\"x\":3}");
        assert!(result.is_ok(), "{result:?}");
    }
    #[test]
    fn tool_choice_parallel_and_optional_text_are_constrained() {
        let tools = vec![
            json!({"type":"function","function":{"name":"lookup","strict":true,"parameters":{"type":"object","properties":{"city":{"enum":["北京"]}},"required":["city"],"additionalProperties":false}}}),
        ];
        let call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"北京\"}}</tool_call>";
        let g = recipe(None, &tools, &json!("required"), false, false)
            .unwrap()
            .unwrap();
        assert!(check(g.clone(), call).is_ok());
        assert!(check(g.clone(), "plain text").is_err());
        assert!(check(g.clone(), &format!("{call}{call}")).is_err());
        assert!(check(g, &call.replace("北京", "bad")).is_err());
        let g = recipe(None, &tools, &json!("auto"), false, false)
            .unwrap()
            .unwrap();
        assert!(check(g.clone(), "Here is code <div>hello</div>").is_ok());
        assert!(check(g.clone(), "Ends in a partial marker <tool_").is_ok());
        let result = check(g.clone(), &format!("Looking up. {call} Done."));
        assert!(result.is_ok(), "{result:?}");
        assert!(check(g, &call.replace("lookup", "invalid")).is_err());
    }
    #[test]
    fn tool_local_references_resolve_against_parameter_schema() {
        let tools = vec![
            json!({"type":"function","function":{"name":"lookup","parameters":{
            "type":"object","$defs":{"city":{"type":"string","enum":["北京"]}},
            "properties":{"city":{"$ref":"#/$defs/city"}},"required":["city"],"additionalProperties":false}}}),
        ];
        let g = recipe(None, &tools, &json!("required"), false, false)
            .unwrap()
            .unwrap();
        let call = "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"city\":\"北京\"}}</tool_call>";
        assert!(check(g.clone(), call).is_ok());
        assert!(check(g, &call.replace("北京", "bad")).is_err());
    }
}
