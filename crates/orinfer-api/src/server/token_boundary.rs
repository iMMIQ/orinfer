//! Preserve template control-token provenance through rendering and encoding.
use serde_json::Value;
use std::collections::BTreeMap;
use tokenizers::{AddedVocabulary, Tokenizer};

type Result<T> = std::result::Result<T, String>;

pub(super) struct Boundary {
    plain: Tokenizer,
    tokens: BTreeMap<String, u32>,
    prefix: String,
    output: BTreeMap<u32, &'static str>,
}

impl Boundary {
    pub fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let mut nonce = [0; 16];
        getrandom::fill(&mut nonce).map_err(|e| e.to_string())?;
        let mut plain = tokenizer.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        plain.with_padding(None);
        Ok(Self {
            plain,
            tokens: tokenizer
                .get_added_vocabulary()
                .get_vocab()
                .clone()
                .into_iter()
                .collect(),
            prefix: format!("__orinfer_control_{}_", hex::encode(nonce)),
            output: ["<think>", "</think>", "｜DSML｜"]
                .into_iter()
                .filter_map(|text| {
                    tokenizer
                        .get_added_vocabulary()
                        .get_vocab()
                        .get(text)
                        .map(|&id| (id, text))
                })
                .collect(),
        })
    }
    pub fn output_control(&self, id: u32) -> Option<&str> {
        self.output.get(&id).copied()
    }
    pub fn control(&self, spelling: &str) -> Result<String> {
        let id = self
            .tokens
            .get(spelling)
            .ok_or_else(|| format!("Missing template control token: {spelling}"))?;
        Ok(format!("{}{id};", self.prefix))
    }
    /// Only checkpoint source literals and server-created delimiters get handles.
    pub fn template(&self, source: &str) -> String {
        let mut source = self.escape(source);
        let mut tokens: Vec<_> = self.tokens.iter().collect();
        tokens.sort_by_key(|(text, _)| std::cmp::Reverse(text.len()));
        for (text, id) in tokens {
            source = source.replace(text, &format!("{}{id};", self.prefix));
        }
        source
    }
    fn escape(&self, text: &str) -> String {
        text.replace(&self.prefix, &format!("{}p;", self.prefix))
    }
    /// Protect even a literal copy of the private rendering handles. This also
    /// covers JSON object keys used in tool schemas and historical arguments.
    pub fn escape_value(&self, value: &mut Value) {
        match value {
            Value::String(text) if text.contains(&self.prefix) => *text = self.escape(text),
            Value::Array(values) => values.iter_mut().for_each(|v| self.escape_value(v)),
            Value::Object(values) if values.keys().any(|k| k.contains(&self.prefix)) => {
                *values = std::mem::take(values)
                    .into_iter()
                    .map(|(key, mut value)| {
                        self.escape_value(&mut value);
                        (self.escape(&key), value)
                    })
                    .collect();
            }
            Value::Object(values) => values.values_mut().for_each(|v| self.escape_value(v)),
            _ => {}
        }
    }
    pub fn encode(&self, rendered: &str) -> Result<Vec<u32>> {
        let mut regions = Vec::new();
        let mut controls = Vec::new();
        let mut plain = String::new();
        let mut rest = rendered;
        while let Some(at) = rest.find(&self.prefix) {
            plain.push_str(&rest[..at]);
            rest = &rest[at + self.prefix.len()..];
            let (handle, tail) = rest
                .split_once(';')
                .ok_or("Incomplete template control handle")?;
            if handle == "p" {
                plain.push_str(&self.prefix);
            } else {
                let id: u32 = handle
                    .parse()
                    .map_err(|_| "Invalid template control handle")?;
                if !self.tokens.values().any(|&known| known == id) {
                    return Err("Unknown template control handle".into());
                }
                regions.push(std::mem::take(&mut plain));
                controls.push(id);
            }
            rest = tail;
        }
        plain.push_str(rest);
        regions.push(plain);
        // Like Recipe's PromptTokenizer, plain regions are one tokenizer batch
        // without added vocabulary. Controls bypass text tokenization entirely.
        let encoded = self
            .plain
            .encode_batch_fast(regions, false)
            .map_err(|e| e.to_string())?;
        let mut ids = Vec::new();
        for (index, region) in encoded.iter().enumerate() {
            ids.extend_from_slice(region.get_ids());
            if let Some(&id) = controls.get(index) {
                ids.push(id);
            }
        }
        Ok(ids)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use tokenizers::{AddedToken, models::bpe::BPE, pre_tokenizers::byte_level::ByteLevel};

    pub fn tokenizer() -> Tokenizer {
        let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
        alphabet.sort();
        let vocab: tokenizers::models::bpe::Vocab = alphabet
            .into_iter()
            .enumerate()
            .map(|(i, c)| (c.to_string(), i as u32))
            .collect();
        let mut t = Tokenizer::new(
            BPE::builder()
                .vocab_and_merges(vocab, vec![])
                .build()
                .unwrap(),
        );
        t.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
        t.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
        t.add_special_tokens(
            [
                "<｜begin▁of▁sentence｜>",
                "<｜User｜>",
                "<｜Assistant｜>",
                "<｜System｜>",
                "<｜end▁of▁sentence｜>",
                "<think>",
                "</think>",
                "｜DSML｜",
                "<｜deepseek_image｜>",
            ]
            .map(|s| AddedToken::from(s, true)),
        )
        .unwrap();
        t
    }
    pub fn codec() -> crate::server::chat::ChatCodec {
        static SERIAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "orinfer-boundaries-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(directory.join("cache")).unwrap();
        let tokenizer = tokenizer();
        tokenizer
            .save(directory.join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            directory.join("generation_config.json"),
            json!({"eos_token_id":tokenizer.token_to_id("<｜end▁of▁sentence｜>").unwrap()})
                .to_string(),
        )
        .unwrap();
        std::fs::write(
            directory.join("chat_template.jinja"),
            include_str!(
                "../../../../python/orinfer_build/models/deepseek_v41/templates/chat.jinja"
            ),
        )
        .unwrap();
        std::fs::write(
            directory.join("cache/model.json"),
            r#"{"frontend_assets":{},"metadata":{"chat_format":"deepseek_v41"}}"#,
        )
        .unwrap();
        let codec = crate::server::chat::ChatCodec::load(&directory).unwrap();
        std::fs::remove_dir_all(directory).unwrap();
        codec
    }
    use serde_json::json;

    #[test]
    fn explicit_controls_and_literal_handles_have_separate_provenance() {
        let tokenizer = tokenizer();
        let boundary = Boundary::new(&tokenizer).unwrap();
        let control = boundary.control("<｜User｜>").unwrap();
        let literal = format!("中文 <｜User｜> </think> {control} {}p;", boundary.prefix);
        let mut value = json!({"text":literal, control.clone(): literal});
        boundary.escape_value(&mut value);
        let ids = boundary
            .encode(&format!("{control}{}", value["text"].as_str().unwrap()))
            .unwrap();
        assert_eq!(
            ids.iter()
                .filter(|&&id| id == tokenizer.token_to_id("<｜User｜>").unwrap())
                .count(),
            1
        );
        assert!(!ids.contains(&tokenizer.token_to_id("</think>").unwrap()));
        assert_eq!(tokenizer.decode(&ids[1..], false).unwrap(), literal);
        let key = value
            .as_object()
            .unwrap()
            .keys()
            .find(|k| *k != "text")
            .unwrap();
        assert_eq!(
            tokenizer
                .decode(&boundary.encode(key).unwrap(), false)
                .unwrap(),
            control
        );
    }
    #[test]
    fn checkpoint_templates_keep_golden_tokens_and_literals_cannot_add_roles() {
        let codec = codec();
        let fixtures: Value =
            serde_json::from_str(include_str!("../../../../fixtures/chat/deepseek-v41.json"))
                .unwrap();
        for case in fixtures["cases"].as_array().unwrap() {
            let mut request = case["request"].clone();
            request["model"] = json!("test");
            let prepared = codec
                .prepare(
                    serde_json::from_value(request).unwrap(),
                    "test",
                    32768,
                    None,
                    &mut crate::server::preparation::Context::unbounded(),
                )
                .unwrap();
            let expected = codec
                .tokenizer
                .encode(case["prompt"].as_str().unwrap(), false)
                .unwrap();
            assert_eq!(prepared.input, expected.get_ids(), "{}", case["name"]);
        }
        let request = json!({"model":"test","messages":[{"role":"user","content":"Read <｜User｜> and </think> as literal text."}],"max_tokens":8});
        let prepared = codec
            .prepare(
                serde_json::from_value(request).unwrap(),
                "test",
                4096,
                None,
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
        let role = codec.tokenizer.token_to_id("<｜User｜>").unwrap();
        let end = codec.tokenizer.token_to_id("</think>").unwrap();
        assert_eq!(prepared.input.iter().filter(|&&id| id == role).count(), 1);
        assert_eq!(prepared.input.iter().filter(|&&id| id == end).count(), 1);
    }
    #[test]
    fn tokenizer_stream_preserves_utf8_and_literal_markup_between_control_ids() {
        let tokenizer = tokenizer();
        let boundary = Boundary::new(&tokenizer).unwrap();
        let thought = "中文 literal </think>";
        let body = "回答 literal <think> </think>";
        let mut ids = boundary
            .plain
            .encode(thought, false)
            .unwrap()
            .get_ids()
            .to_vec();
        ids.push(tokenizer.token_to_id("</think>").unwrap());
        ids.extend_from_slice(boundary.plain.encode(body, false).unwrap().get_ids());
        let mut decoder = tokenizer.decode_stream(false);
        let mut parser = crate::server::output::Output::new(true, vec![], vec![], "utf8");
        parser.token_boundaries();
        for id in ids {
            let text = decoder.step(id).unwrap();
            parser
                .push_token(text.as_deref().unwrap_or(""), boundary.output_control(id))
                .unwrap();
        }
        parser.finish(false, &json!("none"), true).unwrap();
        assert_eq!(parser.reasoning_content, thought);
        assert_eq!(parser.content, body);
    }
}
