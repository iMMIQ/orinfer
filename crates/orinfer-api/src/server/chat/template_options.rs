//! Wire aliases and template variables share the generation/parser controls.
use super::{ChatRequest, Result};

pub(in crate::server) const THINKING_FIELDS: &[&str] =
    &["enable_thinking", "thinking", "reasoning_effort"];
pub(in crate::server) const BUDGET_FIELDS: &[&str] = &["thinking_token_budget", "thinking_budget"];

impl ChatRequest {
    pub(super) fn template_bool(&self, key: &str) -> Result<Option<bool>> {
        self.chat_template_kwargs
            .get(key)
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| format!("chat_template_kwargs.{key}: Expected a boolean"))
            })
            .transpose()
    }
    /// Explicit top-level controls win over template aliases. Normalizing
    /// before rendering also keeps reasoning extraction and grammar in sync.
    pub(in crate::server) fn resolve_template_options(&mut self) -> Result<()> {
        let enabled = self.template_bool("enable_thinking")?;
        let thinking = self.template_bool("thinking")?;
        self.template_bool("preserve_thinking")?;
        if self.enable_thinking.is_none() {
            if enabled.zip(thinking).is_some_and(|(a, b)| a != b) {
                return Err("chat_template_kwargs.thinking: Conflicts with enable_thinking".into());
            }
            self.enable_thinking = enabled.or(thinking);
        }
        if let Some(value) = self
            .chat_template_kwargs
            .get("reasoning_effort")
            .filter(|v| !v.is_null())
        {
            let value = value
                .as_str()
                .ok_or("chat_template_kwargs.reasoning_effort: Expected a string")?;
            if self.reasoning_effort.is_none() {
                self.reasoning_effort = Some(value.to_owned());
            }
        }
        let mut budget = None;
        for &key in BUDGET_FIELDS {
            if let Some(value) = self.chat_template_kwargs.get(key).filter(|v| !v.is_null()) {
                let value = value
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| {
                        format!("chat_template_kwargs.{key}: Expected a nonnegative integer")
                    })?;
                if self.thinking_token_budget.is_none() && budget.is_some_and(|b| b != value) {
                    return Err("chat_template_kwargs.thinking_budget: Conflicts with thinking_token_budget".into());
                }
                budget = Some(value);
            }
        }
        self.thinking_token_budget = self.thinking_token_budget.or(budget);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::server::{chat::ChatCodec, preparation::Context};
    use serde_json::json;
    use tokenizers::{
        Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace,
    };
    #[test]
    fn custom_variables_and_thinking_aliases_reach_the_rendered_tokens() {
        let directory =
            std::env::temp_dir().join(format!("orinfer-template-kwargs-{}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();
        let vocab = [
            "[UNK]", "one", "two", "on", "off", "kept", "dropped", "xhigh",
        ];
        let model = WordLevel::builder()
            .vocab(
                vocab
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v.to_string(), i as u32))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .save(directory.join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(
            directory.join("generation_config.json"),
            r#"{"eos_token_id":0}"#,
        )
        .unwrap();
        std::fs::write(directory.join("chat_template.jinja"),"{{ messages[0].content }} {{ custom_label }} {% if enable_thinking %}on{% else %}off{% endif %} {% if thinking %}on{% else %}off{% endif %} {% if preserve_thinking %}kept{% else %}dropped{% endif %} {{ reasoning_effort }}").unwrap();
        let codec = ChatCodec::load(&directory).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        for override_mode in [None, Some(false)] {
            let mut body = json!({"model":"test","messages":[{"role":"user","content":"one"}],"max_tokens":1,
                "chat_template_kwargs":{"enable_thinking":true,"preserve_thinking":true,"custom_label":"two","messages":[{"role":"user","content":"wrong"}]}});
            if let Some(value) = override_mode {
                body["enable_thinking"] = json!(value);
            }
            let prepared = codec
                .prepare(
                    serde_json::from_value(body).unwrap(),
                    "test",
                    64,
                    None,
                    &mut Context::unbounded(),
                )
                .unwrap();
            let mode = if override_mode == Some(false) { 4 } else { 3 };
            assert_eq!(prepared.input, [1, 2, mode, mode, 5, 7]);
            assert_eq!(prepared.thinking, override_mode.unwrap_or(true));
        }
    }
}
