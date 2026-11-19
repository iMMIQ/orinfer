//! Qwen3 yes/no scoring and the document reranking protocol.
use orinfer_engine::{
    execution::LoadOptions,
    reranking::{Model, Score},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Instant};
use tokenizers::Tokenizer;
type Result<T> = std::result::Result<T, String>;
const PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
const SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";
struct Codec {
    tokenizer: Tokenizer,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    hashes: BTreeMap<String, String>,
    context: usize,
    ids: [u32; 2],
}
impl Codec {
    fn load(root: &Path) -> Result<Self> {
        let frontend = crate::encoder_service::load_frontend_descriptor(root)?;
        Self::new(frontend.tokenizer, frontend.hashes, frontend.context)
    }
    fn new(tokenizer: Tokenizer, hashes: BTreeMap<String, String>, context: usize) -> Result<Self> {
        let tokenizer = crate::encoder_service::unbounded_tokenizer(tokenizer)?;
        let ids = [
            tokenizer.token_to_id("no").ok_or("Missing no token")?,
            tokenizer.token_to_id("yes").ok_or("Missing yes token")?,
        ];
        let encode = |text: &str| {
            tokenizer
                .encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| e.to_string())
        };
        let (prefix, suffix) = (encode(PREFIX)?, encode(SUFFIX)?);
        Ok(Self {
            tokenizer,
            prefix,
            suffix,
            hashes,
            context,
            ids,
        })
    }
    fn prepare(&self, instruction: &str, query: &str, document: &str) -> Result<Vec<u32>> {
        let body = format!("<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}");
        let encoded = self
            .tokenizer
            .encode(body, false)
            .map_err(|e| e.to_string())?;
        if self.prefix.len() + encoded.len() + self.suffix.len() > self.context {
            return Err("Reranking pair exceeds model context".into());
        }
        Ok(self
            .prefix
            .iter()
            .chain(encoded.get_ids())
            .chain(&self.suffix)
            .copied()
            .collect())
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default, deserialize_with = "present")]
    model: Option<String>,
    query: String,
    documents: Vec<Value>,
    #[serde(default, deserialize_with = "present")]
    instruction: Option<String>,
    #[serde(default, deserialize_with = "present")]
    top_n: Option<usize>,
    #[serde(default)]
    return_documents: bool,
}
fn present<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}
fn request(value: Value, model_id: &str) -> Result<Request> {
    if !value.is_object() {
        return Err("Rerank request must be an object".into());
    }
    let request: Request = serde_json::from_value(value).map_err(|e| e.to_string())?;
    if request.model.as_deref().is_some_and(|m| m != model_id) {
        return Err("Unknown reranking model".into());
    }
    if request.query.trim().is_empty() || request.query.len() > 1024 * 1024 {
        return Err("Rerank query must be nonempty text within 1 MiB".into());
    }
    if !(1..=128).contains(&request.documents.len()) {
        return Err("Rerank documents must contain 1..128 items".into());
    }
    if request
        .instruction
        .as_ref()
        .is_some_and(|s| s.len() > 1024 * 1024)
    {
        return Err("Rerank instruction exceeds 1 MiB".into());
    }
    for document in &request.documents {
        let text = document_text(document)?;
        if text.trim().is_empty() || text.len() > 4 * 1024 * 1024 {
            return Err("Rerank document is empty or exceeds 4 MiB".into());
        }
    }
    if request.top_n.is_some_and(|n| !(1..=128).contains(&n)) {
        return Err("top_n must be in 1..128".into());
    }
    Ok(request)
}
fn document_text(document: &Value) -> Result<&str> {
    document
        .as_str()
        .or_else(|| document.get("text").and_then(Value::as_str))
        .ok_or_else(|| "Rerank document must be text or an object with text".into())
}
fn ranked(request: &Request, scores: &[Score]) -> Value {
    let mut indices: Vec<_> = (0..scores.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .logit_difference
            .total_cmp(&scores[a].logit_difference)
            .then(a.cmp(&b))
    });
    let results: Vec<_> = indices
        .into_iter()
        .take(request.top_n.unwrap_or(scores.len()))
        .map(|index| {
            let mut row = json!({"index":index,"relevance_score":scores[index].relevance_score});
            if request.return_documents {
                row["document"] = if request.documents[index].is_string() {
                    json!({"text":request.documents[index]})
                } else {
                    request.documents[index].clone()
                };
            }
            row
        })
        .collect();
    json!({"results":results})
}
fn response(
    codec: &Codec,
    model: &mut Model,
    value: Value,
    id: &str,
) -> Result<(Value, Vec<Score>)> {
    let request = request(value, id)?;
    let inputs: Vec<_> = request
        .documents
        .iter()
        .map(|doc| {
            codec.prepare(
                request.instruction.as_deref().unwrap_or(INSTRUCTION),
                &request.query,
                document_text(doc)?,
            )
        })
        .collect::<Result<_>>()?;
    let total: usize = inputs.iter().map(Vec::len).sum();
    let scores = model.score_many(&inputs.iter().map(Vec::as_slice).collect::<Vec<_>>())?;
    let mut result = ranked(&request, &scores);
    result["model"] = json!(id);
    result["usage"] = json!({"prompt_tokens":total,"total_tokens":total});
    Ok((result, scores))
}
fn load(root: &Path, options: LoadOptions) -> Result<(Codec, Model)> {
    let (codec, model) =
        crate::server::startup::load_parallel(|| Codec::load(root), || Model::load(root, options))?;
    if &codec.hashes != model.frontend_assets()
        || codec.ids != [model.spec().false_token_id, model.spec().true_token_id]
    {
        return Err("Reranker frontend identity changed during loading".into());
    }
    Ok((codec, model))
}
pub fn run_model(root: &Path, requests: &Path, options: LoadOptions) -> Result<Value> {
    let request: Value =
        serde_json::from_slice(&std::fs::read(requests).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let (codec, mut model) = load(root, options)?;
    let id = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("qwen3-reranker-0.6b")
        .to_owned();
    let started = Instant::now();
    let (response, scores) = response(&codec, &mut model, request, &id)?;
    Ok(
        json!({"response":response,"scores":scores,"inference_s":started.elapsed().as_secs_f64(),"load":model.statistics()}),
    )
}
struct RerankingBackend {
    codec: Codec,
    model: Model,
    id: String,
}
impl crate::encoder_service::Backend for RerankingBackend {
    fn respond(&mut self, request: Value) -> Result<Value> {
        response(&self.codec, &mut self.model, request, &self.id).map(|(value, _)| value)
    }
}
pub(crate) fn serve(settings: crate::ServerConfig) -> Result<()> {
    crate::encoder_service::serve(settings, vec!["/v1/rerank", "/rerank"], |settings| {
        let (codec, model) = load(
            &settings.model_dir,
            LoadOptions {
                cuda_graph: settings.cuda_graph,
                verify_weights: settings.verify_weights,
                load_workers: settings.load_workers,
                ..Default::default()
            },
        )?;
        Ok(RerankingBackend {
            codec,
            model,
            id: settings.model.clone(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranking_uses_unsaturated_logits_and_stable_input_indices() {
        let value = json!({"query":"q","documents":["a",{"text":"b","id":4},"a"],"return_documents":true,"top_n":2});
        let r = request(value, "m").unwrap();
        let scores = [1000.0, 1001.0, 1000.0].map(|logit_difference| Score {
            no_logit: 0.0,
            yes_logit: logit_difference as f32,
            logit_difference,
            relevance_score: 1.0,
        });
        let result = ranked(&r, &scores);
        assert_eq!(result["results"][0]["index"], 1);
        assert_eq!(result["results"][0]["document"]["id"], 4);
        assert_eq!(result["results"][1]["index"], 0);
    }
    #[test]
    fn invalid_requests_fail_before_gpu_execution() {
        for value in [
            json!({"query":"","documents":["d"]}),
            json!({"query":"q","documents":[]}),
            json!({"query":"q","documents":["d"],"top_n":0}),
            json!({"query":"q","documents":[{}]}),
            json!({"query":"q","documents":["d"],"model":"other"}),
            json!({"query":"q","documents":["d"],"model":null}),
            json!({"query":"q","documents":["d"],"top_n":null}),
            json!({"query":"q","documents":["d"],"instruction":null}),
            json!({"query":"q","documents":["d"],"return_documents":null}),
            json!({"query":"q","documents":["d"],"unexpected":true}),
            json!(["m", "q", ["d"]]),
        ] {
            assert!(request(value, "m").is_err());
        }
    }
    #[test]
    #[ignore = "Requires the original Qwen3-Reranker tokenizer; CPU only"]
    fn checkpoint_reranker_token_ids_match_transformers() {
        let root =
            std::path::PathBuf::from(std::env::var_os("ORINFER_REFERENCE_CHECKPOINT").unwrap());
        let codec = Codec::new(
            Tokenizer::from_file(root.join("tokenizer.json")).unwrap(),
            BTreeMap::new(),
            32768,
        )
        .unwrap();
        let expected: Value =
            serde_json::from_slice(&std::fs::read(root.join("reference-token-ids.json")).unwrap())
                .unwrap();
        assert_eq!(
            codec
                .prepare(
                    INSTRUCTION,
                    "What is the capital of China?",
                    "The capital of China is Beijing."
                )
                .unwrap(),
            serde_json::from_value::<Vec<u32>>(expected["tokens"].clone()).unwrap()
        );
        assert_eq!(
            codec.ids,
            serde_json::from_value::<[u32; 2]>(expected["ids"].clone()).unwrap()
        );
    }
}
