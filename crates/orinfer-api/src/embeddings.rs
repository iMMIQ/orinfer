//! Embeddings protocol, retrieval prefixes, and Pixtral image preparation.
use base64::{Engine, engine::general_purpose::STANDARD};
use orinfer_engine::{embedding::Model, execution::LoadOptions, vision::ImageInput};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{Cursor, Read},
    path::Path,
    time::Instant,
};
use tokenizers::Tokenizer;
type Result<T> = std::result::Result<T, String>;

struct Codec {
    tokenizer: Tokenizer,
    hashes: BTreeMap<String, String>,
    context: usize,
    patches: usize,
}
struct Prepared {
    tokens: Vec<u32>,
    images: Vec<ImageInput>,
    index: Vec<i32>,
}
impl Codec {
    fn load(root: &Path) -> Result<Self> {
        let descriptor: Value = serde_json::from_slice(
            &std::fs::read(root.join("cache/model.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let expected: BTreeMap<String, String> =
            serde_json::from_value(descriptor["frontend_assets"].clone())
                .map_err(|e| e.to_string())?;
        for name in expected.keys() {
            if Path::new(name).components().count() != 1 {
                return Err("Invalid embedding frontend filename".into());
            }
        }
        let mut tokenizer =
            Tokenizer::from_file(root.join("tokenizer.json")).map_err(|e| e.to_string())?;
        tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        for (token, id) in [("[IMG]", 10), ("[IMG_BREAK]", 12), ("[IMG_END]", 13)] {
            if tokenizer.token_to_id(token) != Some(id) {
                return Err(format!("Unexpected image token {token}"));
            }
        }
        Ok(Self {
            tokenizer,
            hashes: expected,
            context: descriptor["metadata"]["max_context"]
                .as_u64()
                .ok_or("Missing embedding context")? as usize,
            patches: descriptor["metadata"]["embedding"]["max_patches"]
                .as_u64()
                .ok_or("Missing vision capacity")? as usize,
        })
    }
    fn prepare(&self, input: &Value, role: &str, local: bool) -> Result<Prepared> {
        let (text, sources): (&str, Vec<&str>) = if let Some(text) = input.as_str() {
            (text, vec![])
        } else {
            let object = input
                .as_object()
                .ok_or("Embedding input must be text or a text/images object")?;
            if object
                .keys()
                .any(|k| !["text", "images"].contains(&k.as_str()))
            {
                return Err("Unknown multimodal embedding input field".into());
            }
            let text = object.get("text").map_or(Ok(""), |v| {
                v.as_str().ok_or("Embedding text must be a string")
            })?;
            let sources = match object.get("images") {
                None => vec![],
                Some(Value::Array(v)) if v.len() <= 8 => v
                    .iter()
                    .map(|v| v.as_str().ok_or("Embedding image must be a URL/path"))
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                _ => return Err("Embedding images must be a list of at most eight URLs".into()),
            };
            (text, sources)
        };
        if text.len() > 4 * 1024 * 1024 || (text.is_empty() && sources.is_empty()) {
            return Err("Embedding input is empty or exceeds the text limit".into());
        }
        if role == "query" && !sources.is_empty() {
            return Err("Nemotron retrieval queries accept text only".into());
        }
        let mut images = Vec::new();
        let mut prompt = if role == "query" {
            "query: ".to_string()
        } else {
            "passage: ".to_string()
        };
        for source in sources {
            let image = preprocess(&load_image(source, local)?, self.patches)?;
            if !images.is_empty() {
                prompt.push(' ');
            }
            for y in 0..image.grid_height / 2 {
                for _ in 0..image.grid_width / 2 {
                    prompt.push_str("[IMG]");
                }
                prompt.push_str(if y + 1 == image.grid_height / 2 {
                    "[IMG_END]"
                } else {
                    "[IMG_BREAK]"
                });
            }
            images.push(image);
        }
        if !text.is_empty() {
            if !images.is_empty() {
                prompt.push(' ');
            }
            prompt.push_str(&text.replace('[', "[\u{200c}").replace('<', "<\u{200c}"));
        }
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| e.to_string())?;
        let tokens = encoded.get_ids().to_vec();
        if tokens.is_empty() || tokens.len() > self.context {
            return Err("Embedding input exceeds model context".into());
        }
        let mut feature = 0;
        let index = tokens
            .iter()
            .map(|&t| {
                if t == 10 {
                    let i = feature;
                    feature += 1;
                    i
                } else {
                    -1
                }
            })
            .collect();
        Ok(Prepared {
            tokens,
            images,
            index,
        })
    }
}

fn load_image(source: &str, local: bool) -> Result<Vec<u8>> {
    const MAX: usize = 24 * 1024 * 1024;
    if let Some(data) = source.strip_prefix("data:") {
        let (kind, raw) = data.split_once(',').ok_or("Invalid image data URI")?;
        if !["image/png;base64", "image/jpeg;base64", "image/webp;base64"].contains(&kind)
            || raw.len() > MAX.div_ceil(3) * 4
        {
            return Err("Unsupported or oversized embedding image".into());
        }
        return STANDARD.decode(raw).map_err(|e| e.to_string());
    }
    let reader: Box<dyn Read> = if source.starts_with("https://") || source.starts_with("http://") {
        Box::new(
            reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(20))
                .build()
                .map_err(|e| e.to_string())?
                .get(source)
                .send()
                .map_err(|e| e.to_string())?
                .error_for_status()
                .map_err(|e| e.to_string())?,
        )
    } else if local {
        Box::new(std::fs::File::open(source).map_err(|e| e.to_string())?)
    } else {
        return Err("Embedding images require HTTP(S) or a base64 data URI".into());
    };
    let mut bytes = Vec::new();
    reader
        .take((MAX + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX {
        return Err("Embedding image exceeds 24 MiB".into());
    }
    Ok(bytes)
}

fn preprocess(bytes: &[u8], max_patches: usize) -> Result<ImageInput> {
    let mut reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(16384);
    limits.max_image_height = Some(16384);
    limits.max_alloc = Some(256 * 1024 * 1024);
    reader.limits(limits);
    let rgb = reader.decode().map_err(|e| e.to_string())?.to_rgb8();
    let (height, width) = (rgb.height() as usize, rgb.width() as usize);
    if height == 0 || width == 0 {
        return Err("Empty embedding image".into());
    }
    let ratio = (height.max(width) as f64 / 1120.0).max(1.0);
    let h = ((height as f64 / ratio).floor() as usize).div_ceil(28) * 28;
    let w = ((width as f64 / ratio).floor() as usize).div_ceil(28) * 28;
    let (gh, gw) = (h / 14, w / 14);
    if gh * gw > max_patches {
        return Err("Image exceeds compiled embedding patch capacity".into());
    }
    let resized = image::imageops::resize(
        &rgb,
        w as u32,
        h as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let mean = [0.48145466f32, 0.4578275, 0.40821073];
    let std = [0.26862954f32, 0.261_302_6, 0.275_777_1];
    let mut pixels = Vec::with_capacity(gh * gw * 588);
    for py in 0..gh {
        for px in 0..gw {
            for c in 0..3 {
                for y in 0..14 {
                    for x in 0..14 {
                        let v = resized.get_pixel((px * 14 + x) as u32, (py * 14 + y) as u32)[c]
                            as f32
                            / 255.0;
                        pixels.push((v - mean[c]) / std[c]);
                    }
                }
            }
        }
    }
    Ok(ImageInput {
        rgb: Vec::new(),
        grid_height: gh,
        grid_width: gw,
        pixels,
    })
}

fn response(
    codec: &Codec,
    model: &mut Model,
    request: Value,
    model_id: &str,
    local: bool,
) -> Result<Value> {
    let object = request
        .as_object()
        .ok_or("Embedding request must be an object")?;
    for key in object.keys() {
        if ![
            "input",
            "input_type",
            "model",
            "dimensions",
            "encoding_format",
            "user",
        ]
        .contains(&key.as_str())
        {
            return Err(format!("Unknown embedding request field {key}"));
        }
    }
    if object
        .get("model")
        .is_some_and(|m| m.as_str() != Some(model_id))
    {
        return Err("Unknown embedding model".into());
    }
    let role = object
        .get("input_type")
        .map_or(Ok("document"), |v| v.as_str().ok_or("Invalid input_type"))?;
    if !["query", "document"].contains(&role) {
        return Err("input_type must be query or document".into());
    }
    let dimensions = match object.get("dimensions") {
        None => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|&n| n > 0 && n <= model.spec().dimension as u64)
                .ok_or("Invalid embedding dimensions")? as usize,
        ),
    };
    let encoding = object
        .get("encoding_format")
        .map_or(Ok("float"), |v| v.as_str().ok_or("Invalid encoding_format"))?;
    if !["float", "base64"].contains(&encoding) {
        return Err("encoding_format must be float or base64".into());
    }
    let input = object.get("input").ok_or("Missing embedding input")?;
    let inputs = if let Some(v) = input.as_array() {
        v.as_slice()
    } else {
        std::slice::from_ref(input)
    };
    if inputs.is_empty() || inputs.len() > 128 {
        return Err("Embedding input batch must contain 1..128 items".into());
    }
    let mut prepared = Vec::with_capacity(inputs.len());
    let mut prepared_bytes = 0usize;
    for input in inputs {
        let item = codec.prepare(input, role, local)?;
        prepared_bytes += item.tokens.len() * 8
            + item
                .images
                .iter()
                .map(|i| i.pixels.len() * 4)
                .sum::<usize>();
        if prepared_bytes > 256 * 1024 * 1024 {
            return Err("Embedding batch exceeds the 256 MiB preparation budget".into());
        }
        prepared.push(item);
    }
    let mut data = Vec::with_capacity(prepared.len());
    let mut tokens = 0usize;
    let vectors = model.embed_many(
        &prepared
            .iter()
            .map(|p| (p.tokens.as_slice(), p.images.as_slice(), p.index.as_slice()))
            .collect::<Vec<_>>(),
        dimensions,
    )?;
    for (index, (p, vector)) in prepared.into_iter().zip(vectors).enumerate() {
        tokens += p.tokens.len();
        let embedding = if encoding == "base64" {
            let bytes: Vec<_> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
            json!(STANDARD.encode(bytes))
        } else {
            json!(vector)
        };
        data.push(json!({"object":"embedding", "index":index, "embedding":embedding}));
    }
    Ok(
        json!({"object":"list", "data":data, "model":model_id, "usage":{"prompt_tokens":tokens,"total_tokens":tokens}}),
    )
}

pub fn run_model(root: &Path, requests: &Path, options: LoadOptions) -> Result<Value> {
    let request: Value =
        serde_json::from_slice(&std::fs::read(requests).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let (codec, mut model) =
        crate::server::startup::load_parallel(|| Codec::load(root), || Model::load(root, options))?;
    if &codec.hashes != model.frontend_assets() {
        return Err("Embedding frontend changed during loading".into());
    }
    let id = request
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("nemotron-3.5-embed-1b")
        .to_owned();
    let started = Instant::now();
    let result = response(&codec, &mut model, request, &id, true)?;
    Ok(
        json!({"response":result,"inference_s":started.elapsed().as_secs_f64(),"load":model.statistics()}),
    )
}

struct EmbeddingBackend {
    codec: Codec,
    model: Model,
    id: String,
}
impl crate::encoder_service::Backend for EmbeddingBackend {
    fn respond(&mut self, request: Value) -> Result<Value> {
        response(&self.codec, &mut self.model, request, &self.id, false)
    }
}
pub(crate) fn serve(settings: crate::ServerConfig) -> Result<()> {
    crate::encoder_service::serve(settings, vec!["/v1/embeddings"], |settings| {
        let (codec, model) = crate::server::startup::load_parallel(
            || Codec::load(&settings.model_dir),
            || {
                Model::load(
                    &settings.model_dir,
                    LoadOptions {
                        cuda_graph: settings.cuda_graph,
                        verify_weights: settings.verify_weights,
                        load_workers: settings.load_workers,
                        ..Default::default()
                    },
                )
            },
        )?;
        if &codec.hashes != model.frontend_assets() {
            return Err("Embedding frontend changed during loading".into());
        }
        Ok(EmbeddingBackend {
            codec,
            model,
            id: settings.model.clone(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn png(width: u32, height: u32) -> Vec<u8> {
        let image = image::RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
        let mut output = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut output, image::ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }
    #[test]
    fn pixtral_resize_and_channel_patch_order() {
        let image = preprocess(&png(301, 207), 6400).unwrap();
        assert_eq!((image.grid_height, image.grid_width), (16, 22));
        assert_eq!(image.pixels.len(), 16 * 22 * 588);
        for (channel, (mean, std)) in [
            (0.48145466, 0.26862954),
            (0.4578275, 0.261_302_6),
            (0.40821073, 0.275_777_1),
        ]
        .into_iter()
        .enumerate()
        {
            let expected = (1.0 - mean) / std;
            assert!(
                image.pixels[channel * 196..(channel + 1) * 196]
                    .iter()
                    .all(|v| (*v - expected).abs() < 1e-6)
            );
        }
        let resized = preprocess(&png(1600, 400), 6400).unwrap();
        assert_eq!((resized.grid_height, resized.grid_width), (20, 80));
        assert!(preprocess(&png(301, 207), 4).is_err());
    }
    #[test]
    fn image_inputs_enforce_transport_and_shape_limits() {
        assert!(load_image("file:///etc/passwd", false).is_err());
        assert!(load_image("data:text/plain;base64,aGVsbG8=", false).is_err());
        assert!(preprocess(b"invalid", 6400).is_err());
        let data = png(28, 28);
        let uri = format!("data:image/png;base64,{}", STANDARD.encode(&data));
        assert_eq!(load_image(&uri, false).unwrap(), data);
    }

    #[test]
    #[ignore = "Requires the original checkpoint tokenizer; CPU only"]
    fn checkpoint_token_ids_match_transformers_retrieval_template() {
        let root =
            std::path::PathBuf::from(std::env::var_os("ORINFER_REFERENCE_CHECKPOINT").unwrap());
        let codec = Codec {
            tokenizer: Tokenizer::from_file(root.join("tokenizer.json")).unwrap(),
            hashes: BTreeMap::new(),
            context: 32768,
            patches: 6400,
        };
        let query = codec
            .prepare(&json!("Orin 有多少内存？"), "query", false)
            .unwrap();
        assert_eq!(
            query.tokens,
            [
                5272, 1058, 3247, 1259, 116533, 12379, 18092, 20759, 11000, 1719
            ]
        );
        let literal = codec
            .prepare(&json!("literal [IMG] <s>"), "document", false)
            .unwrap();
        assert!(!literal.tokens.contains(&10));
        assert!(literal.index.iter().all(|&i| i == -1));
        assert!(
            codec
                .prepare(&json!({"images":["unused"]}), "query", false)
                .is_err()
        );
    }
}
