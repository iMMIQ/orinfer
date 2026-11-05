//! Images-only OpenAI content parts. CPU work runs outside the async executor.
pub(super) mod cache;
mod deepseek;
#[cfg(test)]
mod history;
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ImageReader, imageops::FilterType};
use orinfer_engine::vision::{ImageInput, ImageLayout, VisionSpec};
use serde_json::{Value, json};
use std::{
    io::{Cursor, Read},
    time::Duration,
};
type Result<T> = std::result::Result<T, String>;
const MAX_BYTES: usize = 24 * 1024 * 1024;
const MIN_PIXELS: usize = 256 * 256;
const MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

fn planned_shape(
    v: &VisionSpec,
    height: usize,
    width: usize,
    detail: &str,
    max_pixels: Option<usize>,
    feature_limit: Option<usize>,
) -> Result<(usize, usize)> {
    if !["auto", "high", "low"].contains(&detail) {
        return Err("image_url.detail must be auto, high or low".into());
    }
    let minimum = v.minimum_pixels();
    let limit = if detail == "low" {
        minimum
    } else {
        max_pixels
            .unwrap_or(usize::MAX)
            .min(v.max_patches * v.patch_size * v.patch_size)
            .min(16_777_216)
    };
    if v.layout == ImageLayout::DeepseekV41 {
        return deepseek::dimensions(v, height, width, limit, feature_limit);
    }
    let factor = v.patch_size * v.merge_size;
    let limit = limit.min(feature_limit.map_or(usize::MAX, |n| n.saturating_mul(factor * factor)));
    if limit < minimum {
        return Err(
            "Too many images for the model's minimum image detail; reduce the image count".into(),
        );
    }
    let result = resize_shape(height, width, factor, limit)?;
    if feature_limit.is_some_and(|n| result.0 / factor * (result.1 / factor) > n) {
        return Err("Image aspect ratio exceeds its automatic image budget at minimum detail; reduce the image count".into());
    }
    Ok(result)
}

/// Per-request limits supplement the execution package and shared host-memory budget.
#[derive(Debug, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageProcessing {
    pub max_pixels: Option<usize>,
    pub max_tokens: Option<usize>,
    /// Fixed per-image budget, independent of the other images in the history.
    pub max_tokens_per_image: Option<usize>,
    pub max_images: Option<usize>,
    pub auto_resize: bool,
}

fn dimensions(data: &[u8]) -> Result<(usize, usize)> {
    let (width, height) = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?
        .into_dimensions()
        .map_err(|e| e.to_string())?;
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or("Image dimensions overflow")?;
    if pixels > 16_777_216 || width > 32768 || height > 32768 {
        return Err("Image exceeds decoded pixel limit".into());
    }
    Ok((height as usize, width as usize))
}

// Capped water filling preserves small images and assigns their unused budget
// to larger ones. Per-aspect minimums also account for patch-grid rounding.
fn feature_limits(demand: &[(usize, usize)], budget: usize) -> Result<Vec<usize>> {
    let minimum: usize = demand.iter().map(|&(_, min)| min).sum();
    if minimum > budget {
        return Err(format!(
            "Images exceed feature capacity at minimum detail: require {minimum} tokens, limit {budget}"
        ));
    }
    let (mut low, mut high) = (0, demand.iter().map(|&(n, _)| n).max().unwrap_or(0));
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        let total: usize = demand.iter().map(|&(n, min)| n.min(mid).max(min)).sum();
        if total <= budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    let mut limits: Vec<_> = demand.iter().map(|&(n, min)| n.min(low).max(min)).collect();
    let mut remaining = budget - limits.iter().sum::<usize>();
    for (limit, &(n, _)) in limits.iter_mut().zip(demand) {
        if remaining > 0 && *limit < n {
            *limit += 1;
            remaining -= 1;
        }
    }
    Ok(limits)
}

fn load(url: &str, preparation: &super::preparation::Context) -> Result<Vec<u8>> {
    preparation.checkpoint(MAX_BYTES)?;
    if let Some(data) = url.strip_prefix("data:") {
        let (mime, payload) = data.split_once(',').ok_or("Invalid image data URI")?;
        if !["image/png;base64", "image/jpeg;base64", "image/webp;base64"].contains(&mime)
            || payload.len() > MAX_BYTES.div_ceil(3) * 4
        {
            return Err("Unsupported or oversized image data URI".into());
        }
        return STANDARD
            .decode(payload)
            .map_err(|e| format!("Image base64: {e}"));
    }
    let parsed = reqwest::Url::parse(url).map_err(|e| format!("Image URL: {e}"))?;
    if !["http", "https"].contains(&parsed.scheme()) {
        return Err("Images require HTTP(S) or a base64 data URI".into());
    }
    let response = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?
        .get(parsed)
        .send()
        .map_err(|e| format!("Image fetch: {e}"))?
        .error_for_status()
        .map_err(|e| format!("Image fetch: {e}"))?;
    if response
        .content_length()
        .is_some_and(|n| n > MAX_BYTES as u64)
    {
        return Err("Image exceeds byte limit".into());
    }
    let mut data = Vec::new();
    response
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > MAX_BYTES {
        return Err("Image exceeds byte limit".into());
    }
    Ok(data)
}
fn round_even(x: f64) -> usize {
    x.round_ties_even() as usize
}
pub fn resize_shape(
    height: usize,
    width: usize,
    factor: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0
        || width == 0
        || height.max(width) as f64 / height.min(width) as f64 > 200.0
        || max_pixels < MIN_PIXELS
    {
        return Err("Invalid image size or aspect ratio".into());
    }
    let (mut h, mut w) = (
        round_even(height as f64 / factor as f64) * factor,
        round_even(width as f64 / factor as f64) * factor,
    );
    let area = (height as f64) * (width as f64);
    if h * w > max_pixels {
        let beta = (area / max_pixels as f64).sqrt();
        h = ((height as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
        w = ((width as f64 / beta / factor as f64).floor() as usize * factor).max(factor);
    } else if h * w < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / area).sqrt();
        h = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        w = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((h, w))
}
#[cfg(test)]
pub fn preprocess(
    data: &[u8],
    v: &VisionSpec,
    detail: &str,
    preparation: &mut super::preparation::Context,
) -> Result<ImageInput> {
    preprocess_with_limit(data, v, detail, None, None, preparation)
}
fn preprocess_with_limit(
    data: &[u8],
    v: &VisionSpec,
    detail: &str,
    max_pixels: Option<usize>,
    feature_limit: Option<usize>,
    preparation: &mut super::preparation::Context,
) -> Result<ImageInput> {
    preparation.checkpoint(0)?;
    let (height, width) = dimensions(data)?;
    let source_pixels = height * width;
    let (rh, rw) = planned_shape(v, height, width, detail, max_pixels, feature_limit)?;
    let cache = preparation.images.clone();
    let identity = cache.as_ref().map(|_| cache::Cache::identity(data));
    let policy = cache::Cache::policy(v, rh, rw)?;
    let gate = cache
        .as_ref()
        .zip(identity)
        .map(|(cache, id)| cache.gate(id));
    let _gate = gate
        .as_ref()
        .map(|gate| gate.lock().unwrap_or_else(|e| e.into_inner()));
    preparation.checkpoint(0)?;
    if let Some(image) = cache
        .as_ref()
        .zip(identity)
        .and_then(|(cache, id)| cache.get(&id, policy))
    {
        preparation.retain(image.host_bytes())?;
        preparation
            .worker_scratch(image.grid_height * image.grid_width * v.encoder_patch_values()? * 2)?;
        return Ok((*image).clone());
    }
    // Decode may hold RGBA/16-bit source, RGB conversion and resized patch FP32 together.
    preparation.checkpoint(data.len() + source_pixels * 12 + rh * rw * 16)?;
    let mut reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    limits.max_image_width = Some(32768);
    limits.max_image_height = Some(32768);
    reader.limits(limits);
    preparation.checkpoint(0)?;
    let decoded = reader
        .decode()
        .map_err(|e| format!("Image decode: {e}"))?
        .to_rgb8();
    let (height, width) = (rh, rw);
    let (content_height, content_width) = if v.layout == ImageLayout::DeepseekV41 {
        deepseek::content_dimensions(
            decoded.height(),
            decoded.width(),
            height as u32,
            width as u32,
        )
    } else {
        (height as u32, width as u32)
    };
    if content_height == 0 || content_width == 0 {
        return Err("Image aspect ratio produces an empty resized image".into());
    }
    // Pillow/torchvision's uint8 bicubic path rounds and clips the horizontal
    // pass before the vertical pass. A single floating-point two-axis resize
    // produces different values at sharp high-contrast edges.
    let horizontal = image::imageops::resize(
        &decoded,
        content_width,
        decoded.height(),
        FilterType::CatmullRom,
    );
    let resized = image::imageops::resize(
        &horizontal,
        content_width,
        content_height,
        FilterType::CatmullRom,
    );
    drop(horizontal);
    drop(decoded);
    let rgb = if content_height != height as u32 || content_width != width as u32 {
        let mut canvas =
            image::RgbImage::from_pixel(width as u32, height as u32, image::Rgb([127; 3]));
        image::imageops::replace(
            &mut canvas,
            &resized,
            ((width as u32 - content_width) as f64 * 0.5).round_ties_even() as i64,
            ((height as u32 - content_height) as f64 * 0.5).round_ties_even() as i64,
        );
        canvas
    } else {
        resized
    };
    let (gh, gw) = (height / v.patch_size, width / v.patch_size);
    let rgb = rgb.into_raw();
    preparation.retain(rgb.len())?;
    preparation.worker_scratch(gh * gw * v.encoder_patch_values()? * 2)?;
    let input = ImageInput {
        grid_height: gh,
        grid_width: gw,
        pixels: Vec::new(),
        rgb,
    };
    v.feature_count(&input)?;
    if let Some((cache, id)) = cache.as_ref().zip(identity) {
        cache.insert(id, policy, data, &input);
    }
    Ok(input)
}
#[cfg(test)]
pub fn messages(
    messages: Vec<Value>,
    spec: Option<&VisionSpec>,
    preparation: &mut super::preparation::Context,
) -> Result<(Vec<Value>, Vec<ImageInput>)> {
    messages_with_options(messages, spec, &ImageProcessing::default(), preparation)
}
pub fn messages_with_options(
    mut messages: Vec<Value>,
    spec: Option<&VisionSpec>,
    options: &ImageProcessing,
    preparation: &mut super::preparation::Context,
) -> Result<(Vec<Value>, Vec<ImageInput>)> {
    let mut descriptors = Vec::new();
    for message in &messages {
        if let Some(parts) = message["content"].as_array() {
            for part in parts.iter().filter(|p| p["type"] == "image_url") {
                if message["role"] != "user"
                    && !(message["role"] == "tool"
                        && spec.is_some_and(|v| v.layout == ImageLayout::DeepseekV41))
                {
                    return Err("Image parts are supported in user messages".into());
                }
                let url = part["image_url"]["url"].as_str();
                if !part["image_url"]["url"].is_null() && url.is_none() {
                    return Err("image_url.url must be a string or null".into());
                }
                let uuid = part
                    .get("uuid")
                    .map(|value| {
                        value
                            .as_str()
                            .filter(|id| !id.is_empty() && id.len() <= 256)
                            .ok_or("Image uuid must be a nonempty string of at most 256 bytes")
                    })
                    .transpose()?;
                if url.is_none() && uuid.is_none() {
                    return Err("image_url requires url or a cached uuid".into());
                }
                let detail = part["image_url"]
                    .get("detail")
                    .map(|v| v.as_str().ok_or("Invalid image detail"))
                    .transpose()?
                    .unwrap_or("auto");
                if !["auto", "high", "low"].contains(&detail) {
                    return Err("image_url.detail must be auto, high or low".into());
                }
                descriptors.push((url, detail, uuid));
            }
        }
    }
    let count = descriptors.len();
    let mut prepared: Vec<Option<ImageInput>> =
        std::iter::repeat_with(|| None).take(count).collect();
    if options.max_pixels.is_some_and(|n| n < MIN_PIXELS)
        || options.max_tokens == Some(0)
        || options.max_tokens_per_image == Some(0)
        || options.max_images == Some(0)
    {
        return Err(
            "image_processing requires max_pixels >= 65536 and positive max_tokens/max_images"
                .into(),
        );
    }
    if options.max_images.is_some_and(|n| count > n) {
        return Err("Image count exceeds image_processing.max_images".into());
    }
    if count > 0 {
        let v = spec.ok_or("This model manifest has no vision encoder")?;
        let factor = v.patch_size * v.merge_size;
        let token_pixels = factor * factor;
        let budget = options
            .max_tokens
            .unwrap_or(v.max_features)
            .min(v.max_features);
        // Group-wide resizing changes old image tokens when another turn adds
        // an image. Native DeepSeek histories use deterministic per-image caps.
        let auto_resize = options.auto_resize && v.layout != ImageLayout::DeepseekV41;
        let minimum_tokens = if v.layout == ImageLayout::DeepseekV41 {
            4
        } else {
            MIN_PIXELS.div_ceil(token_pixels)
        };
        if count > budget / minimum_tokens {
            return Err(format!(
                "Images exceed feature capacity at minimum image detail: {count} images, {budget} available features; reduce the image count"
            ));
        }
        // Inspect dimensions before decoding. Compressed inputs are budgeted too;
        // only the current image has decoder/resize/upload scratch.
        let mut pending = Vec::with_capacity(count);
        let mut total = 0;
        let mut fixed = 0;
        let mut demand = Vec::new();
        for (i, &(url, detail, uuid)) in descriptors.iter().enumerate() {
            let data = if let Some(url) = url {
                load(url, preparation)?
            } else {
                preparation.checkpoint(MAX_BYTES)?;
                let cache = preparation
                    .images
                    .as_ref()
                    .ok_or("Image uuid cache is unavailable; resend image_url.url")?;
                (*cache.source(uuid.ok_or("Missing image uuid")?)?).clone()
            };
            preparation.retain(data.len())?;
            let (h, w) = dimensions(&data)?;
            let max_pixels = if detail == "low" {
                v.minimum_pixels()
            } else {
                options
                    .max_pixels
                    .unwrap_or(usize::MAX)
                    .min(v.max_patches * v.patch_size * v.patch_size)
                    .min(16_777_216)
            };
            let (rh, rw) = planned_shape(
                v,
                h,
                w,
                detail,
                Some(max_pixels),
                options.max_tokens_per_image,
            )?;
            if options.max_pixels.is_some_and(|n| rh * rw > n) {
                return Err(
                    "Image aspect ratio cannot fit image_processing.max_pixels at minimum detail"
                        .into(),
                );
            }
            let features = v.grid_features(rh / v.patch_size, rw / v.patch_size)?;
            let (mh, mw) = planned_shape(v, h, w, detail, Some(v.minimum_pixels()), None)?;
            let minimum = v.grid_features(mh / v.patch_size, mw / v.patch_size)?;
            total += features;
            if detail == "auto" {
                demand.push((features, minimum));
            } else {
                fixed += features;
            }
            if auto_resize {
                pending.push((data, detail, features));
            } else {
                if total > budget {
                    return Err(format!(
                        "Images exceed feature capacity: require at least {total} visual tokens, limit {budget}; lower image_processing.max_pixels/max_tokens_per_image, increase the explicit total budget within the model context, or use fewer images"
                    ));
                }
                prepared[i] = Some(preprocess_with_limit(
                    &data,
                    v,
                    detail,
                    options.max_pixels,
                    Some(features),
                    preparation,
                )?);
                if let Some((cache, uuid)) = preparation.images.as_ref().zip(uuid) {
                    cache.bind(uuid, &data)?;
                }
                preparation.release(data.len());
            }
        }
        let limits = if total > budget {
            if !auto_resize || fixed > budget {
                return Err(format!(
                    "Images exceed feature capacity: require {total} visual tokens, limit {budget}; lower image_processing.max_pixels, enable image_processing.auto_resize for auto images, or use fewer images"
                ));
            }
            feature_limits(&demand, budget - fixed)?
        } else {
            demand.iter().map(|&(n, _)| n).collect()
        };
        let mut limits = limits.into_iter();
        for (i, (data, detail, features)) in pending.into_iter().enumerate() {
            let limit = if detail == "auto" {
                limits.next().ok_or("Missing automatic image limit")?
            } else {
                features
            };
            let image = preprocess_with_limit(
                &data,
                v,
                detail,
                options.max_pixels,
                Some(limit),
                preparation,
            )?;
            preparation.release(data.len());
            prepared[i] = Some(image);
            if let Some((cache, uuid)) = preparation.images.as_ref().zip(descriptors[i].2) {
                cache.bind(uuid, &data)?;
            }
        }
    }
    drop(descriptors);
    let mut images = Vec::with_capacity(count);
    for message in &mut messages {
        let Some(parts) = message["content"].as_array() else {
            continue;
        };
        let mut content = String::new();
        let is_deepseek = spec.is_some_and(|v| v.layout == ImageLayout::DeepseekV41);
        for (index, part) in parts.iter().enumerate() {
            preparation.checkpoint(0)?;
            if is_deepseek && index > 0 {
                content.push_str("\n\n");
            }
            match part["type"].as_str() {
                Some("text") => {
                    content.push_str(part["text"].as_str().ok_or("Text part needs text")?)
                }
                Some("audio") => content.push_str("<|audio_start|><|audio_pad|><|audio_end|>"),
                Some("image_url") => {
                    images.push(
                        prepared[images.len()]
                            .take()
                            .ok_or("Missing prepared image")?,
                    );
                    content.push_str(if is_deepseek {
                        "<｜deepseek_image｜>"
                    } else {
                        MARKER
                    });
                }
                _ => return Err("Supported content parts: text and image_url".into()),
            }
        }
        message["content"] = json!(content);
    }
    Ok((messages, images))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_image_sizes_and_optional_shrinking_obey_independent_limits() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":8,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":1024,"max_features":512,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let part = |size| {
            let image =
                image::RgbImage::from_fn(size, size, |x, y| image::Rgb([x as u8, y as u8, 191]));
            let mut bytes = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image)
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", STANDARD.encode(bytes.get_ref()))}})
        };
        let run = |parts, options: &ImageProcessing| {
            messages_with_options(
                vec![json!({"role":"user","content":parts})],
                Some(&spec),
                options,
                &mut super::super::preparation::Context::unbounded(),
            )
        };
        let small = part(256);
        let large = part(512);
        let parts = vec![small.clone(), small.clone(), large.clone(), small.clone()];
        let (_, images) = run(parts, &ImageProcessing::default()).unwrap();
        assert_eq!(
            images
                .iter()
                .map(|i| (i.grid_height, i.grid_width))
                .collect::<Vec<_>>(),
            [(16, 16), (16, 16), (32, 32), (16, 16)]
        );
        let parts = vec![small, large.clone(), large];
        assert!(
            run(parts.clone(), &ImageProcessing::default())
                .unwrap_err()
                .contains("576")
        );
        let (_, images) = run(
            parts.clone(),
            &ImageProcessing {
                max_tokens: Some(400),
                auto_resize: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((images[0].grid_height, images[0].grid_width), (16, 16));
        assert!(images[1].grid_height > 20);
        assert!(
            images
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>()
                <= 400
        );
        let (_, images) = run(
            parts.clone(),
            &ImageProcessing {
                max_pixels: Some(MIN_PIXELS),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            images
                .iter()
                .all(|i| i.grid_height == 16 && i.grid_width == 16)
        );
        assert!(
            run(
                parts,
                &ImageProcessing {
                    max_images: Some(2),
                    ..Default::default()
                }
            )
            .unwrap_err()
            .contains("max_images")
        );
    }
    #[test]
    #[ignore = "Needs Transformers-generated image references"]
    fn checkpoint_image_processor_reference() {
        let root = std::path::PathBuf::from(std::env::var("ORINFER_IMAGE_REFERENCE").unwrap());
        let reference: Value =
            serde_json::from_slice(&std::fs::read(root.join("reference.json")).unwrap()).unwrap();
        let spec: VisionSpec = serde_json::from_value(reference["vision"].clone()).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let input = preprocess(
                &std::fs::read(root.join(case["image"].as_str().unwrap())).unwrap(),
                &spec,
                "auto",
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
            assert_eq!(
                input.grid_height,
                case["grid"][1].as_u64().unwrap() as usize
            );
            assert_eq!(input.grid_width, case["grid"][2].as_u64().unwrap() as usize);
            let bytes = std::fs::read(root.join(case["pixels"].as_str().unwrap())).unwrap();
            let expected: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            let actual: Vec<_> = input.normalized(&spec).collect();
            assert_eq!(actual.len(), expected.len());
            let (mut square, mut baseline, mut maximum) = (0.0f64, 0.0f64, 0.0f32);
            for (&a, &b) in actual.iter().zip(&expected) {
                square += (a - b) as f64 * (a - b) as f64;
                baseline += b as f64 * b as f64;
                maximum = maximum.max((a - b).abs());
            }
            let relative = (square / baseline).sqrt();
            eprintln!(
                "{} resize relative L2 {relative:.6}, max {maximum:.6}",
                case["image"]
            );
            assert!(relative < 0.02 && maximum < 0.05);
        }
    }
    #[test]
    #[ignore = "Real 12-image RGB preprocessing and memory accounting"]
    fn many_images_share_the_configured_preprocessing_pool() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let image = image::RgbImage::from_pixel(2048, 1024, image::Rgb([32, 128, 224]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.get_ref())
        );
        let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
            {"role":"user","content":(0..12).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
        let estimate =
            crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048);
        let mut context = estimate.context(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(false),
        ));
        let (_, images) = messages(request.messages, Some(&spec), &mut context).unwrap();
        assert_eq!(images.len(), 12);
        assert_eq!(
            images.iter().map(ImageInput::host_bytes).sum::<usize>(),
            72 << 20
        );
        assert_eq!(
            images
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>(),
            24576
        );
        assert!(
            images
                .iter()
                .all(|i| i.pixels.is_empty() && !i.rgb.is_empty())
        );
        context.checkpoint(0).unwrap();
    }
    #[test]
    #[ignore = "Real five-image request near the 32768-feature capacity; needs host RAM"]
    fn automatic_large_images_fit_real_model_capacity() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":32768,"max_features":32768,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":32768,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let image = image::RgbImage::from_pixel(4096, 2048, image::Rgb([32, 128, 224]));
        let mut encoded = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();
        let url = format!(
            "data:image/png;base64,{}",
            STANDARD.encode(encoded.get_ref())
        );
        let request: super::super::ChatRequest = serde_json::from_value(json!({"model":"test","messages":[
            {"role":"user","content":(0..5).map(|_|json!({"type":"image_url","image_url":{"url":url}})).collect::<Vec<_>>()}]})).unwrap();
        assert!(5 * (4096 / 32) * (2048 / 32) > spec.max_features);
        let estimate =
            crate::server::preparation::request_memory(&request, 262144, Some(&spec), 2048);
        let mut context = estimate.context(std::sync::Arc::new(
            std::sync::atomic::AtomicBool::new(false),
        ));
        let (_, images) = messages_with_options(
            request.messages,
            Some(&spec),
            &ImageProcessing {
                auto_resize: true,
                ..Default::default()
            },
            &mut context,
        )
        .unwrap();
        assert_eq!(images.len(), 5);
        let features: usize = images.iter().map(|i| spec.feature_count(i).unwrap()).sum();
        assert!(features <= spec.max_features);
        assert!(features > 30000);
        context.checkpoint(0).unwrap();
    }
    #[test]
    fn automatic_images_share_features_and_preserve_input_order() {
        let spec: VisionSpec = serde_json::from_value(json!({"hidden":5120,"patch_size":16,
            "temporal_patch_size":2,"merge_size":2,"max_patches":1024,"max_features":256,
            "pixels":"p","grid":"g","length":"l","output":"o","features":"f","feature_index":"i",
            "mrope_positions":"r","plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12}))
        .unwrap();
        let encode = |size, color| {
            let image = image::RgbImage::from_pixel(size, size, image::Rgb(color));
            let mut output = Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image)
                .write_to(&mut output, image::ImageFormat::Png)
                .unwrap();
            format!(
                "data:image/png;base64,{}",
                STANDARD.encode(output.get_ref())
            )
        };
        let red = encode(512, [255, 0, 0]);
        let blue = encode(512, [0, 0, 255]);
        let part = |url: &str, detail: &str| json!({"type":"image_url","image_url":{"url":url,"detail":detail}});
        let run = |parts: Vec<Value>| {
            messages_with_options(
                vec![json!({"role":"user","content":parts})],
                Some(&spec),
                &ImageProcessing {
                    auto_resize: true,
                    ..Default::default()
                },
                &mut crate::server::preparation::Context::unbounded(),
            )
        };
        let (rendered, images) = run(vec![part(&red, "auto"), part(&blue, "auto")]).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(rendered[0]["content"], format!("{MARKER}{MARKER}"));
        assert!(
            images
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>()
                <= 256
        );
        assert_eq!(images[0].normalized(&spec).next().unwrap(), 1.0);
        assert_eq!(images[1].normalized(&spec).next().unwrap(), -1.0);
        let (_, reverse) = run(vec![part(&blue, "auto"), part(&red, "auto")]).unwrap();
        assert_eq!(images[0].rgb, reverse[1].rgb);
        assert_eq!(images[1].rgb, reverse[0].rgb);
        let (_, single) = run(vec![part(&red, "auto")]).unwrap();
        assert_eq!(spec.feature_count(&single[0]).unwrap(), 256);
        let error = run(vec![part(&red, "high"), part(&blue, "high")]).unwrap_err();
        assert!(error.contains("feature capacity") && error.contains("512"));
        let small_blue = encode(256, [0, 0, 255]);
        let (_, mixed) = run(vec![part(&red, "auto"), part(&small_blue, "high")]).unwrap();
        assert_eq!((mixed[1].grid_height, mixed[1].grid_width), (16, 16));
        assert_eq!(mixed[0].normalized(&spec).next().unwrap(), 1.0);
        assert_eq!(mixed[1].normalized(&spec).next().unwrap(), -1.0);
        assert!(
            mixed
                .iter()
                .map(|i| spec.feature_count(i).unwrap())
                .sum::<usize>()
                <= 256
        );
        let error = run((0..5)
            .map(|_| part("https://invalid.invalid/never-fetch", "auto"))
            .collect())
        .unwrap_err();
        assert!(error.contains("minimum image detail"));
    }
    #[test]
    fn patch_order_temporal_duplication_and_image_parts() {
        let spec: VisionSpec = serde_json::from_value(json!({
            "hidden":5120,"patch_size":16,"temporal_patch_size":2,"merge_size":2,
            "max_patches":1024,"max_features":256,"pixels":"p","grid":"g","length":"l",
            "output":"o","features":"f","feature_index":"i","mrope_positions":"r",
            "plans":[{"patches":1024,"program":"v"}],"image_token_id":10,
            "vision_start_token_id":11,"vision_end_token_id":12
        }))
        .unwrap();
        let rgb = image::RgbImage::from_fn(256, 256, |x, y| {
            image::Rgb([
                if x / 16 % 2 == 0 { 255 } else { 0 },
                if y / 16 % 2 == 0 { 255 } else { 0 },
                0,
            ])
        });
        let mut data = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(rgb)
            .write_to(&mut data, image::ImageFormat::Png)
            .unwrap();
        let pixels = preprocess(
            data.get_ref(),
            &spec,
            "auto",
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!((pixels.grid_height, pixels.grid_width), (16, 16));
        let normalized: Vec<_> = pixels.normalized(&spec).collect();
        for (row, (red, green)) in [(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]
            .into_iter()
            .enumerate()
        {
            let p = &normalized[row * 1536..(row + 1) * 1536];
            assert!(p[..512].iter().all(|&v| v == red));
            assert!(p[512..1024].iter().all(|&v| v == green));
            assert!(p[1024..].iter().all(|&v| v == -1.0));
        }
        let url = format!("data:image/png;base64,{}", STANDARD.encode(data.get_ref()));
        let input = vec![
            json!({"role":"user","content":[{"type":"text","text":"A"},{"type":"image_url","image_url":{"url":url}},{"type":"text","text":"B"},{"type":"image_url","image_url":{"url":url}}]}),
        ];
        let (out, images) = messages(
            input.clone(),
            Some(&spec),
            &mut crate::server::preparation::Context::unbounded(),
        )
        .unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(out[0]["content"], format!("A{MARKER}B{MARKER}"));
        assert!(
            messages(
                input,
                None,
                &mut crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
        assert!(
            preprocess(
                data.get_ref(),
                &spec,
                "invalid",
                &mut crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
    }
    #[test]
    fn smart_resize_rounding_and_aspect() {
        assert_eq!(resize_shape(256, 256, 32, 1_048_576).unwrap(), (256, 256));
        assert_eq!(resize_shape(80, 112, 32, 1_048_576).unwrap(), (224, 320));
        assert_eq!(
            resize_shape(2000, 1000, 32, 1_048_576).unwrap(),
            (1440, 704)
        );
        assert_eq!(resize_shape(500, 100, 32, 65_536).unwrap(), (576, 128));
        assert!(resize_shape(1, 300, 32, 1_048_576).is_err());
        assert!(
            load(
                "file:///tmp/image.png",
                &crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
        assert!(
            load(
                "data:video/mp4;base64,AA==",
                &crate::server::preparation::Context::unbounded()
            )
            .is_err()
        );
    }
}
