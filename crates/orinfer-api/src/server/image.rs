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
    load_source(url, false)
}

pub(crate) fn load_source(url: &str, allow_local: bool) -> Result<Vec<u8>> {
    if let Some(data) = url.strip_prefix("data:") {
        let (mime, payload) = data.split_once(',').ok_or("Invalid image data URI")?;
        if !["image/png;base64", "image/jpeg;base64", "image/webp;base64"].contains(&mime)
            || payload.len() > MAX_BYTES.div_ceil(3) * 4
        {
            return Err("Unsupported or oversized image data URI".into());
        }
        let bytes = STANDARD
            .decode(payload)
            .map_err(|e| format!("Image base64: {e}"))?;
        if bytes.len() > MAX_BYTES {
            return Err("Image exceeds byte limit".into());
        }
        return Ok(bytes);
    }
    if allow_local && !url.starts_with("http://") && !url.starts_with("https://") {
        return read_bounded(std::fs::File::open(url).map_err(|e| e.to_string())?);
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
    read_bounded(response)
}

fn read_bounded(reader: impl Read) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    reader
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut data)
        .map_err(|e| e.to_string())?;
    if data.len() > MAX_BYTES {
        return Err("Image exceeds byte limit".into());
    }
    Ok(data)
}
pub(crate) fn decode_rgb(data: &[u8], max_side: u32) -> Result<image::RgbImage> {
    let mut reader = ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| e.to_string())?;
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(256 * 1024 * 1024);
    limits.max_image_width = Some(max_side);
    limits.max_image_height = Some(max_side);
    reader.limits(limits);
    Ok(reader
        .decode()
        .map_err(|e| format!("Image decode: {e}"))?
        .to_rgb8())
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
    preparation.checkpoint(0)?;
    let decoded = decode_rgb(data, 32768)?;
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
#[path = "image_tests.rs"]
mod tests;
