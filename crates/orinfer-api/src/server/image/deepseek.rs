//! Official DeepSeek resize geometry and centered RGB padding.
use super::{Result, VisionSpec};

pub(super) fn dimensions(
    v: &VisionSpec,
    height: usize,
    width: usize,
    max_pixels: usize,
    feature_limit: Option<usize>,
) -> Result<(usize, usize)> {
    let patch = v.patch_size;
    let merge = v.merge_size;
    let limit = v
        .max_image_tokens
        .ok_or("DeepSeek requires a per-image token budget")?
        .min(feature_limit.unwrap_or(usize::MAX));
    if height == 0 || width == 0 || patch == 0 || merge == 0 || limit < 4 {
        return Err("Invalid DeepSeek image dimensions or token budget".into());
    }
    let (mut h, mut w) = (height, width);
    if h.checked_mul(w).ok_or("Image dimensions overflow")? < v.minimum_pixels() {
        let ratio = (v.minimum_pixels() as f64 / (h * w) as f64).sqrt();
        h = (h as f64 * ratio) as usize;
        w = (w as f64 * ratio) as usize;
    }
    let (mut best_h, mut best_w) = (h.div_ceil(patch) * patch, w.div_ceil(patch) * patch);
    let token_count =
        |h: usize, w: usize| (h / patch).div_ceil(merge) * ((w / patch).div_ceil(merge) + 1) + 2;
    if token_count(best_h, best_w) > limit {
        let ratio = h as f64 / w as f64;
        let max_w = (((limit - 2) as f64 / ratio) + 0.25).sqrt() - 0.5;
        let max_h = max_w * ratio;
        let cell = patch * merge;
        (best_h, best_w) = if max_w < 1.0 {
            ((limit - 2) / 2 * cell, cell)
        } else if max_h < 1.0 {
            (cell, (limit - 3) * cell)
        } else {
            let beta = (max_w.floor() * cell as f64 / w as f64)
                .min(max_h.floor() * cell as f64 / h as f64);
            (
                (h as f64 * beta / patch as f64).floor() as usize * patch,
                (w as f64 * beta / patch as f64).floor() as usize * patch,
            )
        };
    }
    if best_h * best_w > max_pixels {
        let beta = (max_pixels as f64 / (best_h * best_w) as f64).sqrt();
        best_h = (((best_h as f64 * beta) as usize / patch) * patch).max(patch);
        best_w = (((best_w as f64 * beta) as usize / patch) * patch).max(patch);
    }
    if best_h == 0
        || best_w == 0
        || best_h * best_w > max_pixels
        || token_count(best_h, best_w) > limit
    {
        return Err("Image cannot fit DeepSeek pixel and token budgets".into());
    }
    Ok((best_h, best_w))
}

pub(super) fn content_dimensions(
    height: u32,
    width: u32,
    target_height: u32,
    target_width: u32,
) -> (u32, u32) {
    let source_ratio = width as f64 / height as f64;
    let target_ratio = target_width as f64 / target_height as f64;
    if source_ratio > target_ratio {
        (
            (target_width as f64 / source_ratio).round_ties_even() as u32,
            target_width,
        )
    } else if source_ratio < target_ratio {
        (
            target_height,
            (target_height as f64 * source_ratio).round_ties_even() as u32,
        )
    } else {
        (target_height, target_width)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    #[test]
    fn image_decode_resize_and_padding_match_official_pixel_samples() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/vision/deepseek-v41.json"
        ))
        .unwrap();
        let v: VisionSpec = serde_json::from_value(fixtures["spec"].clone()).unwrap();
        for case in fixtures["pixels"].as_array().unwrap() {
            let data = base64::engine::general_purpose::STANDARD
                .decode(case["png"].as_str().unwrap())
                .unwrap();
            let image = super::super::preprocess(
                &data,
                &v,
                "auto",
                &mut crate::server::preparation::Context::unbounded(),
            )
            .unwrap();
            let h = case["height"].as_u64().unwrap() as usize;
            let w = case["width"].as_u64().unwrap() as usize;
            assert_eq!((image.grid_height * 14, image.grid_width * 14), (h, w));
            for sample in case["samples"].as_array().unwrap() {
                let y = sample[0].as_u64().unwrap() as usize;
                let x = sample[1].as_u64().unwrap() as usize;
                for ch in 0..3 {
                    let expected = sample[2 + ch].as_u64().unwrap() as i16;
                    let actual = image.rgb[(y * w + x) * 3 + ch] as i16;
                    assert!(
                        (actual - expected).abs() <= 1,
                        "{h}x{w} pixel ({y},{x},{ch}): {actual} != {expected}"
                    );
                }
            }
        }
    }
    #[test]
    fn official_grid_vectors_include_portraits_landscapes_and_budget_tails() {
        let fixtures: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/vision/deepseek-v41.json"
        ))
        .unwrap();
        for case in fixtures["grids"].as_array().unwrap() {
            let v: VisionSpec = serde_json::from_value(fixtures["spec"].clone()).unwrap();
            let dims = dimensions(
                &v,
                case["height"].as_u64().unwrap() as usize,
                case["width"].as_u64().unwrap() as usize,
                usize::MAX,
                case["limit"].as_u64().map(|n| n as usize),
            )
            .unwrap();
            assert_eq!(
                [dims.0, dims.1],
                [
                    case["resized_height"].as_u64().unwrap() as usize,
                    case["resized_width"].as_u64().unwrap() as usize
                ],
                "{case}"
            );
        }
    }
}
