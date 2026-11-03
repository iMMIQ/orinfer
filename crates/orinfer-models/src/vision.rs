//! Qwen image spans and interleaved MRoPE positions.
use orinfer_model_sdk::{
    abi::ImageGrid,
    artifact::Result,
    vision::{ImageLayout, VisionSpec},
};
fn feature_count(v: &VisionSpec, image: &ImageGrid) -> Result<usize> {
    v.grid_features(image.grid_height, image.grid_width)
}
/// Physical KV positions stay linear. Only rotary coordinates are compressed
/// across each spatial image grid; generation resumes after their maximum.
pub fn layout(
    v: &VisionSpec,
    tokens: &[u32],
    images: &[ImageGrid],
    context: usize,
) -> Result<(Vec<i32>, Vec<u32>)> {
    if tokens.len() > context {
        return Err("Prompt exceeds context".into());
    }
    let mut index = vec![-1; context];
    let mut positions = Vec::with_capacity(context * 3);
    let (mut cursor, mut base, mut feature, mut image_index) = (0usize, 0usize, 0usize, 0usize);
    while cursor < tokens.len() {
        if tokens[cursor] != v.image_token_id {
            positions.extend([base as u32; 3]);
            base += 1;
            cursor += 1;
            continue;
        }
        let image = images.get(image_index).ok_or("Unbound image tokens")?;
        let count = feature_count(v, image)?;
        let end = cursor.checked_add(count).ok_or("Image span overflow")?;
        if v.layout == ImageLayout::DeepseekV41 {
            if end > tokens.len()
                || tokens[cursor..end].iter().any(|&id| id != v.image_token_id)
                || feature
                    .checked_add(count)
                    .is_none_or(|n| n > v.max_features)
            {
                return Err("Invalid DeepSeek image span or feature capacity".into());
            }
            for row in 0..count {
                index[cursor + row] = (feature + row) as i32;
                positions.extend([(base + row) as u32; 3]);
            }
            base += count;
            cursor = end;
            feature += count;
            image_index += 1;
            continue;
        }
        if cursor == 0
            || tokens[cursor - 1] != v.vision_start_token_id
            || end >= tokens.len()
            || tokens[end] != v.vision_end_token_id
            || tokens[cursor..end].iter().any(|&id| id != v.image_token_id)
            || feature
                .checked_add(count)
                .is_none_or(|n| n > v.max_features)
        {
            return Err("Invalid image token span or feature capacity".into());
        }
        let (height, width) = (
            image.grid_height / v.merge_size,
            image.grid_width / v.merge_size,
        );
        for row in 0..count {
            index[cursor + row] = (feature + row) as i32;
            positions.extend([
                base as u32,
                (base + row / width) as u32,
                (base + row % width) as u32,
            ]);
        }
        base += height.max(width);
        cursor = end;
        feature += count;
        image_index += 1;
    }
    if image_index != images.len() {
        return Err("Unused image features".into());
    }
    for _ in tokens.len()..context {
        positions.extend([base as u32; 3]);
        base += 1;
    }
    if positions.iter().any(|&p| p as usize >= context) {
        return Err("MRoPE exceeds rotary cache".into());
    }
    Ok((index, positions))
}
#[cfg(test)]
mod tests {
    use super::*;
    use orinfer_model_sdk::vision::ImageInput;
    fn positions(
        s: &VisionSpec,
        tokens: &[u32],
        images: &[ImageInput],
        capacity: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let grids: Vec<_> = images
            .iter()
            .map(|i| ImageGrid {
                grid_height: i.grid_height,
                grid_width: i.grid_width,
            })
            .collect();
        layout(s, tokens, &grids, capacity)
    }
    fn spec() -> VisionSpec {
        serde_json::from_value(serde_json::json!({"hidden":8,"patch_size":16,"temporal_patch_size":2,"merge_size":2,"max_patches":64,"max_features":16,"pixels":"p","grid":"g","length":"n","output":"o","features":"f","feature_index":"i","mrope_positions":"r","plans":[{"patches":64,"program":"v"}],"image_token_id":10,"vision_start_token_id":11,"vision_end_token_id":12})).unwrap()
    }
    fn image(h: usize, w: usize) -> ImageInput {
        ImageInput {
            rgb: Vec::new(),
            grid_height: h,
            grid_width: w,
            pixels: vec![0.; h * w * 1536],
        }
    }
    #[test]
    fn deepseek_two_odd_grids_keep_linear_positions_and_distinct_feature_spans() {
        let mut v = spec();
        v.layout = ImageLayout::DeepseekV41;
        v.max_image_tokens = Some(1024);
        v.patch_size = 14;
        v.temporal_patch_size = 1;
        v.merge_size = 3;
        let grids = [
            ImageGrid {
                grid_height: 2,
                grid_width: 4,
            },
            ImageGrid {
                grid_height: 4,
                grid_width: 5,
            },
        ];
        let mut tokens = vec![1];
        tokens.extend([10; 5]);
        tokens.push(2);
        tokens.extend([10; 8]);
        tokens.push(3);
        let (indices, positions) = layout(&v, &tokens, &grids, 32).unwrap();
        assert_eq!(&indices[1..6], &[0, 1, 2, 3, 4]);
        assert_eq!(indices[6], -1);
        assert_eq!(&indices[7..15], &[5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(positions, (0..32).flat_map(|i| [i; 3]).collect::<Vec<_>>());
        tokens[8] = 4;
        assert!(layout(&v, &tokens, &grids, 32).is_err());
    }
    #[test]
    fn multiple_images_and_decode_positions() {
        let s = spec();
        let images = [image(4, 6), image(2, 4)];
        let tokens = s
            .expand(&[1, 11, 10, 12, 2, 11, 10, 12, 3], &images, 32)
            .unwrap();
        let (idx, pos) = positions(&s, &tokens, &images, 32).unwrap();
        assert_eq!(&idx[2..8], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(
            &pos[6..24],
            &[2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 3, 3, 2, 3, 4]
        );
        assert_eq!(&idx[11..13], &[6, 7]);
        assert_eq!(&pos[tokens.len() * 3..tokens.len() * 3 + 3], &[12, 12, 12]);
        assert_eq!(idx[tokens.len()], -1);
        assert!(positions(&s, &tokens, &images[..1], 32).is_err());
        assert!(s.expand(&[10], &images, 32).is_err());
    }
    #[test]
    fn plain_text_and_bad_spans() {
        let s = spec();
        let (idx, pos) = positions(&s, &[1, 2, 3], &[], 8).unwrap();
        assert_eq!(idx, vec![-1; 8]);
        assert_eq!(&pos[21..], &[7, 7, 7]);
        assert!(positions(&s, &[11, 10, 12], &[image(4, 6)], 32).is_err());
        assert!(s.feature_count(&image(3, 4)).is_err());
    }
}
