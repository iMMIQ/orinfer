//! Image feature layout and Qwen interleaved multimodal positions.
use crate::{
    artifact::{Access, Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    #[default]
    F16,
    Bf16,
}
impl Precision {
    fn storage(self) -> Dtype {
        match self {
            Self::F16 => Dtype::F16,
            Self::Bf16 => Dtype::Bf16,
        }
    }
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionPlan {
    pub patches: usize,
    pub program: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionSpec {
    #[serde(default)]
    pub dtype: Precision,
    pub hidden: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub max_patches: usize,
    pub max_features: usize,
    pub pixels: String,
    pub grid: String,
    pub length: String,
    pub output: String,
    pub features: String,
    pub feature_index: String,
    pub mrope_positions: String,
    pub plans: Vec<VisionPlan>,
    pub image_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    pub grid_height: usize,
    pub grid_width: usize,
    /// Normalized RGB, in checkpoint patch/channel/temporal/spatial order.
    pub pixels: Vec<f32>,
}
impl VisionSpec {
    pub fn patch_values(&self) -> Result<usize> {
        self.patch_size
            .checked_mul(self.patch_size)
            .and_then(|n| n.checked_mul(self.temporal_patch_size))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| "Patch size overflow".into())
    }
    pub fn feature_count(&self, image: &ImageInput) -> Result<usize> {
        let n = image
            .grid_height
            .checked_mul(image.grid_width)
            .ok_or("Image grid overflow")?;
        if image.grid_height == 0
            || image.grid_width == 0
            || !image.grid_height.is_multiple_of(self.merge_size)
            || !image.grid_width.is_multiple_of(self.merge_size)
            || n > self.max_patches
            || self.plans.iter().all(|p| p.patches < n)
            || image.pixels.len()
                != n.checked_mul(self.patch_values()?)
                    .ok_or("Pixel size overflow")?
            || image
                .pixels
                .iter()
                .any(|p| !p.is_finite() || !(-1.001..=1.001).contains(p))
        {
            return Err("Invalid image grid, pixel data or vision capacity".into());
        }
        Ok(n / self.merge_size / self.merge_size)
    }
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        // This adapter is images-only. Future adapters declare their own layout.
        if self.hidden == 0
            || self.patch_size != 16
            || self.temporal_patch_size != 2
            || self.merge_size != 2
            || self.max_patches == 0
            || !self.max_patches.is_multiple_of(4)
            || self.max_features == 0
            || self.max_features > m.max_context
            || self.max_patches / 4 > self.max_features
        {
            return Err("Invalid image adapter dimensions".into());
        }
        let names = [
            &self.pixels,
            &self.grid,
            &self.length,
            &self.output,
            &self.features,
            &self.feature_index,
            &self.mrope_positions,
        ];
        if names.into_iter().collect::<BTreeSet<_>>().len() != 7 {
            return Err("Vision control and feature buffers must not alias".into());
        }
        let ids = [
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ];
        if ids.iter().any(|&id| id as usize >= m.vocab)
            || ids.into_iter().collect::<BTreeSet<_>>().len() != 3
        {
            return Err("Invalid vision token IDs".into());
        }
        let mut sizes = BTreeSet::new();
        for p in &self.plans {
            if p.patches == 0
                || !p.patches.is_multiple_of(4)
                || p.patches > self.max_patches
                || !sizes.insert(p.patches)
                || m.programs.get(&p.program).is_none_or(Vec::is_empty)
            {
                return Err("Invalid vision graph plan".into());
            }
        }
        if !sizes.contains(&self.max_patches) {
            return Err("Missing maximum vision plan".into());
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| "Vision size overflow".to_owned())
        };
        for (name, dtype, count) in [
            (
                &self.pixels,
                self.dtype.storage(),
                product(self.max_patches, self.patch_values()?)?,
            ),
            (&self.grid, Dtype::I32, 2),
            (&self.length, Dtype::I32, 1),
            (
                &self.output,
                Dtype::F16,
                product(self.max_patches / 4, self.hidden)?,
            ),
            (
                &self.features,
                Dtype::F16,
                product(self.max_features, self.hidden)?,
            ),
            (&self.feature_index, Dtype::I32, m.max_context),
            (
                &self.mrope_positions,
                Dtype::I32,
                product(m.max_context, 3)?,
            ),
        ] {
            let b = m
                .buffers
                .iter()
                .find(|b| &b.name == name)
                .ok_or("Missing vision buffer")?;
            if b.dtype != dtype
                || b.access == Access::Read
                || b.bytes()? != product(count, dtype.bytes())?
            {
                return Err(format!("Invalid vision buffer {name}"));
            }
        }
        Ok(())
    }

    /// Expand one native image marker per image, preserving message order.
    pub fn expand(
        &self,
        tokens: &[u32],
        images: &[ImageInput],
        context: usize,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut next = 0;
        for &id in tokens {
            if id == self.image_token_id {
                let image = images.get(next).ok_or("Image marker has no image")?;
                out.extend(std::iter::repeat_n(id, self.feature_count(image)?));
                next += 1;
            } else {
                out.push(id);
            }
            if out.len() > context {
                return Err("Image tokens exceed context capacity".into());
            }
        }
        if next != images.len() {
            return Err("Images missing from rendered prompt".into());
        }
        Ok(out)
    }

    /// Physical KV positions stay linear. Only rotary coordinates are compressed
    /// across each spatial image grid; generation resumes after their maximum.
    pub fn layout(
        &self,
        tokens: &[u32],
        images: &[ImageInput],
        context: usize,
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        if tokens.len() > context {
            return Err("Prompt exceeds context".into());
        }
        let mut index = vec![-1; context];
        let mut positions = Vec::with_capacity(context * 3);
        let (mut cursor, mut base, mut feature, mut image_index) = (0usize, 0usize, 0usize, 0usize);
        while cursor < tokens.len() {
            if tokens[cursor] != self.image_token_id {
                positions.extend([base as u32; 3]);
                base += 1;
                cursor += 1;
                continue;
            }
            let image = images.get(image_index).ok_or("Unbound image tokens")?;
            let count = self.feature_count(image)?;
            let end = cursor.checked_add(count).ok_or("Image span overflow")?;
            if cursor == 0
                || tokens[cursor - 1] != self.vision_start_token_id
                || end >= tokens.len()
                || tokens[end] != self.vision_end_token_id
                || tokens[cursor..end]
                    .iter()
                    .any(|&id| id != self.image_token_id)
                || feature
                    .checked_add(count)
                    .is_none_or(|n| n > self.max_features)
            {
                return Err("Invalid image token span or feature capacity".into());
            }
            let (height, width) = (
                image.grid_height / self.merge_size,
                image.grid_width / self.merge_size,
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
}
#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> VisionSpec {
        serde_json::from_value(serde_json::json!({"hidden":8,"patch_size":16,"temporal_patch_size":2,"merge_size":2,"max_patches":64,"max_features":16,"pixels":"p","grid":"g","length":"n","output":"o","features":"f","feature_index":"i","mrope_positions":"r","plans":[{"patches":64,"program":"v"}],"image_token_id":10,"vision_start_token_id":11,"vision_end_token_id":12})).unwrap()
    }
    fn image(h: usize, w: usize) -> ImageInput {
        ImageInput {
            grid_height: h,
            grid_width: w,
            pixels: vec![0.; h * w * 1536],
        }
    }
    #[test]
    fn multiple_images_and_decode_positions() {
        let s = spec();
        let images = [image(4, 6), image(2, 4)];
        let tokens = s
            .expand(&[1, 11, 10, 12, 2, 11, 10, 12, 3], &images, 32)
            .unwrap();
        let (idx, pos) = s.layout(&tokens, &images, 32).unwrap();
        assert_eq!(&idx[2..8], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(
            &pos[6..24],
            &[2, 2, 2, 2, 2, 3, 2, 2, 4, 2, 3, 2, 2, 3, 3, 2, 3, 4]
        );
        assert_eq!(&idx[11..13], &[6, 7]);
        assert_eq!(&pos[tokens.len() * 3..tokens.len() * 3 + 3], &[12, 12, 12]);
        assert_eq!(idx[tokens.len()], -1);
        assert!(s.layout(&tokens, &images[..1], 32).is_err());
        assert!(s.expand(&[10], &images, 32).is_err());
    }
    #[test]
    fn plain_text_and_bad_spans() {
        let s = spec();
        let (idx, pos) = s.layout(&[1, 2, 3], &[], 8).unwrap();
        assert_eq!(idx, vec![-1; 8]);
        assert_eq!(&pos[21..], &[7, 7, 7]);
        assert!(s.layout(&[11, 10, 12], &[image(4, 6)], 32).is_err());
        assert!(s.feature_count(&image(3, 4)).is_err());
    }
}
