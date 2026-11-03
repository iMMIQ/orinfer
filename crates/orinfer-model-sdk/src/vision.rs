//! Checkpoint image feature geometry shared by device backends.
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
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageLayout {
    #[default]
    Qwen,
    DeepseekV41,
}
impl ImageLayout {
    fn is_qwen(&self) -> bool {
        *self == Self::Qwen
    }
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
    #[serde(default, skip_serializing_if = "crate::model::is_zero")]
    pub min_patches: usize,
    pub program: String,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VisionSpec {
    #[serde(default, skip_serializing_if = "ImageLayout::is_qwen")]
    pub layout: ImageLayout,
    /// Per-image LLM span budget, including learned delimiters and row ends.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_image_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_pixels: Option<usize>,
    /// Physical encoder K stride; logical image data retains patch_values().
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_stride: Option<usize>,
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageInput {
    pub grid_height: usize,
    pub grid_width: usize,
    /// Normalized RGB, in checkpoint patch/channel/temporal/spatial order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pixels: Vec<f32>,
    /// Resized interleaved RGB8. Expanded only when this image is encoded.
    /// Exactly one of `pixels` and `rgb` must be populated.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rgb: Vec<u8>,
}
impl ImageInput {
    pub fn host_bytes(&self) -> usize {
        self.pixels.len() * std::mem::size_of::<f32>() + self.rgb.len()
    }
    /// Preserve the checkpoint's merge/patch/channel/temporal/spatial order.
    /// Callers validate the grid and storage with `feature_count` first.
    pub fn normalized<'a>(&'a self, v: &'a VisionSpec) -> impl Iterator<Item = f32> + 'a {
        let patch = v.patch_size;
        let merge = if v.layout == ImageLayout::DeepseekV41 {
            1
        } else {
            v.merge_size
        };
        let width = self.grid_width * patch;
        let blocks = if self.rgb.is_empty() {
            0
        } else {
            self.grid_height / merge
        };
        self.pixels
            .iter()
            .copied()
            .chain((0..blocks).flat_map(move |br| {
                (0..self.grid_width / merge).flat_map(move |bc| {
                    (0..merge * merge).flat_map(move |inner| {
                        (0..3).flat_map(move |ch| {
                            (0..v.temporal_patch_size).flat_map(move |_| {
                                (0..patch * patch).map(move |xy| {
                                    let y = (br * merge + inner / merge) * patch + xy / patch;
                                    let x = (bc * merge + inner % merge) * patch + xy % patch;
                                    self.rgb[(y * width + x) * 3 + ch] as f32 / 127.5 - 1.0
                                })
                            })
                        })
                    })
                })
            }))
    }
    /// Extend each logical patch with the encoder's declared zero K padding.
    /// Callers validate geometry and the stride before collecting upload bytes.
    pub fn encoded<'a>(&'a self, v: &'a VisionSpec) -> impl Iterator<Item = f32> + 'a {
        let logical = v.patch_size * v.patch_size * v.temporal_patch_size * 3;
        let padding = v.pixel_stride.unwrap_or(logical).saturating_sub(logical);
        self.normalized(v).enumerate().flat_map(move |(i, value)| {
            std::iter::once(value).chain(std::iter::repeat_n(
                0.0,
                if (i + 1).is_multiple_of(logical) {
                    padding
                } else {
                    0
                },
            ))
        })
    }
}
impl VisionSpec {
    pub fn encoder_patch_values(&self) -> Result<usize> {
        let logical = self.patch_values()?;
        let stride = self.pixel_stride.unwrap_or(logical);
        if logical == 0 || stride < logical || stride > 65536 {
            return Err("Invalid physical encoder patch stride".into());
        }
        Ok(stride)
    }
    pub fn minimum_pixels(&self) -> usize {
        self.min_pixels.unwrap_or(match self.layout {
            ImageLayout::Qwen => 256 * 256,
            ImageLayout::DeepseekV41 => 295936,
        })
    }
    pub fn grid_features(&self, height: usize, width: usize) -> Result<usize> {
        let patches = height.checked_mul(width).ok_or("Image grid overflow")?;
        if height == 0
            || width == 0
            || self.merge_size == 0
            || patches > self.max_patches
            || self.plans.iter().all(|p| p.patches < patches)
        {
            return Err("Invalid image grid or vision capacity".into());
        }
        match self.layout {
            ImageLayout::Qwen => {
                if !height.is_multiple_of(self.merge_size) || !width.is_multiple_of(self.merge_size)
                {
                    return Err("Image grid must align to the merger".into());
                }
                Ok(patches / self.merge_size / self.merge_size)
            }
            ImageLayout::DeepseekV41 => {
                let features = height
                    .div_ceil(self.merge_size)
                    .checked_mul(width.div_ceil(self.merge_size) + 1)
                    .and_then(|n| n.checked_add(2))
                    .ok_or("Image span overflow")?;
                if self.max_image_tokens.is_none_or(|limit| features > limit) {
                    return Err("Image span exceeds its checkpoint token budget".into());
                }
                Ok(features)
            }
        }
    }
    pub fn patch_values(&self) -> Result<usize> {
        self.patch_size
            .checked_mul(self.patch_size)
            .and_then(|n| n.checked_mul(self.temporal_patch_size))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| "Patch size overflow".into())
    }
    pub fn feature_count(&self, image: &ImageInput) -> Result<usize> {
        self.encoder_patch_values()?;
        let features = self.grid_features(image.grid_height, image.grid_width)?;
        let n = image
            .grid_height
            .checked_mul(image.grid_width)
            .ok_or("Image grid overflow")?;
        if (if image.rgb.is_empty() {
            image.pixels.len()
                != n.checked_mul(self.patch_values()?)
                    .ok_or("Pixel size overflow")?
        } else {
            !image.pixels.is_empty()
                || image.rgb.len()
                    != n.checked_mul(self.patch_size * self.patch_size * 3)
                        .ok_or("Pixel size overflow")?
        }) || image
            .pixels
            .iter()
            .any(|p| !p.is_finite() || !(-1.001..=1.001).contains(p))
        {
            return Err("Invalid image grid, pixel data or vision capacity".into());
        }
        Ok(features)
    }
    /// Bind media to complete placeholder spans before admitting native state.
    pub fn image_spans(
        &self,
        tokens: &[u32],
        images: &[ImageInput],
    ) -> Result<Vec<std::ops::Range<usize>>> {
        let mut spans = Vec::with_capacity(images.len());
        let mut cursor = 0usize;
        let mut total = 0usize;
        for image in images {
            let start = tokens[cursor..]
                .iter()
                .position(|&id| id == self.image_token_id)
                .map(|i| i + cursor)
                .ok_or("Image marker missing")?;
            let features = self.feature_count(image)?;
            total = total
                .checked_add(features)
                .filter(|&n| n <= self.max_features)
                .ok_or("Image feature capacity exceeded")?;
            cursor = start
                .checked_add(features)
                .filter(|&n| n <= tokens.len())
                .ok_or("Image span outside prompt")?;
            if tokens[start..cursor]
                .iter()
                .any(|&id| id != self.image_token_id)
            {
                return Err("Invalid image token span".into());
            }
            spans.push(start..cursor);
        }
        if tokens[cursor..].contains(&self.image_token_id) {
            return Err("Unbound image tokens".into());
        }
        Ok(spans)
    }
    pub fn validate(&self, m: &Manifest) -> Result<()> {
        let (alignment, output_features, output_dtype, dimensions_valid) = match self.layout {
            ImageLayout::Qwen => (
                4,
                self.max_patches / 4,
                Dtype::F16,
                self.patch_size == 16 && self.temporal_patch_size == 2 && self.merge_size == 2,
            ),
            ImageLayout::DeepseekV41 => (
                1,
                self.max_image_tokens.unwrap_or(0),
                Dtype::Bf16,
                self.patch_size == 14
                    && self.temporal_patch_size == 1
                    && self.merge_size == 3
                    && self.max_image_tokens.is_some_and(|n| n >= 4),
            ),
        };
        if self.hidden == 0
            || !dimensions_valid
            || self.max_patches == 0
            || !self.max_patches.is_multiple_of(alignment)
            || self.max_features == 0
            || self.max_features > m.max_context
            || output_features > self.max_features
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
            || self.layout == ImageLayout::Qwen
                && ids.into_iter().collect::<BTreeSet<_>>().len() != 3
        {
            return Err("Invalid vision token IDs".into());
        }
        let mut sizes = BTreeSet::new();
        for p in &self.plans {
            if p.patches == 0
                || p.min_patches > p.patches
                || !p.min_patches.is_multiple_of(alignment)
                || (p.min_patches > 0
                    && m.row_execution.as_ref().is_none_or(|rows| {
                        !rows.programs.contains(&p.program)
                            || rows.program_capacity(&p.program) < p.patches
                    }))
                || !p.patches.is_multiple_of(alignment)
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
                product(self.max_patches, self.encoder_patch_values()?)?,
            ),
            (&self.grid, Dtype::I32, 2),
            (&self.length, Dtype::I32, 1),
            (
                &self.output,
                output_dtype,
                product(output_features, self.hidden)?,
            ),
            (
                &self.features,
                output_dtype,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    fn spec() -> VisionSpec {
        serde_json::from_value(
            serde_json::json!({"hidden":8,"patch_size":16,"temporal_patch_size":2,
            "merge_size":2,"max_patches":64,"max_features":16,"pixels":"p","grid":"g",
            "length":"l","output":"o","features":"f","feature_index":"i","mrope_positions":"r",
            "plans":[{"patches":64,"program":"v"}],"image_token_id":10,"vision_start_token_id":11,
            "vision_end_token_id":12}),
        )
        .unwrap()
    }
    #[test]
    fn deepseek_patches_are_row_major_and_spans_include_learned_delimiters() {
        let mut v = spec();
        v.layout = ImageLayout::DeepseekV41;
        v.max_image_tokens = Some(1024);
        v.patch_size = 14;
        v.temporal_patch_size = 1;
        v.merge_size = 3;
        let image = ImageInput {
            grid_height: 2,
            grid_width: 4,
            rgb: (0..28 * 56 * 3).map(|n| (n * 37 % 256) as u8).collect(),
            ..Default::default()
        };
        assert_eq!(v.feature_count(&image).unwrap(), 5);
        let values: Vec<_> = image.normalized(&v).collect();
        assert_eq!(values.len(), 8 * 588);
        for row in 0..2 {
            for col in 0..4 {
                for ch in 0..3 {
                    for y in 0..14 {
                        for x in 0..14 {
                            let expected =
                                image.rgb[(((row * 14 + y) * 56 + col * 14 + x) * 3) + ch] as f32
                                    / 127.5
                                    - 1.0;
                            assert_eq!(
                                values[(row * 4 + col) * 588 + ch * 196 + y * 14 + x],
                                expected
                            );
                        }
                    }
                }
            }
        }
        v.pixel_stride = Some(640);
        let padded: Vec<_> = image.encoded(&v).collect();
        assert_eq!(padded.len(), 8 * 640);
        for (physical, logical) in padded
            .as_chunks::<640>()
            .0
            .iter()
            .zip(values.as_chunks::<588>().0)
        {
            assert_eq!(&physical[..588], logical);
            assert!(physical[588..].iter().all(|&v| v == 0.0));
        }
        assert_eq!(v.grid_features(4, 5).unwrap(), 8);
        v.max_image_tokens = Some(4);
        assert!(v.feature_count(&image).is_err());
    }
    #[test]
    fn compact_rgb_preserves_patch_channels_and_duplicate_temporal_planes() {
        let v = spec();
        let image = ImageInput {
            grid_height: 2,
            grid_width: 4,
            rgb: (0..32 * 64 * 3).map(|n| (n * 37 % 256) as u8).collect(),
            ..Default::default()
        };
        assert_eq!(v.feature_count(&image).unwrap(), 2);
        let values: Vec<_> = image.normalized(&v).collect();
        assert_eq!(values.len(), 8 * 1536);
        for (patch, (y, x)) in [
            (0, 0),
            (0, 16),
            (16, 0),
            (16, 16),
            (0, 32),
            (0, 48),
            (16, 32),
            (16, 48),
        ]
        .into_iter()
        .enumerate()
        {
            for ch in 0..3 {
                for row in 0..16 {
                    for col in 0..16 {
                        let expected =
                            image.rgb[((y + row) * 64 + x + col) * 3 + ch] as f32 / 127.5 - 1.0;
                        for frame in 0..2 {
                            assert_eq!(
                                values[patch * 1536 + ch * 512 + frame * 256 + row * 16 + col]
                                    .to_bits(),
                                expected.to_bits()
                            );
                        }
                    }
                }
            }
        }
        let old: ImageInput = serde_json::from_value(
            serde_json::json!({"grid_height":2,"grid_width":4,"pixels":values}),
        )
        .unwrap();
        assert_eq!(v.feature_count(&old).unwrap(), 2);
        assert_eq!(old.normalized(&v).collect::<Vec<_>>(), values);
        assert_eq!(image.host_bytes() * 8, old.host_bytes());
        let mut invalid = image.clone();
        invalid.rgb.pop();
        assert!(v.feature_count(&invalid).is_err());
        invalid = image;
        invalid.pixels.push(0.0);
        assert!(v.feature_count(&invalid).is_err());
    }
}
