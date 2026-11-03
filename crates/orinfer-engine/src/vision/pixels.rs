//! Exact checkpoint pixel packing shared by CUDA and Ascend frontends.
use super::{ImageInput, ImageLayout, Precision, VisionSpec};
use crate::artifact::Result;
use std::sync::OnceLock;

/// The caller validates image geometry with `feature_count` before packing.
pub(crate) fn encode(image: &ImageInput, spec: &VisionSpec) -> Result<Vec<u8>> {
    let logical = spec.patch_values()?;
    let stride = spec.encoder_patch_values()? * 2;
    let bytes = image
        .grid_height
        .checked_mul(image.grid_width)
        .and_then(|n| n.checked_mul(stride))
        .ok_or("Image upload extent overflow")?;
    let mut output = vec![0u8; bytes];
    if image.rgb.is_empty() {
        let write = |output: &mut [u8], convert: fn(f32) -> [u8; 2]| {
            for (source, destination) in image
                .pixels
                .chunks_exact(logical)
                .zip(output.chunks_exact_mut(stride))
            {
                for (&value, target) in source
                    .iter()
                    .zip(destination[..logical * 2].as_chunks_mut::<2>().0)
                {
                    target.copy_from_slice(&convert(value));
                }
            }
        };
        match spec.dtype {
            Precision::F16 => write(&mut output, |v| half::f16::from_f32(v).to_le_bytes()),
            Precision::Bf16 => write(&mut output, |v| half::bf16::from_f32(v).to_le_bytes()),
        }
        return Ok(output);
    }
    // RGB8 has only 256 normalized values. These tables preserve the scalar
    // division and rounding exactly while avoiding per-pixel conversion work.
    static FP16: OnceLock<[[u8; 2]; 256]> = OnceLock::new();
    static BF16: OnceLock<[[u8; 2]; 256]> = OnceLock::new();
    let table = match spec.dtype {
        Precision::F16 => FP16.get_or_init(|| {
            std::array::from_fn(|i| half::f16::from_f32(i as f32 / 127.5 - 1.0).to_le_bytes())
        }),
        Precision::Bf16 => BF16.get_or_init(|| {
            std::array::from_fn(|i| half::bf16::from_f32(i as f32 / 127.5 - 1.0).to_le_bytes())
        }),
    };
    let patch = spec.patch_size;
    let merge = if spec.layout == ImageLayout::DeepseekV41 {
        1
    } else {
        spec.merge_size
    };
    let width = image.grid_width * patch;
    let mut patches = output.chunks_exact_mut(stride);
    for br in 0..image.grid_height / merge {
        for bc in 0..image.grid_width / merge {
            for inner in 0..merge * merge {
                let y = (br * merge + inner / merge) * patch;
                let x = (bc * merge + inner % merge) * patch;
                let destination = patches.next().ok_or("Image packing grid differs")?;
                let mut offset = 0;
                for channel in 0..3 {
                    for _ in 0..spec.temporal_patch_size {
                        for row in 0..patch {
                            let first = ((y + row) * width + x) * 3 + channel;
                            for column in 0..patch {
                                destination[offset..offset + 2].copy_from_slice(
                                    &table[image.rgb[first + column * 3] as usize],
                                );
                                offset += 2;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_rgb_matches_scalar_reference_for_both_layouts_and_precisions() {
        for deepseek in [false, true] {
            let spec = serde_json::json!({"layout":if deepseek {"deepseek_v41"}else{"qwen"},"dtype":"bf16","max_image_tokens":1024,"hidden":5120,"patch_size":if deepseek {14}else{16},"temporal_patch_size":if deepseek {1}else{2},"merge_size":if deepseek {3}else{2},"max_patches":64,"max_features":64,"pixels":"p","grid":"g","length":"n","output":"o","features":"f","feature_index":"i","mrope_positions":"r","plans":[{"patches":64,"program":"v"}],"image_token_id":10,"vision_start_token_id":11,"vision_end_token_id":12});
            let mut spec: VisionSpec = serde_json::from_value(spec).unwrap();
            let h = if deepseek { 3 } else { 2 };
            let mut image = ImageInput {
                grid_height: h,
                grid_width: 4,
                rgb: (0..h * 4 * spec.patch_size * spec.patch_size * 3)
                    .map(|i| (i * 37 % 256) as u8)
                    .collect(),
                ..Default::default()
            };
            for precision in [Precision::F16, Precision::Bf16] {
                spec.dtype = precision;
                spec.pixel_stride = Some(spec.patch_values().unwrap() + 32);
                spec.feature_count(&image).unwrap();
                let reference: Vec<_> = image
                    .encoded(&spec)
                    .flat_map(|v| match precision {
                        Precision::F16 => half::f16::from_f32(v).to_le_bytes(),
                        Precision::Bf16 => half::bf16::from_f32(v).to_le_bytes(),
                    })
                    .collect();
                assert_eq!(encode(&image, &spec).unwrap(), reference);
            }
            image.pixels = image.normalized(&spec).collect();
            image.rgb.clear();
            let reference: Vec<_> = image
                .encoded(&spec)
                .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            assert_eq!(encode(&image, &spec).unwrap(), reference);
        }
    }
}
