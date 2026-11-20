//! Complete bidirectional embedding requests through the shared CUDA executor.
pub use crate::encoder_rows::Statistics;
use crate::{
    artifact::Result,
    encoder_rows::LoadedEncoder,
    execution::{ExecutionPhase, LoadOptions},
};
use std::{collections::BTreeMap, path::Path};

pub use orinfer_model_sdk::embedding::Spec;

pub struct Model(LoadedEncoder<Spec>);

impl Model {
    pub fn load(path: &Path, options: LoadOptions) -> Result<Self> {
        LoadedEncoder::load(
            path,
            options,
            |m| m.embedding.clone(),
            "Model package does not provide embeddings",
            "Embedding weight accounting overflow",
        )
        .map(Self)
    }
    pub fn spec(&self) -> &Spec {
        &self.0.spec
    }
    pub fn max_context(&self) -> usize {
        self.0.context
    }
    pub fn frontend_assets(&self) -> &BTreeMap<String, String> {
        &self.0.frontend_assets
    }
    pub fn statistics(&self) -> &Statistics {
        &self.0.statistics
    }

    pub fn embed(
        &mut self,
        tokens: &[u32],
        images: &[crate::vision::ImageInput],
        feature_index: &[i32],
        dimensions: Option<usize>,
    ) -> Result<Vec<f32>> {
        if self.0.spec.rows.is_some() {
            return Ok(self
                .embed_many(&[(tokens, images, feature_index)], dimensions)?
                .remove(0));
        }
        self.validate_input(tokens, images, feature_index, dimensions)?;
        let dimension = dimensions.unwrap_or(self.0.spec.dimension);
        let mut offset = 0;
        self.upload_images(images, &mut offset)?;
        let e = &self.0.execution;
        e.upload_ids(&self.0.input, tokens)?;
        e.upload_ids(&self.0.spec.length, &[tokens.len() as u32])?;
        let index: Vec<u8> = feature_index.iter().flat_map(|i| i.to_le_bytes()).collect();
        e.upload_bytes(&self.0.spec.feature_index, &index)?;
        e.submit_program_rows(&self.0.spec.program, tokens.len(), ExecutionPhase::Prefill)?;
        let bytes = e.download_bytes(&self.0.spec.output, self.0.spec.dimension * 4)?;
        vector(&bytes, dimension)
    }
    fn validate_input(
        &self,
        tokens: &[u32],
        images: &[crate::vision::ImageInput],
        feature_index: &[i32],
        dimensions: Option<usize>,
    ) -> Result<usize> {
        let dimension = dimensions.unwrap_or(self.0.spec.dimension);
        if dimension == 0
            || dimension > self.0.spec.dimension
            || !self.0.tokens_valid(tokens)
            || feature_index.len() != tokens.len()
        {
            return Err("Invalid embedding length, tokens, dimensions or feature index".into());
        }
        let mut features = 0usize;
        for image in images {
            let patches = image
                .grid_height
                .checked_mul(image.grid_width)
                .ok_or("Image grid overflow")?;
            if !image.rgb.is_empty()
                || image.grid_height == 0
                || image.grid_width == 0
                || image.grid_height % self.0.spec.merge_size != 0
                || image.grid_width % self.0.spec.merge_size != 0
                || patches > self.0.spec.max_patches
                || image.pixels.len() != patches * self.0.spec.patch_size.pow(2) * 3
                || image
                    .pixels
                    .iter()
                    .any(|x| !x.is_finite() || x.abs() > 65504.0)
            {
                return Err("Invalid embedding image patches".into());
            }
            features += patches / self.0.spec.merge_size.pow(2);
        }
        if features > self.0.context {
            return Err("Image features exceed embedding context".into());
        }
        let mut next = 0usize;
        for (&token, &index) in tokens.iter().zip(feature_index) {
            if index >= 0 {
                if token != self.0.spec.image_token_id || index as usize != next || next >= features
                {
                    return Err("Invalid embedding image token ownership".into());
                }
                next += 1;
            } else if index != -1 || token == self.0.spec.image_token_id {
                return Err("Unbound embedding image token".into());
            }
        }
        if next != features {
            return Err("Unused embedding image features".into());
        }
        Ok(features)
    }
    fn upload_images(
        &self,
        images: &[crate::vision::ImageInput],
        offset: &mut usize,
    ) -> Result<()> {
        let e = &self.0.execution;
        for image in images {
            let patches = image.grid_height * image.grid_width;
            let pixels: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|&x| half::f16::from_f32(x).to_bits().to_le_bytes())
                .collect();
            e.upload_bytes(&self.0.spec.pixels, &pixels)?;
            e.upload_ids(
                &self.0.spec.grid,
                &[image.grid_height as u32, image.grid_width as u32],
            )?;
            e.upload_ids(&self.0.spec.vision_length, &[patches as u32])?;
            e.submit_program_rows(&self.0.spec.vision_program, patches, ExecutionPhase::Vision)?;
            let count = patches / self.0.spec.merge_size.pow(2);
            e.copy_range_async(
                &self.0.spec.vision_output,
                0,
                &self.0.spec.features,
                *offset * self.0.spec.dimension * 2,
                count * self.0.spec.dimension * 2,
            )?;
            *offset += count;
        }
        Ok(())
    }
    pub fn embed_many(
        &mut self,
        inputs: &[(&[u32], &[crate::vision::ImageInput], &[i32])],
        dimensions: Option<usize>,
    ) -> Result<Vec<Vec<f32>>> {
        let Some(rows) = self.0.spec.rows.clone() else {
            return inputs
                .iter()
                .map(|&(tokens, images, index)| self.embed(tokens, images, index, dimensions))
                .collect();
        };
        for &(tokens, images, index) in inputs {
            self.validate_input(tokens, images, index, dimensions)?;
        }
        let packets = rows.packets(
            &inputs.iter().map(|t| t.0.len()).collect::<Vec<_>>(),
            self.0.context,
        )?;
        let mut output = Vec::with_capacity(inputs.len());
        for packet in packets {
            let mut tokens = vec![0; packet.rows];
            let mut indices = vec![-1i32; packet.rows];
            let mut offset = 0;
            for (owner, &[start, end]) in packet.segments.iter().enumerate() {
                let (ids, images, index) = inputs[packet.first + owner];
                tokens[start as usize..end as usize].copy_from_slice(ids);
                for (target, &source) in indices[start as usize..end as usize].iter_mut().zip(index)
                {
                    if source >= 0 {
                        *target = source + offset as i32;
                    }
                }
                self.upload_images(images, &mut offset)?;
            }
            let e = &self.0.execution;
            e.upload_ids(&self.0.input, &tokens)?;
            e.upload_bytes(
                &self.0.spec.feature_index,
                &indices
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            crate::encoder_rows::upload(e, &rows, &packet)?;
            e.submit_program_rows(&self.0.spec.program, packet.rows, ExecutionPhase::Prefill)?;
            let bytes = e.download_bytes(
                &self.0.spec.output,
                packet.segments.len() * self.0.spec.dimension * 4,
            )?;
            for item in bytes.chunks_exact(self.0.spec.dimension * 4) {
                output.push(vector(item, dimensions.unwrap_or(self.0.spec.dimension))?);
            }
        }
        Ok(output)
    }
}
fn vector(bytes: &[u8], dimension: usize) -> Result<Vec<f32>> {
    let mut output: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .take(dimension)
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let norm = output
        .iter()
        .map(|&x| (x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm <= 1e-12 {
        return Err("Embedding execution produced a nonfinite or zero vector".into());
    }
    for x in &mut output {
        *x = (*x as f64 / norm) as f32;
    }
    Ok(output)
}
