use super::*;

impl ModelRuntime {
    pub(super) fn prepare_visual(
        &self,
        input: &[u32],
        images: &[crate::vision::ImageInput],
        cancelled: &impl Fn() -> bool,
    ) -> Result<()> {
        let Some(v) = &self.manifest.vision else {
            return if images.is_empty() {
                Ok(())
            } else {
                Err("Model has no vision adapter".into())
            };
        };
        let (index, positions) = v.layout(input, images, self.manifest.max_context)?;
        self.upload_bytes(
            &v.feature_index,
            &index
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.upload_ids(&v.mrope_positions, &positions)?;
        let mut offset = 0;
        for image in images {
            if cancelled() {
                return Err("Request cancelled during image encoding".into());
            }
            let features = v.feature_count(image)?;
            let patches = image.grid_height * image.grid_width;
            let plan = v
                .plans
                .iter()
                .filter(|p| p.patches >= patches)
                .min_by_key(|p| p.patches)
                .ok_or("No image graph")?;
            let raw: Vec<u8> = image
                .pixels
                .iter()
                .flat_map(|&x| match v.dtype {
                    crate::vision::Precision::F16 => half::f16::from_f32(x).to_bits().to_le_bytes(),
                    crate::vision::Precision::Bf16 => {
                        half::bf16::from_f32(x).to_bits().to_le_bytes()
                    }
                })
                .collect();
            self.upload_bytes(&v.pixels, &raw)?;
            self.upload_ids(
                &v.grid,
                &[image.grid_height as u32, image.grid_width as u32],
            )?;
            self.upload_ids(&v.length, &[patches as u32])?;
            self.launch_program(&plan.program)?;
            self.execution.copy_range(
                &v.output,
                &v.features,
                offset * v.hidden * 2,
                features * v.hidden * 2,
            )?;
            offset += features;
        }
        Ok(())
    }
}
