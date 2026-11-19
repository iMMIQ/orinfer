//! GPU sampling shares the ordinary request's history, seed and grammar law.
use super::*;

const SAMPLING_NEEDS_RADIX: u32 = 2;

impl ModelRuntime {
    pub(super) fn gpu_sample(
        &self,
        history: &[u32],
        options: &crate::sampling::Options,
        step: usize,
        mask: Option<&[u32]>,
    ) -> Result<Option<u32>> {
        let Some(spec) = &self.manifest.sampling else {
            return Ok(None);
        };
        // Probability reporting and speculative p/q laws retain their current
        // reference path. The batched wrapper uses exactly these same inputs.
        if options.temperature == 0.0 || options.top_logprobs.is_some() {
            return Ok(None);
        }
        for (name, bytes) in inputs(
            history,
            options,
            step,
            mask,
            self.manifest.vocab,
            self.manifest.max_context,
        )? {
            let destination = match name {
                "History" => &spec.history,
                "BiasIds" => &spec.bias_ids,
                "BiasValues" => &spec.bias_values,
                "Mask" => &spec.mask,
                "Control" => &spec.control,
                "Parameters" => &spec.parameters,
                _ => unreachable!(),
            };
            self.upload_bytes(destination, &bytes)?;
        }
        self.execution
            .submit_program(&spec.program, ExecutionPhase::Decode)?;
        let output = self.read_sampling_output(&spec.output, None)?;
        validate_output(&output, self.manifest.vocab).map(Some)
    }

    pub(super) fn read_sampling_output(
        &self,
        output: &str,
        rows: Option<usize>,
    ) -> Result<Vec<u32>> {
        let count = rows.unwrap_or(1) * 2;
        let samples = self.read_controls(output, count)?;
        if !samples
            .as_chunks::<2>()
            .0
            .iter()
            .any(|pair| pair[1] == SAMPLING_NEEDS_RADIX)
        {
            return Ok(samples);
        }
        // The fast stage leaves the radix state live only for rows that need
        // fallback. Reuse the same pointers and preserve completed samples.
        if let Some(rows) = rows {
            self.execution.submit_program_rows(
                "sampling_rows_fallback",
                rows,
                ExecutionPhase::Decode,
            )?;
        } else {
            self.execution
                .submit_program("sampling_fallback", ExecutionPhase::Decode)?;
        }
        self.read_controls(output, count)
    }
}

pub(super) fn validate_output(values: &[u32], vocab: usize) -> Result<u32> {
    if values.len() != 2 || values[1] != 0 || values[0] as usize >= vocab {
        return Err("GPU sampling has invalid logits or empty probability support".into());
    }
    Ok(values[0])
}

pub(super) fn inputs(
    history: &[u32],
    options: &crate::sampling::Options,
    step: usize,
    mask: Option<&[u32]>,
    vocab: usize,
    context: usize,
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    options.validate()?;
    if history.len() > context {
        return Err("Sampling history exceeds context".into());
    }
    let mut result = Vec::new();
    let use_history = options.repetition_penalty != 1.
        || options.presence_penalty != 0.
        || options.frequency_penalty != 0.;
    let ids = |values: &[u32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    if use_history {
        result.push(("History", ids(history)));
    }
    let bias: Vec<_> = options
        .logit_bias
        .iter()
        .filter(|(id, _)| **id < vocab as u32)
        .collect();
    result.push((
        "BiasIds",
        ids(&bias.iter().map(|(id, _)| **id).collect::<Vec<_>>()),
    ));
    result.push((
        "BiasValues",
        bias.iter().flat_map(|(_, v)| v.to_le_bytes()).collect(),
    ));
    if let Some(mask) = mask {
        result.push(("Mask", ids(&padded_mask(mask, vocab))));
    }
    result.push((
        "Control",
        ids(&[
            if use_history { history.len() as u32 } else { 0 },
            bias.len() as u32,
            u32::from(mask.is_some()),
            options.top_k.min(vocab) as u32,
        ]),
    ));
    result.push((
        "Parameters",
        [
            options.temperature,
            options.top_p,
            options.presence_penalty,
            options.frequency_penalty,
            options.repetition_penalty,
            crate::sampling::counter_uniform(options.seed, 0, step as u64),
        ]
        .into_iter()
        .flat_map(f64::to_le_bytes)
        .collect(),
    ));
    Ok(result)
}

pub(super) fn padded_mask(mask: &[u32], vocab: usize) -> Vec<u32> {
    let mut result = vec![0; vocab.div_ceil(32)];
    for (destination, source) in result.iter_mut().zip(mask) {
        *destination = *source;
    }
    if !vocab.is_multiple_of(32)
        && let Some(last) = result.last_mut()
    {
        *last &= u32::MAX >> (32 - vocab % 32);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::padded_mask;
    #[test]
    fn tokenizer_mask_excludes_head_padding_and_unused_tail_bits() {
        assert_eq!(padded_mask(&[u32::MAX], 65), [u32::MAX, 0, 0]);
        assert_eq!(padded_mask(&[u32::MAX; 4], 65), [u32::MAX, u32::MAX, 1]);
        assert_eq!(padded_mask(&[], 32), [0]);
        assert!(padded_mask(&[1], 0).is_empty());
    }
}
