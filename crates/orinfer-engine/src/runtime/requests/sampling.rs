//! Independent-owner GPU sampling after the shared target packet.
use super::super::sampling::{inputs, padded_mask, validate_output};
use super::*;

impl ModelRuntime {
    pub(super) fn gpu_sample_batch(
        &self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
    ) -> Result<BTreeMap<usize, u32>> {
        let Some(batch) = self
            .manifest
            .sampling
            .as_ref()
            .and_then(|s| s.batch.as_ref())
        else {
            return Ok(BTreeMap::new());
        };
        // One sampled owner already has a captured scalar program. Avoid
        // pointer-table/row-plan setup until there is actual work to batch.
        let sampled = selected
            .iter()
            .filter(|(index, _)| {
                let req = &requests[*index];
                !req.prefilling
                    && req.sampling.temperature > 0.
                    && req.sampling.top_logprobs.is_none()
            })
            .take(2)
            .count();
        if sampled < 2 {
            return Ok(BTreeMap::new());
        }
        let mut candidates = Vec::new();
        for &(index, _) in selected {
            let req = &mut requests[index];
            if req.prefilling
                || req.sampling.temperature == 0.
                || req.sampling.top_logprobs.is_some()
            {
                continue;
            }
            if req.owner != self.owner_id || req.released {
                return Err("Request does not own a live sampling arena".into());
            }
            let mask = if req.decoder.constrained() {
                match req.decoder.gpu_mask() {
                    Ok(mask) => mask,
                    Err(_) if req.failure().is_some() => continue,
                    Err(error) => return Err(error),
                }
            } else {
                None
            };
            if mask.as_ref().is_some_and(|mask| {
                padded_mask(mask, self.manifest.vocab)
                    .iter()
                    .all(|&v| v == 0)
            }) {
                req.decoder.fail("Grammar permits no tokens".into());
                continue;
            }
            candidates.push((index, mask));
        }
        if candidates.is_empty() {
            return Ok(BTreeMap::new());
        }
        if candidates.len() > batch.capacity {
            return Err("Sampling rows exceed capacity".into());
        }
        let slots: Vec<_> = candidates.iter().map(|(i, _)| requests[*i].slot).collect();
        profile_mark("sampling_controls_begin");
        let controls = self
            .execution
            .read_sequence_controls(&slots, [&self.manifest.position, &self.manifest.status])?;
        for ((index, _), [position, status]) in candidates.iter().zip(controls) {
            let req = &requests[*index];
            if position < 0 || position as usize != req.input.len() + req.generated || status != 0 {
                return Err("Batched sampling position/status failure".into());
            }
        }
        self.execution.upload_scratch_rows(
            &batch.pointers,
            &batch.scratch,
            batch.stride,
            &self.manifest.logits,
            self.manifest.vocab * 4,
            &slots,
        )?;
        profile_mark("sampling_table");
        for (row, (index, mask)) in candidates.iter().enumerate() {
            let req = &requests[*index];
            for (name, bytes) in inputs(
                &req.history,
                &req.sampling,
                req.generated,
                mask.as_deref(),
                self.manifest.vocab,
                self.manifest.max_context,
            )? {
                self.execution.upload_bytes_at(
                    &batch.scratch,
                    row * batch.stride + batch.offsets[name],
                    &bytes,
                )?;
            }
        }
        profile_mark("sampling_inputs");
        self.execution.submit_program_rows(
            &batch.program,
            candidates.len(),
            ExecutionPhase::Decode,
        )?;
        profile_mark("sampling_submit");
        let output = self.read_controls(&batch.output, candidates.len() * 2)?;
        profile_mark("sampling_readback");
        let mut stats = self.execution.batch_statistics();
        stats.sampling_packets += 1;
        stats.sampling_rows += candidates.len();
        self.execution.set_batch_statistics(stats);
        let mut result = BTreeMap::new();
        for ((index, _), values) in candidates.into_iter().zip(output.as_chunks::<2>().0) {
            let token = match validate_output(values, self.manifest.vocab) {
                Ok(token) => token,
                Err(error) => {
                    requests[index].decoder.fail(error);
                    continue;
                }
            };
            if requests[index].decoder.constrained()
                && let Err(error) = requests[index].decoder.consume(token)
            {
                if requests[index].failure().is_some() {
                    continue;
                }
                return Err(error);
            }
            result.insert(index, token);
        }
        Ok(result)
    }
}
