use super::*;

impl ModelRuntime {
    /// Whole-request convenience API uses the same persistent state machine as
    /// continuous batching. No second production implementation of prefill/MTP.
    pub(crate) fn generate(
        &mut self,
        input: &[u32],
        images: Option<&[crate::vision::ImageInput]>,
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.reap_abandoned()?;
        if !self.reserved_requests.is_empty() {
            return Err("Complete active requests before whole-request generation".into());
        }
        let hints = std::mem::take(&mut self.prefix_hints);
        let mut request = self.start_request(
            requests::GenerationInput {
                input_tokens: input.to_vec(),
                images: images.unwrap_or(&[]).to_vec(),
                max_new_tokens: limit,
                sampling: options.clone(),
                prefix_hints: hints,
            },
            &cancelled,
        )?;
        let schedule = crate::scheduler::Options {
            max_active: 1,
            ..Default::default()
        };
        let mut delivered = 0;
        let mut stopped = false;
        let result = (|| {
            while !request.is_finished() && !stopped {
                if cancelled() {
                    return Err("Request cancelled during generation".into());
                }
                let output = self.advance_requests(&mut [&mut request], &schedule)?;
                for step in output {
                    for token in step.tokens {
                        delivered += 1;
                        if !emit(token) {
                            stopped = true;
                            break;
                        }
                    }
                }
            }
            Ok(delivered)
        })();
        let cache = result.is_ok() && delivered == request.generated_tokens() && !cancelled();
        let cleanup = self.finish_request(&mut request, cache);
        self.prefix_statistics = request.prefix_statistics().clone();
        self.speculation_statistics = self
            .manifest
            .mtp
            .as_ref()
            .map(|_| request.speculation_statistics().clone());
        match (result, cleanup) {
            (Ok(count), Ok(())) => Ok(count),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    }

    pub(super) fn select_target(
        &self,
        history: &[u32],
        options: &crate::sampling::Options,
        step: usize,
    ) -> Result<u32> {
        let m = &self.manifest;
        if self.read_control(&m.status)? != 0 {
            return Err("Model token status failure".into());
        }
        if m.greedy_sampling
            && options.temperature == 0.0
            && !options.is_greedy()
            && options.logit_bias.is_empty()
        {
            options.validate()?;
            // Counts are rebuilt from the authoritative history, including the
            // entire prompt on prefix hits and each speculative commit. They
            // never enter the prefix cache or depend on a resident batch lane.
            self.upload_ids("SamplingHistory", history)?;
            self.upload_ids("SamplingLength", &[history.len() as u32])?;
            let parameters: Vec<u8> = [
                options.repetition_penalty,
                options.presence_penalty,
                options.frequency_penalty,
            ]
            .into_iter()
            .flat_map(f64::to_le_bytes)
            .collect();
            self.upload_bytes("SamplingParameters", &parameters)?;
            self.launch_program("greedy_sampling", ExecutionPhase::Decode)?;
            if self.read_control(&m.status)? != 0 {
                return Err("Nonfinite processed sampling logits".into());
            }
            let token = self.read_control(&m.token)?;
            if token < 0 || token as usize >= m.vocab {
                return Err("Selected token outside vocabulary".into());
            }
            return Ok(token as u32);
        }
        if options.is_greedy() {
            let value = self.read_control(&m.token)?;
            if value < 0 || value as usize >= m.vocab {
                return Err("Selected token outside vocabulary".into());
            }
            Ok(value as u32)
        } else {
            let spec = m
                .buffers
                .iter()
                .find(|b| b.name == m.logits)
                .ok_or("Missing logits")?;
            let raw = self.execution.download_bytes(&m.logits, spec.bytes()?)?;
            crate::sampling::sample(&floats(&raw, spec.dtype), history, options, step)
        }
    }
}
