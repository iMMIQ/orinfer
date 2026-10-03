use super::*;

impl ModelRuntime {
    /// Full chunks use prefill graphs. A remaining tail is teacher-forced through
    /// the M=1 graph: no dummy IDs enter attention, convolution or GDN state.
    pub(crate) fn generate(
        &mut self,
        input: &[u32],
        images: Option<&[crate::vision::ImageInput]>,
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        let m = &self.manifest;
        if input.is_empty()
            || limit == 0
            || input
                .len()
                .checked_add(limit)
                .is_none_or(|n| n > m.max_context)
            || input.iter().any(|&id| id as usize >= m.vocab)
        {
            return Err("Invalid input, generation length or context budget".into());
        }
        options.validate()?;
        self.speculation_statistics = None;
        let mtp = self.manifest.mtp.clone();
        self.execution.reset_sequence(&m.reset_buffers)?;
        self.prepare_visual(input, images.unwrap_or(&[]), &cancelled)?;
        let mut offset = 0;
        let mut last_head = None;
        while offset < input.len() {
            if cancelled() {
                return Err("Request cancelled during prefill".into());
            }
            let remaining = input.len() - offset;
            let plan = m
                .prefill_plans
                .iter()
                .filter(|p| p.chunk_tokens <= remaining)
                .max_by_key(|p| p.chunk_tokens);
            let selected = plan
                .map(|p| {
                    (
                        p.chunk_tokens,
                        p.prefill_program.as_str(),
                        p.head_program.as_str(),
                    )
                })
                .or_else(|| {
                    (m.prefill_plans.is_empty() && m.chunk_tokens <= remaining).then_some((
                        m.chunk_tokens,
                        "prefill",
                        "head",
                    ))
                });
            if let Some((chunk, program, head)) = selected {
                self.upload_ids(&m.input, &input[offset..offset + chunk])?;
                self.launch_program(program, ExecutionPhase::Prefill)?;
                if let Some(spec) = &mtp {
                    self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
                }
                offset += chunk;
                last_head = Some(head);
            } else {
                for id in &input[offset..] {
                    if cancelled() {
                        return Err("Request cancelled during prefill".into());
                    }
                    self.upload_ids(&m.token, std::slice::from_ref(id))?;
                    self.launch_program("decode", ExecutionPhase::Prefill)?;
                    if let Some(spec) = &mtp {
                        self.mtp_capture(spec, 1, ExecutionPhase::Prefill)?;
                    }
                }
                offset = input.len();
                last_head = None;
            }
        }
        if let Some(head) = last_head {
            self.launch_program(head, ExecutionPhase::Prefill)?;
        }
        if self.read_control(&m.position)? as usize != input.len() {
            return Err("Prefill position mismatch".into());
        }
        if let Some(spec) = &mtp {
            return self.mtp_generate(spec, input, limit, options, &cancelled, &mut emit);
        }
        let mut generated = 0;
        let mut history = input.to_vec();
        for step in 0..limit {
            if cancelled() {
                return Err("Request cancelled during decode".into());
            }
            let token = self.select_target(&history, options, step)?;
            self.upload_ids(&m.token, &[token])?;
            generated += 1;
            history.push(token);
            if !emit(token) {
                break;
            }
            if step + 1 < limit {
                self.launch_program("decode", ExecutionPhase::Decode)?;
            }
        }
        if self.read_control(&m.position)? as usize != input.len() + generated - 1 {
            return Err("Decode position mismatch".into());
        }
        Ok(generated)
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
