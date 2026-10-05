use super::*;

impl ModelRuntime {
    /// Full chunks use prefill graphs. A remaining tail is teacher-forced through
    /// the M=1 graph: no dummy IDs enter attention, convolution or GDN state.
    pub(crate) fn generate_reference(
        &mut self,
        input: &[u32],
        images: Option<&[crate::vision::ImageInput]>,
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        emit: impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        self.execution.prepare_legacy_sequence()?;
        let mut prefix = super::prefix::Context::default();
        let outcome = self.generate_reference_inner(
            input,
            images,
            limit,
            options,
            cancelled,
            emit,
            &mut prefix,
        );
        self.prefix_statistics = prefix.statistics.clone();
        self.prefix_hints.clear();
        prefix.kv.clear();
        let cleanup = self.execution.collect_snapshots();
        match outcome {
            Ok(count) => cleanup.map(|()| count),
            Err(error) => Err(error),
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn generate_reference_inner(
        &mut self,
        input: &[u32],
        images: Option<&[crate::vision::ImageInput]>,
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: impl Fn() -> bool,
        mut emit: impl FnMut(u32) -> bool,
        prefix: &mut super::prefix::Context,
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
        let media = self.prefix_media(input, images.unwrap_or(&[]))?;
        let (mut offset, mut warm) = self.restore_prefix(input, &media, prefix)?;
        if offset != 0
            && offset < input.len()
            && let Some(spec) = &mtp
        {
            // Pair the cached endpoint h[P-1] with x[P] before reusing its ring.
            let at = Instant::now();
            self.mtp_warm_state(
                spec,
                &input[offset..offset + 1],
                &cancelled,
                ExecutionPhase::Prefill,
                false,
            )?;
            warm.tokens = offset;
            warm.seconds += at.elapsed().as_secs_f64();
        }
        let mut checkpoints = std::collections::BTreeSet::new();
        let hints = std::mem::take(&mut self.prefix_hints);
        if self.prefix_cache.budget != 0 {
            checkpoints.extend((8192..input.len()).step_by(8192));
            for hint in hints.into_iter().filter(|&p| p > offset && p < input.len()) {
                if self.admit_prefix_checkpoint(hint, input.len(), offset, &checkpoints)? {
                    checkpoints.insert(hint);
                }
            }
            let branch = prefix.statistics.matched_tokens;
            if branch > offset
                && branch < input.len()
                && self.admit_prefix_checkpoint(branch, input.len(), offset, &checkpoints)?
            {
                checkpoints.insert(branch);
            }
        }
        let mut last_head = None;
        while offset < input.len() {
            if cancelled() {
                return Err("Request cancelled during prefill".into());
            }
            let boundary = checkpoints
                .range((
                    std::ops::Bound::Excluded(offset),
                    std::ops::Bound::Unbounded,
                ))
                .next()
                .copied()
                .unwrap_or(input.len());
            let remaining = boundary - offset;
            let m = &self.manifest;
            let plan = m
                .prefill_plans
                .iter()
                .filter(|p| p.chunk_tokens <= remaining)
                .max_by_key(|p| p.chunk_tokens);
            let selected = plan
                .map(|p| {
                    (
                        p.chunk_tokens,
                        p.prefill_program.clone(),
                        p.head_program.clone(),
                    )
                })
                .or_else(|| {
                    (m.prefill_plans.is_empty() && m.chunk_tokens <= remaining).then_some((
                        m.chunk_tokens,
                        "prefill".to_string(),
                        "head".to_string(),
                    ))
                });
            let compute_at = Instant::now();
            let (chunk, head) = if let Some((chunk, program, head)) = selected {
                self.upload_ids(&m.input, &input[offset..offset + chunk])?;
                if !m.batch_profiles.is_empty() {
                    self.upload_ids("BatchSegmentLength", &[chunk as u32])?;
                    self.upload_ids("BatchLastIndex", &[(chunk - 1) as u32])?;
                }
                self.launch_program(&program, ExecutionPhase::Prefill)?;
                (chunk, Some(head))
            } else {
                self.upload_ids(&m.token, &input[offset..offset + 1])?;
                self.launch_program("decode", ExecutionPhase::Prefill)?;
                (1, None)
            };
            let endpoint = offset + chunk;
            let checkpoint = checkpoints.contains(&endpoint);
            let mut compute_s;
            if let Some(spec) = &mtp {
                self.mtp_capture(spec, chunk, ExecutionPhase::Prefill)?;
                compute_s = compute_at.elapsed().as_secs_f64();
                if checkpoint && spec.hidden_ring.is_some() {
                    let at = Instant::now();
                    self.mtp_warm_state(
                        spec,
                        &input[warm.tokens + 1..endpoint],
                        &cancelled,
                        ExecutionPhase::Prefill,
                        false,
                    )?;
                    warm.tokens = endpoint - 1;
                    let seconds = at.elapsed().as_secs_f64();
                    warm.seconds += seconds;
                    compute_s += seconds;
                }
            } else {
                compute_s = compute_at.elapsed().as_secs_f64();
            }
            if checkpoint {
                if let Some(head) = &head {
                    self.launch_program(head, ExecutionPhase::Prefill)?;
                }
                self.store_prefix(&input[..endpoint], &media, warm.tokens, true, prefix)?;
            }
            if let Some(spec) = &mtp
                && spec.hidden_ring.is_some()
                && endpoint < input.len()
            {
                let at = Instant::now();
                self.mtp_warm_state(
                    spec,
                    &input[warm.tokens + 1..endpoint + 1],
                    &cancelled,
                    ExecutionPhase::Prefill,
                    false,
                )?;
                warm.tokens = endpoint;
                let seconds = at.elapsed().as_secs_f64();
                warm.seconds += seconds;
                compute_s += seconds;
            }
            self.prefill_costs.observe(chunk, compute_s);
            offset = endpoint;
            last_head = head;
        }
        if let Some(head) = last_head {
            self.launch_program(&head, ExecutionPhase::Prefill)?;
        }
        if self.read_control(&self.manifest.position)? as usize != input.len() {
            return Err("Prefill position mismatch".into());
        }
        if self.prefix_cache.budget != 0
            && let Some(spec) = &mtp
            && spec.hidden_ring.is_some()
            && warm.tokens < input.len() - 1
        {
            let at = Instant::now();
            self.mtp_warm_state(
                spec,
                &input[warm.tokens + 1..],
                &cancelled,
                ExecutionPhase::Prefill,
                false,
            )?;
            warm.tokens = input.len() - 1;
            warm.seconds += at.elapsed().as_secs_f64();
        }
        self.store_prefix(input, &media, warm.tokens, true, prefix)?;
        let m = &self.manifest;
        if let Some(spec) = &mtp {
            return self.mtp_generate(
                spec,
                (input, warm, &media, prefix),
                limit,
                options,
                &cancelled,
                &mut emit,
            );
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
        self.store_decoded_prefix(&history, &media, &cancelled, prefix)?;
        Ok(generated)
    }
}
