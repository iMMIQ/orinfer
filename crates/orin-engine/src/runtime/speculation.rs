use super::*;

impl ModelRuntime {
    pub(super) fn mtp_capture(
        &self,
        spec: &crate::mtp::Spec,
        tokens: usize,
        phase: ExecutionPhase,
    ) -> Result<()> {
        let plan = spec
            .capture_plans
            .iter()
            .find(|p| p.tokens == tokens)
            .ok_or("Missing MTP hidden capture shape")?;
        self.launch_program(&plan.program, phase)
    }
    pub(super) fn mtp_warm(
        &self,
        spec: &crate::mtp::Spec,
        shifted_ids: &[u32],
        cancelled: &impl Fn() -> bool,
        phase: ExecutionPhase,
    ) -> Result<()> {
        let mut offset = 0;
        while offset < shifted_ids.len() {
            if cancelled() {
                return Err("Request cancelled during MTP warm/refresh".into());
            }
            let remaining = shifted_ids.len() - offset;
            let plan = spec
                .warm_plans
                .iter()
                .filter(|p| p.tokens <= remaining)
                .max_by_key(|p| p.tokens)
                .ok_or("No compatible MTP warm plan")?;
            self.upload_ids(&spec.input, &shifted_ids[offset..offset + plan.tokens])?;
            self.launch_program(&plan.program, phase)?;
            offset += plan.tokens;
            if offset == shifted_ids.len() {
                self.launch_program(&plan.head_program, phase)?;
            }
        }
        Ok(())
    }
    pub(super) fn mtp_generate(
        &mut self,
        spec: &crate::mtp::Spec,
        input: &[u32],
        limit: usize,
        options: &crate::sampling::Options,
        cancelled: &impl Fn() -> bool,
        emit: &mut impl FnMut(u32) -> bool,
    ) -> Result<usize> {
        use std::time::Instant;
        let vocab = self.manifest.vocab;
        let mut stats = crate::mtp::Statistics::default();
        let mut pending = self.select_target(input, options, 0)?;
        self.upload_ids(&self.manifest.token, &[pending])?;
        let mut history = input.to_vec();
        history.push(pending);
        let mut shifted = input[1..].to_vec();
        shifted.push(pending);
        let at = Instant::now();
        self.mtp_warm(spec, &shifted, cancelled, ExecutionPhase::Prefill)?;
        stats.initial_warm_s = at.elapsed().as_secs_f64();
        let mut generated = 1;
        if !emit(pending) || limit == 1 {
            stats.committed_tokens = generated;
            self.speculation_statistics = Some(stats);
            return Ok(generated);
        }
        while generated < limit {
            if cancelled() {
                return Err("Request cancelled during MTP decode".into());
            }
            let position = self.read_control(&self.manifest.position)? as usize;
            let capacity = self.manifest.max_context.saturating_sub(position);
            let remaining_outputs = limit - generated;
            let plan = spec
                .verification_plans
                .iter()
                .filter(|p| {
                    p.tokens <= capacity
                        && p.tokens <= remaining_outputs
                        && p.tokens <= spec.default_verification_tokens
                })
                .max_by_key(|p| p.tokens);
            let Some(plan) = plan else {
                // Final single-token tail or a context edge. The ordinary
                // target graph preserves existing sampling/state semantics.
                self.upload_ids(&self.manifest.token, &[pending])?;
                self.launch_program("decode", ExecutionPhase::Decode)?;
                self.mtp_capture(spec, 1, ExecutionPhase::Decode)?;
                pending = self.select_target(&history, options, generated)?;
                self.upload_ids(&self.manifest.token, &[pending])?;
                history.push(pending);
                generated += 1;
                let stopped = !emit(pending);
                let at = Instant::now();
                self.mtp_warm(spec, &[pending], cancelled, ExecutionPhase::Decode)?;
                stats.refresh_s += at.elapsed().as_secs_f64();
                if stopped {
                    break;
                }
                continue;
            };
            let at = Instant::now();
            let mut drafts = Vec::with_capacity(plan.tokens - 1);
            let mut proposals = Vec::with_capacity(plan.tokens - 1);
            let mut draft_history = history.clone();
            for i in 0..plan.tokens - 1 {
                if cancelled() {
                    return Err("Request cancelled during MTP draft".into());
                }
                if i != 0 {
                    self.launch_program(&spec.draft_program, ExecutionPhase::Decode)?;
                }
                let token = self.read_control(&spec.token)?;
                if self.read_control(&spec.status)? != 0 || token < 0 || token as usize >= vocab {
                    return Err("Invalid MTP draft token".into());
                }
                let token = if options.is_greedy() {
                    token as u32
                } else {
                    let raw = self
                        .execution
                        .download_bytes(&spec.draft_logits, vocab * 4)?;
                    let distribution = crate::sampling::Distribution::from_logits(
                        &floats(&raw, crate::artifact::Dtype::F32),
                        &draft_history,
                        options,
                    )?;
                    let token = distribution.draw(crate::sampling::counter_uniform(
                        options.seed,
                        crate::mtp::DRAFT_STREAM,
                        (generated + i) as u64,
                    ))?;
                    proposals.push(crate::mtp::Proposal {
                        token,
                        distribution,
                    });
                    self.upload_ids(&spec.token, &[token])?;
                    token
                };
                drafts.push(token);
                draft_history.push(token);
            }
            stats.draft_s += at.elapsed().as_secs_f64();
            stats.proposed_tokens += drafts.len();
            let mut verification_input = vec![pending];
            verification_input.extend_from_slice(&drafts);
            let at = Instant::now();
            self.upload_ids(&self.manifest.input, &verification_input)?;
            self.launch_program(&plan.program, ExecutionPhase::Decode)?;
            self.launch_program(&plan.capture_program, ExecutionPhase::Decode)?;
            let target = self.read_controls(&spec.verification_tokens, plan.tokens)?;
            let status = self.read_controls(&spec.verification_status, plan.tokens)?;
            if target.iter().any(|&x| x as usize >= vocab) || status.iter().any(|&x| x != 0) {
                return Err("Invalid MTP target verification result".into());
            }
            stats.verification_s += at.elapsed().as_secs_f64();
            stats.rounds += 1;
            let at = Instant::now();
            let mut committed = if options.is_greedy() {
                crate::mtp::greedy_commit(&drafts, &target)?
            } else {
                let raw = self
                    .execution
                    .download_bytes(&spec.verification_logits, plan.tokens * vocab * 4)?;
                crate::mtp::sampled_commit(
                    &proposals,
                    &floats(&raw, crate::artifact::Dtype::F32),
                    &history,
                    options,
                    generated,
                    vocab,
                )?
            };
            let accepted_drafts = committed.len() - 1;
            stats.sampling_s += at.elapsed().as_secs_f64();
            let mut emitted = 0;
            let mut stopped = false;
            for &token in &committed {
                if cancelled() {
                    return Err("Request cancelled during MTP commit".into());
                }
                emitted += 1;
                generated += 1;
                if !emit(token) {
                    stopped = true;
                    break;
                }
            }
            committed.truncate(emitted);
            history.extend_from_slice(&committed);
            stats.accepted_draft_tokens += accepted_drafts.min(emitted);
            let at = Instant::now();
            if emitted < plan.tokens {
                self.upload_ids(&spec.accepted_inputs, &[emitted as u32])?;
                self.launch_program(&plan.restore_program, ExecutionPhase::Decode)?;
                self.upload_ids(&self.manifest.position, &[(position + emitted) as u32])?;
                self.upload_ids(&spec.target_length, &[(position + emitted) as u32])?;
            }
            stats.restore_s += at.elapsed().as_secs_f64();
            pending = *committed
                .last()
                .ok_or("MTP round committed no target token")?;
            // Keep the already correct MTP slot at position-1. Replace slots
            // from position onward with true target hidden states paired with
            // accepted tokens and the target correction/bonus. The final row
            // also produces the first draft of the next round.
            let at = Instant::now();
            self.upload_ids(&spec.position, &[position as u32])?;
            self.mtp_warm(spec, &committed, cancelled, ExecutionPhase::Decode)?;
            stats.refresh_s += at.elapsed().as_secs_f64();
            if stopped || generated == limit {
                break;
            }
        }
        if self.read_control(&self.manifest.position)? as usize != input.len() + generated - 1 {
            return Err("MTP committed position mismatch".into());
        }
        self.upload_ids(&self.manifest.token, &[pending])?;
        stats.committed_tokens = generated;
        self.speculation_statistics = Some(stats);
        Ok(generated)
    }
}
