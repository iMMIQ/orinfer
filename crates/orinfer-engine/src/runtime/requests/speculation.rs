//! Speculation advances one bounded verification round. Catch-up is lazy while
//! batched target decode is profitable; stale ring entries are never consumed.
use super::*;

impl ModelRuntime {
    pub(super) fn refresh_request_mtp(&self, req: &mut RequestState, head: bool) -> Result<bool> {
        let Some(spec) = &self.manifest.mtp else {
            return Ok(true);
        };
        let position = self.read_control(&self.manifest.position)? as usize;
        let draft_position = self.read_control(&spec.position)? as usize;
        if req.warm.tokens > position || draft_position != req.warm.tokens {
            return Err(format!(
                "Private MTP cursor mismatch: slot {}, target {position}, draft {draft_position}, warm {}",
                req.slot, req.warm.tokens
            ));
        }
        let lag = position - req.warm.tokens;
        let ring = spec
            .hidden_ring
            .as_ref()
            .and_then(|n| self.manifest.buffers.iter().find(|b| b.name == *n))
            .map(|b| b.shape[0])
            .unwrap_or(1);
        if req.mtp_blocked || lag > ring {
            req.mtp_blocked = true;
            return Ok(false);
        }
        if lag != 0 {
            let at = Instant::now();
            self.mtp_warm_state(
                spec,
                &req.history[req.warm.tokens + 1..position + 1],
                &|| false,
                ExecutionPhase::Decode,
                head,
            )?;
            req.warm.tokens = position;
            req.mtp.refresh_s += at.elapsed().as_secs_f64();
        }
        Ok(true)
    }
    pub(super) fn speculative_request_step(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
    ) -> Result<Vec<u32>> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let valid = self.refresh_request_mtp(req, true)?;
        let position = self.read_control(&self.manifest.position)? as usize;
        let remaining = req.limit - req.generated;
        let plan = valid
            .then(|| {
                spec.verification_plans
                    .iter()
                    .filter(|p| {
                        p.tokens <= remaining
                            && p.tokens <= max_rows
                            && p.tokens <= spec.default_verification_tokens
                            && p.tokens <= self.manifest.max_context - position
                    })
                    .max_by_key(|p| p.tokens)
            })
            .flatten();
        let pending = *req.history.last().ok_or("Missing pending token")?;
        let Some(plan) = plan else {
            self.upload_ids(&self.manifest.token, &[pending])?;
            self.prepare_inputs(&[pending], &req.history[..req.history.len() - 1])?;
            self.upload_segment_controls(1)?;
            self.launch_program("decode", ExecutionPhase::Decode)?;
            self.mtp_capture(&spec, 1, ExecutionPhase::Decode)?;
            let out = self.commit_ordinary_token(req)?;
            if valid {
                self.refresh_request_mtp(req, true)?;
            }
            return Ok(out);
        };
        let vocab = self.manifest.vocab;
        if let Some(save) = &spec.draft_snapshot_program {
            self.launch_program(save, ExecutionPhase::Decode)?;
        }
        let at = Instant::now();
        let mut drafts = Vec::with_capacity(plan.tokens - 1);
        let mut proposals = Vec::with_capacity(plan.tokens - 1);
        let mut draft_history = req.history.clone();
        let host_sampling = !req.sampling.is_greedy()
            || req.decoder.constrained()
            || req.sampling.top_logprobs.is_some();
        let mut draft_decoder = req.decoder.fork();
        let draft_options = crate::sampling::Options {
            top_logprobs: None,
            ..req.sampling.clone()
        };
        for i in 0..plan.tokens - 1 {
            if i != 0 {
                self.prepare_program_inputs(
                    &spec.draft_program,
                    &[*drafts.last().ok_or("Missing draft input")?],
                    &[],
                )?;
                self.launch_program(&spec.draft_program, ExecutionPhase::Decode)?;
            }
            let selected = self.read_control(&spec.token)?;
            if self.read_control(&spec.status)? != 0 || selected < 0 || selected as usize >= vocab {
                return Err("Invalid request draft token".into());
            }
            let token = if !host_sampling {
                selected as u32
            } else {
                let raw = self
                    .execution
                    .download_bytes(&spec.draft_logits, vocab * 4)?;
                let law = draft_decoder
                    .law(
                        &floats(&raw, crate::artifact::Dtype::F32),
                        &draft_history,
                        &draft_options,
                    )
                    .inspect_err(|error| {
                        if draft_decoder.failure().is_some() {
                            req.decoder.fail(error.clone());
                        }
                    })?;
                let distribution = law.distribution;
                let token = distribution.draw(crate::sampling::counter_uniform(
                    req.sampling.seed,
                    crate::mtp::DRAFT_STREAM,
                    (req.generated + i) as u64,
                ))?;
                proposals.push(crate::mtp::Proposal {
                    token,
                    distribution,
                });
                self.upload_ids(&spec.token, &[token])?;
                draft_decoder.consume(token).inspect_err(|error| {
                    req.decoder.fail(error.clone());
                })?;
                token
            };
            drafts.push(token);
            draft_history.push(token);
            if draft_decoder.finished() {
                break;
            }
        }
        // Fixed verification profiles may contain unused rows after EOS. They
        // have no authority over the parser or committed probability records.
        req.mtp.proposed_tokens += drafts.len();
        drafts.resize(
            plan.tokens - 1,
            *drafts.last().ok_or("Empty draft profile")?,
        );
        req.mtp.draft_s += at.elapsed().as_secs_f64();
        let mut verification = vec![pending];
        verification.extend_from_slice(&drafts);
        let at = Instant::now();
        self.upload_ids(&self.manifest.input, &verification)?;
        self.prepare_inputs(&verification, &req.history[..req.history.len() - 1])?;
        self.upload_segment_controls(plan.tokens)?;
        self.launch_program(&plan.program, ExecutionPhase::Decode)?;
        self.launch_program(&plan.capture_program, ExecutionPhase::Decode)?;
        let target = self.read_controls(&spec.verification_tokens, plan.tokens)?;
        let status = self.read_controls(&spec.verification_status, plan.tokens)?;
        if target.iter().any(|&id| id as usize >= vocab) || status.iter().any(|&s| s != 0) {
            return Err("Invalid request verification output".into());
        }
        req.mtp.verification_s += at.elapsed().as_secs_f64();
        req.mtp.rounds += 1;
        let at = Instant::now();
        let committed = if !host_sampling {
            crate::mtp::greedy_commit(&drafts, &target)?
        } else {
            let raw = self
                .execution
                .download_bytes(&spec.verification_logits, plan.tokens * vocab * 4)?;
            crate::sampling::verify(
                &proposals,
                &floats(&raw, crate::artifact::Dtype::F32),
                &req.history,
                &req.sampling,
                &mut req.decoder,
                req.generated,
                vocab,
            )?
        };
        if committed.is_empty() || committed.len() > remaining {
            return Err("Invalid speculative commit length".into());
        }
        req.mtp.sampling_s += at.elapsed().as_secs_f64();
        req.mtp.accepted_draft_tokens += if host_sampling {
            committed
                .iter()
                .zip(&proposals)
                .take_while(|(token, proposal)| **token == proposal.token)
                .count()
        } else {
            committed.len() - 1
        };
        let at = Instant::now();
        if spec.commit_always || committed.len() < plan.tokens {
            self.upload_ids(&spec.accepted_inputs, &[committed.len() as u32])?;
            self.launch_program(&plan.restore_program, ExecutionPhase::Decode)?;
            self.upload_ids(
                &self.manifest.position,
                &[(position + committed.len()) as u32],
            )?;
            self.upload_ids(&spec.target_length, &[(position + committed.len()) as u32])?;
        }
        req.mtp.restore_s += at.elapsed().as_secs_f64();
        req.history.extend_from_slice(&committed);
        req.generated += committed.len();
        let at = Instant::now();
        if let Some(restore) = &spec.draft_restore_program {
            self.launch_program(restore, ExecutionPhase::Decode)?;
        }
        self.upload_ids(&spec.position, &[position as u32])?;
        self.mtp_warm(&spec, &committed, &|| false, ExecutionPhase::Decode)?;
        req.warm.tokens = position + committed.len();
        req.mtp.refresh_s += at.elapsed().as_secs_f64();
        self.upload_ids(
            &self.manifest.token,
            &[*committed.last().expect("nonempty commit")],
        )?;
        if self.read_control(&self.manifest.position)? as usize
            != req.input.len() + req.generated - 1
        {
            return Err("Private speculative position mismatch".into());
        }
        req.mtp.committed_tokens = req.generated;
        Ok(committed)
    }
}
