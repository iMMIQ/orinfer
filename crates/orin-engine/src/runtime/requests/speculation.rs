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
            self.launch_program("decode", ExecutionPhase::Decode)?;
            self.mtp_capture(&spec, 1, ExecutionPhase::Decode)?;
            let out = self.commit_ordinary_token(req)?;
            if valid {
                self.refresh_request_mtp(req, true)?;
            }
            return Ok(out);
        };
        let vocab = self.manifest.vocab;
        let at = Instant::now();
        let mut drafts = Vec::with_capacity(plan.tokens - 1);
        let mut proposals = Vec::with_capacity(plan.tokens - 1);
        let mut draft_history = req.history.clone();
        for i in 0..plan.tokens - 1 {
            if i != 0 {
                self.launch_program(&spec.draft_program, ExecutionPhase::Decode)?;
            }
            let selected = self.read_control(&spec.token)?;
            if self.read_control(&spec.status)? != 0 || selected < 0 || selected as usize >= vocab {
                return Err("Invalid request draft token".into());
            }
            let token = if req.sampling.is_greedy() {
                selected as u32
            } else {
                let raw = self
                    .execution
                    .download_bytes(&spec.draft_logits, vocab * 4)?;
                let distribution = crate::sampling::Distribution::from_logits(
                    &floats(&raw, crate::artifact::Dtype::F32),
                    &draft_history,
                    &req.sampling,
                )?;
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
                token
            };
            drafts.push(token);
            draft_history.push(token);
        }
        req.mtp.draft_s += at.elapsed().as_secs_f64();
        req.mtp.proposed_tokens += drafts.len();
        let mut verification = vec![pending];
        verification.extend_from_slice(&drafts);
        let at = Instant::now();
        self.upload_ids(&self.manifest.input, &verification)?;
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
        let committed = if req.sampling.is_greedy() {
            crate::mtp::greedy_commit(&drafts, &target)?
        } else {
            let raw = self
                .execution
                .download_bytes(&spec.verification_logits, plan.tokens * vocab * 4)?;
            crate::mtp::sampled_commit(
                &proposals,
                &floats(&raw, crate::artifact::Dtype::F32),
                &req.history,
                &req.sampling,
                req.generated,
                vocab,
            )?
        };
        if committed.is_empty() || committed.len() > remaining {
            return Err("Invalid speculative commit length".into());
        }
        req.mtp.sampling_s += at.elapsed().as_secs_f64();
        req.mtp.accepted_draft_tokens += committed.len() - 1;
        let at = Instant::now();
        if committed.len() < plan.tokens {
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
