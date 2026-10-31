//! DSpark shares target verification and commits only confirmed state.
use super::*;
mod policy;
pub(super) use policy::Policy;

impl Runtime {
    pub(super) fn advance_dspark(
        &mut self,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        let owners = self.team.model.batch.as_ref().unwrap().owners;
        let baseline = self
            .select_packet(requests, options, None)?
            .ok_or("Missing DSpark decoder packet")?;
        let mixed = baseline.prefill_rows > 0;
        let full = baseline.full;
        let batch = baseline
            .selected
            .iter()
            .filter(|&&(i, _)| !requests[i].prefilling)
            .count()
            .min(owners.min(options.max_batch_tokens));
        let capacity = if options.max_speculative_tokens == 0 {
            self.draft_limit
        } else {
            (options.max_speculative_tokens / batch.max(1))
                .saturating_sub(1)
                .min(self.draft_limit)
        };
        let draft_limit = self.speculation_policy.select(batch, full, mixed, capacity);
        let selected = baseline
            .selected
            .iter()
            .map(|&(i, _)| (i, &requests[i]))
            .filter(|(_, r)| !r.prefilling)
            .take(owners.min(options.max_batch_tokens))
            .map(|(i, r)| {
                (
                    i,
                    if r.history.len() + 4 <= self.team.model.capacity {
                        if r.limit - r.generated > draft_limit {
                            draft_limit.min(self.team.model.capacity - r.history.len())
                        } else {
                            0
                        }
                    } else {
                        0
                    },
                )
            })
            .collect::<Vec<_>>();
        *self
            .statistics
            .mtp_depth_histogram
            .entry(selected.iter().map(|(_, count)| *count).max().unwrap_or(0))
            .or_default() += 1;
        if selected.iter().all(|(_, k)| *k == 0) {
            let limit = self.draft_limit;
            self.draft_limit = 0;
            let result = self.advance_requests_with(requests, options, maintenance);
            self.draft_limit = limit;
            return result;
        }
        let mut segments = vec![];
        let mut inputs = vec![];
        for &(i, count) in &selected {
            if count == 0 {
                continue;
            }
            let request = &requests[i];
            segments.push(BatchSegment {
                slot: request.slot,
                tokens: 5,
            });
            let mut block = vec![128799; 5];
            block[0] = *request.history.last().ok_or("Missing DSpark anchor")?;
            inputs.push(block);
        }
        let mut draft = Layout::verification(segments, owners * 5, owners)?;
        draft.role = super::super::PacketRole::Draft;
        if self.team.needs_preparation(&draft)?
            && !memory::ensure_budget(self, memory::Requirement::workspace())?
        {
            return Err("Insufficient DSpark graph workspace".into());
        }
        let (proposals, draft_s, draft_prepare) = self.team.forward(draft, inputs, false)?;
        let mut inputs = BTreeMap::new();
        let mut draft_tokens = vec![];
        for &(i, count) in &selected {
            let request = &requests[i];
            let tokens = proposals
                .iter()
                .filter(|v| v.owner == request.slot)
                .take(count)
                .map(|v| v.token)
                .collect::<Vec<_>>();
            if tokens.len() != count {
                return Err("Incomplete DSpark proposal pool".into());
            }
            let mut values = vec![*request.history.last().unwrap()];
            values.extend_from_slice(&tokens);
            inputs.insert(i, values);
            draft_tokens.push(tokens);
        }
        let journal = selected
            .iter()
            .filter(|(_, n)| *n > 0)
            .map(|&(i, _)| requests[i].slot)
            .collect::<Vec<_>>();
        let started = Instant::now();
        self.team.exchange(Command::BeginVerification(journal))?;
        let save_s = started.elapsed().as_secs_f64();
        let packet::Selection {
            layout,
            tokens: inputs,
            selected: packet_requests,
            full: packet_full,
            batch: packet_batch,
            prefill_rows,
            prefill_requests,
        } = self
            .select_packet(requests, options, Some(&inputs))?
            .ok_or("Missing DSpark verification packet")?;
        if self.team.needs_preparation(&layout)?
            && !memory::ensure_budget(self, memory::Requirement::workspace())?
        {
            return Err("Insufficient DSpark verification workspace".into());
        }
        let (verified, verify_s, verify_prepare) =
            self.team
                .forward_with(layout, inputs, full || packet_full, maintenance)?;
        let started = Instant::now();
        let mut counts = vec![];
        let mut steps = vec![];
        let mut rollback = vec![];
        for (&(i, count), draft) in selected.iter().zip(draft_tokens) {
            let request = &mut requests[i];
            let target = verified
                .iter()
                .filter(|v| v.owner == request.slot)
                .collect::<Vec<_>>();
            if target.len() != count + 1 {
                return Err("Incomplete DSpark verification rows".into());
            }
            #[cfg(test)]
            if std::env::var_os("ORINFER_A2_DSPARK_TRACE").is_some() && self.speculation.rounds < 8
            {
                eprintln!(
                    "DSPARK_TRACE {}",
                    serde_json::json!({
                        "owner":request.slot,"position":request.history.len()-1,
                        "anchor":request.history.last(),"draft":draft,
                        "target":target.iter().map(|v|v.token).collect::<Vec<_>>(),
                        "confidence":proposals.iter().filter(|v|v.owner==request.slot).map(|v|v.value).collect::<Vec<_>>()
                    })
                );
            }
            let committed = if request.needs_logits() {
                let proposals = draft
                    .iter()
                    .map(|&token| {
                        Ok(crate::mtp::Proposal {
                            token,
                            distribution: crate::sampling::Distribution::point_mass(token, 129280)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let logits = target
                    .iter()
                    .flat_map(|v| v.logits.iter().copied())
                    .collect::<Vec<_>>();
                crate::sampling::verify(
                    &proposals,
                    &logits,
                    &request.history,
                    &request.sampling,
                    &mut request.decoder,
                    request.generated,
                    129280,
                )?
            } else {
                let tokens = crate::mtp::greedy_commit(
                    &draft,
                    &target.iter().map(|v| v.token).collect::<Vec<_>>(),
                )?;
                for &token in &tokens {
                    request.decoder.consume(token)?;
                }
                tokens
            };
            // The last output stays pending. Keep confirmed writes, restore
            // overwritten live slots, and shorten the logical history.
            if committed.len() < count + 1 {
                rollback.push((request.slot, committed.len()));
            }
            counts.push((count, committed.len() - 1));
            *self
                .statistics
                .mtp_request_depth_histogram
                .entry(count)
                .or_default() += 1;
            request.history.extend_from_slice(&committed);
            request.generated += committed.len();
            request.last_output = Instant::now();
            let statistics = &mut request.mtp;
            statistics.rounds += 1;
            statistics.proposed_tokens += count;
            statistics.accepted_draft_tokens += committed.len() - 1;
            statistics.committed_tokens += committed.len();
            self.speculation.proposed_tokens += count;
            self.speculation.accepted_draft_tokens += committed.len() - 1;
            self.speculation.committed_tokens += committed.len();
            self.statistics.decode_tokens += committed.len();
            steps.push(StepOutput {
                request: i,
                tokens: committed,
            });
        }
        let sampling_s = started.elapsed().as_secs_f64();
        self.speculation.sampling_s += sampling_s;
        let started = Instant::now();
        if !rollback.is_empty() {
            self.team
                .exchange(Command::RollbackVerification(rollback))?;
        }
        self.team.exchange(Command::EndVerification)?;
        let restore_s = started.elapsed().as_secs_f64() + save_s;
        let mut confidence = [0.; 5];
        for (step, value) in confidence.iter_mut().enumerate() {
            let scores = proposals
                .iter()
                .skip(step)
                .step_by(5)
                .map(|p| p.value as f64)
                .collect::<Vec<_>>();
            *value = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
        }
        self.speculation_policy.observe(
            selected.len(),
            full,
            mixed,
            [draft_s, verify_s + sampling_s, restore_s],
            &counts,
            confidence,
        );
        self.speculation.rounds += selected.len();
        for &(i, _) in &selected {
            requests[i].mtp.draft_s += draft_s;
            requests[i].mtp.verification_s += verify_s;
            requests[i].mtp.restore_s += restore_s;
        }
        self.speculation.draft_s += draft_s;
        self.speculation.verification_s += verify_s;
        self.speculation.restore_s += restore_s;
        let prefill = packet_requests
            .into_iter()
            .filter(|&(i, _)| requests[i].prefilling)
            .collect();
        steps.extend(self.commit_packet(requests, prefill, &verified)?);
        self.statistics.iterations += 1;
        self.statistics.speculative_iterations += 1;
        self.statistics.compute_s += draft_s + verify_s + restore_s;
        self.statistics.batch_plan_s += draft_prepare + verify_prepare;
        *self
            .statistics
            .batch_histogram
            .entry(packet_batch)
            .or_default() += 1;
        if prefill_rows > 0 {
            self.statistics.mixed_iterations += 1;
            self.statistics.mixed_speculative_iterations += 1;
            self.statistics.bounded_prefill_iterations += 1;
            *self
                .statistics
                .prefill_chunk_histogram
                .entry(prefill_rows)
                .or_default() += 1;
            *self
                .statistics
                .prefill_batch_histogram
                .entry(prefill_requests)
                .or_default() += 1;
            self.statistics.max_bounded_prefill_s = self
                .statistics
                .max_bounded_prefill_s
                .max(draft_s + verify_s + restore_s);
        }
        steps.extend(cached_outputs(requests)?);
        Ok(steps)
    }
}

#[cfg(test)]
mod tests;
