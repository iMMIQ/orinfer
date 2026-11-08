//! DSpark shares target verification and commits only confirmed state.
use super::*;
mod policy;
pub(super) use policy::{Budget, Costs, Feedback, Identity, Policy, PrefillBudget, Work};

impl Runtime {
    pub(super) fn advance_dspark(
        &mut self,
        requests: &mut [&mut Request],
        options: &scheduler::Options,
        maintenance: impl FnOnce(),
    ) -> Result<Vec<StepOutput>> {
        let round_started = Instant::now();
        // Bound decoder admission by the package's verification-row budget;
        // mixed prefill still uses the shared packet capacity and scheduler.
        let bounded_options = scheduler::Options {
            max_batch_tokens: options
                .max_batch_tokens
                .min(self.team.verification_rows.max(1)),
            ..*options
        };
        let options = &bounded_options;
        let owners = self.team.model.batch.as_ref().unwrap().owners;
        let baseline = self
            .select_packet(requests, options, None)?
            .ok_or("Missing DSpark decoder packet")?;
        let shared_work = packet::mark_shared(&baseline.selected, requests);
        let mixed = baseline.prefill_rows > 0;
        let full = baseline.full;
        let indices = baseline
            .selected
            .iter()
            .filter(|&&(i, _)| !requests[i].prefilling)
            .take(owners.min(options.max_batch_tokens))
            .map(|&(i, _)| i)
            .collect::<Vec<_>>();
        // The initial candidate keeps grammar/report and other sampling modes
        // on their established path. Fresh prefill rows may share this packet.
        let rejection = self.rejection_sampling
            && indices.iter().all(|&i| {
                requests[i].sampling.device_categorical() && !requests[i].decoder.constrained()
            });
        let rejection_steps = if rejection {
            indices
                .iter()
                .map(|&i| {
                    let r = &requests[i];
                    Ok((r.slot, crate::mtp::round_step(r.mtp.rounds, r.generated)?))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        } else {
            BTreeMap::new()
        };
        let uniform = self.draft_limit <= self.team.draft_blocks[0];
        for &i in &indices {
            self.speculation_policy.prepare_feedback(
                indices.len(),
                full,
                mixed,
                uniform,
                &mut requests[i].draft_feedback,
            );
        }
        let maxima = indices
            .iter()
            .map(|&i| {
                let r = &requests[i];
                if uniform && r.limit - r.generated <= self.draft_limit {
                    return 0;
                }
                let remaining = self.team.model.capacity.saturating_sub(r.history.len());
                let physical = self
                    .team
                    .draft_blocks
                    .iter()
                    .copied()
                    .filter(|&block| block - 1 <= remaining)
                    .max()
                    .unwrap_or(0);
                self.draft_limit
                    .min(r.limit.saturating_sub(r.generated + 1))
                    .min(remaining)
                    .min(physical)
            })
            .collect::<Vec<_>>();
        // Price selection within the same bounded speculative output window
        // used by capture admission, rather than the one-token ordinary window.
        let draft_window = indices
            .iter()
            .copied()
            .zip(maxima.iter().copied())
            .collect::<BTreeMap<_, _>>();
        let window_block = self
            .team
            .draft_blocks
            .iter()
            .copied()
            .find(|&b| b >= maxima.iter().copied().max().unwrap_or(0))
            .unwrap_or(self.team.draft_blocks[0]);
        let feedback = indices
            .iter()
            .map(|&i| &requests[i].draft_feedback)
            .collect::<Vec<_>>();
        // Use the package's validated verification capacity. Legacy packages
        // retain their conservative boundary below the first row exchange.
        let row_budget = if options.max_speculative_tokens == 0 {
            self.team.verification_rows
        } else {
            // The CLI remains an upper budget within the published capability;
            // the existing row dispatcher selects the measured compute plan.
            options
                .max_speculative_tokens
                .min(self.team.verification_rows)
        };
        let budget_started = Instant::now();
        let depths = self.speculation_policy.select_requests(
            full,
            mixed,
            Budget {
                rows: row_budget,
                latency: options.policy == scheduler::Policy::Latency,
                slack_s: if options.policy == scheduler::Policy::Latency {
                    self.capture_context(
                        &baseline.selected,
                        requests,
                        Some(&draft_window),
                        window_block,
                        options,
                    )
                    .1
                } else {
                    f64::INFINITY
                },
                work: packet::work(&baseline.selected, requests, baseline.prefill_rows),
                prefill_tokens_per_s: self.prefill_tokens_per_s,
                uniform,
                remaining_by_owner: indices
                    .iter()
                    .map(|&i| requests[i].limit - requests[i].generated)
                    .collect(),
                remaining: indices
                    .iter()
                    .map(|&i| requests[i].limit - requests[i].generated)
                    .min()
                    .unwrap_or(0),
            },
            &maxima,
            &feedback,
            &self.team.draft_blocks,
        )?;
        self.statistics.mtp_budget_calls += 1;
        self.statistics.mtp_budget_s += budget_started.elapsed().as_secs_f64();
        let selected = indices.into_iter().zip(depths).collect::<Vec<_>>();
        *self
            .statistics
            .mtp_depth_histogram
            .entry(selected.iter().map(|(_, count)| *count).max().unwrap_or(0))
            .or_default() += 1;
        if selected.iter().all(|(_, k)| *k == 0) {
            // Calibrate the packet whose price was requested. Re-entering
            // selection with drafts disabled can choose another prefill size,
            // leaving the planned ordinary baseline permanently unmeasured.
            return self.advance_selection(baseline, requests, options, maintenance);
        }
        let maximum = selected.iter().map(|(_, count)| *count).max().unwrap();
        self.speculation_policy.begin_round(
            selected.iter().map(|&(i, _)| requests[i].slot).collect(),
            full,
            packet::work(&baseline.selected, requests, baseline.prefill_rows),
            shared_work,
            maximum + 1,
        );
        let block = self
            .team
            .draft_blocks
            .iter()
            .copied()
            .find(|&block| block >= maximum)
            .ok_or("Missing DSpark block capacity")?;
        let mut segments = vec![];
        let mut inputs = vec![];
        for &(i, count) in &selected {
            if count == 0 {
                continue;
            }
            let request = &requests[i];
            segments.push(BatchSegment {
                slot: request.slot,
                tokens: block,
            });
            let mut values = vec![128799; block];
            values[0] = *request.history.last().ok_or("Missing DSpark anchor")?;
            inputs.push(values);
        }
        let mut draft = if self.team.strided_draft {
            let outputs = (0..segments.len()).collect();
            Layout::new(segments, outputs, false, owners * block, owners)?
        } else {
            Layout::verification(segments, owners * block, owners)?
        };
        draft.role = super::super::PacketRole::Draft;
        if self.team.needs_preparation(&draft)?
            && !memory::ensure_budget(self, memory::Requirement::workspace())?
        {
            return Err("Insufficient DSpark graph workspace".into());
        }
        let depths = selected.iter().copied().collect::<BTreeMap<_, _>>();
        // Only these owners execute the draft graph. A final prefill window
        // or a non-speculating tail cannot shorten its reuse forecast.
        let draft_owners = selected
            .iter()
            .copied()
            .filter(|&(_, depth)| depth > 0)
            .collect::<Vec<_>>();
        let (uses, _) =
            self.capture_context(&draft_owners, requests, Some(&depths), block, options);
        // Draft preparation still delays every decoder in the packet.
        let (_, slack) =
            self.capture_context(&baseline.selected, requests, Some(&depths), block, options);
        let capture = self
            .speculation_policy
            .draft_budget(block, draft_owners.len(), uses, slack);
        let mut readout = if self
            .team
            .model
            .sampling
            .as_ref()
            .is_some_and(|s| s.coupled_draft)
        {
            packet::draft_readout(requests)
        } else {
            false.into()
        };
        if rejection {
            readout.store_draft_distributions = true;
            readout.sampling_steps(&rejection_steps);
            for draw in readout.draws.values_mut() {
                draw.seed ^= crate::mtp::DRAFT_STREAM;
            }
        }
        let (proposals, draft_timing) =
            self.team
                .forward_with(draft, inputs, readout, Some(capture), maintenance)?;
        draft_timing.record(&mut self.statistics.batch_execution);
        let draft_s = draft_timing.seconds;
        let draft_prepare = draft_timing.preparation;
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
            .map(|&(i, count)| (requests[i].slot, count + 1))
            .collect::<Vec<_>>();
        let packet::Selection {
            layout,
            tokens: inputs,
            selected: packet_requests,
            full: packet_full,
            batch: packet_batch,
            prefill_rows,
            prefill_requests,
            ..
        } = self
            .packet_for_budget(requests, Some(&inputs), baseline.budget, mixed)?
            .ok_or("Missing DSpark verification packet")?;
        if self.team.needs_preparation(&layout)?
            && !memory::ensure_budget(self, memory::Requirement::workspace())?
        {
            return Err("Insufficient DSpark verification workspace".into());
        }
        let verified_work = packet::work(&packet_requests, requests, prefill_rows);
        let started = Instant::now();
        self.team.exchange(Command::BeginVerification(journal))?;
        let save_s = started.elapsed().as_secs_f64();
        let (uses, slack) =
            self.capture_context(&packet_requests, requests, Some(&depths), block, options);
        let capture = self.speculation_policy.target_budget(
            full || packet_full,
            layout.decode,
            verified_work,
            uses,
            slack,
        );
        let proposal_map = selected
            .iter()
            .zip(&draft_tokens)
            .map(|(&(i, _), tokens)| (i, tokens.clone()))
            .collect();
        let mut readout = packet::target_readout(
            requests,
            full || packet_full,
            &proposal_map,
            self.vocab(),
            &packet_requests,
        )?;
        if rejection {
            readout.sampling_steps(&rejection_steps);
            let stride = self
                .team
                .model
                .sampling
                .as_ref()
                .unwrap()
                .rejection
                .as_ref()
                .unwrap()
                .stride;
            for (&(i, count), tokens) in selected.iter().zip(&draft_tokens) {
                if count == 0 {
                    continue;
                }
                let request = &requests[i];
                readout.rejection.insert(
                    request.slot,
                    tokens
                        .iter()
                        .enumerate()
                        .map(|(position, &token)| {
                            crate::backend::ascend::sampling::RejectionProposal {
                                row: request.slot * stride + position,
                                token,
                                log_uniform: crate::sampling::counter_uniform(
                                    request.sampling.seed,
                                    crate::mtp::ACCEPT_STREAM,
                                    (rejection_steps[&request.slot] + position) as u64,
                                )
                                .ln() as f32,
                            }
                        })
                        .collect(),
                );
            }
        }
        self.speculation_policy
            .prepare_prefill(baseline.prefill_budget);
        let (verified, verify_timing) =
            self.team
                .forward_with(layout, inputs, readout, Some(capture), || {})?;
        verify_timing.record(&mut self.statistics.batch_execution);
        let verify_s = verify_timing.seconds;
        self.statistics.sampling_mask_rows += verify_timing.mask_rows;
        self.statistics.sampling_report_rows += verify_timing.report_rows;
        self.statistics.sampling_prepare_s += verify_timing.mask_prepare_s;
        let verify_prepare = verify_timing.preparation;
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
            let committed = if request.host_sampling()
                && !(request.device_sample() && target.iter().all(|o| o.logits.is_empty()))
            {
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
            } else if request.processed_sampling && request.device_sample() {
                request.verify_device(&draft, &target)?
            } else {
                // Target draws use only their absolute position's randomness.
                // A deterministic guess or a draft sharing that randomness can
                // therefore reuse the matching prefix. Earlier decisions never
                // inspect future draws; the first mismatch is the target draw.
                let mut tokens = crate::mtp::greedy_commit(
                    &draft,
                    &target.iter().map(|v| v.token).collect::<Vec<_>>(),
                )?;
                let count = request.decoder.consume_unconstrained_prefix(&tokens)?;
                if count == 0 {
                    return Err("Empty unconstrained verification prefix".into());
                }
                tokens.truncate(count);
                tokens
            };
            // The last output stays pending. Keep confirmed writes, restore
            // overwritten live slots, and shorten the logical history.
            if committed.len() < count + 1 {
                rollback.push((request.slot, committed.len()));
            }
            if count > 0 {
                let confidence = proposals
                    .iter()
                    .filter(|p| p.owner == request.slot)
                    .map(|p| p.value as f64)
                    .collect::<Vec<_>>();
                request
                    .draft_feedback
                    .observe(&confidence, count, committed.len() - 1, block);
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
        self.speculation_policy.observe(
            selected.len(),
            (full, mixed, uniform),
            Costs {
                draft_replay: draft_timing.replay,
                verify_replay: verify_timing.replay,
                shared: [draft_timing.shared, verify_timing.shared],
                shared_setup: [draft_timing.shared_setup, verify_timing.shared_setup],
                work: verified_work,
                setup: [draft_timing.setup, verify_timing.setup],
                wall_s: Some(round_started.elapsed().as_secs_f64()),
                stages: [
                    draft_s + draft_prepare - draft_timing.setup,
                    verify_s + sampling_s + verify_prepare - verify_timing.setup,
                    restore_s,
                    0.,
                ],
            },
            &counts,
            block,
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

#[cfg(test)]
mod rejection_test;
