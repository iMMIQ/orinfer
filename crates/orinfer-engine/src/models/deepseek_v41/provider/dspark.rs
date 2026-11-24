//! DSpark shares target verification and commits only confirmed state.
use super::*;
mod policy;
pub(super) use policy::{Budget, Costs, Feedback, Identity, Policy, Work};

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
        let keep = self.final_decoder_window();
        let capacity = self.team.rows;
        let mut decode_rows = vec![0; requests.len()];
        let prefill_s = if mixed {
            self.producer_price(&baseline.selected, requests).seconds
        } else {
            0.
        };
        let prefill_estimator = super::prefill::Estimator {
            model: &self.team.model,
            domains: &self.team.target_domains,
            costs: &self.prefill_costs,
            capacity,
        };
        let ceiling_s = options.prefill_budget_ms / 1000.;
        let mut budgets = BTreeMap::<(usize, u64), usize>::new();
        let mut retained_cache = BTreeMap::<Vec<usize>, (Layout, super::super::batch::Key)>::new();
        let mut verification_budget = |rows: &[usize]| {
            let mut sum = 0usize;
            let mut mask = 0u64;
            for (idx, &n) in rows.iter().enumerate() {
                if n > 0 {
                    sum += n;
                    if idx < 64 {
                        mask |= 1u64 << idx;
                    }
                }
            }
            *budgets.entry((sum, mask)).or_insert_with(|| {
                baseline.verification_budget(requests, rows, keep, &prefill_estimator, ceiling_s)
            })
        };
        let allocation = self.speculation_policy.select_requests_with_forecast(
            (full, mixed),
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
                prefill_s,
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
            |depths, reference| {
                for (&i, &depth) in indices.iter().zip(depths) {
                    decode_rows[i] = depth + 1;
                }
                if let Some(reference) = reference {
                    let mut target = baseline.forecast_layout(
                        requests,
                        &decode_rows,
                        verification_budget(&decode_rows),
                        keep,
                        capacity,
                        owners,
                    )?;
                    let contract = self.team.model.batch.as_ref().unwrap();
                    let (retained, retained_key) =
                        if let Some(cached) = retained_cache.get(reference) {
                            cached
                        } else {
                            for (&i, &depth) in indices.iter().zip(reference) {
                                decode_rows[i] = depth + 1;
                            }
                            let built = baseline.forecast_layout(
                                requests,
                                &decode_rows,
                                verification_budget(&decode_rows),
                                keep,
                                capacity,
                                owners,
                            )?;
                            let key = built.bound_work_key(contract, &self.team.work_launches)?;
                            retained_cache
                                .entry(reference.to_vec())
                                .or_insert((built, key))
                        };
                    if !target.bound_work_cached(
                        retained,
                        Some(retained_key),
                        contract,
                        &self.team.work_launches,
                    )? {
                        return Ok(None);
                    }
                    for (&i, &depth) in indices.iter().zip(depths) {
                        decode_rows[i] = depth + 1;
                    }
                }
                let mut predicted = packet::forecast(
                    requests,
                    &baseline.order,
                    &decode_rows,
                    verification_budget(&decode_rows),
                    keep,
                    capacity,
                );
                if predicted.prefill_rows == 0 && predicted.rows > 0 && predicted.rows <= 127 {
                    predicted.rows = packet::bucket_verify_rows(predicted.rows);
                }
                Ok(Some(predicted))
            },
        )?;
        let depths = allocation.depths;
        for (&i, &depth) in indices.iter().zip(&depths) {
            decode_rows[i] = depth + 1;
        }
        let packet_budget = verification_budget(&decode_rows);
        let retained_work = if let Some(reference) = allocation.reference {
            if std::env::var_os("ORINFER_A2_DSPARK_POLICY_TRACE").is_some() {
                super::super::log_async(format!(
                    "DSPARK_ALLOCATION {}",
                    serde_json::json!({
                        "before":reference,"after":depths,"mixed":mixed,
                    })
                ));
            }
            if let Some((cached, _)) = retained_cache.remove(&reference) {
                Some(cached)
            } else {
                for (&i, &depth) in indices.iter().zip(&reference) {
                    decode_rows[i] = depth + 1;
                }
                Some(baseline.forecast_layout(
                    requests,
                    &decode_rows,
                    verification_budget(&decode_rows),
                    keep,
                    capacity,
                    owners,
                )?)
            }
        } else {
            None
        };
        self.statistics.mtp_budget_calls += 1;
        self.statistics.mtp_budget_s += budget_started.elapsed().as_secs_f64();
        let selected = indices.into_iter().zip(depths).collect::<Vec<_>>();
        *self
            .statistics
            .mtp_depth_histogram
            .entry(selected.iter().map(|(_, count)| *count).max().unwrap_or(0))
            .or_default() += 1;
        if selected.iter().all(|(_, k)| *k == 0) {
            // Execute the packet whose ordinary price was requested. A second
            // selection can change its admissions and invalidate calibration.
            let speculative_baseline =
                maxima.iter().any(|&maximum| maximum > 0) && row_budget > selected.len();
            return self.advance_selection(
                baseline,
                requests,
                options,
                speculative_baseline,
                maintenance,
            );
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
        let depths = selected.iter().copied().collect::<BTreeMap<_, _>>();
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
        let (proposals, draft_timing) = if self.team.is_resident(&draft) {
            self.team
                .fast_forward(draft, inputs, readout, maintenance)?
        } else {
            let draft_prepared =
                self.begin_forward_with_workspace(&draft, "Insufficient DSpark graph workspace")?;
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
            let capture =
                self.speculation_policy
                    .draft_budget(block, draft_owners.len(), uses, slack);
            self.team.finish_forward(
                draft_prepared,
                draft,
                inputs,
                readout,
                Some(capture),
                maintenance,
            )?
        };
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
        let mut verification = self
            .packet_for_budget(
                requests,
                Some(&inputs),
                packet_budget,
                mixed,
                &baseline.order,
            )?
            .ok_or("Missing DSpark verification packet")?;
        if let Some(reference) = retained_work
            && !verification.layout.bound_work(
                &reference,
                self.team.model.batch.as_ref().unwrap(),
                &self.team.work_launches,
            )?
        {
            return Err("Verification packet changed its retained launch envelope".into());
        }
        self.bucket_selection(&mut verification, requests)?;
        let packet::Selection {
            layout,
            tokens: inputs,
            selected: packet_requests,
            full: packet_full,
            batch: packet_batch,
            prefill_rows,
            prefill_requests,
            ..
        } = verification;
        // Dispatch GrowCachedAndBeginVerification to all 8 ranks first so CPU
        // readout/rejection/Engram-forecast preparation overlaps the 8-rank IPC.
        let started = self
            .team
            .send_verification_forward(&layout, journal.clone())?;
        let mut verified_work = packet::work(&packet_requests, requests, prefill_rows);
        if prefill_rows == 0 && layout.rows.total > verified_work.rows {
            verified_work.rows = layout.rows.total;
        }
        let prefill_work = self.prefill_work(
            &packet_requests,
            requests,
            full || packet_full,
            layout.decode,
        );
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
        let readout_result = packet::target_readout(
            requests,
            full || packet_full,
            &proposal_map,
            self.vocab(),
            &packet_requests,
        );
        let readout_result = readout_result.map(|mut readout| {
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
            self.prepare_host_inputs(&layout, &packet_requests, requests, &mut readout);
            readout
        });
        let (first_prepared, first_journal_ready) =
            self.team.collect_verification_forward(&layout, started)?;
        let readout = readout_result?;
        let need_workspace = !first_prepared.all_graphs() && !first_prepared.workspace_fits;
        let verify_prepared = if !need_workspace && first_journal_ready {
            first_prepared
        } else {
            let Some(retried) = self.begin_verification_forward_with_workspace(
                &layout,
                journal,
                "Insufficient DSpark verification workspace",
            )?
            else {
                // Journal memory could not be reserved even after prefix/program
                // reclamation; fall back cleanly to ordinary target decode.
                return self.advance_selection(baseline, requests, options, false, || {});
            };
            retried
        };
        let save_s = started.elapsed().as_secs_f64();
        let (verified, verify_timing) = self.team.finish_forward(
            verify_prepared,
            layout,
            inputs,
            readout,
            Some(capture),
            || {},
        )?;
        verify_timing.record(&mut self.statistics.batch_execution);
        self.observe_prefill(prefill_work, &verify_timing);
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
                let core = &mut request.core;
                match crate::sampling::verify_validated(
                    &proposals,
                    &logits,
                    &core.history,
                    &core.sampling,
                    &mut core.decoder,
                    core.generated,
                    129280,
                ) {
                    Ok(tokens) => tokens,
                    Err(error)
                        if core.decoder.failure().is_some()
                            || error == "Rejected proposal has no residual mass" =>
                    {
                        if core.decoder.failure().is_none() {
                            core.decoder.fail(error);
                        }
                        self.health.sampling_request_failures += 1;
                        rollback.push((request.slot, 0));
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            } else if request.processed_sampling && request.device_sample() {
                match request.verify_device(&draft, &target) {
                    Ok(tokens) => tokens,
                    Err(error) if request.decoder.failure().is_some() => {
                        let _ = error;
                        self.health.sampling_request_failures += 1;
                        rollback.push((request.slot, 0));
                        continue;
                    }
                    Err(error) => return Err(error),
                }
            } else {
                // Target draws use only their absolute position's randomness.
                // A deterministic guess or a draft sharing that randomness can
                // therefore reuse the matching prefix. Earlier decisions never
                // inspect future draws; the first mismatch is the target draw,
                // and errors on unaccepted speculative rows are ignored.
                let mut tokens = Vec::with_capacity(target.len());
                let mut row_error = None;
                for (position, output) in target.iter().enumerate() {
                    if let Some(error) = &output.error {
                        row_error = Some(error.clone());
                        break;
                    }
                    tokens.push(output.token);
                    if draft.get(position) != Some(&output.token) {
                        break;
                    }
                }
                if let Some(error) = row_error {
                    request.decoder.fail(error);
                    self.health.sampling_request_failures += 1;
                    rollback.push((request.slot, 0));
                    continue;
                }
                let count = match request.decoder.consume_unconstrained_prefix(&tokens) {
                    Ok(count) => count,
                    Err(_) if request.decoder.failure().is_some() => {
                        self.health.sampling_request_failures += 1;
                        rollback.push((request.slot, 0));
                        continue;
                    }
                    Err(error) => return Err(error),
                };
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
                self.speculation_policy.observe_feedback(
                    request.slot,
                    &mut request.draft_feedback,
                    &confidence,
                    (count, committed.len() - 1),
                    block,
                );
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
        self.team.exchange(Command::FinishVerification(rollback))?;
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

#[cfg(test)]
mod varlen_test;

#[cfg(test)]
mod selection_test;
