//! DSpark shares target verification and commits only confirmed state.
use super::*;
mod policy;
pub(super) use policy::{Budget, Costs, Feedback, Identity, Policy, Work};

struct DraftRequest<'a, 'r> {
    requests: &'a [&'r mut Request],
    baseline_selected: &'a [(usize, usize)],
    selected: &'a [(usize, usize)],
    block: usize,
    owners: usize,
    rejection_steps: Option<&'a BTreeMap<usize, usize>>,
    options: &'a scheduler::Options,
}

struct VerifyReadoutRequest<'a, 'r> {
    requests: &'a [&'r mut Request],
    full: bool,
    selected: &'a [(usize, usize)],
    draft_tokens: &'a [Vec<u32>],
    packet_requests: &'a [(usize, usize)],
    layout: &'a Layout,
    inputs: &'a [Vec<u32>],
    rejection_steps: Option<&'a BTreeMap<usize, usize>>,
}

struct DraftOutcome {
    proposals: Vec<super::super::team::Output>,
    timing: super::super::team::Timing,
    inputs: BTreeMap<usize, Vec<u32>>,
    draft_tokens: Vec<Vec<u32>>,
}

struct RoundMetrics<'a> {
    selected: &'a [(usize, usize)],
    full: bool,
    mixed: bool,
    uniform: bool,
    block: usize,
    round_started: Instant,
    draft_timing: &'a super::super::team::Timing,
    verify_timing: &'a super::super::team::Timing,
    sampling_s: f64,
    restore_s: f64,
    verified_work: Work,
    counts: &'a [(usize, usize)],
    packet_batch: usize,
    prefill_rows: usize,
    prefill_requests: usize,
}

impl Runtime {
    /// Fill unused verification bucket slack (pad_slot rows) with extra verified
    /// drafts from the already-computed parallel draft block at zero extra NPU cost.
    fn expand_verify_bucket_slack(
        &self,
        depths: &mut [usize],
        maxima: &[usize],
        feedback: &[&Feedback],
        capacity: usize,
    ) -> bool {
        let Some(&max_depth) = depths.iter().max() else {
            return false;
        };
        if max_depth == 0 {
            return false;
        }
        let Some(block) = self
            .team
            .draft_blocks
            .iter()
            .copied()
            .find(|&b| b >= max_depth)
        else {
            return false;
        };
        let mut total: usize = depths.iter().map(|&d| d + 1).sum();
        let output_cap = self
            .team
            .model
            .batch
            .as_ref()
            .map_or(capacity, |c| c.output_capacity());
        let target = packet::bucket_verify_rows(total)
            .min(capacity)
            .min(output_cap);
        if total > 127 || target <= total {
            return false;
        }
        let owner_rates: Vec<Vec<f64>> = feedback
            .iter()
            .map(|f| f.rates(block, &[], block))
            .collect();
        let mut expanded = false;
        while total < target {
            let best = (0..depths.len())
                .filter(|&p| depths[p] > 0 && depths[p] < maxima[p].min(block))
                .max_by(|&a, &b| {
                    let ra = owner_rates[a].get(depths[a]).copied().unwrap_or(0.0);
                    let rb = owner_rates[b].get(depths[b]).copied().unwrap_or(0.0);
                    ra.total_cmp(&rb)
                        .then_with(|| depths[b].cmp(&depths[a]))
                        .then_with(|| b.cmp(&a))
                });
            let Some(pos) = best else {
                break;
            };
            depths[pos] += 1;
            total += 1;
            expanded = true;
        }
        expanded
    }

    fn run_dspark_draft(
        &mut self,
        req: DraftRequest<'_, '_>,
        maintenance: impl FnOnce(),
    ) -> Result<DraftOutcome> {
        let DraftRequest {
            requests,
            baseline_selected,
            selected,
            block,
            owners,
            rejection_steps,
            options,
        } = req;
        let mut segments = vec![];
        let mut inputs = vec![];
        for &(i, count) in selected {
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
        if let Some(rejection_steps) = rejection_steps {
            readout.store_draft_distributions = true;
            readout.sampling_steps(rejection_steps);
            for draw in readout.draws.values_mut() {
                draw.seed ^= crate::mtp::DRAFT_STREAM;
            }
        }
        let (proposals, timing) = if self.team.is_resident(&draft) {
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
                self.capture_context(baseline_selected, requests, Some(&depths), block, options);
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
        let mut target_inputs = BTreeMap::new();
        let mut draft_tokens = vec![];
        for &(i, count) in selected {
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
            let mut values = vec![*request.history.last().ok_or("Missing DSpark anchor")?];
            values.extend_from_slice(&tokens);
            target_inputs.insert(i, values);
            draft_tokens.push(tokens);
        }
        Ok(DraftOutcome {
            proposals,
            timing,
            inputs: target_inputs,
            draft_tokens,
        })
    }

    fn build_verification_readout(
        &mut self,
        req: VerifyReadoutRequest<'_, '_>,
    ) -> Result<super::super::team::Readout> {
        let VerifyReadoutRequest {
            requests,
            full,
            selected,
            draft_tokens,
            packet_requests,
            layout,
            inputs,
            rejection_steps,
        } = req;
        let proposal_map = selected
            .iter()
            .zip(draft_tokens)
            .map(|(&(i, _), tokens)| (i, tokens.clone()))
            .collect();
        let mut readout =
            packet::target_readout(requests, full, &proposal_map, self.vocab(), packet_requests)?;
        if let Some(rejection_steps) = rejection_steps {
            readout.sampling_steps(rejection_steps);
            let stride = self
                .team
                .model
                .sampling
                .as_ref()
                .and_then(|s| s.rejection.as_ref())
                .ok_or("Missing DSpark rejection contract")?
                .stride;
            for (&(i, count), tokens) in selected.iter().zip(draft_tokens) {
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
        self.prepare_host_inputs(layout, packet_requests, requests, &mut readout);
        // Overlap Layer 1 Engram random host-RAM gather on verification batches
        // (both pure decode and mixed) with verification IPC, journal copy, and layer 0.
        if let Some(first) = readout.first_input.as_deref() {
            let mut pairs: Vec<(&[u32], &[u32])> = Vec::with_capacity(inputs.len());
            for (lane, tok) in inputs.iter().enumerate() {
                if let Some(&(req_idx, _)) = packet_requests.get(lane) {
                    let req = &requests[req_idx];
                    let worker_hist = if req.prefilling {
                        &req.input[..req.offset.min(req.input.len())]
                    } else {
                        // Worker owner.history precedes the anchor token at tok[0].
                        &req.history[..req.history.len().saturating_sub(1)]
                    };
                    pairs.push((worker_hist, tok.as_slice()));
                } else {
                    pairs.push((&[], tok.as_slice()));
                }
            }
            self.team.prefetch_first_primary(first, &pairs);
        }
        Ok(readout)
    }

    /// Returns `Ok(None)` when a request-scoped sampling failure occurred and
    /// the request was marked failed + scheduled for zero-commit rollback.
    fn verify_request_tokens(
        &mut self,
        request: &mut Request,
        draft: &[u32],
        target: &[&super::super::team::Output],
    ) -> Result<Option<Vec<u32>>> {
        if request.host_sampling()
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
                Ok(tokens) => Ok(Some(tokens)),
                Err(error)
                    if core.decoder.failure().is_some()
                        || error == "Rejected proposal has no residual mass" =>
                {
                    if core.decoder.failure().is_none() {
                        core.decoder.fail(error);
                    }
                    self.health.sampling_request_failures += 1;
                    Ok(None)
                }
                Err(error) => Err(error),
            }
        } else if request.processed_sampling && request.device_sample() {
            match request.verify_device(draft, target) {
                Ok(tokens) => Ok(Some(tokens)),
                Err(_) if request.decoder.failure().is_some() => {
                    self.health.sampling_request_failures += 1;
                    Ok(None)
                }
                Err(error) => Err(error),
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
                return Ok(None);
            }
            let count = match request.decoder.consume_unconstrained_prefix(&tokens) {
                Ok(count) => count,
                Err(_) if request.decoder.failure().is_some() => {
                    self.health.sampling_request_failures += 1;
                    return Ok(None);
                }
                Err(error) => return Err(error),
            };
            if count == 0 {
                return Err("Empty unconstrained verification prefix".into());
            }
            tokens.truncate(count);
            Ok(Some(tokens))
        }
    }

    fn record_dspark_round_metrics(
        &mut self,
        requests: &mut [&mut Request],
        metrics: RoundMetrics<'_>,
    ) {
        let draft_s = metrics.draft_timing.seconds;
        let verify_s = metrics.verify_timing.seconds;
        let draft_prepare = metrics.draft_timing.preparation;
        let verify_prepare = metrics.verify_timing.preparation;
        let restore_s = metrics.restore_s;
        self.speculation_policy.observe(
            metrics.selected.len(),
            (metrics.full, metrics.mixed, metrics.uniform),
            Costs {
                draft_replay: metrics.draft_timing.replay,
                verify_replay: metrics.verify_timing.replay,
                shared: [metrics.draft_timing.shared, metrics.verify_timing.shared],
                shared_setup: [
                    metrics.draft_timing.shared_setup,
                    metrics.verify_timing.shared_setup,
                ],
                work: metrics.verified_work,
                setup: [metrics.draft_timing.setup, metrics.verify_timing.setup],
                wall_s: Some(metrics.round_started.elapsed().as_secs_f64()),
                stages: [
                    draft_s + draft_prepare - metrics.draft_timing.setup,
                    verify_s + metrics.sampling_s + verify_prepare - metrics.verify_timing.setup,
                    restore_s,
                    0.,
                ],
            },
            metrics.counts,
            metrics.block,
        );
        self.speculation.rounds += metrics.selected.len();
        for &(i, _) in metrics.selected {
            requests[i].mtp.draft_s += draft_s;
            requests[i].mtp.verification_s += verify_s;
            requests[i].mtp.restore_s += restore_s;
        }
        self.speculation.draft_s += draft_s;
        self.speculation.verification_s += verify_s;
        self.speculation.restore_s += restore_s;
        self.statistics.iterations += 1;
        self.statistics.speculative_iterations += 1;
        self.statistics.compute_s += draft_s + verify_s + restore_s;
        self.statistics.batch_plan_s += draft_prepare + verify_prepare;
        *self
            .statistics
            .batch_histogram
            .entry(metrics.packet_batch)
            .or_default() += 1;
        if metrics.prefill_rows > 0 {
            self.statistics.mixed_iterations += 1;
            self.statistics.mixed_speculative_iterations += 1;
            self.statistics.bounded_prefill_iterations += 1;
            *self
                .statistics
                .prefill_chunk_histogram
                .entry(metrics.prefill_rows)
                .or_default() += 1;
            *self
                .statistics
                .prefill_batch_histogram
                .entry(metrics.prefill_requests)
                .or_default() += 1;
            self.statistics.max_bounded_prefill_s = self
                .statistics
                .max_bounded_prefill_s
                .max(draft_s + verify_s + restore_s);
        }
    }

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
        let batch_contract = self
            .team
            .model
            .batch
            .as_ref()
            .ok_or("Missing A2 batch contract")?;
        let owners = batch_contract.owners;
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
            Some(
                indices
                    .iter()
                    .map(|&i| {
                        let r = &requests[i];
                        Ok((r.slot, crate::mtp::round_step(r.mtp.rounds, r.generated)?))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?,
            )
        } else {
            None
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
        let mut budgets = BTreeMap::<(usize, usize, bool), usize>::new();
        let mut retained_cache = BTreeMap::<Vec<usize>, (Layout, super::super::batch::Key)>::new();
        let mut verification_budget = |rows: &[usize]| {
            let mut sum = 0usize;
            let mut max_decode_context = 0usize;
            let mut any_needs_logits = false;
            for (idx, &n) in rows.iter().enumerate() {
                if n > 0 {
                    sum += n;
                    let r = &requests[idx];
                    if !r.prefilling {
                        max_decode_context = max_decode_context.max(r.history.len());
                        any_needs_logits |= r.needs_logits();
                    }
                }
            }
            *budgets
                .entry((sum, max_decode_context, any_needs_logits))
                .or_insert_with(|| {
                    baseline.verification_budget(
                        requests,
                        rows,
                        keep,
                        &prefill_estimator,
                        ceiling_s,
                    )
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
                    let (retained, retained_key) = if let Some(cached) =
                        retained_cache.get(reference)
                    {
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
                        let key = built.bound_work_key(batch_contract, &self.team.work_launches)?;
                        retained_cache
                            .entry(reference.to_vec())
                            .or_insert((built, key))
                    };
                    if !target.bound_work_cached(
                        retained,
                        Some(retained_key),
                        batch_contract,
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
        let mut depths = allocation.depths;
        let expanded_bucket = baseline.prefill_rows == 0
            && self.expand_verify_bucket_slack(&mut depths, &maxima, &feedback, capacity);
        for (&i, &depth) in indices.iter().zip(&depths) {
            decode_rows[i] = depth + 1;
        }
        let packet_budget = verification_budget(&decode_rows);
        let retained_work =
            if let Some(reference) = allocation.reference.filter(|_| !expanded_bucket) {
                if super::super::diag().dspark_policy_trace {
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
        let maximum = selected.iter().map(|(_, count)| *count).max().unwrap_or(0);
        *self
            .statistics
            .mtp_depth_histogram
            .entry(maximum)
            .or_default() += 1;
        if maximum == 0 {
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
        let depths = selected.iter().copied().collect::<BTreeMap<_, _>>();
        let DraftOutcome {
            proposals,
            timing: draft_timing,
            inputs,
            draft_tokens,
        } = self.run_dspark_draft(
            DraftRequest {
                requests,
                baseline_selected: &baseline.selected,
                selected: &selected,
                block,
                owners,
                rejection_steps: rejection_steps.as_ref(),
                options,
            },
            maintenance,
        )?;
        draft_timing.record(&mut self.statistics.batch_execution);
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
                self.team
                    .model
                    .batch
                    .as_ref()
                    .ok_or("Missing A2 batch contract")?,
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
        let fast_verify = self.team.can_fast_verify(&layout, &journal);
        // On the slow path, dispatch GrowCachedAndBeginVerification to all 8 ranks
        // first so CPU readout/rejection/Engram preparation overlaps the 8-rank IPC.
        let slow_started = if fast_verify {
            None
        } else {
            Some(
                self.team
                    .send_verification_forward(&layout, journal.clone())?,
            )
        };
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
        let readout_result = self.build_verification_readout(VerifyReadoutRequest {
            requests,
            full: full || packet_full,
            selected: &selected,
            draft_tokens: &draft_tokens,
            packet_requests: &packet_requests,
            layout: &layout,
            inputs: &inputs,
            rejection_steps: rejection_steps.as_ref(),
        });
        let (verified, verify_timing, save_s) = if let Some(started) = slow_started {
            let (first_prepared, first_journal_ready) =
                self.team.collect_verification_forward(&layout, started)?;
            let readout = readout_result?;
            let need_workspace = !first_prepared.all_graphs() && !first_prepared.workspace_fits;
            let verify_prepared = if !need_workspace && first_journal_ready {
                first_prepared
            } else {
                let Some(retried) = self.begin_verification_forward_with_workspace(
                    &layout,
                    journal.clone(),
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
            (verified, verify_timing, save_s)
        } else {
            let readout = readout_result?;
            let (verified, verify_timing) =
                self.team
                    .fast_verify_forward(layout, journal.clone(), inputs, readout)?;
            (verified, verify_timing, 0.)
        };
        verify_timing.record(&mut self.statistics.batch_execution);
        self.observe_prefill(prefill_work, &verify_timing);
        self.statistics.sampling_mask_rows += verify_timing.mask_rows;
        self.statistics.sampling_report_rows += verify_timing.report_rows;
        self.statistics.sampling_prepare_s += verify_timing.mask_prepare_s;
        let started = Instant::now();
        let mut counts = vec![];
        let mut steps = vec![];
        let mut rollback = vec![];
        for (&(i, count), draft) in selected.iter().zip(draft_tokens) {
            let slot = requests[i].slot;
            let target = verified
                .iter()
                .filter(|v| v.owner == slot)
                .collect::<Vec<_>>();
            if target.len() != count + 1 {
                return Err("Incomplete DSpark verification rows".into());
            }
            #[cfg(test)]
            if super::super::diag().dspark_trace && self.speculation.rounds < 8 {
                eprintln!(
                    "DSPARK_TRACE {}",
                    serde_json::json!({
                        "owner":slot,"position":requests[i].history.len()-1,
                        "anchor":requests[i].history.last(),"draft":draft,
                        "target":target.iter().map(|v|v.token).collect::<Vec<_>>(),
                        "confidence":proposals.iter().filter(|v|v.owner==slot).map(|v|v.value).collect::<Vec<_>>()
                    })
                );
            }
            let Some(committed) = self.verify_request_tokens(requests[i], &draft, &target)? else {
                rollback.push((slot, 0));
                continue;
            };
            let request = &mut requests[i];
            // The last output stays pending. Keep confirmed writes, restore
            // overwritten live slots, and shorten the logical history.
            if committed.len() < count + 1 {
                rollback.push((slot, committed.len()));
            }
            if count > 0 {
                let confidence = proposals
                    .iter()
                    .filter(|p| p.owner == slot)
                    .map(|p| p.value as f64)
                    .collect::<Vec<_>>();
                self.speculation_policy.observe_feedback(
                    slot,
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
        self.team.finish_verification(rollback, &journal)?;
        let restore_s = started.elapsed().as_secs_f64() + save_s;
        let prefill = packet_requests
            .into_iter()
            .filter(|&(i, _)| requests[i].prefilling)
            .collect();
        steps.extend(self.commit_packet(requests, prefill, &verified)?);
        self.record_dspark_round_metrics(
            requests,
            RoundMetrics {
                selected: &selected,
                full,
                mixed,
                uniform,
                block,
                round_started,
                draft_timing: &draft_timing,
                verify_timing: &verify_timing,
                sampling_s,
                restore_s,
                verified_work,
                counts: &counts,
                packet_batch,
                prefill_rows,
                prefill_requests,
            },
        );
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
