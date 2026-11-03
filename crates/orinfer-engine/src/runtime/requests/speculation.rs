//! Speculation advances one bounded verification round. Catch-up is lazy while
//! batched target decode is profitable; stale ring entries are never consumed.
use super::*;
pub(in crate::runtime) mod control;

#[derive(Default)]
pub(in crate::runtime) struct Profile {
    time: scheduler::Estimate,
    trials: usize,
}
impl Profile {
    pub fn upper(&self) -> f64 {
        self.time.upper()
    }
    pub fn observe(&mut self, seconds: f64) {
        if self.trials == 1 {
            // The first use may capture a graph or load a lazy CUDA module.
            self.time = Default::default();
        }
        self.time.observe(seconds);
        self.trials += 1;
    }
}

pub(super) enum Preparation {
    Finished(Vec<u32>),
    Round(PreparedRound),
}
pub(super) struct PreparedRound {
    pub(super) plan: crate::mtp::VerificationPlan,
    position: usize,
    remaining: usize,
    vocab: usize,
    host_sampling: bool,
    is_lookup: bool,
    drafts: Vec<u32>,
    proposals: Vec<crate::mtp::Proposal>,
    pub(super) verification: Vec<u32>,
}

struct CommittedRound {
    tokens: Vec<u32>,
    plan: crate::mtp::VerificationPlan,
    position: usize,
    is_lookup: bool,
}

enum DraftPreparation {
    Finished(Vec<u32>),
    Draft(Box<DraftRound>),
}
struct DraftRound {
    round: PreparedRound,
    decoder: crate::sampling::Decoder,
    history: Vec<u32>,
    done: bool,
    failed: bool,
    started: Instant,
}
pub(super) struct BatchChoice {
    pub(super) selected: Vec<(usize, usize)>,
}

pub(super) struct Feedback {
    pub(super) key: control::Key,
    pub(super) width: usize,
    pub(super) lookups: usize,
    pub(super) generated: usize,
    pub(super) indices: Vec<usize>,
}

impl ModelRuntime {
    pub(super) fn maintain_lazy_drafts(
        &mut self,
        requests: &mut [&mut RequestState],
    ) -> Result<()> {
        let Some(spec) = self.manifest.mtp.clone() else {
            return Ok(());
        };
        let Some(ring) = spec
            .hidden_ring
            .as_ref()
            .and_then(|name| self.manifest.buffers.iter().find(|b| b.name == *name))
            .map(|b| b.shape[0])
        else {
            return Ok(());
        };
        let width = spec.verification_widths().into_iter().max().unwrap_or(1);
        let margin = requests.len().saturating_mul(width).max(ring / 4).min(ring);
        let threshold = ring.saturating_sub(margin);
        let mut due: Vec<_> = requests
            .iter()
            .enumerate()
            .filter_map(|(i, r)| {
                if r.prefilling || r.is_finished() || r.mtp_blocked {
                    return None;
                }
                let lag = r
                    .history
                    .len()
                    .saturating_sub(1)
                    .saturating_sub(r.warm.tokens);
                (lag >= threshold && lag > 0).then_some((i, lag))
            })
            .collect();
        due.sort_unstable_by_key(|&(i, lag)| (std::cmp::Reverse(lag), requests[i].slot));
        let mut ordinary_maintenance = false;
        for (i, lag) in due {
            let urgent = lag.saturating_add(width) > ring;
            if !urgent && ordinary_maintenance {
                continue;
            }
            if lag > ring {
                requests[i].mtp_blocked = true;
                continue;
            }
            let tokens = spec.warm_plan(lag)?.1;
            let began = Instant::now();
            self.with_request(requests[i], |model, req| {
                let end = req.warm.tokens + tokens;
                model.mtp_warm_state(
                    &spec,
                    &req.history[req.warm.tokens + 1..end + 1],
                    &|| false,
                    ExecutionPhase::Decode,
                    true,
                )?;
                req.warm.tokens = end;
                req.mtp.refresh_s += began.elapsed().as_secs_f64();
                Ok(())
            })?;
            self.scheduler_statistics.mtp_catchup_tokens += tokens;
            self.scheduler_statistics.mtp_catchup_s += began.elapsed().as_secs_f64();
            ordinary_maintenance = true;
        }
        Ok(())
    }
    fn speculation_key(
        requests: &[&mut RequestState],
        selected: &[(usize, usize)],
    ) -> control::Key {
        let decode: Vec<_> = selected
            .iter()
            .filter(|(i, _)| !requests[*i].prefilling)
            .collect();
        control::Key {
            batch: decode.len(),
            context: decode
                .iter()
                .map(|(i, _)| requests[*i].history.len())
                .max()
                .unwrap_or(1)
                .max(1)
                .ilog2(),
            sampling: decode.iter().any(|(i, _)| {
                !requests[*i].sampling.is_greedy() || requests[*i].sampling.top_logprobs.is_some()
            }),
            prefill: selected
                .iter()
                .filter(|(i, _)| requests[*i].prefilling)
                .map(|(_, n)| n)
                .sum(),
            shared_work: selected.iter().any(|(i, _)| requests[*i].shared_work),
        }
    }
    pub(super) fn speculation_feedback(
        &self,
        requests: &[&mut RequestState],
        work: &Work,
    ) -> Option<Feedback> {
        self.manifest.mtp.as_ref()?;
        let selected = match work {
            Work::Batch(rows) | Work::MixedSpeculative(rows) => rows.clone(),
            Work::SpeculativeBatch(choice) => choice.selected.clone(),
            Work::Speculative(i, width) => vec![(*i, *width)],
            _ => return None,
        };
        let key = Self::speculation_key(requests, &selected);
        if key.batch == 0 {
            return None;
        }
        // Reallocated packets keep the same total row budget as the nominal
        // controller arm; request-local depths need not be equal.
        let rows: usize = selected
            .iter()
            .filter(|(i, _)| !requests[*i].prefilling)
            .map(|(_, n)| *n)
            .sum();
        if !rows.is_multiple_of(key.batch) {
            return None;
        }
        let width = rows / key.batch;
        Some(Feedback {
            key,
            width,
            lookups: requests.iter().map(|r| r.mtp.lookup_rounds).sum(),
            indices: selected
                .iter()
                .filter(|(i, _)| !requests[*i].prefilling)
                .map(|(i, _)| *i)
                .collect(),
            generated: selected
                .iter()
                .filter(|(i, _)| !requests[*i].prefilling)
                .map(|(i, _)| requests[*i].generated)
                .sum(),
        })
    }
    pub(super) fn select_auto_speculation(
        &mut self,
        requests: &[&mut RequestState],
        joint: &[(usize, usize)],
        options: &scheduler::Options,
        slack: f64,
    ) -> Option<BatchChoice> {
        let spec = self.manifest.mtp.as_ref()?;
        let key = Self::speculation_key(requests, joint);
        if key.batch == 0 {
            return None;
        }
        let mixed = key.prefill > 0;
        let plans = if mixed {
            spec.mixed_verification_tokens.clone()
        } else if key.batch > 1 {
            spec.batch_verification_tokens.clone()
        } else {
            spec.verification_widths()
        };
        let mut plans = plans;
        plans.sort_unstable();
        plans.dedup();
        let owners: Vec<_> = joint
            .iter()
            .filter(|(i, _)| !requests[*i].prefilling)
            .map(|(i, _)| *i)
            .collect();
        let remaining_tokens = owners
            .iter()
            .map(|&i| {
                let req = &requests[i];
                (req.limit - req.generated).min(
                    self.manifest
                        .max_context
                        .saturating_sub(req.history.len().saturating_sub(1)),
                )
            })
            .min()?
            .saturating_mul(key.batch);
        let choices: Vec<Vec<_>> = owners
            .iter()
            .map(|&i| {
                let req = &requests[i];
                let limit = (req.limit - req.generated)
                    .min(
                        self.manifest
                            .max_context
                            .saturating_sub(req.history.len().saturating_sub(1)),
                    )
                    .min(spec.default_verification_tokens);
                plans
                    .iter()
                    .copied()
                    .filter(|&n| n >= 2 && n <= limit)
                    .map(|n| (n, req.draft_acceptance.expected(n - 1)))
                    .collect()
            })
            .collect();
        if joint
            .iter()
            .any(|(i, _)| !requests[*i].prefilling && requests[*i].mtp_blocked)
        {
            return None;
        }
        let baseline = self.predict_iteration(requests, joint);
        let row_budget = if options.max_speculative_tokens == 0 {
            4096
        } else {
            options.max_speculative_tokens
        };
        let mut candidates = vec![];
        let mut allocations = BTreeMap::new();
        for width in plans.into_iter().filter(|&n| {
            n >= 2 && n <= spec.default_verification_tokens && n * key.batch <= row_budget
        }) {
            let Some(depths) = control::allocate(&choices, width * key.batch) else {
                continue;
            };
            let depths: BTreeMap<_, _> = owners.iter().copied().zip(depths).collect();
            let selected: Vec<_> = joint
                .iter()
                .map(|&(i, n)| (i, depths.get(&i).copied().unwrap_or(n)))
                .collect();
            let mut segments: Vec<_> = selected
                .iter()
                .map(|&(i, tokens)| BatchSegment {
                    slot: requests[i].slot,
                    tokens,
                })
                .collect();
            if mixed {
                segments.sort_by_key(|s| {
                    !requests[joint
                        .iter()
                        .find(|(i, _)| requests[*i].slot == s.slot)
                        .unwrap()
                        .0]
                        .prefilling
                });
            }
            let packet = mixed || key.batch > 1 || spec.batch_verification_tokens.contains(&width);
            let operation = if mixed {
                orinfer_model_sdk::abi::BatchOperation::MixedRows {
                    prefills: (joint.len() - key.batch) as u8,
                }
            } else {
                orinfer_model_sdk::abi::BatchOperation::Verification
            };
            if packet && !self.model_package.supports_batch(&segments, operation) {
                continue;
            }
            let expected_tokens = selected
                .iter()
                .filter(|(i, _)| !requests[*i].prefilling)
                .map(|(i, n)| requests[*i].draft_acceptance.expected(n - 1))
                .sum();
            allocations.insert(width, selected.clone());
            // A verification round rides the same joined weight pass as one
            // plain iteration; only the extra draft tokens add real cost. A
            // rows*context prior priced each owner as its own prefill chunk
            // and froze batched speculation out of every probe.
            candidates.push(control::Candidate {
                width,
                expected_tokens,
                prior_seconds: baseline + self.mtp_seconds_per_draft * (width - 1) as f64,
                remaining_tokens,
            });
        }
        candidates.sort_by_key(|c| c.width);
        self.dynamic_speculation
            .set_calibration_limit(options.target_tpot_ms / 1000.);
        let decision = self
            .dynamic_speculation
            .choose(key, &candidates, baseline, slack);
        *self
            .scheduler_statistics
            .mtp_depth_histogram
            .entry(decision.map_or(0, |(w, _)| w - 1))
            .or_default() += 1;
        let (width, exploring) = decision?;
        self.scheduler_statistics.mtp_probe_iterations += usize::from(exploring);
        let selected = allocations.remove(&width)?;
        for &(i, n) in &selected {
            if !requests[i].prefilling {
                *self
                    .scheduler_statistics
                    .mtp_request_depth_histogram
                    .entry(n - 1)
                    .or_default() += 1;
            }
        }
        Some(BatchChoice { selected })
    }
    pub(super) fn select_mixed_speculation(
        &mut self,
        requests: &[&mut RequestState],
        joint: &[(usize, usize)],
        options: &scheduler::Options,
        slack: f64,
    ) -> Option<Vec<(usize, usize)>> {
        self.select_auto_speculation(requests, joint, options, slack)
            .map(|choice| choice.selected)
    }
    #[cfg(test)]
    pub(super) fn speculative_mixed_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        self.speculative_mixed_step_with(requests, selected, output, &mut || {})
    }
    pub(super) fn speculative_mixed_step_with(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        let (prefills, decoders): (Vec<_>, Vec<_>) = selected
            .iter()
            .copied()
            .partition(|&(i, _)| requests[i].prefilling);
        if prefills.is_empty()
            || decoders.is_empty()
            || prefills
                .iter()
                .any(|&(i, _)| requests[i].prefill_frame.is_some())
        {
            return Err("Invalid mixed speculative work".into());
        }
        self.speculative_packet_step(requests, &decoders, &prefills, output, overlap)
    }

    #[cfg(test)]
    pub(super) fn speculative_batch_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
    ) -> Result<()> {
        self.speculative_packet_step(requests, selected, &[], output, &mut || {})
    }

    pub(super) fn speculative_packet_step(
        &mut self,
        requests: &mut [&mut RequestState],
        selected: &[(usize, usize)],
        prefills: &[(usize, usize)],
        output: &mut Vec<StepOutput>,
        overlap: &mut dyn FnMut(),
    ) -> Result<()> {
        let mut rounds = vec![];
        let mut drafts = vec![];
        let grouped_draft = self
            .manifest
            .mtp
            .as_ref()
            .is_some_and(|s| !s.batch_draft_profiles.is_empty());
        let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
        let slots: Vec<_> = selected.iter().map(|(i, _)| requests[*i].slot).collect();
        let cursors = self.execution.read_sequence_controls(
            &slots,
            [self.manifest.position.as_str(), spec.position.as_str()],
        )?;
        for (&(i, width), cursor) in selected.iter().zip(cursors) {
            if grouped_draft {
                match self.start_speculative_round(requests[i], width, Some(cursor)) {
                    Ok(DraftPreparation::Finished(tokens)) => {
                        self.scheduler_statistics.decode_tokens += tokens.len();
                        output.push(StepOutput { request: i, tokens });
                    }
                    Ok(DraftPreparation::Draft(build)) => drafts.push((i, *build)),
                    Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    }),
                    Err(error) => return Err(error),
                }
                continue;
            }
            match self.with_request(requests[i], |model, req| {
                model.prepare_speculative_round(req, width)
            }) {
                Ok(Preparation::Finished(tokens)) => {
                    self.scheduler_statistics.decode_tokens += tokens.len();
                    output.push(StepOutput { request: i, tokens });
                }
                Ok(Preparation::Round(round)) => rounds.push((i, round)),
                Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                    request: i,
                    tokens: vec![],
                }),
                Err(error) => return Err(error),
            }
        }
        if grouped_draft {
            self.generate_batch_drafts(requests, &mut drafts)?;
            for (i, build) in drafts {
                if build.failed {
                    output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    });
                    continue;
                }
                rounds.push((i, Self::finish_drafts(requests[i], build)?));
            }
        }
        if rounds.len() == 1 && prefills.is_empty() && !self.deferred_verification() {
            let (i, round) = rounds.pop().unwrap();
            let result = self.with_request(requests[i], |model, req| {
                model.verify_single_round(req, &round)?;
                model.commit_verified_round(req, round)
            });
            let tokens = match result {
                Ok(tokens) => tokens,
                Err(_) if requests[i].failure().is_some() => vec![],
                Err(error) => return Err(error),
            };
            self.scheduler_statistics.decode_tokens += tokens.len();
            output.push(StepOutput { request: i, tokens });
            return Ok(());
        }
        if rounds.is_empty() {
            if !prefills.is_empty() {
                self.execute_selected(requests, prefills, output)?;
            }
            return Ok(());
        }
        let at = Instant::now();
        let mut segments = if prefills.is_empty() {
            vec![]
        } else {
            self.prepare_batch_inputs(requests, prefills)?
        };
        for (i, round) in &rounds {
            let req = &mut requests[*i];
            self.execution.ensure_sequence_program_at(
                req.slot,
                &round.plan.program,
                round.position,
                round.plan.tokens,
            )?;
            self.execution.upload_sequence_ids(
                req.slot,
                &self.manifest.input,
                &round.verification,
            )?;
            self.prepare_sequence_program_inputs(
                req.slot,
                &round.plan.program,
                &round.verification,
                &req.history[..req.history.len() - 1],
            )?;
            self.upload_sequence_segment_controls(req.slot, round.plan.tokens)?;
            segments.push(BatchSegment {
                slot: req.slot,
                tokens: round.plan.tokens,
            });
        }
        if let Some(table) = self
            .manifest
            .mtp
            .as_ref()
            .and_then(|s| s.verification_table.as_ref())
        {
            let mut offset = 0;
            for segment in &segments {
                self.execution.upload_sequence_bytes(
                    segment.slot,
                    &table.row_offset,
                    &(offset as u32).to_le_bytes(),
                )?;
                offset += segment.tokens;
            }
        }
        let operation = if prefills.is_empty() {
            orinfer_model_sdk::abi::BatchOperation::Verification
        } else {
            orinfer_model_sdk::abi::BatchOperation::MixedRows {
                prefills: prefills.len() as u8,
            }
        };
        let (plan, bindings) = self.model_package.cached_batch(&segments, operation)?;
        self.upload_spec_bindings(&segments, operation, bindings)?;
        self.execution.execute_cached_batch_operation_with(
            operation,
            segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            &plan,
            prefills.is_empty(),
            rounds
                .iter()
                .all(|(i, _)| requests[*i].limit - requests[*i].generated >= 2),
            overlap,
        )?;
        let seconds = at.elapsed().as_secs_f64();
        if !prefills.is_empty() {
            self.scheduler_statistics.mixed_iterations += 1;
            self.scheduler_statistics.mixed_speculative_iterations += 1;
            *self
                .scheduler_statistics
                .prefill_batch_histogram
                .entry(prefills.len())
                .or_default() += 1;
            self.commit_batch(requests, prefills, output)?;
        }
        let total_rows: usize = segments.iter().map(|s| s.tokens).sum();
        let grouped_commit = self.manifest.mtp.as_ref().is_some_and(|s| s.batch_commit);
        let mut committed = vec![];
        let control_width = rounds.iter().map(|(_, r)| r.plan.tokens).max().unwrap();
        let slots: Vec<_> = rounds.iter().map(|(i, _)| requests[*i].slot).collect();
        let spec = self.manifest.mtp.as_ref().unwrap();
        let control_values = self.execution.read_sequence_vectors(
            &slots,
            &[&spec.verification_tokens, &spec.verification_status],
            control_width,
        )?;
        for ((i, round), values) in rounds
            .into_iter()
            .zip(control_values.chunks_exact(2 * control_width))
        {
            let width = round.plan.tokens;
            requests[i].mtp.verification_s +=
                seconds * round.plan.tokens as f64 / total_rows as f64;
            if grouped_commit {
                let result = self.sample_verified_round(
                    requests[i],
                    round,
                    Some((
                        &values[..width],
                        &values[control_width..control_width + width],
                    )),
                );
                match result {
                    Ok(round) => committed.push((i, round)),
                    Err(_) if requests[i].failure().is_some() => output.push(StepOutput {
                        request: i,
                        tokens: vec![],
                    }),
                    Err(error) => return Err(error),
                }
                continue;
            }
            let result = self.with_request(requests[i], |model, req| {
                model.commit_verified_round(req, round)
            });
            let tokens = match result {
                Ok(tokens) => tokens,
                Err(_) if requests[i].failure().is_some() => vec![],
                Err(error) => return Err(error),
            };
            self.scheduler_statistics.decode_tokens += tokens.len();
            output.push(StepOutput { request: i, tokens });
        }
        if !committed.is_empty() {
            let at = Instant::now();
            let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
            let mut segments = vec![];
            for (i, round) in &committed {
                self.execution.upload_sequence_ids(
                    requests[*i].slot,
                    &spec.accepted_inputs,
                    &[round.tokens.len() as u32],
                )?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: round.plan.tokens,
                });
            }
            if segments.len() >= 2
                || self.deferred_verification()
                || self.manifest.packet_table().is_some()
            {
                self.commit_spec_packet(&segments)?;
            } else {
                let round = &committed[0].1;
                self.execution.activate_sequence(segments[0].slot)?;
                self.launch_rows(
                    &round.plan.restore_program,
                    round.plan.tokens,
                    ExecutionPhase::Decode,
                )?;
            }
            let restore_s = at.elapsed().as_secs_f64() / committed.len() as f64;
            let grouped_warm = self.manifest.mtp.as_ref().is_some_and(|s| s.batch_warm);
            if grouped_warm {
                for (i, round) in &committed {
                    requests[*i].mtp.restore_s += restore_s;
                    self.restore_verified_cursor(requests[*i], round, true)?;
                }
                self.warm_committed_batch(requests, &committed)?;
            }
            let positions = if grouped_warm {
                let slots: Vec<_> = committed.iter().map(|(i, _)| requests[*i].slot).collect();
                self.execution
                    .read_sequence_controls(&slots, [self.manifest.position.as_str()])?
                    .into_iter()
                    .map(|v| Some(v[0] as usize))
                    .collect::<Vec<_>>()
            } else {
                vec![None; committed.len()]
            };
            for ((i, round), position) in committed.into_iter().zip(positions) {
                let tokens = if grouped_warm {
                    self.finish_warmed_commit(requests[i], round, position)?
                } else {
                    requests[i].mtp.restore_s += restore_s;
                    self.with_request(requests[i], |model, req| {
                        model.finish_verified_round(req, round, true)
                    })?
                };
                self.scheduler_statistics.decode_tokens += tokens.len();
                output.push(StepOutput { request: i, tokens });
            }
        }
        Ok(())
    }
    fn deferred_verification(&self) -> bool {
        self.manifest
            .mtp
            .as_ref()
            .and_then(|s| s.verification_table.as_ref())
            .is_some_and(|t| t.deferred)
    }
    fn commit_spec_packet(&mut self, segments: &[BatchSegment]) -> Result<()> {
        use orinfer_model_sdk::abi::BatchOperation;
        let (plan, bindings) = self
            .model_package
            .cached_batch(segments, BatchOperation::Commit)?;
        self.upload_spec_bindings(segments, BatchOperation::Commit, bindings)?;
        self.execution.execute_cached_batch_operation(
            BatchOperation::Commit,
            segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            &plan,
            true,
            true,
        )
    }
    fn upload_spec_bindings(
        &mut self,
        segments: &[BatchSegment],
        phase: orinfer_model_sdk::abi::BatchOperation,
        bindings: std::rc::Rc<orinfer_model_sdk::abi::StateBindings>,
    ) -> Result<()> {
        use orinfer_model_sdk::abi::BatchOperation;
        let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP contract")?;
        let suffix = self
            .manifest
            .packet_table()
            .map(|t| {
                segments
                    .len()
                    .checked_mul(t.columns.len())
                    .ok_or("MTP table size overflow")
            })
            .transpose()?
            .unwrap_or(0);
        let prefix = bindings
            .len()
            .checked_sub(suffix)
            .ok_or("Truncated MTP packet bindings")?;
        let mut tables = vec![];
        if let Some(table) = self.manifest.packet_table() {
            tables.push((table.buffer.as_str(), prefix..bindings.len()));
        }
        if prefix != 0 {
            let table = if phase == BatchOperation::Commit {
                spec.commit_state_table
                    .as_ref()
                    .ok_or("Missing commit pointer table")?
            } else if matches!(
                phase,
                BatchOperation::Verification | BatchOperation::MixedRows { .. }
            ) {
                let table = spec
                    .verification_table
                    .as_ref()
                    .ok_or("Missing verification pointer table")?;
                let mut offset = 0;
                for segment in segments {
                    self.execution.upload_sequence_ids(
                        segment.slot,
                        &table.row_offset,
                        &[offset as u32],
                    )?;
                    offset += segment.tokens;
                }
                &table.buffer
            } else {
                return Err("Unexpected MTP table bindings".into());
            };
            tables.push((table.as_str(), 0..prefix));
        }
        if tables.is_empty() {
            Ok(())
        } else {
            self.execution
                .upload_sequence_address_ranges(bindings, &tables)
        }
    }
    fn execute_spec_packet(
        &mut self,
        segments: &[BatchSegment],
        phase: orinfer_model_sdk::abi::BatchOperation,
    ) -> Result<()> {
        self.execute_spec_packet_phase(segments, phase, ExecutionPhase::Decode)
    }
    pub(in crate::runtime) fn execute_spec_packet_phase(
        &mut self,
        segments: &[BatchSegment],
        phase: orinfer_model_sdk::abi::BatchOperation,
        execution_phase: ExecutionPhase,
    ) -> Result<()> {
        let (plan, bindings) = self.model_package.cached_batch(segments, phase)?;
        self.upload_spec_bindings(segments, phase, bindings)?;
        self.execution.execute_cached_batch_operation(
            phase,
            segments.iter().map(|s| (s.slot, s.tokens)).collect(),
            &plan,
            execution_phase == ExecutionPhase::Decode,
            true,
        )
    }
    fn warm_committed_batch(
        &mut self,
        requests: &mut [&mut RequestState],
        committed: &[(usize, CommittedRound)],
    ) -> Result<()> {
        use orinfer_model_sdk::abi::BatchOperation;
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let capacity = spec.batch_warm_capacity.max(
            *spec
                .batch_draft_profiles
                .last()
                .ok_or("Missing warm batch capacity")?,
        );
        let at = Instant::now();
        let mut cursor = 0;
        while cursor < committed.len() {
            let mut rows = 0;
            let mut segments = vec![];
            let mut endpoints = vec![];
            while cursor < committed.len() {
                let (i, round) = &committed[cursor];
                let tokens = round.tokens.len();
                if rows + tokens > capacity {
                    break;
                }
                let (warm, live) = spec.warm_plan(tokens)?;
                if live != tokens {
                    return Err("Accepted warm chain exceeds native capacity".into());
                }
                let slot = requests[*i].slot;
                self.execution
                    .upload_sequence_ids(slot, &spec.input, &round.tokens)?;
                self.prepare_sequence_program_inputs(slot, &warm.program, &round.tokens, &[])?;
                self.execution.ensure_sequence_program_at(
                    slot,
                    &warm.program,
                    round.position,
                    tokens,
                )?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens,
                });
                endpoints.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: 1,
                });
                rows += tokens;
                cursor += 1;
            }
            if segments.is_empty() {
                return Err("Accepted warm chain exceeds native packet capacity".into());
            }
            // Each owner processes its complete causal chain once. Only dense
            // projection/MoE rows are joined; heads consume request endpoints.
            self.execute_spec_packet(&segments, BatchOperation::Warm)?;
            self.execute_spec_packet(&endpoints, BatchOperation::DraftHead)?;
        }
        let total: usize = committed.iter().map(|(_, c)| c.tokens.len()).sum();
        let seconds = at.elapsed().as_secs_f64();
        for (i, round) in committed {
            requests[*i].mtp.refresh_s += seconds * round.tokens.len() as f64 / total as f64;
        }
        Ok(())
    }
    fn generate_batch_drafts(
        &mut self,
        requests: &mut [&mut RequestState],
        drafts: &mut [(usize, DraftRound)],
    ) -> Result<()> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        while drafts.iter().any(|(_, d)| !d.done) {
            let active: Vec<_> = drafts
                .iter()
                .enumerate()
                .filter(|(_, (_, d))| !d.done)
                .map(|(i, _)| i)
                .collect();
            let advancing: Vec<_> = active
                .iter()
                .copied()
                .filter(|&i| !drafts[i].1.round.drafts.is_empty())
                .collect();
            let mut segments = vec![];
            for &j in &advancing {
                let (i, draft) = &drafts[j];
                let slot = requests[*i].slot;
                self.prepare_sequence_program_inputs(
                    slot,
                    &spec.draft_program,
                    &[*draft.round.drafts.last().unwrap()],
                    &[],
                )?;
                self.execution.ensure_sequence_program_at(
                    slot,
                    &spec.draft_program,
                    draft.round.position + draft.round.drafts.len() - 1,
                    1,
                )?;
                segments.push(BatchSegment {
                    slot: requests[*i].slot,
                    tokens: 1,
                });
            }
            if !segments.is_empty()
                && (segments.len() >= 2 || self.manifest.packet_table().is_some())
            {
                // A package binds at most its declared draft lanes per packet;
                // wider cohorts split into sequential draft packets.
                let lanes = spec
                    .batch_draft_profiles
                    .last()
                    .copied()
                    .unwrap_or(segments.len());
                for chunk in segments.chunks(lanes.max(1)) {
                    let (plan, bindings) = self
                        .model_package
                        .cached_batch(chunk, orinfer_model_sdk::abi::BatchOperation::Draft)?;
                    self.upload_spec_bindings(
                        chunk,
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        bindings,
                    )?;
                    self.execution.execute_cached_batch_operation(
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        chunk.iter().map(|s| (s.slot, s.tokens)).collect(),
                        &plan,
                        true,
                        true,
                    )?;
                }
            } else if let Some(segment) = segments.first() {
                self.execution.activate_sequence(segment.slot)?;
                self.launch_rows(&spec.draft_program, 1, ExecutionPhase::Decode)?;
            }
            let slots: Vec<_> = active.iter().map(|&j| requests[drafts[j].0].slot).collect();
            let controls = self
                .execution
                .read_sequence_controls(&slots, [spec.token.as_str(), spec.status.as_str()])?;
            for (j, control) in active.into_iter().zip(controls) {
                let (i, draft) = &mut drafts[j];
                match self.consume_draft(requests[*i], draft, Some(control)) {
                    Ok(()) => {}
                    Err(_) if requests[*i].failure().is_some() => {
                        draft.done = true;
                        draft.failed = true;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(())
    }
    pub(super) fn refresh_request_mtp(
        &mut self,
        req: &mut RequestState,
        head: bool,
    ) -> Result<bool> {
        self.refresh_request_mtp_at(req, head, None)
    }
    fn refresh_request_mtp_at(
        &mut self,
        req: &mut RequestState,
        head: bool,
        cursors: Option<[i32; 2]>,
    ) -> Result<bool> {
        let Some(spec) = self.manifest.mtp.clone() else {
            return Ok(true);
        };
        let [position, draft_position] = match cursors {
            Some(values) => values,
            None => self
                .execution
                .read_control_group([&self.manifest.position, &spec.position])?,
        };
        if position < 0 || draft_position < 0 {
            return Err("Invalid private MTP cursor".into());
        }
        let (position, draft_position) = (position as usize, draft_position as usize);
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
                &spec,
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
        let round = match self.prepare_speculative_round(req, max_rows)? {
            Preparation::Finished(tokens) => return Ok(tokens),
            Preparation::Round(round) => round,
        };
        self.verify_single_round(req, &round)?;
        self.commit_verified_round(req, round)
    }
    fn start_speculative_round(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
        cursors: Option<[i32; 2]>,
    ) -> Result<DraftPreparation> {
        if req.failure().is_some() {
            return Ok(DraftPreparation::Finished(vec![]));
        }
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        if cursors.is_none()
            || cursors.is_some_and(|v| v[0] != v[1])
            || spec.draft_snapshot_program.is_some()
        {
            self.execution.activate_sequence(req.slot)?;
        }
        let valid = self.refresh_request_mtp_at(req, true, cursors)?;
        let position = match cursors {
            Some(values) => values[0],
            None => self.read_control(&self.manifest.position)?,
        } as usize;
        let remaining = req.limit - req.generated;
        let plan = valid
            .then(|| {
                spec.best_verification_plan(
                    remaining
                        .min(max_rows)
                        .min(spec.default_verification_tokens)
                        .min(self.manifest.max_context - position),
                )
            })
            .flatten();
        let pending = *req.history.last().ok_or("Missing pending token")?;
        let Some(plan) = plan else {
            self.execution.activate_sequence(req.slot)?;
            self.execute_target_tail(req)?;
            let out = self.commit_ordinary_token(req)?;
            if valid {
                self.refresh_request_mtp(req, true)?;
            }
            return Ok(DraftPreparation::Finished(out));
        };
        let host_sampling = !req.sampling.is_greedy() || req.sampling.top_logprobs.is_some();
        let lookup = (self.prompt_lookup && !host_sampling)
            .then(|| req.prompt_lookup.propose(&req.history, plan.tokens - 1))
            .flatten();
        let is_lookup = lookup.is_some();
        if !is_lookup && let Some(save) = &spec.draft_snapshot_program {
            self.launch_program(save, ExecutionPhase::Decode)?;
        }
        let drafts = lookup.unwrap_or_default();
        if is_lookup {
            req.mtp.lookup_rounds += 1;
            req.mtp.lookup_proposed_tokens += drafts.len();
        }
        Ok(DraftPreparation::Draft(Box::new(DraftRound {
            round: PreparedRound {
                plan: plan.clone(),
                position,
                remaining,
                vocab: self.manifest.vocab,
                host_sampling,
                is_lookup,
                drafts,
                proposals: vec![],
                verification: vec![pending],
            },
            decoder: req.decoder.fork(),
            history: req.history.clone(),
            done: is_lookup,
            failed: false,
            started: Instant::now(),
        })))
    }

    /// Use the model's ordinary one-row packet for an MTP tail whenever that
    /// capability is declared. The packet owns target capture as well as state.
    fn execute_target_tail(&mut self, req: &mut RequestState) -> Result<()> {
        if self.manifest.batch_profiles.contains(&1) {
            let allow_replay = req.limit - req.generated >= 2;
            let segments = self.prepare_batch_inputs(&mut [req], &[(0, 1)])?;
            let (plan, _) = self
                .model_package
                .cached_batch(&segments, orinfer_model_sdk::abi::BatchOperation::Ordinary)?;
            self.execution.execute_batch(
                segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                &plan,
                true,
                allow_replay,
            )
        } else {
            let pending = *req.history.last().ok_or("Missing target tail input")?;
            self.upload_ids(&self.manifest.token, &[pending])?;
            self.prepare_inputs(&[pending], &req.history[..req.history.len() - 1])?;
            self.upload_segment_controls(1)?;
            self.launch_program("decode", ExecutionPhase::Decode)?;
            if let Some(spec) = &self.manifest.mtp {
                self.mtp_capture(spec, 1, ExecutionPhase::Decode)?;
            }
            Ok(())
        }
    }
    fn consume_draft(
        &mut self,
        req: &mut RequestState,
        build: &mut DraftRound,
        controls: Option<[i32; 2]>,
    ) -> Result<()> {
        let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
        let round = &mut build.round;
        let vocab = round.vocab;
        let host_sampling = round.host_sampling;
        let drafts = &mut round.drafts;
        let proposals = &mut round.proposals;
        let draft_decoder = &mut build.decoder;
        let draft_history = &mut build.history;
        let draft_options = crate::sampling::Options {
            top_logprobs: None,
            ..req.sampling.clone()
        };
        let i = drafts.len();
        let [selected, status] = match controls {
            Some(values) => values,
            None => self
                .execution
                .read_control_group([&spec.token, &spec.status])?,
        };
        if status != 0 || selected < 0 || selected as usize >= vocab {
            return Err("Invalid request draft token".into());
        }
        let token = if !host_sampling {
            let token = match draft_decoder
                .greedy_candidate(selected as u32, vocab)
                .inspect_err(|error| {
                    if draft_decoder.failure().is_some() {
                        req.decoder.fail(error.clone());
                    }
                })? {
                Some(token) => token,
                None => {
                    self.execution.activate_sequence(req.slot)?;
                    let raw = self
                        .execution
                        .download_bytes(&spec.draft_logits, vocab * 4)?;
                    draft_decoder
                        .law(
                            &floats(&raw, crate::artifact::Dtype::F32),
                            draft_history,
                            &draft_options,
                        )
                        .inspect_err(|error| {
                            if draft_decoder.failure().is_some() {
                                req.decoder.fail(error.clone());
                            }
                        })?
                        .distribution
                        .draw(0.0)?
                }
            };
            if token != selected as u32 {
                self.execution
                    .upload_sequence_ids(req.slot, &spec.token, &[token])?;
            }
            draft_decoder.consume(token).inspect_err(|error| {
                req.decoder.fail(error.clone());
            })?;
            token
        } else {
            self.execution.activate_sequence(req.slot)?;
            let raw = self
                .execution
                .download_bytes(&spec.draft_logits, vocab * 4)?;
            let law = draft_decoder
                .law(
                    &floats(&raw, crate::artifact::Dtype::F32),
                    draft_history,
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
            self.execution
                .upload_sequence_ids(req.slot, &spec.token, &[token])?;
            draft_decoder.consume(token).inspect_err(|error| {
                req.decoder.fail(error.clone());
            })?;
            token
        };
        drafts.push(token);
        draft_history.push(token);
        build.done = draft_decoder.finished() || drafts.len() == round.plan.tokens - 1;
        Ok(())
    }
    fn finish_drafts(req: &mut RequestState, mut build: DraftRound) -> Result<PreparedRound> {
        let round = &mut build.round;
        req.mtp.proposed_tokens += round.drafts.len();
        // Padding after EOS has no authority over the parser or committed tokens.
        round.drafts.resize(
            round.plan.tokens - 1,
            *round.drafts.last().ok_or("Empty draft profile")?,
        );
        round.verification.extend_from_slice(&round.drafts);
        req.mtp.draft_s += build.started.elapsed().as_secs_f64();
        Ok(build.round)
    }
    pub(super) fn prepare_speculative_round(
        &mut self,
        req: &mut RequestState,
        max_rows: usize,
    ) -> Result<Preparation> {
        let mut build = match self.start_speculative_round(req, max_rows, None)? {
            DraftPreparation::Finished(tokens) => return Ok(Preparation::Finished(tokens)),
            DraftPreparation::Draft(build) => *build,
        };
        while !build.done {
            if !build.round.drafts.is_empty() {
                let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
                self.prepare_program_inputs(
                    &spec.draft_program,
                    &[*build.round.drafts.last().unwrap()],
                    &[],
                )?;
                let program = spec.draft_program.clone();
                if self.manifest.packet_table().is_some() {
                    self.execution.ensure_sequence_program_at(
                        req.slot,
                        &program,
                        build.round.position + build.round.drafts.len() - 1,
                        1,
                    )?;
                    self.execute_spec_packet(
                        &[BatchSegment {
                            slot: req.slot,
                            tokens: 1,
                        }],
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                    )?;
                } else {
                    self.launch_rows(&program, 1, ExecutionPhase::Decode)?;
                }
            }
            self.consume_draft(req, &mut build, None)?;
        }
        Ok(Preparation::Round(Self::finish_drafts(req, build)?))
    }
    pub(super) fn verify_single_round(
        &mut self,
        req: &mut RequestState,
        round: &PreparedRound,
    ) -> Result<()> {
        let plan = &round.plan;
        let verification = &round.verification;
        let at = Instant::now();
        let packet = self
            .manifest
            .mtp
            .as_ref()
            .is_some_and(|spec| spec.batch_verification_tokens.contains(&plan.tokens));
        self.upload_segment_controls(plan.tokens)?;
        if packet {
            self.execution
                .ensure_sequence_program(req.slot, &plan.program)?;
        }
        self.upload_ids(&self.manifest.input, verification)?;
        self.prepare_verification_inputs(
            &plan.program,
            verification,
            &req.history[..req.history.len() - 1],
        )?;
        if packet {
            self.execute_spec_packet(
                &[BatchSegment {
                    slot: req.slot,
                    tokens: plan.tokens,
                }],
                orinfer_model_sdk::abi::BatchOperation::Verification,
            )?;
        } else {
            self.launch_rows(&plan.program, plan.tokens, ExecutionPhase::Decode)?;
            self.launch_rows(&plan.capture_program, plan.tokens, ExecutionPhase::Decode)?;
        }
        req.mtp.verification_s += at.elapsed().as_secs_f64();
        Ok(())
    }
    fn sample_verified_round(
        &mut self,
        req: &mut RequestState,
        round: PreparedRound,
        controls: Option<(&[u32], &[u32])>,
    ) -> Result<CommittedRound> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let PreparedRound {
            plan,
            position,
            remaining,
            vocab,
            host_sampling,
            is_lookup,
            drafts,
            proposals,
            verification: _,
        } = round;
        let owned;
        let (target, status) = if let Some(values) = controls {
            values
        } else {
            owned = (
                self.read_controls(&spec.verification_tokens, plan.tokens)?,
                self.read_controls(&spec.verification_status, plan.tokens)?,
            );
            (owned.0.as_slice(), owned.1.as_slice())
        };
        if target.len() != plan.tokens || status.len() != plan.tokens {
            return Err("Verification control extent mismatch".into());
        }
        if target.iter().any(|&id| id as usize >= vocab) || status.iter().any(|&s| s != 0) {
            return Err("Invalid request verification output".into());
        }
        req.mtp.rounds += 1;
        let at = Instant::now();
        let committed = if !host_sampling {
            let mut committed = Vec::with_capacity(plan.tokens);
            let mut logits = None;
            let mut history = req.history.clone();
            for (row, &selected) in target.iter().enumerate() {
                let token = match req.decoder.greedy_candidate(selected, vocab)? {
                    Some(token) => token,
                    None => {
                        if logits.is_none() {
                            self.execution.activate_sequence(req.slot)?;
                            let raw = self.execution.download_bytes(
                                &spec.verification_logits,
                                plan.tokens * vocab * 4,
                            )?;
                            logits = Some(floats(&raw, crate::artifact::Dtype::F32));
                        }
                        req.decoder
                            .law(
                                &logits.as_ref().unwrap()[row * vocab..(row + 1) * vocab],
                                &history,
                                &req.sampling,
                            )?
                            .distribution
                            .draw(0.0)?
                    }
                };
                req.decoder.consume(token)?;
                committed.push(token);
                history.push(token);
                if drafts.get(row) != Some(&token) || req.decoder.finished() {
                    break;
                }
            }
            committed
        } else {
            self.execution.activate_sequence(req.slot)?;
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
        let accepted = if host_sampling {
            committed
                .iter()
                .zip(&proposals)
                .take_while(|(token, proposal)| **token == proposal.token)
                .count()
        } else {
            committed
                .iter()
                .zip(&drafts)
                .take_while(|(token, draft)| token == draft)
                .count()
        };
        req.mtp.accepted_draft_tokens += accepted;
        if !is_lookup {
            req.draft_acceptance.observe(drafts.len(), accepted);
        }
        if is_lookup {
            req.mtp.lookup_accepted_draft_tokens += accepted;
            req.prompt_lookup
                .observe(accepted, drafts.len(), req.history.len() + committed.len());
        }
        Ok(CommittedRound {
            tokens: committed,
            plan,
            position,
            is_lookup,
        })
    }
    fn restore_verified_cursor(
        &mut self,
        req: &mut RequestState,
        round: &CommittedRound,
        restored: bool,
    ) -> Result<()> {
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        let CommittedRound {
            tokens: committed,
            plan,
            position,
            is_lookup,
        } = round;
        let at = Instant::now();
        if !restored || spec.draft_restore_program.is_some() {
            self.execution.activate_sequence(req.slot)?;
        }
        if !restored && (spec.commit_always || committed.len() < plan.tokens) {
            self.upload_ids(&spec.accepted_inputs, &[committed.len() as u32])?;
            self.launch_rows(&plan.restore_program, plan.tokens, ExecutionPhase::Decode)?;
        }
        if restored || spec.commit_always || committed.len() < plan.tokens {
            self.execution.upload_sequence_ids(
                req.slot,
                &self.manifest.position,
                &[(*position + committed.len()) as u32],
            )?;
            self.execution.upload_sequence_ids(
                req.slot,
                &spec.target_length,
                &[(*position + committed.len()) as u32],
            )?;
        }
        req.mtp.restore_s += at.elapsed().as_secs_f64();
        req.history.extend_from_slice(committed);
        req.generated += committed.len();
        let at = Instant::now();
        if !*is_lookup && let Some(restore) = &spec.draft_restore_program {
            self.launch_program(restore, ExecutionPhase::Decode)?;
        }
        self.execution
            .upload_sequence_ids(req.slot, &spec.position, &[*position as u32])?;
        req.mtp.refresh_s += at.elapsed().as_secs_f64();
        Ok(())
    }
    fn finish_warmed_commit(
        &mut self,
        req: &mut RequestState,
        round: CommittedRound,
        position: Option<usize>,
    ) -> Result<Vec<u32>> {
        let committed = round.tokens;
        req.warm.tokens = round.position + committed.len();
        self.execution.upload_sequence_ids(
            req.slot,
            &self.manifest.token,
            &[*committed.last().expect("nonempty commit")],
        )?;
        if match position {
            Some(p) => p,
            None => self.read_control(&self.manifest.position)? as usize,
        } != req.input.len() + req.generated - 1
        {
            return Err("Private speculative position mismatch".into());
        }
        req.mtp.committed_tokens = req.generated;
        Ok(committed)
    }
    fn finish_verified_round(
        &mut self,
        req: &mut RequestState,
        round: CommittedRound,
        restored: bool,
    ) -> Result<Vec<u32>> {
        self.restore_verified_cursor(req, &round, restored)?;
        let spec = self.manifest.mtp.clone().ok_or("Missing MTP adapter")?;
        if self.manifest.packet_table().is_some() {
            let committed = (0, round);
            self.warm_committed_batch(&mut [req], std::slice::from_ref(&committed))?;
            return self.finish_warmed_commit(req, committed.1, None);
        }
        let at = Instant::now();
        self.mtp_warm(&spec, &round.tokens, &|| false, ExecutionPhase::Decode)?;
        req.mtp.refresh_s += at.elapsed().as_secs_f64();
        self.finish_warmed_commit(req, round, None)
    }
    pub(super) fn commit_verified_round(
        &mut self,
        req: &mut RequestState,
        round: PreparedRound,
    ) -> Result<Vec<u32>> {
        let committed = self.sample_verified_round(req, round, None)?;
        let deferred = self.deferred_verification() || self.manifest.packet_table().is_some();
        if deferred {
            let spec = self.manifest.mtp.as_ref().ok_or("Missing MTP adapter")?;
            self.execution.upload_sequence_ids(
                req.slot,
                &spec.accepted_inputs,
                &[committed.tokens.len() as u32],
            )?;
            self.commit_spec_packet(&[BatchSegment {
                slot: req.slot,
                tokens: committed.plan.tokens,
            }])?;
        }
        self.finish_verified_round(req, committed, deferred)
    }
}

#[cfg(test)]
#[path = "tests/mtp_graph.rs"]
mod graph_tests;

#[cfg(test)]
#[path = "tests/verification.rs"]
mod verification_tests;

#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    #[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
    fn validate_flash_warm_input_ownership() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            model: std::path::PathBuf,
            cases: Vec<GenerationInput>,
            cuda_graph: String,
        }
        let fixture: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
        ))
        .unwrap();
        let mut model = ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                mtp_drafts: Some(3),
                ..Default::default()
            },
        )
        .unwrap();
        let spec = model.manifest.mtp.clone().unwrap();
        let mut requests = Vec::new();
        let mut committed = Vec::new();
        for (i, input) in fixture.cases.into_iter().take(2).enumerate() {
            let mut req = model.start_request(input, &|| false).unwrap();
            while req.prefilling {
                model
                    .with_request(&mut req, |m, r| {
                        m.advance_prefill(r, r.input.len().min(4096), false)
                    })
                    .unwrap();
            }
            let round = model
                .with_request(&mut req, |m, r| {
                    let Preparation::Round(round) = m.prepare_speculative_round(r, 4)? else {
                        return Err("Expected a live speculative round".into());
                    };
                    m.verify_single_round(r, &round)?;
                    let committed = m.sample_verified_round(r, round, None)?;
                    m.restore_verified_cursor(r, &committed, false)?;
                    Ok(committed)
                })
                .unwrap();
            committed.push((i, round));
            requests.push(req);
        }
        assert_eq!(requests.len(), 2);
        for req in &mut requests {
            model
                .with_request(req, |m, _| {
                    // A previous request/round may leave any control value.
                    // Mapping must use this owner's newly uploaded warm length.
                    m.upload_ids("MtpLength", &[0])?;
                    m.upload_ids(&spec.input, &[0; 8])
                })
                .unwrap();
        }
        model
            .warm_committed_batch(&mut requests.iter_mut().collect::<Vec<_>>(), &committed)
            .unwrap();
        for (i, round) in &committed {
            model
                .with_request(&mut requests[*i], |m, _| {
                    assert_eq!(m.read_control("MtpLength")? as usize, round.tokens.len());
                    assert_eq!(
                        m.read_controls(&spec.input, round.tokens.len())?,
                        round.tokens
                    );
                    assert_eq!(
                        m.read_control(&spec.position)? as usize,
                        round.position + round.tokens.len()
                    );
                    Ok(())
                })
                .unwrap();
        }
        for req in &mut requests {
            model.finish_request(req, false).unwrap();
        }
    }
    #[test]
    #[ignore = "Requires real MTP model and exclusive GPU experiment lock"]
    fn validate_batched_drafts_and_commits() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            model: std::path::PathBuf,
            output: std::path::PathBuf,
            cases: Vec<GenerationInput>,
            cuda_graph: String,
        }
        let fixture: Fixture = crate::model::read(&std::path::PathBuf::from(
            std::env::var("ORINFER_BATCH_FIXTURE").unwrap(),
        ))
        .unwrap();
        assert!(!fixture.output.exists());
        let mut model = ModelRuntime::load_with_options(
            &fixture.model,
            LoadOptions {
                cuda_graph: fixture.cuda_graph.parse().unwrap(),
                prefix_cache_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let spec = model.manifest.mtp.clone().unwrap();
        assert!(spec.batch_draft_profiles.last().is_some_and(|&n| n >= 8));
        assert!(spec.batch_commit);
        assert!(spec.batch_warm);
        let options = scheduler::Options {
            target_tpot_ms: 1000.,
            memory_reserve_bytes: 0,
            ..Default::default()
        };
        let source = &fixture.cases[0];
        let input = |lane: usize| {
            let mut tokens = source.input_tokens.clone();
            tokens.extend([32, 32 + lane as u32]);
            GenerationInput {
                audios: vec![],
                input_tokens: tokens,
                images: source.images.clone(),
                max_new_tokens: 16,
                sampling: source.sampling.clone(),
                prefix_hints: vec![],
            }
        };
        let log_probs = |values: &[f32]| {
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let norm = values
                .iter()
                .map(|&x| (x as f64 - max).exp())
                .sum::<f64>()
                .ln()
                + max;
            values.iter().map(|&x| x as f64 - norm).collect::<Vec<_>>()
        };
        let mut rows = vec![];
        for count in [2, 3, 4, 8] {
            let mut baselines = vec![];
            let mut candidates = vec![];
            for lane in 0..count {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    model.advance_requests(&mut [&mut req], &options).unwrap();
                }
                // Fix the shifted token history independently of a rounded
                // draft selection; compare the whole distribution at each step.
                let mut scores = vec![];
                let mut tokens = vec![];
                for _ in 0..3 {
                    let selected = model
                        .with_request(&mut req, |m, _| {
                            let raw = m
                                .execution
                                .download_bytes(&spec.draft_logits, m.manifest.vocab * 4)?;
                            scores.push(log_probs(&floats(&raw, crate::artifact::Dtype::F32)));
                            Ok(m.read_control(&spec.token)? as u32)
                        })
                        .unwrap();
                    tokens.push(selected);
                    model
                        .with_request(&mut req, |m, _| {
                            m.prepare_program_inputs(&spec.draft_program, &[selected], &[])?;
                            m.launch_rows(&spec.draft_program, 1, ExecutionPhase::Decode)
                        })
                        .unwrap();
                }
                baselines.push(scores);
                candidates.push(tokens);
                model.finish_request(&mut req, false).unwrap();
            }
            let mut requests = vec![];
            for lane in 0..count {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    model.advance_requests(&mut [&mut req], &options).unwrap();
                }
                requests.push(req);
            }
            let mut max_kl = 0f64;
            for depth in 0..3 {
                for (lane, req) in requests.iter_mut().enumerate() {
                    let score = model
                        .with_request(req, |m, _| {
                            let raw = m
                                .execution
                                .download_bytes(&spec.draft_logits, m.manifest.vocab * 4)?;
                            Ok(log_probs(&floats(&raw, crate::artifact::Dtype::F32)))
                        })
                        .unwrap();
                    let baseline = &baselines[lane][depth];
                    let kl: f64 = baseline
                        .iter()
                        .zip(&score)
                        .map(|(b, a)| b.exp() * (b - a))
                        .sum();
                    assert!(kl < 0.05, "draft KL {count}/{lane}/{depth}: {kl}");
                    let token = candidates[lane][depth] as usize;
                    assert!(
                        (baseline[token] - score[token]).abs() < 0.1,
                        "draft selected NLL delta"
                    );
                    max_kl = max_kl.max(kl);
                    model
                        .execution
                        .ensure_sequence_program(req.slot, &spec.draft_program)
                        .unwrap();
                    model
                        .prepare_program_inputs(
                            &spec.draft_program,
                            &[candidates[lane][depth]],
                            &[],
                        )
                        .unwrap();
                }
                let segments: Vec<_> = requests
                    .iter()
                    .map(|r| BatchSegment {
                        slot: r.slot,
                        tokens: 1,
                    })
                    .collect();
                let (plan, bindings) = model
                    .model_package
                    .batch_operation(
                        &segments,
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        true,
                    )
                    .unwrap();
                model
                    .upload_spec_bindings(
                        &segments,
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        std::rc::Rc::new(bindings),
                    )
                    .unwrap();
                model
                    .execution
                    .execute_batch_operation(
                        orinfer_model_sdk::abi::BatchOperation::Draft,
                        segments.iter().map(|s| (s.slot, s.tokens)).collect(),
                        &plan,
                        true,
                        true,
                    )
                    .unwrap();
            }
            for req in &mut requests {
                model.finish_request(req, false).unwrap();
            }
            // Exercise complete native rounds with unequal histories, changing
            // membership and tail fallback, then cancellation/arena reuse.
            let mut outputs = vec![vec![]; count];
            requests.clear();
            for (lane, output) in outputs.iter_mut().enumerate() {
                let mut req = model.start_request(input(lane), &|| false).unwrap();
                while req.prefilling {
                    for step in model.advance_requests(&mut [&mut req], &options).unwrap() {
                        output.extend(step.tokens);
                    }
                }
                requests.push(req);
            }
            let at = Instant::now();
            for _ in 0..3 {
                let selected: Vec<_> = requests
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| r.limit - r.generated >= 4)
                    .map(|(i, _)| (i, 4))
                    .collect();
                assert_eq!(selected.len(), count);
                let mut active: Vec<_> = requests.iter_mut().collect();
                let mut steps = vec![];
                model
                    .speculative_batch_step(&mut active, &selected, &mut steps)
                    .unwrap();
                for step in steps {
                    outputs[step.request].extend(step.tokens);
                }
            }
            let round_s = at.elapsed().as_secs_f64();
            while requests.iter().any(|r| !r.is_finished()) {
                let mut active: Vec<_> = requests.iter_mut().collect();
                for step in model.advance_requests(&mut active, &options).unwrap() {
                    outputs[step.request].extend(step.tokens);
                }
            }
            for (lane, req) in requests.iter_mut().enumerate() {
                assert_eq!(req.generated, outputs[lane].len());
                model
                    .with_request(req, |m, r| {
                        assert_eq!(
                            m.read_control(&m.manifest.position)? as usize,
                            r.input.len() + r.generated - 1
                        );
                        Ok(())
                    })
                    .unwrap();
                model.finish_request(req, false).unwrap();
            }
            model.execution.reclaim_idle_state().unwrap();
            assert_eq!(model.execution.resident_kv_bytes(), 0);
            rows.push(serde_json::json!({"batch":count,"draft_max_kl":max_kl,"three_rounds_s":round_s,"outputs":outputs,"position_and_cleanup_passed":true}));
            std::fs::write(&fixture.output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
        }
        let mut peers = vec![];
        for lane in 0..2 {
            let mut req = model.start_request(input(lane), &|| false).unwrap();
            while req.prefilling {
                model.advance_requests(&mut [&mut req], &options).unwrap();
            }
            peers.push(req);
        }
        peers[0]
            .decoder
            .fail("Injected request constraint failure".into());
        let before = peers[1].generated;
        let mut output = vec![];
        model
            .speculative_batch_step(
                &mut peers.iter_mut().collect::<Vec<_>>(),
                &[(0, 4), (1, 4)],
                &mut output,
            )
            .unwrap();
        assert!(output.iter().any(|o| o.request == 0 && o.tokens.is_empty()));
        assert!(peers[1].failure().is_none() && peers[1].generated > before);
        for req in &mut peers {
            model.finish_request(req, false).unwrap();
        }
    }
}
