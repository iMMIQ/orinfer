//! Shared native packet selection for ordinary and speculative target rows.
use super::*;

pub(super) fn readout(requests: &[&mut Request], full: bool) -> super::super::team::Readout {
    super::super::team::Readout {
        full,
        full_owners: Some(
            requests
                .iter()
                .filter(|r| r.prefilling || r.needs_logits())
                .map(|r| r.slot)
                .collect(),
        ),
        draws: requests
            .iter()
            .filter(|r| r.device_sample())
            .map(|r| {
                (
                    r.slot,
                    crate::backend::ascend::sampling::Draw {
                        seed: r.sampling.seed,
                        step: r.generated,
                        temperature: r.sampling.temperature,
                    },
                )
            })
            .collect(),
        parameters: Default::default(),
        ..Default::default()
    }
}

pub(super) fn target_readout(
    requests: &[&mut Request],
    full: bool,
    proposals: &BTreeMap<usize, Vec<u32>>,
    vocab: usize,
    selected: &[(usize, usize)],
) -> Result<super::super::team::Readout> {
    use rayon::prelude::*;
    let mut result = readout(requests, full);
    if vocab == 0 {
        return Err("Missing sampling vocabulary".into());
    }
    let jobs = requests
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            selected.iter().any(|(index, _)| index == i)
                && r.device_sample()
                && r.processed_sampling
                && (r.sampling.top_logprobs.is_some() || r.decoder.has_constraint())
        })
        .map(|(i, r)| {
            (
                r.slot,
                r.decoder.fork(),
                r.sampling.top_logprobs,
                proposals.get(&i).cloned().unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();
    if !jobs.is_empty() {
        result.parameters = Arc::new(crate::sampling::Prepared::new(move || {
            jobs.into_par_iter()
                .map(|(slot, decoder, top_logprobs, tokens)| {
                    Ok((slot, decoder.proposed_rows(&tokens, vocab, top_logprobs)?))
                })
                .collect::<Result<BTreeMap<_, _>>>()
        }));
    }

    Ok(result)
}

pub(super) fn draft_readout(requests: &[&mut Request]) -> super::super::team::Readout {
    let mut result = readout(requests, false);
    result.draws.retain(|owner, _| {
        requests
            .iter()
            .any(|r| r.slot == *owner && !r.decoder.constrained())
    });
    result
}

pub(super) struct Selection {
    pub budget: usize,
    pub order: Vec<usize>,
    pub layout: Layout,
    pub tokens: Vec<Vec<u32>>,
    pub selected: Vec<(usize, usize)>,
    pub full: bool,
    pub batch: usize,
    pub prefill_rows: usize,
    pub prefill_requests: usize,
}

/// Shared scheduling membership survives a cohort's tail becoming single-stream.
pub(super) fn mark_shared(selected: &[(usize, usize)], requests: &mut [&mut Request]) -> bool {
    if selected.len() > 1 {
        for &(i, _) in selected {
            requests[i].shared_work = true;
        }
    }
    selected.iter().any(|&(i, _)| requests[i].shared_work)
}

pub(super) fn work(
    selected: &[(usize, usize)],
    requests: &[&mut Request],
    prefill_rows: usize,
) -> dspark::Work {
    dspark::Work {
        rows: selected.iter().map(|&(_, count)| count).sum(),
        prefill_rows,
        context: selected
            .iter()
            .map(|&(i, _)| {
                if requests[i].prefilling {
                    requests[i].offset
                } else {
                    requests[i].history.len()
                }
            })
            .max()
            .unwrap_or(0),
    }
}
pub(super) fn request_order(requests: &[&mut Request], tokens_per_s: f64) -> Vec<usize> {
    // Freeze ages before sorting: the same policy ranks waiting admissions and
    // active prompt suffixes. Progress resets age, so long prompts can resume.
    let now = Instant::now();
    let mut order = requests
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_finished() && r.ready_output.is_none())
        .map(|(i, r)| {
            let remaining = r.remaining_prefill_tokens();
            (
                i,
                r.prefilling,
                scheduler::Waiting {
                    age_s: now.saturating_duration_since(r.last_prefill).as_secs_f64(),
                    remaining_s: remaining as f64 / tokens_per_s,
                    restore_s: 0.,
                    startup_s: 0.,
                    remaining_tokens: remaining,
                },
            )
        })
        .collect::<Vec<_>>();
    order.sort_unstable_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| {
                if a.1 {
                    scheduler::waiting_order(&a.2, &b.2)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then(a.0.cmp(&b.0))
    });
    order.into_iter().map(|(i, _, _)| i).collect()
}
/// The ordinary packet, verification forecast and real verification use the
/// same admissions, atomic final windows, checkpoint and media boundaries.
fn select_rows(
    requests: &[&mut Request],
    order: &[usize],
    decode_rows: Option<&[usize]>,
    mut budget: usize,
    keep: usize,
    capacity: usize,
) -> (usize, Vec<(usize, usize)>) {
    if let Some(rows) = decode_rows {
        budget = budget.max(rows.iter().sum()).min(capacity);
    }
    let packet_budget = budget;
    let mut selected = vec![];
    // Existing decode rows precede bounded prefill work in a shared packet.
    for prefill in [false, true] {
        for &i in order {
            let request = &requests[i];
            if budget == 0
                || request.is_finished()
                || request.ready_output.is_some()
                || request.prefilling != prefill
                || !prefill && decode_rows.is_some_and(|v| v[i] == 0)
            {
                continue;
            }
            let count = if prefill {
                let next = 4096 - request.offset % 4096;
                let left = request.input.len() - request.offset;
                // A soft latency budget may expand for the first atomic
                // final window. Other owners defer until the next packet.
                let limit = if keep > 0 && left <= keep && selected.is_empty() {
                    budget.max(left)
                } else {
                    budget
                };
                let limit = if keep > 0 && left <= keep {
                    limit
                } else {
                    limit.min(next)
                };
                // Ready text/image rows can run while a later image is
                // still encoding. Never cross an unavailable image span.
                let limit = request.pending_images.front().map_or(limit, |image| {
                    limit.min(image.start.saturating_sub(request.offset))
                });
                let Some(count) = super::super::replay::prompt_rows(
                    left,
                    limit,
                    keep,
                    request.input.len() > capacity,
                ) else {
                    continue;
                };
                count
            } else {
                decode_rows.map_or(1, |v| v[i])
            };
            selected.push((i, count));
            budget = budget.saturating_sub(count);
        }
    }
    (packet_budget, selected)
}

pub(super) fn forecast(
    requests: &[&mut Request],
    order: &[usize],
    decode_rows: &[usize],
    budget: usize,
    keep: usize,
    capacity: usize,
) -> dspark::Work {
    let (_, selected) = select_rows(requests, order, Some(decode_rows), budget, keep, capacity);
    let prefill_rows = selected
        .iter()
        .filter(|&&(i, _)| requests[i].prefilling)
        .map(|&(_, n)| n)
        .sum();
    work(&selected, requests, prefill_rows)
}

impl Runtime {
    pub(super) fn capture_context(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
        depths: Option<&BTreeMap<usize, usize>>,
        block: usize,
        options: &scheduler::Options,
    ) -> (f64, f64) {
        let now = Instant::now();
        let mut uses = f64::INFINITY;
        let mut slack = f64::INFINITY;
        for &(i, count) in selected {
            let r = &requests[i];
            if r.prefilling {
                uses = uses.min((r.input.len() - r.offset) as f64 / count.max(1) as f64);
            } else {
                let depth = depths.and_then(|d| d.get(&i)).copied().unwrap_or(0);
                let guard = if depth == 0 {
                    0
                } else if self.draft_limit <= self.team.draft_blocks[0] {
                    self.draft_limit
                } else {
                    depth
                };
                let remaining = r.limit.saturating_sub(r.generated + guard).max(1);
                let progress = r
                    .draft_feedback
                    .progress(depth, block)
                    .min(r.limit.saturating_sub(r.generated).max(1) as f64);
                uses = uses.min(remaining as f64 / progress);
                if options.policy == scheduler::Policy::Latency {
                    slack = slack.min(
                        super::super::team::capture_interval(
                            options.target_tpot_ms / 1000.,
                            progress,
                        ) - now.duration_since(r.last_output).as_secs_f64(),
                    );
                }
            }
        }
        // Output limits are ceilings. Keep a margin for EOS and role changes.
        (
            (uses * 0.8).max(1.),
            (slack + self.capture_credit_s).max(0.),
        )
    }

    pub(super) fn select_packet(
        &self,
        requests: &[&mut Request],
        options: &scheduler::Options,
        decode_inputs: Option<&BTreeMap<usize, Vec<u32>>>,
    ) -> Result<Option<Selection>> {
        let has_prefill = requests
            .iter()
            .any(|r| !r.is_finished() && r.ready_output.is_none() && r.prefilling);
        let has_decode = requests
            .iter()
            .any(|r| !r.is_finished() && r.ready_output.is_none() && !r.prefilling);
        let budget = if has_prefill {
            if has_decode {
                ((options.prefill_budget_ms / 1000. * self.prefill_tokens_per_s) as usize)
                    .clamp(1, self.team.rows)
            } else {
                // With no decoder waiting for its next token, fill the same
                // native packet instead of buying another weight/collective
                // pass from a latency-oriented throughput prior. Per-request
                // checkpoint boundaries below still bound each segment.
                self.team.rows
            }
        } else {
            self.team
                .model
                .batch
                .as_ref()
                .unwrap()
                .owners
                .min(options.max_batch_tokens)
        };
        self.packet_for_budget(
            requests,
            decode_inputs,
            budget,
            has_prefill,
            &request_order(requests, self.prefill_tokens_per_s),
        )
    }

    pub(super) fn packet_for_budget(
        &self,
        requests: &[&mut Request],
        decode_inputs: Option<&BTreeMap<usize, Vec<u32>>>,
        budget: usize,
        has_prefill: bool,
        order: &[usize],
    ) -> Result<Option<Selection>> {
        let decode_rows = decode_inputs.map(|inputs| {
            (0..requests.len())
                .map(|i| inputs.get(&i).map_or(0, Vec::len))
                .collect::<Vec<_>>()
        });
        let (packet_budget, selected) = select_rows(
            requests,
            order,
            decode_rows.as_deref(),
            budget,
            self.final_decoder_window(),
            self.team.rows,
        );
        let mut segments = vec![];
        let mut tokens = vec![];
        let mut outputs = vec![];
        for &(i, count) in &selected {
            let request = &requests[i];
            let values = if request.prefilling {
                request.input[request.offset..request.offset + count].to_vec()
            } else {
                match decode_inputs {
                    Some(values) => values[&i].clone(),
                    None => vec![*request.history.last().ok_or("Missing A2 decode input")?],
                }
            };
            if !request.prefilling || request.offset + count == request.input.len() {
                outputs.push(segments.len());
            }
            segments.push(BatchSegment {
                slot: request.slot,
                tokens: count,
            });
            tokens.push(values);
        }
        if segments.is_empty() {
            return Ok(None);
        }
        let complete_cache = self.complete_prompt_cache();
        let full = selected.iter().any(|&(i, count)| {
            let r = &requests[i];
            r.needs_logits()
                || complete_cache
                    && r.prefilling
                    && r.cache_prompt
                    && r.offset + count == r.input.len()
        });
        let batch = segments.len();
        let prefill_rows = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .map(|&(_, count)| count)
            .sum::<usize>();
        let prefill_requests = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .count();
        let mut layout = Layout::new(
            segments,
            outputs,
            !has_prefill && decode_inputs.is_none(),
            if has_prefill || decode_inputs.is_some() {
                self.team.rows
            } else {
                self.team.model.batch.as_ref().unwrap().owners
            },
            self.team.model.batch.as_ref().unwrap().owners,
        )?;
        for (lane, &(i, _)) in selected.iter().enumerate() {
            if !requests[i].prefilling {
                layout.decoder_roles[lane] = super::super::replay::Role::Continue;
            }
        }
        if decode_inputs.is_some() {
            layout.outputs.clear();
            layout.output_positions.clear();
            for (lane, &(i, count)) in selected.iter().enumerate() {
                if !requests[i].prefilling {
                    for position in 0..count {
                        layout.outputs.push(lane);
                        layout.output_positions.push(position);
                    }
                } else if requests[i].offset + count == requests[i].input.len() {
                    layout.outputs.push(lane);
                    layout.output_positions.push(count - 1);
                }
            }
        }
        Ok(Some(Selection {
            budget: packet_budget,
            order: order.to_vec(),
            layout,
            tokens,
            selected,
            full,
            batch,
            prefill_rows,
            prefill_requests,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request(prefill: bool, offset: usize, left: usize) -> Request {
        let mut r = super::super::tests::request(Arc::new(Output::default()), Default::default());
        r.ready_output = None;
        r.prefilling = prefill;
        r.offset = offset;
        r.input = vec![1; offset + left];
        r.history = vec![1; offset.max(1)];
        r
    }
    #[test]
    fn verification_uses_free_packet_capacity_without_displacing_final_prompt() {
        let mut owners = (0..16)
            .map(|_| request(false, 100_000, 0))
            .collect::<Vec<_>>();
        owners.push(request(true, 4096, 128));
        let requests = owners.iter_mut().collect::<Vec<_>>();
        let order = (0..17).collect::<Vec<_>>();
        let (_, ordinary) = select_rows(&requests, &order, None, 800, 128, 2048);
        assert_eq!(work(&ordinary, &requests, 128).rows, 144);
        let mut counts = vec![6; 17];
        counts[16] = 0;
        let actual = forecast(&requests, &order, &counts, 800, 128, 2048);
        assert_eq!((actual.rows, actual.prefill_rows), (224, 128));
        // A final SWA window is atomic. An overfull forecast must defer the
        // entire window rather than invent a shorter final decoder prefix.
        let deferred = forecast(&requests, &order, &counts, 160, 128, 2048);
        assert_eq!((deferred.rows, deferred.prefill_rows), (96, 0));
    }
    #[test]
    fn forecast_preserves_checkpoint_media_and_unselected_owner_boundaries() {
        let mut owners = [
            request(false, 100_000, 0),
            request(false, 200_000, 0),
            request(true, 4000, 1000),
        ];
        let mut requests = owners.iter_mut().collect::<Vec<_>>();
        let order = [1, 0, 2];
        let counts = [6, 0, 0];
        let first = forecast(&requests, &order, &counts, 800, 128, 2048);
        assert_eq!(
            (first.rows, first.prefill_rows, first.context),
            (102, 96, 100_000)
        );
        requests[2]
            .pending_images
            .push_back(super::super::vision::Pending {
                start: 4032,
                identity: [7; 32],
                image: None,
            });
        let image = forecast(&requests, &order, &counts, 800, 128, 2048);
        assert_eq!((image.rows, image.prefill_rows), (38, 32));
        requests[2].pending_images.front_mut().unwrap().start = 4000;
        let blocked = forecast(&requests, &order, &counts, 800, 128, 2048);
        assert_eq!((blocked.rows, blocked.prefill_rows), (6, 0));
    }
    #[test]
    fn forecast_keeps_long_prompt_tail_reservations_and_decoder_budget_expansion() {
        let mut owners = [request(false, 100_000, 0), request(true, 0, 2049)];
        let requests = owners.iter_mut().collect::<Vec<_>>();
        let order = [0, 1];
        let partial = forecast(&requests, &order, &[6, 0], 2048, 128, 2048);
        assert_eq!((partial.rows, partial.prefill_rows), (1927, 1921));
        let expanded = forecast(&requests, &order, &[6, 0], 4, 128, 2048);
        assert_eq!((expanded.rows, expanded.prefill_rows), (6, 0));
    }
}
