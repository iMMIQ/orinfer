//! Shared native packet selection for ordinary and speculative target rows.
use super::*;

pub(super) struct Selection {
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
        let mut budget = if has_prefill {
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
        if let Some(inputs) = decode_inputs {
            // Reserve verified decoder rows before admitting prefill work.
            let reserved = inputs.values().map(Vec::len).sum::<usize>();
            budget = budget.max(reserved).min(self.team.rows);
        }
        let mut segments = vec![];
        let mut tokens = vec![];
        let mut outputs = vec![];
        let mut selected = vec![];
        let keep = self.final_decoder_window();
        let order = request_order(requests, self.prefill_tokens_per_s);
        // Existing decode rows precede bounded prefill work in a shared packet.
        for prefill in [false, true] {
            for &i in &order {
                let request = &requests[i];
                if budget == 0
                    || request.is_finished()
                    || request.ready_output.is_some()
                    || !request.pending_images.is_empty()
                    || request.prefilling != prefill
                    || !prefill && decode_inputs.is_some_and(|v| !v.contains_key(&i))
                {
                    continue;
                }
                let count = if prefill {
                    let next = 4096 - request.offset % 4096;
                    let left = request.input.len() - request.offset;
                    // A soft latency budget may expand for the first atomic
                    // final window. Other owners defer until the next packet.
                    let limit = if keep > 0 && left <= keep && segments.is_empty() {
                        budget.max(left)
                    } else {
                        budget
                    };
                    let limit = if keep > 0 && left <= keep {
                        limit
                    } else {
                        limit.min(next)
                    };
                    let Some(count) = super::super::replay::prompt_rows(
                        left,
                        limit,
                        keep,
                        request.input.len() > self.team.rows,
                    ) else {
                        continue;
                    };
                    count
                } else {
                    decode_inputs.map_or(1, |v| v[&i].len())
                };
                let values = if prefill {
                    request.input[request.offset..request.offset + count].to_vec()
                } else {
                    match decode_inputs {
                        Some(values) => values[&i].clone(),
                        None => vec![*request.history.last().ok_or("Missing A2 decode input")?],
                    }
                };
                if !prefill || request.offset + count == request.input.len() {
                    outputs.push(segments.len());
                }
                segments.push(BatchSegment {
                    slot: request.slot,
                    tokens: count,
                });
                tokens.push(values);
                selected.push((i, count));
                budget = budget.saturating_sub(count);
            }
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
