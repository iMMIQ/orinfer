//! Shared native packet selection for ordinary and speculative target rows.
use super::*;

pub(super) fn readout(requests: &[&mut Request], full: bool) -> super::super::team::Readout {
    super::super::team::Readout {
        full,
        continuations: requests
            .iter()
            .filter(|r| r.device_sample() && !r.decoder.constrained())
            .map(|r| r.slot)
            .collect(),
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
            Ok(jobs
                .into_par_iter()
                .map(|(slot, decoder, top_logprobs, tokens)| {
                    let count = tokens.len() + 1;
                    let rows = match decoder.proposed_rows(&tokens, vocab, top_logprobs) {
                        Ok(rows) => rows,
                        Err(error) => vec![
                            crate::backend::ascend::sampling::Parameters {
                                mask: None,
                                top_logprobs: None,
                                error: Some(error),
                            };
                            count
                        ],
                    };
                    (slot, rows)
                })
                .collect::<BTreeMap<_, _>>())
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

impl Selection {
    pub(super) fn verification_budget(
        &self,
        requests: &[&mut Request],
        decode_rows: &[usize],
        keep: usize,
        estimator: &prefill::Estimator<'_>,
        ceiling_s: f64,
    ) -> usize {
        if self.prefill_rows == 0 {
            return self.budget;
        }
        mixed_budget(
            requests,
            &self.order,
            Some(decode_rows),
            keep,
            estimator.capacity,
            ceiling_s,
            |selected| estimator.mixed_price(estimator.work(selected, requests, self.full, false)),
        )
        .min(self.budget)
    }
    pub(super) fn forecast_layout(
        &self,
        requests: &[&mut Request],
        decode_rows: &[usize],
        budget: usize,
        keep: usize,
        capacity: usize,
        owners: usize,
    ) -> Result<Layout> {
        let (_, selected) = select_rows(
            requests,
            &self.order,
            Some(decode_rows),
            budget,
            keep,
            capacity,
        );
        packet_layout(
            requests,
            &selected,
            true,
            self.prefill_rows > 0,
            capacity,
            owners,
        )
    }
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
pub(super) fn request_order(
    requests: &[&mut Request],
    decode_only: bool,
    cost: impl Fn(&Request) -> f64,
) -> Vec<usize> {
    // Freeze ages before sorting: the same policy ranks waiting admissions and
    // active prompt suffixes. Progress resets age, so long prompts can resume.
    let now = Instant::now();
    order_snapshot(requests, decode_only, |_, r| {
        Some(scheduler::Waiting {
            age_s: now.saturating_duration_since(r.last_prefill).as_secs_f64(),
            remaining_s: cost(r),
            restore_s: 0.,
            startup_s: 0.,
            remaining_tokens: r.remaining_prefill_tokens(),
        })
    })
}
fn order_snapshot(
    requests: &[&mut Request],
    decode_only: bool,
    waiting: impl Fn(usize, &Request) -> Option<scheduler::Waiting>,
) -> Vec<usize> {
    let mut order = requests
        .iter()
        .enumerate()
        .filter(|(_, r)| !decode_only || !r.prefilling)
        .filter(|(_, r)| !r.is_finished() && r.ready_output.is_none())
        .filter_map(|(i, r)| waiting(i, r).map(|w| (i, r.prefilling, w)))
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

// Stable scheduling capacities reuse Graph and DSpark calibration; actual
// tails and owner segments still use the same arbitrary-row packet.
fn windows(capacity: usize) -> impl Iterator<Item = usize> {
    (7..usize::BITS)
        .map(|bits| 1usize << bits)
        .take_while(move |&rows| rows < capacity)
        .chain([capacity])
}

/// Price the same selected rows that will execute, including speculative
/// verification prefixes. Native compute capacity and final windows stay intact.
fn mixed_budget(
    requests: &[&mut Request],
    order: &[usize],
    decode_rows: Option<&[usize]>,
    keep: usize,
    capacity: usize,
    ceiling_s: f64,
    price: impl Fn(&[(usize, usize)]) -> f64,
) -> usize {
    let decoders = order
        .iter()
        .filter(|&&i| !requests[i].prefilling)
        .map(|&i| decode_rows.map_or(1, |rows| rows[i]))
        .sum::<usize>();
    let prompt = order
        .iter()
        .filter(|&&i| requests[i].prefilling)
        .map(|&i| (requests[i].input.len() - requests[i].offset).min(keep.max(128)))
        .min()
        .unwrap_or(1);
    let minimum = (decoders + prompt).min(capacity);
    let mut budget = minimum;
    for n in windows(capacity) {
        if n < minimum {
            continue;
        }
        let (_, selected) = select_rows(requests, order, decode_rows, n, keep, capacity);
        if price(&selected) <= ceiling_s {
            budget = n;
        }
    }
    budget
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

/// Canonical verification row buckets (max gap 15 <= DRAFTS + 1 = 17 so device
/// coordinates remain active and all buckets <= 127 stay inside non-SP P[2]).
const VERIFY_BUCKETS: &[usize] = &[16, 32, 48, 64, 80, 96, 112, 127];

/// Canonical Mixed decoder-row buckets (max gap 33 <= 64 <= keep = 128; buckets
/// <= 127 trigger cross-family Tail downgrade from SP P[3] to non-SP P[2]).
const MIXED_DECODER_BUCKETS: &[usize] = &[32, 64, 96, 127, 160, 192, 224, 256];

pub(super) fn bucket_verify_rows(rows: usize) -> usize {
    VERIFY_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= rows)
        .unwrap_or(rows)
}

pub(super) fn bucket_mixed_decoder_rows(rows: usize) -> usize {
    MIXED_DECODER_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= rows)
        .unwrap_or(rows)
}

impl Runtime {
    /// Snap variable verification and mixed Tail/Head dimensions to a small
    /// canonical bucket ladder via a reusable scratch pad owner so ACL Graphs
    /// stay resident across changing batch compositions.
    pub(super) fn bucket_selection(
        &mut self,
        selection: &mut Selection,
        _requests: &[&mut Request],
    ) -> Result<()> {
        if selection.layout.decode
            || selection.layout.role != super::super::PacketRole::Target
            || selection.layout.pad_slot.is_some()
        {
            return Ok(());
        }
        let Some(contract) = self.team.model.batch.as_ref() else {
            return Ok(());
        };
        let owners = contract.owners;
        let output_capacity = contract.output_capacity();
        if owners < 2 || selection.layout.segments.len() >= owners {
            return Ok(());
        }
        let capacity = self.team.rows;
        if selection.prefill_rows == 0 {
            let actual = selection.layout.rows.total;
            if actual == 0 || actual > 127 {
                return Ok(());
            }
            let target = bucket_verify_rows(actual).min(capacity);
            let pad_len = target.saturating_sub(actual);
            selection.layout.enable_bucketing();
            if pad_len > 0
                && selection.layout.outputs.len() + pad_len <= output_capacity
                && let Some(slot) = self.ensure_pad_slot()?
            {
                selection
                    .layout
                    .append_pad(slot, pad_len, pad_len, None, capacity, owners)?;
                selection.tokens.push(vec![0u32; pad_len]);
            }
            return Ok(());
        }
        let keep = self.final_decoder_window();
        if keep == 0 {
            return Ok(());
        }
        let decoder_rows: usize = selection
            .layout
            .segments
            .iter()
            .zip(&selection.layout.decoder_roles)
            .map(|(s, role)| {
                if *role == super::super::replay::Role::Producer {
                    0
                } else {
                    s.tokens.min(keep)
                }
            })
            .sum();
        if decoder_rows == 0 {
            if selection.layout.rows.total == capacity || selection.layout.rows.total == 1024 {
                selection.layout.enable_bucketing();
            }
            return Ok(());
        }
        if decoder_rows > 256 {
            return Ok(());
        }
        let target_dec = bucket_mixed_decoder_rows(decoder_rows);
        let pad_len = target_dec.saturating_sub(decoder_rows);
        let cur_out = selection.layout.outputs.len();
        let target_out = bucket_verify_rows(cur_out)
            .min(target_dec)
            .min(output_capacity);
        let pad_out = target_out.saturating_sub(cur_out);
        selection.layout.enable_bucketing();
        if pad_len == 0 || pad_len > 64 {
            return Ok(());
        }
        let at_budget = selection.layout.rows.total == selection.budget
            || selection.layout.rows.total + pad_len > capacity;
        let shrink_producer = if at_budget {
            selection
                .layout
                .segments
                .iter()
                .zip(&selection.layout.decoder_roles)
                .enumerate()
                .rev()
                .find_map(|(lane, (seg, role))| {
                    (*role == super::super::replay::Role::Producer && seg.tokens > pad_len)
                        .then_some(lane)
                })
        } else {
            None
        };
        if shrink_producer.is_none() && selection.layout.rows.total + pad_len > capacity {
            return Ok(());
        }
        if let Some(slot) = self.ensure_pad_slot()? {
            if let Some(p_lane) = shrink_producer {
                selection.selected[p_lane].1 -= pad_len;
                selection.tokens[p_lane].truncate(selection.selected[p_lane].1);
                selection.prefill_rows -= pad_len;
            }
            selection.layout.append_pad(
                slot,
                pad_len,
                pad_out,
                shrink_producer,
                capacity,
                owners,
            )?;
            selection.tokens.push(vec![0u32; pad_len]);
        }
        Ok(())
    }
    pub(super) fn prepare_host_inputs(
        &mut self,
        layout: &Layout,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
        readout: &mut super::super::team::Readout,
    ) {
        let has_prefill = selected.iter().any(|&(i, _)| requests[i].prefilling);
        let hint = if has_prefill && layout.rows.total >= 128 {
            let previous = self.input_window.take();
            previous.as_ref().and_then(|window| {
                selected.iter().enumerate().find_map(|(lane, &(i, rows))| {
                    let r = &requests[i];
                    let start = r.offset.checked_sub(window.offset)?;
                    (r.prefilling && r.slot == window.owner && start + rows <= window.rows).then(
                        || super::super::host_inputs::Hint {
                            window: Arc::clone(window),
                            destination: layout.rows.starts[lane]..layout.rows.starts[lane] + rows,
                            source: start..start + rows,
                        },
                    )
                })
            })
        } else {
            None
        };
        readout.first_input = self.team.first_input(hint);
        if !has_prefill || layout.rows.total < 256 {
            return;
        }
        let mut offsets = requests.iter().map(|r| r.offset).collect::<Vec<_>>();
        for &(i, n) in selected {
            if requests[i].prefilling {
                offsets[i] += n;
            }
        }
        let now = Instant::now();
        let order = order_snapshot(requests, false, |i, r| {
            let offset = offsets[i];
            if !r.prefilling
                || offset >= r.input.len()
                || r.pending_images
                    .front()
                    .is_some_and(|image| image.start <= offset)
            {
                return None;
            }
            Some(scheduler::Waiting {
                age_s: if offset == r.offset {
                    now.saturating_duration_since(r.last_prefill).as_secs_f64()
                } else {
                    0.
                },
                remaining_s: self.remaining_prefill_cost(offset, r.input.len()),
                restore_s: 0.,
                startup_s: 0.,
                remaining_tokens: r.input.len() - offset,
            })
        });
        if let Some(&i) = order.first() {
            self.input_window =
                self.team
                    .forecast_input(requests[i].slot, offsets[i], &requests[i].input);
            readout.next_input = self.input_window.clone();
        }
    }
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
        let decoders = requests
            .iter()
            .filter(|r| !r.is_finished() && r.ready_output.is_none() && !r.prefilling)
            .count();
        let has_decode = decoders > 0;
        let decode_round = has_prefill
            && has_decode
            && options.policy == scheduler::Policy::Latency
            && self.decode_burst.allows(Instant::now(), decoders);
        let has_prefill = has_prefill && !decode_round;
        let order = request_order(requests, decode_round, |r| {
            self.remaining_prefill_cost(r.offset, r.input.len())
        });
        let budget = if has_prefill {
            if has_decode {
                mixed_budget(
                    requests,
                    &order,
                    None,
                    self.final_decoder_window(),
                    self.team.rows,
                    options.prefill_budget_ms / 1000.,
                    |selected| {
                        let full = self.packet_full(selected, requests);
                        let work = self.prefill_work(selected, requests, full, false);
                        self.prefill_estimator().mixed_price(work)
                    },
                )
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
        self.packet_for_budget(requests, decode_inputs, budget, has_prefill, &order)
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
        let mut tokens = vec![];
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
            tokens.push(values);
        }
        if selected.is_empty() {
            return Ok(None);
        }
        let full = self.packet_full(&selected, requests);
        let batch = selected.len();
        let prefill_rows = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .map(|&(_, count)| count)
            .sum::<usize>();
        let prefill_requests = selected
            .iter()
            .filter(|&&(i, _)| requests[i].prefilling)
            .count();
        let layout = packet_layout(
            requests,
            &selected,
            decode_inputs.is_some(),
            has_prefill,
            self.team.rows,
            self.team.model.batch.as_ref().unwrap().owners,
        )?;
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
    pub(super) fn packet_full(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
    ) -> bool {
        let complete_cache = self.complete_prompt_cache();
        selected.iter().any(|&(i, count)| {
            let r = &requests[i];
            r.needs_logits()
                || complete_cache
                    && r.prefilling
                    && r.cache_prompt
                    && r.offset + count == r.input.len()
        })
    }
}

/// Forecast native geometry using the same membership and row binding as execution.
fn packet_layout(
    requests: &[&mut Request],
    selected: &[(usize, usize)],
    verification: bool,
    has_prefill: bool,
    capacity: usize,
    owners: usize,
) -> Result<Layout> {
    let segments = selected
        .iter()
        .map(|&(i, tokens)| BatchSegment {
            slot: requests[i].slot,
            tokens,
        })
        .collect();
    let outputs = selected
        .iter()
        .enumerate()
        .filter_map(|(lane, &(i, count))| {
            let r = &requests[i];
            (!r.prefilling || r.offset + count == r.input.len()).then_some(lane)
        })
        .collect();
    let mut layout = Layout::new(
        segments,
        outputs,
        !has_prefill && !verification,
        if has_prefill || verification {
            capacity
        } else {
            owners
        },
        owners,
    )?;
    for (lane, &(i, _)) in selected.iter().enumerate() {
        if !requests[i].prefilling {
            layout.decoder_roles[lane] = super::super::replay::Role::Continue;
        }
    }
    if verification {
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
    Ok(layout)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn decode_turn_keeps_owner_indices_and_verification_cannot_reintroduce_prefill() {
        let mut owners = [
            request(false, 100000, 0),
            request(true, 4096, 30000),
            request(false, 80000, 0),
        ];
        let requests = owners.iter_mut().collect::<Vec<_>>();
        let order = request_order(&requests, true, |r| {
            assert!(!r.prefilling);
            0.
        });
        assert_eq!(order.len(), 2);
        assert!(order.contains(&0) && order.contains(&2));
        let predicted = forecast(&requests, &order, &[6, 0, 6], 32, 128, 2048);
        assert_eq!(predicted.prefill_rows, 0);
        assert_eq!(predicted.rows, 12);
        assert_eq!(requests[1].offset, 4096);
    }
    #[test]
    fn expanded_verification_is_priced_before_admitting_prompt_rows() {
        let mut owners = (0..16)
            .map(|_| request(false, 100000, 0))
            .collect::<Vec<_>>();
        owners.push(request(true, 4096, 20000));
        let requests = owners.iter_mut().collect::<Vec<_>>();
        let order = (0..17).rev().collect::<Vec<_>>();
        let price = |selected: &[(usize, usize)]| {
            selected
                .iter()
                .map(|&(i, n)| {
                    n as f64
                        * if requests[i].prefilling {
                            0.00006
                        } else {
                            0.001
                        }
                })
                .sum::<f64>()
        };
        let ordinary = mixed_budget(&requests, &order, None, 128, 2048, 0.2, price);
        let mut rows = vec![6; 17];
        rows[16] = 0;
        let verified = mixed_budget(&requests, &order, Some(&rows), 128, 2048, 0.2, price);
        assert_eq!(ordinary, 2048);
        assert_eq!(verified, 1024);
        let (_, selected) = select_rows(&requests, &order, Some(&rows), verified, 128, 2048);
        assert!(price(&selected) <= 0.2);
        let forecast = forecast(&requests, &order, &rows, verified, 128, 2048);
        assert_eq!((forecast.rows, forecast.prefill_rows), (1024, 928));
        assert_eq!(requests[16].offset, 4096);
    }
    #[test]
    fn scheduling_windows_keep_stable_tiers_and_the_native_capacity_tail() {
        assert_eq!(
            windows(2048).collect::<Vec<_>>(),
            [128, 256, 512, 1024, 2048]
        );
        assert_eq!(
            windows(1792).collect::<Vec<_>>(),
            [128, 256, 512, 1024, 1792]
        );
        assert_eq!(windows(64).collect::<Vec<_>>(), [64]);
    }
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
    fn device_continuation_requires_generated_device_sampled_input() {
        let mut owners = [
            request(false, 128, 0),
            request(true, 0, 128),
            request(false, 128, 0),
        ];
        for (slot, request) in owners.iter_mut().enumerate() {
            request.slot = slot;
            request.device_sampling = true;
        }
        owners[2].device_sampling = false;
        let requests = owners.iter_mut().collect::<Vec<_>>();
        let result = readout(&requests, false);
        assert_eq!(result.continuations.into_iter().collect::<Vec<_>>(), [0]);
        assert!(
            super::super::super::team::Readout::default()
                .continuations
                .is_empty()
        );
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
    #[test]
    fn bucket_ladders_bound_padding_gaps_and_stay_below_sp_threshold() {
        for r in 1..=127 {
            let b = bucket_verify_rows(r);
            assert!(b >= r && b <= 127);
            assert!(b - r <= 16, "verify gap {b}-{r} exceeds device coord limit");
        }
        for r in 1..=256 {
            let b = bucket_mixed_decoder_rows(r);
            assert!(b >= r && b <= 256);
            assert!(b - r <= 64, "mixed gap {b}-{r} exceeds pad token budget");
            if r <= 127 {
                assert!(
                    b <= 127,
                    "mixed decoder_rows {r} spilled into SP bucket {b}"
                );
            }
        }
    }
}
