//! Native geometry adapter for the shared, measured packet cost curves.
use super::*;
use scheduler::prefill::{Path, Prediction, Work};

pub(super) struct Estimator<'a> {
    pub model: &'a super::super::Model,
    pub domains: &'a [(bool, usize, usize)],
    pub costs: &'a scheduler::prefill::Costs,
    pub capacity: usize,
}
impl Estimator<'_> {
    fn cost_domain(&self, rows: usize, decode: bool) -> usize {
        self.domains
            .iter()
            .find(|&&(d, min, max)| d == decode && rows >= min && rows <= max)
            .map_or(usize::MAX, |&(_, min, _)| min * 2 + usize::from(decode))
    }
    pub(super) fn work(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
        full: bool,
        decode: bool,
    ) -> Work {
        let mut w = Work {
            full: full
                && selected.iter().any(|&(i, n)| {
                    let r = &requests[i];
                    if r.prefilling {
                        r.offset + n == r.input.len()
                    } else {
                        r.needs_logits()
                    }
                }),
            ..Default::default()
        };
        let mut volume = 0u64;
        for &(i, n) in selected {
            let r = &requests[i];
            if r.prefilling {
                w.tokens += n;
                volume += n as u64 * r.offset as u64;
                w.final_rows += self.prompt_decoder_rows(n, r.offset + n == r.input.len());
            } else {
                w.decode_rows += n;
                w.decode_context = w.decode_context.max(r.history.len());
            }
        }
        w.context = (volume / w.tokens.max(1) as u64) as usize;
        w.domain = self.cost_domain(w.tokens + w.decode_rows, decode);
        w
    }
    fn prompt_decoder_rows(&self, rows: usize, final_chunk: bool) -> usize {
        self.model
            .batch
            .as_ref()
            .and_then(|c| c.decoder_replay.as_ref())
            .map_or(rows, |c| {
                if c.final_prompt_only && !final_chunk {
                    0
                } else {
                    rows.min(c.keep)
                }
            })
    }
    pub(super) fn producer_price(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
    ) -> Prediction {
        let prompts = selected
            .iter()
            .copied()
            .filter(|&(i, _)| requests[i].prefilling)
            .collect::<Vec<_>>();
        let mut w = self.work(&prompts, requests, false, false);
        // Target pricing already includes the final decoder/head. Opportunity
        // prices only postponed producer tokens, amortized over this packet.
        w.final_rows = 0;
        self.costs.estimate(w)
    }
    pub(super) fn remaining_prefill_cost(&self, offset: usize, end: usize) -> f64 {
        let n = end.saturating_sub(offset);
        if n == 0 {
            return 0.;
        }
        let keep = self
            .model
            .batch
            .as_ref()
            .and_then(|c| c.decoder_replay.as_ref())
            .filter(|r| r.final_prompt_only)
            .map_or(0, |r| r.keep);
        let last = if end > self.capacity && keep > 0 {
            n.min(keep)
        } else {
            (n - 1) % self.capacity + 1
        };
        let producer = n - last;
        let price = |q, c, final_chunk| {
            if q == 0 {
                return 0.;
            }
            self.costs
                .estimate(Work {
                    tokens: q,
                    context: c,
                    final_rows: self.prompt_decoder_rows(q, final_chunk),
                    full: final_chunk,
                    domain: self.cost_domain(q, false),
                    ..Default::default()
                })
                .seconds
        };
        let span = |start: usize, count: usize| {
            let chunks = count / self.capacity;
            let tail = count % self.capacity;
            chunks as f64
                * price(
                    self.capacity,
                    start + chunks.saturating_sub(1) * self.capacity / 2,
                    false,
                )
                + price(tail, start + chunks * self.capacity, false)
        };
        // Head/checkpoint/tail spans preserve the actual greedy partition,
        // including an unaligned restored endpoint, with constant work.
        let head = producer.min((4096 - offset % 4096) % 4096);
        let blocks = (producer - head) / 4096;
        let tail = (producer - head) % 4096;
        span(offset, head)
            + blocks as f64 * span(offset + head + blocks.saturating_sub(1) * 2048, 4096)
            + span(offset + head + blocks * 4096, tail)
            + price(last, end - last, true)
    }
}
impl Runtime {
    fn prefill_estimator(&self) -> Estimator<'_> {
        Estimator {
            model: &self.team.model,
            domains: &self.team.target_domains,
            costs: &self.prefill_costs,
            capacity: self.team.rows,
        }
    }
    pub(super) fn prefill_work(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
        full: bool,
        decode: bool,
    ) -> Work {
        self.prefill_estimator()
            .work(selected, requests, full, decode)
    }
    pub(super) fn producer_price(
        &self,
        selected: &[(usize, usize)],
        requests: &[&mut Request],
    ) -> Prediction {
        self.prefill_estimator().producer_price(selected, requests)
    }
    pub(super) fn remaining_prefill_cost(&self, offset: usize, end: usize) -> f64 {
        self.prefill_estimator().remaining_prefill_cost(offset, end)
    }
    pub(super) fn observe_prefill(&mut self, work: Work, timing: &super::super::team::Timing) {
        let path = if timing.replay {
            Path::Replay
        } else if timing.shared {
            Path::Shared
        } else {
            Path::Direct
        };
        let before = (work.tokens > 0 && std::env::var_os("ORINFER_A2_BATCH_PROFILE").is_some())
            .then(|| self.prefill_costs.estimate(work));
        self.prefill_costs.observe(
            work,
            path,
            timing.seconds + timing.preparation - timing.setup,
            timing.setup,
        );
        if let Some(before) = before {
            let estimate = self.prefill_costs.estimate(work);
            eprintln!(
                "A2_PREFILL_COST {}",
                serde_json::json!({
                    "tokens":work.tokens,"context":work.context,"decode_rows":work.decode_rows,
                    "decode_context":work.decode_context,"final_rows":work.final_rows,
                    "full":work.full,"domain":work.domain,"path":format!("{path:?}"),
                    "seconds":timing.seconds,"observed_s":timing.seconds+timing.preparation-timing.setup,
                    "setup_s":timing.setup,"prediction_s":before.seconds,
                    "prediction_path":format!("{:?}",before.path),"prediction_samples":before.samples,
                    "estimate_s":estimate.seconds,
                    "upper_s":estimate.upper_s,"samples":estimate.samples,
                })
            );
        }
    }
}
