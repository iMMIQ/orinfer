//! One completed packet per request; response work overlaps the next forward.
use super::*;

pub(super) struct Tokens {
    pub tokens: Vec<u32>,
    pub scores: Vec<orinfer_engine::sampling::TokenLogprob>,
}

#[derive(Clone, Default, serde::Serialize)]
pub(in super::super) struct Statistics {
    deferred_packets: usize,
    deferred_tokens: usize,
    maintenance_drains: usize,
    barrier_drains: usize,
    postprocess_s: f64,
    maintenance_postprocess_s: f64,
}

impl Response<'_> {
    pub(super) fn deliver(
        &mut self,
        packet: Tokens,
        defer: bool,
        codec: &ChatCodec,
        model: &str,
        statistics: &mut Statistics,
    ) {
        debug_assert!(self.pending.is_none());
        if self.ending.is_some() {
            return;
        }
        // First output and every possible normal termination are processed
        // before submitting more work for this owner. In particular, user stop
        // strings may span packets, so those requests stay inline throughout.
        let can_defer = defer
            && self.first_token_s.is_some()
            && self.mailbox.job.prepared.stops.is_empty()
            && !packet.tokens.is_empty()
            && (self.mailbox.job.prepared.ignore_eos
                || !packet.tokens.iter().any(|t| codec.eos.contains(t)));
        if can_defer {
            statistics.deferred_packets += 1;
            statistics.deferred_tokens += packet.tokens.len();
            self.pending = Some(packet);
        } else {
            let started = Instant::now();
            self.consume(packet, codec, model);
            statistics.postprocess_s += started.elapsed().as_secs_f64();
        }
    }

    pub(super) fn drain(
        &mut self,
        forward: bool,
        codec: &ChatCodec,
        model: &str,
        statistics: &mut Statistics,
    ) {
        let Some(packet) = self.pending.take() else {
            return;
        };
        if self.ending.is_some() {
            return;
        }
        let started = Instant::now();
        self.consume(packet, codec, model);
        let seconds = started.elapsed().as_secs_f64();
        statistics.postprocess_s += seconds;
        if forward {
            statistics.maintenance_drains += 1;
            statistics.maintenance_postprocess_s += seconds;
        } else {
            statistics.barrier_drains += 1;
        }
    }
}

#[cfg(test)]
mod tests;
