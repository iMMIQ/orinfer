//! Exact persistent-state snapshots; compute workspaces remain request-local.
use super::{Buffer, Worker, batch::Owner, get};
use crate::artifact::Result;
use serde::{Deserialize, Serialize};
use std::ops::Range;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StateLayout {
    pub buffer: String,
    pub row_bytes: usize,
    /// Zero denotes a fixed-size recurrent state, copied in full.
    pub tokens_per_row: usize,
    pub window: Option<usize>,
    #[serde(default)]
    pub circular: bool,
}
impl StateLayout {
    pub(super) fn spans(&self, tokens: usize, bytes: usize) -> Result<Vec<Range<usize>>> {
        if self.row_bytes == 0 || !bytes.is_multiple_of(self.row_bytes) || self.window == Some(0) {
            return Err("Invalid A2 state layout".into());
        }
        if self.tokens_per_row == 0 {
            return Ok(std::iter::once(0..bytes).collect());
        }
        let end = tokens / self.tokens_per_row;
        let start = self.window.map_or(0, |window| end.saturating_sub(window));
        let rows = bytes / self.row_bytes;
        if self.circular {
            if rows == 0 || end - start > rows {
                return Err("A2 state window exceeds ring capacity".into());
            }
            let first = start % rows;
            let count = end - start;
            let left = count.min(rows - first);
            let mut spans = vec![];
            if left > 0 {
                spans.push(first * self.row_bytes..(first + left) * self.row_bytes);
            }
            if count > left {
                spans.push(0..(count - left) * self.row_bytes);
            }
            Ok(spans)
        } else {
            if end > rows {
                return Err("A2 state endpoint exceeds capacity".into());
            }
            Ok((end > start)
                .then_some(start * self.row_bytes..end * self.row_bytes)
                .into_iter()
                .collect())
        }
    }
}
struct Piece {
    name: String,
    offset: usize,
    buffer: Buffer,
}
pub(super) struct Snapshot {
    history: Vec<u32>,
    pieces: Vec<Piece>,
    bank: Buffer,
}
impl Snapshot {
    pub fn bytes(&self) -> usize {
        self.bank.bytes().div_ceil(2 << 20) * (2 << 20)
    }
}
impl Worker {
    pub(super) fn release_owner(&mut self, id: usize) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.remove(&id).ok_or("Unknown A2 owner")?;
        Ok(())
    }
    pub(super) fn save_owner(&self, id: usize, complete: bool) -> Result<Snapshot> {
        let runtime = self.batch.as_ref().ok_or("Missing A2 batch runtime")?;
        if runtime.contract.state.is_empty() {
            return Err("A2 package does not declare persistent state".into());
        }
        let Owner { buffers, history } = runtime.owners.get(&id).ok_or("Unknown A2 owner")?;
        let decoder = runtime
            .contract
            .decoder_replay
            .as_ref()
            .filter(|r| complete && r.final_prompt_only && !r.decoder_state.is_empty());
        if complete && decoder.is_none() {
            return Err("A2 package does not declare complete-prompt decoder state".into());
        }
        let mut spans = vec![];
        let mut bytes = 0usize;
        for layout in runtime
            .contract
            .state
            .iter()
            .chain(decoder.into_iter().flat_map(|r| &r.decoder_state))
        {
            let source = get(buffers, &layout.buffer)?;
            for span in layout.spans(history.len(), source.bytes())? {
                bytes = bytes
                    .checked_add(255)
                    .ok_or("Snapshot alignment overflow")?
                    / 256
                    * 256;
                let offset = bytes;
                bytes = bytes
                    .checked_add(span.len())
                    .ok_or("Snapshot size overflow")?;
                spans.push((layout.buffer.clone(), source.clone(), span, offset));
            }
        }
        let bank = self.session.allocate(
            bytes
                .checked_add(255)
                .ok_or("Snapshot alignment overflow")?,
        )?;
        let address = bank.pointer() as usize;
        let base = address
            .checked_add(255)
            .ok_or("Snapshot address overflow")?
            / 256
            * 256
            - address;
        let mut pieces = vec![];
        let mut tickets = vec![];
        for (name, source, span, offset) in spans {
            let buffer = bank.slice(base + offset, span.len())?;
            tickets.push(
                self.session
                    .copy(&source, span.start, &buffer, 0, span.len())?,
            );
            pieces.push(Piece {
                name,
                offset: span.start,
                buffer,
            });
        }
        for ticket in tickets {
            ticket.wait()?;
        }
        Ok(Snapshot {
            history: history.clone(),
            pieces,
            bank,
        })
    }
    pub(super) fn restore_owner(&mut self, id: usize, snapshot: &Snapshot) -> Result<()> {
        self.reset_owner(id)?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        let owner = runtime.owners.get_mut(&id).ok_or("Unknown A2 owner")?;
        let mut tickets = vec![];
        for piece in &snapshot.pieces {
            let target = get(&owner.buffers, &piece.name)?;
            tickets.push(self.session.copy(
                &piece.buffer,
                0,
                &target,
                piece.offset,
                piece.buffer.bytes(),
            )?);
        }
        for ticket in tickets {
            ticket.wait()?;
        }
        owner.history.clone_from(&snapshot.history);
        runtime.bound_tables = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn state_spans_preserve_partial_compression_and_wrapped_window() {
        let mut layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
        };
        assert_eq!(layout.spans(129, 2048).unwrap(), vec![0..1024]);
        assert!(layout.spans(258, 2048).is_err());
        layout.tokens_per_row = 1;
        layout.window = Some(3);
        layout.circular = true;
        assert_eq!(layout.spans(9, 128).unwrap(), vec![96..128, 0..16]);
        assert_eq!(layout.spans(8, 128).unwrap(), vec![80..128]);
        assert!(layout.spans(8, 0).is_err());
        layout.tokens_per_row = 0;
        assert_eq!(layout.spans(999, 128).unwrap(), vec![0..128]);
    }
}
