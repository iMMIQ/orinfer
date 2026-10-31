//! Exact persistent-state snapshots; compute workspaces remain request-local.
use super::{Buffer, Worker, batch::Owner, get};
use crate::artifact::Result;
use crate::backend::ascend::runtime::Session;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, ops::Range};

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
    /// Closed linear rows are immutable; windows and recurrent state remain private copies.
    #[serde(default)]
    pub append_only: bool,
}
impl StateLayout {
    pub(super) fn spans(&self, tokens: usize, bytes: usize) -> Result<Vec<Range<usize>>> {
        if self.row_bytes == 0
            || !bytes.is_multiple_of(self.row_bytes)
            || self.window == Some(0)
            || self.append_only
                && (self.tokens_per_row == 0 || self.window.is_some() || self.circular)
        {
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
#[derive(Clone)]
struct Piece {
    name: String,
    offset: usize,
    buffer: Buffer,
    bank_id: u64,
    bank_bytes: usize,
}
pub(super) struct Snapshot {
    history: Vec<u32>,
    pieces: Vec<Piece>,
}
type PendingCopy = (String, Buffer, Range<usize>, usize);
struct CapturePlan {
    pieces: Vec<Piece>,
    pending: Vec<PendingCopy>,
    bytes: usize,
}
impl CapturePlan {
    fn allocations(&self, bank_id: u64) -> Result<BTreeMap<u64, usize>> {
        let mut banks: BTreeMap<_, _> = self
            .pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect();
        if !self.pending.is_empty() {
            let bytes = self
                .bytes
                .checked_add(255)
                .ok_or("Snapshot size overflow")?;
            banks.insert(bank_id, super::memory::round_pages(bytes)?);
        }
        Ok(banks)
    }
}
impl Snapshot {
    pub fn allocations(&self) -> BTreeMap<u64, usize> {
        self.pieces
            .iter()
            .map(|p| (p.bank_id, p.bank_bytes))
            .collect()
    }
    fn plan<'a>(
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        parent: Option<&Snapshot>,
    ) -> Result<CapturePlan> {
        if parent.is_some_and(|p| !history.starts_with(&p.history)) {
            return Err("A2 snapshot parent history diverges".into());
        }
        let mut pieces = vec![];
        let mut pending = vec![];
        let mut bytes = 0usize;
        for layout in layouts {
            let source = get(buffers, &layout.buffer)?;
            for mut span in layout.spans(history.len(), source.bytes())? {
                if layout.append_only
                    && let Some(parent) = parent
                {
                    let end = parent.history.len() / layout.tokens_per_row * layout.row_bytes;
                    let mut inherited = 0;
                    for piece in parent.pieces.iter().filter(|p| p.name == layout.buffer) {
                        if piece.offset != inherited
                            || inherited > end
                            || piece.buffer.bytes() > end - inherited
                        {
                            return Err("A2 snapshot parent has noncontiguous closed rows".into());
                        }
                        inherited += piece.buffer.bytes();
                        pieces.push(piece.clone());
                    }
                    if inherited != 0 && inherited != end {
                        return Err("A2 snapshot parent does not cover its closed rows".into());
                    }
                    span.start = inherited;
                }
                if span.is_empty() {
                    continue;
                }
                bytes = bytes
                    .checked_add(255)
                    .ok_or("Snapshot alignment overflow")?
                    / 256
                    * 256;
                let offset = bytes;
                bytes = bytes
                    .checked_add(span.len())
                    .ok_or("Snapshot size overflow")?;
                pending.push((layout.buffer.clone(), source.clone(), span, offset));
            }
        }
        Ok(CapturePlan {
            pieces,
            pending,
            bytes,
        })
    }
    fn capture<'a>(
        session: &Session,
        layouts: impl Iterator<Item = &'a StateLayout>,
        buffers: &BTreeMap<String, Buffer>,
        history: &[u32],
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<Self> {
        let CapturePlan {
            mut pieces,
            pending,
            bytes,
        } = Self::plan(layouts, buffers, history, parent)?;
        if !pending.is_empty() {
            let bank = session.allocate(
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
            let bank_bytes = bank.bytes().div_ceil(2 << 20) * (2 << 20);
            let mut copies = vec![];
            for (name, source, span, offset) in pending {
                let buffer = bank.slice(base + offset, span.len())?;
                copies.push((source, span.start, buffer.clone(), 0, span.len()));
                pieces.push(Piece {
                    name,
                    offset: span.start,
                    buffer,
                    bank_id,
                    bank_bytes,
                });
            }
            session.copy_many(&copies)?.wait()?;
        }
        Ok(Self {
            history: history.to_vec(),
            pieces,
        })
    }
    fn restore_into(&self, session: &Session, buffers: &BTreeMap<String, Buffer>) -> Result<()> {
        let copies = self
            .pieces
            .iter()
            .map(|piece| {
                Ok((
                    piece.buffer.clone(),
                    0,
                    get(buffers, &piece.name)?,
                    piece.offset,
                    piece.buffer.bytes(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if !copies.is_empty() {
            session.copy_many(&copies)?.wait()?;
        }
        Ok(())
    }
}
impl Worker {
    pub(super) fn release_owner(&mut self, id: usize) -> Result<()> {
        self.session.synchronize()?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        runtime.bound_tables = None;
        runtime.owners.remove(&id).ok_or("Unknown A2 owner")?;
        if id == 0 {
            self.clear_base_growth()?;
        }
        Ok(())
    }
    fn snapshot_state(&self, id: usize, complete: bool) -> Result<(&Owner, Vec<&StateLayout>)> {
        let runtime = self.batch.as_ref().ok_or("Missing A2 batch runtime")?;
        if runtime.contract.state.is_empty() {
            return Err("A2 package does not declare persistent state".into());
        }
        let owner = runtime.owners.get(&id).ok_or("Unknown A2 owner")?;
        let decoder = runtime
            .contract
            .decoder_replay
            .as_ref()
            .filter(|r| complete && r.final_prompt_only && !r.decoder_state.is_empty());
        if complete && decoder.is_none() {
            return Err("A2 package does not declare complete-prompt decoder state".into());
        }
        Ok((
            owner,
            runtime
                .contract
                .state
                .iter()
                .chain(decoder.into_iter().flat_map(|r| &r.decoder_state))
                .collect(),
        ))
    }
    pub(super) fn plan_owner_snapshot(
        &self,
        id: usize,
        complete: bool,
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<BTreeMap<u64, usize>> {
        let (owner, layouts) = self.snapshot_state(id, complete)?;
        Snapshot::plan(layouts.into_iter(), &owner.buffers, &owner.history, parent)?
            .allocations(bank_id)
    }
    pub(super) fn save_owner(
        &self,
        id: usize,
        complete: bool,
        bank_id: u64,
        parent: Option<&Snapshot>,
    ) -> Result<Snapshot> {
        let (
            Owner {
                buffers, history, ..
            },
            layouts,
        ) = self.snapshot_state(id, complete)?;
        Snapshot::capture(
            &self.session,
            layouts.into_iter(),
            buffers,
            history,
            bank_id,
            parent,
        )
    }

    pub(super) fn restore_owner(&mut self, id: usize, snapshot: &Snapshot) -> Result<()> {
        self.reset_owner(id)?;
        self.grow_owner(id, snapshot.history.len())?;
        let runtime = self.batch.as_mut().ok_or("Missing A2 batch runtime")?;
        let owner = runtime.owners.get_mut(&id).ok_or("Unknown A2 owner")?;
        snapshot.restore_into(&self.session, &owner.buffers)?;
        owner.history.clone_from(&snapshot.history);
        runtime.bound_tables = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ascend::tests::Fixture;

    #[test]
    fn compressed_snapshot_restores_after_crossing_a_physical_page_boundary() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let grain = 2 << 20;
        let source = session.reserve(grain * 2).unwrap();
        let buffers = BTreeMap::from([("kv".into(), source.clone())]);
        let layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        session.grow(&source, 32).unwrap();
        session
            .upload(&source, 0, &[73; 32])
            .unwrap()
            .wait()
            .unwrap();
        let parent = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[9; 4],
            1,
            None,
        )
        .unwrap();
        session.grow(&source, grain + 32).unwrap();
        session
            .upload(&source, grain, &[91; 32])
            .unwrap()
            .wait()
            .unwrap();
        let history = vec![9; (grain / 16 + 2) * 2];
        let child = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &history,
            2,
            Some(&parent),
        )
        .unwrap();
        drop(parent);
        let target = session.reserve(grain * 2).unwrap();
        session.grow(&target, grain + 32).unwrap();
        child
            .restore_into(&session, &BTreeMap::from([("kv".into(), target.clone())]))
            .unwrap();
        assert_eq!(session.download(&target, 0, 32).unwrap(), vec![73; 32]);
        assert_eq!(session.download(&target, grain, 32).unwrap(), vec![91; 32]);
        drop(child);
        drop(buffers);
        drop(source);
        drop(target);
        drop(session);
        fixture.assert_clean();
    }

    #[test]
    fn evicting_a_parent_requires_replanning_the_new_bank() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let buffers = BTreeMap::from([("kv".into(), session.allocate(128).unwrap())]);
        let layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        let parent = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3],
            1,
            None,
        )
        .unwrap();
        let before = session.allocated_bytes();
        let child = Snapshot::plan(
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            Some(&parent),
        )
        .unwrap();
        assert_eq!(session.allocated_bytes(), before);
        assert_eq!(child.bytes, 32);
        assert!(child.allocations(2).unwrap().contains_key(&1));
        drop(child);
        drop(parent);
        let replanned = Snapshot::plan(
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            None,
        )
        .unwrap();
        assert_eq!(replanned.bytes, 64);
        assert_eq!(replanned.allocations(2).unwrap().len(), 1);
        let saved = Snapshot::capture(
            &session,
            std::iter::once(&layout),
            &buffers,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            2,
            None,
        )
        .unwrap();
        assert_eq!(saved.allocations(), replanned.allocations(2).unwrap());
        drop(replanned);
        drop(saved);
        drop(buffers);
        drop(session);
        fixture.assert_clean();
    }

    #[test]
    fn inherited_closed_rows_survive_parent_eviction_and_keep_mutable_state_private() {
        let mut fixture = Fixture::new();
        let session = fixture.session();
        let layouts = [
            StateLayout {
                buffer: "kv".into(),
                row_bytes: 16,
                tokens_per_row: 2,
                window: None,
                circular: false,
                append_only: true,
            },
            StateLayout {
                buffer: "swa".into(),
                row_bytes: 4,
                tokens_per_row: 1,
                window: Some(3),
                circular: true,
                append_only: false,
            },
            StateLayout {
                buffer: "ring".into(),
                row_bytes: 64,
                tokens_per_row: 0,
                window: None,
                circular: false,
                append_only: false,
            },
        ];
        let buffers = [("kv", 128), ("swa", 32), ("ring", 64)]
            .into_iter()
            .map(|(name, bytes)| (name.to_owned(), session.allocate(bytes).unwrap()))
            .collect::<BTreeMap<_, _>>();
        let mut parent = None;
        let mut reference = None;
        for (index, length) in [3usize, 5, 9].into_iter().enumerate() {
            let kv = (0..128).map(|i| (i / 16 + 1) as u8).collect::<Vec<_>>();
            let mut swa = vec![0; 32];
            for token in 0..length {
                swa[token % 8 * 4..token % 8 * 4 + 4].fill(token as u8 + 1);
            }
            session
                .upload_inputs(&[
                    (buffers["kv"].clone(), kv),
                    (buffers["swa"].clone(), swa),
                    (buffers["ring"].clone(), vec![length as u8; 64]),
                ])
                .unwrap()
                .wait()
                .unwrap();
            let history = (0..length as u32).collect::<Vec<_>>();
            let before_plan = session.allocated_bytes();
            let planned = Snapshot::plan(layouts.iter(), &buffers, &history, parent.as_ref())
                .unwrap()
                .allocations(index as u64 + 1)
                .unwrap();
            assert_eq!(
                session.allocated_bytes(),
                before_plan,
                "Planning cannot allocate device memory"
            );
            let saved = Snapshot::capture(
                &session,
                layouts.iter(),
                &buffers,
                &history,
                index as u64 + 1,
                parent.as_ref(),
            )
            .unwrap();
            assert_eq!(saved.allocations(), planned);
            let kv_pieces = saved
                .pieces
                .iter()
                .filter(|p| p.name == "kv")
                .collect::<Vec<_>>();
            assert_eq!(kv_pieces.len(), index + 1);
            assert_eq!(kv_pieces[0].bank_id, 1);
            assert_eq!(kv_pieces.last().unwrap().offset, [0, 16, 32][index]);
            assert!(
                saved
                    .pieces
                    .iter()
                    .filter(|p| p.name != "kv")
                    .all(|p| p.bank_id == index as u64 + 1)
            );
            assert!(
                Snapshot::capture(
                    &session,
                    layouts.iter(),
                    &buffers,
                    &[99; 9],
                    20,
                    Some(&saved)
                )
                .is_err()
            );
            let mut full_layouts = layouts.clone();
            full_layouts[0].append_only = false;
            reference = Some(
                Snapshot::capture(&session, full_layouts.iter(), &buffers, &history, 100, None)
                    .unwrap(),
            );
            // Dropping the parent must preserve the child's inherited banks;
            // mutable spans still point exclusively to this capture's bank.
            parent = Some(saved);
        }
        drop(buffers);
        let saved = parent.unwrap();
        assert_eq!(saved.allocations().len(), 3);
        let target = || {
            [("kv", 128), ("swa", 32), ("ring", 64)]
                .into_iter()
                .map(|(name, bytes)| {
                    let buffer = session.allocate(bytes).unwrap();
                    session.fill(&buffer, 0).unwrap().wait().unwrap();
                    (name.to_owned(), buffer)
                })
                .collect::<BTreeMap<_, _>>()
        };
        let actual = target();
        let expected = target();
        saved.restore_into(&session, &actual).unwrap();
        reference
            .as_ref()
            .unwrap()
            .restore_into(&session, &expected)
            .unwrap();
        for name in ["kv", "swa", "ring"] {
            assert_eq!(
                session
                    .download(&actual[name], 0, actual[name].bytes())
                    .unwrap(),
                session
                    .download(&expected[name], 0, expected[name].bytes())
                    .unwrap(),
                "{name}"
            );
        }
        assert_eq!(session.download(&actual["kv"], 64, 64).unwrap(), [0; 64]);
        assert_eq!(session.download(&actual["swa"], 4, 20).unwrap(), [0; 20]);
        drop((saved, reference, actual, expected, session));
        fixture.assert_clean();
    }

    #[test]
    fn shared_rows_reject_mutable_geometry() {
        let mut layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: true,
        };
        assert_eq!(layout.spans(5, 128).unwrap(), vec![0..32]);
        layout.tokens_per_row = 0;
        assert!(layout.spans(5, 128).is_err());
        layout.tokens_per_row = 1;
        layout.window = Some(3);
        assert!(layout.spans(5, 128).is_err());
        layout.window = None;
        layout.circular = true;
        assert!(layout.spans(5, 128).is_err());
    }

    #[test]
    fn state_spans_preserve_partial_compression_and_wrapped_window() {
        let mut layout = StateLayout {
            buffer: "kv".into(),
            row_bytes: 16,
            tokens_per_row: 2,
            window: None,
            circular: false,
            append_only: false,
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
