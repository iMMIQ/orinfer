//! Bounded decoder row domains; persistent state remains owner-private.
use super::{Definition, Instruction, Packet, batch};
use crate::{artifact::Result, backend::ascend::kernels::Spec};
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Contract {
    pub keep: usize,
    /// Rebuild decoder state only in the final prompt window.
    #[serde(default)]
    pub final_prompt_only: bool,
    pub tables: BTreeSet<String>,
    pub maps: Vec<RowMap>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Role {
    Producer,
    Final,
    Continue,
}

/// Keep the final prompt window in one packet. Long prompts checkpoint before
/// that window, so an exact prefix hit can rebuild it without replaying a block.
pub(super) fn prompt_rows(left: usize, limit: usize, keep: usize, reserve: bool) -> Option<usize> {
    if left == 0 || limit == 0 {
        return None;
    }
    if keep == 0 {
        return Some(left.min(limit));
    }
    if left <= keep {
        return (left <= limit).then_some(left);
    }
    Some(if reserve || left > limit {
        limit.min(left - keep)
    } else {
        left
    })
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RowMap {
    pub buffer: String,
    pub source_partition: usize,
    pub destination_partition: usize,
    #[serde(default)]
    pub contiguous: bool,
}
pub(super) struct Boundary {
    pub index: usize,
    pub keep: usize,
    prepare: usize,
    select: usize,
}
impl Boundary {
    pub fn find(program: &[Instruction]) -> Result<Option<Self>> {
        let mut result = None;
        for (index, step) in program.iter().enumerate() {
            if let Instruction::DecoderRows {
                keep,
                prepare,
                select,
            } = step
            {
                if result.is_some()
                    || !(128..=512).contains(keep)
                    || *prepare > index
                    || index
                        .checked_add(*select)
                        .is_none_or(|end| end >= program.len())
                {
                    return Err("Invalid decoder row boundary".into());
                }
                if program[index - prepare..index].iter().any(|s| {
                    !matches!(
                        s,
                        Instruction::Gather {
                            trace_only: false,
                            ..
                        }
                    )
                }) || program[index + 1..index + 1 + select].iter().any(|s| {
                    !matches!(
                        s,
                        Instruction::Kernel {
                            owner_work: None,
                            ..
                        }
                    )
                }) {
                    return Err("Decoder selection requires flat gather/kernel boundaries".into());
                }
                result = Some(Self {
                    index,
                    keep: *keep,
                    prepare: *prepare,
                    select: *select,
                });
            }
        }
        Ok(result)
    }
    pub fn skip(&self, index: usize) -> bool {
        (self.index - self.prepare..self.index).contains(&index)
            || (self.index + 1..=self.index + self.select).contains(&index)
    }
}

pub(super) fn validate_contract(
    c: &batch::Contract,
    buffers: &BTreeMap<&str, &Definition>,
    peers: usize,
) -> Result<()> {
    let Some(replay) = &c.decoder_replay else {
        if c.columns.iter().any(|v| v.decoder) {
            return Err("Decoder columns require a replay contract".into());
        }
        return Ok(());
    };
    if !(128..=512).contains(&replay.keep)
        || replay.maps.is_empty()
        || replay.tables.is_empty()
        || replay
            .tables
            .iter()
            .any(|n| !c.work.contains_key(n) && !c.pointers.contains_key(n))
        || c.columns
            .iter()
            .any(|v| v.decoder && (v.decode != Some(false) || v.row_bytes.is_none()))
    {
        return Err("Invalid decoder replay contract".into());
    }
    let mut maps = BTreeSet::new();
    for map in &replay.maps {
        if !maps.insert(&map.buffer)
            || !buffers.contains_key(map.buffer.as_str())
            || ![1, peers].contains(&map.source_partition)
            || ![1, peers].contains(&map.destination_partition)
        {
            return Err("Invalid decoder selection map".into());
        }
    }
    for (table, columns) in &c.pointers {
        if columns.iter().any(|&i| {
            c.columns
                .get(i)
                .is_none_or(|v| v.row_bytes.is_some() && v.decoder != replay.tables.contains(table))
        }) {
            return Err("Owner table mixes full and decoder row domains".into());
        }
    }
    Ok(())
}

pub(super) fn validate_packet(packet: &Packet, contract: Option<&batch::Contract>) -> Result<()> {
    let body = Boundary::find(&packet.body)?;
    let head = Boundary::find(&packet.head)?;
    let replay = contract.and_then(|c| c.decoder_replay.as_ref());
    if packet.batch_decode || replay.is_none() {
        if body.is_some() || head.is_some() || packet.decoder_coordinates.is_some() {
            return Err("Decoder row selection requires a prefill replay contract".into());
        }
        return Ok(());
    }
    let replay = replay.unwrap();
    let body = body.ok_or("Missing decoder body boundary")?;
    let head = head.ok_or("Missing decoder head boundary")?;
    if packet.decoder_coordinates.is_none()
        || body.keep != replay.keep
        || head.keep != replay.keep
        || head.prepare != 0
        || head.select != 0
        || !matches!(body.index.checked_sub(body.prepare + 1).and_then(|i| packet.body.get(i)), Some(Instruction::Snapshot { name, .. }) if name == "layer20")
    {
        return Err("Decoder replay must follow the complete shared KV source layer".into());
    }
    Ok(())
}

pub(super) fn validate_fallback(source: &Spec, fallback: &Spec) -> Result<()> {
    let compatible = source.op == fallback.op
        && source.target == fallback.target
        && source.operands.len() == fallback.operands.len()
        && source
            .operands
            .iter()
            .zip(&fallback.operands)
            .all(|(a, b)| {
                a.name == b.name && a.dtype == b.dtype && a.shape == b.shape && a.access == b.access
            });
    let parameters = |s: &Spec| {
        s.parameters
            .iter()
            .filter(|(k, _)| !["reuse_long", "_owner_source_sha256"].contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<BTreeMap<_, _>>()
    };
    if !compatible
        || parameters(source) != parameters(fallback)
        || source.parameters.get("reuse_long") != Some(&serde_json::json!(false))
        || fallback.parameters.get("reuse_long") != Some(&serde_json::json!(true))
    {
        return Err("Decoder fresh-KV fallback changes its computation or ABI".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use orinfer_model_sdk::architecture::BatchSegment;
    #[test]
    fn final_window_is_atomic_at_chunk_and_checkpoint_tails() {
        for length in [1usize, 17, 127, 128, 129, 2048, 2049, 4096, 4097, 32769] {
            let mut left = length;
            while left > 0 {
                let next = 4096 - (length - left) % 4096;
                let n = prompt_rows(left, 2048.min(next), 128, length > 2048)
                    .or_else(|| prompt_rows(left, 128, 128, true))
                    .unwrap();
                assert!(n > 0 && n <= left);
                if n == left {
                    assert!(n >= 128.min(length));
                }
                left -= n;
                if left > 0 {
                    assert!(left >= 128);
                }
            }
        }
        assert_eq!(prompt_rows(128, 127, 128, true), None);
        assert_eq!(prompt_rows(2048, 2048, 128, false), Some(2048));
        assert_eq!(prompt_rows(2048, 2048, 128, true), Some(1920));
        assert_eq!(prompt_rows(2049, 2048, 128, true), Some(1921));
    }
    #[test]
    fn prompt_decoder_preserves_empty_lanes_and_live_decode_history() {
        let mut layout = batch::Layout::new(
            vec![
                BatchSegment {
                    slot: 17,
                    tokens: 2048,
                },
                BatchSegment {
                    slot: 91,
                    tokens: 1,
                },
                BatchSegment {
                    slot: 7,
                    tokens: 128,
                },
            ],
            vec![1, 2],
            false,
            4096,
            128,
        )
        .unwrap();
        layout.decoder_roles[1] = Role::Continue;
        let selected = layout.decoder(128, true).unwrap();
        assert_eq!(selected.rows.starts, vec![0, 0, 1]);
        assert_eq!(selected.rows.total, 129);
        assert_eq!(
            selected
                .segments
                .iter()
                .map(|s| (s.slot, s.tokens))
                .collect::<Vec<_>>(),
            vec![(17, 0), (91, 1), (7, 128)]
        );
        assert_eq!(selected.outputs, vec![1, 2]);
        assert_eq!(
            selected.decoder(128, true).unwrap().rows.starts,
            selected.rows.starts
        );
        assert_eq!(
            layout.row_map(&selected, 1, 1, 0, false).unwrap(),
            (2048..2177).collect::<Vec<i32>>()
        );
        for rank in 0..8 {
            let map = layout.row_map(&selected, 8, 8, rank, false).unwrap();
            let mut expected = (2048..2177usize)
                .skip(rank)
                .step_by(8)
                .map(|i| (i % 8 * 2177usize.div_ceil(8) + i / 8) as i32)
                .collect::<Vec<_>>();
            expected.resize(129usize.div_ceil(8), -1);
            assert_eq!(map, expected);
        }
        layout.outputs.clear();
        layout.decoder_roles.fill(Role::Producer);
        let empty = layout.decoder(128, true).unwrap();
        assert_eq!(empty.rows.total, 0);
        assert!(layout.row_map(&empty, 1, 1, 0, false).unwrap().is_empty());
        assert_eq!(empty.rows.partition(8, 7).unwrap().lengths, vec![0; 3]);
        layout.outputs = vec![1];
        layout.decoder_roles[1] = Role::Continue;
        let one = layout.decoder(128, true).unwrap();
        assert_eq!(one.rows.total, 1);
        assert_eq!(one.rows.partition(8, 0).unwrap().lengths, vec![0, 1, 0]);
        for rank in 1..8 {
            assert_eq!(one.rows.partition(8, rank).unwrap().lengths, vec![0; 3]);
        }
    }
    #[test]
    fn bounded_maps_preserve_each_owner_under_different_partitions() {
        for lengths in [
            vec![1, 1025, 17, 511],
            vec![2048],
            vec![129, 128, 140, 3],
            vec![17; 32],
        ] {
            let layout = batch::Layout::new(
                lengths
                    .iter()
                    .enumerate()
                    .map(|(slot, &tokens)| BatchSegment { slot, tokens })
                    .collect(),
                vec![0],
                false,
                4096,
                128,
            )
            .unwrap();
            let suffix = layout.suffix(128).unwrap();
            assert_eq!(
                suffix.rows.total,
                lengths.iter().map(|&n| n.min(128)).sum::<usize>()
            );
            let indices = layout.row_map(&suffix, 1, 1, 0, false).unwrap();
            for (owner, &n) in lengths.iter().enumerate() {
                let kept = n.min(128);
                let start = suffix.rows.starts[owner];
                assert_eq!(
                    &indices[start..start + kept],
                    &(layout.rows.starts[owner] + n - kept..layout.rows.starts[owner] + n)
                        .map(|v| v as i32)
                        .collect::<Vec<_>>()
                );
            }
            for source in [1, 8] {
                let all = layout.row_map(&suffix, source, 1, 0, false).unwrap();
                for rank in 0..8 {
                    let mut cyclic = all
                        .iter()
                        .skip(rank)
                        .step_by(8)
                        .copied()
                        .collect::<Vec<_>>();
                    cyclic.resize(all.len().div_ceil(8), -1);
                    assert_eq!(
                        layout.row_map(&suffix, source, 8, rank, false).unwrap(),
                        cyclic
                    );
                    let width = all.len().div_ceil(8);
                    let mut contiguous = all
                        .iter()
                        .skip(rank * width)
                        .take(width)
                        .copied()
                        .collect::<Vec<_>>();
                    contiguous.resize(width, -1);
                    assert_eq!(
                        layout.row_map(&suffix, source, 8, rank, true).unwrap(),
                        contiguous
                    );
                }
            }
        }
    }
    #[test]
    fn marker_rejects_missing_ranges_and_skips_only_selection() {
        let parse = |s: serde_json::Value| serde_json::from_value::<Instruction>(s).unwrap();
        let program = vec![
            parse(
                serde_json::json!({"op":"gather","source":"a","destination":"b","elements":1,"fp32":true}),
            ),
            parse(serde_json::json!({"op":"decoder_rows","keep":128,"prepare":1,"select":1})),
            parse(serde_json::json!({"op":"kernel","manifest":"a","buffers":[]})),
        ];
        let b = Boundary::find(&program).unwrap().unwrap();
        assert!(b.skip(0) && b.skip(2) && !b.skip(1));
        assert!(Boundary::find(&program[1..]).is_err());
        assert!(Boundary::find(&program[..2]).is_err());
    }
}
