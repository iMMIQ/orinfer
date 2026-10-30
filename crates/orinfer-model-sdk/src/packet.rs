//! Common packet operand-table contract for ordinary and speculative rows.
use crate::architecture::BatchSegment;
use crate::{
    artifact::{Dtype, Result},
    model::Manifest,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketTable {
    pub buffer: String,
    pub columns: Vec<String>,
    pub kernels: BTreeMap<String, String>,
    /// Original host argument index -> wrapped host argument index.
    #[serde(default)]
    pub scalar_arguments: BTreeMap<String, Vec<(usize, usize)>>,
    pub copies: Vec<PacketCopy>,
}

impl PacketTable {
    pub fn validate(&self, model: &Manifest) -> Result<()> {
        let table = self;
        let columns: BTreeSet<_> = table.columns.iter().collect();
        let runtime_buffers: BTreeSet<_> = model
            .buffers
            .iter()
            .filter(|buffer| buffer.data.is_none())
            .map(|buffer| buffer.name.as_str())
            .collect();
        let kernels: BTreeSet<_> = model.kernels.iter().map(|k| k.name.as_str()).collect();
        if columns.is_empty()
            || columns.len() != table.columns.len()
            || !model.buffers.iter().any(|b| {
                b.name == table.buffer
                    && b.dtype == Dtype::U64
                    && b.data.is_none()
                    && b.shape == [128 * columns.len()]
            })
            || table
                .columns
                .iter()
                .any(|name| !runtime_buffers.contains(name.as_str()))
            || table.copies.iter().any(|copy| {
                copy.row_bytes == 0 || !model.dynamic_batch_kernels.contains_key(&copy.kernel)
            })
            || table
                .kernels
                .values()
                .chain(table.copies.iter().map(|c| &c.kernel))
                .any(|name| !kernels.contains(name.as_str()))
        {
            return Err("Invalid packet address table".into());
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PacketCopy {
    pub source: String,
    pub destination: String,
    pub kernel: String,
    pub row_bytes: usize,
}

/// Contiguous live rows, independent of the model's computation or state layout.
#[derive(Clone, Debug)]
pub struct Rows {
    pub starts: Vec<usize>,
    pub total: usize,
}

/// A compact row selection that preserves every original owner's position.
/// Owners without selected rows have length zero and keep their original slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowView {
    pub starts: Vec<usize>,
    pub lengths: Vec<usize>,
    pub total: usize,
}

impl RowView {
    pub fn aligned(&self, alignment: usize) -> Result<Self> {
        if alignment == 0 {
            return Err("Zero row alignment".into());
        }
        let mut result = Self {
            starts: vec![],
            lengths: vec![],
            total: 0,
        };
        for &length in &self.lengths {
            let padded = length
                .div_ceil(alignment)
                .checked_mul(alignment)
                .ok_or("Aligned row extent overflow")?;
            result.starts.push(result.total);
            result.lengths.push(padded);
            result.total = result
                .total
                .checked_add(padded)
                .ok_or("Aligned row total overflow")?;
        }
        Ok(result)
    }
}

impl Rows {
    /// Select global rows `first + n * stride`, retaining owner identity.
    pub fn partition(&self, stride: usize, first: usize) -> Result<RowView> {
        if stride == 0 || first >= stride {
            return Err("Invalid row partition".into());
        }
        let before = |end: usize| end.saturating_sub(first).div_ceil(stride);
        let starts = self.starts.iter().map(|&s| before(s)).collect::<Vec<_>>();
        let total = before(self.total);
        let lengths = starts
            .iter()
            .enumerate()
            .map(|(i, &start)| starts.get(i + 1).copied().unwrap_or(total) - start)
            .collect();
        Ok(RowView {
            starts,
            lengths,
            total,
        })
    }

    pub fn new(lengths: impl IntoIterator<Item = usize>, capacity: usize) -> Result<Self> {
        let mut starts = Vec::new();
        let mut total = 0usize;
        for length in lengths {
            if length == 0 {
                return Err("Empty packet segment".into());
            }
            starts.push(total);
            total = total.checked_add(length).ok_or("Packet row overflow")?;
            if total > capacity {
                return Err("Packet row capacity exceeded".into());
            }
        }
        if starts.is_empty() {
            return Err("Empty packet".into());
        }
        Ok(Self { starts, total })
    }

    pub fn packet(segments: &[BatchSegment], owners: usize, capacity: usize) -> Result<Self> {
        let mut slots = BTreeSet::new();
        if segments.len() > owners || segments.iter().any(|s| !slots.insert(s.slot)) {
            return Err("Packet requires distinct owners within capacity".into());
        }
        Self::new(segments.iter().map(|s| s.tokens), capacity)
    }
}

#[cfg(test)]
mod row_view_tests {
    use super::*;

    #[test]
    fn strided_views_cover_each_global_row_once_with_original_owners() {
        for total in 1usize..=2048 {
            let lengths = [1, total.saturating_sub(1)].into_iter().filter(|&n| n != 0);
            let rows = Rows::new(lengths, 2048).unwrap();
            let mut covered = vec![0; total];
            for rank in 0..8 {
                let view = rows.partition(8, rank).unwrap();
                for (owner, &start) in view.starts.iter().enumerate() {
                    for index in start..start + view.lengths[owner] {
                        let global = rank + index * 8;
                        assert!(global >= rows.starts[owner]);
                        assert!(global < rows.starts.get(owner + 1).copied().unwrap_or(total));
                        covered[global] += 1;
                    }
                }
            }
            assert!(covered.iter().all(|&n| n == 1));
        }
    }

    #[test]
    fn empty_owners_and_alignment_do_not_change_owner_positions() {
        let rows = Rows::new([1, 2, 7, 9, 17], 36).unwrap();
        let view = rows.partition(8, 7).unwrap();
        assert_eq!(view.lengths, [0, 0, 1, 1, 2]);
        let aligned = view.aligned(8).unwrap();
        assert_eq!(aligned.starts, [0, 0, 0, 8, 16]);
        assert_eq!(aligned.lengths, [0, 0, 8, 8, 8]);
        assert_eq!(aligned.total, 24);
        assert!(rows.partition(0, 0).is_err());
        assert!(rows.partition(8, 8).is_err());
        assert!(view.aligned(0).is_err());
    }
}
