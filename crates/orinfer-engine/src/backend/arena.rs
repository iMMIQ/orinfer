//! Aligned buffer slices share one allocation without sharing arena ownership.
use crate::{
    artifact::{Buffer, Result},
    weights::TensorIdentity,
};
use std::{alloc::Layout, collections::BTreeMap};

pub(crate) struct ArenaLayout {
    pub offsets: BTreeMap<String, usize>,
    pub allocation_bytes: usize,
    pub padding_bytes: usize,
    alignment: usize,
}
impl ArenaLayout {
    pub fn new<'a>(buffers: impl Iterator<Item = &'a Buffer<TensorIdentity>>) -> Result<Self> {
        Self::try_from_sizes(buffers.map(|buffer| {
            Ok((
                buffer.name.as_str(),
                buffer.bytes()?,
                usize::try_from(buffer.alignment).map_err(|e| e.to_string())?,
            ))
        }))
    }
    pub fn from_sizes<'a>(buffers: impl Iterator<Item = (&'a str, usize, usize)>) -> Result<Self> {
        Self::try_from_sizes(buffers.map(Ok))
    }
    fn try_from_sizes<'a>(
        buffers: impl Iterator<Item = Result<(&'a str, usize, usize)>>,
    ) -> Result<Self> {
        let mut layout = Layout::from_size_align(0, 1).map_err(|e| e.to_string())?;
        let mut offsets = BTreeMap::new();
        let mut payload = 0usize;
        for buffer in buffers {
            let (name, bytes, alignment) = buffer?;
            let next = Layout::from_size_align(bytes, alignment).map_err(|e| e.to_string())?;
            let (extended, offset) = layout.extend(next).map_err(|e| e.to_string())?;
            if offsets.insert(name.to_owned(), offset).is_some() {
                return Err("Duplicate arena buffer".into());
            }
            payload = payload.checked_add(bytes).ok_or("Buffer arena overflow")?;
            layout = extended;
        }
        // CUDA guarantees 256-byte alignment. Other callers reserve base
        // alignment headroom explicitly. Larger contracts add their own here.
        let extra = if layout.align() > 256 {
            layout.align() - 1
        } else {
            0
        };
        let allocation_bytes = layout
            .size()
            .checked_add(extra)
            .ok_or("Buffer arena overflow")?;
        Ok(Self {
            offsets,
            allocation_bytes,
            padding_bytes: allocation_bytes - payload,
            alignment: layout.align(),
        })
    }

    pub fn base(&self, allocation: u64) -> Result<u64> {
        let mask = self.alignment as u64 - 1;
        allocation
            .checked_add(mask)
            .map(|p| p & !mask)
            .ok_or_else(|| "Buffer arena address overflow".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::{Access, Dtype};

    fn buffer(name: &str, bytes: usize, alignment: u64) -> Buffer<TensorIdentity> {
        Buffer {
            name: name.into(),
            dtype: Dtype::U8,
            shape: vec![bytes],
            layout: "contiguous".into(),
            alignment,
            access: Access::Read,
            data: None,
        }
    }

    #[test]
    fn aligned_slices_do_not_overlap_or_escape_the_owner_allocation() {
        let buffers = [buffer("a", 3, 256), buffer("b", 37, 512), buffer("c", 9, 1)];
        let arena = ArenaLayout::new(buffers.iter()).unwrap();
        let base = arena.base(256).unwrap();
        let mut previous_end = 256;
        for buffer in &buffers {
            let address = base + arena.offsets[&buffer.name] as u64;
            assert_eq!(address % buffer.alignment, 0);
            assert!(address >= previous_end);
            previous_end = address + buffer.bytes().unwrap() as u64;
            assert!(previous_end <= 256 + arena.allocation_bytes as u64);
        }
        assert_eq!(arena.padding_bytes, arena.allocation_bytes - 49);
        assert!(arena.base(u64::MAX).is_err());
    }

    #[test]
    fn empty_invalid_and_duplicate_layouts_are_checked_without_cuda() {
        let arena = ArenaLayout::new(std::iter::empty()).unwrap();
        assert_eq!(arena.allocation_bytes, 0);
        for buffers in [
            vec![buffer("a", 3, 3)],
            vec![buffer("a", usize::MAX, 256)],
            vec![buffer("a", 3, 256), buffer("a", 3, 256)],
        ] {
            assert!(ArenaLayout::new(buffers.iter()).is_err());
        }
    }
}
