//! Shared bounded file reads for CUDA and Ascend weight loaders.
use super::TensorSource;
use crate::artifact::Result;
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::ErrorKind,
    os::unix::fs::{FileExt, OpenOptionsExt},
};
pub(crate) const ALIGNMENT: usize = 4096;
struct ReadFile {
    buffered: File,
    direct: Option<File>,
    size: usize,
}
#[derive(Default)]
pub(crate) struct Reader {
    files: BTreeMap<std::path::PathBuf, ReadFile>,
}
fn unsupported_direct(error: &std::io::Error) -> bool {
    matches!(
        error.raw_os_error(),
        Some(libc::EINVAL | libc::EOPNOTSUPP | libc::ENOSYS)
    )
}

impl Reader {
    /// Reuse shard descriptors and accept EOF tails on sector-aligned direct reads.
    pub(crate) fn read(
        &mut self,
        source: &TensorSource,
        offset: usize,
        bytes: usize,
        target: &mut [u8],
    ) -> Result<usize> {
        let absolute = source
            .offset
            .checked_add(offset)
            .ok_or("Weight offset overflow")?;
        if bytes == 0
            || bytes > target.len().saturating_sub(ALIGNMENT)
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > source.bytes)
        {
            return Err("Weight chunk exceeds tensor".into());
        }
        let aligned = absolute / ALIGNMENT * ALIGNMENT;
        let skip = absolute - aligned;
        let extent = (skip + bytes).next_multiple_of(ALIGNMENT);
        if extent > target.len() {
            return Err("Weight chunk exceeds staging slot".into());
        }
        if !self.files.contains_key(&source.path) && self.files.len() >= 16 {
            self.files.pop_first();
        }
        let file = match self.files.entry(source.path.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let buffered = File::open(&source.path)
                    .map_err(|e| format!("{}: {e}", source.path.display()))?;
                let size = usize::try_from(buffered.metadata().map_err(|e| e.to_string())?.len())
                    .map_err(|_| "Weight shard length overflow")?;
                let direct = match OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&source.path)
                {
                    Ok(file) => Some(file),
                    Err(error) if unsupported_direct(&error) => None,
                    Err(error) => return Err(format!("{}: {error}", source.path.display())),
                };
                entry.insert(ReadFile {
                    buffered,
                    direct,
                    size,
                })
            }
        };
        if absolute
            .checked_add(bytes)
            .is_none_or(|end| end > file.size)
        {
            return Err(format!(
                "{}: truncated weight payload",
                source.path.display()
            ));
        }
        if (target.as_ptr() as usize).is_multiple_of(ALIGNMENT)
            && let Some(direct) = &file.direct
        {
            let read = loop {
                match direct.read_at(&mut target[..extent], aligned as u64) {
                    Err(error) if error.kind() == ErrorKind::Interrupted => continue,
                    result => break result,
                }
            };
            match read {
                Ok(n) if n == extent.min(file.size - aligned) => return Ok(skip),
                Ok(_) => {
                    return Err(format!(
                        "{}: short direct weight read",
                        source.path.display()
                    ));
                }
                Err(error) if unsupported_direct(&error) => {
                    file.direct = None;
                }
                Err(error) => return Err(format!("{}: {error}", source.path.display())),
            }
        }
        file.buffered
            .read_exact_at(&mut target[..bytes], absolute as u64)
            .map_err(|e| format!("{}: {e}", source.path.display()))?;
        Ok(0)
    }
}

/// Adjacent source tensors share a read without sharing destination ownership.
#[derive(Debug)]
pub(crate) struct Piece {
    pub key: usize,
    pub offset: usize,
    pub within: usize,
    pub bytes: usize,
}
#[derive(Debug)]
pub(crate) struct Batch {
    pub source: TensorSource,
    pub pieces: Vec<Piece>,
}
pub(crate) struct Plan {
    sources: std::vec::IntoIter<(usize, TensorSource)>,
    current: Option<(usize, TensorSource, usize)>,
    capacity: usize,
}
impl Plan {
    pub fn new(mut sources: Vec<(usize, TensorSource)>, capacity: usize) -> Result<Self> {
        if capacity == 0
            || sources
                .iter()
                .any(|(_, s)| s.bytes == 0 || s.offset.checked_add(s.bytes).is_none())
        {
            return Err("Invalid weight read plan extent".into());
        }
        sources.sort_by(|a, b| (&a.1.path, a.1.offset).cmp(&(&b.1.path, b.1.offset)));
        Ok(Self {
            sources: sources.into_iter(),
            current: None,
            capacity,
        })
    }
    fn advance(&mut self) {
        if self.current.is_none() {
            self.current = self.sources.next().map(|(key, s)| (key, s, 0));
        }
    }
}
impl Iterator for Plan {
    type Item = Batch;
    fn next(&mut self) -> Option<Batch> {
        self.advance();
        let (_, source, offset) = self.current.as_ref()?;
        let mut batch = Batch {
            source: TensorSource {
                path: source.path.clone(),
                offset: source.offset + offset,
                bytes: 0,
            },
            pieces: vec![],
        };
        loop {
            self.advance();
            let Some((key, source, offset)) = self.current.as_mut() else {
                break;
            };
            if source.path != batch.source.path
                || source.offset + *offset != batch.source.offset + batch.source.bytes
                || batch.source.bytes == self.capacity
            {
                break;
            }
            let bytes = (source.bytes - *offset).min(self.capacity - batch.source.bytes);
            batch.pieces.push(Piece {
                key: *key,
                offset: *offset,
                within: batch.source.bytes,
                bytes,
            });
            batch.source.bytes += bytes;
            *offset += bytes;
            if *offset == source.bytes {
                self.current = None;
            }
        }
        Some(batch)
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    #[test]
    fn adjacent_ranges_merge_and_split_without_reordering_destination_bytes() {
        let source = |path: &str, offset, bytes| TensorSource {
            path: path.into(),
            offset,
            bytes,
        };
        let batches: Vec<_> = Plan::new(
            vec![
                (0, source("b", 4, 5)),
                (2, source("a", 16, 3)),
                (1, source("a", 7, 9)),
            ],
            8,
        )
        .unwrap()
        .collect();
        assert_eq!(
            batches.iter().map(|b| b.source.bytes).collect::<Vec<_>>(),
            vec![8, 4, 5]
        );
        let mut destinations = [vec![0; 5], vec![0; 9], vec![0; 3]];
        for batch in batches {
            for p in batch.pieces {
                for i in 0..p.bytes {
                    destinations[p.key][p.offset + i] = (batch.source.offset + p.within + i) as u8;
                }
            }
        }
        assert_eq!(
            destinations,
            [vec![4, 5, 6, 7, 8], (7..16).collect(), vec![16, 17, 18]]
        );
        assert!(Plan::new(vec![(0, source("a", usize::MAX, 1))], 8).is_err());
        assert!(Plan::new(vec![], 0).is_err());
    }
}
