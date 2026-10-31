//! Read-only original checkpoint tensors, before physical execution transforms.
use super::{Index, Shard};
use crate::artifact::{Result, resolve_file};
use memmap2::MmapOptions;
use safetensors::{Dtype, SafeTensors};
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};

pub(crate) struct Checkpoint {
    base: PathBuf,
    index: Index,
    shards: BTreeMap<String, Shard>,
}
pub(crate) struct Tensor<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub data: &'a [u8],
}
impl Checkpoint {
    pub fn open(base: &Path, index: &str) -> Result<Self> {
        let base = base.canonicalize().map_err(|e| e.to_string())?;
        let index = crate::model::read(&resolve_file(&base, index)?)?;
        Ok(Self {
            base,
            index,
            shards: BTreeMap::new(),
        })
    }
    pub fn read(&mut self, name: &str) -> Result<Tensor<'_>> {
        let file = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| format!("Checkpoint tensor missing: {name}"))?
            .clone();
        self.file_tensor(&file, name)
    }
    pub fn path(&self, name: &str) -> Result<PathBuf> {
        let file = self
            .index
            .weight_map
            .get(name)
            .ok_or_else(|| format!("Checkpoint tensor missing: {name}"))?;
        resolve_file(&self.base, file)
    }
    /// Ordered contiguous row shards retain the logical table's original IDs.
    pub fn row_sources(&self, name: &str) -> Result<Vec<(PathBuf, String)>> {
        let names = self.index.row_shards.get(name);
        let parts = names.cloned().unwrap_or_else(|| vec![name.to_owned()]);
        let unique: std::collections::BTreeSet<_> = parts.iter().collect();
        if parts.is_empty() || unique.len() != parts.len() {
            return Err("Empty or duplicate host-row shards".into());
        }
        parts.into_iter().map(|n| Ok((self.path(&n)?, n))).collect()
    }
    pub fn source(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
    ) -> Result<super::TensorSource> {
        let tensor = self.read(name)?;
        if format!("{:?}", tensor.dtype) != dtype || tensor.shape != shape {
            return Err(format!("{name}: prepared tensor dtype/shape differs"));
        }
        let file = &self.index.weight_map[name];
        let shard = &self.shards[file];
        let (start, end) = shard.metadata.info(name).unwrap().data_offsets;
        Ok(super::TensorSource {
            path: shard.path.clone(),
            offset: shard.data_start + start,
            bytes: end - start,
        })
    }
    pub fn file_tensor(&mut self, file: &str, name: &str) -> Result<Tensor<'_>> {
        if !self.shards.contains_key(file) {
            let path = resolve_file(&self.base, file)?;
            let input = File::open(&path).map_err(|e| e.to_string())?;
            // SAFETY: Checkpoint assets are immutable throughout execution. This
            // owner retains each read-only mapping for every borrowed tensor.
            let mapping = unsafe { MmapOptions::new().map(&input) }.map_err(|e| e.to_string())?;
            let (header, metadata) =
                SafeTensors::read_metadata(&mapping).map_err(|e| e.to_string())?;
            self.shards.insert(
                file.to_owned(),
                Shard {
                    path,
                    mapping,
                    metadata,
                    data_start: 8 + header,
                },
            );
        }
        let shard = &self.shards[file];
        let info = shard
            .metadata
            .info(name)
            .ok_or_else(|| format!("{file}: missing {name}"))?;
        let (start, end) = info.data_offsets;
        Ok(Tensor {
            dtype: info.dtype,
            shape: &info.shape,
            data: &shard.mapping[shard.data_start + start..shard.data_start + end],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_sources_validate_header_identity_and_indexed_row_parts() {
        use safetensors::tensor::{TensorView, serialize};
        let dir = tempfile::tempdir().unwrap();
        let payload = [1u8, 2, 3, 4];
        std::fs::write(
            dir.path().join("a.safetensors"),
            serialize(
                [
                    (
                        "a",
                        TensorView::new(Dtype::I8, vec![2, 2], &payload).unwrap(),
                    ),
                    (
                        "b",
                        TensorView::new(Dtype::I8, vec![2, 2], &payload).unwrap(),
                    ),
                ],
                None,
            )
            .unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("index.json"), br#"{"weight_map":{"a":"a.safetensors","b":"a.safetensors"},"row_shards":{"table":["a","b"]}}"#).unwrap();
        let mut checkpoint = Checkpoint::open(dir.path(), "index.json").unwrap();
        let source = checkpoint.source("a", "I8", &[2, 2]).unwrap();
        let bytes = std::fs::read(&source.path).unwrap();
        assert_eq!(&bytes[source.offset..source.offset + source.bytes], payload);
        assert!(checkpoint.source("a", "U8", &[2, 2]).is_err());
        assert!(checkpoint.source("a", "I8", &[4]).is_err());
        let parts = checkpoint.row_sources("table").unwrap();
        assert_eq!(
            parts.iter().map(|(_, n)| n.as_str()).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }
}
