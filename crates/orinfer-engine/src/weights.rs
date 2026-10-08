//! HF sharded safetensors containing the immutable, physical kernel layouts.
use crate::artifact::{Buffer, Dtype, Result, resolve_file, sha256};
use memmap2::{Mmap, MmapOptions};
use safetensors::{SafeTensors, tensor::Metadata};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::File,
    path::{Path, PathBuf},
};

pub use orinfer_model_sdk::model::TensorIdentity;

#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorSource {
    pub path: PathBuf,
    pub offset: usize,
    pub bytes: usize,
}

struct Shard {
    path: PathBuf,
    mapping: Mmap,
    metadata: Metadata,
    data_start: usize,
}

pub(crate) struct Weights {
    base: PathBuf,
    index: Index,
    shards: BTreeMap<String, Shard>,
    verify_payloads: bool,
}

impl Weights {
    pub(crate) fn open(base: &Path, verify_payloads: bool) -> Result<Self> {
        let cache = base.canonicalize().map_err(|e| e.to_string())?;
        let base = cache
            .join("weights")
            .canonicalize()
            .map_err(|e| format!("weights: {e}"))?;
        if !base.starts_with(&cache) {
            return Err("Weights directory leaves model cache".into());
        }
        let index_path = resolve_file(&base, "model.safetensors.index.json")?;
        let index: Index = crate::model::read(&index_path)?;
        if index.weight_map.is_empty() {
            return Err("Empty safetensors weight_map".into());
        }
        Ok(Self {
            base,
            index,
            shards: BTreeMap::new(),
            verify_payloads,
        })
    }

    /// Validate the standard container and binding without faulting tensor payload pages.
    pub(crate) fn source(&mut self, buffer: &Buffer<TensorIdentity>) -> Result<TensorSource> {
        let id = buffer
            .data
            .as_ref()
            .ok_or("Buffer has no tensor identity")?;
        let file = self
            .index
            .weight_map
            .get(&id.tensor)
            .ok_or_else(|| format!("{}: absent from weight_map", id.tensor))?;
        if !self.shards.contains_key(file) {
            let path = resolve_file(&self.base, file)?;
            let input = File::open(&path).map_err(|e| format!("{file}: {e}"))?;
            // SAFETY: Prepared cache files are immutable while the model loads.
            // The read-only mapping owns its lifetime; tensor views never outlive it.
            let mapping =
                unsafe { MmapOptions::new().map(&input) }.map_err(|e| format!("{file}: {e}"))?;
            let (header, metadata) =
                SafeTensors::read_metadata(&mapping).map_err(|e| format!("{file}: {e}"))?;
            self.shards.insert(
                file.clone(),
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
            .info(&id.tensor)
            .ok_or_else(|| format!("{}: missing in {file}", id.tensor))?;
        let dtype = match buffer.dtype {
            Dtype::U8 => safetensors::Dtype::U8,
            Dtype::I8 => safetensors::Dtype::I8,
            Dtype::U16 => safetensors::Dtype::U16,
            Dtype::F16 => safetensors::Dtype::F16,
            Dtype::Bf16 => safetensors::Dtype::BF16,
            Dtype::F32 => safetensors::Dtype::F32,
            Dtype::U32 => safetensors::Dtype::U32,
            Dtype::I32 => safetensors::Dtype::I32,
            Dtype::U64 => safetensors::Dtype::U64,
            Dtype::I64 => safetensors::Dtype::I64,
        };
        if info.dtype != dtype || info.shape != buffer.shape {
            return Err(format!("{}: safetensors dtype/shape mismatch", buffer.name));
        }
        let layout = shard
            .metadata
            .metadata()
            .as_ref()
            .and_then(|metadata| metadata.get(&format!("orin.layout.{}", id.tensor)));
        if layout != Some(&buffer.layout) {
            return Err(format!(
                "{}: safetensors physical layout mismatch",
                buffer.name
            ));
        }
        // The standard parser has validated all offsets, extents and shape products.
        let (begin, end) = info.data_offsets;
        if end - begin != buffer.bytes()? {
            return Err(format!("{}: tensor byte length mismatch", buffer.name));
        }
        Ok(TensorSource {
            path: shard.path.clone(),
            offset: shard.data_start + begin,
            bytes: end - begin,
        })
    }

    pub(crate) fn read<'a>(&'a mut self, buffer: &Buffer<TensorIdentity>) -> Result<&'a [u8]> {
        let source = self.source(buffer)?;
        let id = buffer
            .data
            .as_ref()
            .ok_or("Buffer has no tensor identity")?;
        let shard = &self.shards[&self.index.weight_map[&id.tensor]];
        let data = &shard.mapping[source.offset..source.offset + source.bytes];
        if self.verify_payloads && sha256(data) != id.sha256 {
            return Err(format!("{}: tensor sha256 mismatch", buffer.name));
        }
        Ok(data)
    }

    pub(crate) fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use std::{
        collections::HashMap,
        fs,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "orin-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(path.join("weights")).unwrap();
            Self(path)
        }
        fn write(&self, data: &[u8], dtype: safetensors::Dtype, shape: Vec<usize>) {
            let metadata = HashMap::from([("orin.layout.P".into(), "packed_u4".into())]);
            let tensor = TensorView::new(dtype, shape, data).unwrap();
            let bytes = safetensors::serialize([("P", tensor)], Some(metadata)).unwrap();
            fs::write(self.0.join("weights/shard.safetensors"), bytes).unwrap();
            self.index("shard.safetensors");
        }
        fn index(&self, file: &str) {
            fs::write(
                self.0.join("weights/model.safetensors.index.json"),
                serde_json::to_vec(
                    &serde_json::json!({"metadata":{"total_size":8},"weight_map":{"P":file}}),
                )
                .unwrap(),
            )
            .unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn buffer(data: &[u8], dtype: &str, shape: &[usize]) -> Buffer<TensorIdentity> {
        serde_json::from_value(serde_json::json!({"name":"P","dtype":dtype,"shape":shape,"layout":"packed_u4","alignment":256,"access":"read","data":{"tensor":"P","sha256":sha256(data)}})).unwrap()
    }

    #[test]
    fn preserves_packed_and_float_bits_and_reuses_mapping() {
        for (dtype, name, data) in [
            (
                safetensors::Dtype::I32,
                "i32",
                vec![0x10, 0x32, 0x54, 0x76, 0xff, 0xfe, 0xff, 0xff],
            ),
            (safetensors::Dtype::F16, "f16", vec![0x00, 0x80, 0x01, 0x7e]),
            (
                safetensors::Dtype::BF16,
                "bf16",
                vec![0x00, 0x80, 0xc1, 0x7f],
            ),
        ] {
            let fixture = Fixture::new();
            fixture.write(&data, dtype, vec![2]);
            let mut weights = Weights::open(&fixture.0, true).unwrap();
            let spec = buffer(&data, name, &[2]);
            assert_eq!(weights.read(&spec).unwrap(), data);
            assert_eq!(weights.read(&spec).unwrap(), data);
            assert_eq!(weights.shard_count(), 1);
        }
    }

    #[test]
    fn rejects_wrong_binding_metadata_and_corrupt_payload() {
        let fixture = Fixture::new();
        let data = [0u8; 8];
        fixture.write(&data, safetensors::Dtype::I32, vec![2]);
        let mut weights = Weights::open(&fixture.0, true).unwrap();
        let mut spec = buffer(&data, "i32", &[2]);
        spec.shape = vec![1, 2];
        assert!(weights.read(&spec).is_err());
        spec.shape = vec![2];
        spec.dtype = Dtype::F32;
        assert!(weights.read(&spec).is_err());
        spec.dtype = Dtype::I32;
        spec.layout = "other_u4".into();
        assert!(weights.read(&spec).is_err());
        spec.layout = "packed_u4".into();
        spec.data.as_mut().unwrap().sha256 = sha256(&[1u8; 8]);
        assert!(weights.read(&spec).is_err());
        spec.data.as_mut().unwrap().tensor = "missing".into();
        assert!(weights.read(&spec).is_err());
    }

    #[test]
    fn structural_mode_skips_only_payload_hash() {
        let fixture = Fixture::new();
        let data = [0u8; 8];
        let mut spec = buffer(&data, "i32", &[2]);
        fixture.write(&[1u8; 8], safetensors::Dtype::I32, vec![2]);
        assert_eq!(
            Weights::open(&fixture.0, false)
                .unwrap()
                .read(&spec)
                .unwrap(),
            [1u8; 8]
        );
        assert!(
            Weights::open(&fixture.0, true)
                .unwrap()
                .read(&spec)
                .is_err()
        );
        let mut weights = Weights::open(&fixture.0, false).unwrap();
        spec.shape = vec![1, 2];
        assert!(weights.read(&spec).is_err());
        spec.shape = vec![2];
        spec.dtype = Dtype::F32;
        assert!(weights.read(&spec).is_err());
        spec.dtype = Dtype::I32;
        spec.layout = "other_u4".into();
        assert!(weights.read(&spec).is_err());
    }

    #[test]
    fn rejects_escaping_index_and_truncated_container() {
        let fixture = Fixture::new();
        fixture.write(&[0u8; 8], safetensors::Dtype::I32, vec![2]);
        let spec = buffer(&[0u8; 8], "i32", &[2]);
        for verify in [false, true] {
            fixture.index("../weights/shard.safetensors");
            assert!(
                Weights::open(&fixture.0, verify)
                    .unwrap()
                    .read(&spec)
                    .is_err()
            );
        }
        fixture.index("shard.safetensors");
        fs::write(fixture.0.join("weights/shard.safetensors"), [0u8; 8]).unwrap();
        for verify in [false, true] {
            assert!(
                Weights::open(&fixture.0, verify)
                    .unwrap()
                    .read(&spec)
                    .is_err()
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_outside_weights_directory() {
        let fixture = Fixture::new();
        fs::write(fixture.0.join("outside.safetensors"), [0u8; 8]).unwrap();
        std::os::unix::fs::symlink(
            "../outside.safetensors",
            fixture.0.join("weights/link.safetensors"),
        )
        .unwrap();
        fixture.index("link.safetensors");
        assert!(
            Weights::open(&fixture.0, true)
                .unwrap()
                .read(&buffer(&[0u8; 8], "i32", &[2]))
                .is_err()
        );
    }
}
