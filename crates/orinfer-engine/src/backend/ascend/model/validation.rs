//! CPU-only content verification for native A2 releases.
use super::{Instruction, Model, load_spec, provider};
use crate::{
    artifact::{AssetReader, FileIdentity, Result, read_identity, resolve_file, sha256},
    model::ValidationReport,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
    time::Instant,
};

#[derive(Deserialize)]
struct Release {
    files: BTreeMap<String, Shard>,
}
#[derive(Deserialize)]
struct Shard {
    bytes: u64,
    sha256: String,
    tensors: BTreeMap<String, String>,
}
#[derive(Deserialize)]
struct Index {
    weight_map: BTreeMap<String, String>,
}

fn verify_shards(root: &Path, index: &Index, release: &Release) -> Result<(usize, usize)> {
    let expected: BTreeSet<_> = index.weight_map.values().collect();
    if expected != release.files.keys().collect() || expected.is_empty() {
        return Err("A2 release shards differ from the weight index".into());
    }
    let mut observed = BTreeMap::new();
    let mut bytes = 0usize;
    let mut chunk = vec![0; 8 << 20];
    for (name, record) in &release.files {
        let path = resolve_file(root, name)?;
        let mut file = File::open(&path).map_err(|e| e.to_string())?;
        if file.metadata().map_err(|e| e.to_string())?.len() != record.bytes {
            return Err(format!("{name}: release shard size differs"));
        }
        let mut hash = Sha256::new();
        loop {
            let count = file.read(&mut chunk).map_err(|e| e.to_string())?;
            if count == 0 {
                break;
            }
            hash.update(&chunk[..count]);
        }
        if hex::encode(hash.finalize()) != record.sha256 {
            return Err(format!("{name}: release shard sha256 mismatch"));
        }
        // SAFETY: validation reads immutable release files; the mapping is local.
        let mapping =
            unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&mapping)
            .map_err(|e| format!("{name}: {e}"))?;
        let tensors: BTreeSet<_> = metadata.tensors().into_keys().collect();
        if tensors != record.tensors.keys().cloned().collect() {
            return Err(format!("{name}: release tensor inventory differs"));
        }
        for tensor in tensors {
            if observed.insert(tensor, name.clone()).is_some() {
                return Err("A2 release contains duplicate tensor names".into());
            }
        }
        bytes = bytes
            .checked_add(usize::try_from(record.bytes).map_err(|e| e.to_string())?)
            .ok_or("A2 release size overflow")?;
    }
    if observed != index.weight_map {
        return Err("A2 release tensor inventory differs from the weight index".into());
    }
    Ok((observed.len(), bytes))
}

fn verify_assets(model: &Model, base: &Path) -> Result<()> {
    let mut names = BTreeSet::new();
    for rank in &model.ranks {
        for packet in &rank.packets {
            for instruction in packet.body.iter().chain(&packet.head) {
                if let Instruction::Kernel {
                    manifest,
                    uncompacted_manifest,
                    ..
                } = instruction
                {
                    names.insert(manifest.clone());
                    names.extend(uncompacted_manifest.iter().cloned());
                }
            }
        }
    }
    for variants in model.row_selections.values() {
        for variant in variants {
            names.extend(variant.kernels.iter().cloned());
        }
    }
    let mut seen = BTreeMap::new();
    let mut reader = AssetReader::default();
    let mut verify = |root: &Path, id: &FileIdentity| -> Result<()> {
        let key = (resolve_file(root, &id.file)?, id.tensor.clone());
        if let Some(hash) = seen.get(&key) {
            if hash != &id.sha256 {
                return Err("Conflicting A2 asset identities".into());
            }
        } else {
            reader.read(root, id)?;
            seen.insert(key, id.sha256.clone());
        }
        Ok(())
    };
    for name in names {
        let path = resolve_file(base, &name)?;
        let spec: super::super::kernels::Spec = crate::model::read(&path)?;
        let root = path.parent().ok_or("A2 kernel has no directory")?;
        verify(root, &spec.library)?;
        crate::model_package::validate_library(
            &crate::model_package::library_header(&resolve_file(root, &spec.library.file)?)?,
            &spec.target.host_arch,
        )?;
    }
    for id in model.rope.values().chain(std::iter::once(&model.engram)) {
        verify(base, id)?;
    }
    for rank in &model.ranks {
        for definition in &rank.buffers {
            if let Some(id) = &definition.data {
                verify(base, id)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn validate(path: &Path) -> Result<ValidationReport> {
    let start = Instant::now();
    let descriptor = crate::model::resolve_manifest(path)?;
    let cache = descriptor
        .parent()
        .ok_or("A2 model has no cache directory")?;
    let root = cache.parent().ok_or("A2 model has no root directory")?;
    let model: provider::Descriptor = crate::model::read(&descriptor)?;
    if model.schema_version != 1
        || model.architecture != "deepseek_v41"
        || model.compute_policy != "w4a8"
        || model.metadata.target != "ascend910b"
        || model.metadata.vocab != 129280
    {
        return Err("Unsupported native A2 model descriptor".into());
    }
    crate::loader::validate_frontend_assets(root, &model.frontend_assets)?;
    let base = crate::operators::resolve(cache, &model.execution_package)?;
    let package: provider::Package = serde_json::from_slice(&read_identity(
        &base,
        &FileIdentity {
            file: "package.json".into(),
            tensor: None,
            sha256: model.execution_package.clone(),
        },
    )?)
    .map_err(|e| e.to_string())?;
    if package.schema_version != 1 || package.backend != "ascend" {
        return Err("Invalid native A2 execution package".into());
    }
    read_identity(&base, &package.plan)?;
    let (plan, base, layout) = load_spec(&resolve_file(&base, &package.plan.file)?, root)?;
    if plan.capacity != model.metadata.max_context {
        return Err("A2 service context contract differs".into());
    }
    read_identity(root, &plan.config)?;
    let index: Index =
        serde_json::from_slice(&read_identity(root, &plan.index)?).map_err(|e| e.to_string())?;
    if model.frontend_assets.get("tokenizer.json") != Some(&layout.tokenizer_sha256) {
        return Err("Engram/tokenizer identity differs".into());
    }
    verify_assets(&plan, &base)?;
    let release: Release = crate::model::read(&resolve_file(root, "release-manifest.json")?)?;
    let (tensor_count, weight_bytes) = verify_shards(root, &index, &release)?;
    Ok(ValidationReport {
        manifest_sha256: sha256(&std::fs::read(descriptor).map_err(|e| e.to_string())?),
        architecture: model.architecture,
        compute_policy: model.compute_policy,
        execution_package: model.execution_package,
        tensor_count,
        shard_count: release.files.len(),
        weight_bytes,
        validation_s: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, tensor::TensorView};
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Fixture(std::path::PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn fixture() -> (Fixture, Index, Release) {
        let root = std::env::temp_dir().join(format!(
            "orin-a2-validation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&root).unwrap();
        let data = [1, 2, 3];
        let tensor = TensorView::new(Dtype::I8, vec![3], &data).unwrap();
        let payload = safetensors::serialize([("bank", tensor)], None).unwrap();
        std::fs::write(root.join("layer.safetensors"), &payload).unwrap();
        let index = Index {
            weight_map: BTreeMap::from([("bank".into(), "layer.safetensors".into())]),
        };
        let release = Release {
            files: BTreeMap::from([(
                "layer.safetensors".into(),
                Shard {
                    bytes: payload.len() as u64,
                    sha256: sha256(&payload),
                    tensors: BTreeMap::from([("bank".into(), sha256(&data))]),
                },
            )]),
        };
        (Fixture(root.canonicalize().unwrap()), index, release)
    }

    #[test]
    fn release_validation_rejects_same_size_payload_corruption() {
        let (root, index, release) = fixture();
        assert_eq!(verify_shards(&root.0, &index, &release).unwrap().0, 1);
        let path = root.0.join("layer.safetensors");
        let mut payload = std::fs::read(&path).unwrap();
        *payload.last_mut().unwrap() ^= 1;
        std::fs::write(path, payload).unwrap();
        assert!(
            verify_shards(&root.0, &index, &release)
                .unwrap_err()
                .contains("sha256 mismatch")
        );
    }

    #[test]
    fn release_validation_requires_exact_index_and_container_inventory() {
        let (root, mut index, release) = fixture();
        index
            .weight_map
            .insert("missing".into(), "layer.safetensors".into());
        assert!(
            verify_shards(&root.0, &index, &release)
                .unwrap_err()
                .contains("tensor inventory")
        );
        index
            .weight_map
            .insert("missing".into(), "absent.safetensors".into());
        assert!(
            verify_shards(&root.0, &index, &release)
                .unwrap_err()
                .contains("shards differ")
        );
    }
}
