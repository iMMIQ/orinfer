use super::*;
use serde_json::{Value, json};
use std::io::Write;

fn fixture(root: &Path) -> PathBuf {
    let package = root.join("source");
    fs::create_dir_all(package.join("lib")).unwrap();
    let mut library = vec![0; 20];
    library[..6].copy_from_slice(b"\x7fELF\x02\x01");
    library[16..20].copy_from_slice(&[3, 0, 183, 0]);
    fs::write(package.join("lib/model.so"), &library).unwrap();
    fs::write(package.join("kernel.cubin"), b"kernel").unwrap();
    let bundle = safetensors::serialize(
        [(
            "source",
            safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![6], b"source")
                .unwrap(),
        )],
        None,
    )
    .unwrap();
    fs::write(package.join("assets.safetensors"), bundle).unwrap();
    let manifest = json!({
        "schema_version":1, "runtime_abi":1, "target":"sm_87",
        "architecture":"test_family", "compute_policy":"test_policy",
        "config_signature":{}, "prefill_profiles":[], "buffer_contracts":[], "toolchain":{},
        "execution":{"abi_version":1, "package":"test-model", "version":"1",
            "library":{"file":"lib/model.so", "sha256":sha256(&library)}},
        "catalog":{"binding_signature":"0".repeat(64), "dynamic":[],
            "assets":[
                {"file":"kernel.cubin", "sha256":sha256(b"kernel")},
                {"file":"assets.safetensors", "tensor":"source", "sha256":sha256(b"source")}
            ],
            "definitions":[{"module":0, "source":1, "host_abi":1,
                "symbol":"kernel", "block":[32,1,1], "shared_memory_bytes":0,
                "cooperative":false, "parameters":[]}]
        }
    });
    fs::write(
        package.join("package.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    reseal(&package)
}

fn reseal(package: &Path) -> PathBuf {
    let destination = package
        .parent()
        .unwrap()
        .join(sha256(&fs::read(package.join("package.json")).unwrap()));
    if destination != package {
        fs::rename(package, &destination).unwrap();
    }
    destination
}

fn edit_manifest(package: &Path, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let path = package.join("package.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    edit(&mut manifest);
    fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    reseal(package)
}

fn compress(raw: &[u8], path: &Path, xz: bool) {
    let output = File::create(path).unwrap();
    if xz {
        let mut writer =
            lzma_rust2::XzWriter::new(output, lzma_rust2::XzOptions::with_preset(1)).unwrap();
        writer.write_all(raw).unwrap();
        writer.finish().unwrap();
    } else {
        let mut writer = flate2::write::GzEncoder::new(output, flate2::Compression::fast());
        writer.write_all(raw).unwrap();
        writer.finish().unwrap();
    }
}

fn archive(package: &Path, path: &Path, xz: bool) {
    let mut tar = tar::Builder::new(Vec::new());
    tar.append_dir_all(package.file_name().unwrap(), package)
        .unwrap();
    compress(&tar.into_inner().unwrap(), path, xz);
}

fn assert_empty(cache: &Path) {
    assert_eq!(fs::read_dir(cache).unwrap().count(), 0);
}

#[test]
fn native_a2_installation_checks_plan_identity_before_publishing() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let source = root.join("native");
    fs::create_dir(&source).unwrap();
    let identity = json!({"file":"unused", "sha256":"0".repeat(64)});
    let plan = serde_json::to_vec(&json!({
        "schema_version":1, "config":identity, "index":identity, "engram":identity,
        "capacity":128, "layers":1, "ranks":[], "rope":{}
    }))
    .unwrap();
    fs::write(source.join("model.json"), &plan).unwrap();
    for (index, digest) in ["0".repeat(64), sha256(&plan)].into_iter().enumerate() {
        fs::write(
            source.join("package.json"),
            serde_json::to_vec(&json!({
                "schema_version":1, "backend":"ascend",
                "plan":{"file":"model.json", "sha256":digest}
            }))
            .unwrap(),
        )
        .unwrap();
        let package = reseal(&source);
        let path = root.join(format!("native-{index}.tar.xz"));
        archive(&package, &path, true);
        let cache = root.join("installed");
        let error = install(&path, &cache).unwrap_err();
        assert!(error.contains(if index == 0 {
            "sha256 mismatch"
        } else {
            "40-layer TP8"
        }));
        assert_empty(&cache);
        fs::rename(package, &source).unwrap();
    }
}

#[test]
fn roundtrip_xz_and_gzip_and_duplicate_install() {
    for xz in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let package = fixture(temp.path());
        let archive_path = temp.path().join("package.archive");
        archive(&package, &archive_path, xz);
        let cache = temp.path().join("cache");
        let result = install(&archive_path, &cache).unwrap();
        assert_eq!(result.path.file_name(), package.file_name());
        assert_eq!(
            fs::read(result.path.join("kernel.cubin")).unwrap(),
            b"kernel"
        );
        assert!(
            install(&archive_path, &cache)
                .unwrap_err()
                .contains("already installed")
        );
        assert_eq!(fs::read_dir(&cache).unwrap().count(), 1);
        assert_eq!(
            fs::read(result.path.join("kernel.cubin")).unwrap(),
            b"kernel"
        );
    }
}

#[test]
fn failed_validation_never_publishes() {
    for corruption in [
        "manifest",
        "library",
        "kernel",
        "bundle",
        "abi",
        "catalog",
        "target",
        "elf",
        "tensor",
        "conflicting_hash",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut package = fixture(temp.path());
        match corruption {
            "manifest" => fs::write(package.join("package.json"), b"{}").unwrap(),
            "library" => fs::write(package.join("lib/model.so"), b"corrupt").unwrap(),
            "kernel" => fs::write(package.join("kernel.cubin"), b"corrupt").unwrap(),
            "bundle" => fs::write(package.join("assets.safetensors"), b"corrupt").unwrap(),
            "elf" => {
                fs::write(package.join("lib/model.so"), b"not an ELF").unwrap();
                package = edit_manifest(&package, |m| {
                    m["execution"]["library"]["sha256"] = json!(sha256(b"not an ELF"))
                });
            }
            _ => {
                package = edit_manifest(&package, |m| match corruption {
                    "abi" => m["execution"]["abi_version"] = json!(2),
                    "catalog" => m["catalog"]["definitions"][0]["module"] = json!(99),
                    "target" => m["target"] = json!("sm_90"),
                    "tensor" => m["catalog"]["assets"][1]["tensor"] = json!("missing"),
                    "conflicting_hash" => {
                        let mut other = m["catalog"]["assets"][0].clone();
                        other["sha256"] = json!("0".repeat(64));
                        m["catalog"]["assets"].as_array_mut().unwrap().push(other);
                    }
                    _ => unreachable!(),
                })
            }
        }
        let path = temp.path().join("bad.tar.xz");
        archive(&package, &path, true);
        let cache = temp.path().join("cache");
        assert!(install(&path, &cache).is_err(), "{corruption}");
        assert_empty(&cache);
    }
}

#[test]
fn archive_paths_links_and_duplicates_are_rejected() {
    let digest = "a".repeat(64);
    for (path, kind, error) in [
        (
            format!("{digest}/../outside"),
            tar::EntryType::Regular,
            "Unsafe archive path",
        ),
        (
            format!("/{digest}/outside"),
            tar::EntryType::Regular,
            "Unsafe archive path",
        ),
        (
            format!("{digest}/link"),
            tar::EntryType::Symlink,
            "unique regular files",
        ),
        (
            format!("{digest}/link"),
            tar::EntryType::Link,
            "unique regular files",
        ),
        (
            format!("{digest}/fifo"),
            tar::EntryType::Fifo,
            "unique regular files",
        ),
        (
            format!("{digest}/file"),
            tar::EntryType::Regular,
            "unique regular files",
        ),
        (
            format!("{}/file", "b".repeat(64)),
            tar::EntryType::Regular,
            "Expected one package directory",
        ),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut tar = tar::Builder::new(Vec::new());
        for name in [format!("{digest}/file"), path] {
            let mut header = tar::Header::new_gnu();
            // Deliberately bypass tar-rs path validation to construct hostile input.
            header.as_mut_bytes()[..name.len()].copy_from_slice(name.as_bytes());
            header.set_mode(0o644);
            header.set_entry_type(kind);
            header.set_size(0);
            header.set_cksum();
            tar.append(&header, &[][..]).unwrap();
        }
        let path = temp.path().join("bad.tar.xz");
        compress(&tar.into_inner().unwrap(), &path, true);
        let cache = temp.path().join("cache");
        assert!(install(&path, &cache).unwrap_err().contains(error));
        assert_empty(&cache);
        assert!(!temp.path().join("outside").exists());
    }
}

#[test]
fn truncated_compression_footer_is_rejected() {
    for xz in [true, false] {
        let temp = tempfile::tempdir().unwrap();
        let package = fixture(temp.path());
        let path = temp.path().join("truncated.archive");
        archive(&package, &path, xz);
        let mut raw = fs::read(&path).unwrap();
        raw.truncate(raw.len() - 6);
        fs::write(&path, raw).unwrap();
        let cache = temp.path().join("cache");
        assert!(install(&path, &cache).is_err());
        assert_empty(&cache);
    }
}

#[test]
fn restrictive_directory_modes_do_not_prevent_cleanup() {
    let temp = tempfile::tempdir().unwrap();
    let mut tar = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_mode(0);
    header.set_entry_type(tar::EntryType::Directory);
    header.set_size(0);
    header.set_cksum();
    tar.append_data(&mut header, "a".repeat(64), &[][..])
        .unwrap();
    let path = temp.path().join("bad.tar.xz");
    compress(&tar.into_inner().unwrap(), &path, true);
    let cache = temp.path().join("cache");
    assert!(install(&path, &cache).is_err());
    assert_empty(&cache);
}

#[test]
fn concurrent_installs_publish_once() {
    let temp = tempfile::tempdir().unwrap();
    let package = fixture(temp.path());
    let path = temp.path().join("package.tar.xz");
    archive(&package, &path, true);
    let cache = temp.path().join("cache");
    std::thread::scope(|s| {
        let first = s.spawn(|| install(&path, &cache));
        let second = s.spawn(|| install(&path, &cache));
        let outcomes = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(outcomes.iter().filter(|r| r.is_ok()).count(), 1);
        assert!(
            outcomes
                .iter()
                .find_map(|r| r.as_ref().err())
                .unwrap()
                .contains("already installed")
        );
    });
    assert_eq!(fs::read_dir(cache).unwrap().count(), 1);
}
