//! Install existing schema-1 archives without Python, CUDA or loading native code.
use super::ExecutionPackage;
use crate::artifact::{AssetReader, Result, sha256};
use serde::Serialize;
use std::{
    collections::BTreeSet,
    ffi::CString,
    fs::{self, File},
    io::{self, BufRead, BufReader, Read},
    os::unix::{ffi::OsStrExt, fs::PermissionsExt},
    path::{Component, Path, PathBuf},
};

#[derive(Debug, Serialize)]
pub struct InstalledPackage {
    pub digest: String,
    pub path: PathBuf,
}

/// Validate in a private directory on the cache filesystem, then publish once.
/// Existing packages are never replaced, including concurrent installations.
pub fn install(archive: &Path, cache: &Path) -> Result<InstalledPackage> {
    install_inner(archive, cache).map_err(|e| format!("Package installation: {e}"))
}

fn install_inner(archive: &Path, cache: &Path) -> Result<InstalledPackage> {
    let mut input = BufReader::new(File::open(archive).map_err(|e| e.to_string())?);
    let magic = input.fill_buf().map_err(|e| e.to_string())?;
    let decoder: Box<dyn Read> = if magic.starts_with(b"\xfd7zXZ\0") {
        Box::new(lzma_rust2::XzReader::new_mem_limit(input, true, 256 * 1024))
    } else if magic.starts_with(b"\x1f\x8b") {
        Box::new(flate2::read::MultiGzDecoder::new(input))
    } else {
        return Err("Expected a .tar.xz or .tar.gz execution package".into());
    };
    fs::create_dir_all(cache).map_err(|e| e.to_string())?;
    let staging = tempfile::Builder::new()
        .prefix(".install-")
        .tempdir_in(cache)
        .map_err(|e| e.to_string())?;
    let mut archive = tar::Archive::new(decoder);
    let mut root = None;
    let mut seen = BTreeSet::new();
    for entry in archive.entries().map_err(|e| e.to_string())? {
        let mut entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path().map_err(|e| e.to_string())?.into_owned();
        let mut parts = path.components();
        let Some(Component::Normal(name)) = parts.next() else {
            return Err("Unsafe archive path".into());
        };
        let digest = name.to_str().ok_or("Malformed package digest")?;
        if digest.len() != 64
            || !digest
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("Malformed package digest".into());
        }
        if root.as_deref().is_some_and(|previous| previous != digest) {
            return Err("Expected one package directory".into());
        }
        if parts.any(|p| !matches!(p, Component::Normal(_))) {
            return Err("Unsafe archive path".into());
        }
        root = Some(digest.to_owned());
        let kind = entry.header().entry_type();
        if !(kind.is_file() || kind.is_dir()) || !seen.insert(path.clone()) {
            return Err("Expected unique regular files and directories".into());
        }
        if !entry.unpack_in(staging.path()).map_err(|e| e.to_string())? {
            return Err("Unsafe archive path".into());
        }
        // Packages contain data and dlopen libraries, not standalone executables.
        // Ignore archive permissions so even mode-000 directories remain removable.
        fs::set_permissions(
            staging.path().join(path),
            fs::Permissions::from_mode(if kind.is_dir() { 0o755 } else { 0o644 }),
        )
        .map_err(|e| e.to_string())?;
    }
    // Tar ends before the compression footer. Read through it to verify integrity.
    io::copy(&mut archive.into_inner(), &mut io::sink()).map_err(|e| e.to_string())?;
    let digest = root.ok_or("Empty package archive")?;
    let source = staging
        .path()
        .join(&digest)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    validate(&source, &digest)?;
    let destination = cache.join(&digest);
    rename_new(&source, &destination)?;
    Ok(InstalledPackage {
        digest,
        path: destination.canonicalize().map_err(|e| e.to_string())?,
    })
}

fn validate(root: &Path, digest: &str) -> Result<()> {
    let raw = fs::read(root.join("package.json")).map_err(|e| e.to_string())?;
    if sha256(&raw) != digest {
        return Err("Execution package digest mismatch".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    if value["backend"] == "ascend" {
        let package = serde_json::from_value(value).map_err(|e| e.to_string())?;
        return crate::backend::ascend::model::validation::validate_package(root, &package);
    }
    let package: ExecutionPackage = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    package.validate_metadata()?;
    let execution = &package.execution;
    if execution.abi_version != super::RUNTIME_ABI
        || execution.package.is_empty()
        || execution.version.is_empty()
        || execution.library.tensor.is_some()
    {
        return Err("Incompatible native execution library identity or ABI".into());
    }
    let mut reader = AssetReader::default();
    let target = orinfer_device::Target::from_package(&package.target)?;
    crate::model_package::validate_library(
        &reader.read(root, &execution.library)?,
        &target.host_arch,
    )?;
    let mut checked = BTreeSet::new();
    for asset in &package.catalog.assets {
        if checked.insert((asset.key(), &asset.sha256)) {
            reader.read(root, asset)?;
        }
    }
    Ok(())
}

fn rename_new(source: &Path, destination: &Path) -> Result<()> {
    let source = CString::new(source.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    let destination =
        CString::new(destination.as_os_str().as_bytes()).map_err(|e| e.to_string())?;
    // SAFETY: live NUL-terminated path strings; renameat2 retains neither pointer.
    let status = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if status != 0 {
        let error = io::Error::last_os_error();
        return Err(if error.kind() == io::ErrorKind::AlreadyExists {
            "Execution package already installed; existing files were not changed".into()
        } else {
            error.to_string()
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
