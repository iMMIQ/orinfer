//! Versioned, fixed-shape AOT fixtures. This is not the model manifest format.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Component, Path},
    time::Instant,
};

pub type Result<T> = std::result::Result<T, String>;
pub use crate::cuda::RunReport;

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub target: String,
    pub toolchain: BTreeMap<String, String>,
    pub buffers: Vec<Buffer>,
    pub kernels: Vec<Kernel>,
    pub validation: Validation,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub file: String,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Dtype {
    U8,
    I8,
    F16,
    Bf16,
    F32,
    U32,
    I32,
    U64,
    I64,
}
impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::F16 | Self::Bf16 => 2,
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::U64 | Self::I64 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Buffer<Data = FileIdentity> {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub layout: String,
    pub alignment: u64,
    pub access: Access,
    pub data: Option<Data>,
}
impl<Data> Buffer<Data> {
    pub fn bytes(&self) -> Result<usize> {
        if self.shape.is_empty() || self.shape.contains(&0) {
            return Err(format!("{}: empty shape", self.name));
        }
        self.shape.iter().try_fold(self.dtype.bytes(), |n, d| {
            n.checked_mul(*d)
                .ok_or_else(|| format!("{}: size overflow", self.name))
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Kernel {
    pub name: String,
    pub module: FileIdentity,
    pub source: FileIdentity,
    pub host_abi: FileIdentity,
    pub symbol: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
    pub cooperative: bool,
    pub args: Vec<Argument>,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Argument {
    Buffer { name: String },
    I32 { value: i32 },
    U32 { value: u32 },
    I64 { value: i64 },
    U64 { value: u64 },
    F32 { value: f32 },
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Validation {
    pub output: String,
    pub reference: FileIdentity,
    pub relative_l2_tolerance: f64,
    pub zero_input: String,
    pub repetitions: u32,
}

impl Manifest {
    pub fn validate(&self) -> Result<usize> {
        if self.schema_version != 1 || self.target != "sm_87" {
            return Err("Expected fixture schema 1 and sm_87".into());
        }
        if self.toolchain.is_empty() || self.buffers.is_empty() || self.kernels.is_empty() {
            return Err("Missing toolchain/buffers/kernels".into());
        }
        let mut names = BTreeSet::new();
        let mut bytes = 0usize;
        for b in &self.buffers {
            if b.name.is_empty() || !names.insert(b.name.as_str()) {
                return Err("Duplicate/empty buffer name".into());
            }
            if b.layout.is_empty() || !b.alignment.is_power_of_two() || b.alignment > 256 {
                return Err(format!("{}: unsupported layout/alignment", b.name));
            }
            bytes = bytes
                .checked_add(b.bytes()?)
                .ok_or("Total buffer size overflow")?;
        }
        for k in &self.kernels {
            if k.name.is_empty()
                || k.symbol.is_empty()
                || k.symbol.contains('\0')
                || k.args.is_empty()
            {
                return Err("Invalid kernel name/symbol/arguments".into());
            }
            if k.cooperative {
                return Err("Cooperative launch is not supported by fixture runner".into());
            }
            if k.grid.contains(&0)
                || k.block.contains(&0)
                || k.block
                    .iter()
                    .try_fold(1u32, |n, d| n.checked_mul(*d))
                    .is_none_or(|n| n > 1024)
            {
                return Err(format!("{}: invalid launch dimensions", k.name));
            }
            for a in &k.args {
                match a {
                    Argument::Buffer { name } if !names.contains(name.as_str()) => {
                        return Err(format!("{}: unknown buffer {name}", k.name));
                    }
                    Argument::F32 { value } if !value.is_finite() => {
                        return Err("Nonfinite scalar".into());
                    }
                    _ => {}
                }
            }
        }
        let v = &self.validation;
        if !(v.relative_l2_tolerance.is_finite() && (0.0..=1.0).contains(&v.relative_l2_tolerance))
            || v.repetitions == 0
            || v.repetitions > 1000
        {
            return Err("Invalid validation tolerance/repetitions".into());
        }
        let out = self
            .buffers
            .iter()
            .find(|b| b.name == v.output)
            .ok_or("Unknown validation output")?;
        if !matches!(out.dtype, Dtype::F16 | Dtype::Bf16 | Dtype::F32) || out.access == Access::Read
        {
            return Err("Validation output must be writable f16/bf16/f32".into());
        }
        if v.zero_input == v.output || !names.contains(v.zero_input.as_str()) {
            return Err("Invalid changed-input buffer".into());
        }
        let input = self
            .buffers
            .iter()
            .find(|b| b.name == v.zero_input)
            .unwrap();
        if input.data.is_none() {
            return Err("Changed input requires original data".into());
        }
        Ok(bytes)
    }
}

pub(crate) struct Loaded {
    pub manifest: Manifest,
    pub files: BTreeMap<String, Vec<u8>>,
    pub manifest_sha256: String,
    pub validation_s: f64,
    pub buffer_bytes: usize,
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn read_identity(base: &Path, id: &FileIdentity) -> Result<Vec<u8>> {
    if id.sha256.len() != 64
        || !id
            .sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!("{}: malformed sha256", id.file));
    }
    let canonical = resolve_file(base, &id.file)?;
    let data = fs::read(canonical).map_err(|e| format!("{}: {e}", id.file))?;
    if sha256(&data) != id.sha256 {
        return Err(format!("{}: sha256 mismatch", id.file));
    }
    Ok(data)
}

pub(crate) fn resolve_file(base: &Path, file: &str) -> Result<std::path::PathBuf> {
    let path = Path::new(file);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(format!("{file}: expected relative artifact path"));
    }
    let canonical = base
        .join(path)
        .canonicalize()
        .map_err(|e| format!("{file}: {e}"))?;
    if !canonical.starts_with(base) {
        return Err("Artifact symlink leaves fixture directory".into());
    }
    Ok(canonical)
}

fn load(path: &Path) -> Result<Loaded> {
    let start = Instant::now();
    let raw = fs::read(path).map_err(|e| e.to_string())?;
    let manifest: Manifest = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    let buffer_bytes = manifest.validate()?;
    let base = path
        .canonicalize()
        .map_err(|e| e.to_string())?
        .parent()
        .unwrap()
        .to_owned();
    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut add = |id: &FileIdentity| -> Result<()> {
        if let Some(data) = files.get(&id.file) {
            if sha256(data) != id.sha256 {
                return Err("Conflicting file identities".into());
            }
        } else {
            files.insert(id.file.clone(), read_identity(&base, id)?);
        }
        Ok(())
    };
    for b in &manifest.buffers {
        if let Some(id) = &b.data {
            add(id)?;
        }
    }
    for k in &manifest.kernels {
        for id in [&k.module, &k.source, &k.host_abi] {
            add(id)?;
        }
    }
    add(&manifest.validation.reference)?;
    for b in &manifest.buffers {
        if let Some(id) = &b.data
            && files[&id.file].len() != b.bytes()?
        {
            return Err(format!("{}: data byte count mismatch", b.name));
        }
    }
    let out = manifest
        .buffers
        .iter()
        .find(|b| b.name == manifest.validation.output)
        .unwrap();
    let refbytes = (out.bytes()? / out.dtype.bytes())
        .checked_mul(4)
        .ok_or("Reference size overflow")?;
    if files[&manifest.validation.reference.file].len() != refbytes {
        return Err("Reference byte count mismatch".into());
    }
    Ok(Loaded {
        manifest,
        files,
        manifest_sha256: sha256(&raw),
        validation_s: start.elapsed().as_secs_f64(),
        buffer_bytes,
    })
}

#[derive(Debug, Serialize)]
pub struct ValidationReport {
    pub manifest_sha256: String,
    pub buffers: usize,
    pub kernels: usize,
    pub buffer_bytes: usize,
    pub validation_s: f64,
    pub gpu_initialized: bool,
}
pub fn validate_artifact(path: &Path) -> Result<ValidationReport> {
    let x = load(path)?;
    Ok(ValidationReport {
        manifest_sha256: x.manifest_sha256,
        buffers: x.manifest.buffers.len(),
        kernels: x.manifest.kernels.len(),
        buffer_bytes: x.buffer_bytes,
        validation_s: x.validation_s,
        gpu_initialized: false,
    })
}
pub fn run_artifact(path: &Path) -> Result<RunReport> {
    crate::cuda::run(load(path)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_matches_standard_vectors() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
    fn sample() -> Manifest {
        serde_json::from_value(serde_json::json!({"schema_version":1,"target":"sm_87","toolchain":{"tilelang":"test"},"buffers":[{"name":"A","dtype":"f16","shape":[1,4],"layout":"row_major","alignment":16,"access":"read","data":{"file":"a.bin","sha256":"0".repeat(64)}},{"name":"C","dtype":"f16","shape":[1,4],"layout":"row_major","alignment":16,"access":"write"}],"kernels":[{"name":"test","module":{"file":"k.cubin","sha256":"0".repeat(64)},"source":{"file":"k.cu","sha256":"0".repeat(64)},"host_abi":{"file":"host.txt","sha256":"0".repeat(64)},"symbol":"kernel","grid":[1,1,1],"block":[128,1,1],"shared_memory_bytes":0,"cooperative":false,"args":[{"kind":"buffer","name":"A"},{"kind":"buffer","name":"C"},{"kind":"i32","value":1}]}],"validation":{"output":"C","reference":{"file":"ref.bin","sha256":"0".repeat(64)},"relative_l2_tolerance":0.002,"zero_input":"A","repetitions":20}})).unwrap()
    }
    #[test]
    fn rejects_unsafe_launch_metadata() {
        let mut x = sample();
        assert_eq!(x.validate().unwrap(), 16);
        x.kernels[0].cooperative = true;
        assert!(x.validate().is_err());
        x.kernels[0].cooperative = false;
        x.kernels[0].grid[0] = 0;
        assert!(x.validate().is_err());
    }
    #[test]
    fn rejects_abi_and_extent_errors() {
        let mut x = sample();
        x.kernels[0].args.push(Argument::Buffer {
            name: "missing".into(),
        });
        assert!(x.validate().is_err());
        let mut x = sample();
        x.buffers[0].shape = vec![usize::MAX, 2];
        assert!(x.validate().is_err());
        let mut x = sample();
        x.validation.zero_input = "C".into();
        assert!(x.validate().is_err());
    }
    #[test]
    fn checks_file_hashes_and_directory_boundary() {
        let dir = std::env::temp_dir().join(format!("orin-artifact-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("data.bin"), [1, 2, 3]).unwrap();
        let base = dir.canonicalize().unwrap();
        let mut id = FileIdentity {
            file: "data.bin".into(),
            sha256: sha256(&[1, 2, 3]),
        };
        assert_eq!(read_identity(&base, &id).unwrap(), [1, 2, 3]);
        id.sha256 = sha256(&[1, 2, 4]);
        assert!(read_identity(&base, &id).is_err());
        id.file = "../data.bin".into();
        assert!(read_identity(&base, &id).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
