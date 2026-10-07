use serde::{Deserialize, Serialize};
pub type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, Deserialize, Serialize)]
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
    U16,
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
            Self::F16 | Self::Bf16 | Self::U16 => 2,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Argument {
    Buffer { name: String },
    BufferSlice { name: String, offset: usize },
    I32 { value: i32 },
    U32 { value: u32 },
    I64 { value: i64 },
    U64 { value: u64 },
    F32 { value: f32 },
}
