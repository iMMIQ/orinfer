//! Precision policy is a separate selection axis from model family.
use orinfer_model_sdk::artifact::Result;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Policy {
    /// Predominantly INT8, retaining model-specific FP16/FP32 quality paths.
    Int8Quality,
}
impl Policy {
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "int8_quality" => Ok(Self::Int8Quality),
            _ => Err(format!("Unsupported compute policy: {value}")),
        }
    }
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Int8Quality => "int8_quality",
        }
    }
}
