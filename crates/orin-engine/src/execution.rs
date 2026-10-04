//! Runtime execution policy, independent of checkpoint and operator packages.
use crate::artifact::Result;
use serde::Serialize;
use std::{fmt, str::FromStr};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CudaGraphMode {
    #[default]
    DecodeOnly,
    Full,
    Off,
}

impl fmt::Display for CudaGraphMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DecodeOnly => "decode_only",
            Self::Full => "full",
            Self::Off => "off",
        })
    }
}

impl FromStr for CudaGraphMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "decode_only" => Ok(Self::DecodeOnly),
            "full" => Ok(Self::Full),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "Invalid CUDA Graph mode {value:?}; expected decode_only, full or off"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadOptions {
    pub cuda_graph: CudaGraphMode,
    /// Maximum extra resident prefix snapshots. Zero disables reuse.
    pub prefix_cache_bytes: usize,
}

pub fn parse_cache_mib(value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_mul(1024 * 1024))
        .ok_or_else(|| "Prefix cache MiB must be a nonnegative integer within usize".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExecutionPhase {
    Prefill,
    Decode,
    Vision,
}

impl CudaGraphMode {
    pub(crate) fn uses_graph(self, phase: ExecutionPhase) -> bool {
        match self {
            Self::DecodeOnly => phase == ExecutionPhase::Decode,
            Self::Full => true,
            Self::Off => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_roundtrip_and_invalid_values_fail() {
        for mode in [
            CudaGraphMode::DecodeOnly,
            CudaGraphMode::Full,
            CudaGraphMode::Off,
        ] {
            assert_eq!(mode.to_string().parse::<CudaGraphMode>().unwrap(), mode);
            assert_eq!(serde_json::to_string(&mode).unwrap(), format!("\"{mode}\""));
        }
        for invalid in ["", "on", "0", "1", "decode", "FULL"] {
            assert!(invalid.parse::<CudaGraphMode>().is_err());
        }
        assert_eq!(LoadOptions::default().cuda_graph, CudaGraphMode::DecodeOnly);
        assert_eq!(parse_cache_mib("0").unwrap(), 0);
        assert_eq!(parse_cache_mib("12288").unwrap(), 12usize << 30);
        for invalid in ["-1", "1.5", "", "18446744073709551615"] {
            assert!(parse_cache_mib(invalid).is_err());
        }
    }

    #[test]
    fn decode_only_obeys_call_phase_even_for_a_reused_program() {
        for phase in [
            ExecutionPhase::Prefill,
            ExecutionPhase::Decode,
            ExecutionPhase::Vision,
        ] {
            assert_eq!(
                CudaGraphMode::DecodeOnly.uses_graph(phase),
                phase == ExecutionPhase::Decode
            );
            assert!(CudaGraphMode::Full.uses_graph(phase));
            assert!(!CudaGraphMode::Off.uses_graph(phase));
        }
    }
}
