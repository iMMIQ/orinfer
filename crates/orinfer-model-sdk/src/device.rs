//! Device and execution contracts. This crate never loads an accelerator driver.
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

pub type Result<T> = std::result::Result<T, String>;

/// Process-local ownership of device allocations. An ordinal alone does not
/// distinguish a restarted worker or two sessions on the same device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceOwner {
    backend: Backend,
    ordinal: u32,
    generation: u64,
}
impl DeviceOwner {
    pub fn new(backend: Backend, ordinal: u32) -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            backend,
            ordinal,
            generation: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        }
    }
    pub fn require(self, owner: Self) -> Result<()> {
        if self != owner {
            return Err(
                "Device allocation belongs to a different backend, device or session".into(),
            );
        }
        Ok(())
    }
}

/// CUDA serialization retains its historical flat report fields.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DeviceInfo {
    Cuda {
        name: String,
        sm: [i32; 2],
        driver_version: i32,
        total_bytes: usize,
    },
    Ascend {
        backend: Backend,
        name: String,
        device: u32,
        soc: String,
        total_bytes: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cuda,
    Ascend,
}
impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Cuda => "cuda",
            Self::Ascend => "ascend",
        })
    }
}
impl FromStr for Backend {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "cuda" => Ok(Self::Cuda),
            "ascend" => Ok(Self::Ascend),
            _ => Err(format!(
                "Unknown device backend {value:?}; expected cuda or ascend"
            )),
        }
    }
}

/// Parse a supported CUDA compute architecture string (`sm_80`..`sm_120`).
pub fn parse_cuda_sm(arch: &str) -> Option<[i32; 2]> {
    match arch {
        "sm_80" => Some([8, 0]),
        "sm_86" => Some([8, 6]),
        "sm_87" => Some([8, 7]),
        "sm_89" => Some([8, 9]),
        "sm_90" => Some([9, 0]),
        "sm_100" => Some([10, 0]),
        "sm_103" => Some([10, 3]),
        "sm_120" => Some([12, 0]),
        _ => None,
    }
}

/// Host ABI and device architecture are independent; an A2 or discrete CUDA host can be ARM or x86.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub host_arch: String,
    pub backend: Backend,
    pub device_arch: String,
}
impl Target {
    pub fn validate(&self) -> Result<()> {
        if !matches!(self.host_arch.as_str(), "aarch64" | "x86_64") {
            return Err("Unsupported execution-package host architecture".into());
        }
        match (self.backend, self.device_arch.as_str()) {
            (Backend::Cuda, arch) if parse_cuda_sm(arch).is_some() => Ok(()),
            (Backend::Ascend, "ascend910b") => Ok(()),
            _ => Err("Device architecture differs from its backend or is unsupported".into()),
        }
    }
    /// CUDA compute capability `[major, minor]` for CUDA targets, or `None` for other backends.
    pub fn cuda_sm(&self) -> Option<[i32; 2]> {
        (self.backend == Backend::Cuda)
            .then(|| parse_cuda_sm(&self.device_arch))
            .flatten()
    }
    /// Whether this target represents Jetson AGX Orin (`sm_87` unified-memory path).
    pub fn is_orin(&self) -> bool {
        self.backend == Backend::Cuda && self.device_arch == "sm_87"
    }
    /// Whether this target represents a non-Orin generic CUDA discrete-GPU target.
    pub fn is_generic_cuda(&self) -> bool {
        self.backend == Backend::Cuda && self.device_arch != "sm_87" && self.cuda_sm().is_some()
    }
    /// Existing schema-1/2 packages retain their original target strings (`sm_87`, `ascend910b`).
    /// Generic CUDA targets accept `sm_80`..`sm_120` (defaulting to `x86_64`) and explicit
    /// host-qualified forms such as `sm_90-aarch64` or `sm_80-x86_64`.
    pub fn from_package(value: &str) -> Result<Self> {
        let (backend, device_arch, host_arch) = match value {
            "sm_87" => (Backend::Cuda, "sm_87", "aarch64"),
            "ascend910b" => (Backend::Ascend, "ascend910b", "aarch64"),
            "ascend910b-x86_64" => (Backend::Ascend, "ascend910b", "x86_64"),
            other => {
                if let Some((arch, host)) = other.split_once('-') {
                    if parse_cuda_sm(arch).is_some() && matches!(host, "aarch64" | "x86_64") {
                        (Backend::Cuda, arch, host)
                    } else {
                        return Err(format!("Unsupported execution-package target {value:?}"));
                    }
                } else if parse_cuda_sm(other).is_some() {
                    (Backend::Cuda, other, "x86_64")
                } else {
                    return Err(format!("Unsupported execution-package target {value:?}"));
                }
            }
        };
        Ok(Self {
            host_arch: host_arch.into(),
            backend,
            device_arch: device_arch.into(),
        })
    }
    pub fn require_host(&self, host: &str) -> Result<()> {
        self.validate()?;
        if self.host_arch != host {
            return Err(format!(
                "Execution package requires {} host; running on {host}",
                self.host_arch
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMode {
    #[default]
    DecodeOnly,
    Full,
    Off,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPhase {
    Prefill,
    Decode,
    Vision,
}
impl GraphMode {
    pub fn uses_graph(self, phase: ExecutionPhase) -> bool {
        match self {
            Self::DecodeOnly => phase == ExecutionPhase::Decode,
            Self::Full => true,
            Self::Off => false,
        }
    }
}
impl fmt::Display for GraphMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::DecodeOnly => "decode_only",
            Self::Full => "full",
            Self::Off => "off",
        })
    }
}
impl FromStr for GraphMode {
    type Err = String;
    fn from_str(value: &str) -> Result<Self> {
        match value {
            "decode_only" => Ok(Self::DecodeOnly),
            "full" => Ok(Self::Full),
            "off" => Ok(Self::Off),
            _ => Err(format!(
                "Invalid graph mode {value:?}; expected decode_only, full or off"
            )),
        }
    }
}

/// Logical extents do not assume a CUDA or CANN physical mapping granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRange {
    pub offset: usize,
    pub bytes: usize,
}
impl MemoryRange {
    pub fn end(&self) -> Result<usize> {
        self.offset
            .checked_add(self.bytes)
            .ok_or_else(|| "Memory extent overflow".into())
    }
    pub fn within(&self, capacity: usize) -> Result<()> {
        if self.end()? > capacity {
            return Err(format!(
                "Memory extent exceeds allocation: offset={}, bytes={}, capacity={capacity}",
                self.offset, self.bytes
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cuda_report_preserves_legacy_flat_fields() {
        let report = DeviceInfo::Cuda {
            name: "Orin".into(),
            sm: [8, 7],
            driver_version: 12060,
            total_bytes: 64 << 30,
        };
        assert_eq!(
            serde_json::to_value(report).unwrap(),
            serde_json::json!({
                "name":"Orin", "sm":[8,7], "driver_version":12060, "total_bytes":64u64 << 30,
            })
        );
    }
    #[test]
    fn target_rejects_mixed_backends_and_host_abis() {
        assert_eq!(
            Target::from_package("sm_87").unwrap().backend,
            Backend::Cuda
        );
        assert_eq!(
            Target::from_package("ascend910b").unwrap().backend,
            Backend::Ascend
        );
        assert!(
            Target::from_package("ascend910b-x86_64")
                .unwrap()
                .require_host("aarch64")
                .is_err()
        );
        assert!(
            Target {
                host_arch: "aarch64".into(),
                backend: Backend::Ascend,
                device_arch: "sm_87".into()
            }
            .validate()
            .is_err()
        );
        assert!(Target::from_package("ascend950").is_err());
        assert!(Target::from_package("sm_75").is_err());
        assert!(Target::from_package("sm_80-riscv64").is_err());
    }
    #[test]
    fn generic_cuda_targets_preserve_orin_and_support_discrete_gpus() {
        let orin = Target::from_package("sm_87").unwrap();
        assert_eq!(orin.host_arch, "aarch64");
        assert_eq!(orin.cuda_sm(), Some([8, 7]));
        assert!(orin.is_orin());
        assert!(!orin.is_generic_cuda());

        for (pkg, expected_sm, expected_host) in [
            ("sm_80", [8, 0], "x86_64"),
            ("sm_86", [8, 6], "x86_64"),
            ("sm_89", [8, 9], "x86_64"),
            ("sm_90", [9, 0], "x86_64"),
            ("sm_90-aarch64", [9, 0], "aarch64"),
            ("sm_100-x86_64", [10, 0], "x86_64"),
            ("sm_120", [12, 0], "x86_64"),
        ] {
            let target = Target::from_package(pkg).unwrap();
            assert_eq!(target.backend, Backend::Cuda);
            assert_eq!(target.cuda_sm(), Some(expected_sm));
            assert_eq!(target.host_arch, expected_host);
            assert!(target.is_generic_cuda());
            assert!(!target.is_orin());
            target.require_host(expected_host).unwrap();
        }
    }
    #[test]
    fn extents_detect_overflow_and_out_of_bounds() {
        assert!(
            MemoryRange {
                offset: usize::MAX,
                bytes: 1
            }
            .end()
            .is_err()
        );
        assert!(
            MemoryRange {
                offset: 12,
                bytes: 5
            }
            .within(16)
            .is_err()
        );
        assert!(
            MemoryRange {
                offset: 12,
                bytes: 4
            }
            .within(16)
            .is_ok()
        );
    }
    #[test]
    fn snapshots_cannot_cross_devices_or_worker_incarnations() {
        let first = DeviceOwner::new(Backend::Cuda, 0);
        assert!(first.require(first).is_ok());
        assert!(first.require(DeviceOwner::new(Backend::Cuda, 0)).is_err());
        assert!(first.require(DeviceOwner::new(Backend::Cuda, 1)).is_err());
        assert!(first.require(DeviceOwner::new(Backend::Ascend, 0)).is_err());
    }
}
