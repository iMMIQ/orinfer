//! Orin-only AOT model execution.
//!
//! AOT fixtures and explicit text-model programs execute through the CUDA Driver.

pub mod architecture;
pub mod artifact;
mod cuda;
pub mod error;
pub mod execution;
mod loader;
pub mod model;
pub mod mtp;
pub mod operators;
pub mod ple;
pub mod prefix;
mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod vision;
mod weights;

/// Build target; runtime device discovery must validate it before inference.
pub const TARGET_DEVICE: &str = "Jetson AGX Orin 64GB";
pub const TARGET_ARCH: &str = "aarch64";
pub const CUDA_ARCH: &str = "sm_87";
pub const FIRST_MODEL: &str = "Qwen3.8-27B";
pub const STATUS: &str = "Rust/TileLang resident image/text inference with OpenAI Chat API";

/// Authoritative benchmark plan, embedded without duplicating its thresholds.
pub const BENCHMARK_PLAN: &str = include_str!("../../../configs/benchmark.json");
