//! Shared model runtime with device-owned execution backends.
//!
//! CUDA executes Orin model packages. Ascend provides CANN discovery and an
//! experimental native DSv4.1 provider through the shared request interface.

pub mod architecture;
pub mod artifact;
pub mod backend;
pub mod embedding;
pub mod error;
pub mod execution;
pub mod host_rows;
mod loader;
pub mod model;
mod model_package;
pub mod mtp;
pub mod operators;
pub mod prefix;
pub mod reranking;
mod runtime;
pub mod sampling;
pub mod scheduler;
pub mod vision;
mod weights;

/// Legacy CUDA inference target. Package targets select execution backends.
pub const TARGET_DEVICE: &str = "Jetson AGX Orin 64GB";
pub const TARGET_ARCH: &str = "aarch64";
pub const CUDA_ARCH: &str = "sm_87";
pub const FIRST_MODEL: &str = "Qwen3.8-27B";
pub const STATUS: &str = "Rust/TileLang resident image/text inference with OpenAI Chat API";

/// Authoritative benchmark plan, embedded without duplicating its thresholds.
pub const BENCHMARK_PLAN: &str = include_str!("../../../configs/benchmark.json");

pub mod audio;

mod encoder_rows;
