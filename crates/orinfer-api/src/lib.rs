//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orinfer-engine.
mod config;
pub mod embeddings;
mod encoder_service;
pub mod manager;
pub mod reranking;
mod server;
#[cfg(test)]
mod test_fixtures;
pub use config::{BatchConfig, BatchOrder, PreprocessingLimits, ServerConfig};
pub use server::batch::{BatchReport, run as run_batch};
pub use server::run;
pub use server::startup::has_live_model_workers;
