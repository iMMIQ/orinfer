//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orinfer-engine.
mod config;
mod server;
pub use config::{PreprocessingLimits, ServerConfig};
pub use server::run;
