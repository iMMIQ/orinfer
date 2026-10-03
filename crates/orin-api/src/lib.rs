//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orin-engine.
mod server;
pub use server::run;
