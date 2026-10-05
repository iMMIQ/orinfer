//! CPU-side OpenAI Chat protocol adapter. GPU execution belongs to orinfer-engine.
mod server;
pub use server::run;
