//! Concrete infrastructure adapters for the viprs runtime.
//!
//! This module groups user-facing façades, codec integrations, schedulers, and
//! concrete image sources/sinks that connect the pure domain model to I/O and
//! execution environments.

/// Shared adapter-side caches and eviction policies.
pub mod cache;
pub mod codecs;
pub(crate) mod concretized_bridge;
pub mod foreign;
#[cfg(feature = "fft")]
pub mod freqfilt;
pub mod image_api;
pub(crate) mod instrumentation;
pub(crate) mod lock_instrumentation;
pub mod pipeline;
pub mod process;
pub mod scheduler;
/// Concrete image sinks for files, memory buffers, and concurrent writers.
pub mod sinks;
/// Concrete image sources, generators, and cache-backed source adapters.
pub mod sources;
