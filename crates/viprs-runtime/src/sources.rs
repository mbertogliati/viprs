//! Concrete image sources used by pipelines and ergonomic APIs.
//!
//! These adapters expose in-memory buffers, decoder-backed streams, generated
//! images, and optional memory-mapped files through the shared `DynImageSource`
//! abstraction.

pub mod any;
pub mod color_source;
pub mod create;
pub mod decoder_source;
pub mod generators;
pub mod memory;
#[cfg(feature = "mmap")]
pub mod mmap;
pub mod tile_cache;
pub mod zero;

pub use generators::{
    BlackSource, EyeSource, GaussPrecision, GaussmatSource, GaussnoiseSource, GreySource,
    IdentitySource, SinesSource, TextSource, XyzSource, ZoneSource,
};
