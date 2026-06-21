//! PNG codec — encode via the `png` 0.18 crate and optionally decode via libspng.
//!
//! Supports U8 (8-bit) and U16 (16-bit) band formats.
//! PNG is always lossless; quality/lossy options are silently ignored.
//!
//! Gated behind the `png` Cargo feature flag. When `libspng` is enabled, eager
//! decode prefers libspng and falls back to the `png` crate for unsupported
//! layouts.
//! Interlaced Adam7 region reads are treated as eager random access: the codec
//! materializes one deinterlaced raster and reuses it for subsequent tiles.

mod decode_full;
mod encode;
mod metadata;
mod region_decode;
mod state;

#[cfg(test)]
mod tests;

pub use self::state::{PngCodec, PngEncoder};
