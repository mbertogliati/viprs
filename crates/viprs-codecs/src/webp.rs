//! WebP codec: decode and encode via the `webp` 0.3 crate (libwebp bindings).
//!
//! # Feature gate
//!
//! This file is compiled only when the `webp` Cargo feature is enabled.
//!
//! # Format support
//!
//! Only `U8` pixel data is supported — WebP is an 8-bit format. Calling any
//! method with a non-U8 `F: BandFormat` returns `ViprsError::Codec`.
//!
//! Band counts:
//! - 1 band (grayscale) — expanded to RGB by tripling the channel before
//!   encoding; decoded output is always 3-band RGB (MVP).
//! - 3 bands (RGB)      — passed directly to `Encoder::from_rgb`.
//! - 4 bands (RGBA)     — passed directly to `Encoder::from_rgba`.
//! - other              — `ViprsError::Codec`.
//!
//! Decoded band count is inferred from the WebP stream: images with an alpha
//! channel decode to 4 bands (RGBA); images without alpha decode to 3 bands (RGB).
//!
//! `LoadOptions::shrink_factor` maps to `WebPDecoderConfig.options.use_scaling`
//! for static WebP images. Animated WebP uses `WebPDemuxGetFrame`
//! plus per-fragment `WebPDecode` scaling and canvas composition. As in
//! libvips `webp2vips.c`, shrink-on-load is disabled when any frame is smaller
//! than the canvas because fragment-local scaling would not match whole-canvas
//! scaling semantics. Full-resolution RGB region reads stream rows incrementally;
//! RGBA or shrink-on-load region reads use a cached full-frame decode because
//! libwebp cannot randomly access partial rows for those combinations.
//!
//! # Lossless vs lossy
//!
//! `encode` uses lossy encoding at quality 75 and method 4 by default, matching libvips.
//! Use [`WebpEncodeOptions`] or `encode_with_options()` to override quality,
//! method, and lossless mode.

mod animated;
mod common;
mod encode;
mod static_decode;

#[cfg(test)]
mod tests;

pub use encode::WebpEncodeOptions;

#[cfg(test)]
pub(crate) use common::{
    checked_webp_scratch_allocation_len, set_test_webp_max_scratch_allocation_bytes,
    set_test_webp_max_total_animation_bytes, test_webp_max_scratch_allocation_bytes_override,
    test_webp_max_total_animation_bytes_override, webp_anim_shrink_on_load_plan,
    webp_shrink_on_load_plan,
};
#[cfg(all(test, feature = "_integration"))]
pub(crate) use common::{
    reset_webp_static_region_frame_decode_count, webp_static_region_frame_decode_count,
};
#[cfg(test)]
pub(crate) use static_decode::decode_static_webp_pixels;

/// WebP codec: implements both [`viprs_ports::codec::ImageDecoder`] and [`viprs_ports::codec::ImageEncoder`].
///
/// Zero-sized: all state is derived from the input data at call time.
pub struct WebpCodec;
