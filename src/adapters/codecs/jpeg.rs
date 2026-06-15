//! JPEG codec: decode + encode via libjpeg-turbo (`turbojpeg`).
//!
//! Only `BandFormatId::U8` is supported — JPEG stores 8-bit samples.
//! Calling `decode` or `encode` with any other format returns
//! `ViprsError::Codec` immediately, without doing I/O.
//!
//! `LoadOptions::shrink_factor` maps to libjpeg-turbo DCT-domain scaling.
//! `LoadOptions::max_dimension` computes the closest JPEG shrink
//! factor (1/2/4/8). When even 1/8 decode would still exceed the requested bound,
//! viprs uses 1/8 as the closest match. `ImageDecoder::decode_with_options`
//! still returns a resident raster, so shrink-on-load reduces that resident
//! frame but does not turn JPEG decode into a tile-streaming path by itself.
//!
//! EXIF autorotation is also handled in the adapter. The previous decoder path
//! ignored EXIF orientation and emitted pixels exactly as stored in the file.

mod common;
mod decode;
mod encode;

#[cfg(test)]
mod tests;

pub(crate) use common::{apply_exif_orientation, extract_exif_orientation, orient_u8_image};

/// JPEG codec: implements both [`crate::ports::codec::ImageDecoder`] and
/// [`crate::ports::codec::ImageEncoder`].
///
/// This is a zero-sized type — all state is contained in the options passed
/// at call time. It is `Send + Sync` because it holds no mutable state.
pub struct JpegCodec;
