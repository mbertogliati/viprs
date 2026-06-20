//! Tiff adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "tiff")]

//! TIFF codec — decode and encode via the `tiff` 0.9 crate (pure Rust).
//!
//! Decode support:
//! - sample types: U8, U16, F32
//! - band counts: 1-band grayscale, 3-band RGB, 4-band RGBA/CMYK
//! - multi-page TIFF: `LoadOptions::page` / `LoadOptions::n`
//!
//! Encode support:
//! - sample types: U8, U16, F32
//! - band counts: 1-band grayscale, 3-band RGB, 4-band RGBA (JPEG only for 1/3 band U8)
//! - compression: uncompressed, `LZW`, `Deflate`, `PackBits`, `JPEG`
//! - predictor: horizontal by default for LZW / Deflate, matching libvips
//! - tiled output: `SaveOptions::tile_width` / `SaveOptions::tile_height`
//! - pyramid output: `SaveOptions::pyramid`

use std::borrow::Cow;
use std::cell::RefCell;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::rc::Rc;

use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoder, SamplingFactor};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use tiff::ColorType as TiffColorType;
use tiff::decoder::ifd::Value as TiffValueRef;
use tiff::decoder::{
    BufferLayoutPreference, ChunkType, Decoder, DecodingResult, DecodingSampleType, TiffCodingUnit,
};
use tiff::encoder::colortype::{self as tiff_ct, ColorType};
use tiff::encoder::compression::{CompressionAlgorithm, Deflate, DeflateLevel, Packbits};
use tiff::encoder::{
    DirectoryEncoder, Rational, TiffEncoder as RawTiffEncoder, TiffKind, TiffValue,
};
use tiff::tags::{
    CompressionMethod, PhotometricInterpretation, ResolutionUnit, SampleFormat, Tag, Type,
};
use weezl::{BitOrder, encode::Encoder as WeezlEncoder};

use viprs_core::codec_options::{LoadOptions, SaveOptions};
pub use viprs_core::codec_options::{TiffCompression, TiffPredictor};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, F32, U8, U16};
use viprs_core::image::{Image, ImageMetadata, Interpretation, Region};
use viprs_ports::codec::{ImageDecoder, ImageEncoder, ImageMetadataProbe, TileImageDecoder};

const TIFF_LE_MAGIC: [u8; 4] = [0x49, 0x49, 0x2A, 0x00];
const TIFF_BE_MAGIC: [u8; 4] = [0x4D, 0x4D, 0x00, 0x2A];
const DEFAULT_TIFF_TILE_SIZE: u32 = 128;
const DEFAULT_TIFF_ROWS_PER_STRIP: u32 = 128;
const TIFF_PAGE_NUMBER_TAG: Tag = Tag::Unknown(297);
const TIFF_SUB_IFD_TAG: Tag = Tag::Unknown(330);
const TIFF_ICC_PROFILE_TAG: Tag = Tag::Unknown(34675);
const TIFFTAG_NEWSUBFILETYPE_REDUCED_IMAGE: u32 = 1;

/// TIFF decoder implementation.
#[derive(Debug, Clone, Copy, Default)]
/// The `TiffDecoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::tiff::TiffDecoder>();
/// ```
pub struct TiffDecoder;

/// TIFF encoder implementation.
#[derive(Debug, Clone, Copy)]
/// The `TiffEncoder` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::tiff::TiffEncoder>();
/// ```
pub struct TiffEncoder {
    compression: TiffCompression,
    predictor: TiffPredictor,
}

impl Default for TiffEncoder {
    fn default() -> Self {
        Self {
            compression: TiffCompression::None,
            predictor: TiffPredictor::Horizontal,
        }
    }
}

impl TiffEncoder {
    /// Creates a new encoder with the given compression algorithm.
    #[must_use]
    pub const fn with_compression(compression: TiffCompression) -> Self {
        Self {
            compression,
            predictor: TiffPredictor::Horizontal,
        }
    }

    /// Sets the predictor for the encoder.
    #[must_use]
    pub const fn with_predictor(mut self, predictor: TiffPredictor) -> Self {
        self.predictor = predictor;
        self
    }
}

/// Backward-compatible combined codec used by [`crate::registry::ForeignRegistry`].
///
/// The `TiffCodec` type provides concrete adapter functionality in the `codecs` module.
#[derive(Debug, Clone, Copy, Default)]
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::tiff::TiffCodec>();
/// ```
pub struct TiffCodec {
    encoder: TiffEncoder,
}

impl TiffCodec {
    /// Creates a new codec with the given compression algorithm.
    #[must_use]
    pub const fn with_compression(compression: TiffCompression) -> Self {
        Self {
            encoder: TiffEncoder::with_compression(compression),
        }
    }
}

trait PredictorSample: Copy + bytemuck::Pod {
    fn apply_horizontal_predictor_row(row: &mut [Self], samples_per_pixel: usize);
}

impl PredictorSample for u8 {
    fn apply_horizontal_predictor_row(row: &mut [Self], samples_per_pixel: usize) {
        for index in (samples_per_pixel..row.len()).rev() {
            row[index] = row[index].wrapping_sub(row[index - samples_per_pixel]);
        }
    }
}

impl PredictorSample for u16 {
    fn apply_horizontal_predictor_row(row: &mut [Self], samples_per_pixel: usize) {
        for index in (samples_per_pixel..row.len()).rev() {
            row[index] = row[index].wrapping_sub(row[index - samples_per_pixel]);
        }
    }
}

impl PredictorSample for f32 {
    fn apply_horizontal_predictor_row(row: &mut [Self], samples_per_pixel: usize) {
        for index in (samples_per_pixel..row.len()).rev() {
            row[index] -= row[index - samples_per_pixel];
        }
    }
}

trait PyramidSample: Copy {
    fn average_box(samples: &[Self]) -> Self;
}

impl PyramidSample for u8 {
    fn average_box(samples: &[Self]) -> Self {
        let sum: u32 = samples.iter().copied().map(u32::from).sum();
        (sum / u32::try_from(samples.len()).unwrap_or(1)) as Self
    }
}

impl PyramidSample for u16 {
    fn average_box(samples: &[Self]) -> Self {
        let sum: u64 = samples.iter().copied().map(u64::from).sum();
        (sum / u64::try_from(samples.len()).unwrap_or(1)) as Self
    }
}

impl PyramidSample for f32 {
    fn average_box(samples: &[Self]) -> Self {
        let sum: Self = samples.iter().copied().sum();
        sum / samples.len() as Self
    }
}

struct SubIfdTagValue {
    offset: u32,
    count: usize,
}

impl TiffValue for SubIfdTagValue {
    const BYTE_LEN: u8 = 4;
    const FIELD_TYPE: Type = Type::IFD;

    fn count(&self) -> usize {
        self.count
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Owned(self.offset.to_ne_bytes().to_vec())
    }
}

enum SubIfdPatchTarget {
    Inline,
    Table(u32),
}

#[derive(Clone, Default)]
struct SharedWriteBuffer {
    inner: Rc<RefCell<Cursor<Vec<u8>>>>,
}

impl SharedWriteBuffer {
    fn into_inner(self) -> Vec<u8> {
        match Rc::try_unwrap(self.inner) {
            Ok(buffer) => buffer.into_inner().into_inner(),
            Err(buffer) => buffer.borrow().get_ref().clone(),
        }
    }

    fn with_bytes<R>(
        &self,
        f: impl FnOnce(&[u8]) -> Result<R, ViprsError>,
    ) -> Result<R, ViprsError> {
        let borrowed = self.inner.borrow();
        f(borrowed.get_ref().as_slice())
    }

    fn with_bytes_mut<R>(
        &self,
        f: impl FnOnce(&mut Vec<u8>) -> Result<R, ViprsError>,
    ) -> Result<R, ViprsError> {
        let mut borrowed = self.inner.borrow_mut();
        f(borrowed.get_mut())
    }
}

impl Write for SharedWriteBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.borrow_mut().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.borrow_mut().flush()
    }
}

impl Seek for SharedWriteBuffer {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.inner.borrow_mut().seek(pos)
    }
}

mod decode;
mod encode;
mod pyramid;

use decode::{
    color_type_to_bands, decode_tiff, decode_tiff_region_with_options, is_tiff_header,
    probe_tiff_with_options,
};
use encode::{
    effective_compression, effective_predictor_for_format, encode_tiff_document, pages_for_encode,
    recast_pages_f32, recast_pages_u8, recast_pages_u16, tile_dimensions,
};

#[cfg(test)]
mod tests;

impl ImageDecoder for TiffDecoder {
    fn format_name(&self) -> &'static str {
        "tiff"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        is_tiff_header(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError>
    where
        F::Sample: Clone,
    {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        match F::ID {
            BandFormatId::U8 | BandFormatId::U16 | BandFormatId::F32 => decode_tiff(src, opts),
            other => Err(ViprsError::Codec(format!(
                "tiff: unsupported format {other:?} — only U8, U16, and F32 are supported"
            ))),
        }
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let mut decoder =
            Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
        let (width, height) = decoder
            .dimensions()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        let color_type = decoder
            .colortype()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        let bands = color_type_to_bands(color_type)?;
        Ok((width, height, bands))
    }
}

impl TileImageDecoder for TiffDecoder {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        probe_tiff_with_options(src, opts)
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        decode_tiff_region_with_options::<F>(src, opts, region, output)
    }
}

impl ImageEncoder for TiffEncoder {
    fn format_name(&self) -> &'static str {
        "tiff"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError>
    where
        F::Sample: Clone,
    {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        let compression = effective_compression(self.compression, opts);
        let predictor = effective_predictor_for_format(F::ID, self.predictor, compression, opts);
        let tile = tile_dimensions(opts);
        let pages = pages_for_encode(image)?;

        match (F::ID, image.bands(), compression) {
            (BandFormatId::U8, 1, _) => encode_tiff_document::<tiff_ct::Gray8, U8>(
                &recast_pages_u8(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::U8, 3, _) => encode_tiff_document::<tiff_ct::RGB8, U8>(
                &recast_pages_u8(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::U8, 4, TiffCompression::Jpeg) => Err(ViprsError::Codec(
                "tiff: JPEG compression does not support 4-band images".into(),
            )),
            (BandFormatId::U8, 4, _) => encode_tiff_document::<tiff_ct::RGBA8, U8>(
                &recast_pages_u8(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::U16 | BandFormatId::F32, _, TiffCompression::Jpeg) => Err(
                ViprsError::Codec("tiff: JPEG compression supports only U8 input".into()),
            ),
            (BandFormatId::U16, 1, _) => encode_tiff_document::<tiff_ct::Gray16, U16>(
                &recast_pages_u16(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::U16, 3, _) => encode_tiff_document::<tiff_ct::RGB16, U16>(
                &recast_pages_u16(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::U16, 4, _) => encode_tiff_document::<tiff_ct::RGBA16, U16>(
                &recast_pages_u16(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::F32, 1, _) => encode_tiff_document::<tiff_ct::Gray32Float, F32>(
                &recast_pages_f32(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::F32, 3, _) => encode_tiff_document::<tiff_ct::RGB32Float, F32>(
                &recast_pages_f32(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (BandFormatId::F32, 4, _) => encode_tiff_document::<tiff_ct::RGBA32Float, F32>(
                &recast_pages_f32(&pages)?,
                opts,
                compression,
                predictor,
                tile,
            ),
            (format, bands, _) => Err(ViprsError::Codec(format!(
                "tiff: unsupported encode combination {format:?} with {bands} bands"
            ))),
        }
    }
}

impl ImageDecoder for TiffCodec {
    fn format_name(&self) -> &'static str {
        TiffDecoder.format_name()
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        TiffDecoder.sniff(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError>
    where
        F::Sample: Clone,
    {
        TiffDecoder.decode(src)
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        TiffDecoder.decode_with_options(src, opts)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        TiffDecoder.probe(src)
    }
}

impl TileImageDecoder for TiffCodec {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        TiffDecoder.probe_with_options(src, opts)
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        TiffDecoder.decode_region_into::<F>(src, opts, region, output)
    }
}

impl ImageEncoder for TiffCodec {
    fn format_name(&self) -> &'static str {
        self.encoder.format_name()
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError>
    where
        F::Sample: Clone,
    {
        self.encoder.encode(image)
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
        F::Sample: Clone,
    {
        self.encoder.encode_with_options(image, opts)
    }
}
