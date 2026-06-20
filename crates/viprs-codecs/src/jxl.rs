//! Jxl adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "jxl")]

//! JPEG XL decoder backed by `libjxl` via `jpegxl-rs`.
//!
//! `jxl-oxide` remains on the metadata path because `jpegxl-rs` does not currently
//! expose Exif/XMP box extraction, but pixel decode is delegated to the C reference
//! implementation for parity with libvips throughput.

use std::io::Cursor;

use jpegxl_rs::{ThreadsRunner, decoder_builder};
use jxl_oxide::{AuxBoxData, JxlImage};

use viprs_core::codec_options::LoadOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::ImageDecoder;

const JXL_CODESTREAM_MAGIC: [u8; 2] = [0xFF, 0x0A];
const JXL_CONTAINER_MAGIC: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
];
const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";

/// JPEG XL decoder implementation.
#[derive(Debug, Clone, Copy, Default)]
/// The `JxlCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::jxl::JxlCodec>();
/// ```
pub struct JxlCodec;

#[inline]
fn require_supported_format<F: BandFormat>() -> Result<(), ViprsError> {
    match F::ID {
        BandFormatId::U8 | BandFormatId::U16 => Ok(()),
        _ => Err(ViprsError::Codec(format!(
            "jxl: unsupported format {:?}; only U8 and U16 are supported",
            F::ID
        ))),
    }
}

fn validate_page_selection(opts: &LoadOptions) -> Result<(), ViprsError> {
    let page = opts.page.unwrap_or(0);
    if page > 0 {
        return Err(ViprsError::Codec(format!(
            "jxl: requested page {page}, but only page 0 is currently supported"
        )));
    }

    if let Some(value) = opts.n
        && value != -1
        && value != 1
    {
        return Err(ViprsError::Codec(format!(
            "jxl: n must be 1 or -1 for decode-only support, got {value}"
        )));
    }

    Ok(())
}

fn read_jxl(src: &[u8]) -> Result<JxlImage, ViprsError> {
    JxlImage::builder()
        .read(Cursor::new(src))
        .map_err(|err| ViprsError::Codec(format!("jxl decode: {err}")))
}

fn jxl_interpretation(image: &JxlImage) -> Interpretation {
    let is_wide = image.image_header().metadata.bit_depth.bits_per_sample() > 8;
    match image.pixel_format() {
        jxl_oxide::PixelFormat::Gray | jxl_oxide::PixelFormat::Graya => {
            if is_wide {
                Interpretation::Grey16
            } else {
                Interpretation::BW
            }
        }
        jxl_oxide::PixelFormat::Cmyk | jxl_oxide::PixelFormat::Cmyka => Interpretation::Cmyk,
        jxl_oxide::PixelFormat::Rgb | jxl_oxide::PixelFormat::Rgba => {
            if is_wide {
                Interpretation::Rgb16
            } else {
                Interpretation::Srgb
            }
        }
    }
}

fn exif_metadata(image: &JxlImage) -> Option<Vec<u8>> {
    match image.aux_boxes().first_exif() {
        Ok(AuxBoxData::Data(exif)) => {
            let mut encoded = Vec::with_capacity(EXIF_SIGNATURE.len() + exif.payload().len());
            encoded.extend_from_slice(EXIF_SIGNATURE);
            encoded.extend_from_slice(exif.payload());
            Some(encoded)
        }
        _ => None,
    }
}

fn xmp_metadata(image: &JxlImage) -> Option<Vec<u8>> {
    match image.aux_boxes().first_xml() {
        AuxBoxData::Data(xmp) => Some(xmp.to_vec()),
        _ => None,
    }
}

fn icc_metadata(image: &JxlImage) -> Option<Vec<u8>> {
    image
        .original_icc()
        .map(std::borrow::ToOwned::to_owned)
        .or_else(|| {
            let rendered = image.rendered_icc();
            (!rendered.is_empty()).then_some(rendered)
        })
}

fn jxl_metadata(image: &JxlImage) -> ImageMetadata {
    ImageMetadata {
        interpretation: Some(jxl_interpretation(image)),
        icc_profile: icc_metadata(image),
        exif: exif_metadata(image),
        xmp: xmp_metadata(image),
        n_pages: Some(image.num_loaded_keyframes().max(1) as u32),
        ..ImageMetadata::default()
    }
}

fn decode_samples<F: BandFormat>(src: &[u8]) -> Result<Vec<F::Sample>, ViprsError> {
    thread_local! {
        static JXL_RUNNER: ThreadsRunner<'static> = ThreadsRunner::default();
    }

    JXL_RUNNER.with(|runner| {
        let decoder = decoder_builder()
            .parallel_runner(runner)
            .build()
            .map_err(|err| ViprsError::Codec(format!("jxl decoder init: {err}")))?;

        match F::ID {
            BandFormatId::U8 => {
                let (_, samples) = decoder
                    .decode_with::<u8>(src)
                    .map_err(|err| ViprsError::Codec(format!("jxl decode: {err}")))?;
                bytemuck::allocation::try_cast_vec::<u8, F::Sample>(samples)
                    .map_err(|(err, _)| ViprsError::Codec(format!("jxl: cast error: {err:?}")))
            }
            BandFormatId::U16 => {
                let (_, samples) = decoder
                    .decode_with::<u16>(src)
                    .map_err(|err| ViprsError::Codec(format!("jxl decode: {err}")))?;
                bytemuck::allocation::try_cast_vec::<u16, F::Sample>(samples)
                    .map_err(|(err, _)| ViprsError::Codec(format!("jxl: cast error: {err:?}")))
            }
            other => Err(ViprsError::Codec(format!(
                "jxl: unsupported output format {other:?}; only U8 and U16 are supported"
            ))),
        }
    })
}

impl ImageDecoder for JxlCodec {
    fn format_name(&self) -> &'static str {
        "jxl"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.starts_with(&JXL_CODESTREAM_MAGIC) || header.starts_with(&JXL_CONTAINER_MAGIC)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        require_supported_format::<F>()?;
        validate_page_selection(opts)?;

        let image = read_jxl(src)?;
        let width = image.width();
        let height = image.height();
        let bands = image.pixel_format().channels() as u32;
        let metadata = jxl_metadata(&image);
        let samples = decode_samples::<F>(src)?;

        Image::from_buffer(width, height, bands, samples)
            .map(|image| image.with_metadata(metadata))
            .map_err(|err| ViprsError::Codec(err.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let image = read_jxl(src)?;
        Ok((
            image.width(),
            image.height(),
            image.pixel_format().channels() as u32,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{U8, U16};

    const RGB8_FIXTURE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/images/jxl_lossless_rgb8_2x2.jxl"
    ));
    const GRAY16_FIXTURE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/images/jxl_lossless_gray16_2x2.jxl"
    ));
    const METADATA_FIXTURE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/images/1x1_exif_xmp.jxl"
    ));

    #[test]
    fn sniff_accepts_codestream_and_container_signatures() {
        assert!(JxlCodec.sniff(&JXL_CODESTREAM_MAGIC));
        assert!(JxlCodec.sniff(&JXL_CONTAINER_MAGIC));
        assert!(!JxlCodec.sniff(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn decode_lossless_rgb8_fixture_preserves_pixels() {
        let image = JxlCodec.decode::<U8>(RGB8_FIXTURE).unwrap();

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.bands(), 3);
        assert_eq!(
            image.pixels(),
            &[0, 64, 128, 255, 200, 32, 10, 20, 30, 40, 50, 60]
        );
        assert_eq!(image.metadata().interpretation, Some(Interpretation::Srgb));
    }

    #[test]
    fn decode_lossless_gray16_fixture_preserves_pixels() {
        let image = JxlCodec.decode::<U16>(GRAY16_FIXTURE).unwrap();

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.bands(), 1);
        assert_eq!(image.pixels(), &[0x1234, 0xFEDC, 0xABCD, 0x0001]);
        assert_eq!(
            image.metadata().interpretation,
            Some(Interpretation::Grey16)
        );
    }

    #[test]
    fn decode_container_fixture_exposes_embedded_metadata() {
        let image = JxlCodec.decode::<U8>(METADATA_FIXTURE).unwrap();

        assert_eq!(image.width(), 1);
        assert_eq!(image.height(), 1);
        assert!(!image.pixels().is_empty());
        assert!(
            image
                .metadata()
                .icc_profile
                .as_ref()
                .is_some_and(|icc| !icc.is_empty())
        );
        assert!(
            image
                .metadata()
                .exif
                .as_ref()
                .is_some_and(|exif| exif.starts_with(EXIF_SIGNATURE))
        );
        assert!(
            image
                .metadata()
                .xmp
                .as_ref()
                .is_some_and(|xmp| !xmp.is_empty())
        );
    }

    #[test]
    fn probe_reports_dimensions_and_bands() {
        assert_eq!(JxlCodec.probe(RGB8_FIXTURE).unwrap(), (2, 2, 3));
    }

    #[test]
    fn decode_rejects_nonzero_page_selection() {
        let err = JxlCodec
            .decode_with_options::<U8>(RGB8_FIXTURE, &LoadOptions::default().with_page(1))
            .unwrap_err();
        assert!(err.to_string().contains("page 1"));
    }

    #[test]
    fn decode_rejects_invalid_n_value() {
        let err = JxlCodec
            .decode_with_options::<U8>(RGB8_FIXTURE, &LoadOptions::default().with_n(2))
            .unwrap_err();
        assert!(err.to_string().contains("n must be 1 or -1"));
    }
}
