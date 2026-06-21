//! BMP codec — decode and encode via the `image` crate.
//!
//! The adapter supports 8-bit grayscale, RGB, and RGBA images.

use std::io::Cursor;

use image::ExtendedColorType;
use image::ImageDecoder as _;
use image::codecs::bmp::{BmpDecoder, BmpEncoder};

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

/// BMP codec: implements both [`ImageDecoder`] and [`ImageEncoder`].
pub struct BmpCodec;

fn bmp_color_type_to_bands(color_type: image::ColorType) -> Result<u32, ViprsError> {
    match color_type {
        image::ColorType::L8 => Ok(1),
        image::ColorType::Rgb8 => Ok(3),
        image::ColorType::Rgba8 => Ok(4),
        other => Err(ViprsError::Codec(format!(
            "bmp: unsupported decoded color type {other:?}"
        ))),
    }
}

fn bmp_color_type_to_metadata(color_type: image::ColorType) -> ImageMetadata {
    let interpretation = match color_type {
        image::ColorType::L8 => Some(Interpretation::BW),
        image::ColorType::Rgb8 | image::ColorType::Rgba8 => Some(Interpretation::Srgb),
        _ => None,
    };

    ImageMetadata {
        interpretation,
        ..ImageMetadata::default()
    }
}

fn bmp_extended_color_type(bands: u32) -> Result<ExtendedColorType, ViprsError> {
    match bands {
        1 => Ok(ExtendedColorType::L8),
        3 => Ok(ExtendedColorType::Rgb8),
        4 => Ok(ExtendedColorType::Rgba8),
        other => Err(ViprsError::Codec(format!(
            "bmp: unsupported band count {other} (expected 1, 3, or 4)"
        ))),
    }
}

fn decode_bmp(src: &[u8]) -> Result<(u32, u32, u32, ImageMetadata, Vec<u8>), ViprsError> {
    let decoder = BmpDecoder::new(Cursor::new(src))
        .map_err(|error| ViprsError::Codec(format!("bmp decode: {error}")))?;
    let (width, height) = decoder.dimensions();
    let color_type = decoder.color_type();
    let bands = bmp_color_type_to_bands(color_type)?;
    let total_bytes = usize::try_from(decoder.total_bytes())
        .map_err(|_| ViprsError::Codec("bmp decode: image too large".into()))?;
    let mut pixels = vec![0u8; total_bytes];
    decoder
        .read_image(&mut pixels)
        .map_err(|error| ViprsError::Codec(format!("bmp decode: {error}")))?;
    Ok((
        width,
        height,
        bands,
        bmp_color_type_to_metadata(color_type),
        pixels,
    ))
}

impl ImageDecoder for BmpCodec {
    fn format_name(&self) -> &'static str {
        "bmp"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.starts_with(b"BM")
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "bmp: unsupported format {:?}; only U8 is supported",
                F::ID
            )));
        }

        let (width, height, bands, metadata, pixels) = decode_bmp(src)?;
        let pixels = bytemuck::cast_vec::<u8, F::Sample>(pixels);
        Image::from_buffer(width, height, bands, pixels)
            .map(|image| image.with_metadata(metadata))
            .map_err(|error| ViprsError::Codec(error.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let decoder = BmpDecoder::new(Cursor::new(src))
            .map_err(|error| ViprsError::Codec(format!("bmp probe: {error}")))?;
        let (width, height) = decoder.dimensions();
        let bands = bmp_color_type_to_bands(decoder.color_type())?;
        Ok((width, height, bands))
    }
}

impl ImageEncoder for BmpCodec {
    fn format_name(&self) -> &'static str {
        "bmp"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "bmp: unsupported format {:?}; only U8 is supported",
                F::ID
            )));
        }

        let color_type = bmp_extended_color_type(image.bands())?;
        let pixel_bytes = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());
        let mut output = Vec::new();
        BmpEncoder::new(&mut output)
            .encode(pixel_bytes, image.width(), image.height(), color_type)
            .map_err(|error| ViprsError::Codec(format!("bmp encode: {error}")))?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::U8;

    #[test]
    fn bmp_round_trip_rgb_u8() {
        let codec = BmpCodec;
        let input = Image::<U8>::from_buffer(
            2,
            2,
            3,
            vec![
                255, 0, 0, 0, 255, 0, //
                0, 0, 255, 255, 255, 0,
            ],
        )
        .unwrap();

        let encoded = codec.encode(&input).unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), input.pixels());
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Srgb)
        );
    }
}
