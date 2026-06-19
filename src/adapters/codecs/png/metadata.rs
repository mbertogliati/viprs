use png::{BitDepth, ColorType, Filter, Info, PixelDimensions, Unit};

use crate::domain::codec_options::PngFilterStrategy;
use crate::domain::error::ViprsError;
use crate::domain::image::{ImageMetadata, Interpretation};

use super::state::PNG_XMP_KEYWORD;

/// Map a PNG `ColorType` to a band count.
pub(super) const fn color_type_to_bands(color_type: ColorType) -> u32 {
    match color_type {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb | ColorType::Indexed => 3,
        ColorType::Rgba => 4,
    }
}

/// Map a band count to the corresponding PNG `ColorType`.
pub(super) fn bands_to_color_type(bands: u32) -> Result<ColorType, ViprsError> {
    match bands {
        1 => Ok(ColorType::Grayscale),
        2 => Ok(ColorType::GrayscaleAlpha),
        3 => Ok(ColorType::Rgb),
        4 => Ok(ColorType::Rgba),
        n => Err(ViprsError::Codec(format!(
            "png: unsupported band count {n}"
        ))),
    }
}

const fn png_interpretation(
    color_type: ColorType,
    bit_depth: BitDepth,
    has_srgb: bool,
) -> Interpretation {
    match (color_type, bit_depth) {
        (ColorType::Grayscale | ColorType::GrayscaleAlpha, BitDepth::Sixteen) => {
            Interpretation::Grey16
        }
        (ColorType::Grayscale | ColorType::GrayscaleAlpha, _) => Interpretation::BW,
        (ColorType::Rgb | ColorType::Rgba | ColorType::Indexed, BitDepth::Sixteen) => {
            Interpretation::Rgb16
        }
        (ColorType::Rgb | ColorType::Rgba | ColorType::Indexed, _) => {
            let _ = has_srgb;
            Interpretation::Srgb
        }
    }
}

pub(super) fn build_png_metadata(
    color_type: ColorType,
    bit_depth: BitDepth,
    has_srgb: bool,
    icc_profile: Option<Vec<u8>>,
    exif: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
    xres: Option<f64>,
    yres: Option<f64>,
) -> ImageMetadata {
    ImageMetadata {
        interpretation: Some(png_interpretation(color_type, bit_depth, has_srgb)),
        icc_profile,
        exif,
        xmp,
        xres,
        yres,
        ..ImageMetadata::default()
    }
}

pub(super) fn png_metadata(info: &Info<'_>) -> ImageMetadata {
    let (xres, yres) = match info.pixel_dims {
        Some(pixel_dims) if pixel_dims.unit == Unit::Meter => (
            Some(f64::from(pixel_dims.xppu) / 1_000.0),
            Some(f64::from(pixel_dims.yppu) / 1_000.0),
        ),
        _ => (None, None),
    };

    build_png_metadata(
        info.color_type,
        info.bit_depth,
        info.srgb.is_some(),
        info.icc_profile
            .as_ref()
            .map(|profile| profile.clone().into_owned()),
        info.exif_metadata
            .as_ref()
            .map(|exif| exif.clone().into_owned()),
        png_xmp(info),
        xres,
        yres,
    )
}

fn png_xmp(info: &Info<'_>) -> Option<Vec<u8>> {
    info.utf8_text.iter().find_map(|chunk| {
        if chunk.keyword == PNG_XMP_KEYWORD {
            chunk.get_text().ok().map(String::into_bytes)
        } else {
            None
        }
    })
}

pub(super) const fn png_filter(filter: PngFilterStrategy) -> Filter {
    match filter {
        PngFilterStrategy::Adaptive => Filter::Adaptive,
        PngFilterStrategy::None => Filter::NoFilter,
        PngFilterStrategy::Sub => Filter::Sub,
        PngFilterStrategy::Up => Filter::Up,
        PngFilterStrategy::Avg => Filter::Avg,
        PngFilterStrategy::Paeth => Filter::Paeth,
    }
}

fn pixels_per_mm_to_pixels_per_meter(value: f64) -> Option<u32> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }

    let scaled = (value * 1_000.0).round();
    if scaled > f64::from(u32::MAX) {
        None
    } else {
        Some(scaled as u32)
    }
}

pub(super) fn png_pixel_dims(metadata: &ImageMetadata) -> Option<PixelDimensions> {
    let (Some(xres), Some(yres)) = (metadata.xres, metadata.yres) else {
        return None;
    };
    let xppu = pixels_per_mm_to_pixels_per_meter(xres)?;
    let yppu = pixels_per_mm_to_pixels_per_meter(yres)?;
    Some(PixelDimensions {
        xppu,
        yppu,
        unit: Unit::Meter,
    })
}
