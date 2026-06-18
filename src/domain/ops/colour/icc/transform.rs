use super::{IccImage, IccIntent, IccTransformOptions, Interpretation, ViprsError, icc_error};
use crate::domain::{
    format::{BandFormat, BandFormatId, F32, U8, U16},
    image::{Image, ImageMetadata},
};
use lcms2::{ColorSpaceSignature, DisallowCache, Flags, Intent, PixelFormat, Profile, Transform};
use std::borrow::Cow;

use super::profiles::{
    gray_profile_bytes, lab_profile_bytes, open_profile, profile_load, xyz_profile_bytes,
};

#[derive(Clone, Copy)]
pub(super) enum OutputSpec {
    U8Rgb,
    U8Gray,
    U16Rgb,
    U16Gray,
    U8Cmyk,
    U16Cmyk,
    F32Lab,
    F32Xyz,
}

pub(super) const ICC_OUTPUT_TOO_LARGE_DETAILS: &str =
    "ICC transform output dimensions exceed addressable memory";

fn icc_output_too_large(width: u32, height: u32, bands: u32) -> ViprsError {
    ViprsError::ImageTooLarge {
        width,
        height,
        bands,
        bytes: u128::from(width) * u128::from(height) * u128::from(bands),
        limit_bytes: usize::MAX as u128,
        details: ICC_OUTPUT_TOO_LARGE_DETAILS,
    }
}

pub(super) fn checked_icc_output_pixels(
    width: u32,
    height: u32,
    bands: u32,
) -> Result<usize, ViprsError> {
    (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| icc_output_too_large(width, height, bands))
}

pub(super) fn checked_icc_output_sizes(
    width: u32,
    height: u32,
    out_bands: u32,
) -> Result<(usize, usize), ViprsError> {
    let pixels = checked_icc_output_pixels(width, height, out_bands)?;
    let sample_count = (pixels as u64)
        .checked_mul(out_bands as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| icc_output_too_large(width, height, out_bands))?;
    Ok((pixels, sample_count))
}

fn fallback_input_profile<F: BandFormat>(image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
    match (image.metadata().interpretation, image.bands(), F::ID) {
        (Some(Interpretation::Lab), 3, BandFormatId::F32) => lab_profile_bytes(),
        (Some(Interpretation::Xyz), 3, BandFormatId::F32) => xyz_profile_bytes(),
        (_, 3, BandFormatId::U8 | BandFormatId::U16) => profile_load("srgb"),
        (_, 1, BandFormatId::U8 | BandFormatId::U16) => gray_profile_bytes(),
        _ => Err(icc_error(
            "no embedded ICC profile; provide one explicitly via input_profile option",
        )),
    }
}

pub(super) fn selected_intent(
    requested: IccIntent,
    input_profile: &Profile,
    output_profile: &Profile,
) -> Result<Intent, ViprsError> {
    let output_is_pcs = matches!(
        output_profile.color_space(),
        ColorSpaceSignature::LabData | ColorSpaceSignature::XYZData
    );
    let requested_intent = match requested {
        IccIntent::Perceptual => Intent::Perceptual,
        IccIntent::Relative => Intent::RelativeColorimetric,
        IccIntent::Saturation => Intent::Saturation,
        IccIntent::Absolute => Intent::AbsoluteColorimetric,
        IccIntent::Auto => input_profile.header_rendering_intent(),
    };

    if input_profile.is_intent_supported(requested_intent, 0)
        && (output_is_pcs || output_profile.is_intent_supported(requested_intent, 1))
    {
        return Ok(requested_intent);
    }

    let fallback = input_profile.header_rendering_intent();
    if input_profile.is_intent_supported(fallback, 0)
        && (output_is_pcs || output_profile.is_intent_supported(fallback, 1))
    {
        Ok(fallback)
    } else {
        Err(icc_error(format!(
            "no shared rendering intent between {:?} and {:?}",
            input_profile.color_space(),
            output_profile.color_space()
        )))
    }
}

fn output_spec(output_profile: &Profile, depth: Option<u8>) -> Result<OutputSpec, ViprsError> {
    let depth = depth.unwrap_or(8);
    match (output_profile.color_space(), depth) {
        (ColorSpaceSignature::RgbData, 8) => Ok(OutputSpec::U8Rgb),
        (ColorSpaceSignature::RgbData, 16) => Ok(OutputSpec::U16Rgb),
        (ColorSpaceSignature::RgbData, other) => Err(icc_error(format!(
            "RGB ICC export supports depth 8 or 16, got {other}"
        ))),
        (ColorSpaceSignature::GrayData, 8) => Ok(OutputSpec::U8Gray),
        (ColorSpaceSignature::GrayData, 16) => Ok(OutputSpec::U16Gray),
        (ColorSpaceSignature::GrayData, other) => Err(icc_error(format!(
            "Gray ICC export supports depth 8 or 16, got {other}"
        ))),
        (ColorSpaceSignature::CmykData, 8) => Ok(OutputSpec::U8Cmyk),
        (ColorSpaceSignature::CmykData, 16) => Ok(OutputSpec::U16Cmyk),
        (ColorSpaceSignature::CmykData, other) => Err(icc_error(format!(
            "CMYK ICC export supports depth 8 or 16, got {other}"
        ))),
        (ColorSpaceSignature::LabData, _) => Ok(OutputSpec::F32Lab),
        (ColorSpaceSignature::XYZData, _) => Ok(OutputSpec::F32Xyz),
        (other, _) => Err(icc_error(format!(
            "unsupported output colour space {other:?}; supported: RGB, Gray, CMYK, Lab, XYZ"
        ))),
    }
}

fn input_pixel_format<F: BandFormat>(
    image: &Image<F>,
    input_profile: &Profile,
) -> Result<PixelFormat, ViprsError> {
    input_pixel_format_for_layout(F::ID, input_profile.color_space(), image.bands())
}

pub(super) fn input_pixel_format_for_layout(
    format: BandFormatId,
    color_space: ColorSpaceSignature,
    bands: u32,
) -> Result<PixelFormat, ViprsError> {
    match (format, color_space, bands) {
        (BandFormatId::U8, ColorSpaceSignature::RgbData, 3) => Ok(PixelFormat::RGB_8),
        (BandFormatId::U16, ColorSpaceSignature::RgbData, 3) => Ok(PixelFormat::RGB_16),
        (BandFormatId::U8, ColorSpaceSignature::GrayData, 1) => Ok(PixelFormat::GRAY_8),
        (BandFormatId::U16, ColorSpaceSignature::GrayData, 1) => Ok(PixelFormat::GRAY_16),
        (BandFormatId::U8, ColorSpaceSignature::CmykData, 4) => Ok(PixelFormat::CMYK_8),
        (BandFormatId::U16, ColorSpaceSignature::CmykData, 4) => Ok(PixelFormat::CMYK_16),
        (BandFormatId::F32, ColorSpaceSignature::LabData, 3) => Ok(PixelFormat::Lab_FLT),
        (BandFormatId::F32, ColorSpaceSignature::XYZData, 3) => Ok(PixelFormat::XYZ_FLT),
        (format, color_space, bands) => Err(icc_error(format!(
            "unsupported input format {format:?} for {color_space:?} profile with {bands} bands"
        ))),
    }
}

fn metadata_with_profile(
    source: &ImageMetadata,
    profile: &[u8],
    interpretation: Option<Interpretation>,
) -> ImageMetadata {
    let mut metadata = source.clone();
    metadata.icc_profile = Some(profile.to_vec());
    if let Some(interpretation) = interpretation {
        metadata.interpretation = Some(interpretation);
    }
    metadata
}

fn output_interpretation(spec: OutputSpec) -> Option<Interpretation> {
    match spec {
        OutputSpec::U8Rgb | OutputSpec::U16Rgb => Some(Interpretation::Srgb),
        OutputSpec::U8Gray => Some(Interpretation::BW),
        OutputSpec::U16Gray => Some(Interpretation::Grey16),
        OutputSpec::U8Cmyk | OutputSpec::U16Cmyk => Some(Interpretation::Cmyk),
        OutputSpec::F32Lab => Some(Interpretation::Lab),
        OutputSpec::F32Xyz => Some(Interpretation::Xyz),
    }
}

fn replace_embedded_profile(
    image: IccImage,
    profile: &[u8],
    interpretation: Interpretation,
) -> IccImage {
    match image {
        IccImage::U8(image) => {
            let mut metadata = image.metadata().clone();
            metadata.icc_profile = Some(profile.to_vec());
            metadata.interpretation = Some(interpretation);
            IccImage::U8(image.with_metadata(metadata))
        }
        IccImage::U16(image) => {
            let mut metadata = image.metadata().clone();
            metadata.icc_profile = Some(profile.to_vec());
            metadata.interpretation = Some(interpretation);
            IccImage::U16(image.with_metadata(metadata))
        }
        IccImage::F32(image) => {
            let mut metadata = image.metadata().clone();
            metadata.icc_profile = Some(profile.to_vec());
            metadata.interpretation = Some(interpretation);
            IccImage::F32(image.with_metadata(metadata))
        }
    }
}

fn pcs_profile_for_export<F: BandFormat>(image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
    match image.metadata().interpretation {
        Some(Interpretation::Lab) => lab_profile_bytes(),
        Some(Interpretation::Xyz) => xyz_profile_bytes(),
        Some(other) => Err(icc_error(format!(
            "icc_export expects Lab or Xyz PCS input, got {other:?}"
        ))),
        None => Err(icc_error(
            "icc_export requires Lab or Xyz interpretation to infer the PCS profile",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transform_int_input(
    src: &[u8],
    width: u32,
    height: u32,
    in_fmt: PixelFormat,
    meta: &ImageMetadata,
    input_profile: &Profile,
    output_profile: &Profile,
    output_bytes: &[u8],
    intent: Intent,
    flags: Flags<lcms2::DisallowCache>,
    spec: OutputSpec,
) -> Result<IccImage, ViprsError> {
    let interp = output_interpretation(spec);
    match spec {
        OutputSpec::U8Rgb | OutputSpec::U8Gray | OutputSpec::U8Cmyk => {
            let (out_fmt, out_bands) = match spec {
                OutputSpec::U8Rgb => (PixelFormat::RGB_8, 3u32),
                OutputSpec::U8Gray => (PixelFormat::GRAY_8, 1u32),
                OutputSpec::U8Cmyk => (PixelFormat::CMYK_8, 4u32),
                _ => unreachable!(),
            };
            let transform = Transform::<u8, u8, _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                out_fmt,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let (_pixels, sample_count) = checked_icc_output_sizes(width, height, out_bands)?;
            let mut output = vec![0u8; sample_count];
            transform.transform_pixels(src, &mut output);
            let metadata = metadata_with_profile(meta, output_bytes, interp);
            Image::from_buffer(width, height, out_bands, output)
                .map(|img| IccImage::U8(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
        OutputSpec::U16Rgb | OutputSpec::U16Gray | OutputSpec::U16Cmyk => {
            let (out_fmt, out_bands) = match spec {
                OutputSpec::U16Rgb => (PixelFormat::RGB_16, 3u32),
                OutputSpec::U16Gray => (PixelFormat::GRAY_16, 1u32),
                OutputSpec::U16Cmyk => (PixelFormat::CMYK_16, 4u32),
                _ => unreachable!(),
            };
            let transform = Transform::<u8, u8, _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                out_fmt,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let (_pixels, sample_count) = checked_icc_output_sizes(width, height, out_bands)?;
            let mut output = vec![0u16; sample_count];
            transform.transform_pixels(src, bytemuck::cast_slice_mut(&mut output));
            let metadata = metadata_with_profile(meta, output_bytes, interp);
            Image::from_buffer(width, height, out_bands, output)
                .map(|img| IccImage::U16(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
        OutputSpec::F32Lab => {
            let transform = Transform::<u8, [f32; 3], _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                PixelFormat::Lab_FLT,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let pixels = checked_icc_output_pixels(width, height, 3)?;
            let mut output = vec![[0.0f32; 3]; pixels];
            transform.transform_pixels(src, &mut output);
            let output = bytemuck::allocation::try_cast_vec::<[f32; 3], f32>(output)
                .map_err(|(_e, _b)| icc_error("f32 ICC output cast failed"))?;
            let metadata = metadata_with_profile(meta, output_bytes, interp);
            Image::from_buffer(width, height, 3, output)
                .map(|img| IccImage::F32(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
        OutputSpec::F32Xyz => {
            let transform = Transform::<u8, [f32; 3], _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                PixelFormat::XYZ_FLT,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let pixels = checked_icc_output_pixels(width, height, 3)?;
            let mut output = vec![[0.0f32; 3]; pixels];
            transform.transform_pixels(src, &mut output);
            let output = bytemuck::allocation::try_cast_vec::<[f32; 3], f32>(output)
                .map_err(|(_e, _b)| icc_error("f32 ICC output cast failed"))?;
            let metadata = metadata_with_profile(meta, output_bytes, interp);
            Image::from_buffer(width, height, 3, output)
                .map(|img| IccImage::F32(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
    }
}

fn resolve_input_profile_bytes<'a, F: BandFormat>(
    image: &Image<F>,
    options: &IccTransformOptions<'a>,
) -> Result<Cow<'a, [u8]>, ViprsError> {
    if let Some(profile) = options.input_profile {
        return Ok(Cow::Borrowed(profile));
    }
    if let Some(profile) = image.metadata().icc_profile.as_deref() {
        return Ok(Cow::Owned(profile.to_vec()));
    }
    fallback_input_profile(image).map(Cow::Owned)
}

fn transform_f32_pcs(
    image: &Image<F32>,
    input_profile: &Profile,
    output_profile: &Profile,
    output_bytes: &[u8],
    intent: Intent,
    flags: Flags<lcms2::DisallowCache>,
    spec: OutputSpec,
) -> Result<IccImage, ViprsError> {
    let in_fmt = input_pixel_format(image, input_profile)?;
    let src = bytemuck::cast_slice::<f32, [f32; 3]>(image.pixels());
    let interp = output_interpretation(spec);
    match spec {
        OutputSpec::U8Rgb | OutputSpec::U8Gray | OutputSpec::U8Cmyk => {
            let (out_fmt, out_bands) = match spec {
                OutputSpec::U8Rgb => (PixelFormat::RGB_8, 3u32),
                OutputSpec::U8Gray => (PixelFormat::GRAY_8, 1u32),
                OutputSpec::U8Cmyk => (PixelFormat::CMYK_8, 4u32),
                _ => unreachable!(),
            };
            let transform = Transform::<[f32; 3], u8, _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                out_fmt,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let (_pixels, sample_count) =
                checked_icc_output_sizes(image.width(), image.height(), out_bands)?;
            let mut output = vec![0u8; sample_count];
            transform.transform_pixels(src, &mut output);
            let metadata = metadata_with_profile(image.metadata(), output_bytes, interp);
            Image::from_buffer(image.width(), image.height(), out_bands, output)
                .map(|img| IccImage::U8(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
        OutputSpec::U16Rgb | OutputSpec::U16Gray | OutputSpec::U16Cmyk => {
            let (out_fmt, out_bands) = match spec {
                OutputSpec::U16Rgb => (PixelFormat::RGB_16, 3u32),
                OutputSpec::U16Gray => (PixelFormat::GRAY_16, 1u32),
                OutputSpec::U16Cmyk => (PixelFormat::CMYK_16, 4u32),
                _ => unreachable!(),
            };
            let transform = Transform::<[f32; 3], u8, _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                out_fmt,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let (_pixels, sample_count) =
                checked_icc_output_sizes(image.width(), image.height(), out_bands)?;
            let mut output = vec![0u16; sample_count];
            transform.transform_pixels(src, bytemuck::cast_slice_mut(&mut output));
            let metadata = metadata_with_profile(image.metadata(), output_bytes, interp);
            Image::from_buffer(image.width(), image.height(), out_bands, output)
                .map(|img| IccImage::U16(img.with_metadata(metadata)))
                .map_err(|e| icc_error(e.to_string()))
        }
        OutputSpec::F32Lab => {
            let transform = Transform::<[f32; 3], [f32; 3], _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                PixelFormat::Lab_FLT,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let pixels = checked_icc_output_pixels(image.width(), image.height(), 3)?;
            let mut output = vec![[0.0f32; 3]; pixels];
            transform.transform_pixels(src, &mut output);
            let output = bytemuck::allocation::try_cast_vec::<[f32; 3], f32>(output)
                .map_err(|(_err, _buf)| icc_error("f32 ICC output cast failed"))?;
            let metadata = metadata_with_profile(image.metadata(), output_bytes, interp);
            Image::from_buffer(image.width(), image.height(), 3, output)
                .map(|image| IccImage::F32(image.with_metadata(metadata)))
                .map_err(|err| icc_error(err.to_string()))
        }
        OutputSpec::F32Xyz => {
            let transform = Transform::<[f32; 3], [f32; 3], _, _>::new_flags(
                input_profile,
                in_fmt,
                output_profile,
                PixelFormat::XYZ_FLT,
                intent,
                flags,
            )
            .map_err(super::lcms_error)?;
            let pixels = checked_icc_output_pixels(image.width(), image.height(), 3)?;
            let mut output = vec![[0.0f32; 3]; pixels];
            transform.transform_pixels(src, &mut output);
            let output = bytemuck::allocation::try_cast_vec::<[f32; 3], f32>(output)
                .map_err(|(_err, _buf)| icc_error("f32 ICC output cast failed"))?;
            let metadata = metadata_with_profile(image.metadata(), output_bytes, interp);
            Image::from_buffer(image.width(), image.height(), 3, output)
                .map(|image| IccImage::F32(image.with_metadata(metadata)))
                .map_err(|err| icc_error(err.to_string()))
        }
    }
}

/// Imports an image into PCS (Lab) colour space using its embedded ICC profile.
pub fn icc_import<F: BandFormat>(image: &Image<F>, profile: &[u8]) -> Result<IccImage, ViprsError> {
    let pcs_profile = lab_profile_bytes()?;
    let imported = icc_transform(
        image,
        &pcs_profile,
        &IccTransformOptions {
            input_profile: Some(profile),
            ..IccTransformOptions::default()
        },
    )?;
    Ok(replace_embedded_profile(
        imported,
        profile,
        Interpretation::Lab,
    ))
}

/// Exports an image from PCS back to an output ICC colour space.
pub fn icc_export<F: BandFormat>(
    image: &Image<F>,
    profile: Option<&[u8]>,
) -> Result<IccImage, ViprsError> {
    let output_profile = profile
        .or_else(|| image.metadata().icc_profile.as_deref())
        .ok_or_else(|| icc_error("no ICC profile available to export"))?;
    let pcs_profile = pcs_profile_for_export(image)?;
    icc_transform(
        image,
        output_profile,
        &IccTransformOptions {
            input_profile: Some(&pcs_profile),
            ..IccTransformOptions::default()
        },
    )
}

/// Applies a full ICC colour transform between input and output profiles.
pub fn icc_transform<F: BandFormat>(
    image: &Image<F>,
    output_profile_bytes: &[u8],
    options: &IccTransformOptions<'_>,
) -> Result<IccImage, ViprsError> {
    let input_profile_bytes = resolve_input_profile_bytes(image, options)?;
    let input_profile = open_profile(input_profile_bytes.as_ref(), "input")?;
    let output_profile = open_profile(output_profile_bytes, "output")?;
    let intent = selected_intent(options.intent, &input_profile, &output_profile)?;
    let spec = output_spec(&output_profile, options.depth)?;
    let flags = if options.black_point_compensation {
        Flags::NO_CACHE | Flags::BLACKPOINT_COMPENSATION
    } else {
        Flags::NO_CACHE
    };

    match F::ID {
        BandFormatId::U8 => {
            let src_u8 =
                bytemuck::allocation::try_cast_vec::<F::Sample, u8>(image.pixels().to_vec())
                    .map_err(|(_err, _buf)| icc_error("u8 ICC input cast failed"))?;
            let typed =
                Image::<U8>::from_buffer(image.width(), image.height(), image.bands(), src_u8)
                    .map_err(|err| icc_error(err.to_string()))?;
            let in_fmt = input_pixel_format(&typed, &input_profile)?;
            transform_int_input(
                typed.pixels(),
                image.width(),
                image.height(),
                in_fmt,
                image.metadata(),
                &input_profile,
                &output_profile,
                output_profile_bytes,
                intent,
                flags,
                spec,
            )
        }
        BandFormatId::U16 => {
            let src_u16 =
                bytemuck::allocation::try_cast_vec::<F::Sample, u16>(image.pixels().to_vec())
                    .map_err(|(_err, _buf)| icc_error("u16 ICC input cast failed"))?;
            let typed =
                Image::<U16>::from_buffer(image.width(), image.height(), image.bands(), src_u16)
                    .map_err(|err| icc_error(err.to_string()))?;
            let in_fmt = input_pixel_format(&typed, &input_profile)?;
            let src_bytes: &[u8] = bytemuck::cast_slice(typed.pixels());
            transform_int_input(
                src_bytes,
                image.width(),
                image.height(),
                in_fmt,
                image.metadata(),
                &input_profile,
                &output_profile,
                output_profile_bytes,
                intent,
                flags,
                spec,
            )
        }
        BandFormatId::F32 => {
            let typed = Image::<F32>::from_buffer(
                image.width(),
                image.height(),
                image.bands(),
                bytemuck::allocation::try_cast_vec::<F::Sample, f32>(image.pixels().to_vec())
                    .map_err(|(_err, _buf)| icc_error("f32 ICC input cast failed"))?,
            )
            .map_err(|err| icc_error(err.to_string()))?;
            transform_f32_pcs(
                &typed,
                &input_profile,
                &output_profile,
                output_profile_bytes,
                intent,
                flags,
                spec,
            )
        }
        other => Err(icc_error(format!(
            "unsupported ICC source format {other:?}; supported: U8, U16, F32"
        ))),
    }
}
