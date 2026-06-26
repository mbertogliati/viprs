use super::{
    BandFormat, ImageDecoder, InMemoryImage, LoadOptions, NonZeroU8, Path, Region, ShrinkSample,
    ViprsError,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ThumbnailPreShrinkMode {
    Unsupported,
    Jpeg,
    Webp,
    TiffPyramid,
    /// Software box-filter shrink applied to an already-decoded eager backing.
    ///
    /// Used for PNG, which has no native shrink-on-load. When
    /// `set_thumbnail_shrink_on_load` is called the source reduces the resident
    /// decoded image in-place using a box filter, so downstream pipeline
    /// stages see a much smaller raster (e.g. 8192→1024 at factor 8 = 64×
    /// less data to move through the pipeline).
    SoftwareBoxShrink,
}

#[inline]
pub(super) fn thumbnail_pre_shrink_mode(format_name: &str) -> ThumbnailPreShrinkMode {
    match format_name {
        "jpeg" | "uhdr" => ThumbnailPreShrinkMode::Jpeg,
        "webp" => ThumbnailPreShrinkMode::Webp,
        "tiff" => ThumbnailPreShrinkMode::TiffPyramid,
        "png" => ThumbnailPreShrinkMode::SoftwareBoxShrink,
        _ => ThumbnailPreShrinkMode::Unsupported,
    }
}

#[inline]
pub(super) fn retains_stable_input_for_thumbnail(format_name: &str) -> bool {
    matches!(
        thumbnail_pre_shrink_mode(format_name),
        ThumbnailPreShrinkMode::Jpeg
            | ThumbnailPreShrinkMode::Webp
            | ThumbnailPreShrinkMode::TiffPyramid
    )
}

/// Apply a box-filter shrink by `factor` to an image of any `F: BandFormat`.
///
/// For single-byte formats (U8, UCHAR) uses u32 integer accumulation (no f64).
/// All other formats use f64 accumulation.
/// The shrink runs once at pipeline construction time, not per-tile.
pub(super) fn software_box_shrink_generic<F: BandFormat>(
    image: &InMemoryImage<F>,
    factor: usize,
) -> Result<InMemoryImage<F>, ViprsError>
where
    F::Sample: ShrinkSample,
{
    let src_w = image.width() as usize;
    let src_h = image.height() as usize;
    let bands = image.bands() as usize;
    let dst_w = (src_w / factor).max(1);
    let dst_h = (src_h / factor).max(1);

    // Integer path for 1-byte samples (U8): avoids f64 conversion and rounding.
    if std::mem::size_of::<F::Sample>() == 1 {
        let src_bytes: &[u8] = bytemuck::cast_slice(image.pixels());
        let mut dst_bytes = vec![0u8; dst_w * dst_h * bands];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                let sy0 = dy * factor;
                let sx0 = dx * factor;
                let sy1 = (sy0 + factor).min(src_h);
                let sx1 = (sx0 + factor).min(src_w);
                let divisor = ((sy1 - sy0) * (sx1 - sx0)).max(1) as u32;

                for b in 0..bands {
                    let mut sum: u32 = 0;
                    for sy in sy0..sy1 {
                        let row_base = sy * src_w * bands;
                        for sx in sx0..sx1 {
                            sum += u32::from(src_bytes[row_base + sx * bands + b]);
                        }
                    }
                    dst_bytes[(dy * dst_w + dx) * bands + b] =
                        ((sum + divisor / 2) / divisor) as u8;
                }
            }
        }

        let dst_samples: Vec<F::Sample> = bytemuck::cast_vec(dst_bytes);
        return InMemoryImage::from_buffer(dst_w as u32, dst_h as u32, bands as u32, dst_samples)
            .map(|img| img.with_metadata(image.metadata().clone()));
    }

    let mut dst_samples = vec![F::Sample::from_f64_clamped(0.0); dst_w * dst_h * bands];
    let src = image.pixels();

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sy0 = dy * factor;
            let sx0 = dx * factor;
            let sy1 = (sy0 + factor).min(src_h);
            let sx1 = (sx0 + factor).min(src_w);
            let count = ((sy1 - sy0) * (sx1 - sx0)).max(1) as f64;

            for b in 0..bands {
                let mut sum = 0.0_f64;
                for sy in sy0..sy1 {
                    let row_base = sy * src_w * bands;
                    for sx in sx0..sx1 {
                        sum += src[row_base + sx * bands + b].to_f64();
                    }
                }
                dst_samples[(dy * dst_w + dx) * bands + b] =
                    F::Sample::from_f64_clamped(sum / count);
            }
        }
    }

    InMemoryImage::from_buffer(dst_w as u32, dst_h as u32, bands as u32, dst_samples)
        .map(|img| img.with_metadata(image.metadata().clone()))
}

#[inline]
pub(super) const fn normalize_shrink_factor(factor: u8) -> u8 {
    match factor {
        2 | 4 | 8 | 16 => factor,
        _ => 1,
    }
}

pub(super) fn normalize_streaming_options(opts: &LoadOptions, shrink_factor: u8) -> LoadOptions {
    let mut normalized = opts.clone();
    normalized.shrink_factor = if shrink_factor > 1 {
        NonZeroU8::new(shrink_factor)
    } else {
        None
    };
    normalized
}

#[inline]
pub(super) fn shrunk_dimension(dimension: u32, factor: u8) -> u32 {
    if dimension == 0 || factor <= 1 {
        dimension
    } else {
        (dimension / u32::from(factor)).max(1)
    }
}

#[inline]
fn decode_time_shrink_applied(
    original_width: u32,
    original_height: u32,
    decoded_width: u32,
    decoded_height: u32,
    requested_factor: u8,
) -> bool {
    requested_factor > 1
        && (decoded_width < original_width || decoded_height < original_height)
        && decoded_width <= shrunk_dimension(original_width, requested_factor)
        && decoded_height <= shrunk_dimension(original_height, requested_factor)
}

pub(super) fn eager_backing_shrink_factor<D: ImageDecoder, F: BandFormat>(
    decoder: &D,
    src: &[u8],
    requested_factor: u8,
    image: &InMemoryImage<F>,
) -> u8 {
    if requested_factor <= 1 {
        return 1;
    }

    match decoder.probe(src) {
        Ok((original_width, original_height, _))
            if decode_time_shrink_applied(
                original_width,
                original_height,
                image.width(),
                image.height(),
                requested_factor,
            ) =>
        {
            requested_factor
        }
        _ => 1,
    }
}

pub(super) fn eager_backing_shrink_factor_from_path<D: ImageDecoder, F: BandFormat>(
    decoder: &D,
    path: &Path,
    requested_factor: u8,
    image: &InMemoryImage<F>,
) -> u8 {
    if requested_factor <= 1 {
        return 1;
    }

    match decoder.probe_path(path) {
        Ok((original_width, original_height, _))
            if decode_time_shrink_applied(
                original_width,
                original_height,
                image.width(),
                image.height(),
                requested_factor,
            ) =>
        {
            requested_factor
        }
        _ => 1,
    }
}

pub(super) fn expected_output_len<F: BandFormat>(
    region: Region,
    bands: u32,
    context: &'static str,
) -> Result<usize, ViprsError> {
    region
        .pixel_count()
        .checked_mul(bands as usize)
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<F::Sample>()))
        .ok_or_else(|| ViprsError::Codec(format!("{context}: output buffer length overflow")))
}

pub(super) fn checked_region_end(
    region: Region,
    width: u32,
    height: u32,
    context: &'static str,
) -> Result<(i64, i64), ViprsError> {
    let end_x = i64::from(region.x) + i64::from(region.width);
    let end_y = i64::from(region.y) + i64::from(region.height);

    if end_x > i64::from(i32::MAX) || end_y > i64::from(i32::MAX) {
        return Err(ViprsError::Codec(format!(
            "{context}: region {region:?} is out of bounds for {width}x{height} image"
        )));
    }

    Ok((end_x, end_y))
}

pub(super) fn materialize_residual_thumbnail_shrink<F: BandFormat>(
    image: InMemoryImage<F>,
    requested_factor: u8,
    backing_factor: u8,
) -> Result<(InMemoryImage<F>, u8), ViprsError>
where
    F::Sample: ShrinkSample,
{
    if requested_factor <= backing_factor || !requested_factor.is_multiple_of(backing_factor) {
        return Ok((image, backing_factor));
    }

    let residual_factor = requested_factor / backing_factor;
    if residual_factor <= 1 {
        return Ok((image, backing_factor));
    }

    let shrunken = software_box_shrink_generic(&image, usize::from(residual_factor))?;
    Ok((shrunken, requested_factor))
}
