use libwebp_sys::{VP8StatusCode, WEBP_CSP_MODE, WebPDecode, WebPDecoderConfig, WebPRGBABuffer};
#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
    sync::{Arc, OnceLock, RwLock},
};

use super::super::shrink_on_load::{ShrinkOnLoadBackend, ShrinkOnLoadPlan};
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId};
use crate::domain::image::{Image, Interpretation, Region};

const WEBP_SHRINK_BACKEND: ShrinkOnLoadBackend = ShrinkOnLoadBackend::WebpDecoderConfigScaling;
const WEBP_ANIM_SHRINK_BACKEND: ShrinkOnLoadBackend = ShrinkOnLoadBackend::WebpDemuxFragmentScaling;
pub(super) const WEBP_ICC_CHUNK_FOURCC: &[u8; 5] = b"ICCP\0";
pub(super) const WEBP_XMP_CHUNK_FOURCC: &[u8; 5] = b"XMP \0";
pub(super) const WEBP_DEFAULT_QUALITY: u8 = 75;
pub(super) const WEBP_DEFAULT_METHOD: u8 = 4;
pub(super) const WEBP_DEFAULT_LOSSLESS: bool = false;
pub(super) const WEBP_INCREMENTAL_CHUNK_SIZE: usize = 16 * 1024;
pub(super) const WEBP_MODE_LAST: usize = 13;
const WEBP_MAX_SCRATCH_ALLOCATION_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const WEBP_MAX_TOTAL_ANIMATION_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub(super) const WEBP_STATIC_REGION_CACHE_CAPACITY: usize = 4;
#[cfg(test)]
thread_local! {
    static WEBP_MAX_SCRATCH_ALLOCATION_BYTES_OVERRIDE: Cell<Option<u64>> = const { Cell::new(None) };
    static WEBP_MAX_TOTAL_ANIMATION_BYTES_OVERRIDE: Cell<Option<u64>> = const { Cell::new(None) };
}
#[cfg(test)]
pub(super) static WEBP_STATIC_REGION_FRAME_DECODES: AtomicUsize = AtomicUsize::new(0);

type StaticWebpRegionCache = HashMap<StaticWebpRegionCacheKey, Arc<CachedStaticWebpFrame>>;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct StaticWebpRegionCacheKey {
    hash: u64,
    src_len: usize,
    shrink_factor: u8,
}

#[derive(Debug)]
pub(super) struct CachedStaticWebpFrame {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bands: u32,
    pub(super) pixels: Vec<u8>,
}

#[inline]
pub(super) fn webp_max_scratch_allocation_bytes() -> u64 {
    #[cfg(test)]
    {
        return WEBP_MAX_SCRATCH_ALLOCATION_BYTES_OVERRIDE
            .with(|limit| limit.get())
            .unwrap_or(WEBP_MAX_SCRATCH_ALLOCATION_BYTES);
    }

    #[cfg(not(test))]
    {
        WEBP_MAX_SCRATCH_ALLOCATION_BYTES
    }
}

#[inline]
pub(super) fn webp_max_total_animation_bytes() -> u64 {
    #[cfg(test)]
    {
        return WEBP_MAX_TOTAL_ANIMATION_BYTES_OVERRIDE
            .with(|limit| limit.get())
            .unwrap_or(WEBP_MAX_TOTAL_ANIMATION_BYTES);
    }

    #[cfg(not(test))]
    {
        WEBP_MAX_TOTAL_ANIMATION_BYTES
    }
}

pub(super) fn webp_static_region_cache() -> &'static RwLock<StaticWebpRegionCache> {
    static CACHE: OnceLock<RwLock<StaticWebpRegionCache>> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

pub(super) fn webp_static_region_cache_key(
    src: &[u8],
    shrink_factor: u8,
) -> StaticWebpRegionCacheKey {
    let mut hasher = DefaultHasher::new();
    src.hash(&mut hasher);
    StaticWebpRegionCacheKey {
        hash: hasher.finish(),
        src_len: src.len(),
        shrink_factor,
    }
}

#[cfg(test)]
pub(crate) fn reset_webp_static_region_frame_decode_count() {
    WEBP_STATIC_REGION_FRAME_DECODES.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn webp_static_region_frame_decode_count() -> usize {
    WEBP_STATIC_REGION_FRAME_DECODES.load(Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn test_webp_max_scratch_allocation_bytes_override() -> Option<u64> {
    WEBP_MAX_SCRATCH_ALLOCATION_BYTES_OVERRIDE.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn set_test_webp_max_scratch_allocation_bytes(limit: Option<u64>) {
    WEBP_MAX_SCRATCH_ALLOCATION_BYTES_OVERRIDE.with(|current| current.set(limit));
}

#[cfg(test)]
pub(crate) fn test_webp_max_total_animation_bytes_override() -> Option<u64> {
    WEBP_MAX_TOTAL_ANIMATION_BYTES_OVERRIDE.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn set_test_webp_max_total_animation_bytes(limit: Option<u64>) {
    WEBP_MAX_TOTAL_ANIMATION_BYTES_OVERRIDE.with(|current| current.set(limit));
}

pub(super) fn validate_webp_riff_size(src: &[u8]) -> Result<(), ViprsError> {
    if src.len() < 12 || &src[0..4] != b"RIFF" || &src[8..12] != b"WEBP" {
        return Ok(());
    }

    let declared_size = u32::from_le_bytes([src[4], src[5], src[6], src[7]]) as usize;
    let declared_total = declared_size
        .checked_add(8)
        .ok_or_else(|| ViprsError::Codec("webp: RIFF size overflow".into()))?;
    if declared_total != src.len() {
        return Err(ViprsError::Codec(format!(
            "webp: RIFF size {declared_total} does not match input length {}",
            src.len()
        )));
    }

    Ok(())
}
// ── Helpers ───────────────────────────────────────────────────────────────────

/// Assert at the call site that `F` is `U8`; otherwise return a typed error.
///
/// WebP is an 8-bit format: there is no safe, lossless mapping from U16/F32
/// pixel values to WebP, so we reject non-U8 formats rather than silently
/// truncating data.
#[inline]
pub(super) fn require_u8<F: BandFormat>() -> Result<(), ViprsError> {
    if F::ID != BandFormatId::U8 {
        return Err(ViprsError::Codec(format!(
            "webp: unsupported format {:?}; only U8 is supported",
            F::ID
        )));
    }
    Ok(())
}

pub(super) fn image_from_u8_pixels<F: BandFormat>(
    width: u32,
    height: u32,
    bands: u32,
    pixels_u8: Vec<u8>,
) -> Result<Image<F>, ViprsError> {
    let samples =
        bytemuck::allocation::try_cast_vec::<u8, F::Sample>(pixels_u8).map_err(|(_e, _v)| {
            ViprsError::Codec("webp: sample cast failed (internal error)".into())
        })?;
    Image::from_buffer(width, height, bands, samples).map_err(|e| ViprsError::Codec(e.to_string()))
}

pub(super) fn copy_clamped_region_from_u8_pixels<F: BandFormat>(
    width: u32,
    height: u32,
    bands: u32,
    pixels_u8: &[u8],
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    require_u8::<F>()?;

    let expected = region
        .pixel_count()
        .checked_mul(bands as usize)
        .ok_or_else(|| ViprsError::Codec("webp: output buffer size overflow".into()))?;
    if output.len() != expected {
        return Err(ViprsError::Codec(format!(
            "webp: output buffer size mismatch (got {}, expected {expected})",
            output.len()
        )));
    }
    if width == 0 || height == 0 {
        return Ok(());
    }

    let stride = width as usize * bands as usize;
    for row in 0..region.height as i32 {
        let src_y = (region.y + row).clamp(0, height as i32 - 1) as usize;
        for col in 0..region.width as i32 {
            let src_x = (region.x + col).clamp(0, width as i32 - 1) as usize;
            let src_start = src_y * stride + src_x * bands as usize;
            let dst_start = (row as usize * region.width as usize + col as usize) * bands as usize;
            output[dst_start..dst_start + bands as usize]
                .copy_from_slice(&pixels_u8[src_start..src_start + bands as usize]);
        }
    }

    Ok(())
}

#[inline]
pub(super) fn webp_interpretation(bands: u32) -> Interpretation {
    match bands {
        1 | 2 => Interpretation::BW,
        _ => Interpretation::Srgb,
    }
}

pub(crate) fn webp_shrink_on_load_plan(requested_factor: u8) -> ShrinkOnLoadPlan {
    ShrinkOnLoadPlan::new(requested_factor, WEBP_SHRINK_BACKEND)
}

pub(crate) fn webp_anim_shrink_on_load_plan(requested_factor: u8) -> ShrinkOnLoadPlan {
    ShrinkOnLoadPlan::new(requested_factor, WEBP_ANIM_SHRINK_BACKEND)
}

pub(super) fn webp_scaled_dimension(dimension: u32, factor: u8) -> u32 {
    if factor == 1 {
        dimension
    } else {
        (dimension / u32::from(factor)).max(1)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WebpFrameArea {
    pub(super) left: u32,
    pub(super) top: u32,
    pub(super) width: u32,
    pub(super) height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WebpWindowDecodePlan {
    pub(super) crop_left: u32,
    pub(super) crop_top: u32,
    pub(super) crop_width: u32,
    pub(super) crop_height: u32,
    pub(super) output_width: u32,
    pub(super) output_height: u32,
    pub(super) output_offset_x: u32,
    pub(super) output_offset_y: u32,
}

pub(super) fn webp_clamped_coverage_region(
    region: Region,
    image_width: u32,
    image_height: u32,
) -> Option<Region> {
    if region.is_empty() || image_width == 0 || image_height == 0 {
        return None;
    }

    let max_x = i64::from(image_width.saturating_sub(1));
    let max_y = i64::from(image_height.saturating_sub(1));
    let x0 = i64::from(region.x).clamp(0, max_x) as u32;
    let y0 = i64::from(region.y).clamp(0, max_y) as u32;
    let x1 = (i64::from(region.x) + i64::from(region.width) - 1).clamp(0, max_x) as u32;
    let y1 = (i64::from(region.y) + i64::from(region.height) - 1).clamp(0, max_y) as u32;

    Some(Region::new(
        x0 as i32,
        y0 as i32,
        x1.saturating_sub(x0) + 1,
        y1.saturating_sub(y0) + 1,
    ))
}

pub(super) fn webp_window_decode_plan(
    source_left: u32,
    source_top: u32,
    output_width: u32,
    output_height: u32,
    shrink_factor: u8,
) -> Result<WebpWindowDecodePlan, ViprsError> {
    let output_offset_x = if shrink_factor == 1 {
        source_left & 1
    } else {
        0
    };
    let output_offset_y = if shrink_factor == 1 {
        source_top & 1
    } else {
        0
    };
    let crop_left = source_left.saturating_sub(output_offset_x);
    let crop_top = source_top.saturating_sub(output_offset_y);
    let crop_width = output_width
        .checked_mul(u32::from(shrink_factor))
        .and_then(|width| width.checked_add(output_offset_x))
        .ok_or_else(|| ViprsError::Codec("webp: crop width overflow".into()))?;
    let crop_height = output_height
        .checked_mul(u32::from(shrink_factor))
        .and_then(|height| height.checked_add(output_offset_y))
        .ok_or_else(|| ViprsError::Codec("webp: crop height overflow".into()))?;

    Ok(WebpWindowDecodePlan {
        crop_left,
        crop_top,
        crop_width,
        crop_height,
        output_width: output_width
            .checked_add(output_offset_x)
            .ok_or_else(|| ViprsError::Codec("webp: output width overflow".into()))?,
        output_height: output_height
            .checked_add(output_offset_y)
            .ok_or_else(|| ViprsError::Codec("webp: output height overflow".into()))?,
        output_offset_x,
        output_offset_y,
    })
}

pub(super) fn webp_decode_window_into_buffer(
    src: *const u8,
    src_len: usize,
    input_width: u32,
    input_height: u32,
    bands: u32,
    plan: WebpWindowDecodePlan,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let expected = (plan.output_width as usize)
        .checked_mul(plan.output_height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::Codec("webp: output buffer size overflow".into()))?;
    if output.len() != expected {
        return Err(ViprsError::Codec(format!(
            "webp: output buffer size mismatch (got {}, expected {expected})",
            output.len()
        )));
    }

    let full_decode = plan.crop_left == 0
        && plan.crop_top == 0
        && plan.crop_width == input_width
        && plan.crop_height == input_height;
    let scaled_decode =
        plan.output_width != plan.crop_width || plan.output_height != plan.crop_height;
    let stride = i32::try_from(
        plan.output_width
            .checked_mul(bands)
            .ok_or_else(|| ViprsError::Codec("webp: row stride overflows u32".into()))?,
    )
    .map_err(|_| ViprsError::Codec("webp: row stride overflows i32".into()))?;

    let mut config = WebPDecoderConfig::new()
        .map_err(|()| ViprsError::Codec("webp: decoder config init failed".into()))?;
    config.output.colorspace = if bands == 4 {
        WEBP_CSP_MODE::MODE_RGBA
    } else {
        WEBP_CSP_MODE::MODE_RGB
    };
    config.output.width = i32::try_from(plan.output_width)
        .map_err(|_| ViprsError::Codec("webp: output width overflows i32".into()))?;
    config.output.height = i32::try_from(plan.output_height)
        .map_err(|_| ViprsError::Codec("webp: output height overflows i32".into()))?;
    config.output.is_external_memory = 1;
    config.output.u.RGBA = WebPRGBABuffer {
        rgba: output.as_mut_ptr(),
        stride,
        size: output.len(),
    };
    config.options.use_threads = 1;
    if !full_decode {
        config.options.use_cropping = 1;
        config.options.crop_left = i32::try_from(plan.crop_left)
            .map_err(|_| ViprsError::Codec("webp: crop left overflows i32".into()))?;
        config.options.crop_top = i32::try_from(plan.crop_top)
            .map_err(|_| ViprsError::Codec("webp: crop top overflows i32".into()))?;
        config.options.crop_width = i32::try_from(plan.crop_width)
            .map_err(|_| ViprsError::Codec("webp: crop width overflows i32".into()))?;
        config.options.crop_height = i32::try_from(plan.crop_height)
            .map_err(|_| ViprsError::Codec("webp: crop height overflows i32".into()))?;
    }
    if scaled_decode {
        config.options.use_scaling = 1;
        config.options.scaled_width = config.output.width;
        config.options.scaled_height = config.output.height;
    }

    let status = {
        // SAFETY: `src` points to a valid WebP bitstream that remains alive for the
        // duration of the call, `output` owns the external buffer described by
        // `config.output`, and libwebp writes at most that buffer.
        unsafe { WebPDecode(src, src_len, std::ptr::from_mut(&mut config)) }
    };
    if status != VP8StatusCode::VP8_STATUS_OK {
        return Err(ViprsError::Codec(format!(
            "webp: decode failed with status {status:?}"
        )));
    }

    Ok(())
}

pub(super) fn copy_clamped_region_from_u8_window<F: BandFormat>(
    image_width: u32,
    image_height: u32,
    bands: u32,
    window: Region,
    pixels_u8: &[u8],
    window_offset_x: u32,
    window_offset_y: u32,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    require_u8::<F>()?;

    let expected_output = checked_webp_region_output_len(
        region,
        bands,
        "webp: requested region exceeds addressable memory",
    )?;
    if output.len() != expected_output {
        return Err(ViprsError::Codec(format!(
            "webp: output buffer size mismatch (got {}, expected {expected_output})",
            output.len()
        )));
    }

    let decoded_width = window
        .width
        .checked_add(window_offset_x)
        .ok_or_else(|| ViprsError::Codec("webp: decoded window width overflow".into()))?;
    let decoded_height = window
        .height
        .checked_add(window_offset_y)
        .ok_or_else(|| ViprsError::Codec("webp: decoded window height overflow".into()))?;
    let expected_window = (decoded_width as usize)
        .checked_mul(decoded_height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::Codec("webp: window buffer size overflow".into()))?;
    if pixels_u8.len() != expected_window {
        return Err(ViprsError::Codec(format!(
            "webp: window buffer size mismatch (got {}, expected {expected_window})",
            pixels_u8.len()
        )));
    }

    if image_width == 0 || image_height == 0 || region.is_empty() {
        return Ok(());
    }

    let stride = decoded_width as usize * bands as usize;
    for row in 0..region.height as i32 {
        let src_y = (region.y + row).clamp(0, image_height as i32 - 1) as usize;
        let window_y = src_y
            .saturating_sub(window.y as usize)
            .checked_add(window_offset_y as usize)
            .ok_or_else(|| ViprsError::Codec("webp: window y overflow".into()))?;
        for col in 0..region.width as i32 {
            let src_x = (region.x + col).clamp(0, image_width as i32 - 1) as usize;
            let window_x = src_x
                .saturating_sub(window.x as usize)
                .checked_add(window_offset_x as usize)
                .ok_or_else(|| ViprsError::Codec("webp: window x overflow".into()))?;
            let src_start = window_y
                .checked_mul(stride)
                .and_then(|row_start| row_start.checked_add(window_x * bands as usize))
                .ok_or_else(|| ViprsError::Codec("webp: window index overflow".into()))?;
            let dst_start = (row as usize * region.width as usize + col as usize) * bands as usize;
            output[dst_start..dst_start + bands as usize]
                .copy_from_slice(&pixels_u8[src_start..src_start + bands as usize]);
        }
    }

    Ok(())
}
pub(crate) fn checked_webp_scratch_allocation_len(
    width: u32,
    height: u32,
    bands: u32,
    details: &'static str,
) -> Result<usize, ViprsError> {
    let limit_bytes = webp_max_scratch_allocation_bytes();
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixel_count| pixel_count.checked_mul(u64::from(bands)))
        .ok_or_else(|| {
            let total_bytes = u128::from(width) * u128::from(height) * u128::from(bands);
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: u128::from(limit_bytes),
                details,
            }
        })?;
    if bytes > limit_bytes {
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(bytes),
            limit_bytes: u128::from(limit_bytes),
            details,
        });
    }

    usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
        width,
        height,
        bands,
        bytes: u128::from(bytes),
        limit_bytes: u128::try_from(usize::MAX).unwrap_or(u128::MAX),
        details,
    })
}

pub(super) fn checked_webp_region_output_len(
    region: Region,
    bands: u32,
    details: &'static str,
) -> Result<usize, ViprsError> {
    region
        .checked_pixel_count()
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width: region.width,
            height: region.height,
            bands,
            bytes: u128::from(region.width) * u128::from(region.height) * u128::from(bands),
            limit_bytes: usize::MAX as u128,
            details,
        })
}
