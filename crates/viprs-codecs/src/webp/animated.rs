use libwebp_sys::{
    WebPChunkIterator, WebPData, WebPDemuxDelete, WebPDemuxGetChunk, WebPDemuxGetFrame,
    WebPDemuxGetI, WebPDemuxInternal, WebPDemuxNextFrame, WebPDemuxReleaseChunkIterator,
    WebPDemuxReleaseIterator, WebPFormatFeature, WebPGetDemuxABIVersion, WebPIterator,
    WebPMuxAnimBlend, WebPMuxAnimDispose,
};

use super::super::shrink_on_load::ShrinkOnLoadPlan;
use super::common::{
    WebpFrameArea, WebpWindowDecodePlan, checked_webp_region_output_len,
    checked_webp_scratch_allocation_len, copy_clamped_region_from_u8_window, image_from_u8_pixels,
    webp_anim_shrink_on_load_plan, webp_clamped_coverage_region, webp_decode_window_into_buffer,
    webp_interpretation, webp_max_total_animation_bytes, webp_window_decode_plan,
};
use viprs_core::codec_options::LoadOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::BandFormat;
use viprs_core::image::{Image, ImageMetadata, Region};

pub(super) struct WebpDemux(*mut libwebp_sys::WebPDemuxer);

impl WebpDemux {
    pub(super) fn new(src: &[u8]) -> Result<Self, ViprsError> {
        let data = WebPData {
            bytes: src.as_ptr(),
            size: src.len(),
        };
        let demux = {
            // SAFETY: `data` points to `src`, which stays alive for the lifetime of the
            // returned demux handle; we pass a null parser state because we require a
            // complete in-memory WebP stream.
            unsafe {
                WebPDemuxInternal(
                    std::ptr::from_ref(&data),
                    0,
                    std::ptr::null_mut(),
                    WebPGetDemuxABIVersion(),
                )
            }
        };
        if demux.is_null() {
            return Err(ViprsError::Codec("webp: unable to parse container".into()));
        }
        Ok(Self(demux))
    }

    #[inline]
    pub(super) fn feature(&self, feature: WebPFormatFeature) -> u32 {
        // SAFETY: `self.0` is a live demux handle created by `WebPDemuxInternal` and
        // released only in `Drop`.
        unsafe { WebPDemuxGetI(self.0, feature) }
    }

    fn first_frame(&self) -> Result<WebpFrameIter, ViprsError> {
        let mut iter = std::mem::MaybeUninit::<WebPIterator>::zeroed();
        let ok = {
            // SAFETY: `self.0` is a valid demux handle, and `iter` points to writable
            // uninitialized storage for libwebp to populate.
            unsafe { WebPDemuxGetFrame(self.0, 1, iter.as_mut_ptr()) }
        };
        if ok == 0 {
            return Err(ViprsError::Codec(
                "webp: unable to read animation frames".into(),
            ));
        }
        Ok(WebpFrameIter {
            iter: {
                // SAFETY: `WebPDemuxGetFrame` succeeded, so libwebp initialized the
                // iterator structure completely.
                unsafe { iter.assume_init() }
            },
        })
    }

    pub(super) fn chunk(&self, fourcc: &[u8; 5]) -> Option<Vec<u8>> {
        let mut iter = std::mem::MaybeUninit::<WebPChunkIterator>::zeroed();
        let ok = {
            // SAFETY: `self.0` is a valid demux handle, `fourcc` is a null-terminated
            // 4CC string, and `iter` points to writable storage for libwebp.
            unsafe { WebPDemuxGetChunk(self.0, fourcc.as_ptr().cast(), 1, iter.as_mut_ptr()) }
        };
        if ok == 0 {
            return None;
        }
        let mut iter = {
            // SAFETY: `WebPDemuxGetChunk` succeeded and initialized the iterator.
            unsafe { iter.assume_init() }
        };
        let bytes = {
            // SAFETY: libwebp keeps `iter.chunk.bytes` alive until the chunk iterator is
            // released below, and `iter.chunk.size` describes exactly that borrowed chunk.
            unsafe { std::slice::from_raw_parts(iter.chunk.bytes, iter.chunk.size) }.to_vec()
        };
        // SAFETY: `iter` was initialized by `WebPDemuxGetChunk` and has not been released yet.
        unsafe { WebPDemuxReleaseChunkIterator(std::ptr::from_mut(&mut iter)) };
        Some(bytes)
    }
}

impl Drop for WebpDemux {
    fn drop(&mut self) {
        // SAFETY: `self.0` was returned by `WebPDemuxInternal` and has not been freed yet, so dropping the handle releases libwebp-owned state exactly once.
        unsafe { WebPDemuxDelete(self.0) };
    }
}

struct WebpFrameIter {
    iter: WebPIterator,
}

impl WebpFrameIter {
    #[inline]
    const fn current(&self) -> &WebPIterator {
        &self.iter
    }

    fn advance(&mut self) -> Result<bool, ViprsError> {
        if self.iter.frame_num >= self.iter.num_frames {
            return Ok(false);
        }
        // SAFETY: `self.iter` is an iterator initialized by libwebp and remains valid until released in `Drop`.
        let ok = unsafe { WebPDemuxNextFrame(std::ptr::from_mut(&mut self.iter)) };
        if ok == 0 {
            return Err(ViprsError::Codec(
                "webp: truncated animated frame sequence".into(),
            ));
        }
        Ok(true)
    }
}

impl Drop for WebpFrameIter {
    fn drop(&mut self) {
        // SAFETY: `self.iter` was initialized by libwebp and has not been released yet, so releasing it here matches `WebPDemuxGetFrame`.
        unsafe { WebPDemuxReleaseIterator(std::ptr::from_mut(&mut self.iter)) };
    }
}

fn webp_round_ties_even_div(value: u32, divisor: u8) -> u32 {
    let divisor = u32::from(divisor);
    let quotient = value / divisor;
    let remainder = value % divisor;
    let doubled = remainder.saturating_mul(2);

    if doubled < divisor {
        quotient
    } else if doubled > divisor || quotient % 2 == 1 {
        quotient + 1
    } else {
        quotient
    }
}

pub(super) fn webp_scaled_animation_dimension(dimension: u32, factor: u8) -> u32 {
    if factor == 1 {
        dimension
    } else {
        webp_round_ties_even_div(dimension, factor).max(1)
    }
}

fn webp_scaled_animation_offset(offset: u32, factor: u8) -> u32 {
    if factor == 1 {
        offset
    } else {
        webp_round_ties_even_div(offset, factor)
    }
}

fn webp_frame_area(iter: &WebPIterator, shrink_factor: u8) -> Result<WebpFrameArea, ViprsError> {
    let left = u32::try_from(iter.x_offset)
        .map_err(|_| ViprsError::Codec("webp: negative frame x offset".into()))?;
    let top = u32::try_from(iter.y_offset)
        .map_err(|_| ViprsError::Codec("webp: negative frame y offset".into()))?;
    let width = u32::try_from(iter.width)
        .map_err(|_| ViprsError::Codec("webp: negative frame width".into()))?;
    let height = u32::try_from(iter.height)
        .map_err(|_| ViprsError::Codec("webp: negative frame height".into()))?;

    Ok(WebpFrameArea {
        left: webp_scaled_animation_offset(left, shrink_factor),
        top: webp_scaled_animation_offset(top, shrink_factor),
        width: webp_scaled_animation_dimension(width, shrink_factor),
        height: webp_scaled_animation_dimension(height, shrink_factor),
    })
}

fn webp_decode_rgba_fragment_window(
    fragment: WebPData,
    input_width: u32,
    input_height: u32,
    source_left: u32,
    source_top: u32,
    output_width: u32,
    output_height: u32,
    shrink_factor: u8,
    pixels_u8: &mut Vec<u8>,
) -> Result<WebpWindowDecodePlan, ViprsError> {
    let plan = webp_window_decode_plan(
        source_left,
        source_top,
        output_width,
        output_height,
        shrink_factor,
    )?;
    let len = (plan.output_width as usize)
        .checked_mul(plan.output_height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(4))
        .ok_or_else(|| ViprsError::Codec("webp: animated frame dimensions overflow".into()))?;
    pixels_u8.resize(len, 0);
    webp_decode_window_into_buffer(
        fragment.bytes,
        fragment.size,
        input_width,
        input_height,
        4,
        plan,
        pixels_u8,
    )?;
    Ok(plan)
}

fn webp_blend_rgba_pixel(src: &[u8], dst: &mut [u8]) {
    let src_alpha = src[3];
    if src_alpha == 0 {
        return;
    }

    let dst_alpha = dst[3];
    let dst_factor = ((u32::from(dst_alpha) * (255 - u32::from(src_alpha)) + 127) >> 8) as u8;
    let out_alpha = src_alpha.saturating_add(dst_factor);
    let scale = if out_alpha == 0 {
        0
    } else {
        (1_u32 << 24) / u32::from(out_alpha)
    };

    let blend = |src_channel: u8, dst_channel: u8| -> u8 {
        (((u32::from(src_channel) * u32::from(src_alpha)
            + u32::from(dst_channel) * u32::from(dst_factor))
            * scale
            + (1 << 12))
            >> 24) as u8
    };

    dst[0] = blend(src[0], dst[0]);
    dst[1] = blend(src[1], dst[1]);
    dst[2] = blend(src[2], dst[2]);
    dst[3] = out_alpha;
}

fn webp_paint_rgba_area(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    pixels_u8: &[u8],
    area: WebpFrameArea,
    blend: bool,
) -> Result<(), ViprsError> {
    let expected_len = (area.width as usize)
        .checked_mul(area.height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(4))
        .ok_or_else(|| ViprsError::Codec("webp: animated frame dimensions overflow".into()))?;
    if pixels_u8.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "webp: animated frame buffer length mismatch (got {}, expected {expected_len})",
            pixels_u8.len()
        )));
    }

    let clipped_width = area.width.min(canvas_width.saturating_sub(area.left));
    let clipped_height = area.height.min(canvas_height.saturating_sub(area.top));
    if clipped_width == 0 || clipped_height == 0 {
        return Ok(());
    }

    let canvas_stride = canvas_width as usize * 4;
    let frame_stride = area.width as usize * 4;
    for y in 0..clipped_height as usize {
        let src_row = y * frame_stride;
        let dst_row = (area.top as usize + y) * canvas_stride + area.left as usize * 4;
        if blend {
            for x in 0..clipped_width as usize {
                let src_base = src_row + x * 4;
                let dst_base = dst_row + x * 4;
                webp_blend_rgba_pixel(
                    &pixels_u8[src_base..src_base + 4],
                    &mut canvas[dst_base..dst_base + 4],
                );
            }
        } else {
            let byte_len = clipped_width as usize * 4;
            canvas[dst_row..dst_row + byte_len]
                .copy_from_slice(&pixels_u8[src_row..src_row + byte_len]);
        }
    }

    Ok(())
}

fn webp_clear_rgba_area(
    canvas: &mut [u8],
    canvas_width: u32,
    canvas_height: u32,
    area: WebpFrameArea,
) {
    let clipped_width = area.width.min(canvas_width.saturating_sub(area.left));
    let clipped_height = area.height.min(canvas_height.saturating_sub(area.top));
    if clipped_width == 0 || clipped_height == 0 {
        return;
    }

    let stride = canvas_width as usize * 4;
    let row_bytes = clipped_width as usize * 4;
    for y in 0..clipped_height as usize {
        let row_start = (area.top as usize + y) * stride + area.left as usize * 4;
        canvas[row_start..row_start + row_bytes].fill(0);
    }
}

pub(super) fn webp_animation_shrink_factor(
    demux: &WebpDemux,
    canvas_width: u32,
    canvas_height: u32,
    shrink_plan: ShrinkOnLoadPlan,
) -> Result<u8, ViprsError> {
    if shrink_plan.factor() == 1 {
        return Ok(1);
    }

    let mut frames = demux.first_frame()?;
    loop {
        let frame = frames.current();
        let frame_width = u32::try_from(frame.width)
            .map_err(|_| ViprsError::Codec("webp: negative frame width".into()))?;
        let frame_height = u32::try_from(frame.height)
            .map_err(|_| ViprsError::Codec("webp: negative frame height".into()))?;
        if frame_width != canvas_width || frame_height != canvas_height {
            return Ok(1);
        }
        if !frames.advance()? {
            break;
        }
    }

    Ok(shrink_plan.factor())
}

pub(super) fn decode_animated_webp<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
    icc_profile: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
) -> Result<Image<F>, ViprsError> {
    let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
    let shrink_plan = webp_anim_shrink_on_load_plan(shrink_factor);
    let demux = WebpDemux::new(src)?;
    let canvas_width = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_WIDTH);
    let canvas_height = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_HEIGHT);
    let frame_count = demux.feature(WebPFormatFeature::WEBP_FF_FRAME_COUNT);
    if frame_count == 0 {
        return Err(ViprsError::Codec(
            "webp: animated decode produced no frames".into(),
        ));
    }

    let effective_shrink_factor =
        webp_animation_shrink_factor(&demux, canvas_width, canvas_height, shrink_plan)?;
    let width = webp_scaled_animation_dimension(canvas_width, effective_shrink_factor);
    let height = webp_scaled_animation_dimension(canvas_height, effective_shrink_factor);
    let canvas_len = checked_webp_scratch_allocation_len(
        width,
        height,
        4,
        "webp: eager animated decode allocation exceeds safe limit",
    )?;
    let animation_limit_bytes = webp_max_total_animation_bytes();
    let mut frame_pixels = Vec::new();
    let mut frames = Vec::with_capacity(frame_count as usize);
    let mut previous_canvas: Option<Vec<u8>> = None;
    let mut total_allocated_bytes = 0_u64;
    let mut dispose_method = WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE;
    let mut dispose_area = WebpFrameArea::default();
    let mut iter = demux.first_frame()?;

    loop {
        let mut canvas = vec![0u8; canvas_len];
        if let Some(previous) = previous_canvas.as_ref() {
            canvas.copy_from_slice(previous);
        }
        if dispose_method == WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND {
            webp_clear_rgba_area(&mut canvas, width, height, dispose_area);
        }

        let frame = *iter.current();
        let input_width = u32::try_from(frame.width)
            .map_err(|_| ViprsError::Codec("webp: negative frame width".into()))?;
        let input_height = u32::try_from(frame.height)
            .map_err(|_| ViprsError::Codec("webp: negative frame height".into()))?;
        let area = webp_frame_area(&frame, effective_shrink_factor)?;
        let _ = webp_decode_rgba_fragment_window(
            frame.fragment,
            input_width,
            input_height,
            0,
            0,
            area.width,
            area.height,
            effective_shrink_factor,
            &mut frame_pixels,
        )?;
        webp_paint_rgba_area(
            &mut canvas,
            width,
            height,
            &frame_pixels,
            area,
            frame.frame_num > 1 && frame.blend_method == WebPMuxAnimBlend::WEBP_MUX_BLEND,
        )?;
        if let Some(previous) = previous_canvas.take() {
            frames.push(image_from_u8_pixels::<F>(width, height, 4, previous)?);
        }
        total_allocated_bytes = total_allocated_bytes
            .checked_add(canvas_len as u64)
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width,
                height,
                bands: 4,
                bytes: u128::from(total_allocated_bytes) + u128::from(canvas_len as u64),
                limit_bytes: u128::from(animation_limit_bytes),
                details: "webp: eager animated decode accumulated frames exceed safe limit",
            })?;
        if total_allocated_bytes > animation_limit_bytes {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands: 4,
                bytes: u128::from(total_allocated_bytes),
                limit_bytes: u128::from(animation_limit_bytes),
                details: "webp: eager animated decode accumulated frames exceed safe limit",
            });
        }
        previous_canvas = Some(canvas);
        dispose_method = frame.dispose_method;
        dispose_area = area;
        if !iter.advance()? {
            break;
        }
    }

    if let Some(last_canvas) = previous_canvas.take() {
        frames.push(image_from_u8_pixels::<F>(width, height, 4, last_canvas)?);
    }

    let Some(first_frame) = frames.first().cloned() else {
        return Err(ViprsError::Codec(
            "webp: animated decode produced no frames".into(),
        ));
    };

    let mut metadata = ImageMetadata {
        interpretation: Some(webp_interpretation(first_frame.bands())),
        n_pages: Some(frames.len() as u32),
        icc_profile,
        xmp,
        ..ImageMetadata::default()
    };
    if frames.len() > 1 {
        metadata.page_height = Some(first_frame.height());
    }

    let image = first_frame.with_metadata(metadata);
    if frames.len() > 1 {
        Ok(image.with_frames(frames))
    } else {
        Ok(image)
    }
}

fn webp_overlap_region(window: Region, area: WebpFrameArea) -> Option<Region> {
    let left = (window.x as u32).max(area.left);
    let top = (window.y as u32).max(area.top);
    let right = (window.x as u32)
        .saturating_add(window.width)
        .min(area.left.saturating_add(area.width));
    let bottom = (window.y as u32)
        .saturating_add(window.height)
        .min(area.top.saturating_add(area.height));
    if left >= right || top >= bottom {
        return None;
    }

    Some(Region::new(
        left as i32,
        top as i32,
        right - left,
        bottom - top,
    ))
}

fn webp_paint_rgba_window(
    canvas: &mut [u8],
    window: Region,
    pixels_u8: &[u8],
    decode_plan: WebpWindowDecodePlan,
    area: Region,
    blend: bool,
) -> Result<(), ViprsError> {
    let expected = (decode_plan.output_width as usize)
        .checked_mul(decode_plan.output_height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(4))
        .ok_or_else(|| ViprsError::Codec("webp: decoded window dimensions overflow".into()))?;
    if pixels_u8.len() != expected {
        return Err(ViprsError::Codec(format!(
            "webp: decoded window buffer length mismatch (got {}, expected {expected})",
            pixels_u8.len()
        )));
    }

    let canvas_stride = window.width as usize * 4;
    let src_stride = decode_plan.output_width as usize * 4;
    let src_offset_x = decode_plan.output_offset_x as usize;
    let src_offset_y = decode_plan.output_offset_y as usize;
    let dst_offset_x = area.x.saturating_sub(window.x) as usize;
    let dst_offset_y = area.y.saturating_sub(window.y) as usize;

    for y in 0..area.height as usize {
        let src_row = (src_offset_y + y) * src_stride + src_offset_x * 4;
        let dst_row = (dst_offset_y + y) * canvas_stride + dst_offset_x * 4;
        if blend {
            for x in 0..area.width as usize {
                let src_base = src_row + x * 4;
                let dst_base = dst_row + x * 4;
                webp_blend_rgba_pixel(
                    &pixels_u8[src_base..src_base + 4],
                    &mut canvas[dst_base..dst_base + 4],
                );
            }
        } else {
            let byte_len = area.width as usize * 4;
            canvas[dst_row..dst_row + byte_len]
                .copy_from_slice(&pixels_u8[src_row..src_row + byte_len]);
        }
    }

    Ok(())
}

pub(super) fn decode_animated_webp_region_into<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let expected_output = checked_webp_region_output_len(
        region,
        4,
        "webp: requested region exceeds addressable memory",
    )?;
    if output.len() != expected_output {
        return Err(ViprsError::Codec(format!(
            "webp: output buffer size mismatch (got {}, expected {expected_output})",
            output.len()
        )));
    }

    let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
    let shrink_plan = webp_anim_shrink_on_load_plan(shrink_factor);
    let demux = WebpDemux::new(src)?;
    let canvas_width = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_WIDTH);
    let canvas_height = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_HEIGHT);
    let frame_count = demux.feature(WebPFormatFeature::WEBP_FF_FRAME_COUNT);
    if frame_count == 0 {
        return Err(ViprsError::Codec(
            "webp: animated decode produced no frames".into(),
        ));
    }

    let effective_shrink_factor =
        webp_animation_shrink_factor(&demux, canvas_width, canvas_height, shrink_plan)?;
    let width = webp_scaled_animation_dimension(canvas_width, effective_shrink_factor);
    let height = webp_scaled_animation_dimension(canvas_height, effective_shrink_factor);
    let Some(coverage) = webp_clamped_coverage_region(region, width, height) else {
        return Ok(());
    };
    let strip_window = Region::new(0, coverage.y, width, coverage.height);

    let canvas_len = checked_webp_region_output_len(
        strip_window,
        4,
        "webp: animated canvas dimensions exceed addressable memory",
    )?;
    let mut canvas = vec![0u8; canvas_len];
    let mut frame_pixels = Vec::new();
    let mut dispose_method = WebPMuxAnimDispose::WEBP_MUX_DISPOSE_NONE;
    let mut dispose_area = WebpFrameArea::default();
    let mut iter = demux.first_frame()?;

    loop {
        if dispose_method == WebPMuxAnimDispose::WEBP_MUX_DISPOSE_BACKGROUND
            && let Some(dispose_overlap) = webp_overlap_region(strip_window, dispose_area)
        {
            let clear_area = WebpFrameArea {
                left: u32::try_from(dispose_overlap.x.saturating_sub(strip_window.x))
                    .map_err(|_| ViprsError::Codec("webp: negative dispose x overlap".into()))?,
                top: u32::try_from(dispose_overlap.y.saturating_sub(strip_window.y))
                    .map_err(|_| ViprsError::Codec("webp: negative dispose y overlap".into()))?,
                width: dispose_overlap.width,
                height: dispose_overlap.height,
            };
            webp_clear_rgba_area(
                &mut canvas,
                strip_window.width,
                strip_window.height,
                clear_area,
            );
        }

        let frame = *iter.current();
        let input_width = u32::try_from(frame.width)
            .map_err(|_| ViprsError::Codec("webp: negative frame width".into()))?;
        let input_height = u32::try_from(frame.height)
            .map_err(|_| ViprsError::Codec("webp: negative frame height".into()))?;
        let area = webp_frame_area(&frame, effective_shrink_factor)?;
        if let Some(overlap) = webp_overlap_region(strip_window, area) {
            let source_left = if overlap.x as u32 > area.left {
                (overlap.x as u32 - area.left) * u32::from(effective_shrink_factor)
            } else {
                0
            };
            let source_top = (overlap.y as u32 - area.top) * u32::from(effective_shrink_factor);
            let plan = webp_decode_rgba_fragment_window(
                frame.fragment,
                input_width,
                input_height,
                source_left,
                source_top,
                overlap.width,
                overlap.height,
                effective_shrink_factor,
                &mut frame_pixels,
            )?;
            webp_paint_rgba_window(
                &mut canvas,
                strip_window,
                &frame_pixels,
                plan,
                overlap,
                frame.frame_num > 1 && frame.blend_method == WebPMuxAnimBlend::WEBP_MUX_BLEND,
            )?;
        }

        dispose_method = frame.dispose_method;
        dispose_area = area;
        if !iter.advance()? {
            break;
        }
    }

    copy_clamped_region_from_u8_window::<F>(
        width,
        height,
        4,
        strip_window,
        &canvas,
        0,
        0,
        region,
        output,
    )
}
