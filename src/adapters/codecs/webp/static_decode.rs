#[cfg(test)]
use std::sync::atomic::Ordering;
use std::{
    ffi::{c_int, c_void},
    sync::Arc,
};

use libwebp_sys::{
    VP8StatusCode, WEBP_CSP_MODE, WebPDecoderConfig, WebPFormatFeature, WebPIAppend, WebPIDecode,
    WebPIDelete,
};
use webp::BitstreamFeatures;

use super::WebpCodec;
use super::animated::{
    WebpDemux, decode_animated_webp, decode_animated_webp_region_into,
    webp_animation_shrink_factor, webp_scaled_animation_dimension,
};
#[cfg(test)]
use super::common::WEBP_STATIC_REGION_FRAME_DECODES;
use super::common::{
    CachedStaticWebpFrame, WEBP_ICC_CHUNK_FOURCC, WEBP_INCREMENTAL_CHUNK_SIZE, WEBP_MODE_LAST,
    WEBP_STATIC_REGION_CACHE_CAPACITY, WEBP_XMP_CHUNK_FOURCC, WebpWindowDecodePlan,
    checked_webp_scratch_allocation_len, copy_clamped_region_from_u8_window, image_from_u8_pixels,
    require_u8, validate_webp_riff_size, webp_anim_shrink_on_load_plan,
    webp_clamped_coverage_region, webp_decode_window_into_buffer, webp_interpretation,
    webp_scaled_dimension, webp_shrink_on_load_plan, webp_static_region_cache,
    webp_static_region_cache_key, webp_window_decode_plan,
};
use crate::domain::codec_options::LoadOptions;
use crate::domain::error::ViprsError;
use crate::domain::format::BandFormat;
use crate::domain::image::{Image, ImageMetadata, Region};
use crate::ports::codec::{ImageDecoder, ImageMetadataProbe, TileImageDecoder};

type WebpSamplerRowFunc = unsafe extern "C" fn(*const u8, *const u8, *const u8, *mut u8, c_int);
type WebpUpsampleLinePairFunc = unsafe extern "C" fn(
    *const u8,
    *const u8,
    *const u8,
    *const u8,
    *const u8,
    *const u8,
    *mut u8,
    *mut u8,
    c_int,
);

#[repr(C)]
struct Vp8Io {
    width: c_int,
    height: c_int,
    mb_y: c_int,
    mb_w: c_int,
    mb_h: c_int,
    y: *const u8,
    u: *const u8,
    v: *const u8,
    y_stride: c_int,
    uv_stride: c_int,
    opaque: *mut c_void,
    put: Option<unsafe extern "C" fn(*const Vp8Io) -> c_int>,
    setup: Option<unsafe extern "C" fn(*mut Vp8Io) -> c_int>,
    teardown: Option<unsafe extern "C" fn(*const Vp8Io)>,
    fancy_upsampling: c_int,
    data_size: usize,
    data: *const u8,
    bypass_filtering: c_int,
    use_cropping: c_int,
    crop_left: c_int,
    crop_right: c_int,
    crop_top: c_int,
    crop_bottom: c_int,
    use_scaling: c_int,
    scaled_width: c_int,
    scaled_height: c_int,
    a: *const u8,
}

unsafe extern "C" {
    fn WebPIoInitFromOptions(
        options: *const libwebp_sys::WebPDecoderOptions,
        io: *mut Vp8Io,
        out_colorspace: WEBP_CSP_MODE,
    ) -> c_int;
    fn WebPISetIOHooks(
        idec: *mut libwebp_sys::WebPIDecoder,
        put: Option<unsafe extern "C" fn(*const Vp8Io) -> c_int>,
        setup: Option<unsafe extern "C" fn(*mut Vp8Io) -> c_int>,
        teardown: Option<unsafe extern "C" fn(*const Vp8Io)>,
        user_data: *mut c_void,
    ) -> c_int;
    fn WebPInitSamplers();
    fn WebPInitUpsamplers();
    fn WebPSamplerProcessPlane(
        y: *const u8,
        y_stride: c_int,
        u: *const u8,
        v: *const u8,
        uv_stride: c_int,
        dst: *mut u8,
        dst_stride: c_int,
        width: c_int,
        height: c_int,
        func: Option<WebpSamplerRowFunc>,
    );
    static mut WebPSamplers: [Option<WebpSamplerRowFunc>; WEBP_MODE_LAST];
    static mut WebPUpsamplers: [Option<WebpUpsampleLinePairFunc>; WEBP_MODE_LAST];
}

struct WebpStripDecodeState {
    decode_start: usize,
    decode_end: usize,
    stride: usize,
    bands: usize,
    options: libwebp_sys::WebPDecoderOptions,
    output: *mut u8,
    sampler: Option<WebpSamplerRowFunc>,
    upsampler: Option<WebpUpsampleLinePairFunc>,
    tmp_y: Vec<u8>,
    tmp_u: Vec<u8>,
    tmp_v: Vec<u8>,
    scratch_top: Vec<u8>,
    scratch_bottom: Vec<u8>,
    completed: bool,
}

impl WebpStripDecodeState {
    fn new(
        decode_start: usize,
        decode_end: usize,
        stride: usize,
        bands: usize,
        options: libwebp_sys::WebPDecoderOptions,
        output: &mut [u8],
        sampler: Option<WebpSamplerRowFunc>,
        upsampler: Option<WebpUpsampleLinePairFunc>,
    ) -> Self {
        Self {
            decode_start,
            decode_end,
            stride,
            bands,
            options,
            output: output.as_mut_ptr(),
            sampler,
            upsampler,
            tmp_y: Vec::new(),
            tmp_u: Vec::new(),
            tmp_v: Vec::new(),
            scratch_top: vec![0; stride],
            scratch_bottom: vec![0; stride],
            completed: false,
        }
    }

    #[inline]
    fn is_rgba(&self) -> bool {
        self.bands == 4
    }

    #[inline]
    unsafe fn row_ptr(&mut self, row: usize, top: bool) -> *mut u8 {
        if row >= self.decode_start && row < self.decode_end {
            let row_offset = (row - self.decode_start) * self.stride;
            // SAFETY: `row_offset` is bounded by the strip output size prepared by the caller.
            unsafe { self.output.add(row_offset) }
        } else if top {
            self.scratch_top.as_mut_ptr()
        } else {
            self.scratch_bottom.as_mut_ptr()
        }
    }

    fn copy_alpha_rows(
        &mut self,
        alpha: *const u8,
        alpha_base_row: usize,
        row_width: usize,
        start_row: usize,
        rows: usize,
    ) {
        if !self.is_rgba() || rows == 0 {
            return;
        }

        for row in start_row..start_row + rows {
            if row < self.decode_start || row >= self.decode_end {
                continue;
            }
            let rel = row - alpha_base_row;
            let alpha_row = if alpha.is_null() {
                None
            } else {
                // SAFETY: callers pass the alpha pointer corresponding to `alpha_base_row`.
                Some(unsafe { std::slice::from_raw_parts(alpha.add(rel * row_width), row_width) })
            };
            let dst_offset = (row - self.decode_start) * self.stride;
            let dst_row = {
                // SAFETY: `dst_offset` points inside the output strip owned by the state, so the
                // output pointer plus `self.stride` bytes forms a writable destination row.
                unsafe { std::slice::from_raw_parts_mut(self.output.add(dst_offset), self.stride) }
            };
            for x in 0..row_width {
                dst_row[x * 4 + 3] = alpha_row.map_or(255, |alpha| alpha[x]);
            }
        }
    }

    fn mark_completed(&mut self, ready_end: usize) -> c_int {
        if ready_end >= self.decode_end {
            self.completed = true;
            0
        } else {
            1
        }
    }
}

unsafe extern "C" fn webp_strip_setup(io: *mut Vp8Io) -> c_int {
    // SAFETY: libwebp calls the hook with the `Vp8Io` instance it owns for the active decode.
    let io = unsafe { &mut *io };
    // SAFETY: `opaque` points to the state passed to `WebPISetIOHooks`.
    let state = unsafe { &mut *(io.opaque.cast::<WebpStripDecodeState>()) };
    let intermediate_mode = if state.is_rgba() {
        WEBP_CSP_MODE::MODE_YUVA
    } else {
        WEBP_CSP_MODE::MODE_YUV
    };
    // SAFETY: `state.options` outlives the callback, and `io` is libwebp's live decode state.
    let ok =
        unsafe { WebPIoInitFromOptions(std::ptr::from_ref(&state.options), io, intermediate_mode) };
    if ok == 0 {
        return 0;
    }
    if io.fancy_upsampling != 0 {
        let uv_width = ((io.mb_w as usize) + 1) >> 1;
        state.tmp_y.resize(io.mb_w as usize, 0);
        state.tmp_u.resize(uv_width, 0);
        state.tmp_v.resize(uv_width, 0);
    }
    1
}

unsafe extern "C" fn webp_strip_teardown(_io: *const Vp8Io) {}

unsafe fn webp_strip_write_sampled(io: &Vp8Io, state: &mut WebpStripDecodeState) -> c_int {
    let overlap_start = (io.mb_y as usize).max(state.decode_start);
    let overlap_end = (io.mb_y as usize + io.mb_h as usize).min(state.decode_end);
    if overlap_start < overlap_end {
        let dst_offset = (overlap_start - state.decode_start) * state.stride;
        let y_offset = (overlap_start - io.mb_y as usize) * io.y_stride as usize;
        let uv_offset = ((overlap_start - io.mb_y as usize) / 2) * io.uv_stride as usize;
        // SAFETY: the overlap slices and destination row span valid memory owned by libwebp and
        // by the caller-provided output strip respectively.
        unsafe {
            WebPSamplerProcessPlane(
                io.y.add(y_offset),
                io.y_stride,
                io.u.add(uv_offset),
                io.v.add(uv_offset),
                io.uv_stride,
                state.output.add(dst_offset),
                state.stride as c_int,
                io.mb_w,
                (overlap_end - overlap_start) as c_int,
                state.sampler,
            );
        }
        state.copy_alpha_rows(
            io.a,
            io.mb_y as usize,
            io.width as usize,
            overlap_start,
            overlap_end - overlap_start,
        );
    }
    state.mark_completed(io.mb_y as usize + io.mb_h as usize)
}

unsafe fn webp_strip_write_fancy(io: &Vp8Io, state: &mut WebpStripDecodeState) -> c_int {
    let Some(upsample) = state.upsampler else {
        return 0;
    };

    let mut cur_y = io.y;
    let mut cur_u = io.u;
    let mut cur_v = io.v;
    let mut y = io.mb_y as usize;
    let y_end = io.mb_y as usize + io.mb_h as usize;
    let uv_width = ((io.mb_w as usize) + 1) >> 1;
    let last_call = io.crop_top as usize + y_end >= io.crop_bottom as usize;
    let mut ready_end = if y == 0 {
        // SAFETY: the upsampler writes one scanline into either the target row or scratch row.
        unsafe {
            upsample(
                cur_y,
                std::ptr::null(),
                cur_u,
                cur_v,
                cur_u,
                cur_v,
                state.row_ptr(0, true),
                std::ptr::null_mut(),
                io.mb_w,
            );
        }
        1
    } else {
        // SAFETY: the saved rows come from the previous callback and remain valid until reused.
        unsafe {
            upsample(
                state.tmp_y.as_ptr(),
                cur_y,
                state.tmp_u.as_ptr(),
                state.tmp_v.as_ptr(),
                cur_u,
                cur_v,
                state.row_ptr(y - 1, true),
                state.row_ptr(y, false),
                io.mb_w,
            );
        }
        y + 1
    };

    while y + 2 < y_end {
        let top_u = cur_u;
        let top_v = cur_v;
        // SAFETY: each callback exposes contiguous Y and UV rows for the current macroblock strip.
        unsafe {
            cur_u = cur_u.add(io.uv_stride as usize);
            cur_v = cur_v.add(io.uv_stride as usize);
            cur_y = cur_y.add(2 * io.y_stride as usize);
            upsample(
                cur_y.sub(io.y_stride as usize),
                cur_y,
                top_u,
                top_v,
                cur_u,
                cur_v,
                state.row_ptr(y + 1, true),
                state.row_ptr(y + 2, false),
                io.mb_w,
            );
        }
        ready_end = y + 3;
        y += 2;
    }

    // SAFETY: advance to the trailing row that may need to be carried across callbacks.
    unsafe {
        cur_y = cur_y.add(io.y_stride as usize);
    }

    if !last_call {
        // SAFETY: the last undecoded row and its chroma line stay valid for the duration of the callback.
        unsafe {
            std::ptr::copy_nonoverlapping(cur_y, state.tmp_y.as_mut_ptr(), io.mb_w as usize);
            std::ptr::copy_nonoverlapping(cur_u, state.tmp_u.as_mut_ptr(), uv_width);
            std::ptr::copy_nonoverlapping(cur_v, state.tmp_v.as_mut_ptr(), uv_width);
        }
        ready_end = ready_end.min(y_end.saturating_sub(1));
    } else if y_end & 1 == 0 {
        // SAFETY: the mirrored final row writes into either the target row or scratch row.
        unsafe {
            upsample(
                cur_y,
                std::ptr::null(),
                cur_u,
                cur_v,
                cur_u,
                cur_v,
                state.row_ptr(y_end - 1, true),
                std::ptr::null_mut(),
                io.mb_w,
            );
        }
        ready_end = y_end;
    }

    let alpha_start = if io.mb_y == 0 {
        0
    } else {
        io.mb_y as usize - 1
    };
    let alpha_rows = if last_call {
        state
            .decode_end
            .min(io.crop_bottom as usize)
            .saturating_sub(alpha_start)
    } else if io.mb_y == 0 {
        io.mb_h as usize - 1
    } else {
        io.mb_h as usize
    };
    let alpha_ptr = if io.a.is_null() || alpha_start == io.mb_y as usize {
        io.a
    } else {
        // SAFETY: libwebp keeps the alpha plane for the current block persistent, so the row
        // immediately above `io.a` is still readable during the fancy-upsampling callback.
        unsafe { io.a.sub(io.width as usize) }
    };
    state.copy_alpha_rows(
        alpha_ptr,
        alpha_start,
        io.width as usize,
        alpha_start,
        alpha_rows,
    );
    state.mark_completed(ready_end)
}

unsafe extern "C" fn webp_strip_put(io: *const Vp8Io) -> c_int {
    // SAFETY: libwebp calls the hook with the active decode context.
    let io = unsafe { &*io };
    // SAFETY: `opaque` points to the strip state installed at decoder setup time.
    let state = unsafe { &mut *(io.opaque.cast::<WebpStripDecodeState>()) };
    if io.mb_w <= 0 || io.mb_h <= 0 {
        return 0;
    }
    if io.fancy_upsampling != 0 {
        // SAFETY: the hook only touches memory ranges that belong to this decode callback.
        unsafe { webp_strip_write_fancy(io, state) }
    } else {
        // SAFETY: the hook only touches memory ranges that belong to this decode callback.
        unsafe { webp_strip_write_sampled(io, state) }
    }
}

fn webp_decode_strip_incremental_into_buffer(
    src: &[u8],
    input_width: u32,
    _input_height: u32,
    bands: u32,
    strip_top: u32,
    strip_height: u32,
    output: &mut [u8],
) -> Result<WebpWindowDecodePlan, ViprsError> {
    let plan = webp_window_decode_plan(0, strip_top, input_width, strip_height, 1)?;
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

    let colorspace = if bands == 4 {
        WEBP_CSP_MODE::MODE_RGBA
    } else {
        WEBP_CSP_MODE::MODE_RGB
    };
    let mode_index = colorspace as usize;
    if mode_index >= WEBP_MODE_LAST {
        return Err(ViprsError::Codec("webp: colorspace index overflow".into()));
    }

    // SAFETY: libwebp exposes these function tables globally after explicit initialization.
    let (sampler, upsampler) = unsafe {
        WebPInitSamplers();
        WebPInitUpsamplers();
        (WebPSamplers[mode_index], WebPUpsamplers[mode_index])
    };
    if sampler.is_none() {
        return Err(ViprsError::Codec(
            "webp: sampler initialization failed".into(),
        ));
    }
    if upsampler.is_none() {
        return Err(ViprsError::Codec(
            "webp: upsampler initialization failed".into(),
        ));
    }

    let mut config = WebPDecoderConfig::new()
        .map_err(|()| ViprsError::Codec("webp: decoder config init failed".into()))?;
    config.output.colorspace = colorspace;
    config.options.use_threads = 1;
    let mut state = WebpStripDecodeState::new(
        plan.crop_top as usize,
        plan.crop_top as usize + plan.crop_height as usize,
        plan.output_width as usize * bands as usize,
        bands as usize,
        config.options,
        output,
        sampler,
        upsampler,
    );

    // SAFETY: `config` and `state` remain alive for the decoder lifetime, and the hooks only
    // borrow `state` during `WebPIAppend`.
    let idec = unsafe { WebPIDecode(std::ptr::null(), 0, std::ptr::from_mut(&mut config)) };
    if idec.is_null() {
        return Err(ViprsError::Codec(
            "webp: incremental decoder init failed".into(),
        ));
    }

    let result = (|| {
        // SAFETY: `idec` is live and `state` stays pinned on the stack for the whole decode.
        let hooks_ok = unsafe {
            WebPISetIOHooks(
                idec,
                Some(webp_strip_put),
                Some(webp_strip_setup),
                Some(webp_strip_teardown),
                std::ptr::from_mut(&mut state).cast::<c_void>(),
            )
        };
        if hooks_ok == 0 {
            return Err(ViprsError::Codec("webp: io hook setup failed".into()));
        }

        let mut offset = 0usize;
        while offset < src.len() && !state.completed {
            let end = (offset + WEBP_INCREMENTAL_CHUNK_SIZE).min(src.len());
            let chunk = &src[offset..end];
            // SAFETY: `chunk` is valid input for the duration of the append call.
            let status = unsafe { WebPIAppend(idec, chunk.as_ptr(), chunk.len()) };
            match status {
                VP8StatusCode::VP8_STATUS_OK | VP8StatusCode::VP8_STATUS_SUSPENDED => {}
                VP8StatusCode::VP8_STATUS_USER_ABORT if state.completed => break,
                _ => {
                    return Err(ViprsError::Codec(format!(
                        "webp: incremental decode failed with status {status:?}"
                    )));
                }
            }
            offset = end;
        }

        if !state.completed {
            return Err(ViprsError::Codec(
                "webp: incremental decode ended before strip completed".into(),
            ));
        }

        Ok(plan)
    })();

    // SAFETY: `idec` was created above and must be released exactly once.
    unsafe { WebPIDelete(idec) };
    result
}
fn decode_static_webp<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
    icc_profile: Option<Vec<u8>>,
    xmp: Option<Vec<u8>>,
) -> Result<Image<F>, ViprsError> {
    let (width, height, bands, pixels_u8) = decode_static_webp_pixels(src, opts)?;
    let image = image_from_u8_pixels::<F>(width, height, bands, pixels_u8)?;
    Ok(image.with_metadata(ImageMetadata {
        interpretation: Some(webp_interpretation(bands)),
        n_pages: Some(1),
        icc_profile,
        xmp,
        ..ImageMetadata::default()
    }))
}

pub(crate) fn decode_static_webp_pixels(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<(u32, u32, u32, Vec<u8>), ViprsError> {
    let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
    let shrink_plan = webp_shrink_on_load_plan(shrink_factor);
    let features = BitstreamFeatures::new(src)
        .ok_or_else(|| ViprsError::Codec("webp: decode failed".into()))?;
    let bands = if features.has_alpha() { 4 } else { 3 };
    let width = webp_scaled_dimension(features.width(), shrink_plan.factor());
    let height = webp_scaled_dimension(features.height(), shrink_plan.factor());
    let len = checked_webp_scratch_allocation_len(
        width,
        height,
        bands,
        "webp: eager static decode allocation exceeds safe limit",
    )?;
    let mut pixels_u8 = vec![0u8; len];
    let plan = webp_window_decode_plan(0, 0, width, height, shrink_plan.factor())?;
    webp_decode_window_into_buffer(
        src.as_ptr(),
        src.len(),
        features.width(),
        features.height(),
        bands,
        plan,
        &mut pixels_u8,
    )?;

    Ok((width, height, bands, pixels_u8))
}

fn cached_static_webp_frame(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<Arc<CachedStaticWebpFrame>, ViprsError> {
    let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
    let cache_key =
        webp_static_region_cache_key(src, webp_shrink_on_load_plan(shrink_factor).factor());
    if let Some(frame) = webp_static_region_cache()
        .read()
        .map_err(|_| ViprsError::Codec("webp: static region cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(frame);
    }

    #[cfg(test)]
    WEBP_STATIC_REGION_FRAME_DECODES.fetch_add(1, Ordering::Relaxed);
    let (width, height, bands, pixels) = decode_static_webp_pixels(src, opts)?;
    let decoded = Arc::new(CachedStaticWebpFrame {
        width,
        height,
        bands,
        pixels,
    });
    let mut cache = webp_static_region_cache()
        .write()
        .map_err(|_| ViprsError::Codec("webp: static region cache poisoned".into()))?;
    if let Some(existing) = cache.get(&cache_key).cloned() {
        return Ok(existing);
    }
    if cache.len() >= WEBP_STATIC_REGION_CACHE_CAPACITY {
        cache.clear();
    }
    cache.insert(cache_key, Arc::clone(&decoded));
    Ok(decoded)
}
fn decode_static_webp_region_into<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
    let shrink_plan = webp_shrink_on_load_plan(shrink_factor);
    let features = BitstreamFeatures::new(src)
        .ok_or_else(|| ViprsError::Codec("webp: decode failed".into()))?;
    let bands = if features.has_alpha() { 4 } else { 3 };
    let width = webp_scaled_dimension(features.width(), shrink_plan.factor());
    let height = webp_scaled_dimension(features.height(), shrink_plan.factor());
    let Some(coverage) = webp_clamped_coverage_region(region, width, height) else {
        return Ok(());
    };
    let strip_window = Region::new(0, coverage.y, width, coverage.height);
    if shrink_plan.factor() == 1 && bands == 3 {
        let scratch_height = coverage.height + (coverage.y.rem_euclid(2) as u32);
        let scratch_len = checked_webp_scratch_allocation_len(
            width,
            scratch_height,
            bands,
            "webp: static strip scratch allocation exceeds safe limit",
        )?;
        let mut window_pixels = vec![0u8; scratch_len];
        let plan = webp_decode_strip_incremental_into_buffer(
            src,
            features.width(),
            features.height(),
            bands,
            coverage.y as u32,
            coverage.height,
            &mut window_pixels,
        )?;
        copy_clamped_region_from_u8_window::<F>(
            width,
            height,
            bands,
            strip_window,
            &window_pixels,
            plan.output_offset_x,
            plan.output_offset_y,
            region,
            output,
        )?;
        return Ok(());
    }

    let cached = cached_static_webp_frame(src, opts)?;
    if cached.width != width || cached.height != height || cached.bands != bands {
        return Err(ViprsError::Codec(
            "webp: cached static region frame metadata mismatch".into(),
        ));
    }
    copy_clamped_region_from_u8_window::<F>(
        width,
        height,
        bands,
        Region::new(0, 0, cached.width, cached.height),
        &cached.pixels,
        0,
        0,
        region,
        output,
    )
}
impl ImageDecoder for WebpCodec {
    fn format_name(&self) -> &'static str {
        "webp"
    }

    /// Recognise a WebP stream by its RIFF/WEBP container magic.
    ///
    /// WebP container layout:
    ///   bytes [0..4]  = b"RIFF"
    ///   bytes [4..8]  = file size (LE u32, ignored here)
    ///   bytes [8..12] = b"WEBP"
    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.len() >= 12 && &header[0..4] == b"RIFF" && &header[8..12] == b"WEBP"
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
        require_u8::<F>()?;
        validate_webp_riff_size(src)?;
        let demux = WebpDemux::new(src)?;
        let icc_profile = demux.chunk(WEBP_ICC_CHUNK_FOURCC);
        let xmp = demux.chunk(WEBP_XMP_CHUNK_FOURCC);
        // max_dimension: not supported — decoder does not expose pre-decode scaling.
        let features = BitstreamFeatures::new(src)
            .ok_or_else(|| ViprsError::Codec("webp: decode failed".into()))?;
        if features.has_animation() {
            decode_animated_webp(src, opts, icc_profile, xmp)
        } else {
            decode_static_webp(src, opts, icc_profile, xmp)
        }
    }

    /// Probe `src` from the WebP bitstream header without decoding pixels.
    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        validate_webp_riff_size(src)?;
        let features = BitstreamFeatures::new(src)
            .ok_or_else(|| ViprsError::Codec("webp: probe failed".into()))?;
        let bands = if features.has_animation() || features.has_alpha() {
            4
        } else {
            3
        };
        Ok((features.width(), features.height(), bands))
    }
}

impl TileImageDecoder for WebpCodec {
    fn probe_with_options(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let demux = WebpDemux::new(src)?;
        let icc_profile = demux.chunk(WEBP_ICC_CHUNK_FOURCC);
        let xmp = demux.chunk(WEBP_XMP_CHUNK_FOURCC);
        let features = BitstreamFeatures::new(src)
            .ok_or_else(|| ViprsError::Codec("webp: probe failed".into()))?;
        if features.has_animation() {
            let canvas_width = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_WIDTH);
            let canvas_height = demux.feature(WebPFormatFeature::WEBP_FF_CANVAS_HEIGHT);
            let frame_count = demux.feature(WebPFormatFeature::WEBP_FF_FRAME_COUNT);
            let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
            let shrink_plan = webp_anim_shrink_on_load_plan(shrink_factor);
            let effective_shrink_factor =
                webp_animation_shrink_factor(&demux, canvas_width, canvas_height, shrink_plan)?;
            let metadata = ImageMetadata {
                interpretation: Some(webp_interpretation(4)),
                n_pages: Some(frame_count.max(1)),
                icc_profile,
                xmp,
                ..ImageMetadata::default()
            };
            return Ok(ImageMetadataProbe::new(
                webp_scaled_animation_dimension(canvas_width, effective_shrink_factor),
                webp_scaled_animation_dimension(canvas_height, effective_shrink_factor),
                4,
            )
            .with_metadata(metadata));
        }

        let shrink_factor = opts.shrink_factor.map_or(1, std::num::NonZeroU8::get);
        let shrink_plan = webp_shrink_on_load_plan(shrink_factor);
        let bands = if features.has_alpha() { 4 } else { 3 };
        let metadata = ImageMetadata {
            interpretation: Some(webp_interpretation(bands)),
            n_pages: Some(1),
            icc_profile,
            xmp,
            ..ImageMetadata::default()
        };
        Ok(ImageMetadataProbe::new(
            webp_scaled_dimension(features.width(), shrink_plan.factor()),
            webp_scaled_dimension(features.height(), shrink_plan.factor()),
            bands,
        )
        .with_metadata(metadata))
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
        require_u8::<F>()?;
        validate_webp_riff_size(src)?;
        let features = BitstreamFeatures::new(src)
            .ok_or_else(|| ViprsError::Codec("webp: decode failed".into()))?;
        if features.has_animation() {
            return decode_animated_webp_region_into::<F>(src, opts, region, output);
        }

        decode_static_webp_region_into::<F>(src, opts, region, output)
    }
}
