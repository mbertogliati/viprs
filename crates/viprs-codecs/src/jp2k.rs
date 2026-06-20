//! `Jp2K` adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "jp2k")]

//! JPEG 2000 codec using `jpeg2k` for decode and `OpenJPEG` FFI for encode.

use std::ffi::CString;
use std::num::NonZeroUsize;
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::ptr;

use jpeg2k::{
    ColorSpace as J2kColorSpace,
    format::{J2K_CODESTREAM_MAGIC, JP2_RFC3745_MAGIC},
};
use openjpeg_sys as sys;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const DEFAULT_QUALITY: u8 = 48;
const MAX_COMPONENTS: usize = 4;
const DEFAULT_DECODER_THREADS_CAP: usize = 8;
const DEFAULT_TILE_SIZE: u32 = 512;
const MEDIUM_IMAGE_TILE_SIZE: u32 = 1024;
const MEDIUM_IMAGE_TILE_MIN_DIM: u32 = 1536;
const MEDIUM_IMAGE_TILE_MAX_DIM: u32 = 4096;
const DEFAULT_CODEBLOCK_SIZE: i32 = 64;
const DEFAULT_PRECINCT_SIZE: i32 = 256;
const PROFILE_RES_SPEC: usize = 7;

/// The `Jp2kCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::jp2k::Jp2kCodec>();
/// ```
pub struct Jp2kCodec;

#[inline]
fn require_supported_format<F: BandFormat>() -> Result<(), ViprsError> {
    match F::ID {
        BandFormatId::U8 | BandFormatId::U16 => Ok(()),
        _ => Err(ViprsError::Codec(format!(
            "jp2k: unsupported format {:?}; only U8 and U16 are supported",
            F::ID
        ))),
    }
}

fn validate_load_options(opts: &LoadOptions) -> Result<(), ViprsError> {
    if let Some(value) = opts.n
        && value != -1
        && value != 1
    {
        return Err(ViprsError::Codec(format!(
            "jp2k: n must be 1 or -1, got {value}"
        )));
    }

    Ok(())
}

#[inline]
fn codec_format_from_bytes(src: &[u8]) -> Result<sys::CODEC_FORMAT, ViprsError> {
    if src.starts_with(J2K_CODESTREAM_MAGIC) {
        Ok(sys::CODEC_FORMAT::OPJ_CODEC_J2K)
    } else if src.starts_with(JP2_RFC3745_MAGIC) {
        Ok(sys::CODEC_FORMAT::OPJ_CODEC_JP2)
    } else {
        Err(ViprsError::Codec(
            "jp2k: unsupported codestream format for metadata probe".into(),
        ))
    }
}

#[inline]
fn codec_format_from_path(path: &Path) -> Result<sys::CODEC_FORMAT, ViprsError> {
    match path
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("j2k") => Ok(sys::CODEC_FORMAT::OPJ_CODEC_J2K),
        Some("jp2" | "jpf" | "jpx") => Ok(sys::CODEC_FORMAT::OPJ_CODEC_JP2),
        _ => Err(ViprsError::Codec(format!(
            "jp2k: unsupported file extension for metadata probe: {}",
            path.display()
        ))),
    }
}

struct WrappedSlice<'a> {
    offset: usize,
    buf: &'a [u8],
}

impl<'a> WrappedSlice<'a> {
    const fn new(buf: &'a [u8]) -> Self {
        Self { offset: 0, buf }
    }

    const fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.offset)
    }

    fn seek(&mut self, new_offset: usize) -> usize {
        self.offset = self.buf.len().min(new_offset);
        self.offset
    }

    fn consume(&mut self, n_bytes: usize) -> usize {
        let offset = self.offset.saturating_add(n_bytes);
        self.offset = self.buf.len().min(offset);
        self.offset
    }

    fn read_into(&mut self, out_buffer: &mut [u8]) -> Option<usize> {
        let remaining = self.remaining();
        if remaining == 0 {
            return None;
        }

        let n_read = remaining.min(out_buffer.len());
        let start = self.offset;
        let end = self.consume(n_read);
        out_buffer[..n_read].copy_from_slice(&self.buf[start..end]);
        Some(n_read)
    }
}

unsafe extern "C" fn buf_read_stream_free_fn(p_data: *mut c_void) {
    if p_data.is_null() {
        return;
    }

    // SAFETY: OpenJPEG passes back the Box allocation created in openjpeg_stream_from_bytes.
    unsafe { drop(Box::from_raw(p_data.cast::<WrappedSlice<'static>>())) };
}

unsafe extern "C" fn buf_read_stream_read_fn(
    p_buffer: *mut c_void,
    nb_bytes: usize,
    p_data: *mut c_void,
) -> usize {
    if p_buffer.is_null() || nb_bytes == 0 {
        return usize::MAX;
    }

    // SAFETY: OpenJPEG provides the same WrappedSlice pointer registered as user data.
    let slice = unsafe { &mut *p_data.cast::<WrappedSlice<'static>>() };
    // SAFETY: OpenJPEG requests a writable output buffer of nb_bytes bytes.
    let out_buf = unsafe { std::slice::from_raw_parts_mut(p_buffer.cast::<u8>(), nb_bytes) };
    slice.read_into(out_buf).unwrap_or(usize::MAX)
}

unsafe extern "C" fn buf_read_stream_skip_fn(nb_bytes: i64, p_data: *mut c_void) -> i64 {
    // SAFETY: OpenJPEG provides the same WrappedSlice pointer registered as user data.
    let slice = unsafe { &mut *p_data.cast::<WrappedSlice<'static>>() };
    slice.consume(nb_bytes.max(0) as usize) as i64
}

unsafe extern "C" fn buf_read_stream_seek_fn(nb_bytes: i64, p_data: *mut c_void) -> i32 {
    // SAFETY: OpenJPEG provides the same WrappedSlice pointer registered as user data.
    let slice = unsafe { &mut *p_data.cast::<WrappedSlice<'static>>() };
    let seek_offset = nb_bytes.max(0) as usize;
    let new_offset = slice.seek(seek_offset);
    i32::from(seek_offset == new_offset)
}

struct OpenJpegStream(*mut sys::opj_stream_t);

impl Drop for OpenJpegStream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: pointer is created by OpenJPEG stream constructors and freed once here.
            unsafe { sys::opj_stream_destroy(self.0) };
        }
    }
}

struct OpenJpegCodec(*mut sys::opj_codec_t);

impl Drop for OpenJpegCodec {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: pointer is created by opj_create_decompress and freed once here.
            unsafe { sys::opj_destroy_codec(self.0) };
        }
    }
}

struct OpenJpegImage(*mut sys::opj_image_t);

impl Drop for OpenJpegImage {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: pointer is created by opj_read_header and freed once here.
            unsafe { sys::opj_image_destroy(self.0) };
        }
    }
}

struct OpenJpegInfo(*mut sys::opj_codestream_info_v2_t);

impl Drop for OpenJpegInfo {
    fn drop(&mut self) {
        let mut info_ptr = self.0;
        if !info_ptr.is_null() {
            // SAFETY: pointer is created by opj_get_cstr_info and freed once here.
            unsafe { sys::opj_destroy_cstr_info(&raw mut info_ptr) };
        }
    }
}

struct OpenJpegDecodedImage {
    image: OpenJpegImage,
    resolution_count: u32,
}

impl OpenJpegDecodedImage {
    #[inline]
    fn num_components(&self) -> u32 {
        // SAFETY: self.image owns a live opj_image_t for the lifetime of this wrapper.
        unsafe { (*self.image.0).numcomps }
    }

    #[inline]
    fn components(&self) -> &[sys::opj_image_comp_t] {
        let len = self.num_components() as usize;
        // SAFETY: OpenJPEG stores `numcomps` contiguous component descriptors in `comps`.
        unsafe { std::slice::from_raw_parts((*self.image.0).comps, len) }
    }

    #[inline]
    fn color_space(&self) -> J2kColorSpace {
        // SAFETY: self.image owns a live opj_image_t for the lifetime of this wrapper.
        match unsafe { (*self.image.0).color_space } {
            sys::COLOR_SPACE::OPJ_CLRSPC_SRGB => J2kColorSpace::SRGB,
            sys::COLOR_SPACE::OPJ_CLRSPC_GRAY => J2kColorSpace::Gray,
            sys::COLOR_SPACE::OPJ_CLRSPC_SYCC => J2kColorSpace::SYCC,
            sys::COLOR_SPACE::OPJ_CLRSPC_EYCC => J2kColorSpace::EYCC,
            sys::COLOR_SPACE::OPJ_CLRSPC_CMYK => J2kColorSpace::CMYK,
            sys::COLOR_SPACE::OPJ_CLRSPC_UNSPECIFIED => J2kColorSpace::Unspecified,
            sys::COLOR_SPACE::OPJ_CLRSPC_UNKNOWN => J2kColorSpace::Unknown,
        }
    }
}

fn openjpeg_stream_from_bytes(src: &[u8]) -> Result<OpenJpegStream, ViprsError> {
    let data_ptr = Box::into_raw(Box::new(WrappedSlice::new(src))).cast::<c_void>();

    // SAFETY: OpenJPEG returns an owned input stream pointer or null on failure.
    let stream_ptr = unsafe { sys::opj_stream_default_create(1) };
    if stream_ptr.is_null() {
        // SAFETY: stream creation failed, so ownership of data_ptr stays with Rust.
        unsafe { drop(Box::from_raw(data_ptr.cast::<WrappedSlice<'static>>())) };
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG input stream".into(),
        ));
    }
    let stream_guard = OpenJpegStream(stream_ptr);

    // SAFETY: stream_guard owns the stream and data_ptr remains valid until stream destruction.
    unsafe {
        sys::opj_stream_set_read_function(stream_guard.0, Some(buf_read_stream_read_fn));
        sys::opj_stream_set_skip_function(stream_guard.0, Some(buf_read_stream_skip_fn));
        sys::opj_stream_set_seek_function(stream_guard.0, Some(buf_read_stream_seek_fn));
        sys::opj_stream_set_user_data_length(stream_guard.0, src.len() as u64);
        sys::opj_stream_set_user_data(stream_guard.0, data_ptr, Some(buf_read_stream_free_fn));
    }

    Ok(stream_guard)
}

fn openjpeg_stream_from_path(path: &Path) -> Result<OpenJpegStream, ViprsError> {
    let c_path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|err| ViprsError::Codec(format!("jp2k: invalid input path: {err}")))?;
    // SAFETY: c_path lives until stream creation returns; OpenJPEG copies the filename immediately.
    let stream_ptr = unsafe { sys::opj_stream_create_default_file_stream(c_path.as_ptr(), 1) };
    if stream_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG file stream".into(),
        ));
    }
    Ok(OpenJpegStream(stream_ptr))
}

fn decoder_params(opts: &LoadOptions) -> sys::opj_dparameters_t {
    let mut decoder_params = std::mem::MaybeUninit::<sys::opj_dparameters_t>::uninit();
    // SAFETY: OpenJPEG initializes every field in opj_dparameters_t.
    let mut decoder_params = unsafe {
        sys::opj_set_default_decoder_parameters(decoder_params.as_mut_ptr());
        decoder_params.assume_init()
    };
    decoder_params.cp_reduce = opts.page.unwrap_or(0);
    decoder_params
}

fn configure_decoder_threads(codec: *mut sys::opj_codec_t, opts: &LoadOptions) {
    let thread_count = opts
        .decoder_threads
        .map_or_else(
            || {
                std::thread::available_parallelism()
                    .map_or(1, NonZeroUsize::get)
                    .min(DEFAULT_DECODER_THREADS_CAP)
            },
            NonZeroUsize::get,
        )
        .min(i32::MAX as usize) as i32;
    if thread_count <= 1 {
        return;
    }

    // SAFETY: codec is a live decoder handle configured by OpenJPEG for the current decode call.
    unsafe {
        if sys::opj_has_thread_support() == 1 {
            let _ = sys::opj_codec_set_threads(codec, thread_count);
        }
    }
}

fn setup_openjpeg_decoder(
    codec_format: sys::CODEC_FORMAT,
    opts: &LoadOptions,
    decoder_params: &mut sys::opj_dparameters_t,
) -> Result<OpenJpegCodec, ViprsError> {
    // SAFETY: OpenJPEG returns an owned decoder pointer or null on failure.
    let codec_ptr = unsafe { sys::opj_create_decompress(codec_format) };
    if codec_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG decoder".into(),
        ));
    }
    let codec_guard = OpenJpegCodec(codec_ptr);

    // SAFETY: codec pointer is valid for the lifetime of the decode call.
    unsafe {
        sys::opj_set_error_handler(
            codec_guard.0,
            Some(opj_error_callback),
            std::ptr::null_mut(),
        );
    }

    // SAFETY: codec pointer and decoder params are initialized and valid.
    if unsafe { sys::opj_setup_decoder(codec_guard.0, decoder_params) } == 0 {
        return Err(ViprsError::Codec("jp2k: opj_setup_decoder failed".into()));
    }
    configure_decoder_threads(codec_guard.0, opts);

    Ok(codec_guard)
}

fn resolution_count_from_codec(codec: *mut sys::opj_codec_t) -> u32 {
    // SAFETY: codec remains valid for the duration of this call.
    let info_ptr = unsafe { sys::opj_get_cstr_info(codec) };
    if info_ptr.is_null() {
        return 1;
    }
    let info_guard = OpenJpegInfo(info_ptr);

    // SAFETY: info_ptr is valid until destroyed by OpenJpegInfo drop.
    let default_tile = unsafe { &(*info_guard.0).m_default_tile_info };
    if default_tile.tccp_info.is_null() {
        return 1;
    }

    // SAFETY: tccp_info is non-null and owned by the codestream info structure.
    unsafe { (*default_tile.tccp_info).numresolutions }.max(1)
}

fn decode_openjpeg_stream(
    codec_format: sys::CODEC_FORMAT,
    stream: &OpenJpegStream,
    opts: &LoadOptions,
) -> Result<OpenJpegDecodedImage, ViprsError> {
    let mut decoder_params = decoder_params(opts);
    let codec_guard = setup_openjpeg_decoder(codec_format, opts, &mut decoder_params)?;

    let mut image_ptr: *mut sys::opj_image_t = ptr::null_mut();
    // SAFETY: stream and codec are initialized and live for the duration of the call.
    if unsafe { sys::opj_read_header(stream.0, codec_guard.0, &raw mut image_ptr) } == 0 {
        return Err(ViprsError::Codec("jp2k: opj_read_header failed".into()));
    }
    let image_guard = OpenJpegImage(image_ptr);
    let resolution_count = resolution_count_from_codec(codec_guard.0);

    // SAFETY: codec, stream, and image remain live until the end of this function.
    let decoded = unsafe {
        sys::opj_decode(codec_guard.0, stream.0, image_guard.0) != 0
            && sys::opj_end_decompress(codec_guard.0, stream.0) != 0
    };
    if !decoded {
        return Err(ViprsError::Codec("jp2k: decoding failed".into()));
    }

    Ok(OpenJpegDecodedImage {
        image: image_guard,
        resolution_count,
    })
}

fn decode_openjpeg_bytes(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<OpenJpegDecodedImage, ViprsError> {
    let codec_format = codec_format_from_bytes(src)?;
    let stream = openjpeg_stream_from_bytes(src)?;
    decode_openjpeg_stream(codec_format, &stream, opts)
}

fn decode_openjpeg_path(
    path: &Path,
    opts: &LoadOptions,
) -> Result<OpenJpegDecodedImage, ViprsError> {
    let codec_format = codec_format_from_path(path)?;
    let stream = openjpeg_stream_from_path(path)?;
    decode_openjpeg_stream(codec_format, &stream, opts)
}

fn probe_openjpeg_stream(
    codec_format: sys::CODEC_FORMAT,
    stream: &OpenJpegStream,
) -> Result<(u32, u32, u32), ViprsError> {
    let probe_opts = LoadOptions::default();
    let mut decoder_params = decoder_params(&probe_opts);
    let codec_guard = setup_openjpeg_decoder(codec_format, &probe_opts, &mut decoder_params)?;

    let mut image_ptr: *mut sys::opj_image_t = ptr::null_mut();
    // SAFETY: stream and codec are initialized and live for the duration of the call.
    if unsafe { sys::opj_read_header(stream.0, codec_guard.0, &raw mut image_ptr) } == 0 {
        return Err(ViprsError::Codec(
            "jp2k probe: opj_read_header failed".into(),
        ));
    }
    let image_guard = OpenJpegImage(image_ptr);
    let image = OpenJpegDecodedImage {
        image: image_guard,
        resolution_count: 1,
    };
    let (width, height, bands) = validate_decode_layout(&image)?;
    Ok((width, height, bands))
}

fn probe_openjpeg_bytes(src: &[u8]) -> Result<(u32, u32, u32), ViprsError> {
    let codec_format = codec_format_from_bytes(src)?;
    let stream = openjpeg_stream_from_bytes(src)?;
    probe_openjpeg_stream(codec_format, &stream)
}

fn probe_openjpeg_path(path: &Path) -> Result<(u32, u32, u32), ViprsError> {
    let codec_format = codec_format_from_path(path)?;
    let stream = openjpeg_stream_from_path(path)?;
    probe_openjpeg_stream(codec_format, &stream)
}

fn decode_tiled_stream_fast<F: BandFormat>(
    codec_format: sys::CODEC_FORMAT,
    stream_ptr: *mut sys::opj_stream_t,
    opts: &LoadOptions,
) -> Result<Option<Image<F>>, ViprsError> {
    let stream_guard = OpenJpegStream(stream_ptr);
    // SAFETY: OpenJPEG returns an owned decoder pointer or null on failure.
    let codec_ptr = unsafe { sys::opj_create_decompress(codec_format) };
    if codec_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG decoder".into(),
        ));
    }
    let codec_guard = OpenJpegCodec(codec_ptr);

    let mut decoder_params = std::mem::MaybeUninit::<sys::opj_dparameters_t>::uninit();
    // SAFETY: OpenJPEG initializes every field in opj_dparameters_t.
    let mut decoder_params = unsafe {
        sys::opj_set_default_decoder_parameters(decoder_params.as_mut_ptr());
        decoder_params.assume_init()
    };
    decoder_params.cp_reduce = opts.page.unwrap_or(0);

    // SAFETY: codec pointer is valid for the lifetime of this decode call.
    unsafe {
        sys::opj_set_error_handler(
            codec_guard.0,
            Some(opj_error_callback),
            std::ptr::null_mut(),
        );
    }

    // SAFETY: codec pointer and decoder params are initialized and valid.
    if unsafe { sys::opj_setup_decoder(codec_guard.0, &raw mut decoder_params) } == 0 {
        return Err(ViprsError::Codec("jp2k: opj_setup_decoder failed".into()));
    }
    configure_decoder_threads(codec_guard.0, opts);

    let mut image_ptr: *mut sys::opj_image_t = ptr::null_mut();
    // SAFETY: stream and codec are initialized and live for the duration of the call.
    if unsafe { sys::opj_read_header(stream_guard.0, codec_guard.0, &raw mut image_ptr) } == 0 {
        return Err(ViprsError::Codec("jp2k: opj_read_header failed".into()));
    }
    let image_guard = OpenJpegImage(image_ptr);

    // SAFETY: codec remains valid after opj_read_header and OpenJPEG owns the returned pointer.
    let info_ptr = unsafe { sys::opj_get_cstr_info(codec_guard.0) };
    if info_ptr.is_null() {
        return Ok(None);
    }
    let info_guard = OpenJpegInfo(info_ptr);

    // SAFETY: both pointers are owned by OpenJPEG guards for this scope.
    let image = unsafe { &*image_guard.0 };
    // SAFETY: both pointers are owned by OpenJPEG guards for this scope.
    let info = unsafe { &*info_guard.0 };
    let Some(plan) = fast_tiled_decode_plan::<F>(image, info, opts)? else {
        return Ok(None);
    };

    let pixel_count = (plan.width as usize)
        .checked_mul(plan.height as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image dimensions overflow".into()))?;
    let output_len = pixel_count
        .checked_mul(plan.bands as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: output buffer overflow".into()))?;
    let metadata = raw_decoded_metadata(image, plan.bands, plan.resolution_count);

    match F::ID {
        BandFormatId::U8 if plan.precision == 8 => {
            let mut pixels = vec![0_u8; output_len];
            for tile_y in 0..plan.tiles_down {
                for tile_x in 0..plan.tiles_across {
                    let tile_index = tile_y * plan.tiles_across + tile_x;
                    // SAFETY: codec, stream, and image belong to this decode session; tile indices
                    // are bounded by the codestream tile grid.
                    if unsafe {
                        sys::opj_get_decoded_tile(
                            codec_guard.0,
                            stream_guard.0,
                            image_guard.0,
                            tile_index,
                        )
                    } == 0
                    {
                        return Err(ViprsError::Codec(format!(
                            "jp2k decode: failed to decode tile {tile_index}"
                        )));
                    }

                    let tile_left = tile_x.saturating_mul(plan.tile_width);
                    let tile_top = tile_y.saturating_mul(plan.tile_height);
                    // SAFETY: `image_guard` still owns the image filled by `opj_get_decoded_tile`
                    // above, so dereferencing it yields the current decoded tile image.
                    let tile_image = unsafe { &*image_guard.0 };
                    match plan.bands {
                        1 => copy_fast_tiled_u8::<1>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        2 => copy_fast_tiled_u8::<2>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        3 => copy_fast_tiled_u8::<3>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        _ => copy_fast_tiled_u8::<4>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                    }
                }
            }

            let samples = bytemuck::allocation::try_cast_vec::<u8, F::Sample>(pixels)
                .map_err(|(err, _)| ViprsError::Codec(format!("jp2k: cast error: {err:?}")))?;
            Image::from_buffer(plan.width, plan.height, plan.bands, samples)
                .map(|image| image.with_metadata(metadata))
                .map(Some)
                .map_err(|err| ViprsError::Codec(err.to_string()))
        }
        BandFormatId::U16 if plan.precision == 16 => {
            let mut pixels = vec![0_u16; output_len];
            for tile_y in 0..plan.tiles_down {
                for tile_x in 0..plan.tiles_across {
                    let tile_index = tile_y * plan.tiles_across + tile_x;
                    // SAFETY: codec, stream, and image belong to this decode session; tile indices
                    // are bounded by the codestream tile grid.
                    if unsafe {
                        sys::opj_get_decoded_tile(
                            codec_guard.0,
                            stream_guard.0,
                            image_guard.0,
                            tile_index,
                        )
                    } == 0
                    {
                        return Err(ViprsError::Codec(format!(
                            "jp2k decode: failed to decode tile {tile_index}"
                        )));
                    }

                    let tile_left = tile_x.saturating_mul(plan.tile_width);
                    let tile_top = tile_y.saturating_mul(plan.tile_height);
                    // SAFETY: `image_guard` still owns the image filled by `opj_get_decoded_tile`
                    // above, so dereferencing it yields the current decoded tile image.
                    let tile_image = unsafe { &*image_guard.0 };
                    match plan.bands {
                        1 => copy_fast_tiled_u16::<1>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        2 => copy_fast_tiled_u16::<2>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        3 => copy_fast_tiled_u16::<3>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                        _ => copy_fast_tiled_u16::<4>(
                            &mut pixels,
                            tile_image,
                            &plan,
                            tile_left,
                            tile_top,
                        )?,
                    }
                }
            }

            let samples = bytemuck::allocation::try_cast_vec::<u16, F::Sample>(pixels)
                .map_err(|(err, _)| ViprsError::Codec(format!("jp2k: cast error: {err:?}")))?;
            Image::from_buffer(plan.width, plan.height, plan.bands, samples)
                .map(|image| image.with_metadata(metadata))
                .map(Some)
                .map_err(|err| ViprsError::Codec(err.to_string()))
        }
        _ => Ok(None),
    }
}

fn decode_tiled_bytes_fast<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<Option<Image<F>>, ViprsError> {
    let codec_format = codec_format_from_bytes(src)?;
    let data_ptr = Box::into_raw(Box::new(WrappedSlice::new(src))).cast::<c_void>();

    // SAFETY: OpenJPEG returns an owned input stream pointer or null on failure.
    let stream_ptr = unsafe { sys::opj_stream_default_create(1) };
    if stream_ptr.is_null() {
        // SAFETY: stream creation failed, so ownership of data_ptr stays with Rust.
        unsafe { drop(Box::from_raw(data_ptr.cast::<WrappedSlice<'static>>())) };
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG input stream".into(),
        ));
    }

    // SAFETY: stream_ptr is valid and takes ownership of `data_ptr` until destroyed.
    unsafe {
        sys::opj_stream_set_read_function(stream_ptr, Some(buf_read_stream_read_fn));
        sys::opj_stream_set_skip_function(stream_ptr, Some(buf_read_stream_skip_fn));
        sys::opj_stream_set_seek_function(stream_ptr, Some(buf_read_stream_seek_fn));
        sys::opj_stream_set_user_data_length(stream_ptr, src.len() as u64);
        sys::opj_stream_set_user_data(stream_ptr, data_ptr, Some(buf_read_stream_free_fn));
    }

    decode_tiled_stream_fast(codec_format, stream_ptr, opts)
}

fn decode_tiled_path_fast<F: BandFormat>(
    path: &Path,
    opts: &LoadOptions,
) -> Result<Option<Image<F>>, ViprsError> {
    let codec_format = codec_format_from_path(path)?;
    let c_path = CString::new(path.to_string_lossy().as_bytes())
        .map_err(|err| ViprsError::Codec(format!("jp2k: invalid input path: {err}")))?;
    // SAFETY: c_path lives until stream creation returns; OpenJPEG copies the filename immediately.
    let stream_ptr = unsafe { sys::opj_stream_create_default_file_stream(c_path.as_ptr(), 1) };
    if stream_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG file stream".into(),
        ));
    }

    decode_tiled_stream_fast(codec_format, stream_ptr, opts)
}

const fn map_interpretation(color_space: J2kColorSpace, bands: u32, wide: bool) -> Interpretation {
    match color_space {
        J2kColorSpace::Gray => {
            if wide {
                Interpretation::Grey16
            } else {
                Interpretation::BW
            }
        }
        J2kColorSpace::CMYK => Interpretation::Cmyk,
        _ => {
            if bands == 1 {
                if wide {
                    Interpretation::Grey16
                } else {
                    Interpretation::BW
                }
            } else if wide {
                Interpretation::Rgb16
            } else {
                Interpretation::Srgb
            }
        }
    }
}

fn decoded_metadata(image: &OpenJpegDecodedImage, bands: u32, wide: bool) -> ImageMetadata {
    let mut extra = std::collections::HashMap::new();
    let bits = image
        .components()
        .iter()
        .map(|component| component.prec)
        .max()
        .unwrap_or(if wide { 16 } else { 8 });
    extra.insert("jp2k.bits-per-sample".into(), bits.to_string());
    extra.insert("jp2k.components".into(), image.num_components().to_string());

    ImageMetadata {
        interpretation: Some(map_interpretation(image.color_space(), bands, wide)),
        icc_profile: None,
        exif: None,
        xmp: None,
        n_pages: Some(image.resolution_count),
        extra,
        ..ImageMetadata::default()
    }
}

struct DecodedComponent<'a> {
    data: &'a [i32],
    precision: u32,
    signed: bool,
}

struct FastTiledDecodePlan {
    width: u32,
    height: u32,
    bands: u32,
    precision: u32,
    resolution_count: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
}

const PARALLEL_PACK_MIN_PIXELS: usize = 512 * 512;
const FAST_TILED_MIN_PIXELS: usize = 4096 * 4096;

#[inline]
const fn raw_components(image: &sys::opj_image_t) -> &[sys::opj_image_comp_t] {
    if image.comps.is_null() || image.numcomps == 0 {
        &[]
    } else {
        // SAFETY: OpenJPEG owns `comps` for the lifetime of `image`, and `numcomps`
        // gives the valid element count.
        unsafe { std::slice::from_raw_parts(image.comps, image.numcomps as usize) }
    }
}

#[inline]
const fn raw_component_data(component: &sys::opj_image_comp_t) -> &[i32] {
    let len = (component.w as usize).saturating_mul(component.h as usize);
    if component.data.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: OpenJPEG allocates `component.data` for `w * h` decoded samples.
        unsafe { std::slice::from_raw_parts(component.data, len) }
    }
}

#[inline]
fn raw_resolution_count(info: &sys::opj_codestream_info_v2_t) -> u32 {
    if info.m_default_tile_info.tccp_info.is_null() {
        1
    } else {
        // SAFETY: `tccp_info` belongs to `info` and is valid while `info` lives.
        unsafe { (*info.m_default_tile_info.tccp_info).numresolutions }.max(1)
    }
}

const fn raw_map_interpretation(
    color_space: sys::OPJ_COLOR_SPACE,
    bands: u32,
    wide: bool,
) -> Interpretation {
    match color_space {
        sys::COLOR_SPACE::OPJ_CLRSPC_GRAY => {
            if wide {
                Interpretation::Grey16
            } else {
                Interpretation::BW
            }
        }
        _ => {
            if bands == 1 {
                if wide {
                    Interpretation::Grey16
                } else {
                    Interpretation::BW
                }
            } else if wide {
                Interpretation::Rgb16
            } else {
                Interpretation::Srgb
            }
        }
    }
}

fn raw_decoded_metadata(
    image: &sys::opj_image_t,
    bands: u32,
    resolution_count: u32,
) -> ImageMetadata {
    let components = raw_components(image);
    let max_precision = components
        .iter()
        .map(|component| component.prec)
        .max()
        .unwrap_or(8);
    let wide = max_precision > 8;
    let mut extra = std::collections::HashMap::new();
    extra.insert("jp2k.bits-per-sample".into(), max_precision.to_string());
    extra.insert("jp2k.components".into(), image.numcomps.to_string());

    ImageMetadata {
        interpretation: Some(raw_map_interpretation(image.color_space, bands, wide)),
        icc_profile: None,
        exif: None,
        xmp: None,
        n_pages: Some(resolution_count),
        extra,
        ..ImageMetadata::default()
    }
}

fn fast_tiled_decode_plan<F: BandFormat>(
    image: &sys::opj_image_t,
    info: &sys::opj_codestream_info_v2_t,
    opts: &LoadOptions,
) -> Result<Option<FastTiledDecodePlan>, ViprsError> {
    if opts.page.unwrap_or(0) != 0 || (info.tw == 1 && info.th == 1) {
        return Ok(None);
    }

    match image.color_space {
        sys::COLOR_SPACE::OPJ_CLRSPC_UNKNOWN
        | sys::COLOR_SPACE::OPJ_CLRSPC_UNSPECIFIED
        | sys::COLOR_SPACE::OPJ_CLRSPC_GRAY
        | sys::COLOR_SPACE::OPJ_CLRSPC_SRGB => {}
        _ => return Ok(None),
    }

    if image.x0 != 0 || image.y0 != 0 {
        return Ok(None);
    }

    let components = raw_components(image);
    let first = components
        .first()
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image has no components".into()))?;
    let pixel_count = (first.w as usize)
        .checked_mul(first.h as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image dimensions overflow".into()))?;
    if pixel_count < FAST_TILED_MIN_PIXELS {
        return Ok(None);
    }
    let has_alpha = components.iter().any(|component| component.alpha == 1);
    let bands = match (components, has_alpha) {
        ([_], _) => 1,
        ([_, _], true) => 2,
        ([_, _, _], false) => 3,
        ([_, _, _, _], _) => 4,
        _ => return Ok(None),
    };

    let precision = match F::ID {
        BandFormatId::U8 => 8,
        BandFormatId::U16 => 16,
        _ => return Ok(None),
    };

    if components.iter().any(|component| {
        component.dx != 1
            || component.dy != 1
            || component.x0 != 0
            || component.y0 != 0
            || component.sgnd != 0
            || component.prec != precision
            || component.w != first.w
            || component.h != first.h
    }) {
        return Ok(None);
    }

    Ok(Some(FastTiledDecodePlan {
        width: first.w,
        height: first.h,
        bands,
        precision,
        resolution_count: raw_resolution_count(info),
        tile_width: info.tdx,
        tile_height: info.tdy,
        tiles_across: info.tw,
        tiles_down: info.th,
    }))
}

fn copy_fast_tiled_u8<const BANDS: usize>(
    pixels: &mut [u8],
    image: &sys::opj_image_t,
    plan: &FastTiledDecodePlan,
    tile_left: u32,
    tile_top: u32,
) -> Result<(), ViprsError> {
    let components = raw_components(image);
    let first = components
        .first()
        .ok_or_else(|| ViprsError::Codec("jp2k decode: tiled image has no components".into()))?;
    let tile_width = first.w.min(plan.width.saturating_sub(tile_left)) as usize;
    let tile_height = first.h.min(plan.height.saturating_sub(tile_top)) as usize;
    let tile_stride = first.w as usize;
    let width = plan.width as usize;

    let component_rows: [&[i32]; BANDS] =
        std::array::from_fn(|band| raw_component_data(&components[band]));
    if component_rows
        .iter()
        .any(|component| component.len() < tile_stride.saturating_mul(tile_height))
    {
        return Err(ViprsError::Codec(
            "jp2k decode: tiled component buffer shorter than expected".into(),
        ));
    }

    for row in 0..tile_height {
        let dest_row = ((tile_top as usize + row) * width + tile_left as usize) * BANDS;
        let dest = &mut pixels[dest_row..dest_row + tile_width * BANDS];
        let src_row = row * tile_stride;
        for x in 0..tile_width {
            let dst = x * BANDS;
            for band in 0..BANDS {
                dest[dst + band] = component_rows[band][src_row + x] as u8;
            }
        }
    }

    Ok(())
}

fn copy_fast_tiled_u16<const BANDS: usize>(
    pixels: &mut [u16],
    image: &sys::opj_image_t,
    plan: &FastTiledDecodePlan,
    tile_left: u32,
    tile_top: u32,
) -> Result<(), ViprsError> {
    let components = raw_components(image);
    let first = components
        .first()
        .ok_or_else(|| ViprsError::Codec("jp2k decode: tiled image has no components".into()))?;
    let tile_width = first.w.min(plan.width.saturating_sub(tile_left)) as usize;
    let tile_height = first.h.min(plan.height.saturating_sub(tile_top)) as usize;
    let tile_stride = first.w as usize;
    let width = plan.width as usize;

    let component_rows: [&[i32]; BANDS] =
        std::array::from_fn(|band| raw_component_data(&components[band]));
    if component_rows
        .iter()
        .any(|component| component.len() < tile_stride.saturating_mul(tile_height))
    {
        return Err(ViprsError::Codec(
            "jp2k decode: tiled component buffer shorter than expected".into(),
        ));
    }

    for row in 0..tile_height {
        let dest_row = ((tile_top as usize + row) * width + tile_left as usize) * BANDS;
        let dest = &mut pixels[dest_row..dest_row + tile_width * BANDS];
        let src_row = row * tile_stride;
        for x in 0..tile_width {
            let dst = x * BANDS;
            for band in 0..BANDS {
                dest[dst + band] = component_rows[band][src_row + x] as u16;
            }
        }
    }

    Ok(())
}

#[inline]
const fn decoded_component(component: &sys::opj_image_comp_t) -> DecodedComponent<'_> {
    DecodedComponent {
        data: raw_component_data(component),
        precision: component.prec,
        signed: component.sgnd == 1,
    }
}

fn validate_decode_layout(image: &OpenJpegDecodedImage) -> Result<(u32, u32, u32), ViprsError> {
    match image.color_space() {
        J2kColorSpace::Unknown
        | J2kColorSpace::Unspecified
        | J2kColorSpace::Gray
        | J2kColorSpace::SRGB => {}
        color_space => {
            return Err(ViprsError::Codec(format!(
                "jp2k decode: unsupported color space {color_space:?}"
            )));
        }
    }

    let components = image.components();
    let first = components
        .first()
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image has no components".into()))?;
    if components.iter().any(|component| component.data.is_null()) {
        return Err(ViprsError::Codec(
            "jp2k decode: decoded component payload is missing".into(),
        ));
    }
    let width = first.w;
    let height = first.h;
    let has_alpha = components.iter().any(|component| component.alpha == 1);
    let bands = match (components, has_alpha) {
        ([_], _) => 1,
        ([_, _], true) => 2,
        ([_, _, _], false) => 3,
        ([_, _, _, _], _) => 4,
        _ => {
            return Err(ViprsError::Codec(format!(
                "jp2k decode: unsupported component layout ({})",
                image.num_components()
            )));
        }
    };

    let max_precision = components
        .iter()
        .map(|component| component.prec)
        .max()
        .unwrap_or(0);
    if !(1..=16).contains(&max_precision) {
        return Err(ViprsError::Codec(format!(
            "jp2k decode: unsupported precision {max_precision}"
        )));
    }

    Ok((width, height, bands))
}

#[inline]
fn scale_to_u8(component: &DecodedComponent<'_>, value: i32) -> u8 {
    if component.signed {
        let old_max = 1_i64 << (component.precision - 1);
        ((((i64::from(value)) * 128) / old_max) + 128) as u8
    } else if component.precision == 8 {
        value as u8
    } else {
        let old_max = (1_u64 << component.precision) - 1;
        (((value as u64) * u64::from(u8::MAX)) / old_max) as u8
    }
}

#[inline]
fn scale_to_u16(component: &DecodedComponent<'_>, value: i32) -> u16 {
    if component.signed {
        let old_max = 1_i64 << (component.precision - 1);
        ((((i64::from(value)) * 32_768) / old_max) + 32_768) as u16
    } else if component.precision == 16 {
        value as u16
    } else {
        let old_max = (1_u64 << component.precision) - 1;
        (((value as u64) * u64::from(u16::MAX)) / old_max) as u16
    }
}

#[inline]
fn decode_sample_u8(component: &DecodedComponent<'_>, wide_precision: bool, idx: usize) -> u8 {
    let value = component.data[idx];
    if wide_precision {
        (scale_to_u16(component, value) >> 8) as u8
    } else {
        scale_to_u8(component, value)
    }
}

#[inline]
fn decode_sample_u16(component: &DecodedComponent<'_>, wide_precision: bool, idx: usize) -> u16 {
    let value = component.data[idx];
    if wide_precision {
        scale_to_u16(component, value)
    } else {
        u16::from(scale_to_u8(component, value)) * 257
    }
}

fn fill_unsigned_u8_rows<const BANDS: usize>(
    pixels: &mut [u8],
    width: usize,
    components: [&DecodedComponent<'_>; BANDS],
) {
    let row_stride = width * BANDS;
    fill_rows(
        pixels,
        row_stride,
        components[0].data.len() >= PARALLEL_PACK_MIN_PIXELS,
        |row_idx, row| {
            let start = row_idx * width;
            let end = start + width;
            let component_rows = components.map(|component| &component.data[start..end]);
            for (x, pixel) in row.chunks_exact_mut(BANDS).take(width).enumerate() {
                for band in 0..BANDS {
                    pixel[band] = component_rows[band][x] as u8;
                }
            }
        },
    );
}

fn fill_unsigned_u16_rows<const BANDS: usize>(
    pixels: &mut [u16],
    width: usize,
    components: [&DecodedComponent<'_>; BANDS],
) {
    let row_stride = width * BANDS;
    fill_rows(
        pixels,
        row_stride,
        components[0].data.len() >= PARALLEL_PACK_MIN_PIXELS,
        |row_idx, row| {
            let start = row_idx * width;
            let end = start + width;
            let component_rows = components.map(|component| &component.data[start..end]);
            for (x, pixel) in row.chunks_exact_mut(BANDS).take(width).enumerate() {
                for band in 0..BANDS {
                    pixel[band] = component_rows[band][x] as u16;
                }
            }
        },
    );
}

fn fill_rows<T, F>(pixels: &mut [T], row_stride: usize, parallel: bool, fill_row: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    #[cfg(feature = "rayon")]
    if parallel {
        pixels
            .par_chunks_exact_mut(row_stride)
            .enumerate()
            .for_each(|(row_idx, row)| fill_row(row_idx, row));
        return;
    }

    for (row_idx, row) in pixels.chunks_exact_mut(row_stride).enumerate() {
        fill_row(row_idx, row);
    }
}

fn decode_pixels_u8(image: &OpenJpegDecodedImage) -> Result<(u32, u32, u32, Vec<u8>), ViprsError> {
    let (width, height, bands) = validate_decode_layout(image)?;
    let components = image.components();
    let max_precision = components
        .iter()
        .map(|component| component.prec)
        .max()
        .unwrap_or(0);
    let wide_precision = max_precision > 8;
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image dimensions overflow".into()))?;
    let output_len = pixel_count
        .checked_mul(bands as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: output buffer overflow".into()))?;
    let mut pixels = vec![0_u8; output_len];

    if max_precision == 8 && components.iter().all(|component| component.sgnd == 0) {
        match components {
            [r] => {
                let r = decoded_component(r);
                fill_unsigned_u8_rows::<1>(&mut pixels, width as usize, [&r]);
                return Ok((width, height, bands, pixels));
            }
            [r, a] if a.alpha == 1 => {
                let r = decoded_component(r);
                let a = decoded_component(a);
                fill_unsigned_u8_rows::<2>(&mut pixels, width as usize, [&r, &a]);
                return Ok((width, height, bands, pixels));
            }
            [r, g, b] if !components.iter().any(|component| component.alpha == 1) => {
                let r = decoded_component(r);
                let g = decoded_component(g);
                let b = decoded_component(b);
                fill_unsigned_u8_rows::<3>(&mut pixels, width as usize, [&r, &g, &b]);
                return Ok((width, height, bands, pixels));
            }
            [r, g, b, a] => {
                let r = decoded_component(r);
                let g = decoded_component(g);
                let b = decoded_component(b);
                let a = decoded_component(a);
                fill_unsigned_u8_rows::<4>(&mut pixels, width as usize, [&r, &g, &b, &a]);
                return Ok((width, height, bands, pixels));
            }
            _ => {}
        }
    }

    match components {
        [r] => {
            let r = decoded_component(r);
            for (idx, pixel) in pixels.iter_mut().enumerate() {
                *pixel = decode_sample_u8(&r, wide_precision, idx);
            }
        }
        [r, a] if a.alpha == 1 => {
            let r = decoded_component(r);
            let a = decoded_component(a);
            for idx in 0..pixel_count {
                let offset = idx * 2;
                pixels[offset] = decode_sample_u8(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u8(&a, wide_precision, idx);
            }
        }
        [r, g, b] if !components.iter().any(|component| component.alpha == 1) => {
            let r = decoded_component(r);
            let g = decoded_component(g);
            let b = decoded_component(b);
            for idx in 0..pixel_count {
                let offset = idx * 3;
                pixels[offset] = decode_sample_u8(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u8(&g, wide_precision, idx);
                pixels[offset + 2] = decode_sample_u8(&b, wide_precision, idx);
            }
        }
        [r, g, b, a] => {
            let r = decoded_component(r);
            let g = decoded_component(g);
            let b = decoded_component(b);
            let a = decoded_component(a);
            for idx in 0..pixel_count {
                let offset = idx * 4;
                pixels[offset] = decode_sample_u8(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u8(&g, wide_precision, idx);
                pixels[offset + 2] = decode_sample_u8(&b, wide_precision, idx);
                pixels[offset + 3] = decode_sample_u8(&a, wide_precision, idx);
            }
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "jp2k decode: unsupported component layout ({})",
                image.num_components()
            )));
        }
    }

    Ok((width, height, bands, pixels))
}

fn decode_pixels_u16(
    image: &OpenJpegDecodedImage,
) -> Result<(u32, u32, u32, Vec<u16>), ViprsError> {
    let (width, height, bands) = validate_decode_layout(image)?;
    let components = image.components();
    let max_precision = components
        .iter()
        .map(|component| component.prec)
        .max()
        .unwrap_or(0);
    let wide_precision = max_precision > 8;
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: image dimensions overflow".into()))?;
    let output_len = pixel_count
        .checked_mul(bands as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k decode: output buffer overflow".into()))?;
    let mut pixels = vec![0_u16; output_len];

    if max_precision == 16 && components.iter().all(|component| component.sgnd == 0) {
        match components {
            [r] => {
                let r = decoded_component(r);
                fill_unsigned_u16_rows::<1>(&mut pixels, width as usize, [&r]);
                return Ok((width, height, bands, pixels));
            }
            [r, a] if a.alpha == 1 => {
                let r = decoded_component(r);
                let a = decoded_component(a);
                fill_unsigned_u16_rows::<2>(&mut pixels, width as usize, [&r, &a]);
                return Ok((width, height, bands, pixels));
            }
            [r, g, b] if !components.iter().any(|component| component.alpha == 1) => {
                let r = decoded_component(r);
                let g = decoded_component(g);
                let b = decoded_component(b);
                fill_unsigned_u16_rows::<3>(&mut pixels, width as usize, [&r, &g, &b]);
                return Ok((width, height, bands, pixels));
            }
            [r, g, b, a] => {
                let r = decoded_component(r);
                let g = decoded_component(g);
                let b = decoded_component(b);
                let a = decoded_component(a);
                fill_unsigned_u16_rows::<4>(&mut pixels, width as usize, [&r, &g, &b, &a]);
                return Ok((width, height, bands, pixels));
            }
            _ => {}
        }
    }

    match components {
        [r] => {
            let r = decoded_component(r);
            for (idx, pixel) in pixels.iter_mut().enumerate() {
                *pixel = decode_sample_u16(&r, wide_precision, idx);
            }
        }
        [r, a] if a.alpha == 1 => {
            let r = decoded_component(r);
            let a = decoded_component(a);
            for idx in 0..pixel_count {
                let offset = idx * 2;
                pixels[offset] = decode_sample_u16(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u16(&a, wide_precision, idx);
            }
        }
        [r, g, b] if !components.iter().any(|component| component.alpha == 1) => {
            let r = decoded_component(r);
            let g = decoded_component(g);
            let b = decoded_component(b);
            for idx in 0..pixel_count {
                let offset = idx * 3;
                pixels[offset] = decode_sample_u16(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u16(&g, wide_precision, idx);
                pixels[offset + 2] = decode_sample_u16(&b, wide_precision, idx);
            }
        }
        [r, g, b, a] => {
            let r = decoded_component(r);
            let g = decoded_component(g);
            let b = decoded_component(b);
            let a = decoded_component(a);
            for idx in 0..pixel_count {
                let offset = idx * 4;
                pixels[offset] = decode_sample_u16(&r, wide_precision, idx);
                pixels[offset + 1] = decode_sample_u16(&g, wide_precision, idx);
                pixels[offset + 2] = decode_sample_u16(&b, wide_precision, idx);
                pixels[offset + 3] = decode_sample_u16(&a, wide_precision, idx);
            }
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "jp2k decode: unsupported component layout ({})",
                image.num_components()
            )));
        }
    }

    Ok((width, height, bands, pixels))
}

#[inline]
fn is_jp2k_extension(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jp2" | "j2k" | "jpf" | "jpx"
            )
        })
}

const fn opj_color_space_for_bands(bands: u32) -> sys::OPJ_COLOR_SPACE {
    match bands {
        1 => sys::COLOR_SPACE::OPJ_CLRSPC_GRAY,
        4 => sys::COLOR_SPACE::OPJ_CLRSPC_CMYK,
        _ => sys::COLOR_SPACE::OPJ_CLRSPC_SRGB,
    }
}

unsafe extern "C" fn opj_error_callback(msg: *const c_char, _data: *mut c_void) {
    if msg.is_null() {
        return;
    }
    // SAFETY: OpenJPEG guarantees msg is a NUL-terminated C string for callback lifetime.
    let message = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    eprintln!("jp2k: OpenJPEG error: {message}");
}

struct EncodedJp2Buffer {
    offset: usize,
    bytes: Vec<u8>,
}

impl EncodedJp2Buffer {
    fn new(capacity: usize) -> Self {
        Self {
            offset: 0,
            bytes: Vec::with_capacity(capacity),
        }
    }

    fn reserve_for(&mut self, len: usize) {
        if len <= self.bytes.capacity() {
            return;
        }

        let target = len.checked_next_power_of_two().unwrap_or(len);
        let additional = target.saturating_sub(self.bytes.len());
        self.bytes.reserve(additional);
    }

    fn ensure_len(&mut self, len: usize) {
        if self.bytes.len() < len {
            self.bytes.resize(len, 0);
        }
    }

    fn write_from(&mut self, input: &[u8]) -> usize {
        let end = self.offset.saturating_add(input.len());
        self.reserve_for(end);
        if self.bytes.len() < end {
            // SAFETY: reserve_for ensures capacity for `end` bytes, and the range
            // [self.offset, end) is fully initialized by copy_from_slice below.
            unsafe { self.bytes.set_len(end) };
        }
        self.bytes[self.offset..end].copy_from_slice(input);
        self.offset = end;
        input.len()
    }

    fn skip_by(&mut self, delta: i64) -> Option<i64> {
        let new_offset = if delta >= 0 {
            self.offset.checked_add(delta as usize)?
        } else {
            self.offset.checked_sub(delta.unsigned_abs() as usize)?
        };
        self.ensure_len(new_offset);
        self.offset = new_offset;
        Some(delta)
    }

    fn seek_to(&mut self, offset: i64) -> bool {
        let Ok(new_offset) = usize::try_from(offset) else {
            return false;
        };
        self.ensure_len(new_offset);
        self.offset = new_offset;
        true
    }
}

unsafe extern "C" fn encoded_jp2_buffer_free_fn(p_data: *mut c_void) {
    if p_data.is_null() {
        return;
    }

    // SAFETY: OpenJPEG passes back the Box allocation registered in encode_to_jp2_bytes.
    unsafe { drop(Box::from_raw(p_data.cast::<EncodedJp2Buffer>())) };
}

unsafe extern "C" fn encoded_jp2_buffer_write_fn(
    p_buffer: *mut c_void,
    nb_bytes: usize,
    p_data: *mut c_void,
) -> usize {
    if p_buffer.is_null() || nb_bytes == 0 {
        return 0;
    }

    // SAFETY: OpenJPEG provides the same EncodedJp2Buffer pointer registered as user data.
    let buffer = unsafe { &mut *p_data.cast::<EncodedJp2Buffer>() };
    // SAFETY: OpenJPEG provides a readable buffer of nb_bytes bytes for the callback lifetime.
    let input = unsafe { std::slice::from_raw_parts(p_buffer.cast::<u8>(), nb_bytes) };
    buffer.write_from(input)
}

unsafe extern "C" fn encoded_jp2_buffer_skip_fn(nb_bytes: i64, p_data: *mut c_void) -> i64 {
    // SAFETY: OpenJPEG provides the same EncodedJp2Buffer pointer registered as user data.
    let buffer = unsafe { &mut *p_data.cast::<EncodedJp2Buffer>() };
    buffer.skip_by(nb_bytes).unwrap_or(-1)
}

unsafe extern "C" fn encoded_jp2_buffer_seek_fn(nb_bytes: i64, p_data: *mut c_void) -> i32 {
    // SAFETY: OpenJPEG provides the same EncodedJp2Buffer pointer registered as user data.
    let buffer = unsafe { &mut *p_data.cast::<EncodedJp2Buffer>() };
    i32::from(buffer.seek_to(nb_bytes))
}

#[inline]
fn default_tile_dimension(explicit: Option<u32>, image_dim: u32) -> i32 {
    explicit
        .unwrap_or_else(|| {
            if (MEDIUM_IMAGE_TILE_MIN_DIM..=MEDIUM_IMAGE_TILE_MAX_DIM).contains(&image_dim) {
                MEDIUM_IMAGE_TILE_SIZE
            } else {
                DEFAULT_TILE_SIZE
            }
        })
        .max(1)
        .min(image_dim.max(1)) as i32
}

#[inline]
fn estimated_encoded_capacity(
    source_bytes: usize,
    tile_capacity: usize,
    lossless: bool,
    quality: Option<u8>,
) -> usize {
    let compression_divisor = if lossless {
        2
    } else if quality.unwrap_or(DEFAULT_QUALITY) >= 90 {
        3
    } else {
        4
    };
    let estimated = source_bytes / compression_divisor;
    let max_tile_multiple = tile_capacity.saturating_mul(16);
    estimated.clamp(tile_capacity.max(4096), max_tile_multiple.max(4096))
}

fn apply_lossy_profile(encoder_params: &mut sys::opj_cparameters_t, quality: u8) {
    encoder_params.irreversible = 1;
    encoder_params.prog_order = sys::PROG_ORDER::OPJ_RPCL;
    encoder_params.cblockw_init = DEFAULT_CODEBLOCK_SIZE;
    encoder_params.cblockh_init = DEFAULT_CODEBLOCK_SIZE;
    encoder_params.cp_disto_alloc = 1;
    encoder_params.cp_fixed_quality = 1;
    encoder_params.tcp_numlayers = 1;
    encoder_params.numresolution = PROFILE_RES_SPEC as i32;
    encoder_params.csty = 1;
    encoder_params.res_spec = PROFILE_RES_SPEC as i32;

    for idx in 0..PROFILE_RES_SPEC {
        encoder_params.prcw_init[idx] = DEFAULT_PRECINCT_SIZE;
        encoder_params.prch_init[idx] = DEFAULT_PRECINCT_SIZE;
        encoder_params.tcp_distoratio[idx] = f32::from(quality) + (10 * idx) as f32;
    }
}

#[inline]
fn libvips_num_resolutions(width: u32, height: u32) -> i32 {
    let min_dim = width.min(height).max(1);
    let log2_min = (u32::BITS - 1 - min_dim.leading_zeros()) as i32;
    (log2_min - 5).max(1)
}

#[inline]
fn openjpeg_thread_count() -> i32 {
    std::thread::available_parallelism()
        .map_or(1, usize::from)
        .max(1) as i32
}

fn pack_tile_component_data<F: BandFormat>(
    image: &Image<F>,
    bands_usize: usize,
    tile_left: u32,
    tile_top: u32,
    tile_width: u32,
    tile_height: u32,
    output: &mut Vec<u8>,
) -> Result<(), ViprsError> {
    let sample_size = std::mem::size_of::<F::Sample>();
    let tile_pixels = tile_width as usize * tile_height as usize;
    let image_width = image.width() as usize;
    let total_bytes = tile_pixels
        .checked_mul(bands_usize)
        .and_then(|samples| samples.checked_mul(sample_size))
        .unwrap_or(0);
    output.clear();
    output.resize(total_bytes, 0);

    match F::ID {
        BandFormatId::U8 => {
            let pixels_ptr = image.pixels().as_ptr();
            for component_idx in 0..bands_usize {
                let component_bytes =
                    &mut output[component_idx * tile_pixels..(component_idx + 1) * tile_pixels];
                for local_y in 0..tile_height as usize {
                    let src_row = (tile_top as usize + local_y) * image_width;
                    let dst_row = local_y * tile_width as usize;
                    for local_x in 0..tile_width as usize {
                        let pixel_idx = src_row + tile_left as usize + local_x;
                        let sample_idx = pixel_idx * bands_usize + component_idx;
                        let sample = {
                            // SAFETY: `pixels_ptr` comes from `image.pixels()`, and `sample_idx`
                            // stays within `expected_samples` because `tile_left/top`, `local_x/y`,
                            // and `component_idx` are bounded by the current tile and band count.
                            unsafe { *pixels_ptr.add(sample_idx).cast::<u8>() }
                        };
                        component_bytes[dst_row + local_x] = sample;
                    }
                }
            }
        }
        BandFormatId::U16 => {
            let pixels_ptr = image.pixels().as_ptr();
            for component_idx in 0..bands_usize {
                let start = component_idx * tile_pixels * sample_size;
                let end = start + tile_pixels * sample_size;
                let component_bytes = &mut output[start..end];
                for local_y in 0..tile_height as usize {
                    let src_row = (tile_top as usize + local_y) * image_width;
                    let dst_row = local_y * tile_width as usize;
                    for local_x in 0..tile_width as usize {
                        let pixel_idx = src_row + tile_left as usize + local_x;
                        let sample_idx = pixel_idx * bands_usize + component_idx;
                        let dst_offset = (dst_row + local_x) * sample_size;
                        let sample = {
                            // SAFETY: `pixels_ptr` comes from `image.pixels()`, and `sample_idx`
                            // stays within `expected_samples` because `tile_left/top`, `local_x/y`,
                            // and `component_idx` are bounded by the current tile and band count.
                            unsafe { *pixels_ptr.add(sample_idx).cast::<u16>() }
                        }
                        .to_ne_bytes();
                        component_bytes[dst_offset..dst_offset + sample_size]
                            .copy_from_slice(&sample);
                    }
                }
            }
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "jp2k encode: unsupported format {:?}",
                F::ID
            )));
        }
    }
    Ok(())
}

fn encode_to_jp2_bytes<F: BandFormat>(
    image: &Image<F>,
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let bands = image.bands();
    if bands == 0 || bands as usize > MAX_COMPONENTS {
        return Err(ViprsError::Codec(format!(
            "jp2k: unsupported band count {bands}; expected 1..={MAX_COMPONENTS}"
        )));
    }

    let width = image.width();
    let height = image.height();
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ViprsError::Codec("jp2k: image dimensions overflow".into()))?;
    let bands_usize = bands as usize;
    let expected_samples = pixel_count
        .checked_mul(bands_usize)
        .ok_or_else(|| ViprsError::Codec("jp2k: sample count overflow".into()))?;
    if image.pixels().len() != expected_samples {
        return Err(ViprsError::Codec(format!(
            "jp2k: pixel buffer length mismatch, got {}, expected {}",
            image.pixels().len(),
            expected_samples
        )));
    }

    let mut component_params = vec![
        sys::opj_image_cmptparm_t {
            dx: 1,
            dy: 1,
            w: width,
            h: height,
            x0: 0,
            y0: 0,
            prec: if F::ID == BandFormatId::U16 { 16 } else { 8 },
            bpp: if F::ID == BandFormatId::U16 { 16 } else { 8 },
            sgnd: 0,
        };
        bands_usize
    ];

    let mut encoder_params = {
        let mut params = std::mem::MaybeUninit::<sys::opj_cparameters_t>::uninit();
        // SAFETY: OpenJPEG fills all fields in the passed output struct.
        unsafe {
            sys::opj_set_default_encoder_parameters(params.as_mut_ptr());
            params.assume_init()
        }
    };

    let lossless = opts.lossless == Some(true);
    encoder_params.tcp_numlayers = 1;
    if lossless {
        encoder_params.tcp_rates[0] = 0.0;
        encoder_params.irreversible = 0;
    } else {
        let quality = opts.quality.unwrap_or(DEFAULT_QUALITY).clamp(1, 100);
        apply_lossy_profile(&mut encoder_params, quality);
    }

    let tile_width = default_tile_dimension(opts.tile_width, width);
    let tile_height = default_tile_dimension(opts.tile_height, height);
    encoder_params.tile_size_on = 1;
    encoder_params.cp_tdx = tile_width;
    encoder_params.cp_tdy = tile_height;
    // c_char is i8 on x86/macOS but u8 on ARM Linux — cast via i32 for portability.
    encoder_params.tcp_mct = if bands >= 3 {
        1 as std::os::raw::c_char
    } else {
        0 as std::os::raw::c_char
    };
    encoder_params.numresolution = libvips_num_resolutions(width, height);

    // SAFETY: component_params points to initialized component descriptors for this call.
    let image_ptr = unsafe {
        sys::opj_image_tile_create(
            bands,
            component_params.as_mut_ptr(),
            opj_color_space_for_bands(bands),
        )
    };
    if image_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to allocate OpenJPEG image".into(),
        ));
    }

    let image_guard = OpenJpegImage(image_ptr);

    // SAFETY: image_ptr points to valid OpenJPEG image returned above.
    unsafe {
        (*image_guard.0).x0 = 0;
        (*image_guard.0).y0 = 0;
        (*image_guard.0).x1 = width;
        (*image_guard.0).y1 = height;
    }

    // SAFETY: codec is created by OpenJPEG and released by OpenJpegCodec drop.
    let codec_ptr = unsafe { sys::opj_create_compress(sys::CODEC_FORMAT::OPJ_CODEC_JP2) };
    if codec_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create OpenJPEG compressor".into(),
        ));
    }

    let codec_guard = OpenJpegCodec(codec_ptr);

    // SAFETY: codec pointer is valid for the lifetime of this encode call.
    unsafe {
        sys::opj_set_error_handler(
            codec_guard.0,
            Some(opj_error_callback),
            std::ptr::null_mut(),
        );
    }

    // SAFETY: codec/image/params pointers are valid and initialized.
    if unsafe { sys::opj_setup_encoder(codec_guard.0, &raw mut encoder_params, image_guard.0) } == 0
    {
        return Err(ViprsError::Codec("jp2k: opj_setup_encoder failed".into()));
    }

    let pixel_bytes = std::mem::size_of::<F::Sample>();
    let tile_width_u32 = tile_width as u32;
    let tile_height_u32 = tile_height as u32;
    let tile_capacity = (tile_width_u32 as usize)
        .checked_mul(tile_height_u32 as usize)
        .and_then(|pixels| pixels.checked_mul(bands_usize))
        .and_then(|samples| samples.checked_mul(pixel_bytes))
        .ok_or_else(|| ViprsError::Codec("jp2k: tile buffer size overflow".into()))?;
    let source_bytes = expected_samples
        .checked_mul(pixel_bytes)
        .ok_or_else(|| ViprsError::Codec("jp2k: source buffer size overflow".into()))?;

    // SAFETY: all pointers are valid and encoder setup has completed.
    if unsafe { sys::opj_codec_set_threads(codec_guard.0, openjpeg_thread_count()) } == 0 {
        return Err(ViprsError::Codec(
            "jp2k: failed to configure OpenJPEG encoder threads".into(),
        ));
    }

    // SAFETY: OpenJPEG returns an owned output stream pointer or null on failure.
    let stream_ptr = unsafe { sys::opj_stream_default_create(0) };
    if stream_ptr.is_null() {
        return Err(ViprsError::Codec(
            "jp2k: failed to create output stream".into(),
        ));
    }

    let stream_guard = OpenJpegStream(stream_ptr);
    let encoded_buffer_ptr = Box::into_raw(Box::new(EncodedJp2Buffer::new(
        estimated_encoded_capacity(source_bytes, tile_capacity, lossless, opts.quality),
    )))
    .cast::<c_void>();

    // SAFETY: stream_guard owns the stream and encoded_buffer_ptr remains valid until stream destruction.
    unsafe {
        sys::opj_stream_set_write_function(stream_guard.0, Some(encoded_jp2_buffer_write_fn));
        sys::opj_stream_set_skip_function(stream_guard.0, Some(encoded_jp2_buffer_skip_fn));
        sys::opj_stream_set_seek_function(stream_guard.0, Some(encoded_jp2_buffer_seek_fn));
        sys::opj_stream_set_user_data(
            stream_guard.0,
            encoded_buffer_ptr,
            Some(encoded_jp2_buffer_free_fn),
        );
    }

    // SAFETY: all pointers are valid and initialized.
    let started = unsafe { sys::opj_start_compress(codec_guard.0, image_guard.0, stream_guard.0) };
    if started == 0 {
        return Err(ViprsError::Codec("jp2k: opj_start_compress failed".into()));
    }

    let tiles_across = width.div_ceil(tile_width_u32);
    let tiles_down = height.div_ceil(tile_height_u32);
    let mut tile_buffer = Vec::with_capacity(tile_capacity);
    let mut encoded = 1;
    for tile_y in 0..tiles_down {
        let top = tile_y * tile_height_u32;
        let current_tile_height = (height - top).min(tile_height_u32);
        for tile_x in 0..tiles_across {
            let left = tile_x * tile_width_u32;
            let current_tile_width = (width - left).min(tile_width_u32);
            pack_tile_component_data(
                image,
                bands_usize,
                left,
                top,
                current_tile_width,
                current_tile_height,
                &mut tile_buffer,
            )?;
            let tile_index = tile_y * tiles_across + tile_x;
            // SAFETY: encoder has started, tile buffer is planar per OpenJPEG contract, and tiles are written sequentially.
            encoded = unsafe {
                sys::opj_write_tile(
                    codec_guard.0,
                    tile_index,
                    tile_buffer.as_mut_ptr(),
                    tile_buffer.len() as u32,
                    stream_guard.0,
                )
            };
            if encoded == 0 {
                break;
            }
        }
        if encoded == 0 {
            break;
        }
    }
    // SAFETY: balanced with opj_start_compress; required to finalize stream.
    let ended = unsafe { sys::opj_end_compress(codec_guard.0, stream_guard.0) };
    if encoded == 0 || ended == 0 {
        return Err(ViprsError::Codec("jp2k: encoding failed".into()));
    }

    // SAFETY: `encoded_buffer_ptr` is the Box allocation registered as stream user data and
    // has not been freed because ownership is transferred back only after this successful encode.
    let encoded_buffer_ptr = unsafe { &mut *encoded_buffer_ptr.cast::<EncodedJp2Buffer>() };
    Ok(std::mem::take(&mut encoded_buffer_ptr.bytes))
}

fn finish_decoded_image<F: BandFormat>(
    decoded: &OpenJpegDecodedImage,
) -> Result<Image<F>, ViprsError> {
    match F::ID {
        BandFormatId::U8 => {
            let (width, height, bands, samples) = decode_pixels_u8(decoded)?;
            let metadata = decoded_metadata(decoded, bands, false);
            let samples = bytemuck::allocation::try_cast_vec::<u8, F::Sample>(samples)
                .map_err(|(err, _)| ViprsError::Codec(format!("jp2k: cast error: {err:?}")))?;
            Image::from_buffer(width, height, bands, samples)
                .map(|image| image.with_metadata(metadata))
                .map_err(|err| ViprsError::Codec(err.to_string()))
        }
        BandFormatId::U16 => {
            let (width, height, bands, samples) = decode_pixels_u16(decoded)?;
            let metadata = decoded_metadata(decoded, bands, true);
            let samples = bytemuck::allocation::try_cast_vec::<u16, F::Sample>(samples)
                .map_err(|(err, _)| ViprsError::Codec(format!("jp2k: cast error: {err:?}")))?;
            Image::from_buffer(width, height, bands, samples)
                .map(|image| image.with_metadata(metadata))
                .map_err(|err| ViprsError::Codec(err.to_string()))
        }
        _ => Err(ViprsError::Codec(format!(
            "jp2k decode: unsupported target format {:?}",
            F::ID
        ))),
    }
}

impl ImageDecoder for Jp2kCodec {
    fn format_name(&self) -> &'static str {
        "jp2k"
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        is_jp2k_extension(path)
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.starts_with(JP2_RFC3745_MAGIC) || header.starts_with(J2K_CODESTREAM_MAGIC)
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
        validate_load_options(opts)?;

        if let Some(decoded) = decode_tiled_bytes_fast::<F>(src, opts)? {
            return Ok(decoded);
        }

        let decoded = decode_openjpeg_bytes(src, opts)?;
        finish_decoded_image(&decoded)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        require_supported_format::<F>()?;
        validate_load_options(opts)?;

        if let Some(decoded) = decode_tiled_path_fast::<F>(path, opts)? {
            return Ok(decoded);
        }

        let decoded = decode_openjpeg_path(path, opts)?;
        finish_decoded_image(&decoded)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        probe_openjpeg_bytes(src)
    }

    fn probe_path(&self, path: &Path) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        probe_openjpeg_path(path)
    }
}

impl ImageEncoder for Jp2kCodec {
    fn format_name(&self) -> &'static str {
        "jp2k"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        require_supported_format::<F>()?;
        encode_to_jp2_bytes(image, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::codec_options::SaveOptions;
    use viprs_core::format::{F32, U8, U16};

    fn psnr_u8(actual: &[u8], expected: &[u8]) -> f64 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "PSNR inputs must have same length"
        );
        if actual.is_empty() {
            return f64::INFINITY;
        }

        let mse = actual
            .iter()
            .zip(expected.iter())
            .map(|(&got, &want)| {
                let diff = f64::from(got) - f64::from(want);
                diff * diff
            })
            .sum::<f64>()
            / actual.len() as f64;

        if mse == 0.0 {
            f64::INFINITY
        } else {
            20.0 * 255.0_f64.log10() - 10.0 * mse.log10()
        }
    }

    #[test]
    fn sniff_recognises_jp2k_signatures() {
        let codec = Jp2kCodec;
        assert!(codec.sniff(&JP2_RFC3745_MAGIC));
        assert!(codec.sniff(&J2K_CODESTREAM_MAGIC));
        assert!(!codec.sniff(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn round_trip_lossless_u8_preserves_samples() {
        let codec = Jp2kCodec;
        let image = Image::<U8>::from_buffer(
            3,
            2,
            3,
            vec![
                0, 10, 20, 30, 40, 50, 60, 70, 80, // row 1
                90, 100, 110, 120, 130, 140, 150, 160, 170, // row 2
            ],
        )
        .unwrap();

        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default().lossless())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (3, 2, 3)
        );
        assert_eq!(decoded.metadata().n_pages, Some(1));
        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn round_trip_u16_preserves_samples() {
        let codec = Jp2kCodec;
        let image =
            Image::<U16>::from_buffer(2, 2, 1, vec![0u16, 1024u16, 4096u16, 65535u16]).unwrap();

        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default().lossless())
            .unwrap();
        let decoded = codec.decode::<U16>(&encoded).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (2, 2, 1)
        );
        assert_eq!(decoded.pixels(), image.pixels());
    }

    #[test]
    fn round_trip_lossy_u8_preserves_visual_quality() {
        let codec = Jp2kCodec;
        let width = 32;
        let height = 32;
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 7 + y * 3) % 256) as u8,
                        ((x * 5 + y * 11) % 256) as u8,
                        ((x * 13 + y * 9) % 256) as u8,
                    ]
                })
            })
            .collect();
        let image = Image::<U8>::from_buffer(width, height, 3, pixels).unwrap();

        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default().with_quality(100))
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(
            (decoded.width(), decoded.height(), decoded.bands()),
            (width, height, 3)
        );
        let psnr = psnr_u8(decoded.pixels(), image.pixels());
        assert!(
            psnr > 40.0,
            "JP2K lossy round-trip PSNR too low: {psnr:.2} dB"
        );
    }

    #[test]
    fn rejects_non_u8_u16_formats() {
        let codec = Jp2kCodec;
        let image = Image::<F32>::from_buffer(1, 1, 1, vec![1.0f32]).unwrap();
        let err = codec.encode(&image).unwrap_err();
        assert!(err.to_string().contains("only U8 and U16"));
    }

    #[test]
    fn narrow_precision_u16_path_matches_previous_u8_expansion() {
        let component = DecodedComponent {
            data: &[128],
            precision: 8,
            signed: false,
        };

        assert_eq!(decode_sample_u16(&component, false, 0), 32896);
    }

    #[test]
    fn wide_precision_u8_path_matches_previous_u16_shift() {
        let component = DecodedComponent {
            data: &[32768],
            precision: 16,
            signed: false,
        };

        assert_eq!(decode_sample_u8(&component, true, 0), 128);
    }

    #[test]
    fn unsigned_u8_fast_path_interleaves_rgb_rows() {
        let r = DecodedComponent {
            data: &[1, 2, 3, 4],
            precision: 8,
            signed: false,
        };
        let g = DecodedComponent {
            data: &[10, 20, 30, 40],
            precision: 8,
            signed: false,
        };
        let b = DecodedComponent {
            data: &[100, 110, 120, 130],
            precision: 8,
            signed: false,
        };
        let mut pixels = vec![0_u8; 12];

        fill_unsigned_u8_rows::<3>(&mut pixels, 2, [&r, &g, &b]);

        assert_eq!(pixels, vec![1, 10, 100, 2, 20, 110, 3, 30, 120, 4, 40, 130]);
    }

    #[test]
    fn unsigned_u16_fast_path_interleaves_rows() {
        let grey = DecodedComponent {
            data: &[1000, 2000, 3000, 4000],
            precision: 16,
            signed: false,
        };
        let alpha = DecodedComponent {
            data: &[5000, 6000, 7000, 8000],
            precision: 16,
            signed: false,
        };
        let mut pixels = vec![0_u16; 8];

        fill_unsigned_u16_rows::<2>(&mut pixels, 2, [&grey, &alpha]);

        assert_eq!(pixels, vec![1000, 5000, 2000, 6000, 3000, 7000, 4000, 8000]);
    }

    #[test]
    fn default_tile_dimension_uses_larger_tiles_for_medium_images() {
        assert_eq!(default_tile_dimension(None, 512), 512);
        assert_eq!(default_tile_dimension(None, 2048), 1024);
        assert_eq!(default_tile_dimension(None, 8192), 512);
    }

    #[test]
    fn encoded_buffer_write_grows_without_zero_filling_appends() {
        let mut buffer = EncodedJp2Buffer {
            offset: 0,
            bytes: Vec::with_capacity(1),
        };

        assert_eq!(buffer.write_from(&[1, 2, 3]), 3);
        assert_eq!(buffer.write_from(&[4, 5]), 2);
        assert_eq!(buffer.bytes, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn fast_tiled_copy_clips_edge_tiles_to_output_bounds() {
        let mut pixels = vec![0_u8; 3 * 2 * 3];
        let r = [1, 4];
        let g = [11, 14];
        let b = [21, 24];
        let components = [
            sys::opj_image_comp_t {
                dx: 1,
                dy: 1,
                w: 1,
                h: 2,
                x0: 2,
                y0: 0,
                prec: 8,
                bpp: 8,
                sgnd: 0,
                resno_decoded: 0,
                factor: 0,
                data: r.as_ptr() as *mut i32,
                alpha: 0,
            },
            sys::opj_image_comp_t {
                dx: 1,
                dy: 1,
                w: 1,
                h: 2,
                x0: 2,
                y0: 0,
                prec: 8,
                bpp: 8,
                sgnd: 0,
                resno_decoded: 0,
                factor: 0,
                data: g.as_ptr() as *mut i32,
                alpha: 0,
            },
            sys::opj_image_comp_t {
                dx: 1,
                dy: 1,
                w: 1,
                h: 2,
                x0: 2,
                y0: 0,
                prec: 8,
                bpp: 8,
                sgnd: 0,
                resno_decoded: 0,
                factor: 0,
                data: b.as_ptr() as *mut i32,
                alpha: 0,
            },
        ];
        let image = sys::opj_image_t {
            x0: 0,
            y0: 0,
            x1: 3,
            y1: 2,
            numcomps: 3,
            color_space: sys::COLOR_SPACE::OPJ_CLRSPC_SRGB,
            comps: components.as_ptr() as *mut sys::opj_image_comp_t,
            icc_profile_buf: ptr::null_mut(),
            icc_profile_len: 0,
        };
        let plan = FastTiledDecodePlan {
            width: 3,
            height: 2,
            bands: 3,
            precision: 8,
            resolution_count: 1,
            tile_width: 2,
            tile_height: 2,
            tiles_across: 2,
            tiles_down: 1,
        };

        copy_fast_tiled_u8::<3>(&mut pixels, &image, &plan, 2, 0).unwrap();

        assert_eq!(
            pixels,
            vec![0, 0, 0, 0, 0, 0, 1, 11, 21, 0, 0, 0, 0, 0, 0, 4, 14, 24]
        );
    }
}
