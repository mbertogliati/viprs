//! Heif Support adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

use std::{ffi::c_void, ptr, slice, sync::OnceLock};

use libheif_rs::{CompressionFormat, ImageHandle, LibHeif};
use libheif_sys as lh;

use crate::domain::codec_options::HeifSubsampling;
use crate::domain::error::ViprsError;
use crate::domain::format::BandFormat;
use crate::domain::image::{Image, ImageMetadata};

static LIBHEIF: OnceLock<Result<LibHeif, String>> = OnceLock::new();
const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";

#[derive(Clone, Copy)]
enum ExifEndian {
    Little,
    Big,
}

/// Metadata to embed in an encoded HEIF/AVIF bitstream.
///
/// Pass `Default::default()` (all `None`) to strip all metadata.
#[derive(Clone, Copy, Default)]
/// The `HeifWriteMetadata` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```ignore
/// let _ = core::mem::size_of::<viprs::adapters::codecs::heif_support::HeifWriteMetadata>();
/// ```
pub(crate) struct HeifWriteMetadata<'a> {
    pub exif: Option<&'a [u8]>,
    pub xmp: Option<&'a [u8]>,
    /// Raw ICC profile bytes (typically "rICC" or "prof" colour profile).
    pub icc_profile: Option<&'a [u8]>,
}

/// `shared_libheif` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::shared_libheif;
/// ```
pub(crate) fn shared_libheif(context: &str) -> Result<&'static LibHeif, ViprsError> {
    match LIBHEIF.get_or_init(|| LibHeif::new_checked().map_err(|e| e.to_string())) {
        Ok(lib_heif) => Ok(lib_heif),
        Err(message) => Err(ViprsError::Codec(format!(
            "{context}: init libheif: {message}"
        ))),
    }
}

#[inline]
/// `checked_interleaved_sample_count` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::checked_interleaved_sample_count;
/// ```
pub(crate) fn checked_interleaved_sample_count(
    width: u32,
    height: u32,
    bands: u32,
    details: &'static str,
) -> Result<usize, ViprsError> {
    (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(bands as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(width) * u128::from(height) * u128::from(bands),
            limit_bytes: usize::MAX as u128,
            details,
        })
}

#[inline]
/// `checked_interleaved_row_bytes` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::checked_interleaved_row_bytes;
/// ```
pub(crate) fn checked_interleaved_row_bytes(
    width: u32,
    bands: u32,
    bytes_per_sample: usize,
    details: &'static str,
) -> Result<usize, ViprsError> {
    checked_interleaved_sample_count(width, 1, bands, details).and_then(|row_samples| {
        row_samples
            .checked_mul(bytes_per_sample)
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width,
                height: 1,
                bands,
                bytes: u128::from(width) * u128::from(bands) * u128::from(bytes_per_sample as u64),
                limit_bytes: usize::MAX as u128,
                details,
            })
    })
}

#[inline]
/// `checked_interleaved_byte_count` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::checked_interleaved_byte_count;
/// ```
pub(crate) fn checked_interleaved_byte_count(
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: usize,
    details: &'static str,
) -> Result<usize, ViprsError> {
    (width as u64)
        .checked_mul(height as u64)
        .and_then(|n| n.checked_mul(bands as u64))
        .and_then(|n| n.checked_mul(bytes_per_sample as u64))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(width)
                * u128::from(height)
                * u128::from(bands)
                * u128::from(bytes_per_sample as u64),
            limit_bytes: usize::MAX as u128,
            details,
        })
}

fn ok_error() -> lh::heif_error {
    lh::heif_error {
        code: lh::heif_error_code_heif_error_Ok,
        subcode: lh::heif_suberror_code_heif_suberror_Unspecified,
        message: ptr::null(),
    }
}

fn format_error(prefix: &str, error: lh::heif_error) -> ViprsError {
    let message = if error.message.is_null() {
        String::new()
    } else {
        // SAFETY: libheif guarantees `message` is either null or a valid NUL-terminated string.
        unsafe { std::ffi::CStr::from_ptr(error.message) }
            .to_string_lossy()
            .into_owned()
    };
    ViprsError::Codec(format!(
        "{prefix}: {message} ({}.{} )",
        error.code, error.subcode
    ))
}

#[inline]
fn format_interleaved_error(
    context: &str,
    bands: u32,
    bit_depth: u8,
    error: lh::heif_error,
) -> ViprsError {
    if context == "heif"
        && bands == 4
        && bit_depth == 16
        && error.subcode == lh::heif_suberror_code_heif_suberror_Unsupported_bit_depth
    {
        return ViprsError::Codec(
            "heif: linked libheif encoder/container does not support 16-bit interleaved RGBA"
                .into(),
        );
    }

    format_error(context, error)
}

unsafe extern "C" fn vec_writer(
    _ctx: *mut lh::heif_context,
    data: *const c_void,
    size: usize,
    userdata: *mut c_void,
) -> lh::heif_error {
    let buffer = userdata.cast::<Vec<u8>>();
    if buffer.is_null() {
        return lh::heif_error {
            code: lh::heif_error_code_heif_error_Usage_error,
            subcode: lh::heif_suberror_code_heif_suberror_Null_pointer_argument,
            message: ptr::null(),
        };
    }
    if data.is_null() && size != 0 {
        return lh::heif_error {
            code: lh::heif_error_code_heif_error_Usage_error,
            subcode: lh::heif_suberror_code_heif_suberror_Null_pointer_argument,
            message: ptr::null(),
        };
    }

    // SAFETY: libheif calls this with the same `userdata` pointer passed to `heif_context_write`, which we set to a valid `Vec<u8>` for the duration of the write call.
    let buffer = unsafe { &mut *buffer };
    let start = buffer.len();
    buffer.reserve(size);
    // SAFETY: `data` points to `size` bytes owned by libheif for the duration of the callback, and `reserve(size)` ensures capacity for the append.
    unsafe {
        ptr::copy_nonoverlapping(data.cast::<u8>(), buffer.as_mut_ptr().add(start), size);
        buffer.set_len(start + size);
    }
    ok_error()
}

fn encode_context_to_bytes(
    context: &str,
    ctx: *mut lh::heif_context,
) -> Result<Vec<u8>, ViprsError> {
    let mut output: Vec<u8> = Vec::new();
    let mut writer = lh::heif_writer {
        writer_api_version: 1,
        write: Some(vec_writer),
    };

    let error = {
        // SAFETY: `libheif` calls `writer->write(ctx, data, size, userdata)`
        // synchronously from `heif_context_write()` and does not retain either
        // `writer` or `userdata` after the call returns; upstream
        // `libheif/api/libheif/heif_context.cc` confirms this. `output` therefore outlives the
        // callback, the raw pointer stays valid for the full call, and Rust does not
        // touch `output` again until `heif_context_write()` returns, so there is no
        // mutable aliasing or data race. We keep the direct C path because
        // `libheif-rs` 0.20 `write_to_bytes()` still uses a broken `vector_writer`.
        unsafe {
            lh::heif_context_write(
                ctx,
                std::ptr::from_mut(&mut writer),
                std::ptr::from_mut(&mut output).cast(),
            )
        }
    };
    if error.code != lh::heif_error_code_heif_error_Ok {
        return Err(format_error(context, error));
    }

    Ok(output)
}

#[inline]
fn is_unsupported_parameter(error: lh::heif_error) -> bool {
    error.subcode == lh::heif_suberror_code_heif_suberror_Unsupported_parameter
}

#[inline]
fn available_threads() -> i32 {
    std::thread::available_parallelism().map_or(1, |threads| {
        i32::try_from(threads.get()).unwrap_or(i32::MAX)
    })
}

fn set_encoder_integer_parameter(
    context: &str,
    encoder: *mut lh::heif_encoder,
    name: &[u8],
    value: i32,
) -> Result<(), ViprsError> {
    let error = {
        // SAFETY: `encoder` is a live libheif encoder pointer owned by the caller, and
        // `name` is a NUL-terminated parameter name that stays alive for the call.
        unsafe { lh::heif_encoder_set_parameter_integer(encoder, name.as_ptr().cast(), value) }
    };
    if error.code != lh::heif_error_code_heif_error_Ok && !is_unsupported_parameter(error) {
        return Err(format_error(context, error));
    }

    Ok(())
}

fn set_encoder_string_parameter(
    context: &str,
    encoder: *mut lh::heif_encoder,
    name: &[u8],
    value: &[u8],
) -> Result<(), ViprsError> {
    let error = {
        // SAFETY: `encoder` is a live libheif encoder pointer owned by the caller, and
        // both `name` and `value` are NUL-terminated strings that stay alive until the
        // function returns.
        unsafe {
            lh::heif_encoder_set_parameter_string(
                encoder,
                name.as_ptr().cast(),
                value.as_ptr().cast(),
            )
        }
    };
    if error.code != lh::heif_error_code_heif_error_Ok && !is_unsupported_parameter(error) {
        return Err(format_error(context, error));
    }

    Ok(())
}

fn set_encoder_boolean_parameter(
    context: &str,
    encoder: *mut lh::heif_encoder,
    name: &[u8],
    value: bool,
) -> Result<(), ViprsError> {
    let error = {
        // SAFETY: `encoder` is a live libheif encoder pointer owned by the caller, and
        // `name` is a NUL-terminated parameter name that stays alive for the call.
        unsafe {
            lh::heif_encoder_set_parameter_boolean(encoder, name.as_ptr().cast(), i32::from(value))
        }
    };
    if error.code != lh::heif_error_code_heif_error_Ok && !is_unsupported_parameter(error) {
        return Err(format_error(context, error));
    }

    Ok(())
}

fn configure_encoder_threads(
    context: &str,
    encoder: *mut lh::heif_encoder,
) -> Result<(), ViprsError> {
    // SAFETY: `encoder` is live for the duration of this function, and `heif_encoder_list_parameters()` returns a null-terminated array of parameter pointers.
    unsafe {
        let mut param = lh::heif_encoder_list_parameters(encoder);
        while !param.is_null() {
            let Some(parameter) = (*param).as_ref() else {
                break;
            };

            let name = lh::heif_encoder_parameter_get_name(parameter);
            if !name.is_null() {
                // SAFETY: libheif guarantees a valid NUL-terminated parameter name.
                let parameter_name = std::ffi::CStr::from_ptr(name).to_bytes();
                if parameter_name == b"threads" {
                    let mut have_minimum = 0;
                    let mut have_maximum = 0;
                    let mut minimum = 0;
                    let mut maximum = 0;
                    let error = lh::heif_encoder_parameter_get_valid_integer_values(
                        parameter,
                        std::ptr::from_mut(&mut have_minimum),
                        std::ptr::from_mut(&mut have_maximum),
                        std::ptr::from_mut(&mut minimum),
                        std::ptr::from_mut(&mut maximum),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    );
                    if error.code != lh::heif_error_code_heif_error_Ok {
                        return Err(format_error(context, error));
                    }

                    let mut threads = available_threads();
                    if have_minimum != 0 {
                        threads = threads.max(minimum);
                    }
                    if have_maximum != 0 {
                        threads = threads.min(maximum);
                    }
                    return set_encoder_integer_parameter(context, encoder, b"threads\0", threads);
                }
            }

            param = param.add(1);
        }
    }

    Ok(())
}

/// `encode_interleaved` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::encode_interleaved;
/// ```
pub(crate) fn encode_interleaved(
    context: &str,
    compression: CompressionFormat,
    width: u32,
    height: u32,
    bands: u32,
    bit_depth: u8,
    pixels: &[u8],
    lossless: bool,
    quality: u8,
    effort: Option<u8>,
    subsampling: HeifSubsampling,
    metadata: HeifWriteMetadata<'_>,
) -> Result<Vec<u8>, ViprsError> {
    let _ = shared_libheif(context)?;

    let bytes_per_sample = if bit_depth > 8 { 2usize } else { 1usize };
    let row_bytes = width as usize * bands as usize * bytes_per_sample;
    let expected_len = row_bytes * height as usize;
    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "{context}: expected {expected_len} input bytes, got {}",
            pixels.len()
        )));
    }

    let chroma = match (bit_depth > 8, bands) {
        (false, 3) => lh::heif_chroma_heif_chroma_interleaved_RGB,
        (false, 4) => lh::heif_chroma_heif_chroma_interleaved_RGBA,
        (true, 3) => lh::heif_chroma_heif_chroma_interleaved_RRGGBB_BE,
        (true, 4) => lh::heif_chroma_heif_chroma_interleaved_RRGGBBAA_BE,
        _ => {
            return Err(ViprsError::Codec(format!(
                "{context}: unsupported band count {bands}"
            )));
        }
    };

    // SAFETY: all libheif pointers created below are checked for null before dereference, every borrowed plane slice stays within the `stride` and dimensions reported by libheif, and each allocated libheif object is released exactly once before returning from this function.
    unsafe {
        let ctx = lh::heif_context_alloc();
        if ctx.is_null() {
            return Err(ViprsError::Codec(format!(
                "{context}: heif_context_alloc failed"
            )));
        }

        let mut encoder: *mut lh::heif_encoder = ptr::null_mut();
        let mut image: *mut lh::heif_image = ptr::null_mut();
        let mut options: *mut lh::heif_encoding_options = ptr::null_mut();
        let mut handle: *mut lh::heif_image_handle = ptr::null_mut();
        let mut nclx_profile: *mut lh::heif_color_profile_nclx = ptr::null_mut();

        let result = (|| {
            let error = lh::heif_context_get_encoder_for_format(
                ctx,
                compression as _,
                std::ptr::from_mut(&mut encoder),
            );
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_error(context, error));
            }

            let error = lh::heif_encoder_set_lossy_quality(encoder, i32::from(quality));
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_error(context, error));
            }

            let error = lh::heif_encoder_set_lossless(encoder, i32::from(lossless));
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_error(context, error));
            }

            let effort = i32::from(effort.unwrap_or(4).min(9));
            let speed = 9 - effort;
            set_encoder_integer_parameter(context, encoder, b"speed\0", speed)?;

            if !(lossless && compression == CompressionFormat::Av1) {
                let chroma_parameter = match subsampling {
                    HeifSubsampling::Auto if lossless || quality >= 90 => b"444\0".as_slice(),
                    HeifSubsampling::Auto | HeifSubsampling::Subsample420 => b"420\0".as_slice(),
                    HeifSubsampling::Subsample422 => b"422\0".as_slice(),
                    HeifSubsampling::Subsample444 => b"444\0".as_slice(),
                };
                set_encoder_string_parameter(context, encoder, b"chroma\0", chroma_parameter)?;
            }

            configure_encoder_threads(context, encoder)?;
            set_encoder_boolean_parameter(context, encoder, b"auto-tiles\0", true)?;
            set_encoder_boolean_parameter(context, encoder, b"enable-intrabc\0", false)?;

            let error = lh::heif_image_create(
                width as _,
                height as _,
                lh::heif_colorspace_heif_colorspace_RGB,
                chroma,
                std::ptr::from_mut(&mut image),
            );
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_interleaved_error(context, bands, bit_depth, error));
            }

            let error = lh::heif_image_add_plane(
                image,
                lh::heif_channel_heif_channel_interleaved,
                width as _,
                height as _,
                i32::from(bit_depth),
            );
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_interleaved_error(context, bands, bit_depth, error));
            }

            let mut stride = 0;
            let plane = lh::heif_image_get_plane(
                image,
                lh::heif_channel_heif_channel_interleaved,
                std::ptr::from_mut(&mut stride),
            );
            if plane.is_null() {
                return Err(ViprsError::Codec(format!(
                    "{context}: heif_image_get_plane failed"
                )));
            }

            for row in 0..height as usize {
                let src_start = row * row_bytes;
                let src_end = src_start + row_bytes;
                let dst_start = row * stride as usize;
                // SAFETY: `plane` points to a buffer allocated by libheif with at
                // least `height * stride` bytes. Each row copy stays within bounds.
                let dst = slice::from_raw_parts_mut(plane.add(dst_start), row_bytes);
                dst.copy_from_slice(&pixels[src_start..src_end]);
            }

            // Embed ICC profile into the heif_image before encoding.
            // libvips uses "rICC" for reduced (embedded) profiles; we replicate that.
            // see: heifsave.c vips_foreign_save_heif_add_icc
            if let Some(icc) = metadata.icc_profile.filter(|data| !data.is_empty()) {
                let error = lh::heif_image_set_raw_color_profile(
                    image,
                    b"rICC\0".as_ptr().cast(),
                    icc.as_ptr().cast(),
                    icc.len(),
                );
                if error.code != lh::heif_error_code_heif_error_Ok {
                    return Err(format_error(context, error));
                }
            }

            options = lh::heif_encoding_options_alloc();
            if options.is_null() {
                return Err(ViprsError::Codec(format!(
                    "{context}: heif_encoding_options_alloc failed"
                )));
            }
            (*options).save_alpha_channel = u8::from(bands == 4);
            if lossless && compression == CompressionFormat::Av1 {
                nclx_profile = lh::heif_nclx_color_profile_alloc();
                if nclx_profile.is_null() {
                    return Err(ViprsError::Codec(format!(
                        "{context}: heif_nclx_color_profile_alloc failed"
                    )));
                }
                (*nclx_profile).matrix_coefficients =
                    lh::heif_matrix_coefficients_heif_matrix_coefficients_RGB_GBR;
                (*options).output_nclx_profile = nclx_profile;
                (*options).macOS_compatibility_workaround_no_nclx_profile = 0;
            }

            let error = lh::heif_context_encode_image(
                ctx,
                image,
                encoder,
                options,
                std::ptr::from_mut(&mut handle),
            );
            if error.code != lh::heif_error_code_heif_error_Ok {
                return Err(format_interleaved_error(context, bands, bit_depth, error));
            }

            if let Some(exif) = metadata.exif.filter(|data| !data.is_empty()) {
                let exif_len = i32::try_from(exif.len()).map_err(|_| {
                    ViprsError::Codec(format!("{context}: EXIF metadata exceeds libheif limits"))
                })?;
                let error =
                    lh::heif_context_add_exif_metadata(ctx, handle, exif.as_ptr().cast(), exif_len);
                if error.code != lh::heif_error_code_heif_error_Ok {
                    return Err(format_error(context, error));
                }
            }

            if let Some(xmp) = metadata.xmp.filter(|data| !data.is_empty()) {
                let xmp_len = i32::try_from(xmp.len()).map_err(|_| {
                    ViprsError::Codec(format!("{context}: XMP metadata exceeds libheif limits"))
                })?;
                let error =
                    lh::heif_context_add_XMP_metadata(ctx, handle, xmp.as_ptr().cast(), xmp_len);
                if error.code != lh::heif_error_code_heif_error_Ok {
                    return Err(format_error(context, error));
                }
            }

            let encoded = encode_context_to_bytes(context, ctx)?;

            Ok(encoded)
        })();

        if !handle.is_null() {
            lh::heif_image_handle_release(handle);
        }
        if !options.is_null() {
            lh::heif_encoding_options_free(options);
        }
        if !nclx_profile.is_null() {
            lh::heif_nclx_color_profile_free(nclx_profile);
        }
        if !image.is_null() {
            lh::heif_image_release(image);
        }
        if !encoder.is_null() {
            lh::heif_encoder_release(encoder);
        }
        lh::heif_context_free(ctx);

        result
    }
}

/// `read_metadata` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::read_metadata;
/// ```
pub(crate) fn read_metadata(
    context: &str,
    handle: &ImageHandle,
) -> Result<ImageMetadata, ViprsError> {
    let mut metadata = ImageMetadata::default();

    // Read raw ICC colour profile (rICC or prof) from the image handle.
    // see: heifload.c — libvips reads both rICC and prof via heif_image_handle_get_raw_color_profile.
    if let Some(profile) = handle.color_profile_raw() {
        metadata.icc_profile = Some(profile.data);
    }

    let metadata_count = handle.number_of_metadata_blocks(0);
    if metadata_count <= 0 {
        return Ok(metadata);
    }

    let mut item_ids = vec![0; metadata_count as usize];
    let item_count = handle.metadata_block_ids(&mut item_ids, 0);
    for item_id in item_ids.into_iter().take(item_count) {
        let metadata_type = handle.metadata_type(item_id).unwrap_or_default();
        let content_type = handle.metadata_content_type(item_id).unwrap_or_default();
        let mut data = handle
            .metadata(item_id)
            .map_err(|e| ViprsError::Codec(format!("{context}: metadata read: {e}")))?;

        if metadata_type.eq_ignore_ascii_case("exif") {
            if data.len() > 4 {
                data.drain(..4);
            } else {
                data.clear();
            }
            metadata.orientation = extract_exif_orientation(&data);
            metadata.exif = Some(data);
        } else if content_type.eq_ignore_ascii_case("application/rdf+xml") {
            metadata.xmp = Some(data);
        }
    }

    Ok(metadata)
}

fn read_u16(bytes: &[u8], endian: ExifEndian) -> Option<u16> {
    let pair: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
    Some(match endian {
        ExifEndian::Little => u16::from_le_bytes(pair),
        ExifEndian::Big => u16::from_be_bytes(pair),
    })
}

fn read_u32(bytes: &[u8], endian: ExifEndian) -> Option<u32> {
    let quad: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
    Some(match endian {
        ExifEndian::Little => u32::from_le_bytes(quad),
        ExifEndian::Big => u32::from_be_bytes(quad),
    })
}

fn extract_exif_orientation(exif: &[u8]) -> Option<u8> {
    let tiff = exif.strip_prefix(EXIF_SIGNATURE)?;
    if tiff.len() < 8 {
        return None;
    }

    let endian = match &tiff[..2] {
        b"II" => ExifEndian::Little,
        b"MM" => ExifEndian::Big,
        _ => return None,
    };
    if read_u16(&tiff[2..4], endian)? != 42 {
        return None;
    }

    let ifd0_offset = read_u32(&tiff[4..8], endian)? as usize;
    let entry_count = read_u16(tiff.get(ifd0_offset..ifd0_offset + 2)?, endian)? as usize;
    let entries_start = ifd0_offset + 2;

    for entry_index in 0..entry_count {
        let entry_offset = entries_start + entry_index * 12;
        let entry = tiff.get(entry_offset..entry_offset + 12)?;
        if read_u16(&entry[0..2], endian)? != 0x0112 {
            continue;
        }
        if read_u16(&entry[2..4], endian)? != 3 || read_u32(&entry[4..8], endian)? != 1 {
            return None;
        }

        let value = read_u16(&entry[8..10], endian)?;
        return (1..=8).contains(&value).then_some(value as u8);
    }

    None
}

fn apply_orientation_to_pixels<T: Copy>(
    pixels: Vec<T>,
    width: u32,
    height: u32,
    bands: u32,
    orientation: u8,
    context: &str,
) -> Result<(u32, u32, Vec<T>), ViprsError> {
    if !(2..=8).contains(&orientation) || width == 0 || height == 0 || bands == 0 {
        return Ok((width, height, pixels));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let bands_usize = bands as usize;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .and_then(|pixel_count| pixel_count.checked_mul(bands_usize))
        .ok_or_else(|| ViprsError::Codec(format!("{context}: decoded dimensions overflow")))?;
    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "{context}: decoded buffer length mismatch (got {}, expected {expected_len})",
            pixels.len()
        )));
    }

    let (out_width, out_height) = match orientation {
        5..=8 => (height, width),
        _ => (width, height),
    };
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;
    let mut out = vec![pixels[0]; out_width_usize * out_height_usize * bands_usize];

    for out_y in 0..out_height_usize {
        for out_x in 0..out_width_usize {
            let (src_x, src_y) = match orientation {
                2 => (width_usize - 1 - out_x, out_y),
                3 => (width_usize - 1 - out_x, height_usize - 1 - out_y),
                4 => (out_x, height_usize - 1 - out_y),
                5 => (out_y, out_x),
                6 => (out_y, height_usize - 1 - out_x),
                7 => (width_usize - 1 - out_y, height_usize - 1 - out_x),
                8 => (width_usize - 1 - out_y, out_x),
                _ => (out_x, out_y),
            };
            let src_base = (src_y * width_usize + src_x) * bands_usize;
            let dst_base = (out_y * out_width_usize + out_x) * bands_usize;
            out[dst_base..dst_base + bands_usize]
                .copy_from_slice(&pixels[src_base..src_base + bands_usize]);
        }
    }

    Ok((out_width, out_height, out))
}

/// `normalize_decoded_image` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs::adapters::codecs::heif_support::normalize_decoded_image;
/// ```
pub(crate) fn normalize_decoded_image<F: BandFormat>(
    image: Image<F>,
    no_rotate: bool,
    context: &str,
) -> Result<Image<F>, ViprsError>
where
    F::Sample: Copy,
{
    let mut metadata = image.metadata().clone();
    let orientation = metadata
        .orientation
        .or_else(|| metadata.exif.as_deref().and_then(extract_exif_orientation));
    metadata.orientation = orientation;

    if no_rotate {
        return Ok(image.with_metadata(metadata));
    }

    if orientation.is_some() {
        metadata.remove_orientation();
        metadata.orientation = Some(1);
    }

    if let Some(orientation @ 2..=8) = orientation {
        let width = image.width();
        let height = image.height();
        let bands = image.bands();
        let (width, height, pixels) = apply_orientation_to_pixels(
            image.into_buffer(),
            width,
            height,
            bands,
            orientation,
            context,
        )?;
        Image::from_buffer(width, height, bands, pixels)
            .map(|image| image.with_metadata(metadata))
            .map_err(|e| ViprsError::Codec(format!("{context}: {e}")))
    } else {
        Ok(image.with_metadata(metadata))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExifEndian, HeifWriteMetadata, apply_orientation_to_pixels, available_threads,
        checked_interleaved_byte_count, checked_interleaved_row_bytes,
        checked_interleaved_sample_count, encode_interleaved, extract_exif_orientation,
        format_error, format_interleaved_error, is_unsupported_parameter, normalize_decoded_image,
        ok_error, read_u16, read_u32, vec_writer,
    };
    use crate::domain::{
        codec_options::HeifSubsampling,
        format::U8,
        image::{Image, ImageMetadata},
    };
    use libheif_rs::CompressionFormat;
    use std::{ffi::c_void, ptr};

    fn exif_blob(orientation: u16) -> Vec<u8> {
        let mut exif = Vec::with_capacity(32);
        exif.extend_from_slice(b"Exif\0\0");
        exif.extend_from_slice(b"II");
        exif.extend_from_slice(&42u16.to_le_bytes());
        exif.extend_from_slice(&8u32.to_le_bytes());
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&0x0112u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&0u16.to_le_bytes());
        exif.extend_from_slice(&0u32.to_le_bytes());
        exif
    }

    fn orient_pixels(
        pixels: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        orientation: u8,
    ) -> (u32, u32, Vec<u8>) {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let bands_usize = bands as usize;
        let (out_width, out_height) = match orientation {
            5..=8 => (height, width),
            _ => (width, height),
        };
        let out_width_usize = out_width as usize;
        let out_height_usize = out_height as usize;
        let mut out = vec![0u8; out_width_usize * out_height_usize * bands_usize];

        for out_y in 0..out_height_usize {
            for out_x in 0..out_width_usize {
                let (src_x, src_y) = match orientation {
                    2 => (width_usize - 1 - out_x, out_y),
                    3 => (width_usize - 1 - out_x, height_usize - 1 - out_y),
                    4 => (out_x, height_usize - 1 - out_y),
                    5 => (out_y, out_x),
                    6 => (out_y, height_usize - 1 - out_x),
                    7 => (width_usize - 1 - out_y, height_usize - 1 - out_x),
                    8 => (width_usize - 1 - out_y, out_x),
                    _ => (out_x, out_y),
                };
                let src_base = (src_y * width_usize + src_x) * bands_usize;
                let dst_base = (out_y * out_width_usize + out_x) * bands_usize;
                out[dst_base..dst_base + bands_usize]
                    .copy_from_slice(&pixels[src_base..src_base + bands_usize]);
            }
        }

        (out_width, out_height, out)
    }

    fn exif_orientation(exif: &[u8]) -> Option<u8> {
        exif.get(18..20)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .and_then(|value| u8::try_from(value).ok())
            .filter(|value| (1..=8).contains(value))
    }

    fn decoded_image_fixture() -> Image<U8> {
        let pixels = vec![
            10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
        ];
        Image::from_buffer(2, 3, 3, pixels)
            .unwrap()
            .with_metadata(ImageMetadata {
                exif: Some(exif_blob(6)),
                xmp: Some(br#"<x:xmpmeta><rdf:RDF>heif-support</rdf:RDF></x:xmpmeta>"#.to_vec()),
                orientation: Some(6),
                ..ImageMetadata::default()
            })
    }

    #[test]
    fn vec_writer_appends_multiple_chunks() {
        let mut output: Vec<u8> = Vec::new();
        let first = [1_u8, 2, 3];
        let second = [4_u8, 5];

        // SAFETY: the callback receives valid chunk pointers and a live `Vec<u8>` userdata pointer for the duration of each call.
        let first_err = unsafe {
            vec_writer(
                ptr::null_mut(),
                first.as_ptr().cast::<c_void>(),
                first.len(),
                std::ptr::from_mut(&mut output).cast::<c_void>(),
            )
        };
        // SAFETY: same invariant as above for the second chunk.
        let second_err = unsafe {
            vec_writer(
                ptr::null_mut(),
                second.as_ptr().cast::<c_void>(),
                second.len(),
                std::ptr::from_mut(&mut output).cast::<c_void>(),
            )
        };

        assert_eq!(first_err.code, ok_error().code);
        assert_eq!(second_err.code, ok_error().code);
        assert_eq!(output, vec![1_u8, 2, 3, 4, 5]);
    }

    #[test]
    fn vec_writer_rejects_null_data_when_size_is_nonzero() {
        let mut output: Vec<u8> = Vec::new();

        // SAFETY: `userdata` points to a live `Vec<u8>`, and the test exercises the explicit null-data guard in `vec_writer`.
        let error = unsafe {
            vec_writer(
                ptr::null_mut(),
                ptr::null(),
                1,
                std::ptr::from_mut(&mut output).cast::<c_void>(),
            )
        };

        assert_eq!(
            error.code,
            libheif_sys::heif_error_code_heif_error_Usage_error
        );
        assert!(output.is_empty());
    }

    #[test]
    fn vec_writer_rejects_null_userdata() {
        let data = [1_u8, 2, 3];

        // SAFETY: the test passes a valid data pointer and explicitly exercises the null-userdata guard.
        let error = unsafe {
            vec_writer(
                ptr::null_mut(),
                data.as_ptr().cast::<c_void>(),
                data.len(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            error.code,
            libheif_sys::heif_error_code_heif_error_Usage_error
        );
        assert_eq!(
            error.subcode,
            libheif_sys::heif_suberror_code_heif_suberror_Null_pointer_argument
        );
    }

    #[test]
    fn checked_interleaved_helpers_validate_counts_and_overflow() {
        assert_eq!(
            checked_interleaved_sample_count(4, 3, 2, "sample overflow").unwrap(),
            24
        );
        assert_eq!(
            checked_interleaved_row_bytes(4, 3, 2, "row overflow").unwrap(),
            24
        );
        assert_eq!(
            checked_interleaved_byte_count(4, 3, 2, 2, "byte overflow").unwrap(),
            48
        );

        let sample_err =
            checked_interleaved_sample_count(u32::MAX, u32::MAX, 4, "sample overflow").unwrap_err();
        assert!(matches!(
            sample_err,
            crate::domain::error::ViprsError::ImageTooLarge { details, .. } if details == "sample overflow"
        ));

        let row_err =
            checked_interleaved_row_bytes(u32::MAX, u32::MAX, 2, "row overflow").unwrap_err();
        assert!(matches!(
            row_err,
            crate::domain::error::ViprsError::ImageTooLarge { details, .. } if details == "row overflow"
        ));

        let byte_err =
            checked_interleaved_byte_count(u32::MAX, u32::MAX, 4, 2, "byte overflow").unwrap_err();
        assert!(matches!(
            byte_err,
            crate::domain::error::ViprsError::ImageTooLarge { details, .. } if details == "byte overflow"
        ));
    }

    #[test]
    fn format_error_includes_optional_message() {
        static MESSAGE: &[u8] = b"boom\0";

        let with_message = format_error(
            "heif",
            libheif_sys::heif_error {
                code: 7,
                subcode: 11,
                message: MESSAGE.as_ptr().cast(),
            },
        );
        assert_eq!(with_message.to_string(), "codec error: heif: boom (7.11 )");

        let without_message = format_error(
            "heif",
            libheif_sys::heif_error {
                code: 3,
                subcode: 5,
                message: ptr::null(),
            },
        );
        assert_eq!(without_message.to_string(), "codec error: heif:  (3.5 )");
    }

    #[test]
    fn format_interleaved_error_special_cases_heif_rgba_16bit() {
        let special = format_interleaved_error(
            "heif",
            4,
            16,
            libheif_sys::heif_error {
                code: libheif_sys::heif_error_code_heif_error_Usage_error,
                subcode: libheif_sys::heif_suberror_code_heif_suberror_Unsupported_bit_depth,
                message: ptr::null(),
            },
        );
        assert_eq!(
            special.to_string(),
            "codec error: heif: linked libheif encoder/container does not support 16-bit interleaved RGBA"
        );

        let passthrough = format_interleaved_error(
            "avif",
            4,
            16,
            libheif_sys::heif_error {
                code: 9,
                subcode: 12,
                message: ptr::null(),
            },
        );
        assert_eq!(passthrough.to_string(), "codec error: avif:  (9.12 )");
    }

    #[test]
    fn helper_parsers_cover_endianness_and_orientation_validation() {
        assert_eq!(read_u16(&[0x34, 0x12], ExifEndian::Little), Some(0x1234));
        assert_eq!(read_u16(&[0x12, 0x34], ExifEndian::Big), Some(0x1234));
        assert_eq!(
            read_u32(&[0x78, 0x56, 0x34, 0x12], ExifEndian::Little),
            Some(0x1234_5678)
        );
        assert_eq!(
            read_u32(&[0x12, 0x34, 0x56, 0x78], ExifEndian::Big),
            Some(0x1234_5678)
        );

        let mut big_endian_exif = Vec::with_capacity(32);
        big_endian_exif.extend_from_slice(b"Exif\0\0MM");
        big_endian_exif.extend_from_slice(&42u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&8u32.to_be_bytes());
        big_endian_exif.extend_from_slice(&1u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&0x0112u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&3u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&1u32.to_be_bytes());
        big_endian_exif.extend_from_slice(&8u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&0u16.to_be_bytes());
        big_endian_exif.extend_from_slice(&0u32.to_be_bytes());

        assert_eq!(extract_exif_orientation(&big_endian_exif), Some(8));
        assert_eq!(extract_exif_orientation(b"not-exif"), None);
        assert_eq!(extract_exif_orientation(&exif_blob(9)), None);
    }

    #[test]
    fn orientation_helpers_cover_identity_and_length_mismatch() {
        let pixels = vec![1_u8, 2, 3, 4, 5, 6];
        let identity = apply_orientation_to_pixels(pixels.clone(), 1, 2, 3, 1, "heif").unwrap();
        assert_eq!(identity, (1, 2, pixels.clone()));

        let mismatch =
            apply_orientation_to_pixels(vec![1_u8, 2, 3], 1, 2, 3, 6, "heif").unwrap_err();
        assert!(
            mismatch
                .to_string()
                .contains("decoded buffer length mismatch"),
            "{mismatch}"
        );
    }

    #[test]
    fn normalize_decoded_image_no_rotate_preserves_orientation_metadata() {
        let image = decoded_image_fixture();
        let normalized = normalize_decoded_image(image.clone(), true, "heif").unwrap();

        assert_eq!(normalized.width(), image.width());
        assert_eq!(normalized.height(), image.height());
        assert_eq!(normalized.pixels(), image.pixels());
        assert_eq!(normalized.metadata().orientation, Some(6));
        assert_eq!(normalized.metadata().exif, image.metadata().exif);
    }

    #[test]
    fn normalize_decoded_image_infers_orientation_from_exif() {
        let image = Image::<U8>::from_buffer(2, 1, 3, vec![1_u8, 2, 3, 4, 5, 6])
            .unwrap()
            .with_metadata(ImageMetadata {
                exif: Some(exif_blob(2)),
                orientation: None,
                ..ImageMetadata::default()
            });

        let normalized = normalize_decoded_image(image, false, "heif").unwrap();
        assert_eq!(normalized.metadata().orientation, Some(1));
        assert_eq!(normalized.pixels(), &[4_u8, 5, 6, 1, 2, 3]);
    }

    #[test]
    fn unsupported_parameter_subcode_is_detected() {
        assert!(is_unsupported_parameter(libheif_sys::heif_error {
            code: libheif_sys::heif_error_code_heif_error_Usage_error,
            subcode: libheif_sys::heif_suberror_code_heif_suberror_Unsupported_parameter,
            message: ptr::null(),
        }));
        assert!(!is_unsupported_parameter(libheif_sys::heif_error {
            code: libheif_sys::heif_error_code_heif_error_Usage_error,
            subcode: libheif_sys::heif_suberror_code_heif_suberror_Invalid_parameter_value,
            message: ptr::null(),
        }));
    }

    #[test]
    fn available_threads_reports_positive_value() {
        assert!(available_threads() >= 1);
    }

    #[test]
    fn encode_interleaved_rejects_invalid_input_before_encoding() {
        let wrong_len = encode_interleaved(
            "heif",
            CompressionFormat::Hevc,
            2,
            1,
            3,
            8,
            &[1_u8, 2, 3],
            false,
            50,
            None,
            HeifSubsampling::Auto,
            HeifWriteMetadata::default(),
        )
        .unwrap_err();
        assert!(wrong_len.to_string().contains("expected 6 input bytes"));

        let unsupported_bands = encode_interleaved(
            "heif",
            CompressionFormat::Hevc,
            1,
            1,
            2,
            8,
            &[1_u8, 2],
            false,
            50,
            None,
            HeifSubsampling::Auto,
            HeifWriteMetadata::default(),
        )
        .unwrap_err();
        assert!(
            unsupported_bands
                .to_string()
                .contains("unsupported band count 2")
        );
    }

    #[test]
    fn orientation_helper_covers_remaining_flip_branches() {
        let pixels = vec![1_u8, 2, 3, 4];
        let cases = [
            (2, vec![2_u8, 1, 4, 3]),
            (3, vec![4_u8, 3, 2, 1]),
            (4, vec![3_u8, 4, 1, 2]),
            (5, vec![1_u8, 3, 2, 4]),
            (7, vec![4_u8, 2, 3, 1]),
            (8, vec![2_u8, 4, 1, 3]),
        ];

        for (orientation, expected) in cases {
            let (_, _, oriented) =
                apply_orientation_to_pixels(pixels.clone(), 2, 2, 1, orientation, "heif").unwrap();
            assert_eq!(oriented, expected);
        }
    }

    #[test]
    fn normalize_decoded_image_rotates_and_scrubs_exif_for_heif() {
        let image = decoded_image_fixture();
        let normalized = normalize_decoded_image(image.clone(), false, "heif").unwrap();
        let (expected_width, expected_height, expected_pixels) = orient_pixels(
            image.pixels(),
            image.width(),
            image.height(),
            image.bands(),
            6,
        );

        assert_eq!(normalized.width(), expected_width);
        assert_eq!(normalized.height(), expected_height);
        assert_eq!(normalized.pixels(), expected_pixels.as_slice());
        assert_eq!(normalized.metadata().orientation, Some(1));
        assert_eq!(
            exif_orientation(normalized.metadata().exif.as_deref().unwrap_or(&[])),
            None
        );
        assert_eq!(normalized.metadata().xmp, image.metadata().xmp);
    }

    #[test]
    fn normalize_decoded_image_rotates_and_scrubs_exif_for_avif() {
        let image = decoded_image_fixture();
        let normalized = normalize_decoded_image(image.clone(), false, "avif").unwrap();
        let (expected_width, expected_height, expected_pixels) = orient_pixels(
            image.pixels(),
            image.width(),
            image.height(),
            image.bands(),
            6,
        );

        assert_eq!(normalized.width(), expected_width);
        assert_eq!(normalized.height(), expected_height);
        assert_eq!(normalized.pixels(), expected_pixels.as_slice());
        assert_eq!(normalized.metadata().orientation, Some(1));
        assert_eq!(
            exif_orientation(normalized.metadata().exif.as_deref().unwrap_or(&[])),
            None
        );
        assert_eq!(normalized.metadata().xmp, image.metadata().xmp);
    }
}
