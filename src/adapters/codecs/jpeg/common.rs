use std::ffi::CStr;

use turbojpeg::{Colorspace, PixelFormat, Subsamp, raw};

use super::super::shrink_on_load::{ShrinkOnLoadBackend, ShrinkOnLoadPlan};
use crate::domain::codec_options::JpegSubsampling;
use crate::domain::error::ViprsError;
use crate::domain::format::U8;
use crate::domain::image::{Image, ImageMetadata, Interpretation};

pub(super) const ICC_PROFILE_SIGNATURE: &[u8] = b"ICC_PROFILE\0";
pub(super) const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";
const XMP_SIGNATURE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const JPEG_SHRINK_FACTORS: [u8; 4] = [1, 2, 4, 8];
const MAX_APP_SEGMENT_PAYLOAD: usize = u16::MAX as usize - 2;
const MAX_ICC_CHUNK_PAYLOAD: usize = MAX_APP_SEGMENT_PAYLOAD - ICC_PROFILE_SIGNATURE.len() - 2;
pub(super) const MAX_JPEG_DECODED_IMAGE_BYTES: usize = 1 << 30;
const MAX_JPEG_ICC_PROFILE_BYTES: usize = 1 << 20;
const MAX_JPEG_EXIF_FIELD_BYTES: usize = 256 * 1024;
pub(super) const JPEG_STREAM_FLUSH_BYTES: usize = 16 * 1024;
const JPEG_SHRINK_BACKEND: ShrinkOnLoadBackend = ShrinkOnLoadBackend::JpegTurboScaledIdct;

pub(super) struct TurboJpegHandle {
    pub(super) ptr: raw::tjhandle,
}

impl TurboJpegHandle {
    pub(super) fn new(init: raw::TJINIT) -> Result<Self, ViprsError> {
        // SAFETY: `tj3Init` is called with a valid TurboJPEG init constant and returns
        // either a handle owned by the caller or null on failure.
        let ptr = unsafe { raw::tj3Init(init as i32) };
        if ptr.is_null() {
            return Err(turbojpeg_error(ptr, "jpeg"));
        }
        Ok(Self { ptr })
    }

    pub(super) fn set(&mut self, param: raw::TJPARAM, value: i32) -> Result<(), ViprsError> {
        // SAFETY: `self.ptr` is a live handle created by `tj3Init`; `param` and `value`
        // are forwarded verbatim to TurboJPEG.
        let result = unsafe { raw::tj3Set(self.ptr, param as i32, value) };
        if result != 0 {
            return Err(turbojpeg_error(self.ptr, "jpeg"));
        }
        Ok(())
    }

    pub(super) fn get(&mut self, param: raw::TJPARAM) -> i32 {
        // SAFETY: `self.ptr` is a live TurboJPEG handle and `param` is a valid query id.
        unsafe { raw::tj3Get(self.ptr, param as i32) }
    }

    pub(super) fn set_scaling_factor(
        &mut self,
        factor: raw::tjscalingfactor,
    ) -> Result<(), ViprsError> {
        // SAFETY: `self.ptr` is a live TurboJPEG handle and `factor` is one of the
        // DCT-domain scale ratios accepted by libjpeg-turbo.
        let result = unsafe { raw::tj3SetScalingFactor(self.ptr, factor) };
        if result != 0 {
            return Err(turbojpeg_error(self.ptr, "jpeg"));
        }
        Ok(())
    }
}

impl Drop for TurboJpegHandle {
    fn drop(&mut self) {
        // SAFETY: destroying a null handle is explicitly allowed; otherwise the handle
        // was created by `tj3Init` and has not been transferred elsewhere.
        unsafe { raw::tj3Destroy(self.ptr) };
    }
}

pub(super) fn turbojpeg_error(handle: raw::tjhandle, codec_name: &str) -> ViprsError {
    // SAFETY: TurboJPEG returns a valid null-terminated static error string for any
    // handle value, including null on global init failures.
    let message = unsafe { CStr::from_ptr(raw::tj3GetErrorStr(handle)) }
        .to_string_lossy()
        .into_owned();
    ViprsError::Codec(format!("{codec_name}: {message}"))
}

fn colorspace_to_decode_shape(
    colorspace: Colorspace,
) -> (PixelFormat, u32, Option<Interpretation>) {
    match colorspace {
        Colorspace::Gray => (PixelFormat::GRAY, 1, Some(Interpretation::BW)),
        Colorspace::CMYK | Colorspace::YCCK => (PixelFormat::CMYK, 4, Some(Interpretation::Cmyk)),
        Colorspace::RGB | Colorspace::YCbCr => (PixelFormat::RGB, 3, Some(Interpretation::Srgb)),
    }
}

fn decode_shape_from_handle(
    handle: &mut TurboJpegHandle,
) -> Result<(PixelFormat, u32, Option<Interpretation>), ViprsError> {
    let colorspace = match handle.get(raw::TJPARAM_TJPARAM_COLORSPACE) as u32 {
        raw::TJCS_TJCS_GRAY => Colorspace::Gray,
        raw::TJCS_TJCS_CMYK => Colorspace::CMYK,
        raw::TJCS_TJCS_YCCK => Colorspace::YCCK,
        raw::TJCS_TJCS_RGB => Colorspace::RGB,
        raw::TJCS_TJCS_YCbCr => Colorspace::YCbCr,
        other => {
            return Err(ViprsError::Codec(format!(
                "jpeg: unsupported TurboJPEG colorspace {other}"
            )));
        }
    };
    Ok(colorspace_to_decode_shape(colorspace))
}

pub(super) fn subsampling_to_sampling_factor(subsampling: JpegSubsampling, quality: u8) -> Subsamp {
    match subsampling {
        JpegSubsampling::Auto => {
            if quality < 90 {
                Subsamp::Sub2x2
            } else {
                Subsamp::None
            }
        }
        JpegSubsampling::Off => Subsamp::None,
        JpegSubsampling::Subsample420 => Subsamp::Sub2x2,
        JpegSubsampling::Subsample422 => Subsamp::Sub2x1,
        JpegSubsampling::Subsample440 => Subsamp::Sub1x2,
    }
}

#[inline]
pub(super) fn turbojpeg_quality(quality: u8) -> Result<i32, ViprsError> {
    if quality > 100 {
        return Err(ViprsError::Codec(
            "jpeg: quality must be in range 0..=100".into(),
        ));
    }

    // TurboJPEG rejects 0, but viprs follows libvips' public contract where
    // 0 means "lowest quality / maximum compression".
    Ok(i32::from(quality.max(1)))
}

pub(super) fn raw_scaling_factor_for_shrink(shrink_factor: u8) -> raw::tjscalingfactor {
    raw::tjscalingfactor {
        num: 1,
        denom: i32::from(shrink_factor),
    }
}

#[derive(Copy, Clone)]
enum ExifEndian {
    Little,
    Big,
}

pub(super) fn visit_jpeg_segments(src: &[u8], mut visitor: impl FnMut(u8, &[u8]) -> bool) {
    if src.len() < 4 || src[0] != 0xFF || src[1] != 0xD8 {
        return;
    }

    let mut offset = 2usize;
    while offset + 1 < src.len() {
        while offset < src.len() && src[offset] != 0xFF {
            offset += 1;
        }
        while offset < src.len() && src[offset] == 0xFF {
            offset += 1;
        }
        if offset >= src.len() {
            break;
        }

        let marker = src[offset];
        offset += 1;

        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        if matches!(marker, 0x01 | 0xD0..=0xD7) {
            continue;
        }
        if offset + 2 > src.len() {
            break;
        }

        let segment_len = u16::from_be_bytes([src[offset], src[offset + 1]]) as usize;
        if segment_len < 2 || offset + segment_len > src.len() {
            break;
        }

        let payload = &src[offset + 2..offset + segment_len];
        if !visitor(marker, payload) {
            break;
        }

        offset += segment_len;
    }
}

fn is_sof_marker(marker: u8) -> bool {
    matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF)
}

fn validate_jpeg_structure(src: &[u8]) -> Result<(), ViprsError> {
    let mut sof_count = 0usize;
    let mut error = None;

    visit_jpeg_segments(src, |marker, payload| {
        if is_sof_marker(marker) {
            sof_count += 1;
            if sof_count > 1 {
                error = Some("jpeg: multiple SOF markers".to_string());
                return false;
            }
            if payload.len() < 6 {
                error = Some("jpeg: truncated SOF segment".to_string());
                return false;
            }
        }
        true
    });

    if let Some(message) = error {
        return Err(ViprsError::Codec(message));
    }
    if sof_count == 0 {
        return Err(ViprsError::Codec("jpeg: missing SOF marker".into()));
    }

    Ok(())
}

pub(super) struct JpegPreflight {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bands: u32,
    pub(super) pixel_format: PixelFormat,
    pub(super) interpretation: Option<Interpretation>,
    pub(super) exif: Option<Vec<u8>>,
    pub(super) icc_profile: Option<Vec<u8>>,
    pub(super) xmp: Option<Vec<u8>>,
    pub(super) orientation: Option<u8>,
}

pub(super) fn preflight_jpeg(src: &[u8]) -> Result<JpegPreflight, ViprsError> {
    validate_jpeg_structure(src)?;
    let exif = extract_exif(src);
    if let Some(exif_payload) = exif.as_deref() {
        validate_exif_payload(exif_payload)?;
    }
    let icc_profile = extract_icc_profile(src)?;
    let xmp = extract_xmp(src);
    let orientation = exif.as_deref().and_then(extract_exif_orientation);

    let mut dec = TurboJpegHandle::new(raw::TJINIT_TJINIT_DECOMPRESS)?;
    let src_len = src
        .len()
        .try_into()
        .map_err(|_| ViprsError::Codec("jpeg: source length overflow".into()))?;
    // SAFETY: `dec` owns a valid decompressor handle; `src` remains alive for the
    // duration of the call; TurboJPEG writes only to decoder state.
    let header_result = unsafe { raw::tj3DecompressHeader(dec.ptr, src.as_ptr(), src_len) };
    if header_result != 0 {
        return Err(turbojpeg_error(dec.ptr, "jpeg"));
    }

    let (pixel_format, bands, interpretation) = decode_shape_from_handle(&mut dec)?;
    let width = u32::try_from(dec.get(raw::TJPARAM_TJPARAM_JPEGWIDTH))
        .map_err(|_| ViprsError::Codec("jpeg: header width overflow".into()))?;
    let height = u32::try_from(dec.get(raw::TJPARAM_TJPARAM_JPEGHEIGHT))
        .map_err(|_| ViprsError::Codec("jpeg: header height overflow".into()))?;
    if width == 0 || height == 0 {
        return Err(ViprsError::Codec("jpeg: zero-sized image".into()));
    }

    Ok(JpegPreflight {
        width,
        height,
        bands,
        pixel_format,
        interpretation,
        exif,
        icc_profile,
        xmp,
        orientation,
    })
}

fn extract_icc_profile(src: &[u8]) -> Result<Option<Vec<u8>>, ViprsError> {
    let mut expected_chunks: Option<usize> = None;
    let mut chunks: Vec<Option<Vec<u8>>> = Vec::new();
    let mut total_len = 0usize;
    let mut error = None;

    visit_jpeg_segments(src, |marker, payload| {
        if marker == 0xE2 && payload.starts_with(ICC_PROFILE_SIGNATURE) {
            if payload.len() < ICC_PROFILE_SIGNATURE.len() + 2 {
                error = Some("jpeg: truncated ICC profile chunk header".to_string());
                return false;
            }

            let sequence_number = payload[ICC_PROFILE_SIGNATURE.len()] as usize;
            let chunk_count = payload[ICC_PROFILE_SIGNATURE.len() + 1] as usize;

            if sequence_number == 0 || chunk_count == 0 {
                error = Some("jpeg: invalid ICC profile chunk numbering".to_string());
                return false;
            }

            if let Some(expected) = expected_chunks {
                if expected != chunk_count {
                    error = Some("jpeg: ICC profile chunk count mismatch".to_string());
                    return false;
                }
            } else {
                expected_chunks = Some(chunk_count);
                chunks.resize_with(chunk_count, || None);
            }

            if sequence_number > chunk_count {
                error = Some("jpeg: ICC profile chunk sequence out of range".to_string());
                return false;
            }

            let chunk_index = sequence_number - 1;
            if chunks[chunk_index].is_some() {
                error = Some("jpeg: duplicate ICC profile chunk".to_string());
                return false;
            }

            let chunk_len = payload[ICC_PROFILE_SIGNATURE.len() + 2..].len();
            total_len = match total_len.checked_add(chunk_len) {
                Some(total) => total,
                None => {
                    error = Some("jpeg: ICC profile length overflow".to_string());
                    return false;
                }
            };
            if total_len > MAX_JPEG_ICC_PROFILE_BYTES {
                error = Some(format!(
                    "jpeg: ICC profile exceeds safety limit {MAX_JPEG_ICC_PROFILE_BYTES} bytes"
                ));
                return false;
            }

            chunks[chunk_index] = Some(payload[ICC_PROFILE_SIGNATURE.len() + 2..].to_vec());
        }
        true
    });

    if let Some(message) = error {
        return Err(ViprsError::Codec(message));
    }
    if chunks.is_empty() {
        return Ok(None);
    }
    if chunks.iter().any(Option::is_none) {
        return Err(ViprsError::Codec(
            "jpeg: incomplete ICC profile chunk sequence".into(),
        ));
    }

    let mut profile = Vec::with_capacity(total_len);
    for chunk in chunks.into_iter().flatten() {
        profile.extend_from_slice(&chunk);
    }
    Ok(Some(profile))
}

fn extract_exif(src: &[u8]) -> Option<Vec<u8>> {
    let mut exif = None;
    visit_jpeg_segments(src, |marker, payload| {
        if marker == 0xE1 && payload.starts_with(EXIF_SIGNATURE) {
            exif = Some(payload.to_vec());
            return false;
        }
        true
    });
    exif
}

fn extract_xmp(src: &[u8]) -> Option<Vec<u8>> {
    let mut xmp = None;
    visit_jpeg_segments(src, |marker, payload| {
        if marker == 0xE1 && payload.starts_with(XMP_SIGNATURE) {
            xmp = Some(payload[XMP_SIGNATURE.len()..].to_vec());
            return false;
        }
        true
    });
    xmp
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

fn exif_field_type_size(field_type: u16) -> Option<usize> {
    match field_type {
        1 | 2 | 6 | 7 => Some(1),
        3 | 8 => Some(2),
        4 | 9 | 11 => Some(4),
        5 | 10 | 12 => Some(8),
        _ => None,
    }
}

/// `extract_exif_orientation` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```rust
/// let _ = viprs::adapters::codecs::jpeg::extract_exif_orientation;
/// ```
pub(crate) fn extract_exif_orientation(exif: &[u8]) -> Option<u8> {
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
        let tag = read_u16(&entry[0..2], endian)?;
        if tag != 0x0112 {
            continue;
        }

        let field_type = read_u16(&entry[2..4], endian)?;
        let component_count = read_u32(&entry[4..8], endian)?;
        if field_type != 3 || component_count != 1 {
            return None;
        }

        let value = read_u16(&entry[8..10], endian)?;
        return (1..=8).contains(&value).then_some(value as u8);
    }

    None
}

fn validate_exif_payload(exif: &[u8]) -> Result<(), ViprsError> {
    let tiff = exif
        .strip_prefix(EXIF_SIGNATURE)
        .ok_or_else(|| ViprsError::Codec("jpeg: malformed EXIF payload".into()))?;
    if tiff.len() < 8 {
        return Err(ViprsError::Codec("jpeg: truncated EXIF TIFF header".into()));
    }

    let endian = match &tiff[..2] {
        b"II" => ExifEndian::Little,
        b"MM" => ExifEndian::Big,
        _ => {
            return Err(ViprsError::Codec(
                "jpeg: invalid EXIF byte order marker".into(),
            ));
        }
    };

    if read_u16(&tiff[2..4], endian) != Some(42) {
        return Err(ViprsError::Codec("jpeg: invalid EXIF TIFF magic".into()));
    }

    let ifd0_offset = read_u32(&tiff[4..8], endian)
        .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF IFD0 offset".into()))?
        as usize;
    let entry_count = read_u16(
        tiff.get(ifd0_offset..ifd0_offset + 2)
            .ok_or_else(|| ViprsError::Codec("jpeg: EXIF IFD0 offset out of bounds".into()))?,
        endian,
    )
    .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF IFD0 entry count".into()))?
        as usize;
    let entries_start = ifd0_offset + 2;
    let entries_len = entry_count
        .checked_mul(12)
        .ok_or_else(|| ViprsError::Codec("jpeg: EXIF entry table overflow".into()))?;
    let entries_end = entries_start
        .checked_add(entries_len)
        .ok_or_else(|| ViprsError::Codec("jpeg: EXIF entry table overflow".into()))?;
    if entries_end > tiff.len() {
        return Err(ViprsError::Codec("jpeg: truncated EXIF entry table".into()));
    }

    for entry in tiff[entries_start..entries_end].chunks_exact(12) {
        let tag = read_u16(&entry[0..2], endian)
            .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF tag".into()))?;
        let field_type = read_u16(&entry[2..4], endian)
            .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF field type".into()))?;
        let component_count = read_u32(&entry[4..8], endian)
            .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF field count".into()))?
            as usize;
        let bytes_per_component = exif_field_type_size(field_type)
            .ok_or_else(|| ViprsError::Codec("jpeg: unsupported EXIF field type".into()))?;
        let value_len = component_count
            .checked_mul(bytes_per_component)
            .ok_or_else(|| ViprsError::Codec("jpeg: EXIF field size overflow".into()))?;
        if value_len >= MAX_JPEG_EXIF_FIELD_BYTES {
            return Err(ViprsError::Codec(format!(
                "jpeg: EXIF field exceeds safety limit {MAX_JPEG_EXIF_FIELD_BYTES} bytes"
            )));
        }
        if value_len > 4 {
            let value_offset = read_u32(&entry[8..12], endian)
                .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF value offset".into()))?
                as usize;
            let value_end = value_offset
                .checked_add(value_len)
                .ok_or_else(|| ViprsError::Codec("jpeg: EXIF value range overflow".into()))?;
            if value_end > tiff.len() {
                return Err(ViprsError::Codec(
                    "jpeg: EXIF value extends past payload bounds".into(),
                ));
            }
        }

        if tag != 0x0112 {
            continue;
        }

        if field_type != 3 || component_count != 1 {
            return Err(ViprsError::Codec(
                "jpeg: malformed EXIF orientation entry".into(),
            ));
        }

        let value = read_u16(&entry[8..10], endian)
            .ok_or_else(|| ViprsError::Codec("jpeg: truncated EXIF orientation value".into()))?;
        if !(1..=8).contains(&value) {
            return Err(ViprsError::Codec(
                "jpeg: invalid EXIF orientation value".into(),
            ));
        }
    }

    Ok(())
}

pub(super) fn checked_decoded_image_len(
    codec_name: &str,
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: usize,
    max_bytes: usize,
) -> Result<usize, ViprsError> {
    let decoded_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample))
        .ok_or_else(|| ViprsError::Codec(format!("{codec_name}: decoded dimensions overflow")))?;
    if decoded_len > max_bytes {
        return Err(ViprsError::Codec(format!(
            "{codec_name}: decoded image requires {decoded_len} bytes, exceeds safety limit {max_bytes}"
        )));
    }
    Ok(decoded_len)
}

pub(super) fn shrink_dimension_for_factor(dimension: u32, factor: u8) -> u32 {
    if factor == 1 {
        dimension
    } else {
        (dimension / u32::from(factor)).max(1)
    }
}

pub(super) fn shrink_factor_for_max_dimension(width: u32, height: u32, max_dimension: u32) -> u8 {
    for factor in JPEG_SHRINK_FACTORS {
        if shrink_dimension_for_factor(width, factor) <= max_dimension
            && shrink_dimension_for_factor(height, factor) <= max_dimension
        {
            return factor;
        }
    }

    8
}

pub(super) fn jpeg_shrink_on_load_plan(requested_factor: u8) -> ShrinkOnLoadPlan {
    ShrinkOnLoadPlan::new(requested_factor, JPEG_SHRINK_BACKEND)
}

pub(super) fn normalize_exif_app1_payload(exif: &[u8]) -> Vec<u8> {
    if exif.starts_with(EXIF_SIGNATURE) {
        exif.to_vec()
    } else {
        let mut payload = Vec::with_capacity(EXIF_SIGNATURE.len() + exif.len());
        payload.extend_from_slice(EXIF_SIGNATURE);
        payload.extend_from_slice(exif);
        payload
    }
}

pub(super) fn normalize_xmp_app1_payload(xmp: &[u8]) -> Vec<u8> {
    if xmp.starts_with(XMP_SIGNATURE) {
        xmp.to_vec()
    } else {
        let mut payload = Vec::with_capacity(XMP_SIGNATURE.len() + xmp.len());
        payload.extend_from_slice(XMP_SIGNATURE);
        payload.extend_from_slice(xmp);
        payload
    }
}

pub(super) fn insert_segment_after_soi(
    jpeg: &mut Vec<u8>,
    marker: u8,
    payload: &[u8],
) -> Result<(), ViprsError> {
    if payload.len() > MAX_APP_SEGMENT_PAYLOAD {
        return Err(ViprsError::Codec(format!(
            "jpeg: APP{:X} payload too large: {} bytes",
            marker,
            payload.len()
        )));
    }

    let segment_len = u16::try_from(payload.len() + 2)
        .map_err(|_| ViprsError::Codec("jpeg: APP segment length overflow".into()))?;
    let mut segment = Vec::with_capacity(payload.len() + 4);
    segment.extend_from_slice(&[0xFF, marker]);
    segment.extend_from_slice(&segment_len.to_be_bytes());
    segment.extend_from_slice(payload);
    jpeg.splice(2..2, segment);
    Ok(())
}

fn insert_icc_profile_segments(jpeg: &mut Vec<u8>, profile: &[u8]) -> Result<(), ViprsError> {
    if profile.is_empty() {
        return Ok(());
    }

    let chunk_count = profile.len().div_ceil(MAX_ICC_CHUNK_PAYLOAD);
    let chunk_count_u8 = u8::try_from(chunk_count)
        .map_err(|_| ViprsError::Codec("jpeg: ICC profile needs too many APP2 chunks".into()))?;
    for (index, chunk) in profile.chunks(MAX_ICC_CHUNK_PAYLOAD).enumerate().rev() {
        let sequence_number = u8::try_from(index + 1)
            .map_err(|_| ViprsError::Codec("jpeg: ICC profile chunk index overflow".into()))?;
        let mut payload = Vec::with_capacity(ICC_PROFILE_SIGNATURE.len() + 2 + chunk.len());
        payload.extend_from_slice(ICC_PROFILE_SIGNATURE);
        payload.push(sequence_number);
        payload.push(chunk_count_u8);
        payload.extend_from_slice(chunk);
        insert_segment_after_soi(jpeg, 0xE2, &payload)?;
    }
    Ok(())
}

pub(super) fn insert_metadata_segments(
    jpeg: &mut Vec<u8>,
    image: &ImageMetadata,
) -> Result<(), ViprsError> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err(ViprsError::Codec(
            "jpeg: encoder emitted invalid SOI marker".into(),
        ));
    }

    if let Some(xmp) = image.xmp.as_deref() {
        insert_segment_after_soi(jpeg, 0xE1, &normalize_xmp_app1_payload(xmp))?;
    }
    if let Some(exif) = image.exif.as_deref() {
        insert_segment_after_soi(jpeg, 0xE1, &normalize_exif_app1_payload(exif))?;
    }
    if let Some(icc_profile) = image.icc_profile.as_deref() {
        insert_icc_profile_segments(jpeg, icc_profile)?;
    }

    Ok(())
}

pub(super) fn crop_strict_shrink_edges(
    pixels: Vec<u8>,
    decoded_width: u32,
    decoded_height: u32,
    target_width: u32,
    target_height: u32,
    bands: u32,
) -> Result<Vec<u8>, ViprsError> {
    if decoded_width == target_width && decoded_height == target_height {
        return Ok(pixels);
    }
    if decoded_width < target_width || decoded_height < target_height {
        return Err(ViprsError::Codec(format!(
            "jpeg: native shrink produced {decoded_width}x{decoded_height}, smaller than requested {target_width}x{target_height}"
        )));
    }

    let decoded_width = decoded_width as usize;
    let decoded_height = decoded_height as usize;
    let target_width = target_width as usize;
    let target_height = target_height as usize;
    let bands = bands as usize;
    let expected_len = decoded_width
        .checked_mul(decoded_height)
        .and_then(|pixel_count| pixel_count.checked_mul(bands))
        .ok_or_else(|| ViprsError::Codec("jpeg: decoded dimensions overflow".into()))?;

    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "jpeg: decoded buffer length mismatch (got {}, expected {expected_len})",
            pixels.len()
        )));
    }

    let row_bytes = target_width * bands;
    let decoded_row_bytes = decoded_width * bands;
    let mut cropped = vec![0u8; target_height * row_bytes];
    for y in 0..target_height {
        let src = y * decoded_row_bytes;
        let dst = y * row_bytes;
        cropped[dst..dst + row_bytes].copy_from_slice(&pixels[src..src + row_bytes]);
    }

    Ok(cropped)
}

/// `apply_exif_orientation` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```rust
/// let _ = viprs::adapters::codecs::jpeg::apply_exif_orientation;
/// ```
pub(crate) fn apply_exif_orientation(
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    orientation: u8,
    codec_name: &str,
) -> Result<(u32, u32, Vec<u8>), ViprsError> {
    if !(2..=8).contains(&orientation) || width == 0 || height == 0 || bands == 0 {
        return Ok((width, height, pixels));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let bands_usize = bands as usize;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .and_then(|pixel_count| pixel_count.checked_mul(bands_usize))
        .ok_or_else(|| ViprsError::Codec(format!("{codec_name}: decoded dimensions overflow")))?;

    if pixels.len() != expected_len {
        return Err(ViprsError::Codec(format!(
            "{codec_name}: decoded buffer length mismatch (got {}, expected {expected_len})",
            pixels.len()
        )));
    }

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

    Ok((out_width, out_height, out))
}

/// `orient_u8_image` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```rust
/// let _ = viprs::adapters::codecs::jpeg::orient_u8_image;
/// ```
pub(crate) fn orient_u8_image(
    image: &Image<U8>,
    orientation: u8,
    codec_name: &str,
) -> Result<Image<U8>, ViprsError> {
    if !(2..=8).contains(&orientation) {
        return Ok(image.clone());
    }

    let (width, height, pixels) = apply_exif_orientation(
        image.pixels().to_vec(),
        image.width(),
        image.height(),
        image.bands(),
        orientation,
        codec_name,
    )?;
    let mut metadata = image.metadata().clone();
    metadata.orientation = Some(1);

    Image::from_buffer(width, height, image.bands(), pixels)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(format!("{codec_name}: {e}")))
}
