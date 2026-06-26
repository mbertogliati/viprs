//! Uhdr adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "uhdr")]

//! Ultra HDR (JPEGR) decode adapter.
//!
//! Decodes the SDR base JPEG and, when present, extracts the secondary gain map
//! JPEG plus the Adobe/Android gain map metadata needed by `uhdr2scrgb`.

use crate::JpegCodec;
use crate::jpeg::{extract_exif_orientation, orient_u8_image};
use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, U8};
use viprs_core::image::{InMemoryImage, UhdrGainMap, UhdrGainMapMetadata};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];
const JPEG_EOI: [u8; 2] = [0xFF, 0xD9];
const MPF_SIGNATURE: &[u8] = b"MPF\0";
const HDRGM_NAMESPACE: &str = "http://ns.adobe.com/hdr-gain-map/1.0/";
// REASON: the lib target does not read APP1 XMP packets directly, but tests and
// future metadata helpers share the canonical signature bytes from this module.
#[allow(dead_code)]
const XMP_SIGNATURE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const DEFAULT_GAINMAP_OFFSET: f32 = 1.0 / 64.0;

/// The `UhdrCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::uhdr::UhdrCodec>();
/// ```
pub struct UhdrCodec;

#[derive(Default)]
struct UhdrMarkers {
    mpf_segments: usize,
    gainmap_segments: usize,
}

#[derive(Clone, Copy, Debug)]
enum TiffEndian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug)]
struct MpfEntry {
    size: u32,
    offset: u32,
}

#[derive(Clone, Copy, Debug)]
struct ParsedGainMapMetadata {
    map: UhdrGainMapMetadata,
    hdr_capacity_min: f32,
    hdr_capacity_max: f32,
    base_rendition_is_hdr: bool,
}

fn visit_jpeg_segments(src: &[u8], mut visitor: impl FnMut(u8, &[u8]) -> bool) {
    if src.len() < 4 || src[..2] != JPEG_SOI {
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

fn scan_markers(src: &[u8]) -> UhdrMarkers {
    let mut markers = UhdrMarkers::default();
    visit_jpeg_segments(src, |marker, payload| {
        if marker == 0xE2 && payload.starts_with(MPF_SIGNATURE) {
            markers.mpf_segments += 1;
        }
        if marker == 0xE1 && payload.windows(7).any(|window| window == b"GainMap") {
            markers.gainmap_segments += 1;
        }

        true
    });
    markers
}

fn read_u16(bytes: &[u8], endian: TiffEndian) -> Option<u16> {
    let pair: [u8; 2] = bytes.get(..2)?.try_into().ok()?;
    Some(match endian {
        TiffEndian::Little => u16::from_le_bytes(pair),
        TiffEndian::Big => u16::from_be_bytes(pair),
    })
}

fn read_u32(bytes: &[u8], endian: TiffEndian) -> Option<u32> {
    let quad: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
    Some(match endian {
        TiffEndian::Little => u32::from_le_bytes(quad),
        TiffEndian::Big => u32::from_be_bytes(quad),
    })
}

fn parse_mpf_entries(payload: &[u8]) -> Result<Vec<MpfEntry>, ViprsError> {
    let tiff = payload
        .strip_prefix(MPF_SIGNATURE)
        .ok_or_else(|| ViprsError::Codec("uhdr: MPF APP2 segment missing MPF signature".into()))?;
    if tiff.len() < 8 {
        return Err(ViprsError::Codec("uhdr: truncated MPF TIFF header".into()));
    }

    let endian = match &tiff[..2] {
        b"II" => TiffEndian::Little,
        b"MM" => TiffEndian::Big,
        _ => {
            return Err(ViprsError::Codec(
                "uhdr: invalid MPF TIFF endianness".into(),
            ));
        }
    };
    if read_u16(&tiff[2..4], endian) != Some(42) {
        return Err(ViprsError::Codec("uhdr: invalid MPF TIFF magic".into()));
    }

    let ifd_offset = read_u32(&tiff[4..8], endian)
        .and_then(|offset| usize::try_from(offset).ok())
        .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF IFD offset".into()))?;
    let entry_count = read_u16(
        tiff.get(ifd_offset..ifd_offset + 2)
            .ok_or_else(|| ViprsError::Codec("uhdr: truncated MPF IFD entry count".into()))?,
        endian,
    )
    .map(usize::from)
    .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF IFD entry count".into()))?;
    let entries_start = ifd_offset + 2;

    let mut image_count = None;
    let mut mp_entries_offset = None;
    let mut mp_entries_len = None;

    for entry_index in 0..entry_count {
        let entry_offset = entries_start + entry_index * 12;
        let entry = tiff
            .get(entry_offset..entry_offset + 12)
            .ok_or_else(|| ViprsError::Codec("uhdr: truncated MPF IFD entry".into()))?;
        let tag = read_u16(&entry[0..2], endian)
            .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF tag".into()))?;
        let count = read_u32(&entry[4..8], endian)
            .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF count".into()))?;
        let value = read_u32(&entry[8..12], endian)
            .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF value".into()))?;

        match tag {
            0xB001 => image_count = Some(value),
            0xB002 => {
                mp_entries_offset = Some(usize::try_from(value).map_err(|_| {
                    ViprsError::Codec("uhdr: MPF entry payload offset overflows usize".into())
                })?);
                mp_entries_len = Some(usize::try_from(count).map_err(|_| {
                    ViprsError::Codec("uhdr: MPF entry payload length overflows usize".into())
                })?);
            }
            _ => {}
        }
    }

    let image_count = usize::try_from(image_count.unwrap_or(0))
        .map_err(|_| ViprsError::Codec("uhdr: MPF image count overflows usize".into()))?;
    let entry_data_offset = mp_entries_offset
        .ok_or_else(|| ViprsError::Codec("uhdr: MPF image entry directory missing".into()))?;
    let entry_data_len = mp_entries_len
        .ok_or_else(|| ViprsError::Codec("uhdr: MPF image entry payload missing".into()))?;
    let entry_data = tiff
        .get(entry_data_offset..entry_data_offset + entry_data_len)
        .ok_or_else(|| ViprsError::Codec("uhdr: truncated MPF image entry payload".into()))?;
    let expected_len = image_count
        .checked_mul(16)
        .ok_or_else(|| ViprsError::Codec("uhdr: MPF image entry table length overflow".into()))?;
    if entry_data.len() < expected_len {
        return Err(ViprsError::Codec(
            "uhdr: MPF image entry payload shorter than declared image count".into(),
        ));
    }

    let mut entries = Vec::with_capacity(image_count);
    for chunk in entry_data[..expected_len].chunks_exact(16) {
        let size = read_u32(&chunk[4..8], endian)
            .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF image size".into()))?;
        let offset = read_u32(&chunk[8..12], endian)
            .ok_or_else(|| ViprsError::Codec("uhdr: invalid MPF image offset".into()))?;
        entries.push(MpfEntry { size, offset });
    }

    Ok(entries)
}

fn extract_mpf_entries(src: &[u8]) -> Result<Option<Vec<MpfEntry>>, ViprsError> {
    let mut parsed = None;
    visit_jpeg_segments(src, |marker, payload| {
        if marker == 0xE2 && payload.starts_with(MPF_SIGNATURE) {
            parsed = Some(parse_mpf_entries(payload));
            false
        } else {
            true
        }
    });

    parsed.transpose()
}

fn extract_item_attribute(tag: &str, semantic: &str, attribute: &str) -> Option<usize> {
    let tag_pos = tag.find(&format!("Item:Semantic=\"{semantic}\""))?;
    let attr_pos = tag[tag_pos..].find(&format!("{attribute}=\""))?;
    let start = tag_pos + attr_pos + attribute.len() + 2;
    let value = &tag[start..tag[start..].find('"')? + start];
    value.trim().parse().ok()
}

fn extract_container_gainmap_length(xmp: &[u8]) -> Option<usize> {
    let xml = String::from_utf8_lossy(xmp);
    for fragment in xml.split("<Container:Item").skip(1) {
        let end = fragment.find("/>").or_else(|| fragment.find('>'))?;
        let tag = &fragment[..end];
        if let Some(length) = extract_item_attribute(tag, "GainMap", "Item:Length") {
            return Some(length);
        }
    }

    None
}

fn extract_gainmap_jpeg(
    src: &[u8],
    entries: &[MpfEntry],
    fallback_length: Option<usize>,
) -> Option<Vec<u8>> {
    if entries.len() < 2 {
        return None;
    }

    let primary_size = usize::try_from(entries[0].size).ok()?;
    let gainmap_size = usize::try_from(entries[1].size)
        .ok()
        .filter(|size| *size > 0)
        .or(fallback_length)?;

    let mut starts = Vec::with_capacity(3);
    starts.push(primary_size);
    if let Ok(offset) = usize::try_from(entries[1].offset) {
        starts.push(offset);
        if let Some(relative) = primary_size.checked_add(offset) {
            starts.push(relative);
        }
    }

    for start in starts {
        let end = start.checked_add(gainmap_size)?;
        let bytes = src.get(start..end)?;
        if bytes.starts_with(&JPEG_SOI) && bytes.ends_with(&JPEG_EOI) {
            return Some(bytes.to_vec());
        }
    }

    None
}

fn extract_attr(xml: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = xml.find(&needle)? + needle.len();
    let end = start + xml[start..].find('"')?;
    Some(xml[start..end].to_string())
}

fn extract_element(xml: &str, name: &str) -> Option<String> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let start = xml.find(&open)? + open.len();
    let end = start + xml[start..].find(&close)?;
    Some(xml[start..end].to_string())
}

fn parse_float_sequence(raw: &str) -> Option<Vec<f32>> {
    let values: Vec<f32> = raw
        .split(|c: char| c == ',' || c.is_ascii_whitespace())
        .filter(|part| !part.is_empty())
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<_, _>>()
        .ok()?;
    (!values.is_empty()).then_some(values)
}

fn parse_float_array_from_xml(xml: &str, name: &str) -> Option<Vec<f32>> {
    if let Some(attr) = extract_attr(xml, name)
        && let Some(values) = parse_float_sequence(&attr)
    {
        return Some(values);
    }

    let element = extract_element(xml, name)?;
    if element.contains("<rdf:li") {
        let mut values = Vec::new();
        let mut rest = element.as_str();
        while let Some(open_pos) = rest.find("<rdf:li") {
            rest = &rest[open_pos..];
            let start = rest.find('>')? + 1;
            let after_start = &rest[start..];
            let end = after_start.find("</rdf:li>")?;
            let text = after_start[..end].trim();
            if !text.is_empty() {
                values.push(text.parse::<f32>().ok()?);
            }
            rest = &after_start[end + "</rdf:li>".len()..];
        }
        return (!values.is_empty()).then_some(values);
    }

    parse_float_sequence(element.trim())
}

fn parse_bool_attr(xml: &str, name: &str) -> Option<bool> {
    extract_attr(xml, name).and_then(|value| match value.trim() {
        "True" | "true" | "1" => Some(true),
        "False" | "false" | "0" => Some(false),
        _ => None,
    })
}

fn expand_channels(values: Option<Vec<f32>>, default: [f32; 3]) -> Result<[f32; 3], ViprsError> {
    values.map_or(Ok(default), |values| match values.as_slice() {
        [value] => Ok([*value; 3]),
        [red, green, blue] => Ok([*red, *green, *blue]),
        _ => Err(ViprsError::Codec(
            "uhdr: gain map XMP arrays must contain 1 or 3 values".into(),
        )),
    })
}

fn parse_gainmap_xmp(xmp: &[u8]) -> Result<Option<ParsedGainMapMetadata>, ViprsError> {
    let xml = String::from_utf8_lossy(xmp);
    if !xml.contains(HDRGM_NAMESPACE) {
        return Ok(None);
    }

    let gain_map_min = expand_channels(
        parse_float_array_from_xml(&xml, "hdrgm:GainMapMin"),
        [0.0; 3],
    )?;
    let gain_map_max = expand_channels(
        Some(
            parse_float_array_from_xml(&xml, "hdrgm:GainMapMax").ok_or_else(|| {
                ViprsError::Codec("uhdr: gain map XMP missing hdrgm:GainMapMax".into())
            })?,
        ),
        [0.0; 3],
    )?;
    let gamma = expand_channels(parse_float_array_from_xml(&xml, "hdrgm:Gamma"), [1.0; 3])?;
    let offset_sdr = expand_channels(
        parse_float_array_from_xml(&xml, "hdrgm:OffsetSDR"),
        [DEFAULT_GAINMAP_OFFSET; 3],
    )?;
    let offset_hdr = expand_channels(
        parse_float_array_from_xml(&xml, "hdrgm:OffsetHDR"),
        [DEFAULT_GAINMAP_OFFSET; 3],
    )?;
    let hdr_capacity_min = parse_float_array_from_xml(&xml, "hdrgm:HDRCapacityMin")
        .and_then(|values| values.first().copied())
        .unwrap_or(0.0);
    let hdr_capacity_max = parse_float_array_from_xml(&xml, "hdrgm:HDRCapacityMax")
        .and_then(|values| values.first().copied())
        .ok_or_else(|| {
            ViprsError::Codec("uhdr: gain map XMP missing hdrgm:HDRCapacityMax".into())
        })?;
    let base_rendition_is_hdr = parse_bool_attr(&xml, "hdrgm:BaseRenditionIsHDR").unwrap_or(false);

    Ok(Some(ParsedGainMapMetadata {
        map: UhdrGainMapMetadata {
            gamma,
            min_content_boost: gain_map_min.map(f32::exp2),
            max_content_boost: gain_map_max.map(f32::exp2),
            offset_hdr,
            offset_sdr,
        },
        hdr_capacity_min,
        hdr_capacity_max,
        base_rendition_is_hdr,
    }))
}

fn parse_gainmap_metadata(image: &InMemoryImage<U8>) -> Result<Option<ParsedGainMapMetadata>, ViprsError> {
    if let Some(xmp) = image.metadata().xmp.as_deref()
        && let Some(parsed) = parse_gainmap_xmp(xmp)?
    {
        return Ok(Some(parsed));
    }

    // Ultra HDR defines normative gainmap metadata in XMP `hdrgm:*`; EXIF-only
    // samples need vendor tag reverse-engineering before viprs can support them.
    Ok(None)
}

fn decode_gainmap(
    src: &[u8],
    opts: &LoadOptions,
    container_xmp: Option<&[u8]>,
    primary_orientation: Option<u8>,
) -> Result<Option<UhdrGainMap>, ViprsError> {
    let Some(entries) = extract_mpf_entries(src)? else {
        return Ok(None);
    };
    let fallback_length = container_xmp.and_then(extract_container_gainmap_length);
    let Some(gainmap_bytes) = extract_gainmap_jpeg(src, &entries, fallback_length) else {
        return Ok(None);
    };

    let mut gainmap_opts = opts.clone();
    // The gainmap must ignore its own EXIF orientation. Matching any rotation
    // already applied to the primary image is tracked separately; the Ultra HDR
    // spec requires the stored gainmap orientation to match the primary image.
    gainmap_opts.no_rotate = true;
    let mut gainmap_image = JpegCodec.decode_with_options::<U8>(&gainmap_bytes, &gainmap_opts)?;
    if let Some(orientation) = primary_orientation {
        gainmap_image = orient_u8_image(&gainmap_image, orientation, "uhdr gainmap")?;
    }
    let Some(parsed) = parse_gainmap_metadata(&gainmap_image)? else {
        return Ok(None);
    };

    Ok(Some(UhdrGainMap {
        image: Box::new(gainmap_image),
        metadata: parsed.map,
        hdr_capacity_min: parsed.hdr_capacity_min,
        hdr_capacity_max: parsed.hdr_capacity_max,
        base_rendition_is_hdr: parsed.base_rendition_is_hdr,
    }))
}

impl ImageDecoder for UhdrCodec {
    fn format_name(&self) -> &'static str {
        "uhdr"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        scan_markers(header).mpf_segments > 0
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "uhdr: unsupported format {:?}; only U8 is supported",
                F::ID
            )));
        }

        let markers = scan_markers(src);
        if markers.mpf_segments == 0 {
            return Err(ViprsError::Codec(
                "uhdr: MPF segment not found; not a valid Ultra HDR stream".into(),
            ));
        }

        let decoded = JpegCodec.decode_with_options::<F>(src, opts)?;
        let mut metadata = decoded.metadata().clone();
        let primary_orientation = (!opts.no_rotate)
            .then(|| metadata.exif.as_deref().and_then(extract_exif_orientation))
            .flatten();
        let gainmap = decode_gainmap(src, opts, metadata.xmp.as_deref(), primary_orientation)?;

        metadata.extra.insert("uhdr.base".into(), "jpeg".into());
        metadata
            .extra
            .insert("uhdr.mpf_segments".into(), markers.mpf_segments.to_string());
        metadata.extra.insert(
            "uhdr.gainmap_segments".into(),
            markers.gainmap_segments.to_string(),
        );

        if let Some(gainmap) = gainmap {
            metadata.extra.insert(
                "uhdr.gainmap_width".into(),
                gainmap.image().width().to_string(),
            );
            metadata.extra.insert(
                "uhdr.gainmap_height".into(),
                gainmap.image().height().to_string(),
            );
            metadata.extra.insert(
                "uhdr.gainmap_bands".into(),
                gainmap.image().bands().to_string(),
            );
            metadata.uhdr_gainmap = Some(gainmap);
        }

        Ok(decoded.with_metadata(metadata))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        JpegCodec.probe(src)
    }
}

impl ImageEncoder for UhdrCodec {
    fn format_name(&self) -> &'static str {
        "uhdr"
    }

    fn encode<F: BandFormat>(&self, _image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
        Err(ViprsError::Codec(
            "uhdr: encode is not implemented (decode-only codec)".into(),
        ))
    }

    fn encode_with_options<F: BandFormat>(
      &self,
      _image: &InMemoryImage<F>,
      _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        Err(ViprsError::Codec(
            "uhdr: encode is not implemented (decode-only codec)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::PathBuf};

    use crate::registry::ForeignRegistry;
    use viprs_core::codec_options::SaveOptions;
    use viprs_core::format::U16;

    fn build_jpeg_fixture() -> Vec<u8> {
        let image =
            Image::<U8>::from_buffer(2, 2, 3, vec![0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
                .unwrap();
        JpegCodec.encode(&image).unwrap()
    }

    fn encode_jpeg_quality_100(image: &Image<U8>) -> Vec<u8> {
        JpegCodec
            .encode_with_options(image, &SaveOptions::default().with_quality(100))
            .unwrap()
    }

    fn xmp_app1_payload(xmp: &[u8]) -> Vec<u8> {
        if xmp.starts_with(XMP_SIGNATURE) {
            xmp.to_vec()
        } else {
            let mut payload = Vec::with_capacity(XMP_SIGNATURE.len() + xmp.len());
            payload.extend_from_slice(XMP_SIGNATURE);
            payload.extend_from_slice(xmp);
            payload
        }
    }

    fn with_embedded_xmp(jpeg: &[u8], xmp: &[u8]) -> Vec<u8> {
        let payload = xmp_app1_payload(xmp);
        with_app1_payload(jpeg, &payload)
    }

    fn exif_app1_payload(exif: &[u8]) -> Vec<u8> {
        if exif.starts_with(b"Exif\0\0") {
            exif.to_vec()
        } else {
            let mut payload = Vec::with_capacity(6 + exif.len());
            payload.extend_from_slice(b"Exif\0\0");
            payload.extend_from_slice(exif);
            payload
        }
    }

    fn with_embedded_exif(jpeg: &[u8], exif: &[u8]) -> Vec<u8> {
        let payload = exif_app1_payload(exif);
        with_app1_payload(jpeg, &payload)
    }

    fn with_app1_payload(jpeg: &[u8], payload: &[u8]) -> Vec<u8> {
        let segment_len = u16::try_from(payload.len() + 2).unwrap();
        let mut with_app1 = Vec::with_capacity(jpeg.len() + payload.len() + 4);
        with_app1.extend_from_slice(&jpeg[..2]);
        with_app1.extend_from_slice(&[0xFF, 0xE1]);
        with_app1.extend_from_slice(&segment_len.to_be_bytes());
        with_app1.extend_from_slice(payload);
        with_app1.extend_from_slice(&jpeg[2..]);
        with_app1
    }

    fn exif_orientation_payload(orientation: u16) -> Vec<u8> {
        let mut payload = Vec::with_capacity(32);
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(b"II");
        payload.extend_from_slice(&42u16.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&0x0112u16.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&orientation.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload
    }

    fn proprietary_exif_payload(tag: u16, value: u32) -> Vec<u8> {
        let mut payload = Vec::with_capacity(32);
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(b"II");
        payload.extend_from_slice(&42u16.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&tag.to_le_bytes());
        payload.extend_from_slice(&4u16.to_le_bytes());
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&value.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload
    }

    fn container_xmp(gainmap_length: usize) -> Vec<u8> {
        format!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:Container="http://ns.google.com/photos/1.0/container/"
      xmlns:Item="http://ns.google.com/photos/1.0/container/item/"
      xmlns:hdrgm="{HDRGM_NAMESPACE}"
      hdrgm:Version="1.0">
      <Container:Directory>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource">
            <Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/>
          </rdf:li>
          <rdf:li rdf:parseType="Resource">
            <Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{gainmap_length}"/>
          </rdf:li>
        </rdf:Seq>
      </Container:Directory>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#
        )
        .into_bytes()
    }

    fn gainmap_xmp() -> Vec<u8> {
        br#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
      hdrgm:Version="1.0"
      hdrgm:GainMapMin="0"
      hdrgm:GainMapMax="2"
      hdrgm:Gamma="1"
      hdrgm:OffsetSDR="0.125"
      hdrgm:OffsetHDR="0.25"
      hdrgm:HDRCapacityMin="0"
      hdrgm:HDRCapacityMax="2"
      hdrgm:BaseRenditionIsHDR="False"/>
  </rdf:RDF>
</x:xmpmeta>"#
            .to_vec()
    }

    fn mpf_segment(primary_size: u32, secondary_size: u32) -> Vec<u8> {
        let mut payload = Vec::with_capacity(4 + 82);
        payload.extend_from_slice(MPF_SIGNATURE);
        payload.extend_from_slice(b"MM");
        payload.extend_from_slice(&42u16.to_be_bytes());
        payload.extend_from_slice(&8u32.to_be_bytes());
        payload.extend_from_slice(&3u16.to_be_bytes());

        payload.extend_from_slice(&0xB000u16.to_be_bytes());
        payload.extend_from_slice(&7u16.to_be_bytes());
        payload.extend_from_slice(&4u32.to_be_bytes());
        payload.extend_from_slice(b"0100");

        payload.extend_from_slice(&0xB001u16.to_be_bytes());
        payload.extend_from_slice(&4u16.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&2u32.to_be_bytes());

        payload.extend_from_slice(&0xB002u16.to_be_bytes());
        payload.extend_from_slice(&7u16.to_be_bytes());
        payload.extend_from_slice(&32u32.to_be_bytes());
        payload.extend_from_slice(&50u32.to_be_bytes());

        payload.extend_from_slice(&0u32.to_be_bytes());

        payload.extend_from_slice(&0x0003_0000u32.to_be_bytes());
        payload.extend_from_slice(&primary_size.to_be_bytes());
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes());

        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&secondary_size.to_be_bytes());
        payload.extend_from_slice(&primary_size.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes());
        payload.extend_from_slice(&0u16.to_be_bytes());

        let mut segment = Vec::with_capacity(payload.len() + 4);
        segment.extend_from_slice(&[0xFF, 0xE2]);
        segment.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
        segment.extend_from_slice(&payload);
        segment
    }

    fn inject_uhdr_stream(base: Vec<u8>, gainmap: Vec<u8>) -> Vec<u8> {
        let initial_segment = mpf_segment(0, gainmap.len() as u32);
        let primary_size = (base.len() + initial_segment.len()) as u32;
        let segment = mpf_segment(primary_size, gainmap.len() as u32);

        let mut stream = Vec::with_capacity(base.len() + gainmap.len() + segment.len());
        stream.extend_from_slice(&base[..2]);
        stream.extend_from_slice(&segment);
        stream.extend_from_slice(&base[2..]);
        stream.extend_from_slice(&gainmap);
        stream
    }

    fn synthetic_uhdr_fixture() -> Vec<u8> {
        let gainmap_image =
            Image::<U8>::from_buffer(1, 1, 1, vec![255]).expect("gainmap image should be valid");
        let gainmap = with_embedded_xmp(&JpegCodec.encode(&gainmap_image).unwrap(), &gainmap_xmp());
        let base = with_embedded_xmp(&build_jpeg_fixture(), &container_xmp(gainmap.len()));
        inject_uhdr_stream(base, gainmap)
    }

    fn synthetic_uhdr_fixture_from_parts(base: Vec<u8>, gainmap: Vec<u8>) -> Vec<u8> {
        inject_uhdr_stream(base, gainmap)
    }

    fn test_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from("target/uhdr-codec-tests");
        path.push(name);
        path
    }

    #[test]
    fn sniff_detects_mpf_segment() {
        let bytes = synthetic_uhdr_fixture();
        assert!(UhdrCodec.sniff(&bytes));
        assert!(!UhdrCodec.sniff(&build_jpeg_fixture()));
    }

    #[test]
    fn decode_uses_base_jpeg_and_exposes_gainmap_attachment() {
        let bytes = synthetic_uhdr_fixture();
        let image = UhdrCodec.decode::<U8>(&bytes).unwrap();

        assert_eq!((image.width(), image.height(), image.bands()), (2, 2, 3));
        assert_eq!(
            image.metadata().extra.get("uhdr.base"),
            Some(&"jpeg".to_string())
        );
        assert_eq!(
            image.metadata().extra.get("uhdr.mpf_segments"),
            Some(&"1".to_string())
        );

        let gainmap = image
            .metadata()
            .uhdr_gainmap()
            .expect("UHDR decode should attach the gainmap image");
        assert_eq!((gainmap.image().width(), gainmap.image().height()), (1, 1));
        assert_eq!(gainmap.image().bands(), 1);
        assert_eq!(gainmap.image().pixels(), &[255]);
        assert_eq!(gainmap.metadata.gamma, [1.0; 3]);
        assert_eq!(gainmap.metadata.min_content_boost, [1.0; 3]);
        assert_eq!(gainmap.metadata.max_content_boost, [4.0; 3]);
        assert_eq!(gainmap.metadata.offset_sdr, [0.125; 3]);
        assert_eq!(gainmap.metadata.offset_hdr, [0.25; 3]);
        assert_eq!(gainmap.hdr_capacity_max, 2.0);
        assert!(!gainmap.base_rendition_is_hdr);
    }

    #[test]
    fn decode_rejects_non_u8_output_format() {
        let bytes = synthetic_uhdr_fixture();
        let err = UhdrCodec.decode::<U16>(&bytes).unwrap_err();
        assert!(err.to_string().contains("only U8 is supported"));
    }

    #[test]
    fn probe_returns_jpeg_dimensions() {
        let bytes = synthetic_uhdr_fixture();
        let probe = UhdrCodec.probe(&bytes).unwrap();
        assert_eq!(probe, (2, 2, 3));
    }

    #[test]
    fn registry_prefers_uhdr_decoder_over_plain_jpeg() {
        let path = test_path("registry-uhdr.jpg");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, synthetic_uhdr_fixture()).unwrap();

        let decoded = ForeignRegistry::default().load(&path).unwrap();
        assert!(decoded.metadata().uhdr_gainmap().is_some());

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn decode_rotates_gainmap_with_primary_orientation() {
        let base_image = Image::<U8>::from_buffer(
            2,
            3,
            3,
            vec![
                0, 0, 0, 32, 32, 32, 64, 64, 64, 96, 96, 96, 128, 128, 128, 160, 160, 160,
            ],
        )
        .unwrap();
        let gainmap_image =
            Image::<U8>::from_buffer(2, 3, 1, vec![0, 255, 255, 0, 0, 255]).unwrap();
        let gainmap = with_embedded_xmp(&encode_jpeg_quality_100(&gainmap_image), &gainmap_xmp());
        let expected_gainmap = orient_u8_image(
            &JpegCodec
                .decode_with_options::<U8>(&gainmap, &LoadOptions::default().no_rotate())
                .unwrap(),
            6,
            "test gainmap",
        )
        .unwrap();
        let base = with_embedded_xmp(
            &with_embedded_exif(
                &encode_jpeg_quality_100(&base_image),
                &exif_orientation_payload(6),
            ),
            &container_xmp(gainmap.len()),
        );
        let bytes = synthetic_uhdr_fixture_from_parts(base, gainmap);

        let image = UhdrCodec.decode::<U8>(&bytes).unwrap();
        let gainmap = image.metadata().uhdr_gainmap().unwrap();

        assert_eq!((image.width(), image.height()), (3, 2));
        assert_eq!(gainmap.image().metadata().orientation, Some(1));
        assert_eq!(
            (gainmap.image().width(), gainmap.image().height()),
            (expected_gainmap.width(), expected_gainmap.height())
        );
        assert_eq!(gainmap.image().pixels(), expected_gainmap.pixels());
    }

    #[test]
    fn decode_with_no_rotate_keeps_gainmap_storage_orientation() {
        let base_image = Image::<U8>::from_buffer(
            2,
            3,
            3,
            vec![
                0, 0, 0, 32, 32, 32, 64, 64, 64, 96, 96, 96, 128, 128, 128, 160, 160, 160,
            ],
        )
        .unwrap();
        let gainmap_image =
            Image::<U8>::from_buffer(2, 3, 1, vec![0, 255, 255, 0, 0, 255]).unwrap();
        let gainmap = with_embedded_xmp(&encode_jpeg_quality_100(&gainmap_image), &gainmap_xmp());
        let stored_gainmap = JpegCodec
            .decode_with_options::<U8>(&gainmap, &LoadOptions::default().no_rotate())
            .unwrap();
        let base = with_embedded_xmp(
            &with_embedded_exif(
                &encode_jpeg_quality_100(&base_image),
                &exif_orientation_payload(6),
            ),
            &container_xmp(gainmap.len()),
        );
        let bytes = synthetic_uhdr_fixture_from_parts(base, gainmap);

        let image = UhdrCodec
            .decode_with_options::<U8>(&bytes, &LoadOptions::default().no_rotate())
            .unwrap();
        let gainmap = image.metadata().uhdr_gainmap().unwrap();

        assert_eq!((image.width(), image.height()), (2, 3));
        assert_eq!(
            (gainmap.image().width(), gainmap.image().height()),
            (stored_gainmap.width(), stored_gainmap.height())
        );
        assert_eq!(gainmap.image().pixels(), stored_gainmap.pixels());
    }

    #[test]
    fn decode_exif_only_gainmap_metadata_is_ignored() {
        let gainmap_image = Image::<U8>::from_buffer(1, 1, 1, vec![255]).unwrap();
        let gainmap = with_embedded_exif(
            &encode_jpeg_quality_100(&gainmap_image),
            &proprietary_exif_payload(0xC7A1, 0x0102_0304),
        );
        let base = with_embedded_xmp(&build_jpeg_fixture(), &container_xmp(gainmap.len()));
        let bytes = synthetic_uhdr_fixture_from_parts(base, gainmap);

        let image = UhdrCodec.decode::<U8>(&bytes).unwrap();

        assert!(image.metadata().uhdr_gainmap().is_none());
    }
}
