//! Analyze 7.5 codec — decode SPM/Analyze `.hdr`/`.img` pairs.
//!
//! The Analyze 7.5 format (also called SPM Analyze) consists of:
//! - A 348-byte binary header (`.hdr`) in big-endian (MSB-first) byte order.
//! - A raw pixel data file (`.img`) with no header.
//!
//! For codec API purposes (single `&[u8]` buffer), the input/output is the
//! concatenation `header_bytes ++ pixel_bytes` (348 + n bytes total).
//! This mirrors how libvips stores the "dsr" blob alongside pixels in memory.
//!
//! Supported data types (matches libvips `analyze2vips.c`):
//! - `DT_UNSIGNED_CHAR` (2)  → U8,  1 band
//! - `DT_SIGNED_SHORT`  (4)  → I16, 1 band
//! - `DT_SIGNED_INT`    (8)  → I32, 1 band
//! - `DT_FLOAT`         (16) → F32, 1 band
//! - `DT_DOUBLE`        (64) → F64, 1 band
//! - `DT_RGB`           (128) → U8, 3 bands
//!
//! Byte order: the `.hdr` and `.img` payload are always big-endian
//! (SPARC/MSB). The decoder reads them explicitly as big-endian bytes so the
//! result does not depend on host endianness.
//!
//! Reference: `.libvips_repo/libvips/foreign/analyze2vips.c`,
//!            `.libvips_repo/libvips/foreign/dbh.h`.

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8};
use viprs_core::image::InMemoryImage;
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Fixed size of the Analyze 7.5 `dsr` header struct.
pub const ANALYZE_HEADER_SIZE: usize = 348;

/// Analyze data type codes (from `dbh.h`).
const DT_UNSIGNED_CHAR: i16 = 2;
const DT_SIGNED_SHORT: i16 = 4;
const DT_SIGNED_INT: i16 = 8;
const DT_FLOAT: i16 = 16;
const DT_DOUBLE: i16 = 64;
const DT_RGB: i16 = 128;

// ── Header layout offsets (byte offsets into the 348-byte `dsr` struct) ───────
//
// These are taken directly from dbh.h with careful accounting of field sizes:
//   struct header_key (40 bytes):
//     sizeof_hdr:     int32  @ 0
//     (other fields)
//   struct image_dimension (108 bytes) starts at offset 40:
//     dim[8]:         int16×8 @ 40..56
//     (gap / vox_units 4 bytes, cal_units 8 bytes, unused1 2 bytes) @ 56..70
//     datatype:       int16 @ 70
//     bitpix:         int16 @ 72
//   struct data_history (200 bytes) starts at offset 148

const OFFSET_SIZEOF_HDR: usize = 0;
const OFFSET_DIM: usize = 40; // dim[0..7]: 8 × i16
const OFFSET_DATATYPE: usize = 70;
// bitpix is at 72, pixdim at 76, etc. — not needed for decode.

// ── Byte-swap helpers ─────────────────────────────────────────────────────────

#[inline]
fn read_i16_be(buf: &[u8], offset: usize) -> i16 {
    i16::from_be_bytes([buf[offset], buf[offset + 1]])
}

#[inline]
fn read_i32_be(buf: &[u8], offset: usize) -> i32 {
    i32::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

#[inline]
fn write_i16_be(buf: &mut [u8], offset: usize, v: i16) {
    let bytes = v.to_be_bytes();
    buf[offset] = bytes[0];
    buf[offset + 1] = bytes[1];
}

#[inline]
fn write_i32_be(buf: &mut [u8], offset: usize, v: i32) {
    let bytes = v.to_be_bytes();
    buf[offset..offset + 4].copy_from_slice(&bytes);
}

// ── Header decode ─────────────────────────────────────────────────────────────

struct AnalyzeInfo {
    width: u32,
    height: u32,
    bands: u32,
    datatype: i16,
    bytes_per_sample: usize,
}

fn parse_analyze_header(src: &[u8]) -> Result<AnalyzeInfo, ViprsError> {
    if src.len() < ANALYZE_HEADER_SIZE {
        return Err(ViprsError::Codec(format!(
            "analyze: buffer too short for header: {} < {}",
            src.len(),
            ANALYZE_HEADER_SIZE
        )));
    }

    // Validate sizeof_hdr (must be 348 in big-endian).
    let sizeof_hdr = read_i32_be(src, OFFSET_SIZEOF_HDR);
    if sizeof_hdr != ANALYZE_HEADER_SIZE as i32 {
        return Err(ViprsError::Codec(format!(
            "analyze: sizeof_hdr is {sizeof_hdr}, expected {ANALYZE_HEADER_SIZE}"
        )));
    }

    // dim[0] is the number of dimensions (must be 2–7).
    let ndim = read_i16_be(src, OFFSET_DIM);
    if !(2..=7).contains(&ndim) {
        return Err(ViprsError::Codec(format!(
            "analyze: {ndim}-dimensional images not supported"
        )));
    }

    // dim[1] = x, dim[2] = y; higher dimensions collapse into height.
    let dim_x = i64::from(read_i16_be(src, OFFSET_DIM + 2));
    if dim_x <= 0 {
        return Err(ViprsError::Codec("analyze: width must be positive".into()));
    }

    let mut height_64: i64 = i64::from(read_i16_be(src, OFFSET_DIM + 4));
    for i in 3..=(ndim as usize) {
        height_64 *= i64::from(read_i16_be(src, OFFSET_DIM + i * 2));
        if height_64 <= 0 {
            return Err(ViprsError::Codec(
                "analyze: height product overflowed or is non-positive".into(),
            ));
        }
    }

    let datatype = read_i16_be(src, OFFSET_DATATYPE);

    let (bands, bytes_per_sample) = match datatype {
        DT_UNSIGNED_CHAR => (1u32, 1usize),
        DT_SIGNED_SHORT => (1, 2),
        DT_SIGNED_INT | DT_FLOAT => (1, 4),
        DT_DOUBLE => (1, 8),
        DT_RGB => (3, 1),
        other => {
            return Err(ViprsError::Codec(format!(
                "analyze: datatype {other} not supported"
            )));
        }
    };

    Ok(AnalyzeInfo {
        width: dim_x as u32,
        height: height_64 as u32,
        bands,
        datatype,
        bytes_per_sample,
    })
}

// ── Core decode ──────────────────────────────────────────────────────────────

/// Decode the concatenated `hdr+img` buffer.
///
/// The caller must provide the full byte sequence as produced by
/// [`encode_analyze`]: 348 header bytes followed by raw pixel data.
fn decode_analyze(
    src: &[u8],
    target_format: BandFormatId,
) -> Result<Box<dyn std::any::Any + Send>, ViprsError> {
    let info = parse_analyze_header(src)?;

    let pixel_bytes = src
        .get(ANALYZE_HEADER_SIZE..)
        .ok_or_else(|| ViprsError::Codec("analyze: buffer too short for pixel data".into()))?;

    let expected_bytes = (info.width as usize)
        .checked_mul(info.height as usize)
        .and_then(|n| n.checked_mul(info.bands as usize))
        .and_then(|n| n.checked_mul(info.bytes_per_sample))
        .ok_or_else(|| ViprsError::Codec("analyze: pixel data size overflow".into()))?;

    if pixel_bytes.len() < expected_bytes {
        return Err(ViprsError::Codec(format!(
            "analyze: pixel data too short: {} < {expected_bytes}",
            pixel_bytes.len()
        )));
    }

    let pixel_bytes = &pixel_bytes[..expected_bytes];

    // Build the image in the native datatype, then box it.
    match (info.datatype, target_format) {
        (DT_UNSIGNED_CHAR | DT_RGB, BandFormatId::U8) => {
            let pixels = pixel_bytes.to_vec();
            let img = InMemoryImage::<U8>::from_buffer(info.width, info.height, info.bands, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            Ok(Box::new(img))
        }
        (DT_SIGNED_SHORT, BandFormatId::I16) => {
            let pixels: Vec<i16> = pixel_bytes
                .chunks_exact(2)
                .map(|c| i16::from_be_bytes([c[0], c[1]]))
                .collect();
            let img = InMemoryImage::<I16>::from_buffer(info.width, info.height, 1, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            Ok(Box::new(img))
        }
        (DT_SIGNED_INT, BandFormatId::I32) => {
            let pixels: Vec<i32> = pixel_bytes
                .chunks_exact(4)
                .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let img = InMemoryImage::<I32>::from_buffer(info.width, info.height, 1, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            Ok(Box::new(img))
        }
        (DT_FLOAT, BandFormatId::F32) => {
            let pixels: Vec<f32> = pixel_bytes
                .chunks_exact(4)
                .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let img = InMemoryImage::<F32>::from_buffer(info.width, info.height, 1, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            Ok(Box::new(img))
        }
        (DT_DOUBLE, BandFormatId::F64) => {
            let pixels: Vec<f64> = pixel_bytes
                .chunks_exact(8)
                .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            let img = InMemoryImage::<F64>::from_buffer(info.width, info.height, 1, pixels)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            Ok(Box::new(img))
        }
        (native_dt, requested) => Err(ViprsError::Codec(format!(
            "analyze: native datatype {native_dt} is not compatible with requested format {requested:?}"
        ))),
    }
}

// ── Core encode ──────────────────────────────────────────────────────────────

/// Encode an image into a concatenated `hdr+img` buffer.
///
/// The header is always written big-endian (Analyze specification).
fn encode_analyze<F: BandFormat>(image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
    // Map BandFormatId to Analyze datatype + bytes_per_sample.
    let (datatype, bytes_per_sample, ndim): (i16, usize, i16) = match (F::ID, image.bands()) {
        (BandFormatId::U8, 1) => (DT_UNSIGNED_CHAR, 1, 2),
        (BandFormatId::U8, 3) => (DT_RGB, 1, 2),
        (BandFormatId::I16, 1) => (DT_SIGNED_SHORT, 2, 2),
        (BandFormatId::I32, 1) => (DT_SIGNED_INT, 4, 2),
        (BandFormatId::F32, 1) => (DT_FLOAT, 4, 2),
        (BandFormatId::F64, 1) => (DT_DOUBLE, 8, 2),
        (fmt, bands) => {
            return Err(ViprsError::Codec(format!(
                "analyze: unsupported format {fmt:?} with {bands} band(s) for encoding"
            )));
        }
    };

    let bitpix = (bytes_per_sample * 8) as i16;

    // Build the 348-byte header, all zeroed initially.
    let mut header = vec![0u8; ANALYZE_HEADER_SIZE];

    // header_key.sizeof_hdr = 348 (big-endian i32)
    write_i32_be(&mut header, OFFSET_SIZEOF_HDR, ANALYZE_HEADER_SIZE as i32);

    // header_key.regular = 'r'
    header[38] = b'r';

    // image_dimension.dim[0] = ndim
    write_i16_be(&mut header, OFFSET_DIM, ndim);
    // dim[1] = width, dim[2] = height
    write_i16_be(&mut header, OFFSET_DIM + 2, image.width() as i16);
    write_i16_be(&mut header, OFFSET_DIM + 4, image.height() as i16);
    // datatype
    write_i16_be(&mut header, OFFSET_DATATYPE, datatype);
    // bitpix at offset 72
    write_i16_be(&mut header, 72, bitpix);

    // Pixel data is written as big-endian bytes so it decodes identically on
    // every host. U8/RGB samples are copied verbatim.
    let pixel_bytes = encode_pixels_be::<F>(image, datatype, bytes_per_sample)?;

    let mut out = header;
    out.extend_from_slice(&pixel_bytes);
    Ok(out)
}

fn encode_pixels_be<F: BandFormat>(
  image: &InMemoryImage<F>,
  datatype: i16,
  _bytes_per_sample: usize,
) -> Result<Vec<u8>, ViprsError> {
    match datatype {
        DT_UNSIGNED_CHAR | DT_RGB => {
            Ok(bytemuck::cast_slice::<F::Sample, u8>(image.pixels()).to_vec())
        }
        DT_SIGNED_SHORT => {
            let samples = bytemuck::cast_slice::<F::Sample, i16>(image.pixels());
            let mut out = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                out.extend_from_slice(&s.to_be_bytes());
            }
            Ok(out)
        }
        DT_SIGNED_INT => {
            let samples = bytemuck::cast_slice::<F::Sample, i32>(image.pixels());
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                out.extend_from_slice(&s.to_be_bytes());
            }
            Ok(out)
        }
        DT_FLOAT => {
            let samples = bytemuck::cast_slice::<F::Sample, f32>(image.pixels());
            let mut out = Vec::with_capacity(samples.len() * 4);
            for &s in samples {
                out.extend_from_slice(&s.to_be_bytes());
            }
            Ok(out)
        }
        DT_DOUBLE => {
            let samples = bytemuck::cast_slice::<F::Sample, f64>(image.pixels());
            let mut out = Vec::with_capacity(samples.len() * 8);
            for &s in samples {
                out.extend_from_slice(&s.to_be_bytes());
            }
            Ok(out)
        }
        other => Err(ViprsError::Codec(format!(
            "analyze: unexpected datatype {other} in encode_pixels_be"
        ))),
    }
}

// ── Public codec struct ──────────────────────────────────────────────────────

/// Codec for the SPM Analyze 7.5 format.
///
/// Input/output buffer layout: 348-byte big-endian header followed immediately
/// by the raw pixel data (no gap).  This corresponds to concatenating the
/// `.hdr` and `.img` files that the format uses on disk.
#[derive(Clone, Copy, Debug)]
/// The `AnalyzeCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::analyze::AnalyzeCodec>();
/// ```
pub struct AnalyzeCodec;

impl AnalyzeCodec {
    /// Sniff an Analyze header: `sizeof_hdr` at offset 0 must be 348 (BE).
    fn sniff_header(header: &[u8]) -> bool {
        if header.len() < 4 {
            return false;
        }
        let sizeof_hdr = i32::from_be_bytes([header[0], header[1], header[2], header[3]]);
        sizeof_hdr == ANALYZE_HEADER_SIZE as i32
    }
}

impl ImageDecoder for AnalyzeCodec {
    fn format_name(&self) -> &'static str {
        "analyze"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        Self::sniff_header(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
    {
        let boxed = decode_analyze(src, F::ID)?;
        boxed.downcast::<InMemoryImage<F>>().map(|b| *b).map_err(|_| {
            ViprsError::Codec(format!(
                "analyze: decoded image type does not match requested format {:?}",
                F::ID
            ))
        })
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let info = parse_analyze_header(src)?;
        Ok((info.width, info.height, info.bands))
    }
}

impl ImageEncoder for AnalyzeCodec {
    fn format_name(&self) -> &'static str {
        "analyze"
    }

    fn encode<F: BandFormat>(&self, image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
      &self,
      image: &InMemoryImage<F>,
      _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        encode_analyze(image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{F32, F64, I16, I32, U8};

    const SAMPLE_JPEG: &[u8] = include_bytes!("../../../tests/fixtures/images/sample.jpg");
    const SAMPLE_PNG: &[u8] = include_bytes!("../../../tests/fixtures/images/sample.png");

    /// Build a minimal valid 348-byte Analyze header for the given dimensions
    /// and datatype.
    fn make_header(width: u16, height: u16, datatype: i16, bitpix: i16) -> Vec<u8> {
        let mut hdr = vec![0u8; ANALYZE_HEADER_SIZE];
        write_i32_be(&mut hdr, OFFSET_SIZEOF_HDR, ANALYZE_HEADER_SIZE as i32);
        hdr[38] = b'r'; // regular
        write_i16_be(&mut hdr, OFFSET_DIM, 2); // 2D
        write_i16_be(&mut hdr, OFFSET_DIM + 2, width as i16);
        write_i16_be(&mut hdr, OFFSET_DIM + 4, height as i16);
        write_i16_be(&mut hdr, OFFSET_DATATYPE, datatype);
        write_i16_be(&mut hdr, 72, bitpix);
        hdr
    }

    #[test]
    fn sniff_accepts_valid_analyze_header() {
        let codec = AnalyzeCodec;
        let hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        assert!(codec.sniff(&hdr));
    }

    #[test]
    fn sniff_rejects_non_analyze() {
        let codec = AnalyzeCodec;
        assert!(!codec.sniff(SAMPLE_JPEG));
        assert!(!codec.sniff(SAMPLE_PNG));
    }

    #[test]
    fn sniff_rejects_headers_shorter_than_sizeof_hdr_field() {
        assert!(!AnalyzeCodec::sniff_header(&[0x00, 0x00, 0x01]));
    }

    #[test]
    fn decoder_and_encoder_report_format_name() {
        let codec = AnalyzeCodec;
        assert_eq!(ImageDecoder::format_name(&codec), "analyze");
        assert_eq!(ImageEncoder::format_name(&codec), "analyze");
    }

    #[test]
    fn round_trip_u8_grayscale() {
        let codec = AnalyzeCodec;
        let original = InMemoryImage::<U8>::from_buffer(4, 2, 1, (0..8u8).collect()).unwrap();
        let encoded = codec.encode(&original).unwrap();
        assert_eq!(encoded.len(), ANALYZE_HEADER_SIZE + 8);
        let decoded = codec.decode::<U8>(&encoded).unwrap();
        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 1);
        assert_eq!(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_u8_rgb() {
        let codec = AnalyzeCodec;
        let pixels: Vec<u8> = (0..24).collect();
        let original = InMemoryImage::<U8>::from_buffer(4, 2, 3, pixels).unwrap();
        let encoded = codec.encode(&original).unwrap();
        assert_eq!(encoded.len(), ANALYZE_HEADER_SIZE + 24);
        let decoded = codec.decode::<U8>(&encoded).unwrap();
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), original.pixels());
    }

    #[test]
    fn trait_entrypoints_round_trip_rgb_and_probe_dimensions() {
        let codec = AnalyzeCodec;
        let image = InMemoryImage::<U8>::from_buffer(3, 2, 3, (0u8..18).collect()).unwrap();

        let encoded = codec
            .encode_with_options(&image, &SaveOptions::default())
            .unwrap();
        let decoded = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default())
            .unwrap();
        let dims = codec.probe(&encoded).unwrap();

        assert_eq!(decoded.pixels(), image.pixels());
        assert_eq!(dims, (3, 2, 3));
    }

    #[test]
    fn round_trip_i16() {
        let codec = AnalyzeCodec;
        let pixels: Vec<i16> = vec![-1000, 0, 1000, i16::MAX];
        let original = InMemoryImage::<I16>::from_buffer(4, 1, 1, pixels).unwrap();
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<I16>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), original.pixels());
    }

    #[test]
    fn encode_writes_big_endian_header_and_i16_payload() {
        let codec = AnalyzeCodec;
        let original = InMemoryImage::<I16>::from_buffer(2, 1, 1, vec![0x1234, -2]).unwrap();
        let encoded = codec.encode(&original).unwrap();

        assert_eq!(
            &encoded[..ANALYZE_HEADER_SIZE],
            &make_header(2, 1, DT_SIGNED_SHORT, 16)
        );
        assert_eq!(&encoded[ANALYZE_HEADER_SIZE..], &[0x12, 0x34, 0xff, 0xfe]);
    }

    #[test]
    fn decode_reads_big_endian_i16_payload() {
        let codec = AnalyzeCodec;
        let mut buf = make_header(2, 1, DT_SIGNED_SHORT, 16);
        buf.extend_from_slice(&[0x12, 0x34, 0xff, 0xfe]);

        let decoded = codec.decode::<I16>(&buf).unwrap();
        assert_eq!(decoded.pixels(), &[0x1234, -2]);
    }

    #[test]
    fn round_trip_i32() {
        let codec = AnalyzeCodec;
        let pixels: Vec<i32> = vec![i32::MIN, -1, 0, i32::MAX];
        let original = InMemoryImage::<I32>::from_buffer(4, 1, 1, pixels).unwrap();
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<I32>(&encoded).unwrap();
        assert_eq!(decoded.pixels(), original.pixels());
    }

    #[test]
    fn round_trip_f32() {
        let codec = AnalyzeCodec;
        let pixels: Vec<f32> = vec![-1.5, 0.0, 1.5, f32::MAX];
        let original = InMemoryImage::<F32>::from_buffer(4, 1, 1, pixels).unwrap();
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F32>(&encoded).unwrap();
        for (a, b) in decoded.pixels().iter().zip(original.pixels().iter()) {
            assert!((a - b).abs() <= f32::EPSILON * a.abs().max(1.0));
        }
    }

    #[test]
    fn decode_reads_big_endian_f32_payload() {
        let codec = AnalyzeCodec;
        let mut buf = make_header(2, 1, DT_FLOAT, 32);
        buf.extend_from_slice(&0x3f800000u32.to_be_bytes());
        buf.extend_from_slice(&0xc0200000u32.to_be_bytes());

        let decoded = codec.decode::<F32>(&buf).unwrap();
        assert_eq!(decoded.pixels()[0].to_bits(), 0x3f800000);
        assert_eq!(decoded.pixels()[1].to_bits(), 0xc0200000);
    }

    #[test]
    fn round_trip_f64() {
        let codec = AnalyzeCodec;
        let pixels: Vec<f64> = vec![-1.0, 0.5, 1.0, std::f64::consts::PI];
        let original = InMemoryImage::<F64>::from_buffer(4, 1, 1, pixels).unwrap();
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F64>(&encoded).unwrap();
        for (a, b) in decoded.pixels().iter().zip(original.pixels().iter()) {
            assert!((a - b).abs() <= f64::EPSILON * a.abs().max(1.0));
        }
    }

    #[test]
    fn probe_returns_correct_dimensions() {
        let codec = AnalyzeCodec;
        // Prepend a valid header to some pixel bytes so probe can read dimensions.
        let mut buf = make_header(10, 5, DT_UNSIGNED_CHAR, 8);
        buf.extend(vec![0u8; 50]); // 10 × 5 pixels
        let (w, h, b) = codec.probe(&buf).unwrap();
        assert_eq!((w, h, b), (10, 5, 1));
    }

    #[test]
    fn decode_truncated_pixel_data_errors() {
        let codec = AnalyzeCodec;
        let mut buf = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        buf.extend(vec![0u8; 4]); // only 4 bytes, need 16
        assert!(codec.decode::<U8>(&buf).is_err());
    }

    #[test]
    fn decode_invalid_sizeof_hdr_errors() {
        let codec = AnalyzeCodec;
        let mut hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        // Corrupt sizeof_hdr to 0.
        write_i32_be(&mut hdr, 0, 0);
        hdr.extend(vec![0u8; 16]);
        assert!(codec.decode::<U8>(&hdr).is_err());
    }

    #[test]
    fn decode_format_mismatch_errors() {
        let codec = AnalyzeCodec;
        // Encode as U8, then try to decode as I16.
        let original = InMemoryImage::<U8>::from_buffer(2, 2, 1, vec![0; 4]).unwrap();
        let encoded = codec.encode(&original).unwrap();
        assert!(codec.decode::<I16>(&encoded).is_err());
    }

    #[test]
    fn parse_header_rejects_short_buffers() {
        let err = match parse_analyze_header(&SAMPLE_PNG[..ANALYZE_HEADER_SIZE - 1]) {
            Ok(_) => panic!("short buffer should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("buffer too short for header"));
    }

    #[test]
    fn parse_header_rejects_unsupported_dimension_count() {
        let mut hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        write_i16_be(&mut hdr, OFFSET_DIM, 1);

        let err = match parse_analyze_header(&hdr) {
            Ok(_) => panic!("unsupported dimension count should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("1-dimensional images not supported")
        );
    }

    #[test]
    fn parse_header_rejects_non_positive_width() {
        let mut hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        write_i16_be(&mut hdr, OFFSET_DIM + 2, 0);

        let err = match parse_analyze_header(&hdr) {
            Ok(_) => panic!("non-positive width should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("width must be positive"));
    }

    #[test]
    fn parse_header_rejects_non_positive_height_product() {
        let mut hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        write_i16_be(&mut hdr, OFFSET_DIM, 3);
        write_i16_be(&mut hdr, OFFSET_DIM + 6, 0);

        let err = match parse_analyze_header(&hdr) {
            Ok(_) => panic!("non-positive height product should fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string()
                .contains("height product overflowed or is non-positive")
        );
    }

    #[test]
    fn parse_header_rejects_unsupported_datatype() {
        let mut hdr = make_header(4, 4, DT_UNSIGNED_CHAR, 8);
        write_i16_be(&mut hdr, OFFSET_DATATYPE, 32);

        let err = match parse_analyze_header(&hdr) {
            Ok(_) => panic!("unsupported datatype should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("datatype 32 not supported"));
    }

    #[test]
    fn parse_header_multiplies_higher_dimensions_into_height() {
        let mut hdr = make_header(3, 2, DT_UNSIGNED_CHAR, 8);
        write_i16_be(&mut hdr, OFFSET_DIM, 4);
        write_i16_be(&mut hdr, OFFSET_DIM + 6, 5);
        write_i16_be(&mut hdr, OFFSET_DIM + 8, 2);

        let info = parse_analyze_header(&hdr).unwrap();

        assert_eq!((info.width, info.height, info.bands), (3, 20, 1));
    }

    #[test]
    fn encode_rejects_unsupported_format_band_combo() {
        let image = InMemoryImage::<I16>::from_buffer(2, 1, 2, vec![1, 2, 3, 4]).unwrap();

        let err = encode_analyze(&image).unwrap_err();
        assert!(
            err.to_string()
                .contains("unsupported format I16 with 2 band(s) for encoding")
        );
    }

    #[test]
    fn encode_pixels_rejects_unexpected_datatype() {
        let image = InMemoryImage::<U8>::from_buffer(2, 1, 1, vec![1, 2]).unwrap();

        let err = encode_pixels_be(&image, 999, 1).unwrap_err();
        assert!(
            err.to_string()
                .contains("unexpected datatype 999 in encode_pixels_be")
        );
    }
}
