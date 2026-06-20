//! VIPS native `.v` / `.vips` codec.
//!
//! Parity target: libvips `vipsload` / `vipssave` backed by iofuncs/vips.c.
//! This implementation writes and reads the native 64-byte VIPS header and
//! interleaved pixel payload.

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{Image, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const VIPS_MAGIC_INTEL_BYTES: [u8; 4] = [0xB6, 0xA6, 0xF2, 0x08];
const VIPS_MAGIC_SPARC_BYTES: [u8; 4] = [0x08, 0xF2, 0xA6, 0xB6];
const VIPS_HEADER_SIZE: usize = 64;

const VIPS_CODING_NONE: i32 = 0;

const VIPS_FORMAT_UCHAR: i32 = 0;
const VIPS_FORMAT_USHORT: i32 = 2;
const VIPS_FORMAT_SHORT: i32 = 3;
const VIPS_FORMAT_UINT: i32 = 4;
const VIPS_FORMAT_INT: i32 = 5;
const VIPS_FORMAT_FLOAT: i32 = 6;
const VIPS_FORMAT_DOUBLE: i32 = 8;

const VIPS_INTERPRETATION_MULTIBAND: i32 = 0;
const VIPS_INTERPRETATION_B_W: i32 = 1;
const VIPS_INTERPRETATION_HISTOGRAM: i32 = 10;
const VIPS_INTERPRETATION_XYZ: i32 = 12;
const VIPS_INTERPRETATION_LAB: i32 = 13;
const VIPS_INTERPRETATION_CMYK: i32 = 15;
const VIPS_INTERPRETATION_LABQ: i32 = 16;
const VIPS_INTERPRETATION_RGB: i32 = 17;
const VIPS_INTERPRETATION_CMC: i32 = 18;
const VIPS_INTERPRETATION_LCH: i32 = 19;
const VIPS_INTERPRETATION_LABS: i32 = 21;
const VIPS_INTERPRETATION_SRGB: i32 = 22;
const VIPS_INTERPRETATION_YXY: i32 = 23;
const VIPS_INTERPRETATION_FOURIER: i32 = 24;
const VIPS_INTERPRETATION_RGB16: i32 = 25;
const VIPS_INTERPRETATION_GREY16: i32 = 26;
const VIPS_INTERPRETATION_MATRIX: i32 = 27;
const VIPS_INTERPRETATION_SCRGB: i32 = 28;
const VIPS_INTERPRETATION_HSV: i32 = 29;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HeaderEndianness {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VipsHeader {
    width: u32,
    height: u32,
    bands: u32,
    band_format: BandFormatId,
    coding: i32,
    interpretation: Option<Interpretation>,
    xres: f64,
    yres: f64,
}

/// The `VipsCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::vips_format::VipsCodec>();
/// ```
pub struct VipsCodec;

impl VipsCodec {
    /// `decode_vips` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// let _ = viprs_codecs::vips_format::VipsCodec::decode_vips::<viprs_core::format::U8>;
    /// ```
    pub fn decode_vips<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        let (header, endianness) = parse_header(src)?;
        if header.band_format != F::ID {
            return Err(ViprsError::Codec(format!(
                "vips: header band format {:?} does not match requested {:?}",
                header.band_format,
                F::ID
            )));
        }
        if header.coding != VIPS_CODING_NONE {
            return Err(ViprsError::Codec(format!(
                "vips: unsupported coding {} (only coding=0 is supported)",
                header.coding
            )));
        }

        let sample_size = std::mem::size_of::<F::Sample>();
        let sample_count = pixel_count(header.width, header.height, header.bands)?;
        let expected_bytes = sample_count
            .checked_mul(sample_size)
            .ok_or_else(|| ViprsError::Codec("vips: pixel payload byte count overflow".into()))?;

        if src.len() < VIPS_HEADER_SIZE + expected_bytes {
            return Err(ViprsError::Codec(format!(
                "vips: truncated pixel payload: got {} bytes, need {}",
                src.len().saturating_sub(VIPS_HEADER_SIZE),
                expected_bytes
            )));
        }

        let payload = &src[VIPS_HEADER_SIZE..VIPS_HEADER_SIZE + expected_bytes];
        let mut samples = vec![bytemuck::Zeroable::zeroed(); sample_count];
        let sample_bytes = bytemuck::cast_slice_mut::<F::Sample, u8>(&mut samples);
        sample_bytes.copy_from_slice(payload);

        if needs_swap(endianness) && sample_size > 1 {
            for chunk in sample_bytes.chunks_exact_mut(sample_size) {
                chunk.reverse();
            }
        }

        let metadata = ImageMetadata {
            interpretation: header.interpretation,
            xres: Some(header.xres),
            yres: Some(header.yres),
            ..ImageMetadata::default()
        };
        Image::from_buffer(header.width, header.height, header.bands, samples)
            .map_err(|err| ViprsError::Codec(err.to_string()))
            .map(|image| image.with_metadata(metadata))
    }

    /// `encode_vips` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// let _ = viprs_codecs::vips_format::VipsCodec::encode_vips::<viprs_core::format::U8>;
    /// ```
    pub fn encode_vips<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        let sample_size = std::mem::size_of::<F::Sample>();
        let pixel_bytes = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());
        let length_i32 = i32::try_from(pixel_bytes.len()).map_err(|_| {
            ViprsError::Codec("vips: pixel payload exceeds i32 length field".into())
        })?;

        let endian = native_endianness();
        let mut header = [0u8; VIPS_HEADER_SIZE];

        write_u32(&mut header[0..4], magic_for_endianness(endian), endian);
        write_i32(
            &mut header[4..8],
            i32::try_from(image.width())
                .map_err(|_| ViprsError::Codec("vips: width overflows i32".into()))?,
            endian,
        );
        write_i32(
            &mut header[8..12],
            i32::try_from(image.height())
                .map_err(|_| ViprsError::Codec("vips: height overflows i32".into()))?,
            endian,
        );
        write_i32(
            &mut header[12..16],
            i32::try_from(image.bands())
                .map_err(|_| ViprsError::Codec("vips: bands overflows i32".into()))?,
            endian,
        );
        write_i32(
            &mut header[16..20],
            i32::try_from(sample_size * 8)
                .map_err(|_| ViprsError::Codec("vips: bbits overflows i32".into()))?,
            endian,
        );
        write_i32(&mut header[20..24], band_format_to_vips(F::ID), endian);
        write_i32(&mut header[24..28], VIPS_CODING_NONE, endian);
        write_i32(
            &mut header[28..32],
            interpretation_to_vips(image.metadata().interpretation),
            endian,
        );
        write_f32(
            &mut header[32..36],
            image.metadata().xres.unwrap_or(1.0) as f32,
            endian,
        );
        write_f32(
            &mut header[36..40],
            image.metadata().yres.unwrap_or(1.0) as f32,
            endian,
        );
        write_i32(&mut header[40..44], length_i32, endian);
        write_i16(&mut header[44..46], 0, endian); // Compression
        write_i16(&mut header[46..48], 0, endian); // Level
        write_i32(&mut header[48..52], 0, endian); // Xoffset
        write_i32(&mut header[52..56], 0, endian); // Yoffset
        // bytes 56..64 are zero-padding

        let mut out = Vec::with_capacity(VIPS_HEADER_SIZE + pixel_bytes.len());
        out.extend_from_slice(&header);
        out.extend_from_slice(pixel_bytes);
        Ok(out)
    }
}

impl ImageDecoder for VipsCodec {
    fn format_name(&self) -> &'static str {
        "vips"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_vips(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_vips(src)
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        self.decode_vips(src)
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let (header, _) = parse_header(src)?;
        Ok((header.width, header.height, header.bands))
    }
}

impl ImageEncoder for VipsCodec {
    fn format_name(&self) -> &'static str {
        "vips"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_vips(image)
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        self.encode_vips(image)
    }
}

fn sniff_vips(header: &[u8]) -> bool {
    header.len() >= 4
        && (header[0..4] == VIPS_MAGIC_INTEL_BYTES || header[0..4] == VIPS_MAGIC_SPARC_BYTES)
}

fn parse_header(src: &[u8]) -> Result<(VipsHeader, HeaderEndianness), ViprsError> {
    if src.len() < VIPS_HEADER_SIZE {
        return Err(ViprsError::Codec(format!(
            "vips: header too short: got {} bytes, need {}",
            src.len(),
            VIPS_HEADER_SIZE
        )));
    }
    if !sniff_vips(src) {
        return Err(ViprsError::Codec("vips: invalid magic".into()));
    }

    let endianness = if src[0..4] == VIPS_MAGIC_INTEL_BYTES {
        HeaderEndianness::Little
    } else {
        HeaderEndianness::Big
    };

    let width_i32 = read_i32(&src[4..8], endianness);
    let height_i32 = read_i32(&src[8..12], endianness);
    let bands_i32 = read_i32(&src[12..16], endianness);
    if width_i32 <= 0 || height_i32 <= 0 || bands_i32 <= 0 {
        return Err(ViprsError::Codec(format!(
            "vips: invalid dimensions in header: width={width_i32}, height={height_i32}, bands={bands_i32}"
        )));
    }

    let band_format_raw = read_i32(&src[20..24], endianness);
    let band_format = vips_to_band_format(band_format_raw)?;
    let coding = read_i32(&src[24..28], endianness);
    let interp_raw = read_i32(&src[28..32], endianness);
    let xres = f64::from(read_f32(&src[32..36], endianness)).max(0.0);
    let yres = f64::from(read_f32(&src[36..40], endianness)).max(0.0);

    Ok((
        VipsHeader {
            width: width_i32 as u32,
            height: height_i32 as u32,
            bands: bands_i32 as u32,
            band_format,
            coding,
            interpretation: vips_to_interpretation(interp_raw),
            xres,
            yres,
        },
        endianness,
    ))
}

fn pixel_count(width: u32, height: u32, bands: u32) -> Result<usize, ViprsError> {
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|value| value.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::Codec("vips: dimensions overflow usize".into()))
}

const fn needs_swap(file_endian: HeaderEndianness) -> bool {
    matches!(
        (file_endian, native_endianness()),
        (HeaderEndianness::Little, HeaderEndianness::Big)
            | (HeaderEndianness::Big, HeaderEndianness::Little)
    )
}

const fn native_endianness() -> HeaderEndianness {
    if cfg!(target_endian = "big") {
        HeaderEndianness::Big
    } else {
        HeaderEndianness::Little
    }
}

const fn magic_for_endianness(endian: HeaderEndianness) -> u32 {
    match endian {
        HeaderEndianness::Little => u32::from_le_bytes(VIPS_MAGIC_INTEL_BYTES),
        HeaderEndianness::Big => u32::from_be_bytes(VIPS_MAGIC_SPARC_BYTES),
    }
}

const fn band_format_to_vips(id: BandFormatId) -> i32 {
    match id {
        BandFormatId::U8 => VIPS_FORMAT_UCHAR,
        BandFormatId::U16 => VIPS_FORMAT_USHORT,
        BandFormatId::I16 => VIPS_FORMAT_SHORT,
        BandFormatId::U32 => VIPS_FORMAT_UINT,
        BandFormatId::I32 => VIPS_FORMAT_INT,
        BandFormatId::F32 => VIPS_FORMAT_FLOAT,
        BandFormatId::F64 => VIPS_FORMAT_DOUBLE,
    }
}

fn vips_to_band_format(value: i32) -> Result<BandFormatId, ViprsError> {
    let id = match value {
        VIPS_FORMAT_UCHAR => BandFormatId::U8,
        VIPS_FORMAT_USHORT => BandFormatId::U16,
        VIPS_FORMAT_SHORT => BandFormatId::I16,
        VIPS_FORMAT_UINT => BandFormatId::U32,
        VIPS_FORMAT_INT => BandFormatId::I32,
        VIPS_FORMAT_FLOAT => BandFormatId::F32,
        VIPS_FORMAT_DOUBLE => BandFormatId::F64,
        _ => {
            return Err(ViprsError::Codec(format!(
                "vips: unsupported band format value {value}"
            )));
        }
    };
    Ok(id)
}

fn interpretation_to_vips(interp: Option<Interpretation>) -> i32 {
    match interp.unwrap_or(Interpretation::Multiband) {
        Interpretation::Multiband => VIPS_INTERPRETATION_MULTIBAND,
        Interpretation::BW => VIPS_INTERPRETATION_B_W,
        Interpretation::Histogram => VIPS_INTERPRETATION_HISTOGRAM,
        Interpretation::Xyz => VIPS_INTERPRETATION_XYZ,
        Interpretation::Lab => VIPS_INTERPRETATION_LAB,
        Interpretation::Cmyk => VIPS_INTERPRETATION_CMYK,
        Interpretation::Labq => VIPS_INTERPRETATION_LABQ,
        Interpretation::Rgb => VIPS_INTERPRETATION_RGB,
        Interpretation::Cmc => VIPS_INTERPRETATION_CMC,
        Interpretation::Lch => VIPS_INTERPRETATION_LCH,
        Interpretation::Labs => VIPS_INTERPRETATION_LABS,
        Interpretation::Srgb => VIPS_INTERPRETATION_SRGB,
        Interpretation::Yxy => VIPS_INTERPRETATION_YXY,
        Interpretation::Fourier => VIPS_INTERPRETATION_FOURIER,
        Interpretation::Rgb16 => VIPS_INTERPRETATION_RGB16,
        Interpretation::Grey16 => VIPS_INTERPRETATION_GREY16,
        Interpretation::Matrix => VIPS_INTERPRETATION_MATRIX,
        Interpretation::Scrgb => VIPS_INTERPRETATION_SCRGB,
        Interpretation::Hsv => VIPS_INTERPRETATION_HSV,
    }
}

const fn vips_to_interpretation(value: i32) -> Option<Interpretation> {
    match value {
        VIPS_INTERPRETATION_MULTIBAND => Some(Interpretation::Multiband),
        VIPS_INTERPRETATION_B_W => Some(Interpretation::BW),
        VIPS_INTERPRETATION_HISTOGRAM => Some(Interpretation::Histogram),
        VIPS_INTERPRETATION_XYZ => Some(Interpretation::Xyz),
        VIPS_INTERPRETATION_LAB => Some(Interpretation::Lab),
        VIPS_INTERPRETATION_CMYK => Some(Interpretation::Cmyk),
        VIPS_INTERPRETATION_LABQ => Some(Interpretation::Labq),
        VIPS_INTERPRETATION_RGB => Some(Interpretation::Rgb),
        VIPS_INTERPRETATION_CMC => Some(Interpretation::Cmc),
        VIPS_INTERPRETATION_LCH => Some(Interpretation::Lch),
        VIPS_INTERPRETATION_LABS => Some(Interpretation::Labs),
        VIPS_INTERPRETATION_SRGB => Some(Interpretation::Srgb),
        VIPS_INTERPRETATION_YXY => Some(Interpretation::Yxy),
        VIPS_INTERPRETATION_FOURIER => Some(Interpretation::Fourier),
        VIPS_INTERPRETATION_RGB16 => Some(Interpretation::Rgb16),
        VIPS_INTERPRETATION_GREY16 => Some(Interpretation::Grey16),
        VIPS_INTERPRETATION_MATRIX => Some(Interpretation::Matrix),
        VIPS_INTERPRETATION_SCRGB => Some(Interpretation::Scrgb),
        VIPS_INTERPRETATION_HSV => Some(Interpretation::Hsv),
        _ => None,
    }
}

fn read_i32(src: &[u8], endian: HeaderEndianness) -> i32 {
    let bytes: [u8; 4] = [src[0], src[1], src[2], src[3]];
    match endian {
        HeaderEndianness::Little => i32::from_le_bytes(bytes),
        HeaderEndianness::Big => i32::from_be_bytes(bytes),
    }
}

fn read_f32(src: &[u8], endian: HeaderEndianness) -> f32 {
    let bytes: [u8; 4] = [src[0], src[1], src[2], src[3]];
    match endian {
        HeaderEndianness::Little => f32::from_le_bytes(bytes),
        HeaderEndianness::Big => f32::from_be_bytes(bytes),
    }
}

const fn write_i16(dst: &mut [u8], value: i16, endian: HeaderEndianness) {
    let bytes = match endian {
        HeaderEndianness::Little => value.to_le_bytes(),
        HeaderEndianness::Big => value.to_be_bytes(),
    };
    dst.copy_from_slice(&bytes);
}

const fn write_i32(dst: &mut [u8], value: i32, endian: HeaderEndianness) {
    let bytes = match endian {
        HeaderEndianness::Little => value.to_le_bytes(),
        HeaderEndianness::Big => value.to_be_bytes(),
    };
    dst.copy_from_slice(&bytes);
}

const fn write_u32(dst: &mut [u8], value: u32, endian: HeaderEndianness) {
    let bytes = match endian {
        HeaderEndianness::Little => value.to_le_bytes(),
        HeaderEndianness::Big => value.to_be_bytes(),
    };
    dst.copy_from_slice(&bytes);
}

const fn write_f32(dst: &mut [u8], value: f32, endian: HeaderEndianness) {
    let bytes = match endian {
        HeaderEndianness::Little => value.to_le_bytes(),
        HeaderEndianness::Big => value.to_be_bytes(),
    };
    dst.copy_from_slice(&bytes);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use viprs_core::format::{F32, F64, I16, I32, U8, U16, U32};

    fn assert_round_trip<F: BandFormat>(width: u32, height: u32, bands: u32, pixels: Vec<F::Sample>)
    where
        F::Sample: PartialEq + bytemuck::Pod,
    {
        let codec = VipsCodec;
        let image = Image::<F>::from_buffer(width, height, bands, pixels.clone())
            .expect("test image should be valid");
        let encoded = codec.encode(&image).expect("encode should succeed");
        let decoded = codec.decode::<F>(&encoded).expect("decode should succeed");

        assert_eq!(decoded.width(), width);
        assert_eq!(decoded.height(), height);
        assert_eq!(decoded.bands(), bands);
        assert!(decoded.pixels() == pixels.as_slice());
    }

    #[test]
    fn sniff_accepts_native_magic_values() {
        assert!(VipsCodec.sniff(&VIPS_MAGIC_INTEL_BYTES));
        assert!(VipsCodec.sniff(&VIPS_MAGIC_SPARC_BYTES));
        assert!(!VipsCodec.sniff(b"RAW!"));
    }

    #[test]
    fn round_trip_u8() {
        assert_round_trip::<U8>(3, 2, 2, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn round_trip_u16() {
        assert_round_trip::<U16>(2, 2, 1, vec![1, 257, 1024, 65535]);
    }

    #[test]
    fn round_trip_i16() {
        assert_round_trip::<I16>(2, 2, 1, vec![-1, 0, 1, 32767]);
    }

    #[test]
    fn round_trip_u32() {
        assert_round_trip::<U32>(2, 2, 1, vec![1, 2, 3, u32::MAX]);
    }

    #[test]
    fn round_trip_i32() {
        assert_round_trip::<I32>(2, 2, 1, vec![i32::MIN, -1, 0, i32::MAX]);
    }

    #[test]
    fn round_trip_f32() {
        assert_round_trip::<F32>(2, 2, 1, vec![0.0, 1.5, -2.75, std::f32::consts::PI]);
    }

    #[test]
    fn round_trip_f64() {
        assert_round_trip::<F64>(2, 2, 1, vec![0.0, 1.5, -2.75, std::f64::consts::TAU]);
    }

    #[test]
    fn round_trip_preserves_basic_metadata() {
        let codec = VipsCodec;
        let metadata = ImageMetadata {
            interpretation: Some(Interpretation::Srgb),
            xres: Some(12.5),
            yres: Some(7.25),
            ..ImageMetadata::default()
        };
        let image = Image::<U8>::from_buffer(2, 1, 3, vec![1, 2, 3, 4, 5, 6])
            .expect("test image")
            .with_metadata(metadata.clone());

        let encoded = codec.encode(&image).expect("encode");
        let decoded = codec.decode::<U8>(&encoded).expect("decode");

        assert_eq!(decoded.metadata().interpretation, metadata.interpretation);
        assert_eq!(decoded.metadata().xres, metadata.xres);
        assert_eq!(decoded.metadata().yres, metadata.yres);
    }

    #[test]
    fn decode_errors_on_band_format_mismatch() {
        let codec = VipsCodec;
        let image = Image::<U16>::from_buffer(2, 1, 1, vec![7, 9]).expect("image");
        let encoded = codec.encode(&image).expect("encode");
        let err = codec.decode::<U8>(&encoded).expect_err("must fail");
        assert!(
            err.to_string().contains("does not match requested"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn decode_errors_on_unsupported_coding() {
        let codec = VipsCodec;
        let image = Image::<U8>::from_buffer(1, 1, 1, vec![42]).expect("image");
        let mut encoded = codec.encode(&image).expect("encode");
        let endian = if encoded[0..4] == VIPS_MAGIC_INTEL_BYTES {
            HeaderEndianness::Little
        } else {
            HeaderEndianness::Big
        };
        write_i32(&mut encoded[24..28], 2, endian);

        let err = codec.decode::<U8>(&encoded).expect_err("must fail");
        assert!(
            err.to_string().contains("unsupported coding"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn registry_round_trip_for_vips_extensions() {
        let path_v = Path::new("target/foreign-registry-unit-tests/vips-native.v");
        let path_vips = Path::new("target/foreign-registry-unit-tests/vips-native.vips");
        std::fs::create_dir_all("target/foreign-registry-unit-tests").expect("dir");

        let image = Image::<U8>::from_buffer(2, 2, 1, vec![1, 2, 3, 4]).expect("image");
        let registry = crate::registry::ForeignRegistry::default();
        registry.save(&image, path_v).expect("save .v");
        let decoded = registry.load(path_v).expect("load .v");
        registry.save(&decoded, path_vips).expect("save .vips");
        let decoded_vips = registry.load(path_vips).expect("load .vips");

        assert_eq!(decoded_vips.pixels(), &[1, 2, 3, 4]);

        std::fs::remove_file(path_v).expect("cleanup .v");
        std::fs::remove_file(path_vips).expect("cleanup .vips");
    }
}
