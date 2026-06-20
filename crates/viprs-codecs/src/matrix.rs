//! Matrix codec — decode/encode libvips `.mat` matrix files.
//!
//! Format specification (from libvips `matrixload`/`matrixsave`):
//! - First line: `width height [scale [offset]]` — integers for width/height,
//!   optional floats for scale (default 1.0) and offset (default 0.0).
//! - Subsequent lines: space-separated `f64` values; one row per line.
//! - All numbers use `.` as decimal separator (C locale).
//! - Scale and offset are stored as image metadata `"scale"` / `"offset"`.
//!
//! Output is a single-band `F64` image.
//!
//! Reference: `.libvips_repo/libvips/foreign/matrixload.c`, `matrixsave.c`.

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, F64};
use viprs_core::image::{Image, ImageMetadata};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

// ── Header parse ─────────────────────────────────────────────────────────────

struct ParsedMatrixHeader {
    width: u32,
    height: u32,
    scale: f64,
    offset: f64,
    data_offset: usize,
}

/// Parse tokens from a byte slice, treating `' '`, `'\t'`, `','`, `'"'` as
/// delimiters (matches libvips `vips_break_token` usage).
fn parse_matrix_header(src: &[u8]) -> Result<ParsedMatrixHeader, ViprsError> {
    // Find the end of the first line.
    let line_end = src.iter().position(|&b| b == b'\n').unwrap_or(src.len());
    let line = std::str::from_utf8(&src[..line_end])
        .map_err(|_| ViprsError::Codec("matrix: header is not valid UTF-8".into()))?
        .trim();

    // Tokenise on spaces, tabs, commas, quotes.
    let tokens: Vec<&str> = line
        .split([' ', '\t', ',', '"'])
        .filter(|s| !s.is_empty())
        .collect();

    if tokens.len() < 2 {
        return Err(ViprsError::Codec(
            "matrix: header must have at least width and height".into(),
        ));
    }

    let width_f: f64 = tokens[0]
        .parse()
        .map_err(|_| ViprsError::Codec(format!("matrix: invalid width '{}'", tokens[0])))?;
    let height_f: f64 = tokens[1]
        .parse()
        .map_err(|_| ViprsError::Codec(format!("matrix: invalid height '{}'", tokens[1])))?;

    if width_f.fract() != 0.0 || height_f.fract() != 0.0 {
        return Err(ViprsError::Codec(
            "matrix: width and height must be integers".into(),
        ));
    }

    let width = width_f as i64;
    let height = height_f as i64;

    if width <= 0 || width > 100_000 || height <= 0 || height > 100_000 {
        return Err(ViprsError::Codec(format!(
            "matrix: width ({width}) and height ({height}) must be in range 1..=100000"
        )));
    }

    let scale = if tokens.len() > 2 {
        tokens[2]
            .parse::<f64>()
            .map_err(|_| ViprsError::Codec(format!("matrix: invalid scale '{}'", tokens[2])))?
    } else {
        1.0
    };

    if scale == 0.0 {
        return Err(ViprsError::Codec("matrix: scale must not be zero".into()));
    }

    let offset = if tokens.len() > 3 {
        tokens[3]
            .parse::<f64>()
            .map_err(|_| ViprsError::Codec(format!("matrix: invalid offset '{}'", tokens[3])))?
    } else {
        0.0
    };

    let data_offset = if line_end < src.len() {
        line_end + 1
    } else {
        src.len()
    };

    Ok(ParsedMatrixHeader {
        width: width as u32,
        height: height as u32,
        scale,
        offset,
        data_offset,
    })
}

// ── Core decode ──────────────────────────────────────────────────────────────

fn decode_matrix(src: &[u8]) -> Result<Image<F64>, ViprsError> {
    let header = parse_matrix_header(src)?;
    let total = (header.width as usize)
        .checked_mul(header.height as usize)
        .ok_or_else(|| ViprsError::Codec("matrix: image dimensions overflow".into()))?;

    let mut pixels = Vec::with_capacity(total);
    let data = &src[header.data_offset..];

    for (row_idx, line_bytes) in data.split(|&b| b == b'\n').enumerate() {
        if row_idx >= header.height as usize {
            break;
        }
        let line = std::str::from_utf8(line_bytes)
            .map_err(|_| ViprsError::Codec(format!("matrix: row {row_idx} is not valid UTF-8")))?;

        let values: Vec<&str> = line
            .split([' ', '\t', ',', '"'])
            .filter(|s| !s.is_empty())
            .collect();

        if values.len() < header.width as usize {
            return Err(ViprsError::Codec(format!(
                "matrix: row {row_idx} has {} columns, expected {}",
                values.len(),
                header.width
            )));
        }

        for (col, &token) in values[..header.width as usize].iter().enumerate() {
            let v = token.parse::<f64>().map_err(|_| {
                ViprsError::Codec(format!(
                    "matrix: bad number '{token}' at row {row_idx} col {col}"
                ))
            })?;
            pixels.push(v);
        }
    }

    if pixels.len() < total {
        return Err(ViprsError::Codec(format!(
            "matrix: expected {} pixels, got {}",
            total,
            pixels.len()
        )));
    }

    let mut metadata = ImageMetadata::default();
    metadata
        .extra
        .insert("scale".to_string(), header.scale.to_string());
    metadata
        .extra
        .insert("offset".to_string(), header.offset.to_string());

    Image::from_buffer(header.width, header.height, 1, pixels)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

// ── Core encode ──────────────────────────────────────────────────────────────

fn encode_matrix<F: BandFormat>(image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
    if image.bands() != 1 {
        return Err(ViprsError::Codec(
            "matrix: only single-band images can be saved in matrix format".into(),
        ));
    }

    let scale = image
        .metadata()
        .extra
        .get("scale")
        .map_or(Ok(1.0_f64), |value| {
            value.parse::<f64>().map_err(|_| {
                ViprsError::Codec(format!(
                    "matrix: metadata extra['scale'] is not a valid number: '{value}'"
                ))
            })
        })?;
    let offset = image
        .metadata()
        .extra
        .get("offset")
        .map_or(Ok(0.0_f64), |value| {
            value.parse::<f64>().map_err(|_| {
                ViprsError::Codec(format!(
                    "matrix: metadata extra['offset'] is not a valid number: '{value}'"
                ))
            })
        })?;

    if scale == 0.0 {
        return Err(ViprsError::Codec(
            "matrix: metadata extra['scale'] must not be zero".into(),
        ));
    }

    let samples: Vec<f64> = match F::ID {
        BandFormatId::F64 => bytemuck::cast_slice::<F::Sample, f64>(image.pixels()).to_vec(),
        BandFormatId::F32 => bytemuck::cast_slice::<F::Sample, f32>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
        BandFormatId::U8 => bytemuck::cast_slice::<F::Sample, u8>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
        BandFormatId::U16 => bytemuck::cast_slice::<F::Sample, u16>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
        BandFormatId::I16 => bytemuck::cast_slice::<F::Sample, i16>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
        BandFormatId::U32 => bytemuck::cast_slice::<F::Sample, u32>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
        BandFormatId::I32 => bytemuck::cast_slice::<F::Sample, i32>(image.pixels())
            .iter()
            .map(|&s| f64::from(s))
            .collect(),
    };

    let mut output = Vec::new();

    // Header line: `width height [scale offset]` (omit if defaults).
    if scale != 1.0 || offset != 0.0 {
        output.extend_from_slice(
            format!(
                "{} {} {} {}\n",
                image.width(),
                image.height(),
                scale,
                offset
            )
            .as_bytes(),
        );
    } else {
        output.extend_from_slice(format!("{} {}\n", image.width(), image.height()).as_bytes());
    }

    let width = image.width() as usize;
    let height = image.height() as usize;

    for row in 0..height {
        for col in 0..width {
            if col > 0 {
                output.push(b' ');
            }
            let s = format!("{}", samples[row * width + col]);
            output.extend_from_slice(s.as_bytes());
        }
        output.push(b'\n');
    }

    Ok(output)
}

// ── Sniff heuristic ──────────────────────────────────────────────────────────

fn sniff_matrix(header: &[u8]) -> bool {
    // A matrix file starts with two integers on the first line.  We try a
    // lightweight parse rather than reading the full header.
    let line_end = header
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(header.len());
    let line = match std::str::from_utf8(&header[..line_end]) {
        Ok(s) => s.trim(),
        Err(_) => return false,
    };
    let mut tokens = line.split([' ', '\t', ',', '"']).filter(|s| !s.is_empty());
    let w_ok = tokens
        .next()
        .and_then(|t| t.parse::<f64>().ok())
        .is_some_and(|v| v.fract() == 0.0 && v > 0.0);
    let h_ok = tokens
        .next()
        .and_then(|t| t.parse::<f64>().ok())
        .is_some_and(|v| v.fract() == 0.0 && v > 0.0);
    w_ok && h_ok
}

// ── Public codec struct ──────────────────────────────────────────────────────

/// Codec for the libvips matrix format (`.mat` files).
///
/// Produces single-band `F64` images on decode with `"scale"` and `"offset"`
/// stored in image metadata `extra` map. Encode accepts any single-band image.
#[derive(Clone, Copy, Debug)]
/// The `MatrixCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::matrix::MatrixCodec>();
/// ```
pub struct MatrixCodec;

impl ImageDecoder for MatrixCodec {
    fn format_name(&self) -> &'static str {
        "matrix"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        sniff_matrix(header)
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        if F::ID != BandFormatId::F64 {
            return Err(ViprsError::Codec(format!(
                "matrix: only F64 output is supported, requested {:?}",
                F::ID
            )));
        }
        let image = decode_matrix(src)?;
        let metadata = image.metadata().clone();
        let (w, h, b) = (image.width(), image.height(), image.bands());
        let raw: Vec<f64> = image.into_buffer();
        let samples: Vec<F::Sample> = bytemuck::cast_vec(raw);
        Image::from_buffer(w, h, b, samples)
            .map(|decoded| decoded.with_metadata(metadata))
            .map_err(|e| ViprsError::Codec(e.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let header = parse_matrix_header(src)?;
        Ok((header.width, header.height, 1))
    }
}

impl ImageEncoder for MatrixCodec {
    fn format_name(&self) -> &'static str {
        "matrix"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        _opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        encode_matrix(image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{F32, F64, I16, I32, U8, U16, U32};

    const SIMPLE_MAT: &[u8] = b"3 2\n1 2 3\n4 5 6\n";

    fn codec_error_message(error: ViprsError) -> String {
        match error {
            ViprsError::Codec(message) => message,
            other => panic!("expected codec error, got {other:?}"),
        }
    }

    #[test]
    fn decode_simple_matrix() {
        let codec = MatrixCodec;
        let image = codec.decode::<F64>(SIMPLE_MAT).unwrap();
        assert_eq!(image.width(), 3);
        assert_eq!(image.height(), 2);
        assert_eq!(image.bands(), 1);
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_header_with_scale_and_offset() {
        let src = b"2 2 2.0 1.0\n1 2\n3 4\n";
        let codec = MatrixCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(image.metadata().extra.get("scale"), Some(&"2".to_string()));
        assert_eq!(image.metadata().extra.get("offset"), Some(&"1".to_string()));
    }

    #[test]
    fn decode_header_with_scale_only() {
        let src = b"2 1 3.0\n10 20\n";
        let codec = MatrixCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.pixels(), &[10.0, 20.0]);
    }

    #[test]
    fn encode_round_trip_f64() {
        let codec = MatrixCodec;
        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert("scale".to_string(), "2.5".to_string());
        metadata
            .extra
            .insert("offset".to_string(), "1.25".to_string());
        let original = Image::<F64>::from_buffer(3, 2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .with_metadata(metadata);
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F64>(&encoded).unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
        assert_eq!(
            decoded.metadata().extra.get("scale"),
            Some(&"2.5".to_string())
        );
        assert_eq!(
            decoded.metadata().extra.get("offset"),
            Some(&"1.25".to_string())
        );
        for (a, b) in decoded.pixels().iter().zip(original.pixels().iter()) {
            assert!((a - b).abs() < f64::EPSILON, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn encode_u8_single_band() {
        let codec = MatrixCodec;
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![0, 128, 64, 255]).unwrap();
        let encoded = codec.encode(&image).unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        // First line is the dimension header.
        assert!(text.starts_with("2 2\n"), "header mismatch: {text}");
        assert!(text.contains("128"));
    }

    #[test]
    fn encode_multi_band_errors() {
        let codec = MatrixCodec;
        let image = Image::<U8>::from_buffer(2, 1, 3, vec![0; 6]).unwrap();
        assert!(codec.encode(&image).is_err());
    }

    #[test]
    fn decode_wrong_format_errors() {
        let codec = MatrixCodec;
        assert!(codec.decode::<U8>(SIMPLE_MAT).is_err());
    }

    #[test]
    fn decode_short_row_errors() {
        let src = b"3 2\n1 2\n4 5 6\n"; // first row only has 2 values
        let codec = MatrixCodec;
        assert!(codec.decode::<F64>(src).is_err());
    }

    #[test]
    fn decode_supports_quotes_commas_and_tabs() {
        let src = b"\"2\",\t2,\t4.5,\t-1.25\n\"1\", 2, 99\n3,\t4,\t100\n";
        let codec = MatrixCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            image.metadata().extra.get("scale"),
            Some(&"4.5".to_string())
        );
        assert_eq!(
            image.metadata().extra.get("offset"),
            Some(&"-1.25".to_string())
        );
    }

    #[test]
    fn decode_missing_rows_errors() {
        let src = b"2 2\n1 2";
        let codec = MatrixCodec;
        let error = codec.decode::<F64>(src).unwrap_err();
        assert!(codec_error_message(error).contains("expected 4 pixels, got 2"));
    }

    #[test]
    fn decode_invalid_number_errors() {
        let src = b"2 1\n1 nope\n";
        let codec = MatrixCodec;
        let error = codec.decode::<F64>(src).unwrap_err();
        assert!(codec_error_message(error).contains("bad number 'nope'"));
    }

    #[test]
    fn decode_non_utf8_header_errors() {
        let src = b"2 \xFF\n1 2\n";
        let codec = MatrixCodec;
        let error = codec.decode::<F64>(src).unwrap_err();
        assert!(codec_error_message(error).contains("header is not valid UTF-8"));
    }

    #[test]
    fn decode_non_utf8_row_errors() {
        let src = b"2 1\n1 \xFF\n";
        let codec = MatrixCodec;
        let error = codec.decode::<F64>(src).unwrap_err();
        assert!(codec_error_message(error).contains("row 0 is not valid UTF-8"));
    }

    #[test]
    fn probe_returns_dimensions() {
        let codec = MatrixCodec;
        let (w, h, b) = codec.probe(SIMPLE_MAT).unwrap();
        assert_eq!((w, h, b), (3, 2, 1));
    }

    #[test]
    fn probe_accepts_header_without_newline() {
        let codec = MatrixCodec;
        let (w, h, b) = codec.probe(b"7 9").unwrap();
        assert_eq!((w, h, b), (7, 9, 1));
    }

    #[test]
    fn probe_rejects_out_of_range_dimensions() {
        let codec = MatrixCodec;
        let error = codec.probe(b"0 2\n1 2\n").unwrap_err();
        assert!(codec_error_message(error).contains("must be in range 1..=100000"));
    }

    #[test]
    fn sniff_accepts_valid_header() {
        let codec = MatrixCodec;
        assert!(codec.sniff(b"3 2\n1 2 3\n"));
        assert!(codec.sniff(b"100 200\n"));
        assert!(codec.sniff(b"\"3\",\t2"));
    }

    #[test]
    fn sniff_rejects_non_matrix() {
        let codec = MatrixCodec;
        // JPEG magic
        assert!(!codec.sniff(&[0xFF, 0xD8, 0xFF]));
        // CSV header line without two integers
        assert!(!codec.sniff(b"1.5,2.5\n"));
        assert!(!codec.sniff(b"0 2\n"));
        assert!(!codec.sniff(b"2.5 3\n"));
    }

    #[test]
    fn decode_header_invalid_scale_zero_errors() {
        let src = b"2 2 0.0\n1 2\n3 4\n";
        let codec = MatrixCodec;
        assert!(codec.decode::<F64>(src).is_err());
    }

    #[test]
    fn encode_header_has_dimensions() {
        let mut metadata = ImageMetadata::default();
        metadata.extra.insert("scale".to_string(), "3".to_string());
        metadata.extra.insert("offset".to_string(), "1".to_string());
        let image = Image::<F64>::from_buffer(2, 1, 1, vec![1.0, 2.0])
            .unwrap()
            .with_metadata(metadata);
        let codec = MatrixCodec;
        let encoded = codec.encode(&image).unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        assert!(
            text.starts_with("2 1 3 1\n"),
            "expected header '2 1 3 1': {text}"
        );
    }

    #[test]
    fn encode_default_metadata_omits_scale_and_offset() {
        let codec = MatrixCodec;
        let image = Image::<F64>::from_buffer(2, 1, 1, vec![1.5, 2.5]).unwrap();
        let encoded = codec.encode(&image).unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        assert!(text.starts_with("2 1\n"));
        assert!(!text.starts_with("2 1 1 0\n"));
    }

    #[test]
    fn encode_invalid_scale_metadata_errors() {
        let codec = MatrixCodec;
        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert("scale".to_string(), "not-a-number".to_string());
        let image = Image::<F64>::from_buffer(1, 1, 1, vec![1.0])
            .unwrap()
            .with_metadata(metadata);
        let error = codec.encode(&image).unwrap_err();
        assert!(codec_error_message(error).contains("extra['scale'] is not a valid number"));
    }

    #[test]
    fn encode_invalid_offset_metadata_errors() {
        let codec = MatrixCodec;
        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert("offset".to_string(), "not-a-number".to_string());
        let image = Image::<F64>::from_buffer(1, 1, 1, vec![1.0])
            .unwrap()
            .with_metadata(metadata);
        let error = codec.encode(&image).unwrap_err();
        assert!(codec_error_message(error).contains("extra['offset'] is not a valid number"));
    }

    #[test]
    fn encode_zero_scale_metadata_errors() {
        let codec = MatrixCodec;
        let mut metadata = ImageMetadata::default();
        metadata.extra.insert("scale".to_string(), "0".to_string());
        let image = Image::<F64>::from_buffer(1, 1, 1, vec![1.0])
            .unwrap()
            .with_metadata(metadata);
        let error = codec.encode(&image).unwrap_err();
        assert!(codec_error_message(error).contains("must not be zero"));
    }

    #[test]
    fn encode_supports_all_numeric_band_formats() {
        let codec = MatrixCodec;

        let f32_image = Image::<F32>::from_buffer(2, 1, 1, vec![1.25, -2.5]).unwrap();
        let f32_encoded = codec.encode(&f32_image).unwrap();
        let f32_text = std::str::from_utf8(&f32_encoded).unwrap();
        assert!(f32_text.contains("1.25 -2.5"));

        let u16_image = Image::<U16>::from_buffer(2, 1, 1, vec![1, 65535]).unwrap();
        let u16_encoded = codec.encode(&u16_image).unwrap();
        let u16_text = std::str::from_utf8(&u16_encoded).unwrap();
        assert!(u16_text.contains("1 65535"));

        let i16_image = Image::<I16>::from_buffer(2, 1, 1, vec![-2, 3]).unwrap();
        let i16_encoded = codec.encode(&i16_image).unwrap();
        let i16_text = std::str::from_utf8(&i16_encoded).unwrap();
        assert!(i16_text.contains("-2 3"));

        let u32_image = Image::<U32>::from_buffer(2, 1, 1, vec![42, 1_000_000]).unwrap();
        let u32_encoded = codec.encode(&u32_image).unwrap();
        let u32_text = std::str::from_utf8(&u32_encoded).unwrap();
        assert!(u32_text.contains("42 1000000"));

        let i32_image = Image::<I32>::from_buffer(2, 1, 1, vec![-42, 7]).unwrap();
        let i32_encoded = codec.encode(&i32_image).unwrap();
        let i32_text = std::str::from_utf8(&i32_encoded).unwrap();
        assert!(i32_text.contains("-42 7"));
    }
}
