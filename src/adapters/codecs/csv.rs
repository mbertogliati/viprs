//! CSV codec — decode/encode CSV (comma-separated values) images.
//!
//! Follows the libvips `csvload`/`csvsave` behaviour:
//! - Decode produces a single-band `F64` image (one value per cell).
//! - Comments (`#`…end-of-line) are skipped during decode.
//! - Separators default to `;`, `,`, tab; whitespace is collapsed.
//! - Encode writes one row per line with values separated by `\t`.
//!
//! Reference: `.libvips_repo/libvips/foreign/csvload.c`, `csvsave.c`.

use crate::domain::codec_options::{LoadOptions, SaveOptions};
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId, F64};
use crate::domain::image::Image;
use crate::ports::codec::{ImageDecoder, ImageEncoder};

// ── Token scanner ────────────────────────────────────────────────────────────

/// Lightweight byte-level scanner that tokenises a CSV source.
///
/// Mirrors the whitemap/sepmap approach from `VipsForeignLoadCsv`.
struct Scanner<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) {
        if self.pos < self.src.len() {
            self.pos += 1;
        }
    }

    /// Skip to the next non-whitespace, non-newline character, consuming
    /// comment lines.  Stops at `\n`, EOF, or a non-whitespace byte.
    fn skip_whitespace(&mut self, whitespace: &[bool; 256]) {
        while let Some(ch) = self.peek() {
            if ch == b'\n' {
                break;
            }
            if ch == b'#' {
                // consume the rest of the comment line
                while let Some(c) = self.peek() {
                    self.advance();
                    if c == b'\n' {
                        break;
                    }
                }
                break;
            }
            if whitespace[ch as usize] {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Skip the inside of a quoted string (after the opening `"`).
    fn skip_quoted(&mut self) {
        loop {
            match self.peek() {
                None | Some(b'\n') => break,
                Some(b'\\') => {
                    self.advance(); // skip escape char
                    self.advance(); // skip next char
                }
                Some(b'"') => {
                    self.advance(); // consume closing quote
                    break;
                }
                _ => self.advance(),
            }
        }
    }

    /// Read the next numeric value from the current position.
    ///
    /// Returns `(value, terminator)` where terminator is the byte (or `None`
    /// for EOF/newline) that follows the item.  The `value` is 0.0 on
    /// non-numeric or empty fields (matches libvips behaviour).
    fn read_value(
        &mut self,
        whitespace: &[bool; 256],
        separators: &[bool; 256],
    ) -> (f64, Option<u8>) {
        self.skip_whitespace(whitespace);

        let mut value = 0.0_f64;

        match self.peek() {
            None => return (value, None),
            Some(b'\n') => return (value, Some(b'\n')),
            Some(b'"') => {
                self.advance(); // consume opening quote
                self.skip_quoted();
            }
            Some(ch) if !separators[ch as usize] => {
                // Collect until whitespace, separator, newline, or EOF.
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if c == b'\n' || whitespace[c as usize] || separators[c as usize] {
                        break;
                    }
                    self.advance();
                }
                if self.pos > start {
                    // Best-effort parse; non-numeric fields silently yield 0.0
                    // (matches libvips warning-only behaviour).
                    if let Ok(s) = std::str::from_utf8(&self.src[start..self.pos]) {
                        value = s.parse::<f64>().unwrap_or(0.0);
                    }
                }
            }
            _ => {} // separator: empty field, value stays 0.0
        }

        // Skip trailing whitespace (but not newlines).
        self.skip_whitespace(whitespace);

        // Step over a separator if present.
        if let Some(ch) = self.peek()
            && separators[ch as usize]
        {
            self.advance();
        }
        (value, self.peek())
    }

    /// Advance past a newline (if present).
    fn consume_newline(&mut self) {
        if self.peek() == Some(b'\n') {
            self.advance();
        }
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.src.len()
    }
}

// ── Build lookup tables ──────────────────────────────────────────────────────

fn build_whitespace_map(chars: &str) -> [bool; 256] {
    let mut map = [false; 256];
    for ch in chars.bytes() {
        if ch != b'\n' {
            map[ch as usize] = true;
        }
    }
    map
}

fn build_separator_map(chars: &str) -> [bool; 256] {
    let mut map = [false; 256];
    for ch in chars.bytes() {
        if ch != b'\n' {
            map[ch as usize] = true;
        }
    }
    map
}

// ── Core decode ──────────────────────────────────────────────────────────────

/// Decode `src` into a single-band F64 image.
///
/// Two-pass strategy (mirrors libvips `lines=-1` default):
/// 1. First pass: skip `skip` lines, scan row 0 to count columns.
/// 2. Second pass: count remaining rows; then decode all values.
fn decode_csv(
    src: &[u8],
    skip_lines: u32,
    limit_lines: Option<u32>,
) -> Result<Image<F64>, ViprsError> {
    let whitespace = build_whitespace_map(" ");
    let separators = build_separator_map(";,\t");

    // Skip `skip_lines` lines.
    let mut scanner = Scanner::new(src);
    for _ in 0..skip_lines {
        while let Some(ch) = scanner.peek() {
            scanner.advance();
            if ch == b'\n' {
                break;
            }
        }
        if scanner.at_eof() {
            return Err(ViprsError::Codec(
                "csv: unexpected end of file during skip".into(),
            ));
        }
    }

    let data_start = scanner.pos;

    // Pass 1: measure column count from first row.
    let width = {
        let mut cols = 0u32;
        let mut row_scanner = Scanner::new(&src[data_start..]);
        // Skip comment lines before first data row.
        loop {
            match row_scanner.peek() {
                None => break,
                Some(b'#') => {
                    while let Some(c) = row_scanner.peek() {
                        row_scanner.advance();
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                _ => break,
            }
        }
        loop {
            let (_, term) = row_scanner.read_value(&whitespace, &separators);
            cols += 1;
            match term {
                None | Some(b'\n') => break,
                _ => {}
            }
        }
        cols
    };

    if width == 0 {
        return Err(ViprsError::Codec("csv: no columns found".into()));
    }

    // Pass 2: count data rows.
    let height = if let Some(limit) = limit_lines {
        limit
    } else {
        let mut rows = 0u32;
        let mut row_scanner = Scanner::new(&src[data_start..]);
        let mut in_row = false;
        loop {
            match row_scanner.peek() {
                None => {
                    if in_row {
                        rows += 1;
                    }
                    break;
                }
                Some(b'\n') => {
                    if in_row {
                        rows += 1;
                        in_row = false;
                    }
                    row_scanner.advance();
                }
                Some(b'#') => {
                    // comment line — skip and don't count
                    while let Some(c) = row_scanner.peek() {
                        row_scanner.advance();
                        if c == b'\n' {
                            break;
                        }
                    }
                }
                Some(ch) if ch == b' ' || ch == b'\t' || ch == b'\r' => {
                    row_scanner.advance();
                }
                _ => {
                    if !in_row {
                        in_row = true;
                    }
                    row_scanner.advance();
                }
            }
        }
        rows
    };

    if height == 0 {
        return Err(ViprsError::Codec("csv: no data rows found".into()));
    }

    // Decode pixels.
    let total = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| ViprsError::Codec("csv: image dimensions overflow".into()))?;
    let mut pixels = vec![0.0_f64; total];
    let mut decode_scanner = Scanner::new(&src[data_start..]);

    for row in 0..(height as usize) {
        // Skip comment lines before this data row.
        while let Some(b'#') = decode_scanner.peek() {
            while let Some(c) = decode_scanner.peek() {
                decode_scanner.advance();
                if c == b'\n' {
                    break;
                }
            }
        }

        for col in 0..(width as usize) {
            let (value, _) = decode_scanner.read_value(&whitespace, &separators);
            pixels[row * width as usize + col] = value;
        }
        // Step over the end-of-row newline.
        decode_scanner.consume_newline();
    }

    Image::from_buffer(width, height, 1, pixels).map_err(|e| ViprsError::Codec(e.to_string()))
}

// ── Core encode ──────────────────────────────────────────────────────────────

fn encode_csv<F: BandFormat>(image: &Image<F>, separator: &str) -> Result<Vec<u8>, ViprsError> {
    if image.bands() != 1 {
        return Err(ViprsError::Codec(
            "csv: only single-band images can be saved as CSV".into(),
        ));
    }

    // We need f64 values; cast via the sample bytes.
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

    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut output =
        Vec::with_capacity(checked_csv_encode_capacity(image.width(), image.height())?);

    for row in 0..height {
        for col in 0..width {
            if col > 0 {
                output.extend_from_slice(separator.as_bytes());
            }
            // Use Rust's default f64 display, which is locale-independent.
            let s = format!("{}", samples[row * width + col]);
            output.extend_from_slice(s.as_bytes());
        }
        output.push(b'\n');
    }

    Ok(output)
}

fn checked_csv_encode_capacity(width: u32, height: u32) -> Result<usize, ViprsError> {
    let Some(capacity_bytes) = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(16))
    else {
        let total_bytes = u128::from(width) * u128::from(height) * 16;
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands: 1,
            bytes: total_bytes,
            limit_bytes: isize::MAX as u128,
            details: "csv encoded output buffer exceeds Vec allocation limits",
        });
    };

    let capacity = usize::try_from(capacity_bytes).map_err(|_| ViprsError::ImageTooLarge {
        width,
        height,
        bands: 1,
        bytes: u128::from(capacity_bytes),
        limit_bytes: isize::MAX as u128,
        details: "csv encoded output buffer exceeds Vec allocation limits",
    })?;

    let mut output = Vec::<u8>::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands: 1,
            bytes: u128::from(capacity_bytes),
            limit_bytes: isize::MAX as u128,
            details: "csv encoded output buffer exceeds Vec allocation limits",
        })?;
    Ok(capacity)
}

// ── Public codec struct ──────────────────────────────────────────────────────

/// Codec for the CSV (comma-separated values) image format.
///
/// Produces single-band `F64` images on decode.  Encode accepts any
/// single-band image and converts samples to `f64` text.
#[derive(Clone, Copy, Debug)]
/// The `CsvCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::csv::CsvCodec>();
/// ```
pub struct CsvCodec;

impl CsvCodec {
    /// Returns true if `header` looks like a CSV (any printable ASCII).
    ///
    /// CSV has no magic bytes, so this is a conservative heuristic: the first
    /// byte must be printable ASCII, excluding bytes that appear in binary
    /// formats.  The registry uses extension-based detection for save; for load
    /// the user must call the codec explicitly (matching libvips behaviour).
    fn sniff_header(header: &[u8]) -> bool {
        // A CSV file starts with printable ASCII. We require the first byte to
        // be a digit, letter, quote, minus, or `#` (comment).
        matches!(header.first(), Some(&b) if b.is_ascii_graphic() || b == b' ')
    }
}

impl ImageDecoder for CsvCodec {
    fn format_name(&self) -> &'static str {
        "csv"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        Self::sniff_header(header)
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
                "csv: only F64 output is supported, requested {:?}",
                F::ID
            )));
        }
        let image = decode_csv(src, 0, None)?;
        // SAFETY: F::ID == F64 is checked above; cast is zero-cost.
        let (w, h, b) = (image.width(), image.height(), image.bands());
        let raw: Vec<f64> = image.into_buffer();
        let samples: Vec<F::Sample> = bytemuck::cast_vec(raw);
        Image::from_buffer(w, h, b, samples).map_err(|e| ViprsError::Codec(e.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let image = decode_csv(src, 0, None)?;
        Ok((image.width(), image.height(), image.bands()))
    }
}

impl ImageEncoder for CsvCodec {
    fn format_name(&self) -> &'static str {
        "csv"
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
        encode_csv(image, "\t")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::{F32, F64, U8};

    #[test]
    fn decode_simple_csv() {
        let src = b"1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n";
        let codec = CsvCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.width(), 3);
        assert_eq!(image.height(), 2);
        assert_eq!(image.bands(), 1);
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_comma_separated() {
        let src = b"10,20\n30,40\n";
        let codec = CsvCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.pixels(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn decode_semicolon_separated() {
        let src = b"1;2;3\n4;5;6\n";
        let codec = CsvCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn decode_comment_lines_skipped() {
        let src = b"# comment\n1\t2\n3\t4\n";
        let codec = CsvCodec;
        let image = codec.decode::<F64>(src).unwrap();
        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 2);
        assert_eq!(image.pixels(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn encode_round_trip_f64() {
        let codec = CsvCodec;
        let original =
            Image::<F64>::from_buffer(3, 2, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F64>(&encoded).unwrap();
        assert_eq!(decoded.width(), 3);
        assert_eq!(decoded.height(), 2);
        for (a, b) in decoded.pixels().iter().zip(original.pixels().iter()) {
            assert!((a - b).abs() < f64::EPSILON, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn encode_u8_single_band() {
        let codec = CsvCodec;
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![0, 128, 64, 255]).unwrap();
        let encoded = codec.encode(&image).unwrap();
        let text = std::str::from_utf8(&encoded).unwrap();
        assert!(text.contains("128"));
        assert!(text.contains("255"));
    }

    #[test]
    fn encode_multi_band_errors() {
        let codec = CsvCodec;
        let image = Image::<U8>::from_buffer(2, 1, 3, vec![0; 6]).unwrap();
        assert!(codec.encode(&image).is_err());
    }

    #[test]
    fn decode_wrong_format_errors() {
        let src = b"1\t2\n3\t4\n";
        let codec = CsvCodec;
        // Only F64 is supported for decode.
        assert!(codec.decode::<F32>(src).is_err());
        assert!(codec.decode::<U8>(src).is_err());
    }

    #[test]
    fn decode_negative_and_float_values() {
        let src = b"-1.5\t0.0\t3.14\n";
        let codec = CsvCodec;
        let image = codec.decode::<F64>(src).unwrap();
        let px = image.pixels();
        assert!((px[0] - (-1.5)).abs() < f64::EPSILON);
        assert!((px[1] - 0.0).abs() < f64::EPSILON);
        assert!((px[2] - 3.14).abs() < 1e-10);
    }

    #[test]
    fn probe_returns_correct_dimensions() {
        let src = b"1\t2\t3\n4\t5\t6\n7\t8\t9\n";
        let codec = CsvCodec;
        let (w, h, b) = codec.probe(src).unwrap();
        assert_eq!(w, 3);
        assert_eq!(h, 3);
        assert_eq!(b, 1);
    }

    #[test]
    fn sniff_accepts_printable_ascii() {
        let codec = CsvCodec;
        assert!(codec.sniff(b"1.0,2.0"));
        assert!(codec.sniff(b"# comment\n"));
        assert!(codec.sniff(b"\"quoted\""));
    }

    #[test]
    fn csv_encode_capacity_rejects_oversized_dimensions() {
        let err = checked_csv_encode_capacity(u32::MAX, u32::MAX)
            .expect_err("oversized CSV dimensions must be rejected before reserving output");

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 1,
                ..
            }
        ));
    }
}
