//! Netpbm codec family — decode P1-P6 and encode PBM/PGM/PPM binary streams.

use viprs_core::codec_options::{LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, U8, U16};
use viprs_core::image::{InMemoryImage, ImageMetadata, Interpretation};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PnmMagic {
    P1,
    P2,
    P3,
    P4,
    P5,
    P6,
}

impl PnmMagic {
    fn parse(token: &[u8]) -> Result<Self, ViprsError> {
        match token {
            b"P1" => Ok(Self::P1),
            b"P2" => Ok(Self::P2),
            b"P3" => Ok(Self::P3),
            b"P4" => Ok(Self::P4),
            b"P5" => Ok(Self::P5),
            b"P6" => Ok(Self::P6),
            _ => Err(ViprsError::Codec("pnm: unsupported magic number".into())),
        }
    }

    const fn bands(self) -> u32 {
        match self {
            Self::P1 | Self::P2 | Self::P4 | Self::P5 => 1,
            Self::P3 | Self::P6 => 3,
        }
    }

    const fn is_ascii(self) -> bool {
        matches!(self, Self::P1 | Self::P2 | Self::P3)
    }

    const fn is_bitmap(self) -> bool {
        matches!(self, Self::P1 | Self::P4)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PnmEncodeKind {
    Pbm,
    Pgm,
    Ppm,
    Pnm,
}

#[derive(Clone, Copy, Debug)]
struct ParsedPnm {
    magic: PnmMagic,
    width: u32,
    height: u32,
    max_value: Option<u32>,
    data_offset: usize,
}

struct TokenCursor<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> TokenCursor<'a> {
    const fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn next_token(&mut self) -> Option<&'a [u8]> {
        while self.pos < self.src.len() {
            match self.src[self.pos] {
                byte if byte.is_ascii_whitespace() => self.pos += 1,
                b'#' => {
                    while self.pos < self.src.len() && self.src[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }

        let start = self.pos;
        while self.pos < self.src.len() {
            let byte = self.src[self.pos];
            if byte.is_ascii_whitespace() || byte == b'#' {
                break;
            }
            self.pos += 1;
        }

        (start != self.pos).then_some(&self.src[start..self.pos])
    }
}

fn parse_decimal(token: &[u8], context: &str) -> Result<u32, ViprsError> {
    let text = std::str::from_utf8(token)
        .map_err(|_| ViprsError::Codec(format!("pnm: {context} is not valid ASCII")))?;
    text.parse::<u32>()
        .map_err(|_| ViprsError::Codec(format!("pnm: invalid {context} '{text}'")))
}

fn parse_pnm_header(src: &[u8]) -> Result<ParsedPnm, ViprsError> {
    let mut cursor = TokenCursor::new(src);
    let magic = PnmMagic::parse(
        cursor
            .next_token()
            .ok_or_else(|| ViprsError::Codec("pnm: missing magic number".into()))?,
    )?;
    let width = parse_decimal(
        cursor
            .next_token()
            .ok_or_else(|| ViprsError::Codec("pnm: missing width".into()))?,
        "width",
    )?;
    let height = parse_decimal(
        cursor
            .next_token()
            .ok_or_else(|| ViprsError::Codec("pnm: missing height".into()))?,
        "height",
    )?;
    if width == 0 || height == 0 {
        return Err(ViprsError::Codec(
            "pnm: width and height must be greater than zero".into(),
        ));
    }

    let max_value = if magic.is_bitmap() {
        None
    } else {
        let value = parse_decimal(
            cursor
                .next_token()
                .ok_or_else(|| ViprsError::Codec("pnm: missing max value".into()))?,
            "max value",
        )?;
        if value == 0 || value > u32::from(u16::MAX) {
            return Err(ViprsError::Codec(format!(
                "pnm: max value {value} is out of range 1..=65535"
            )));
        }
        Some(value)
    };

    let data_offset = if magic.is_ascii() {
        cursor.pos
    } else {
        if cursor.pos >= src.len() || !src[cursor.pos].is_ascii_whitespace() {
            return Err(ViprsError::Codec(
                "pnm: binary header must be followed by whitespace".into(),
            ));
        }
        let mut pos = cursor.pos + 1;
        while pos < src.len() && src[pos] == b'#' {
            while pos < src.len() && src[pos] != b'\n' {
                pos += 1;
            }
            if pos < src.len() {
                pos += 1;
            }
        }
        pos
    };

    Ok(ParsedPnm {
        magic,
        width,
        height,
        max_value,
        data_offset,
    })
}

fn image_metadata_for_pnm(magic: PnmMagic, bit_depth: BandFormatId) -> ImageMetadata {
    let interpretation = match (magic.bands(), bit_depth) {
        (1, BandFormatId::U16) => Some(Interpretation::Grey16),
        (1, _) => Some(Interpretation::BW),
        (3, BandFormatId::U16) => Some(Interpretation::Rgb16),
        (3, _) => Some(Interpretation::Srgb),
        _ => None,
    };
    ImageMetadata {
        interpretation,
        ..ImageMetadata::default()
    }
}

fn expected_sample_count(header: &ParsedPnm) -> Result<usize, ViprsError> {
    let pixels = usize::try_from(header.width)
        .ok()
        .and_then(|width| {
            usize::try_from(header.height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| ViprsError::Codec("pnm: image dimensions overflow".into()))?;
    pixels
        .checked_mul(usize::try_from(header.magic.bands()).unwrap_or(0))
        .ok_or_else(|| ViprsError::Codec("pnm: sample count overflow".into()))
}

fn reserve_output_capacity<T>(
    output: &mut Vec<T>,
    width: u32,
    height: u32,
    bands: u32,
    details: &'static str,
) -> Result<usize, ViprsError> {
    let total_bytes = u128::from(width)
        * u128::from(height)
        * u128::from(bands)
        * std::mem::size_of::<T>() as u128;

    let Some(sample_count) = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(u64::from(bands)))
    else {
        return Err(ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: total_bytes,
            limit_bytes: isize::MAX as u128,
            details,
        });
    };

    let capacity = usize::try_from(sample_count).map_err(|_| ViprsError::ImageTooLarge {
        width,
        height,
        bands,
        bytes: total_bytes,
        limit_bytes: isize::MAX as u128,
        details,
    })?;

    output
        .try_reserve_exact(capacity)
        .map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: total_bytes,
            limit_bytes: isize::MAX as u128,
            details,
        })?;
    Ok(capacity)
}

fn decode_ascii_samples_u8(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u8>, ViprsError> {
    let mut cursor = TokenCursor::new(&src[header.data_offset..]);
    let mut output = Vec::new();
    let sample_count = reserve_output_capacity(
        &mut output,
        header.width,
        header.height,
        header.magic.bands(),
        "pnm ASCII output buffer exceeds Vec allocation limits",
    )?;
    if header.magic.is_bitmap() {
        for _ in 0..sample_count {
            let token = cursor
                .next_token()
                .ok_or_else(|| ViprsError::Codec("pnm: truncated bitmap data".into()))?;
            let value = parse_decimal(token, "bitmap sample")?;
            output.push(if value == 0 { u8::MAX } else { 0 });
        }
    } else {
        let max_value = header.max_value.unwrap_or_else(|| u32::from(u8::MAX));
        for _ in 0..sample_count {
            let token = cursor
                .next_token()
                .ok_or_else(|| ViprsError::Codec("pnm: truncated ASCII raster".into()))?;
            let value = parse_decimal(token, "sample")?;
            if value > max_value {
                return Err(ViprsError::Codec(format!(
                    "pnm: sample {value} exceeds max value {max_value}"
                )));
            }
            output.push(u8::try_from(value).map_err(|_| {
                ViprsError::Codec(format!("pnm: sample {value} does not fit into U8 output"))
            })?);
        }
    }
    Ok(output)
}

fn decode_ascii_samples_u16(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u16>, ViprsError> {
    let mut cursor = TokenCursor::new(&src[header.data_offset..]);
    let mut output = Vec::new();
    let sample_count = reserve_output_capacity(
        &mut output,
        header.width,
        header.height,
        header.magic.bands(),
        "pnm ASCII output buffer exceeds Vec allocation limits",
    )?;
    if header.magic.is_bitmap() {
        for _ in 0..sample_count {
            let token = cursor
                .next_token()
                .ok_or_else(|| ViprsError::Codec("pnm: truncated bitmap data".into()))?;
            let value = parse_decimal(token, "bitmap sample")?;
            output.push(if value == 0 { u16::MAX } else { 0 });
        }
    } else {
        let max_value = header.max_value.unwrap_or_else(|| u32::from(u16::MAX));
        for _ in 0..sample_count {
            let token = cursor
                .next_token()
                .ok_or_else(|| ViprsError::Codec("pnm: truncated ASCII raster".into()))?;
            let value = parse_decimal(token, "sample")?;
            if value > max_value {
                return Err(ViprsError::Codec(format!(
                    "pnm: sample {value} exceeds max value {max_value}"
                )));
            }
            output.push(u16::try_from(value).map_err(|_| {
                ViprsError::Codec(format!("pnm: sample {value} does not fit into U16 output"))
            })?);
        }
    }
    Ok(output)
}

fn decode_binary_bitmap_u8(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u8>, ViprsError> {
    let mut output = Vec::new();
    reserve_output_capacity(
        &mut output,
        header.width,
        header.height,
        1,
        "pnm bitmap output buffer exceeds Vec allocation limits",
    )?;
    let row_bytes = usize::try_from(header.width.div_ceil(8))
        .map_err(|_| ViprsError::Codec("pnm: row width overflow".into()))?;
    let height = usize::try_from(header.height)
        .map_err(|_| ViprsError::Codec("pnm: height overflow".into()))?;
    let expected = row_bytes
        .checked_mul(height)
        .ok_or_else(|| ViprsError::Codec("pnm: bitmap byte count overflow".into()))?;
    let raster = src
        .get(header.data_offset..header.data_offset + expected)
        .ok_or_else(|| ViprsError::Codec("pnm: truncated binary bitmap raster".into()))?;
    let width = usize::try_from(header.width)
        .map_err(|_| ViprsError::Codec("pnm: width overflow".into()))?;
    for row in raster.chunks_exact(row_bytes) {
        for x in 0..width {
            let byte = row[x / 8];
            let mask = 1u8 << (7 - (x % 8));
            output.push(if byte & mask == 0 { u8::MAX } else { 0 });
        }
    }
    Ok(output)
}

fn decode_binary_bitmap_u16(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u16>, ViprsError> {
    let mut output = Vec::new();
    reserve_output_capacity(
        &mut output,
        header.width,
        header.height,
        1,
        "pnm bitmap output buffer exceeds Vec allocation limits",
    )?;
    let row_bytes = usize::try_from(header.width.div_ceil(8))
        .map_err(|_| ViprsError::Codec("pnm: row width overflow".into()))?;
    let height = usize::try_from(header.height)
        .map_err(|_| ViprsError::Codec("pnm: height overflow".into()))?;
    let expected = row_bytes
        .checked_mul(height)
        .ok_or_else(|| ViprsError::Codec("pnm: bitmap byte count overflow".into()))?;
    let raster = src
        .get(header.data_offset..header.data_offset + expected)
        .ok_or_else(|| ViprsError::Codec("pnm: truncated binary bitmap raster".into()))?;
    let width = usize::try_from(header.width)
        .map_err(|_| ViprsError::Codec("pnm: width overflow".into()))?;
    for row in raster.chunks_exact(row_bytes) {
        for x in 0..width {
            let byte = row[x / 8];
            let mask = 1u8 << (7 - (x % 8));
            output.push(if byte & mask == 0 { u16::MAX } else { 0 });
        }
    }
    Ok(output)
}

fn decode_binary_samples_u8(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u8>, ViprsError> {
    if header.magic.is_bitmap() {
        return decode_binary_bitmap_u8(header, src);
    }

    let expected = expected_sample_count(header)?;
    let raster = src
        .get(header.data_offset..header.data_offset + expected)
        .ok_or_else(|| ViprsError::Codec("pnm: truncated binary raster".into()))?;
    Ok(raster.to_vec())
}

fn decode_binary_samples_u16(header: &ParsedPnm, src: &[u8]) -> Result<Vec<u16>, ViprsError> {
    if header.magic.is_bitmap() {
        return decode_binary_bitmap_u16(header, src);
    }

    let samples = expected_sample_count(header)?;
    let bytes = samples
        .checked_mul(2)
        .ok_or_else(|| ViprsError::Codec("pnm: binary raster byte count overflow".into()))?;
    let raster = src
        .get(header.data_offset..header.data_offset + bytes)
        .ok_or_else(|| ViprsError::Codec("pnm: truncated binary raster".into()))?;
    Ok(raster
        .chunks_exact(2)
        .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_pnm_u8(src: &[u8]) -> Result<(ParsedPnm, InMemoryImage<U8>), ViprsError> {
    let header = parse_pnm_header(src)?;
    let pixels = if header.magic.is_ascii() {
        decode_ascii_samples_u8(&header, src)?
    } else {
        decode_binary_samples_u8(&header, src)?
    };
    let metadata = image_metadata_for_pnm(header.magic, BandFormatId::U8);
    let image = InMemoryImage::from_buffer(header.width, header.height, header.magic.bands(), pixels)
        .map(|image| image.with_metadata(metadata))
        .map_err(|error| ViprsError::Codec(error.to_string()))?;
    Ok((header, image))
}

fn decode_pnm_u16(src: &[u8]) -> Result<(ParsedPnm, InMemoryImage<U16>), ViprsError> {
    let header = parse_pnm_header(src)?;
    if !header.magic.is_bitmap() && header.max_value.unwrap_or_default() <= u8::MAX.into() {
        return Err(ViprsError::Codec(
            "pnm: decoding into U16 requires a 16-bit Netpbm source".into(),
        ));
    }
    let pixels = if header.magic.is_ascii() {
        decode_ascii_samples_u16(&header, src)?
    } else {
        decode_binary_samples_u16(&header, src)?
    };
    let metadata = image_metadata_for_pnm(header.magic, BandFormatId::U16);
    let image = InMemoryImage::from_buffer(header.width, header.height, header.magic.bands(), pixels)
        .map(|image| image.with_metadata(metadata))
        .map_err(|error| ViprsError::Codec(error.to_string()))?;
    Ok((header, image))
}

#[derive(Clone, Copy)]
/// The `PnmCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs_codecs::pnm::PnmCodec>();
/// ```
pub struct PnmCodec {
    kind: PnmEncodeKind,
}

impl PnmCodec {
    /// Creates a PBM (Portable Bitmap) codec instance.
    #[must_use]
    pub const fn pbm() -> Self {
        Self {
            kind: PnmEncodeKind::Pbm,
        }
    }

    /// Creates a PGM (Portable Graymap) codec instance.
    #[must_use]
    pub const fn pgm() -> Self {
        Self {
            kind: PnmEncodeKind::Pgm,
        }
    }

    /// Creates a PPM (Portable Pixmap) codec instance.
    #[must_use]
    pub const fn ppm() -> Self {
        Self {
            kind: PnmEncodeKind::Ppm,
        }
    }

    /// Creates a generic PNM codec instance (auto-detects subformat).
    #[must_use]
    pub const fn pnm() -> Self {
        Self {
            kind: PnmEncodeKind::Pnm,
        }
    }

    fn sniff_header(header: &[u8]) -> bool {
        matches!(
            header.get(..2),
            Some(b"P1" | b"P2" | b"P3" | b"P4" | b"P5" | b"P6")
        )
    }
}

fn select_pnm_magic_for_image<F: BandFormat>(
  kind: PnmEncodeKind,
  image: &InMemoryImage<F>,
) -> Result<PnmMagic, ViprsError> {
    match kind {
        PnmEncodeKind::Pbm => {
            if image.bands() != 1 {
                return Err(ViprsError::Codec(
                    "pnm: PBM output requires a single-band image".into(),
                ));
            }
            Ok(PnmMagic::P4)
        }
        PnmEncodeKind::Pgm => {
            if image.bands() != 1 {
                return Err(ViprsError::Codec(
                    "pnm: PGM output requires a single-band image".into(),
                ));
            }
            Ok(PnmMagic::P5)
        }
        PnmEncodeKind::Ppm => {
            if image.bands() != 3 {
                return Err(ViprsError::Codec(
                    "pnm: PPM output requires a three-band image".into(),
                ));
            }
            Ok(PnmMagic::P6)
        }
        PnmEncodeKind::Pnm => match image.bands() {
            1 if F::ID == BandFormatId::U8 && is_binary_bitmap(image) => Ok(PnmMagic::P4),
            1 => Ok(PnmMagic::P5),
            3 => Ok(PnmMagic::P6),
            other => Err(ViprsError::Codec(format!(
                "pnm: generic .pnm output requires 1 or 3 bands, got {other}"
            ))),
        },
    }
}

fn is_binary_bitmap<F: BandFormat>(image: &InMemoryImage<F>) -> bool {
    match F::ID {
        BandFormatId::U8 => bytemuck::cast_slice::<F::Sample, u8>(image.pixels())
            .iter()
            .all(|&sample| sample == 0 || sample == u8::MAX),
        BandFormatId::U16 => bytemuck::cast_slice::<F::Sample, u16>(image.pixels())
            .iter()
            .all(|&sample| sample == 0 || sample == u16::MAX),
        _ => false,
    }
}

fn encode_pbm_u8(image: &InMemoryImage<U8>) -> Vec<u8> {
    let width = image.width() as usize;
    let row_bytes = width.div_ceil(8);
    let mut raster = Vec::with_capacity(row_bytes * image.height() as usize);
    for row in image.pixels().chunks_exact(width) {
        let mut current = 0u8;
        for (x, sample) in row.iter().copied().enumerate() {
            current <<= 1;
            if sample <= 127 {
                current |= 1;
            }
            if x % 8 == 7 {
                raster.push(current);
                current = 0;
            }
        }
        let remainder = width % 8;
        if remainder != 0 {
            current <<= 8 - remainder;
            raster.push(current);
        }
    }
    raster
}

fn encode_pbm_u16(image: &InMemoryImage<U16>) -> Vec<u8> {
    let width = image.width() as usize;
    let row_bytes = width.div_ceil(8);
    let mut raster = Vec::with_capacity(row_bytes * image.height() as usize);
    for row in image.pixels().chunks_exact(width) {
        let mut current = 0u8;
        for (x, sample) in row.iter().copied().enumerate() {
            current <<= 1;
            if sample <= (u16::MAX / 2) {
                current |= 1;
            }
            if x % 8 == 7 {
                raster.push(current);
                current = 0;
            }
        }
        let remainder = width % 8;
        if remainder != 0 {
            current <<= 8 - remainder;
            raster.push(current);
        }
    }
    raster
}

fn encode_pnm<F: BandFormat>(kind: PnmEncodeKind, image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
    let magic = select_pnm_magic_for_image(kind, image)?;
    let mut output = Vec::new();
    let magic_header = match magic {
        PnmMagic::P4 => b"P4\n".as_slice(),
        PnmMagic::P5 => b"P5\n".as_slice(),
        PnmMagic::P6 => b"P6\n".as_slice(),
        other => {
            return Err(ViprsError::Codec(format!(
                "pnm: unsupported Netpbm variant {other:?} for binary encoding"
            )));
        }
    };
    output.extend_from_slice(magic_header);
    output.extend_from_slice(format!("{} {}\n", image.width(), image.height()).as_bytes());

    if magic != PnmMagic::P4 {
        let max_value = match F::ID {
            BandFormatId::U8 => u8::MAX.to_string(),
            BandFormatId::U16 => u16::MAX.to_string(),
            other => {
                return Err(ViprsError::Codec(format!(
                    "pnm: unsupported format {other:?}; only U8 and U16 are supported"
                )));
            }
        };
        output.extend_from_slice(max_value.as_bytes());
        output.push(b'\n');
    }

    match (magic, F::ID) {
        (PnmMagic::P4, BandFormatId::U8) => {
            let pixels = bytemuck::cast_slice::<F::Sample, u8>(image.pixels()).to_vec();
            let image =
                InMemoryImage::<U8>::from_buffer(image.width(), image.height(), image.bands(), pixels)
                    .map_err(|error| ViprsError::Codec(error.to_string()))?;
            output.extend_from_slice(&encode_pbm_u8(&image));
        }
        (PnmMagic::P4, BandFormatId::U16) => {
            let pixels = bytemuck::cast_slice::<F::Sample, u16>(image.pixels()).to_vec();
            let image =
                InMemoryImage::<U16>::from_buffer(image.width(), image.height(), image.bands(), pixels)
                    .map_err(|error| ViprsError::Codec(error.to_string()))?;
            output.extend_from_slice(&encode_pbm_u16(&image));
        }
        (_, BandFormatId::U8) => {
            output.extend_from_slice(bytemuck::cast_slice::<F::Sample, u8>(image.pixels()));
        }
        (_, BandFormatId::U16) => {
            for sample in bytemuck::cast_slice::<F::Sample, u16>(image.pixels()) {
                output.extend_from_slice(&sample.to_be_bytes());
            }
        }
        (_, other) => {
            return Err(ViprsError::Codec(format!(
                "pnm: unsupported format {other:?}; only U8 and U16 are supported"
            )));
        }
    }

    Ok(output)
}

impl ImageDecoder for PnmCodec {
    fn format_name(&self) -> &'static str {
        "pnm"
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
        match F::ID {
            BandFormatId::U8 => {
                let (_, image) = decode_pnm_u8(src)?;
                let (width, height, bands) = (image.width(), image.height(), image.bands());
                let metadata = image.metadata().clone();
                let pixels = bytemuck::cast_vec::<u8, F::Sample>(image.into_buffer());
                InMemoryImage::from_buffer(width, height, bands, pixels)
                    .map(|decoded| decoded.with_metadata(metadata))
                    .map_err(|error| ViprsError::Codec(error.to_string()))
            }
            BandFormatId::U16 => {
                let (_, image) = decode_pnm_u16(src)?;
                let (width, height, bands) = (image.width(), image.height(), image.bands());
                let metadata = image.metadata().clone();
                let pixels = bytemuck::cast_vec::<u16, F::Sample>(image.into_buffer());
                InMemoryImage::from_buffer(width, height, bands, pixels)
                    .map(|decoded| decoded.with_metadata(metadata))
                    .map_err(|error| ViprsError::Codec(error.to_string()))
            }
            other => Err(ViprsError::Codec(format!(
                "pnm: unsupported format {other:?}; only U8 and U16 are supported"
            ))),
        }
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let header = parse_pnm_header(src)?;
        Ok((header.width, header.height, header.magic.bands()))
    }
}

impl ImageEncoder for PnmCodec {
    fn format_name(&self) -> &'static str {
        "pnm"
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
        encode_pnm(self.kind, image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{F32, U8, U16};

    #[test]
    fn pnm_decodes_all_p1_to_p6_variants() {
        let codec = PnmCodec::pnm();
        let fixtures = [
            (b"P1\n2 2\n0 1 1 0\n".as_slice(), vec![255, 0, 0, 255], 1),
            (b"P2\n2 2\n255\n0 1 2 3\n".as_slice(), vec![0, 1, 2, 3], 1),
            (
                b"P3\n2 1\n255\n255 0 0 0 255 0\n".as_slice(),
                vec![255, 0, 0, 0, 255, 0],
                3,
            ),
            (b"P4\n5 1\n\x50".as_slice(), vec![255, 0, 255, 0, 255], 1),
            (
                b"P5\n2 2\n255\n\x00\x01\x02\x03".as_slice(),
                vec![0, 1, 2, 3],
                1,
            ),
            (
                b"P6\n2 1\n255\n\xff\x00\x00\x00\xff\x00".as_slice(),
                vec![255, 0, 0, 0, 255, 0],
                3,
            ),
        ];

        for (fixture, expected, bands) in fixtures {
            let decoded = codec.decode::<U8>(fixture).unwrap();
            assert_eq!(decoded.bands(), bands);
            assert_eq!(decoded.pixels(), expected);
        }
    }

    #[test]
    fn pnm_round_trip_grayscale_and_rgb() {
        let pgm = PnmCodec::pgm();
        let ppm = PnmCodec::ppm();

        let grayscale = Image::<U8>::from_buffer(2, 2, 1, vec![0, 1, 2, 3]).unwrap();
        let rgb = Image::<U8>::from_buffer(2, 1, 3, vec![255, 0, 0, 0, 255, 0]).unwrap();

        let encoded_gray = pgm.encode(&grayscale).unwrap();
        let decoded_gray = pgm.decode::<U8>(&encoded_gray).unwrap();
        assert_eq!(decoded_gray.pixels(), grayscale.pixels());

        let encoded_rgb = ppm.encode(&rgb).unwrap();
        let decoded_rgb = ppm.decode::<U8>(&encoded_rgb).unwrap();
        assert_eq!(decoded_rgb.pixels(), rgb.pixels());
    }

    #[test]
    fn pnm_round_trip_binary_bitmap_and_u16() {
        let pbm = PnmCodec::pbm();
        let pnm = PnmCodec::pnm();

        let bitmap = Image::<U8>::from_buffer(5, 1, 1, vec![255, 0, 255, 0, 255]).unwrap();
        let encoded_bitmap = pbm.encode(&bitmap).unwrap();
        assert!(encoded_bitmap.starts_with(b"P4\n"));
        let decoded_bitmap = pbm.decode::<U8>(&encoded_bitmap).unwrap();
        assert_eq!(decoded_bitmap.pixels(), bitmap.pixels());

        let gray16 = Image::<U16>::from_buffer(2, 1, 1, vec![0, u16::MAX]).unwrap();
        let encoded_gray16 = pnm.encode(&gray16).unwrap();
        assert!(encoded_gray16.starts_with(b"P5\n"));
        let decoded_gray16 = pnm.decode::<U16>(&encoded_gray16).unwrap();
        assert_eq!(decoded_gray16.pixels(), gray16.pixels());
    }

    #[test]
    fn pnm_header_probe_sniff_and_comments_cover_metadata_paths() {
        let codec = PnmCodec::pnm();
        assert!(codec.sniff(b"P6\n1 1\n255\n"));
        assert!(!codec.sniff(b"PX\n"));

        let src = b"P6\n# width follows\n2 # inline comment\n1\n255\n\x01\x02\x03\x04\x05\x06";
        assert_eq!(codec.probe(src).unwrap(), (2, 1, 3));
        let decoded = codec.decode::<U8>(src).unwrap();
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Srgb)
        );

        let gray16 = codec.decode::<U16>(b"P5\n1 1\n65535\n\x12\x34").unwrap();
        assert_eq!(
            gray16.metadata().interpretation,
            Some(Interpretation::Grey16)
        );
        assert_eq!(gray16.pixels(), &[0x1234]);

        let rgb16 = codec
            .decode::<U16>(b"P6\n1 1\n65535\n\x00\x01\x00\x02\x00\x03")
            .unwrap();
        assert_eq!(rgb16.metadata().interpretation, Some(Interpretation::Rgb16));
        assert_eq!(rgb16.pixels(), &[1, 2, 3]);
    }

    #[test]
    fn pnm_decode_u16_bitmap_and_ascii_paths() {
        let codec = PnmCodec::pnm();

        let ascii_bitmap = codec.decode::<U16>(b"P1\n4 1\n0 1 0 1\n").unwrap();
        assert_eq!(ascii_bitmap.pixels(), &[u16::MAX, 0, u16::MAX, 0]);

        let binary_bitmap = codec.decode::<U16>(b"P4\n9 1\n\x55\x00").unwrap();
        assert_eq!(
            binary_bitmap.pixels(),
            &[u16::MAX, 0, u16::MAX, 0, u16::MAX, 0, u16::MAX, 0, u16::MAX]
        );

        let ascii_gray = codec
            .decode::<U16>(b"P2\n3 1\n65535\n0 32768 65535\n")
            .unwrap();
        assert_eq!(ascii_gray.pixels(), &[0, 32768, u16::MAX]);

        let ascii_rgb = codec.decode::<U16>(b"P3\n1 1\n65535\n1 2 65535\n").unwrap();
        assert_eq!(ascii_rgb.pixels(), &[1, 2, u16::MAX]);
    }

    #[test]
    fn pnm_decode_rejects_header_and_raster_errors() {
        let codec = PnmCodec::pnm();
        let bad_inputs: &[&[u8]] = &[
            b"",
            b"P9\n1 1\n",
            b"P5\nx 1\n255\n",
            b"P5\n1 y\n255\n",
            b"P5\n0 1\n255\n",
            b"P5\n1 1\n0\n",
            b"P5\n1 1\n65536\n",
            b"P5\n1 1\n255",
            b"P5\n1 1\n255\n",
            b"P6\n1 1\n255\n\x01\x02",
            b"P2\n1 1\n255\n256\n",
            b"P2\n1 1\n65535\n256\n",
            b"P2\n1 1\n10\n11\n",
            b"P2\n1 1\n255\n",
            b"P1\n1 1\n",
            b"P1\n1 1\nnot-a-bit\n",
        ];

        for input in bad_inputs {
            assert!(codec.decode::<U8>(input).is_err(), "{input:?}");
        }

        assert!(codec.decode::<U16>(b"P5\n1 1\n255\n\x00").is_err());
        assert!(codec.decode::<U16>(b"P2\n1 1\n10\n11\n").is_err());
        assert!(codec.decode::<F32>(b"P5\n1 1\n255\n\x00").is_err());
    }

    #[test]
    fn pnm_encode_selects_variants_and_rejects_invalid_requests() {
        let pbm = PnmCodec::pbm();
        let pgm = PnmCodec::pgm();
        let ppm = PnmCodec::ppm();
        let pnm = PnmCodec::pnm();

        let bitmap16 = Image::<U16>::from_buffer(
            9,
            1,
            1,
            vec![0, u16::MAX, 0, u16::MAX, 0, u16::MAX, 0, u16::MAX, 0],
        )
        .unwrap();
        let encoded = pbm.encode(&bitmap16).unwrap();
        assert!(encoded.starts_with(b"P4\n9 1\n"));
        assert_eq!(&encoded[7..], &[0xaa, 0x80]);

        let gray = Image::<U8>::from_buffer(2, 1, 1, vec![10, 20]).unwrap();
        assert!(pgm.encode(&gray).unwrap().starts_with(b"P5\n"));
        assert!(pnm.encode(&gray).unwrap().starts_with(b"P5\n"));

        let bitmap = Image::<U8>::from_buffer(2, 1, 1, vec![0, u8::MAX]).unwrap();
        assert!(pnm.encode(&bitmap).unwrap().starts_with(b"P4\n"));

        let rgb = Image::<U8>::from_buffer(1, 1, 3, vec![1, 2, 3]).unwrap();
        assert!(ppm.encode(&rgb).unwrap().starts_with(b"P6\n"));

        let two_band = Image::<U8>::from_buffer(1, 1, 2, vec![1, 2]).unwrap();
        assert!(pbm.encode(&rgb).is_err());
        assert!(pgm.encode(&rgb).is_err());
        assert!(ppm.encode(&gray).is_err());
        assert!(pnm.encode(&two_band).is_err());

        let float_gray = Image::<F32>::from_buffer(1, 1, 1, vec![1.0]).unwrap();
        assert!(pgm.encode(&float_gray).is_err());
    }

    #[test]
    fn pnm_binary_comments_format_names_and_private_bitmap_helpers_are_covered() {
        let codec = PnmCodec::pnm();
        assert_eq!(ImageDecoder::format_name(&codec), "pnm");
        assert_eq!(ImageEncoder::format_name(&codec), "pnm");

        let decoded = codec
            .decode::<U8>(b"P5\n1 1\n255\n# raster comment\n*")
            .unwrap();
        assert_eq!(decoded.pixels(), &[42]);

        let rgb16 = Image::<U16>::from_buffer(1, 1, 3, vec![1, 258, u16::MAX]).unwrap();
        let encoded = codec.encode(&rgb16).unwrap();
        assert!(encoded.starts_with(b"P6\n1 1\n65535\n"));
        assert_eq!(&encoded[13..], &[0, 1, 1, 2, 255, 255]);

        let full_byte_u8 =
            Image::<U8>::from_buffer(8, 1, 1, vec![0, 255, 0, 255, 0, 255, 0, 255]).unwrap();
        assert_eq!(encode_pbm_u8(&full_byte_u8), vec![0b1010_1010]);

        let full_byte_u16 = Image::<U16>::from_buffer(
            8,
            1,
            1,
            vec![0, u16::MAX, 0, u16::MAX, 0, u16::MAX, 0, u16::MAX],
        )
        .unwrap();
        assert_eq!(encode_pbm_u16(&full_byte_u16), vec![0b1010_1010]);
        assert!(is_binary_bitmap(&full_byte_u16));

        let float_gray = Image::<F32>::from_buffer(1, 1, 1, vec![0.0]).unwrap();
        assert!(!is_binary_bitmap(&float_gray));
    }

    #[test]
    fn pnm_decode_rejects_oversized_binary_bitmap_dimensions_before_raster_allocation() {
        let codec = PnmCodec::pbm();
        let src = format!("P4\n{} {}\n", u32::MAX, u32::MAX).into_bytes();

        let err = codec
            .decode::<U8>(&src)
            .expect_err("oversized bitmap dimensions must be rejected before raster allocation");

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

    #[test]
    fn pnm_decode_rejects_oversized_ascii_bitmap_dimensions_before_raster_allocation() {
        let codec = PnmCodec::pnm();
        let src = format!("P1\n{} {}\n", u32::MAX, u32::MAX).into_bytes();

        let err = codec
            .decode::<U8>(&src)
            .expect_err("oversized bitmap dimensions must be rejected before raster allocation");

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

    #[test]
    fn pnm_decode_rejects_oversized_ascii_grayscale_dimensions_before_raster_allocation() {
        let codec = PnmCodec::pnm();
        let grayscale_u8 = format!("P2\n{} {}\n255\n", u32::MAX, u32::MAX).into_bytes();
        let grayscale_u16 = format!("P2\n{} {}\n65535\n", u32::MAX, u32::MAX).into_bytes();

        let err_u8 = codec.decode::<U8>(&grayscale_u8).expect_err(
            "oversized grayscale dimensions must be rejected before U8 raster allocation",
        );
        let err_u16 = codec.decode::<U16>(&grayscale_u16).expect_err(
            "oversized grayscale dimensions must be rejected before U16 raster allocation",
        );

        assert!(matches!(
            err_u8,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 1,
                ..
            }
        ));
        assert!(matches!(
            err_u16,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 1,
                ..
            }
        ));
    }
}
