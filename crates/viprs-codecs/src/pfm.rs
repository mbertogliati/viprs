//! Pfm adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "pfm")]

//! Portable Float Map codec — decode and encode PF/Pf float images.

use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{Image, ImageMetadata, Interpretation},
};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const PFM_SCALE_KEY: &str = "pfm-scale";
const PFM_ENDIAN_KEY: &str = "pfm-endian";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PfmKind {
    Color,
    Grayscale,
}

impl PfmKind {
    fn parse(token: &[u8]) -> Result<Self, ViprsError> {
        match token {
            b"PF" => Ok(Self::Color),
            b"Pf" => Ok(Self::Grayscale),
            _ => Err(ViprsError::Codec("pfm: unsupported magic number".into())),
        }
    }

    const fn bands(self) -> u32 {
        match self {
            Self::Color => 3,
            Self::Grayscale => 1,
        }
    }

    const fn interpretation(self) -> Interpretation {
        match self {
            Self::Color => Interpretation::Scrgb,
            Self::Grayscale => Interpretation::BW,
        }
    }

    const fn magic(self) -> &'static str {
        match self {
            Self::Color => "PF",
            Self::Grayscale => "Pf",
        }
    }
}

#[derive(Clone, Debug)]
struct ParsedPfm {
    kind: PfmKind,
    width: u32,
    height: u32,
    scale: f32,
    little_endian: bool,
    data_offset: usize,
}

#[derive(Clone, Copy, Debug, Default)]
/// The `PfmCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::pfm::PfmCodec>();
/// ```
pub struct PfmCodec;

fn next_line<'a>(src: &'a [u8], offset: &mut usize) -> Option<&'a [u8]> {
    if *offset >= src.len() {
        return None;
    }

    let start = *offset;
    let end = src[start..]
        .iter()
        .position(|&byte| byte == b'\n')
        .map_or(src.len(), |index| start + index);
    *offset = if end < src.len() { end + 1 } else { end };

    let mut line = &src[start..end];
    if let Some(stripped) = line.strip_suffix(b"\r") {
        line = stripped;
    }
    Some(line)
}

fn next_non_comment_line<'a>(src: &'a [u8], offset: &mut usize) -> Result<&'a [u8], ViprsError> {
    while let Some(line) = next_line(src, offset) {
        let trimmed = line
            .iter()
            .copied()
            .skip_while(u8::is_ascii_whitespace)
            .collect::<Vec<_>>();
        if trimmed.is_empty() || trimmed.first() == Some(&b'#') {
            continue;
        }
        return Ok(line);
    }

    Err(ViprsError::Codec("pfm: truncated header".into()))
}

fn parse_ascii_u32(token: &str, field: &str) -> Result<u32, ViprsError> {
    token
        .parse::<u32>()
        .map_err(|_| ViprsError::Codec(format!("pfm: invalid {field} '{token}'")))
}

fn parse_ascii_f32(token: &str, field: &str) -> Result<f32, ViprsError> {
    token
        .parse::<f32>()
        .map_err(|_| ViprsError::Codec(format!("pfm: invalid {field} '{token}'")))
}

fn parse_pfm_header(src: &[u8]) -> Result<ParsedPfm, ViprsError> {
    let mut offset = 0usize;
    let magic = next_non_comment_line(src, &mut offset)?;
    let kind = PfmKind::parse(magic)?;

    let dimensions_line = std::str::from_utf8(next_non_comment_line(src, &mut offset)?)
        .map_err(|_| ViprsError::Codec("pfm: dimensions line is not valid ASCII".into()))?;
    let mut dimensions = dimensions_line.split_whitespace();
    let width = parse_ascii_u32(
        dimensions
            .next()
            .ok_or_else(|| ViprsError::Codec("pfm: missing width".into()))?,
        "width",
    )?;
    let height = parse_ascii_u32(
        dimensions
            .next()
            .ok_or_else(|| ViprsError::Codec("pfm: missing height".into()))?,
        "height",
    )?;
    if width == 0 || height == 0 {
        return Err(ViprsError::Codec(
            "pfm: width and height must be greater than zero".into(),
        ));
    }
    if dimensions.next().is_some() {
        return Err(ViprsError::Codec(
            "pfm: dimensions line must contain exactly two integers".into(),
        ));
    }

    let scale_line = std::str::from_utf8(next_non_comment_line(src, &mut offset)?)
        .map_err(|_| ViprsError::Codec("pfm: scale line is not valid ASCII".into()))?;
    let scale = parse_ascii_f32(scale_line.trim(), "scale")?;
    if !scale.is_finite() || scale == 0.0 {
        return Err(ViprsError::Codec(
            "pfm: scale must be finite and non-zero".into(),
        ));
    }

    Ok(ParsedPfm {
        kind,
        width,
        height,
        scale: scale.abs(),
        little_endian: scale.is_sign_negative(),
        data_offset: offset,
    })
}

fn sample_count(width: u32, height: u32, bands: u32) -> Result<usize, ViprsError> {
    usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .and_then(|pixels| {
            usize::try_from(bands)
                .ok()
                .and_then(|b| pixels.checked_mul(b))
        })
        .ok_or_else(|| ViprsError::Codec("pfm: sample count overflow".into()))
}

fn decode_pfm(src: &[u8]) -> Result<(ParsedPfm, Vec<f32>), ViprsError> {
    let header = parse_pfm_header(src)?;
    let expected_samples = sample_count(header.width, header.height, header.kind.bands())?;
    let expected_bytes = expected_samples
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ViprsError::Codec("pfm: byte count overflow".into()))?;
    let raster = src
        .get(header.data_offset..header.data_offset + expected_bytes)
        .ok_or_else(|| ViprsError::Codec("pfm: truncated raster".into()))?;

    let mut pixels = Vec::with_capacity(expected_samples);
    for chunk in raster.chunks_exact(4) {
        let value = if header.little_endian {
            f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        };
        pixels.push(value);
    }

    Ok((header, pixels))
}

fn metadata_for_pfm(header: &ParsedPfm) -> ImageMetadata {
    let mut metadata = ImageMetadata {
        interpretation: Some(header.kind.interpretation()),
        ..ImageMetadata::default()
    };
    metadata
        .extra
        .insert(PFM_SCALE_KEY.into(), header.scale.to_string());
    metadata.extra.insert(
        PFM_ENDIAN_KEY.into(),
        if header.little_endian {
            "little".into()
        } else {
            "big".into()
        },
    );
    metadata
}

fn endian_from_metadata(metadata: &ImageMetadata) -> bool {
    metadata
        .extra
        .get(PFM_ENDIAN_KEY)
        .map(|value| value.eq_ignore_ascii_case("little"))
        .unwrap_or(false)
}

fn scale_from_metadata(metadata: &ImageMetadata) -> f32 {
    metadata
        .extra
        .get(PFM_SCALE_KEY)
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0)
}

impl ImageDecoder for PfmCodec {
    fn format_name(&self) -> &'static str {
        "pfm"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.starts_with(b"PF") || header.starts_with(b"Pf")
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
        if F::ID != BandFormatId::F32 {
            return Err(ViprsError::Codec(format!(
                "pfm: unsupported format {:?}; only F32 is supported",
                F::ID
            )));
        }

        let (header, pixels) = decode_pfm(src)?;
        let pixels = bytemuck::cast_vec::<f32, F::Sample>(pixels);
        Image::from_buffer(header.width, header.height, header.kind.bands(), pixels)
            .map(|image| image.with_metadata(metadata_for_pfm(&header)))
            .map_err(|error| ViprsError::Codec(error.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let header = parse_pfm_header(src)?;
        Ok((header.width, header.height, header.kind.bands()))
    }
}

impl ImageEncoder for PfmCodec {
    fn format_name(&self) -> &'static str {
        "pfm"
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
        if F::ID != BandFormatId::F32 {
            return Err(ViprsError::Codec(format!(
                "pfm: unsupported format {:?}; only F32 is supported",
                F::ID
            )));
        }

        let kind = match image.bands() {
            1 => PfmKind::Grayscale,
            3 => PfmKind::Color,
            other => {
                return Err(ViprsError::Codec(format!(
                    "pfm: unsupported band count {other}; expected 1 or 3"
                )));
            }
        };

        let little_endian = endian_from_metadata(image.metadata());
        let scale = scale_from_metadata(image.metadata());
        let scale_value = if little_endian { -scale } else { scale };
        let samples = bytemuck::cast_slice::<F::Sample, f32>(image.pixels());

        let mut output = Vec::new();
        output.extend_from_slice(kind.magic().as_bytes());
        output.extend_from_slice(b"\n");
        output.extend_from_slice(format!("{} {}\n", image.width(), image.height()).as_bytes());
        output.extend_from_slice(format!("{scale_value}\n").as_bytes());

        for &sample in samples {
            let bytes = if little_endian {
                sample.to_le_bytes()
            } else {
                sample.to_be_bytes()
            };
            output.extend_from_slice(&bytes);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{F32, U8};

    fn image_with_metadata(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<f32>,
        scale: f32,
        little_endian: bool,
    ) -> Image<F32> {
        let mut metadata = ImageMetadata::default();
        metadata
            .extra
            .insert(PFM_SCALE_KEY.into(), scale.to_string());
        metadata.extra.insert(
            PFM_ENDIAN_KEY.into(),
            if little_endian {
                "little".into()
            } else {
                "big".into()
            },
        );
        Image::<F32>::from_buffer(width, height, bands, pixels)
            .unwrap()
            .with_metadata(metadata)
    }

    #[test]
    fn pfm_round_trip_rgb_big_endian() {
        let codec = PfmCodec;
        let original = image_with_metadata(
            2,
            2,
            3,
            vec![
                0.0, 0.25, 0.5, 0.75, 1.0, 1.25, //
                1.5, 1.75, 2.0, 2.25, 2.5, 2.75,
            ],
            1.0,
            false,
        );

        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(decoded.pixels(), original.pixels());
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        assert_eq!(
            decoded.metadata().extra.get(PFM_SCALE_KEY),
            Some(&"1".into())
        );
        assert_eq!(
            decoded.metadata().extra.get(PFM_ENDIAN_KEY),
            Some(&"big".into())
        );
    }

    #[test]
    fn pfm_decode_little_endian_grayscale() {
        let mut encoded = b"Pf\n2 2\n-2.5\n".to_vec();
        for sample in [0.5f32, 1.5, 2.5, 3.5] {
            encoded.extend_from_slice(&sample.to_le_bytes());
        }

        let decoded = PfmCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 1);
        assert_eq!(decoded.pixels(), &[0.5, 1.5, 2.5, 3.5]);
        assert_eq!(decoded.metadata().interpretation, Some(Interpretation::BW));
        assert_eq!(
            decoded.metadata().extra.get(PFM_SCALE_KEY),
            Some(&"2.5".into())
        );
        assert_eq!(
            decoded.metadata().extra.get(PFM_ENDIAN_KEY),
            Some(&"little".into())
        );
    }

    #[test]
    fn pfm_round_trip_grayscale_little_endian() {
        let codec = PfmCodec;
        let original = image_with_metadata(2, 2, 1, vec![0.5, 1.0, 1.5, 2.0], 2.0, true);

        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.pixels(), original.pixels());
        assert_eq!(
            decoded.metadata().extra.get(PFM_SCALE_KEY),
            Some(&"2".into())
        );
        assert_eq!(
            decoded.metadata().extra.get(PFM_ENDIAN_KEY),
            Some(&"little".into())
        );
    }

    #[test]
    fn pfm_parse_header_skips_comments() {
        let mut encoded = b"PF
# generated by test
2 1
# scale follows
1.0
"
        .to_vec();
        encoded.extend(std::iter::repeat_n(0u8, 2 * 3 * 4));

        let parsed = parse_pfm_header(&encoded).unwrap();

        assert_eq!(parsed.width, 2);
        assert_eq!(parsed.height, 1);
        assert_eq!(parsed.kind, PfmKind::Color);
    }

    #[test]
    fn pfm_rejects_zero_scale() {
        let encoded = b"Pf
1 1
0
";
        assert!(parse_pfm_header(encoded).is_err());
    }

    #[test]
    fn pfm_encode_rejects_invalid_band_count() {
        let image = Image::<F32>::from_buffer(1, 1, 2, vec![0.0, 1.0]).unwrap();
        assert!(PfmCodec.encode(&image).is_err());
    }

    #[test]
    fn pfm_sniff_rejects_non_pfm_magic() {
        assert!(!PfmCodec.sniff(
            b"P6
1 1
255
"
        ));
    }

    #[test]
    fn pfm_probe_reports_dimensions() {
        let mut encoded = b"PF\n3 1\n1.0\n".to_vec();
        encoded.extend(std::iter::repeat_n(0u8, 3 * 3 * 4));
        assert_eq!(PfmCodec.probe(&encoded).unwrap(), (3, 1, 3));
    }

    #[test]
    fn pfm_rejects_non_f32_requests() {
        let mut encoded = b"Pf\n1 1\n1.0\n".to_vec();
        encoded.extend_from_slice(&0.5f32.to_be_bytes());
        let result = PfmCodec.decode::<U8>(&encoded);
        assert!(result.is_err());
    }
}
