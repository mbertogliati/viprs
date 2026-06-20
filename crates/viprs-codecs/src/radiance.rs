//! Radiance adapter module.
//!
//! This module provides concrete codec-related infrastructure used by the
//! adapter layer for loading, saving, or normalizing external image formats.

#![cfg(feature = "radiance")]

//! Radiance HDR codec — decode and encode RGBE `.hdr` files as F32 RGB images.

use viprs_core::{
    codec_options::{LoadOptions, SaveOptions},
    error::ViprsError,
    format::{BandFormat, BandFormatId},
    image::{Image, ImageMetadata, Interpretation},
};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

const RADIANCE_MAGIC: &str = "#?RADIANCE";
const RGBE_MAGIC: &str = "#?RGBE";
const RADIANCE_FORMAT: &str = "32-bit_rle_rgbe";
const MIN_SCANLINE_LEN: usize = 8;
const MAX_SCANLINE_LEN: usize = 0x7fff;
const MIN_RUN_LEN: usize = 4;
const FORMAT_KEY: &str = "radiance.format";
const EXPOSURE_KEY: &str = "radiance.exposure";
const ASPECT_KEY: &str = "radiance.aspect";

#[derive(Clone, Copy, Debug, Default)]
/// The `RadianceCodec` type provides concrete adapter functionality in the `codecs` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```rust
/// let _ = core::mem::size_of::<viprs::adapters::codecs::radiance::RadianceCodec>();
/// ```
pub struct RadianceCodec;

#[derive(Clone, Debug)]
struct ParsedRadianceHeader {
    width: u32,
    height: u32,
    data_offset: usize,
    exposure: Option<f32>,
    aspect: Option<f32>,
}

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

fn parse_resolution_line(line: &str) -> Result<(u32, u32), ViprsError> {
    let mut tokens = line.split_whitespace();
    let first_axis = tokens
        .next()
        .ok_or_else(|| ViprsError::Codec("radiance: missing first resolution axis".into()))?;
    let first_value = tokens
        .next()
        .ok_or_else(|| ViprsError::Codec("radiance: missing first resolution value".into()))?;
    let second_axis = tokens
        .next()
        .ok_or_else(|| ViprsError::Codec("radiance: missing second resolution axis".into()))?;
    let second_value = tokens
        .next()
        .ok_or_else(|| ViprsError::Codec("radiance: missing second resolution value".into()))?;
    if tokens.next().is_some() {
        return Err(ViprsError::Codec(
            "radiance: resolution line has too many fields".into(),
        ));
    }

    let first = first_value.parse::<u32>().map_err(|_| {
        ViprsError::Codec(format!(
            "radiance: invalid resolution value '{first_value}'"
        ))
    })?;
    let second = second_value.parse::<u32>().map_err(|_| {
        ViprsError::Codec(format!(
            "radiance: invalid resolution value '{second_value}'"
        ))
    })?;
    if first == 0 || second == 0 {
        return Err(ViprsError::Codec(
            "radiance: width and height must be greater than zero".into(),
        ));
    }

    let y_major = first_axis.ends_with('Y') && second_axis.ends_with('X');
    let x_major = first_axis.ends_with('X') && second_axis.ends_with('Y');
    if !y_major && !x_major {
        return Err(ViprsError::Codec(
            "radiance: unsupported resolution axis order".into(),
        ));
    }

    if y_major {
        Ok((second, first))
    } else {
        Ok((first, second))
    }
}

fn parse_radiance_header(src: &[u8]) -> Result<ParsedRadianceHeader, ViprsError> {
    let mut offset = 0usize;
    let magic = next_line(src, &mut offset)
        .ok_or_else(|| ViprsError::Codec("radiance: missing magic header line".into()))?;
    let magic = std::str::from_utf8(magic)
        .map_err(|_| ViprsError::Codec("radiance: magic line is not valid ASCII".into()))?;
    if magic != RADIANCE_MAGIC && magic != RGBE_MAGIC {
        return Err(ViprsError::Codec("radiance: invalid magic header".into()));
    }

    let mut saw_format = false;
    let mut exposure = None;
    let mut aspect = None;

    loop {
        let line = next_line(src, &mut offset)
            .ok_or_else(|| ViprsError::Codec("radiance: truncated header".into()))?;
        if line.is_empty() {
            break;
        }
        let line = std::str::from_utf8(line)
            .map_err(|_| ViprsError::Codec("radiance: header is not valid ASCII".into()))?;
        if let Some(format) = line.strip_prefix("FORMAT=") {
            let format = format.trim();
            if format != RADIANCE_FORMAT {
                return Err(ViprsError::Codec(format!(
                    "radiance: unsupported format '{format}'"
                )));
            }
            saw_format = true;
        } else if let Some(value) = line.strip_prefix("EXPOSURE=") {
            exposure = value.trim().parse::<f32>().ok().filter(|v| v.is_finite());
        } else if let Some(value) = line.strip_prefix("PIXASPECT=") {
            aspect = value.trim().parse::<f32>().ok().filter(|v| v.is_finite());
        }
    }

    if !saw_format {
        return Err(ViprsError::Codec(
            "radiance: missing FORMAT=32-bit_rle_rgbe".into(),
        ));
    }

    let resolution_line = next_line(src, &mut offset)
        .ok_or_else(|| ViprsError::Codec("radiance: missing resolution line".into()))?;
    let resolution_line = std::str::from_utf8(resolution_line)
        .map_err(|_| ViprsError::Codec("radiance: resolution line is not valid ASCII".into()))?;
    let (width, height) = parse_resolution_line(resolution_line)?;

    Ok(ParsedRadianceHeader {
        width,
        height,
        data_offset: offset,
        exposure,
        aspect,
    })
}

fn rgbe_to_rgb(pixel: [u8; 4]) -> [f32; 3] {
    if pixel[3] == 0 {
        return [0.0, 0.0, 0.0];
    }

    let factor = 2.0_f32.powi(i32::from(pixel[3]) - 136);
    [
        (f32::from(pixel[0]) + 0.5) * factor,
        (f32::from(pixel[1]) + 0.5) * factor,
        (f32::from(pixel[2]) + 0.5) * factor,
    ]
}

fn rgb_to_rgbe(rgb: &[f32]) -> [u8; 4] {
    let red = rgb[0].max(0.0);
    let green = rgb[1].max(0.0);
    let blue = rgb[2].max(0.0);
    let max_component = red.max(green).max(blue);
    if max_component <= 1.0e-32 {
        return [0, 0, 0, 0];
    }

    let exponent = max_component.log2().floor() as i32 + 1;
    let scale = 255.9999_f32 * 2.0_f32.powi(-exponent);

    [
        if red > 0.0 { (red * scale) as u8 } else { 0 },
        if green > 0.0 {
            (green * scale) as u8
        } else {
            0
        },
        if blue > 0.0 { (blue * scale) as u8 } else { 0 },
        (exponent + 128) as u8,
    ]
}

fn decode_old_style_scanline(
    src: &[u8],
    width: usize,
) -> Result<(Vec<[u8; 4]>, usize), ViprsError> {
    let mut scanline = vec![[0u8; 4]; width];
    let mut pos = 0usize;
    let mut pixel_index = 0usize;
    let mut rshift = 0usize;

    while pixel_index < width {
        let chunk = src
            .get(pos..pos + 4)
            .ok_or_else(|| ViprsError::Codec("radiance: truncated old-style scanline".into()))?;
        pos += 4;
        let pixel = [chunk[0], chunk[1], chunk[2], chunk[3]];

        if pixel[0] == 1 && pixel[1] == 1 && pixel[2] == 1 {
            if pixel_index == 0 {
                return Err(ViprsError::Codec(
                    "radiance: repeat marker at start of old-style scanline".into(),
                ));
            }
            let repeat = usize::from(pixel[3]) << rshift;
            for _ in 0..repeat {
                if pixel_index >= width {
                    break;
                }
                scanline[pixel_index] = scanline[pixel_index - 1];
                pixel_index += 1;
            }
            rshift += 8;
            if rshift > 24 {
                return Err(ViprsError::Codec(
                    "radiance: old-style run length overflow".into(),
                ));
            }
        } else {
            scanline[pixel_index] = pixel;
            pixel_index += 1;
            rshift = 0;
        }
    }

    Ok((scanline, pos))
}

fn decode_scanline(src: &[u8], width: usize) -> Result<(Vec<[u8; 4]>, usize), ViprsError> {
    if width < MIN_SCANLINE_LEN || width > MAX_SCANLINE_LEN || src.len() < 4 {
        return decode_old_style_scanline(src, width);
    }

    if src[0] != 2 || src[1] != 2 || (src[2] & 0x80) != 0 {
        return decode_old_style_scanline(src, width);
    }

    let encoded_width = (usize::from(src[2]) << 8) | usize::from(src[3]);
    if encoded_width != width {
        return Err(ViprsError::Codec(
            "radiance: scanline length mismatch".into(),
        ));
    }

    let mut scanline = vec![[0u8; 4]; width];
    let mut pos = 4usize;
    for channel in 0..4 {
        let mut x = 0usize;
        while x < width {
            let code = *src
                .get(pos)
                .ok_or_else(|| ViprsError::Codec("radiance: truncated RLE stream".into()))?;
            pos += 1;
            if code > 128 {
                let run = usize::from(code & 0x7f);
                if run == 0 || x + run > width {
                    return Err(ViprsError::Codec("radiance: invalid RLE run".into()));
                }
                let value = *src
                    .get(pos)
                    .ok_or_else(|| ViprsError::Codec("radiance: truncated RLE run value".into()))?;
                pos += 1;
                for target in &mut scanline[x..x + run] {
                    target[channel] = value;
                }
                x += run;
            } else {
                let run = usize::from(code);
                if run == 0 || x + run > width {
                    return Err(ViprsError::Codec("radiance: invalid RLE literal".into()));
                }
                let values = src
                    .get(pos..pos + run)
                    .ok_or_else(|| ViprsError::Codec("radiance: truncated RLE literal".into()))?;
                pos += run;
                for (target, value) in scanline[x..x + run].iter_mut().zip(values.iter().copied()) {
                    target[channel] = value;
                }
                x += run;
            }
        }
    }

    Ok((scanline, pos))
}

fn metadata_for_radiance(header: &ParsedRadianceHeader) -> ImageMetadata {
    let mut metadata = ImageMetadata {
        interpretation: Some(Interpretation::Scrgb),
        ..ImageMetadata::default()
    };
    metadata
        .extra
        .insert(FORMAT_KEY.into(), RADIANCE_FORMAT.into());
    if let Some(exposure) = header.exposure {
        metadata
            .extra
            .insert(EXPOSURE_KEY.into(), exposure.to_string());
    }
    if let Some(aspect) = header.aspect {
        metadata.extra.insert(ASPECT_KEY.into(), aspect.to_string());
    }
    metadata
}

fn decode_radiance(src: &[u8]) -> Result<(ParsedRadianceHeader, Vec<f32>), ViprsError> {
    let header = parse_radiance_header(src)?;
    let width = usize::try_from(header.width)
        .map_err(|_| ViprsError::Codec("radiance: width exceeds usize".into()))?;
    let height = usize::try_from(header.height)
        .map_err(|_| ViprsError::Codec("radiance: height exceeds usize".into()))?;
    let sample_count = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| ViprsError::Codec("radiance: sample count overflow".into()))?;
    let mut pixels = vec![0.0f32; sample_count];
    let mut pos = header.data_offset;

    for row in 0..height {
        let (scanline, used) = decode_scanline(
            src.get(pos..)
                .ok_or_else(|| ViprsError::Codec("radiance: truncated raster".into()))?,
            width,
        )?;
        pos += used;
        for (column, rgbe) in scanline.iter().enumerate() {
            let rgb = rgbe_to_rgb(*rgbe);
            let base = (row * width + column) * 3;
            pixels[base..base + 3].copy_from_slice(&rgb);
        }
    }

    Ok((header, pixels))
}

fn encode_rle_channel(values: &[u8], output: &mut Vec<u8>) {
    let mut position = 0usize;
    while position < values.len() {
        let mut run_start = position;
        let mut run_len = 1usize;
        while run_start < values.len() {
            run_len = 1;
            while run_len < 127
                && run_start + run_len < values.len()
                && values[run_start + run_len] == values[run_start]
            {
                run_len += 1;
            }
            if run_len >= MIN_RUN_LEN {
                break;
            }
            run_start += run_len;
        }

        while position < run_start {
            let literal_len = (run_start - position).min(128);
            output.push(literal_len as u8);
            output.extend_from_slice(&values[position..position + literal_len]);
            position += literal_len;
        }

        if run_start < values.len() {
            output.push((128 + run_len) as u8);
            output.push(values[run_start]);
            position = run_start + run_len;
        }
    }
}

fn encode_scanline(scanline: &[[u8; 4]], output: &mut Vec<u8>) {
    let width = scanline.len();
    if !(MIN_SCANLINE_LEN..=MAX_SCANLINE_LEN).contains(&width) {
        for pixel in scanline {
            output.extend_from_slice(pixel);
        }
        return;
    }

    output.extend_from_slice(&[2, 2, (width >> 8) as u8, (width & 0xff) as u8]);
    for channel in 0..4 {
        let values = scanline
            .iter()
            .map(|pixel| pixel[channel])
            .collect::<Vec<_>>();
        encode_rle_channel(&values, output);
    }
}

impl ImageDecoder for RadianceCodec {
    fn format_name(&self) -> &'static str {
        "radiance"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        header.starts_with(RADIANCE_MAGIC.as_bytes()) || header.starts_with(RGBE_MAGIC.as_bytes())
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
                "radiance: unsupported format {:?}; only F32 is supported",
                F::ID
            )));
        }

        let (header, pixels) = decode_radiance(src)?;
        let pixels = bytemuck::cast_vec::<f32, F::Sample>(pixels);
        Image::from_buffer(header.width, header.height, 3, pixels)
            .map(|image| image.with_metadata(metadata_for_radiance(&header)))
            .map_err(|error| ViprsError::Codec(error.to_string()))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let header = parse_radiance_header(src)?;
        Ok((header.width, header.height, 3))
    }
}

impl ImageEncoder for RadianceCodec {
    fn format_name(&self) -> &'static str {
        "radiance"
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
                "radiance: unsupported format {:?}; only F32 is supported",
                F::ID
            )));
        }
        if image.bands() != 3 {
            return Err(ViprsError::Codec(format!(
                "radiance: unsupported band count {}; expected 3",
                image.bands()
            )));
        }

        let samples = bytemuck::cast_slice::<F::Sample, f32>(image.pixels());
        let width = usize::try_from(image.width())
            .map_err(|_| ViprsError::Codec("radiance: width exceeds usize".into()))?;
        let height = usize::try_from(image.height())
            .map_err(|_| ViprsError::Codec("radiance: height exceeds usize".into()))?;
        let mut output = Vec::new();
        output.extend_from_slice(b"#?RADIANCE\n");
        output.extend_from_slice(format!("FORMAT={RADIANCE_FORMAT}\n").as_bytes());
        if let Some(exposure) = image
            .metadata()
            .extra
            .get(EXPOSURE_KEY)
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite())
        {
            output.extend_from_slice(format!("EXPOSURE={exposure}\n").as_bytes());
        }
        if let Some(aspect) = image
            .metadata()
            .extra
            .get(ASPECT_KEY)
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            output.extend_from_slice(format!("PIXASPECT={aspect}\n").as_bytes());
        }
        output.extend_from_slice(b"\n");
        output
            .extend_from_slice(format!("-Y {} +X {}\n", image.height(), image.width()).as_bytes());

        for row in 0..height {
            let row_start = row
                .checked_mul(width)
                .and_then(|pixels| pixels.checked_mul(3))
                .ok_or_else(|| ViprsError::Codec("radiance: row offset overflow".into()))?;
            let scanline = samples[row_start..row_start + width * 3]
                .chunks_exact(3)
                .map(rgb_to_rgbe)
                .collect::<Vec<_>>();
            encode_scanline(&scanline, &mut output);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::format::{F32, U8};

    fn approx_eq(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (lhs, rhs)) in left.iter().zip(right.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= tolerance,
                "sample {index} mismatch: left={lhs}, right={rhs}, diff={diff}"
            );
        }
    }

    #[test]
    fn radiance_round_trip_rgb_f32() {
        let codec = RadianceCodec;
        let original =
            Image::<F32>::from_buffer(8, 2, 3, (0..48).map(|index| index as f32 * 0.125).collect())
                .unwrap();

        let encoded = codec.encode(&original).unwrap();
        let decoded = codec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), 8);
        assert_eq!(decoded.height(), 2);
        assert_eq!(decoded.bands(), 3);
        assert_eq!(
            decoded.metadata().interpretation,
            Some(Interpretation::Scrgb)
        );
        approx_eq(decoded.pixels(), original.pixels(), 0.02);
    }

    #[test]
    fn radiance_decodes_small_raw_scanline() {
        let mut encoded = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n".to_vec();
        encoded.extend_from_slice(&rgb_to_rgbe(&[0.5, 1.0, 1.5]));
        encoded.extend_from_slice(&rgb_to_rgbe(&[2.0, 2.5, 3.0]));

        let decoded = RadianceCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 1);
        approx_eq(decoded.pixels(), &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 0.02);
    }

    #[test]
    fn radiance_parse_resolution_supports_both_axis_orders() {
        assert_eq!(parse_resolution_line("-Y 3 +X 4").unwrap(), (4, 3));
        assert_eq!(parse_resolution_line("+X 4 -Y 3").unwrap(), (4, 3));
    }

    #[test]
    fn radiance_parse_header_preserves_optional_metadata() {
        let encoded = b"#?RADIANCE
FORMAT=32-bit_rle_rgbe
EXPOSURE=1.5
PIXASPECT=2

-Y 1 +X 1
";
        let parsed = parse_radiance_header(encoded).unwrap();
        assert_eq!(parsed.exposure, Some(1.5));
        assert_eq!(parsed.aspect, Some(2.0));
    }

    #[test]
    fn radiance_parse_header_rejects_missing_format() {
        let encoded = b"#?RADIANCE
EXPOSURE=1.5

-Y 1 +X 1
";
        assert!(parse_radiance_header(encoded).is_err());
    }

    #[test]
    fn radiance_decode_old_style_scanline_repeats_previous_pixel() {
        let source = [10u8, 20, 30, 129, 1, 1, 1, 2];
        let (scanline, used) = decode_old_style_scanline(&source, 3).unwrap();
        assert_eq!(used, source.len());
        assert_eq!(scanline, vec![[10, 20, 30, 129]; 3]);
    }

    #[test]
    fn radiance_decode_scanline_reads_rle_runs_and_literals() {
        let encoded = vec![
            2, 2, 0, 8, 132, 1, 4, 2, 3, 4, 5, 8, 8, 7, 6, 5, 4, 3, 2, 1, 136, 9, 136, 129,
        ];
        let (scanline, used) = decode_scanline(&encoded, 8).unwrap();
        assert_eq!(used, encoded.len());
        assert_eq!(scanline[0], [1, 8, 9, 129]);
        assert_eq!(scanline[4], [2, 4, 9, 129]);
        assert_eq!(scanline[7], [5, 1, 9, 129]);
    }

    #[test]
    fn radiance_encode_scanline_uses_raw_for_small_width() {
        let scanline = vec![[1u8, 2, 3, 4], [5, 6, 7, 8]];
        let mut encoded = Vec::new();
        encode_scanline(&scanline, &mut encoded);
        assert_eq!(encoded, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn radiance_encode_scanline_uses_rle_for_wide_rows() {
        let scanline = vec![[1u8, 2, 3, 4]; 8];
        let mut encoded = Vec::new();
        encode_scanline(&scanline, &mut encoded);
        assert!(encoded.starts_with(&[2, 2, 0, 8]));
    }

    #[test]
    fn radiance_round_trip_preserves_optional_metadata() {
        let mut metadata = ImageMetadata::default();
        metadata.extra.insert(EXPOSURE_KEY.into(), "1.5".into());
        metadata.extra.insert(ASPECT_KEY.into(), "2".into());
        let original = Image::<F32>::from_buffer(1, 1, 3, vec![0.5, 1.0, 1.5])
            .unwrap()
            .with_metadata(metadata);

        let encoded = RadianceCodec.encode(&original).unwrap();
        let decoded = RadianceCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(
            decoded.metadata().extra.get(EXPOSURE_KEY),
            Some(&"1.5".into())
        );
        assert_eq!(decoded.metadata().extra.get(ASPECT_KEY), Some(&"2".into()));
    }

    #[test]
    fn radiance_encode_rejects_non_rgb_images() {
        let image = Image::<F32>::from_buffer(1, 1, 1, vec![0.5]).unwrap();
        assert!(RadianceCodec.encode(&image).is_err());
    }

    #[test]
    fn radiance_sniff_rejects_non_radiance_headers() {
        assert!(!RadianceCodec.sniff(
            b"P6
1 1
255
"
        ));
    }

    #[test]
    fn radiance_parse_header_rejects_invalid_magic() {
        let encoded = b"#?NOTRADIANCE
FORMAT=32-bit_rle_rgbe

-Y 1 +X 1
";
        assert!(parse_radiance_header(encoded).is_err());
    }

    #[test]
    fn radiance_parse_header_rejects_unsupported_format() {
        let encoded = b"#?RADIANCE
FORMAT=32-bit_rle_xyze

-Y 1 +X 1
";
        assert!(parse_radiance_header(encoded).is_err());
    }

    #[test]
    fn radiance_parse_resolution_rejects_invalid_axis_order() {
        assert!(parse_resolution_line("+Z 1 +X 2").is_err());
    }

    #[test]
    fn radiance_decode_old_style_rejects_repeat_at_start() {
        let source = [1u8, 1, 1, 2];
        assert!(decode_old_style_scanline(&source, 1).is_err());
    }

    #[test]
    fn radiance_decode_scanline_rejects_mismatched_width() {
        let encoded = [2u8, 2, 0, 7];
        assert!(decode_scanline(&encoded, 8).is_err());
    }

    #[test]
    fn radiance_decode_scanline_rejects_invalid_run() {
        let encoded = [2u8, 2, 0, 8, 128, 1];
        assert!(decode_scanline(&encoded, 8).is_err());
    }

    #[test]
    fn radiance_decode_scanline_rejects_invalid_literal() {
        let encoded = [2u8, 2, 0, 8, 0, 1];
        assert!(decode_scanline(&encoded, 8).is_err());
    }

    #[test]
    fn radiance_parse_header_accepts_rgbe_magic_and_crlf() {
        let encoded = b"#?RGBE
FORMAT=32-bit_rle_rgbe

-Y 1 +X 1
";
        let parsed = parse_radiance_header(encoded).unwrap();
        assert_eq!((parsed.width, parsed.height), (1, 1));
    }

    #[test]
    fn radiance_decode_scanline_falls_back_to_old_style_when_missing_rle_magic() {
        let mut encoded = Vec::new();
        for _ in 0..8 {
            encoded.extend_from_slice(&[10u8, 20, 30, 129]);
        }
        let (scanline, used) = decode_scanline(&encoded, 8).unwrap();
        assert_eq!(used, encoded.len());
        assert_eq!(scanline[0], [10, 20, 30, 129]);
        assert_eq!(scanline[7], [10, 20, 30, 129]);
    }

    #[test]
    fn radiance_probe_reports_dimensions() {
        let encoded = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 3 +X 4\n";
        assert_eq!(RadianceCodec.probe(encoded).unwrap(), (4, 3, 3));
    }

    #[test]
    fn radiance_rejects_non_f32_requests() {
        let encoded = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\0\0\0\0";
        let result = RadianceCodec.decode::<U8>(encoded);
        assert!(result.is_err());
    }
}
