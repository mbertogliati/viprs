use std::borrow::Cow;
use std::io::Write;

use crc32fast::Hasher;
use flate2::{Compression as ZlibCompression, write::ZlibEncoder};
use png::{
    BitDepth, ColorType, DeflateCompression, Encoder as RawPngEncoder, Filter, Info,
    SrgbRenderingIntent, text_metadata::ITXtChunk,
};

use crate::adapters::codecs::web_colour::{normalize_web_output_u8, normalize_web_output_u16};
use crate::adapters::instrumentation::viprs_span;
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId, U8, U16};
use crate::domain::image::{Image, ImageMetadata, Interpretation};

use super::metadata::{bands_to_color_type, png_pixel_dims};
use super::state::{PNG_XMP_KEYWORD, PngEncoder};

fn map_png_encoding_error(error: png::EncodingError) -> ViprsError {
    match error {
        png::EncodingError::IoError(io_error) => ViprsError::Io(io_error),
        other => ViprsError::Codec(other.to_string()),
    }
}

fn encode_pixels<F: BandFormat>(image: &Image<F>) -> Result<(BitDepth, Cow<'_, [u8]>), ViprsError> {
    match F::ID {
        BandFormatId::U8 => {
            let bytes: &[u8] = bytemuck::cast_slice(image.pixels());
            Ok((BitDepth::Eight, Cow::Borrowed(bytes)))
        }
        BandFormatId::U16 => {
            let samples = bytemuck::cast_slice::<F::Sample, u16>(image.pixels());
            let mut be_bytes = vec![0u8; samples.len() * 2];
            for (dst, sample) in be_bytes.chunks_exact_mut(2).zip(samples.iter().copied()) {
                dst.copy_from_slice(&sample.to_be_bytes());
            }
            Ok((BitDepth::Sixteen, Cow::Owned(be_bytes)))
        }
        _ => Err(ViprsError::Codec(format!(
            "png: unsupported format {:?}",
            F::ID
        ))),
    }
}

fn encode_png<F: BandFormat>(
    image: &Image<F>,
    encoder: PngEncoder,
    strip_metadata: bool,
) -> Result<Vec<u8>, ViprsError> {
    viprs_span!(tracing::Level::INFO, "viprs.encode", format = "png");
    let mut output = Vec::new();
    encode_png_to_writer(image, encoder, strip_metadata, &mut output)?;
    Ok(output)
}

pub(super) fn encode_png_web_ready<F: BandFormat>(
    image: &Image<F>,
    encoder: &PngEncoder,
    strip_metadata: bool,
) -> Result<Vec<u8>, ViprsError> {
    if !image.metadata().has_icc_profile() {
        return encode_png(image, *encoder, strip_metadata);
    }

    match F::ID {
        BandFormatId::U8 => {
            // SAFETY: this match arm is reached only when `F::ID == BandFormatId::U8`,
            // which guarantees `F::Sample == u8` and makes the image layout identical.
            let image = unsafe { &*std::ptr::from_ref(image).cast::<Image<U8>>() }.clone();
            let normalized = normalize_web_output_u8(&image)?;
            encode_png(normalized.as_ref(), *encoder, strip_metadata)
        }
        BandFormatId::U16 => {
            // SAFETY: this match arm is reached only when `F::ID == BandFormatId::U16`,
            // which guarantees `F::Sample == u16` and makes the image layout identical.
            let image = unsafe { &*std::ptr::from_ref(image).cast::<Image<U16>>() }.clone();
            let normalized = normalize_web_output_u16(&image)?;
            encode_png(normalized.as_ref(), *encoder, strip_metadata)
        }
        _ => encode_png(image, *encoder, strip_metadata),
    }
}

pub(super) fn encode_png_to_writer_web_ready<F: BandFormat>(
    image: &Image<F>,
    encoder: &PngEncoder,
    strip_metadata: bool,
    writer: &mut dyn Write,
) -> Result<(), ViprsError> {
    if !image.metadata().has_icc_profile() {
        return encode_png_to_writer(image, *encoder, strip_metadata, writer);
    }

    match F::ID {
        BandFormatId::U8 => {
            // SAFETY: this match arm is reached only when `F::ID == BandFormatId::U8`,
            // which guarantees `F::Sample == u8` and makes the image layout identical.
            let image = unsafe { &*std::ptr::from_ref(image).cast::<Image<U8>>() }.clone();
            let normalized = normalize_web_output_u8(&image)?;
            encode_png_to_writer(normalized.as_ref(), *encoder, strip_metadata, writer)
        }
        BandFormatId::U16 => {
            // SAFETY: this match arm is reached only when `F::ID == BandFormatId::U16`,
            // which guarantees `F::Sample == u16` and makes the image layout identical.
            let image = unsafe { &*std::ptr::from_ref(image).cast::<Image<U16>>() }.clone();
            let normalized = normalize_web_output_u16(&image)?;
            encode_png_to_writer(normalized.as_ref(), *encoder, strip_metadata, writer)
        }
        _ => encode_png_to_writer(image, *encoder, strip_metadata, writer),
    }
}

fn encode_png_to_writer<F: BandFormat, W: Write>(
    image: &Image<F>,
    encoder: PngEncoder,
    strip_metadata: bool,
    output: W,
) -> Result<(), ViprsError> {
    let color_type = bands_to_color_type(image.bands())?;
    let (bit_depth, pixel_bytes) = encode_pixels(image)?;
    if encoder.interlace {
        let interlaced = encode_interlaced_png(
            image.width(),
            image.height(),
            color_type,
            bit_depth,
            pixel_bytes.as_ref(),
            image.metadata(),
            encoder,
            strip_metadata,
        )?;
        let mut output = output;
        output.write_all(&interlaced)?;
        return Ok(());
    }

    let mut info = Info::with_size(image.width(), image.height());
    info.color_type = color_type;
    info.bit_depth = bit_depth;
    info.interlaced = encoder.interlace;
    if !strip_metadata {
        info.pixel_dims = png_pixel_dims(image.metadata());
        info.icc_profile = image.metadata().icc_profile.as_deref().map(Cow::Borrowed);
        info.exif_metadata = image.metadata().exif.as_deref().map(Cow::Borrowed);
        if let Some(xmp) = image.metadata().xmp.as_deref() {
            let text = String::from_utf8(xmp.to_vec())
                .map_err(|_| ViprsError::Codec("png: XMP metadata must be valid UTF-8".into()))?;
            info.utf8_text.push(ITXtChunk::new(PNG_XMP_KEYWORD, text));
        }
    }
    if !strip_metadata
        && matches!(image.metadata().interpretation, Some(Interpretation::Srgb))
        && image.metadata().icc_profile.is_none()
    {
        info.srgb = Some(SrgbRenderingIntent::Perceptual);
    }

    let mut raw_encoder = RawPngEncoder::with_info(output, info).map_err(map_png_encoding_error)?;
    match encoder.compression.min(9) {
        0 => {
            raw_encoder.set_deflate_compression(DeflateCompression::NoCompression);
            raw_encoder.set_filter(encoder.filter);
        }
        level => {
            raw_encoder.set_deflate_compression(DeflateCompression::Level(level));
            raw_encoder.set_filter(encoder.filter);
        }
    }

    let mut writer = raw_encoder.write_header().map_err(map_png_encoding_error)?;
    writer
        .write_image_data(pixel_bytes.as_ref())
        .map_err(map_png_encoding_error)?;
    drop(writer);

    Ok(())
}

pub(super) const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
pub(super) const MAX_PNG_DECODED_IMAGE_BYTES: usize = 1 << 30;
pub(super) const ADAM7_PASSES: [(u32, u32, u32, u32); 7] = [
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
];

fn encode_interlaced_png(
    width: u32,
    height: u32,
    color_type: ColorType,
    bit_depth: BitDepth,
    pixel_bytes: &[u8],
    metadata: &ImageMetadata,
    encoder: PngEncoder,
    strip_metadata: bool,
) -> Result<Vec<u8>, ViprsError> {
    let mut output = Vec::new();
    output.extend_from_slice(&PNG_SIGNATURE);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.push(bit_depth as u8);
    ihdr.push(color_type as u8);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(1);
    write_chunk(&mut output, *b"IHDR", &ihdr);

    if !strip_metadata {
        if let Some(pixel_dims) = png_pixel_dims(metadata) {
            let mut phys = Vec::with_capacity(9);
            phys.extend_from_slice(&pixel_dims.xppu.to_be_bytes());
            phys.extend_from_slice(&pixel_dims.yppu.to_be_bytes());
            phys.push(pixel_dims.unit as u8);
            write_chunk(&mut output, *b"pHYs", &phys);
        }
        if let Some(icc_profile) = metadata.icc_profile.as_deref() {
            write_iccp_chunk(&mut output, icc_profile)?;
        } else if matches!(metadata.interpretation, Some(Interpretation::Srgb)) {
            write_chunk(
                &mut output,
                *b"sRGB",
                &[SrgbRenderingIntent::Perceptual as u8],
            );
        }
        if let Some(exif) = metadata.exif.as_deref() {
            write_chunk(&mut output, *b"eXIf", exif);
        }
    }

    let scanlines = adam7_scanlines(
        width,
        height,
        color_type,
        bit_depth,
        pixel_bytes,
        encoder.filter,
    );
    let compression = if encoder.compression == 0 {
        ZlibCompression::none()
    } else {
        ZlibCompression::new(u32::from(encoder.compression.min(9)))
    };
    let mut compressed = ZlibEncoder::new(Vec::new(), compression);
    compressed
        .write_all(&scanlines)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let compressed = compressed
        .finish()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    write_chunk(&mut output, *b"IDAT", &compressed);
    write_chunk(&mut output, *b"IEND", &[]);

    Ok(output)
}

fn adam7_scanlines(
    width: u32,
    height: u32,
    color_type: ColorType,
    bit_depth: BitDepth,
    pixel_bytes: &[u8],
    filter: Filter,
) -> Vec<u8> {
    let bytes_per_sample = match bit_depth {
        BitDepth::Sixteen => 2usize,
        _ => 1,
    };
    let bytes_per_pixel = color_type.samples() * bytes_per_sample;
    let total_scanline_bytes = ADAM7_PASSES
        .iter()
        .map(|&(x_start, y_start, x_step, y_step)| {
            let pass_width = adam7_extent(width, x_start, x_step) as usize;
            let pass_height = adam7_extent(height, y_start, y_step) as usize;
            pass_height * (1 + pass_width * bytes_per_pixel)
        })
        .sum();
    let mut scanlines = Vec::with_capacity(total_scanline_bytes);

    for (x_start, y_start, x_step, y_step) in ADAM7_PASSES {
        let pass_width = adam7_extent(width, x_start, x_step);
        let pass_height = adam7_extent(height, y_start, y_step);
        if pass_width == 0 || pass_height == 0 {
            continue;
        }

        let row_len = pass_width as usize * bytes_per_pixel;
        let mut previous_row = vec![0u8; row_len];
        let mut raw_row = vec![0u8; row_len];
        let mut filtered_row = vec![0u8; row_len];
        let mut candidate_row = vec![0u8; row_len];
        let mut has_previous_row = false;
        for row in 0..pass_height {
            let y = y_start + row * y_step;
            for column in 0..pass_width {
                let x = x_start + column * x_step;
                let pixel_offset = ((y * width + x) as usize) * bytes_per_pixel;
                let dst = column as usize * bytes_per_pixel;
                raw_row[dst..dst + bytes_per_pixel]
                    .copy_from_slice(&pixel_bytes[pixel_offset..pixel_offset + bytes_per_pixel]);
            }

            let previous = has_previous_row.then_some(previous_row.as_slice());
            let filter_type = filter_row_into(
                &raw_row,
                previous,
                bytes_per_pixel,
                filter,
                &mut filtered_row,
                &mut candidate_row,
            );
            scanlines.push(filter_type);
            scanlines.extend_from_slice(&filtered_row);
            previous_row.copy_from_slice(&raw_row);
            has_previous_row = true;
        }
    }

    scanlines
}

pub(super) fn adam7_extent(extent: u32, start: u32, step: u32) -> u32 {
    if extent <= start {
        0
    } else {
        (extent - start).div_ceil(step)
    }
}

fn adaptive_filter_row_into(
    row: &[u8],
    previous_row: Option<&[u8]>,
    bytes_per_pixel: usize,
    output: &mut [u8],
    scratch: &mut [u8],
) -> u8 {
    output.copy_from_slice(row);
    let mut best_filter = 0u8;
    let mut best_score = filter_score(output);

    for (filter_type, candidate_filter) in [
        (1u8, Filter::Sub),
        (2u8, Filter::Up),
        (3u8, Filter::Avg),
        (4u8, Filter::Paeth),
    ] {
        filter_row_bytes_into(
            row,
            previous_row,
            bytes_per_pixel,
            candidate_filter,
            scratch,
        );
        let score = filter_score(scratch);
        if score < best_score {
            best_score = score;
            best_filter = filter_type;
            output.copy_from_slice(scratch);
        }
    }

    best_filter
}

fn filter_row_into(
    row: &[u8],
    previous_row: Option<&[u8]>,
    bytes_per_pixel: usize,
    filter: Filter,
    output: &mut [u8],
    scratch: &mut [u8],
) -> u8 {
    match filter {
        Filter::Adaptive => {
            adaptive_filter_row_into(row, previous_row, bytes_per_pixel, output, scratch)
        }
        Filter::NoFilter => {
            filter_row_bytes_into(row, previous_row, bytes_per_pixel, filter, output);
            0
        }
        Filter::Sub => {
            filter_row_bytes_into(row, previous_row, bytes_per_pixel, filter, output);
            1
        }
        Filter::Up => {
            filter_row_bytes_into(row, previous_row, bytes_per_pixel, filter, output);
            2
        }
        Filter::Avg => {
            filter_row_bytes_into(row, previous_row, bytes_per_pixel, filter, output);
            3
        }
        Filter::Paeth => {
            filter_row_bytes_into(row, previous_row, bytes_per_pixel, filter, output);
            4
        }
        _ => adaptive_filter_row_into(row, previous_row, bytes_per_pixel, output, scratch),
    }
}

fn filter_row_bytes_into(
    row: &[u8],
    previous_row: Option<&[u8]>,
    bytes_per_pixel: usize,
    filter: Filter,
    output: &mut [u8],
) {
    let previous_row = previous_row.unwrap_or(&[]);
    for (index, &byte) in row.iter().enumerate() {
        let left = if index >= bytes_per_pixel {
            row[index - bytes_per_pixel]
        } else {
            0
        };
        let up = previous_row.get(index).copied().unwrap_or(0);
        let up_left = if index >= bytes_per_pixel {
            previous_row
                .get(index - bytes_per_pixel)
                .copied()
                .unwrap_or(0)
        } else {
            0
        };
        output[index] = match filter {
            Filter::Sub => byte.wrapping_sub(left),
            Filter::Up => byte.wrapping_sub(up),
            Filter::Avg => {
                let average = left.midpoint(up);
                byte.wrapping_sub(average)
            }
            Filter::Paeth => byte.wrapping_sub(paeth_predictor(left, up, up_left)),
            Filter::NoFilter | _ => byte,
        };
    }
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = i32::from(a);
    let b = i32::from(b);
    let c = i32::from(c);
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

fn filter_score(filtered: &[u8]) -> u64 {
    filtered
        .iter()
        .map(|&byte| u64::from(i16::from(byte as i8).unsigned_abs()))
        .sum()
}

fn write_chunk(output: &mut Vec<u8>, chunk_type: [u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    output.extend_from_slice(&chunk_type);
    output.extend_from_slice(data);

    let mut hasher = Hasher::new();
    hasher.update(&chunk_type);
    hasher.update(data);
    output.extend_from_slice(&hasher.finalize().to_be_bytes());
}

fn write_iccp_chunk(output: &mut Vec<u8>, icc_profile: &[u8]) -> Result<(), ViprsError> {
    let mut data = Vec::with_capacity(5 + icc_profile.len());
    data.extend_from_slice(b"icc");
    data.push(0);
    data.push(0);

    let mut encoder = ZlibEncoder::new(Vec::new(), ZlibCompression::default());
    encoder
        .write_all(icc_profile)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    data.extend_from_slice(
        &encoder
            .finish()
            .map_err(|e| ViprsError::Codec(e.to_string()))?,
    );
    write_chunk(output, *b"iCCP", &data);
    Ok(())
}
