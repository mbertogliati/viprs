use std::fs::File;
use std::io::{BufRead, BufReader, Seek};
use std::path::Path;

#[cfg(feature = "libspng")]
use super::state::PNG_XMP_KEYWORD;
use crc32fast::Hasher;
#[cfg(feature = "libspng")]
use png::ColorType;
use png::{BitDepth, Decoder as PngDecoder, Transformations};
#[cfg(feature = "libspng")]
use spng::{
    BitDepth as SpngBitDepth, ColorType as SpngColorType, CrcAction as SpngCrcAction,
    DecodeFlags as SpngDecodeFlags, Format as SpngFormat, Info as SpngInfo,
    raw::RawContext as SpngRawContext,
};

use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::Image;
#[cfg(feature = "libspng")]
use viprs_core::image::ImageMetadata;

use super::encode::MAX_PNG_DECODED_IMAGE_BYTES;
#[cfg(feature = "libspng")]
use super::metadata::build_png_metadata;
use super::metadata::{color_type_to_bands, png_metadata};
use super::state::{PNG_FILE_READER_CAPACITY, PngSequentialPathSession};

const PNG_SIGNATURE_LEN: usize = 8;
const PNG_CHUNK_HEADER_LEN: usize = 8;
const PNG_CHUNK_CRC_LEN: usize = 4;

pub(super) fn validate_png_critical_chunk_crcs(src: &[u8]) -> Result<(), ViprsError> {
    if src.len() < PNG_SIGNATURE_LEN {
        return Err(ViprsError::Codec(
            "png: input shorter than signature".into(),
        ));
    }

    let mut offset = PNG_SIGNATURE_LEN;
    while offset + PNG_CHUNK_HEADER_LEN + PNG_CHUNK_CRC_LEN <= src.len() {
        let chunk_len = u32::from_be_bytes([
            src[offset],
            src[offset + 1],
            src[offset + 2],
            src[offset + 3],
        ]) as usize;
        let chunk_type_start = offset + 4;
        let chunk_data_start = chunk_type_start + 4;
        let chunk_data_end = chunk_data_start.checked_add(chunk_len).ok_or_else(|| {
            ViprsError::Codec("png: chunk length overflows addressable memory".into())
        })?;
        let crc_end = chunk_data_end
            .checked_add(PNG_CHUNK_CRC_LEN)
            .ok_or_else(|| {
                ViprsError::Codec("png: chunk CRC overflows addressable memory".into())
            })?;
        if crc_end > src.len() {
            return Err(ViprsError::Codec("png: truncated chunk payload".into()));
        }

        let chunk_type = &src[chunk_type_start..chunk_data_start];
        let is_critical = chunk_type[0].is_ascii_uppercase();
        if is_critical {
            let expected = u32::from_be_bytes([
                src[chunk_data_end],
                src[chunk_data_end + 1],
                src[chunk_data_end + 2],
                src[chunk_data_end + 3],
            ]);
            let mut hasher = Hasher::new();
            hasher.update(&src[chunk_type_start..chunk_data_end]);
            let actual = hasher.finalize();
            if actual != expected {
                let name = std::str::from_utf8(chunk_type).unwrap_or("????");
                return Err(ViprsError::Codec(format!(
                    "png: invalid CRC for critical chunk {name}"
                )));
            }
        }

        if chunk_type == b"IEND" {
            return Ok(());
        }
        offset = crc_end;
    }

    Err(ViprsError::Codec("png: missing IEND chunk".into()))
}

pub(super) fn png_reader<R>(src: R) -> Result<png::Reader<R>, ViprsError>
where
    R: BufRead + Seek,
{
    png_decoder(src)
        .read_info()
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

pub(super) fn png_decoder<R>(src: R) -> PngDecoder<R>
where
    R: BufRead + Seek,
{
    let mut decoder = PngDecoder::new(src);
    decoder.set_transformations(Transformations::EXPAND);
    decoder
}

pub(super) fn png_file_reader(path: &Path) -> Result<BufReader<File>, ViprsError> {
    Ok(BufReader::with_capacity(
        PNG_FILE_READER_CAPACITY,
        File::open(path)?,
    ))
}

pub(super) fn open_png_sequential_path_session(
    path: &Path,
) -> Result<Option<PngSequentialPathSession>, ViprsError> {
    let reader = png_reader(png_file_reader(path)?)?;
    let info = reader.info();
    if info.interlaced {
        return Ok(None);
    }

    let width = info.width;
    let height = info.height;
    let bit_depth = info.bit_depth;
    let bands = color_type_to_bands(info.color_type);
    let row_len = reader
        .output_line_size(width)
        .ok_or_else(|| ViprsError::Codec("png: cannot determine row buffer size".into()))?;

    Ok(Some(PngSequentialPathSession {
        path: path.to_path_buf(),
        reader,
        row_scratch: vec![0; row_len],
        width,
        height,
        bands,
        bit_depth,
        next_source_y: 0,
    }))
}

#[cfg(feature = "libspng")]
fn libspng_output_format(
    color_type: SpngColorType,
    bit_depth: SpngBitDepth,
) -> Result<SpngFormat, ViprsError> {
    match bit_depth {
        // Mirror libvips: keep native PNG channel layout whenever libspng can
        // emit host-endian samples directly, and only request an expanded format
        // when the file is palette-indexed.
        SpngBitDepth::Eight => match color_type {
            SpngColorType::Indexed => Ok(SpngFormat::Rgb8),
            SpngColorType::Grayscale
            | SpngColorType::GrayscaleAlpha
            | SpngColorType::Truecolor
            | SpngColorType::TruecolorAlpha => Ok(SpngFormat::Png),
        },
        SpngBitDepth::Sixteen => Ok(SpngFormat::Png),
        depth => Err(ViprsError::Codec(format!(
            "png: libspng does not support bit depth {depth:?}"
        ))),
    }
}

/// Accumulate one source row into a **u16** `acc` using a horizontal box-shrink by `factor`.
///
/// Only valid when `factor * factor * 255 ≤ u16::MAX` (holds for factor ≤ 16).
/// u16 accumulators double the NEON lane count vs u32, giving ~2× accumulation throughput.
#[inline]
fn accumulate_row_u16<const BANDS: usize>(
    src_row: &[u8],
    acc: &mut [u16],
    factor: usize,
    dst_w: usize,
    src_w: usize,
) {
    let full = dst_w.min(src_w / factor);
    for (dx, block) in src_row[..full * factor * BANDS]
        .chunks_exact(factor * BANDS)
        .enumerate()
    {
        let acc_base = dx * BANDS;
        let mut local = [0u16; BANDS];
        for pixel in block.chunks_exact(BANDS) {
            for b in 0..BANDS {
                local[b] += u16::from(pixel[b]);
            }
        }
        for b in 0..BANDS {
            acc[acc_base + b] += local[b];
        }
    }
    if full < dst_w {
        let dx = full;
        let src_base = full * factor * BANDS;
        let acc_base = dx * BANDS;
        let mut local = [0u16; BANDS];
        for pixel in src_row[src_base..].chunks_exact(BANDS) {
            for b in 0..BANDS {
                local[b] += u16::from(pixel[b]);
            }
        }
        for b in 0..BANDS {
            acc[acc_base + b] += local[b];
        }
    }
}

/// Write one output row from a **u16** `acc`.
///
/// For power-of-two totals uses a multiply-shift to avoid division.
#[inline]
fn finalize_row_u16<const BANDS: usize>(
    acc: &[u16],
    dst: &mut [u8],
    dst_w: usize,
    total: u16,
    amend: u16,
) {
    let multiplier = (1u32 << 24) / u32::from(total);
    for dx in 0..dst_w {
        let acc_base = dx * BANDS;
        let dst_base = dx * BANDS;
        for b in 0..BANDS {
            let sum = acc[acc_base + b].saturating_add(amend);
            dst[dst_base + b] = ((u32::from(sum) * multiplier) >> 24) as u8;
        }
    }
}

/// Accumulate one source row into `acc` using a horizontal box-shrink by `factor`.
///
/// `BANDS` is monomorphized at compile time so the inner `for b in 0..BANDS` loop
/// is unrolled and the compiler can auto-vectorize the pixel accumulation.
/// Pixels within each output block are processed sequentially (contiguous read) so
/// the access pattern matches what LLVM/NEON expect for vectorization.
#[inline]
fn accumulate_row<const BANDS: usize>(
    src_row: &[u8],
    acc: &mut [u32],
    factor: usize,
    dst_w: usize,
    src_w: usize,
) {
    // Full blocks: factor*BANDS bytes per output pixel — sequential read.
    let full = dst_w.min(src_w / factor);
    for (dx, block) in src_row[..full * factor * BANDS]
        .chunks_exact(factor * BANDS)
        .enumerate()
    {
        let acc_base = dx * BANDS;
        // Keep per-block sums in local vars so the compiler can keep them in
        // registers across the inner loop and vectorize the accumulation.
        let mut local = [0u32; BANDS];
        for pixel in block.chunks_exact(BANDS) {
            for b in 0..BANDS {
                local[b] += u32::from(pixel[b]);
            }
        }
        for b in 0..BANDS {
            acc[acc_base + b] += local[b];
        }
    }
    // Partial trailing block (only when src_w is not a multiple of factor).
    if full < dst_w {
        let dx = full;
        let src_base = full * factor * BANDS;
        let acc_base = dx * BANDS;
        let mut local = [0u32; BANDS];
        for pixel in src_row[src_base..].chunks_exact(BANDS) {
            for b in 0..BANDS {
                local[b] += u32::from(pixel[b]);
            }
        }
        for b in 0..BANDS {
            acc[acc_base + b] += local[b];
        }
    }
}

/// Write one output row from `acc`.
///
/// Uses a fixed-point multiply-shift (same as libvips `UCHAR_SHRINK`) to avoid
/// integer division in the inner loop. `multiplier = (1<<32) / (256 * total)`,
/// then `out = (sum * multiplier) >> 24`. For power-of-two totals this is exact.
#[inline]
fn finalize_row<const BANDS: usize>(
    acc: &[u32],
    dst: &mut [u8],
    dst_w: usize,
    total: u32,
    amend: u32,
) {
    // multiplier = (1<<32) / (256 * total) so that (sum * multiplier) >> 24 == sum/total.
    let multiplier = ((1u64 << 32) / (u64::from(total) * 256)) as u32;
    for dx in 0..dst_w {
        let acc_base = dx * BANDS;
        let dst_base = dx * BANDS;
        for b in 0..BANDS {
            let sum = acc[acc_base + b] + amend;
            // SAFETY: multiplier construction guarantees result fits in u8.
            dst[dst_base + b] = ((u64::from(sum) * u64::from(multiplier)) >> 24) as u8;
        }
    }
}

/// Decode a PNG row-by-row and apply an integer box-filter shrink by `factor` inline.
///
/// This avoids allocating the full decoded raster (e.g. 200 MB for 8192×8192 RGB).
/// Peak resident memory is one source-width row buffer plus the shrunken output.
///
/// Returns `(dst_width, dst_height, bands, pixels_u8)`.
///
/// `factor` must be a power of two in {2, 4, 8, 16}; if the image dimensions are not
/// exact multiples the last partial block is averaged over its actual pixel count.
pub(super) fn decode_png_with_box_shrink_u8<R: BufRead + Seek>(
    src: R,
    factor: usize,
) -> Result<(u32, u32, u32, Vec<u8>), ViprsError> {
    let dec = png_decoder(src);
    let mut reader = dec
        .read_info()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    let info = reader.info();
    let src_w = info.width as usize;
    let src_h = info.height as usize;
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    let bands = color_type_to_bands(color_type) as usize;

    if bit_depth != BitDepth::Eight {
        return Err(ViprsError::Codec(format!(
            "png: box-shrink decode only supports 8-bit images, got {bit_depth:?}"
        )));
    }

    let dst_w = src_w.div_ceil(factor);
    let dst_h = src_h.div_ceil(factor);
    let has_partial_columns = !src_w.is_multiple_of(factor);

    let mut src_row = vec![0u8; src_w * bands];
    let mut dst_pixels = vec![0u8; dst_w * dst_h * bands];

    // Use u16 accumulators when the total (factor²) fits: factor²×255 ≤ u16::MAX.
    // For factor=16 (our primary case): 256×255 = 65280 ≤ 65535 ✓
    // For factor=8:  64×255 = 16320 ≤ 65535 ✓
    // For factor=4:  16×255 = 4080  ≤ 65535 ✓
    // For factor=2:  4×255  = 1020  ≤ 65535 ✓
    let use_u16_acc = u16::try_from(factor * factor * 255).is_ok();

    if use_u16_acc {
        // Fast path: u16 accumulator — twice as many NEON lanes as u32.
        let total_u16 = (factor * factor) as u16;
        let amend_u16 = total_u16 / 2;
        let mut acc: Vec<u16> = vec![0u16; dst_w * bands];

        let mut src_y = 0usize;
        while src_y < src_h {
            let block_end = (src_y + factor).min(src_h);
            let v_pixels = block_end - src_y;
            let dy = src_y / factor;

            acc.fill(0);

            for _ in src_y..block_end {
                reader
                    .read_row(&mut src_row)
                    .map_err(|e| ViprsError::Codec(e.to_string()))?
                    .ok_or_else(|| ViprsError::Codec("png: unexpected end of image".into()))?;

                match bands {
                    1 => accumulate_row_u16::<1>(&src_row, &mut acc, factor, dst_w, src_w),
                    2 => accumulate_row_u16::<2>(&src_row, &mut acc, factor, dst_w, src_w),
                    3 => accumulate_row_u16::<3>(&src_row, &mut acc, factor, dst_w, src_w),
                    _ => accumulate_row_u16::<4>(&src_row, &mut acc, factor, dst_w, src_w),
                }
            }

            let dst_row = &mut dst_pixels[dy * dst_w * bands..(dy + 1) * dst_w * bands];
            if v_pixels == factor && !has_partial_columns {
                match bands {
                    1 => finalize_row_u16::<1>(&acc, dst_row, dst_w, total_u16, amend_u16),
                    2 => finalize_row_u16::<2>(&acc, dst_row, dst_w, total_u16, amend_u16),
                    3 => finalize_row_u16::<3>(&acc, dst_row, dst_w, total_u16, amend_u16),
                    _ => finalize_row_u16::<4>(&acc, dst_row, dst_w, total_u16, amend_u16),
                }
            } else {
                for dx in 0..dst_w {
                    let sx0 = dx * factor;
                    let h_pixels = ((sx0 + factor).min(src_w)) - sx0;
                    let total = ((v_pixels * h_pixels) as u32).max(1);
                    for b in 0..bands {
                        let sum = u32::from(acc[dx * bands + b]);
                        dst_row[dx * bands + b] = ((sum + total / 2) / total) as u8;
                    }
                }
            }

            src_y = block_end;
        }
    } else {
        // Fallback: u32 accumulator for large factors.
        let total_u32 = (factor * factor) as u32;
        let amend_u32 = total_u32 / 2;
        let mut acc: Vec<u32> = vec![0u32; dst_w * bands];

        let mut src_y = 0usize;
        while src_y < src_h {
            let block_end = (src_y + factor).min(src_h);
            let v_pixels = block_end - src_y;
            let dy = src_y / factor;

            acc.fill(0);

            for _ in src_y..block_end {
                reader
                    .read_row(&mut src_row)
                    .map_err(|e| ViprsError::Codec(e.to_string()))?
                    .ok_or_else(|| ViprsError::Codec("png: unexpected end of image".into()))?;

                match bands {
                    1 => accumulate_row::<1>(&src_row, &mut acc, factor, dst_w, src_w),
                    2 => accumulate_row::<2>(&src_row, &mut acc, factor, dst_w, src_w),
                    3 => accumulate_row::<3>(&src_row, &mut acc, factor, dst_w, src_w),
                    _ => accumulate_row::<4>(&src_row, &mut acc, factor, dst_w, src_w),
                }
            }

            let dst_row = &mut dst_pixels[dy * dst_w * bands..(dy + 1) * dst_w * bands];
            if v_pixels == factor && !has_partial_columns {
                match bands {
                    1 => finalize_row::<1>(&acc, dst_row, dst_w, total_u32, amend_u32),
                    2 => finalize_row::<2>(&acc, dst_row, dst_w, total_u32, amend_u32),
                    3 => finalize_row::<3>(&acc, dst_row, dst_w, total_u32, amend_u32),
                    _ => finalize_row::<4>(&acc, dst_row, dst_w, total_u32, amend_u32),
                }
            } else {
                for dx in 0..dst_w {
                    let sx0 = dx * factor;
                    let h_pixels = ((sx0 + factor).min(src_w)) - sx0;
                    let total = ((v_pixels * h_pixels) as u32).max(1);
                    for b in 0..bands {
                        let sum = acc[dx * bands + b];
                        dst_row[dx * bands + b] = ((sum + total / 2) / total) as u8;
                    }
                }
            }

            src_y = block_end;
        }
    }

    Ok((dst_w as u32, dst_h as u32, bands as u32, dst_pixels))
}

pub(super) fn decode_png_with_png_crate_reader<F: BandFormat, R>(
    src: R,
) -> Result<Image<F>, ViprsError>
where
    R: BufRead + Seek,
{
    let dec = png_decoder(src);
    let mut reader = dec
        .read_info()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    let bands = color_type_to_bands(color_type);
    let metadata = png_metadata(info);

    let buf_size = reader
        .output_buffer_size()
        .ok_or_else(|| ViprsError::Codec("png: cannot determine buffer size".into()))?;
    if buf_size > MAX_PNG_DECODED_IMAGE_BYTES {
        return Err(ViprsError::Codec(format!(
            "png: decoded image requires {buf_size} bytes, exceeds safety limit {MAX_PNG_DECODED_IMAGE_BYTES}"
        )));
    }
    let mut buf = vec![0; buf_size];
    reader
        .next_frame(&mut buf)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    let samples: Vec<F::Sample> = match (F::ID, bit_depth) {
        (BandFormatId::U8, BitDepth::Eight) => {
            // SAFETY: F::ID == U8 implies F::Sample == u8 (bytemuck::Pod,
            // align 1, size 1). The cast is therefore a no-op in memory.
            bytemuck::allocation::try_cast_vec::<u8, F::Sample>(buf)
                .map_err(|(e, _)| ViprsError::Codec(format!("png: cast error: {e:?}")))?
        }
        (BandFormatId::U16, BitDepth::Sixteen) => {
            // PNG stores 16-bit samples in big-endian order. Convert to the
            // platform's native endian before handing data to the caller.
            let native: Vec<u16> = buf
                .chunks_exact(2)
                .map(|b| u16::from_be_bytes([b[0], b[1]]))
                .collect();
            // SAFETY: F::ID == U16 implies F::Sample == u16 (bytemuck::Pod).
            bytemuck::allocation::try_cast_vec::<u16, F::Sample>(native)
                .map_err(|(e, _)| ViprsError::Codec(format!("png: cast error: {e:?}")))?
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                F::ID,
                bit_depth
            )));
        }
    };

    Image::from_buffer(width, height, bands, samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

#[cfg(feature = "libspng")]
fn spng_output_bands(output_format: SpngFormat, input_color_type: SpngColorType) -> u32 {
    match output_format {
        SpngFormat::G8 => 1,
        SpngFormat::Ga8 | SpngFormat::Ga16 => 2,
        SpngFormat::Rgb8 => 3,
        SpngFormat::Rgba8 | SpngFormat::Rgba16 => 4,
        SpngFormat::Png | SpngFormat::Raw => input_color_type.samples() as u32,
    }
}

#[cfg(feature = "libspng")]
fn png_metadata_from_libspng<R: std::io::Read>(
    ctx: &SpngRawContext<R>,
    info: SpngInfo,
) -> ImageMetadata {
    let color_type = match info.color_type {
        SpngColorType::Grayscale => ColorType::Grayscale,
        SpngColorType::Truecolor => ColorType::Rgb,
        SpngColorType::Indexed => ColorType::Indexed,
        SpngColorType::GrayscaleAlpha => ColorType::GrayscaleAlpha,
        SpngColorType::TruecolorAlpha => ColorType::Rgba,
    };
    let bit_depth = match info.bit_depth {
        SpngBitDepth::One => BitDepth::One,
        SpngBitDepth::Two => BitDepth::Two,
        SpngBitDepth::Four => BitDepth::Four,
        SpngBitDepth::Eight => BitDepth::Eight,
        SpngBitDepth::Sixteen => BitDepth::Sixteen,
    };
    let (xres, yres) = match ctx.get_phys() {
        Ok(phys) if phys.unit_specifier == 1 => (
            Some(f64::from(phys.ppu_x) / 1_000.0),
            Some(f64::from(phys.ppu_y) / 1_000.0),
        ),
        _ => (None, None),
    };
    let icc_profile = ctx
        .get_iccp()
        .ok()
        .map(|profile| profile.profile().to_vec());
    let exif = ctx.get_exif().ok().map(|exif| exif.data().to_vec());
    let xmp = ctx.get_text().ok().and_then(|texts| {
        texts.iter().find_map(|text| {
            (text.keyword().ok() == Some(PNG_XMP_KEYWORD))
                .then(|| text.text().ok().map(str::as_bytes).map(ToOwned::to_owned))
                .flatten()
        })
    });

    build_png_metadata(
        color_type,
        bit_depth,
        ctx.get_srgb().is_ok(),
        icc_profile,
        exif,
        xmp,
        xres,
        yres,
    )
}

#[cfg(feature = "libspng")]
fn new_libspng_context<R>() -> Result<SpngRawContext<R>, ViprsError> {
    let mut ctx = SpngRawContext::new().map_err(|e| ViprsError::Codec(e.to_string()))?;
    ctx.set_crc_action(SpngCrcAction::Error, SpngCrcAction::Error)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(ctx)
}

#[cfg(feature = "libspng")]
fn decode_libspng_rows<R>(
    ctx: &mut SpngRawContext<R>,
    output_format: SpngFormat,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    ctx.decode_image(output, output_format, SpngDecodeFlags::empty())
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

#[cfg(feature = "libspng")]
fn decode_png_with_libspng_reader<F: BandFormat, R: std::io::Read>(
    src: R,
) -> Result<Image<F>, ViprsError> {
    let mut ctx = new_libspng_context()?;
    ctx.set_png_stream(src)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    let ihdr = ctx
        .get_ihdr()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let color_type =
        SpngColorType::try_from(ihdr.color_type).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let bit_depth =
        SpngBitDepth::try_from(ihdr.bit_depth).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let info = SpngInfo {
        width: ihdr.width,
        height: ihdr.height,
        color_type,
        bit_depth,
    };
    let width = info.width;
    let height = info.height;
    let bands = color_type_to_bands(match color_type {
        SpngColorType::Grayscale => ColorType::Grayscale,
        SpngColorType::Truecolor => ColorType::Rgb,
        SpngColorType::Indexed => ColorType::Indexed,
        SpngColorType::GrayscaleAlpha => ColorType::GrayscaleAlpha,
        SpngColorType::TruecolorAlpha => ColorType::Rgba,
    });
    let output_format = libspng_output_format(color_type, bit_depth)?;
    let output_size = ctx
        .decoded_image_size(output_format)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut buf = vec![0; output_size];
    decode_libspng_rows(&mut ctx, output_format, &mut buf)?;
    let metadata = png_metadata_from_libspng(&ctx, info);

    let decoded_bands = spng_output_bands(output_format, color_type);
    if decoded_bands != bands {
        return Err(ViprsError::Codec(format!(
            "png: libspng decoded {decoded_bands} bands, expected {bands}"
        )));
    }

    let samples: Vec<F::Sample> = match (F::ID, bit_depth) {
        (BandFormatId::U8, SpngBitDepth::Eight) => {
            bytemuck::allocation::try_cast_vec::<u8, F::Sample>(buf)
                .map_err(|(e, _)| ViprsError::Codec(format!("png: cast error: {e:?}")))?
        }
        (BandFormatId::U16, SpngBitDepth::Sixteen) => {
            let native: Vec<u16> = buf
                .chunks_exact(2)
                .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
                .collect();
            bytemuck::allocation::try_cast_vec::<u16, F::Sample>(native)
                .map_err(|(e, _)| ViprsError::Codec(format!("png: cast error: {e:?}")))?
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                F::ID,
                bit_depth
            )));
        }
    };

    Image::from_buffer(width, height, bands, samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

#[cfg(feature = "libspng")]
pub(super) fn decode_png_with_libspng<F: BandFormat>(src: &[u8]) -> Result<Image<F>, ViprsError> {
    decode_png_with_libspng_reader::<F, _>(std::io::Cursor::new(src))
        .or_else(|_| decode_png_with_png_crate_reader(std::io::Cursor::new(src)))
}
