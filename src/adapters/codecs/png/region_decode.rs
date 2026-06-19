use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek, Write};
use std::path::Path;

use png::{BitDepth, Decoder as PngReader};

use crate::adapters::instrumentation::viprs_span;
use crate::domain::codec_options::{LoadOptions, SaveOptions};
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId};
use crate::domain::image::{Image, Region};
use crate::ports::codec::{ImageDecoder, ImageEncoder, ImageMetadataProbe, TileImageDecoder};

#[cfg(feature = "libspng")]
use super::decode_full::decode_png_with_libspng;
use super::decode_full::{
    decode_png_with_box_shrink_u8, decode_png_with_png_crate_reader,
    open_png_sequential_path_session, png_file_reader, png_reader,
};
use super::encode::{
    ADAM7_PASSES, MAX_PNG_DECODED_IMAGE_BYTES, adam7_extent, encode_png_to_writer_web_ready,
    encode_png_web_ready,
};
use super::metadata::{color_type_to_bands, png_filter, png_metadata};
#[cfg(test)]
use super::state::PNG_ROW_DECODE_PROBE;
use super::state::{PngCodec, PngEncoder, PngInterlacedRaster, PngSequentialPathSession};

fn validate_png_region_output<F: BandFormat>(
    region: Region,
    bands: u32,
    output: &[u8],
) -> Result<(), ViprsError> {
    let expected = region
        .checked_pixel_count()
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .and_then(|samples| samples.checked_mul(std::mem::size_of::<F::Sample>()))
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width: region.width,
            height: region.height,
            bands,
            bytes: u128::from(region.width)
                * u128::from(region.height)
                * u128::from(bands)
                * u128::from(std::mem::size_of::<F::Sample>() as u64),
            limit_bytes: usize::MAX as u128,
            details: "png region output buffer exceeds addressable memory",
        })?;

    if output.len() != expected {
        return Err(ViprsError::Codec(format!(
            "png: output buffer size mismatch (got {}, expected {expected})",
            output.len()
        )));
    }

    Ok(())
}

pub(super) fn clamp_region_coordinate(origin: i32, offset: u32, limit: u32) -> i32 {
    (origin + offset as i32).clamp(0, limit as i32 - 1)
}

fn copy_png_row_u8(
    row: &[u8],
    image_width: u32,
    bands: usize,
    region: Region,
    out_y: u32,
    output: &mut [u8],
) {
    let pixel_stride = bands;
    for out_x in 0..region.width {
        let src_x = clamp_region_coordinate(region.x, out_x, image_width) as usize;
        let src = src_x * pixel_stride;
        let dst = (out_y as usize * region.width as usize + out_x as usize) * pixel_stride;
        output[dst..dst + pixel_stride].copy_from_slice(&row[src..src + pixel_stride]);
    }
}

fn copy_png_row_u16(
    row: &[u8],
    image_width: u32,
    bands: usize,
    region: Region,
    out_y: u32,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let output_samples: &mut [u16] = bytemuck::try_cast_slice_mut(output)
        .map_err(|_| ViprsError::Codec("png: output buffer size mismatch".into()))?;
    for out_x in 0..region.width {
        let src_x = clamp_region_coordinate(region.x, out_x, image_width) as usize;
        let src = src_x * bands * 2;
        let dst = (out_y as usize * region.width as usize + out_x as usize) * bands;
        for band in 0..bands {
            let sample = src + band * 2;
            output_samples[dst + band] = u16::from_be_bytes([row[sample], row[sample + 1]]);
        }
    }

    Ok(())
}

fn decode_png_region_rows<F: BandFormat, R>(
    reader: &mut png::Reader<R>,
    row: &mut [u8],
    bit_depth: BitDepth,
    bands: u32,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError>
where
    R: BufRead + Seek,
{
    validate_png_region_output::<F>(region, bands, output)?;
    if region.width == 0 || region.height == 0 {
        return Ok(());
    }

    let width = reader.info().width;
    let height = reader.info().height;
    let max_source_y = (0..region.height)
        .map(|out_y| clamp_region_coordinate(region.y, out_y, height))
        .max()
        .unwrap_or(0) as u32;

    for source_y in 0..=max_source_y {
        #[cfg(test)]
        let _row_decode_probe = PNG_ROW_DECODE_PROBE.enter();
        reader
            .read_row(row)
            .map_err(|e| ViprsError::Codec(e.to_string()))?
            .ok_or_else(|| ViprsError::Codec("png: unexpected end of image rows".into()))?;

        let needed = (0..region.height)
            .any(|out_y| clamp_region_coordinate(region.y, out_y, height) as u32 == source_y);
        if !needed {
            continue;
        }

        let bands = bands as usize;
        for out_y in 0..region.height {
            if clamp_region_coordinate(region.y, out_y, height) as u32 != source_y {
                continue;
            }

            match (F::ID, bit_depth) {
                (BandFormatId::U8, BitDepth::Eight) => {
                    copy_png_row_u8(&row, width, bands, region, out_y, output);
                }
                (BandFormatId::U16, BitDepth::Sixteen) => {
                    copy_png_row_u16(&row, width, bands, region, out_y, output)?;
                }
                _ => {
                    return Err(ViprsError::Codec(format!(
                        "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                        F::ID,
                        bit_depth
                    )));
                }
            }
        }
    }

    Ok(())
}

fn copy_png_full_width_row_u16(
    row: &[u8],
    bands: usize,
    region: Region,
    out_y: u32,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let output_samples: &mut [u16] = bytemuck::try_cast_slice_mut(output)
        .map_err(|_| ViprsError::Codec("png: output buffer size mismatch".into()))?;
    let dst = out_y as usize * region.width as usize * bands;
    for (sample_idx, chunk) in row.chunks_exact(2).enumerate() {
        output_samples[dst + sample_idx] = u16::from_be_bytes([chunk[0], chunk[1]]);
    }
    Ok(())
}

fn decode_png_sequential_session_into<F: BandFormat>(
    session: &mut PngSequentialPathSession,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    validate_png_region_output::<F>(region, session.bands, output)?;
    if region.width == 0 || region.height == 0 {
        return Ok(());
    }

    if region.x != 0 || region.width != session.width || region.y < 0 {
        return Err(ViprsError::Codec(
            "png: sequential session requires in-bounds full-width strips".into(),
        ));
    }

    let start_y = region.y as u32;
    if start_y != session.next_source_y {
        return Err(ViprsError::Codec(format!(
            "png: sequential strip order mismatch (got y={}, expected {})",
            region.y, session.next_source_y
        )));
    }

    let available_rows = session.height.saturating_sub(session.next_source_y);
    let rows_to_decode = available_rows.min(region.height);
    let bands = session.bands as usize;

    for out_y in 0..rows_to_decode {
        session
            .reader
            .read_row(&mut session.row_scratch)
            .map_err(|e| ViprsError::Codec(e.to_string()))?
            .ok_or_else(|| ViprsError::Codec("png: unexpected end of image rows".into()))?;
        match (F::ID, session.bit_depth) {
            (BandFormatId::U8, BitDepth::Eight) => {
                let row_bytes = session.width as usize * bands;
                let dst = out_y as usize * row_bytes;
                output[dst..dst + row_bytes].copy_from_slice(&session.row_scratch[..row_bytes]);
            }
            (BandFormatId::U16, BitDepth::Sixteen) => {
                copy_png_full_width_row_u16(&session.row_scratch, bands, region, out_y, output)?;
            }
            _ => {
                return Err(ViprsError::Codec(format!(
                    "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                    F::ID,
                    session.bit_depth
                )));
            }
        }
        session.next_source_y += 1;
    }

    if rows_to_decode < region.height {
        if rows_to_decode == 0 {
            return Err(ViprsError::Codec(
                "png: sequential strip request starts past end of image".into(),
            ));
        }
        let row_bytes = session.width as usize * bands * std::mem::size_of::<F::Sample>();
        for out_y in rows_to_decode..region.height {
            match (F::ID, session.bit_depth) {
                (BandFormatId::U8, BitDepth::Eight) => {
                    let src = (rows_to_decode as usize - 1) * row_bytes;
                    let dst = out_y as usize * row_bytes;
                    let (head, tail) = output.split_at_mut(dst);
                    let previous = &head[src..src + row_bytes];
                    tail[..row_bytes].copy_from_slice(previous);
                }
                (BandFormatId::U16, BitDepth::Sixteen) => {
                    copy_png_full_width_row_u16(
                        &session.row_scratch,
                        bands,
                        region,
                        out_y,
                        output,
                    )?;
                }
                _ => {
                    return Err(ViprsError::Codec(
                        "png: unexpected sequential decode format/bit-depth mismatch".into(),
                    ));
                }
            }
        }
    }

    Ok(())
}

fn png_bytes_per_sample(bit_depth: BitDepth) -> Result<usize, ViprsError> {
    match bit_depth {
        BitDepth::Eight => Ok(1),
        BitDepth::Sixteen => Ok(2),
        _ => Err(ViprsError::Codec(format!(
            "png: unsupported bit depth {bit_depth:?}"
        ))),
    }
}

pub(super) fn decode_png_full_raster_with_png_crate<R>(
    src: R,
) -> Result<PngInterlacedRaster, ViprsError>
where
    R: BufRead + Seek,
{
    #[cfg(test)]
    PNG_ROW_DECODE_PROBE.record_full_raster_decode();
    let mut reader = PngReader::new(src)
        .read_info()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let bit_depth = info.bit_depth;
    let bands = color_type_to_bands(info.color_type);
    let pixels_len = reader
        .output_buffer_size()
        .ok_or_else(|| ViprsError::Codec("png: cannot determine buffer size".into()))?;
    if pixels_len > MAX_PNG_DECODED_IMAGE_BYTES {
        return Err(ViprsError::Codec(format!(
            "png: decoded image requires {pixels_len} bytes, exceeds safety limit {MAX_PNG_DECODED_IMAGE_BYTES}"
        )));
    }
    let mut pixels = vec![0; pixels_len];
    reader
        .next_frame(&mut pixels)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(PngInterlacedRaster {
        width,
        height,
        bands,
        bit_depth,
        pixels,
    })
}

fn decode_png_region_from_full_raster<F: BandFormat>(
    raster: &PngInterlacedRaster,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    validate_png_region_output::<F>(region, raster.bands, output)?;
    if region.width == 0 || region.height == 0 {
        return Ok(());
    }

    let bands = raster.bands as usize;
    let row_bytes = raster.width as usize * bands * png_bytes_per_sample(raster.bit_depth)?;
    match (F::ID, raster.bit_depth) {
        (BandFormatId::U8, BitDepth::Eight) => {
            for out_y in 0..region.height {
                let src_y = clamp_region_coordinate(region.y, out_y, raster.height) as usize;
                let row_start = src_y * row_bytes;
                let row = &raster.pixels[row_start..row_start + row_bytes];
                copy_png_row_u8(row, raster.width, bands, region, out_y, output);
            }
        }
        (BandFormatId::U16, BitDepth::Sixteen) => {
            for out_y in 0..region.height {
                let src_y = clamp_region_coordinate(region.y, out_y, raster.height) as usize;
                let row_start = src_y * row_bytes;
                let row = &raster.pixels[row_start..row_start + row_bytes];
                copy_png_row_u16(row, raster.width, bands, region, out_y, output)?;
            }
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                F::ID,
                raster.bit_depth
            )));
        }
    }

    Ok(())
}

fn decode_png_region_interlaced_rows<F: BandFormat, R>(
    reader: &mut png::Reader<R>,
    bit_depth: BitDepth,
    bands: u32,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError>
where
    R: BufRead + Seek,
{
    validate_png_region_output::<F>(region, bands, output)?;
    if region.width == 0 || region.height == 0 {
        return Ok(());
    }

    let width = reader.info().width;
    let height = reader.info().height;
    let bands = bands as usize;
    let bytes_per_sample = png_bytes_per_sample(bit_depth)?;
    let bytes_per_pixel = bands * bytes_per_sample;

    match (F::ID, bit_depth) {
        (BandFormatId::U8, BitDepth::Eight) => {
            for (x_start, y_start, x_step, y_step) in ADAM7_PASSES {
                let pass_width = adam7_extent(width, x_start, x_step);
                let pass_height = adam7_extent(height, y_start, y_step);
                if pass_width == 0 || pass_height == 0 {
                    continue;
                }

                for pass_row in 0..pass_height {
                    let y = y_start + pass_row * y_step;
                    #[cfg(test)]
                    let _row_decode_probe = PNG_ROW_DECODE_PROBE.enter();
                    let row = reader
                        .next_interlaced_row()
                        .map_err(|e| ViprsError::Codec(e.to_string()))?
                        .ok_or_else(|| {
                            ViprsError::Codec("png: unexpected end of Adam7 rows".into())
                        })?;
                    if !matches!(row.interlace(), png::InterlaceInfo::Adam7(_)) {
                        return Err(ViprsError::Codec(
                            "png: expected Adam7 row while decoding interlaced PNG".into(),
                        ));
                    }

                    let pass_row = row.data();
                    let expected_len = pass_width as usize * bytes_per_pixel;
                    if pass_row.len() != expected_len {
                        return Err(ViprsError::Codec(format!(
                            "png: Adam7 row size mismatch (got {}, expected {expected_len})",
                            pass_row.len()
                        )));
                    }

                    for out_y in 0..region.height {
                        if clamp_region_coordinate(region.y, out_y, height) as u32 != y {
                            continue;
                        }

                        for out_x in 0..region.width {
                            let src_x = clamp_region_coordinate(region.x, out_x, width) as u32;
                            if src_x < x_start {
                                continue;
                            }
                            let delta = src_x - x_start;
                            if delta % x_step != 0 {
                                continue;
                            }

                            let pass_column = (delta / x_step) as usize;
                            let src = pass_column * bytes_per_pixel;
                            let dst = (out_y as usize * region.width as usize + out_x as usize)
                                * bytes_per_pixel;
                            output[dst..dst + bytes_per_pixel]
                                .copy_from_slice(&pass_row[src..src + bytes_per_pixel]);
                        }
                    }
                }
            }
        }
        (BandFormatId::U16, BitDepth::Sixteen) => {
            let output_samples: &mut [u16] = bytemuck::try_cast_slice_mut(output)
                .map_err(|_| ViprsError::Codec("png: output buffer size mismatch".into()))?;

            for (x_start, y_start, x_step, y_step) in ADAM7_PASSES {
                let pass_width = adam7_extent(width, x_start, x_step);
                let pass_height = adam7_extent(height, y_start, y_step);
                if pass_width == 0 || pass_height == 0 {
                    continue;
                }

                for pass_row in 0..pass_height {
                    let y = y_start + pass_row * y_step;
                    #[cfg(test)]
                    let _row_decode_probe = PNG_ROW_DECODE_PROBE.enter();
                    let row = reader
                        .next_interlaced_row()
                        .map_err(|e| ViprsError::Codec(e.to_string()))?
                        .ok_or_else(|| {
                            ViprsError::Codec("png: unexpected end of Adam7 rows".into())
                        })?;
                    if !matches!(row.interlace(), png::InterlaceInfo::Adam7(_)) {
                        return Err(ViprsError::Codec(
                            "png: expected Adam7 row while decoding interlaced PNG".into(),
                        ));
                    }

                    let pass_row = row.data();
                    let expected_len = pass_width as usize * bytes_per_pixel;
                    if pass_row.len() != expected_len {
                        return Err(ViprsError::Codec(format!(
                            "png: Adam7 row size mismatch (got {}, expected {expected_len})",
                            pass_row.len()
                        )));
                    }

                    for out_y in 0..region.height {
                        if clamp_region_coordinate(region.y, out_y, height) as u32 != y {
                            continue;
                        }

                        for out_x in 0..region.width {
                            let src_x = clamp_region_coordinate(region.x, out_x, width) as u32;
                            if src_x < x_start {
                                continue;
                            }
                            let delta = src_x - x_start;
                            if delta % x_step != 0 {
                                continue;
                            }

                            let pass_column = (delta / x_step) as usize;
                            let src = pass_column * bytes_per_pixel;
                            let dst =
                                (out_y as usize * region.width as usize + out_x as usize) * bands;
                            for band in 0..bands {
                                let sample = src + band * bytes_per_sample;
                                output_samples[dst + band] =
                                    u16::from_be_bytes([pass_row[sample], pass_row[sample + 1]]);
                            }
                        }
                    }
                }
            }
        }
        _ => {
            return Err(ViprsError::Codec(format!(
                "png: format/bit-depth mismatch — requested {:?} but file has {:?}",
                F::ID,
                bit_depth
            )));
        }
    }

    if reader
        .next_interlaced_row()
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .is_some()
    {
        return Err(ViprsError::Codec(
            "png: Adam7 decoder left unread interlaced rows".into(),
        ));
    }

    Ok(())
}

// ── ImageDecoder ──────────────────────────────────────────────────────────────

impl ImageDecoder for PngCodec {
    fn format_name(&self) -> &'static str {
        "png"
    }

    fn can_decode_path(&self, path: &Path) -> bool {
        path.extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
    }

    /// Returns `true` when `header` starts with the 8-byte PNG magic signature.
    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        const PNG_MAGIC: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
        header.len() >= 8 && header[..8] == PNG_MAGIC
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        viprs_span!(tracing::Level::INFO, "viprs.decode", format = "png");
        // The pure-Rust `png` crate (fdeflate) is faster than libspng's
        // bundled static zlib for full-frame decode. Use libspng only as
        // a fallback for layouts the Rust decoder cannot handle.
        let result = decode_png_with_png_crate_reader::<F, _>(Cursor::new(src));
        if result.is_ok() {
            return result;
        }

        #[cfg(feature = "libspng")]
        if let Ok(image) = decode_png_with_libspng(src) {
            return Ok(image);
        }

        result
    }

    /// PNG does not support shrink-on-load; all `opts` fields are ignored.
    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        // max_dimension: not supported — decoder does not expose pre-decode scaling.
        self.decode(src)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        // When a shrink hint is present and the target format is U8, decode
        // row-by-row with an inline box filter. This avoids materialising the
        // full decoded image (e.g. 200 MB for 8192×8192 RGB) before shrinking.
        if F::ID == BandFormatId::U8 {
            if let Some(factor) = opts.shrink_factor {
                let factor = factor.get() as usize;
                if factor > 1 {
                    // The pure-Rust `png` crate (miniz_oxide + BufReader) is competitive with
                    // libvips for this box-shrink path. The libspng progressive-row API is
                    // slower here because the C→Rust FFI boundary is crossed 8192 times (once
                    // per row), while miniz_oxide stays entirely in Rust with branch-predictor-
                    // friendly sequential memory access.
                    let result =
                        decode_png_with_box_shrink_u8(BufReader::new(File::open(path)?), factor);

                    let (width, height, bands, pixels) = result?;
                    let samples: Vec<F::Sample> =
                        bytemuck::allocation::try_cast_vec::<u8, F::Sample>(pixels).map_err(
                            |(e, _)| ViprsError::Codec(format!("png: cast error: {e:?}")),
                        )?;
                    return Image::from_buffer(width, height, bands, samples)
                        .map_err(|e| ViprsError::Codec(e.to_string()));
                }
            }
        }

        // The pure-Rust `png` crate (fdeflate decompressor) outperforms
        // libspng's bundled static zlib for full-frame decode from disk.
        // Reserve libspng for streaming/region decode where its progressive
        // API is beneficial.

        decode_png_with_png_crate_reader(png_file_reader(path)?)
    }

    /// Probe `src` for image dimensions without decoding pixel data.
    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let reader = png_reader(Cursor::new(src))?;
        let info = reader.info();
        let bands = color_type_to_bands(info.color_type);
        Ok((info.width, info.height, bands))
    }
}

impl TileImageDecoder for PngCodec {
    fn probe_with_options(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let reader = png_reader(Cursor::new(src))?;
        let info = reader.info();
        let bands = color_type_to_bands(info.color_type);
        Ok(ImageMetadataProbe::new(info.width, info.height, bands)
            .with_metadata(png_metadata(info)))
    }

    fn probe_path_with_options(
        &self,
        path: &Path,
        _opts: &LoadOptions,
    ) -> Result<ImageMetadataProbe, ViprsError>
    where
        Self: Sized,
    {
        let reader = png_reader(BufReader::new(File::open(path)?))?;
        let info = reader.info();
        let bands = color_type_to_bands(info.color_type);
        Ok(ImageMetadataProbe::new(info.width, info.height, bands)
            .with_metadata(png_metadata(info)))
    }

    fn decode_region_into<F: BandFormat>(
        &self,
        src: &[u8],
        _opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let mut reader = png_reader(Cursor::new(src))?;
        let info = reader.info();
        let bit_depth = info.bit_depth;
        let interlaced = info.interlaced;
        let bands = color_type_to_bands(info.color_type);
        if interlaced {
            let _ = (bit_depth, bands);
            // Adam7 region reads are not tile-bounded: materialize the deinterlaced
            // raster once and serve future regions from the cached full frame.
            let raster = self.interlaced_raster_from_bytes(src)?;
            decode_png_region_from_full_raster::<F>(&raster, region, output)
        } else {
            let row_len = reader
                .output_line_size(info.width)
                .ok_or_else(|| ViprsError::Codec("png: cannot determine row buffer size".into()))?;
            let mut row_scratch = self.take_row_scratch(row_len)?;
            let result = decode_png_region_rows::<F, _>(
                &mut reader,
                &mut row_scratch[..],
                bit_depth,
                bands,
                region,
                output,
            );
            let store_result = self.store_row_scratch(row_scratch);
            result.and(store_result)
        }
    }

    fn decode_region_from_path<F: BandFormat>(
        &self,
        path: &Path,
        _opts: &LoadOptions,
        region: Region,
        output: &mut [u8],
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let mut session = self.take_sequential_path_session()?;
        let should_restart = matches!(
            session.as_ref(),
            Some(existing)
                if existing.path == path
                    && region.x == 0
                    && region.y == 0
                    && region.width == existing.width
        );
        if should_restart {
            session = None;
        }

        if let Some(mut existing) = session.take() {
            if existing.path == path
                && region.x == 0
                && region.width == existing.width
                && region.y >= 0
                && region.y as u32 == existing.next_source_y
            {
                let result = decode_png_sequential_session_into::<F>(&mut existing, region, output);
                let store_result = self.store_sequential_path_session(Some(existing));
                return result.and(store_result);
            }
        }

        if region.x == 0
            && region.y == 0
            && let Some(mut session) = open_png_sequential_path_session(path)?
            && region.width == session.width
        {
            let result = decode_png_sequential_session_into::<F>(&mut session, region, output);
            let store_result = self.store_sequential_path_session(Some(session));
            return result.and(store_result);
        }

        let mut reader = png_reader(BufReader::new(File::open(path)?))?;
        let info = reader.info();
        let bit_depth = info.bit_depth;
        let interlaced = info.interlaced;
        let bands = color_type_to_bands(info.color_type);
        if interlaced {
            let _ = (bit_depth, bands);
            // Stable on-disk interlaced PNGs reuse a cached eager backing because
            // restarting Adam7 row decode for every tile is O(tiles × height).
            let raster = self.interlaced_raster_from_path(path)?;
            decode_png_region_from_full_raster::<F>(&raster, region, output)
        } else {
            let row_len = reader
                .output_line_size(info.width)
                .ok_or_else(|| ViprsError::Codec("png: cannot determine row buffer size".into()))?;
            let mut row_scratch = self.take_row_scratch(row_len)?;
            let result = decode_png_region_rows::<F, _>(
                &mut reader,
                &mut row_scratch[..],
                bit_depth,
                bands,
                region,
                output,
            );
            let store_result = self.store_row_scratch(row_scratch);
            result.and(store_result)
        }
    }
}

// ── ImageEncoder ──────────────────────────────────────────────────────────────

impl ImageEncoder for PngCodec {
    fn format_name(&self) -> &'static str {
        "png"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        PngEncoder::default().encode(image)
    }

    /// PNG is always lossless; `opts.quality`, `opts.lossless`, and most other
    /// fields are ignored. Only `opts.compression_level` is honoured if set.
    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        let mut encoder = PngEncoder::default();
        if let Some(level) = opts.compression_level {
            encoder.compression = level;
        }
        if let Some(interlace) = opts.interlace {
            encoder.interlace = interlace;
        }
        if let Some(filter) = opts.png_filter {
            encoder.filter = png_filter(filter);
        }
        encode_png_web_ready(image, &encoder, opts.strip_metadata == Some(true))
    }

    fn encode_to_writer<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
        writer: &mut dyn Write,
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let mut encoder = PngEncoder::default();
        if let Some(level) = opts.compression_level {
            encoder.compression = level;
        }
        if let Some(interlace) = opts.interlace {
            encoder.interlace = interlace;
        }
        if let Some(filter) = opts.png_filter {
            encoder.filter = png_filter(filter);
        }
        encode_png_to_writer_web_ready(image, &encoder, opts.strip_metadata == Some(true), writer)
    }
}

impl ImageEncoder for PngEncoder {
    fn format_name(&self) -> &'static str {
        "png"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        encode_png_web_ready(image, self, false)
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        let mut configured = *self;
        if let Some(level) = opts.compression_level {
            configured.compression = level;
        }
        if let Some(interlace) = opts.interlace {
            configured.interlace = interlace;
        }
        if let Some(filter) = opts.png_filter {
            configured.filter = png_filter(filter);
        }
        encode_png_web_ready(image, &configured, opts.strip_metadata == Some(true))
    }

    fn encode_to_writer<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
        writer: &mut dyn Write,
    ) -> Result<(), ViprsError>
    where
        Self: Sized,
    {
        let mut configured = *self;
        if let Some(level) = opts.compression_level {
            configured.compression = level;
        }
        if let Some(interlace) = opts.interlace {
            configured.interlace = interlace;
        }
        if let Some(filter) = opts.png_filter {
            configured.filter = png_filter(filter);
        }
        encode_png_to_writer_web_ready(
            image,
            &configured,
            opts.strip_metadata == Some(true),
            writer,
        )
    }
}
