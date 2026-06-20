use super::pyramid::{
    first_ifd_offset, ifd_entry_value_pos, next_ifd_pointer_pos, patch_subifd_offsets,
    pyramid_levels, tiff_read_u32, tiff_write_u32, write_subifd_tag,
};
#[allow(clippy::wildcard_imports)]
// REASON: TIFF encode helpers share many sibling codec symbols.
use super::*;

pub(super) fn deflate_level(level: Option<u8>) -> DeflateLevel {
    match level.unwrap_or(6) {
        0..=3 => DeflateLevel::Fast,
        4..=6 => DeflateLevel::Balanced,
        _ => DeflateLevel::Best,
    }
}

pub(super) fn effective_compression(
    default: TiffCompression,
    opts: &SaveOptions,
) -> TiffCompression {
    opts.tiff_compression.unwrap_or(default)
}

pub(super) fn effective_predictor(
    default: TiffPredictor,
    compression: TiffCompression,
    opts: &SaveOptions,
) -> TiffPredictor {
    match opts.tiff_predictor.unwrap_or(default) {
        TiffPredictor::Horizontal
            if matches!(compression, TiffCompression::Lzw | TiffCompression::Deflate) =>
        {
            TiffPredictor::Horizontal
        }
        _ => TiffPredictor::None,
    }
}

pub(super) fn effective_predictor_for_format(
    format: BandFormatId,
    default: TiffPredictor,
    compression: TiffCompression,
    opts: &SaveOptions,
) -> TiffPredictor {
    let predictor = effective_predictor(default, compression, opts);
    if matches!(format, BandFormatId::F32) {
        // The `tiff` crate rejects predictor tag 2 for IEEE float samples and does not expose
        // floating-point predictor 3, so float TIFF output must disable prediction.
        TiffPredictor::None
    } else {
        predictor
    }
}

pub(super) const fn compression_tag(compression: TiffCompression) -> u16 {
    match compression {
        TiffCompression::None => CompressionMethod::None.to_u16(),
        TiffCompression::Lzw => CompressionMethod::LZW.to_u16(),
        TiffCompression::Deflate => CompressionMethod::Deflate.to_u16(),
        TiffCompression::PackBits => CompressionMethod::PackBits.to_u16(),
        TiffCompression::Jpeg => CompressionMethod::ModernJPEG.to_u16(),
    }
}

pub(super) fn tile_dimensions(opts: &SaveOptions) -> Option<(u32, u32)> {
    match (opts.tile_width, opts.tile_height) {
        (None, None) => None,
        (Some(width), Some(height)) => Some((width.max(1), height.max(1))),
        (Some(width), None) => Some((width.max(1), DEFAULT_TIFF_TILE_SIZE)),
        (None, Some(height)) => Some((DEFAULT_TIFF_TILE_SIZE, height.max(1))),
    }
}

pub(super) fn compression_quality(opts: &SaveOptions) -> u8 {
    opts.quality.unwrap_or(100)
}

pub(super) fn pages_for_encode<F: BandFormat>(image: &Image<F>) -> Result<Vec<Image<F>>, ViprsError>
where
    F::Sample: Clone,
{
    if let Some(frames) = image.frames() {
        return Ok(frames.to_vec());
    }

    let page_height = image
        .metadata()
        .page_height
        .unwrap_or_else(|| image.height());
    if page_height == 0 || page_height >= image.height() {
        return Ok(vec![image.clone()]);
    }
    if !image.height().is_multiple_of(page_height) {
        return Err(ViprsError::Codec(format!(
            "tiff: page_height {page_height} must divide image height {}",
            image.height()
        )));
    }

    let rows = image.height() / page_height;
    let row_stride = (image.width() * image.bands()) as usize;
    let page_stride = row_stride * page_height as usize;
    let mut pages = Vec::with_capacity(rows as usize);
    for page_index in 0..rows as usize {
        let start = page_index * page_stride;
        let end = start + page_stride;
        let mut page = Image::from_buffer(
            image.width(),
            page_height,
            image.bands(),
            image.pixels()[start..end].to_vec(),
        )
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
        page = page.with_metadata(ImageMetadata {
            page_height: Some(page_height),
            n_pages: Some(rows),
            ..image.metadata().clone()
        });
        pages.push(page);
    }
    Ok(pages)
}
pub(super) fn recast_pages_u8<F: BandFormat>(
    pages: &[Image<F>],
) -> Result<Vec<Image<U8>>, ViprsError> {
    pages
        .iter()
        .map(|page| {
            Image::<U8>::from_buffer(
                page.width(),
                page.height(),
                page.bands(),
                bytemuck::cast_slice(page.pixels()).to_vec(),
            )
            .map(|image| image.with_metadata(page.metadata().clone()))
            .map_err(|e| ViprsError::Codec(e.to_string()))
        })
        .collect()
}

pub(super) fn recast_pages_u16<F: BandFormat>(
    pages: &[Image<F>],
) -> Result<Vec<Image<U16>>, ViprsError> {
    pages
        .iter()
        .map(|page| {
            Image::<U16>::from_buffer(
                page.width(),
                page.height(),
                page.bands(),
                bytemuck::cast_slice(page.pixels()).to_vec(),
            )
            .map(|image| image.with_metadata(page.metadata().clone()))
            .map_err(|e| ViprsError::Codec(e.to_string()))
        })
        .collect()
}

pub(super) fn recast_pages_f32<F: BandFormat>(
    pages: &[Image<F>],
) -> Result<Vec<Image<F32>>, ViprsError> {
    pages
        .iter()
        .map(|page| {
            Image::<F32>::from_buffer(
                page.width(),
                page.height(),
                page.bands(),
                bytemuck::cast_slice(page.pixels()).to_vec(),
            )
            .map(|image| image.with_metadata(page.metadata().clone()))
            .map_err(|e| ViprsError::Codec(e.to_string()))
        })
        .collect()
}

pub(super) fn write_rational_resolution<W, K>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    metadata: &ImageMetadata,
) -> Result<(), ViprsError>
where
    W: Write + Seek,
    K: TiffKind,
{
    let xres = metadata.xres.unwrap_or(1.0);
    let yres = metadata.yres.unwrap_or(1.0);
    if metadata.xres.is_some() || metadata.yres.is_some() {
        directory
            .write_tag(Tag::ResolutionUnit, ResolutionUnit::Centimeter.to_u16())
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        directory
            .write_tag(
                Tag::XResolution,
                Rational {
                    n: (xres * 10_000.0).round().max(1.0) as u32,
                    d: 10_000,
                },
            )
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        directory
            .write_tag(
                Tag::YResolution,
                Rational {
                    n: (yres * 10_000.0).round().max(1.0) as u32,
                    d: 10_000,
                },
            )
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    } else {
        directory
            .write_tag(Tag::ResolutionUnit, ResolutionUnit::None.to_u16())
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        directory
            .write_tag(Tag::XResolution, Rational { n: 1, d: 1 })
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        directory
            .write_tag(Tag::YResolution, Rational { n: 1, d: 1 })
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    Ok(())
}

pub(super) fn write_icc_profile<W, K>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    metadata: &ImageMetadata,
) -> Result<(), ViprsError>
where
    W: Write + Seek,
    K: TiffKind,
{
    if let Some(icc_profile) = metadata.icc_profile.as_deref() {
        directory
            .write_tag(TIFF_ICC_PROFILE_TAG, icc_profile)
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    Ok(())
}

pub(super) fn write_common_tags<W, K, C: ColorType>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    width: u32,
    height: u32,
    compression: TiffCompression,
    predictor: TiffPredictor,
    tile: Option<(u32, u32)>,
    page_number: Option<(u16, u16)>,
    reduced_resolution: bool,
) -> Result<(), ViprsError>
where
    W: Write + Seek,
    K: TiffKind,
{
    let sample_format: Vec<u16> = C::SAMPLE_FORMAT.iter().map(SampleFormat::to_u16).collect();
    directory
        .write_tag(Tag::ImageWidth, width)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::ImageLength, height)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::Compression, compression_tag(compression))
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::BitsPerSample, C::BITS_PER_SAMPLE)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::SampleFormat, &sample_format[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::PhotometricInterpretation, C::TIFF_VALUE.to_u16())
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(
            Tag::SamplesPerPixel,
            u16::try_from(C::BITS_PER_SAMPLE.len())
                .map_err(|e| ViprsError::Codec(e.to_string()))?,
        )
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::PlanarConfiguration, 1u16)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::Orientation, 1u16)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    if reduced_resolution {
        directory
            .write_tag(Tag::NewSubfileType, TIFFTAG_NEWSUBFILETYPE_REDUCED_IMAGE)
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }
    if let Some((page, total_pages)) = page_number {
        let page_numbers: [u16; 2] = (page, total_pages).into();
        directory
            .write_tag(TIFF_PAGE_NUMBER_TAG, &page_numbers[..])
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }
    if matches!(predictor, TiffPredictor::Horizontal)
        && matches!(compression, TiffCompression::Lzw | TiffCompression::Deflate)
    {
        directory
            .write_tag(Tag::Predictor, 2u16)
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }
    if let Some((tile_width, tile_height)) = tile {
        directory
            .write_tag(Tag::TileWidth, tile_width)
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        directory
            .write_tag(Tag::TileLength, tile_height)
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    Ok(())
}

pub(super) fn apply_predictor<S: PredictorSample>(
    samples: &mut [S],
    width: usize,
    height: usize,
    samples_per_pixel: usize,
    predictor: TiffPredictor,
) {
    if !matches!(predictor, TiffPredictor::Horizontal) {
        return;
    }

    let row_len = width * samples_per_pixel;
    for row in 0..height {
        let start = row * row_len;
        let end = start + row_len;
        S::apply_horizontal_predictor_row(&mut samples[start..end], samples_per_pixel);
    }
}

pub(super) fn compress_bytes(
    bytes: &[u8],
    compression: TiffCompression,
    compression_level: Option<u8>,
) -> Result<Vec<u8>, ViprsError> {
    let mut output = Vec::new();
    match compression {
        TiffCompression::None => output.extend_from_slice(bytes),
        TiffCompression::Lzw => {
            let mut encoder = WeezlEncoder::with_tiff_size_switch(BitOrder::Msb, 8);
            output.reserve(bytes.len().saturating_div(2).max(4096));
            encoder
                .into_vec(&mut output)
                .encode_all(bytes)
                .status
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
        }
        TiffCompression::Deflate => {
            Deflate::with_level(deflate_level(compression_level))
                .write_to(&mut output, bytes)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
        }
        TiffCompression::PackBits => {
            Packbits
                .write_to(&mut output, bytes)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
        }
        TiffCompression::Jpeg => {
            return Err(ViprsError::Codec(
                "tiff: jpeg compression is handled by encode_jpeg_chunk".into(),
            ));
        }
    }
    Ok(output)
}

pub(super) fn encode_jpeg_chunk(
    bytes: &[u8],
    width: u32,
    height: u32,
    bands: u32,
    quality: u8,
) -> Result<Vec<u8>, ViprsError> {
    let color_type = match bands {
        1 => JpegColorType::Luma,
        3 => JpegColorType::Rgb,
        _ => {
            return Err(ViprsError::Codec(format!(
                "tiff: JPEG compression supports only 1-band or 3-band U8 images, got {bands} bands"
            )));
        }
    };

    let mut output = Vec::new();
    let mut encoder = JpegEncoder::new(&mut output, quality);
    if bands == 3 {
        encoder.set_sampling_factor(SamplingFactor::F_1_1);
    }
    encoder
        .encode(
            bytes,
            u16::try_from(width).map_err(|e| ViprsError::Codec(e.to_string()))?,
            u16::try_from(height).map_err(|e| ViprsError::Codec(e.to_string()))?,
            color_type,
        )
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(output)
}

pub(super) fn write_strips<W, K, C, F>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    image: &Image<F>,
    compression: TiffCompression,
    predictor: TiffPredictor,
    compression_level: Option<u8>,
    quality: u8,
) -> Result<(), ViprsError>
where
    W: Write + std::io::Seek,
    K: TiffKind,
    C: ColorType,
    F: BandFormat<Sample = C::Inner>,
    C::Inner: PredictorSample,
    [F::Sample]: TiffValue,
{
    let rows_per_strip = match compression {
        TiffCompression::PackBits => 1u32,
        TiffCompression::Jpeg => image.height(),
        TiffCompression::None if matches!(predictor, TiffPredictor::None) => image.height(),
        _ => image.height().min(DEFAULT_TIFF_ROWS_PER_STRIP),
    };
    let bands = image.bands() as usize;
    let row_stride = image.width() as usize * bands;
    let strip_count = image.height().div_ceil(rows_per_strip);
    let mut offsets = Vec::with_capacity(strip_count as usize);
    let mut byte_counts = Vec::with_capacity(strip_count as usize);

    directory
        .write_tag(Tag::RowsPerStrip, rows_per_strip)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    for strip_index in 0..strip_count {
        let start_row = strip_index * rows_per_strip;
        let strip_height = (image.height() - start_row).min(rows_per_strip);
        let start = start_row as usize * row_stride;
        let end = start + strip_height as usize * row_stride;
        let slice = &image.pixels()[start..end];

        let encoded = if matches!(compression, TiffCompression::Jpeg) {
            Cow::Owned(encode_jpeg_chunk(
                bytemuck::cast_slice(slice),
                image.width(),
                strip_height,
                image.bands(),
                quality,
            )?)
        } else if matches!(compression, TiffCompression::None)
            && matches!(predictor, TiffPredictor::None)
        {
            Cow::Borrowed(bytemuck::cast_slice(slice))
        } else {
            let mut predicted = slice.to_vec();
            apply_predictor(
                &mut predicted,
                image.width() as usize,
                strip_height as usize,
                bands,
                predictor,
            );
            Cow::Owned(compress_bytes(
                bytemuck::cast_slice(&predicted),
                compression,
                compression_level,
            )?)
        };

        let offset = directory
            .write_data(encoded.as_ref())
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        offsets.push(u32::try_from(offset).map_err(|e| ViprsError::Codec(e.to_string()))?);
        byte_counts
            .push(u32::try_from(encoded.len()).map_err(|e| ViprsError::Codec(e.to_string()))?);
    }

    directory
        .write_tag(Tag::StripOffsets, &offsets[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::StripByteCounts, &byte_counts[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(())
}

pub(super) fn write_tiles<W, K, C, F>(
    directory: &mut DirectoryEncoder<'_, W, K>,
    image: &Image<F>,
    compression: TiffCompression,
    predictor: TiffPredictor,
    compression_level: Option<u8>,
    quality: u8,
    tile_width: u32,
    tile_height: u32,
) -> Result<(), ViprsError>
where
    W: Write + std::io::Seek,
    K: TiffKind,
    C: ColorType,
    F: BandFormat<Sample = C::Inner>,
    C::Inner: PredictorSample + Default,
    [F::Sample]: TiffValue,
{
    let bands = image.bands() as usize;
    let src_row_stride = image.width() as usize * bands;
    let tile_row_stride = tile_width as usize * bands;
    let tiles_x = image.width().div_ceil(tile_width);
    let tiles_y = image.height().div_ceil(tile_height);
    let mut offsets = Vec::with_capacity((tiles_x * tiles_y) as usize);
    let mut byte_counts = Vec::with_capacity((tiles_x * tiles_y) as usize);

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            let mut tile = vec![C::Inner::default(); tile_row_stride * tile_height as usize];
            let src_x = tile_x * tile_width;
            let src_y = tile_y * tile_height;
            let copy_width = (image.width() - src_x).min(tile_width) as usize;
            let copy_height = (image.height() - src_y).min(tile_height) as usize;

            for row in 0..copy_height {
                let src_start = (src_y as usize + row) * src_row_stride + src_x as usize * bands;
                let src_end = src_start + copy_width * bands;
                let dst_start = row * tile_row_stride;
                let dst_end = dst_start + copy_width * bands;
                tile[dst_start..dst_end].copy_from_slice(&image.pixels()[src_start..src_end]);
            }

            let encoded = if matches!(compression, TiffCompression::Jpeg) {
                encode_jpeg_chunk(
                    bytemuck::cast_slice(&tile),
                    tile_width,
                    tile_height,
                    image.bands(),
                    quality,
                )?
            } else {
                apply_predictor(
                    &mut tile,
                    tile_width as usize,
                    tile_height as usize,
                    bands,
                    predictor,
                );
                compress_bytes(bytemuck::cast_slice(&tile), compression, compression_level)?
            };

            let offset = directory
                .write_data(&encoded[..])
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            offsets.push(u32::try_from(offset).map_err(|e| ViprsError::Codec(e.to_string()))?);
            byte_counts
                .push(u32::try_from(encoded.len()).map_err(|e| ViprsError::Codec(e.to_string()))?);
        }
    }

    directory
        .write_tag(Tag::TileOffsets, &offsets[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    directory
        .write_tag(Tag::TileByteCounts, &byte_counts[..])
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(())
}

#[allow(deprecated)]
// REASON: tiff crate deprecation, upgrade tracked in backlog.
pub(super) fn write_page<C, F>(
    encoder: &mut RawTiffEncoder<SharedWriteBuffer>,
    image: &Image<F>,
    compression: TiffCompression,
    predictor: TiffPredictor,
    compression_level: Option<u8>,
    quality: u8,
    tile: Option<(u32, u32)>,
    page_number: Option<(u16, u16)>,
    reduced_resolution: bool,
    subifd_count: usize,
) -> Result<Option<SubIfdPatchTarget>, ViprsError>
where
    C: ColorType,
    F: BandFormat<Sample = C::Inner>,
    C::Inner: PredictorSample + Default,
    [F::Sample]: TiffValue,
{
    let mut directory = encoder
        .image_directory()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    write_common_tags::<_, _, C>(
        &mut directory,
        image.width(),
        image.height(),
        compression,
        predictor,
        tile,
        page_number,
        reduced_resolution,
    )?;
    write_rational_resolution(&mut directory, image.metadata())?;
    write_icc_profile(&mut directory, image.metadata())?;
    let subifd_table_offset = write_subifd_tag(&mut directory, subifd_count)?;

    match tile {
        Some((tile_width, tile_height)) => write_tiles::<_, _, C, F>(
            &mut directory,
            image,
            compression,
            predictor,
            compression_level,
            quality,
            tile_width,
            tile_height,
        )?,
        None => write_strips::<_, _, C, F>(
            &mut directory,
            image,
            compression,
            predictor,
            compression_level,
            quality,
        )?,
    }

    directory
        .finish()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    Ok(subifd_table_offset)
}

pub(super) fn encode_tiff_document<C, F>(
    pages: &[Image<F>],
    opts: &SaveOptions,
    compression: TiffCompression,
    predictor: TiffPredictor,
    tile: Option<(u32, u32)>,
) -> Result<Vec<u8>, ViprsError>
where
    C: ColorType,
    F: BandFormat<Sample = C::Inner>,
    C::Inner: PredictorSample + Default + Clone + PyramidSample,
    [F::Sample]: TiffValue,
{
    let output = SharedWriteBuffer::default();
    let mut encoder =
        RawTiffEncoder::new(output.clone()).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let quality = compression_quality(opts);
    let pyramid = opts.pyramid == Some(true) && pages.len() == 1;
    let tile = if pyramid {
        Some(tile.unwrap_or((DEFAULT_TIFF_TILE_SIZE, DEFAULT_TIFF_TILE_SIZE)))
    } else {
        tile
    };
    let total_pages = u16::try_from(pages.len()).map_err(|e| ViprsError::Codec(e.to_string()))?;

    for (index, page) in pages.iter().enumerate() {
        if pyramid {
            let levels = pyramid_levels(page, tile)?;
            let subifd_table_offset = write_page::<C, F>(
                &mut encoder,
                page,
                compression,
                predictor,
                opts.compression_level,
                quality,
                tile,
                Some((
                    u16::try_from(index).map_err(|e| ViprsError::Codec(e.to_string()))?,
                    total_pages,
                )),
                false,
                levels.len(),
            )?;
            let root_ifd_offset = output.with_bytes(first_ifd_offset)?;
            let root_next_ifd_pos =
                output.with_bytes(|bytes| next_ifd_pointer_pos(bytes, root_ifd_offset))?;
            let inline_subifd_value_pos = match subifd_table_offset {
                Some(SubIfdPatchTarget::Inline) => Some(output.with_bytes(|bytes| {
                    ifd_entry_value_pos(bytes, root_ifd_offset, TIFF_SUB_IFD_TAG)
                })?),
                _ => None,
            };
            let mut next_link_pos = root_next_ifd_pos;
            let mut subifd_offsets = Vec::with_capacity(levels.len());

            for level in levels {
                write_page::<C, F>(
                    &mut encoder,
                    &level,
                    compression,
                    predictor,
                    opts.compression_level,
                    quality,
                    tile,
                    None,
                    true,
                    0,
                )?;
                let subifd_offset =
                    output.with_bytes(|bytes| tiff_read_u32(bytes, next_link_pos))?;
                subifd_offsets.push(subifd_offset);
                next_link_pos =
                    output.with_bytes(|bytes| next_ifd_pointer_pos(bytes, subifd_offset))?;
            }

            match subifd_table_offset {
                Some(SubIfdPatchTarget::Table(table_offset)) => {
                    output.with_bytes_mut(|bytes| {
                        patch_subifd_offsets(bytes, table_offset, &subifd_offsets)
                    })?;
                }
                Some(SubIfdPatchTarget::Inline) => {
                    if let (Some(value_pos), Some(&subifd_offset)) =
                        (inline_subifd_value_pos, subifd_offsets.first())
                    {
                        output.with_bytes_mut(|bytes| {
                            tiff_write_u32(bytes, value_pos, subifd_offset)
                        })?;
                    }
                }
                None => {}
            }
            if !subifd_offsets.is_empty() {
                output.with_bytes_mut(|bytes| tiff_write_u32(bytes, root_next_ifd_pos, 0))?;
            }
        } else {
            write_page::<C, F>(
                &mut encoder,
                page,
                compression,
                predictor,
                opts.compression_level,
                quality,
                tile,
                Some((
                    u16::try_from(index).map_err(|e| ViprsError::Codec(e.to_string()))?,
                    total_pages,
                )),
                false,
                0,
            )?;
        }
    }

    drop(encoder);
    Ok(output.into_inner())
}
