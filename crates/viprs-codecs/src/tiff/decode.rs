use super::pyramid::{
    choose_pyramid_level, current_page_subifd_offsets, patch_first_ifd_offset, probe_ifd_dimensions,
};
#[allow(clippy::wildcard_imports)]
// REASON: TIFF decode helpers share many sibling codec types/constants.
use super::*;

pub(super) fn is_tiff_header(header: &[u8]) -> bool {
    header.len() >= 4 && (header[..4] == TIFF_LE_MAGIC || header[..4] == TIFF_BE_MAGIC)
}

pub(super) fn color_type_to_bands(color_type: TiffColorType) -> Result<u32, ViprsError> {
    match color_type {
        TiffColorType::Gray(_) => Ok(1),
        TiffColorType::RGB(_)
        | TiffColorType::YCbCr(_)
        | TiffColorType::Palette(_)
        | TiffColorType::Lab(_) => Ok(3),
        TiffColorType::RGBA(_) | TiffColorType::CMYK(_) => Ok(4),
        TiffColorType::GrayA(_) => Err(ViprsError::Codec(
            "tiff: grayscale+alpha is not supported; expected 1, 3, or 4 bands".into(),
        )),
        TiffColorType::CMYKA(_) => Err(ViprsError::Codec(
            "tiff: CMYK+alpha is not supported; expected 1, 3, or 4 bands".into(),
        )),
        TiffColorType::Multiband { num_samples, .. } => match num_samples {
            1 => Ok(1),
            3 => Ok(3),
            4 => Ok(4),
            other => Err(ViprsError::Codec(format!(
                "tiff: unsupported multiband sample count {other}; expected 1, 3, or 4 bands"
            ))),
        },
        _ => Err(ViprsError::Codec(
            "tiff: unsupported color type; expected 1, 3, or 4 bands".into(),
        )),
    }
}

pub(super) fn bit_depth(color_type: TiffColorType) -> u8 {
    color_type.bit_depth()
}

pub(super) fn interpretation_from_tags(
    photometric: Option<PhotometricInterpretation>,
    color_type: TiffColorType,
    result_is_float: bool,
) -> Interpretation {
    let bands = color_type_to_bands(color_type).unwrap_or(0);
    let bits = bit_depth(color_type);

    match photometric {
        Some(PhotometricInterpretation::WhiteIsZero | PhotometricInterpretation::BlackIsZero) => {
            if result_is_float {
                Interpretation::Multiband
            } else if bits > 8 {
                Interpretation::Grey16
            } else {
                Interpretation::BW
            }
        }
        Some(PhotometricInterpretation::RGB | PhotometricInterpretation::YCbCr) if bands >= 3 => {
            if result_is_float {
                Interpretation::Scrgb
            } else if bits > 8 {
                Interpretation::Rgb16
            } else {
                Interpretation::Srgb
            }
        }
        Some(PhotometricInterpretation::CIELab) if bands >= 3 => Interpretation::Lab,
        Some(PhotometricInterpretation::CMYK) if bands >= 4 => Interpretation::Cmyk,
        _ => Interpretation::Multiband,
    }
}

#[allow(deprecated)]
// REASON: tiff crate deprecation, upgrade tracked in backlog.
pub(super) fn resolution_in_pixels_per_mm(
    decoder: &mut Decoder<Cursor<&[u8]>>,
    tag: Tag,
    unit: ResolutionUnit,
) -> Result<Option<f64>, ViprsError> {
    let value = decoder
        .find_tag(tag)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .map(|value| match value {
            TiffValueRef::Rational(numerator, denominator) => {
                if denominator == 0 {
                    0.0
                } else {
                    f64::from(numerator) / f64::from(denominator)
                }
            }
            TiffValueRef::RationalBig(numerator, denominator) => {
                if denominator == 0 {
                    0.0
                } else {
                    numerator as f64 / denominator as f64
                }
            }
            TiffValueRef::Float(value) => f64::from(value),
            TiffValueRef::Double(value) => value,
            TiffValueRef::Unsigned(value) => f64::from(value),
            TiffValueRef::UnsignedBig(value) => value as f64,
            TiffValueRef::Short(value) => f64::from(value),
            _ => 0.0,
        });

    Ok(value.map(|resolution| match unit {
        ResolutionUnit::Inch => resolution / 25.4,
        ResolutionUnit::Centimeter => resolution / 10.0,
        _ => resolution,
    }))
}

pub(super) fn extract_metadata(
    decoder: &mut Decoder<Cursor<&[u8]>>,
    color_type: TiffColorType,
    result_is_float: bool,
    height: u32,
    total_pages: u32,
    loaded_pages: u32,
) -> Result<ImageMetadata, ViprsError> {
    let photometric = decoder
        .find_tag_unsigned::<u16>(Tag::PhotometricInterpretation)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .and_then(PhotometricInterpretation::from_u16);
    let orientation = decoder
        .find_tag_unsigned::<u16>(Tag::Orientation)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .and_then(|value| u8::try_from(value).ok());
    let resolution_unit = decoder
        .find_tag_unsigned::<u16>(Tag::ResolutionUnit)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .and_then(ResolutionUnit::from_u16)
        .unwrap_or(ResolutionUnit::None);

    let xres = resolution_in_pixels_per_mm(decoder, Tag::XResolution, resolution_unit)?;
    let yres = resolution_in_pixels_per_mm(decoder, Tag::YResolution, resolution_unit)?;
    let icc_profile = decoder
        .find_tag_unsigned_vec::<u8>(TIFF_ICC_PROFILE_TAG)
        .map_err(|e| ViprsError::Codec(e.to_string()))?;

    Ok(ImageMetadata {
        interpretation: Some(interpretation_from_tags(
            photometric,
            color_type,
            result_is_float,
        )),
        orientation,
        icc_profile,
        xres,
        yres,
        n_pages: Some(total_pages),
        page_height: (total_pages > 1 || loaded_pages > 1).then_some(height),
        ..ImageMetadata::default()
    })
}

pub(super) const fn decoding_result_name(result: &DecodingResult) -> &'static str {
    match result {
        DecodingResult::U8(_) => "U8",
        DecodingResult::U16(_) => "U16",
        DecodingResult::U32(_) => "U32",
        DecodingResult::U64(_) => "U64",
        DecodingResult::F16(_) => "F16",
        DecodingResult::F32(_) => "F32",
        DecodingResult::F64(_) => "F64",
        DecodingResult::I8(_) => "I8",
        DecodingResult::I16(_) => "I16",
        DecodingResult::I32(_) => "I32",
        DecodingResult::I64(_) => "I64",
    }
}

pub(super) fn count_pages(src: &[u8]) -> Result<u32, ViprsError> {
    let mut decoder =
        Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut pages = 1u32;
    while decoder.more_images() {
        decoder
            .next_image()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        pages += 1;
    }
    Ok(pages)
}

pub(super) fn decode_current_page<F: BandFormat>(
    src: &[u8],
    page_index: u32,
    decoder: &mut Decoder<Cursor<&[u8]>>,
    total_pages: u32,
    loaded_pages: u32,
) -> Result<InMemoryImage<F>, ViprsError> {
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let color_type = decoder
        .colortype()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let bands = color_type_to_bands(color_type)?;
    let decoded = match try_decode_page_strips_in_parallel::<F>(
        src, page_index, decoder, width, height, bands, color_type,
    )? {
        Some(decoded) => decoded,
        None => decode_current_page_serial::<F>(decoder)?,
    };
    let metadata = extract_metadata(
        decoder,
        color_type,
        decoded.result_is_float,
        height,
        total_pages,
        loaded_pages,
    )?;

    InMemoryImage::from_buffer(width, height, bands, decoded.samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

pub struct DecodedPage<F: BandFormat> {
    samples: Vec<F::Sample>,
    result_is_float: bool,
}

pub(super) fn decode_current_page_serial<F: BandFormat>(
    decoder: &mut Decoder<Cursor<&[u8]>>,
) -> Result<DecodedPage<F>, ViprsError> {
    let decoded = decoder
        .read_image()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let result_is_float = matches!(&decoded, DecodingResult::F32(_) | DecodingResult::F64(_));
    let samples: Vec<F::Sample> = match (F::ID, decoded) {
        (BandFormatId::U8, DecodingResult::U8(data)) => bytemuck::allocation::try_cast_vec(data)
            .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        (BandFormatId::U16, DecodingResult::U16(data)) => bytemuck::allocation::try_cast_vec(data)
            .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        (BandFormatId::F32, DecodingResult::F32(data)) => bytemuck::allocation::try_cast_vec(data)
            .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        (requested, result) => {
            return Err(ViprsError::Codec(format!(
                "tiff: requested {:?}, but file decoded as {}",
                requested,
                decoding_result_name(&result)
            )));
        }
    };

    Ok(DecodedPage {
        samples,
        result_is_float,
    })
}

pub(super) fn decoder_for_page(
    src: &[u8],
    page_index: u32,
) -> Result<Decoder<Cursor<&[u8]>>, ViprsError> {
    let mut decoder =
        Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    for _ in 0..page_index {
        decoder
            .next_image()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }
    Ok(decoder)
}

#[cfg(feature = "rayon")]
pub(super) fn try_decode_page_strips_in_parallel<F: BandFormat>(
    src: &[u8],
    page_index: u32,
    decoder: &mut Decoder<Cursor<&[u8]>>,
    width: u32,
    height: u32,
    bands: u32,
    color_type: TiffColorType,
) -> Result<Option<DecodedPage<F>>, ViprsError> {
    if decoder.get_chunk_type() != ChunkType::Strip {
        return Ok(None);
    }

    let planar_configuration = decoder
        .find_tag_unsigned::<u16>(Tag::PlanarConfiguration)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .unwrap_or(1);
    if planar_configuration != 1 {
        return Ok(None);
    }

    let rows_per_strip = decoder
        .find_tag_unsigned::<u32>(Tag::RowsPerStrip)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .unwrap_or(height);
    let strip_count = height.div_ceil(rows_per_strip);
    if strip_count <= 1 || decoder.chunk_dimensions().0 != width {
        return Ok(None);
    }

    let sample_formats = decoder
        .find_tag_unsigned_vec::<u16>(Tag::SampleFormat)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .unwrap_or_default();
    let sample_format = sample_formats
        .first()
        .copied()
        .unwrap_or_else(|| SampleFormat::Uint.to_u16());
    if sample_formats.iter().any(|&value| value != sample_format) {
        return Ok(None);
    }

    let decoded_format = match (sample_format, color_type.bit_depth()) {
        (value, 8) if value == SampleFormat::Uint.to_u16() => BandFormatId::U8,
        (value, 16) if value == SampleFormat::Uint.to_u16() => BandFormatId::U16,
        (value, 32) if value == SampleFormat::IEEEFP.to_u16() => BandFormatId::F32,
        _ => return Ok(None),
    };
    if decoded_format != F::ID {
        return Ok(None);
    }

    let row_stride = usize::try_from(width)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .checked_mul(usize::try_from(bands).map_err(|e| ViprsError::Codec(e.to_string()))?)
        .ok_or_else(|| ViprsError::Codec("tiff: strip row stride overflow".into()))?;
    let strip_sample_len = row_stride
        .checked_mul(usize::try_from(rows_per_strip).map_err(|e| ViprsError::Codec(e.to_string()))?)
        .ok_or_else(|| ViprsError::Codec("tiff: strip buffer overflow".into()))?;
    let total_samples = row_stride
        .checked_mul(usize::try_from(height).map_err(|e| ViprsError::Codec(e.to_string()))?)
        .ok_or_else(|| ViprsError::Codec("tiff: output buffer overflow".into()))?;
    let samples: Vec<F::Sample> = match F::ID {
        BandFormatId::U8 => bytemuck::allocation::try_cast_vec::<u8, F::Sample>(
            decode_page_strips_in_parallel_samples::<u8>(
                src,
                page_index,
                strip_sample_len,
                total_samples,
            )?,
        )
        .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        BandFormatId::U16 => bytemuck::allocation::try_cast_vec::<u16, F::Sample>(
            decode_page_strips_in_parallel_samples::<u16>(
                src,
                page_index,
                strip_sample_len,
                total_samples,
            )?,
        )
        .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        BandFormatId::F32 => bytemuck::allocation::try_cast_vec::<f32, F::Sample>(
            decode_page_strips_in_parallel_samples::<f32>(
                src,
                page_index,
                strip_sample_len,
                total_samples,
            )?,
        )
        .map_err(|(e, _)| ViprsError::Codec(format!("tiff: cast error: {e:?}")))?,
        _ => return Ok(None),
    };

    Ok(Some(DecodedPage {
        samples,
        result_is_float: decoded_format == BandFormatId::F32,
    }))
}

#[cfg(feature = "rayon")]
pub(super) fn decode_page_strips_in_parallel_samples<
    T: bytemuck::Pod + bytemuck::Zeroable + Send,
>(
    src: &[u8],
    page_index: u32,
    strip_sample_len: usize,
    total_samples: usize,
) -> Result<Vec<T>, ViprsError> {
    let mut samples = vec![T::zeroed(); total_samples];
    samples
        .par_chunks_mut(strip_sample_len)
        .enumerate()
        .try_for_each(|(strip_index, strip_samples)| -> Result<(), ViprsError> {
            let mut strip_decoder = decoder_for_page(src, page_index)?;
            strip_decoder
                .read_chunk_bytes(
                    u32::try_from(strip_index).map_err(|e| ViprsError::Codec(e.to_string()))?,
                    bytemuck::cast_slice_mut(strip_samples),
                )
                .map_err(|e| ViprsError::Codec(e.to_string()))
        })?;
    Ok(samples)
}

#[cfg(not(feature = "rayon"))]
pub(super) fn try_decode_page_strips_in_parallel<F: BandFormat>(
    _src: &[u8],
    _page_index: u32,
    _decoder: &mut Decoder<Cursor<&[u8]>>,
    _width: u32,
    _height: u32,
    _bands: u32,
    _color_type: TiffColorType,
) -> Result<Option<DecodedPage<F>>, ViprsError>
where
    F::Sample: Clone,
{
    Ok(None)
}

pub(super) fn normalize_page_selection(
    total_pages: u32,
    opts: &LoadOptions,
) -> Result<(u32, u32), ViprsError> {
    let page = opts.page.unwrap_or(0);
    if page >= total_pages {
        return Err(ViprsError::Codec(format!(
            "tiff: requested page {page}, but file only has {total_pages} page(s)"
        )));
    }

    let remaining = total_pages - page;
    let requested = match opts.n {
        None => 1,
        Some(-1) => remaining,
        Some(value) if value > 0 => u32::try_from(value)
            .map_err(|_| ViprsError::Codec(format!("tiff: invalid page count {value}")))?,
        Some(value) => {
            return Err(ViprsError::Codec(format!(
                "tiff: n must be positive or -1, got {value}"
            )));
        }
    };

    Ok((page, requested.min(remaining)))
}

pub(super) fn decode_page_with_pyramid_selection<F: BandFormat>(
    src: &[u8],
    page_index: u32,
    total_pages: u32,
    loaded_pages: u32,
    opts: &LoadOptions,
) -> Result<InMemoryImage<F>, ViprsError>
where
    F::Sample: Clone,
{
    let mut decoder =
        Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    for _ in 0..page_index {
        decoder
            .next_image()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    let root_dimensions = decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let subifd_offsets = current_page_subifd_offsets(&mut decoder)?;
    let mut subifd_dimensions = Vec::with_capacity(subifd_offsets.len());
    for offset in subifd_offsets {
        subifd_dimensions.push((offset, probe_ifd_dimensions(src, offset)?));
    }

    if let Some(selected_offset) = choose_pyramid_level(root_dimensions, &subifd_dimensions, opts) {
        let patched = patch_first_ifd_offset(src, selected_offset)?;
        let mut selected_decoder = Decoder::new(Cursor::new(patched.as_slice()))
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
        return decode_current_page::<F>(
            patched.as_slice(),
            0,
            &mut selected_decoder,
            total_pages,
            loaded_pages,
        );
    }

    decode_current_page::<F>(src, page_index, &mut decoder, total_pages, loaded_pages)
}

pub(super) fn decode_tiff<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<InMemoryImage<F>, ViprsError>
where
    F::Sample: Clone,
{
    let total_pages = count_pages(src)?;
    let (start_page, page_count) = normalize_page_selection(total_pages, opts)?;

    let mut pages = Vec::with_capacity(page_count as usize);
    for index in 0..page_count {
        pages.push(decode_page_with_pyramid_selection::<F>(
            src,
            start_page + index,
            total_pages,
            page_count,
            opts,
        )?);
    }

    if page_count == 1 {
        return pages
            .into_iter()
            .next()
            .ok_or_else(|| ViprsError::Codec("tiff: missing decoded page".into()));
    }

    let first = pages
        .first()
        .ok_or_else(|| ViprsError::Codec("tiff: missing decoded pages".into()))?;
    let width = first.width();
    let bands = first.bands();
    let page_height = first.height();
    let metadata = first.metadata().clone();

    if pages
        .iter()
        .any(|page| page.width() != width || page.height() != page_height || page.bands() != bands)
    {
        return Err(ViprsError::Codec(
            "tiff: requested pages must share width, height, and band count".into(),
        ));
    }

    let total_len: usize = pages.iter().map(|page| page.pixels().len()).sum();
    let mut pixels = Vec::with_capacity(total_len);
    for page in &pages {
        pixels.extend_from_slice(page.pixels());
    }

    InMemoryImage::from_buffer(width, page_height * page_count, bands, pixels)
        .map(|image| image.with_metadata(metadata).with_frames(pages))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

pub(super) fn streaming_tiff_page_selection(opts: &LoadOptions) -> Result<(), ViprsError> {
    let requested = opts.n.unwrap_or(1);
    if requested == 1 {
        Ok(())
    } else {
        Err(ViprsError::Codec(
            "tiff: streaming decode currently supports a single page selection".into(),
        ))
    }
}

pub(super) fn expected_tiff_sample_type<F: BandFormat>() -> Result<DecodingSampleType, ViprsError> {
    match F::ID {
        BandFormatId::U8 => Ok(DecodingSampleType::U8),
        BandFormatId::U16 => Ok(DecodingSampleType::U16),
        BandFormatId::F32 => Ok(DecodingSampleType::F32),
        other => Err(ViprsError::Codec(format!(
            "tiff: unsupported format {other:?} — only U8, U16, and F32 are supported"
        ))),
    }
}

pub(super) fn probe_tiff_with_options(
    src: &[u8],
    opts: &LoadOptions,
) -> Result<ImageMetadataProbe, ViprsError> {
    streaming_tiff_page_selection(opts)?;
    let total_pages = count_pages(src)?;
    let (start_page, page_count) = normalize_page_selection(total_pages, opts)?;
    let mut decoder =
        Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    for _ in 0..start_page {
        decoder
            .next_image()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    let root_dimensions = decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let subifd_offsets = current_page_subifd_offsets(&mut decoder)?;
    let mut subifd_dimensions = Vec::with_capacity(subifd_offsets.len());
    for offset in subifd_offsets {
        subifd_dimensions.push((offset, probe_ifd_dimensions(src, offset)?));
    }

    let selected_offset = choose_pyramid_level(root_dimensions, &subifd_dimensions, opts);
    let selected_storage = selected_offset
        .map(|offset| patch_first_ifd_offset(src, offset))
        .transpose()?;
    let decoder_src = selected_storage.as_deref().unwrap_or(src);
    let mut selected_decoder =
        Decoder::new(Cursor::new(decoder_src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let (width, height) = selected_decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let color_type = selected_decoder
        .colortype()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let bands = color_type_to_bands(color_type)?;
    let metadata = ImageMetadata {
        interpretation: Some(interpretation_from_tags(None, color_type, false)),
        n_pages: Some(total_pages),
        page_height: (total_pages > 1 || page_count > 1).then_some(height),
        ..ImageMetadata::default()
    };

    Ok(ImageMetadataProbe::new(width, height, bands).with_metadata(metadata))
}

pub(super) fn decode_tiff_region_with_options<F: BandFormat>(
    src: &[u8],
    opts: &LoadOptions,
    region: Region,
    output: &mut [u8],
) -> Result<(), ViprsError> {
    let expected_sample_type = expected_tiff_sample_type::<F>()?;
    streaming_tiff_page_selection(opts)?;
    let probe = probe_tiff_with_options(src, opts)?;
    let bands = probe.bands;
    let sample_size = std::mem::size_of::<F::Sample>();
    let expected_output_len = region
        .checked_pixel_count()
        .and_then(|pixel_count| pixel_count.checked_mul(bands as usize))
        .and_then(|samples| samples.checked_mul(sample_size))
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width: region.width,
            height: region.height,
            bands,
            bytes: u128::from(region.width)
                * u128::from(region.height)
                * u128::from(bands)
                * u128::from(sample_size as u64),
            limit_bytes: usize::MAX as u128,
            details: "tiff region output buffer exceeds addressable memory",
        })?;
    if output.len() != expected_output_len {
        return Err(ViprsError::Codec(format!(
            "tiff: output buffer size mismatch (got {}, expected {expected_output_len})",
            output.len()
        )));
    }
    if probe.width == 0 || probe.height == 0 {
        return Ok(());
    }

    let total_pages = count_pages(src)?;
    let (start_page, _page_count) = normalize_page_selection(total_pages, opts)?;
    let mut decoder =
        Decoder::new(Cursor::new(src)).map_err(|e| ViprsError::Codec(e.to_string()))?;
    for _ in 0..start_page {
        decoder
            .next_image()
            .map_err(|e| ViprsError::Codec(e.to_string()))?;
    }

    let root_dimensions = decoder
        .dimensions()
        .map_err(|e| ViprsError::Codec(e.to_string()))?;
    let subifd_offsets = current_page_subifd_offsets(&mut decoder)?;
    let mut subifd_dimensions = Vec::with_capacity(subifd_offsets.len());
    for offset in subifd_offsets {
        subifd_dimensions.push((offset, probe_ifd_dimensions(src, offset)?));
    }
    let selected_offset = choose_pyramid_level(root_dimensions, &subifd_dimensions, opts);
    let selected_storage = selected_offset
        .map(|offset| patch_first_ifd_offset(src, offset))
        .transpose()?;
    let decoder_src = selected_storage.as_deref().unwrap_or(src);
    let mut selected_decoder =
        Decoder::new(Cursor::new(decoder_src)).map_err(|e| ViprsError::Codec(e.to_string()))?;

    let (chunk_width, chunk_height) = selected_decoder.chunk_dimensions();
    let chunks_across = probe.width.div_ceil(chunk_width).max(1);
    let clamped_x0 = region.x.clamp(0, probe.width as i32 - 1);
    let clamped_y0 = region.y.clamp(0, probe.height as i32 - 1);
    let clamped_x1 = (region.x + region.width as i32 - 1).clamp(0, probe.width as i32 - 1);
    let clamped_y1 = (region.y + region.height as i32 - 1).clamp(0, probe.height as i32 - 1);
    let decoded_width = usize::try_from(clamped_x1 - clamped_x0 + 1)
        .map_err(|_| ViprsError::Codec("tiff: decoded width overflow".into()))?;
    let decoded_height = usize::try_from(clamped_y1 - clamped_y0 + 1)
        .map_err(|_| ViprsError::Codec("tiff: decoded height overflow".into()))?;
    let decoded_len = decoded_width
        .checked_mul(decoded_height)
        .and_then(|pixels| pixels.checked_mul(bands as usize))
        .and_then(|samples| samples.checked_mul(sample_size))
        .ok_or_else(|| ViprsError::Codec("tiff: decoded window size overflow".into()))?;
    let mut decoded = vec![0u8; decoded_len];
    let unit_x0 = u32::try_from(clamped_x0)
        .map_err(|_| ViprsError::Codec("tiff: clamped x overflow".into()))?
        / chunk_width.max(1);
    let unit_y0 = u32::try_from(clamped_y0)
        .map_err(|_| ViprsError::Codec("tiff: clamped y overflow".into()))?
        / chunk_height.max(1);
    let unit_x1 = u32::try_from(clamped_x1)
        .map_err(|_| ViprsError::Codec("tiff: clamped x overflow".into()))?
        / chunk_width.max(1);
    let unit_y1 = u32::try_from(clamped_y1)
        .map_err(|_| ViprsError::Codec("tiff: clamped y overflow".into()))?
        / chunk_height.max(1);

    for unit_y in unit_y0..=unit_y1 {
        for unit_x in unit_x0..=unit_x1 {
            let unit_index = unit_y
                .checked_mul(chunks_across)
                .and_then(|row| row.checked_add(unit_x))
                .ok_or_else(|| ViprsError::Codec("tiff: coding unit index overflow".into()))?;
            let layout = selected_decoder
                .image_coding_unit_layout(TiffCodingUnit(unit_index))
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            if layout.sample_type != Some(expected_sample_type) {
                return Err(ViprsError::Codec(format!(
                    "tiff: requested {:?}, but chunk layout is {:?}",
                    F::ID,
                    layout.sample_type
                )));
            }

            let mut chunk = vec![0u8; layout.complete_len];
            selected_decoder
                .read_coding_unit_bytes(TiffCodingUnit(unit_index), &mut chunk)
                .map_err(|e| ViprsError::Codec(e.to_string()))?;
            let (unit_data_width, unit_data_height) =
                selected_decoder.chunk_data_dimensions(unit_index);
            copy_tiff_coding_unit_into_window(
                &chunk,
                &layout,
                bands,
                sample_size,
                (unit_x * chunk_width, unit_y * chunk_height),
                (unit_data_width, unit_data_height),
                (clamped_x0 as u32, clamped_y0 as u32),
                decoded_width,
                decoded_height,
                &mut decoded,
            )?;
        }
    }

    for row in 0..region.height as i32 {
        let src_y = (region.y + row).clamp(0, probe.height as i32 - 1) - clamped_y0;
        for col in 0..region.width as i32 {
            let src_x = (region.x + col).clamp(0, probe.width as i32 - 1) - clamped_x0;
            let src_index =
                ((src_y as usize * decoded_width) + src_x as usize) * bands as usize * sample_size;
            let dst_index = ((row as usize * region.width as usize) + col as usize)
                * bands as usize
                * sample_size;
            output[dst_index..dst_index + bands as usize * sample_size]
                .copy_from_slice(&decoded[src_index..src_index + bands as usize * sample_size]);
        }
    }

    Ok(())
}

pub(super) fn copy_tiff_coding_unit_into_window(
    chunk: &[u8],
    layout: &BufferLayoutPreference,
    bands: u32,
    sample_size: usize,
    unit_origin: (u32, u32),
    unit_size: (u32, u32),
    window_origin: (u32, u32),
    window_width: usize,
    window_height: usize,
    decoded: &mut [u8],
) -> Result<(), ViprsError> {
    let row_stride = layout
        .row_stride
        .map(std::num::NonZeroUsize::get)
        .ok_or_else(|| ViprsError::Codec("tiff: missing coding-unit row stride".into()))?;
    let plane_stride = layout
        .plane_stride
        .map_or(chunk.len(), std::num::NonZeroUsize::get);
    let overlap_x0 = unit_origin.0.max(window_origin.0);
    let overlap_y0 = unit_origin.1.max(window_origin.1);
    let overlap_x1 = (unit_origin.0 + unit_size.0).min(window_origin.0 + window_width as u32);
    let overlap_y1 = (unit_origin.1 + unit_size.1).min(window_origin.1 + window_height as u32);
    if overlap_x0 >= overlap_x1 || overlap_y0 >= overlap_y1 {
        return Ok(());
    }

    if layout.planes == 1 {
        let bytes_per_pixel = bands as usize * sample_size;
        for y in overlap_y0..overlap_y1 {
            let chunk_row = (y - unit_origin.1) as usize;
            let window_row = (y - window_origin.1) as usize;
            let src_x = (overlap_x0 - unit_origin.0) as usize;
            let dst_x = (overlap_x0 - window_origin.0) as usize;
            let pixel_count = (overlap_x1 - overlap_x0) as usize;
            let src_start = chunk_row * row_stride + src_x * bytes_per_pixel;
            let dst_start = (window_row * window_width + dst_x) * bytes_per_pixel;
            let byte_len = pixel_count * bytes_per_pixel;
            decoded[dst_start..dst_start + byte_len]
                .copy_from_slice(&chunk[src_start..src_start + byte_len]);
        }
        return Ok(());
    }

    if layout.planes != bands as usize {
        return Err(ViprsError::Codec(format!(
            "tiff: unsupported plane layout {} for {bands} bands",
            layout.planes
        )));
    }

    for y in overlap_y0..overlap_y1 {
        let chunk_row = (y - unit_origin.1) as usize;
        let window_row = (y - window_origin.1) as usize;
        for x in overlap_x0..overlap_x1 {
            let chunk_col = (x - unit_origin.0) as usize;
            let window_col = (x - window_origin.0) as usize;
            let dst_start = (window_row * window_width + window_col) * bands as usize * sample_size;
            for band in 0..bands as usize {
                let src_start =
                    band * plane_stride + chunk_row * row_stride + chunk_col * sample_size;
                decoded[dst_start + band * sample_size..dst_start + (band + 1) * sample_size]
                    .copy_from_slice(&chunk[src_start..src_start + sample_size]);
            }
        }
    }
    Ok(())
}
