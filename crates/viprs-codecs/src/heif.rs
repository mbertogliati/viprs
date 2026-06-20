//! HEIF/HEIC codec — decode and encode via `libheif-rs`.
//!
//! HEIF encode prefers an HEVC encoder and falls back to AV1 when that is the
//! only encoder available in the linked libheif build. Multi-page/page-sequence
//! inputs decode into [`AnimationFrame`]s for animated thumbnail flows.

use super::heif_support::{
    HeifWriteMetadata, checked_interleaved_byte_count, checked_interleaved_row_bytes,
    checked_interleaved_sample_count, encode_interleaved, normalize_decoded_image, read_metadata,
    shared_libheif,
};

use libheif_rs::{
    ColorSpace, CompressionFormat, HeifContext, ImageHandle, ItemId, LibHeif, RgbChroma,
};

use viprs_core::codec_options::{
    HeifBitDepth, HeifCompression, HeifSubsampling, LoadOptions, SaveOptions,
};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, U8, U16};
use viprs_core::image::{AnimationFrame, FrameDisposal, Image, ImageMetadata};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

/// HEIF/HEIC codec.
pub struct HeifCodec;

struct HeifPageSelection {
    item_ids: Vec<ItemId>,
    first_page: usize,
    page_count: usize,
    total_pages: u32,
}

const HEIF_DEFAULT_QUALITY: u8 = 50;
const HEIF_DECODE_PLANE_TOO_LARGE: &str = "heif decode plane exceeds addressable memory";
const HEIF_ENCODE_BUFFER_TOO_LARGE: &str = "heif encode buffer exceeds addressable memory";

#[inline]
fn clamp_quality(quality: Option<u8>, default: u8) -> u8 {
    quality.unwrap_or(default).min(100)
}

fn available_encoders(lib_heif: &LibHeif) -> String {
    let available = lib_heif.encoder_descriptors(16, None, None);
    if available.is_empty() {
        return "none".into();
    }
    available
        .iter()
        .map(|descriptor| {
            format!(
                "{} ({:?})",
                descriptor.id(),
                descriptor.compression_format()
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn heif_compression(lib_heif: &LibHeif) -> Result<CompressionFormat, ViprsError> {
    if lib_heif.encoder_for_format(CompressionFormat::Hevc).is_ok() {
        return Ok(CompressionFormat::Hevc);
    }
    if lib_heif.encoder_for_format(CompressionFormat::Av1).is_ok() {
        return Ok(CompressionFormat::Av1);
    }

    Err(ViprsError::Codec(format!(
        "heif: no HEVC or AV1 encoder available in libheif; available encoders: {}",
        available_encoders(lib_heif)
    )))
}

fn configured_heif_compression(
    lib_heif: &LibHeif,
    opts: &SaveOptions,
) -> Result<CompressionFormat, ViprsError> {
    let requested = match opts.heif_compression.unwrap_or(HeifCompression::Auto) {
        HeifCompression::Auto => return heif_compression(lib_heif),
        HeifCompression::Hevc => CompressionFormat::Hevc,
        HeifCompression::Avc => CompressionFormat::Avc,
        HeifCompression::Jpeg => CompressionFormat::Jpeg,
        HeifCompression::Av1 => CompressionFormat::Av1,
    };

    lib_heif.encoder_for_format(requested).map_err(|_| {
        ViprsError::Codec(format!(
            "heif: requested {requested:?} encoder is unavailable; available encoders: {}",
            available_encoders(lib_heif)
        ))
    })?;
    Ok(requested)
}

#[inline]
fn resolved_bit_depth(opts: &SaveOptions, is_u16: bool, has_alpha: bool) -> u8 {
    opts.heif_bit_depth
        .unwrap_or(if is_u16 && has_alpha {
            HeifBitDepth::Sixteen
        } else if is_u16 {
            HeifBitDepth::Twelve
        } else {
            HeifBitDepth::Eight
        })
        .as_u8()
}

#[inline]
fn resolved_subsampling(opts: &SaveOptions) -> HeifSubsampling {
    if opts.lossless == Some(true) {
        HeifSubsampling::Subsample444
    } else {
        opts.heif_subsampling.unwrap_or(HeifSubsampling::Auto)
    }
}

const fn decoded_chroma(bit_depth: u8, has_alpha: bool) -> RgbChroma {
    if bit_depth > 8 {
        if has_alpha {
            RgbChroma::HdrRgbaBe
        } else {
            RgbChroma::HdrRgbBe
        }
    } else if has_alpha {
        RgbChroma::Rgba
    } else {
        RgbChroma::Rgb
    }
}

fn normalize_u16_samples_for_bit_depth(samples: &mut [u16], bit_depth: u8) {
    let shift = 16u8.saturating_sub(bit_depth);
    if shift > 0 {
        for sample in samples {
            *sample <<= shift;
        }
    }
}

fn heif_page_metadata(
    mut metadata: ImageMetadata,
    page_height: u32,
    total_pages: u32,
) -> ImageMetadata {
    metadata.n_pages = Some(total_pages);
    metadata.page_height = (total_pages > 1).then_some(page_height);
    metadata
}

fn heif_top_level_image_ids(ctx: &HeifContext) -> Result<Vec<ItemId>, ViprsError> {
    let total_pages = ctx.number_of_top_level_images();
    if total_pages == 0 {
        return Err(ViprsError::Codec(
            "heif: container does not contain any top-level images".into(),
        ));
    }

    let mut item_ids = vec![0; total_pages];
    let listed = ctx.top_level_image_ids(&mut item_ids);
    item_ids.truncate(listed);
    if item_ids.is_empty() {
        return Err(ViprsError::Codec(
            "heif: failed to enumerate top-level images".into(),
        ));
    }

    Ok(item_ids)
}

fn heif_primary_page(ctx: &HeifContext, item_ids: &[ItemId]) -> Result<usize, ViprsError> {
    for (index, &item_id) in item_ids.iter().enumerate() {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("heif: image_handle({item_id}): {e}")))?;
        if handle.is_primary() {
            return Ok(index);
        }
    }

    Err(ViprsError::Codec(
        "heif: failed to locate primary image among top-level images".into(),
    ))
}

fn heif_selected_page_range(
    opts: &LoadOptions,
    total_pages: usize,
    primary_page: usize,
) -> Result<(usize, usize), ViprsError> {
    let page = if opts.page.is_none() && opts.n.is_none() {
        primary_page
    } else {
        usize::try_from(opts.page.unwrap_or(0))
            .map_err(|_| ViprsError::Codec("heif: page index exceeds platform limits".into()))?
    };
    let requested = opts.n.unwrap_or(1);
    let pages_to_decode = if requested == -1 {
        total_pages.saturating_sub(page)
    } else if requested > 0 {
        usize::try_from(requested)
            .map_err(|_| ViprsError::Codec(format!("heif: invalid n={requested}")))?
    } else {
        0
    };

    if page >= total_pages || pages_to_decode == 0 || page + pages_to_decode > total_pages {
        return Err(ViprsError::Codec(format!(
            "heif: bad page number (page={page}, n={requested}, total_pages={total_pages})"
        )));
    }

    Ok((page, pages_to_decode))
}

fn select_heif_pages(
    ctx: &HeifContext,
    opts: &LoadOptions,
) -> Result<HeifPageSelection, ViprsError> {
    let item_ids = heif_top_level_image_ids(ctx)?;
    let primary_page = heif_primary_page(ctx, &item_ids)?;
    let (first_page, page_count) = heif_selected_page_range(opts, item_ids.len(), primary_page)?;

    Ok(HeifPageSelection {
        total_pages: item_ids.len() as u32,
        item_ids,
        first_page,
        page_count,
    })
}

fn cast_decoded_frame<S: BandFormat, D: BandFormat>(
    image: Image<S>,
    context: &str,
) -> Result<Image<D>, ViprsError> {
    let width = image.width();
    let height = image.height();
    let bands = image.bands();
    let metadata = image.metadata().clone();
    let samples = bytemuck::allocation::try_cast_vec::<S::Sample, D::Sample>(image.into_buffer())
        .map_err(|(e, _)| ViprsError::Codec(format!("{context}: cast error: {e:?}")))?;
    Image::from_buffer(width, height, bands, samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

fn cast_decoded_image<S: BandFormat, D: BandFormat>(
    image: Image<S>,
    context: &str,
) -> Result<Image<D>, ViprsError>
where
    S::Sample: Clone,
{
    let animation_frames = image.animation_frames().map(ToOwned::to_owned);
    let mut cast = cast_decoded_frame::<S, D>(image, context)?;
    if let Some(frames) = animation_frames {
        let converted_frames = frames
            .into_iter()
            .map(|frame| {
                let delay_ms = frame.delay_ms();
                let disposal = frame.disposal();
                cast_decoded_frame::<S, D>(frame.into_image(), context)
                    .map(|image| AnimationFrame::new(image, delay_ms, disposal))
            })
            .collect::<Result<Vec<_>, _>>()?;
        cast = cast.with_animation_frames(converted_frames);
    }
    Ok(cast)
}

#[inline]
fn checked_heif_decode_sample_count(
    width: u32,
    height: u32,
    bands: u32,
) -> Result<usize, ViprsError> {
    checked_interleaved_sample_count(width, height, bands, HEIF_DECODE_PLANE_TOO_LARGE)
}

#[inline]
fn checked_heif_row_bytes(
    width: u32,
    bands: u32,
    bytes_per_sample: usize,
) -> Result<usize, ViprsError> {
    checked_interleaved_row_bytes(width, bands, bytes_per_sample, HEIF_DECODE_PLANE_TOO_LARGE)
}

fn decode_interleaved_plane<T>(
    _ctx: &HeifContext,
    handle: &ImageHandle,
    total_pages: u32,
    build: impl FnOnce(u32, u32, u32, u8, ImageMetadata, &[u8], usize) -> Result<T, ViprsError>,
) -> Result<T, ViprsError> {
    let metadata = read_metadata("heif", handle)?;
    let width = handle.width();
    let height = handle.height();
    let has_alpha = handle.has_alpha_channel();
    let bands = if has_alpha { 4 } else { 3 };
    let bit_depth = handle
        .luma_bits_per_pixel()
        .max(handle.chroma_bits_per_pixel())
        .max(8);

    let lib_heif = shared_libheif("heif")?;
    let image = lib_heif
        .decode(
            handle,
            ColorSpace::Rgb(decoded_chroma(bit_depth, has_alpha)),
            None,
        )
        .map_err(|e| ViprsError::Codec(format!("heif: decode: {e}")))?;
    let plane = image
        .planes()
        .interleaved
        .ok_or_else(|| ViprsError::Codec("heif: no interleaved plane".into()))?;

    build(
        width,
        height,
        bands,
        bit_depth,
        heif_page_metadata(metadata, height, total_pages),
        plane.data,
        plane.stride,
    )
}

fn decode_single_page_u8(
    ctx: &HeifContext,
    handle: &ImageHandle,
    total_pages: u32,
    opts: &LoadOptions,
) -> Result<Image<U8>, ViprsError> {
    decode_interleaved_plane(
        ctx,
        handle,
        total_pages,
        |width, height, bands, bit_depth, metadata, plane_data, stride| {
            let sample_count = checked_heif_decode_sample_count(width, height, bands)?;
            let row_samples = checked_heif_row_bytes(width, bands, 1)?;
            let mut samples = vec![0u8; sample_count];
            if bit_depth > 8 {
                let shift = 16u8.saturating_sub(bit_depth);
                for row in 0..height as usize {
                    let src_start = row * stride;
                    let src_end = src_start + (row_samples * 2);
                    let src_row = &plane_data[src_start..src_end];
                    let dst_row = &mut samples[(row * row_samples)..((row + 1) * row_samples)];
                    for (dst, chunk) in dst_row.iter_mut().zip(src_row.chunks_exact(2)) {
                        *dst = ((u16::from_be_bytes([chunk[0], chunk[1]]) << shift) >> 8) as u8;
                    }
                }
            } else {
                for row in 0..height as usize {
                    let src_start = row * stride;
                    let src_end = src_start + row_samples;
                    let dst_start = row * row_samples;
                    let dst_end = dst_start + row_samples;
                    samples[dst_start..dst_end].copy_from_slice(&plane_data[src_start..src_end]);
                }
            }

            let image = Image::from_buffer(width, height, bands, samples)
                .map_err(|e| ViprsError::Codec(e.to_string()))?
                .with_metadata(metadata);
            normalize_decoded_image(image, opts.no_rotate, "heif")
        },
    )
}

fn decode_single_page_u16(
    ctx: &HeifContext,
    handle: &ImageHandle,
    total_pages: u32,
    opts: &LoadOptions,
) -> Result<Image<U16>, ViprsError> {
    decode_interleaved_plane(
        ctx,
        handle,
        total_pages,
        |width, height, bands, bit_depth, metadata, plane_data, stride| {
            let sample_count = checked_heif_decode_sample_count(width, height, bands)?;
            let mut samples = Vec::with_capacity(sample_count);
            let row_bytes =
                checked_heif_row_bytes(width, bands, if bit_depth > 8 { 2 } else { 1 })?;
            for row in 0..height as usize {
                let row_start = row * stride;
                let row_end = row_start + row_bytes;
                let row_data = &plane_data[row_start..row_end];
                if bit_depth > 8 {
                    samples.extend(
                        row_data
                            .chunks_exact(2)
                            .map(|chunk| u16::from_be_bytes([chunk[0], chunk[1]])),
                    );
                } else {
                    samples.extend(row_data.iter().copied().map(u16::from));
                }
            }
            normalize_u16_samples_for_bit_depth(&mut samples, bit_depth);
            let image = Image::from_buffer(width, height, bands, samples)
                .map_err(|e| ViprsError::Codec(e.to_string()))?
                .with_metadata(metadata);
            normalize_decoded_image(image, opts.no_rotate, "heif")
        },
    )
}

fn decode_to_u8(src: &[u8], opts: &LoadOptions) -> Result<Image<U8>, ViprsError> {
    let ctx = HeifContext::read_from_bytes(src)
        .map_err(|e| ViprsError::Codec(format!("heif: read_from_bytes: {e}")))?;
    let selection = select_heif_pages(&ctx, opts)?;
    if selection.page_count == 1 {
        let item_id = selection.item_ids[selection.first_page];
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("heif: image_handle({item_id}): {e}")))?;
        return decode_single_page_u8(&ctx, &handle, selection.total_pages, opts);
    }

    let mut frames = Vec::with_capacity(selection.page_count);
    for &item_id in
        &selection.item_ids[selection.first_page..selection.first_page + selection.page_count]
    {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("heif: image_handle({item_id}): {e}")))?;
        let frame = decode_single_page_u8(&ctx, &handle, selection.total_pages, opts)?;
        frames.push(AnimationFrame::new(frame, 0, FrameDisposal::Keep));
    }

    let page_height = frames[0].image().height();
    let mut image = Image::from_frames(frames).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut metadata = image.metadata().clone();
    metadata.n_pages = Some(selection.total_pages);
    metadata.page_height = (selection.total_pages > 1).then_some(page_height);
    image = image.with_metadata(metadata);
    Ok(image)
}

fn decode_to_u16(src: &[u8], opts: &LoadOptions) -> Result<Image<U16>, ViprsError> {
    let ctx = HeifContext::read_from_bytes(src)
        .map_err(|e| ViprsError::Codec(format!("heif: read_from_bytes: {e}")))?;
    let selection = select_heif_pages(&ctx, opts)?;
    if selection.page_count == 1 {
        let item_id = selection.item_ids[selection.first_page];
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("heif: image_handle({item_id}): {e}")))?;
        return decode_single_page_u16(&ctx, &handle, selection.total_pages, opts);
    }

    let mut frames = Vec::with_capacity(selection.page_count);
    for &item_id in
        &selection.item_ids[selection.first_page..selection.first_page + selection.page_count]
    {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("heif: image_handle({item_id}): {e}")))?;
        let frame = decode_single_page_u16(&ctx, &handle, selection.total_pages, opts)?;
        frames.push(AnimationFrame::new(frame, 0, FrameDisposal::Keep));
    }

    let page_height = frames[0].image().height();
    let mut image = Image::from_frames(frames).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut metadata = image.metadata().clone();
    metadata.n_pages = Some(selection.total_pages);
    metadata.page_height = (selection.total_pages > 1).then_some(page_height);
    image = image.with_metadata(metadata);
    Ok(image)
}

#[inline]
fn metadata_to_write<'a>(metadata: &'a ImageMetadata, opts: &SaveOptions) -> HeifWriteMetadata<'a> {
    if opts.strip_metadata == Some(true) {
        HeifWriteMetadata::default()
    } else {
        HeifWriteMetadata {
            exif: metadata.exif.as_deref(),
            xmp: metadata.xmp.as_deref(),
            icc_profile: metadata.icc_profile.as_deref(),
        }
    }
}

fn encode_u8_with_libheif(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
    metadata: &ImageMetadata,
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let bands = match bands {
        1 | 3 | 4 => bands,
        n => {
            return Err(ViprsError::Codec(format!(
                "heif: unsupported band count {n} — only 1, 3, and 4 bands are supported"
            )));
        }
    };
    let output_bands = if bands == 1 { 3 } else { bands };
    let bit_depth = resolved_bit_depth(opts, false, bands == 4);
    let storage;
    let pixel_bytes = if bands != 1 && bit_depth == 8 {
        pixels
    } else {
        let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
        let byte_count = checked_interleaved_byte_count(
            width,
            height,
            output_bands,
            bytes_per_sample,
            HEIF_ENCODE_BUFFER_TOO_LARGE,
        )?;
        let mut expanded = Vec::with_capacity(byte_count);

        match bands {
            1 => {
                for &sample in pixels {
                    if bit_depth > 8 {
                        let encoded = (u16::from(sample) << (bit_depth - 8)).to_be_bytes();
                        expanded.extend_from_slice(&encoded);
                        expanded.extend_from_slice(&encoded);
                        expanded.extend_from_slice(&encoded);
                    } else {
                        expanded.extend_from_slice(&[sample, sample, sample]);
                    }
                }
            }
            3 | 4 => {
                for &sample in pixels {
                    expanded
                        .extend_from_slice(&(u16::from(sample) << (bit_depth - 8)).to_be_bytes());
                }
            }
            _ => {
                return Err(ViprsError::Codec(
                    "heif: unsupported band count after validation".into(),
                ));
            }
        }

        storage = expanded;
        storage.as_slice()
    };

    let lib_heif = shared_libheif("heif")?;
    let compression = configured_heif_compression(lib_heif, opts)?;
    encode_interleaved(
        "heif",
        compression,
        width,
        height,
        output_bands,
        bit_depth,
        pixel_bytes,
        opts.lossless == Some(true),
        clamp_quality(opts.quality, HEIF_DEFAULT_QUALITY),
        opts.effort,
        resolved_subsampling(opts),
        metadata_to_write(metadata, opts),
    )
}

fn encode_u16_with_libheif(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u16],
    metadata: &ImageMetadata,
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let bands = match bands {
        1 | 3 | 4 => bands,
        n => {
            return Err(ViprsError::Codec(format!(
                "heif: unsupported band count {n} — only 1, 3, and 4 bands are supported"
            )));
        }
    };
    let output_bands = if bands == 1 { 3 } else { bands };
    let bit_depth = resolved_bit_depth(opts, true, bands == 4);
    let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
    let byte_count = checked_interleaved_byte_count(
        width,
        height,
        output_bands,
        bytes_per_sample,
        HEIF_ENCODE_BUFFER_TOO_LARGE,
    )?;
    let mut pixel_bytes = Vec::with_capacity(byte_count);

    match bands {
        1 => {
            for &sample in pixels {
                let narrowed = sample >> (16 - bit_depth);
                if bit_depth > 8 {
                    let encoded = narrowed.to_be_bytes();
                    pixel_bytes.extend_from_slice(&encoded);
                    pixel_bytes.extend_from_slice(&encoded);
                    pixel_bytes.extend_from_slice(&encoded);
                } else {
                    let encoded = narrowed as u8;
                    pixel_bytes.extend_from_slice(&[encoded, encoded, encoded]);
                }
            }
        }
        3 | 4 => {
            for &sample in pixels {
                let narrowed = sample >> (16 - bit_depth);
                if bit_depth > 8 {
                    pixel_bytes.extend_from_slice(&narrowed.to_be_bytes());
                } else {
                    pixel_bytes.push(narrowed as u8);
                }
            }
        }
        _ => {
            return Err(ViprsError::Codec(
                "heif: unsupported band count after validation".into(),
            ));
        }
    }

    let lib_heif = shared_libheif("heif")?;
    let compression = configured_heif_compression(lib_heif, opts)?;
    encode_interleaved(
        "heif",
        compression,
        width,
        height,
        output_bands,
        bit_depth,
        &pixel_bytes,
        opts.lossless == Some(true),
        clamp_quality(opts.quality, HEIF_DEFAULT_QUALITY),
        opts.effort,
        resolved_subsampling(opts),
        metadata_to_write(metadata, opts),
    )
}

impl ImageDecoder for HeifCodec {
    fn format_name(&self) -> &'static str {
        "heif"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        if header.len() < 12 || &header[4..8] != b"ftyp" {
            return false;
        }
        matches!(
            &header[8..12],
            b"heic" | b"heix" | b"hevc" | b"heim" | b"heis" | b"hevm" | b"hevs" | b"mif1"
        )
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<Image<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<Image<F>, ViprsError>
    where
        Self: Sized,
    {
        match F::ID {
            BandFormatId::U8 => {
                let decoded = decode_to_u8(src, opts)?;
                cast_decoded_image::<U8, F>(decoded, "heif")
            }
            BandFormatId::U16 => {
                let image = decode_to_u16(src, opts)?;
                cast_decoded_image::<U16, F>(image, "heif")
            }
            _ => Err(ViprsError::Codec(format!(
                "heif: unsupported format {:?} — only U8 and U16 are supported",
                F::ID
            ))),
        }
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let ctx = HeifContext::read_from_bytes(src)
            .map_err(|e| ViprsError::Codec(format!("heif: probe: {e}")))?;
        let handle = ctx
            .primary_image_handle()
            .map_err(|e| ViprsError::Codec(format!("heif: probe handle: {e}")))?;
        let bands = if handle.has_alpha_channel() { 4 } else { 3 };
        Ok((handle.width(), handle.height(), bands))
    }
}

impl ImageEncoder for HeifCodec {
    fn format_name(&self) -> &'static str {
        "heif"
    }

    fn encode<F: BandFormat>(&self, image: &Image<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &Image<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        match F::ID {
            BandFormatId::U8 => {
                let pixels = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());
                encode_u8_with_libheif(
                    image.width(),
                    image.height(),
                    image.bands(),
                    pixels,
                    image.metadata(),
                    opts,
                )
            }
            BandFormatId::U16 => {
                let pixels = bytemuck::cast_slice::<F::Sample, u16>(image.pixels());
                encode_u16_with_libheif(
                    image.width(),
                    image.height(),
                    image.bands(),
                    pixels,
                    image.metadata(),
                    opts,
                )
            }
            _ => Err(ViprsError::Codec(format!(
                "heif: unsupported format {:?} — only U8 and U16 are supported",
                F::ID
            ))),
        }
    }
}

#[cfg(all(test, feature = "_integration"))]
mod tests {
    use super::*;
    use libheif_rs::{Channel, EncoderQuality, HeifContext, Image as LibHeifImage};
    use viprs_core::format::{F32, U8, U16};
    use viprs_core::image::{FrameDisposal, ImageMetadata};

    const BENCH_512_HEIF: &[u8] =
        include_bytes!("../../../tests/fixtures/images/bench_512x512.heic");
    const BENCH_2048_HEIF: &[u8] =
        include_bytes!("../../../tests/fixtures/images/bench_2048x2048.heic");
    const U8_ALLOC_ENV: &str = "VIPRS_HEIF_U8_ALLOC_CHILD";
    const U8_ALLOC_CHILD_TEST: &str =
        "adapters::codecs::heif::tests::u8_decode_peak_live_bytes_stay_near_output_buffer_child";

    fn heif_encoder_available() -> bool {
        shared_libheif("heif").is_ok_and(|lib_heif| {
            lib_heif.encoder_for_format(CompressionFormat::Hevc).is_ok()
                || lib_heif.encoder_for_format(CompressionFormat::Av1).is_ok()
        })
    }

    fn heif_u8_decode_alloc_stats() -> crate::test_support::AllocStats {
        crate::test_support::reset_alloc_stats();
        let decoded = HeifCodec.decode::<U8>(BENCH_2048_HEIF).unwrap();
        let stats = crate::test_support::alloc_stats();
        assert_eq!(decoded.width(), 2048);
        assert_eq!(decoded.height(), 2048);
        stats
    }

    fn rgb_u8_gradient(width: u32, height: u32) -> Image<U8> {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        (x.wrapping_mul(17) + y.wrapping_mul(3)) as u8,
                        (x.wrapping_mul(5) + y.wrapping_mul(11)) as u8,
                        (x ^ y) as u8,
                    ]
                })
            })
            .collect();
        Image::<U8>::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn rgb_u16_gradient(width: u32, height: u32) -> Image<U16> {
        let pixels: Vec<u16> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 257 + y * 19) & 0xFFFF) as u16,
                        ((x * 97 + y * 521) & 0xFFFF) as u16,
                        ((x * 409 + y * 73) & 0xFFFF) as u16,
                    ]
                })
            })
            .collect();
        Image::<U16>::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn rgba_u16_gradient(width: u32, height: u32) -> Image<U16> {
        let pixels: Vec<u16> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 257 + y * 19 + 1) & 0xFFFF) as u16,
                        ((x * 97 + y * 521 + 3) & 0xFFFF) as u16,
                        ((x * 409 + y * 73 + 5) & 0xFFFF) as u16,
                        ((x * 887 + y * 29 + 7) & 0xFFFF) as u16,
                    ]
                })
            })
            .collect();
        Image::<U16>::from_buffer(width, height, 4, pixels).unwrap()
    }

    fn heif_sequence_encoder_available() -> bool {
        shared_libheif("heif").is_ok_and(|lib_heif| {
            lib_heif.encoder_for_format(CompressionFormat::Hevc).is_ok()
                || lib_heif.encoder_for_format(CompressionFormat::Av1).is_ok()
        })
    }

    fn libheif_rgb_image(width: u32, height: u32, pixels: &[u8]) -> LibHeifImage {
        let mut image = LibHeifImage::new(width, height, ColorSpace::Rgb(RgbChroma::Rgb)).unwrap();
        image
            .create_plane(Channel::Interleaved, width, height, 8)
            .unwrap();
        let mut planes = image.planes_mut();
        let plane = planes.interleaved.as_mut().unwrap();
        let row_bytes = width as usize * 3;
        for row in 0..height as usize {
            let src_start = row * row_bytes;
            let src_end = src_start + row_bytes;
            let dst_start = row * plane.stride;
            let dst_end = dst_start + row_bytes;
            plane.data[dst_start..dst_end].copy_from_slice(&pixels[src_start..src_end]);
        }
        image
    }

    fn two_page_heif(primary_page: usize) -> Vec<u8> {
        let lib_heif = LibHeif::new();
        let mut context = HeifContext::new().unwrap();
        let compression = if lib_heif.encoder_for_format(CompressionFormat::Hevc).is_ok() {
            CompressionFormat::Hevc
        } else {
            CompressionFormat::Av1
        };
        let mut encoder = lib_heif.encoder_for_format(compression).unwrap();
        encoder.set_quality(EncoderQuality::Lossy(100)).unwrap();

        let first = rgb_u8_gradient(1, 1);
        let second = Image::<U8>::from_buffer(1, 1, 3, vec![0, 0, 255]).unwrap();
        let first_heif = libheif_rgb_image(1, 1, bytemuck::cast_slice(first.pixels()));
        let second_heif = libheif_rgb_image(1, 1, bytemuck::cast_slice(second.pixels()));

        let mut first_handle = context
            .encode_image(&first_heif, &mut encoder, None)
            .unwrap();
        let mut second_handle = context
            .encode_image(&second_heif, &mut encoder, None)
            .unwrap();
        match primary_page {
            0 => context.set_primary_image(&mut first_handle).unwrap(),
            1 => context.set_primary_image(&mut second_handle).unwrap(),
            _ => panic!("primary_page must be 0 or 1"),
        }

        context.write_to_bytes().unwrap()
    }

    fn exif_blob(orientation: u16) -> Vec<u8> {
        let mut exif = Vec::with_capacity(32);
        exif.extend_from_slice(b"Exif\0\0");
        exif.extend_from_slice(b"II");
        exif.extend_from_slice(&42u16.to_le_bytes());
        exif.extend_from_slice(&8u32.to_le_bytes());
        exif.extend_from_slice(&1u16.to_le_bytes());
        exif.extend_from_slice(&0x0112u16.to_le_bytes());
        exif.extend_from_slice(&3u16.to_le_bytes());
        exif.extend_from_slice(&1u32.to_le_bytes());
        exif.extend_from_slice(&orientation.to_le_bytes());
        exif.extend_from_slice(&0u16.to_le_bytes());
        exif.extend_from_slice(&0u32.to_le_bytes());
        exif
    }

    fn exif_orientation(exif: &[u8]) -> Option<u8> {
        exif.get(18..20)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .and_then(|value| u8::try_from(value).ok())
            .filter(|value| (1..=8).contains(value))
    }

    fn apply_orientation(
        pixels: &[u8],
        width: u32,
        height: u32,
        bands: u32,
        orientation: u8,
    ) -> (u32, u32, Vec<u8>) {
        let width_usize = width as usize;
        let height_usize = height as usize;
        let bands_usize = bands as usize;
        let (out_width, out_height) = match orientation {
            5..=8 => (height, width),
            _ => (width, height),
        };
        let out_width_usize = out_width as usize;
        let out_height_usize = out_height as usize;
        let mut out = vec![0u8; out_width_usize * out_height_usize * bands_usize];

        for out_y in 0..out_height_usize {
            for out_x in 0..out_width_usize {
                let (src_x, src_y) = match orientation {
                    2 => (width_usize - 1 - out_x, out_y),
                    3 => (width_usize - 1 - out_x, height_usize - 1 - out_y),
                    4 => (out_x, height_usize - 1 - out_y),
                    5 => (out_y, out_x),
                    6 => (out_y, height_usize - 1 - out_x),
                    7 => (width_usize - 1 - out_y, height_usize - 1 - out_x),
                    8 => (width_usize - 1 - out_y, out_x),
                    _ => (out_x, out_y),
                };
                let src_base = (src_y * width_usize + src_x) * bands_usize;
                let dst_base = (out_y * out_width_usize + out_x) * bands_usize;
                out[dst_base..dst_base + bands_usize]
                    .copy_from_slice(&pixels[src_base..src_base + bands_usize]);
            }
        }

        (out_width, out_height, out)
    }

    #[test]
    fn sniff_recognises_heic_brand() {
        let codec = HeifCodec;
        let mut header = vec![0u8; 12];
        header[4..8].copy_from_slice(b"ftyp");
        header[8..12].copy_from_slice(b"heic");
        assert!(codec.sniff(&header));
    }

    #[test]
    fn sniff_rejects_avif_brand() {
        let codec = HeifCodec;
        let mut header = vec![0u8; 12];
        header[4..8].copy_from_slice(b"ftyp");
        header[8..12].copy_from_slice(b"avif");
        assert!(!codec.sniff(&header));
    }

    #[test]
    fn heif_decode_rejects_u32_max_dimensions() {
        let result = checked_heif_decode_sample_count(u32::MAX, u32::MAX, 4);
        assert!(matches!(
            result,
            Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                ..
            }) if width == u32::MAX && height == u32::MAX && bands == 4
        ));
    }

    #[test]
    fn heif_u8_encode_rejects_oversized_grayscale_expansion() {
        let err = encode_u8_with_libheif(
            3_100_000_000,
            2_000_000_000,
            1,
            &[],
            &ImageMetadata::default(),
            &SaveOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                ..
            } if width == 3_100_000_000 && height == 2_000_000_000 && bands == 3
        ));
    }

    #[test]
    fn heif_u16_encode_rejects_oversized_byte_widening() {
        let err = encode_u16_with_libheif(
            3_100_000_000,
            1_000_000_000,
            3,
            &[],
            &ImageMetadata::default(),
            &SaveOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                ..
            } if width == 3_100_000_000 && height == 1_000_000_000 && bands == 3
        ));
    }

    #[test]
    fn clamp_quality_uses_default_and_caps_at_100() {
        assert_eq!(
            clamp_quality(None, HEIF_DEFAULT_QUALITY),
            HEIF_DEFAULT_QUALITY
        );
        assert_eq!(clamp_quality(Some(77), HEIF_DEFAULT_QUALITY), 77);
        assert_eq!(clamp_quality(Some(255), HEIF_DEFAULT_QUALITY), 100);
    }

    #[test]
    fn resolved_bit_depth_matches_sample_type_and_alpha_defaults() {
        assert_eq!(resolved_bit_depth(&SaveOptions::default(), false, false), 8);
        assert_eq!(resolved_bit_depth(&SaveOptions::default(), true, false), 12);
        assert_eq!(resolved_bit_depth(&SaveOptions::default(), true, true), 16);
        assert_eq!(
            resolved_bit_depth(
                &SaveOptions::default().with_heif_bit_depth(HeifBitDepth::Ten),
                true,
                true,
            ),
            10
        );
    }

    #[test]
    fn resolved_subsampling_uses_lossless_override() {
        assert_eq!(
            resolved_subsampling(&SaveOptions::default()),
            HeifSubsampling::Auto
        );
        assert_eq!(
            resolved_subsampling(
                &SaveOptions::default().with_heif_subsampling(HeifSubsampling::Subsample420)
            ),
            HeifSubsampling::Subsample420
        );
        assert_eq!(
            resolved_subsampling(
                &SaveOptions::default()
                    .lossless()
                    .with_heif_subsampling(HeifSubsampling::Subsample420)
            ),
            HeifSubsampling::Subsample444
        );
    }

    #[test]
    fn decoded_chroma_tracks_bit_depth_and_alpha() {
        assert_eq!(decoded_chroma(8, false), RgbChroma::Rgb);
        assert_eq!(decoded_chroma(8, true), RgbChroma::Rgba);
        assert_eq!(decoded_chroma(10, false), RgbChroma::HdrRgbBe);
        assert_eq!(decoded_chroma(10, true), RgbChroma::HdrRgbaBe);
    }

    #[test]
    fn normalize_u16_samples_for_bit_depth_left_aligns_samples() {
        let mut samples = vec![0x0001_u16, 0x00AB, 0x03FF];
        normalize_u16_samples_for_bit_depth(&mut samples, 10);
        assert_eq!(samples, vec![0x0040, 0x2AC0, 0xFFC0]);
    }

    #[test]
    fn metadata_to_write_preserves_or_strips_metadata() {
        let metadata = ImageMetadata {
            exif: Some(exif_blob(6)),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF>heif</rdf:RDF></x:xmpmeta>"#.to_vec()),
            icc_profile: Some(vec![0, 1, 2, 3]),
            ..ImageMetadata::default()
        };

        let preserved = metadata_to_write(&metadata, &SaveOptions::default());
        assert_eq!(preserved.exif, metadata.exif.as_deref());
        assert_eq!(preserved.xmp, metadata.xmp.as_deref());
        assert_eq!(preserved.icc_profile, metadata.icc_profile.as_deref());

        let stripped = metadata_to_write(&metadata, &SaveOptions::default().strip_metadata());
        assert!(stripped.exif.is_none());
        assert!(stripped.xmp.is_none());
        assert!(stripped.icc_profile.is_none());
    }

    #[test]
    fn probe_reports_fixture_dimensions() {
        let codec = HeifCodec;
        let (width, height, bands) = codec.probe(BENCH_512_HEIF).unwrap();

        assert_eq!((width, height, bands), (512, 512, 3));
    }

    #[test]
    fn decode_u16_fixture_expands_u8_samples_to_high_byte() {
        let codec = HeifCodec;
        let decoded_u8 = codec.decode::<U8>(BENCH_512_HEIF).unwrap();
        let decoded_u16 = codec.decode::<U16>(BENCH_512_HEIF).unwrap();

        assert_eq!(decoded_u16.width(), decoded_u8.width());
        assert_eq!(decoded_u16.height(), decoded_u8.height());
        assert_eq!(decoded_u16.bands(), decoded_u8.bands());
        for (&sample_u8, &sample_u16) in decoded_u8.pixels().iter().zip(decoded_u16.pixels()) {
            assert_eq!(sample_u16 >> 8, u16::from(sample_u8));
        }
    }

    #[test]
    fn decode_rejects_unsupported_band_format() {
        let codec = HeifCodec;
        let err = codec.decode::<F32>(BENCH_512_HEIF).unwrap_err();

        assert!(
            matches!(err, ViprsError::Codec(message) if message.contains("unsupported format"))
        );
    }

    #[test]
    fn decode_with_multi_page_request_returns_animation_frames() {
        if !heif_sequence_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let encoded = two_page_heif(1);

        let default_decoded = codec.decode::<U8>(&encoded).unwrap();
        let page0 = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(0))
            .unwrap();
        let decoded = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(0).with_n(2))
            .unwrap();

        assert_eq!(
            default_decoded.pixels(),
            decoded.animation_frames().unwrap()[1].image().pixels()
        );
        assert_eq!(page0.metadata().n_pages, Some(2));
        assert_eq!(decoded.metadata().n_pages, Some(2));
        assert_eq!(decoded.metadata().page_height, Some(1));

        let frames = decoded
            .animation_frames()
            .expect("multi-page HEIF should expose animation frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].delay_ms(), 0);
        assert_eq!(frames[1].delay_ms(), 0);
        assert_eq!(frames[0].disposal(), FrameDisposal::Keep);
        assert_eq!(frames[1].disposal(), FrameDisposal::Keep);
        assert_eq!(frames[0].image().pixels(), page0.pixels());
        assert_eq!(frames[1].image().pixels(), default_decoded.pixels());
    }

    #[test]
    fn encode_rejects_unsupported_band_count_for_u8_images() {
        let codec = HeifCodec;
        let image = Image::<U8>::from_buffer(1, 1, 2, vec![10, 20]).unwrap();
        let err = codec.encode::<U8>(&image).unwrap_err();

        assert!(
            matches!(err, ViprsError::Codec(message) if message.contains("unsupported band count 2"))
        );
    }

    #[test]
    fn encode_rejects_unsupported_band_count_for_u16_images() {
        let codec = HeifCodec;
        let image = Image::<U16>::from_buffer(1, 1, 2, vec![10, 20]).unwrap();
        let err = codec.encode::<U16>(&image).unwrap_err();

        assert!(
            matches!(err, ViprsError::Codec(message) if message.contains("unsupported band count 2"))
        );
    }

    #[test]
    fn encode_rejects_unsupported_band_format() {
        let codec = HeifCodec;
        let image = Image::<F32>::from_buffer(1, 1, 3, vec![0.0, 0.5, 1.0]).unwrap();
        let err = codec.encode::<F32>(&image).unwrap_err();

        assert!(
            matches!(err, ViprsError::Codec(message) if message.contains("unsupported format"))
        );
    }

    #[test]
    fn configured_heif_compression_reports_available_and_unavailable_encoders() {
        let lib_heif = shared_libheif("heif").unwrap();
        let available = available_encoders(lib_heif);
        assert!(!available.is_empty());

        let auto = configured_heif_compression(lib_heif, &SaveOptions::default()).unwrap();
        assert!(matches!(
            auto,
            CompressionFormat::Hevc | CompressionFormat::Av1
        ));

        let err = configured_heif_compression(
            lib_heif,
            &SaveOptions::default().with_heif_compression(HeifCompression::Jpeg),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Codec(message)
                if message.contains("requested Jpeg encoder is unavailable")
                    && message.contains("available encoders:")
        ));
    }

    #[test]
    fn apply_orientation_covers_all_heif_exif_transforms() {
        let pixels = vec![1_u8, 2, 3, 4];
        let cases = [
            (2, 2, 2, vec![2_u8, 1, 4, 3]),
            (3, 2, 2, vec![4_u8, 3, 2, 1]),
            (4, 2, 2, vec![3_u8, 4, 1, 2]),
            (5, 2, 2, vec![1_u8, 3, 2, 4]),
            (6, 2, 2, vec![3_u8, 1, 4, 2]),
            (7, 2, 2, vec![4_u8, 2, 3, 1]),
            (8, 2, 2, vec![2_u8, 4, 1, 3]),
        ];

        for (orientation, expected_width, expected_height, expected_pixels) in cases {
            let (width, height, oriented) = apply_orientation(&pixels, 2, 2, 1, orientation);
            assert_eq!((width, height), (expected_width, expected_height));
            assert_eq!(oriented, expected_pixels);
        }
    }

    #[test]
    fn grayscale_u8_encode_expands_to_rgb_channels() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let image = Image::<U8>::from_buffer(2, 2, 1, vec![0, 64, 128, 255]).unwrap();
        let encoded = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().lossless())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.bands(), 3);
        for pixel in decoded.pixels().chunks_exact(3) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }
    }

    #[test]
    fn grayscale_u8_ten_bit_round_trip_can_decode_to_u8_and_u16() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let image = Image::<U8>::from_buffer(2, 1, 1, vec![16, 240]).unwrap();
        let encoded = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Ten),
            )
            .unwrap();
        let decoded_u8 = codec.decode::<U8>(&encoded).unwrap();
        let decoded_u16 = codec.decode::<U16>(&encoded).unwrap();

        assert_eq!(decoded_u8.bands(), 3);
        assert_eq!(decoded_u16.bands(), 3);
        for (pixel_u8, pixel_u16) in decoded_u8
            .pixels()
            .chunks_exact(3)
            .zip(decoded_u16.pixels().chunks_exact(3))
        {
            assert_eq!(pixel_u8[0], pixel_u8[1]);
            assert_eq!(pixel_u8[1], pixel_u8[2]);
            assert_eq!(pixel_u16[0], pixel_u16[1]);
            assert_eq!(pixel_u16[1], pixel_u16[2]);
            assert_eq!(pixel_u16[0] >> 8, u16::from(pixel_u8[0]));
        }
    }

    #[test]
    fn grayscale_u16_encode_expands_to_rgb_for_eight_and_twelve_bit_paths() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let image = Image::<U16>::from_buffer(2, 1, 1, vec![0x1234, 0xFEDC]).unwrap();

        let encoded_8 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Eight),
            )
            .unwrap();
        let decoded_8 = codec.decode::<U8>(&encoded_8).unwrap();
        for pixel in decoded_8.pixels().chunks_exact(3) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }

        let encoded_12 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Twelve),
            )
            .unwrap();
        let decoded_12 = codec.decode::<U16>(&encoded_12).unwrap();
        for pixel in decoded_12.pixels().chunks_exact(3) {
            assert_eq!(pixel[0], pixel[1]);
            assert_eq!(pixel[1], pixel[2]);
        }
    }

    #[test]
    fn round_trip_u8_rgb_solid_colour() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let pixels: Vec<u8> = [32u8, 160, 224].repeat(4 * 4);
        let original = Image::<U8>::from_buffer(4, 4, 3, pixels).unwrap();

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().with_quality(100))
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
        assert_eq!(decoded.bands(), 3);

        for (index, (&orig, &decoded_sample)) in original
            .pixels()
            .iter()
            .zip(decoded.pixels().iter())
            .enumerate()
        {
            let diff = (i32::from(orig) - i32::from(decoded_sample)).abs();
            assert!(
                diff <= 16,
                "pixel sample {index}: original={orig}, decoded={decoded_sample}, diff={diff} > tolerance=16"
            );
        }
    }

    #[test]
    fn round_trip_u16_rgb_10bit_within_tolerance() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let pixels: Vec<u16> = [96u16 << 6, 480u16 << 6, 860u16 << 6].repeat(4 * 4);
        let original = Image::<U16>::from_buffer(4, 4, 3, pixels).unwrap();

        let encoded = codec
            .encode_with_options::<U16>(
                &original,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Ten),
            )
            .unwrap();
        let decoded = codec.decode::<U16>(&encoded).unwrap();

        assert_eq!(decoded.width(), 4);
        assert_eq!(decoded.height(), 4);
        assert_eq!(decoded.bands(), 3);

        for (index, (&orig, &decoded_sample)) in original
            .pixels()
            .iter()
            .zip(decoded.pixels().iter())
            .enumerate()
        {
            let orig_10 = i32::from(orig >> 6);
            let decoded_10 = i32::from(decoded_sample >> 6);
            let diff = (orig_10 - decoded_10).abs();
            assert!(
                diff <= 16,
                "pixel sample {index}: original_10={orig_10}, decoded_10={decoded_10}, diff={diff} > tolerance=16"
            );
        }
    }

    #[test]
    fn lossless_round_trip_u8_rgb_is_exact() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let original = rgb_u8_gradient(16, 16);

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().lossless())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.width(), original.width());
        assert_eq!(decoded.height(), original.height());
        assert_eq!(decoded.bands(), original.bands());
        for (&decoded_sample, &original_sample) in decoded.pixels().iter().zip(original.pixels()) {
            assert!((i16::from(decoded_sample) - i16::from(original_sample)).abs() <= 1);
        }
    }

    #[test]
    fn subsampling_changes_heif_bitstream() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let image = rgb_u8_gradient(64, 64);

        let subsampled = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default()
                    .with_quality(75)
                    .with_heif_subsampling(HeifSubsampling::Subsample420),
            )
            .unwrap();
        let full_chroma = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default()
                    .with_quality(75)
                    .with_heif_subsampling(HeifSubsampling::Subsample444),
            )
            .unwrap();

        assert_ne!(subsampled, full_chroma);
        assert_ne!(subsampled.len(), full_chroma.len());
    }

    #[test]
    fn bit_depth_changes_heif_output_size() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let image = rgb_u16_gradient(32, 32);

        let encoded_8 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Eight),
            )
            .unwrap();
        let encoded_10 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Ten),
            )
            .unwrap();
        let encoded_12 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .lossless()
                    .with_heif_bit_depth(HeifBitDepth::Twelve),
            )
            .unwrap();

        let handle_8 = HeifContext::read_from_bytes(&encoded_8)
            .unwrap()
            .primary_image_handle()
            .unwrap();
        let handle_10 = HeifContext::read_from_bytes(&encoded_10)
            .unwrap()
            .primary_image_handle()
            .unwrap();
        let handle_12 = HeifContext::read_from_bytes(&encoded_12)
            .unwrap()
            .primary_image_handle()
            .unwrap();

        assert_eq!(handle_8.luma_bits_per_pixel(), 8);
        assert_eq!(handle_10.luma_bits_per_pixel(), 10);
        assert_eq!(handle_12.luma_bits_per_pixel(), 12);
        assert_ne!(encoded_8.len(), encoded_10.len());
        assert_ne!(encoded_10.len(), encoded_12.len());
    }

    #[test]
    fn lossless_round_trip_u16_rgba_preserves_16bit_samples() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let original = rgba_u16_gradient(16, 16);

        match codec.encode_with_options::<U16>(&original, &SaveOptions::default().lossless()) {
            Ok(encoded) => {
                let handle = HeifContext::read_from_bytes(&encoded)
                    .unwrap()
                    .primary_image_handle()
                    .unwrap();
                assert_eq!(handle.luma_bits_per_pixel(), 16);
                assert!(handle.has_alpha_channel());

                let decoded = codec.decode::<U16>(&encoded).unwrap();

                assert_eq!(decoded.width(), original.width());
                assert_eq!(decoded.height(), original.height());
                assert_eq!(decoded.bands(), original.bands());
                assert_eq!(decoded.pixels(), original.pixels());
            }
            Err(ViprsError::Codec(message)) => {
                assert_eq!(
                    message,
                    "heif: linked libheif encoder/container does not support 16-bit interleaved RGBA"
                );
            }
            Err(other) => panic!("unexpected HEIF encode error: {other}"),
        }
    }

    #[test]
    fn decode_applies_heif_exif_orientation_and_normalizes_metadata() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let original = rgb_u8_gradient(3, 2).with_metadata(ImageMetadata {
            exif: Some(exif_blob(6)),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF>heif</rdf:RDF></x:xmpmeta>"#.to_vec()),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().lossless())
            .unwrap();
        let stored = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().no_rotate())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();
        let stored_orientation = stored.metadata().orientation.unwrap_or(1);

        assert!(stored.metadata().exif.is_some());
        assert!(stored.metadata().orientation.is_some());
        assert_eq!(decoded.metadata().xmp, original.metadata().xmp);
        assert_eq!(decoded.metadata().orientation, Some(1));
        assert_eq!(
            exif_orientation(decoded.metadata().exif.as_deref().unwrap_or(&[])),
            None
        );

        let (expected_width, expected_height, expected_pixels) = apply_orientation(
            stored.pixels(),
            stored.width(),
            stored.height(),
            stored.bands(),
            stored_orientation,
        );
        assert_eq!(decoded.width(), expected_width);
        assert_eq!(decoded.height(), expected_height);
        assert_eq!(decoded.pixels(), expected_pixels.as_slice());
    }

    #[test]
    fn round_trip_preserves_heif_icc_profile() {
        if !heif_encoder_available() {
            return;
        }

        // Minimal 4-byte fake ICC profile. Real profiles start with a 128-byte
        // header, but libheif stores raw bytes without validation — any non-empty
        // blob is preserved faithfully.
        let icc = vec![0x00u8, 0x00, 0x04, 0x00];
        let codec = HeifCodec;
        let original = rgb_u8_gradient(8, 8).with_metadata(ImageMetadata {
            icc_profile: Some(icc.clone()),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().lossless())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.metadata().icc_profile, Some(icc));
    }

    #[test]
    fn strip_metadata_omits_heif_icc_exif_xmp() {
        if !heif_encoder_available() {
            return;
        }

        let codec = HeifCodec;
        let original = rgb_u8_gradient(8, 8).with_metadata(ImageMetadata {
            exif: Some(exif_blob(1)),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF>heif-strip</rdf:RDF></x:xmpmeta>"#.to_vec()),
            icc_profile: Some(vec![0x00u8, 0x00, 0x04, 0x00]),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(
                &original,
                &SaveOptions::default().lossless().strip_metadata(),
            )
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert!(
            decoded.metadata().exif.is_none(),
            "strip_metadata should omit EXIF"
        );
        assert!(
            decoded.metadata().xmp.is_none(),
            "strip_metadata should omit XMP"
        );
        assert!(
            decoded.metadata().icc_profile.is_none(),
            "strip_metadata should omit ICC profile"
        );
    }

    #[test]
    fn u8_decode_peak_live_bytes_stay_near_output_buffer() {
        let decoded = HeifCodec.decode::<U8>(BENCH_2048_HEIF).unwrap();
        let output_bytes = decoded.pixels().len() as u64;
        let stats = crate::test_support::run_alloc_stats_child(U8_ALLOC_CHILD_TEST, U8_ALLOC_ENV);

        assert!(
            stats.peak_live_bytes <= output_bytes * 2,
            "HEIF U8 decode should stay near one output buffer: output_bytes={output_bytes}, stats={stats:?}"
        );
    }

    #[test]
    fn u8_decode_peak_live_bytes_stay_near_output_buffer_child() {
        if !crate::test_support::should_run_alloc_stats_child(U8_ALLOC_ENV) {
            return;
        }

        crate::test_support::emit_alloc_stats(heif_u8_decode_alloc_stats());
    }

    #[test]
    fn format_name_is_heif() {
        let codec = HeifCodec;
        assert_eq!(<HeifCodec as ImageDecoder>::format_name(&codec), "heif");
        assert_eq!(<HeifCodec as ImageEncoder>::format_name(&codec), "heif");
    }

    #[test]
    fn decode_empty_slice_returns_codec_error() {
        let codec = HeifCodec;
        let result = codec.decode::<U8>(&[]);
        assert!(matches!(result, Err(ViprsError::Codec(_))));
    }
}
