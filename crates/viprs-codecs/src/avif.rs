//! AVIF codec — decode via `libheif-rs`, encode via `ravif` (lossy U8) or
//! `libheif-rs` (U16 / 10-bit, plus optional U8 lossless when the `heif`
//! feature is enabled).
//!
//! Lossy U8 encode stays pure Rust via `ravif`. `ravif`/`rav1e` still do not
//! implement true AV1 lossless, so AVIF lossless for `U8` images is only
//! attempted through libheif when the `heif` feature is enabled, and even then
//! availability depends on the linked AV1 encoder. U16 encode uses libheif so
//! 10-bit AVIF output is available behind the existing AVIF feature.
//! Page sequences decode into [`AnimationFrame`]s for animated thumbnail flows.

use super::heif_support::{
    HeifWriteMetadata, checked_interleaved_byte_count, checked_interleaved_row_bytes,
    checked_interleaved_sample_count, encode_interleaved, normalize_decoded_image, read_metadata,
    shared_libheif,
};

use libheif_rs::{ColorSpace, CompressionFormat, HeifContext, ImageHandle, ItemId, RgbChroma};
use ravif::{BitDepth, ColorModel, Encoder as RavifEncoder, Img, RGB8, RGBA8};
use viprs_core::codec_options::{HeifBitDepth, HeifSubsampling, LoadOptions, SaveOptions};
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId, U8, U16};
use viprs_core::image::{AnimationFrame, FrameDisposal, ImageMetadata, InMemoryImage};
use viprs_ports::codec::{ImageDecoder, ImageEncoder};

/// AVIF codec: implements both [`ImageDecoder`] and [`ImageEncoder`].
pub struct AvifCodec;

const AVIF_DEFAULT_QUALITY: u8 = 50;
const AVIF_DEFAULT_SPEED: u8 = 9;
const AVIF_DECODE_PLANE_TOO_LARGE: &str = "avif decode plane exceeds addressable memory";
const AVIF_ENCODE_BUFFER_TOO_LARGE: &str = "avif encode buffer exceeds addressable memory";

#[inline]
fn clamp_quality(quality: Option<u8>, default: u8) -> u8 {
    quality.unwrap_or(default).min(100)
}

fn normalize_u16_samples(samples: &mut [u16], bit_depth: u8) {
    let shift = 16u8.saturating_sub(bit_depth);
    if shift > 0 {
        for sample in samples {
            *sample <<= shift;
        }
    }
}

#[inline]
fn avif_speed(effort: Option<u8>) -> u8 {
    effort.map_or(AVIF_DEFAULT_SPEED, |effort| {
        10u8.saturating_sub(effort.min(9))
    })
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

struct AvifPageSelection {
    item_ids: Vec<ItemId>,
    first_page: usize,
    page_count: usize,
    total_pages: u32,
}

fn avif_page_metadata(
    mut metadata: ImageMetadata,
    page_height: u32,
    total_pages: u32,
) -> ImageMetadata {
    metadata.n_pages = Some(total_pages);
    metadata.page_height = (total_pages > 1).then_some(page_height);
    metadata
}

#[inline]
fn checked_avif_decode_sample_count(
    width: u32,
    height: u32,
    bands: u32,
) -> Result<usize, ViprsError> {
    checked_interleaved_sample_count(width, height, bands, AVIF_DECODE_PLANE_TOO_LARGE)
}

#[inline]
fn checked_avif_row_bytes(
    width: u32,
    bands: u32,
    bytes_per_sample: usize,
) -> Result<usize, ViprsError> {
    checked_interleaved_row_bytes(width, bands, bytes_per_sample, AVIF_DECODE_PLANE_TOO_LARGE)
}

fn avif_top_level_image_ids(ctx: &HeifContext) -> Result<Vec<ItemId>, ViprsError> {
    let total_pages = ctx.number_of_top_level_images();
    if total_pages == 0 {
        return Err(ViprsError::Codec(
            "avif: container does not contain any top-level images".into(),
        ));
    }

    let mut item_ids = vec![0; total_pages];
    let listed = ctx.top_level_image_ids(&mut item_ids);
    item_ids.truncate(listed);
    if item_ids.is_empty() {
        return Err(ViprsError::Codec(
            "avif: failed to enumerate top-level images".into(),
        ));
    }

    Ok(item_ids)
}

fn avif_primary_page(ctx: &HeifContext, item_ids: &[ItemId]) -> Result<usize, ViprsError> {
    for (index, &item_id) in item_ids.iter().enumerate() {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("avif: image_handle({item_id}): {e}")))?;
        if handle.is_primary() {
            return Ok(index);
        }
    }

    Err(ViprsError::Codec(
        "avif: failed to locate primary image among top-level images".into(),
    ))
}

fn avif_selected_page_range(
    opts: &LoadOptions,
    total_pages: usize,
    primary_page: usize,
) -> Result<(usize, usize), ViprsError> {
    let page = if opts.page.is_none() && opts.n.is_none() {
        primary_page
    } else {
        usize::try_from(opts.page.unwrap_or(0))
            .map_err(|_| ViprsError::Codec("avif: page index exceeds platform limits".into()))?
    };
    let requested = opts.n.unwrap_or(1);
    let pages_to_decode = if requested == -1 {
        total_pages.saturating_sub(page)
    } else if requested > 0 {
        usize::try_from(requested)
            .map_err(|_| ViprsError::Codec(format!("avif: invalid n={requested}")))?
    } else {
        0
    };

    if page >= total_pages || pages_to_decode == 0 || page + pages_to_decode > total_pages {
        return Err(ViprsError::Codec(format!(
            "avif: bad page number (page={page}, n={requested}, total_pages={total_pages})"
        )));
    }

    Ok((page, pages_to_decode))
}

fn select_avif_pages(
    ctx: &HeifContext,
    opts: &LoadOptions,
) -> Result<AvifPageSelection, ViprsError> {
    let item_ids = avif_top_level_image_ids(ctx)?;
    let primary_page = avif_primary_page(ctx, &item_ids)?;
    let (first_page, page_count) = avif_selected_page_range(opts, item_ids.len(), primary_page)?;
    let total_pages = item_ids.len() as u32;

    Ok(AvifPageSelection {
        item_ids,
        first_page,
        page_count,
        total_pages,
    })
}

fn decode_u16_page(
    handle: &ImageHandle,
    opts: &LoadOptions,
    total_pages: u32,
) -> Result<InMemoryImage<U16>, ViprsError> {
    let metadata = read_metadata("avif", handle)?;
    let width = handle.width();
    let height = handle.height();
    let has_alpha = handle.has_alpha_channel();
    let bands = if has_alpha { 4 } else { 3 };
    let bit_depth = handle
        .luma_bits_per_pixel()
        .max(handle.chroma_bits_per_pixel())
        .max(8);

    let lib_heif = shared_libheif("avif")?;
    let decoded = lib_heif
        .decode(
            handle,
            ColorSpace::Rgb(decoded_chroma(bit_depth, has_alpha)),
            None,
        )
        .map_err(|e| ViprsError::Codec(format!("avif: decode: {e}")))?;
    let plane = decoded
        .planes()
        .interleaved
        .ok_or_else(|| ViprsError::Codec("avif: no interleaved plane".into()))?;

    let sample_count = checked_avif_decode_sample_count(width, height, bands)?;
    let mut samples = Vec::with_capacity(sample_count);
    let row_bytes = checked_avif_row_bytes(width, bands, if bit_depth > 8 { 2 } else { 1 })?;
    for row in 0..height as usize {
        let row_start = row * plane.stride;
        let row_end = row_start + row_bytes;
        let row_data = &plane.data[row_start..row_end];
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
    normalize_u16_samples(&mut samples, bit_depth);
    let image = InMemoryImage::from_buffer(width, height, bands, samples)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .with_metadata(avif_page_metadata(metadata, height, total_pages));
    normalize_decoded_image(image, opts.no_rotate, "avif")
}

fn decode_u8_page(
    handle: &ImageHandle,
    opts: &LoadOptions,
    total_pages: u32,
) -> Result<InMemoryImage<U8>, ViprsError> {
    let metadata = read_metadata("avif", handle)?;
    let width = handle.width();
    let height = handle.height();
    let has_alpha = handle.has_alpha_channel();
    let bands = if has_alpha { 4 } else { 3 };

    let lib_heif = shared_libheif("avif")?;
    let decoded = lib_heif
        .decode(handle, ColorSpace::Rgb(decoded_chroma(8, has_alpha)), None)
        .map_err(|e| ViprsError::Codec(format!("avif: decode: {e}")))?;
    let plane = decoded
        .planes()
        .interleaved
        .ok_or_else(|| ViprsError::Codec("avif: no interleaved plane".into()))?;

    let sample_count = checked_avif_decode_sample_count(width, height, bands)?;
    let row_bytes = checked_avif_row_bytes(width, bands, 1)?;
    let mut samples = Vec::with_capacity(sample_count);
    if plane.stride == row_bytes {
        samples.extend_from_slice(&plane.data[..sample_count]);
    } else {
        for row in 0..height as usize {
            let row_start = row * plane.stride;
            let row_end = row_start + row_bytes;
            samples.extend_from_slice(&plane.data[row_start..row_end]);
        }
    }

    let image = InMemoryImage::from_buffer(width, height, bands, samples)
        .map_err(|e| ViprsError::Codec(e.to_string()))?
        .with_metadata(avif_page_metadata(metadata, height, total_pages));
    normalize_decoded_image(image, opts.no_rotate, "avif")
}

fn decode_u16_samples(src: &[u8], opts: &LoadOptions) -> Result<InMemoryImage<U16>, ViprsError> {
    let ctx = HeifContext::read_from_bytes(src)
        .map_err(|e| ViprsError::Codec(format!("avif: read_from_bytes: {e}")))?;
    let selection = select_avif_pages(&ctx, opts)?;
    if selection.page_count == 1 {
        let item_id = selection.item_ids[selection.first_page];
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("avif: image_handle({item_id}): {e}")))?;
        return decode_u16_page(&handle, opts, selection.total_pages);
    }

    let mut frames = Vec::with_capacity(selection.page_count);
    for &item_id in
        &selection.item_ids[selection.first_page..selection.first_page + selection.page_count]
    {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("avif: image_handle({item_id}): {e}")))?;
        let frame = decode_u16_page(&handle, opts, selection.total_pages)?;
        frames.push(AnimationFrame::new(frame, 0, FrameDisposal::Keep));
    }

    let page_height = frames[0].image().height();
    let mut image =
        InMemoryImage::from_frames(frames).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut metadata = image.metadata().clone();
    metadata.n_pages = Some(selection.total_pages);
    metadata.page_height = (selection.total_pages > 1).then_some(page_height);
    image = image.with_metadata(metadata);
    Ok(image)
}

fn decode_u8_samples(src: &[u8], opts: &LoadOptions) -> Result<InMemoryImage<U8>, ViprsError> {
    let ctx = HeifContext::read_from_bytes(src)
        .map_err(|e| ViprsError::Codec(format!("avif: read_from_bytes: {e}")))?;
    let selection = select_avif_pages(&ctx, opts)?;
    if selection.page_count == 1 {
        let item_id = selection.item_ids[selection.first_page];
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("avif: image_handle({item_id}): {e}")))?;
        return decode_u8_page(&handle, opts, selection.total_pages);
    }

    let mut frames = Vec::with_capacity(selection.page_count);
    for &item_id in
        &selection.item_ids[selection.first_page..selection.first_page + selection.page_count]
    {
        let handle = ctx
            .image_handle(item_id)
            .map_err(|e| ViprsError::Codec(format!("avif: image_handle({item_id}): {e}")))?;
        let frame = decode_u8_page(&handle, opts, selection.total_pages)?;
        frames.push(AnimationFrame::new(frame, 0, FrameDisposal::Keep));
    }

    let page_height = frames[0].image().height();
    let mut image =
        InMemoryImage::from_frames(frames).map_err(|e| ViprsError::Codec(e.to_string()))?;
    let mut metadata = image.metadata().clone();
    metadata.n_pages = Some(selection.total_pages);
    metadata.page_height = (selection.total_pages > 1).then_some(page_height);
    image = image.with_metadata(metadata);
    Ok(image)
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

fn cast_decoded_frame<S: BandFormat, D: BandFormat>(
    image: InMemoryImage<S>,
    context: &str,
) -> Result<InMemoryImage<D>, ViprsError> {
    let width = image.width();
    let height = image.height();
    let bands = image.bands();
    let metadata = image.metadata().clone();
    let samples = bytemuck::allocation::try_cast_vec::<S::Sample, D::Sample>(image.into_buffer())
        .map_err(|(e, _)| ViprsError::Codec(format!("{context}: cast error: {e:?}")))?;
    InMemoryImage::from_buffer(width, height, bands, samples)
        .map(|image| image.with_metadata(metadata))
        .map_err(|e| ViprsError::Codec(e.to_string()))
}

fn cast_decoded_image<S: BandFormat, D: BandFormat>(
    image: InMemoryImage<S>,
    context: &str,
) -> Result<InMemoryImage<D>, ViprsError>
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
fn resolved_bit_depth(opts: &SaveOptions, is_u16: bool) -> u8 {
    opts.heif_bit_depth
        .unwrap_or(if is_u16 {
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

#[inline]
fn needs_libheif_u8_encode_path(metadata: &ImageMetadata, opts: &SaveOptions) -> bool {
    opts.heif_subsampling.is_some()
        || opts.heif_bit_depth.is_some()
        || (opts.strip_metadata != Some(true)
            && (metadata.exif.is_some()
                || metadata.xmp.is_some()
                || metadata.icc_profile.is_some()))
}

#[cfg(feature = "heif")]
fn avif_u8_lossless_unavailable_error(source: &ViprsError) -> ViprsError {
    ViprsError::Codec(format!(
        "avif: true lossless U8 encoding is unavailable in this build: ravif/rav1e do not implement AV1 lossless, and the linked libheif AV1 encoder could not complete the lossless path ({source})"
    ))
}

#[cfg(not(feature = "heif"))]
fn avif_u8_lossless_unavailable_error() -> ViprsError {
    ViprsError::Codec(
        "avif: true lossless U8 encoding is unavailable in this build: ravif/rav1e do not implement AV1 lossless, and the libheif-backed fallback is only enabled with the `heif` feature".into(),
    )
}

fn encode_u8_with_ravif(
    width: u32,
    height: u32,
    bands: u32,
    pixels: &[u8],
    opts: &SaveOptions,
) -> Result<Vec<u8>, ViprsError> {
    let quality = clamp_quality(opts.quality, AVIF_DEFAULT_QUALITY) as f32;
    let lossless = opts.lossless == Some(true);
    let speed = if lossless && opts.effort.is_none() {
        1
    } else {
        avif_speed(opts.effort)
    };
    let width = width as usize;
    let height = height as usize;

    let enc = if lossless {
        // ravif maps quality=100 to rav1e quantizer 0 and this branch already
        // rules out the other common AVIF rounding sources:
        // - ColorModel::RGB writes an identity-matrix RGB/GBR stream
        // - ravif hard-codes 4:4:4 chroma here
        // - BitDepth::Eight avoids any 8-bit -> 10-bit -> 8-bit expansion
        // The remaining ±2 drift is therefore in rav1e's q=0 coding path, not
        // in an RGB↔YUV conversion or bit-depth promotion. Keep this branch
        // aligned with libvips' matrix/subsampling choices and treat the
        // round-trip as near-lossless in tests until rav1e grows true AV1
        // lossless support.
        RavifEncoder::new()
            .with_quality(100.0)
            .with_alpha_quality(100.0)
            .with_internal_color_model(ColorModel::RGB)
            .with_bit_depth(BitDepth::Eight)
            .with_speed(speed)
    } else {
        RavifEncoder::new()
            .with_quality(quality)
            .with_alpha_quality(100.0)
            .with_bit_depth(BitDepth::Eight)
            .with_speed(speed)
    };

    let encoded = match bands {
        1 => {
            let rgb_pixels: Vec<RGB8> = pixels.iter().map(|&g| RGB8 { r: g, g, b: g }).collect();
            enc.encode_rgb(Img::new(rgb_pixels.as_slice(), width, height))
        }
        3 => {
            let rgb_pixels = bytemuck::cast_slice::<u8, RGB8>(pixels);
            enc.encode_rgb(Img::new(rgb_pixels, width, height))
        }
        4 => {
            let rgba_pixels = bytemuck::cast_slice::<u8, RGBA8>(pixels);
            enc.encode_rgba(Img::new(rgba_pixels, width, height))
        }
        n => {
            return Err(ViprsError::Codec(format!(
                "avif: unsupported band count {n} — only 1, 3, and 4 bands are supported"
            )));
        }
    }
    .map_err(|e| ViprsError::Codec(format!("avif encode: {e}")))?;

    Ok(encoded.avif_file)
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
                "avif: unsupported band count {n} — only 1, 3, and 4 bands are supported"
            )));
        }
    };
    let output_bands = if bands == 1 { 3 } else { bands };
    let bit_depth = resolved_bit_depth(opts, false);
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
            AVIF_ENCODE_BUFFER_TOO_LARGE,
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
                    "avif: unsupported band count after validation".into(),
                ));
            }
        }

        storage = expanded;
        storage.as_slice()
    };

    encode_interleaved(
        "avif",
        CompressionFormat::Av1,
        width,
        height,
        output_bands,
        bit_depth,
        pixel_bytes,
        opts.lossless == Some(true),
        clamp_quality(opts.quality, AVIF_DEFAULT_QUALITY),
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
                "avif: unsupported band count {n} — only 1, 3, and 4 bands are supported"
            )));
        }
    };
    let output_bands = if bands == 1 { 3 } else { bands };
    let bit_depth = resolved_bit_depth(opts, true);
    let bytes_per_sample = if bit_depth > 8 { 2 } else { 1 };
    let byte_count = checked_interleaved_byte_count(
        width,
        height,
        output_bands,
        bytes_per_sample,
        AVIF_ENCODE_BUFFER_TOO_LARGE,
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
                "avif: unsupported band count after validation".into(),
            ));
        }
    }

    encode_interleaved(
        "avif",
        CompressionFormat::Av1,
        width,
        height,
        output_bands,
        bit_depth,
        &pixel_bytes,
        opts.lossless == Some(true),
        clamp_quality(opts.quality, AVIF_DEFAULT_QUALITY),
        opts.effort,
        resolved_subsampling(opts),
        metadata_to_write(metadata, opts),
    )
}

impl ImageDecoder for AvifCodec {
    fn format_name(&self) -> &'static str {
        "avif"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        if header.len() < 12 {
            return false;
        }
        &header[4..8] == b"ftyp" && (&header[8..12] == b"avif" || &header[8..12] == b"avis")
    }

    fn decode<F: BandFormat>(&self, src: &[u8]) -> Result<InMemoryImage<F>, ViprsError> {
        self.decode_with_options(src, &LoadOptions::default())
    }

    fn decode_with_options<F: BandFormat>(
        &self,
        src: &[u8],
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
    {
        match F::ID {
            BandFormatId::U8 => {
                let image = decode_u8_samples(src, opts)?;
                cast_decoded_image::<U8, F>(image, "avif")
            }
            BandFormatId::U16 => {
                let image = decode_u16_samples(src, opts)?;
                cast_decoded_image::<U16, F>(image, "avif")
            }
            _ => Err(ViprsError::Codec(format!(
                "avif: unsupported format {:?} — only U8 and U16 are supported",
                F::ID
            ))),
        }
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        let ctx = HeifContext::read_from_bytes(src)
            .map_err(|e| ViprsError::Codec(format!("avif: probe: {e}")))?;
        let handle = ctx
            .primary_image_handle()
            .map_err(|e| ViprsError::Codec(format!("avif: probe handle: {e}")))?;
        let bands = if handle.has_alpha_channel() { 4 } else { 3 };
        Ok((handle.width(), handle.height(), bands))
    }
}

impl ImageEncoder for AvifCodec {
    fn format_name(&self) -> &'static str {
        "avif"
    }

    fn encode<F: BandFormat>(&self, image: &InMemoryImage<F>) -> Result<Vec<u8>, ViprsError> {
        self.encode_with_options(image, &SaveOptions::default())
    }

    fn encode_with_options<F: BandFormat>(
        &self,
        image: &InMemoryImage<F>,
        opts: &SaveOptions,
    ) -> Result<Vec<u8>, ViprsError>
    where
        Self: Sized,
    {
        match F::ID {
            BandFormatId::U8 => {
                let pixels = bytemuck::cast_slice::<F::Sample, u8>(image.pixels());
                if opts.lossless == Some(true) {
                    #[cfg(feature = "heif")]
                    {
                        return encode_u8_with_libheif(
                            image.width(),
                            image.height(),
                            image.bands(),
                            pixels,
                            image.metadata(),
                            opts,
                        )
                        .map_err(|source| avif_u8_lossless_unavailable_error(&source));
                    }

                    #[cfg(not(feature = "heif"))]
                    {
                        return Err(avif_u8_lossless_unavailable_error());
                    }
                }
                if needs_libheif_u8_encode_path(image.metadata(), opts) {
                    let _ = shared_libheif("avif")?;
                    encode_u8_with_libheif(
                        image.width(),
                        image.height(),
                        image.bands(),
                        pixels,
                        image.metadata(),
                        opts,
                    )
                } else {
                    encode_u8_with_ravif(image.width(), image.height(), image.bands(), pixels, opts)
                }
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
                "avif: unsupported format {:?} — only U8 and U16 are supported",
                F::ID
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libheif_rs::{Channel, EncoderQuality, Image as LibHeifImage, LibHeif};
    use proptest::prelude::*;
    use viprs_core::image::ImageMetadata;

    prop_compose! {
        fn rgb_u8_image()
            (width in 1u32..=32u32, height in 1u32..=32u32)
            (
                width in Just(width),
                height in Just(height),
                pixels in proptest::collection::vec(any::<u8>(), (width * height * 3) as usize),
            ) -> (u32, u32, Vec<u8>) {
                (width, height, pixels)
            }
    }

    fn av1_encoder_available() -> bool {
        shared_libheif("avif")
            .is_ok_and(|lib_heif| lib_heif.encoder_for_format(CompressionFormat::Av1).is_ok())
    }

    fn require_av1_encoder() {
        assert!(
            av1_encoder_available(),
            "requires libheif AV1 encoder support. Run this test in the HEIF/AVIF encoder-contract lane."
        );
    }

    fn rgb_u8_gradient(width: u32, height: u32) -> InMemoryImage<U8> {
        let pixels: Vec<u8> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        (x.wrapping_mul(13) + y.wrapping_mul(7)) as u8,
                        (x.wrapping_mul(3) + y.wrapping_mul(17)) as u8,
                        (x ^ y.wrapping_mul(9)) as u8,
                    ]
                })
            })
            .collect();
        InMemoryImage::<U8>::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn rgb_u16_gradient(width: u32, height: u32) -> InMemoryImage<U16> {
        let pixels: Vec<u16> = (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| {
                    [
                        ((x * 193 + y * 29) & 0xFFFF) as u16,
                        ((x * 389 + y * 97) & 0xFFFF) as u16,
                        ((x * 641 + y * 53) & 0xFFFF) as u16,
                    ]
                })
            })
            .collect();
        InMemoryImage::<U16>::from_buffer(width, height, 3, pixels).unwrap()
    }

    fn solid_rgb_u8(width: u32, height: u32, rgb: [u8; 3]) -> InMemoryImage<U8> {
        let pixels = rgb.repeat((width * height) as usize);
        InMemoryImage::<U8>::from_buffer(width, height, 3, pixels).unwrap()
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

    fn two_page_avif(primary_page: usize) -> Vec<u8> {
        let lib_heif = LibHeif::new();
        let mut context = HeifContext::new().unwrap();
        let mut encoder = lib_heif.encoder_for_format(CompressionFormat::Av1).unwrap();
        encoder.set_quality(EncoderQuality::Lossy(100)).unwrap();

        let first = solid_rgb_u8(1, 1, [255, 0, 0]);
        let second = solid_rgb_u8(1, 1, [0, 0, 255]);
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

    fn assert_rgb_within_tolerance(actual: &[u8], expected: [u8; 3], tolerance: u8) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual_sample, expected_sample)) in actual.iter().zip(expected).enumerate() {
            let diff = (i16::from(actual_sample) - i16::from(expected_sample)).abs();
            assert!(
                diff <= i16::from(tolerance),
                "channel {index}: actual={actual_sample}, expected={expected_sample}, diff={diff} > tolerance={tolerance}"
            );
        }
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
    fn sniff_recognises_avif_brand() {
        let codec = AvifCodec;
        let mut header = vec![0u8; 12];
        header[4..8].copy_from_slice(b"ftyp");
        header[8..12].copy_from_slice(b"avif");
        assert!(codec.sniff(&header));
    }

    #[test]
    fn sniff_rejects_jpeg() {
        let codec = AvifCodec;
        let header = [
            0xFF_u8, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        ];
        assert!(!codec.sniff(&header));
    }

    #[test]
    fn avif_decode_rejects_u32_max_dimensions() {
        let result = checked_avif_decode_sample_count(u32::MAX, u32::MAX, 4);
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
    fn avif_u8_encode_rejects_oversized_grayscale_expansion() {
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
    fn avif_u16_encode_rejects_oversized_byte_widening() {
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
    fn round_trip_u8_rgb_solid_colour() {
        let codec = AvifCodec;
        let pixels: Vec<u8> = [0u8, 150, 150].repeat(4 * 4);
        let original = InMemoryImage::<U8>::from_buffer(4, 4, 3, pixels).unwrap();

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().with_quality(90))
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
                diff <= 15,
                "pixel sample {index}: original={orig}, decoded={decoded_sample}, diff={diff} > tolerance=15"
            );
        }
    }

    #[test]
    fn decode_with_page_selects_non_primary_avif_image() {
        if !av1_encoder_available() {
            return;
        }

        let codec = AvifCodec;
        let encoded = two_page_avif(1);

        let default_decoded = codec.decode::<U8>(&encoded).unwrap();
        let page0 = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(0))
            .unwrap();
        let page1 = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(1))
            .unwrap();

        assert_rgb_within_tolerance(default_decoded.pixels(), [0, 0, 255], 1);
        assert_rgb_within_tolerance(page0.pixels(), [255, 0, 0], 1);
        assert_rgb_within_tolerance(page1.pixels(), [0, 0, 255], 1);
        assert_eq!(default_decoded.pixels(), page1.pixels());
        assert_eq!(page0.metadata().n_pages, Some(2));
        assert_eq!(page0.metadata().page_height, Some(1));
        assert_eq!(page1.metadata().n_pages, Some(2));
        assert_eq!(page1.metadata().page_height, Some(1));
    }

    #[test]
    fn decode_with_invalid_page_returns_clear_error() {
        if !av1_encoder_available() {
            return;
        }

        let codec = AvifCodec;
        let encoded = two_page_avif(0);
        let err = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(2))
            .unwrap_err();
        let message = err.to_string();

        assert!(message.contains("bad page number"), "{message}");
    }

    #[test]
    fn decode_with_multi_page_request_returns_animation_frames() {
        if !av1_encoder_available() {
            return;
        }

        let codec = AvifCodec;
        let encoded = two_page_avif(0);
        let decoded = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().with_page(0).with_n(2))
            .unwrap();

        assert_rgb_within_tolerance(decoded.pixels(), [255, 0, 0], 1);
        assert_eq!(decoded.metadata().n_pages, Some(2));
        assert_eq!(decoded.metadata().page_height, Some(1));

        let frames = decoded
            .animation_frames()
            .expect("multi-page AVIF should expose animation frames");
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].delay_ms(), 0);
        assert_eq!(frames[1].delay_ms(), 0);
        assert_eq!(frames[0].disposal(), viprs_core::image::FrameDisposal::Keep);
        assert_eq!(frames[1].disposal(), viprs_core::image::FrameDisposal::Keep);
        assert_rgb_within_tolerance(frames[0].image().pixels(), [255, 0, 0], 1);
        assert_rgb_within_tolerance(frames[1].image().pixels(), [0, 0, 255], 1);
    }

    #[test]
    fn default_quality_matches_libvips_q50() {
        let codec = AvifCodec;
        let image = rgb_u8_gradient(32, 32);

        let default_encoded = codec.encode::<U8>(&image).unwrap();
        let explicit_q50 = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().with_quality(50))
            .unwrap();

        assert_eq!(default_encoded, explicit_q50);
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn round_trip_u16_rgb_10bit_within_tolerance() {
        require_av1_encoder();

        let codec = AvifCodec;
        let pixels: Vec<u16> = [128u16 << 6, 512u16 << 6, 900u16 << 6].repeat(4 * 4);
        let original = InMemoryImage::<U16>::from_buffer(4, 4, 3, pixels).unwrap();

        let encoded = codec
            .encode_with_options::<U16>(
                &original,
                &SaveOptions::default()
                    .with_quality(100)
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
                diff <= 32,
                "pixel sample {index}: original_10={orig_10}, decoded_10={decoded_10}, diff={diff} > tolerance=32"
            );
        }
    }

    #[test]
    fn lossless_u8_uses_available_backend_honestly() {
        let codec = AvifCodec;
        let image = rgb_u8_gradient(32, 32);

        let quality_100 = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().with_quality(100))
            .unwrap();
        assert!(!quality_100.is_empty());

        #[cfg(feature = "heif")]
        if av1_encoder_available() {
            let lossless = codec
                .encode_with_options::<U8>(&image, &SaveOptions::default().lossless())
                .unwrap();
            let decoded = codec.decode::<U8>(&lossless).unwrap();

            assert_ne!(quality_100, lossless);
            for (index, (&orig, &decoded_sample)) in image
                .pixels()
                .iter()
                .zip(decoded.pixels().iter())
                .enumerate()
            {
                let diff = (i32::from(orig) - i32::from(decoded_sample)).abs();
                assert_eq!(
                    diff, 0,
                    "pixel sample {index}: original={orig}, decoded={decoded_sample}, diff={diff} > tolerance=0"
                );
            }
            return;
        }

        let lossless_error = codec
            .encode_with_options::<U8>(&image, &SaveOptions::default().lossless())
            .unwrap_err();

        assert!(
            lossless_error
                .to_string()
                .contains("true lossless U8 encoding is unavailable"),
            "unexpected error: {lossless_error}"
        );
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn subsampling_changes_avif_bitstream() {
        require_av1_encoder();

        let codec = AvifCodec;
        let image = rgb_u8_gradient(64, 64);

        let subsampled = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default()
                    .with_quality(80)
                    .with_heif_subsampling(HeifSubsampling::Subsample420),
            )
            .unwrap();
        let full_chroma = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default()
                    .with_quality(80)
                    .with_heif_subsampling(HeifSubsampling::Subsample444),
            )
            .unwrap();

        assert_ne!(subsampled, full_chroma);
        assert_ne!(subsampled.len(), full_chroma.len());
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn bit_depth_changes_avif_output_size() {
        require_av1_encoder();

        let codec = AvifCodec;
        let image = rgb_u16_gradient(32, 32);

        let encoded_8 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .with_quality(100)
                    .with_heif_bit_depth(HeifBitDepth::Eight),
            )
            .unwrap();
        let encoded_10 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .with_quality(100)
                    .with_heif_bit_depth(HeifBitDepth::Ten),
            )
            .unwrap();
        let encoded_12 = codec
            .encode_with_options::<U16>(
                &image,
                &SaveOptions::default()
                    .with_quality(100)
                    .with_heif_bit_depth(HeifBitDepth::Twelve),
            )
            .unwrap();

        let handle_8 = libheif_rs::HeifContext::read_from_bytes(&encoded_8)
            .unwrap()
            .primary_image_handle()
            .unwrap();
        let handle_10 = libheif_rs::HeifContext::read_from_bytes(&encoded_10)
            .unwrap()
            .primary_image_handle()
            .unwrap();
        let handle_12 = libheif_rs::HeifContext::read_from_bytes(&encoded_12)
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
    fn encode_effort_changes_avif_bitstream() {
        let codec = AvifCodec;
        let pixels: Vec<u8> = (0..32)
            .flat_map(|y| (0..32).flat_map(move |x| [x as u8, y as u8, (x ^ y) as u8]))
            .collect();
        let image = InMemoryImage::<U8>::from_buffer(32, 32, 3, pixels).unwrap();

        let fast = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default().with_quality(80).with_effort(0),
            )
            .unwrap();
        let thorough = codec
            .encode_with_options::<U8>(
                &image,
                &SaveOptions::default().with_quality(80).with_effort(9),
            )
            .unwrap();

        assert_ne!(fast, thorough);
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn decode_applies_avif_exif_orientation_and_normalizes_metadata() {
        require_av1_encoder();

        let codec = AvifCodec;
        let original = rgb_u8_gradient(3, 2).with_metadata(ImageMetadata {
            exif: Some(exif_blob(6)),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF>avif</rdf:RDF></x:xmpmeta>"#.to_vec()),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().lossless())
            .unwrap();
        let stored = codec
            .decode_with_options::<U8>(&encoded, &LoadOptions::default().no_rotate())
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(stored.metadata().orientation, Some(6));
        assert_eq!(
            exif_orientation(stored.metadata().exif.as_deref().unwrap_or(&[])),
            Some(6)
        );
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
            6,
        );
        assert_eq!(decoded.width(), expected_width);
        assert_eq!(decoded.height(), expected_height);
        assert_eq!(decoded.pixels(), expected_pixels.as_slice());
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn round_trip_preserves_avif_icc_profile() {
        require_av1_encoder();

        // Minimal fake ICC profile blob. libheif stores it verbatim.
        let icc = vec![0x00u8, 0x00, 0x04, 0x00];
        let codec = AvifCodec;
        let original = rgb_u8_gradient(8, 8).with_metadata(ImageMetadata {
            icc_profile: Some(icc.clone()),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(&original, &SaveOptions::default().with_quality(90))
            .unwrap();
        let decoded = codec.decode::<U8>(&encoded).unwrap();

        assert_eq!(decoded.metadata().icc_profile, Some(icc));
    }

    #[test]
    #[ignore = "requires libheif AV1 encoder support"]
    fn strip_metadata_omits_avif_icc_exif_xmp() {
        require_av1_encoder();

        let codec = AvifCodec;
        let original = rgb_u8_gradient(8, 8).with_metadata(ImageMetadata {
            exif: Some(exif_blob(1)),
            xmp: Some(br#"<x:xmpmeta><rdf:RDF>avif-strip</rdf:RDF></x:xmpmeta>"#.to_vec()),
            icc_profile: Some(vec![0x00u8, 0x00, 0x04, 0x00]),
            ..ImageMetadata::default()
        });

        let encoded = codec
            .encode_with_options::<U8>(
                &original,
                &SaveOptions::default().with_quality(90).strip_metadata(),
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

    proptest! {
        // AVIF encode/decode is CPU-intensive (~50ms native, much worse under
        // emulation). 16 cases is sufficient to exercise the rounding invariant
        // across varied image dimensions and pixel patterns.
        #![proptest_config(ProptestConfig::with_cases(16))]
        #[test]
        fn prop_near_lossless_round_trip_rgb_stays_within_ravif_rounding(
            (width, height, pixels) in rgb_u8_image(),
        ) {
            let codec = AvifCodec;
            let original = InMemoryImage::<U8>::from_buffer(width, height, 3, pixels).unwrap();

            let encoded = encode_u8_with_ravif(
                original.width(),
                original.height(),
                original.bands(),
                original.pixels(),
                &SaveOptions::default().lossless(),
            )
            .unwrap();
            let decoded = codec.decode::<U8>(&encoded).unwrap();

            prop_assert_eq!(decoded.width(), width);
            prop_assert_eq!(decoded.height(), height);
            prop_assert_eq!(decoded.bands(), 3);
            // ravif's lossless-looking branch already uses RGB identity-matrix
            // signalling, 4:4:4 chroma, and 8-bit storage, so the remaining ±2
            // drift comes from rav1e's q=0 path rather than color conversion or
            // bit-depth expansion.
            for (&orig, &decoded_sample) in original.pixels().iter().zip(decoded.pixels().iter()) {
                let diff = (i32::from(orig) - i32::from(decoded_sample)).abs();
                // ravif/rav1e's "lossless" RGB path is quantizer-0 4:4:4, but it
                // still introduces up to ±2 sample rounding on some inputs.
                prop_assert!(diff <= 2);
            }
        }
    }

    #[test]
    fn format_name_is_avif() {
        let codec = AvifCodec;
        assert_eq!(<AvifCodec as ImageDecoder>::format_name(&codec), "avif");
        assert_eq!(<AvifCodec as ImageEncoder>::format_name(&codec), "avif");
    }
}
