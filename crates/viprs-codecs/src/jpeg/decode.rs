#[cfg(feature = "mmap")]
use std::fs::File;
use std::path::Path;

#[cfg(feature = "mmap")]
use memmap2::Mmap;
use turbojpeg::raw;

use super::common::{
    JpegPreflight, MAX_JPEG_DECODED_IMAGE_BYTES, TurboJpegHandle, checked_decoded_image_len,
    crop_strict_shrink_edges, jpeg_shrink_on_load_plan, preflight_jpeg, probe_jpeg_header,
    raw_scaling_factor_for_shrink, shrink_dimension_for_factor, shrink_factor_for_max_dimension,
    turbojpeg_error,
};
use super::{JpegCodec, apply_exif_orientation};
use crate::viprs_span;
use viprs_core::codec_options::LoadOptions;
use viprs_core::error::ViprsError;
use viprs_core::format::{BandFormat, BandFormatId};
use viprs_core::image::{ImageMetadata, InMemoryImage};
use viprs_ports::codec::ImageDecoder;

fn scaled_dimension_for_factor(dimension: u32, factor: u8) -> u32 {
    if factor == 1 {
        dimension
    } else {
        dimension.div_ceil(u32::from(factor))
    }
}

impl ImageDecoder for JpegCodec {
    fn format_name(&self) -> &'static str {
        "jpeg"
    }

    fn sniff(&self, header: &[u8]) -> bool
    where
        Self: Sized,
    {
        // JPEG files begin with the SOI marker: FF D8 FF.
        header.len() >= 3 && header[0] == 0xFF && header[1] == 0xD8 && header[2] == 0xFF
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
        viprs_span!(tracing::Level::INFO, "viprs.decode", format = "jpeg");
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "jpeg: unsupported format {:?} — only U8 is supported",
                F::ID
            )));
        }
        let JpegPreflight {
            width: source_width,
            height: source_height,
            bands,
            pixel_format,
            interpretation,
            exif,
            icc_profile,
            xmp,
            orientation,
        } = preflight_jpeg(src)?;
        let mut dec = TurboJpegHandle::new(raw::TJINIT_TJINIT_DECOMPRESS)?;
        // Keep libjpeg-turbo's accurate chroma upsampling path enabled.
        // The fast upsampler is measurably less faithful on 4:2:0 fixtures and
        // drifted from libvips by up to 16 samples in the JPEG invert E2E test.
        dec.set(raw::TJPARAM_TJPARAM_FASTUPSAMPLE, 0)?;
        let src_len = src
            .len()
            .try_into()
            .map_err(|_| ViprsError::Codec("jpeg: source length overflow".into()))?;
        // SAFETY: `dec` owns a valid decompressor handle; `src` remains alive for the
        // duration of the call; TurboJPEG writes only to internal decoder state here.
        let header_result = unsafe { raw::tj3DecompressHeader(dec.ptr, src.as_ptr(), src_len) };
        if header_result != 0 {
            return Err(turbojpeg_error(dec.ptr, "jpeg"));
        }
        let requested_shrink = opts.shrink_factor.map_or_else(
            || {
                opts.max_dimension.map_or(1, |max_dimension| {
                    shrink_factor_for_max_dimension(source_width, source_height, max_dimension)
                })
            },
            std::num::NonZeroU8::get,
        );
        let shrink_plan = jpeg_shrink_on_load_plan(requested_shrink);
        if shrink_plan.factor() > 1 && dec.get(raw::TJPARAM_TJPARAM_LOSSLESS) != 0 {
            return Err(ViprsError::Codec(
                "jpeg: lossless JPEG cannot use shrink-on-load scaling".into(),
            ));
        }
        let target_width = shrink_dimension_for_factor(source_width, shrink_plan.factor());
        let target_height = shrink_dimension_for_factor(source_height, shrink_plan.factor());
        let scaled_width = scaled_dimension_for_factor(source_width, shrink_plan.factor());
        let scaled_height = scaled_dimension_for_factor(source_height, shrink_plan.factor());
        if shrink_plan.factor() > 1 {
            dec.set_scaling_factor(raw_scaling_factor_for_shrink(shrink_plan.factor()))?;
        }

        let pitch = (scaled_width as usize)
            .checked_mul(pixel_format.size())
            .ok_or_else(|| ViprsError::Codec("jpeg: row stride overflow".into()))?;
        let decoded_len = checked_decoded_image_len(
            "jpeg",
            scaled_width,
            scaled_height,
            bands,
            1,
            MAX_JPEG_DECODED_IMAGE_BYTES,
        )?;
        // `decode_with_options` must return an owning `Image`, so one full output-sized
        // allocation is the minimum contract for this API. TurboJPEG exposes partial
        // packed-pixel decode via `tj3SetCroppingRegion`, but that belongs in a
        // `TileImageDecoder::decode_region_into` implementation. This eager
        // `ImageDecoder` path still materializes one full shrunken raster, which then
        // becomes the Image backing allocation via `Vec::from_raw_parts` below. Any
        // truly tile-bounded JPEG path must be implemented as a streaming decoder
        // rather than by splitting this method into smaller eager chunks.
        let mut decoded_pixels = Vec::with_capacity(decoded_len);
        let pitch_i32 = i32::try_from(pitch)
            .map_err(|_| ViprsError::Codec("jpeg: row stride overflow".into()))?;
        // SAFETY: `dec` owns a valid handle configured for decompression; `decoded_pixels`
        // has enough spare capacity for `scaled_width` × `scaled_height` pixels in
        // `pixel_format`, and TurboJPEG fully initializes that destination on success.
        let decode_result = unsafe {
            raw::tj3Decompress8(
                dec.ptr,
                src.as_ptr(),
                src_len,
                decoded_pixels
                    .spare_capacity_mut()
                    .as_mut_ptr()
                    .cast::<u8>(),
                pitch_i32,
                pixel_format as i32,
            )
        };
        if decode_result != 0 {
            return Err(turbojpeg_error(dec.ptr, "jpeg"));
        }
        // SAFETY: `tj3Decompress8` wrote exactly `decoded_len` initialized bytes into the
        // reserved buffer on the success path above, so exposing that initialized prefix is sound.
        unsafe { decoded_pixels.set_len(decoded_len) };

        let mut width = scaled_width;
        let mut height = scaled_height;
        let mut pixels_u8 = crop_strict_shrink_edges(
            decoded_pixels,
            width,
            height,
            target_width,
            target_height,
            bands,
        )?;
        width = target_width;
        height = target_height;

        if !opts.no_rotate
            && let Some(value) = orientation
        {
            (width, height, pixels_u8) =
                apply_exif_orientation(pixels_u8, width, height, bands, value, "jpeg")?;
        }

        // SAFETY: F::ID == BandFormatId::U8 was checked above, which means
        // F::Sample is `u8` (size 1, align 1, Pod).  A Vec<u8> is therefore
        // layout-identical to Vec<F::Sample>.  We consume the original Vec to
        // avoid any double-free; the reconstructed Vec takes ownership of the
        // same heap allocation.
        let samples: Vec<F::Sample> = unsafe {
            let mut v = std::mem::ManuallyDrop::new(pixels_u8);
            Vec::from_raw_parts(v.as_mut_ptr().cast::<F::Sample>(), v.len(), v.capacity())
        };

        let metadata = ImageMetadata {
            interpretation,
            icc_profile,
            exif,
            xmp,
            orientation: if opts.no_rotate {
                orientation
            } else {
                orientation.map(|_| 1)
            },
            ..ImageMetadata::default()
        };

        InMemoryImage::from_buffer(width, height, bands, samples)
            .map(|image| image.with_metadata(metadata))
            .map_err(|e| ViprsError::Codec(format!("jpeg: {e}")))
    }

    fn probe(&self, src: &[u8]) -> Result<(u32, u32, u32), ViprsError>
    where
        Self: Sized,
    {
        probe_jpeg_header(src)
    }

    fn decode_path_with_options<F: BandFormat>(
        &self,
        path: &Path,
        opts: &LoadOptions,
    ) -> Result<InMemoryImage<F>, ViprsError>
    where
        Self: Sized,
    {
        #[cfg(feature = "mmap")]
        {
            let file = File::open(path)?;
            // SAFETY: the file descriptor stays alive for the lifetime of `mmap`,
            // the mapping is read-only, and the bytes are consumed only within
            // this function before either handle is dropped.
            let mmap = unsafe { Mmap::map(&file)? };
            self.decode_with_options(&mmap, opts)
        }

        #[cfg(not(feature = "mmap"))]
        {
            let src = std::fs::read(path)?;
            self.decode_with_options(&src, opts)
        }
    }
}
