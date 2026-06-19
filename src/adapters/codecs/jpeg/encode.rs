use std::borrow::Cow;
use std::io::Write;

use jpeg_encoder::{
    ColorType as StreamingJpegColorType, Encoder as StreamingJpegEncoder, JfifWrite,
    SamplingFactor as StreamingSamplingFactor,
};
use turbojpeg::{PixelFormat, Subsamp, raw};

use super::super::web_colour::normalize_web_output_u8;
use super::JpegCodec;
use super::common::{
    JPEG_STREAM_FLUSH_BYTES, TurboJpegHandle, insert_metadata_segments,
    normalize_exif_app1_payload, normalize_xmp_app1_payload, subsampling_to_sampling_factor,
    turbojpeg_error, turbojpeg_quality,
};
use crate::adapters::instrumentation::viprs_span;
use crate::domain::codec_options::{JpegSubsampling, SaveOptions};
use crate::domain::error::ViprsError;
use crate::domain::format::{BandFormat, BandFormatId, U8};
use crate::domain::image::{Image, ImageMetadata, Interpretation};
use crate::ports::codec::ImageEncoder;

fn map_jpeg_encoding_error(error: jpeg_encoder::EncodingError) -> ViprsError {
    match error {
        jpeg_encoder::EncodingError::IoError(io_error) => ViprsError::Io(io_error),
        other => ViprsError::Codec(format!("jpeg: {other}")),
    }
}

struct EncodeInput<'a> {
    pixels: Cow<'a, [u8]>,
    pixel_format: PixelFormat,
    pitch: usize,
}

struct PeriodicFlushWriter<'a> {
    inner: &'a mut dyn Write,
    written_since_flush: usize,
}

impl<'a> PeriodicFlushWriter<'a> {
    fn new(inner: &'a mut dyn Write) -> Self {
        Self {
            inner,
            written_since_flush: 0,
        }
    }
}

impl Write for PeriodicFlushWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let written = self.inner.write(buf)?;
        self.written_since_flush = self.written_since_flush.saturating_add(written);
        if self.written_since_flush >= JPEG_STREAM_FLUSH_BYTES {
            self.inner.flush()?;
            self.written_since_flush = 0;
        }
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.written_since_flush = 0;
        self.inner.flush()
    }
}

fn streaming_jpeg_color_type(
    pixel_format: PixelFormat,
) -> Result<StreamingJpegColorType, ViprsError> {
    match pixel_format {
        PixelFormat::GRAY => Ok(StreamingJpegColorType::Luma),
        PixelFormat::RGB => Ok(StreamingJpegColorType::Rgb),
        PixelFormat::CMYK => Ok(StreamingJpegColorType::Cmyk),
        other => Err(ViprsError::Codec(format!(
            "jpeg: unsupported streaming pixel format {other:?}"
        ))),
    }
}

fn streaming_sampling_factor(quality: u8, subsampling: JpegSubsampling) -> StreamingSamplingFactor {
    match subsampling {
        JpegSubsampling::Auto => {
            if quality < 90 {
                StreamingSamplingFactor::F_2_2
            } else {
                StreamingSamplingFactor::F_1_1
            }
        }
        JpegSubsampling::Off => StreamingSamplingFactor::F_1_1,
        JpegSubsampling::Subsample420 => StreamingSamplingFactor::F_2_2,
        JpegSubsampling::Subsample422 => StreamingSamplingFactor::F_2_1,
        JpegSubsampling::Subsample440 => StreamingSamplingFactor::F_1_2,
    }
}

fn add_streaming_metadata_segments<W: JfifWrite>(
    encoder: &mut StreamingJpegEncoder<W>,
    image: &ImageMetadata,
) -> Result<(), ViprsError> {
    if let Some(xmp) = image.xmp.as_deref() {
        encoder
            .add_app_segment(1, &normalize_xmp_app1_payload(xmp))
            .map_err(map_jpeg_encoding_error)?;
    }
    if let Some(exif) = image.exif.as_deref() {
        encoder
            .add_app_segment(1, &normalize_exif_app1_payload(exif))
            .map_err(map_jpeg_encoding_error)?;
    }
    if let Some(icc_profile) = image.icc_profile.as_deref() {
        encoder
            .add_icc_profile(icc_profile)
            .map_err(map_jpeg_encoding_error)?;
    }
    Ok(())
}

fn prepare_encode_input<'a, F: BandFormat>(
    image: &'a Image<F>,
) -> Result<EncodeInput<'a>, ViprsError> {
    // SAFETY: callers check `F::ID == BandFormatId::U8` before reaching this helper,
    // therefore `F::Sample` is layout-identical to `u8`.
    let pixel_bytes: &'a [u8] = bytemuck::cast_slice(image.pixels());
    match image.bands() {
        1 => Ok(EncodeInput {
            pixels: Cow::Borrowed(pixel_bytes),
            pixel_format: PixelFormat::GRAY,
            pitch: image.width() as usize,
        }),
        3 => Ok(EncodeInput {
            pixels: Cow::Borrowed(pixel_bytes),
            pixel_format: PixelFormat::RGB,
            pitch: image.width() as usize * 3,
        }),
        4 if image.metadata().interpretation == Some(Interpretation::Cmyk) => Ok(EncodeInput {
            pixels: Cow::Borrowed(pixel_bytes),
            pixel_format: PixelFormat::CMYK,
            pitch: image.width() as usize * 4,
        }),
        4 => {
            let mut rgb = Vec::with_capacity(image.width() as usize * image.height() as usize * 3);
            for rgba in pixel_bytes.chunks_exact(4) {
                rgb.extend_from_slice(&rgba[..3]);
            }
            Ok(EncodeInput {
                pixels: Cow::Owned(rgb),
                pixel_format: PixelFormat::RGB,
                pitch: image.width() as usize * 3,
            })
        }
        n => Err(ViprsError::Codec(format!(
            "jpeg: cannot encode image with {n} bands (only 1, 3, 4 supported)"
        ))),
    }
}
impl ImageEncoder for JpegCodec {
    fn format_name(&self) -> &'static str {
        "jpeg"
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
        viprs_span!(tracing::Level::INFO, "viprs.encode", format = "jpeg");
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "jpeg: unsupported format {:?} — only U8 is supported",
                F::ID
            )));
        }

        // SAFETY: the `F::ID == BandFormatId::U8` guard above guarantees `F::Sample == u8`.
        // `Image<F>` and `Image<U8>` therefore have identical layouts for this call site.
        let image =
            normalize_web_output_u8(unsafe { &*std::ptr::from_ref(image).cast::<Image<U8>>() })?;
        let image = image.as_ref();
        let quality = opts.quality.unwrap_or(75);
        let prepared = prepare_encode_input(image)?;
        let subsampling = if prepared.pixel_format == PixelFormat::GRAY {
            Subsamp::Gray
        } else {
            subsampling_to_sampling_factor(
                opts.jpeg_subsampling.unwrap_or(JpegSubsampling::Auto),
                quality,
            )
        };

        let mut enc = TurboJpegHandle::new(raw::TJINIT_TJINIT_COMPRESS)?;
        enc.set(raw::TJPARAM_TJPARAM_QUALITY, turbojpeg_quality(quality)?)?;
        enc.set(raw::TJPARAM_TJPARAM_SUBSAMP, subsampling as i32)?;
        if let Some(interlace) = opts.interlace {
            enc.set(raw::TJPARAM_TJPARAM_PROGRESSIVE, i32::from(interlace))?;
        }
        if let Some(restart_interval) = opts.restart_interval {
            enc.set(
                raw::TJPARAM_TJPARAM_RESTARTBLOCKS,
                i32::from(restart_interval),
            )?;
        }

        let width = i32::try_from(image.width())
            .map_err(|_| ViprsError::Codec("jpeg: width overflow".into()))?;
        let pitch = i32::try_from(prepared.pitch)
            .map_err(|_| ViprsError::Codec("jpeg: row stride overflow".into()))?;
        let height = i32::try_from(image.height())
            .map_err(|_| ViprsError::Codec("jpeg: height overflow".into()))?;
        let mut jpeg_ptr = std::ptr::null_mut();
        let mut jpeg_len: raw::size_t = 0;
        // SAFETY: `enc` owns a valid compressor handle; the input slice is alive for the
        // duration of the call; TurboJPEG allocates and returns the destination buffer.
        let encode_result = unsafe {
            raw::tj3Compress8(
                enc.ptr,
                prepared.pixels.as_ptr(),
                width,
                pitch,
                height,
                prepared.pixel_format as i32,
                std::ptr::addr_of_mut!(jpeg_ptr),
                std::ptr::addr_of_mut!(jpeg_len),
            )
        };
        if encode_result != 0 {
            return Err(turbojpeg_error(enc.ptr, "jpeg"));
        }
        if jpeg_ptr.is_null() {
            return Err(ViprsError::Codec(
                "jpeg: encoder returned null output buffer".into(),
            ));
        }

        let jpeg_len_usize = usize::try_from(jpeg_len)
            .map_err(|_| ViprsError::Codec("jpeg: encoded output length overflow".into()))?;
        let mut output = vec![0u8; jpeg_len_usize];
        // SAFETY: `jpeg_ptr` points to `jpeg_len_usize` bytes allocated by TurboJPEG.
        unsafe {
            std::ptr::copy_nonoverlapping(jpeg_ptr, output.as_mut_ptr(), jpeg_len_usize);
            raw::tj3Free(jpeg_ptr.cast());
        }

        if opts.strip_metadata != Some(true) {
            insert_metadata_segments(&mut output, image.metadata())?;
        }

        Ok(output)
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
        viprs_span!(tracing::Level::INFO, "viprs.encode", format = "jpeg");
        if F::ID != BandFormatId::U8 {
            return Err(ViprsError::Codec(format!(
                "jpeg: unsupported format {:?} — only U8 is supported",
                F::ID
            )));
        }
        if image.width() > u32::from(u16::MAX) || image.height() > u32::from(u16::MAX) {
            let buf = self.encode_with_options(image, opts)?;
            writer.write_all(&buf)?;
            writer.flush()?;
            return Ok(());
        }

        let quality = opts.quality.unwrap_or(75);
        let prepared = prepare_encode_input(image)?;
        let color_type = streaming_jpeg_color_type(prepared.pixel_format)?;
        let mut writer = PeriodicFlushWriter::new(writer);
        let mut encoder = StreamingJpegEncoder::new(&mut writer, quality.max(1));

        if prepared.pixel_format != PixelFormat::GRAY {
            encoder.set_sampling_factor(streaming_sampling_factor(
                quality,
                opts.jpeg_subsampling.unwrap_or(JpegSubsampling::Auto),
            ));
        }
        if let Some(interlace) = opts.interlace {
            encoder.set_progressive(interlace);
        }
        if let Some(restart_interval) = opts.restart_interval {
            encoder.set_restart_interval(restart_interval);
        }
        if opts.strip_metadata != Some(true) {
            add_streaming_metadata_segments(&mut encoder, image.metadata())?;
        }

        encoder
            .encode(
                prepared.pixels.as_ref(),
                u16::try_from(image.width())
                    .map_err(|_| ViprsError::Codec("jpeg: width overflow".into()))?,
                u16::try_from(image.height())
                    .map_err(|_| ViprsError::Codec("jpeg: height overflow".into()))?,
                color_type,
            )
            .map_err(map_jpeg_encoding_error)?;
        writer.flush()?;
        Ok(())
    }
}
