//! One-call processing helper for common server workflows.
//!
//! ```no_run
//! use viprs::adapters::process::{EncodeOptions, ProcessOptions, process};
//!
//! let input_bytes = vec![0xFF, 0xD8, 0xFF];
//! let mut output = Vec::new();
//! let _result = process(
//!     &input_bytes,
//!     &mut output,
//!     |img| Ok(img),
//!     &EncodeOptions::Jpeg { quality: 85 },
//!     &ProcessOptions::default(),
//! );
//! ```

#[cfg(any(
    feature = "jpeg",
    feature = "png",
    feature = "webp",
    feature = "tiff",
    feature = "gif",
    feature = "avif",
    feature = "heif",
    feature = "jp2k",
    feature = "exr",
    feature = "bmp"
))]
use crate::ports::codec::ImageEncoder;
use std::io::Write;

#[cfg(feature = "avif")]
use crate::adapters::codecs::AvifCodec;
#[cfg(feature = "bmp")]
use crate::adapters::codecs::BmpCodec;
#[cfg(feature = "exr")]
use crate::adapters::codecs::ExrCodec;
#[cfg(feature = "gif")]
use crate::adapters::codecs::GifCodec;
#[cfg(feature = "heif")]
use crate::adapters::codecs::HeifCodec;
#[cfg(feature = "jp2k")]
use crate::adapters::codecs::Jp2kCodec;
#[cfg(feature = "jpeg")]
use crate::adapters::codecs::JpegCodec;
#[cfg(feature = "png")]
use crate::adapters::codecs::PngCodec;
#[cfg(feature = "tiff")]
use crate::adapters::codecs::TiffCodec;
#[cfg(feature = "webp")]
use crate::adapters::codecs::WebpCodec;
#[cfg(feature = "exr")]
use crate::domain::{format::F32, image::AnimationFrame, ops::conversion::cast::CastSample};
use crate::{
    adapters::codecs::registry::ForeignRegistry,
    domain::{
        cancel::CancellationToken,
        codec_options::{LoadOptions, SaveOptions},
        error::ViprsError,
        format::U8,
        image::{Image, ImageMetadata},
        limits::DecodeLimits,
    },
};

/// Output encoding format and quality options.
///
/// Supports all formats that have an enabled feature flag. The input format
/// is auto-detected by the registry; output format is chosen here.
#[derive(Clone, Debug)]
pub enum EncodeOptions {
    /// JPEG output with quality 1-100.
    Jpeg {
        /// Lossy JPEG quality factor in the range 1-100.
        quality: u8,
    },
    /// PNG output with compression level 0-9.
    Png {
        /// Deflate compression level in the range 0-9.
        compression: u8,
    },
    /// WebP output. `quality` 1-100 for lossy, or `lossless = true`.
    WebP {
        /// Lossy WebP quality factor in the range 1-100.
        quality: u8,
        /// Whether to use WebP's lossless coding mode.
        lossless: bool,
    },
    /// TIFF output.
    Tiff,
    /// GIF output.
    Gif,
    /// AVIF output with quality 1-100.
    Avif {
        /// Lossy AVIF quality factor in the range 1-100.
        quality: u8,
    },
    /// HEIF output with quality 1-100.
    Heif {
        /// Lossy HEIF quality factor in the range 1-100.
        quality: u8,
    },
    /// JPEG 2000 output with quality 1-100.
    Jp2k {
        /// Lossy JPEG 2000 quality factor in the range 1-100.
        quality: u8,
    },
    /// `OpenEXR` output (lossless HDR).
    Exr,
    /// BMP output (uncompressed).
    Bmp,
}

/// Processing control options.
#[derive(Clone, Debug, Default)]
pub struct ProcessOptions {
    /// Optional cancellation token used for cooperative cancellation.
    pub cancel_token: Option<CancellationToken>,
    /// Decode and execution limits enforced before expensive work continues.
    pub limits: DecodeLimits,
    /// Whether output metadata should be cleared before encoding.
    pub strip_metadata: bool,
}

/// Result of a successful processing operation.
#[derive(Clone, Debug)]
pub struct ProcessResult {
    /// Detected input format name (for example, `"jpeg"` or `"png"`).
    pub input_format: &'static str,
    /// Input image dimensions as `(width, height)`.
    pub input_dimensions: (u32, u32),
    /// Encoder format name used for the response payload.
    pub output_format: &'static str,
    /// Number of bytes written to the output writer.
    pub bytes_written: u64,
}

/// Process an image: decode from bytes, apply operations, encode to writer.
///
/// This is the primary server API — handles the full lifecycle in one call.
/// The `ops` closure receives the decoded image and returns the processed result.
///
/// # Errors
///
/// Returns `ViprsError` on decode failure, operation failure, or encode/write failure.
/// Returns `ViprsError::Cancelled` if the cancel token is triggered.
pub fn process<W, F>(
    input: &[u8],
    output: &mut W,
    ops: F,
    encode_opts: &EncodeOptions,
    process_opts: &ProcessOptions,
) -> Result<ProcessResult, ViprsError>
where
    W: Write,
    F: FnOnce(Image<U8>) -> Result<Image<U8>, ViprsError>,
{
    validate_encode_options(encode_opts)?;
    check_cancelled(process_opts.cancel_token.as_ref())?;

    let load_options = LoadOptions::default();
    let (decoded, input_format) =
        ForeignRegistry::shared().load_from_memory_with_options(input, &load_options)?;
    process_opts
        .limits
        .validate_u8_image(decoded.width(), decoded.height(), decoded.bands())?;
    let input_dimensions = (decoded.width(), decoded.height());

    check_cancelled(process_opts.cancel_token.as_ref())?;
    let mut processed = ops(decoded)?;
    check_cancelled(process_opts.cancel_token.as_ref())?;

    process_opts.limits.validate_u8_image(
        processed.width(),
        processed.height(),
        processed.bands(),
    )?;
    if process_opts.strip_metadata {
        processed = processed.with_metadata(ImageMetadata::default());
    }

    let encoded = encode_image(&processed, encode_opts, process_opts)?;
    output.write_all(&encoded)?;

    Ok(ProcessResult {
        input_format,
        input_dimensions,
        output_format: encode_opts.format_name(),
        bytes_written: encoded.len() as u64,
    })
}

/// Pipeline-based image processing: operations run tile-by-tile through the
/// demand-driven scheduler, so only a sliding window of tiles is resident at once
/// during the processing phase.
///
/// Unlike [`process`], this function expresses operations as pipeline builder
/// steps. The scheduler requests and processes tiles on demand, keeping peak
/// working-set memory proportional to `(tile_height × width × bands × threads)`
/// rather than a second full copy of the image.
///
/// The `build_pipeline` closure receives a `PipelineBuilder` (backed by a
/// `MemorySource` of the decoded image) and must return a built `CompiledPipeline`.
///
/// # Errors
///
/// Returns `ViprsError` on decode, pipeline, or encode failure.
/// Returns `ViprsError::Cancelled` if the cancel token is triggered.
#[cfg(feature = "rayon")]
pub fn process_pipeline<W, F>(
    input: &[u8],
    output: &mut W,
    build_pipeline: F,
    encode_opts: &EncodeOptions,
    process_opts: &ProcessOptions,
) -> Result<ProcessResult, ViprsError>
where
    W: Write,
    F: FnOnce(
        crate::adapters::pipeline::PipelineBuilder,
    ) -> Result<crate::adapters::pipeline::CompiledPipeline, ViprsError>,
{
    use crate::adapters::pipeline::PipelineBuilder;
    use crate::adapters::scheduler::rayon_scheduler::RayonScheduler;
    use crate::adapters::sinks::memory::MemorySink;
    use crate::adapters::sources::memory::MemorySource;
    use crate::ports::scheduler::TileScheduler;

    validate_encode_options(encode_opts)?;
    check_cancelled(process_opts.cancel_token.as_ref())?;

    // Decode input via the format registry.
    let load_options = LoadOptions::default();
    let (decoded, input_format) =
        ForeignRegistry::shared().load_from_memory_with_options(input, &load_options)?;
    process_opts
        .limits
        .validate_u8_image(decoded.width(), decoded.height(), decoded.bands())?;
    let input_dimensions = (decoded.width(), decoded.height());

    check_cancelled(process_opts.cancel_token.as_ref())?;

    // Build a MemorySource from decoded pixels — the pipeline reads tiles on demand.
    let metadata = decoded.metadata().clone();
    let source = MemorySource::<U8>::new(
        decoded.width(),
        decoded.height(),
        decoded.bands(),
        decoded.into_buffer(),
    )?;
    let builder = PipelineBuilder::from_source(source);

    // Let the caller configure pipeline operations.
    let pipeline = build_pipeline(builder)?;

    check_cancelled(process_opts.cancel_token.as_ref())?;

    // Execute pipeline tile-by-tile via the rayon scheduler.
    let scheduler = RayonScheduler::new(RayonScheduler::default_threads())?;
    let mut sink = MemorySink::for_pipeline(&pipeline)?;

    if let Some(ref token) = process_opts.cancel_token {
        scheduler.run_cancellable(&pipeline, &mut sink, token)?;
    } else {
        scheduler.run(&pipeline, &mut sink)?;
    }

    // Extract result and encode.
    let out_metadata = if process_opts.strip_metadata {
        ImageMetadata::default()
    } else {
        metadata
    };
    let image = sink.into_image::<U8>(
        pipeline.width,
        pipeline.height,
        pipeline.output_bands,
        out_metadata,
    )?;

    let encoded = encode_image(&image, encode_opts, process_opts)?;
    output.write_all(&encoded)?;

    Ok(ProcessResult {
        input_format,
        input_dimensions,
        output_format: encode_opts.format_name(),
        bytes_written: encoded.len() as u64,
    })
}

fn validate_encode_options(encode_opts: &EncodeOptions) -> Result<(), ViprsError> {
    match *encode_opts {
        EncodeOptions::Jpeg { quality } if !(1..=100).contains(&quality) => Err(ViprsError::Codec(
            format!("process: JPEG quality must be in 1..=100, got {quality}"),
        )),
        EncodeOptions::Png { compression } if compression > 9 => Err(ViprsError::Codec(format!(
            "process: PNG compression must be in 0..=9, got {compression}"
        ))),
        EncodeOptions::WebP { quality, .. } if !(1..=100).contains(&quality) => {
            Err(ViprsError::Codec(format!(
                "process: WebP quality must be in 1..=100, got {quality}"
            )))
        }
        EncodeOptions::Avif { quality } if !(1..=100).contains(&quality) => Err(ViprsError::Codec(
            format!("process: AVIF quality must be in 1..=100, got {quality}"),
        )),
        EncodeOptions::Heif { quality } if !(1..=100).contains(&quality) => Err(ViprsError::Codec(
            format!("process: HEIF quality must be in 1..=100, got {quality}"),
        )),
        EncodeOptions::Jp2k { quality } if !(1..=100).contains(&quality) => Err(ViprsError::Codec(
            format!("process: JP2K quality must be in 1..=100, got {quality}"),
        )),
        _ => Ok(()),
    }
}

fn check_cancelled(cancel_token: Option<&CancellationToken>) -> Result<(), ViprsError> {
    if let Some(token) = cancel_token {
        token.check_cancelled()?;
    }
    Ok(())
}

fn encode_image(
    image: &Image<U8>,
    encode_opts: &EncodeOptions,
    process_opts: &ProcessOptions,
) -> Result<Vec<u8>, ViprsError> {
    let save_options = save_options(encode_opts, process_opts.strip_metadata);
    match *encode_opts {
        EncodeOptions::Jpeg { .. } => encode_jpeg(image, &save_options),
        EncodeOptions::Png { .. } => encode_png(image, &save_options),
        EncodeOptions::WebP { .. } => encode_webp(image, &save_options),
        EncodeOptions::Tiff => encode_tiff(image, &save_options),
        EncodeOptions::Gif => encode_gif(image, &save_options),
        EncodeOptions::Avif { .. } => encode_avif(image, &save_options),
        EncodeOptions::Heif { .. } => encode_heif(image, &save_options),
        EncodeOptions::Jp2k { .. } => encode_jp2k(image, &save_options),
        EncodeOptions::Exr => encode_exr(image, &save_options),
        EncodeOptions::Bmp => encode_bmp(image, &save_options),
    }
}

fn save_options(encode_opts: &EncodeOptions, strip_metadata: bool) -> SaveOptions {
    let mut options = SaveOptions {
        strip_metadata: Some(strip_metadata),
        ..SaveOptions::default()
    };

    match *encode_opts {
        EncodeOptions::Png { compression } => {
            options.compression_level = Some(compression);
        }
        EncodeOptions::WebP { quality, lossless } => {
            options.quality = Some(quality);
            options.lossless = Some(lossless);
        }
        EncodeOptions::Jpeg { quality }
        | EncodeOptions::Avif { quality }
        | EncodeOptions::Heif { quality }
        | EncodeOptions::Jp2k { quality } => {
            options.quality = Some(quality);
        }
        EncodeOptions::Tiff | EncodeOptions::Gif | EncodeOptions::Exr | EncodeOptions::Bmp => {}
    }

    options
}

impl EncodeOptions {
    const fn format_name(&self) -> &'static str {
        match self {
            Self::Jpeg { .. } => "jpeg",
            Self::Png { .. } => "png",
            Self::WebP { .. } => "webp",
            Self::Tiff => "tiff",
            Self::Gif => "gif",
            Self::Avif { .. } => "avif",
            Self::Heif { .. } => "heif",
            Self::Jp2k { .. } => "jp2k",
            Self::Exr => "exr",
            Self::Bmp => "bmp",
        }
    }
}

#[cfg(feature = "jpeg")]
fn encode_jpeg(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    JpegCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "jpeg"))]
const fn encode_jpeg(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: jpeg",
        details: "enable Cargo feature `jpeg` to use EncodeOptions::Jpeg",
    })
}

#[cfg(feature = "png")]
fn encode_png(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    PngCodec::default().encode_with_options(image, options)
}

#[cfg(not(feature = "png"))]
const fn encode_png(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: png",
        details: "enable Cargo feature `png` to use EncodeOptions::Png",
    })
}

#[cfg(feature = "webp")]
fn encode_webp(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    WebpCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "webp"))]
const fn encode_webp(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: webp",
        details: "enable Cargo feature `webp` to use EncodeOptions::WebP",
    })
}

#[cfg(feature = "tiff")]
fn encode_tiff(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    TiffCodec::default().encode_with_options(image, options)
}

#[cfg(not(feature = "tiff"))]
const fn encode_tiff(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: tiff",
        details: "enable Cargo feature `tiff` to use EncodeOptions::Tiff",
    })
}

#[cfg(feature = "gif")]
fn encode_gif(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    GifCodec::default().encode_with_options(image, options)
}

#[cfg(not(feature = "gif"))]
const fn encode_gif(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: gif",
        details: "enable Cargo feature `gif` to use EncodeOptions::Gif",
    })
}

#[cfg(feature = "avif")]
fn encode_avif(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    AvifCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "avif"))]
const fn encode_avif(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: avif",
        details: "enable Cargo feature `avif` to use EncodeOptions::Avif",
    })
}

#[cfg(feature = "heif")]
fn encode_heif(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    HeifCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "heif"))]
const fn encode_heif(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: heif",
        details: "enable Cargo feature `heif` to use EncodeOptions::Heif",
    })
}

#[cfg(feature = "jp2k")]
fn encode_jp2k(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Jp2kCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "jp2k"))]
const fn encode_jp2k(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: jp2k",
        details: "enable Cargo feature `jp2k` to use EncodeOptions::Jp2k",
    })
}

#[cfg(feature = "exr")]
fn encode_exr(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    ExrCodec.encode_with_options(&convert_u8_image_to_f32(image)?, options)
}

#[cfg(not(feature = "exr"))]
const fn encode_exr(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: exr",
        details: "enable Cargo feature `exr` to use EncodeOptions::Exr",
    })
}

#[cfg(feature = "exr")]
fn convert_u8_image_to_f32(image: &Image<U8>) -> Result<Image<F32>, ViprsError> {
    let pixels = image
        .pixels()
        .iter()
        .copied()
        .map(|sample| sample.cast_to())
        .collect();

    let mut converted =
        Image::<F32>::from_buffer(image.width(), image.height(), image.bands(), pixels)?
            .with_metadata(image.metadata().clone());

    if let Some(animation_frames) = image.animation_frames() {
        let converted_frames = animation_frames
            .iter()
            .map(convert_u8_animation_frame_to_f32)
            .collect::<Result<Vec<_>, _>>()?;
        converted = converted.with_animation_frames(converted_frames);
    } else if let Some(frames) = image.frames() {
        let converted_frames = frames
            .iter()
            .map(convert_u8_image_to_f32)
            .collect::<Result<Vec<_>, _>>()?;
        converted = converted.with_frames(converted_frames);
    }

    Ok(converted)
}

#[cfg(feature = "exr")]
fn convert_u8_animation_frame_to_f32(
    frame: &AnimationFrame<U8>,
) -> Result<AnimationFrame<F32>, ViprsError> {
    Ok(AnimationFrame::new(
        convert_u8_image_to_f32(frame.image())?,
        frame.delay_ms(),
        frame.disposal(),
    ))
}

#[cfg(feature = "bmp")]
fn encode_bmp(image: &Image<U8>, options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    BmpCodec.encode_with_options(image, options)
}

#[cfg(not(feature = "bmp"))]
const fn encode_bmp(_image: &Image<U8>, _options: &SaveOptions) -> Result<Vec<u8>, ViprsError> {
    Err(ViprsError::Unimplemented {
        feature: "process encode: bmp",
        details: "enable Cargo feature `bmp` to use EncodeOptions::Bmp",
    })
}

#[cfg(test)]
mod tests {
    use super::{EncodeOptions, ProcessOptions, process};
    use crate::domain::{cancel::CancellationToken, error::ViprsError};

    #[cfg(feature = "exr")]
    use crate::{
        adapters::codecs::ExrCodec,
        domain::{
            codec_options::SaveOptions,
            format::{F32, U8},
            image::Image,
        },
        ports::codec::ImageDecoder,
    };
    #[cfg(feature = "png")]
    use crate::{
        adapters::codecs::PngCodec,
        domain::{format::U8, image::Image},
        ports::codec::{ImageDecoder, ImageEncoder},
    };

    #[cfg(feature = "png")]
    #[test]
    fn png_round_trip_writes_output() {
        let input = Image::<U8>::from_buffer(2, 1, 1, vec![8, 16]).unwrap();
        let encoded = PngCodec::default().encode(&input).unwrap();
        let mut output = Vec::new();

        let result = process(
            &encoded,
            &mut output,
            |image| Ok(image),
            &EncodeOptions::Png { compression: 6 },
            &ProcessOptions::default(),
        )
        .unwrap();

        let decoded = PngCodec::default().decode::<U8>(&output).unwrap();
        assert_eq!(decoded.pixels(), &[8, 16]);
        assert_eq!(result.input_format, "png");
        assert_eq!(result.output_format, "png");
        assert_eq!(result.bytes_written, output.len() as u64);
    }

    #[test]
    fn cancelled_process_aborts_before_decode() {
        let token = CancellationToken::new();
        token.cancel();
        let options = ProcessOptions {
            cancel_token: Some(token),
            ..ProcessOptions::default()
        };

        let err = process(
            b"not-an-image",
            &mut Vec::new(),
            |image| Ok(image),
            &EncodeOptions::Png { compression: 0 },
            &options,
        )
        .unwrap_err();

        assert!(matches!(err, ViprsError::Cancelled));
    }

    #[cfg(feature = "exr")]
    #[test]
    fn exr_encode_auto_casts_u8_input_to_f32() {
        let image = Image::<U8>::from_buffer(2, 1, 1, vec![0, 255]).unwrap();

        let encoded = super::encode_exr(&image, &SaveOptions::default()).unwrap();
        let decoded = ExrCodec.decode::<F32>(&encoded).unwrap();

        assert_eq!(decoded.pixels(), &[0.0, 1.0]);
    }
}
