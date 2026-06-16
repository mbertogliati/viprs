#![allow(clippy::unnecessary_wraps)]
// REASON: image-api constructors preserve a uniform fallible adapter interface across source types.

use super::{
    BandFormat, BandFormatId, BuildError, CompiledPipeline, DecodeLimits, ForeignRegistry, Image,
    JPEG_HEADER, LoadOptions, MemorySource, PNG_HEADER, Path, PipelineBuilder, Pod, RayonScheduler,
    Read, ResourceLimits, ViprsError, size_of,
};

#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use super::{Arc, DecoderSource};
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use crate::domain::format::U8;

#[cfg(feature = "jpeg")]
use super::JpegCodec;
#[cfg(feature = "png")]
use super::PngCodec;
#[cfg(feature = "png")]
use super::PNG_IHDR_BIT_DEPTH_OFFSET;
#[cfg(feature = "png")]
use crate::domain::format::U16;
#[cfg(feature = "png")]
use std::fs;
#[cfg(feature = "webp")]
use super::{WEBP_MAGIC, WEBP_RIFF_HEADER, WebpCodec};

/// High-level façade for decode → pipeline → encode workflows.
///
/// `ImageApi` is the main user-facing adapter for request/response image
/// processing. It lets callers build a compiled pipeline with fluent methods and
/// execute it only when they need encoded bytes or a saved file.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs::adapters::image_api::ImageApi;
///
/// ImageApi::open("photo.jpg")?
///     .invert()?
///     .thumbnail(400)?
///     .save("out.jpg")?;
/// # Ok::<(), viprs::domain::error::ViprsError>(())
/// ```
pub struct ImageApi {
    pub(in crate::adapters::image_api) builder: PipelineBuilder,
    pub(in crate::adapters::image_api) resource_limits: Option<ResourceLimits>,
}

impl std::fmt::Debug for ImageApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageApi").finish()
    }
}

/// Configured decode entrypoint that applies shared request limits to every run.
///
/// Use this loader when a service wants one preconfigured set of decode and
/// execution limits reused across many requests.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs::{
///     adapters::image_api::ImageApi,
///     domain::limits::ResourceLimits,
/// };
///
/// let loader = ImageApi::with_limits(ResourceLimits::default());
/// let _image = loader.open("photo.jpg")?;
/// # Ok::<(), viprs::domain::error::ViprsError>(())
/// ```
#[derive(Clone, Debug)]
pub struct ImageApiLoader {
    limits: ResourceLimits,
}

/// Optional thumbnail planning controls for the fluent [`ImageApi`] façade.
///
/// This type solves the small set of thumbnail policy knobs that should remain
/// ergonomic without exposing the full lower-level thumbnail planner.
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "icc")] {
/// use viprs::adapters::image_api::ImageApiThumbnailOptions;
///
/// let options = ImageApiThumbnailOptions::default().with_auto_normalize_to_srgb(true);
/// let _ = options;
/// # }
/// ```
#[cfg(feature = "icc")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageApiThumbnailOptions {
    pub(in crate::adapters::image_api) auto_normalize_to_srgb: bool,
}

#[cfg(feature = "icc")]
impl ImageApiThumbnailOptions {
    /// Enable or disable explicit ICC-to-sRGB normalization after thumbnail planning.
    ///
    /// This is useful when thumbnail output must always land in web-friendly sRGB
    /// even if the source embeds a different ICC profile.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "icc")] {
    /// use viprs::adapters::image_api::ImageApiThumbnailOptions;
    ///
    /// let options = ImageApiThumbnailOptions::default().with_auto_normalize_to_srgb(true);
    /// assert_ne!(options, ImageApiThumbnailOptions::default());
    /// # }
    /// ```
    #[must_use]
    pub fn with_auto_normalize_to_srgb(mut self, enabled: bool) -> Self {
        self.auto_normalize_to_srgb = enabled;
        self
    }
}

impl ImageApi {
    /// Create a configured loader that applies the provided limits to every decode
    /// and execution. Clone a single [`ResourceLimits`] value in service state to
    /// share its concurrency gate across requests.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::{
    ///     adapters::image_api::ImageApi,
    ///     domain::limits::ResourceLimits,
    /// };
    ///
    /// let _loader = ImageApi::with_limits(ResourceLimits::default());
    /// ```
    #[must_use]
    pub const fn with_limits(limits: ResourceLimits) -> ImageApiLoader {
        ImageApiLoader { limits }
    }

    /// Open an image from a file path, auto-detecting the codec.
    ///
    /// This solves the most common entrypoint for file-based workflows without
    /// requiring callers to pick a decoder manually.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs::adapters::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?;
    /// let _ = image;
    /// # Ok::<(), viprs::domain::error::ViprsError>(())
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ViprsError> {
        Self::open_with_options(path.as_ref(), &LoadOptions::default(), None)
    }

    /// Decode an image from any `Read` source.
    ///
    /// Use this when image bytes already live in an HTTP body, archive entry, or
    /// any other streaming reader.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::io::Cursor;
    /// use viprs::adapters::image_api::ImageApi;
    ///
    /// let bytes = Cursor::new(vec![0_u8; 0]);
    /// let _ = ImageApi::from_reader(bytes);
    /// ```
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self, ViprsError> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }

    /// Decode bytes while validating decoded resource limits.
    ///
    /// This protects services that accept untrusted uploads by rejecting images
    /// whose decoded dimensions or sample sizes exceed a budget.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::{
    ///     adapters::image_api::ImageApi,
    ///     domain::limits::DecodeLimits,
    /// };
    ///
    /// let _ = ImageApi::from_bytes_with_limits(&[], DecodeLimits::default());
    /// ```
    pub fn from_bytes_with_limits(buf: &[u8], limits: DecodeLimits) -> Result<Self, ViprsError> {
        Self::from_bytes_with_options(
            buf,
            None,
            &LoadOptions {
                limits: Some(limits),
                ..LoadOptions::default()
            },
            None,
        )
    }

    /// Decode bytes into an in-memory source using codec auto-detection.
    ///
    /// This is the most direct way to process already-buffered uploads without
    /// writing them to disk first.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::adapters::image_api::ImageApi;
    ///
    /// let _ = ImageApi::from_bytes(&[]);
    /// ```
    pub fn from_bytes(buf: &[u8]) -> Result<Self, ViprsError> {
        Self::from_bytes_with_options(buf, None, &LoadOptions::default(), None)
    }

    pub(in crate::adapters::image_api) const fn from_builder(
        builder: PipelineBuilder,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            builder,
            resource_limits,
        })
    }

    pub(in crate::adapters::image_api) fn open_with_options(
        path: &Path,
        opts: &LoadOptions,
        resource_limits: Option<&ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        if path.is_dir() {
            return Self::from_registry_path(path, opts, resource_limits);
        }

        match normalized_extension(path) {
            #[cfg(feature = "jpeg")]
            Some("jpg" | "jpeg") => {
                Self::from_jpeg_path_with_options(path, opts, resource_limits.cloned())
            }
            #[cfg(feature = "png")]
            Some("png") => Self::from_png_path_with_options(path, opts, resource_limits.cloned()),
            #[cfg(feature = "webp")]
            Some("webp") => Self::from_webp_path_with_options(path, opts, resource_limits.cloned()),
            _ => Self::from_registry_path(path, opts, resource_limits),
        }
    }

    pub(in crate::adapters::image_api) fn from_bytes_with_options(
        buf: &[u8],
        path_hint: Option<&Path>,
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        if is_jpeg(buf) {
            return Self::from_jpeg_bytes_with_options(buf, opts, resource_limits.as_ref());
        }

        if is_png(buf) {
            return Self::from_png_bytes_with_options(buf, opts, resource_limits.as_ref());
        }

        #[cfg(feature = "webp")]
        if is_webp(buf) {
            return Self::from_webp_bytes_with_options(buf, opts, resource_limits);
        }

        Self::from_registry_bytes_or_path(buf, path_hint, opts, resource_limits)
    }

    pub(in crate::adapters::image_api) fn build_scheduler(
        resource_limits: Option<&ResourceLimits>,
    ) -> Result<RayonScheduler, ViprsError> {
        let scheduler = RayonScheduler::new(RayonScheduler::default_threads())?;
        Ok(match resource_limits {
            Some(resource_limits) => {
                scheduler.with_execution_limiter(resource_limits.execution_limiter())
            }
            None => scheduler,
        })
    }

    pub(in crate::adapters::image_api) fn from_image_with_limits<F>(
        image: Image<F>,
        limits: Option<&DecodeLimits>,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError>
    where
        F: BandFormat,
        F::Sample: Pod,
    {
        if let Some(limits) = limits {
            limits.validate(
                image.width(),
                image.height(),
                image.bands(),
                size_of::<F::Sample>() as u32,
            )?;
        }
        let width = image.width();
        let height = image.height();
        let bands = image.bands();
        let metadata = image.metadata().clone();
        let source = MemorySource::<F>::new(width, height, bands, image.into_buffer())?
            .with_metadata(metadata);

        Ok(Self {
            builder: PipelineBuilder::from_source(source),
            resource_limits,
        })
    }

    #[allow(dead_code)]
    // REASON: this constructor is kept for future externally supplied source adapters.
    pub(in crate::adapters::image_api) fn from_source_with_limits(
        source: impl crate::ports::source::DynImageSource + 'static,
        bytes_per_sample: u32,
        limits: Option<&DecodeLimits>,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        validate_source_limits(
            source.width(),
            source.height(),
            source.bands(),
            bytes_per_sample,
            limits,
        )?;
        Ok(Self {
            builder: PipelineBuilder::from_source(source),
            resource_limits,
        })
    }

    pub(in crate::adapters::image_api) fn from_registry_bytes_or_path(
        buf: &[u8],
        path_hint: Option<&Path>,
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        let registry = ForeignRegistry::shared();
        match registry.load_from_memory_with_options(buf, opts) {
            Ok((image, _format_name)) => {
                Self::from_image_with_limits(image, opts.limits.as_ref(), resource_limits)
            }
            Err(memory_error) => {
                if let Some(path) = path_hint {
                    return match registry.load_with_options(path, opts) {
                        Ok(image) => Self::from_image_with_limits(
                            image,
                            opts.limits.as_ref(),
                            resource_limits,
                        ),
                        Err(path_error) => Err(map_image_api_decode_error(path_error)),
                    };
                }

                Err(map_image_api_decode_error(memory_error))
            }
        }
    }

    pub(in crate::adapters::image_api) fn from_registry_path(
        path: &Path,
        opts: &LoadOptions,
        resource_limits: Option<&ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        let image = ForeignRegistry::shared().load_with_options(path, opts)?;
        Self::from_image_with_limits(image, opts.limits.as_ref(), resource_limits.cloned())
    }

    pub(in crate::adapters::image_api) const fn from_jpeg_bytes_with_options(
        buf: &[u8],
        opts: &LoadOptions,
        resource_limits: Option<&ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        #[cfg(feature = "jpeg")]
        {
            let source = DecoderSource::<_, U8>::probed_shared_with_options(
                JpegCodec,
                Arc::<[u8]>::from(buf),
                opts.clone(),
            )?;
            return Self::from_source_with_limits(
                source,
                1,
                opts.limits.as_ref(),
                resource_limits.cloned(),
            );
        }

        #[cfg(not(feature = "jpeg"))]
        {
            let _ = opts;
            let _ = buf;
            let _ = resource_limits;
            Err(ViprsError::Unimplemented {
                feature: "image_api decode: jpeg",
                details: "enable Cargo feature `jpeg` to use ImageApi::from_bytes with JPEG input",
            })
        }
    }

    pub(in crate::adapters::image_api) fn from_png_bytes_with_options(
        buf: &[u8],
        opts: &LoadOptions,
        resource_limits: Option<&ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        #[cfg(feature = "png")]
        {
            let shared = Arc::<[u8]>::from(buf);
            return match png_bit_depth(buf) {
                Some(16) => Self::from_source_with_limits(
                    DecoderSource::<_, U16>::streaming_shared(
                        PngCodec::default(),
                        Arc::clone(&shared),
                        opts.clone(),
                    )?,
                    2,
                    opts.limits.as_ref(),
                    resource_limits.cloned(),
                ),
                Some(_) => Self::from_source_with_limits(
                    DecoderSource::<_, U8>::streaming_shared(
                        PngCodec::default(),
                        shared,
                        opts.clone(),
                    )?,
                    1,
                    opts.limits.as_ref(),
                    resource_limits.cloned(),
                ),
                None => Err(ViprsError::Codec("image_api: malformed PNG header".into())),
            };
        }

        #[cfg(not(feature = "png"))]
        {
            let _ = opts;
            let _ = buf;
            let _ = resource_limits;
            Err(ViprsError::Unimplemented {
                feature: "image_api decode: png",
                details: "enable Cargo feature `png` to use ImageApi::from_bytes with PNG input",
            })
        }
    }

    #[cfg(feature = "jpeg")]
    pub(in crate::adapters::image_api) fn from_jpeg_path_with_options(
        path: &Path,
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        let source =
            DecoderSource::<_, U8>::probed_path_with_options(JpegCodec, path, opts.clone())?;
        Self::from_source_with_limits(source, 1, opts.limits.as_ref(), resource_limits)
    }

    #[cfg(feature = "png")]
    pub(in crate::adapters::image_api) fn from_png_path_with_options(
        path: &Path,
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        match png_bit_depth(&fs::read(path)?) {
            Some(16) => Self::from_source_with_limits(
                DecoderSource::<_, U16>::streaming_path(PngCodec::default(), path, opts.clone())?,
                2,
                opts.limits.as_ref(),
                resource_limits,
            ),
            Some(_) => Self::from_source_with_limits(
                DecoderSource::<_, U8>::streaming_path(PngCodec::default(), path, opts.clone())?,
                1,
                opts.limits.as_ref(),
                resource_limits,
            ),
            None => Err(ViprsError::Codec("image_api: malformed PNG header".into())),
        }
    }

    #[cfg(feature = "webp")]
    pub(in crate::adapters::image_api) fn from_webp_bytes_with_options(
        buf: &[u8],
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        let source = DecoderSource::<_, U8>::streaming_shared(
            WebpCodec,
            Arc::<[u8]>::from(buf),
            opts.clone(),
        )?;
        Self::from_source_with_limits(source, 1, opts.limits.as_ref(), resource_limits)
    }

    #[cfg(feature = "webp")]
    pub(in crate::adapters::image_api) fn from_webp_path_with_options(
        path: &Path,
        opts: &LoadOptions,
        resource_limits: Option<ResourceLimits>,
    ) -> Result<Self, ViprsError> {
        let source = DecoderSource::<_, U8>::streaming_path(WebpCodec, path, opts.clone())?;
        Self::from_source_with_limits(source, 1, opts.limits.as_ref(), resource_limits)
    }

    pub(in crate::adapters::image_api) fn validate_output_limits(
        resource_limits: Option<&ResourceLimits>,
        pipeline: &CompiledPipeline,
    ) -> Result<(), ViprsError> {
        let Some(resource_limits) = resource_limits else {
            return Ok(());
        };
        resource_limits.validate_output(
            pipeline.width,
            pipeline.height,
            pipeline.output_bands,
            output_bytes_per_sample(pipeline.output_format),
        )
    }
}

impl ImageApiLoader {
    /// Open an image from a file path with shared resource limits.
    ///
    /// This mirrors [`ImageApi::open`] while consistently applying the loader's
    /// configured decode and execution budgets.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs::{
    ///     adapters::image_api::ImageApi,
    ///     domain::limits::ResourceLimits,
    /// };
    ///
    /// let loader = ImageApi::with_limits(ResourceLimits::default());
    /// let _image = loader.open("photo.jpg")?;
    /// # Ok::<(), viprs::domain::error::ViprsError>(())
    /// ```
    pub fn open(&self, path: impl AsRef<Path>) -> Result<ImageApi, ViprsError> {
        ImageApi::open_with_options(
            path.as_ref(),
            &LoadOptions {
                limits: Some(self.limits.decode_limits()),
                ..LoadOptions::default()
            },
            Some(&self.limits),
        )
    }

    /// Decode an image from any `Read` source with shared resource limits.
    ///
    /// Use this when uploads arrive as readers and the same service-level resource
    /// policy must be enforced for every request.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::io::Cursor;
    /// use viprs::{
    ///     adapters::image_api::ImageApi,
    ///     domain::limits::ResourceLimits,
    /// };
    ///
    /// let loader = ImageApi::with_limits(ResourceLimits::default());
    /// let _ = loader.from_reader(Cursor::new(Vec::<u8>::new()));
    /// ```
    pub fn from_reader<R: Read>(&self, mut reader: R) -> Result<ImageApi, ViprsError> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf)?;
        self.from_bytes(&buf)
    }

    /// Decode bytes into an in-memory source using the configured limits.
    ///
    /// This mirrors [`ImageApi::from_bytes`] while ensuring decode size and
    /// scheduler concurrency stay within the loader's configured limits.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs::{
    ///     adapters::image_api::ImageApi,
    ///     domain::limits::ResourceLimits,
    /// };
    ///
    /// let loader = ImageApi::with_limits(ResourceLimits::default());
    /// let _ = loader.from_bytes(&[]);
    /// ```
    pub fn from_bytes(&self, buf: &[u8]) -> Result<ImageApi, ViprsError> {
        ImageApi::from_bytes_with_options(
            buf,
            None,
            &LoadOptions {
                limits: Some(self.limits.decode_limits()),
                ..LoadOptions::default()
            },
            Some(self.limits.clone()),
        )
    }
}

#[inline]
fn is_jpeg(buf: &[u8]) -> bool {
    buf.starts_with(&JPEG_HEADER)
}

#[inline]
fn is_png(buf: &[u8]) -> bool {
    buf.starts_with(&PNG_HEADER)
}

#[inline]
#[cfg(feature = "webp")]
fn is_webp(buf: &[u8]) -> bool {
    buf.len() >= 12 && buf[..4] == WEBP_RIFF_HEADER && buf[8..12] == WEBP_MAGIC
}

#[inline]
#[cfg(feature = "png")]
fn png_bit_depth(buf: &[u8]) -> Option<u8> {
    buf.get(PNG_IHDR_BIT_DEPTH_OFFSET).copied()
}

fn map_image_api_decode_error(err: ViprsError) -> ViprsError {
    match err {
        ViprsError::Codec(message)
            if message.contains("no decoder matched") || message.contains("unsupported format") =>
        {
            ViprsError::Codec("image_api: unsupported input format".into())
        }
        other => other,
    }
}

#[inline]
const fn output_bytes_per_sample(format: BandFormatId) -> u32 {
    match format {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

#[allow(dead_code)]
// REASON: shared decode-limit validation stays available for future source constructors.
fn validate_source_limits(
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: u32,
    limits: Option<&DecodeLimits>,
) -> Result<(), ViprsError> {
    if let Some(limits) = limits {
        limits.validate(width, height, bands, bytes_per_sample)?;
    }
    Ok(())
}

fn normalized_extension(path: &Path) -> Option<&str> {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|ext| {
            if ext.eq_ignore_ascii_case("jpg") {
                "jpg"
            } else if ext.eq_ignore_ascii_case("jpeg") {
                "jpeg"
            } else if ext.eq_ignore_ascii_case("png") {
                "png"
            } else if ext.eq_ignore_ascii_case("webp") {
                "webp"
            } else {
                ext
            }
        })
}
