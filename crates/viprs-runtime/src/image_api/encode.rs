#[cfg(feature = "jpeg")]
use super::JpegCodec;
#[cfg(feature = "png")]
use super::PngCodec;
#[cfg(feature = "webp")]
use super::WebpCodec;
use super::{
    BandFormatId, F32, F64, ForeignRegistry, I16, I32, ImagePipeline2, Path, U8, U16, U32, ViprsError,
    Write,
};
#[cfg(any(feature = "jpeg", feature = "png", feature = "webp"))]
use super::{ImageEncoder, SaveOptions};

impl ImagePipeline2 {
    /// Build, run, and encode the output as JPEG.
    ///
    /// This solves HTTP and background-job workflows that need encoded bytes
    /// instead of writing directly to a file.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let jpeg = ImagePipeline2::open("photo.png")?.flatten()?.encode_jpeg(85)?;
    /// let _ = jpeg;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_jpeg(self, quality: u8) -> Result<Vec<u8>, ViprsError> {
        #[cfg(feature = "jpeg")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;
            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = JpegCodec;
            let options = SaveOptions {
                quality: Some(quality),
                ..SaveOptions::default()
            };

            with_output_image!(pipeline, &scheduler, |image| {
                codec.encode_with_options(&image, &options)
            })
        }

        #[cfg(not(feature = "jpeg"))]
        {
            let _ = quality;
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: jpeg",
                details: "enable Cargo feature `jpeg` to use ImageApi::encode_jpeg",
            })
        }
    }

    /// Build, run, and stream the output as JPEG into `writer`.
    ///
    /// Use this to avoid a second full output buffer when a caller already owns a
    /// writer such as an HTTP response body or file handle.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let mut bytes = Vec::new();
    /// ImagePipeline2::open("photo.png")?.flatten()?.encode_jpeg_to(&mut bytes, 85)?;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_jpeg_to<W: Write>(self, writer: &mut W, quality: u8) -> Result<(), ViprsError> {
        #[cfg(feature = "jpeg")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;

            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = JpegCodec;
            let options = SaveOptions {
                quality: Some(quality),
                ..SaveOptions::default()
            };

            with_output_image!(pipeline, &scheduler, |image| {
                codec.encode_to_writer(&image, &options, writer)
            })
        }

        #[cfg(not(feature = "jpeg"))]
        {
            let _ = writer;
            let _ = quality;
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: jpeg",
                details: "enable Cargo feature `jpeg` to use ImageApi::encode_jpeg_to",
            })
        }
    }

    /// Build, run, and encode the output as PNG.
    ///
    /// This returns the final encoded bytes while preserving lossless output and
    /// alpha when the codec feature is enabled.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let png = ImagePipeline2::open("photo.jpg")?.encode_png()?;
    /// let _ = png;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_png(self) -> Result<Vec<u8>, ViprsError> {
        #[cfg(feature = "png")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;
            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = PngCodec::default();

            with_output_image!(pipeline, &scheduler, |image| codec.encode(&image))
        }

        #[cfg(not(feature = "png"))]
        {
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: png",
                details: "enable Cargo feature `png` to use ImageApi::encode_png",
            })
        }
    }

    /// Build, run, and write the output as PNG into `writer`.
    ///
    /// PNG output is exposed through the same streaming surface, but the codec may
    /// still buffer internally depending on the chosen options.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let mut bytes = Vec::new();
    /// ImagePipeline2::open("photo.jpg")?.encode_png_to(&mut bytes)?;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_png_to<W: Write>(self, writer: &mut W) -> Result<(), ViprsError> {
        #[cfg(feature = "png")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;

            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = PngCodec::default();
            let options = SaveOptions::default();

            with_output_image!(pipeline, &scheduler, |image| {
                codec.encode_to_writer(&image, &options, writer)
            })
        }

        #[cfg(not(feature = "png"))]
        {
            let _ = writer;
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: png",
                details: "enable Cargo feature `png` to use ImageApi::encode_png_to",
            })
        }
    }

    /// Build, run, and encode the output as WebP.
    ///
    /// Use this for compact web delivery when the WebP feature is enabled and the
    /// caller needs an owned byte buffer.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let webp = ImagePipeline2::open("photo.jpg")?.encode_webp(80)?;
    /// let _ = webp;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_webp(self, quality: u8) -> Result<Vec<u8>, ViprsError> {
        #[cfg(feature = "webp")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;
            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = WebpCodec;
            let options = SaveOptions {
                quality: Some(quality),
                ..SaveOptions::default()
            };

            with_output_image!(pipeline, &scheduler, |image| {
                codec.encode_with_options(&image, &options)
            })
        }

        #[cfg(not(feature = "webp"))]
        {
            let _ = quality;
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: webp",
                details: "enable Cargo feature `webp` to use ImageApi::encode_webp",
            })
        }
    }

    /// Build, run, and write the output as WebP into `writer`.
    ///
    /// WebP currently uses the buffered codec path internally, then forwards the
    /// final byte stream to `writer`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// let mut bytes = Vec::new();
    /// ImagePipeline2::open("photo.jpg")?.encode_webp_to(&mut bytes, 80)?;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn encode_webp_to<W: Write>(self, writer: &mut W, quality: u8) -> Result<(), ViprsError> {
        #[cfg(feature = "webp")]
        {
            let resource_limits = self.resource_limits.clone();
            let pipeline = self.builder.build()?;

            Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
            let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
            let codec = WebpCodec;
            let options = SaveOptions {
                quality: Some(quality),
                ..SaveOptions::default()
            };

            with_output_image!(pipeline, &scheduler, |image| {
                codec.encode_to_writer(&image, &options, writer)
            })
        }

        #[cfg(not(feature = "webp"))]
        {
            let _ = writer;
            let _ = quality;
            Err(ViprsError::Unimplemented {
                feature: "image_api encode: webp",
                details: "enable Cargo feature `webp` to use ImageApi::encode_webp_to",
            })
        }
    }

    /// Build, run, and save the output using the destination path extension.
    ///
    /// This is the simplest terminal operation for file-oriented workflows
    /// because it delegates encoder selection to the foreign registry.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImagePipeline2;
    ///
    /// ImagePipeline2::open("photo.jpg")?.thumbnail(400)?.save("thumb.webp")?;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn save(self, path: impl AsRef<Path>) -> Result<(), ViprsError> {
        let resource_limits = self.resource_limits.clone();
        let pipeline = self.builder.build()?;
        Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
        let scheduler = Self::build_scheduler(resource_limits.as_ref())?;
        let path = path.as_ref();

        with_output_image!(pipeline, &scheduler, |image| {
            ForeignRegistry::shared().save_as(&image, path)
        })
    }
}
