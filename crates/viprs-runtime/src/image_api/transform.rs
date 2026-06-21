use super::{
    BandFormatId, BuildError, DEFAULT_SHARPEN_M1, DEFAULT_SHARPEN_M2, DEFAULT_SHARPEN_SIGMA,
    DEFAULT_SHARPEN_X1, DEFAULT_SHARPEN_Y2, DEFAULT_SHARPEN_Y3, F32, F64, I16, I32, ImageApi,
    InterpolationKernel, PipelineOp, SmartcropOp, Thumbnail, ThumbnailTarget, U8, U16, U32,
    ViprsError,
};

#[cfg(feature = "icc")]
use super::ImageApiThumbnailOptions;

impl ImageApi {
    /// Apply any pipeline operation.
    ///
    /// This is the generic escape hatch for fluent chains when you already have a
    /// pipeline operation value and want the same ergonomic API.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::{image_api::ImageApi, domain::ops::point::Invert};
    ///
    /// let image = ImageApi::open("photo.jpg")?.apply(Invert)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn apply<O: PipelineOp>(self, op: O) -> Result<Self, BuildError> {
        Self::from_builder(
            self.builder.apply(op)?.flush_into_identity()?,
            self.resource_limits,
        )
    }

    /// Convenience invert operation.
    ///
    /// Use this for the common negative-image transform without constructing the
    /// lower-level operation manually.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.invert()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn invert(self) -> Result<Self, BuildError> {
        Self::from_builder(
            self.builder.invert()?.flush_into_identity()?,
            self.resource_limits,
        )
    }

    /// Convenience linear operation.
    ///
    /// This solves simple exposure/offset adjustments with a familiar
    /// `output = input * scale + offset` transform.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.linear(1.1, 2.0)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn linear(self, scale: f64, offset: f64) -> Result<Self, BuildError> {
        Self::from_builder(
            self.builder.linear(scale, offset)?.flush_into_identity()?,
            self.resource_limits,
        )
    }

    /// Alpha-composite the image onto a white background and drop the alpha band.
    ///
    /// This is the ergonomic way to prepare transparent inputs for formats that
    /// do not preserve alpha, such as baseline JPEG.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("logo.png")?.flatten()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn flatten(self) -> Result<Self, BuildError> {
        self.flatten_with(255, 255, 255)
    }

    /// Alpha-composite the image onto an RGB background and drop the alpha band.
    ///
    /// Use this when the output background must match brand or page colors
    /// instead of defaulting to white.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("logo.png")?.flatten_with(240, 240, 240)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn flatten_with(self, r: u8, g: u8, b: u8) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self
                .builder
                .flatten([
                    f32::from(r) / 255.0,
                    f32::from(g) / 255.0,
                    f32::from(b) / 255.0,
                    1.0,
                ])?
                .flush_into_identity()?,
            resource_limits: self.resource_limits,
        })
    }

    /// Multiply colour bands by alpha before resampling or compositing.
    ///
    /// Premultiplication avoids dark halos around transparent edges when later
    /// resampling or blending the image.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("overlay.png")?.premultiply()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn premultiply(self) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.premultiply()?.flush_into_identity()?,
            resource_limits: self.resource_limits,
        })
    }

    /// Divide colour bands by alpha after premultiplied processing.
    ///
    /// Use this to return to straight-alpha pixels after operations that were
    /// safer or more accurate in premultiplied form.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("overlay.png")?.premultiply()?.unpremultiply()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn unpremultiply(self) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.unpremultiply()?.flush_into_identity()?,
            resource_limits: self.resource_limits,
        })
    }

    /// Resize to a thumbnail width using the default Lanczos3 pipeline plan.
    ///
    /// This is the primary convenience method for web thumbnails and mirrors the
    /// common `thumbnail(width)` workflow from libvips-style APIs.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.thumbnail(400)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn thumbnail(self, width: u32) -> Result<Self, BuildError> {
        #[cfg(feature = "icc")]
        {
            self.thumbnail_with_options(width, ImageApiThumbnailOptions::default())
        }

        #[cfg(not(feature = "icc"))]
        Self::from_builder(
            self.builder.thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(width),
                InterpolationKernel::Lanczos3,
            ))?,
            self.resource_limits,
        )
    }

    /// Resize to a thumbnail width using explicit thumbnail-planning options.
    ///
    /// This solves the handful of workflows that need thumbnailing plus explicit
    /// profile normalization policy.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # #[cfg(feature = "icc")] {
    /// use viprs_runtime::image_api::{ImageApi, ImageApiThumbnailOptions};
    ///
    /// let options = ImageApiThumbnailOptions::default().with_auto_normalize_to_srgb(true);
    /// let image = ImageApi::open("photo.jpg")?.thumbnail_with_options(400, options)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// # }
    /// ```
    #[cfg(feature = "icc")]
    pub fn thumbnail_with_options(
        self,
        width: u32,
        options: ImageApiThumbnailOptions,
    ) -> Result<Self, BuildError> {
        let builder = self.builder.thumbnail(Thumbnail::new(
            ThumbnailTarget::Width(width),
            InterpolationKernel::Lanczos3,
        ))?;
        let builder = if options.auto_normalize_to_srgb {
            builder.normalize_to_srgb()?
        } else {
            builder
        };
        Self::from_builder(builder, self.resource_limits)
    }

    /// Explicitly normalize ICC-managed pixels to sRGB within the fluent pipeline.
    ///
    /// Use this when downstream consumers expect web-standard sRGB bytes even if
    /// the source embeds another color profile.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # #[cfg(feature = "icc")] {
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.normalize_to_srgb()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// # }
    /// ```
    #[cfg(feature = "icc")]
    pub fn normalize_to_srgb(self) -> Result<Self, BuildError> {
        Self::from_builder(self.builder.normalize_to_srgb()?, self.resource_limits)
    }

    /// Crop the current image to an attention-guided region of `width × height`.
    ///
    /// This convenience method materializes the current pipeline output, analyses it with
    /// the domain smartcrop scorer, then rebuilds the façade around an extracted-area view.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.smartcrop(300, 300)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn smartcrop(self, width: u32, height: u32) -> Result<Self, ViprsError> {
        let resource_limits = self.resource_limits.clone();
        let pipeline = self.builder.build()?;

        Self::validate_output_limits(resource_limits.as_ref(), &pipeline)?;
        let scheduler = Self::build_scheduler(resource_limits.as_ref())?;

        with_output_image!(pipeline, &scheduler, |image| {
            let crop = SmartcropOp::analyze(&image, width, height);
            let crop_width = width.min(image.width()).max(1);
            let crop_height = height.min(image.height()).max(1);
            let cropped = Self::from_image_with_limits(image, None, resource_limits)?;
            Ok(Self {
                builder: cropped.builder.extract_area(
                    crop.crop_left(),
                    crop.crop_top(),
                    crop_width,
                    crop_height,
                )?,
                resource_limits: cropped.resource_limits,
            })
        })
    }

    /// Convenience sharpen operation using libvips-compatible default parameters.
    ///
    /// This is the shortest path to a libvips-style sharpen without exposing the
    /// full parameter set.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.sharpen()?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn sharpen(self) -> Result<Self, BuildError> {
        self.sharpen_with(
            DEFAULT_SHARPEN_SIGMA,
            DEFAULT_SHARPEN_X1,
            DEFAULT_SHARPEN_Y2,
            DEFAULT_SHARPEN_Y3,
            DEFAULT_SHARPEN_M1,
            DEFAULT_SHARPEN_M2,
        )
    }

    /// Convenience sharpen operation with full control over the libvips sharpen parameters.
    ///
    /// Use this when you need to match libvips sharpen tuning from an existing
    /// pipeline or benchmark.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_api::ImageApi;
    ///
    /// let image = ImageApi::open("photo.jpg")?.sharpen_with(0.5, 2.0, 10.0, 20.0, 0.0, 3.0)?;
    /// let _ = image;
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn sharpen_with(
        self,
        sigma: f32,
        x1: f32,
        y2: f32,
        y3: f32,
        m1: f32,
        m2: f32,
    ) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.sharpen(sigma, x1, y2, y3, m1, m2)?,
            resource_limits: self.resource_limits,
        })
    }
}
