use viprs_core::{colorspace::Colorspace, error::BuildError};

use super::super::pipeline::{CommitState, Committed, ImagePipeline};

impl<State> ImagePipeline<State>
where
    State: CommitState,
{
    /// Convert the current image to a target colorspace.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the source colorspace is unknown or no route exists.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    /// use viprs_core::colorspace::SRgb;
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?).colourspace::<SRgb>()?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn colourspace<To: Colorspace>(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.colourspace::<To>()?,
        ))
    }

    /// Insert an ICC-managed normalization stage to sRGB when needed.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when ICC normalization setup fails.
    #[cfg(feature = "icc")]
    pub fn normalize_to_srgb(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.normalize_to_srgb()?,
        ))
    }

    /// Premultiply alpha.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format is unsupported.
    pub fn premultiply(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.premultiply()?,
        ))
    }

    /// Unpremultiply alpha.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format is unsupported.
    pub fn unpremultiply(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.unpremultiply()?,
        ))
    }

    /// Flatten alpha against a background color.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format is unsupported.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("overlay.png")?)
    ///     .flatten([255.0, 255.0, 255.0, 255.0])?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn flatten(self, background: [f32; 4]) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.flatten(background)?,
        ))
    }
}
