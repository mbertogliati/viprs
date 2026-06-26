use viprs_core::error::BuildError;

use super::super::{BandFormatId, pipeline::ImagePipeline};

impl ImagePipeline {
    /// Apply sample inversion to the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline shape cannot accept the operation.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
    ///
    /// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .invert()?
    ///     .raw_pixels()
    ///     .run_blocking(Sink::memory())?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn invert(self) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.invert()?.flush_into_identity()?,
        })
    }

    /// Apply `output = input * scale + offset` per sample.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?).linear(1.2, -4.0)?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn linear(self, scale: f64, offset: f64) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.linear(scale, offset)?.flush_into_identity()?,
        })
    }

    /// Convert samples to another band format.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the conversion is unsupported.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{BandFormatId, Format, ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?).cast(BandFormatId::F32)?;
    /// assert_eq!(pipeline.output_format(), Format::F32);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn cast(self, target: BandFormatId) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.cast(target)?,
        })
    }

    /// Extract the most-significant byte from each integer band.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format is unsupported.
    pub fn msb(self) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.msb()?,
        })
    }
}
