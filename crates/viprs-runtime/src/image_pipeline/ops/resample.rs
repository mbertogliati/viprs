use viprs_core::error::BuildError;

use super::super::{
    InterpolationKernel, Resize, Thumbnail,
    pipeline::{CommitState, Committed, ImagePipeline},
};

impl<State> ImagePipeline<State>
where
    State: CommitState,
{
    /// Reduce image width by a factor.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor or kernel is invalid.
    pub fn reduce_h(
        self,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.reduce_h(factor, kernel)?,
        ))
    }

    /// Shrink image width by an integer factor.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor is invalid.
    pub fn shrink_h(self, factor: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.shrink_h(factor)?,
        ))
    }

    /// Shrink image width by an integer factor with ceil rounding.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor is invalid.
    pub fn shrink_h_with_ceil(
        self,
        factor: u32,
        ceil: bool,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.shrink_h_with_ceil(factor, ceil)?,
        ))
    }

    /// Shrink both axes by integer factors.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factors are invalid.
    pub fn shrink(
        self,
        h_factor: u32,
        v_factor: u32,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.shrink(h_factor, v_factor)?,
        ))
    }

    /// Reduce image height by a factor.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor or kernel is invalid.
    pub fn reduce_v(
        self,
        factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.reduce_v(factor, kernel)?,
        ))
    }

    /// Reduce both axes by floating-point factors.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factors or kernel are invalid.
    pub fn reduce(
        self,
        h_factor: f64,
        v_factor: f64,
        kernel: InterpolationKernel,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.reduce(h_factor, v_factor, kernel)?,
        ))
    }

    /// Shrink image height by an integer factor.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor is invalid.
    pub fn shrink_v(self, factor: u32) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.shrink_v(factor)?,
        ))
    }

    /// Shrink image height by an integer factor with ceil rounding.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when factor is invalid.
    pub fn shrink_v_with_ceil(
        self,
        factor: u32,
        ceil: bool,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.shrink_v_with_ceil(factor, ceil)?,
        ))
    }

    /// Apply an affine transform.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the matrix, dimensions, or kernel are invalid.
    #[allow(clippy::too_many_arguments)]
    // REASON: Mirrors the existing libvips-style affine parameter contract.
    pub fn affine(
        self,
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        output_w: u32,
        output_h: u32,
        kernel: InterpolationKernel,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?
                .builder
                .affine(matrix, tx, ty, output_w, output_h, kernel)?,
        ))
    }

    /// Apply a similarity transform.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current pipeline cannot accept the operation.
    pub fn similarity(
        self,
        scale: f64,
        angle: f64,
        kernel: InterpolationKernel,
    ) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.similarity(scale, angle, kernel)?,
        ))
    }

    /// Resize the image using a prepared resize plan.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when resize planning or execution setup fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, InterpolationKernel, Resize};
    ///
    /// let resize = Resize::new(0.5, 0.5, InterpolationKernel::Lanczos3);
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?).resize(resize)?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn resize(self, resize: Resize) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.resize(resize)?,
        ))
    }

    /// Create a thumbnail using a prepared thumbnail plan.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when thumbnail planning or execution setup fails.
    pub fn thumbnail(self, thumbnail: Thumbnail) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(
            self.commit()?.builder.thumbnail(thumbnail)?,
        ))
    }
}
