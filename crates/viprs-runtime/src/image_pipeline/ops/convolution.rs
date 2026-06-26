use viprs_core::error::BuildError;

use super::super::pipeline::ImagePipeline;

impl ImagePipeline {
    /// Apply a 2D convolution.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format or kernel is unsupported.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    ///
    /// let kernel = vec![vec![0.0, 1.0, 0.0], vec![1.0, 4.0, 1.0], vec![0.0, 1.0, 0.0]];
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?).conv2d(kernel)?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn conv2d(self, kernel: Vec<Vec<f64>>) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.conv2d(kernel)?,
        })
    }

    /// Apply a separable Gaussian blur.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when the current format is unsupported.
    pub fn gauss_blur(self, sigma: f32) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.gauss_blur(sigma)?,
        })
    }

    /// Apply libvips-style sharpen semantics.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when colorspace or format conversion is unsupported.
    #[allow(clippy::too_many_arguments)]
    // REASON: Mirrors the existing libvips-style sharpen parameter contract.
    pub fn sharpen(
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
        })
    }
}
