use crate::pipeline::PipelineBuilder;
use viprs_core::error::BuildError;

use super::{Format, Input, RawOutputPipeline};

/// Public image pipeline builder and execution object.
///
/// The type owns a lazy pipeline description. Transform methods add work; an
/// explicit output contract is required before execution.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
///
/// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
///     .raw_pixels()
///     .run_blocking(Sink::memory())?;
/// assert!(!output.as_bytes().is_empty());
/// # Ok::<(), viprs_core::error::ViprsError>(())
/// ```
///
/// ```compile_fail
/// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
///
/// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg").unwrap());
/// let _ = pipeline.run_blocking(Sink::memory());
/// ```
pub struct ImagePipeline {
    builder: PipelineBuilder,
}

impl ImagePipeline {
    /// Start a pipeline from a first-class input.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?);
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub fn from_input(input: Input) -> Self {
        Self {
            builder: input.into_builder(),
        }
    }

    /// Apply sample inversion to the pipeline.
    ///
    /// This is the first operation method on the new public surface; later
    /// issues migrate the rest of the operation vocabulary here.
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

    /// Select raw interleaved pixels as the pipeline output contract.
    ///
    /// This method does not execute the pipeline. It only makes the output
    /// materialization boundary explicit before `run`, `run_with`, or
    /// `run_blocking`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
    ///
    /// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .raw_pixels()
    ///     .run_blocking(Sink::memory())?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub fn raw_pixels(self) -> RawOutputPipeline {
        RawOutputPipeline::from_builder(self.builder)
    }

    /// Return the current pipeline output format.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{Format, ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?);
    /// assert_eq!(pipeline.output_format(), Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub fn output_format(&self) -> Format {
        Format::from(self.builder.current_format())
    }
}

#[cfg(test)]
mod tests {
    use super::{ImagePipeline, Input};
    use crate::image_pipeline::{ProcessingConfig, Sink};
    use viprs_core::format::U8;

    fn image_fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/images")
            .join(name)
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn path_input_runs_without_predecoded_pixel_input() {
        let input = Input::path(image_fixture("sample.jpg")).unwrap();
        assert!(input.path_ref().is_some());

        let output = ImagePipeline::from_input(input)
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert!(output.width() > 0);
        assert!(output.height() > 0);
        assert_eq!(output.bands(), 3);
        assert!(!output.as_bytes().is_empty());
    }

    #[cfg(feature = "jpeg")]
    #[test]
    fn path_input_executes_existing_operation() {
        let original = ImagePipeline::from_input(Input::path(image_fixture("sample.jpg")).unwrap())
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();
        let inverted = ImagePipeline::from_input(Input::path(image_fixture("sample.jpg")).unwrap())
            .invert()
            .unwrap()
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert_eq!(original.width(), inverted.width());
        assert_eq!(original.height(), inverted.height());
        assert_ne!(original.as_bytes(), inverted.as_bytes());
    }

    #[cfg(feature = "png")]
    #[test]
    fn png_path_input_runs_without_predecoded_pixel_input() {
        let output = ImagePipeline::from_input(Input::path(image_fixture("sample.png")).unwrap())
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert!(output.width() > 0);
        assert!(output.height() > 0);
        assert!(!output.as_bytes().is_empty());
    }

    #[cfg(feature = "webp")]
    #[test]
    fn webp_path_input_runs_without_predecoded_pixel_input() {
        let output = ImagePipeline::from_input(Input::path(image_fixture("sample.webp")).unwrap())
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert!(output.width() > 0);
        assert!(output.height() > 0);
        assert!(!output.as_bytes().is_empty());
    }

    #[test]
    fn explicit_memory_input_runs_memory_pipeline() {
        let output = ImagePipeline::from_input(Input::memory::<U8>(2, 1, 1, vec![10, 20]).unwrap())
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert_eq!(output.as_bytes(), &[10, 20]);
        assert_eq!(output.width(), 2);
        assert_eq!(output.height(), 1);
        assert_eq!(output.bands(), 1);
    }

    #[test]
    fn explicit_memory_input_executes_existing_operation() {
        let output = ImagePipeline::from_input(Input::memory::<U8>(2, 1, 1, vec![0, 255]).unwrap())
            .invert()
            .unwrap()
            .raw_pixels()
            .run_blocking(Sink::memory())
            .unwrap();

        assert_eq!(output.as_bytes(), &[255, 0]);
    }

    #[test]
    fn run_with_uses_processing_config() {
        let output = ImagePipeline::from_input(Input::memory::<U8>(1, 1, 1, vec![42]).unwrap())
            .raw_pixels()
            .run_with_blocking(ProcessingConfig::default().with_threads(1), Sink::memory())
            .unwrap();

        assert_eq!(output.as_bytes(), &[42]);
    }
}
