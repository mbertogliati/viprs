use std::future::{Ready, ready};

use crate::{
    pipeline::PipelineBuilder, ports::scheduler::TileScheduler, sinks::memory::MemorySink,
};
use viprs_core::error::{BuildError, ViprsError};

use super::{
    Format, Input, PipelineOutput, ProcessingConfig,
    sink::{Sink, SinkKind},
};

/// Public image pipeline builder and execution object.
///
/// The type owns a lazy pipeline description. Transform methods add work; only
/// `run`, `run_with`, or `run_blocking` execute it.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
///
/// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
///     .run_blocking(Sink::memory())?;
/// assert!(!output.as_bytes().is_empty());
/// # Ok::<(), viprs_core::error::ViprsError>(())
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
    ///     .run_blocking(Sink::memory())?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn invert(self) -> Result<Self, BuildError> {
        Ok(Self {
            builder: self.builder.invert()?.flush_into_identity()?,
        })
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

    /// Execute the pipeline with the default processing config.
    ///
    /// This async-shaped method is intentionally thin in the first skeleton; it
    /// executes through the same blocking engine as [`Self::run_blocking`].
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when building, scheduling, or writing output fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example() -> Result<(), viprs_core::error::ViprsError> {
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
    ///
    /// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .run(Sink::memory())
    ///     .await?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(self, sink: Sink) -> Ready<Result<PipelineOutput, ViprsError>> {
        ready(self.run_blocking(sink))
    }

    /// Execute the pipeline with explicit processing config.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when config validation, building, scheduling, or
    /// writing output fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example() -> Result<(), viprs_core::error::ViprsError> {
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, ProcessingConfig, Sink};
    ///
    /// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .run_with(ProcessingConfig::default().with_threads(1), Sink::memory())
    ///     .await?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run_with(
        self,
        config: ProcessingConfig,
        sink: Sink,
    ) -> Ready<Result<PipelineOutput, ViprsError>> {
        ready(self.run_with_blocking(config, sink))
    }

    /// Execute the pipeline synchronously with default processing config.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when building, scheduling, or writing output fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input, Sink};
    ///
    /// let output = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .run_blocking(Sink::memory())?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn run_blocking(self, sink: Sink) -> Result<PipelineOutput, ViprsError> {
        self.run_with_blocking(ProcessingConfig::default(), sink)
    }

    pub(in crate::image_pipeline) fn run_with_blocking(
        self,
        config: ProcessingConfig,
        sink: Sink,
    ) -> Result<PipelineOutput, ViprsError> {
        let Sink { kind } = sink;
        match kind {
            SinkKind::Memory => {
                let pipeline = self.builder.build()?;
                config.validate_output(
                    pipeline.width,
                    pipeline.height,
                    pipeline.output_bands,
                    Format::from(pipeline.output_format).bytes_per_sample() as u32,
                )?;
                let scheduler = config.into_scheduler()?;
                let mut memory_sink = MemorySink::for_pipeline(&pipeline)?;
                scheduler.run(&pipeline, &mut memory_sink)?;
                Ok(PipelineOutput::from_parts(
                    pipeline.width,
                    pipeline.height,
                    pipeline.output_bands,
                    Format::from(pipeline.output_format),
                    memory_sink.into_buffer(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ImagePipeline, Input, ProcessingConfig, Sink};
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
            .run_blocking(Sink::memory())
            .unwrap();
        let inverted = ImagePipeline::from_input(Input::path(image_fixture("sample.jpg")).unwrap())
            .invert()
            .unwrap()
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
            .run_blocking(Sink::memory())
            .unwrap();

        assert!(output.width() > 0);
        assert!(output.height() > 0);
        assert!(!output.as_bytes().is_empty());
    }

    #[test]
    fn explicit_memory_input_runs_memory_pipeline() {
        let output = ImagePipeline::from_input(Input::memory::<U8>(2, 1, 1, vec![10, 20]).unwrap())
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
            .run_blocking(Sink::memory())
            .unwrap();

        assert_eq!(output.as_bytes(), &[255, 0]);
    }

    #[test]
    fn run_with_uses_processing_config() {
        let output = ImagePipeline::from_input(Input::memory::<U8>(1, 1, 1, vec![42]).unwrap())
            .run_with_blocking(ProcessingConfig::default().with_threads(1), Sink::memory())
            .unwrap();

        assert_eq!(output.as_bytes(), &[42]);
    }
}
