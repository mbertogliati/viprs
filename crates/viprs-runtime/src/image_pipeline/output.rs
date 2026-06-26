use std::{
    fs::File,
    future::{Ready, ready},
    io::Write,
    path::Path,
};

use crate::{
    pipeline::internal::PipelinePlan, ports::scheduler::TileScheduler, sinks::memory::MemorySink,
};
use viprs_core::error::{BuildError, ViprsError};

use super::{Format, ProcessingConfig, Sink, sink::SinkKind};

/// Output-ready pipeline whose contract is raw interleaved pixels.
///
/// This type is returned by [`crate::image_pipeline::ImagePipeline::raw_pixels`].
/// It owns the terminal raw-pixel execution methods so an uncontracted pipeline
/// cannot be run directly.
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
pub struct RawOutputPipeline {
    builder: Result<PipelinePlan, BuildError>,
}

impl RawOutputPipeline {
    pub(in crate::image_pipeline) const fn from_builder(
        builder: Result<PipelinePlan, BuildError>,
    ) -> Self {
        Self { builder }
    }

    /// Execute the raw-pixel output contract with the default processing config.
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
    ///     .raw_pixels()
    ///     .run(Sink::memory())
    ///     .await?;
    /// assert!(!output.as_bytes().is_empty());
    /// # Ok(())
    /// # }
    /// ```
    pub fn run(self, sink: Sink) -> Ready<Result<PipelineOutput, ViprsError>> {
        ready(self.run_blocking(sink))
    }

    /// Execute the raw-pixel output contract with explicit processing config.
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
    ///     .raw_pixels()
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

    /// Execute the raw-pixel output contract synchronously.
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
    ///     .raw_pixels()
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
            SinkKind::Memory => self.run_memory(config),
            SinkKind::Writer(writer) => self.run_writer(config, writer),
            SinkKind::Path(path) => self.run_path(config, &path),
        }
    }

    fn run_memory(self, config: ProcessingConfig) -> Result<PipelineOutput, ViprsError> {
        self.render_raw(config)
    }

    fn run_writer(
        self,
        config: ProcessingConfig,
        mut writer: Box<dyn Write + Send>,
    ) -> Result<PipelineOutput, ViprsError> {
        let output = self.render_raw(config)?;
        writer.write_all(output.as_bytes())?;
        writer.flush()?;
        Ok(output)
    }

    fn run_path(self, config: ProcessingConfig, path: &Path) -> Result<PipelineOutput, ViprsError> {
        let output = self.render_raw(config)?;
        let mut file = File::create(path)?;
        file.write_all(output.as_bytes())?;
        file.flush()?;
        Ok(output)
    }

    fn render_raw(self, config: ProcessingConfig) -> Result<PipelineOutput, ViprsError> {
        let pipeline = self.builder?.compile()?;
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

/// Bytes produced by a completed pipeline run.
///
/// This is the minimal first output contract for the public skeleton. Future
/// output contracts will add encoded images and lazy values.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
///
/// let output = PipelineOutput::from_parts(1, 1, 1, Format::U8, vec![255]);
/// assert_eq!(output.as_bytes(), &[255]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineOutput {
    width: u32,
    height: u32,
    bands: u32,
    format: Format,
    bytes: Vec<u8>,
}

impl PipelineOutput {
    /// Build output bytes from validated execution parts.
    ///
    /// This constructor is mainly for tests and adapters that already validated
    /// the shape contract.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// let output = PipelineOutput::from_parts(1, 1, 1, Format::U8, vec![3]);
    /// assert_eq!(output.width(), 1);
    /// ```
    #[must_use]
    pub const fn from_parts(
        width: u32,
        height: u32,
        bands: u32,
        format: Format,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            width,
            height,
            bands,
            format,
            bytes,
        }
    }

    /// Return output width in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(2, 1, 1, Format::U8, vec![0, 1]).width(), 2);
    /// ```
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Return output height in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(1, 2, 1, Format::U8, vec![0, 1]).height(), 2);
    /// ```
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Return output band count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(1, 1, 3, Format::U8, vec![0, 1, 2]).bands(), 3);
    /// ```
    #[must_use]
    pub const fn bands(&self) -> u32 {
        self.bands
    }

    /// Return output sample format.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(1, 1, 1, Format::U8, vec![0]).format(), Format::U8);
    /// ```
    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    /// Borrow output bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(1, 1, 1, Format::U8, vec![9]).as_bytes(), &[9]);
    /// ```
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the output and return the owned bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::{Format, PipelineOutput};
    ///
    /// assert_eq!(PipelineOutput::from_parts(1, 1, 1, Format::U8, vec![9]).into_bytes(), vec![9]);
    /// ```
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}
