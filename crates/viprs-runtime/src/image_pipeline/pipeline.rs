use std::marker::PhantomData;

use viprs_core::error::BuildError;

use super::{DemandHint, Format, Input, RawOutputPipeline};
use crate::pipeline::{
    CommitBuilderState, CommittedBuilderState,
    internal::{Fusing as BuilderFusing, PipelinePlan},
};

/// Public state for a pipeline with no pending fused point operations.
///
/// This is the default `ImagePipeline` state. It marks a pipeline whose staged
/// operation graph is ready for non-fusable operations or output contracts.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::{Committed, ImagePipeline, Input};
///
/// let pipeline: ImagePipeline<Committed> = ImagePipeline::from_input(Input::path("photo.jpg")?);
/// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
/// # Ok::<(), viprs_core::error::ViprsError>(())
/// ```
pub struct Committed;

/// Public state for a pipeline accumulating a statically fused point-operation chain.
///
/// The chain is materialized only when [`ImagePipeline::commit`] is called, when
/// a non-fusable operation is appended, or when an output contract is selected.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::{Fusing, ImagePipeline, Input};
/// use viprs_runtime::domain::ops::point::Invert;
///
/// let pipeline: ImagePipeline<Fusing<Invert>> =
///     ImagePipeline::from_input(Input::path("photo.jpg")?).invert()?;
/// let committed = pipeline.commit()?;
/// assert_eq!(committed.output_format(), viprs_runtime::image_pipeline::Format::U8);
/// # Ok::<(), viprs_core::error::ViprsError>(())
/// ```
pub type Fusing<C> = BuilderFusing<C>;

/// Capability for an `ImagePipeline` state that can be committed.
///
/// The public API uses this trait as the boundary between typestate-preserving
/// fusable operations and operations that require a concrete planned graph.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::{CommitState, Committed};
///
/// fn accepts_committable<S: CommitState>() {}
/// accepts_committable::<Committed>();
/// ```
pub trait CommitState: private::Sealed {
    #[doc(hidden)]
    type BuilderState: CommitBuilderState;
}

impl CommitState for Committed {
    type BuilderState = CommittedBuilderState;
}

impl<C> CommitState for Fusing<C>
where
    C: viprs_core::concretize::Concretize + Clone,
{
    type BuilderState = Self;
}

mod private {
    pub trait Sealed {}

    impl Sealed for super::Committed {}

    impl<C> Sealed for super::Fusing<C> where C: viprs_core::concretize::Concretize {}
}

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
pub struct ImagePipeline<State = Committed>
where
    State: CommitState,
{
    pub(super) builder: PipelinePlan<State::BuilderState>,
    state: PhantomData<State>,
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
        Self::from_builder(input.into_builder())
    }
}

impl<State> ImagePipeline<State>
where
    State: CommitState,
{
    pub(super) const fn from_builder(builder: PipelinePlan<State::BuilderState>) -> Self {
        Self {
            builder,
            state: PhantomData,
        }
    }

    /// Commit any pending fused point operations into the internal execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError`] when pending operations cannot be materialized for the
    /// current image format or band count.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .invert()?
    ///     .commit()?;
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn commit(self) -> Result<ImagePipeline<Committed>, BuildError> {
        Ok(ImagePipeline::from_builder(self.builder.commit_plan()?))
    }

    /// Declare the colorspace of the current pipeline stage.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::{ColorspaceId, ImagePipeline, Input};
    ///
    /// let pipeline = ImagePipeline::from_input(Input::path("photo.jpg")?)
    ///     .with_colorspace(ColorspaceId::SRgb);
    /// assert_eq!(pipeline.output_format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub fn with_colorspace(mut self, colorspace: viprs_core::colorspace::ColorspaceId) -> Self {
        self.builder = self.builder.with_colorspace(colorspace);
        self
    }

    /// Enable sequential access for the internal planner.
    ///
    /// This is a scheduling hint, not an image operation.
    #[must_use]
    pub fn with_sequential_access(mut self, sequential: bool) -> Self {
        self.builder = self.builder.with_sequential_access(sequential);
        self
    }

    /// Enable libvips-style sequential streaming.
    ///
    /// This is a scheduling hint, not an image operation.
    #[must_use]
    pub fn sequential(mut self, lines_ahead: usize) -> Self {
        self.builder = self.builder.sequential(lines_ahead);
        self
    }

    /// Enable a bounded scanline cache.
    ///
    /// This is a scheduling hint, not an image operation.
    #[must_use]
    pub fn linecache(mut self, lines_ahead: usize) -> Self {
        self.builder = self.builder.linecache(lines_ahead);
        self
    }

    /// Override demand hint selection for the internal planner.
    ///
    /// This is a scheduling hint, not an image operation.
    #[must_use]
    pub fn with_demand_hint_override(mut self, demand_hint: DemandHint) -> Self {
        self.builder = self.builder.with_demand_hint_override(demand_hint);
        self
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
        RawOutputPipeline::from_builder(self.commit().map(|pipeline| pipeline.builder))
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
