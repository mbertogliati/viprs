/// Public output destination vocabulary for pipeline execution.
///
/// The current skeleton supports explicit in-memory output. Writer and path
/// sinks will extend this type without changing `ImagePipeline::run_blocking`.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::image_pipeline::Sink;
///
/// let sink = Sink::memory();
/// assert!(sink.is_memory());
/// ```
pub struct Sink {
    pub(in crate::image_pipeline) kind: SinkKind,
}

pub(in crate::image_pipeline) enum SinkKind {
    Memory,
}

impl Sink {
    /// Create an in-memory sink that returns rendered bytes.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// let sink = Sink::memory();
    /// assert!(sink.is_memory());
    /// ```
    #[must_use]
    pub const fn memory() -> Self {
        Self {
            kind: SinkKind::Memory,
        }
    }

    /// Return whether this sink stores output in memory.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// assert!(Sink::memory().is_memory());
    /// ```
    #[must_use]
    pub const fn is_memory(&self) -> bool {
        matches!(self.kind, SinkKind::Memory)
    }
}
