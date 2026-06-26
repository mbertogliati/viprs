use std::{
    io::Write,
    path::{Path, PathBuf},
};

/// Public output destination vocabulary for pipeline execution.
///
/// Sinks describe where a selected output contract writes its bytes. For
/// [`crate::image_pipeline::RawOutputPipeline`], writer and path sinks receive
/// raw interleaved pixels; they do not imply image encoding.
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
    // Writer sinks are an I/O boundary: callers choose the concrete writer at
    // runtime, so static dispatch cannot name the type after it enters `Sink`.
    Writer(Box<dyn Write + Send>),
    Path(PathBuf),
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

    /// Create a sink that writes raw output bytes to an existing writer.
    ///
    /// The writer receives exactly the byte stream selected by the output
    /// contract. For `.raw_pixels()`, this means raw interleaved samples rather
    /// than an encoded image container.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// let sink = Sink::writer(Vec::<u8>::new());
    /// assert!(sink.is_writer());
    /// ```
    #[must_use]
    pub fn writer<W>(writer: W) -> Self
    where
        W: Write + Send + 'static,
    {
        Self {
            kind: SinkKind::Writer(Box::new(writer)),
        }
    }

    /// Create a filesystem sink that writes raw output bytes to a path.
    ///
    /// The file contains exactly the byte stream selected by the output
    /// contract. For `.raw_pixels()`, this means raw interleaved samples rather
    /// than an encoded image container.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// let sink = Sink::path("out.raw");
    /// assert!(sink.is_path());
    /// ```
    #[must_use]
    pub fn path(path: impl AsRef<Path>) -> Self {
        Self {
            kind: SinkKind::Path(path.as_ref().to_path_buf()),
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

    /// Return whether this sink writes output to a caller-provided writer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// assert!(Sink::writer(Vec::<u8>::new()).is_writer());
    /// ```
    #[must_use]
    pub const fn is_writer(&self) -> bool {
        matches!(self.kind, SinkKind::Writer(_))
    }

    /// Return whether this sink writes output to a filesystem path.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Sink;
    ///
    /// assert!(Sink::path("out.raw").is_path());
    /// ```
    #[must_use]
    pub const fn is_path(&self) -> bool {
        matches!(self.kind, SinkKind::Path(_))
    }
}
