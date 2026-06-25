use super::Format;

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
