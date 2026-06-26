use std::{
    io::Read,
    path::{Path, PathBuf},
};

use crate::{
    image_api::ImageApi, pipeline::internal::PipelinePlan, ports::source::ImageSource,
    sources::memory::MemorySource,
};
use viprs_core::{error::ViprsError, format::BandFormat};

use super::Format;

/// Public input vocabulary for constructing an `ImagePipeline`.
///
/// Path input is the primary API shape and delegates source selection to the
/// existing loader. Encoded byte input stages compressed data before handing it
/// to the same loader. In-memory input is an explicit boundary for tests and
/// callers that already own pixels.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::image_pipeline::Input;
///
/// let input = Input::path("photo.jpg")?;
/// assert_eq!(input.format(), viprs_runtime::image_pipeline::Format::U8);
/// # Ok::<(), viprs_core::error::ViprsError>(())
/// ```
pub struct Input {
    builder: PipelinePlan,
    path: Option<PathBuf>,
    width: u32,
    height: u32,
    bands: u32,
    format: Format,
}

impl Input {
    /// Create an input from a stable filesystem path.
    ///
    /// Source selection stays in the existing image loader so this vocabulary
    /// does not duplicate codec knowledge.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when the existing loader cannot open the path.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// let input = Input::path("photo.jpg")?;
    /// assert_eq!(input.format(), viprs_runtime::image_pipeline::Format::U8);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn path(path: impl AsRef<Path>) -> Result<Self, ViprsError> {
        let path = path.as_ref().to_path_buf();
        let builder = ImageApi::open(&path)?.into_pipeline_builder();
        Ok(Self::from_builder(builder, Some(path)))
    }

    /// Create an input from already-buffered encoded image bytes.
    ///
    /// This is an encoded input boundary for uploads, cache objects, and other
    /// callers that already staged compressed bytes in memory. The existing
    /// loader promotes memory-backed codec input into stable pipeline storage;
    /// this constructor does not accept decoded raster pixels.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when the existing loader cannot identify or plan a
    /// decoder for the encoded bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// let bytes = std::fs::read("photo.jpg")?;
    /// let input = Input::bytes(bytes)?;
    /// assert!(input.path_ref().is_none());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn bytes(bytes: impl AsRef<[u8]>) -> Result<Self, ViprsError> {
        let builder = ImageApi::from_bytes(bytes.as_ref())?.into_pipeline_builder();
        Ok(Self::from_builder(builder, None))
    }

    /// Create an input by fully buffering encoded bytes from a reader.
    ///
    /// This is not true streaming. The reader is drained into memory first, then
    /// the staged encoded bytes are handed to the same loader used by
    /// [`Input::bytes`]. Use it for APIs that expose `Read` boundaries while
    /// keeping the buffering cost explicit in this contract.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when reading fails or when the existing loader
    /// cannot identify or plan a decoder for the encoded bytes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::io::Cursor;
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// let bytes = std::fs::read("photo.jpg")?;
    /// let input = Input::reader(Cursor::new(bytes))?;
    /// assert!(input.path_ref().is_none());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn reader<R: Read>(mut reader: R) -> Result<Self, ViprsError> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Self::bytes(&bytes)
    }

    /// Create an in-memory input from row-major interleaved samples.
    ///
    /// This is an explicit memory boundary for tests and callers that already
    /// own decoded pixels. It is not the default file/codec path.
    ///
    /// # Errors
    ///
    /// Returns [`ViprsError`] when dimensions and buffer length do not match.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::format::U8;
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// let input = Input::memory::<U8>(2, 1, 1, vec![10, 20])?;
    /// assert_eq!(input.width(), 2);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    pub fn memory<F>(
        width: u32,
        height: u32,
        bands: u32,
        pixels: Vec<F::Sample>,
    ) -> Result<Self, ViprsError>
    where
        F: BandFormat,
    {
        let source = MemorySource::<F>::new(width, height, bands, pixels)?;
        Ok(Self::from_source(source, None))
    }

    /// Return the input width in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::format::U8;
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// assert_eq!(Input::memory::<U8>(1, 1, 1, vec![0])?.width(), 1);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Return the input height in pixels.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::format::U8;
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// assert_eq!(Input::memory::<U8>(1, 2, 1, vec![0, 1])?.height(), 2);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Return the input band count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::format::U8;
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// assert_eq!(Input::memory::<U8>(1, 1, 3, vec![0, 1, 2])?.bands(), 3);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub const fn bands(&self) -> u32 {
        self.bands
    }

    /// Return the input sample format.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_core::format::U16;
    /// use viprs_runtime::image_pipeline::{Format, Input};
    ///
    /// assert_eq!(Input::memory::<U16>(1, 1, 1, vec![0])?.format(), Format::U16);
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub const fn format(&self) -> Format {
        self.format
    }

    /// Return the stable path for path-backed inputs.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::image_pipeline::Input;
    ///
    /// let input = Input::path("photo.jpg")?;
    /// assert!(input.path_ref().is_some());
    /// # Ok::<(), viprs_core::error::ViprsError>(())
    /// ```
    #[must_use]
    pub fn path_ref(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub(in crate::image_pipeline) fn into_builder(self) -> PipelinePlan {
        self.builder
    }

    fn from_source<S>(source: S, path: Option<PathBuf>) -> Self
    where
        S: ImageSource + 'static,
    {
        Self::from_builder(PipelinePlan::from_source(source), path)
    }

    fn from_builder(builder: PipelinePlan, path: Option<PathBuf>) -> Self {
        let (width, height) = builder.current_dimensions();
        let bands = builder.current_bands();
        let format = Format::from(builder.current_format());
        Self {
            builder,
            path,
            width,
            height,
            bands,
            format,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::Input;

    fn image_fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/images")
            .join(name)
    }

    #[cfg(feature = "png")]
    #[test]
    fn bytes_input_uses_existing_encoded_decode_path() {
        let path = image_fixture("sample.png");
        let encoded = std::fs::read(&path).unwrap();
        let expected = Input::path(path).unwrap();

        let input = Input::bytes(encoded).unwrap();

        assert!(input.path_ref().is_none());
        assert_eq!(input.width(), expected.width());
        assert_eq!(input.height(), expected.height());
        assert_eq!(input.bands(), expected.bands());
        assert_eq!(input.format(), expected.format());
    }

    #[cfg(feature = "png")]
    #[test]
    fn reader_input_buffers_encoded_bytes_before_decode_planning() {
        let path = image_fixture("sample.png");
        let encoded = std::fs::read(&path).unwrap();
        let expected = Input::path(path).unwrap();

        let input = Input::reader(Cursor::new(encoded)).unwrap();

        assert!(input.path_ref().is_none());
        assert_eq!(input.width(), expected.width());
        assert_eq!(input.height(), expected.height());
        assert_eq!(input.bands(), expected.bands());
        assert_eq!(input.format(), expected.format());
    }
}
