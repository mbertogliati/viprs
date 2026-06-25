use viprs_core::format::BandFormatId;

/// Public sample format vocabulary for pipeline inputs and outputs.
///
/// This type keeps callers away from lower-level domain identifiers while still
/// mapping one-to-one to the engine's concrete band formats.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::image_pipeline::Format;
///
/// assert_eq!(Format::U8.bytes_per_sample(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// Unsigned 8-bit integer samples.
    U8,
    /// Unsigned 16-bit integer samples.
    U16,
    /// Signed 16-bit integer samples.
    I16,
    /// Unsigned 32-bit integer samples.
    U32,
    /// Signed 32-bit integer samples.
    I32,
    /// 32-bit floating point samples.
    F32,
    /// 64-bit floating point samples.
    F64,
}

impl Format {
    pub(in crate::image_pipeline) const fn from_id(format: BandFormatId) -> Self {
        match format {
            BandFormatId::U8 => Self::U8,
            BandFormatId::U16 => Self::U16,
            BandFormatId::I16 => Self::I16,
            BandFormatId::U32 => Self::U32,
            BandFormatId::I32 => Self::I32,
            BandFormatId::F32 => Self::F32,
            BandFormatId::F64 => Self::F64,
        }
    }

    /// Return the byte width of one sample in this format.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::image_pipeline::Format;
    ///
    /// assert_eq!(Format::F32.bytes_per_sample(), 4);
    /// ```
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

impl From<BandFormatId> for Format {
    fn from(format: BandFormatId) -> Self {
        Self::from_id(format)
    }
}
