//! Shared extend modes used by conversion and resampling operations.

/// Fill mode for pixels outside the embedded source image.
#[derive(Debug, Clone, PartialEq)]
pub enum ExtendMode {
    /// Fill with the zero value of the sample type.
    Black,
    /// Fill with the maximum white value for the sample type.
    White,
    /// Fill with a caller-supplied background vector.
    ///
    /// A single value is expanded to every band. Otherwise the vector length
    /// must match the image band count.
    Background(Vec<f64>),
    /// Replicate the nearest source edge pixel.
    Copy,
    /// Backwards-compatible alias for [`ExtendMode::Copy`].
    Edge,
    /// Tile the source periodically.
    Repeat,
    /// Tile the source periodically with every other tile mirrored.
    Mirror,
}
