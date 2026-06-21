/// Broadcast layouts supported by binary pixel operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhsLayout {
    /// The RHS already matches the full flattened input layout.
    Direct,
    /// The RHS is a single scalar broadcast across all samples.
    Scalar,
    /// The RHS contains one value per image band.
    PerBand,
    /// The RHS is a single-band image broadcast across bands.
    SingleBandImage,
}

/// Detects how a flattened RHS input should be broadcast across the LHS samples.
#[must_use]
pub const fn detect_rhs_layout(
    rhs_len: usize,
    sample_len: usize,
    bands: usize,
) -> Option<RhsLayout> {
    if rhs_len == sample_len {
        Some(RhsLayout::Direct)
    } else if rhs_len == 1 {
        Some(RhsLayout::Scalar)
    } else if rhs_len == bands {
        Some(RhsLayout::PerBand)
    } else if bands != 0 && rhs_len == sample_len / bands {
        Some(RhsLayout::SingleBandImage)
    } else {
        None
    }
}
