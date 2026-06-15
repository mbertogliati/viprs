#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RhsLayout {
    Direct,
    Scalar,
    PerBand,
    SingleBandImage,
}

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
