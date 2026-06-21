//! Shared operation helpers extracted from the split ops crates.

/// Type-level sample casts shared by conversion and colour operations.
pub mod cast_sample;
/// Rich extend modes shared by conversion and resampling operations.
pub mod extend_mode;
/// Gaussian-kernel construction shared by convolution and composite operations.
pub mod gauss_kernel;
/// Sample inversion trait and primitive implementations shared by pixel operations.
pub mod invertible;
/// Band recombination types shared by arithmetic and LUT operations.
pub mod recomb;
/// RHS broadcast-shape detection shared by arithmetic and boolean operations.
pub mod rhs_broadcast;
/// Sample conversion traits shared by convolution and resampling operations.
pub mod sample_conv;
