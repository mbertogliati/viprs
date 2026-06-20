//! Core domain types and traits for viprs.
//!
//! Non-ops domain types are provided by the `viprs-core` crate and re-exported here
//! for backward compatibility.

// Re-export everything from viprs-core
pub use viprs_core::cancel;
pub use viprs_core::codec_options;
pub use viprs_core::coeff;
pub use viprs_core::colorspace;
pub use viprs_core::colour;
/// Runtime routing for dynamic colorspace-conversion graphs.
pub mod colour_dispatcher;
pub mod concretize;
pub use viprs_core::draw;
pub use viprs_core::error;
pub use viprs_core::format;
pub use viprs_core::image;
pub use viprs_core::kernel;
pub use viprs_core::limits;
pub use viprs_core::op;
pub use viprs_core::reducer;
pub use viprs_core::reorder;
pub use viprs_core::resample;
pub use viprs_core::simd;
pub use viprs_core::stats;

// Still owned by this crate directly
pub mod ops;
pub mod reducers;

pub use op::DemandHint;
