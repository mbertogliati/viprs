//! Core domain types and traits for viprs.
//!
//! This module is the public entry point for the pure domain layer: image containers,
//! band formats, colorspaces, operation traits, and supporting runtime metadata.
//! Most callers import domain building blocks from here before assembling pipelines or
//! implementing new operations.

/// Cancellation primitives for stopping long-running pipeline work.
pub mod cancel;
pub mod codec_options;
pub mod coeff;
pub mod colorspace;
/// Colour-conversion traits and helpers shared across colour operations.
pub mod colour;
pub mod concretize;
pub mod draw;
/// Typed error enums used across the domain and adapter layers.
pub mod error;
/// Band-format traits, identifiers, and sample math helpers.
pub mod format;
/// Core image, region, and tile container types.
pub mod image;
pub mod kernel;
/// Resource-limit types and validation helpers.
pub mod limits;
pub mod op;
pub mod reducer;
pub mod reorder;
/// Resampling traits, filters, and high-level resize configuration.
pub mod resample;
/// Shared helper types extracted from operation crates to avoid cross-crate dependencies.
pub mod shared_ops;
/// SIMD abstraction helpers shared by performance-sensitive operations.
pub mod simd;
/// SIMD dispatch macros and function-pointer selectors used by split ops crates.
pub mod simd_util;
/// Aggregated image statistics produced by reducers.
pub mod stats;

pub use op::DemandHint;
