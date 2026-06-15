//! Reduction operators that summarize tiles into scalar values, profiles, and histograms.
//!
//! This module groups the domain reducers that collapse pixel data into measurements or
//! compact derived images while preserving the tile-based execution model used by viprs.

pub mod getpoint;
pub mod hist_equal;
pub mod histogram;
pub mod hough;
pub mod label_regions;
pub mod profile;
pub mod project;
pub mod stats;

pub use getpoint::GetpointReducer;
pub use hist_equal::HistEqualReducer;
pub use histogram::{HistFindNDimReducer, HistFindNDimResult, HistFindReducer};
pub use hough::{HoughCircleReducer, HoughLineReducer};
pub use label_regions::LabelRegionsReducer;
pub use profile::{ProfileOp, ProfileResult};
pub use project::{ProjectOp, ProjectResult};
pub use stats::{StatsReducer, image_avg, image_deviate, image_max, image_min};
