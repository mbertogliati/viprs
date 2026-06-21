//! Histogram analysis and histogram-driven image adjustment operations.
/// Provides the `case` module for this domain area.
pub mod case;
/// Provides the `clahe` module for this domain area.
pub mod clahe;
/// Provides the `hist_cum` module for this domain area.
pub mod hist_cum;
/// Provides the `hist_entropy` module for this domain area.
pub mod hist_entropy;
/// Provides the `hist_equal` module for this domain area.
pub mod hist_equal;
/// Provides domain support for `hist_find_facades`.
pub mod hist_find_facades;
/// Provides the `hist_find_indexed` module for this domain area.
pub mod hist_find_indexed;
/// Provides the `hist_ismonotonic` module for this domain area.
pub mod hist_ismonotonic;
/// Provides the `hist_match` module for this domain area.
pub mod hist_match;
/// Provides the `hist_norm` module for this domain area.
pub mod hist_norm;
/// Provides the `hist_percent` module for this domain area.
pub mod hist_percent;
/// Provides the `hist_plot` module for this domain area.
pub mod hist_plot;
/// Provides the `stdif` module for this domain area.
pub mod stdif;

pub use case::CaseOp;
pub use clahe::ClaheOp;
pub use hist_cum::HistCumOp;
pub use hist_entropy::HistEntropyOp;
pub use hist_equal::HistEqualOp;
pub use hist_find_facades::{HistFindOp, HistFindResult};
pub use hist_find_indexed::{HistFindIndexedOp, HistFindIndexedResult};
#[cfg(all(test, feature = "_integration"))]
pub(crate) mod hist_find_ndim_compat;
#[cfg(all(test, feature = "_integration"))]
pub(crate) use hist_find_ndim_compat::HistFindNDimOp;
pub use hist_ismonotonic::HistIsMonotonicOp;
pub use hist_match::HistMatchOp;
pub use hist_norm::{HistNormOp, HistNormTypedOp, hist_norm_promoted_format};
pub use hist_percent::HistPercentOp;
pub use hist_plot::HistPlotOp;
pub use stdif::StdifOp;
