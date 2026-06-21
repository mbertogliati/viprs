//! Morphological operations such as dilation, erosion, and rank-based filters.
pub mod close;
/// Provides domain support for `count_lines`.
pub mod count_lines;
pub mod dilate;
pub mod erode;
/// Provides domain support for `labelregions`.
pub mod labelregions;
pub mod median;
/// Provides the `nearest` module for this domain area.
pub mod nearest;
pub mod open;
/// Provides the `rank` module for this domain area.
pub mod rank;

pub use close::Close;
pub use count_lines::CountLinesOp;
pub use dilate::Dilate;
pub use erode::Erode;
pub use labelregions::LabelRegionsOp;
pub use median::Median;
pub use nearest::NearestOp;
pub use open::Open;
pub use rank::RankOp;
