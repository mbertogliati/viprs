//! Structural operations that reshape image geometry or band organization.
/// Provides the `embed` module for this domain area.
pub mod embed;
/// Provides the `extract_area` module for this domain area.
pub mod extract_area;
pub mod flatten;
/// Provides the `flip_horizontal` module for this domain area.
pub mod flip_horizontal;
/// Provides the `flip_vertical` module for this domain area.
pub mod flip_vertical;
/// Provides the `insert` module for this domain area.
pub mod insert;
/// Provides the `join` module for this domain area.
pub mod join;
pub mod premultiply;
/// Provides the `replicate` module for this domain area.
pub mod replicate;
/// Provides the `rotate180` module for this domain area.
pub mod rotate180;
/// Provides the `rotate270` module for this domain area.
pub mod rotate270;
/// Provides the `rotate90` module for this domain area.
pub mod rotate90;
/// Provides the `subsample` module for this domain area.
pub mod subsample;
pub mod unpremultiply;
/// Provides the `zoom` module for this domain area.
pub mod zoom;

pub use embed::{Embed, ExtendMode};
pub use extract_area::ExtractArea;
pub use flatten::Flatten;
pub use flip_horizontal::FlipHorizontal;
pub use flip_vertical::FlipVertical;
pub use insert::Insert;
pub use join::{Join, JoinDirection};
pub use premultiply::Premultiply;
pub use replicate::Replicate;
pub use rotate90::Rotate90;
pub use rotate180::Rotate180;
pub use rotate270::Rotate270;
pub use subsample::Subsample;
pub use unpremultiply::Unpremultiply;
pub use zoom::Zoom;
