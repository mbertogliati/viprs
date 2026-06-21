//! Relational comparison operations that produce masks or sample-wise predicates.
/// Provides the `equal` module for this domain area.
pub mod equal;
/// Provides domain support for `less`.
pub mod less;
/// Provides the `less_eq` module for this domain area.
pub mod less_eq;
/// Provides the `more` module for this domain area.
pub mod more;
/// Provides the `more_eq` module for this domain area.
pub mod more_eq;
/// Provides the `not_equal` module for this domain area.
pub mod not_equal;

pub use equal::Equal;
pub use less::Less;
pub use less_eq::LessEq;
pub use more::More;
pub use more_eq::MoreEq;
pub use not_equal::NotEqual;

/// Sample bound for relational ops.
///
/// Libvips relational operations always produce uchar output (`0` or `255`), but
/// the input comparison rules vary with the source format. This trait keeps the
/// per-sample comparison bound local to the relational module.
pub trait CmpSample: Copy + PartialOrd + PartialEq {}

impl CmpSample for u8 {}

impl CmpSample for u16 {}

impl CmpSample for i16 {}

impl CmpSample for u32 {}

impl CmpSample for i32 {}

impl CmpSample for f32 {}

impl CmpSample for f64 {}
