//! Arithmetic operations that transform, combine, and measure pixel sample values.
/// Provides domain support for `abs`.
pub mod abs;
/// Provides domain support for `acos`.
pub mod acos;
/// Provides the `add` module for this domain area.
pub mod add;
/// Provides domain support for `add_images`.
pub mod add_images;
/// Provides the `asin` module for this domain area.
pub mod asin;
/// Provides the `atan` module for this domain area.
pub mod atan;
/// Provides the `clamp` module for this domain area.
pub mod clamp;
/// Provides the `complex_conj` module for this domain area.
pub mod complex_conj;
/// Provides the `complex_form` module for this domain area.
pub mod complex_form;
/// Provides the `complex_imag` module for this domain area.
pub mod complex_imag;
/// Provides the `complex_real` module for this domain area.
pub mod complex_real;
/// Provides domain support for `cos`.
pub mod cos;
/// Provides the `divide` module for this domain area.
pub mod divide;
/// Provides the `exp` module for this domain area.
pub mod exp;
/// Provides the `find_trim` module for this domain area.
pub mod find_trim;
/// Provides the `invert` module for this domain area.
pub mod invert;
/// Provides the `linear` module for this domain area.
pub mod linear;
/// Provides the `log` module for this domain area.
pub mod log;
/// Provides the `math2` module for this domain area.
pub mod math2;
/// Provides the `maxpair` module for this domain area.
pub mod maxpair;
/// Provides the `measure` module for this domain area.
pub mod measure;
/// Provides the `minpair` module for this domain area.
pub mod minpair;
/// Provides the `multiply` module for this domain area.
pub mod multiply;
/// Provides the `polar` module for this domain area.
pub mod polar;
/// Provides the `power` module for this domain area.
pub mod power;
/// Provides the `recomb` module for this domain area.
pub mod recomb;
/// Provides the `rect` module for this domain area.
pub mod rect;
/// Provides domain support for `reduce_facades`.
pub mod reduce_facades;
/// Provides the `remainder` module for this domain area.
pub mod remainder;
pub(crate) mod rhs_broadcast;
/// Provides the `round` module for this domain area.
pub mod round;
/// Provides the `sign` module for this domain area.
pub mod sign;
/// Provides the `sin` module for this domain area.
pub mod sin;
/// Provides the `sqrt` module for this domain area.
pub mod sqrt;
/// Provides the `subtract` module for this domain area.
pub mod subtract;
/// Provides the `sum` module for this domain area.
pub mod sum;
/// Provides the `tan` module for this domain area.
pub mod tan;

pub use abs::Abs;
pub use acos::ACos;
pub use add::Add;
pub use add_images::AddImages;
pub use asin::ASin;
pub use atan::ATan;
pub use clamp::ClampOp;
pub use complex_conj::ComplexConjOp;
pub use complex_form::ComplexFormOp;
pub use complex_imag::ComplexImagOp;
pub use complex_real::ComplexRealOp;
pub use cos::Cos;
pub use divide::Divide;
pub use exp::Exp;
pub use find_trim::{FindTrimOp, TrimBox};
pub use invert::Invert;
pub use linear::{Linear, LinearKernelU8};
pub use log::Log;
pub use math2::{Math2, Math2Mode};
pub use maxpair::MaxPair;
pub use measure::{MeasureOp, MeasureResult};
pub use minpair::MinPair;
pub use multiply::Multiply;
pub use polar::PolarOp;
pub use power::Power;
pub use recomb::{Matrix, Recomb, Recomb64, RecombOp};
pub use rect::RectOp;
pub use reduce_facades::{
    AvgOp, DeviateOp, ExtremaResult, GetpointOp, MaxOp, MinOp, ScalarStats, StatsOp, StatsResult,
    StatsRow,
};
pub use remainder::Remainder;
pub use round::{Ceil, Floor, Round};
pub use sign::Sign;
pub use sin::Sin;
pub use sqrt::Sqrt;
pub use subtract::Subtract;
pub use sum::SumOp;
pub use tan::Tan;
