//! Format-erased point operations for static fusion.
//!
//! Each struct describes WHAT to do without knowing the pixel format.
//! All implement `Concretize` — the trait that enables compile-time fusion.

mod abs;
mod boolean;
mod clamp;
mod gamma;
mod invert;
mod linear;
mod trig;

pub use abs::Abs;
pub use boolean::{BoolAnd, BoolOr, BoolXor, Lshift, Rshift};
pub use clamp::Clamp;
pub use gamma::Gamma;
pub use invert::Invert;
pub use linear::Linear;
pub use trig::{ACos, ASin, ATan, Ceil, Cos, Exp, Floor, Log, Power, Round, Sign, Sin, Sqrt, Tan};
