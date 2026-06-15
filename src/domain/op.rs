//! Operation port traits.
//!
//! Two complementary traits:
//! - `Op` — monomorphized, hot-path, NOT object-safe by design.
//! - `DynOperation` — object-safe, used in dynamic pipeline graphs.
//! - `OperationBridge<T>` — bridges a static `Op` to `DynOperation`.
//!
//! # Design rationale
//!
//! `Op` uses associated types (`type Input`, `type Output`) instead of a generic
//! parameter `F` so that each implementor fixes its own input/output formats. This
//! makes `Cast<From, To>` a first-class `Op` rather than a special case that must
//! bypass the bridge.

mod bridge;
mod demand;
mod dynamic_op;
mod static_op;
#[cfg(test)]
mod tests;

pub use bridge::{DynViewOp, OperationBridge, ViewBridge, ViewOp};
pub use demand::{CoordinateDrivenSourceSpec, DemandHint, NodeSpec, SourceReadPlan};
pub use dynamic_op::DynOperation;
pub use static_op::{Op, PixelLocalOp};
