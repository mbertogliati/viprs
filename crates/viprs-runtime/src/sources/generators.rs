//! Generators image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

pub mod black;
mod common;
pub mod eye;
pub mod gaussmat;
pub mod gaussnoise;
pub mod grey;
pub mod identity;
pub mod sines;
pub mod text;
pub mod xyz;
pub mod zone;

pub use black::BlackSource;
pub use eye::EyeSource;
pub use gaussmat::{GaussPrecision, GaussmatSource};
pub use gaussnoise::GaussnoiseSource;
pub use grey::GreySource;
pub use identity::IdentitySource;
pub use sines::SinesSource;
pub use text::TextSource;
pub use xyz::XyzSource;
pub use zone::ZoneSource;
