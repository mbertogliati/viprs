//! Backward-compatible re-exports for the foreign codec registry.
//!
//! This module preserves the historical `adapters::foreign` entrypoint while
//! forwarding callers to the codec registry and codec trait definitions that now
//! live in `adapters::codecs` and `ports::codec`.

/// Runtime codec registry used to load and save images by sniffed format or file extension.
///
/// This re-export keeps older call sites working while pointing them at the
/// canonical registry implementation in `adapters::codecs::registry`.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs::adapters::foreign::ForeignRegistry;
///
/// let registry = ForeignRegistry::shared();
/// let _ = registry;
/// ```
pub use crate::adapters::codecs::registry::ForeignRegistry;
/// Object-safe codec boundary implemented by concrete codec adapters.
///
/// Use this trait when a registry or plugin system must handle codecs selected
/// at runtime rather than through static dispatch.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs::adapters::foreign::ImageCodec;
///
/// fn accepts_runtime_codec(_codec: &dyn ImageCodec) {}
/// ```
pub use crate::ports::codec::ImageCodec;
