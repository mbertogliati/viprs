//! Foreign codec registry: sniff-based load + extension-based save.
//!
//! This is the runtime plugin boundary for codecs. Concrete codec adapters
//! remain generic over `F: BandFormat`; the registry bridges them behind an
//! object-safe trait so callers can say `Image::<U8>::load("x.png")`.

mod bridges;
mod deepzoom;
mod deferred;
mod runtime;

#[cfg(all(test, feature = "_integration"))]
mod tests;

#[cfg(any(test, feature = "dcraw", feature = "openslide"))]
pub(crate) use bridges::boxed_extension_decoder;
pub(crate) use bridges::{boxed_codec, boxed_decoder};
pub use deepzoom::ImageCodecExt;
#[cfg(all(test, feature = "deepzoom"))]
pub(crate) use deepzoom::to_u8_image;
pub(crate) use deepzoom::{is_deepzoom_extension, save_deepzoom};
#[cfg(all(test, feature = "_integration"))]
pub(crate) use deferred::pdf_header_sniff;
pub(crate) use deferred::{deferred_decode_error, deferred_encode_error};
pub use runtime::ForeignRegistry;
#[cfg(all(test, feature = "_integration"))]
pub(crate) use runtime::read_header;
