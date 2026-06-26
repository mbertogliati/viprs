//! `DecoderSource<D>` — adapter that bridges [`ImageDecoder`] into [`ImageSource`].
//!
//! # Design rationale
//!
//! The codec and source traits are orthogonal by design: codecs convert compressed
//! streams into pixels; `ImageSource` supplies pixel tiles on demand to the pipeline
//! scheduler. Keeping them separate means codec implementors do not need to know
//! about pipeline nodes, and source implementors do not need format-specific parsing.
//!
//! `DecoderSource<D>` is the glue. Existing constructors eagerly decode a single
//! backing image via `D::decode_with_options` / `D::decode_path_with_options`
//! for backward compatibility. The `streaming*` constructors require
//! [`TileImageDecoder`] and retain only the compressed input or stable path plus
//! probed metadata, then decode each requested tile directly into the scheduler
//! output buffer.
//!
//! # Shrink-on-load
//!
//! Shrink-on-load is expressed through [`LoadOptions::shrink_factor`]. Eager
//! sources forward the hint to the decoder first; if the codec ignores it, the
//! source falls back to a post-decode shrink view over the resident backing
//! image. Streaming sources pass the normalized factor through probe/tile decode.
//!
//! # Access mode
//!
//! `DecoderSource<D>` is parameterised with a phantom `AccessMode` type so callers
//! can distinguish sequential and random-access loaders at compile time. Callers use
//! [`TileCache`](crate::sources::tile_cache::TileCache) to promote a
//! sequential source to random access when an operation requires arbitrary reads.
//!
//! ```text
//!   DecoderSource<JpegDecoder, Sequential>   → impl SequentialSource
//!   DecoderSource<TiffDecoder, RandomAccess> → impl RandomAccessSource
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::num::NonZeroU8;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::domain::codec_options::LoadOptions;
use crate::domain::error::ViprsError;
use crate::domain::format::BandFormat;
use crate::domain::image::{DemandHint, InMemoryImage, ImageMetadata, Region};
use crate::domain::ops::resample::shrinkh::ShrinkSample;
use crate::ports::codec::{ImageDecoder, ImageMetadataProbe, TileImageDecoder};
use crate::ports::source::ImageSource;

mod backing;
mod input;
mod shrink;
mod source_impl;

#[cfg(test)]
mod tests;

use self::backing::DecoderBacking;
use self::input::{
    DecodeRegionFn, ProbeInputFn, StableDecoderInput, decode_region_with, probe_input_with,
    streaming_backing_shrink_factor, streaming_eager_decode,
};
use self::shrink::{
    ThumbnailPreShrinkMode, checked_region_end, eager_backing_shrink_factor,
    eager_backing_shrink_factor_from_path, expected_output_len,
    materialize_residual_thumbnail_shrink, normalize_shrink_factor, normalize_streaming_options,
    retains_stable_input_for_thumbnail, shrunk_dimension, software_box_shrink_generic,
    thumbnail_pre_shrink_mode,
};

pub use self::backing::{DecoderSource, RandomAccess, Sequential};
pub use self::input::DecoderInput;
