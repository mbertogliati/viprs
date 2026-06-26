//! `viprs` is a native Rust reimplementation of libvips with a demand-driven,
//! horizontally-threaded pipeline architecture.
//!
//! For common application workflows, the crate exposes a compact surface via
//! [`prelude`]: [`ImagePipeline`] + [`ViprsError`]. Power users can opt into the
//! explicit advanced surfaces under [`pipeline`], [`ops`], and [`codecs`].
//!
//! # Quick start
//!
//! ```no_run
//! # #[cfg(feature = "jpeg")]
//! # fn main() -> Result<(), viprs::ViprsError> {
//! use std::fs;
//! use viprs::{
//!     adapters::{
//!         codecs::JpegCodec,
//!         pipeline::ImagePipeline,
//!         scheduler::rayon_scheduler::RayonScheduler,
//!     },
//!     domain::{
//!         codec_options::SaveOptions,
//!         format::U8,
//!     },
//!     ports::codec::{ImageDecoder, ImageEncoder},
//!     sources::decoder_source::DecoderSource,
//! };
//!
//! let source = DecoderSource::<_, U8>::probed_path(JpegCodec, "input.jpg")?;
//! let pipeline = ImagePipeline::from_source(source)
//!     .thumbnail(400)?
//!     .invert()?
//!     .build()?;
//! let scheduler = RayonScheduler::new(RayonScheduler::default_threads())?;
//! let image = pipeline.run_to_image::<U8, _>(&scheduler)?;
//! let encoded = JpegCodec.encode_with_options(&image, &SaveOptions::default())?;
//! fs::write("thumb.jpg", encoded)?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "jpeg"))]
//! # fn main() {}
//! ```
//!
//! # Advanced surfaces
//!
//! - [`prelude`] for the fluent end-user API
//! - [`pipeline`] for explicit pipeline construction and execution internals
//! - [`ops`] for direct operation imports
//! - [`codecs`] for advanced encode/decode control
//!
//! # Feature flags
//!
//! - Default runtime features: `rayon`, `mmap`, `simd-pulp`
//! - Common codecs: `jpeg`, `png`, `webp`, `tiff`, `gif`
//! - Advanced/native integrations: `heif`, `avif`, `openslide`, `icc`, `jp2k`, `fft`
//!
//! Native codec flags intentionally preserve the existing C-backed integrations
//! for performance. See the repository `README.md` for the full feature matrix,
//! per-OS install commands, and runnable examples.
//!
//! Organized as a hexagon: `adapters/ -> ports/ <- domain/`.
//! This file contains only module declarations and public re-exports.

pub mod adapters;
pub mod domain;
#[cfg(feature = "ffi")]
pub mod ffi;
pub mod ports;
#[cfg(test)]
mod test_support;

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: test_support::CountingAllocator = test_support::CountingAllocator;

/// Minimal end-user import surface for the fluent image façade.
pub mod prelude {
    pub use crate::{ImageCodecExt, ImagePipeline, ResourceLimits, ViprsError};
}

/// Explicit advanced pipeline surface for manual graph construction and execution.
pub mod pipeline {
    pub use crate::adapters::pipeline::Commit;
    pub use crate::adapters::pipeline::{
        CompiledNode, CompiledOp, CompiledPipeline, ImagePipeline, InputSlicePtr, LineCacheConfig,
        PipelineArena, PipelineOp, ThreadBufferPool,
    };
    pub use crate::adapters::scheduler::rayon_scheduler::RayonScheduler;
    pub use crate::adapters::sinks::discard::DiscardSink;
    pub use crate::adapters::sinks::memory::MemorySink;
    pub use crate::adapters::sources::{memory::MemorySource, zero::ZeroSource};
    pub use crate::domain::error::BuildError;
    pub use crate::domain::image::DemandHint;
    pub use crate::domain::op::{
        DynOperation, DynViewOp, NodeSpec, OperationBridge, SourceReadPlan, ViewBridge,
    };
    pub use crate::ports::scheduler::{ReducingScheduler, TileScheduler};
    pub use crate::ports::sink::{ConcurrentSink, ImageSink};
    pub use crate::ports::source::{DynImageSource, ImageSource};
}

/// Direct operation namespace mirroring `domain::ops` for explicit composition.
pub mod ops {
    pub use crate::domain::ops::*;
}

/// Codec namespace for advanced decode / encode control.
pub mod codecs {
    pub use crate::ImageCodecExt;
    pub use crate::adapters::codecs::*;
    pub use crate::domain::codec_options::{LoadOptions, RawEndianness, SaveOptions};
    pub use crate::ports::codec::{ImageDecoder, ImageEncoder};
}

pub use adapters::codecs::registry::ImageCodecExt;
pub use adapters::sources;
pub use domain::error::ViprsError;
pub use domain::limits::{DecodeLimits, ResourceLimits};

#[cfg(feature = "fft")]
pub use adapters::freqfilt::{fwfft, invfft};
pub use adapters::pipeline::{CompiledPipeline, ImagePipeline};
pub use adapters::sources::{BlackSource, any::AnySource};
pub use domain::error::BuildError;
pub use domain::format::{BandFormat, BandFormatId, F32, F64, U8, U16};
pub use domain::image::{
    DemandHint, ImageMetadata, InMemoryImage, Interpretation, Region, Tile, TileMut,
};
pub use domain::op::{DynOperation, Op, OperationBridge};
#[cfg(feature = "fft")]
pub use domain::ops::freqfilt::{FwFftOp, InvFftOp};
pub use domain::ops::point::Linear;
pub use domain::ops::{
    arithmetic::{Add, AvgOp, DeviateOp, Multiply, RecombOp, Subtract},
    create::{EyeOp, GaussmatOp, GaussmatPrecision, SinesOp, TonelutOp},
    freqfilt::COMPLEX_BANDS,
    histogram::HistFindOp,
};
pub use ports::scheduler::TileScheduler;
pub use ports::source::ImageSource;

#[cfg(test)]
mod public_api_tests {
    use std::any::TypeId;

    use super::{adapters, codecs, domain, ops, pipeline, prelude};

    #[test]
    fn prelude_reexports_simple_api_surface() {
        assert_eq!(
            TypeId::of::<prelude::ImagePipeline>(),
            TypeId::of::<crate::ImagePipeline>()
        );
        assert_eq!(
            TypeId::of::<prelude::ViprsError>(),
            TypeId::of::<crate::ViprsError>()
        );
    }

    #[test]
    fn pipeline_reexports_advanced_pipeline_types() {
        assert_eq!(
            TypeId::of::<pipeline::ImagePipeline>(),
            TypeId::of::<adapters::pipeline::ImagePipeline>(),
        );
        assert_eq!(
            TypeId::of::<pipeline::PipelineArena>(),
            TypeId::of::<adapters::pipeline::PipelineArena>(),
        );
        assert_eq!(
            TypeId::of::<pipeline::CompiledPipeline>(),
            TypeId::of::<adapters::pipeline::CompiledPipeline>(),
        );
        assert_eq!(
            TypeId::of::<pipeline::NodeSpec>(),
            TypeId::of::<domain::op::NodeSpec>()
        );
        assert_eq!(
            TypeId::of::<pipeline::BuildError>(),
            TypeId::of::<domain::error::BuildError>(),
        );
    }

    #[test]
    fn ops_reexports_operation_modules() {
        assert_eq!(
            TypeId::of::<ops::point::Invert>(),
            TypeId::of::<domain::ops::point::Invert>(),
        );
        assert_eq!(
            TypeId::of::<ops::resample::thumbnail::ThumbnailTarget>(),
            TypeId::of::<domain::ops::resample::thumbnail::ThumbnailTarget>(),
        );
    }

    #[test]
    fn codecs_reexport_advanced_codec_surface() {
        assert_eq!(
            TypeId::of::<codecs::RawCodec>(),
            TypeId::of::<adapters::codecs::RawCodec>()
        );
        assert_eq!(
            TypeId::of::<codecs::LoadOptions>(),
            TypeId::of::<domain::codec_options::LoadOptions>(),
        );
        assert_eq!(
            TypeId::of::<codecs::SaveOptions>(),
            TypeId::of::<domain::codec_options::SaveOptions>(),
        );
    }
}
