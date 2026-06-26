//! Builder support for compiled image pipelines.
//!
//! These helpers participate in turning fluent pipeline descriptions into
//! scheduler-ready execution plans.

use super::{
    Angle, Angle45, ArenaNodeOp, BandFormat, BandFormatId, BuildError, Cast, Colorspace,
    ColorspaceId, ColourspaceDispatcher, CompiledPipeline, Conv2d, CopyOp, DemandHint,
    DynImageSource, DynOperation, DynViewOp, EmbedBridge, ExtendMode, ExtractArea, F32, F64,
    FlattenBridge, Flip, GaussBlurH, GaussBlurV, GaussOutputFormat, Gravity, GridBridge, I16, I32,
    ImageMetadata, InterpolationKernel, Interpretation, Lab, LabSSharpen, LabSToLab, LabToLabS,
    LineCacheAccess, LineCacheRequest, MsbOp, NodeIdx, NonZeroU8, NonZeroUsize, OperationBridge,
    PipelineArena, Premultiply, ReduceBridge, ReduceHBridge, ReduceVBridge, ReplicateBridge,
    Resize, ResizeNode, RotBridge, ShrinkBridge, ShrinkHBridge, ShrinkVBridge, SimilarityBridge,
    SubsampleBridge, Thumbnail, ThumbnailNode, U8, U16, U32, Unpremultiply, ViewBridge, Wrap,
    ZoomBridge, flatten_has_alpha, format_sample_size,
};
#[cfg(feature = "icc")]
use crate::domain::ops::colour::icc::build_normalize_to_srgb_op;
use crate::{
    adapters::concretized_bridge::flush_concretize_chain,
    domain::{
        concretize::Concretize,
        ops::conversion::{LineCacheOp, SequentialOp},
    },
};

mod state;
pub use state::AffineBridge;
pub use state::PipelineOp;
// REASON: Preserve the previous builder.rs module surface for crate-internal callers.
#[allow(unused_imports)]
pub use state::{Commit, Committed, Fusing};

mod core;
pub use core::ImagePipeline;

mod colour;
mod geometry;
mod resample;
