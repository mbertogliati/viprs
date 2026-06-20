//! Concrete pipeline construction and compilation adapters.
//!
//! These types bridge ergonomic builder calls to compiled execution plans that
//! schedulers can run tile-by-tile over concrete image sources.

use crate::{
    adapters::{cache::OperationTileCache, sinks::memory::MemorySink, sources::zero::ZeroSource},
    domain::ops::{
        colour::{lab_to_labs::LabToLabS, labs_to_lab::LabSToLab},
        conversion::{
            cast::Cast,
            copy::CopyOp,
            embed::{EmbedBridge, ExtendMode, Gravity},
            flip::Flip,
            grid::GridBridge,
            msb::MsbOp,
            replicate::ReplicateBridge,
            rot::{Angle, RotBridge},
            rot45::Angle45,
            wrap::Wrap,
            zoom::ZoomBridge,
        },
        convolution::{
            GaussBlurH, GaussBlurV, LabSSharpen, conv2d::Conv2d, gauss_blur::GaussOutputFormat,
        },
        resample::{
            reduce::ReduceBridge,
            reduceh::ReduceHBridge,
            reducev::ReduceVBridge,
            resize::{Resize, ResizeNode},
            shrink::ShrinkBridge,
            shrinkh::ShrinkHBridge,
            shrinkv::ShrinkVBridge,
            similarity::SimilarityBridge,
            thumbnail::{Thumbnail, ThumbnailNode},
        },
        structural::{
            extract_area::ExtractArea,
            flatten::{FlattenBridge, flatten_has_alpha},
            premultiply::Premultiply,
            subsample::SubsampleBridge,
            unpremultiply::Unpremultiply,
        },
    },
    domain::{
        colorspace::{Colorspace, ColorspaceId, Lab},
        colour_dispatcher::ColourspaceDispatcher,
        error::{BuildError, ViprsError},
        format::{BandFormat, BandFormatId, F32, F64, I16, I32, U8, U16, U32},
        image::{DemandHint, Image, ImageMetadata, Interpretation, Region},
        kernel::InterpolationKernel,
        op::{DynOperation, DynViewOp, NodeSpec, OperationBridge, SourceReadPlan, ViewBridge},
        reorder::{ReorderError, ReorderNode, reorder_dag},
    },
    ports::{scheduler::TileScheduler, source::DynImageSource},
};
use std::{
    num::{NonZeroU8, NonZeroUsize},
    sync::Arc,
};

#[cfg(test)]
use crate::domain::colorspace::{Cmyk, Lch, Oklab, Oklch, SRgb, ScRgb, Ucs, Xyz, Yxy};

/// Index of a node inside a [`PipelineArena`] or [`CompiledPipeline`].
///
/// This alias makes graph wiring code easier to read by distinguishing node
/// references from plain integers.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::pipeline::NodeIdx;
///
/// let node: NodeIdx = 0;
/// assert_eq!(node, 0);
/// ```
pub type NodeIdx = usize;
/// Index of a scratch/output buffer inside a compiled pipeline execution plan.
///
/// This alias documents when an integer refers to buffer storage rather than to
/// a graph node.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::pipeline::BufferIdx;
///
/// let buffer: BufferIdx = 0;
/// assert_eq!(buffer, 0);
/// ```
pub type BufferIdx = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LineCacheAccess {
    Sequential,
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LineCacheRequest {
    lines_ahead: Option<usize>,
    access: LineCacheAccess,
}

impl LineCacheRequest {
    pub(crate) fn new(lines_ahead: usize, access: LineCacheAccess) -> Self {
        Self {
            lines_ahead: (lines_ahead != 0).then_some(lines_ahead),
            access,
        }
    }

    pub(crate) fn resolve(self, tile_height: u32) -> LineCacheConfig {
        let auto_lines = (tile_height as usize).saturating_mul(2).max(1);
        let lines_ahead = self
            .lines_ahead
            .map_or(auto_lines, |lines_ahead| lines_ahead.max(1));
        LineCacheConfig { lines_ahead }
    }

    pub(crate) const fn access(self) -> LineCacheAccess {
        self.access
    }
}

/// Bounded scanline cache configuration for libvips-style `linecache` / `sequential`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineCacheConfig {
    /// Maximum number of full-width lines retained in memory at once.
    pub lines_ahead: usize,
}

mod compiled;
pub use compiled::{CompiledNode, CompiledOp, CompiledPipeline, InputSlicePtr, ThreadBufferPool};

mod arena;
pub use arena::PipelineArena;
use arena::{ArenaNodeOp, format_sample_size};

mod builder;
pub use builder::{Flush, PipelineBuilder, PipelineOp};

#[cfg(test)]
mod tests;
