use crate::domain::{
    colorspace::ColorspaceId,
    error::{BuildError, ViprsError},
    format::BandFormatId,
    image::{ImageMetadata, Region},
};
use std::any::Any;

use super::{CoordinateDrivenSourceSpec, DemandHint, NodeSpec, SourceReadPlan};

/// Object-safe version of Op for use in dynamic pipeline graphs.
///
/// `dyn DynOperation` is acceptable in pipeline registries because the concrete type
/// is NOT known at pipeline-construction time — the set of chained operations is built
/// at runtime. This is the exact exemption allowed by CLAUDE.md rule 1.
pub trait DynOperation: Send + Sync {
    /// Returns or performs input format.
    fn input_format(&self) -> BandFormatId;
    /// Returns or performs output format.
    fn output_format(&self) -> BandFormatId;
    /// Returns or performs bands.
    fn bands(&self) -> u32;
    /// Returns the tile-demand pattern required by this operation.
    fn demand_hint(&self) -> DemandHint;

    /// Returns `true` if this operation is pixel-local: output(x,y) depends only on
    /// input(x,y). Pixel-local ops satisfy both:
    /// 1. `required_input_region(r) == r` for all regions `r`.
    /// 2. `node_spec(tile_w, tile_h) == NodeSpec::identity(tile_w, tile_h)`.
    ///
    /// The default is `false`. `OperationBridge` overrides this using its `pixel_local`
    /// flag, which is set to `true` by `OperationBridge::new_pixel_local`. The planner
    /// and scheduler use this to preserve pixel-local source-read behavior.
    fn is_pixel_local(&self) -> bool {
        false
    }

    /// Transform image metadata alongside this operation.
    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        source.clone()
    }

    /// Number of input buffers this operation reads from. Default: 1.
    ///
    /// Override to return > 1 for multi-input ops (`BandJoin`, `Composite`, etc.).
    /// The pipeline compiler allocates `input_slot_count()` upstream buffer slots
    /// per node and passes them via `dyn_process_region_multi`.
    fn input_slot_count(&self) -> usize {
        1
    }

    /// Format expected for a specific input slot.
    ///
    /// Single-input ops use the historical `input_format()` contract. Multi-input
    /// ops override this when individual slots carry distinct formats.
    fn input_format_slot(&self, slot: usize) -> BandFormatId {
        debug_assert!(
            slot < self.input_slot_count(),
            "input slot out of range for input_format_slot"
        );
        self.input_format()
    }

    /// Band count expected for a specific input slot.
    ///
    /// Defaults to the output band count for existing single-input operations.
    /// Multi-input ops override this when slots have distinct band counts.
    fn input_bands_slot(&self, slot: usize) -> u32 {
        debug_assert!(
            slot < self.input_slot_count(),
            "input slot out of range for input_bands_slot"
        );
        self.bands()
    }

    /// Map an output tile back to the input tile needed in the given slot.
    ///
    /// `slot` must be in `0..self.input_slot_count()`. For single-input ops,
    /// delegates to `required_input_region` (slot 0 is the only slot).
    /// Override for multi-input ops to return distinct regions per slot.
    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        debug_assert_eq!(
            slot, 0,
            "single-input op called with slot != 0 — did you override input_slot_count?"
        );
        self.required_input_region(output)
    }

    /// Optional sparse read plan for this operation and input slot.
    ///
    /// The scheduler always honors this plan for source-direct inputs. For
    /// non-source inputs it may also propagate `PointGrid` upstream through
    /// single-input pixel-local operations; otherwise it falls back to the
    /// rectangular `produced_region()`. Callers that need the packed fallback
    /// rectangle can use `bounding_source_region()`.
    fn source_read_plan_slot(&self, output: &Region, slot: usize) -> SourceReadPlan {
        SourceReadPlan::rect(self.required_input_region_slot(output, slot))
    }

    /// Declares that one source-direct slot needs another slot to be materialized first.
    ///
    /// The scheduler uses this to run a two-pass prepare path for multi-input ops like
    /// `MapImOp`: materialize the dependency slot, derive the source bounds, then fetch the
    /// dependent root region. Default: no cross-slot dependency.
    fn coordinate_driven_source_spec(&self) -> Option<CoordinateDrivenSourceSpec> {
        None
    }

    /// Runtime source-direct plan for coordinate-driven demand.
    ///
    /// Called only when `coordinate_driven_source_spec()` returned a matching `(slot,
    /// dependency_slot)` pair and the dependency tile has already been materialized.
    /// Returning `None` keeps the scheduler on the static source plan path.
    fn source_read_plan_slot_with_materialized_dependency(
        &self,
        output: &Region,
        slot: usize,
        dependency_slot: usize,
        dependency_region: Region,
        dependency: &[u8],
    ) -> Option<SourceReadPlan> {
        let _ = (output, slot, dependency_slot, dependency_region, dependency);
        None
    }

    /// Map an output tile back to the required input tile (single-input ops only).
    ///
    /// Kept for backward compatibility and as the delegate target of
    /// `required_input_region_slot`. Multi-input ops should override
    /// `required_input_region_slot` directly.
    fn required_input_region(&self, output: &Region) -> Region;

    /// Output image width given the input width. Default: identity.
    fn output_width(&self, input_w: u32) -> u32 {
        input_w
    }

    /// Output image height given the input height. Default: identity.
    fn output_height(&self, input_h: u32) -> u32 {
        input_h
    }

    /// Output image dimensions given the input image dimensions.
    fn output_size(&self, input_w: u32, input_h: u32) -> (u32, u32) {
        (self.output_width(input_w), self.output_height(input_h))
    }

    /// The colorspace this operation produces, if it changes the colorspace.
    ///
    /// Default is `None` — the vast majority of operations are colorspace-agnostic
    /// (arithmetic, convolution, cast, etc.) and do not alter the colorspace.
    ///
    /// `ColourConvertBridge` overrides this to return `Some(To::ID)` so that
    /// `PipelineBuilder` can track the current colorspace after a colour conversion
    /// without needing a separate `DynColourOperation` trait.
    fn output_colorspace(&self) -> Option<ColorspaceId> {
        None
    }

    /// Buffer-sizing spec for this node.
    ///
    /// Returns the tile dimensions this node reads from its upstream buffer
    /// and writes to its downstream buffer. The default (`NodeSpec::identity`)
    /// is correct for all pixel-local operations. Override only for ops that
    /// change tile geometry (convolution, Rotate90).
    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    /// Returns or performs dyn start.
    fn dyn_start(&self) -> Box<dyn Any + Send>;

    /// Geometry-aware variant of [`DynOperation::dyn_start`].
    ///
    /// Bridges override this when the underlying static op uses tile dimensions to size scratch buffers.
    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        let _ = (tile_w, tile_h);
        self.dyn_start()
    }

    /// Band-aware geometry version of `dyn_start`.
    fn dyn_start_with_tile_and_bands(
        &self,
        tile_w: u32,
        tile_h: u32,
        bands: u32,
    ) -> Box<dyn Any + Send> {
        let _ = bands;
        self.dyn_start_with_tile(tile_w, tile_h)
    }

    /// Returns or performs validate build contract.
    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        let _ = (input_bands, output_bands);
        Ok(())
    }

    /// Returns or performs validate region contract.
    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        let _ = (input_region, input_bands, output_region, output_bands);
        Ok(())
    }

    /// Process a tile of pixels (single-input path).
    ///
    /// `input_region` and `output_region` may differ for operations that have a halo
    /// (e.g. convolution): the input tile is larger than the output tile by the kernel
    /// radius on each side. For pixel-local operations `input_region == output_region`.
    ///
    /// For multi-input ops, prefer `dyn_process_region_multi`. The scheduler calls
    /// this method only for nodes with `input_slot_count() == 1`.
    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    );

    /// Process a tile of pixels from multiple input slots (DAG merge nodes).
    ///
    /// `inputs[i]` contains the raw bytes for input slot `i`.
    /// `input_regions[i]` is the region those bytes cover.
    /// `inputs.len() == input_regions.len() == self.input_slot_count()`.
    ///
    /// Default implementation: delegates to `dyn_process_region` for single-input ops.
    /// Multi-input ops MUST override this method.
    ///
    /// The scheduler builds `inputs` and `input_regions` as `Vec`s per tile (one
    /// heap allocation of pointers, not pixel data).
    fn dyn_process_region_multi(
        &self,
        state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(
            inputs.len(),
            1,
            "multi-input op must override dyn_process_region_multi"
        );
        debug_assert_eq!(input_regions.len(), 1);
        if let (Some(&input), Some(&input_region)) = (inputs.first(), input_regions.first()) {
            self.dyn_process_region(state, input, output, input_region, output_region);
        }
    }
}
