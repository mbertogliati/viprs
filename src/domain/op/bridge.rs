use crate::domain::{
    error::{BuildError, ViprsError},
    format::{BandFormat, BandFormatId},
    image::{ImageMetadata, Region, Tile, TileMut},
};
use std::any::Any;

use super::{DemandHint, DynOperation, NodeSpec, Op, PixelLocalOp, SourceReadPlan};

/// Bridge from a static [`Op`] implementation to object-safe [`DynOperation`].
///
/// This is the single type-erasure boundary that lets compiled pipelines store heterogeneous ops
/// while keeping each concrete pixel kernel monomorphized internally.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{format::U8, image::{Region, Tile, TileMut}, op::{Op, OperationBridge}};
/// struct Copy;
/// impl Op for Copy {
///     type Input = U8;
///     type Output = U8;
///     type State = ();
///     fn required_input_region(&self, output: &Region) -> Region { *output }
///     fn start(&self) -> Self::State {}
///     fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
///         output.data.copy_from_slice(input.data);
///     }
/// }
/// let bridge = OperationBridge::new(Copy, 1);
/// assert_eq!(bridge.bands, 1);
/// ```
pub struct OperationBridge<T: Op> {
    /// Stores the `op` value for this item.
    pub op: T,
    input_bands: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// True when `T: PixelLocalOp`. Set by `new_pixel_local`, false by default.
    /// Checked at runtime by `is_pixel_local()` for pixel-local pipeline planning.
    pixel_local: bool,
}

impl<T: Op> OperationBridge<T>
where
    T::Input: BandFormat,
    T::Output: BandFormat,
    <T::Input as BandFormat>::Sample: bytemuck::Pod,
    <T::Output as BandFormat>::Sample: bytemuck::Pod,
{
    /// Construct an `OperationBridge`, honouring `T::OUTPUT_BANDS`.
    ///
    /// If `T::OUTPUT_BANDS` is `Some(n)`, the supplied `bands` argument is
    /// ignored and `n` is used instead. This ensures ops that always produce a
    /// fixed number of bands (e.g., `BandSplit` → 1) cannot be mis-configured
    /// by a caller that passes the wrong value.
    pub fn new(op: T, bands: u32) -> Self {
        let effective_bands = T::OUTPUT_BANDS.map_or(bands, |n| n as u32);
        Self {
            op,
            input_bands: bands,
            bands: effective_bands,
            pixel_local: false,
        }
    }

    /// Construct an `OperationBridge` for a pixel-local op (`T: PixelLocalOp`).
    ///
    /// Sets the `pixel_local` flag to `true`, which is surfaced via
    /// `DynOperation::is_pixel_local()`.
    pub fn new_pixel_local(op: T, bands: u32) -> Self
    where
        T: PixelLocalOp,
    {
        let effective_bands = T::OUTPUT_BANDS.map_or(bands, |n| n as u32);
        Self {
            op,
            input_bands: bands,
            bands: effective_bands,
            pixel_local: true,
        }
    }

    /// Smart constructor for ops whose input and output band counts are both runtime values.
    ///
    /// Unlike `new`, this constructor always uses the supplied band counts, even if
    /// `T::OUTPUT_BANDS` is `Some(n)`. Use this when the input and/or output band count
    /// is only known at runtime (e.g., `ExtractBands`, `RecombOp`).
    pub const fn with_dynamic_bands(op: T, input_bands: u32, output_bands: u32) -> Self {
        Self {
            op,
            input_bands,
            bands: output_bands,
            pixel_local: false,
        }
    }

    /// Dynamic-band constructor for pixel-local ops.
    pub const fn with_dynamic_bands_pixel_local(op: T, input_bands: u32, output_bands: u32) -> Self
    where
        T: PixelLocalOp,
    {
        Self {
            op,
            input_bands,
            bands: output_bands,
            pixel_local: true,
        }
    }
}

impl<T> DynOperation for OperationBridge<T>
where
    T: Op + Send + Sync,
    T::Input: BandFormat,
    T::Output: BandFormat,
    <T::Input as BandFormat>::Sample: bytemuck::Pod,
    <T::Output as BandFormat>::Sample: bytemuck::Pod,
{
    fn input_format(&self) -> BandFormatId {
        T::Input::ID
    }

    fn output_format(&self) -> BandFormatId {
        T::Output::ID
    }

    fn bands(&self) -> u32 {
        self.bands
    }

    fn demand_hint(&self) -> DemandHint {
        self.op.demand_hint()
    }

    fn is_pixel_local(&self) -> bool {
        self.pixel_local
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        self.op.transform_metadata(source)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        self.op.required_input_region(output)
    }

    fn source_read_plan_slot(&self, output: &Region, slot: usize) -> SourceReadPlan {
        SourceReadPlan::rect(self.required_input_region_slot(output, slot))
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        self.op.node_spec(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(self.op.start())
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn Any + Send> {
        Box::new(self.op.start_with_tile(tile_w, tile_h))
    }

    fn dyn_start_with_tile_and_bands(
        &self,
        tile_w: u32,
        tile_h: u32,
        bands: u32,
    ) -> Box<dyn Any + Send> {
        let _ = bands;
        Box::new(
            self.op
                .start_with_tile_and_bands(tile_w, tile_h, self.input_bands),
        )
    }

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        self.op
            .validate_region_contract(input_region, input_bands, output_region, output_bands)
    }

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        self.op.validate_build_contract(input_bands, output_bands)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        // The state was created by dyn_start() which boxes T::State. A downcast failure
        // means the pipeline was constructed with mismatched operation types — a bug in
        // bridge construction, not in user-supplied data. In release mode we return
        // without writing to output (leaving it unchanged) rather than panicking, so
        // that a construction bug does not corrupt downstream sinks.
        let Some(state) = state.downcast_mut::<T::State>() else {
            debug_assert!(
                false,
                "invariant violated: state type mismatch in bridge — bug in pipeline construction"
            );
            return;
        };

        // ThreadBufferPool pre-allocates buffers with the correct byte size for T::Input::Sample.
        // A cast failure indicates the pool was constructed with wrong buffer sizes — a
        // bug in pipeline construction. In release mode we return with empty slices rather
        // than panicking; the output tile is left unchanged.
        let Ok(input_samples) =
            bytemuck::try_cast_slice::<u8, <T::Input as BandFormat>::Sample>(input)
        else {
            debug_assert!(
                false,
                "buffer size/alignment mismatch — bug in ThreadBufferPool construction"
            );
            return;
        };
        let Ok(output_samples) =
            bytemuck::try_cast_slice_mut::<u8, <T::Output as BandFormat>::Sample>(output)
        else {
            debug_assert!(
                false,
                "buffer size/alignment mismatch — bug in ThreadBufferPool construction"
            );
            return;
        };

        let input_tile = Tile::new(input_region, self.input_bands, input_samples);
        let mut output_tile = TileMut::new(output_region, self.bands, output_samples);
        self.op.process_region(state, &input_tile, &mut output_tile);
    }
}

/// Zero-copy coordinate-transform operation.
///
/// A `ViewOp` declares only a coordinate mapping — it has no `process_region`
/// because it produces no new pixel data. The scheduler passes the upstream buffer
/// directly to the downstream node without copying.
///
/// Implement this trait (instead of `Op`) for operations whose only effect is
/// to shift, crop, or remap which region of the upstream buffer is visible.
/// `ExtractArea` is the canonical example.
///
/// `dyn DynViewOp` in `CompiledNode` is acceptable for the same reason as
/// `dyn DynOperation`: the concrete type is not known at pipeline-construction time.
pub trait ViewOp: Send + Sync {
    /// Associated type for format.
    type Format: BandFormat;

    /// Returns the tile-demand pattern required by this operation.
    fn demand_hint(&self) -> DemandHint;

    /// Map an output region back to the required input region.
    /// Same contract as `Op::required_input_region`.
    fn required_input_region(&self, output: &Region) -> Region;

    /// Clip a requested output region to the pixels this view can actually supply.
    ///
    /// Most views are identity over their declared output extent, so the default is
    /// to accept the full requested region unchanged.
    fn valid_output_region(&self, output: &Region) -> Region {
        *output
    }

    /// Output image width given the input width.
    /// Override for ops that change image dimensions (e.g. `ExtractArea`).
    fn output_width(&self, input_width: u32) -> u32 {
        input_width
    }

    /// Output image height given the input height.
    fn output_height(&self, input_height: u32) -> u32 {
        input_height
    }
}

/// Object-safe version of `ViewOp` for use in `CompiledPipeline`.
pub trait DynViewOp: Send + Sync {
    /// Returns or performs format.
    fn format(&self) -> BandFormatId;
    /// Returns or performs bands.
    fn bands(&self) -> u32;
    /// Returns the tile-demand pattern required by this operation.
    fn demand_hint(&self) -> DemandHint;
    /// Returns the input region required to produce `output`.
    fn required_input_region(&self, output: &Region) -> Region;
    /// Returns or performs valid output region.
    fn valid_output_region(&self, output: &Region) -> Region {
        *output
    }
    /// Returns or performs output width.
    fn output_width(&self, input_width: u32) -> u32;
    /// Returns or performs output height.
    fn output_height(&self, input_height: u32) -> u32;
    /// Buffer-sizing spec for this view node.
    ///
    /// View nodes are zero-copy and share the upstream buffer. The default
    /// `NodeSpec` has `input == output == tile` which is correct for
    /// `ExtractArea`. Override if a future view op changes tile geometry.
    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w,
            input_tile_h: tile_h,
            output_tile_w: self.output_width(tile_w),
            output_tile_h: self.output_height(tile_h),
            coordinate_driven_source: None,
        }
    }
}

/// Bridge from a static [`ViewOp`] to object-safe [`DynViewOp`].
///
/// This preserves zero-copy coordinate mapping while letting compiled pipelines store views and
/// transform nodes uniformly.
///
/// # Examples
/// ```rust
/// # use viprs::domain::{format::U8, image::Region, op::{DemandHint, ViewBridge, ViewOp}};
/// struct IdentityView;
/// impl ViewOp for IdentityView {
///     type Format = U8;
///     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
///     fn required_input_region(&self, output: &Region) -> Region { *output }
/// }
/// let bridge = ViewBridge::new(IdentityView, 1);
/// assert_eq!(bridge.bands, 1);
/// ```
pub struct ViewBridge<T: ViewOp> {
    /// Stores the `op` value for this item.
    pub op: T,
    /// Number of bands associated with this item.
    pub bands: u32,
}

impl<T: ViewOp> ViewBridge<T> {
    /// Create a dynamic view bridge with the given runtime band count.
    ///
    /// This packages a zero-copy view op for storage inside a compiled pipeline.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs::domain::{format::U8, image::Region, op::{DemandHint, ViewBridge, ViewOp}};
    /// struct IdentityView;
    /// impl ViewOp for IdentityView {
    ///     type Format = U8;
    ///     fn demand_hint(&self) -> DemandHint { DemandHint::ThinStrip }
    ///     fn required_input_region(&self, output: &Region) -> Region { *output }
    /// }
    /// let bridge = ViewBridge::new(IdentityView, 1);
    /// assert_eq!(bridge.bands, 1);
    /// ```
    pub const fn new(op: T, bands: u32) -> Self {
        Self { op, bands }
    }
}

impl<T: ViewOp + Send + Sync> DynViewOp for ViewBridge<T>
where
    T::Format: BandFormat,
{
    fn format(&self) -> BandFormatId {
        T::Format::ID
    }
    fn bands(&self) -> u32 {
        self.bands
    }
    fn demand_hint(&self) -> DemandHint {
        self.op.demand_hint()
    }
    fn required_input_region(&self, output: &Region) -> Region {
        self.op.required_input_region(output)
    }
    fn valid_output_region(&self, output: &Region) -> Region {
        self.op.valid_output_region(output)
    }
    fn output_width(&self, input_width: u32) -> u32 {
        self.op.output_width(input_width)
    }
    fn output_height(&self, input_height: u32) -> u32 {
        self.op.output_height(input_height)
    }
}
