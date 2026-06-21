use crate::{
    error::{BuildError, ViprsError},
    format::BandFormat,
    image::{ImageMetadata, Region, Tile, TileMut},
};

use super::{DemandHint, NodeSpec};

/// Marker trait for pixel-local operations.
///
/// An operation is **pixel-local** if and only if both of the following hold:
///
/// 1. `required_input_region(r) == r` for all regions `r` — the input region
///    equals the output region (no neighbourhood access, no halo).
/// 2. `node_spec(tile_w, tile_h) == NodeSpec::identity(tile_w, tile_h)` for all
///    `tile_w, tile_h` — the tile geometry is unchanged (no transpose, no padding).
///
/// These guarantees are upheld by convention, not enforced by the type system at
/// runtime. Any implementor that violates either invariant would silently produce
/// incorrect output when fused. Implementors must document compliance
/// in their `required_input_region` and `node_spec` overrides (or lack thereof).
///
/// # Why a marker trait instead of a `const IS_PIXEL_LOCAL: bool`
///
/// A marker trait allows compile-time fusion chains via `Concretize` tuples
/// (e.g., `(A, (B, C))`) with zero runtime overhead. A `const` flag would
/// require a runtime check that the optimizer cannot guarantee to eliminate, and
/// cannot participate in trait bounds.
///
/// # Fusion safety
///
/// Two `PixelLocalOp` implementations `A` and `B` can be fused into
/// `FusedOp<A, B>` when `A::Output == B::Input`. The fused op inherits the
/// pixel-local contract and itself implements `PixelLocalOp`, permitting deeper
/// fusion chains without any additional boilerplate from the caller.
pub trait PixelLocalOp: Op {}

/// Static image-processing contract with compile-time input and output formats.
///
/// `Op` lets the compiler monomorphize hot pixel loops while still giving the pipeline enough
/// metadata to plan tile demand and state allocation.
///
/// # Examples
/// ```rust
/// # use viprs_core::{
/// #     format::U8,
/// #     image::{Region, Tile, TileMut},
/// #     op::Op,
/// # };
/// struct Copy;
///
/// impl Op for Copy {
///     type Input = U8;
///     type Output = U8;
///     type State = ();
///
///     fn required_input_region(&self, output: &Region) -> Region { *output }
///     fn start(&self) -> Self::State {}
///     fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
///         output.data.copy_from_slice(input.data);
///     }
/// }
/// ```
pub trait Op: Send + Sync {
    /// Pixel format read by this operation.
    type Input: BandFormat;
    /// Pixel format written by this operation.
    ///
    /// This may differ from [`Self::Input`] for format conversions such as `Cast<U8, F32>`.
    type Output: BandFormat;
    /// Per-thread mutable state for scratch buffers or accumulators.
    type State: Send + 'static;

    /// Number of output bands this operation produces, if known at compile time.
    ///
    /// `None` (default) means the band count is determined by the caller at
    /// bridge-construction time and passed to `OperationBridge::new`.
    /// `Some(n)` means the op always produces exactly `n` bands regardless of
    /// input band count — e.g., `BandSplit` always produces 1 band.
    ///
    /// `OperationBridge::new` reads this const: if `Some(n)`, it ignores the
    /// `bands` argument and uses `n` instead.
    const OUTPUT_BANDS: Option<usize> = None;

    /// Preferred tile geometry for this operation.
    ///
    /// libvips defaults geometric transforms to square tiles, so `SmallTile` is
    /// the default when an op does not declare a stronger preference.
    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    /// Returns the tile-demand pattern required by this operation.
    fn demand_hint(&self) -> DemandHint {
        self.preferred_tile_geometry()
    }

    /// Return the input region required to compute `output`.
    ///
    /// Pixel-local operations return `output` unchanged, while halo-based operations expand it.
    fn required_input_region(&self, output: &Region) -> Region;

    /// Buffer-sizing spec for this operation.
    ///
    /// The default (`NodeSpec::identity`) is correct for all pixel-local operations.
    /// Override for ops that change tile geometry, e.g. convolution (halo) or Rotate90.
    /// `OperationBridge` delegates this to `DynOperation::node_spec`.
    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    /// Create the per-thread state for one pipeline execution.
    ///
    /// Implementations allocate or reset scratch state here so `process_region` can stay hot-path friendly.
    fn start(&self) -> Self::State;

    /// Create per-thread state for a specific scheduler tile size.
    ///
    /// Operations with geometry-dependent scratch buffers override this to pre-allocate once per thread.
    fn start_with_tile(&self, tile_w: u32, tile_h: u32) -> Self::State {
        let _ = (tile_w, tile_h);
        self.start()
    }

    /// Create per-thread state for a specific tile size and input band count.
    ///
    /// Override this when scratch memory depends on runtime bands so `process_region` stays allocation free.
    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
        let _ = bands;
        self.start_with_tile(tile_w, tile_h)
    }

    /// Validate pipeline-time configuration before this op is inserted.
    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        let _ = (input_bands, output_bands);
        Ok(())
    }

    /// Validate a tile pair before processing.
    ///
    /// The default implementation accepts any tile pair. Ops with strict
    /// region/band contracts override this to return a typed error.
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

    /// Transform image metadata alongside the pixel operation.
    ///
    /// The default preserves all metadata unchanged. Operations that change image
    /// semantics (for example interpretation, ICC profile, or EXIF orientation)
    /// override this hook so `run_to_image()` materializes correct output metadata.
    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        source.clone()
    }

    /// Validate and process a tile pair, surfacing contract violations as typed errors.
    fn execute_region(
        &self,
        state: &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    ) -> Result<(), ViprsError> {
        self.validate_region_contract(input.region, input.bands, output.region, output.bands)?;
        self.process_region(state, input, output);
        Ok(())
    }

    /// Process one input tile and write the matching output tile.
    ///
    /// Implementations keep all mutable scratch in `state` so the hot pixel path stays thread-local and allocation free.
    fn process_region(
        &self,
        state: &mut Self::State,
        input: &Tile<Self::Input>,
        output: &mut TileMut<Self::Output>,
    );
}
