use super::{
    BandFormat, BuildError, Concretize, DemandHint, DynOperation, InterpolationKernel, LineCacheOp,
    PipelineBuilder, SequentialOp, flush_concretize_chain,
};

/// Trait for anything that can be applied to a [`PipelineBuilder`] via `.apply()`.
///
/// Two blanket implementations cover all cases:
/// - `Concretize` types (point ops) → fused into a single vectorized loop
/// - `Box<dyn DynOperation>` → executed as a separate pipeline node
///
/// The user never sees this trait — they just call `builder.apply(op)`.
pub trait PipelineOp<State = Identity> {
    /// Builder state produced after this operation has been incorporated.
    type NextState: Flush;

    /// Applies this operation to the current builder state.
    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<State>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError>;
}

/// Local marker trait limiting the concretize blanket impl to known point-op chain types.
pub trait ConcretizePipelineOpAllowed {}

impl ConcretizePipelineOpAllowed for () {}
impl<A, B> ConcretizePipelineOpAllowed for (A, B)
where
    A: ConcretizePipelineOpAllowed,
    B: ConcretizePipelineOpAllowed,
{
}

impl<C> ConcretizePipelineOpAllowed for viprs_core::concretize::Chain<C> where
    C: ConcretizePipelineOpAllowed
{
}

impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Abs {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::BoolAnd {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::BoolOr {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::BoolXor {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Lshift {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Rshift {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Clamp {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Gamma {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Invert {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Linear {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::ACos {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::ASin {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::ATan {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Ceil {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Cos {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Exp {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Floor {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Log {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Power {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Round {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Sign {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Sin {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Sqrt {}
impl ConcretizePipelineOpAllowed for viprs_ops_pixel::point::Tan {}

/// Point ops implementing `Concretize` are auto-fused.
impl<C> PipelineOp<Identity> for C
where
    C: Concretize + Clone + ConcretizePipelineOpAllowed,
{
    type NextState = Fusing<C>;

    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<Identity>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError> {
        Ok(builder.into_state(Fusing { chain: self }))
    }
}

impl<C, D> PipelineOp<Fusing<C>> for D
where
    C: Concretize + Clone + ConcretizePipelineOpAllowed,
    D: Concretize + Clone + ConcretizePipelineOpAllowed,
{
    type NextState = Fusing<(C, D)>;

    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<Fusing<C>>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError> {
        let PipelineBuilder {
            arena,
            last_node,
            current_format,
            bands,
            current_colorspace,
            current_interpretation,
            current_icc_profile,
            pending,
        } = builder;
        let Fusing { chain } = pending;
        Ok(PipelineBuilder {
            arena,
            last_node,
            current_format,
            bands,
            current_colorspace,
            current_interpretation,
            current_icc_profile,
            pending: Fusing {
                chain: (chain, self),
            },
        })
    }
}

/// Pre-built dynamic operations pass through to `then()`.
impl<S: Flush> PipelineOp<S> for Box<dyn DynOperation> {
    type NextState = Identity;

    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<S>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError> {
        builder.then(self)
    }
}

impl<S: Flush> PipelineOp<S> for SequentialOp {
    type NextState = S;

    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<S>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError> {
        Ok(builder.configure_sequential_streaming(self.lines_ahead()))
    }
}

impl<S: Flush> PipelineOp<S> for LineCacheOp {
    type NextState = S;

    fn apply_to_pipeline(
        self,
        builder: PipelineBuilder<S>,
    ) -> Result<PipelineBuilder<Self::NextState>, BuildError> {
        Ok(builder.configure_linecache(self.lines_ahead()))
    }
}

const LBB_REDUCE_REASON: &str =
    "Lbb is a nonlinear 2-D affine interpolator in libvips; use resize()/affine() for LBB parity";
const NOHALO_REDUCE_REASON: &str = "Nohalo is a nonlinear 2-D interpolator in libvips and has no separable reduce kernel; use resize()/affine() mapping or one of: Nearest, Bilinear, Bicubic/CatmullRom, Lanczos2, Lanczos3";

/// No point-op chain is being accumulated, so the builder can accept a fresh op.
pub struct Identity;

/// A statically typed `Concretize` chain that has not yet been emitted into the arena.
pub struct Fusing<C: Concretize> {
    chain: C,
}

/// Capability to flush the current builder state into concrete pipeline nodes.
pub trait Flush: Sized {
    /// Materializes any deferred work held by this state into the builder arena.
    fn flush(builder: &mut PipelineBuilder<Self>) -> Result<(), BuildError>;
}

impl Flush for Identity {
    fn flush(_builder: &mut PipelineBuilder<Self>) -> Result<(), BuildError> {
        Ok(())
    }
}

impl<C> Flush for Fusing<C>
where
    C: Concretize + Clone,
{
    fn flush(builder: &mut PipelineBuilder<Self>) -> Result<(), BuildError> {
        let op = flush_concretize_chain(
            &builder.pending.chain,
            builder.current_format,
            builder.bands,
        )?;
        builder.push_dyn_op(op)
    }
}

#[inline]
pub(in crate::pipeline::builder) fn validate_reduce_kernel(
    op: &'static str,
    kernel: InterpolationKernel,
) -> Result<(), BuildError> {
    if kernel == InterpolationKernel::Lbb {
        return Err(BuildError::UnsupportedKernel {
            op,
            kernel,
            reason: LBB_REDUCE_REASON,
        });
    }

    if kernel == InterpolationKernel::Nohalo {
        return Err(BuildError::UnsupportedKernel {
            op,
            kernel,
            reason: NOHALO_REDUCE_REASON,
        });
    }

    Ok(())
}

#[inline]
pub(in crate::pipeline::builder) fn validate_reduce_factors(
    h_factor: f64,
    v_factor: f64,
) -> Result<(), BuildError> {
    if !h_factor.is_finite() || !v_factor.is_finite() {
        return Err(BuildError::InvalidReduceParameters {
            h_factor,
            v_factor,
            reason: "factors must be finite",
        });
    }

    if h_factor < 1.0 || v_factor < 1.0 {
        return Err(BuildError::InvalidReduceParameters {
            h_factor,
            v_factor,
            reason: "factors must be >= 1.0",
        });
    }

    Ok(())
}

#[inline]
pub(in crate::pipeline::builder) fn validate_extract_area_bounds(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    image_width: u32,
    image_height: u32,
) -> Result<(), BuildError> {
    if width == 0 || height == 0 {
        return Err(BuildError::InvalidExtractAreaParameters {
            x,
            y,
            width,
            height,
            image_width,
            image_height,
        });
    }

    let crop_right = x.checked_add(width);
    let crop_bottom = y.checked_add(height);
    if crop_right.is_none_or(|right| right > image_width)
        || crop_bottom.is_none_or(|bottom| bottom > image_height)
    {
        return Err(BuildError::InvalidExtractAreaParameters {
            x,
            y,
            width,
            height,
            image_width,
            image_height,
        });
    }

    Ok(())
}

/// `DynOperation` wrapper for `Affine` that overrides `output_width` and
/// `output_height` with the fixed dimensions supplied at construction time.
///
/// `pub(crate)` — callers use `PipelineBuilder::affine`, not this type directly.
pub struct AffineBridge<F: BandFormat>
where
    F::Sample: bytemuck::Pod
        + crate::domain::ops::resample::sample_conv::ToF64
        + crate::domain::ops::resample::sample_conv::FromF64,
{
    inner: crate::domain::op::OperationBridge<crate::domain::ops::resample::affine::Affine<F>>,
    output_w: u32,
    output_h: u32,
    demand_hint: DemandHint,
}

impl<F: BandFormat + Send + Sync> AffineBridge<F>
where
    F::Sample: bytemuck::Pod
        + crate::domain::ops::resample::sample_conv::ToF64
        + crate::domain::ops::resample::sample_conv::FromF64,
{
    /// Creates an affine bridge that reports caller-supplied output dimensions and extend mode.
    pub fn new_with_extend(
        matrix: [f64; 4],
        tx: f64,
        ty: f64,
        kernel: InterpolationKernel,
        input_w: u32,
        input_h: u32,
        output_w: u32,
        output_h: u32,
        bands: u32,
        demand_hint: DemandHint,
        extend: crate::domain::ops::resample::affine::ExtendMode,
    ) -> Result<Self, crate::domain::error::BuildError> {
        use crate::domain::ops::resample::affine::Affine;
        let affine = Affine::try_new(matrix, tx, ty, kernel, output_w, output_h)
            .map(|affine| {
                affine
                    .with_extend(extend)
                    .with_source_bounds(crate::domain::image::Region::new(0, 0, input_w, input_h))
            })
            .map_err(|error| match error {
                crate::domain::error::ViprsError::DegenerateAffineTransform {
                    matrix,
                    output_width,
                    output_height,
                    reason,
                } => crate::domain::error::BuildError::DegenerateAffineTransform {
                    matrix,
                    output_width,
                    output_height,
                    reason,
                },
                _ => crate::domain::error::BuildError::InvalidAffineMatrix {
                    matrix,
                    reason: "affine validation failed",
                },
            })?;
        Ok(Self {
            inner: crate::domain::op::OperationBridge::new(affine, bands),
            output_w,
            output_h,
            demand_hint,
        })
    }
}

impl<F: BandFormat> crate::domain::op::DynOperation for AffineBridge<F>
where
    F::Sample: bytemuck::Pod
        + crate::domain::ops::resample::sample_conv::ToF64
        + crate::domain::ops::resample::sample_conv::FromF64
        + Send,
{
    fn input_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.input_format()
    }

    fn output_format(&self) -> crate::domain::format::BandFormatId {
        self.inner.output_format()
    }

    fn bands(&self) -> u32 {
        self.inner.bands()
    }

    fn demand_hint(&self) -> crate::domain::image::DemandHint {
        self.demand_hint
    }

    fn required_input_region(
        &self,
        output: &crate::domain::image::Region,
    ) -> crate::domain::image::Region {
        self.inner.required_input_region(output)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> crate::domain::op::NodeSpec {
        self.inner.node_spec(tile_w, tile_h)
    }

    fn output_width(&self, _input_w: u32) -> u32 {
        self.output_w
    }

    fn output_height(&self, _input_h: u32) -> u32 {
        self.output_h
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start()
    }

    fn dyn_start_with_tile(&self, tile_w: u32, tile_h: u32) -> Box<dyn std::any::Any + Send> {
        self.inner.dyn_start_with_tile(tile_w, tile_h)
    }

    fn dyn_process_region(
        &self,
        state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        input_region: crate::domain::image::Region,
        output_region: crate::domain::image::Region,
    ) {
        self.inner
            .dyn_process_region(state, input, output, input_region, output_region);
    }
}
