#[cfg(feature = "icc")]
use super::build_normalize_to_srgb_op;
use super::colour::interpretation_to_colorspace;
use super::{
    ArenaNodeOp, BandFormatId, BuildError, ColorspaceId, CompiledPipeline, DemandHint,
    DynImageSource, DynOperation, Flush, Identity, ImageMetadata, Interpretation, LineCacheAccess,
    LineCacheRequest, NodeIdx, NonZeroUsize, PipelineArena, PipelineOp, format_sample_size,
};

/// Primary constructor is `from_source`.
///
/// The source fixes width, height, format, and band count. Every call to `then`
/// validates the format against the current pipeline output. Convenience methods
/// (`linear`, `invert`, `cast`) dispatch over the current format and hide
/// `OperationBridge` from callers.
pub struct PipelineBuilder<Op = Identity> {
    pub(in crate::pipeline::builder) arena: PipelineArena,
    pub(in crate::pipeline::builder) last_node: Option<NodeIdx>,
    /// Format of the last operation's output (or the source's format if no ops yet).
    pub(in crate::pipeline::builder) current_format: BandFormatId,
    /// Band count at the current stage (matches the source until an op changes it).
    pub(in crate::pipeline::builder) bands: u32,
    /// Colorspace of the current pipeline output. `None` means unknown (e.g., raw
    /// decoder output not yet probed). `colourspace()` requires this to be `Some`
    /// so it can select the correct `ColourConvert` implementation.
    pub(in crate::pipeline::builder) current_colorspace: Option<ColorspaceId>,
    /// Interpretation of the current pipeline output when known.
    pub(in crate::pipeline::builder) current_interpretation: Option<Interpretation>,
    /// Embedded ICC profile for the current pipeline stage when known.
    pub(in crate::pipeline::builder) current_icc_profile: Option<Vec<u8>>,
    pub(in crate::pipeline::builder) pending: Op,
}

impl PipelineBuilder<Identity> {
    /// Primary constructor. The source defines width, height, format, and band count.
    ///
    /// Accepts `impl DynImageSource + 'static` — no `Box::new` at the call site.
    /// `dyn DynImageSource` is required inside `PipelineArena` because the concrete
    /// source type varies at runtime.
    pub fn from_source(source: impl DynImageSource + 'static) -> Self {
        let metadata = source.metadata();
        let current_interpretation = metadata.interpretation;
        let current_colorspace = current_interpretation.and_then(interpretation_to_colorspace);
        let current_icc_profile = metadata.icc_profile;
        let format = source.format();
        let bands = source.bands();
        Self {
            arena: PipelineArena::with_source(Box::new(source)),
            last_node: None,
            current_format: format,
            bands,
            current_colorspace,
            current_interpretation,
            current_icc_profile,
            pending: Identity,
        }
    }

    /// Test-only constructor. Creates an internal `ZeroSource<U8>` (format=U8, bands=1).
    /// **Do not use in production code** — connect a real source via `from_source` so
    /// that the pipeline processes actual image data.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            arena: PipelineArena::new(width, height),
            last_node: None,
            current_format: BandFormatId::U8,
            bands: 1,
            current_colorspace: None,
            current_interpretation: None,
            current_icc_profile: None,
            pending: Identity,
        }
    }

    /// Declare the colorspace of the current pipeline stage.
    ///
    /// Call this after `from_source` when the source colorspace is known (e.g. after
    /// probing a JPEG decoder). Required before calling `colourspace()`.
    #[must_use]
    pub const fn with_colorspace(mut self, colorspace: ColorspaceId) -> Self {
        self.current_colorspace = Some(colorspace);
        self
    }
}

impl<Op: Flush> PipelineBuilder<Op> {
    fn configure_line_cache(
        mut self,
        lines_ahead: usize,
        access: LineCacheAccess,
        sequential: bool,
    ) -> Self {
        self.arena.set_sequential(sequential);
        self.arena
            .set_line_cache_request(Some(LineCacheRequest::new(lines_ahead, access)));
        self.arena
            .set_demand_hint_override(Some(DemandHint::ThinStrip));
        self
    }

    pub(in crate::pipeline::builder) fn configure_sequential_streaming(
        self,
        lines_ahead: usize,
    ) -> Self {
        self.configure_line_cache(lines_ahead, LineCacheAccess::Sequential, true)
    }

    pub(in crate::pipeline::builder) fn configure_linecache(self, lines_ahead: usize) -> Self {
        self.configure_line_cache(lines_ahead, LineCacheAccess::Random, false)
    }

    pub(in crate::pipeline::builder) fn into_state<NextOp>(
        self,
        pending: NextOp,
    ) -> PipelineBuilder<NextOp> {
        PipelineBuilder {
            arena: self.arena,
            last_node: self.last_node,
            current_format: self.current_format,
            bands: self.bands,
            current_colorspace: self.current_colorspace,
            current_interpretation: self.current_interpretation,
            current_icc_profile: self.current_icc_profile,
            pending,
        }
    }

    /// `current_format` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::current_format;
    /// ```
    #[allow(dead_code)]
    // REASON: diagnostics and future builder extensions still query the staged output format.
    pub(crate) const fn current_format(&self) -> BandFormatId {
        self.current_format
    }

    /// `node_count` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::node_count;
    /// ```
    #[allow(dead_code)]
    // REASON: diagnostics and future builder extensions still query arena node counts.
    pub(crate) const fn node_count(&self) -> usize {
        self.arena.nodes.len()
    }

    /// Hint that the compiled pipeline will be consumed in top-to-bottom order.
    ///
    /// When enabled, schedulers must not issue tiles out of row-major order. This
    /// trades horizontal parallelism for lower peak memory in linear pipelines
    /// (decode → ops → encode) where previously produced tiles will never be read again.
    #[must_use]
    pub const fn with_sequential_access(mut self, sequential: bool) -> Self {
        self.arena.set_sequential(sequential);
        self
    }

    /// Enable libvips-style sequential streaming with a bounded full-width line cache.
    ///
    /// `lines_ahead == 0` selects the default budget of `tile_height * 2`, resolved
    /// from the compiled thin-strip geometry.
    #[must_use]
    pub fn sequential(self, lines_ahead: usize) -> Self {
        self.configure_sequential_streaming(lines_ahead)
    }

    /// Enable a bounded scanline cache without forcing sequential scheduling.
    ///
    /// Like libvips `linecache`, this keeps a small full-width cache of thin strips while
    /// leaving the rest of the pipeline free to schedule tiles in random-access order.
    #[must_use]
    pub fn linecache(self, lines_ahead: usize) -> Self {
        self.configure_linecache(lines_ahead)
    }

    /// Override the compiled pipeline demand hint before buffer sizing and scheduling metadata
    /// are finalized.
    #[must_use]
    pub const fn with_demand_hint_override(mut self, demand_hint: DemandHint) -> Self {
        self.arena.set_demand_hint_override(Some(demand_hint));
        self
    }

    /// `current_dimensions` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::current_dimensions;
    /// ```
    pub(crate) fn current_dimensions(&self) -> (u32, u32) {
        let mut width = self.arena.width;
        let mut height = self.arena.height;
        for node in &self.arena.nodes {
            match &node.op {
                ArenaNodeOp::Transform(op) => {
                    (width, height) = op.output_size(width, height);
                }
                ArenaNodeOp::View(op) => {
                    width = op.output_width(width);
                    height = op.output_height(height);
                }
            }
        }
        (width, height)
    }

    fn current_output_bytes(&self) -> usize {
        let (width, height) = self.current_dimensions();
        (width as usize)
            .saturating_mul(height as usize)
            .saturating_mul(self.bands as usize)
            .saturating_mul(format_sample_size(self.current_format))
    }

    pub(in crate::pipeline::builder) fn flush_pending(&mut self) -> Result<(), BuildError> {
        Op::flush(self)
    }

    #[inline]
    const fn validate_non_zero_bands(&self) -> Result<(), BuildError> {
        if self.bands == 0 {
            return Err(BuildError::InvalidImage {
                reason: "zero-band image",
            });
        }

        Ok(())
    }

    /// `flush_into_identity` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::flush_into_identity;
    /// ```
    pub fn flush_into_identity(mut self) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.flush_pending()?;
        Ok(self.into_state(Identity))
    }

    pub(in crate::pipeline::builder) fn push_dyn_op(
        &mut self,
        op: Box<dyn DynOperation>,
    ) -> Result<(), BuildError> {
        if op.input_format() != self.current_format {
            return Err(BuildError::FormatMismatch {
                produced: self.current_format,
                expected: op.input_format(),
                hint: "the operation's input format does not match the current pipeline output; insert a Cast operation",
            });
        }
        op.validate_build_contract(self.bands, op.bands())?;
        self.current_format = op.output_format();
        self.bands = op.bands();
        let metadata = ImageMetadata {
            interpretation: self.current_interpretation,
            icc_profile: self.current_icc_profile.clone(),
            ..ImageMetadata::default()
        };
        let transformed_metadata = op.transform_metadata(&metadata);
        self.current_interpretation = transformed_metadata.interpretation;
        self.current_icc_profile = transformed_metadata.icc_profile;
        if let Some(interpretation) = self.current_interpretation {
            self.current_colorspace = interpretation_to_colorspace(interpretation);
        }
        if let Some(cs) = op.output_colorspace() {
            self.current_colorspace = Some(cs);
        }
        let idx = self.arena.add_node(op);
        if let Some(prev) = self.last_node {
            self.arena.connect(prev, idx)?;
        }
        self.last_node = Some(idx);
        Ok(())
    }

    /// Insert an explicit ICC-managed normalization stage to sRGB when the current
    /// pipeline stage carries a non-sRGB embedded profile.
    #[cfg(feature = "icc")]
    /// `normalize_to_srgb` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::normalize_to_srgb;
    /// ```
    pub fn normalize_to_srgb(self) -> Result<PipelineBuilder<Identity>, BuildError> {
        let builder = self.flush_into_identity()?;
        let op = build_normalize_to_srgb_op(
            builder.current_format,
            builder.bands,
            builder.current_interpretation,
            builder.current_icc_profile.as_deref(),
        )?;
        match op {
            Some(op) => builder.then(op),
            None => Ok(builder),
        }
    }

    /// Low-level escape hatch: append a pre-built `DynOperation`.
    ///
    /// Validates that `op.input_format() == self.current_format`. Prefer the typed
    /// convenience methods (`linear`, `invert`, `cast`) which build the bridge internally.
    pub fn then(self, op: Box<dyn DynOperation>) -> Result<PipelineBuilder<Identity>, BuildError> {
        self.validate_non_zero_bands()?;
        let mut builder = self.flush_into_identity()?;
        builder.push_dyn_op(op)?;
        Ok(builder)
    }

    /// Enable pipeline-owned tile caching for the current last operation.
    ///
    /// The cache is disabled by default. `max_bytes` is the requested minimum
    /// pipeline-wide LRU budget; if this method is called multiple times, the latest
    /// request replaces the previous one while marking additional operations as
    /// cacheable.
    pub fn cache_last_op(
        self,
        max_bytes: NonZeroUsize,
    ) -> Result<PipelineBuilder<Identity>, BuildError> {
        let mut builder = self.flush_into_identity()?;
        // Full-frame reruns revisit the last op in row-major order. If the cache budget
        // cannot hold the whole output, large affine-family images churn the LRU and pay
        // the allocation/locking overhead without preserving tiles for the next pass.
        // Promote the requested budget to at least one full output frame so cache-on
        // benchmarks measure reuse instead of an intentionally cold cache.
        let effective_max_bytes =
            NonZeroUsize::new(builder.current_output_bytes().max(max_bytes.get()))
                .map_or(max_bytes, |effective_max_bytes| effective_max_bytes);
        let last = builder.last_node.ok_or(BuildError::NoNodes)?;
        let Some(cache_node) = builder
            .arena
            .nodes
            .iter()
            .enumerate()
            .rev()
            .find_map(|(idx, node)| matches!(node.op, ArenaNodeOp::Transform(_)).then_some(idx))
        else {
            let _ = last;
            return Ok(builder);
        };
        builder
            .arena
            .enable_cache(cache_node, effective_max_bytes)?;
        Ok(builder)
    }

    /// `build` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::builder::build;
    /// ```
    pub fn build(mut self) -> Result<CompiledPipeline, BuildError> {
        self.flush_pending()?;
        self.arena.compile()
    }

    /// Apply any operation to the pipeline.
    ///
    /// This is the primary API. Accepts both:
    /// - **Point ops** (`Concretize`) — automatically fused by the compiler
    /// - **Complex ops** (`Box<dyn DynOperation>`) — executed as separate pipeline nodes
    ///
    /// The optimization is transparent: the user calls `.apply()` for everything,
    /// and the system picks the optimal execution strategy.
    ///
    /// ```ignore
    /// use viprs_core::ops::point::{Invert, Linear};
    /// let pipeline = PipelineBuilder::from_source(src)
    ///     .apply(Invert)?          // point op → fused
    ///     .apply(Linear::new(2.0, -0.5))?  // point op → fused
    ///     .apply(some_dyn_op)?     // complex → separate node
    ///     .build()?;
    /// ```
    pub fn apply<O: PipelineOp<Op>>(
        self,
        op: O,
    ) -> Result<PipelineBuilder<O::NextState>, BuildError> {
        self.validate_non_zero_bands()?;
        op.apply_to_pipeline(self)
    }

    // ── Convenience methods ───────────────────────────────────────────────────

    /// Apply `output = input * scale + offset` per-sample.
    ///
    /// Extends the pending point-op fusion chain when possible; the chain flushes only when
    /// a non-fusable pipeline boundary is reached or `build()` is called. `scale` and
    /// `offset` are preserved as floating-point parameters. Integer pipelines evaluate in
    /// floating-point and clip the result back to the target sample range, matching libvips
    /// `linear`.
    pub fn linear(
        self,
        scale: f64,
        offset: f64,
    ) -> Result<
        PipelineBuilder<<crate::domain::ops::point::Linear as PipelineOp<Op>>::NextState>,
        BuildError,
    >
    where
        crate::domain::ops::point::Linear: PipelineOp<Op>,
    {
        if !scale.is_finite() || !offset.is_finite() {
            return Err(BuildError::InvalidLinearParameters { scale, offset });
        }
        self.apply(crate::domain::ops::point::Linear::new(scale, offset))
    }

    /// Invert all samples element-wise.
    ///
    /// Extends the pending point-op fusion chain when possible; the chain flushes only when
    /// a non-fusable pipeline boundary is reached or `build()` is called. The inversion
    /// semantic is type-dependent: for U8 it is `255 - x`, for F32/F64 it is `1.0 - x`,
    /// for signed integers it is negation. See `Invertible` impls.
    pub fn invert(
        self,
    ) -> Result<
        PipelineBuilder<<crate::domain::ops::point::Invert as PipelineOp<Op>>::NextState>,
        BuildError,
    >
    where
        crate::domain::ops::point::Invert: PipelineOp<Op>,
    {
        self.apply(crate::domain::ops::point::Invert)
    }
}
