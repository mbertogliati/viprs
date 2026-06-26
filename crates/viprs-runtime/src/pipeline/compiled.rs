//! Immutable compiled pipeline structures used by schedulers.
//!
//! Compilation resolves graph wiring, buffer ownership, and source metadata into
//! concrete execution data structures that worker threads can traverse cheaply.

use super::{
    Arc, BandFormat, BandFormatId, BufferIdx, BuildError, DemandHint, DynImageSource, DynOperation,
    DynViewOp, ImageMetadata, InMemoryImage, LineCacheAccess, LineCacheConfig, MemorySink,
    NodeSpec, OperationTileCache, Region, SourceReadPlan, TileScheduler, ViprsError,
};

/// The operation stored in a `CompiledNode`.
///
/// `Transform` nodes write to a new buffer; `View` nodes share the upstream buffer
/// (their `output_buf == input_buf`).
///
/// `dyn DynOperation` / `dyn DynViewOp` are acceptable here for the same reason as
/// in `PipelineArena`: the concrete type is not known at pipeline-construction time.
pub enum CompiledOp {
    /// Node that materializes a fresh output tile buffer.
    Transform(Box<dyn DynOperation>),
    /// Node that reinterprets an upstream buffer without allocating a new one.
    View(Box<dyn DynViewOp>),
}

const fn sample_bytes(id: BandFormatId) -> usize {
    match id {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

impl CompiledOp {
    /// Return the number of output bands produced by this compiled node operation.
    ///
    /// This helps schedulers size tile buffers without branching on transform vs.
    /// view node types.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledOp;
    ///
    /// fn bands(op: &CompiledOp) -> u32 { op.bands() }
    /// ```
    #[must_use]
    pub fn bands(&self) -> u32 {
        match self {
            Self::Transform(t) => t.bands(),
            Self::View(v) => v.bands(),
        }
    }

    /// Return the concrete output sample format for this operation.
    ///
    /// This solves runtime dispatch when the scheduler or sink needs to know the
    /// final sample layout of a compiled node.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledOp;
    ///
    /// fn output_format(op: &CompiledOp) { let _ = op.output_format(); }
    /// ```
    #[must_use]
    pub fn output_format(&self) -> BandFormatId {
        match self {
            Self::Transform(t) => t.output_format(),
            Self::View(v) => v.format(),
        }
    }

    /// Compute the upstream region needed to produce an output region for this node.
    ///
    /// This lets schedulers pull the correct source tiles before executing a
    /// transform or view node.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::{pipeline::CompiledOp, domain::image::Region};
    ///
    /// fn needs(op: &CompiledOp, output: Region) { let _ = op.required_input_region(&output); }
    /// ```
    #[must_use]
    pub fn required_input_region(&self, output: &Region) -> Region {
        match self {
            Self::Transform(t) => t.required_input_region(output),
            Self::View(v) => v.required_input_region(output),
        }
    }

    /// Return the scheduler demand hint exposed by this operation.
    ///
    /// Demand hints guide tile geometry selection and execution order.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledOp;
    ///
    /// fn hint(op: &CompiledOp) { let _ = op.demand_hint(); }
    /// ```
    #[must_use]
    pub fn demand_hint(&self) -> DemandHint {
        match self {
            Self::Transform(t) => t.demand_hint(),
            Self::View(v) => v.demand_hint(),
        }
    }

    /// Build the node specification used during tile geometry planning.
    ///
    /// This exposes the exact per-node tile contract that compilation preserved
    /// from the original operation.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledOp;
    ///
    /// fn spec(op: &CompiledOp) { let _ = op.node_spec(128, 128); }
    /// ```
    #[must_use]
    pub fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        match self {
            Self::Transform(t) => t.node_spec(tile_w, tile_h),
            Self::View(v) => v.node_spec(tile_w, tile_h),
        }
    }

    /// Transform output metadata for this operation.
    ///
    /// This solves color/profile propagation during compilation so sinks see the
    /// final image metadata after the full pipeline runs.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::{pipeline::CompiledOp, domain::image::ImageMetadata};
    ///
    /// fn metadata(op: &CompiledOp, source: &ImageMetadata) { let _ = op.transform_metadata(source); }
    /// ```
    #[must_use]
    pub fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        match self {
            Self::Transform(t) => t.transform_metadata(source),
            Self::View(_) => source.clone(),
        }
    }
}

/// One compiled pipeline node with resolved buffers and upstream wiring.
///
/// This type solves the scheduler's need for a stable, index-based execution
/// plan after graph compilation has finished.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::pipeline::CompiledNode;
///
/// fn inspect(node: &CompiledNode) { let _ = node.output_buf(); }
/// ```
pub struct CompiledNode {
    pub(crate) op: CompiledOp,
    /// Input buffer indices, one per input slot.
    ///
    /// Single-input ops: `input_bufs.len() == 1`, `input_bufs[0]` is the upstream buffer.
    /// View nodes: `input_bufs[0] == output_buf` (zero-copy, same buffer).
    /// Multi-input ops (DAG merge nodes): `input_bufs[i]` is the buffer for slot `i`.
    ///
    /// Invariant (enforced by `CompiledNode::new`):
    /// - Transform nodes: `∀ib ∈ input_bufs: ib < output_buf`
    /// - View nodes: `∀ib ∈ input_bufs: ib == output_buf`
    ///
    /// The scheduler relies on this for the split-borrow pattern and for the
    /// unsafe raw-pointer optimisation.
    input_bufs: Vec<BufferIdx>,
    input_upstreams: Vec<Option<usize>>,
    input_buffer_producers: Vec<Option<usize>>,
    output_buf: BufferIdx,
    cache_op_id: Option<usize>,
}

impl CompiledNode {
    /// Construct a `CompiledNode`, validating the buffer-index invariant.
    ///
    /// - Transform nodes: every `input_buf` must be strictly less than `output_buf`.
    /// - View nodes: every `input_buf` must equal `output_buf` (zero-copy, shared buffer).
    pub(crate) fn new(
        op: CompiledOp,
        input_bufs: Vec<BufferIdx>,
        input_upstreams: Vec<Option<usize>>,
        input_buffer_producers: Vec<Option<usize>>,
        output_buf: BufferIdx,
        cache_op_id: Option<usize>,
    ) -> Result<Self, BuildError> {
        debug_assert_eq!(input_bufs.len(), input_upstreams.len());
        debug_assert_eq!(input_bufs.len(), input_buffer_producers.len());
        for &ib in &input_bufs {
            let valid = match &op {
                CompiledOp::Transform(_) => ib < output_buf,
                CompiledOp::View(_) => ib == output_buf,
            };
            if !valid {
                return Err(BuildError::InvalidBufferOrder {
                    input_buf: ib,
                    output_buf,
                });
            }
        }
        Ok(Self {
            op,
            input_bufs,
            input_upstreams,
            input_buffer_producers,
            output_buf,
            cache_op_id,
        })
    }

    /// Return the input buffer indices consumed by this node.
    ///
    /// This exposes the scheduler wiring chosen during compilation.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.input_bufs(); }
    /// ```
    #[must_use]
    pub fn input_bufs(&self) -> &[BufferIdx] {
        &self.input_bufs
    }

    /// Return the output buffer index produced by this node.
    ///
    /// This tells the scheduler where downstream nodes should read this node's
    /// materialized tile data.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.output_buf(); }
    /// ```
    #[must_use]
    pub const fn output_buf(&self) -> BufferIdx {
        self.output_buf
    }

    /// Return the upstream node indices feeding each input slot.
    ///
    /// This helps profiling and diagnostics recover the original graph topology.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.input_upstreams(); }
    /// ```
    #[must_use]
    pub fn input_upstreams(&self) -> &[Option<usize>] {
        &self.input_upstreams
    }

    /// Return the producer node associated with each input buffer.
    ///
    /// This is useful for diagnostics that need to distinguish shared buffers
    /// from direct upstream edges.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.input_buffer_producers(); }
    /// ```
    #[must_use]
    pub fn input_buffer_producers(&self) -> &[Option<usize>] {
        &self.input_buffer_producers
    }

    /// Borrow the compiled operation stored in this node.
    ///
    /// This gives schedulers access to the node's execution contract without
    /// exposing mutable graph state.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.op(); }
    /// ```
    #[must_use]
    pub const fn op(&self) -> &CompiledOp {
        &self.op
    }

    /// Return the cache identifier for this node when pipeline tile caching is enabled.
    ///
    /// Nodes without pipeline-owned caching return `None`.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledNode;
    ///
    /// fn inspect(node: &CompiledNode) { let _ = node.cache_op_id(); }
    /// ```
    #[must_use]
    pub const fn cache_op_id(&self) -> Option<usize> {
        self.cache_op_id
    }
}

/// Immutable execution plan produced by pipeline compilation.
///
/// This is the bridge between the fluent builder API and tile schedulers: all
/// graph validation, buffer sizing, and metadata propagation are already done.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::pipeline::CompiledPipeline;
///
/// fn inspect(pipeline: &CompiledPipeline) { let _ = pipeline.width; }
/// ```
pub struct CompiledPipeline {
    /// The source that fills buffer index 0 (`buffer\[0\]`) before each tile is processed.
    ///
    /// `dyn DynImageSource` is kept intentionally: the dynamic pipeline builder accepts
    /// sources discovered at runtime, and the per-tile vtable call in
    /// `read_region` below the project's >5% regression threshold. The dyn boundary stays
    /// at pipeline construction, while the tile copy dominates the cost on the scheduler path.
    pub source: Box<dyn DynImageSource>,
    /// Compiled transform and view nodes in execution order.
    pub nodes: Vec<CompiledNode>,
    /// Total number of worker-visible tile buffers required by the plan.
    pub buffer_count: usize,
    /// Byte size for each worker buffer slot.
    pub buffer_sizes: Vec<usize>,
    /// Whether the pipeline must execute as a top-to-bottom sequential stream.
    pub sequential: bool,
    /// Sequential streaming cache budget resolved at compile time.
    pub sequential_line_cache: Option<LineCacheConfig>,
    pub(crate) line_cache_access: Option<LineCacheAccess>,
    /// Demand hint chosen for the compiled output graph.
    pub demand_hint: DemandHint,
    /// Final output width in pixels.
    pub width: u32,
    /// Final output height in pixels.
    pub height: u32,
    /// Final output sample format.
    pub output_format: BandFormatId,
    /// Final output band count.
    pub output_bands: u32,
    /// Final metadata after all compilation-time propagation rules.
    pub output_metadata: ImageMetadata,
    /// Sample format assigned to each worker buffer slot.
    pub buffer_formats: Vec<BandFormatId>,
    /// Band count assigned to each worker buffer slot.
    pub buffer_bands: Vec<u32>,
    /// Producer node index for each buffer slot, when one exists.
    pub buffer_producers: Vec<Option<usize>>,
    /// Largest output region each node may be asked to materialize.
    pub node_max_output_regions: Vec<Region>,
    /// Optional pipeline-owned tile cache keyed by compiled node id.
    pub tile_cache: Option<OperationTileCache>,
}

impl CompiledPipeline {
    /// Run the compiled pipeline into an owned [`InMemoryImage`].
    ///
    /// This solves in-memory execution for callers that need the fully rendered
    /// output image instead of a streaming sink.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::{pipeline::CompiledPipeline, scheduler::rayon_scheduler::RayonScheduler, domain::format::U8};
    ///
    /// fn run(pipeline: &CompiledPipeline, scheduler: &RayonScheduler) {
    ///     let _ = pipeline.run_to_image::<U8, _>(scheduler);
    /// }
    /// ```
    pub fn run_to_image<F, S>(&self, scheduler: &S) -> Result<InMemoryImage<F>, ViprsError>
    where
        F: BandFormat,
        S: TileScheduler<Self>,
    {
        if F::ID != self.output_format {
            return Err(ViprsError::Scheduler(format!(
                "run_to_image: format mismatch — pipeline output is {:?}, caller supplied {:?}",
                self.output_format,
                F::ID,
            )));
        }

        let mut sink = MemorySink::for_pipeline(self)?;
        scheduler.run(self, &mut sink)?;
        sink.into_image::<F>(
            self.width,
            self.height,
            self.output_bands,
            self.output_metadata.clone(),
        )
    }

    /// Clear every tile cached by this pipeline, if tile caching is enabled.
    ///
    /// Use this to reclaim memory or reset cache state between benchmark passes.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::CompiledPipeline;
    ///
    /// fn clear(pipeline: &CompiledPipeline) { let _ = pipeline.clear_tile_cache(); }
    /// ```
    pub fn clear_tile_cache(&self) -> Result<(), ViprsError> {
        if let Some(cache) = &self.tile_cache {
            cache.clear()?;
        }
        Ok(())
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
/// Raw pointer wrapper used for temporary multi-input slice indirection.
///
/// This solves worker-local scratch storage without copying slice metadata into
/// heavier synchronization primitives.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::pipeline::InputSlicePtr;
///
/// let ptr = InputSlicePtr(std::ptr::slice_from_raw_parts(std::ptr::null(), 0));
/// let _ = ptr;
/// ```
pub struct InputSlicePtr(
    /// Borrowed slice pointer staged for temporary multi-input dispatch.
    pub *const [u8],
);

// SAFETY: `InputSlicePtr` is used only as worker-local scratch inside
// `ThreadBufferPool`. The scheduler overwrites these pointers immediately before
// each multi-input dispatch and never shares them across threads.
unsafe impl Send for InputSlicePtr {}

/// Worker-local buffers and per-node state reused across tile executions.
///
/// Reusing one pool per worker avoids per-tile allocations while keeping the
/// scheduler's temporary storage isolated to each thread.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::pipeline::{CompiledPipeline, ThreadBufferPool};
///
/// fn allocate(pipeline: &CompiledPipeline) { let _ = ThreadBufferPool::new(pipeline); }
/// ```
pub struct ThreadBufferPool {
    /// Reusable worker-local tile buffers, indexed like `CompiledPipeline::buffer_sizes`.
    pub buffers: Vec<Vec<u8>>,
    /// One entry per pipeline node. `None` for `View` nodes (no per-thread state needed).
    pub op_states: Vec<Option<Box<dyn std::any::Any + Send>>>,
    /// Scratch input regions computed for each node's input slots.
    pub scratch_regions: Vec<Vec<Region>>,
    /// Cached source-read plans for each node output region.
    pub node_output_read_plans: Vec<Option<SourceReadPlan>>,
    /// Output region most recently assigned to each node for this worker.
    pub node_output_regions: Vec<Region>,
    /// Marks which `node_output_regions` entries are initialized.
    pub node_output_assigned: Vec<bool>,
    /// Source region required for the tile currently being processed.
    pub source_region: Region,
    /// Read plan for the current source region.
    pub source_read_plan: SourceReadPlan,
    /// Marks whether `source_region` and `source_read_plan` are initialized.
    pub source_region_assigned: bool,
    /// Whether the current source request may use sparse zero-fill fallback.
    pub source_sparse_fallback: bool,
    /// Borrowed source slice retained for zero-copy source reads.
    pub source_borrowed: Option<InputSlicePtr>,
    /// Worker-local owned copies used when multi-input nodes need contiguous scratch buffers.
    pub input_scratch_buffers: Vec<Vec<Vec<u8>>>,
    /// Raw slice pointers passed into multi-input operations without extra allocation.
    pub multi_input_refs: Vec<Vec<InputSlicePtr>>,
    /// Per-node cached output tiles materialized by the pipeline tile cache.
    pub cached_tiles: Vec<Option<Arc<[u8]>>>,
    pub(crate) sequential_line_cache:
        Option<crate::scheduler::rayon_scheduler::SequentialLineCache>,
}

impl ThreadBufferPool {
    /// Allocate the reusable worker-local scratch storage for a compiled pipeline.
    ///
    /// This sizes buffers, operation state, and region scratch space from the
    /// compiled execution plan once per worker thread.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::{CompiledPipeline, ThreadBufferPool};
    ///
    /// fn allocate(pipeline: &CompiledPipeline) { let _ = ThreadBufferPool::new(pipeline); }
    /// ```
    #[must_use]
    pub fn new(pipeline: &CompiledPipeline) -> Self {
        let tile_w = pipeline.demand_hint.tile_width(pipeline.width);
        let tile_h = pipeline
            .demand_hint
            .tile_height(pipeline.width, pipeline.height);
        let buffers = pipeline.buffer_sizes.iter().map(|_| Vec::new()).collect();
        let op_states = pipeline
            .nodes
            .iter()
            .enumerate()
            .map(|(node_idx, n)| match &n.op {
                CompiledOp::Transform(t) => {
                    let input_bands = n
                        .input_bufs()
                        .first()
                        .map_or_else(|| t.bands(), |&input_buf| pipeline.buffer_bands[input_buf]);
                    let state_region = pipeline.node_max_output_regions[node_idx];
                    Some(t.dyn_start_with_tile_and_bands(
                        state_region.width.max(tile_w),
                        state_region.height.max(tile_h),
                        input_bands,
                    ))
                }
                CompiledOp::View(_) => None,
            })
            .collect();
        let scratch_regions = pipeline
            .nodes
            .iter()
            .map(|node| match &node.op {
                CompiledOp::Transform(op) => {
                    vec![Region::new(0, 0, 0, 0); op.input_slot_count()]
                }
                CompiledOp::View(_) => Vec::new(),
            })
            .collect();
        let input_scratch_buffers = pipeline
            .nodes
            .iter()
            .map(|node| match &node.op {
                CompiledOp::Transform(op) => node
                    .input_bufs()
                    .iter()
                    .take(op.input_slot_count())
                    .map(|_| Vec::new())
                    .collect(),
                CompiledOp::View(_) => Vec::new(),
            })
            .collect();
        let multi_input_refs = pipeline
            .nodes
            .iter()
            .map(|node| match &node.op {
                CompiledOp::Transform(op) => {
                    vec![
                        InputSlicePtr(std::ptr::slice_from_raw_parts(std::ptr::null::<u8>(), 0,));
                        op.input_slot_count()
                    ]
                }
                CompiledOp::View(_) => Vec::new(),
            })
            .collect();
        let source_pixel_bytes =
            pipeline.buffer_bands[0] as usize * sample_bytes(pipeline.buffer_formats[0]);
        Self {
            buffers,
            op_states,
            scratch_regions,
            node_output_read_plans: vec![None; pipeline.nodes.len()],
            node_output_regions: vec![Region::new(0, 0, 0, 0); pipeline.nodes.len()],
            node_output_assigned: vec![false; pipeline.nodes.len()],
            source_region: Region::new(0, 0, 0, 0),
            source_read_plan: SourceReadPlan::rect(Region::new(0, 0, 0, 0)),
            source_region_assigned: false,
            source_sparse_fallback: false,
            source_borrowed: None,
            input_scratch_buffers,
            multi_input_refs,
            cached_tiles: vec![None; pipeline.nodes.len()],
            sequential_line_cache: pipeline.sequential_line_cache.map(|config| {
                crate::scheduler::rayon_scheduler::SequentialLineCache::new(
                    pipeline.source.width(),
                    pipeline.source.height(),
                    source_pixel_bytes,
                    config.lines_ahead,
                    tile_h as usize,
                    pipeline
                        .line_cache_access
                        .unwrap_or(LineCacheAccess::Sequential),
                )
            }),
        }
    }
}
