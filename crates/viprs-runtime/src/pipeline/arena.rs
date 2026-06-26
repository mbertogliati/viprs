//! Mutable DAG builder for compiled image pipelines.
//!
//! `PipelineArena` holds concrete operations, sources, and graph edges until the
//! build step freezes them into a scheduler-ready [`CompiledPipeline`].

use super::{
    BandFormatId, BufferIdx, BuildError, CompiledNode, CompiledOp, CompiledPipeline, DemandHint,
    DynImageSource, DynOperation, DynViewOp, LineCacheRequest, NodeIdx, NonZeroUsize,
    OperationTileCache, Region, ReorderError, ReorderNode, U8, ZeroSource, reorder_dag,
};

/// The operation stored in an `ArenaNode`.
///
/// `Transform` nodes have a `process_region`; `View` nodes are coordinate-transform-only
/// (zero-copy).
pub enum ArenaNodeOp {
    Transform(Box<dyn DynOperation>),
    View(Box<dyn DynViewOp>),
}

/// The `ArenaNode` type provides concrete adapter functionality in the `pipeline` module.
/// Use this type when you need the runtime behavior implemented by this adapter.
///
/// # Examples
///
/// ```ignore
/// let _ = core::mem::size_of::<viprs_runtime::pipeline::arena::ArenaNode>();
/// ```
pub struct ArenaNode {
    pub op: ArenaNodeOp,
    pub cache_enabled: bool,
}

/// A mutable DAG of operations. Call `compile()` to freeze it into a `CompiledPipeline`.
///
/// Construct via `PipelineArena::with_source` so that the source defines `width`/`height`.
/// `PipelineArena::new` is kept for tests that use a `ZeroSource` implicitly.
///
/// # Examples
///
/// ```rust,no_run
/// use viprs_runtime::pipeline::PipelineArena;
///
/// let arena = PipelineArena::new(64, 64);
/// let _ = arena;
/// ```
pub struct PipelineArena {
    pub(super) nodes: Vec<ArenaNode>,
    /// Directed edges: `(upstream_node, downstream_node, input_slot)`.
    ///
    /// `input_slot` identifies which input slot of `downstream_node` this edge connects to.
    /// Single-input ops always use slot 0. Multi-input ops (e.g. `BandJoin`) use slots 0..N.
    edges: Vec<(NodeIdx, NodeIdx, u8)>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    cache_max_bytes: Option<NonZeroUsize>,
    sequential: bool,
    line_cache_request: Option<LineCacheRequest>,
    demand_hint_override: Option<DemandHint>,
    /// The pixel source for this pipeline. `dyn DynImageSource` is required here because the
    /// concrete source type is not known at pipeline-construction time — it is supplied by
    /// the caller and varies at runtime.
    pub(super) source: Option<Box<dyn DynImageSource>>,
}

impl PipelineArena {
    /// Construct an arena that will use a `ZeroSource<U8>` as the pixel source.
    /// Prefer `with_source` in production code; `new` is intended for tests only.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let arena = PipelineArena::new(32, 32);
    /// let _ = arena;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            width,
            height,
            cache_max_bytes: None,
            sequential: false,
            line_cache_request: None,
            demand_hint_override: None,
            source: None,
        }
    }

    /// Construct an arena whose dimensions and pixel data come from `source`.
    ///
    /// `dyn DynImageSource` here is the documented exception from CLAUDE.md rule 1:
    /// the concrete source type is not known at compile time — it is supplied by the
    /// caller and varies at runtime. `DynImageSource` is object-safe (no associated types).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use viprs_runtime::{
    ///     pipeline::PipelineArena, sources::ZeroSource,
    ///     domain::format::U8,
    /// };
    ///
    /// let arena = PipelineArena::with_source(Box::new(ZeroSource::<U8>::new(16, 16, 1)));
    /// let _ = arena;
    /// ```
    #[must_use]
    pub fn with_source(source: Box<dyn DynImageSource>) -> Self {
        let width = source.width();
        let height = source.height();
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            width,
            height,
            cache_max_bytes: None,
            sequential: false,
            line_cache_request: None,
            demand_hint_override: None,
            source: Some(source),
        }
    }

    /// `set_sequential` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::arena::set_sequential;
    /// ```
    pub(crate) const fn set_sequential(&mut self, sequential: bool) {
        self.sequential = sequential;
    }

    /// `set_line_cache_request` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::arena::set_line_cache_request;
    /// ```
    pub(crate) const fn set_line_cache_request(
        &mut self,
        line_cache_request: Option<LineCacheRequest>,
    ) {
        self.line_cache_request = line_cache_request;
    }

    /// `set_demand_hint_override` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs_runtime::pipeline::arena::set_demand_hint_override;
    /// ```
    pub(crate) const fn set_demand_hint_override(&mut self, demand_hint: Option<DemandHint>) {
        self.demand_hint_override = demand_hint;
    }

    /// Add a transform node to the arena and return its node index.
    ///
    /// This solves explicit DAG construction for callers that already own boxed
    /// transform operations.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(8, 8);
    /// let _ = &mut arena;
    /// ```
    pub fn add_node(&mut self, op: Box<dyn DynOperation>) -> NodeIdx {
        let idx = self.nodes.len();
        self.nodes.push(ArenaNode {
            op: ArenaNodeOp::Transform(op),
            cache_enabled: false,
        });
        idx
    }

    /// Add a zero-copy view node to the arena and return its node index.
    ///
    /// View nodes transform coordinates or metadata without allocating a new
    /// output buffer.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(8, 8);
    /// let _ = &mut arena;
    /// ```
    pub fn add_view_node(&mut self, op: Box<dyn DynViewOp>) -> NodeIdx {
        let idx = self.nodes.len();
        self.nodes.push(ArenaNode {
            op: ArenaNodeOp::View(op),
            cache_enabled: false,
        });
        idx
    }

    /// Mark a transform node as eligible for pipeline-owned tile caching.
    ///
    /// This lets callers preserve expensive intermediate tiles when the graph
    /// fans out or revisits the same operation output.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use std::num::NonZeroUsize;
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(8, 8);
    /// let _ = arena.enable_cache(0, NonZeroUsize::new(1024).unwrap());
    /// ```
    pub fn enable_cache(
        &mut self,
        node: NodeIdx,
        max_bytes: NonZeroUsize,
    ) -> Result<(), BuildError> {
        let arena_node = self
            .nodes
            .get_mut(node)
            .ok_or(BuildError::InvalidNodeIndex(node))?;
        match &arena_node.op {
            ArenaNodeOp::Transform(_) => {
                arena_node.cache_enabled = true;
                self.cache_max_bytes = Some(max_bytes);
                Ok(())
            }
            ArenaNodeOp::View(_) => Err(BuildError::CacheRequiresTransform { node }),
        }
    }

    /// Connect `upstream` output to `downstream` input slot 0.
    ///
    /// Sugar for `connect_to_slot(upstream, downstream, 0)`. Use for all single-input ops
    /// and for the primary (first) input of multi-input ops.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(8, 8);
    /// let _ = arena.connect(0, 1);
    /// ```
    pub fn connect(&mut self, upstream: NodeIdx, downstream: NodeIdx) -> Result<(), BuildError> {
        self.connect_to_slot(upstream, downstream, 0)
    }

    /// Connect `upstream` output to a specific input slot of `downstream`.
    ///
    /// Use for DAG merge nodes where `downstream` reads from multiple upstreams.
    /// `slot` must be in `0..downstream.input_slot_count()` — validated at compile time.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(8, 8);
    /// let _ = arena.connect_to_slot(0, 2, 1);
    /// ```
    pub fn connect_to_slot(
        &mut self,
        upstream: NodeIdx,
        downstream: NodeIdx,
        slot: u8,
    ) -> Result<(), BuildError> {
        if upstream >= self.nodes.len() {
            return Err(BuildError::InvalidNodeIndex(upstream));
        }
        if downstream >= self.nodes.len() {
            return Err(BuildError::InvalidNodeIndex(downstream));
        }
        let slot_count = node_input_slot_count(&self.nodes[downstream].op);
        if slot as usize >= slot_count {
            return Err(BuildError::InvalidInputSlot {
                node: downstream,
                slot,
                slot_count,
            });
        }

        let produced = node_format(&self.nodes[upstream].op);
        let expected = node_input_format_slot(&self.nodes[downstream].op, slot as usize);
        if produced != expected {
            return Err(BuildError::FormatMismatch {
                produced,
                expected,
                hint: "insert an explicit Cast operation between these two stages",
            });
        }
        self.edges.push((upstream, downstream, slot));
        Ok(())
    }

    /// Update the output dimensions of the pipeline.
    ///
    /// Called by `PipelinePlan` when an op changes the image dimensions (e.g.,
    /// `ExtractArea`). The arena stores the _output_ dimensions so that `compile()`
    /// propagates the correct width/height to `CompiledPipeline`, which the scheduler
    /// uses to generate tiles.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let mut arena = PipelineArena::new(32, 32);
    /// arena.set_dimensions(16, 16);
    /// ```
    pub const fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Freeze the mutable graph into a scheduler-ready compiled pipeline.
    ///
    /// This validates node ordering, buffer wiring, and cache policy before the
    /// execution layer runs any tile work.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use viprs_runtime::pipeline::PipelineArena;
    ///
    /// let arena = PipelineArena::new(8, 8);
    /// let _compiled = arena.compile()?;
    /// # Ok::<(), viprs_core::error::BuildError>(())
    /// ```
    pub fn compile(self) -> Result<CompiledPipeline, BuildError> {
        let cache_max_bytes = self.cache_max_bytes;

        // Materialise a ZeroSource<U8> when no real source was provided.
        // `dyn DynImageSource` is required here for the same reason as in `with_source`
        // — the concrete type is only known at construction time.
        let source: Box<dyn DynImageSource> = self
            .source
            .unwrap_or_else(|| Box::new(ZeroSource::<U8>::new(self.width, self.height, 1)));

        let n = self.nodes.len();

        let reorder_nodes: Vec<ReorderNode> = self
            .nodes
            .iter()
            .map(|node| match &node.op {
                ArenaNodeOp::Transform(op) => ReorderNode::transform(op.node_spec(1, 1)),
                ArenaNodeOp::View(view) => ReorderNode::view(view.node_spec(1, 1)),
            })
            .collect();
        let reorder_edges: Vec<(NodeIdx, NodeIdx)> =
            self.edges.iter().map(|&(u, v, _slot)| (u, v)).collect();
        let topo_order = reorder_dag(&reorder_nodes, &reorder_edges)
            .map(|plan| plan.order)
            .map_err(|err| match err {
                ReorderError::InvalidNode { node, .. } => BuildError::InvalidNodeIndex(node),
                ReorderError::Cycle => BuildError::Cycle,
            })?;

        // Build upstream/downstream count maps for cache eligibility and downstream
        // buffer wiring.
        // `out_degree[i]` = number of outgoing edges from node i in the original graph.
        let mut out_degree = vec![0usize; n];
        // `in_edges[i]` = list of (upstream_idx, slot) for node i.
        let mut in_edges: Vec<Vec<(NodeIdx, u8)>> = vec![Vec::new(); n];
        for &(u, v, slot) in &self.edges {
            out_degree[u] += 1;
            in_edges[v].push((u, slot));
        }

        // Cache helps in two cases:
        // - DAG boundaries, where tiles can be reused within one pipeline execution.
        // - Output sinks, where repeated full-frame runs can reuse the final tile set.
        //
        // Interior linear nodes still pay the lock/Arc cost without gaining reuse, so a
        // requested cache on a node with exactly one input and one output remains disabled.
        let node_cache_enabled: Vec<bool> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                node.cache_enabled && (out_degree[idx] != 1 || in_edges[idx].len() > 1)
            })
            .collect();

        // Convert nodes into Option<ArenaNodeOp> for take() consumption later.
        let mut node_ops: Vec<Option<ArenaNodeOp>> = self
            .nodes
            .into_iter()
            .map(|arena_node| Some(arena_node.op))
            .collect();

        // Direct mapping: plan-level concretization now owns point-op fusion.
        let canonical_idx: Vec<NodeIdx> = (0..n).collect();

        // Maps each original NodeIdx to the BufferIdx it writes to.
        // Used by the DAG buffer assignment to look up which buffer an upstream node
        // produced, so merge nodes can find both input buffers.
        let mut node_output_bufs: Vec<Option<BufferIdx>> = vec![None; n];

        let demand_hint = self.demand_hint_override.unwrap_or_else(|| {
            node_ops
                .iter()
                .filter_map(|o| o.as_ref())
                .map(|op| match op {
                    ArenaNodeOp::Transform(t) => t.demand_hint(),
                    ArenaNodeOp::View(v) => v.demand_hint(),
                })
                .max()
                .unwrap_or_else(|| source.demand_hint())
        });

        // Pre-compute the output dimensions of the pipeline so that tile_w/tile_h
        // match what generate_tiles() will use at runtime (which calls
        // demand_hint.tile_width(pipeline.width)). For ops that transpose dimensions
        // (e.g. Rotate90), self.width != out_w, and sizing buffers against self.width
        // would underallocate the output buffer.
        let (pipeline_out_w, pipeline_out_h) = {
            let mut w = self.width;
            let mut h = self.height;
            for &idx in &topo_order {
                if let Some(op) = node_ops[idx].as_ref() {
                    match op {
                        ArenaNodeOp::Transform(t) => {
                            (w, h) = t.output_size(w, h);
                        }
                        ArenaNodeOp::View(v) => {
                            w = v.output_width(w);
                            h = v.output_height(h);
                        }
                    }
                }
            }
            (w, h)
        };

        // buffer[0] = source input; its size is computed from the source.
        let (tile_w, tile_h) = if pipeline_out_w == 0 || pipeline_out_h == 0 {
            (0, 0)
        } else {
            (
                demand_hint.tile_width(pipeline_out_w),
                demand_hint.tile_height(pipeline_out_w, pipeline_out_h),
            )
        };

        // Per-node buffer sizing via NodeSpec. buffer[0] is the source buffer;
        // its size is the max of the minimum source size and what the first
        // Transform node declares it needs.
        let source_bps = format_sample_size(source.format());
        let source_bands = source.bands() as usize;
        let min_source_size = tile_w as usize * tile_h as usize * source_bands * source_bps;

        let mut canonical_in_edges: Vec<Vec<(NodeIdx, u8)>> = vec![Vec::new(); n];
        for &(src, dst, slot) in &self.edges {
            let src = canonical_idx[src];
            let dst = canonical_idx[dst];
            if src != dst {
                canonical_in_edges[dst].push((src, slot));
            }
        }

        let mut buffer_owner_orig: Vec<Option<NodeIdx>> = vec![None; n];
        for &orig_idx in &topo_order {
            buffer_owner_orig[orig_idx] = match node_ops[orig_idx].as_ref() {
                Some(ArenaNodeOp::Transform(_)) => Some(orig_idx),
                Some(ArenaNodeOp::View(_)) => canonical_in_edges[orig_idx]
                    .iter()
                    .find(|&&(_, slot)| slot == 0)
                    .and_then(|&(src, _)| buffer_owner_orig[src]),
                None => None,
            };
        }

        let mut remaining_buffer_consumers = vec![0usize; n];
        for &orig_idx in &topo_order {
            let Some(ArenaNodeOp::Transform(_)) = node_ops[orig_idx].as_ref() else {
                continue;
            };
            for &(src, _slot) in &canonical_in_edges[orig_idx] {
                if let Some(owner) = buffer_owner_orig[src] {
                    remaining_buffer_consumers[owner] += 1;
                }
            }
        }

        let buffer_signatures: Vec<Option<(BandFormatId, u32)>> = node_ops
            .iter()
            .map(|op| match op {
                Some(ArenaNodeOp::Transform(t)) => Some((t.output_format(), t.bands())),
                _ => None,
            })
            .collect();

        // buffer_sizes[0] is a placeholder updated after the loop.
        let mut buffer_sizes: Vec<usize> = vec![0usize];
        let mut compiled_nodes: Vec<CompiledNode> = Vec::with_capacity(n);
        let mut node_compiled_idx: Vec<Option<usize>> = vec![None; n];
        let mut buffer_owner_compiled_idx: Vec<Option<usize>> = vec![None; n];
        let mut free_buffers: Vec<(BufferIdx, BandFormatId, u32)> = Vec::new();
        let mut max_buffer_idx = 0usize;

        for &orig_idx in &topo_order {
            // Kahn's algorithm enqueues a node only when its in-degree reaches zero,
            // which happens exactly once per node. `take()` returning None would mean
            // the same orig_idx appeared twice in topo_order — a bug in the sort,
            // not in user data. The debug_assert catches it in debug builds; the
            // `ok_or` surfaces it as a typed error in release builds.
            debug_assert!(
                node_ops[orig_idx].is_some(),
                "Kahn's sort emitted node {orig_idx} more than once"
            );
            let arena_op = node_ops[orig_idx]
                .take()
                .ok_or(BuildError::DuplicateNodeInTopoOrder(orig_idx))?;

            match arena_op {
                ArenaNodeOp::View(v) => {
                    // Zero-copy: view nodes share the upstream buffer — no new buffer needed.
                    // For DAG: a view node's upstream is always a single edge (slot 0).
                    let upstream = canonical_in_edges[orig_idx]
                        .iter()
                        .find(|&&(_, slot)| slot == 0)
                        .map(|&(src, _)| src);
                    let input_buf = upstream.and_then(|src| node_output_bufs[src]).unwrap_or(0);
                    let input_upstream = upstream.and_then(|src| node_compiled_idx[src]);
                    let input_buffer_producer =
                        upstream.and_then(|src| buffer_owner_compiled_idx[src]);
                    node_output_bufs[orig_idx] = Some(input_buf);
                    let compiled_idx = compiled_nodes.len();
                    compiled_nodes.push(CompiledNode::new(
                        CompiledOp::View(v),
                        vec![input_buf],
                        vec![input_upstream],
                        vec![input_buffer_producer],
                        input_buf,
                        None,
                    )?);
                    node_compiled_idx[orig_idx] = Some(compiled_idx);
                    buffer_owner_compiled_idx[orig_idx] = input_buffer_producer;
                }
                ArenaNodeOp::Transform(t) => {
                    let spec = t.node_spec(tile_w, tile_h);

                    // Collect input buffers for each slot. For single-input ops, slot 0
                    // comes from the upstream node (or source buffer 0 for the first node).
                    // For multi-input ops, each slot maps to a distinct upstream buffer.
                    let slot_count = t.input_slot_count();
                    let mut input_bufs: Vec<BufferIdx> = Vec::with_capacity(slot_count);
                    let mut input_upstreams = Vec::with_capacity(slot_count);
                    let mut input_buffer_producers = Vec::with_capacity(slot_count);
                    let mut input_buffer_owner_origs = Vec::with_capacity(slot_count);
                    for slot in 0..slot_count {
                        let upstream = canonical_in_edges[orig_idx]
                            .iter()
                            .find(|&&(_, s)| s as usize == slot)
                            .map(|&(src, _)| src);
                        let buf = upstream.and_then(|src| node_output_bufs[src]).unwrap_or(0);
                        input_bufs.push(buf);
                        input_upstreams.push(upstream.and_then(|src| node_compiled_idx[src]));
                        input_buffer_producers
                            .push(upstream.and_then(|src| buffer_owner_compiled_idx[src]));
                        input_buffer_owner_origs
                            .push(upstream.and_then(|src| buffer_owner_orig[src]));
                    }

                    let max_input_buf = input_bufs.iter().copied().max().unwrap_or(0);
                    let output_format = t.output_format();
                    let output_bands = t.bands();
                    let reuse_pos = free_buffers
                        .iter()
                        .enumerate()
                        .filter(|(_, (idx, format, bands))| {
                            *idx > max_input_buf
                                && *format == output_format
                                && *bands == output_bands
                        })
                        .min_by_key(|(_, (idx, _, _))| *idx)
                        .map(|(pos, _)| pos);
                    let output_buf = reuse_pos.map_or_else(
                        || {
                            max_buffer_idx += 1;
                            max_buffer_idx
                        },
                        |pos| free_buffers.swap_remove(pos).0,
                    );

                    // Ensure each input buffer is large enough for what this node declares.
                    for (slot, &ib) in input_bufs.iter().enumerate() {
                        let input_needed = spec.input_tile_w as usize
                            * spec.input_tile_h as usize
                            * t.input_bands_slot(slot) as usize
                            * format_sample_size(t.input_format_slot(slot));
                        if ib < buffer_sizes.len() {
                            buffer_sizes[ib] = buffer_sizes[ib].max(input_needed);
                        } else {
                            buffer_sizes.resize(ib + 1, 0);
                            buffer_sizes[ib] = input_needed;
                        }
                    }

                    // Size the output buffer from this node's declared output geometry.
                    let output_size = spec.output_tile_w as usize
                        * spec.output_tile_h as usize
                        * t.bands() as usize
                        * format_sample_size(t.output_format());
                    if output_buf >= buffer_sizes.len() {
                        buffer_sizes.resize(output_buf + 1, 0);
                    }
                    buffer_sizes[output_buf] = output_size;

                    node_output_bufs[orig_idx] = Some(output_buf);
                    let compiled_idx = compiled_nodes.len();
                    let cache_op_id = node_cache_enabled[orig_idx].then_some(compiled_idx);
                    compiled_nodes.push(CompiledNode::new(
                        CompiledOp::Transform(t),
                        input_bufs,
                        input_upstreams,
                        input_buffer_producers,
                        output_buf,
                        cache_op_id,
                    )?);
                    node_compiled_idx[orig_idx] = Some(compiled_idx);
                    buffer_owner_compiled_idx[orig_idx] = Some(compiled_idx);

                    for owner_orig in input_buffer_owner_origs.into_iter().flatten() {
                        if remaining_buffer_consumers[owner_orig] == 0 {
                            continue;
                        }
                        remaining_buffer_consumers[owner_orig] -= 1;
                        if remaining_buffer_consumers[owner_orig] == 0
                            && let (Some(freed_buf), Some((format, bands))) =
                                (node_output_bufs[owner_orig], buffer_signatures[owner_orig])
                        {
                            free_buffers.push((freed_buf, format, bands));
                        }
                    }
                }
            }
        }

        // buffer[0] must be large enough for the full source region implied by the
        // entire compiled chain, not just the first node's local `node_spec`.
        // A downstream halo op can back-propagate extra rows/columns through an
        // upstream halo op (e.g. GaussBlurH → GaussBlurV), so the source read for
        // one scheduler tile may exceed the first node's declared input tile size.
        // Compute that exact back-propagated region the same way the scheduler does
        // at runtime and size buffer[0] accordingly.
        let source_region_for_tile =
            source_region_for_scheduler_tile(&compiled_nodes, tile_w, tile_h);
        let source_region_size = source_region_for_tile.pixel_count() * source_bands * source_bps;
        buffer_sizes[0] = buffer_sizes[0].max(min_source_size).max(source_region_size);

        let buffer_count = max_buffer_idx + 1;

        let mut buffer_formats = vec![source.format(); buffer_count];
        let mut buffer_bands = vec![source.bands(); buffer_count];
        let mut buffer_producers = vec![None; buffer_count];
        for (node_idx, node) in compiled_nodes.iter().enumerate() {
            if let CompiledOp::Transform(op) = &node.op {
                buffer_formats[node.output_buf()] = op.output_format();
                buffer_bands[node.output_buf()] = op.bands();
                buffer_producers[node.output_buf()] = Some(node_idx);
            }
        }

        // Buffer sizes for intermediate outputs must cover the largest single-tile region a
        // downstream tile can demand from them, not the union of all tile coordinates across
        // the image. Unioning every tile's region inflates pointwise pipelines to full-image
        // scratch buffers, which defeats libvips-style demand execution.
        let mut buffer_max_pixels = vec![0usize; buffer_count];
        let mut node_max_output_regions = vec![Region::new(0, 0, 0, 0); compiled_nodes.len()];
        let (cols, rows) = if tile_w == 0 || tile_h == 0 {
            (0, 0)
        } else {
            (
                pipeline_out_w.div_ceil(tile_w),
                pipeline_out_h.div_ceil(tile_h),
            )
        };

        for row in 0..rows {
            for col in 0..cols {
                let x_u32 = col * tile_w;
                let y_u32 = row * tile_h;
                let x = i32::try_from(x_u32).map_err(|_| BuildError::ImageTooLarge {
                    width: pipeline_out_w,
                    height: pipeline_out_h,
                    bands: source.bands(),
                    bytes: u128::from(pipeline_out_w) * u128::from(pipeline_out_h),
                    limit_bytes: i32::MAX as u128,
                    details: "pipeline tile origin exceeds signed coordinate range",
                })?;
                let y = i32::try_from(y_u32).map_err(|_| BuildError::ImageTooLarge {
                    width: pipeline_out_w,
                    height: pipeline_out_h,
                    bands: source.bands(),
                    bytes: u128::from(pipeline_out_w) * u128::from(pipeline_out_h),
                    limit_bytes: i32::MAX as u128,
                    details: "pipeline tile origin exceeds signed coordinate range",
                })?;
                let output_region = Region::new(
                    x,
                    y,
                    tile_w.min(pipeline_out_w - x_u32),
                    tile_h.min(pipeline_out_h - y_u32),
                );

                let mut tile_regions = vec![Region::new(0, 0, 0, 0); buffer_count];
                tile_regions[buffer_count - 1] = output_region;

                for (node_idx, node) in compiled_nodes.iter().enumerate().rev() {
                    let output_buf = node.output_buf();
                    let node_output_region = tile_regions[output_buf];
                    if node_output_region.is_empty() {
                        continue;
                    }
                    node_max_output_regions[node_idx] =
                        max_region_extent(node_max_output_regions[node_idx], node_output_region);

                    match &node.op {
                        CompiledOp::Transform(op) => {
                            for slot in 0..op.input_slot_count() {
                                let input_buf = node.input_bufs()[slot];
                                let required =
                                    if input_buf == 0 && node.input_upstreams()[slot].is_none() {
                                        op.source_read_plan_slot(&node_output_region, slot)
                                            .produced_region()
                                    } else {
                                        op.required_input_region_slot(&node_output_region, slot)
                                    };
                                tile_regions[input_buf] =
                                    union_region(tile_regions[input_buf], required);
                            }
                        }
                        CompiledOp::View(view) => {
                            let input_buf = node.input_bufs()[0];
                            let required = view.required_input_region(&node_output_region);
                            tile_regions[input_buf] =
                                union_region(tile_regions[input_buf], required);
                        }
                    }
                }

                for (buf_idx, region) in tile_regions.into_iter().enumerate() {
                    buffer_max_pixels[buf_idx] =
                        buffer_max_pixels[buf_idx].max(region.pixel_count());
                }
            }
        }

        for (buf_idx, pixel_count) in buffer_max_pixels.iter().copied().enumerate() {
            if pixel_count == 0 {
                continue;
            }
            let size = pixel_count
                * buffer_bands[buf_idx] as usize
                * format_sample_size(buffer_formats[buf_idx]);
            buffer_sizes[buf_idx] = buffer_sizes[buf_idx].max(size);
        }

        // Propagate output image dimensions through the compiled node chain.
        // output_width/output_height are a separate responsibility from node_spec —
        // they track full image dimensions, not tile geometry.
        let mut out_w = self.width;
        let mut out_h = self.height;
        for node in &compiled_nodes {
            match &node.op {
                CompiledOp::Transform(t) => {
                    (out_w, out_h) = t.output_size(out_w, out_h);
                }
                CompiledOp::View(v) => {
                    out_w = v.output_width(out_w);
                    out_h = v.output_height(out_h);
                }
            }
        }

        let output_format = compiled_nodes.last().map_or_else(
            || source.format(),
            |n| match &n.op {
                CompiledOp::Transform(t) => t.output_format(),
                CompiledOp::View(v) => v.format(),
            },
        );

        let output_bands = compiled_nodes.last().map_or_else(
            || source.bands(),
            |n| match &n.op {
                CompiledOp::Transform(t) => t.bands(),
                CompiledOp::View(v) => v.bands(),
            },
        );

        let mut output_metadata = source.metadata();
        for node in &compiled_nodes {
            output_metadata = node.op.transform_metadata(&output_metadata);
        }

        let tile_cache = if node_cache_enabled.iter().any(|enabled| *enabled) {
            Some(OperationTileCache::new(
                cache_max_bytes.ok_or(BuildError::CacheMissingCapacity)?,
            ))
        } else {
            None
        };
        let sequential_line_cache = self
            .line_cache_request
            .map(|request| request.resolve(tile_h));
        let line_cache_access = self.line_cache_request.map(LineCacheRequest::access);

        Ok(CompiledPipeline {
            source,
            nodes: compiled_nodes,
            buffer_count,
            buffer_sizes: { buffer_sizes },
            sequential: self.sequential,
            sequential_line_cache,
            line_cache_access,
            demand_hint,
            width: out_w,
            height: out_h,
            output_format,
            output_bands,
            output_metadata,
            buffer_formats,
            buffer_bands,
            buffer_producers,
            node_max_output_regions,
            tile_cache,
        })
    }
}

/// Return the output format of an `ArenaNodeOp` (used for format-compatibility checks).
fn node_format(op: &ArenaNodeOp) -> BandFormatId {
    match op {
        ArenaNodeOp::Transform(t) => t.output_format(),
        ArenaNodeOp::View(v) => v.format(),
    }
}

fn node_input_format_slot(op: &ArenaNodeOp, slot: usize) -> BandFormatId {
    match op {
        ArenaNodeOp::Transform(t) => t.input_format_slot(slot),
        // ViewOp has no separate input format — it does not transform pixels.
        ArenaNodeOp::View(v) => v.format(),
    }
}

fn node_input_slot_count(op: &ArenaNodeOp) -> usize {
    match op {
        ArenaNodeOp::Transform(t) => t.input_slot_count(),
        ArenaNodeOp::View(_) => 1,
    }
}

/// `format_sample_size` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs_runtime::pipeline::arena::format_sample_size;
/// ```
pub(super) const fn format_sample_size(id: BandFormatId) -> usize {
    match id {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

fn union_region(lhs: Region, rhs: Region) -> Region {
    if lhs.is_empty() {
        return rhs;
    }
    if rhs.is_empty() {
        return lhs;
    }

    let left = lhs.x.min(rhs.x);
    let top = lhs.y.min(rhs.y);
    let right =
        (i64::from(lhs.x) + i64::from(lhs.width)).max(i64::from(rhs.x) + i64::from(rhs.width));
    let bottom =
        (i64::from(lhs.y) + i64::from(lhs.height)).max(i64::from(rhs.y) + i64::from(rhs.height));

    Region::new(
        left,
        top,
        saturating_u32_extent(i64::from(left), right),
        saturating_u32_extent(i64::from(top), bottom),
    )
}

fn max_region_extent(lhs: Region, rhs: Region) -> Region {
    if lhs.is_empty() {
        return Region::new(0, 0, rhs.width, rhs.height);
    }
    if rhs.is_empty() {
        return Region::new(0, 0, lhs.width, lhs.height);
    }

    Region::new(0, 0, lhs.width.max(rhs.width), lhs.height.max(rhs.height))
}

/// `source_region_for_scheduler_tile` exposes adapter behavior needed by the surrounding module.
/// Call it when you need the concrete operation implemented here.
///
/// # Examples
///
/// ```ignore
/// let _ = viprs_runtime::pipeline::arena::source_region_for_scheduler_tile;
/// ```
pub fn source_region_for_scheduler_tile(
    compiled_nodes: &[CompiledNode],
    tile_w: u32,
    tile_h: u32,
) -> Region {
    let mut source_region = Region::new(0, 0, tile_w, tile_h);
    for node in compiled_nodes.iter().rev() {
        source_region = match &node.op {
            CompiledOp::Transform(op) => {
                let coordinate_driven = op.coordinate_driven_source_spec().filter(|spec| {
                    spec.source_slot < node.input_bufs().len()
                        && node.input_bufs()[spec.source_slot] == 0
                        && node.input_upstreams()[spec.source_slot].is_none()
                });
                coordinate_driven.map_or_else(
                    || op.required_input_region(&source_region),
                    |spec| {
                        op.source_read_plan_slot(&source_region, spec.source_slot)
                            .produced_region()
                    },
                )
            }
            CompiledOp::View(view) => view.required_input_region(&source_region),
        };
    }
    source_region
}

fn saturating_u32_extent(start: i64, end: i64) -> u32 {
    let extent = end.saturating_sub(start);
    u32::try_from(extent.clamp(0, i64::from(u32::MAX))).map_or(u32::MAX, |value| value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_region_handles_i32_boundary_without_wrapping() {
        let lhs = Region::new(i32::MAX - 1, 0, 4, 1);
        let rhs = Region::new(i32::MAX, 0, 1, 1);

        assert_eq!(union_region(lhs, rhs), Region::new(i32::MAX - 1, 0, 4, 1));
    }
}
