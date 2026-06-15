use super::runtime::WORKER_POOL;
use super::{
    Arc, BandFormatId, CompiledOp, CompiledPipeline, DEFAULT_TARGET_L2_BYTES, DemandHint,
    DynImageSource, InputSlicePtr, Instant, L2_TILE_BUDGET_DIVISOR, MemorySink,
    PipelineOutputTarget, Region, SequentialLineCache, SourceReadPlan, ThreadBufferPool,
    TileGeometry, TileStrip, ViprsError,
};

pub(super) fn with_worker_pool<T>(
    pipeline: &CompiledPipeline,
    f: impl FnOnce(&mut ThreadBufferPool) -> Result<T, ViprsError>,
) -> Result<T, ViprsError> {
    WORKER_POOL.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let pool = borrow.get_or_insert_with(|| ThreadBufferPool::new(pipeline));
        f(pool)
    })
}

pub(super) const fn sample_bytes(id: BandFormatId) -> usize {
    match id {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

pub(super) fn checked_region_byte_size(
    region: Region,
    bands: usize,
    bps: usize,
) -> Result<usize, ViprsError> {
    let bands_u64 = u64::try_from(bands).unwrap_or(u64::MAX);
    let bps_u64 = u64::try_from(bps).unwrap_or(u64::MAX);
    let bands_u32 = u32::try_from(bands).unwrap_or(u32::MAX);
    let total_bytes =
        u128::from(region.width) * u128::from(region.height) * bands as u128 * bps as u128;

    let Some(bytes) = u64::from(region.width)
        .checked_mul(u64::from(region.height))
        .and_then(|pixel_count| pixel_count.checked_mul(bands_u64))
        .and_then(|sample_count| sample_count.checked_mul(bps_u64))
    else {
        return Err(ViprsError::ImageTooLarge {
            width: region.width,
            height: region.height,
            bands: bands_u32,
            bytes: total_bytes,
            limit_bytes: usize::MAX as u128,
            details: "scheduler byte count exceeds addressable memory",
        });
    };

    usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
        width: region.width,
        height: region.height,
        bands: bands_u32,
        bytes: u128::from(bytes),
        limit_bytes: usize::MAX as u128,
        details: "scheduler byte count exceeds addressable memory",
    })
}

#[inline]
pub(super) fn ensure_pool_buffers(pool: &mut ThreadBufferPool, pipeline: &CompiledPipeline) {
    debug_assert_eq!(pool.buffers.len(), pipeline.buffer_sizes.len());
    for (buffer, &required_len) in pool.buffers.iter_mut().zip(&pipeline.buffer_sizes) {
        if buffer.len() != required_len {
            buffer.resize(required_len, 0);
        }
    }
}

#[inline]
pub(super) const fn input_ptrs_as_slices(inputs: &[InputSlicePtr]) -> &[&[u8]] {
    // SAFETY: each entry is written immediately before use from a live `&[u8]` slice backed by pool storage that outlives the returned borrow, and `*const [u8]` shares the same fat-pointer layout as `&[u8]`.
    unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<&[u8]>(), inputs.len()) }
}

#[inline]
pub(super) const fn borrowed_input_slice(ptr: &InputSlicePtr) -> &[u8] {
    // SAFETY: `InputSlicePtr` is written from a live borrowed slice immediately before use and
    // never escapes the tile execution that owns the source borrow.
    unsafe { &*ptr.0 }
}

#[inline]
pub(super) fn source_borrowed_slice(pool: &ThreadBufferPool) -> Option<&[u8]> {
    pool.source_borrowed.as_ref().map(borrowed_input_slice)
}

#[inline]
pub(super) fn try_borrow_source_plan(
    source: &dyn DynImageSource,
    plan: SourceReadPlan,
) -> Option<&[u8]> {
    match plan {
        SourceReadPlan::Rect { region } => source.borrow_region(region),
        SourceReadPlan::PointGrid { .. } => None,
    }
}

pub(super) fn read_source_plan_with_cache(
    cache: Option<&mut SequentialLineCache>,
    source: &dyn DynImageSource,
    plan: SourceReadPlan,
    output: &mut [u8],
) -> Result<bool, ViprsError> {
    let Some(cache) = cache else {
        return Ok(false);
    };
    let SourceReadPlan::Rect { region } = plan else {
        return Ok(false);
    };
    cache.read_region(source, region, output)
}

pub(super) fn union_region(lhs: Region, rhs: Region) -> Region {
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

pub(super) fn merge_source_read_plan(
    lhs: SourceReadPlan,
    rhs: SourceReadPlan,
) -> (SourceReadPlan, bool) {
    if lhs == rhs {
        return (lhs, false);
    }

    (
        SourceReadPlan::rect(union_region(
            lhs.bounding_source_region(),
            rhs.bounding_source_region(),
        )),
        true,
    )
}

pub(super) fn region_contains(outer: Region, inner: Region) -> bool {
    if inner.is_empty() {
        return true;
    }

    outer.x <= inner.x
        && outer.y <= inner.y
        && i64::from(outer.x) + i64::from(outer.width)
            >= i64::from(inner.x) + i64::from(inner.width)
        && i64::from(outer.y) + i64::from(outer.height)
            >= i64::from(inner.y) + i64::from(inner.height)
}

pub(super) fn read_source_plan(
    source: &dyn DynImageSource,
    plan: SourceReadPlan,
    output: &mut [u8],
    pixel_bytes: usize,
) -> Result<(), ViprsError> {
    match plan {
        SourceReadPlan::Rect { region } => source.read_region(region, output),
        SourceReadPlan::PointGrid {
            input_region,
            source_origin_x,
            source_origin_y,
            x_step,
            y_step,
        } => {
            output.fill(0);
            let source_width = i64::from(source.width());
            let source_height = i64::from(source.height());
            for row in 0..input_region.height {
                for col in 0..input_region.width {
                    let source_x = i64::from(source_origin_x) + i64::from(col) * i64::from(x_step);
                    let source_y = i64::from(source_origin_y) + i64::from(row) * i64::from(y_step);
                    if (0..source_width).contains(&source_x)
                        && (0..source_height).contains(&source_y)
                    {
                        let source_region = Region::new(
                            i32::try_from(source_x).map_err(|_| {
                                ViprsError::Scheduler(format!(
                                    "point-grid x coordinate {source_x} exceeds signed coordinate range"
                                ))
                            })?,
                            i32::try_from(source_y).map_err(|_| {
                                ViprsError::Scheduler(format!(
                                    "point-grid y coordinate {source_y} exceeds signed coordinate range"
                                ))
                            })?,
                            1,
                            1,
                        );
                        let dst = (row as usize * input_region.width as usize + col as usize)
                            * pixel_bytes;
                        source.read_region(source_region, &mut output[dst..dst + pixel_bytes])?;
                    }
                }
            }
            Ok(())
        }
    }
}

pub(super) fn saturating_u32_extent(start: i64, end: i64) -> u32 {
    let extent = end.saturating_sub(start);
    u32::try_from(extent.clamp(0, i64::from(u32::MAX))).map_or(u32::MAX, |value| value)
}

pub(super) fn timed_profile<T, F>(slot: &mut u128, f: F) -> T
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = f();
    *slot += start.elapsed().as_nanos();
    result
}

pub(super) fn copy_packed_subregion(
    source: &[u8],
    source_region: Region,
    subregion: Region,
    scratch: &mut [u8],
    pixel_bytes: usize,
) {
    if subregion.is_empty() {
        return;
    }

    debug_assert!(
        region_contains(source_region, subregion),
        "subregion must be contained within the producer region"
    );

    let src_row_bytes = source_region.width as usize * pixel_bytes;
    let dst_row_bytes = subregion.width as usize * pixel_bytes;
    let x_offset = (subregion.x - source_region.x) as usize;
    let y_offset = (subregion.y - source_region.y) as usize;

    for row in 0..subregion.height as usize {
        let src_start = (y_offset + row) * src_row_bytes + x_offset * pixel_bytes;
        let dst_start = row * dst_row_bytes;
        scratch[dst_start..dst_start + dst_row_bytes]
            .copy_from_slice(&source[src_start..src_start + dst_row_bytes]);
    }
}

pub(super) fn copy_region_into_requested(
    source: &[u8],
    available_region: Region,
    requested_region: Region,
    scratch: &mut [u8],
    pixel_bytes: usize,
) {
    if requested_region.is_empty() {
        return;
    }

    scratch.fill(0);
    if available_region.is_empty() {
        return;
    }

    debug_assert!(
        region_contains(requested_region, available_region),
        "available region must be contained within the requested region"
    );

    let src_row_bytes = available_region.width as usize * pixel_bytes;
    let dst_row_bytes = requested_region.width as usize * pixel_bytes;
    let x_offset = (available_region.x - requested_region.x) as usize;
    let y_offset = (available_region.y - requested_region.y) as usize;

    for row in 0..available_region.height as usize {
        let src_start = row * src_row_bytes;
        let dst_start = (y_offset + row) * dst_row_bytes + x_offset * pixel_bytes;
        scratch[dst_start..dst_start + src_row_bytes]
            .copy_from_slice(&source[src_start..src_start + src_row_bytes]);
    }
}

pub(super) fn ensure_scratch_buffer(scratch: &mut Vec<u8>, required_len: usize) -> &mut [u8] {
    if scratch.len() < required_len {
        scratch.resize(required_len, 0);
    }
    &mut scratch[..required_len]
}

pub(super) fn logical_producer_region(
    pipeline: &CompiledPipeline,
    pool: &ThreadBufferPool,
    node: &crate::adapters::pipeline::CompiledNode,
    slot: usize,
    input_region: Region,
    producer_idx: Option<usize>,
) -> Region {
    if let Some(upstream_idx) = node
        .input_upstreams()
        .get(slot)
        .and_then(|upstream| *upstream)
    {
        return match &pipeline.nodes[upstream_idx].op {
            CompiledOp::View(view) => view.valid_output_region(&input_region),
            CompiledOp::Transform(_) => pool.node_output_regions[upstream_idx],
        };
    }

    producer_idx.map_or(pool.source_region, |upstream_idx| {
        pool.node_output_regions[upstream_idx]
    })
}

pub(super) fn cached_or_buffered_output<'a>(
    cached_tile: Option<&'a Arc<[u8]>>,
    buffers: &'a [Vec<u8>],
    output_buf: usize,
    out_bytes: usize,
) -> &'a [u8] {
    cached_tile.map_or(&buffers[output_buf][..out_bytes], |tile| &tile[..out_bytes])
}

pub(super) fn producer_output<'a>(
    source_borrowed: Option<&'a [u8]>,
    cached_tile: Option<&'a Arc<[u8]>>,
    buffers: &'a [Vec<u8>],
    input_buf: usize,
    producer_bytes: usize,
) -> &'a [u8] {
    source_borrowed.map_or_else(
        || {
            cached_tile.map_or(&buffers[input_buf][..producer_bytes], |tile| {
                &tile[..producer_bytes]
            })
        },
        |borrowed| &borrowed[..producer_bytes],
    )
}

pub(super) fn materialize_cached_tile(out_bytes: usize, fill: impl FnOnce(&mut [u8])) -> Arc<[u8]> {
    let mut tile = Arc::<[u8]>::new_uninit_slice(out_bytes);
    let Some(buffer) = Arc::get_mut(&mut tile) else {
        debug_assert!(false, "freshly allocated Arc tile must be uniquely owned");
        // SAFETY: `tile` was created by `Arc::new_uninit_slice` in this function and has not been cloned.
        unsafe { std::hint::unreachable_unchecked() }
    };
    let output_slice =
        // SAFETY: `MaybeUninit<u8>` has identical layout to `u8`, the Arc allocation is still uniquely owned here, and `fill` writes every byte before `assume_init` publishes the slice.
        unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), out_bytes) };
    fill(output_slice);
    // SAFETY: `fill` initialized every byte in the Arc allocation.
    unsafe { tile.assume_init() }
}

pub(super) fn propagate_required_region(
    pool: &mut ThreadBufferPool,
    input_buf: usize,
    input_upstream: Option<usize>,
    required_region: Region,
) {
    if input_buf == 0 && input_upstream.is_none() {
        if pool.source_region_assigned {
            let (plan, sparse_fallback) = merge_source_read_plan(
                pool.source_read_plan,
                SourceReadPlan::rect(required_region),
            );
            pool.source_read_plan = plan;
            pool.source_sparse_fallback |= sparse_fallback;
        } else {
            pool.source_read_plan = SourceReadPlan::rect(required_region);
            pool.source_region_assigned = true;
        }
        return;
    }

    if let Some(producer_idx) = input_upstream {
        propagate_node_output_plan(pool, producer_idx, SourceReadPlan::rect(required_region));
    }
}

pub(super) fn propagate_source_read_plan(
    pool: &mut ThreadBufferPool,
    input_buf: usize,
    input_upstream: Option<usize>,
    plan: SourceReadPlan,
) {
    if input_buf == 0 && input_upstream.is_none() {
        if pool.source_region_assigned {
            let (merged, sparse_fallback) = merge_source_read_plan(pool.source_read_plan, plan);
            pool.source_read_plan = merged;
            pool.source_sparse_fallback |= sparse_fallback;
        } else {
            pool.source_read_plan = plan;
            pool.source_region_assigned = true;
        }
        return;
    }

    if let Some(producer_idx) = input_upstream {
        propagate_node_output_plan(pool, producer_idx, plan);
    }
}

pub(super) fn propagate_node_output_plan(
    pool: &mut ThreadBufferPool,
    producer_idx: usize,
    plan: SourceReadPlan,
) {
    if pool.node_output_assigned[producer_idx] {
        let existing = pool.node_output_read_plans[producer_idx]
            .unwrap_or_else(|| SourceReadPlan::rect(pool.node_output_regions[producer_idx]));
        let (merged, _) = merge_source_read_plan(existing, plan);
        pool.node_output_regions[producer_idx] = merged.produced_region();
        pool.node_output_read_plans[producer_idx] = Some(merged);
    } else {
        pool.node_output_regions[producer_idx] = plan.produced_region();
        pool.node_output_read_plans[producer_idx] = Some(plan);
        pool.node_output_assigned[producer_idx] = true;
    }
}

pub(super) fn prepare_multi_input_slot(
    pipeline: &CompiledPipeline,
    pool: &mut ThreadBufferPool,
    node: &crate::adapters::pipeline::CompiledNode,
    node_idx: usize,
    buffers: &[Vec<u8>],
    slot: usize,
) -> Result<InputSlicePtr, ViprsError> {
    let input_buf = node.input_bufs()[slot];
    let input_region = pool.scratch_regions[node_idx][slot];
    let producer_idx = if input_buf == 0 {
        None
    } else {
        node.input_buffer_producers()[slot]
    };
    let producer_region =
        logical_producer_region(pipeline, pool, node, slot, input_region, producer_idx);
    let input_bands = pipeline.buffer_bands[input_buf] as usize;
    let input_bps = sample_bytes(pipeline.buffer_formats[input_buf]);
    let in_bytes = checked_region_byte_size(input_region, input_bands, input_bps)?;
    let producer_bytes = checked_region_byte_size(producer_region, input_bands, input_bps)?;
    let source_borrowed = if input_buf == 0 && producer_idx.is_none() {
        pool.source_borrowed.as_ref().map(borrowed_input_slice)
    } else {
        None
    };
    let producer = producer_output(
        source_borrowed,
        producer_idx.and_then(|upstream_idx| pool.cached_tiles[upstream_idx].as_ref()),
        buffers,
        input_buf,
        producer_bytes,
    );

    if producer_region == input_region {
        Ok(InputSlicePtr(&raw const producer[..in_bytes]))
    } else {
        let scratch =
            ensure_scratch_buffer(&mut pool.input_scratch_buffers[node_idx][slot], in_bytes);
        if region_contains(input_region, producer_region) {
            copy_region_into_requested(
                producer,
                producer_region,
                input_region,
                scratch,
                input_bands * input_bps,
            );
        } else {
            debug_assert!(
                region_contains(producer_region, input_region),
                "producer and requested regions must be nested"
            );
            copy_packed_subregion(
                producer,
                producer_region,
                input_region,
                scratch,
                input_bands * input_bps,
            );
        }
        Ok(InputSlicePtr(&raw const scratch[..in_bytes]))
    }
}

/// Partition the image described by `pipeline` into non-overlapping tiles.
///
/// The last column and row of tiles are clamped at the image boundary so no
/// tile extends beyond `(pipeline.width, pipeline.height)`.
#[allow(dead_code)]
// REASON: alternate scheduler entry points still use the default-budget helpers in follow-up work.
pub(super) fn generate_tiles(pipeline: &CompiledPipeline) -> Result<Vec<Region>, ViprsError> {
    generate_tiles_with_l2_budget(pipeline, DEFAULT_TARGET_L2_BYTES)
}

pub(super) fn generate_tiles_with_l2_budget(
    pipeline: &CompiledPipeline,
    target_l2_bytes: usize,
) -> Result<Vec<Region>, ViprsError> {
    let geometry = tile_geometry_for_l2_budget(pipeline, target_l2_bytes)?;

    let mut tiles = Vec::with_capacity((geometry.cols * geometry.rows) as usize);
    for row in 0..geometry.rows {
        for col in 0..geometry.cols {
            let x = col * geometry.tile_width;
            let y = row * geometry.tile_height;
            let w = geometry.tile_width.min(pipeline.width - x);
            let h = geometry.tile_height.min(pipeline.height - y);
            let x = i32::try_from(x).map_err(|_| {
                ViprsError::Scheduler(format!(
                    "tile x origin {x} exceeds signed coordinate range for {}x{} output",
                    pipeline.width, pipeline.height
                ))
            })?;
            let y = i32::try_from(y).map_err(|_| {
                ViprsError::Scheduler(format!(
                    "tile y origin {y} exceeds signed coordinate range for {}x{} output",
                    pipeline.width, pipeline.height
                ))
            })?;
            tiles.push(Region::new(x, y, w, h));
        }
    }
    Ok(tiles)
}

#[allow(dead_code)]
// REASON: alternate scheduler entry points still use the default-budget helpers in follow-up work.
pub(super) fn tile_geometry(pipeline: &CompiledPipeline) -> Result<TileGeometry, ViprsError> {
    tile_geometry_for_l2_budget(pipeline, DEFAULT_TARGET_L2_BYTES)
}

pub(super) fn tile_geometry_for_l2_budget(
    pipeline: &CompiledPipeline,
    target_l2_bytes: usize,
) -> Result<TileGeometry, ViprsError> {
    if pipeline.width == 0 || pipeline.height == 0 {
        return Ok(TileGeometry {
            tile_width: 0,
            tile_height: 0,
            cols: 0,
            rows: 0,
        });
    }

    let tile_width = pipeline.demand_hint.tile_width(pipeline.width);
    let tile_height = tile_height_for_hint(pipeline, target_l2_bytes)?;
    let cols = pipeline.width.div_ceil(tile_width);
    let rows = pipeline.height.div_ceil(tile_height);

    Ok(TileGeometry {
        tile_width,
        tile_height,
        cols,
        rows,
    })
}

pub(super) fn tile_height_for_hint(
    pipeline: &CompiledPipeline,
    target_l2_bytes: usize,
) -> Result<u32, ViprsError> {
    let default_height = pipeline
        .demand_hint
        .tile_height(pipeline.width, pipeline.height)
        .min(pipeline.height)
        .max(1);

    match pipeline.demand_hint {
        DemandHint::FatStrip | DemandHint::Any => {
            let l2_budget = target_l2_bytes
                .saturating_div(L2_TILE_BUDGET_DIVISOR)
                .max(1);
            let bytes_per_row = checked_region_byte_size(
                Region::new(0, 0, pipeline.width, 1),
                pipeline.output_bands as usize,
                sample_bytes(pipeline.output_format),
            )?;
            let l2_height = l2_budget.saturating_div(bytes_per_row.max(1)).max(1) as u32;
            Ok(default_height.min(l2_height).min(pipeline.height).max(1))
        }
        _ => Ok(default_height),
    }
}

#[allow(dead_code)]
// REASON: alternate scheduler entry points still use the default-budget helpers in follow-up work.
pub(super) fn generate_tile_strips(
    pipeline: &CompiledPipeline,
    strip_height_tiles: usize,
) -> Result<Vec<TileStrip>, ViprsError> {
    generate_tile_strips_with_l2_budget(pipeline, strip_height_tiles, DEFAULT_TARGET_L2_BYTES)
}

pub(super) fn generate_tile_strips_with_l2_budget(
    pipeline: &CompiledPipeline,
    strip_height_tiles: usize,
    target_l2_bytes: usize,
) -> Result<Vec<TileStrip>, ViprsError> {
    let geometry = tile_geometry_for_l2_budget(pipeline, target_l2_bytes)?;
    let tiles = generate_tiles_with_l2_budget(pipeline, target_l2_bytes)?;
    let strip_height_tiles = strip_height_tiles.max(1) as u32;
    let mut strips = Vec::with_capacity(geometry.rows.div_ceil(strip_height_tiles) as usize);

    let mut row = 0;
    while row < geometry.rows {
        let row_end = (row + strip_height_tiles).min(geometry.rows);
        let start = (row * geometry.cols) as usize;
        let end = (row_end * geometry.cols) as usize;
        strips.push(TileStrip {
            regions: tiles[start..end].to_vec(),
        });
        row = row_end;
    }

    Ok(strips)
}

#[inline]
pub(super) fn pipeline_output_target(pipeline: &CompiledPipeline) -> Option<PipelineOutputTarget> {
    let last = pipeline.nodes.last()?;
    let CompiledOp::Transform(op) = &last.op else {
        return Some(PipelineOutputTarget {
            output_buf: last.output_buf(),
            output_bands: pipeline.output_bands,
            output_format: pipeline.output_format,
        });
    };

    Some(PipelineOutputTarget {
        output_buf: last.output_buf(),
        output_bands: op.bands(),
        output_format: op.output_format(),
    })
}

#[inline]
pub(super) fn source_only_output_bytes(
    pipeline: &CompiledPipeline,
    region: Region,
) -> Result<usize, ViprsError> {
    checked_region_byte_size(
        region,
        pipeline.output_bands as usize,
        sample_bytes(pipeline.output_format),
    )
}

#[inline]
pub(super) fn pipeline_output_bytes(
    target: PipelineOutputTarget,
    region: Region,
) -> Result<usize, ViprsError> {
    checked_region_byte_size(
        region,
        target.output_bands as usize,
        sample_bytes(target.output_format),
    )
}

#[inline]
pub(super) fn pipeline_output_slice(
    target: PipelineOutputTarget,
    pool: &ThreadBufferPool,
    out_bytes: usize,
) -> &[u8] {
    cached_or_buffered_output(
        pool.cached_tiles.last().and_then(Option::as_ref),
        &pool.buffers,
        target.output_buf,
        out_bytes,
    )
}

#[inline]
pub(super) fn source_only_output_slice(pool: &ThreadBufferPool, out_bytes: usize) -> &[u8] {
    source_borrowed_slice(pool).map_or(&pool.buffers[0][..out_bytes], |borrowed| {
        &borrowed[..out_bytes]
    })
}

pub(super) fn direct_source_region_for_output(
    pipeline: &CompiledPipeline,
    output_region: Region,
) -> Option<Region> {
    let mut required_region = output_region;
    for node in pipeline.nodes.iter().rev() {
        match &node.op {
            CompiledOp::View(view) => {
                required_region = view.required_input_region(&required_region);
            }
            CompiledOp::Transform(_) => return None,
        }
    }
    Some(required_region)
}

pub(super) fn pipeline_reads_source_directly(pipeline: &CompiledPipeline) -> bool {
    pipeline
        .nodes
        .iter()
        .all(|node| matches!(node.op, CompiledOp::View(_)))
}

pub(super) fn can_direct_write_all_regions(sink: &MemorySink, regions: &[Region]) -> bool {
    regions
        .iter()
        .copied()
        .all(|region| sink.is_contiguous_full_width_region(region))
}
