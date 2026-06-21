use super::{
    Arc, CancellationToken, CompiledOp, CompiledPipeline, InputSlicePtr, PipelineOutputTarget,
    Region, SchedulerContractError, SourceReadPlan, ThreadBufferPool, TileLockScope,
    TileRunProfile, ViprsError, ensure_not_cancelled,
    planning::{
        borrowed_input_slice, checked_region_byte_size, copy_packed_subregion,
        copy_region_into_requested, ensure_pool_buffers, ensure_scratch_buffer,
        generate_tile_strips_with_l2_budget, input_ptrs_as_slices, logical_producer_region,
        materialize_cached_tile, pipeline_output_bytes, pipeline_output_slice,
        prepare_multi_input_slot, producer_output, propagate_required_region,
        propagate_source_read_plan, read_source_plan, read_source_plan_with_cache, region_contains,
        sample_bytes, source_only_output_bytes, source_only_output_slice, timed_profile,
        try_borrow_source_plan,
    },
    viprs_span,
};

pub(super) fn run_tiles_sequential_non_empty(
    pipeline: &CompiledPipeline,
    output_target: PipelineOutputTarget,
    strip_height_tiles: usize,
    target_l2_bytes: usize,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let strips =
        generate_tile_strips_with_l2_budget(pipeline, strip_height_tiles, target_l2_bytes)?;
    let mut pool = ThreadBufferPool::new(pipeline);

    for strip in strips {
        for &region in &strip.regions {
            let _tile_scope = TileLockScope::new();
            execute_tile(pipeline, region, &mut pool, None)?;
            let out_bytes = pipeline_output_bytes(output_target, region)?;
            let output_slice = pipeline_output_slice(output_target, &pool, out_bytes);
            write_tile(region, output_slice)?;
        }
    }

    Ok(())
}

pub(super) fn run_tiles_sequential_non_empty_cancellable(
    pipeline: &CompiledPipeline,
    output_target: PipelineOutputTarget,
    strip_height_tiles: usize,
    target_l2_bytes: usize,
    token: &CancellationToken,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let strips =
        generate_tile_strips_with_l2_budget(pipeline, strip_height_tiles, target_l2_bytes)?;
    let mut pool = ThreadBufferPool::new(pipeline);

    for strip in strips {
        for &region in &strip.regions {
            ensure_not_cancelled(token)?;
            let _tile_scope = TileLockScope::new();
            execute_tile(pipeline, region, &mut pool, None)?;
            let out_bytes = pipeline_output_bytes(output_target, region)?;
            let output_slice = pipeline_output_slice(output_target, &pool, out_bytes);
            write_tile(region, output_slice)?;
        }
    }

    Ok(())
}

pub(super) fn run_tiles_sequential_source_only(
    pipeline: &CompiledPipeline,
    strip_height_tiles: usize,
    target_l2_bytes: usize,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let strips =
        generate_tile_strips_with_l2_budget(pipeline, strip_height_tiles, target_l2_bytes)?;
    let mut pool = ThreadBufferPool::new(pipeline);

    for strip in strips {
        for &region in &strip.regions {
            let _tile_scope = TileLockScope::new();
            execute_tile(pipeline, region, &mut pool, None)?;
            let out_bytes = source_only_output_bytes(pipeline, region)?;
            let output_slice = source_only_output_slice(&pool, out_bytes);
            write_tile(region, output_slice)?;
        }
    }

    Ok(())
}

pub(super) fn run_tiles_sequential_source_only_cancellable(
    pipeline: &CompiledPipeline,
    strip_height_tiles: usize,
    target_l2_bytes: usize,
    token: &CancellationToken,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let strips =
        generate_tile_strips_with_l2_budget(pipeline, strip_height_tiles, target_l2_bytes)?;
    let mut pool = ThreadBufferPool::new(pipeline);

    for strip in strips {
        for &region in &strip.regions {
            ensure_not_cancelled(token)?;
            let _tile_scope = TileLockScope::new();
            execute_tile(pipeline, region, &mut pool, None)?;
            let out_bytes = source_only_output_bytes(pipeline, region)?;
            let output_slice = source_only_output_slice(&pool, out_bytes);
            write_tile(region, output_slice)?;
        }
    }

    Ok(())
}

pub(super) fn run_single_tile_non_empty(
    pipeline: &CompiledPipeline,
    output_target: PipelineOutputTarget,
    region: Region,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let mut pool = ThreadBufferPool::new(pipeline);
    let _tile_scope = TileLockScope::new();
    execute_tile(pipeline, region, &mut pool, None)?;
    let out_bytes = pipeline_output_bytes(output_target, region)?;
    let output_slice = pipeline_output_slice(output_target, &pool, out_bytes);
    write_tile(region, output_slice)
}

pub(super) fn run_single_tile_non_empty_cancellable(
    pipeline: &CompiledPipeline,
    output_target: PipelineOutputTarget,
    region: Region,
    token: &CancellationToken,
    write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    ensure_not_cancelled(token)?;
    run_single_tile_non_empty(pipeline, output_target, region, write_tile)
}

pub(super) fn run_single_tile_source_only(
    pipeline: &CompiledPipeline,
    region: Region,
    mut write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    let mut pool = ThreadBufferPool::new(pipeline);
    let _tile_scope = TileLockScope::new();
    execute_tile(pipeline, region, &mut pool, None)?;
    let out_bytes = source_only_output_bytes(pipeline, region)?;
    let output_slice = source_only_output_slice(&pool, out_bytes);
    write_tile(region, output_slice)
}

pub(super) fn run_single_tile_source_only_cancellable(
    pipeline: &CompiledPipeline,
    region: Region,
    token: &CancellationToken,
    write_tile: impl FnMut(Region, &[u8]) -> Result<(), ViprsError>,
) -> Result<(), ViprsError> {
    ensure_not_cancelled(token)?;
    run_single_tile_source_only(pipeline, region, write_tile)
}

/// Execute a single tile through the full pipeline node chain in place.
///
/// After this call, the last node's pixels for `output_region` are available in
/// `pool.buffers[last_node.output_buf()]` or `pool.cached_tiles[last_node]` when
/// the node materialized directly into the cache.
///
/// **Split-borrow invariant** (single- and multi-input): `compile()` assigns
/// buffer indices monotonically, so for every Transform node all `input_buf`
/// indices are strictly less than `output_buf`. `split_at_mut(output_buf)`
/// therefore yields disjoint slices for both paths without copying.
pub(super) fn execute_tile(
    pipeline: &CompiledPipeline,
    output_region: Region,
    pool: &mut ThreadBufferPool,
    profile: Option<&mut TileRunProfile>,
) -> Result<(), ViprsError> {
    viprs_span!(tracing::Level::TRACE, "viprs.tile");
    let node_count = pipeline.nodes.len();
    let mut profile = profile;
    ensure_pool_buffers(pool, pipeline);

    if node_count == 0 {
        let out_bytes = source_only_output_bytes(pipeline, output_region)?;
        if out_bytes == 0 {
            return Ok(());
        }

        let plan = SourceReadPlan::rect(output_region);
        let output = &mut pool.buffers[0][..out_bytes];
        let pixel_bytes = pipeline.output_bands as usize * sample_bytes(pipeline.output_format);
        let cached = {
            let cache = pool.sequential_line_cache.as_mut();
            if let Some(profile) = profile.as_mut() {
                profile.source_read_count += 1;
                timed_profile(&mut profile.source_read_ns, || {
                    read_source_plan_with_cache(cache, pipeline.source.as_ref(), plan, output)
                })?
            } else {
                read_source_plan_with_cache(cache, pipeline.source.as_ref(), plan, output)?
            }
        };
        if cached {
            pool.source_region = output_region;
            pool.source_borrowed = None;
        } else {
            let borrowed = if let Some(profile) = profile.as_mut() {
                timed_profile(&mut profile.source_read_ns, || {
                    try_borrow_source_plan(pipeline.source.as_ref(), plan)
                })
            } else {
                try_borrow_source_plan(pipeline.source.as_ref(), plan)
            };
            if let Some(borrowed) = borrowed {
                pool.source_region = output_region;
                pool.source_borrowed = Some(InputSlicePtr(std::ptr::from_ref::<[u8]>(borrowed)));
            } else {
                pool.source_borrowed = None;
                if let Some(profile) = profile.as_mut() {
                    timed_profile(&mut profile.source_read_ns, || {
                        read_source_plan(pipeline.source.as_ref(), plan, output, pixel_bytes)
                    })?;
                } else {
                    read_source_plan(pipeline.source.as_ref(), plan, output, pixel_bytes)?;
                }
            }
        }

        return Ok(());
    }

    if node_count == 1 {
        let node = &pipeline.nodes[0];
        if let CompiledOp::Transform(op) = &node.op
            && op.input_slot_count() == 1
            && node.input_bufs()[0] == 0
            && node.output_buf() > 0
        {
            let out_bytes = checked_region_byte_size(
                output_region,
                op.bands() as usize,
                sample_bytes(op.output_format()),
            )?;
            if let (Some(cache), Some(cache_op_id)) = (&pipeline.tile_cache, node.cache_op_id())
                && let Some(cached_tile) = cache.get(cache_op_id, output_region)?
            {
                if let Some(profile) = profile.as_mut() {
                    profile.nodes[0].cache_hits += 1;
                }
                if cached_tile.len() != out_bytes {
                    return Err(ViprsError::Scheduler(format!(
                        "cached tile length {} does not match expected output length {} for single-node fast path",
                        cached_tile.len(),
                        out_bytes,
                    )));
                }
                pool.cached_tiles[0] = Some(cached_tile);
                return Ok(());
            }

            let source_plan = op.source_read_plan_slot(&output_region, 0);
            let input_region = source_plan.produced_region();
            let source_bands = pipeline.buffer_bands[0] as usize;
            let source_bps = sample_bytes(pipeline.buffer_formats[0]);
            let source_bytes = checked_region_byte_size(input_region, source_bands, source_bps)?;

            let cached = if source_bytes > 0 {
                let cache = pool.sequential_line_cache.as_mut();
                if let Some(profile) = profile.as_mut() {
                    profile.source_read_count += 1;
                    timed_profile(&mut profile.source_read_ns, || {
                        read_source_plan_with_cache(
                            cache,
                            pipeline.source.as_ref(),
                            source_plan,
                            &mut pool.buffers[0][..source_bytes],
                        )
                    })?
                } else {
                    read_source_plan_with_cache(
                        cache,
                        pipeline.source.as_ref(),
                        source_plan,
                        &mut pool.buffers[0][..source_bytes],
                    )?
                }
            } else {
                false
            };
            let borrowed = if source_bytes > 0 && !cached {
                if let Some(profile) = profile.as_mut() {
                    timed_profile(&mut profile.source_read_ns, || {
                        try_borrow_source_plan(pipeline.source.as_ref(), source_plan)
                    })
                } else {
                    try_borrow_source_plan(pipeline.source.as_ref(), source_plan)
                }
            } else {
                None
            };
            if source_bytes > 0 && borrowed.is_none() && !cached {
                if let Some(profile) = profile.as_mut() {
                    timed_profile(&mut profile.source_read_ns, || {
                        read_source_plan(
                            pipeline.source.as_ref(),
                            source_plan,
                            &mut pool.buffers[0][..source_bytes],
                            source_bands * source_bps,
                        )
                    })?;
                } else {
                    read_source_plan(
                        pipeline.source.as_ref(),
                        source_plan,
                        &mut pool.buffers[0][..source_bytes],
                        source_bands * source_bps,
                    )?;
                }
            }

            let state = pool.op_states[0].as_mut().ok_or_else(|| {
                ViprsError::from(SchedulerContractError::MissingTransformState { node: 0 })
            })?;
            let (left, right) = pool.buffers.split_at_mut(node.output_buf());
            let input_slice = borrowed.unwrap_or(&left[0][..source_bytes]);

            op.validate_region_contract(
                input_region,
                pipeline.buffer_bands[0],
                output_region,
                op.bands(),
            )?;

            if let (Some(cache), Some(cache_op_id)) = (&pipeline.tile_cache, node.cache_op_id()) {
                let tile = if let Some(profile) = profile.as_mut() {
                    profile.nodes[0].exec_count += 1;
                    materialize_cached_tile(out_bytes, |output_slice| {
                        timed_profile(&mut profile.nodes[0].process_ns, || {
                            op.dyn_process_region(
                                state.as_mut(),
                                input_slice,
                                output_slice,
                                input_region,
                                output_region,
                            );
                        });
                    })
                } else {
                    materialize_cached_tile(out_bytes, |output_slice| {
                        op.dyn_process_region(
                            state.as_mut(),
                            input_slice,
                            output_slice,
                            input_region,
                            output_region,
                        );
                    })
                };
                cache.insert(cache_op_id, output_region, Arc::clone(&tile))?;
                pool.cached_tiles[0] = Some(tile);
            } else {
                let output_slice = &mut right[0][..out_bytes];
                if let Some(profile) = profile.as_mut() {
                    profile.nodes[0].exec_count += 1;
                    timed_profile(&mut profile.nodes[0].process_ns, || {
                        op.dyn_process_region(
                            state.as_mut(),
                            input_slice,
                            output_slice,
                            input_region,
                            output_region,
                        );
                    });
                } else {
                    op.dyn_process_region(
                        state.as_mut(),
                        input_slice,
                        output_slice,
                        input_region,
                        output_region,
                    );
                }
            }
            return Ok(());
        }
    }

    let empty_region = Region::new(0, 0, 0, 0);
    for (regions, assigned) in pool
        .scratch_regions
        .iter_mut()
        .zip(pool.node_output_assigned.iter_mut())
    {
        for region in regions {
            *region = empty_region;
        }
        *assigned = false;
    }
    pool.node_output_regions.fill(empty_region);
    pool.node_output_read_plans.fill(None);
    pool.source_region = empty_region;
    pool.source_read_plan = SourceReadPlan::rect(empty_region);
    pool.source_region_assigned = false;
    pool.source_sparse_fallback = false;
    pool.source_borrowed = None;
    pool.cached_tiles.fill(None);

    pool.node_output_regions[node_count - 1] = output_region;
    pool.node_output_assigned[node_count - 1] = true;

    for i in (0..node_count).rev() {
        if !pool.node_output_assigned[i] {
            continue;
        }

        let node_output_region = pool.node_output_regions[i];
        let node_output_plan = pool.node_output_read_plans[i]
            .unwrap_or_else(|| SourceReadPlan::rect(node_output_region));
        match &pipeline.nodes[i].op {
            CompiledOp::Transform(op) => {
                let coordinate_driven = op.coordinate_driven_source_spec();
                if let (Some(cache), Some(cache_op_id)) =
                    (&pipeline.tile_cache, pipeline.nodes[i].cache_op_id())
                    && let Some(cached_tile) = cache.get(cache_op_id, node_output_region)?
                {
                    if let Some(profile) = profile.as_mut() {
                        profile.nodes[i].cache_hits += 1;
                    }
                    pool.cached_tiles[i] = Some(cached_tile);
                    continue;
                }

                for slot in 0..op.input_slot_count() {
                    let input_buf = pipeline.nodes[i].input_bufs()[slot];
                    let direct_source =
                        input_buf == 0 && pipeline.nodes[i].input_upstreams()[slot].is_none();
                    if direct_source
                        && coordinate_driven.is_some_and(|spec| spec.source_slot == slot)
                    {
                        pool.scratch_regions[i][slot] = empty_region;
                        continue;
                    }
                    let source_plan = if op.input_slot_count() == 1 && op.is_pixel_local() {
                        node_output_plan
                    } else {
                        op.source_read_plan_slot(&node_output_region, slot)
                    };
                    let input_region = source_plan.produced_region();
                    pool.scratch_regions[i][slot] = input_region;
                    if direct_source || matches!(source_plan, SourceReadPlan::PointGrid { .. }) {
                        propagate_source_read_plan(
                            pool,
                            input_buf,
                            pipeline.nodes[i].input_upstreams()[slot],
                            source_plan,
                        );
                    } else {
                        propagate_required_region(
                            pool,
                            input_buf,
                            pipeline.nodes[i].input_upstreams()[slot],
                            input_region,
                        );
                    }
                }
            }
            CompiledOp::View(view) => {
                let valid_output_region = view.valid_output_region(&node_output_region);
                let input_region = view.required_input_region(&valid_output_region);
                propagate_required_region(
                    pool,
                    pipeline.nodes[i].input_bufs()[0],
                    pipeline.nodes[i].input_upstreams()[0],
                    input_region,
                );
            }
        }
    }

    if pool.source_sparse_fallback {
        for (i, node) in pipeline.nodes.iter().enumerate() {
            let CompiledOp::Transform(op) = &node.op else {
                continue;
            };
            if !pool.node_output_assigned[i] {
                continue;
            }

            for slot in 0..op.input_slot_count() {
                if node.input_bufs()[slot] == 0
                    && op
                        .coordinate_driven_source_spec()
                        .is_none_or(|spec| spec.source_slot != slot)
                {
                    pool.scratch_regions[i][slot] = op
                        .source_read_plan_slot(&pool.node_output_regions[i], slot)
                        .bounding_source_region();
                }
            }
        }
    }

    if pool.source_region_assigned && !pool.source_read_plan.produced_region().is_empty() {
        let source_bands = pipeline.buffer_bands[0] as usize;
        let source_bps = sample_bytes(pipeline.buffer_formats[0]);
        pool.source_region = pool.source_read_plan.produced_region();
        let source_bytes = checked_region_byte_size(pool.source_region, source_bands, source_bps)?;
        // `CompiledPipeline::source` stays erased on purpose: this single
        // per-tile vtable call below the accepted regression budget, while the tile copy
        // dominates the cost for real scheduler tile sizes.
        let cached = {
            let cache = pool.sequential_line_cache.as_mut();
            if let Some(profile) = profile.as_mut() {
                profile.source_read_count += 1;
                timed_profile(&mut profile.source_read_ns, || {
                    read_source_plan_with_cache(
                        cache,
                        pipeline.source.as_ref(),
                        pool.source_read_plan,
                        &mut pool.buffers[0][..source_bytes],
                    )
                })?
            } else {
                read_source_plan_with_cache(
                    cache,
                    pipeline.source.as_ref(),
                    pool.source_read_plan,
                    &mut pool.buffers[0][..source_bytes],
                )?
            }
        };
        if cached {
            pool.source_borrowed = None;
        } else {
            let borrowed = if let Some(profile) = profile.as_mut() {
                timed_profile(&mut profile.source_read_ns, || {
                    try_borrow_source_plan(pipeline.source.as_ref(), pool.source_read_plan)
                })
            } else {
                try_borrow_source_plan(pipeline.source.as_ref(), pool.source_read_plan)
            };
            if let Some(borrowed) = borrowed {
                pool.source_borrowed = Some(InputSlicePtr(std::ptr::from_ref::<[u8]>(borrowed)));
            } else {
                pool.source_borrowed = None;
                if let Some(profile) = profile.as_mut() {
                    timed_profile(&mut profile.source_read_ns, || {
                        read_source_plan(
                            pipeline.source.as_ref(),
                            pool.source_read_plan,
                            &mut pool.buffers[0][..source_bytes],
                            source_bands * source_bps,
                        )
                    })?;
                } else {
                    read_source_plan(
                        pipeline.source.as_ref(),
                        pool.source_read_plan,
                        &mut pool.buffers[0][..source_bytes],
                        source_bands * source_bps,
                    )?;
                }
            }
        }
    }

    // Phase 2: execute nodes front to back.
    //
    // View nodes are zero-copy: their output_buf == input_buf and Phase 1 already
    // computed the correct source region, so the upstream buffer contains the right
    // data. Nothing to do for View nodes.
    //
    // Both single- and multi-input Transform nodes use split_at_mut(output_buf) to
    // obtain non-overlapping input/output slices without copying. All input_buf
    // indices are < output_buf (compile() monotone assignment), so every input lands
    // in the left half and the output lands at right[0]. `op_states` and `buffers`
    // are disjoint fields of `ThreadBufferPool`, so NLL allows holding a mutable
    // borrow of each simultaneously.
    //
    for (i, node) in pipeline.nodes.iter().enumerate() {
        if !pool.node_output_assigned[i] {
            continue;
        }

        let CompiledOp::Transform(op) = &node.op else {
            // View node: upstream buffer already contains the correct pixels.
            continue;
        };

        let output_region_for_node = pool.node_output_regions[i];

        let out_bytes = checked_region_byte_size(
            output_region_for_node,
            op.bands() as usize,
            sample_bytes(op.output_format()),
        )?;

        if let Some(cached_tile) = pool.cached_tiles[i].as_ref() {
            if cached_tile.len() != out_bytes {
                return Err(ViprsError::Scheduler(format!(
                    "cached tile length {} does not match expected output length {} for node {i}",
                    cached_tile.len(),
                    out_bytes,
                )));
            }
            continue;
        }

        if op.input_slot_count() == 1 {
            let input_buf = node.input_bufs()[0];
            let input_region = pool.scratch_regions[i][0];
            let producer_idx = if input_buf == 0 {
                None
            } else {
                node.input_buffer_producers()[0]
            };
            let producer_region =
                logical_producer_region(pipeline, pool, node, 0, input_region, producer_idx);
            let input_bands = pipeline.buffer_bands[input_buf] as usize;
            let input_bps = sample_bytes(pipeline.buffer_formats[input_buf]);
            let in_bytes = checked_region_byte_size(input_region, input_bands, input_bps)?;
            let producer_bytes = checked_region_byte_size(producer_region, input_bands, input_bps)?;
            let pixel_bytes = input_bands * input_bps;
            let (left, right) = pool.buffers.split_at_mut(node.output_buf());
            let source_borrowed = if input_buf == 0 && producer_idx.is_none() {
                pool.source_borrowed.as_ref().map(borrowed_input_slice)
            } else {
                None
            };
            let producer = producer_output(
                source_borrowed,
                producer_idx.and_then(|upstream_idx| pool.cached_tiles[upstream_idx].as_ref()),
                left,
                input_buf,
                producer_bytes,
            );
            let input_slice = if producer_region == input_region {
                &producer[..in_bytes]
            } else {
                let scratch =
                    ensure_scratch_buffer(&mut pool.input_scratch_buffers[i][0], in_bytes);
                if region_contains(input_region, producer_region) {
                    copy_region_into_requested(
                        producer,
                        producer_region,
                        input_region,
                        scratch,
                        pixel_bytes,
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
                        pixel_bytes,
                    );
                }
                &scratch[..in_bytes]
            };

            op.validate_region_contract(
                input_region,
                input_bands as u32,
                output_region_for_node,
                op.bands(),
            )?;

            let state = pool.op_states[i].as_mut().ok_or_else(|| {
                ViprsError::from(SchedulerContractError::MissingTransformState { node: i })
            })?;
            if let (Some(cache), Some(cache_op_id)) = (&pipeline.tile_cache, node.cache_op_id()) {
                let tile = if let Some(profile) = profile.as_mut() {
                    profile.nodes[i].exec_count += 1;
                    materialize_cached_tile(out_bytes, |output_slice| {
                        timed_profile(&mut profile.nodes[i].process_ns, || {
                            op.dyn_process_region(
                                state.as_mut(),
                                input_slice,
                                output_slice,
                                input_region,
                                output_region_for_node,
                            );
                        });
                    })
                } else {
                    materialize_cached_tile(out_bytes, |output_slice| {
                        op.dyn_process_region(
                            state.as_mut(),
                            input_slice,
                            output_slice,
                            input_region,
                            output_region_for_node,
                        );
                    })
                };
                cache.insert(cache_op_id, output_region_for_node, Arc::clone(&tile))?;
                pool.cached_tiles[i] = Some(tile);
            } else {
                let output_slice = &mut right[0][..out_bytes];
                if let Some(profile) = profile.as_mut() {
                    profile.nodes[i].exec_count += 1;
                    timed_profile(&mut profile.nodes[i].process_ns, || {
                        op.dyn_process_region(
                            state.as_mut(),
                            input_slice,
                            output_slice,
                            input_region,
                            output_region_for_node,
                        );
                    });
                } else {
                    op.dyn_process_region(
                        state.as_mut(),
                        input_slice,
                        output_slice,
                        input_region,
                        output_region_for_node,
                    );
                }
            }
        } else {
            let slot_count = op.input_slot_count();
            let buffers_ptr = std::ptr::from_ref(&pool.buffers);
            let coordinate_driven = op.coordinate_driven_source_spec().filter(|spec| {
                spec.source_slot < slot_count
                    && spec.dependency_slot < slot_count
                    && node.input_bufs()[spec.source_slot] == 0
                    && node.input_upstreams()[spec.source_slot].is_none()
            });

            for slot in 0..slot_count {
                if coordinate_driven.is_some_and(|spec| spec.source_slot == slot) {
                    continue;
                }
                // SAFETY: `buffers_ptr` points at `pool.buffers`, which is not mutated until after all input refs are prepared and we split out the output buffer below.
                let buffers = unsafe { &*buffers_ptr };
                pool.multi_input_refs[i][slot] =
                    prepare_multi_input_slot(pipeline, pool, node, i, buffers, slot)?;
            }

            if let Some(spec) = coordinate_driven {
                let dependency_region = pool.scratch_regions[i][spec.dependency_slot];
                let dependency_inputs =
                    input_ptrs_as_slices(&pool.multi_input_refs[i][..slot_count]);
                let dependency = dependency_inputs[spec.dependency_slot];
                let Some(source_plan) = op.source_read_plan_slot_with_materialized_dependency(
                    &output_region_for_node,
                    spec.source_slot,
                    spec.dependency_slot,
                    dependency_region,
                    dependency,
                ) else {
                    return Err(ViprsError::Scheduler(format!(
                        "coordinate-driven slot {} for node {i} did not produce a runtime source plan",
                        spec.source_slot
                    )));
                };

                let source_region = source_plan.produced_region();
                pool.scratch_regions[i][spec.source_slot] = source_region;
                let input_bands = pipeline.buffer_bands[0] as usize;
                let input_bps = sample_bytes(pipeline.buffer_formats[0]);
                let in_bytes = checked_region_byte_size(source_region, input_bands, input_bps)?;
                let scratch = ensure_scratch_buffer(
                    &mut pool.input_scratch_buffers[i][spec.source_slot],
                    in_bytes,
                );
                if !read_source_plan_with_cache(
                    pool.sequential_line_cache.as_mut(),
                    pipeline.source.as_ref(),
                    source_plan,
                    scratch,
                )? {
                    read_source_plan(
                        pipeline.source.as_ref(),
                        source_plan,
                        scratch,
                        input_bands * input_bps,
                    )?;
                }
                pool.multi_input_refs[i][spec.source_slot] =
                    InputSlicePtr(&raw const scratch[..in_bytes]);
            }

            let input_refs = input_ptrs_as_slices(&pool.multi_input_refs[i][..slot_count]);

            let state = pool.op_states[i].as_mut().ok_or_else(|| {
                ViprsError::from(SchedulerContractError::MissingTransformState { node: i })
            })?;
            if let (Some(cache), Some(cache_op_id)) = (&pipeline.tile_cache, node.cache_op_id()) {
                let tile = if let Some(profile) = profile.as_mut() {
                    profile.nodes[i].exec_count += 1;
                    materialize_cached_tile(out_bytes, |output_slice| {
                        timed_profile(&mut profile.nodes[i].process_ns, || {
                            op.dyn_process_region_multi(
                                state.as_mut(),
                                input_refs,
                                output_slice,
                                &pool.scratch_regions[i],
                                output_region_for_node,
                            );
                        });
                    })
                } else {
                    materialize_cached_tile(out_bytes, |output_slice| {
                        op.dyn_process_region_multi(
                            state.as_mut(),
                            input_refs,
                            output_slice,
                            &pool.scratch_regions[i],
                            output_region_for_node,
                        );
                    })
                };
                cache.insert(cache_op_id, output_region_for_node, Arc::clone(&tile))?;
                pool.cached_tiles[i] = Some(tile);
            } else {
                let (_, right) = pool.buffers.split_at_mut(node.output_buf());
                let output_slice = &mut right[0][..out_bytes];
                if let Some(profile) = profile.as_mut() {
                    profile.nodes[i].exec_count += 1;
                    timed_profile(&mut profile.nodes[i].process_ns, || {
                        op.dyn_process_region_multi(
                            state.as_mut(),
                            input_refs,
                            output_slice,
                            &pool.scratch_regions[i],
                            output_region_for_node,
                        );
                    });
                } else {
                    op.dyn_process_region_multi(
                        state.as_mut(),
                        input_refs,
                        output_slice,
                        &pool.scratch_regions[i],
                        output_region_for_node,
                    );
                }
            }
        }
    }

    Ok(())
}
