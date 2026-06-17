#![allow(clippy::significant_drop_tightening)]
// REASON: runtime coordination objects intentionally live across scoped scheduling phases.

use super::{
    Arc, BandFormat, BandFormatId, CancellationToken, Cell, CompiledPipeline, ConcurrentSink,
    DEFAULT_TARGET_L2_BYTES, DemandHint, ExecutionPermit, ExecutionSemaphore, ImageSink,
    IndexedParallelIterator, Instant, IntoParallelRefIterator, MAX_SCOPED_STRIPS_PER_WORKER,
    MAX_STRIP_HEIGHT_TILES, MemorySink, Mutex, NodeRunProfile, ParallelIterator,
    PipelineOutputTarget, PipelineRunProfile, Range, RayonScheduler, ReducingScheduler, RefCell,
    Region, SCOPED_STRIP_WORK_BYTES_PER_WORKER_L2_MULTIPLIER, STATIC_PARTITION_TASKS_PER_THREAD,
    THIN_STRIP_MAX_LARGE_SOURCE_WORKERS, THIN_STRIP_MIN_PARALLEL_WORKERS,
    THIN_STRIP_POOL_BUDGET_SOURCE_MULTIPLIER, THIN_STRIP_WORKER_CAP_SOURCE_BYTES_THRESHOLD,
    ThreadBufferPool, Tile, TileGeometry, TileLockScope, TileReducer, TileRunProfile,
    TileScheduler, TileStrip, ViprsError, catch_scheduler_panic, ensure_not_cancelled,
    execute::{
        execute_tile, run_single_tile_non_empty, run_single_tile_non_empty_cancellable,
        run_single_tile_source_only, run_single_tile_source_only_cancellable,
        run_tiles_sequential_non_empty, run_tiles_sequential_non_empty_cancellable,
        run_tiles_sequential_source_only, run_tiles_sequential_source_only_cancellable,
    },
    lock_instrumentation,
    planning::{
        can_direct_write_all_regions, checked_region_byte_size, direct_source_region_for_output,
        pipeline_output_bytes, pipeline_output_slice, pipeline_output_target,
        pipeline_reads_source_directly, sample_bytes, source_only_output_bytes,
        source_only_output_slice, tile_geometry_for_l2_budget, timed_profile, with_worker_pool,
    },
    scheduler_panic_to_error, viprs_span,
};

#[cfg(feature = "lock_instrumentation")]
use super::lock_instrumentation::LockInstrumentationSnapshot;

impl TileRunProfile {
    fn new(node_count: usize) -> Self {
        Self {
            nodes: vec![NodeRunProfile::default(); node_count],
            ..Self::default()
        }
    }
}

thread_local! {
    pub(super) static EXECUTION_LIMIT_DEPTH: Cell<usize> = const { Cell::new(0) };
    pub(super) static WORKER_POOL: RefCell<Option<ThreadBufferPool>> = const { RefCell::new(None) };
}

struct ExecutionLimitDepthGuard {
    outermost: bool,
}

impl ExecutionLimitDepthGuard {
    fn enter() -> Self {
        let outermost = EXECUTION_LIMIT_DEPTH.with(|depth| {
            let current = depth.get();
            depth.set(current + 1);
            current == 0
        });
        Self { outermost }
    }

    const fn outermost(&self) -> bool {
        self.outermost
    }
}

impl Drop for ExecutionLimitDepthGuard {
    fn drop(&mut self) {
        EXECUTION_LIMIT_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

impl PipelineRunProfile {
    fn new(node_count: usize) -> Self {
        Self {
            nodes: vec![NodeRunProfile::default(); node_count],
            ..Self::default()
        }
    }

    fn merge_tile(&mut self, tile: &TileRunProfile) {
        self.tile_count += 1;
        self.tile_execute_ns += tile.tile_execute_ns;
        self.sink_write_ns += tile.sink_write_ns;
        self.source_read_ns += tile.source_read_ns;
        self.source_read_count += tile.source_read_count;
        for (aggregate, local) in self.nodes.iter_mut().zip(&tile.nodes) {
            aggregate.exec_count += local.exec_count;
            aggregate.cache_hits += local.cache_hits;
            aggregate.process_ns += local.process_ns;
        }
    }

    fn merge_profile(&mut self, other: &Self) {
        self.tile_count += other.tile_count;
        self.tile_execute_ns += other.tile_execute_ns;
        self.sink_write_ns += other.sink_write_ns;
        self.source_read_ns += other.source_read_ns;
        self.source_read_count += other.source_read_count;
        for (aggregate, local) in self.nodes.iter_mut().zip(&other.nodes) {
            aggregate.exec_count += local.exec_count;
            aggregate.cache_hits += local.cache_hits;
            aggregate.process_ns += local.process_ns;
        }
    }

    fn finish(&mut self, run_start: Instant) {
        self.total_ns = run_start.elapsed().as_nanos();
        #[cfg(feature = "lock_instrumentation")]
        {
            let mut lock_stats = lock_instrumentation::snapshot();
            lock_stats.tile_count = self.tile_count;
            self.lock_stats = lock_stats;
        }
    }
}

impl RayonScheduler {
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::new;
    /// ```
    pub const fn new(num_threads: usize) -> Result<Self, ViprsError> {
        Ok(Self {
            pools: Mutex::new(Vec::new()),
            num_threads,
            strip_height_tiles: None,
            target_l2_bytes: DEFAULT_TARGET_L2_BYTES,
            execution_limiter: None,
        })
    }

    /// `default_threads` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::default_threads;
    /// ```
    #[must_use]
    pub fn default_threads() -> usize {
        std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
    }

    #[must_use]
    /// `with_strip_height_tiles` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::with_strip_height_tiles;
    /// ```
    pub fn with_strip_height_tiles(mut self, strip_height_tiles: usize) -> Self {
        self.strip_height_tiles = Some(strip_height_tiles.max(1));
        self
    }

    #[must_use]
    /// `with_l2_cache_bytes` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::with_l2_cache_bytes;
    /// ```
    pub fn with_l2_cache_bytes(mut self, target_l2_bytes: usize) -> Self {
        self.target_l2_bytes = target_l2_bytes.max(1);
        self
    }

    #[must_use]
    /// `with_max_concurrent_pipelines` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::with_max_concurrent_pipelines;
    /// ```
    pub fn with_max_concurrent_pipelines(mut self, max_concurrent: usize) -> Self {
        self.execution_limiter = Some(Arc::new(ExecutionSemaphore::new(max_concurrent.max(1))));
        self
    }

    #[must_use]
    /// `with_execution_limiter` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::with_execution_limiter;
    /// ```
    pub(crate) fn with_execution_limiter(
        mut self,
        execution_limiter: Arc<ExecutionSemaphore>,
    ) -> Self {
        self.execution_limiter = Some(execution_limiter);
        self
    }

    pub(super) fn with_execution_permit<T>(
        &self,
        f: impl FnOnce() -> Result<T, ViprsError>,
    ) -> Result<T, ViprsError> {
        let depth_guard = ExecutionLimitDepthGuard::enter();
        let _permit: Option<ExecutionPermit<'_>> = if depth_guard.outermost() {
            self.execution_limiter
                .as_deref()
                .map(ExecutionSemaphore::acquire)
        } else {
            None
        };
        f()
    }

    pub(super) fn thread_pool(&self) -> Result<Arc<rayon::ThreadPool>, ViprsError> {
        self.thread_pool_capped(self.num_threads)
    }

    pub(super) fn thread_pool_capped(
        &self,
        limit: usize,
    ) -> Result<Arc<rayon::ThreadPool>, ViprsError> {
        let thread_count = self.num_threads.min(limit.max(1));
        let mut guard = self
            .pools
            .lock()
            .map_err(|_| ViprsError::Scheduler("scheduler pool mutex poisoned".into()))?;
        if let Some((_, pool)) = guard.iter().find(|(threads, _)| *threads == thread_count) {
            return Ok(Arc::clone(pool));
        }

        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .map_err(|e| ViprsError::Scheduler(e.to_string()))?,
        );
        guard.push((thread_count, Arc::clone(&pool)));
        Ok(pool)
    }

    pub(super) fn effective_strip_height_tiles(
        &self,
        pipeline: &CompiledPipeline,
    ) -> Result<usize, ViprsError> {
        if let Some(strip_height_tiles) = self.strip_height_tiles {
            return Ok(strip_height_tiles.max(1));
        }

        // Thin-strip pipelines already use full-width tiles; grouping multiple tile rows into a
        // strip reduces parallelism and showed up as the dominant gauss_blur bottleneck in profile
        // runs. Keep them at one tile row per strip so worker distribution stays fine-grained.
        if pipeline.demand_hint == DemandHint::ThinStrip {
            return Ok(1);
        }

        let geometry = tile_geometry_for_l2_budget(pipeline, self.target_l2_bytes)?;
        let Some(last) = pipeline.nodes.last() else {
            return Ok(1);
        };
        let tile_bytes = checked_region_byte_size(
            Region::new(0, 0, geometry.tile_width, geometry.tile_height),
            last.op.bands() as usize,
            sample_bytes(last.op.output_format()),
        )?
        .max(1);

        Ok(self
            .target_l2_bytes
            .saturating_div(tile_bytes)
            .min(geometry.rows.div_ceil(self.num_threads as u32) as usize)
            .clamp(1, MAX_STRIP_HEIGHT_TILES))
    }

    pub(super) fn tile_geometry_for_execution(
        &self,
        pipeline: &CompiledPipeline,
    ) -> Result<TileGeometry, ViprsError> {
        let mut geometry = tile_geometry_for_l2_budget(pipeline, self.target_l2_bytes)?;

        if matches!(pipeline.demand_hint, DemandHint::FatStrip | DemandHint::Any)
            && self.num_threads > 1
            && geometry.rows < (self.num_threads as u32).max(16)
            && geometry.tile_height > 1
        {
            let desired_rows = (self.num_threads as u32)
                .max(16)
                .min(pipeline.height)
                .max(1);
            let parallel_tile_height = pipeline.height.div_ceil(desired_rows).max(1);
            if parallel_tile_height < geometry.tile_height {
                geometry.tile_height = parallel_tile_height;
                geometry.rows = pipeline.height.div_ceil(parallel_tile_height);
            }
        }

        Ok(geometry)
    }

    pub(super) fn generate_tiles_for_execution(
        &self,
        pipeline: &CompiledPipeline,
    ) -> Result<Vec<Region>, ViprsError> {
        let geometry = self.tile_geometry_for_execution(pipeline)?;

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

    pub(super) fn generate_tile_strips_for_execution(
        &self,
        pipeline: &CompiledPipeline,
        strip_height_tiles: usize,
    ) -> Result<Vec<TileStrip>, ViprsError> {
        let geometry = self.tile_geometry_for_execution(pipeline)?;
        let tiles = self.generate_tiles_for_execution(pipeline)?;
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

    pub(super) fn static_work_ranges(
        &self,
        demand_hint: DemandHint,
        work_items: usize,
    ) -> Option<Vec<Range<usize>>> {
        Self::static_work_ranges_for_workers(demand_hint, self.num_threads, work_items)
    }

    pub(super) fn static_work_ranges_for_workers(
        demand_hint: DemandHint,
        worker_count: usize,
        work_items: usize,
    ) -> Option<Vec<Range<usize>>> {
        if worker_count <= 1 || work_items <= 1 {
            return None;
        }

        let keep_thin_strip_static = demand_hint == DemandHint::ThinStrip;
        if !keep_thin_strip_static
            && work_items > worker_count.saturating_mul(STATIC_PARTITION_TASKS_PER_THREAD)
        {
            return None;
        }

        let chunk_len = work_items.div_ceil(worker_count).max(1);
        let mut ranges = Vec::with_capacity(work_items.div_ceil(chunk_len));
        let mut start = 0;
        while start < work_items {
            let end = (start + chunk_len).min(work_items);
            ranges.push(start..end);
            start = end;
        }
        Some(ranges)
    }

    #[cfg(feature = "lock_instrumentation")]
    /// `lock_instrumentation_snapshot` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::lock_instrumentation_snapshot;
    /// ```
    pub fn lock_instrumentation_snapshot() -> LockInstrumentationSnapshot {
        lock_instrumentation::snapshot()
    }

    /// `run_with_profile` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let _ = viprs::adapters::scheduler::rayon_scheduler::run_with_profile;
    /// ```
    pub fn run_with_profile(
        &self,
        pipeline: &CompiledPipeline,
        sink: &mut dyn ImageSink,
    ) -> Result<PipelineRunProfile, ViprsError> {
        self.with_execution_permit(|| catch_scheduler_panic("rayon scheduler profile panic", || {
            let _run_guard = lock_instrumentation::prepare_run();
            let run_start = Instant::now();
            if let Some(concurrent_sink) = sink.as_concurrent_sink() {
                return self.run_concurrent_with_profile(pipeline, concurrent_sink, run_start);
            }

            let mut profile = PipelineRunProfile::new(pipeline.nodes.len());
            let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;

            if pipeline.sequential {
                let strips =
                    self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
                let mut pool = ThreadBufferPool::new(pipeline);
                let output_target = pipeline_output_target(pipeline);

                for strip in strips {
                    for &region in &strip.regions {
                        let _tile_scope = TileLockScope::new();
                        let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                        let execute_start = Instant::now();
                        execute_tile(pipeline, region, &mut pool, Some(&mut tile_profile))?;
                        tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                        let out_bytes = output_target.map_or_else(
                            || source_only_output_bytes(pipeline, region),
                            |target| pipeline_output_bytes(target, region),
                        )?;
                        let output_slice = output_target.map_or_else(
                            || source_only_output_slice(&pool, out_bytes),
                            |target| pipeline_output_slice(target, &pool, out_bytes),
                        );
                        timed_profile(&mut tile_profile.sink_write_ns, || {
                            sink.write_region(region, output_slice)
                        })?;
                        profile.merge_tile(&tile_profile);
                    }
                }

                profile.finish(run_start);
                return Ok(profile);
            }

            let strips = self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
            let output_target = pipeline_output_target(pipeline);
            let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
            let thread_pool = self.thread_pool_capped(worker_count)?;

            thread_local! {
                static PROFILE_POOL: RefCell<Option<ThreadBufferPool>> = const { RefCell::new(None) };
            }

            let sink_mutex = Mutex::new(sink);
            let aggregate = Mutex::new(PipelineRunProfile::new(pipeline.nodes.len()));

            thread_pool.install(|| {
                strips.par_iter().with_min_len(1).try_for_each(|strip| {
                    PROFILE_POOL.with(|cell| -> Result<(), ViprsError> {
                        let mut borrow = cell.borrow_mut();
                        let pool = borrow.get_or_insert_with(|| ThreadBufferPool::new(pipeline));

                        for &region in &strip.regions {
                            let _tile_scope = TileLockScope::new();
                            let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());

                            let execute_start = Instant::now();
                            execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                            tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                            let out_bytes = output_target.map_or_else(
                                || source_only_output_bytes(pipeline, region),
                                |target| pipeline_output_bytes(target, region),
                            )?;
                            let output_slice = output_target.map_or_else(
                                || source_only_output_slice(pool, out_bytes),
                                |target| pipeline_output_slice(target, pool, out_bytes),
                            );

                            timed_profile(&mut tile_profile.sink_write_ns, || {
                                lock_instrumentation::record_lock_acquisition();
                                sink_mutex
                                    .lock()
                                    .map_err(|_| {
                                        ViprsError::Scheduler("sink mutex poisoned".into())
                                    })?
                                    .write_region(region, output_slice)
                            })?;

                            aggregate
                                .lock()
                                .map_err(|_| {
                                    ViprsError::Scheduler("profile mutex poisoned".into())
                                })?
                                .merge_tile(&tile_profile);
                        }
                        Ok(())
                    })
                })
            })?;

            profile = aggregate
                .into_inner()
                .map_err(|_| ViprsError::Scheduler("profile mutex poisoned".into()))?;
            profile.finish(run_start);
            Ok(profile)
        }))
    }

    pub(super) fn run_concurrent_with_profile(
        &self,
        pipeline: &CompiledPipeline,
        sink: &dyn ConcurrentSink,
        run_start: Instant,
    ) -> Result<PipelineRunProfile, ViprsError> {
        let mut profile = PipelineRunProfile::new(pipeline.nodes.len());
        let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;
        let strips = self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
        if let Some(memory_sink) = sink.as_any().downcast_ref::<MemorySink>() {
            let all_regions: Vec<Region> = strips
                .iter()
                .flat_map(|strip| strip.regions.iter().copied())
                .collect();
            if pipeline_reads_source_directly(pipeline)
                && can_direct_write_all_regions(memory_sink, &all_regions)
            {
                for strip in &strips {
                    for &region in &strip.regions {
                        let Some(source_region) = direct_source_region_for_output(pipeline, region)
                        else {
                            return Err(ViprsError::Scheduler(
                                "direct source region unexpectedly unavailable".into(),
                            ));
                        };
                        let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                        tile_profile.source_read_count += 1;
                        timed_profile(&mut tile_profile.source_read_ns, || {
                            // SAFETY: the direct-source fast path processes full-width strips
                            // sequentially, so each region has exclusive access to its sink bytes.
                            let write_result = unsafe {
                                memory_sink
                                    .with_full_width_region_mut_concurrent(region, |output| {
                                        pipeline.source.read_region(source_region, output)
                                    })?
                            };
                            write_result.ok_or_else(|| {
                                ViprsError::Scheduler(
                                    "direct source path lost contiguous sink region".into(),
                                )
                            })?
                        })?;
                        tile_profile.tile_execute_ns = tile_profile.source_read_ns;
                        profile.merge_tile(&tile_profile);
                    }
                }
                profile.finish(run_start);
                return Ok(profile);
            }
        }

        if pipeline.sequential {
            let tiles = self.generate_tiles_for_execution(pipeline)?;
            let mut pool = ThreadBufferPool::new(pipeline);
            let output_target = pipeline_output_target(pipeline);

            for region in tiles {
                let _tile_scope = TileLockScope::new();
                let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                let execute_start = Instant::now();
                execute_tile(pipeline, region, &mut pool, Some(&mut tile_profile))?;
                tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                let out_bytes = output_target.map_or_else(
                    || source_only_output_bytes(pipeline, region),
                    |target| pipeline_output_bytes(target, region),
                )?;
                let output_slice = output_target.map_or_else(
                    || source_only_output_slice(&pool, out_bytes),
                    |target| pipeline_output_slice(target, &pool, out_bytes),
                );
                timed_profile(&mut tile_profile.sink_write_ns, || {
                    sink.write_region_concurrent(region, output_slice)
                })?;
                profile.merge_tile(&tile_profile);
            }

            profile.finish(run_start);
            return Ok(profile);
        }

        if self.should_use_scoped_strips(pipeline, &strips)? {
            profile = if let Some(output_target) = pipeline_output_target(pipeline) {
                self.profile_scoped_strips(pipeline, &strips, &|pool, strip, local_profile| {
                    for &region in &strip.regions {
                        let _tile_scope = TileLockScope::new();
                        let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                        let execute_start = Instant::now();
                        execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                        tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                        let out_bytes = pipeline_output_bytes(output_target, region)?;
                        let output_slice = pipeline_output_slice(output_target, pool, out_bytes);
                        timed_profile(&mut tile_profile.sink_write_ns, || {
                            sink.write_region_concurrent(region, output_slice)
                        })?;
                        local_profile.merge_tile(&tile_profile);
                    }
                    Ok(())
                })?
            } else {
                self.profile_scoped_strips(pipeline, &strips, &|pool, strip, local_profile| {
                    for &region in &strip.regions {
                        let _tile_scope = TileLockScope::new();
                        let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                        let execute_start = Instant::now();
                        execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                        tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                        let out_bytes = source_only_output_bytes(pipeline, region)?;
                        let output_slice = source_only_output_slice(pool, out_bytes);
                        timed_profile(&mut tile_profile.sink_write_ns, || {
                            sink.write_region_concurrent(region, output_slice)
                        })?;
                        local_profile.merge_tile(&tile_profile);
                    }
                    Ok(())
                })?
            };
            profile.finish(run_start);
            return Ok(profile);
        }

        let tiles = self.generate_tiles_for_execution(pipeline)?;
        if let Some(output_target) = pipeline_output_target(pipeline) {
            profile = self.parallel_profile_tiles(
                pipeline,
                output_target,
                &tiles,
                |region, output_slice, tile_profile| {
                    timed_profile(&mut tile_profile.sink_write_ns, || {
                        sink.write_region_concurrent(region, output_slice)
                    })
                },
            )?;
        } else if let Some(ranges) = self.static_work_ranges(pipeline.demand_hint, tiles.len()) {
            profile = self.thread_pool()?.install(|| {
                ranges
                    .par_iter()
                    .try_fold(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut local_profile, range| {
                            with_worker_pool(pipeline, |pool| {
                                for &region in &tiles[range.clone()] {
                                    let _tile_scope = TileLockScope::new();
                                    let mut tile_profile =
                                        TileRunProfile::new(pipeline.nodes.len());
                                    let execute_start = Instant::now();
                                    execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                                    tile_profile.tile_execute_ns +=
                                        execute_start.elapsed().as_nanos();

                                    let out_bytes = source_only_output_bytes(pipeline, region)?;
                                    let output_slice = source_only_output_slice(pool, out_bytes);

                                    timed_profile(&mut tile_profile.sink_write_ns, || {
                                        sink.write_region_concurrent(region, output_slice)
                                    })?;
                                    local_profile.merge_tile(&tile_profile);
                                }
                                Ok(local_profile)
                            })
                        },
                    )
                    .try_reduce(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut left, right| {
                            left.merge_profile(&right);
                            Ok(left)
                        },
                    )
            })?;
        } else {
            profile = self.thread_pool()?.install(|| {
                tiles
                    .par_iter()
                    .with_min_len(8)
                    .try_fold(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut local_profile, &region| {
                            let _tile_scope = TileLockScope::new();
                            with_worker_pool(pipeline, |pool| {
                                let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                                let execute_start = Instant::now();
                                execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                                tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                                let out_bytes = source_only_output_bytes(pipeline, region)?;
                                let output_slice = source_only_output_slice(pool, out_bytes);

                                timed_profile(&mut tile_profile.sink_write_ns, || {
                                    sink.write_region_concurrent(region, output_slice)
                                })?;
                                local_profile.merge_tile(&tile_profile);
                                Ok(local_profile)
                            })
                        },
                    )
                    .try_reduce(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut left, right| {
                            left.merge_profile(&right);
                            Ok(left)
                        },
                    )
            })?;
        }
        profile.finish(run_start);
        Ok(profile)
    }

    pub(super) fn parallel_profile_tiles(
        &self,
        pipeline: &CompiledPipeline,
        output_target: PipelineOutputTarget,
        tiles: &[Region],
        write_tile: impl Fn(Region, &[u8], &mut TileRunProfile) -> Result<(), ViprsError> + Sync,
    ) -> Result<PipelineRunProfile, ViprsError> {
        if let Some(ranges) = self.static_work_ranges(pipeline.demand_hint, tiles.len()) {
            self.thread_pool()?.install(|| {
                ranges
                    .par_iter()
                    .try_fold(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut local_profile, range| {
                            with_worker_pool(pipeline, |pool| {
                                for &region in &tiles[range.clone()] {
                                    let _tile_scope = TileLockScope::new();
                                    let mut tile_profile =
                                        TileRunProfile::new(pipeline.nodes.len());
                                    let execute_start = Instant::now();
                                    execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                                    tile_profile.tile_execute_ns +=
                                        execute_start.elapsed().as_nanos();

                                    let out_bytes = pipeline_output_bytes(output_target, region)?;
                                    let output_slice =
                                        pipeline_output_slice(output_target, pool, out_bytes);

                                    write_tile(region, output_slice, &mut tile_profile)?;
                                    local_profile.merge_tile(&tile_profile);
                                }
                                Ok(local_profile)
                            })
                        },
                    )
                    .try_reduce(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut left, right| {
                            left.merge_profile(&right);
                            Ok(left)
                        },
                    )
            })
        } else {
            self.thread_pool()?.install(|| {
                tiles
                    .par_iter()
                    .with_min_len(8)
                    .try_fold(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut local_profile, &region| {
                            let _tile_scope = TileLockScope::new();
                            with_worker_pool(pipeline, |pool| {
                                let mut tile_profile = TileRunProfile::new(pipeline.nodes.len());
                                let execute_start = Instant::now();
                                execute_tile(pipeline, region, pool, Some(&mut tile_profile))?;
                                tile_profile.tile_execute_ns += execute_start.elapsed().as_nanos();

                                let out_bytes = pipeline_output_bytes(output_target, region)?;
                                let output_slice =
                                    pipeline_output_slice(output_target, pool, out_bytes);

                                write_tile(region, output_slice, &mut tile_profile)?;
                                local_profile.merge_tile(&tile_profile);
                                Ok(local_profile)
                            })
                        },
                    )
                    .try_reduce(
                        || PipelineRunProfile::new(pipeline.nodes.len()),
                        |mut left, right| {
                            left.merge_profile(&right);
                            Ok(left)
                        },
                    )
            })
        }
    }

    #[inline]
    pub(super) fn should_use_scoped_strips(
        &self,
        pipeline: &CompiledPipeline,
        strips: &[TileStrip],
    ) -> Result<bool, ViprsError> {
        let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
        self.should_use_scoped_strips_for_workload(
            pipeline.demand_hint,
            strips.len(),
            worker_count,
            Region::new(0, 0, pipeline.width, pipeline.height),
            pipeline.output_bands,
            pipeline.output_format,
        )
    }

    #[inline]
    pub(super) fn should_use_scoped_strips_for_workload(
        &self,
        demand_hint: DemandHint,
        strip_count: usize,
        worker_count: usize,
        output_region: Region,
        output_bands: u32,
        output_format: BandFormatId,
    ) -> Result<bool, ViprsError> {
        if demand_hint != DemandHint::ThinStrip || worker_count <= 1 || strip_count <= 1 {
            return Ok(false);
        }

        let strips_per_worker = strip_count.div_ceil(worker_count);
        if strips_per_worker > MAX_SCOPED_STRIPS_PER_WORKER {
            return Ok(false);
        }

        let total_work_bytes = checked_region_byte_size(
            output_region,
            output_bands as usize,
            sample_bytes(output_format),
        )?
        .max(1);
        let bytes_per_worker = total_work_bytes.div_ceil(worker_count);
        let max_bytes_per_worker = self
            .target_l2_bytes
            .saturating_mul(SCOPED_STRIP_WORK_BYTES_PER_WORKER_L2_MULTIPLIER);

        Ok(bytes_per_worker <= max_bytes_per_worker)
    }

    pub(super) fn effective_strip_worker_count(
        &self,
        pipeline: &CompiledPipeline,
        strip_count: usize,
    ) -> Result<usize, ViprsError> {
        let pool_bytes = pipeline.buffer_sizes.iter().copied().sum::<usize>().max(1);
        let source_bytes = checked_region_byte_size(
            Region::new(0, 0, pipeline.source.width(), pipeline.source.height()),
            pipeline.source.bands() as usize,
            sample_bytes(pipeline.source.format()),
        )?
        .max(1);

        Ok(self.effective_strip_worker_count_for_workload(
            pipeline.demand_hint,
            strip_count,
            pool_bytes,
            source_bytes,
            pipeline.width,
        ))
    }

    pub(super) fn effective_strip_worker_count_for_workload(
        &self,
        demand_hint: DemandHint,
        strip_count: usize,
        pool_bytes: usize,
        source_bytes: usize,
        output_width: u32,
    ) -> usize {
        let worker_count = self.num_threads.min(strip_count).max(1);
        let strip_memory_dominates = demand_hint == DemandHint::ThinStrip || output_width <= 512;
        if !strip_memory_dominates
            || worker_count <= 1
            || source_bytes < THIN_STRIP_WORKER_CAP_SOURCE_BYTES_THRESHOLD
        {
            return worker_count;
        }

        let memory_limited_workers = source_bytes
            .max(1)
            .saturating_mul(THIN_STRIP_POOL_BUDGET_SOURCE_MULTIPLIER)
            .div_ceil(pool_bytes.max(1));
        let min_parallel_workers = THIN_STRIP_MIN_PARALLEL_WORKERS.min(worker_count);
        let large_source_worker_cap = THIN_STRIP_MAX_LARGE_SOURCE_WORKERS.min(worker_count);

        memory_limited_workers
            .max(min_parallel_workers)
            .min(large_source_worker_cap)
            .min(worker_count)
    }

    pub(super) fn run_scoped_strips(
        &self,
        pipeline: &CompiledPipeline,
        strips: &[TileStrip],
        run_strip: &(impl Fn(&mut ThreadBufferPool, &TileStrip) -> Result<(), ViprsError> + Sync),
    ) -> Result<(), ViprsError> {
        let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
        if worker_count <= 1 {
            let mut pool = ThreadBufferPool::new(pipeline);
            for strip in strips {
                run_strip(&mut pool, strip)?;
            }
            return Ok(());
        }

        let chunk_len = strips.len().div_ceil(worker_count);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for chunk in strips.chunks(chunk_len) {
                handles.push(scope.spawn(move || -> Result<(), ViprsError> {
                    let mut pool = ThreadBufferPool::new(pipeline);
                    for strip in chunk {
                        run_strip(&mut pool, strip)?;
                    }
                    Ok(())
                }));
            }

            for handle in handles {
                let join_result = handle.join().map_err(|payload| {
                    scheduler_panic_to_error("scoped strip worker panic", payload)
                })?;
                join_result?;
            }
            Ok(())
        })
    }

    pub(super) fn profile_scoped_strips(
        &self,
        pipeline: &CompiledPipeline,
        strips: &[TileStrip],
        run_strip: &(
             impl Fn(
            &mut ThreadBufferPool,
            &TileStrip,
            &mut PipelineRunProfile,
        ) -> Result<(), ViprsError>
             + Sync
         ),
    ) -> Result<PipelineRunProfile, ViprsError> {
        let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
        if worker_count <= 1 {
            let mut pool = ThreadBufferPool::new(pipeline);
            let mut profile = PipelineRunProfile::new(pipeline.nodes.len());
            for strip in strips {
                run_strip(&mut pool, strip, &mut profile)?;
            }
            return Ok(profile);
        }

        let chunk_len = strips.len().div_ceil(worker_count);
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for chunk in strips.chunks(chunk_len) {
                handles.push(
                    scope.spawn(move || -> Result<PipelineRunProfile, ViprsError> {
                        let mut pool = ThreadBufferPool::new(pipeline);
                        let mut profile = PipelineRunProfile::new(pipeline.nodes.len());
                        for strip in chunk {
                            run_strip(&mut pool, strip, &mut profile)?;
                        }
                        Ok(profile)
                    }),
                );
            }

            let mut aggregate = PipelineRunProfile::new(pipeline.nodes.len());
            for handle in handles {
                let join_result = handle.join().map_err(|payload| {
                    scheduler_panic_to_error("scoped strip profile worker panic", payload)
                })?;
                aggregate.merge_profile(&join_result?);
            }
            Ok(aggregate)
        })
    }
}

impl TileScheduler<CompiledPipeline> for RayonScheduler {
    fn run(&self, pipeline: &CompiledPipeline, sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
        self.with_execution_permit(|| {
            catch_scheduler_panic("rayon scheduler panic", || {
                #[cfg(feature = "tracing")]
                let tile_count = self.generate_tiles_for_execution(pipeline)?.len();
                viprs_span!(
                    tracing::Level::INFO,
                    "viprs.pipeline.run",
                    tiles = tile_count,
                    threads = self.num_threads
                );
                let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;
                let _run_guard = lock_instrumentation::prepare_run();
                let output_target = pipeline_output_target(pipeline);
                if let Some(concurrent_sink) = sink.as_concurrent_sink() {
                    return self.run_concurrent(pipeline, concurrent_sink);
                }

                if pipeline.sequential {
                    return if let Some(output_target) = output_target {
                        run_tiles_sequential_non_empty(
                            pipeline,
                            output_target,
                            strip_height_tiles,
                            self.target_l2_bytes,
                            |region, output_slice| sink.write_region(region, output_slice),
                        )
                    } else {
                        run_tiles_sequential_source_only(
                            pipeline,
                            strip_height_tiles,
                            self.target_l2_bytes,
                            |region, output_slice| sink.write_region(region, output_slice),
                        )
                    };
                }
                let strips =
                    self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
                let tiles = self.generate_tiles_for_execution(pipeline)?;
                if let [region] = tiles.as_slice() {
                    return if let Some(output_target) = output_target {
                        run_single_tile_non_empty(
                            pipeline,
                            output_target,
                            *region,
                            |tile_region, output_slice| {
                                sink.write_region(tile_region, output_slice)
                            },
                        )
                    } else {
                        run_single_tile_source_only(
                            pipeline,
                            *region,
                            |tile_region, output_slice| {
                                sink.write_region(tile_region, output_slice)
                            },
                        )
                    };
                }

                let sink_mutex = Mutex::new(sink);
                let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
                let thread_pool = self.thread_pool_capped(worker_count)?;

                output_target.map_or_else(
                    || {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes =
                                                        source_only_output_bytes(pipeline, region)?;
                                                    let output_slice =
                                                        source_only_output_slice(pool, out_bytes);

                                                    lock_instrumentation::record_lock_acquisition();
                                                    sink_mutex
                                                        .lock()
                                                        .map_err(|_| {
                                                            ViprsError::Scheduler(
                                                                "sink mutex poisoned".into(),
                                                            )
                                                        })?
                                                        .write_region(region, output_slice)?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                with_worker_pool(pipeline, |pool| -> Result<(), ViprsError> {
                                    for strip in &strips[range.clone()] {
                                        for &region in &strip.regions {
                                            let _tile_scope = TileLockScope::new();
                                            execute_tile(pipeline, region, pool, None)?;

                                            let out_bytes =
                                                source_only_output_bytes(pipeline, region)?;
                                            let output_slice =
                                                source_only_output_slice(pool, out_bytes);

                                            lock_instrumentation::record_lock_acquisition();
                                            sink_mutex
                                                .lock()
                                                .map_err(|_| {
                                                    ViprsError::Scheduler(
                                                        "sink mutex poisoned".into(),
                                                    )
                                                })?
                                                .write_region(region, output_slice)?;
                                        }
                                    }
                                    Ok(())
                                })
                            })
                                })
                            },
                        )
                    },
                    |output_target| {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes = pipeline_output_bytes(
                                                        output_target,
                                                        region,
                                                    )?;
                                                    let output_slice = pipeline_output_slice(
                                                        output_target,
                                                        pool,
                                                        out_bytes,
                                                    );

                                                    lock_instrumentation::record_lock_acquisition();
                                                    sink_mutex
                                                        .lock()
                                                        .map_err(|_| {
                                                            ViprsError::Scheduler(
                                                                "sink mutex poisoned".into(),
                                                            )
                                                        })?
                                                        .write_region(region, output_slice)?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                with_worker_pool(pipeline, |pool| -> Result<(), ViprsError> {
                                    for strip in &strips[range.clone()] {
                                        for &region in &strip.regions {
                                            let _tile_scope = TileLockScope::new();
                                            execute_tile(pipeline, region, pool, None)?;

                                            let out_bytes =
                                                pipeline_output_bytes(output_target, region)?;
                                            let output_slice = pipeline_output_slice(
                                                output_target,
                                                pool,
                                                out_bytes,
                                            );

                                            lock_instrumentation::record_lock_acquisition();
                                            sink_mutex
                                                .lock()
                                                .map_err(|_| {
                                                    ViprsError::Scheduler(
                                                        "sink mutex poisoned".into(),
                                                    )
                                                })?
                                                .write_region(region, output_slice)?;
                                        }
                                    }
                                    Ok(())
                                })
                            })
                                })
                            },
                        )
                    },
                )
            })
        })
    }

    fn run_cancellable(
        &self,
        pipeline: &CompiledPipeline,
        sink: &mut dyn ImageSink,
        token: &CancellationToken,
    ) -> Result<(), ViprsError> {
        self.with_execution_permit(|| {
            catch_scheduler_panic("rayon scheduler cancellable panic", || {
                ensure_not_cancelled(token)?;
                let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;
                let _run_guard = lock_instrumentation::prepare_run();
                let output_target = pipeline_output_target(pipeline);
                if let Some(concurrent_sink) = sink.as_concurrent_sink() {
                    return self.run_concurrent_cancellable(pipeline, concurrent_sink, token);
                }

                if pipeline.sequential {
                    return if let Some(output_target) = output_target {
                        run_tiles_sequential_non_empty_cancellable(
                            pipeline,
                            output_target,
                            strip_height_tiles,
                            self.target_l2_bytes,
                            token,
                            |region, output_slice| sink.write_region(region, output_slice),
                        )
                    } else {
                        run_tiles_sequential_source_only_cancellable(
                            pipeline,
                            strip_height_tiles,
                            self.target_l2_bytes,
                            token,
                            |region, output_slice| sink.write_region(region, output_slice),
                        )
                    };
                }

                let strips =
                    self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
                let tiles = self.generate_tiles_for_execution(pipeline)?;
                if let [region] = tiles.as_slice() {
                    return if let Some(output_target) = output_target {
                        run_single_tile_non_empty_cancellable(
                            pipeline,
                            output_target,
                            *region,
                            token,
                            |tile_region, output_slice| {
                                sink.write_region(tile_region, output_slice)
                            },
                        )
                    } else {
                        run_single_tile_source_only_cancellable(
                            pipeline,
                            *region,
                            token,
                            |tile_region, output_slice| {
                                sink.write_region(tile_region, output_slice)
                            },
                        )
                    };
                }

                let sink_mutex = Mutex::new(sink);
                let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
                let thread_pool = self.thread_pool_capped(worker_count)?;

                output_target.map_or_else(
                    || {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    ensure_not_cancelled(token)?;
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes =
                                                        source_only_output_bytes(pipeline, region)?;
                                                    let output_slice =
                                                        source_only_output_slice(pool, out_bytes);

                                                    lock_instrumentation::record_lock_acquisition();
                                                    sink_mutex
                                                        .lock()
                                                        .map_err(|_| {
                                                            ViprsError::Scheduler(
                                                                "sink mutex poisoned".into(),
                                                            )
                                                        })?
                                                        .write_region(region, output_slice)?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                with_worker_pool(pipeline, |pool| -> Result<(), ViprsError> {
                                    for strip in &strips[range.clone()] {
                                        for &region in &strip.regions {
                                            ensure_not_cancelled(token)?;
                                            let _tile_scope = TileLockScope::new();
                                            execute_tile(pipeline, region, pool, None)?;

                                            let out_bytes =
                                                source_only_output_bytes(pipeline, region)?;
                                            let output_slice =
                                                source_only_output_slice(pool, out_bytes);

                                            lock_instrumentation::record_lock_acquisition();
                                            sink_mutex
                                                .lock()
                                                .map_err(|_| {
                                                    ViprsError::Scheduler(
                                                        "sink mutex poisoned".into(),
                                                    )
                                                })?
                                                .write_region(region, output_slice)?;
                                        }
                                    }
                                    Ok(())
                                })
                            })
                                })
                            },
                        )
                    },
                    |output_target| {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    ensure_not_cancelled(token)?;
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes = pipeline_output_bytes(
                                                        output_target,
                                                        region,
                                                    )?;
                                                    let output_slice = pipeline_output_slice(
                                                        output_target,
                                                        pool,
                                                        out_bytes,
                                                    );

                                                    lock_instrumentation::record_lock_acquisition();
                                                    sink_mutex
                                                        .lock()
                                                        .map_err(|_| {
                                                            ViprsError::Scheduler(
                                                                "sink mutex poisoned".into(),
                                                            )
                                                        })?
                                                        .write_region(region, output_slice)?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                with_worker_pool(pipeline, |pool| -> Result<(), ViprsError> {
                                    for strip in &strips[range.clone()] {
                                        for &region in &strip.regions {
                                            ensure_not_cancelled(token)?;
                                            let _tile_scope = TileLockScope::new();
                                            execute_tile(pipeline, region, pool, None)?;

                                            let out_bytes =
                                                pipeline_output_bytes(output_target, region)?;
                                            let output_slice = pipeline_output_slice(
                                                output_target,
                                                pool,
                                                out_bytes,
                                            );

                                            lock_instrumentation::record_lock_acquisition();
                                            sink_mutex
                                                .lock()
                                                .map_err(|_| {
                                                    ViprsError::Scheduler(
                                                        "sink mutex poisoned".into(),
                                                    )
                                                })?
                                                .write_region(region, output_slice)?;
                                        }
                                    }
                                    Ok(())
                                })
                            })
                                })
                            },
                        )
                    },
                )
            })
        })
    }

    /// Lock-free concurrent path: each rayon thread writes its tile directly via
    /// `ConcurrentSink::write_region_concurrent`, with no `Mutex` in the hot path.
    ///
    /// # Safety invariant
    ///
    /// The call to `write_region_concurrent` from multiple rayon threads simultaneously
    /// is sound because:
    ///
    /// 1. `generate_tiles` produces disjoint regions — no two tiles share any pixel.
    /// 2. `ConcurrentSink` requires `Send + Sync` and its contract specifies that
    ///    concurrent calls with non-overlapping regions must be safe.
    /// 3. `MemorySink` upholds this contract via `UnsafeCell<Vec<u8>>` and
    ///    its `Sync` implementation and the safety comment on `MemorySink`.
    ///
    /// Any `ConcurrentSink` implementation that does not uphold the disjoint-write
    /// contract violates the trait's documented invariant, not this scheduler.
    fn run_concurrent(
        &self,
        pipeline: &CompiledPipeline,
        sink: &dyn ConcurrentSink,
    ) -> Result<(), ViprsError> {
        self.with_execution_permit(|| {
            catch_scheduler_panic("rayon scheduler concurrent panic", || {
                #[cfg(feature = "tracing")]
                let tile_count = self.generate_tiles_for_execution(pipeline)?.len();
                viprs_span!(
                    tracing::Level::INFO,
                    "viprs.pipeline.run",
                    tiles = tile_count,
                    threads = self.num_threads
                );
                let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;
                let _run_guard = lock_instrumentation::prepare_run();
                let output_target = pipeline_output_target(pipeline);
                let strips =
                    self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
                let tiles = self.generate_tiles_for_execution(pipeline)?;
                if let Some(memory_sink) = sink.as_any().downcast_ref::<MemorySink>() {
                    let all_regions: Vec<Region> = strips
                        .iter()
                        .flat_map(|strip| strip.regions.iter().copied())
                        .collect();
                    if pipeline_reads_source_directly(pipeline)
                        && can_direct_write_all_regions(memory_sink, &all_regions)
                    {
                        for strip in &strips {
                            for &region in &strip.regions {
                                let Some(source_region) =
                                    direct_source_region_for_output(pipeline, region)
                                else {
                                    return Err(ViprsError::Scheduler(
                                        "direct source region unexpectedly unavailable".into(),
                                    ));
                                };
                                // SAFETY: the direct-source fast path processes full-width strips
                                // sequentially, so each region has exclusive access to its sink bytes.
                                let write_result = unsafe {
                                    memory_sink
                                        .with_full_width_region_mut_concurrent(region, |output| {
                                            pipeline.source.read_region(source_region, output)
                                        })?
                                };
                                write_result.ok_or_else(|| {
                                    ViprsError::Scheduler(
                                        "direct source path lost contiguous sink region".into(),
                                    )
                                })??;
                            }
                        }
                        return Ok(());
                    }
                }
                if pipeline.sequential {
                    return output_target.map_or_else(
                        || {
                            run_tiles_sequential_source_only(
                                pipeline,
                                strip_height_tiles,
                                self.target_l2_bytes,
                                |region, output_slice| {
                                    sink.write_region_concurrent(region, output_slice)
                                },
                            )
                        },
                        |output_target| {
                            run_tiles_sequential_non_empty(
                                pipeline,
                                output_target,
                                strip_height_tiles,
                                self.target_l2_bytes,
                                |region, output_slice| {
                                    sink.write_region_concurrent(region, output_slice)
                                },
                            )
                        },
                    );
                }

                if let [region] = tiles.as_slice() {
                    return output_target.map_or_else(
                        || {
                            run_single_tile_source_only(
                                pipeline,
                                *region,
                                |tile_region, output_slice| {
                                    sink.write_region_concurrent(tile_region, output_slice)
                                },
                            )
                        },
                        |output_target| {
                            run_single_tile_non_empty(
                                pipeline,
                                output_target,
                                *region,
                                |tile_region, output_slice| {
                                    sink.write_region_concurrent(tile_region, output_slice)
                                },
                            )
                        },
                    );
                }

                if self.should_use_scoped_strips(pipeline, &strips)? {
                    return output_target.map_or_else(
                        || {
                            self.run_scoped_strips(pipeline, &strips, &|pool, strip| {
                                for &region in &strip.regions {
                                    let _tile_scope = TileLockScope::new();
                                    execute_tile(pipeline, region, pool, None)?;

                                    let out_bytes = source_only_output_bytes(pipeline, region)?;
                                    let output_slice = source_only_output_slice(pool, out_bytes);

                                    sink.write_region_concurrent(region, output_slice)?;
                                }
                                Ok(())
                            })
                        },
                        |output_target| {
                            self.run_scoped_strips(pipeline, &strips, &|pool, strip| {
                                for &region in &strip.regions {
                                    let _tile_scope = TileLockScope::new();
                                    execute_tile(pipeline, region, pool, None)?;

                                    let out_bytes = pipeline_output_bytes(output_target, region)?;
                                    let output_slice =
                                        pipeline_output_slice(output_target, pool, out_bytes);

                                    sink.write_region_concurrent(region, output_slice)?;
                                }
                                Ok(())
                            })
                        },
                    );
                }

                let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
                let thread_pool = self.thread_pool_capped(worker_count)?;
                output_target.map_or_else(
                    || {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes =
                                                        source_only_output_bytes(pipeline, region)?;
                                                    let output_slice =
                                                        source_only_output_slice(pool, out_bytes);

                                                    sink.write_region_concurrent(
                                                        region,
                                                        output_slice,
                                                    )?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for strip in &strips[range.clone()] {
                                                    for &region in &strip.regions {
                                                        let _tile_scope = TileLockScope::new();
                                                        execute_tile(pipeline, region, pool, None)?;

                                                        let out_bytes = source_only_output_bytes(
                                                            pipeline, region,
                                                        )?;
                                                        let output_slice = source_only_output_slice(
                                                            pool, out_bytes,
                                                        );

                                                        sink.write_region_concurrent(
                                                            region,
                                                            output_slice,
                                                        )?;
                                                    }
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                        )
                    },
                    |output_target| {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes = pipeline_output_bytes(
                                                        output_target,
                                                        region,
                                                    )?;
                                                    let output_slice = pipeline_output_slice(
                                                        output_target,
                                                        pool,
                                                        out_bytes,
                                                    );

                                                    sink.write_region_concurrent(
                                                        region,
                                                        output_slice,
                                                    )?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for strip in &strips[range.clone()] {
                                                    for &region in &strip.regions {
                                                        let _tile_scope = TileLockScope::new();
                                                        execute_tile(pipeline, region, pool, None)?;

                                                        let out_bytes = pipeline_output_bytes(
                                                            output_target,
                                                            region,
                                                        )?;
                                                        let output_slice = pipeline_output_slice(
                                                            output_target,
                                                            pool,
                                                            out_bytes,
                                                        );

                                                        sink.write_region_concurrent(
                                                            region,
                                                            output_slice,
                                                        )?;
                                                    }
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                        )
                    },
                )
            })
        })
    }

    fn run_concurrent_cancellable(
        &self,
        pipeline: &CompiledPipeline,
        sink: &dyn ConcurrentSink,
        token: &CancellationToken,
    ) -> Result<(), ViprsError> {
        self.with_execution_permit(|| {
            catch_scheduler_panic("rayon scheduler concurrent cancellable panic", || {
                ensure_not_cancelled(token)?;
                let strip_height_tiles = self.effective_strip_height_tiles(pipeline)?;
                let _run_guard = lock_instrumentation::prepare_run();
                let output_target = pipeline_output_target(pipeline);
                let strips =
                    self.generate_tile_strips_for_execution(pipeline, strip_height_tiles)?;
                if let Some(memory_sink) = sink.as_any().downcast_ref::<MemorySink>() {
                    let all_regions: Vec<Region> = strips
                        .iter()
                        .flat_map(|strip| strip.regions.iter().copied())
                        .collect();
                    if pipeline_reads_source_directly(pipeline)
                        && can_direct_write_all_regions(memory_sink, &all_regions)
                    {
                        for strip in &strips {
                            for &region in &strip.regions {
                                ensure_not_cancelled(token)?;
                                let Some(source_region) =
                                    direct_source_region_for_output(pipeline, region)
                                else {
                                    return Err(ViprsError::Scheduler(
                                        "direct source region unexpectedly unavailable".into(),
                                    ));
                                };
                                // SAFETY: the direct-source fast path processes full-width strips
                                // sequentially, so each region has exclusive access to its sink bytes.
                                let write_result = unsafe {
                                    memory_sink
                                        .with_full_width_region_mut_concurrent(region, |output| {
                                            pipeline.source.read_region(source_region, output)
                                        })?
                                };
                                write_result.ok_or_else(|| {
                                    ViprsError::Scheduler(
                                        "direct source path lost contiguous sink region".into(),
                                    )
                                })??;
                            }
                        }
                        return Ok(());
                    }
                }
                if pipeline.sequential {
                    return output_target.map_or_else(
                        || {
                            run_tiles_sequential_source_only_cancellable(
                                pipeline,
                                strip_height_tiles,
                                self.target_l2_bytes,
                                token,
                                |region, output_slice| {
                                    sink.write_region_concurrent(region, output_slice)
                                },
                            )
                        },
                        |output_target| {
                            run_tiles_sequential_non_empty_cancellable(
                                pipeline,
                                output_target,
                                strip_height_tiles,
                                self.target_l2_bytes,
                                token,
                                |region, output_slice| {
                                    sink.write_region_concurrent(region, output_slice)
                                },
                            )
                        },
                    );
                }

                let tiles = self.generate_tiles_for_execution(pipeline)?;
                if let [region] = tiles.as_slice() {
                    return output_target.map_or_else(
                        || {
                            run_single_tile_source_only_cancellable(
                                pipeline,
                                *region,
                                token,
                                |tile_region, output_slice| {
                                    sink.write_region_concurrent(tile_region, output_slice)
                                },
                            )
                        },
                        |output_target| {
                            run_single_tile_non_empty_cancellable(
                                pipeline,
                                output_target,
                                *region,
                                token,
                                |tile_region, output_slice| {
                                    sink.write_region_concurrent(tile_region, output_slice)
                                },
                            )
                        },
                    );
                }

                if self.should_use_scoped_strips(pipeline, &strips)? {
                    return output_target.map_or_else(
                        || {
                            self.run_scoped_strips(pipeline, &strips, &|pool, strip| {
                                for &region in &strip.regions {
                                    ensure_not_cancelled(token)?;
                                    let _tile_scope = TileLockScope::new();
                                    execute_tile(pipeline, region, pool, None)?;

                                    let out_bytes = source_only_output_bytes(pipeline, region)?;
                                    let output_slice = source_only_output_slice(pool, out_bytes);

                                    sink.write_region_concurrent(region, output_slice)?;
                                }
                                Ok(())
                            })
                        },
                        |output_target| {
                            self.run_scoped_strips(pipeline, &strips, &|pool, strip| {
                                for &region in &strip.regions {
                                    ensure_not_cancelled(token)?;
                                    let _tile_scope = TileLockScope::new();
                                    execute_tile(pipeline, region, pool, None)?;

                                    let out_bytes = pipeline_output_bytes(output_target, region)?;
                                    let output_slice =
                                        pipeline_output_slice(output_target, pool, out_bytes);

                                    sink.write_region_concurrent(region, output_slice)?;
                                }
                                Ok(())
                            })
                        },
                    );
                }

                let worker_count = self.effective_strip_worker_count(pipeline, strips.len())?;
                let thread_pool = self.thread_pool_capped(worker_count)?;
                output_target.map_or_else(
                    || {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    ensure_not_cancelled(token)?;
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes =
                                                        source_only_output_bytes(pipeline, region)?;
                                                    let output_slice =
                                                        source_only_output_slice(pool, out_bytes);

                                                    sink.write_region_concurrent(
                                                        region,
                                                        output_slice,
                                                    )?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for strip in &strips[range.clone()] {
                                                    for &region in &strip.regions {
                                                        ensure_not_cancelled(token)?;
                                                        let _tile_scope = TileLockScope::new();
                                                        execute_tile(pipeline, region, pool, None)?;

                                                        let out_bytes = source_only_output_bytes(
                                                            pipeline, region,
                                                        )?;
                                                        let output_slice = source_only_output_slice(
                                                            pool, out_bytes,
                                                        );

                                                        sink.write_region_concurrent(
                                                            region,
                                                            output_slice,
                                                        )?;
                                                    }
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                        )
                    },
                    |output_target| {
                        Self::static_work_ranges_for_workers(
                            pipeline.demand_hint,
                            worker_count,
                            strips.len(),
                        )
                        .map_or_else(
                            || {
                                thread_pool.install(|| {
                                    strips.par_iter().with_min_len(1).try_for_each(|strip| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for &region in &strip.regions {
                                                    ensure_not_cancelled(token)?;
                                                    let _tile_scope = TileLockScope::new();
                                                    execute_tile(pipeline, region, pool, None)?;

                                                    let out_bytes = pipeline_output_bytes(
                                                        output_target,
                                                        region,
                                                    )?;
                                                    let output_slice = pipeline_output_slice(
                                                        output_target,
                                                        pool,
                                                        out_bytes,
                                                    );

                                                    sink.write_region_concurrent(
                                                        region,
                                                        output_slice,
                                                    )?;
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                            |ranges| {
                                thread_pool.install(|| {
                                    ranges.par_iter().try_for_each(|range| {
                                        with_worker_pool(
                                            pipeline,
                                            |pool| -> Result<(), ViprsError> {
                                                for strip in &strips[range.clone()] {
                                                    for &region in &strip.regions {
                                                        ensure_not_cancelled(token)?;
                                                        let _tile_scope = TileLockScope::new();
                                                        execute_tile(pipeline, region, pool, None)?;

                                                        let out_bytes = pipeline_output_bytes(
                                                            output_target,
                                                            region,
                                                        )?;
                                                        let output_slice = pipeline_output_slice(
                                                            output_target,
                                                            pool,
                                                            out_bytes,
                                                        );

                                                        sink.write_region_concurrent(
                                                            region,
                                                            output_slice,
                                                        )?;
                                                    }
                                                }
                                                Ok(())
                                            },
                                        )
                                    })
                                })
                            },
                        )
                    },
                )
            })
        })
    }
}

impl ReducingScheduler<CompiledPipeline> for RayonScheduler {
    /// Run the pipeline in a single pass, writing tiles to `sink` and accumulating
    /// per-tile outputs through `reducer` simultaneously.
    ///
    /// Each tile is written to `sink` (same contract as `run_concurrent`) and then
    /// reduced on the same producing rayon thread via `reducer.accumulate_into`.
    /// The rayon fold state carries `(ThreadBufferPool, R::Scratch, Option<R::Partial>)`:
    /// `R::Scratch` is initialized once per fold partition with `Default::default()`
    /// and reused across all tiles processed by that worker.
    ///
    /// The `&[u8]` output buffer from `execute_tile` is reinterpreted as `&[F::Sample]`
    /// via `bytemuck::cast_slice`, which is valid because `F::Sample: bytemuck::Pod`
    /// (guaranteed by `BandFormat`'s associated type bound). No copy is required.
    ///
    /// `F` must match the output format of the last pipeline node. A runtime check
    /// guards against format mismatches by comparing `BandFormat::ID` against
    /// `pipeline.output_format`.
    fn run_with_reducer<F, R>(
        &self,
        pipeline: &CompiledPipeline,
        sink: &dyn ConcurrentSink,
        reducer: &R,
    ) -> Result<R::Output, ViprsError>
    where
        F: BandFormat,
        R: TileReducer<F>,
    {
        self.with_execution_permit(|| catch_scheduler_panic("rayon scheduler reducer panic", || {
            let _run_guard = lock_instrumentation::prepare_run();
            let output_target = pipeline_output_target(pipeline);
            if F::ID != pipeline.output_format {
                return Err(ViprsError::Scheduler(format!(
                    "run_with_reducer: format mismatch — pipeline output is {:?}, caller supplied {:?}",
                    pipeline.output_format,
                    F::ID,
                )));
            }

            if pipeline.sequential {
                let strips = self.generate_tile_strips_for_execution(
                    pipeline,
                    self.effective_strip_height_tiles(pipeline)?,
                )?;
                let mut pool = ThreadBufferPool::new(pipeline);
                let mut scratch = R::Scratch::default();
                let mut partial = None::<R::Partial>;

                for strip in strips {
                    for &region in &strip.regions {
                        let _tile_scope = TileLockScope::new();
                        execute_tile(pipeline, region, &mut pool, None)?;

                        let out_bytes = output_target.map_or_else(
                            || source_only_output_bytes(pipeline, region),
                            |target| pipeline_output_bytes(target, region),
                        )?;
                        let output_slice = output_target.map_or_else(
                            || source_only_output_slice(&pool, out_bytes),
                            |target| pipeline_output_slice(target, &pool, out_bytes),
                        );

                        sink.write_region_concurrent(region, output_slice)?;

                        let typed_slice: &[F::Sample] = bytemuck::cast_slice(output_slice);
                        let tile = Tile::<F>::new(region, pipeline.output_bands, typed_slice);
                        reducer.accumulate_into(&tile, &region, &mut scratch, &mut partial);
                    }
                }

                return Ok(reducer.finalize(partial.ok_or_else(|| {
                    ViprsError::Scheduler(
                        "run_with_reducer: no tiles produced — image has zero area".into(),
                    )
                })?));
            }

            let strips = self.generate_tile_strips_for_execution(
                pipeline,
                self.effective_strip_height_tiles(pipeline)?,
            )?;

            let (_, _, combined) = self.thread_pool()?.install(|| {
                strips
                    .par_iter()
                    .try_fold(
                        || {
                            (
                                ThreadBufferPool::new(pipeline),
                                R::Scratch::default(),
                                None::<R::Partial>,
                            )
                        },
                        |(mut pool, mut scratch, mut partial), strip| -> Result<_, ViprsError> {
                            for &region in &strip.regions {
                                let _tile_scope = TileLockScope::new();
                                execute_tile(pipeline, region, &mut pool, None)?;

                                let out_bytes = output_target.map_or_else(
                                    || source_only_output_bytes(pipeline, region),
                                    |target| pipeline_output_bytes(target, region),
                                )?;
                                let output_slice = output_target.map_or_else(
                                    || source_only_output_slice(&pool, out_bytes),
                                    |target| pipeline_output_slice(target, &pool, out_bytes),
                                );

                                sink.write_region_concurrent(region, output_slice)?;

                                // SAFETY: `bytemuck::cast_slice` is sound because
                                // `F::Sample: Pod` and `output_slice` was sized for exactly
                                // `pixel_count * bands * size_of::<F::Sample>()`.
                                let typed_slice: &[F::Sample] = bytemuck::cast_slice(output_slice);
                                let tile =
                                    Tile::<F>::new(region, pipeline.output_bands, typed_slice);

                                reducer.accumulate_into(&tile, &region, &mut scratch, &mut partial);
                            }

                            Ok((pool, scratch, partial))
                        },
                    )
                    .try_reduce(
                        || {
                            (
                                ThreadBufferPool::new(pipeline),
                                R::Scratch::default(),
                                None::<R::Partial>,
                            )
                        },
                        |(left_pool, left_scratch, left_partial), (_, _, right_partial)| {
                            let combined = match (left_partial, right_partial) {
                                (Some(left), Some(right)) => Some(reducer.combine(left, right)),
                                (Some(left), None) => Some(left),
                                (None, Some(right)) => Some(right),
                                (None, None) => None,
                            };
                            Ok((left_pool, left_scratch, combined))
                        },
                    )
            })?;

            Ok(reducer.finalize(combined.ok_or_else(|| {
                ViprsError::Scheduler(
                    "run_with_reducer: no tiles produced — image has zero area".into(),
                )
            })?))
        }))
    }
}
