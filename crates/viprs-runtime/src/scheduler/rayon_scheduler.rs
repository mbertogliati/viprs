//! Rayon Scheduler scheduler adapter.
//!
//! These types execute compiled pipelines while managing concurrency, worker
//! scratch buffers, and tile traversal policy.

#![allow(clippy::struct_field_names)]
// REASON: the scheduler profile types intentionally mirror their serialized metric names.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;

use crate::{
    adapters::{
        instrumentation::viprs_span,
        lock_instrumentation::{self, TileLockScope},
        pipeline::{
            CompiledOp, CompiledPipeline, InputSlicePtr, LineCacheAccess, ThreadBufferPool,
        },
        sinks::memory::MemorySink,
    },
    domain::{
        cancel::CancellationToken,
        error::{SchedulerContractError, ViprsError},
        format::{BandFormat, BandFormatId},
        image::{DemandHint, Region, Tile},
        limits::{ExecutionPermit, ExecutionSemaphore},
        op::SourceReadPlan,
        reducer::TileReducer,
    },
    ports::{
        scheduler::{ReducingScheduler, TileScheduler},
        sink::{ConcurrentSink, ImageSink},
        source::DynImageSource,
    },
};

#[cfg(feature = "lock_instrumentation")]
use crate::adapters::lock_instrumentation::LockInstrumentationSnapshot;

/// Tile scheduler backed by a Rayon thread pool.
///
/// # Concurrency model
///
/// - Each rayon worker reuses a thread-local `ThreadBufferPool`, so tile execution keeps
///   mutable scratch state local to the worker instead of funneling through a global lock.
/// - `run` serializes only the final sink write through a `Mutex` when the sink cannot accept
///   concurrent writes.
/// - `run_concurrent` and the concurrent branch of `run_with_profile` bypass that `Mutex`
///   entirely by calling `ConcurrentSink::write_region_concurrent` on disjoint tile regions.
/// - `run_with_profile` aggregates per-tile timings with rayon `try_fold` / `try_reduce`,
///   so profiling adds no extra shared lock acquisition per output tile.
///
/// Peak memory is `O(threads × tile_size)` — tiles are written as they complete
/// rather than collected first. Zero allocator calls occur on the pixel path after
/// `ThreadBufferPool` initialization per thread.
pub struct RayonScheduler {
    pools: Mutex<Vec<(usize, Arc<rayon::ThreadPool>)>>,
    num_threads: usize,
    strip_height_tiles: Option<usize>,
    target_l2_bytes: usize,
    execution_limiter: Option<Arc<ExecutionSemaphore>>,
}

const DEFAULT_TARGET_L2_BYTES: usize = 2 * 1024 * 1024;
const MAX_STRIP_HEIGHT_TILES: usize = 32;
const L2_TILE_BUDGET_DIVISOR: usize = 1;
const STATIC_PARTITION_TASKS_PER_THREAD: usize = 2;
const MAX_SCOPED_STRIPS_PER_WORKER: usize = 8;
const SCOPED_STRIP_WORK_BYTES_PER_WORKER_L2_MULTIPLIER: usize = 4;
const THIN_STRIP_POOL_BUDGET_SOURCE_MULTIPLIER: usize = 4;
const THIN_STRIP_MIN_PARALLEL_WORKERS: usize = 2;
const THIN_STRIP_WORKER_CAP_SOURCE_BYTES_THRESHOLD: usize = 64 * 1024 * 1024;
const THIN_STRIP_MAX_LARGE_SOURCE_WORKERS: usize = 6;

fn scheduler_panic_to_error(context: &str, payload: Box<dyn Any + Send>) -> ViprsError {
    let payload = match payload.downcast::<ViprsError>() {
        Ok(err) => return *err,
        Err(payload) => payload,
    };
    let payload = match payload.downcast::<String>() {
        Ok(message) => return ViprsError::Scheduler(format!("{context}: {}", *message)),
        Err(payload) => payload,
    };
    if let Ok(message) = payload.downcast::<&'static str>() {
        return ViprsError::Scheduler(format!("{context}: {}", *message));
    }
    ViprsError::Scheduler(format!("{context}: non-string panic payload"))
}

fn catch_scheduler_panic<T>(
    context: &str,
    f: impl FnOnce() -> Result<T, ViprsError>,
) -> Result<T, ViprsError> {
    panic::catch_unwind(AssertUnwindSafe(f))
        .map_err(|payload| scheduler_panic_to_error(context, payload))?
}

fn ensure_not_cancelled(token: &CancellationToken) -> Result<(), ViprsError> {
    if token.is_cancelled() {
        return Err(ViprsError::Cancelled);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TileGeometry {
    tile_width: u32,
    tile_height: u32,
    cols: u32,
    rows: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TileStrip {
    regions: Vec<Region>,
}

#[derive(Clone, Copy)]
struct PipelineOutputTarget {
    output_buf: usize,
    output_bands: u32,
    output_format: BandFormatId,
}

#[derive(Clone, Debug, Default)]
/// Per-node runtime counters collected during profiled scheduler runs.
///
/// This helps benchmark and diagnostics code attribute execution time and cache
/// behavior to specific compiled nodes.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::scheduler::rayon_scheduler::NodeRunProfile;
///
/// let profile = NodeRunProfile::default();
/// assert_eq!(profile.exec_count, 0);
/// ```
pub struct NodeRunProfile {
    /// Number of times this node executed during the profiled run.
    pub exec_count: u64,
    /// Number of tile requests served from the node-local cache.
    pub cache_hits: u64,
    /// Total nanoseconds spent inside the node's processing path.
    pub process_ns: u128,
}

#[derive(Clone, Debug, Default)]
/// Aggregate profile for one pipeline execution.
///
/// This summarizes total runtime, source reads, sink writes, and per-node costs
/// so callers can compare execution plans and scheduler settings.
///
/// # Examples
///
/// ```rust
/// use viprs_runtime::scheduler::rayon_scheduler::PipelineRunProfile;
///
/// let profile = PipelineRunProfile::default();
/// assert_eq!(profile.tile_count, 0);
/// ```
pub struct PipelineRunProfile {
    /// End-to-end runtime of the profiled scheduler execution.
    pub total_ns: u128,
    /// Time spent executing tile work, excluding source and sink bookkeeping.
    pub tile_execute_ns: u128,
    /// Time spent handing completed regions to the sink.
    pub sink_write_ns: u128,
    /// Time spent reading source regions.
    pub source_read_ns: u128,
    /// Number of output tiles the scheduler processed.
    pub tile_count: u64,
    /// Number of source region reads issued during the run.
    pub source_read_count: u64,
    /// Per-node execution counters collected alongside the aggregate totals.
    pub nodes: Vec<NodeRunProfile>,
    #[cfg(feature = "lock_instrumentation")]
    /// Optional lock contention snapshot captured when lock instrumentation is enabled.
    pub lock_stats: LockInstrumentationSnapshot,
}

#[derive(Default)]
struct TileRunProfile {
    tile_execute_ns: u128,
    sink_write_ns: u128,
    source_read_ns: u128,
    source_read_count: u64,
    nodes: Vec<NodeRunProfile>,
}

mod execute;
mod line_cache;
mod planning;
mod runtime;

#[cfg(test)]
use execute::*;
pub(crate) use line_cache::SequentialLineCache;
#[cfg(test)]
use planning::*;

#[cfg(test)]
mod tests;
