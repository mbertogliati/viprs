//! `TileScheduler` port — the capability interface for driving a compiled pipeline.
//!
//! `TileScheduler<P>` is generic over the pipeline type `P`. This keeps the port
//! free of any dependency on `adapters/` (CLAUDE.md rule 3). Concrete schedulers
//! (e.g. `RayonScheduler`) implement `TileScheduler<CompiledPipeline>` in `adapters/`.
//!
//! `ReducingScheduler<P>` extends `TileScheduler<P>` with single-pass in-flight
//! reduction via a dedicated opt-in trait.

use std::sync::Mutex;

use viprs_core::{
    cancel::CancellationToken, error::ViprsError, format::BandFormat, image::Region,
    reducer::TileReducer,
};
use crate::sink::{ConcurrentSink, ImageSink};

/// Drives a compiled pipeline by requesting and delivering tiles.
///
/// The scheduler is responsible for dividing the image into tiles,
/// executing operations, and writing results to the sink.
///
/// The trait is generic over `P` — the pipeline type — so that the port does not
/// import `CompiledPipeline` from `adapters/`. Concrete implementations fix `P`
/// to their specific pipeline representation (e.g. `CompiledPipeline`).
///
/// # Examples
///
/// ```rust
/// use viprs::domain::{cancel::CancellationToken, error::ViprsError, image::Region};
/// use viprs::ports::{
///     scheduler::TileScheduler,
///     sink::ImageSink,
/// };
///
/// struct NoopScheduler;
/// struct FakePipeline;
/// struct Sink;
///
/// impl ImageSink for Sink {
///     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
///         Ok(())
///     }
///
///     fn finish(self: Box<Self>) -> Result<(), ViprsError> {
///         Ok(())
///     }
/// }
///
/// impl TileScheduler<FakePipeline> for NoopScheduler {
///     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
///         Ok(())
///     }
/// }
///
/// let scheduler = NoopScheduler;
/// let pipeline = FakePipeline;
/// let token = CancellationToken::new();
/// let mut sink = Sink;
/// scheduler.run_cancellable(&pipeline, &mut sink, &token)?;
/// # Ok::<(), ViprsError>(())
/// ```
pub trait TileScheduler<P>: Send + Sync {
    /// Run the pipeline writing results into `sink`.
    ///
    /// This method solves full pipeline execution for schedulers that know how
    /// to request tiles and deliver them to a destination sink.
    ///
    /// # Errors
    ///
    /// Returns `ViprsError` if tile execution or sink writes fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, image::Region};
    /// # use viprs::ports::{scheduler::TileScheduler, sink::ImageSink};
    /// # struct NoopScheduler;
    /// # struct FakePipeline;
    /// # struct Sink;
    /// # impl ImageSink for Sink {
    /// #     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// # impl TileScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// let scheduler = NoopScheduler;
    /// let pipeline = FakePipeline;
    /// let mut sink = Sink;
    /// scheduler.run(&pipeline, &mut sink)?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn run(&self, pipeline: &P, sink: &mut dyn ImageSink) -> Result<(), ViprsError>;

    /// Run the pipeline writing results into `sink`, aborting cooperatively when `token`
    /// is cancelled.
    ///
    /// The default implementation preserves backward compatibility by checking the token once
    /// before dispatch and then delegating to `run`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{cancel::CancellationToken, error::ViprsError, image::Region};
    /// # use viprs::ports::{scheduler::TileScheduler, sink::ImageSink};
    /// # struct NoopScheduler;
    /// # struct FakePipeline;
    /// # struct Sink;
    /// # impl ImageSink for Sink {
    /// #     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// # impl TileScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// let scheduler = NoopScheduler;
    /// let pipeline = FakePipeline;
    /// let token = CancellationToken::new();
    /// let mut sink = Sink;
    /// scheduler.run_cancellable(&pipeline, &mut sink, &token)?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn run_cancellable(
        &self,
        pipeline: &P,
        sink: &mut dyn ImageSink,
        token: &CancellationToken,
    ) -> Result<(), ViprsError> {
        if token.is_cancelled() {
            return Err(ViprsError::Cancelled);
        }
        self.run(pipeline, sink)
    }

    /// Run the pipeline writing results into a `ConcurrentSink`.
    ///
    /// Schedulers that can parallelize writes (e.g. `RayonScheduler`) override this
    /// to eliminate the per-tile `Mutex` that `run` requires. Schedulers that cannot
    /// parallelize writes fall back to this default, which wraps the sink in a
    /// `Mutex`-backed `ImageSink` adapter and delegates to `run`, guaranteeing
    /// correctness without requiring every scheduler to implement the concurrent path.
    ///
    /// The caller must ensure that the regions produced by the pipeline's tile
    /// generation are disjoint — this invariant is upheld by `generate_tiles`.
    ///
    /// `dyn ConcurrentSink` is acceptable here (CLAUDE.md rule 1): the sink is a
    /// runtime-extensible registry point set at pipeline construction time, not a
    /// per-pixel dispatch.
    ///
    /// This keeps the concurrent-write path opt-in without complicating the base trait.
    ///
    /// # Errors
    ///
    /// Returns `ViprsError` if tile execution or sink writes fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{error::ViprsError, image::Region};
    /// # use viprs::ports::{scheduler::TileScheduler, sink::{ConcurrentSink, ImageSink}};
    /// # struct NoopScheduler;
    /// # struct FakePipeline;
    /// # impl TileScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// # struct SharedSink;
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let scheduler = NoopScheduler;
    /// let pipeline = FakePipeline;
    /// let sink = SharedSink;
    /// scheduler.run_concurrent(&pipeline, &sink)?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn run_concurrent(&self, pipeline: &P, sink: &dyn ConcurrentSink) -> Result<(), ViprsError> {
        // Default fallback: wrap `ConcurrentSink` in a mutex-backed `ImageSink` shim
        // and delegate to `run`. This is correct but serializes writes — schedulers
        // that can parallelize should override this method.
        let mut shim = ConcurrentSinkShim {
            inner: Mutex::new(sink),
        };
        self.run(pipeline, &mut shim)
    }

    /// Run the pipeline writing results into a `ConcurrentSink`, aborting cooperatively when
    /// `token` is cancelled.
    ///
    /// The default implementation preserves backward compatibility by checking the token once
    /// before dispatch and then delegating to `run_concurrent`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs::domain::{cancel::CancellationToken, error::ViprsError, image::Region};
    /// # use viprs::ports::{scheduler::TileScheduler, sink::{ConcurrentSink, ImageSink}};
    /// # struct NoopScheduler;
    /// # struct FakePipeline;
    /// # impl TileScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// # }
    /// # struct SharedSink;
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let scheduler = NoopScheduler;
    /// let pipeline = FakePipeline;
    /// let token = CancellationToken::new();
    /// let sink = SharedSink;
    /// scheduler.run_concurrent_cancellable(&pipeline, &sink, &token)?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn run_concurrent_cancellable(
        &self,
        pipeline: &P,
        sink: &dyn ConcurrentSink,
        token: &CancellationToken,
    ) -> Result<(), ViprsError> {
        if token.is_cancelled() {
            return Err(ViprsError::Cancelled);
        }
        self.run_concurrent(pipeline, sink)
    }
}

/// Scheduler that accumulates a `TileReducer` result while generating tiles.
///
/// Unlike the two-pass approach (where the reducer runs over a
/// `MemorySink` buffer after the pipeline completes), `ReducingScheduler` folds
/// each tile's output into a per-thread `R::Partial` *as it is produced*, then
/// merges all per-thread partials into the final `R::Output` before returning.
///
/// This eliminates the second pipeline pass for the "normalize" pattern
/// (find stats → apply scale), at the cost of additional scheduler complexity.
///
/// # Design choice (Option B)
///
/// This is a *separate* super-trait of `TileScheduler<P>` rather than an extra
/// method on `TileScheduler<P>`. Adding `run_with_reducer<F, R>` directly to
/// `TileScheduler<P>` would force every scheduler implementor to supply it, and
/// would make the trait's generic bounds combinatorially complex (P × F × R).
/// A separate opt-in trait lets schedulers that cannot reduce (e.g. a future
/// `SingleThreadScheduler`) remain simple.
///
/// # Object safety
///
/// `ReducingScheduler<P>` is **not** object-safe because `run_with_reducer` is
/// generic over `F` and `R`. This is intentional: the reduction path is a
/// compile-time-monomorphized hot path. `dyn ReducingScheduler` is never needed
/// because the scheduler is always a concrete type known at pipeline-construction
/// time (see CLAUDE.md rule 1).
///
/// # Per-worker partial state
///
/// `run_with_reducer` keeps `(ThreadBufferPool, R::Scratch, Option<R::Partial>)`
/// inside the rayon fold state. `R::Scratch` is initialized with
/// `Default::default()` once per worker fold partition and reused across many
/// tiles through `TileReducer::accumulate_into`, eliminating per-tile allocations
/// for reducers that provide scratch-aware implementations. The scheduler merges
/// fold-local partials via `reducer.combine` in a final reduce step.
///
/// # Reducer target: output of last node
///
/// The reducer receives the *output* of the last pipeline node (not the source
/// pixels). This matches the common use-case — statistics over the *result* of
/// the pipeline. Access to additional synchronized inputs is modeled at the
/// reducer layer via `domain::reducer::BiSourceReducer`: the reducer owns any
/// prevalidated side input it needs, and the scheduler continues to stream only
/// the primary output tile. Access to the source buffer at index 0
/// (`pool.buffers\[0\]`) would
/// require a separate `ReducingScheduler` variant and is deferred to a future ADR.
///
/// # Examples
///
/// ```ignore
/// use viprs::domain::{
///     error::ViprsError,
///     format::U8,
///     image::Region,
///     reducer::TileReducer,
/// };
/// use viprs::ports::{
///     scheduler::{ReducingScheduler, TileScheduler},
///     sink::{ConcurrentSink, ImageSink},
/// };
///
/// struct NoopScheduler;
/// struct FakePipeline;
/// struct SharedSink;
/// struct CountingReducer;
///
/// impl ImageSink for SharedSink {
///     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
///         Ok(())
///     }
///     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
/// }
///
/// impl ConcurrentSink for SharedSink {
///     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
///         Ok(())
///     }
///     fn as_any(&self) -> &dyn std::any::Any { self }
///     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
/// }
///
/// impl TileScheduler<FakePipeline> for NoopScheduler {
///     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> {
///         Ok(())
///     }
/// }
///
/// impl TileReducer<U8> for CountingReducer {
///     type Partial = usize;
///     type Output = usize;
///     type Scratch = ();
///
///     fn reduce_tile(&self, tile: &[u8], _bands: u32) -> Result<Self::Partial, ViprsError> {
///         Ok(tile.len())
///     }
///
///     fn accumulate_into(
///         &self,
///         partial: &mut Option<Self::Partial>,
///         tile: &[u8],
///         bands: u32,
///         _scratch: &mut Self::Scratch,
///     ) -> Result<(), ViprsError> {
///         let next = self.reduce_tile(tile, bands)?;
///         *partial = Some(partial.unwrap_or(0) + next);
///         Ok(())
///     }
///
///     fn combine(&self, left: Self::Partial, right: Self::Partial) -> Result<Self::Partial, ViprsError> {
///         Ok(left + right)
///     }
///
///     fn finalize(&self, partial: Option<Self::Partial>) -> Result<Self::Output, ViprsError> {
///         Ok(partial.unwrap_or(0))
///     }
/// }
///
/// impl ReducingScheduler<FakePipeline> for NoopScheduler {
///     fn run_with_reducer<F, R>(
///         &self,
///         _pipeline: &FakePipeline,
///         _sink: &dyn ConcurrentSink,
///         _reducer: &R,
///     ) -> Result<R::Output, ViprsError>
///     where
///         F: viprs::domain::format::BandFormat,
///         R: TileReducer<F>,
///     {
///         Err(ViprsError::Scheduler("example scheduler".into()))
///     }
/// }
/// ```
pub trait ReducingScheduler<P>: TileScheduler<P> {
    /// Run the pipeline, writing tiles to `sink` and simultaneously accumulating
    /// per-tile outputs through `reducer`.
    ///
    /// The pipeline runs exactly once. Each output tile is:
    ///   1. Written to `sink` (same semantics as `run_concurrent`).
    ///   2. Passed to `reducer.reduce_tile` on the producing thread.
    ///
    /// Per-thread `R::Partial` values are combined with `reducer.combine` after
    /// all tiles complete. `reducer.finalize` is called exactly once.
    ///
    /// `F` must match the output format of the last pipeline node. Passing the
    /// wrong `F` is a logic error; the implementation may panic or return
    /// `ViprsError::Scheduler` if it can detect the mismatch at runtime.
    ///
    /// # Errors
    ///
    /// Returns `ViprsError` if tile execution, sink writes, or reducer logic fails.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use viprs::domain::{error::ViprsError, format::{BandFormat, U8}, image::Region, reducer::TileReducer};
    /// # use viprs::ports::{scheduler::{ReducingScheduler, TileScheduler}, sink::{ConcurrentSink, ImageSink}};
    /// # struct NoopScheduler;
    /// # struct FakePipeline;
    /// # struct SharedSink;
    /// # struct CountingReducer;
    /// # impl ImageSink for SharedSink {
    /// #     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> { Ok(()) }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> { Ok(()) }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// # impl TileScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run(&self, _pipeline: &FakePipeline, _sink: &mut dyn ImageSink) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// # impl TileReducer<U8> for CountingReducer {
    /// #     type Partial = usize;
    /// #     type Output = usize;
    /// #     type Scratch = ();
    /// #     fn reduce_tile(&self, tile: &[u8], _bands: u32) -> Result<Self::Partial, ViprsError> { Ok(tile.len()) }
    /// #     fn accumulate_into(&self, partial: &mut Option<Self::Partial>, tile: &[u8], bands: u32, _scratch: &mut Self::Scratch) -> Result<(), ViprsError> {
    /// #         let next = self.reduce_tile(tile, bands)?;
    /// #         *partial = Some(partial.unwrap_or(0) + next);
    /// #         Ok(())
    /// #     }
    /// #     fn combine(&self, left: Self::Partial, right: Self::Partial) -> Result<Self::Partial, ViprsError> { Ok(left + right) }
    /// #     fn finalize(&self, partial: Option<Self::Partial>) -> Result<Self::Output, ViprsError> { Ok(partial.unwrap_or(0)) }
    /// # }
    /// # impl ReducingScheduler<FakePipeline> for NoopScheduler {
    /// #     fn run_with_reducer<F, R>(&self, _pipeline: &FakePipeline, _sink: &dyn ConcurrentSink, _reducer: &R) -> Result<R::Output, ViprsError>
    /// #     where
    /// #         F: BandFormat,
    /// #         R: TileReducer<F>,
    /// #     {
    /// #         Err(ViprsError::Scheduler("example scheduler".into()))
    /// #     }
    /// # }
    /// let scheduler = NoopScheduler;
    /// let pipeline = FakePipeline;
    /// let sink = SharedSink;
    /// let reducer = CountingReducer;
    /// let _ = scheduler.run_with_reducer::<U8, _>(&pipeline, &sink, &reducer);
    /// ```
    fn run_with_reducer<F, R>(
        &self,
        pipeline: &P,
        sink: &dyn ConcurrentSink,
        reducer: &R,
    ) -> Result<R::Output, ViprsError>
    where
        F: BandFormat,
        R: TileReducer<F>;
}

/// Adapter that wraps a `&dyn ConcurrentSink` behind `ImageSink` for the fallback path.
///
/// Schedulers that override `run_concurrent` do not use this type at all.
/// It exists only so the default `run_concurrent` implementation can call `run`.
struct ConcurrentSinkShim<'a> {
    inner: Mutex<&'a dyn ConcurrentSink>,
}

impl ImageSink for ConcurrentSinkShim<'_> {
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.inner
            .lock()
            .map_err(|_| ViprsError::Scheduler("sink mutex poisoned".into()))?
            .write_region_concurrent(region, data)
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}
