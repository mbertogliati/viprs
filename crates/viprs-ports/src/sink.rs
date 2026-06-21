//! Sink port traits for pipeline outputs.
//!
//! Sinks are the final destination for tiles produced by a scheduler. They
//! abstract over infrastructure concerns such as writing into memory, files, or
//! foreign buffers while keeping the domain independent from those details.

use std::any::Any;

use viprs_core::{error::ViprsError, image::Region};

/// A consumer of pixel data — the exit point of a pipeline.
///
/// The scheduler calls `write_region` for each tile as it is produced,
/// then calls `finish` once all tiles have been delivered.
///
/// # Examples
///
/// ```rust
/// use viprs_ports::sink::ImageSink;
/// use viprs_core::{error::ViprsError, image::Region};
///
/// struct CollectingSink(Vec<u8>);
///
/// impl ImageSink for CollectingSink {
///     fn write_region(&mut self, _region: Region, data: &[u8]) -> Result<(), ViprsError> {
///         self.0.extend_from_slice(data);
///         Ok(())
///     }
///
///     fn finish(self: Box<Self>) -> Result<(), ViprsError> {
///         Ok(())
///     }
/// }
/// ```
pub trait ImageSink: Send {
    /// Writes one produced region into the sink's backing destination.
    ///
    /// This method lets schedulers stream tile output without knowing whether the
    /// destination is memory, a file, or another external system.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ImageSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct CollectingSink(Vec<u8>);
    /// # impl ImageSink for CollectingSink {
    /// #     fn write_region(&mut self, _region: Region, data: &[u8]) -> Result<(), ViprsError> {
    /// #         self.0.extend_from_slice(data);
    /// #         Ok(())
    /// #     }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let mut sink = CollectingSink(Vec::new());
    /// sink.write_region(Region::new(0, 0, 1, 1), &[255])?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError>;

    /// Returns a concurrent view of this sink when the implementation supports parallel writes.
    ///
    /// This hook lets schedulers opt into lock-free or internally synchronized
    /// sink writes without changing the base sequential sink contract.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ImageSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct CollectingSink;
    /// # impl ImageSink for CollectingSink {
    /// #     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let sink = CollectingSink;
    /// assert!(sink.as_concurrent_sink().is_none());
    /// ```
    fn as_concurrent_sink(&self) -> Option<&dyn ConcurrentSink> {
        None
    }

    /// Finalizes the sink after the scheduler has delivered all regions.
    ///
    /// This method solves shutdown and flush concerns for destinations that must
    /// commit buffered data once the pipeline has completed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ImageSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct CollectingSink;
    /// # impl ImageSink for CollectingSink {
    /// #     fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// Box::new(CollectingSink).finish()?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn finish(self: Box<Self>) -> Result<(), ViprsError>;
}

/// Sink that can receive writes concurrently from multiple threads.
///
/// The implementation guarantees that `write_region_concurrent` is safe to call
/// from multiple threads simultaneously, given that the regions passed are
/// non-overlapping (invariant guaranteed by `generate_tiles` in the scheduler).
///
/// `dyn ConcurrentSink` is acceptable in the scheduler (CLAUDE.md rule 1)
/// because it is a runtime registry point, not a per-pixel hot path.
///
/// # Examples
///
/// ```rust
/// use viprs_ports::sink::ConcurrentSink;
/// use viprs_core::{error::ViprsError, image::Region};
///
/// struct SharedSink;
///
/// impl ConcurrentSink for SharedSink {
///     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
///         Ok(())
///     }
///
///     fn as_any(&self) -> &dyn std::any::Any {
///         self
///     }
///
///     fn finish(self: Box<Self>) -> Result<(), ViprsError> {
///         Ok(())
///     }
/// }
/// ```
pub trait ConcurrentSink: Send + Sync + Any {
    /// Writes one region while allowing multiple threads to call the sink at the same time.
    ///
    /// This method solves parallel sink delivery for schedulers that can produce
    /// disjoint tiles concurrently.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ConcurrentSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct SharedSink;
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let sink = SharedSink;
    /// sink.write_region_concurrent(Region::new(0, 0, 1, 1), &[0])?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn write_region_concurrent(&self, region: Region, data: &[u8]) -> Result<(), ViprsError>;

    /// Exposes the concrete sink for downcasting in adapter-specific code.
    ///
    /// This method solves integration points that need to recover a concrete sink
    /// type after crossing an object-safe boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ConcurrentSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct SharedSink;
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// let sink = SharedSink;
    /// assert!(sink.as_any().downcast_ref::<SharedSink>().is_some());
    /// ```
    fn as_any(&self) -> &dyn Any;

    /// Finalizes a concurrent sink after all parallel writes have completed.
    ///
    /// This method solves flush or commit work for sinks that buffer concurrent
    /// writes before exposing the finished image to callers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use viprs_ports::sink::ConcurrentSink;
    /// # use viprs_core::{error::ViprsError, image::Region};
    /// # struct SharedSink;
    /// # impl ConcurrentSink for SharedSink {
    /// #     fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
    /// #         Ok(())
    /// #     }
    /// #     fn as_any(&self) -> &dyn std::any::Any { self }
    /// #     fn finish(self: Box<Self>) -> Result<(), ViprsError> { Ok(()) }
    /// # }
    /// Box::new(SharedSink).finish()?;
    /// # Ok::<(), ViprsError>(())
    /// ```
    fn finish(self: Box<Self>) -> Result<(), ViprsError>;
}
