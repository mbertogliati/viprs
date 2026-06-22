//! Discard sink — forces full pipeline evaluation without retaining output pixels.
//!
//! Use this in benchmarks where the goal is to measure processing throughput
//! without inflating RSS with the output buffer.

use std::any::Any;

use viprs_core::{error::ViprsError, image::Region};
use viprs_ports::sink::{ConcurrentSink, ImageSink};

/// A sink that accepts every output region and immediately discards it.
///
/// No pixel data is stored. This is the correct sink for no-E2E benchmarks
/// that need to force full pipeline evaluation without output-buffer RSS inflation
/// or codec encode cost.
///
/// # Examples
///
/// ```rust
/// use viprs_core::image::Region;
/// use viprs_ports::sink::ImageSink;
/// use viprs_runtime::sinks::discard::DiscardSink;
///
/// let mut sink = DiscardSink::new();
/// sink.write_region(Region::new(0, 0, 16, 16), &[0_u8; 768])?;
/// viprs_ports::sink::ImageSink::finish(Box::new(DiscardSink::new()))?;
/// # Ok::<(), viprs_core::error::ViprsError>(())
/// ```
pub struct DiscardSink;

impl DiscardSink {
    /// Creates a new discard sink.
    ///
    /// No allocation occurs. All output regions written to this sink are
    /// silently discarded.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for DiscardSink {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageSink for DiscardSink {
    #[inline]
    fn write_region(&mut self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
        Ok(())
    }

    fn as_concurrent_sink(&self) -> Option<&dyn ConcurrentSink> {
        Some(self)
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}

impl ConcurrentSink for DiscardSink {
    #[inline]
    fn write_region_concurrent(&self, _region: Region, _data: &[u8]) -> Result<(), ViprsError> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DiscardSink;
    use viprs_core::image::Region;
    use viprs_ports::sink::ImageSink;

    #[test]
    fn discards_sequential_regions() {
        let mut sink = DiscardSink::new();
        sink.write_region(Region::new(0, 0, 4, 4), &[1_u8; 64])
            .expect("discard sink accepts sequential writes");
        ImageSink::finish(Box::new(sink)).expect("discard sink finishes after sequential writes");
    }

    #[test]
    fn exposes_concurrent_sink_view() {
        let sink = DiscardSink::new();
        let concurrent = sink
            .as_concurrent_sink()
            .expect("discard sink exposes concurrent view");
        concurrent
            .write_region_concurrent(Region::new(0, 0, 2, 2), &[0_u8; 16])
            .expect("discard sink accepts concurrent writes");
    }
}
