#![allow(clippy::unnecessary_wraps)]
// REASON: the sink API matches fallible sink constructors used by other backends.

use std::cell::UnsafeCell;

use crate::{
    adapters::pipeline::CompiledPipeline,
    domain::{
        error::ViprsError,
        format::{BandFormat, BandFormatId},
        image::{Image, ImageMetadata, Region},
    },
    ports::sink::{ConcurrentSink, ImageSink},
};

/// An in-memory sink that accumulates tile writes into a contiguous row-major buffer.
///
/// # Concurrency model
///
/// `buffer` is wrapped in `UnsafeCell` to support concurrent writes via `ConcurrentSink`.
/// The safety invariant — that no two threads write to the same byte range — is upheld by
/// `generate_tiles` in the scheduler, which produces disjoint tiles. See the `Sync` impl
/// below for the full argument.
// SAFETY invariant: writes are always to non-overlapping byte ranges
// (guaranteed by generate_tiles in the scheduler). No two threads ever write
// to the same byte simultaneously.
pub struct MemorySink {
    buffer: UnsafeCell<Vec<u8>>,
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: usize,
}

// SAFETY: Concurrent writes to `buffer` are safe because `generate_tiles` guarantees
// disjoint output tiles. Each call to `write_region_concurrent` writes only to the
// row range corresponding to its `Region`, which does not overlap with any other
// in-flight tile. This invariant must be upheld by every caller of
// `write_region_concurrent`; it is documented on `ConcurrentSink`.
unsafe impl Sync for MemorySink {}

impl MemorySink {
    fn checked_buffer_len(
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
    ) -> Result<usize, ViprsError> {
        let bytes_per_sample_u64 =
            u64::try_from(bytes_per_sample).map_err(|_| ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: u128::from(width)
                    * u128::from(height)
                    * u128::from(bands)
                    * bytes_per_sample as u128,
                limit_bytes: usize::MAX as u128,
                details: "memory sink byte count exceeds addressable memory",
            })?;
        let total_bytes =
            u128::from(width) * u128::from(height) * u128::from(bands) * bytes_per_sample as u128;

        let Some(bytes) = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|pixel_count| pixel_count.checked_mul(u64::from(bands)))
            .and_then(|sample_count| sample_count.checked_mul(bytes_per_sample_u64))
        else {
            return Err(ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: total_bytes,
                limit_bytes: usize::MAX as u128,
                details: "memory sink byte count exceeds addressable memory",
            });
        };

        usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(bytes),
            limit_bytes: usize::MAX as u128,
            details: "memory sink byte count exceeds addressable memory",
        })
    }

    /// Create a new sink pre-sized for an image with the given dimensions.
    ///
    /// `bytes_per_sample` must match the sample size of the pipeline's output format.
    /// Prefer [`MemorySink::for_pipeline`] to derive this automatically.
    pub fn new(
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
    ) -> Result<Self, ViprsError> {
        let size = Self::checked_buffer_len(width, height, bands, bytes_per_sample)?;
        Ok(Self {
            buffer: UnsafeCell::new(vec![0u8; size]),
            width,
            height,
            bands,
            bytes_per_sample,
        })
    }

    /// Create a sink correctly sized for the output of `pipeline`.
    ///
    /// Derives `bytes_per_sample` from `pipeline.output_format`, eliminating the
    /// class of bugs caused by passing the wrong value manually.
    pub fn for_pipeline(pipeline: &CompiledPipeline) -> Result<Self, ViprsError> {
        let bps = match pipeline.output_format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 | BandFormatId::I16 => 2,
            BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
            BandFormatId::F64 => 8,
        };
        Self::new(pipeline.width, pipeline.height, pipeline.output_bands, bps)
    }

    /// Consume the sink and return the underlying pixel buffer.
    pub fn into_buffer(self) -> Vec<u8> {
        self.buffer.into_inner()
    }

    /// Consumes the sink and rehydrates the collected bytes into an owned [`Image`].
    pub fn into_image<F: BandFormat>(
        self,
        width: u32,
        height: u32,
        bands: u32,
        metadata: ImageMetadata,
    ) -> Result<Image<F>, ViprsError> {
        let bytes = self.into_buffer();
        let sample_size = std::mem::size_of::<F::Sample>();
        if !bytes.len().is_multiple_of(sample_size) {
            return Err(ViprsError::Scheduler(format!(
                "memory sink buffer length {} is not divisible by sample size {}",
                bytes.len(),
                sample_size,
            )));
        }

        let samples: Vec<F::Sample> = bytes
            .chunks_exact(sample_size)
            .map(bytemuck::pod_read_unaligned::<F::Sample>)
            .collect();

        Image::from_buffer(width, height, bands, samples).map(|image| image.with_metadata(metadata))
    }

    /// Scatter `data` (a contiguous tile in row-major order) into the correct
    /// position inside `buf`, which covers the full image in row-major order.
    ///
    /// Both `ImageSink` and `ConcurrentSink` call this helper so the scatter
    /// logic is not duplicated. The caller is responsible for obtaining `buf`
    /// and ensuring exclusivity of the written ranges.
    fn scatter_to_buffer(
        buf: &mut [u8],
        width: u32,
        bands: u32,
        bytes_per_sample: usize,
        region: Region,
        data: &[u8],
    ) -> Result<(), ViprsError> {
        let stride = width as usize * bands as usize * bytes_per_sample;
        let pixel_bytes = bands as usize * bytes_per_sample;

        if region.x == 0 && region.width == width {
            let tile_bytes = region.height as usize * region.width as usize * pixel_bytes;
            let dst_start = region.y as usize * stride;
            let dst_end = dst_start + tile_bytes;
            buf[dst_start..dst_end].copy_from_slice(data);
            return Ok(());
        }

        for row in 0..region.height as usize {
            let src_start = row * region.width as usize * pixel_bytes;
            let src_end = src_start + region.width as usize * pixel_bytes;
            let dst_y = region.y as usize + row;
            let dst_x = region.x as usize;
            let dst_start = dst_y * stride + dst_x * pixel_bytes;
            let dst_end = dst_start + region.width as usize * pixel_bytes;
            buf[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }
        Ok(())
    }

    fn validate_region(&self, region: Region) -> Result<usize, ViprsError> {
        let end_x = i64::from(region.x) + i64::from(region.width);
        let end_y = i64::from(region.y) + i64::from(region.height);

        if region.x < 0
            || region.y < 0
            || end_x > i64::from(self.width)
            || end_y > i64::from(self.height)
        {
            return Err(ViprsError::Scheduler(format!(
                "memory sink region {region:?} is out of bounds for {}x{} image",
                self.width, self.height
            )));
        }

        Ok(self.width as usize * self.bands as usize * self.bytes_per_sample)
    }

    fn validate_data_len(&self, region: Region, data_len: usize) -> Result<(), ViprsError> {
        let Some(expected) = region
            .checked_pixel_count()
            .and_then(|pixel_count| {
                usize::try_from(self.bands)
                    .ok()
                    .and_then(|bands| pixel_count.checked_mul(bands))
            })
            .and_then(|sample_count| sample_count.checked_mul(self.bytes_per_sample))
        else {
            return Err(ViprsError::Scheduler(format!(
                "memory sink region byte count exceeds addressable memory for {region:?}"
            )));
        };

        if data_len != expected {
            return Err(ViprsError::Scheduler(format!(
                "memory sink buffer length {data_len} does not match expected {expected} for {region:?}"
            )));
        }

        Ok(())
    }

    pub(crate) const fn is_contiguous_full_width_region(&self, region: Region) -> bool {
        region.x == 0 && region.width == self.width
    }

    pub(crate) unsafe fn with_full_width_region_mut_concurrent<R>(
        &self,
        region: Region,
        f: impl FnOnce(&mut [u8]) -> R,
    ) -> Result<Option<R>, ViprsError> {
        let stride = self.validate_region(region)?;
        if !self.is_contiguous_full_width_region(region) {
            return Ok(None);
        }

        let start = region.y as usize * stride;
        let end = start + region.height as usize * stride;
        // SAFETY: callers must uphold the same non-overlapping-region invariant as
        // `write_region_concurrent`; this function only lends a mutable view to `f`
        // for the duration of this call, and full-width strips have disjoint byte
        // ranges across tiles when that invariant holds.
        let buf = unsafe { &mut *self.buffer.get() };
        Ok(Some(f(&mut buf[start..end])))
    }
}

impl ImageSink for MemorySink {
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.validate_region(region)?;
        self.validate_data_len(region, data.len())?;
        // Exclusive `&mut self` — no concurrent access possible here.
        let buf = self.buffer.get_mut();
        Self::scatter_to_buffer(
            buf,
            self.width,
            self.bands,
            self.bytes_per_sample,
            region,
            data,
        )
    }

    fn as_concurrent_sink(&self) -> Option<&dyn ConcurrentSink> {
        Some(self)
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}

impl ConcurrentSink for MemorySink {
    fn write_region_concurrent(&self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.validate_region(region)?;
        self.validate_data_len(region, data.len())?;
        // SAFETY: tiles produced by `generate_tiles` are disjoint, so each concurrent call writes a unique byte range in `self.buffer`; the `Sync` implementation on `MemorySink` owns that invariant.
        let buf = unsafe { &mut *self.buffer.get() };
        Self::scatter_to_buffer(
            buf,
            self.width,
            self.bands,
            self.bytes_per_sample,
            region,
            data,
        )
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::pipeline::PipelineBuilder;
    use crate::domain::op::{Op, OperationBridge};
    use crate::domain::{format::BandFormatId, image::Region};
    use crate::domain::{
        format::F32,
        format::U8,
        image::{DemandHint, Tile, TileMut},
    };
    use std::sync::Arc;

    // ── helpers ──────────────────────────────────────────────────────────────

    // `bands` is passed to OperationBridge::new, not used inside process_region itself.
    #[allow(dead_code)]
    struct PassThrough {
        bands: u32,
    }

    impl Op for PassThrough {
        type Input = U8;
        type Output = U8;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, _: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
            output.data.copy_from_slice(input.data);
        }
    }

    struct F32PassThrough;
    impl Op for F32PassThrough {
        type Input = F32;
        type Output = F32;
        type State = ();
        fn demand_hint(&self) -> DemandHint {
            DemandHint::Any
        }
        fn required_input_region(&self, r: &Region) -> Region {
            *r
        }
        fn start(&self) {}
        #[inline]
        fn process_region(&self, _: &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
            output.data.copy_from_slice(input.data);
        }
    }

    // ── write_region (ImageSink) ──────────────────────────────────────────────

    #[test]
    fn memory_sink_write_and_read_back() {
        // 4x4 single-band image, u8 (1 byte per sample)
        let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();

        // Write first tile: top-left 2x2 with values [1,2,3,4]
        sink.write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();
        // Write second tile: bottom-right 2x2 with values [5,6,7,8]
        sink.write_region(Region::new(2, 2, 2, 2), &[5u8, 6, 7, 8])
            .unwrap();

        let buf = sink.into_buffer();
        // Row 0: [1, 2, 0, 0]
        // Row 1: [3, 4, 0, 0]
        // Row 2: [0, 0, 5, 6]
        // Row 3: [0, 0, 7, 8]
        assert_eq!(buf[0], 1);
        assert_eq!(buf[1], 2);
        assert_eq!(buf[4], 3);
        assert_eq!(buf[5], 4);
        assert_eq!(buf[10], 5);
        assert_eq!(buf[11], 6);
        assert_eq!(buf[14], 7);
        assert_eq!(buf[15], 8);
        // Unwritten pixels remain 0
        assert_eq!(buf[2], 0);
        assert_eq!(buf[8], 0);
    }

    // ── write_region_concurrent (ConcurrentSink) ─────────────────────────────

    #[test]
    fn concurrent_write_matches_sequential_write() {
        // Verify that write_region_concurrent produces the same result as write_region
        // for identical inputs, using a 4x4 image with two disjoint 2x2 tiles.
        let mut seq = MemorySink::new(4, 4, 1, 1).unwrap();
        seq.write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();
        seq.write_region(Region::new(2, 2, 2, 2), &[5u8, 6, 7, 8])
            .unwrap();
        let seq_buf = seq.into_buffer();

        let conc = MemorySink::new(4, 4, 1, 1).unwrap();
        conc.write_region_concurrent(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();
        conc.write_region_concurrent(Region::new(2, 2, 2, 2), &[5u8, 6, 7, 8])
            .unwrap();
        let conc_buf = conc.into_buffer();

        assert_eq!(seq_buf, conc_buf);
    }

    #[test]
    fn write_region_rejects_negative_origin() {
        let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();

        let result = sink.write_region(Region::new(-1, 0, 1, 1), &[7]);

        assert!(matches!(result, Err(ViprsError::Scheduler(_))));
    }

    #[test]
    fn write_region_rejects_oversized_width() {
        let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();

        let result = sink.write_region(Region::new(0, 0, u32::MAX, 1), &[7]);

        assert!(matches!(result, Err(ViprsError::Scheduler(_))));
    }

    #[test]
    fn write_region_rejects_regions_whose_x_end_overflows_i32() {
        let mut sink = MemorySink::new(1, 1, 1, 1).unwrap();

        let err = sink
            .write_region(Region::new(i32::MAX, 0, 1, 1), &[1])
            .unwrap_err();

        assert!(matches!(err, ViprsError::Scheduler(message) if message.contains("out of bounds")));
    }

    #[test]
    fn memory_sink_write_region_rejects_undersized_tile_buffer() {
        let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();

        let result = sink.write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3]);

        assert!(matches!(result, Err(ViprsError::Scheduler(_))));
    }

    #[test]
    fn memory_sink_write_region_accepts_exact_tile_buffer() {
        let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();

        sink.write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();

        assert_eq!(sink.into_buffer()[..8], [1, 2, 0, 0, 3, 4, 0, 0]);
    }

    #[test]
    fn concurrent_write_region_rejects_negative_origin() {
        let sink = MemorySink::new(4, 4, 1, 1).unwrap();

        let result = sink.write_region_concurrent(Region::new(-1, 0, 1, 1), &[7]);

        assert!(matches!(result, Err(ViprsError::Scheduler(_))));
    }

    #[test]
    fn memory_sink_write_region_concurrent_rejects_undersized_tile_buffer() {
        let sink = MemorySink::new(4, 4, 1, 1).unwrap();

        let result = sink.write_region_concurrent(Region::new(0, 0, 2, 2), &[1u8, 2, 3]);

        assert!(matches!(result, Err(ViprsError::Scheduler(_))));
    }

    #[test]
    fn memory_sink_write_region_concurrent_accepts_exact_tile_buffer() {
        let sink = MemorySink::new(4, 4, 1, 1).unwrap();

        sink.write_region_concurrent(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();

        assert_eq!(sink.into_buffer()[..8], [1, 2, 0, 0, 3, 4, 0, 0]);
    }

    #[test]
    fn full_width_region_helper_limits_mut_borrow_to_callback() {
        let sink = MemorySink::new(4, 4, 1, 1).unwrap();
        let region = Region::new(0, 1, 4, 2);

        // SAFETY: this test performs a single callback on a full-width strip, so no
        // overlapping mutable access to the sink buffer occurs.
        let wrote = unsafe {
            sink.with_full_width_region_mut_concurrent(region, |output| {
                output.copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
            })
        }
        .unwrap();

        assert_eq!(wrote, Some(()));
        assert_eq!(
            sink.into_buffer(),
            vec![0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]
        );
    }

    #[test]
    fn full_width_region_helper_rejects_partial_width_region() {
        let sink = MemorySink::new(4, 4, 1, 1).unwrap();

        // SAFETY: the helper returns `None` before borrowing the buffer when the region
        // is not full-width, so no mutable access is created here.
        let result =
            unsafe { sink.with_full_width_region_mut_concurrent(Region::new(1, 0, 2, 2), |_| ()) }
                .unwrap();

        assert_eq!(result, None);
    }

    #[test]
    fn concurrent_writes_from_multiple_threads_are_safe() {
        // Spawn two threads that each write one disjoint tile. Verify the final
        // buffer matches what sequential writes would produce.
        //
        // The tiles are:
        //   Thread 0 → top half    (rows 0..4):  bytes 0..4
        //   Thread 1 → bottom half (rows 4..8):  bytes 4..8
        //
        // 8x8 single-band image, 1 bps.
        let sink = Arc::new(MemorySink::new(8, 8, 1, 1).unwrap());

        let sink_a = Arc::clone(&sink);
        let h0 = std::thread::spawn(move || {
            // Top 4 rows (0..4)
            let data: Vec<u8> = (1u8..=32).collect(); // 8*4 = 32 bytes
            sink_a
                .write_region_concurrent(Region::new(0, 0, 8, 4), &data)
                .unwrap();
        });

        let sink_b = Arc::clone(&sink);
        let h1 = std::thread::spawn(move || {
            // Bottom 4 rows (4..8)
            let data: Vec<u8> = (33u8..=64).collect(); // 8*4 = 32 bytes
            sink_b
                .write_region_concurrent(Region::new(0, 4, 8, 4), &data)
                .unwrap();
        });

        h0.join().unwrap();
        h1.join().unwrap();

        // Arc::try_unwrap is safe here because both threads have finished.
        let buf = Arc::try_unwrap(sink)
            .ok()
            .expect("arc should have exactly one strong reference")
            .into_buffer();
        let expected: Vec<u8> = (1u8..=64).collect();
        assert_eq!(buf, expected);
    }

    // ── for_pipeline ─────────────────────────────────────────────────────────

    #[test]
    fn for_pipeline_infers_bps_for_u8() {
        let pipeline = PipelineBuilder::new(16, 16)
            .then(Box::new(OperationBridge::new(
                PassThrough { bands: 1 },
                1u32,
            )))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(pipeline.output_format, BandFormatId::U8);

        let sink = MemorySink::for_pipeline(&pipeline).unwrap();
        // 16 * 16 * 1 band * 1 bps = 256 bytes
        assert_eq!(sink.into_buffer().len(), 256);
    }

    #[test]
    fn for_pipeline_infers_bps_for_f32() {
        use crate::adapters::sources::zero::ZeroSource;
        use crate::domain::format::F32;

        // Use a F32 source so that the pipeline's current_format matches F32PassThrough.
        let pipeline = PipelineBuilder::from_source(ZeroSource::<F32>::new(16, 16, 1))
            .then(Box::new(OperationBridge::new(F32PassThrough, 1u32)))
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(pipeline.output_format, BandFormatId::F32);

        let sink = MemorySink::for_pipeline(&pipeline).unwrap();
        // 16 * 16 * 1 band * 4 bps = 1024 bytes
        assert_eq!(sink.into_buffer().len(), 1024);
    }
}
