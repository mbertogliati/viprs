#![allow(clippy::significant_drop_tightening)]
// REASON: sink buffer swaps intentionally keep locks until both halves of the exchange are consistent.

use std::{
    path::Path,
    sync::{Arc, Condvar, Mutex},
    thread::{self, JoinHandle},
};

use crossbeam_channel::{Receiver, Sender, bounded};

use crate::{
    adapters::pipeline::CompiledPipeline,
    domain::{error::ViprsError, format::BandFormatId, image::Region},
    ports::sink::{ConcurrentSink, ImageSink},
};

use super::file_sink::{FileSinkWriter, RawFileWriter};

struct FlushMessage {
    region: Region,
    data: Vec<u8>,
    used_len: usize,
}

struct StripBuffer {
    data: Vec<u8>,
    start_row: u32,
    rows: u32,
    written_bytes: usize,
}

impl StripBuffer {
    fn new(capacity: usize, rows: u32) -> Self {
        Self {
            data: vec![0; capacity],
            start_row: 0,
            rows,
            written_bytes: 0,
        }
    }

    const fn used_len(&self, stride: usize) -> usize {
        self.rows as usize * stride
    }

    const fn reset(&mut self, start_row: u32, rows: u32) {
        self.start_row = start_row;
        self.rows = rows;
        self.written_bytes = 0;
    }
}

struct DoubleBufferState {
    next_x: i32,
    next_y: i32,
    writer_error: Option<String>,
    current: StripBuffer,
}

struct SharedState {
    state: Mutex<DoubleBufferState>,
    ready: Condvar,
}

impl SharedState {
    const fn new(current: StripBuffer) -> Self {
        Self {
            state: Mutex::new(DoubleBufferState {
                next_x: 0,
                next_y: 0,
                writer_error: None,
                current,
            }),
            ready: Condvar::new(),
        }
    }

    fn store_writer_error(&self, err: &ViprsError) -> Result<(), ViprsError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| ViprsError::Scheduler("double-buffer sink state mutex poisoned".into()))?;
        state.writer_error = Some(err.to_string());
        self.ready.notify_all();
        Ok(())
    }

    fn writer_error(state: &DoubleBufferState) -> Option<ViprsError> {
        state.writer_error.as_ref().map(|message| {
            ViprsError::Scheduler(format!("double-buffer sink writer failed: {message}"))
        })
    }
}

/// Disk sink that overlaps tile production with write-behind flushes.
///
/// The pipeline fills a full-width strip buffer while a dedicated background thread
/// flushes the previously completed strip to the encoder or file writer. Buffer
/// ownership moves across channels, so pixel data is never shared via
/// `Arc<Mutex<_>>`.
pub struct DoubleBufferSink {
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: usize,
    rows_per_buffer: u32,
    flush_sender: Sender<FlushMessage>,
    available_receiver: Receiver<Vec<u8>>,
    shared: Arc<SharedState>,
    writer_handle: Mutex<Option<JoinHandle<Result<(), ViprsError>>>>,
}

impl DoubleBufferSink {
    fn checked_total_bytes(
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
        details: &'static str,
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
                details,
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
                details,
            });
        };

        usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(bytes),
            limit_bytes: usize::MAX as u128,
            details,
        })
    }

    /// Creates a double-buffered sink for a row-major output stream.
    pub fn new(
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
        rows_per_buffer: u32,
        mut writer: Box<dyn FileSinkWriter>,
    ) -> Result<Self, ViprsError> {
        if rows_per_buffer == 0 {
            return Err(ViprsError::Scheduler(
                "double-buffer sink requires rows_per_buffer > 0".into(),
            ));
        }

        let capacity = Self::checked_total_bytes(
            width,
            rows_per_buffer,
            bands,
            bytes_per_sample,
            "double-buffer sink buffer capacity exceeds addressable memory",
        )?;
        let initial_rows = height.min(rows_per_buffer);
        let current = StripBuffer::new(capacity, initial_rows);

        let (flush_sender, flush_receiver) = bounded(1);
        let (available_sender, available_receiver) = bounded(1);
        available_sender.send(vec![0; capacity]).map_err(|_| {
            ViprsError::Scheduler("double-buffer sink failed to seed spare buffer".into())
        })?;

        let shared = Arc::new(SharedState::new(current));
        let writer_shared = Arc::clone(&shared);
        let writer_handle = thread::spawn(move || {
            Self::writer_loop(
                &mut *writer,
                &flush_receiver,
                &available_sender,
                &writer_shared,
            )
        });

        Ok(Self {
            width,
            height,
            bands,
            bytes_per_sample,
            rows_per_buffer,
            flush_sender,
            available_receiver,
            shared,
            writer_handle: Mutex::new(Some(writer_handle)),
        })
    }

    /// Creates a double-buffered sink sized from a compiled pipeline output.
    pub fn for_pipeline(
        pipeline: &CompiledPipeline,
        rows_per_buffer: u32,
        writer: Box<dyn FileSinkWriter>,
    ) -> Result<Self, ViprsError> {
        let bytes_per_sample = match pipeline.output_format {
            BandFormatId::U8 => 1,
            BandFormatId::U16 | BandFormatId::I16 => 2,
            BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
            BandFormatId::F64 => 8,
        };
        Self::new(
            pipeline.width,
            pipeline.height,
            pipeline.output_bands,
            bytes_per_sample,
            rows_per_buffer,
            writer,
        )
    }

    /// Creates a RAW file sink that writes full-width strips directly to `path`.
    pub fn create_raw(
        path: impl AsRef<Path>,
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
        rows_per_buffer: u32,
    ) -> Result<Self, ViprsError> {
        Self::new(
            width,
            height,
            bands,
            bytes_per_sample,
            rows_per_buffer,
            Box::new(RawFileWriter::create(path)?),
        )
    }

    const fn stride(width: u32, bands: u32, bytes_per_sample: usize) -> usize {
        width as usize * bands as usize * bytes_per_sample
    }

    fn validate_region(&self, region: Region, data_len: usize) -> Result<(), ViprsError> {
        let end_x = i64::from(region.x) + i64::from(region.width);
        let end_y = i64::from(region.y) + i64::from(region.height);
        if region.x < 0
            || region.y < 0
            || end_x > i64::from(self.width)
            || end_y > i64::from(self.height)
        {
            return Err(ViprsError::Scheduler(format!(
                "double-buffer sink region {region:?} is out of bounds for {}x{} image",
                self.width, self.height
            )));
        }

        let expected = Self::checked_total_bytes(
            region.width,
            region.height,
            self.bands,
            self.bytes_per_sample,
            "double-buffer sink region byte count exceeds addressable memory",
        )?;
        if data_len != expected {
            return Err(ViprsError::Scheduler(format!(
                "double-buffer sink buffer length {data_len} does not match expected {expected} for {region:?}"
            )));
        }

        Ok(())
    }

    fn writer_loop(
        writer: &mut dyn FileSinkWriter,
        flush_receiver: &Receiver<FlushMessage>,
        available_sender: &Sender<Vec<u8>>,
        shared: &Arc<SharedState>,
    ) -> Result<(), ViprsError> {
        while let Ok(message) = flush_receiver.recv() {
            if let Err(err) = writer.write_region(message.region, &message.data[..message.used_len])
            {
                let _ = shared.store_writer_error(&err);
                return Err(err);
            }

            if available_sender.send(message.data).is_err() {
                return Err(ViprsError::Scheduler(
                    "double-buffer sink spare buffer channel closed".into(),
                ));
            }
        }

        if let Err(err) = writer.finish() {
            let _ = shared.store_writer_error(&err);
            return Err(err);
        }

        Ok(())
    }

    fn scatter_to_strip(
        &self,
        buffer: &mut [u8],
        strip_start_row: u32,
        region: Region,
        data: &[u8],
    ) -> Result<(), ViprsError> {
        let stride = Self::stride(self.width, self.bands, self.bytes_per_sample);
        let pixel_bytes = self.bands as usize * self.bytes_per_sample;
        let start_row = u32::try_from(region.y).map_err(|_| {
            ViprsError::Scheduler(format!(
                "double-buffer sink received negative y coordinate in {region:?}"
            ))
        })?;

        for row in 0..region.height as usize {
            let src_start = row * region.width as usize * pixel_bytes;
            let src_end = src_start + region.width as usize * pixel_bytes;
            let dst_y = start_row as usize + row - strip_start_row as usize;
            let dst_x = region.x as usize;
            let dst_start = dst_y * stride + dst_x * pixel_bytes;
            let dst_end = dst_start + region.width as usize * pixel_bytes;
            buffer[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }

        Ok(())
    }

    const fn advance_expected_tile(state: &mut DoubleBufferState, region: Region, width: u32) {
        let next_x = region.x + region.width as i32;
        if next_x >= width as i32 {
            state.next_x = 0;
            state.next_y = region.y + region.height as i32;
        } else {
            state.next_x = next_x;
            state.next_y = region.y;
        }
    }

    fn rotate_buffer(&self, state: &mut DoubleBufferState) -> Result<(), ViprsError> {
        let stride = Self::stride(self.width, self.bands, self.bytes_per_sample);
        let used_len = state.current.used_len(stride);
        if state.current.written_bytes != used_len {
            return Ok(());
        }

        let flush_region = Region::new(
            0,
            state.current.start_row as i32,
            self.width,
            state.current.rows,
        );
        let mut flush_buffer = Vec::new();
        std::mem::swap(&mut flush_buffer, &mut state.current.data);
        self.flush_sender
            .send(FlushMessage {
                region: flush_region,
                data: flush_buffer,
                used_len,
            })
            .map_err(|_| {
                SharedState::writer_error(state).unwrap_or_else(|| {
                    ViprsError::Scheduler(
                        "double-buffer sink writer channel closed before strip flush".into(),
                    )
                })
            })?;

        let next_start_row = state.current.start_row + state.current.rows;
        let next_rows = self
            .height
            .saturating_sub(next_start_row)
            .min(self.rows_per_buffer);
        if next_rows > 0 {
            let next_buffer = self.available_receiver.recv().map_err(|_| {
                SharedState::writer_error(state).unwrap_or_else(|| {
                    ViprsError::Scheduler(
                        "double-buffer sink lost its spare buffer before swap".into(),
                    )
                })
            })?;
            state.current.data = next_buffer;
        }
        state.current.reset(next_start_row, next_rows);
        Ok(())
    }

    fn write_region_into_buffers(
        &self,
        state: &mut DoubleBufferState,
        region: Region,
        data: &[u8],
    ) -> Result<(), ViprsError> {
        let pixel_bytes = self.bands as usize * self.bytes_per_sample;
        let row_bytes = region.width as usize * pixel_bytes;
        let mut rows_written = 0u32;

        while rows_written < region.height {
            if let Some(err) = SharedState::writer_error(state) {
                return Err(err);
            }

            if state.current.rows == 0 {
                return Err(ViprsError::Scheduler(
                    "double-buffer sink exhausted strip buffers before all rows were written"
                        .into(),
                ));
            }

            let piece_y = u32::try_from(region.y).map_err(|_| {
                ViprsError::Scheduler(format!(
                    "double-buffer sink received negative y coordinate in {region:?}"
                ))
            })? + rows_written;
            let strip_end = state.current.start_row + state.current.rows;
            if piece_y < state.current.start_row || piece_y >= strip_end {
                return Err(ViprsError::Scheduler(format!(
                    "double-buffer sink expected rows {}..{} but received {region:?}",
                    state.current.start_row, strip_end
                )));
            }

            let piece_rows = (strip_end - piece_y).min(region.height - rows_written);
            let start = rows_written as usize * row_bytes;
            let end = start + piece_rows as usize * row_bytes;
            let piece_region = Region::new(region.x, piece_y as i32, region.width, piece_rows);
            let used_len =
                state
                    .current
                    .used_len(Self::stride(self.width, self.bands, self.bytes_per_sample));

            self.scatter_to_strip(
                &mut state.current.data[..used_len],
                state.current.start_row,
                piece_region,
                &data[start..end],
            )?;
            state.current.written_bytes += end - start;
            rows_written += piece_rows;

            if state.current.written_bytes == used_len {
                self.rotate_buffer(state)?;
            }
        }

        Ok(())
    }
}

impl ImageSink for DoubleBufferSink {
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.write_region_concurrent(region, data)
    }

    fn as_concurrent_sink(&self) -> Option<&dyn ConcurrentSink> {
        Some(self)
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        ConcurrentSink::finish(self)
    }
}

impl ConcurrentSink for DoubleBufferSink {
    fn write_region_concurrent(&self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.validate_region(region, data.len())?;

        let mut state =
            self.shared.state.lock().map_err(|_| {
                ViprsError::Scheduler("double-buffer sink state mutex poisoned".into())
            })?;

        while state.next_x != region.x || state.next_y != region.y {
            if let Some(err) = SharedState::writer_error(&state) {
                return Err(err);
            }
            state = self.shared.ready.wait(state).map_err(|_| {
                ViprsError::Scheduler("double-buffer sink state mutex poisoned".into())
            })?;
        }

        if let Some(err) = SharedState::writer_error(&state) {
            return Err(err);
        }

        self.write_region_into_buffers(&mut state, region, data)?;
        Self::advance_expected_tile(&mut state, region, self.width);
        self.shared.ready.notify_all();
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        let state =
            self.shared.state.lock().map_err(|_| {
                ViprsError::Scheduler("double-buffer sink state mutex poisoned".into())
            })?;
        if state.current.rows > 0 || state.current.written_bytes > 0 {
            if let Some(err) = SharedState::writer_error(&state) {
                return Err(err);
            }
            return Err(ViprsError::Scheduler(
                "double-buffer sink finished before all strips were flushed".into(),
            ));
        }
        drop(state);

        // Drain any stale spare buffer so the writer can return its last strip buffer
        // without blocking on the bounded recycle channel during shutdown.
        while self.available_receiver.try_recv().is_ok() {}

        drop(self.flush_sender);

        let handle = self
            .writer_handle
            .into_inner()
            .map_err(|_| {
                ViprsError::Scheduler("double-buffer sink writer handle mutex poisoned".into())
            })?
            .ok_or_else(|| {
                ViprsError::Scheduler("double-buffer sink writer thread already joined".into())
            })?;

        handle.join().unwrap_or_else(|_| {
            Err(ViprsError::Scheduler(
                "double-buffer sink writer thread panicked".into(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct RecordingWriter {
        output: Arc<Mutex<Vec<u8>>>,
    }

    impl FileSinkWriter for RecordingWriter {
        fn write_region(&mut self, _region: Region, data: &[u8]) -> Result<(), ViprsError> {
            let mut output = self.output.lock().map_err(|_| {
                ViprsError::Scheduler("double-buffer sink test output mutex poisoned".into())
            })?;
            output.extend_from_slice(data);
            Ok(())
        }
    }

    fn oversized_dimensions_sink(
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
    ) -> DoubleBufferSink {
        let (flush_sender, flush_receiver) = bounded(1);
        drop(flush_receiver);
        let (_available_sender, available_receiver) = bounded(1);
        let shared = Arc::new(SharedState::new(StripBuffer::new(0, 0)));

        DoubleBufferSink {
            width,
            height,
            bands,
            bytes_per_sample,
            rows_per_buffer: 1,
            flush_sender,
            available_receiver,
            shared,
            writer_handle: Mutex::new(Some(std::thread::spawn(|| Ok(())))),
        }
    }

    #[test]
    fn double_buffer_sink_flushes_all_tiles() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::new(
            DoubleBufferSink::new(
                4,
                1000,
                1,
                1,
                1,
                Box::new(RecordingWriter {
                    output: Arc::clone(&output),
                }),
            )
            .unwrap(),
        );

        std::thread::scope(|scope| {
            for chunk in 0..8u32 {
                let sink = Arc::clone(&sink);
                scope.spawn(move || {
                    let start = chunk * 125;
                    let end = (start + 125).min(1000);
                    for row in start..end {
                        let value = row as u8;
                        sink.write_region_concurrent(
                            Region::new(0, row as i32, 4, 1),
                            &[value, value, value, value],
                        )
                        .unwrap();
                    }
                });
            }
        });

        let sink = Arc::try_unwrap(sink)
            .ok()
            .expect("double-buffer sink should have one strong reference");
        ConcurrentSink::finish(Box::new(sink)).unwrap();

        let output = output.lock().unwrap();
        assert_eq!(output.len(), 4000);
        for row in 0..1000usize {
            let expected = [(row as u8); 4];
            assert_eq!(&output[row * 4..row * 4 + 4], expected.as_slice());
        }
    }

    #[test]
    fn double_buffer_sink_new_rejects_overflowing_buffer_capacity() {
        let output = Arc::new(Mutex::new(Vec::new()));
        let err = match DoubleBufferSink::new(
            u32::MAX,
            1,
            u32::MAX,
            2,
            1,
            Box::new(RecordingWriter { output }),
        ) {
            Ok(_) => panic!("double-buffer sink should reject overflowing buffer capacity"),
            Err(err) => err,
        };

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                details: "double-buffer sink buffer capacity exceeds addressable memory",
                ..
            }
        ));
    }

    #[test]
    fn double_buffer_sink_rejects_regions_whose_byte_count_overflows() {
        let sink = oversized_dimensions_sink(u32::MAX, u32::MAX, u32::MAX, 2);

        let err = sink
            .write_region_concurrent(Region::new(0, 0, u32::MAX, u32::MAX), &[])
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                details: "double-buffer sink region byte count exceeds addressable memory",
                ..
            }
        ));

        ConcurrentSink::finish(Box::new(sink)).unwrap();
    }

    #[test]
    fn double_buffer_sink_rejects_regions_whose_x_end_overflows_i32() {
        let sink = oversized_dimensions_sink(1, 1, 1, 1);

        let err = sink
            .write_region_concurrent(Region::new(i32::MAX, 0, 1, 1), &[1])
            .unwrap_err();
        assert!(matches!(err, ViprsError::Scheduler(message) if message.contains("out of bounds")));

        ConcurrentSink::finish(Box::new(sink)).unwrap();
    }

    #[test]
    fn double_buffer_sink_rejects_regions_whose_y_end_overflows_i32() {
        let sink = oversized_dimensions_sink(1, 1, 1, 1);

        let err = sink
            .write_region_concurrent(Region::new(0, i32::MAX, 1, 1), &[1])
            .unwrap_err();
        assert!(matches!(err, ViprsError::Scheduler(message) if message.contains("out of bounds")));

        ConcurrentSink::finish(Box::new(sink)).unwrap();
    }
}
