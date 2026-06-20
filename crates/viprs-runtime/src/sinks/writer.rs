use std::io::Write;

use crate::{
    domain::{error::ViprsError, image::Region},
    ports::sink::ImageSink,
};

/// Type-erased encoder invoked by [`WriterSink::finish`].
///
/// The closure runs once after the full output image has been assembled in
/// row-major order. `dyn FnOnce` is acceptable here because dispatch happens at
/// sink teardown, outside the per-tile pixel path.
pub type EncodeFn<W> =
    Box<dyn FnOnce(&[u8], u32, u32, u32, &mut W) -> Result<(), ViprsError> + Send>;

/// A sink that accumulates tile writes into a full image buffer and then
/// streams the encoded result to a [`Write`] on [`ImageSink::finish`].
///
/// Memory usage is bounded by `width × height × bands × bytes_per_sample`.
/// The buffer is allocated exactly once during construction.
pub struct WriterSink<W: Write + Send> {
    writer: W,
    buffer: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    bytes_per_sample: usize,
    encode_fn: EncodeFn<W>,
}

impl<W: Write + Send> WriterSink<W> {
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
                details: "writer sink byte count exceeds addressable memory",
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
                details: "writer sink byte count exceeds addressable memory",
            });
        };

        usize::try_from(bytes).map_err(|_| ViprsError::ImageTooLarge {
            width,
            height,
            bands,
            bytes: u128::from(bytes),
            limit_bytes: usize::MAX as u128,
            details: "writer sink byte count exceeds addressable memory",
        })
    }

    /// Create a sink pre-sized for an image and configured with a one-shot encoder.
    pub fn new(
        writer: W,
        width: u32,
        height: u32,
        bands: u32,
        bytes_per_sample: usize,
        encode_fn: EncodeFn<W>,
    ) -> Result<Self, ViprsError> {
        let size = Self::checked_buffer_len(width, height, bands, bytes_per_sample)?;
        Ok(Self {
            writer,
            buffer: vec![0u8; size],
            width,
            height,
            bands,
            bytes_per_sample,
            encode_fn,
        })
    }

    fn validate_region(&self, region: Region) -> Result<(), ViprsError> {
        let end_x = i64::from(region.x) + i64::from(region.width);
        let end_y = i64::from(region.y) + i64::from(region.height);

        if region.x < 0
            || region.y < 0
            || end_x > i64::from(self.width)
            || end_y > i64::from(self.height)
        {
            return Err(ViprsError::Scheduler(format!(
                "writer sink region {region:?} is out of bounds for {}x{} image",
                self.width, self.height
            )));
        }

        Ok(())
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
                "writer sink region byte count exceeds addressable memory for {region:?}"
            )));
        };

        if data_len != expected {
            return Err(ViprsError::Scheduler(format!(
                "writer sink buffer length {data_len} does not match expected {expected} for {region:?}"
            )));
        }

        Ok(())
    }

    fn scatter_to_buffer(&mut self, region: Region, data: &[u8]) {
        let stride = self.width as usize * self.bands as usize * self.bytes_per_sample;
        let pixel_bytes = self.bands as usize * self.bytes_per_sample;
        let row_bytes = region.width as usize * pixel_bytes;

        for row in 0..region.height as usize {
            let src_start = row * row_bytes;
            let src_end = src_start + row_bytes;
            let dst_y = region.y as usize + row;
            let dst_x = region.x as usize;
            let dst_start = dst_y * stride + dst_x * pixel_bytes;
            let dst_end = dst_start + row_bytes;
            self.buffer[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }
    }
}

impl<W: Write + Send> ImageSink for WriterSink<W> {
    fn write_region(&mut self, region: Region, data: &[u8]) -> Result<(), ViprsError> {
        self.validate_region(region)?;
        self.validate_data_len(region, data.len())?;
        self.scatter_to_buffer(region, data);
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<(), ViprsError> {
        let Self {
            mut writer,
            buffer,
            width,
            height,
            bands,
            bytes_per_sample: _,
            encode_fn,
        } = *self;

        encode_fn(&buffer, width, height, bands, &mut writer)?;
        writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::WriterSink;
    use crate::{domain::error::ViprsError, domain::image::Region, ports::sink::ImageSink};
    use std::{
        io::{self, Write},
        sync::{Arc, Mutex},
    };

    #[derive(Clone, Default)]
    struct SharedVecWriter {
        written: Arc<Mutex<Vec<u8>>>,
    }

    impl SharedVecWriter {
        fn into_inner(self) -> Vec<u8> {
            self.written.lock().unwrap().clone()
        }
    }

    impl Write for SharedVecWriter {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.written.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn writer_sink_writes_tiles_row_by_row_on_finish() {
        let writer = SharedVecWriter::default();
        let written = writer.clone();
        let mut sink = WriterSink::new(
            writer,
            4,
            4,
            1,
            1,
            Box::new(|pixels, _, _, _, writer| {
                writer.write_all(pixels)?;
                Ok(())
            }),
        )
        .unwrap();

        sink.write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3, 4])
            .unwrap();
        sink.write_region(Region::new(2, 2, 2, 2), &[5u8, 6, 7, 8])
            .unwrap();

        Box::new(sink).finish().unwrap();

        assert_eq!(
            written.into_inner(),
            vec![1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8]
        );
    }

    #[test]
    fn writer_sink_finish_passes_dimensions_to_encoder() {
        let writer = SharedVecWriter::default();
        let dims = Arc::new(Mutex::new(None));
        let dims_written = Arc::clone(&dims);
        let sink = WriterSink::new(
            writer,
            3,
            2,
            4,
            1,
            Box::new(move |pixels, width, height, bands, _| {
                *dims_written.lock().unwrap() = Some((pixels.len(), width, height, bands));
                Ok(())
            }),
        )
        .unwrap();

        Box::new(sink).finish().unwrap();

        assert_eq!(*dims.lock().unwrap(), Some((24, 3, 2, 4)));
    }

    #[test]
    fn writer_sink_rejects_out_of_bounds_region() {
        let mut sink = WriterSink::new(
            SharedVecWriter::default(),
            4,
            4,
            1,
            1,
            Box::new(|_, _, _, _, _| Ok(())),
        )
        .unwrap();

        let err = sink
            .write_region(Region::new(-1, 0, 1, 1), &[1])
            .unwrap_err();

        assert!(matches!(err, ViprsError::Scheduler(message) if message.contains("out of bounds")));
    }

    #[test]
    fn writer_sink_rejects_undersized_tile_buffer() {
        let mut sink = WriterSink::new(
            SharedVecWriter::default(),
            4,
            4,
            1,
            1,
            Box::new(|_, _, _, _, _| Ok(())),
        )
        .unwrap();

        let err = sink
            .write_region(Region::new(0, 0, 2, 2), &[1u8, 2, 3])
            .unwrap_err();

        assert!(matches!(err, ViprsError::Scheduler(message) if message.contains("buffer length")));
    }

    #[test]
    fn writer_sink_new_rejects_images_that_exceed_addressable_memory() {
        let result = WriterSink::new(
            SharedVecWriter::default(),
            u32::MAX,
            u32::MAX,
            4,
            8,
            Box::new(|_, _, _, _, _| Ok(())),
        );

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }
}
