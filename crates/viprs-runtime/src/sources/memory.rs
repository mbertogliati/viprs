//! Memory image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use bytemuck::Pod;

use crate::{
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{DemandHint, ImageMetadata, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

/// An in-memory image source backed by a `Vec<F::Sample>`.
///
/// Coordinates outside image bounds are handled via clamp-to-edge extension.
pub struct MemorySource<F: BandFormat> {
    width: u32,
    height: u32,
    bands: u32,
    data: Vec<F::Sample>,
    metadata: ImageMetadata,
}

impl<F: BandFormat> MemorySource<F> {
    /// `new` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::memory::new;
    /// ```
    pub fn new(
        width: u32,
        height: u32,
        bands: u32,
        data: Vec<F::Sample>,
    ) -> Result<Self, ViprsError> {
        let expected: usize = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(u64::from(bands)))
            .and_then(|n| n.try_into().ok())
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width,
                height,
                bands,
                bytes: u128::from(width)
                    * u128::from(height)
                    * u128::from(bands)
                    * std::mem::size_of::<F::Sample>() as u128,
                limit_bytes: usize::MAX as u128,
                details: "memory source dimensions exceed addressable memory",
            })?;
        if data.len() != expected {
            return Err(ViprsError::RegionOutOfBounds {
                requested: format!("buffer length {} != expected {}", data.len(), expected),
                width,
                height,
            });
        }
        Ok(Self {
            width,
            height,
            bands,
            data,
            metadata: ImageMetadata::default(),
        })
    }

    #[must_use]
    /// `with_metadata` exposes adapter behavior needed by the surrounding module.
    /// Call it when you need the concrete operation implemented here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let _ = viprs::adapters::sources::memory::with_metadata;
    /// ```
    pub fn with_metadata(mut self, metadata: ImageMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    fn checked_region_end(&self, region: Region) -> Result<(i64, i64), ViprsError> {
        let end_x = i64::from(region.x) + i64::from(region.width);
        let end_y = i64::from(region.y) + i64::from(region.height);

        if end_x > i64::from(i32::MAX) || end_y > i64::from(i32::MAX) {
            return Err(ViprsError::RegionOutOfBounds {
                requested: format!("region {region:?} exceeds i32 addressable coordinates"),
                width: self.width,
                height: self.height,
            });
        }

        Ok((end_x, end_y))
    }

    fn validate_output_len(&self, region: Region, output_len: usize) -> Result<(), ViprsError> {
        let sample_size = std::mem::size_of::<F::Sample>();
        let expected = region
            .checked_pixel_count()
            .and_then(|pixels| pixels.checked_mul(self.bands as usize))
            .and_then(|samples| samples.checked_mul(sample_size))
            .ok_or_else(|| {
                ViprsError::Scheduler(format!(
                    "output length overflow for region {region:?} with {} bands and sample size {sample_size}",
                    self.bands
                ))
            })?;

        if output_len < expected {
            return Err(ViprsError::Scheduler(format!(
                "output length {output_len} is smaller than expected {expected} for region {region:?}"
            )));
        }

        Ok(())
    }
}

impl<F: BandFormat> ImageSource for MemorySource<F>
where
    F::Sample: Pod,
{
    type Format = F;

    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn bands(&self) -> u32 {
        self.bands
    }
    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn metadata(&self) -> ImageMetadata {
        self.metadata.clone()
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let bps = std::mem::size_of::<F::Sample>();
        let bands = self.bands as usize;
        let (end_x, end_y) = self.checked_region_end(region)?;
        self.validate_output_len(region, output.len())?;
        if self.width == 0 || self.height == 0 {
            output.fill(0);
            return Ok(());
        }
        let row_samples = region.width as usize * bands;
        let src_stride = self.width as usize * bands;

        let in_bounds = region.x >= 0
            && region.y >= 0
            && end_x <= i64::from(self.width)
            && end_y <= i64::from(self.height);

        if in_bounds {
            if region.x == 0 && region.width == self.width {
                let src_start = region.y as usize * src_stride;
                let src_end = src_start + region.height as usize * src_stride;
                let src_bytes: &[u8] = bytemuck::cast_slice(&self.data[src_start..src_end]);
                output.copy_from_slice(src_bytes);
                return Ok(());
            }

            for row in 0..region.height as usize {
                let src_row_start =
                    (region.y as usize + row) * src_stride + region.x as usize * bands;
                let src_row = &self.data[src_row_start..src_row_start + row_samples];
                let src_bytes: &[u8] = bytemuck::cast_slice(src_row);
                let dst_byte_start = row * row_samples * bps;
                output[dst_byte_start..dst_byte_start + row_samples * bps]
                    .copy_from_slice(src_bytes);
            }
            return Ok(());
        }

        let pixel_bytes = bands * bps;
        let row_bytes = self.width as usize * pixel_bytes;
        let image_bytes: &[u8] = bytemuck::cast_slice(&self.data);
        let dst_row_bytes = region.width as usize * pixel_bytes;
        let width_i64 = i64::from(self.width);
        let height_i64 = i64::from(self.height);

        for row in 0..region.height as usize {
            let dst_row_start = row * dst_row_bytes;
            let dst_row = &mut output[dst_row_start..dst_row_start + dst_row_bytes];

            let src_y = (i64::from(region.y) + row as i64).clamp(0, height_i64 - 1) as usize;
            let src_row_start = src_y * row_bytes;
            let src_row = &image_bytes[src_row_start..src_row_start + row_bytes];

            let src_x0 = i64::from(region.x).clamp(0, width_i64) as usize;
            let src_x1 = end_x.clamp(0, width_i64) as usize;
            let center_pixels = src_x1.saturating_sub(src_x0);
            let left_pad = if region.x < 0 {
                i64::from(region.x).unsigned_abs() as usize
            } else {
                0
            }
            .min(region.width as usize);
            let right_pad = region.width as usize - left_pad - center_pixels;

            let left_pixel = &src_row[..pixel_bytes];
            for pixel in 0..left_pad {
                let dst = pixel * pixel_bytes;
                dst_row[dst..dst + pixel_bytes].copy_from_slice(left_pixel);
            }

            if center_pixels > 0 {
                let src = src_x0 * pixel_bytes;
                let dst = left_pad * pixel_bytes;
                let len = center_pixels * pixel_bytes;
                dst_row[dst..dst + len].copy_from_slice(&src_row[src..src + len]);
            }

            let right_pixel_start = (self.width as usize - 1) * pixel_bytes;
            let right_pixel = &src_row[right_pixel_start..right_pixel_start + pixel_bytes];
            for pixel in 0..right_pad {
                let dst = (left_pad + center_pixels + pixel) * pixel_bytes;
                dst_row[dst..dst + pixel_bytes].copy_from_slice(right_pixel);
            }
        }
        Ok(())
    }

    fn borrow_region(&self, region: Region) -> Option<&[u8]> {
        let Ok((end_x, end_y)) = self.checked_region_end(region) else {
            return None;
        };
        let in_bounds = region.x >= 0
            && region.y >= 0
            && end_x <= i64::from(self.width)
            && end_y <= i64::from(self.height);
        if !in_bounds || region.x != 0 || region.width != self.width {
            return None;
        }

        let bands = self.bands as usize;
        let src_stride = self.width as usize * bands;
        let src_start = region.y as usize * src_stride;
        let src_end = src_start + region.height as usize * src_stride;
        Some(bytemuck::cast_slice(&self.data[src_start..src_end]))
    }
}

/// `MemorySource` holds the full pixel buffer in memory — every region can be
/// served in any order without constraint.
impl<F: BandFormat> RandomAccessSource for MemorySource<F> where F::Sample: bytemuck::Pod {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, image::Interpretation};

    fn make_4x4() -> MemorySource<U8> {
        // 4x4 single-band image; pixel at (x, y) = y * 4 + x (values 0..15)
        let data: Vec<u8> = (0u8..16).collect();
        MemorySource::new(4, 4, 1, data).unwrap()
    }

    #[test]
    fn memory_source_with_metadata_preserves_metadata() {
        let mut metadata = ImageMetadata::default();
        metadata.interpretation = Some(Interpretation::Srgb);

        let src = MemorySource::<U8>::new(1, 1, 1, vec![9u8])
            .unwrap()
            .with_metadata(metadata.clone());

        assert_eq!(src.metadata(), metadata);
    }

    #[test]
    fn memory_source_returns_correct_pixels() {
        let src = make_4x4();
        // Read a 2x2 central region starting at (1,1)
        let region = Region::new(1, 1, 2, 2);
        let mut output = vec![0u8; 4];
        src.read_region(region, &mut output).unwrap();
        // pixel (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10
        assert_eq!(output, vec![5, 6, 9, 10]);
    }

    #[test]
    fn memory_source_clamps_negative_region() {
        let src = make_4x4();
        // Region starts at (-2, -2) with size 2x2 — all coordinates clamp to (0,0)
        let region = Region::new(-2, -2, 2, 2);
        let mut output = vec![0u8; 4];
        src.read_region(region, &mut output).unwrap();
        // All four pixels should clamp to (0,0) => value 0
        assert_eq!(output, vec![0, 0, 0, 0]);
    }

    #[test]
    fn memory_source_rejects_short_output_buffer() {
        let src = make_4x4();
        let mut output = vec![0u8; 3];

        let err = src
            .read_region(Region::new(0, 0, 2, 2), &mut output)
            .unwrap_err();

        assert!(
            matches!(err, ViprsError::Scheduler(ref message) if message.contains("output length 3 is smaller than expected 4"))
        );
    }

    #[test]
    fn memory_source_accepts_exactly_sized_output_buffer() {
        let src = make_4x4();
        let mut output = vec![0u8; 4];

        src.read_region(Region::new(0, 0, 2, 2), &mut output)
            .unwrap();

        assert_eq!(output, vec![0, 1, 4, 5]);
    }

    #[test]
    fn memory_source_borrow_region_returns_full_width_strip() {
        let src = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
        let borrowed = src.borrow_region(Region::new(0, 1, 4, 2)).unwrap();
        assert_eq!(borrowed, &[4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn memory_source_borrow_region_rejects_partial_width_strip() {
        let src = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
        assert!(src.borrow_region(Region::new(1, 1, 2, 2)).is_none());
    }

    #[test]
    fn memory_source_rejects_regions_whose_x_end_overflows_i32() {
        let src = make_4x4();
        let mut output = vec![0u8; 1];

        let err = src
            .read_region(Region::new(i32::MAX, 0, 1, 1), &mut output)
            .unwrap_err();

        assert!(matches!(err, ViprsError::RegionOutOfBounds { .. }));
    }

    #[test]
    fn memory_source_zero_width_clamped_read_returns_zeroes() {
        let src = MemorySource::<U8>::new(0, 1, 1, Vec::new()).unwrap();
        let mut output = vec![255u8; 4];

        src.read_region(Region::new(-2, 0, 4, 1), &mut output)
            .unwrap();

        assert_eq!(output, vec![0u8; 4]);
    }

    #[test]
    fn memory_source_zero_height_clamped_read_returns_zeroes() {
        let src = MemorySource::<U8>::new(1, 0, 1, Vec::new()).unwrap();
        let mut output = vec![255u8; 4];

        src.read_region(Region::new(0, -2, 1, 4), &mut output)
            .unwrap();

        assert_eq!(output, vec![0u8; 4]);
    }

    #[test]
    fn memory_source_new_rejects_oversized_dimensions() {
        let result = MemorySource::<U8>::new(u32::MAX, u32::MAX, 2, Vec::new());

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }
}
