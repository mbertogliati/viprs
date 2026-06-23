//! Operations extracted from the viprs root crate.

pub mod convolution;
pub mod morphology;
pub mod resample;
pub mod structural;

#[cfg(test)]
pub(crate) mod test_support {
    use bytemuck::Pod;
    use viprs_core::{
        error::ViprsError,
        format::BandFormat,
        image::{ImageMetadata, Region},
    };

    pub(crate) struct TestMemorySource<F: BandFormat> {
        width: u32,
        height: u32,
        bands: u32,
        data: Vec<F::Sample>,
    }

    impl<F> TestMemorySource<F>
    where
        F: BandFormat,
        F::Sample: Pod,
    {
        pub(crate) fn new(
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
                    details: "test memory source dimensions exceed addressable memory",
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
            })
        }

        pub(crate) fn read_region(
            &self,
            region: Region,
            output: &mut [u8],
        ) -> Result<(), ViprsError> {
            let sample_size = std::mem::size_of::<F::Sample>();
            let pixel_bytes = self.bands as usize * sample_size;
            let expected = region
                .checked_pixel_count()
                .and_then(|pixels| pixels.checked_mul(self.bands as usize))
                .and_then(|samples| samples.checked_mul(sample_size))
                .ok_or_else(|| {
                    ViprsError::Scheduler(format!(
                        "output length overflow for region {region:?} with {} bands",
                        self.bands
                    ))
                })?;

            if output.len() < expected {
                return Err(ViprsError::Scheduler(format!(
                    "output length {} is smaller than expected {} for region {region:?}",
                    output.len(),
                    expected
                )));
            }
            if self.width == 0 || self.height == 0 {
                output[..expected].fill(0);
                return Ok(());
            }

            let image_bytes: &[u8] = bytemuck::cast_slice(&self.data);
            let row_bytes = self.width as usize * pixel_bytes;
            let dst_row_bytes = region.width as usize * pixel_bytes;
            let width_i64 = i64::from(self.width);
            let height_i64 = i64::from(self.height);

            for row in 0..region.height as usize {
                let dst_row_start = row * dst_row_bytes;
                let dst_row = &mut output[dst_row_start..dst_row_start + dst_row_bytes];
                let src_y = (i64::from(region.y) + row as i64).clamp(0, height_i64 - 1) as usize;

                for col in 0..region.width as usize {
                    let src_x = (i64::from(region.x) + col as i64).clamp(0, width_i64 - 1) as usize;
                    let src = src_y * row_bytes + src_x * pixel_bytes;
                    let dst = col * pixel_bytes;
                    dst_row[dst..dst + pixel_bytes]
                        .copy_from_slice(&image_bytes[src..src + pixel_bytes]);
                }
            }

            Ok(())
        }

        #[allow(dead_code)]
        pub(crate) fn metadata(&self) -> ImageMetadata {
            ImageMetadata::default()
        }
    }
}
