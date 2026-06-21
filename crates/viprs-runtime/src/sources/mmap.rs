//! Mmap image source adapter.
//!
//! This module exposes concrete source implementations or helpers that feed
//! pixels into compiled pipelines.

use std::{fs::File, marker::PhantomData, path::Path};

use bytemuck::Pod;
use memmap2::Mmap;

use crate::{
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{DemandHint, Region},
    },
    ports::source::{ImageSource, RandomAccessSource},
};

/// A file-backed image source using memory-mapped I/O.
///
/// Maps a raw pixel file into the process address space. The file must contain
/// exactly `width * height * bands * sizeof(F::Sample)` bytes of raw, unencoded
/// pixel data with no header. Region reads are zero-copy: the OS page cache serves
/// pixels directly from mapped pages.
///
/// Coordinates outside image bounds are handled via clamp-to-edge extension,
/// matching the contract in `ImageSource::read_region`.
///
/// Rationale behind using `memmap2` and the single `unsafe`
/// block in `MmapSource::open`.
pub struct MmapSource<F: BandFormat> {
    /// Owned file handle — kept alive so the OS does not reclaim the file
    /// for the lifetime of the mapping. On Unix, an open fd prevents deletion;
    /// on Windows, an open handle prevents modification.
    _file: File,
    mmap: Mmap,
    width: u32,
    height: u32,
    bands: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> MmapSource<F>
where
    F::Sample: Pod,
{
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

    /// Opens a raw pixel file and maps it into memory.
    ///
    /// `width`, `height`, and `bands` describe the image layout. The file must
    /// contain exactly `width * height * bands * size_of::<F::Sample>()` bytes.
    /// Returns `ViprsError::RegionOutOfBounds` if the file size does not match.
    pub fn open(path: &Path, width: u32, height: u32, bands: u32) -> Result<Self, ViprsError> {
        let file = File::open(path)?;
        let expected_bytes: usize = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| n.checked_mul(u64::from(bands)))
            .and_then(|n| n.checked_mul(std::mem::size_of::<F::Sample>() as u64))
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
                details: "mmap source dimensions exceed addressable memory",
            })?;

        let file_len = file.metadata()?.len() as usize;
        if file_len != expected_bytes {
            return Err(ViprsError::RegionOutOfBounds {
                requested: format!(
                    "file size {} bytes != expected {} bytes ({}x{}x{}x{})",
                    file_len,
                    expected_bytes,
                    width,
                    height,
                    bands,
                    std::mem::size_of::<F::Sample>(),
                ),
                width,
                height,
            });
        }

        // SAFETY: the file stays open for the life of the mapping, the mapping is read-only, and callers must not concurrently mutate or truncate the file from outside the process.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            _file: file,
            mmap,
            width,
            height,
            bands,
            _format: PhantomData,
        })
    }
}

impl<F: BandFormat> ImageSource for MmapSource<F>
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
        // mmap lets the OS decide which pages to bring in — SmallTile maximises
        // page-cache locality for random-access patterns.
        DemandHint::SmallTile
    }

    fn read_region(&self, region: Region, output: &mut [u8]) -> Result<(), ViprsError> {
        let bps = std::mem::size_of::<F::Sample>();
        let bands = self.bands as usize;
        let _ = self.checked_region_end(region)?;
        self.validate_output_len(region, output.len())?;
        if self.width == 0 || self.height == 0 {
            output.fill(0);
            return Ok(());
        }
        let width_i64 = i64::from(self.width);
        let height_i64 = i64::from(self.height);

        for row in 0..region.height as usize {
            for col in 0..region.width as usize {
                let src_x = (i64::from(region.x) + col as i64).clamp(0, width_i64 - 1) as usize;
                let src_y = (i64::from(region.y) + row as i64).clamp(0, height_i64 - 1) as usize;

                let src_pixel_idx = src_y * self.width as usize + src_x;
                let src_byte_start = src_pixel_idx * bands * bps;
                let src_bytes = &self.mmap[src_byte_start..src_byte_start + bands * bps];

                let dst_pixel_idx = row * region.width as usize + col;
                let dst_byte_start = dst_pixel_idx * bands * bps;
                output[dst_byte_start..dst_byte_start + bands * bps].copy_from_slice(src_bytes);
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

        let row_bytes =
            self.width as usize * self.bands as usize * std::mem::size_of::<F::Sample>();
        let start = region.y as usize * row_bytes;
        let end = start + region.height as usize * row_bytes;
        Some(&self.mmap[start..end])
    }
}

/// `MmapSource` is backed by the OS page cache — every region is accessible
/// in any order without restriction.
impl<F: BandFormat> RandomAccessSource for MmapSource<F> where F::Sample: bytemuck::Pod {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::format::U8;
    use std::{
        io::Write,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT_TEST_FILE_ID: AtomicUsize = AtomicUsize::new(0);

    struct TestRawFile {
        path: PathBuf,
    }

    impl TestRawFile {
        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestRawFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Write `data` to a uniquely named test artifact file and return its handle.
    fn write_raw_file(name: &str, data: &[u8]) -> TestRawFile {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(".test-artifacts")
            .join("mmap-source");
        std::fs::create_dir_all(&dir).unwrap();
        let file_id = NEXT_TEST_FILE_ID.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{file_id}_{name}"));
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
        TestRawFile { path }
    }

    fn make_4x4_file(name: &str) -> TestRawFile {
        // 4x4 single-band raw image; pixel at (x, y) = y * 4 + x (values 0..15)
        let data: Vec<u8> = (0u8..16).collect();
        write_raw_file(name, &data)
    }

    #[test]
    fn mmap_source_open_succeeds_for_correct_file() {
        let path = make_4x4_file("viprs_mmap_test_open_ok.raw");
        let result = MmapSource::<U8>::open(path.path(), 4, 4, 1);
        assert!(result.is_ok(), "expected Ok");
    }

    #[test]
    fn mmap_source_open_fails_for_wrong_size() {
        let path = write_raw_file("viprs_mmap_test_wrong_size.raw", &[0u8; 10]);
        let result = MmapSource::<U8>::open(path.path(), 4, 4, 1);
        assert!(
            result.is_err(),
            "expected Err for file size mismatch, got Ok"
        );
    }

    #[test]
    fn mmap_source_returns_correct_pixels() {
        let path = make_4x4_file("viprs_mmap_test_pixels.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();

        let region = Region::new(1, 1, 2, 2);
        let mut output = vec![0u8; 4];
        src.read_region(region, &mut output).unwrap();
        // pixel (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10
        assert_eq!(output, vec![5, 6, 9, 10]);
    }

    #[test]
    fn mmap_source_clamps_negative_region() {
        let path = make_4x4_file("viprs_mmap_test_clamp_neg.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();

        // Region starts at (-2, -2) with size 2x2 — all coordinates clamp to (0,0)
        let region = Region::new(-2, -2, 2, 2);
        let mut output = vec![0xffu8; 4];
        src.read_region(region, &mut output).unwrap();
        // All four pixels should clamp to (0,0) => value 0
        assert_eq!(output, vec![0, 0, 0, 0]);
    }

    #[test]
    fn mmap_source_rejects_short_output_buffer() {
        let path = make_4x4_file("viprs_mmap_test_short_output.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();
        let mut output = vec![0u8; 3];

        let err = src
            .read_region(Region::new(0, 0, 2, 2), &mut output)
            .unwrap_err();

        assert!(
            matches!(err, ViprsError::Scheduler(ref message) if message.contains("output length 3 is smaller than expected 4"))
        );
    }

    #[test]
    fn mmap_source_accepts_exactly_sized_output_buffer() {
        let path = make_4x4_file("viprs_mmap_test_exact_output.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();
        let mut output = vec![0u8; 4];

        src.read_region(Region::new(0, 0, 2, 2), &mut output)
            .unwrap();

        assert_eq!(output, vec![0, 1, 4, 5]);
    }

    #[test]
    fn mmap_source_clamps_beyond_right_bottom_edge() {
        let path = make_4x4_file("viprs_mmap_test_clamp_edge.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();

        // Region starts at (3, 3) with size 2x2 — right/bottom pixels clamp to (3,3)
        let region = Region::new(3, 3, 2, 2);
        let mut output = vec![0u8; 4];
        src.read_region(region, &mut output).unwrap();
        // pixel (3,3)=15; all four should be 15
        assert_eq!(output, vec![15, 15, 15, 15]);
    }

    #[test]
    fn mmap_source_dimensions_match_constructor() {
        let path = make_4x4_file("viprs_mmap_test_dims.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();
        assert_eq!(ImageSource::width(&src), 4);
        assert_eq!(ImageSource::height(&src), 4);
        assert_eq!(ImageSource::bands(&src), 1);
    }

    #[test]
    fn mmap_source_rejects_regions_whose_x_end_overflows_i32() {
        let path = make_4x4_file("viprs_mmap_test_x_overflow.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();
        let mut output = vec![0u8; 1];

        let err = src
            .read_region(Region::new(i32::MAX, 0, 1, 1), &mut output)
            .unwrap_err();

        assert!(matches!(err, ViprsError::RegionOutOfBounds { .. }));
    }

    #[test]
    fn mmap_source_demand_hint_is_small_tile() {
        let path = make_4x4_file("viprs_mmap_test_hint.raw");
        let src = MmapSource::<U8>::open(path.path(), 4, 4, 1).unwrap();
        assert_eq!(ImageSource::demand_hint(&src), DemandHint::SmallTile);
    }

    #[test]
    fn mmap_source_open_rejects_oversized_dimensions() {
        let path = write_raw_file("viprs_mmap_test_oversized.raw", &[]);
        let result = MmapSource::<U8>::open(path.path(), u32::MAX, u32::MAX, 2);

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }
}
