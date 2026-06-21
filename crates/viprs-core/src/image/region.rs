use crate::{error::ViprsError, format::BandFormat};

use super::core::checked_image_buffer_len;

/// An axis-aligned rectangle in image coordinates.
///
/// Regions let operations describe tile demand, halo expansion, and clipping without
/// materializing full images.
///
/// # Examples
/// ```rust
/// # use viprs_core::image::Region;
/// let region = Region::new(-1, 2, 3, 4);
/// assert_eq!(region.width, 3);
/// assert_eq!(region.height, 4);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Horizontal factor associated with this condition.
    pub x: i32,
    /// Vertical factor associated with this condition.
    pub y: i32,
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
}

/// Clamp a wide signed coordinate into the `i32` range used by image regions.
#[must_use]
pub const fn clamp_i64_to_i32(value: i64) -> i32 {
    if value < i32::MIN as i64 {
        i32::MIN
    } else if value > i32::MAX as i64 {
        i32::MAX
    } else {
        value as i32
    }
}

impl Region {
    /// Create a rectangle with explicit origin and size.
    ///
    /// This keeps region construction explicit when mapping output demand back to source
    /// coordinates.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::Region;
    /// let region = Region::new(1, 2, 5, 6);
    /// assert_eq!(region.x, 1);
    /// assert_eq!(region.y, 2);
    /// ```
    #[must_use]
    pub const fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Return the number of pixels covered by the region.
    ///
    /// This is useful when sizing tile buffers from geometry alone.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::Region;
    /// let region = Region::new(0, 0, 4, 3);
    /// assert_eq!(region.pixel_count(), 12);
    /// ```
    ///
    /// # Panics
    ///
    /// Debug builds assert if the region does not fit in addressable memory.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        match checked_region_pixel_count(*self) {
            Ok(pixel_count) => pixel_count,
            Err(error) => {
                debug_assert!(false, "region pixel count overflow for {self:?}: {error}");
                usize::MAX
            }
        }
    }

    /// Return the pixel count if it fits in addressable memory.
    ///
    /// This lets callers validate huge regions without panicking on overflow.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::Region;
    /// let region = Region::new(0, 0, 2, 2);
    /// assert_eq!(region.checked_pixel_count(), Some(4));
    /// ```
    #[must_use]
    pub fn checked_pixel_count(&self) -> Option<usize> {
        checked_region_pixel_count(*self).ok()
    }

    /// Expand the region outward by the given amounts on each side.
    /// top/bottom expand height, left/right expand width.
    /// x,y shift by -left and -top respectively (can go negative).
    #[must_use]
    pub fn expand(&self, top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            x: clamp_i64_to_i32(i64::from(self.x) - i64::from(left)),
            y: clamp_i64_to_i32(i64::from(self.y) - i64::from(top)),
            width: self.width.saturating_add(left).saturating_add(right),
            height: self.height.saturating_add(top).saturating_add(bottom),
        }
    }

    /// Clip this region to image bounds [0, `image_width`) x [0, `image_height`).
    #[must_use]
    pub fn clip_to(&self, image_width: u32, image_height: u32) -> Self {
        let image_width = i64::from(image_width);
        let image_height = i64::from(image_height);
        let x0 = i64::from(self.x).clamp(0, image_width);
        let y0 = i64::from(self.y).clamp(0, image_height);
        let x1 = (i64::from(self.x) + i64::from(self.width)).clamp(0, image_width);
        let y1 = (i64::from(self.y) + i64::from(self.height)).clamp(0, image_height);
        let width = (x1 - x0) as u32;
        let height = (y1 - y0) as u32;
        Self {
            x: x0 as i32,
            y: y0 as i32,
            width,
            height,
        }
    }

    /// Return `true` when the region has no addressable pixels.
    ///
    /// Empty regions allow schedulers to short-circuit work without special sentinel types.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::image::Region;
    /// assert!(Region::new(0, 0, 0, 5).is_empty());
    /// ```
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// An immutable view into a rectangular tile of pixel data.
///
/// Invariant: `data.len() == region.pixel_count() * bands as usize`
pub struct Tile<'a, F: BandFormat> {
    /// Stores the `region` value for this item.
    pub region: Region,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `data` value for this item.
    pub data: &'a [F::Sample],
}

impl<'a, F: BandFormat> Tile<'a, F> {
    /// Build an immutable typed tile view over already-packed pixel data.
    ///
    /// This validates that geometry, band count, and slice length agree before a kernel reads
    /// the tile.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::{Region, Tile}};
    /// let data = [1_u8, 2, 3, 4];
    /// let tile = Tile::<U8>::new(Region::new(0, 0, 2, 2), 1, &data);
    /// assert_eq!(tile.bands, 1);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the region/band geometry overflows or if `data.len()` does not match the
    /// packed tile shape.
    pub fn new(region: Region, bands: u32, data: &'a [F::Sample]) -> Self {
        let expected_len = match checked_tile_data_len(region, bands) {
            Ok(expected_len) => expected_len,
            Err(error) => {
                debug_assert!(
                    false,
                    "tile shape overflow for {region:?} with {bands} bands: {error}"
                );
                data.len()
            }
        };
        assert_eq!(
            data.len(),
            expected_len,
            "tile data length mismatch for {region:?} with {bands} bands"
        );
        Self {
            region,
            bands,
            data,
        }
    }
}

/// A mutable view into a rectangular tile of pixel data.
///
/// Invariant: `data.len() == region.pixel_count() * bands as usize`
pub struct TileMut<'a, F: BandFormat> {
    /// Stores the `region` value for this item.
    pub region: Region,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `data` value for this item.
    pub data: &'a mut [F::Sample],
}

impl<'a, F: BandFormat> TileMut<'a, F> {
    /// Build a mutable typed tile view over already-packed pixel data.
    ///
    /// This gives in-place operations a checked buffer contract before they write samples.
    ///
    /// # Examples
    /// ```rust
    /// # use viprs_core::{format::U8, image::{Region, TileMut}};
    /// let mut data = [0_u8; 4];
    /// let tile = TileMut::<U8>::new(Region::new(0, 0, 2, 2), 1, &mut data);
    /// assert_eq!(tile.region.width, 2);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the region/band geometry overflows or if `data.len()` does not match the
    /// packed tile shape.
    pub fn new(region: Region, bands: u32, data: &'a mut [F::Sample]) -> Self {
        let expected_len = match checked_tile_data_len(region, bands) {
            Ok(expected_len) => expected_len,
            Err(error) => {
                debug_assert!(
                    false,
                    "tile shape overflow for {region:?} with {bands} bands: {error}"
                );
                data.len()
            }
        };
        assert_eq!(
            data.len(),
            expected_len,
            "tile data length mismatch for {region:?} with {bands} bands"
        );
        Self {
            region,
            bands,
            data,
        }
    }
}

fn checked_region_pixel_count(region: Region) -> Result<usize, ViprsError> {
    let Some(pixel_count) = u64::from(region.width).checked_mul(u64::from(region.height)) else {
        let total_pixels = u128::from(region.width) * u128::from(region.height);
        return Err(ViprsError::ImageTooLarge {
            width: region.width,
            height: region.height,
            bands: 1,
            bytes: total_pixels,
            limit_bytes: usize::MAX as u128,
            details: "region pixel count exceeds addressable memory",
        });
    };

    usize::try_from(pixel_count).map_err(|_| ViprsError::ImageTooLarge {
        width: region.width,
        height: region.height,
        bands: 1,
        bytes: u128::from(pixel_count),
        limit_bytes: usize::MAX as u128,
        details: "region pixel count exceeds addressable memory",
    })
}

fn checked_tile_data_len(region: Region, bands: u32) -> Result<usize, ViprsError> {
    checked_image_buffer_len(region.width, region.height, bands)
}
