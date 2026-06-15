use crate::domain::{
    error::MosaicingError,
    error::ViprsError,
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Represents a tie point offset.
pub struct TiePointOffset {
    /// Stores the `dx` value for this item.
    pub dx: i32,
    /// Stores the `dy` value for this item.
    pub dy: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents a tie point match.
pub struct TiePointMatch {
    /// Stores the `offset` value for this item.
    pub offset: TiePointOffset,
    /// Stores the `score` value for this item.
    pub score: f64,
}

/// Applies the `tie points` mosaicing operation to related images. Use it when matching,
/// aligning, or merging overlapping image content.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::mosaicing::tie_points::TiePointSearchOp;
///
/// let op = TiePointSearchOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct TiePointSearchOp {
    search_radius: i32,
    minimum_overlap_pixels: usize,
}

#[inline]
fn checked_overlap_samples(overlap: Region, bands: u32) -> Result<usize, ViprsError> {
    overlap
        .checked_pixel_count()
        .and_then(|n| n.checked_mul(bands as usize))
        .ok_or_else(|| ViprsError::ImageTooLarge {
            width: overlap.width,
            height: overlap.height,
            bands,
            bytes: u128::from(overlap.width) * u128::from(overlap.height) * u128::from(bands),
            limit_bytes: usize::MAX as u128,
            details: "tie-point overlap sample count exceeds addressable memory",
        })
}

impl TiePointSearchOp {
    #[must_use]
    /// Creates a new `TiePointSearchOp`.
    pub const fn new(search_radius: u32) -> Self {
        Self {
            search_radius: search_radius as i32,
            minimum_overlap_pixels: 16,
        }
    }

    #[must_use]
    /// Returns this value configured with minimum overlap.
    pub fn with_minimum_overlap(mut self, minimum_overlap_pixels: usize) -> Self {
        self.minimum_overlap_pixels = minimum_overlap_pixels.max(1);
        self
    }

    /// Returns or performs search.
    pub fn search<F>(
        &self,
        reference: &Tile<F>,
        secondary: &Tile<F>,
        overlap_hint: Region,
    ) -> Result<TiePointMatch, ViprsError>
    where
        F: BandFormat,
        F::Sample: ToF64,
    {
        if reference.bands != secondary.bands {
            return Err(MosaicingError::BandCountMismatch {
                reference_bands: reference.bands,
                secondary_bands: secondary.bands,
            }
            .into());
        }

        let mut best: Option<TiePointMatch> = None;
        for dy in -self.search_radius..=self.search_radius {
            for dx in -self.search_radius..=self.search_radius {
                let shifted_secondary = Region::new(
                    secondary.region.x + dx,
                    secondary.region.y + dy,
                    secondary.region.width,
                    secondary.region.height,
                );
                let Some(candidate) = intersect_regions(overlap_hint, reference.region)
                    .and_then(|region| intersect_regions(region, shifted_secondary))
                else {
                    continue;
                };
                if candidate.pixel_count() < self.minimum_overlap_pixels {
                    continue;
                }

                let Some(score) = correlation_score(reference, secondary, candidate, dx, dy)?
                else {
                    continue;
                };
                let candidate_match = TiePointMatch {
                    offset: TiePointOffset { dx, dy },
                    score,
                };
                match best {
                    Some(current)
                        if current.score > candidate_match.score
                            || ((current.score - candidate_match.score).abs() <= f64::EPSILON
                                && manhattan(current.offset)
                                    <= manhattan(candidate_match.offset)) => {}
                    _ => best = Some(candidate_match),
                }
            }
        }

        best.ok_or_else(|| MosaicingError::NoValidOverlapWindow.into())
    }
}

fn correlation_score<F>(
    reference: &Tile<F>,
    secondary: &Tile<F>,
    overlap: Region,
    dx: i32,
    dy: i32,
) -> Result<Option<f64>, ViprsError>
where
    F: BandFormat,
    F::Sample: ToF64,
{
    let bands = reference.bands as usize;
    let samples = checked_overlap_samples(overlap, reference.bands)?;
    if samples == 0 {
        return Ok(None);
    }

    let ref_width = reference.region.width as usize;
    let sec_width = secondary.region.width as usize;
    let mut ref_sum = 0.0f64;
    let mut sec_sum = 0.0f64;

    for row in 0..overlap.height as usize {
        let global_y = overlap.y + row as i32;
        let Some(ref_y) = usize::try_from(global_y - reference.region.y).ok() else {
            return Ok(None);
        };
        let Some(sec_y) = usize::try_from(global_y - (secondary.region.y + dy)).ok() else {
            return Ok(None);
        };
        for col in 0..overlap.width as usize {
            let global_x = overlap.x + col as i32;
            let Some(ref_x) = usize::try_from(global_x - reference.region.x).ok() else {
                return Ok(None);
            };
            let Some(sec_x) = usize::try_from(global_x - (secondary.region.x + dx)).ok() else {
                return Ok(None);
            };
            let ref_base = (ref_y * ref_width + ref_x) * bands;
            let sec_base = (sec_y * sec_width + sec_x) * bands;
            for band in 0..bands {
                ref_sum += reference.data[ref_base + band].to_f64();
                sec_sum += secondary.data[sec_base + band].to_f64();
            }
        }
    }

    let ref_mean = ref_sum / samples as f64;
    let sec_mean = sec_sum / samples as f64;
    let mut covariance = 0.0f64;
    let mut ref_variance = 0.0f64;
    let mut sec_variance = 0.0f64;
    let mut squared_error = 0.0f64;

    for row in 0..overlap.height as usize {
        let global_y = overlap.y + row as i32;
        let Some(ref_y) = usize::try_from(global_y - reference.region.y).ok() else {
            return Ok(None);
        };
        let Some(sec_y) = usize::try_from(global_y - (secondary.region.y + dy)).ok() else {
            return Ok(None);
        };
        for col in 0..overlap.width as usize {
            let global_x = overlap.x + col as i32;
            let Some(ref_x) = usize::try_from(global_x - reference.region.x).ok() else {
                return Ok(None);
            };
            let Some(sec_x) = usize::try_from(global_x - (secondary.region.x + dx)).ok() else {
                return Ok(None);
            };
            let ref_base = (ref_y * ref_width + ref_x) * bands;
            let sec_base = (sec_y * sec_width + sec_x) * bands;
            for band in 0..bands {
                let ref_centered = reference.data[ref_base + band].to_f64() - ref_mean;
                let sec_centered = secondary.data[sec_base + band].to_f64() - sec_mean;
                covariance = ref_centered.mul_add(sec_centered, covariance);
                ref_variance = ref_centered.mul_add(ref_centered, ref_variance);
                sec_variance = sec_centered.mul_add(sec_centered, sec_variance);
                let diff = reference.data[ref_base + band].to_f64()
                    - secondary.data[sec_base + band].to_f64();
                squared_error = diff.mul_add(diff, squared_error);
            }
        }
    }

    let denominator = (ref_variance * sec_variance).sqrt();
    if denominator <= f64::EPSILON {
        let mean_squared_error = squared_error / samples as f64;
        Ok(Some(1.0 / (1.0 + mean_squared_error)))
    } else {
        Ok(Some(covariance / denominator))
    }
}

fn intersect_regions(lhs: Region, rhs: Region) -> Option<Region> {
    let x0 = i64::from(lhs.x).max(i64::from(rhs.x));
    let y0 = i64::from(lhs.y).max(i64::from(rhs.y));
    let x1 = (i64::from(lhs.x) + i64::from(lhs.width)).min(i64::from(rhs.x) + i64::from(rhs.width));
    let y1 =
        (i64::from(lhs.y) + i64::from(lhs.height)).min(i64::from(rhs.y) + i64::from(rhs.height));
    if x1 <= x0 || y1 <= y0 {
        None
    } else {
        Some(Region::new(
            i32::try_from(x0).ok()?,
            i32::try_from(y0).ok()?,
            u32::try_from(x1 - x0).ok()?,
            u32::try_from(y1 - y0).ok()?,
        ))
    }
}

const fn manhattan(offset: TiePointOffset) -> i32 {
    offset.dx.abs() + offset.dy.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{error::ViprsError, format::U8};
    use proptest::prelude::*;

    fn patterned(width: usize, height: usize, seed: u8) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let value = (x * x * 17 + y * 29 + x * y * 13 + usize::from(seed)) % 251;
                pixels.push((value + 1) as u8);
            }
        }
        pixels
    }

    #[test]
    fn finds_known_horizontal_shift() {
        let data = patterned(5, 5, 7);
        let reference = Tile::<U8>::new(Region::new(0, 0, 5, 5), 1, &data);
        let secondary = Tile::<U8>::new(Region::new(1, 0, 5, 5), 1, &data);
        let overlap = Region::new(0, 0, 5, 5);
        let search = TiePointSearchOp::new(2);

        let result = search.search(&reference, &secondary, overlap).unwrap();

        assert_eq!(result.offset, TiePointOffset { dx: -1, dy: 0 });
        assert!(result.score > 0.99);
    }

    #[test]
    fn finds_known_vertical_shift() {
        let data = patterned(4, 4, 19);
        let reference = Tile::<U8>::new(Region::new(0, 0, 4, 4), 1, &data);
        let secondary = Tile::<U8>::new(Region::new(0, 1, 4, 4), 1, &data);
        let overlap = Region::new(0, 0, 4, 4);
        let search = TiePointSearchOp::new(2).with_minimum_overlap(4);

        let result = search.search(&reference, &secondary, overlap).unwrap();

        assert_eq!(result.offset, TiePointOffset { dx: 0, dy: -1 });
        assert!(result.score > 0.99);
    }

    proptest! {
        #[test]
        fn identical_tiles_with_zero_offset_keep_origin(seed in 0u8..=250) {
            let values = patterned(5, 5, seed);
            let region = Region::new(0, 0, 5, 5);
            let reference = Tile::<U8>::new(region, 1, &values);
            let secondary = Tile::<U8>::new(region, 1, &values);
            let search = TiePointSearchOp::new(2);
            let result = search.search(&reference, &secondary, region).unwrap();

            prop_assert_eq!(result.offset, TiePointOffset { dx: 0, dy: 0 });
            prop_assert!(result.score > 0.99);
        }

        #[test]
        fn exact_copy_prefers_inverse_region_shift(seed in 0u8..=250) {
            let values = patterned(5, 5, seed);
            let reference = Tile::<U8>::new(Region::new(0, 0, 5, 5), 1, &values);
            let secondary = Tile::<U8>::new(Region::new(1, 0, 5, 5), 1, &values);
            let overlap = Region::new(0, 0, 5, 5);
            let search = TiePointSearchOp::new(2);
            let result = search.search(&reference, &secondary, overlap).unwrap();
            prop_assert_eq!(result.offset, TiePointOffset { dx: -1, dy: 0 });
        }
    }

    #[test]
    fn checked_overlap_samples_rejects_overflow() {
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);

        let err = checked_overlap_samples(huge, 2).unwrap_err();

        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: 2,
                ..
            }
        ));
    }

    #[test]
    fn intersect_regions_handles_large_positive_origins_without_panicking() {
        let lhs = Region::new(i32::MAX - 2, i32::MAX - 2, 8, 8);
        let rhs = Region::new(i32::MAX - 1, i32::MAX - 1, 1, 1);

        let overlap = intersect_regions(lhs, rhs);

        assert_eq!(overlap, Some(Region::new(i32::MAX - 1, i32::MAX - 1, 1, 1)));
    }
}
