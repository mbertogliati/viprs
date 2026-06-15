//! Reducers for connected-component labeling on thresholded images.

use std::collections::HashMap;

use crate::domain::{
    error::ViprsError,
    format::U8,
    image::{Image, Region, Tile},
    reducer::TileReducer,
};

/// Labels connected regions whose samples exceed a threshold.
///
/// This reducer solves connected-component extraction for `U8` tiles by collecting active pixel
/// positions and assigning stable region ids in the final pass.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::LabelRegionsReducer,
/// };
///
/// let reducer = LabelRegionsReducer { threshold: 0 };
/// let region = Region::new(0, 0, 2, 2);
/// let tile = Tile::<U8>::new(region, 1, &[1, 0, 0, 1]);
/// let labels = reducer.finalize(reducer.reduce_tile(&tile, &region)).unwrap();
///
/// assert_eq!(labels.width(), 2);
/// ```
pub struct LabelRegionsReducer {
    /// Stores the `threshold` value for this item.
    pub threshold: u8,
}

/// Stores the per-tile active-pixel positions used during region labeling.
///
/// This partial result solves cross-tile labeling by carrying the coordinates and bounds needed
/// to merge tile-local discoveries before the final connected-component pass.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::LabelRegionsReducer,
/// };
///
/// let reducer = LabelRegionsReducer { threshold: 0 };
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 1, &[1]);
/// let partial = reducer.reduce_tile(&tile, &region);
///
/// let _ = partial;
/// ```
pub struct TileLabelsPartial {
    max_width: u32,
    max_height: u32,
    active_positions: Vec<(u32, u32)>,
    error: Option<ViprsError>,
}

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, idx: usize) -> usize {
        if self.parent[idx] != idx {
            let root = self.find(self.parent[idx]);
            self.parent[idx] = root;
        }
        self.parent[idx]
    }

    fn union(&mut self, lhs: usize, rhs: usize) {
        let lhs_root = self.find(lhs);
        let rhs_root = self.find(rhs);
        if lhs_root == rhs_root {
            return;
        }

        match self.rank[lhs_root].cmp(&self.rank[rhs_root]) {
            std::cmp::Ordering::Less => self.parent[lhs_root] = rhs_root,
            std::cmp::Ordering::Greater => self.parent[rhs_root] = lhs_root,
            std::cmp::Ordering::Equal => {
                self.parent[rhs_root] = lhs_root;
                self.rank[lhs_root] += 1;
            }
        }
    }
}

impl TileReducer<U8> for LabelRegionsReducer {
    type Partial = TileLabelsPartial;
    type Output = Result<Image<crate::domain::format::U32>, ViprsError>;
    /// Per-thread reusable list of active pixel positions for one tile.
    type Scratch = Vec<(u32, u32)>;

    fn reduce_tile(&self, tile: &Tile<U8>, region: &Region) -> Self::Partial {
        let mut active_positions = Vec::new();
        match self.collect_active_positions(tile, region, &mut active_positions) {
            Ok(()) => match Self::checked_region_extents(region) {
                Ok((max_width, max_height)) => TileLabelsPartial {
                    max_width,
                    max_height,
                    active_positions,
                    error: None,
                },
                Err(error) => Self::partial_from_error(region, error),
            },
            Err(error) => Self::partial_from_error(region, error),
        }
    }

    fn accumulate_into(
        &self,
        tile: &Tile<U8>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        if let Err(error) = self.collect_active_positions(tile, region, scratch) {
            let tile_partial = Self::partial_from_error(region, error);
            *partial = Some(match partial.take() {
                Some(existing) => self.combine(existing, tile_partial),
                None => tile_partial,
            });
            return;
        }
        let (tile_max_width, tile_max_height) = match Self::checked_region_extents(region) {
            Ok(extents) => extents,
            Err(error) => {
                let tile_partial = Self::partial_from_error(region, error);
                *partial = Some(match partial.take() {
                    Some(existing) => self.combine(existing, tile_partial),
                    None => tile_partial,
                });
                return;
            }
        };

        match partial {
            Some(existing) => {
                existing.max_width = existing.max_width.max(tile_max_width);
                existing.max_height = existing.max_height.max(tile_max_height);
                if existing.error.is_some() {
                    return;
                }
                existing.active_positions.reserve(scratch.len());
                existing.active_positions.extend_from_slice(scratch);
            }
            None => {
                *partial = Some(TileLabelsPartial {
                    max_width: tile_max_width,
                    max_height: tile_max_height,
                    active_positions: scratch.clone(),
                    error: None,
                });
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, mut b: Self::Partial) -> Self::Partial {
        a.max_width = a.max_width.max(b.max_width);
        a.max_height = a.max_height.max(b.max_height);
        if a.error.is_none() {
            a.error = b.error.take();
        }
        if a.error.is_some() {
            return a;
        }
        a.active_positions.append(&mut b.active_positions);
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        if let Some(error) = combined.error {
            return Err(error);
        }
        let width = combined.max_width;
        let height = combined.max_height;
        let pixel_count = Self::checked_pixel_count(
            width,
            height,
            "labelregions final image exceeds addressable memory",
        )?;
        // u32 dimensions fit into usize on Viprs' supported 32/64-bit targets.
        let width_usize = width as usize;
        let height_usize = height as usize;
        let mut active = vec![false; pixel_count];
        for (x, y) in combined.active_positions {
            active[y as usize * width_usize + x as usize] = true;
        }

        let mut union_find = UnionFind::new(pixel_count);
        for y in 0..height_usize {
            for x in 0..width_usize {
                let idx = y * width_usize + x;
                if !active[idx] {
                    continue;
                }
                if x + 1 < width_usize && active[idx + 1] {
                    union_find.union(idx, idx + 1);
                }
                if y + 1 < height_usize && active[idx + width_usize] {
                    union_find.union(idx, idx + width_usize);
                }
            }
        }

        let mut root_to_label = HashMap::new();
        let mut next_label = 1u32;
        let mut labels = vec![0u32; pixel_count];
        for (idx, is_active) in active.iter().copied().enumerate() {
            if !is_active {
                continue;
            }
            let root = union_find.find(idx);
            let label = root_to_label.entry(root).or_insert_with(|| {
                let current = next_label;
                next_label += 1;
                current
            });
            labels[idx] = *label;
        }

        Image::from_buffer(width, height, 1, labels)
    }
}

impl LabelRegionsReducer {
    #[inline]
    fn image_too_large(width: u32, height: u32, details: &'static str) -> ViprsError {
        ViprsError::ImageTooLarge {
            width,
            height,
            bands: 1,
            bytes: u128::from(width) * u128::from(height),
            limit_bytes: usize::MAX as u128,
            details,
        }
    }

    #[inline]
    fn checked_pixel_count(
        width: u32,
        height: u32,
        details: &'static str,
    ) -> Result<usize, ViprsError> {
        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Self::image_too_large(width, height, details))?;

        std::alloc::Layout::array::<usize>(pixel_count)
            .map(|_| pixel_count)
            .map_err(|_| Self::image_too_large(width, height, details))
    }

    #[inline]
    fn checked_position_capacity(
        width: u32,
        height: u32,
        details: &'static str,
    ) -> Result<usize, ViprsError> {
        let pixel_count = u64::from(width)
            .checked_mul(u64::from(height))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Self::image_too_large(width, height, details))?;

        std::alloc::Layout::array::<(u32, u32)>(pixel_count)
            .map(|_| pixel_count)
            .map_err(|_| Self::image_too_large(width, height, details))
    }

    #[inline]
    fn checked_region_extents(region: &Region) -> Result<(u32, u32), ViprsError> {
        let width = i64::from(region.x)
            .saturating_add(i64::from(region.width))
            .max(0);
        let width = u32::try_from(width).map_err(|_| {
            Self::image_too_large(
                region.width,
                region.height,
                "labelregions final image extent exceeds addressable coordinates",
            )
        })?;
        let height = i64::from(region.y)
            .saturating_add(i64::from(region.height))
            .max(0);
        let height = u32::try_from(height).map_err(|_| {
            Self::image_too_large(
                region.width,
                region.height,
                "labelregions final image extent exceeds addressable coordinates",
            )
        })?;
        Ok((width, height))
    }

    #[inline]
    fn partial_from_error(region: &Region, error: ViprsError) -> TileLabelsPartial {
        match Self::checked_region_extents(region) {
            Ok((max_width, max_height)) => TileLabelsPartial {
                max_width,
                max_height,
                active_positions: Vec::new(),
                error: Some(error),
            },
            Err(extent_error) => TileLabelsPartial {
                max_width: 0,
                max_height: 0,
                active_positions: Vec::new(),
                error: Some(extent_error),
            },
        }
    }

    #[inline]
    fn collect_active_positions(
        &self,
        tile: &Tile<U8>,
        region: &Region,
        active_positions: &mut Vec<(u32, u32)>,
    ) -> Result<(), ViprsError> {
        let width = region.width as usize;
        let height = region.height as usize;
        let bands = tile.bands as usize;
        active_positions.clear();
        active_positions.reserve(Self::checked_position_capacity(
            region.width,
            region.height,
            "labelregions active-position buffer exceeds addressable memory",
        )?);

        for y in 0..height {
            for x in 0..width {
                let pixel_base = (y * width + x) * bands;
                let is_active = tile.data[pixel_base..pixel_base + bands]
                    .iter()
                    .any(|&sample| sample > self.threshold);
                if is_active {
                    let image_x = i64::from(region.x) + x as i64;
                    let image_y = i64::from(region.y) + y as i64;
                    if image_x < 0
                        || image_y < 0
                        || image_x > i64::from(u32::MAX)
                        || image_y > i64::from(u32::MAX)
                    {
                        continue;
                    }
                    active_positions.push((image_x as u32, image_y as u32));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Tile;

    #[test]
    fn union_find_attaches_lower_rank_root_to_higher_rank_root() {
        let mut union_find = UnionFind::new(3);

        union_find.union(1, 2);
        union_find.union(0, 1);

        assert_eq!(union_find.find(0), union_find.find(1));
        assert_eq!(union_find.parent[0], 1);
    }

    #[test]
    fn label_regions_assigns_distinct_labels_to_disconnected_blobs() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let top_region = Region::new(0, 0, 4, 2);
        let bottom_region = Region::new(0, 2, 4, 2);
        let top = Tile::<U8>::new(top_region, 1, &[1, 1, 0, 0, 1, 1, 0, 0]);
        let bottom = Tile::<U8>::new(bottom_region, 1, &[0, 0, 1, 1, 0, 0, 1, 1]);

        let combined = reducer.combine(
            reducer.reduce_tile(&top, &top_region),
            reducer.reduce_tile(&bottom, &bottom_region),
        );
        let labels = reducer.finalize(combined).unwrap();
        let data = labels.pixels();

        assert_eq!(data[0], data[1]);
        assert_eq!(data[0], data[4]);
        assert_eq!(data[10], data[11]);
        assert_eq!(data[10], data[14]);
        assert_ne!(data[0], 0);
        assert_ne!(data[10], 0);
        assert_ne!(data[0], data[10]);
    }

    #[test]
    fn accumulate_into_reuses_scratch_buffer_and_matches_reduce_tile_path() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let top_region = Region::new(0, 0, 4, 2);
        let bottom_region = Region::new(0, 2, 4, 2);
        let top = Tile::<U8>::new(top_region, 1, &[1, 1, 0, 0, 1, 1, 0, 0]);
        let bottom = Tile::<U8>::new(bottom_region, 1, &[0, 0, 1, 1, 0, 0, 1, 1]);

        let combined_reduce_tile = reducer.combine(
            reducer.reduce_tile(&top, &top_region),
            reducer.reduce_tile(&bottom, &bottom_region),
        );
        let expected = reducer
            .finalize(combined_reduce_tile)
            .unwrap()
            .pixels()
            .to_vec();

        let mut partial = None;
        let mut scratch = Vec::with_capacity(16);
        let scratch_ptr = scratch.as_ptr();
        reducer.accumulate_into(&top, &top_region, &mut scratch, &mut partial);
        reducer.accumulate_into(&bottom, &bottom_region, &mut scratch, &mut partial);

        assert_eq!(scratch_ptr, scratch.as_ptr());
        let actual = reducer
            .finalize(partial.expect("partial should exist after accumulate_into calls"))
            .unwrap()
            .pixels()
            .to_vec();
        assert_eq!(actual, expected);
    }

    #[test]
    fn negative_regions_skip_out_of_bounds_pixels_instead_of_wrapping_coordinates() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let region = Region::new(-1, -1, 3, 3);
        let tile = Tile::<U8>::new(
            region,
            1,
            &[
                1, 1, 1, //
                1, 1, 1, //
                1, 1, 1,
            ],
        );

        let mut partial = None;
        let mut scratch = Vec::new();
        reducer.accumulate_into(&tile, &region, &mut scratch, &mut partial);
        let labels = reducer
            .finalize(partial.expect("partial should exist after accumulate_into call"))
            .unwrap()
            .pixels()
            .to_vec();

        assert_eq!(
            labels.len(),
            4,
            "output dimensions should clip to visible extents"
        );
        assert_ne!(labels[0], 0);
        assert!(labels.iter().all(|&value| value == labels[0]));
    }

    #[test]
    fn finalize_builds_non_trivial_label_map_from_partial() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let partial = TileLabelsPartial {
            max_width: 5,
            max_height: 4,
            active_positions: vec![(0, 0), (1, 0), (0, 1), (3, 0), (4, 0), (4, 1), (1, 3)],
            error: None,
        };

        let labels = reducer.finalize(partial).unwrap();
        let data = labels.pixels();

        assert_eq!(data[0], data[1]);
        assert_eq!(data[0], data[5]);
        assert_eq!(data[3], data[4]);
        assert_eq!(data[3], data[9]);
        assert_ne!(data[0], 0);
        assert_ne!(data[3], 0);
        assert_ne!(data[16], 0);
        assert_ne!(data[0], data[3]);
        assert_ne!(data[0], data[16]);
        assert_ne!(data[3], data[16]);
        assert_eq!(data[2], 0);
        assert_eq!(data[6], 0);
        assert_eq!(data[19], 0);
    }

    #[test]
    fn collect_active_positions_returns_image_too_large_for_oversized_region() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let region = Region::new(0, 0, u32::MAX, u32::MAX);
        let tile = Tile::<U8> {
            region,
            bands: 1,
            data: &[],
        };
        let mut active_positions = Vec::new();

        let result = reducer.collect_active_positions(&tile, &region, &mut active_positions);

        assert!(matches!(
            result,
            Err(ViprsError::ImageTooLarge {
                details: "labelregions active-position buffer exceeds addressable memory",
                ..
            })
        ));
        assert!(active_positions.is_empty());
    }

    #[test]
    fn reduce_tile_and_finalize_propagate_collection_errors() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let region = Region::new(0, 0, u32::MAX, u32::MAX);
        let tile = Tile::<U8> {
            region,
            bands: 1,
            data: &[],
        };

        let result = reducer.finalize(reducer.reduce_tile(&tile, &region));

        assert!(matches!(
            result,
            Err(ViprsError::ImageTooLarge {
                details: "labelregions active-position buffer exceeds addressable memory",
                ..
            })
        ));
    }

    #[test]
    fn accumulate_into_keeps_error_and_expands_bounds_after_failed_partial() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let oversized_region = Region::new(0, 0, u32::MAX, u32::MAX);
        let oversized_tile = Tile::<U8> {
            region: oversized_region,
            bands: 1,
            data: &[],
        };
        let valid_region = Region::new(2, 3, 2, 1);
        let valid_tile = Tile::<U8>::new(valid_region, 1, &[1, 1]);
        let mut partial = None;
        let mut scratch = Vec::new();

        reducer.accumulate_into(
            &oversized_tile,
            &oversized_region,
            &mut scratch,
            &mut partial,
        );
        reducer.accumulate_into(&valid_tile, &valid_region, &mut scratch, &mut partial);

        let partial = partial.expect("partial should exist after accumulation");
        assert_eq!(partial.max_width, u32::MAX);
        assert_eq!(partial.max_height, u32::MAX);
        assert!(partial.active_positions.is_empty());
        assert!(matches!(
            reducer.finalize(partial),
            Err(ViprsError::ImageTooLarge {
                details: "labelregions active-position buffer exceeds addressable memory",
                ..
            })
        ));
    }

    #[test]
    fn combine_returns_rhs_error_without_merging_positions() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let left = TileLabelsPartial {
            max_width: 2,
            max_height: 2,
            active_positions: vec![(0, 0), (1, 1)],
            error: None,
        };
        let right = TileLabelsPartial {
            max_width: 4,
            max_height: 3,
            active_positions: vec![(3, 2)],
            error: Some(ViprsError::ImageTooLarge {
                width: 4,
                height: 3,
                bands: 1,
                bytes: 12,
                limit_bytes: 8,
                details: "labelregions active-position buffer exceeds addressable memory",
            }),
        };

        let combined = reducer.combine(left, right);

        assert_eq!(combined.max_width, 4);
        assert_eq!(combined.max_height, 3);
        assert_eq!(combined.active_positions, vec![(0, 0), (1, 1)]);
        assert!(matches!(
            combined.error,
            Some(ViprsError::ImageTooLarge {
                details: "labelregions active-position buffer exceeds addressable memory",
                ..
            })
        ));
    }

    #[test]
    fn accumulate_into_merges_late_collection_error_into_existing_partial() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let valid_region = Region::new(0, 0, 2, 2);
        let valid_tile = Tile::<U8>::new(valid_region, 1, &[1, 0, 0, 1]);
        let oversized_region = Region::new(0, 0, u32::MAX, u32::MAX);
        let oversized_tile = Tile::<U8> {
            region: oversized_region,
            bands: 1,
            data: &[],
        };
        let mut partial = None;
        let mut scratch = Vec::new();

        reducer.accumulate_into(&valid_tile, &valid_region, &mut scratch, &mut partial);
        reducer.accumulate_into(
            &oversized_tile,
            &oversized_region,
            &mut scratch,
            &mut partial,
        );

        let partial = partial.expect("partial should exist after accumulation");
        assert_eq!(partial.max_width, u32::MAX);
        assert_eq!(partial.max_height, u32::MAX);
        assert_eq!(partial.active_positions, vec![(0, 0), (1, 1)]);
        assert!(matches!(
            reducer.finalize(partial),
            Err(ViprsError::ImageTooLarge {
                details: "labelregions active-position buffer exceeds addressable memory",
                ..
            })
        ));
    }

    #[test]
    fn combine_preserves_existing_error_without_merging_rhs_positions() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let left = TileLabelsPartial {
            max_width: 3,
            max_height: 4,
            active_positions: vec![(2, 3)],
            error: Some(ViprsError::ImageTooLarge {
                width: 3,
                height: 4,
                bands: 1,
                bytes: 12,
                limit_bytes: 8,
                details: "labelregions final image exceeds addressable memory",
            }),
        };
        let right = TileLabelsPartial {
            max_width: 6,
            max_height: 2,
            active_positions: vec![(5, 1)],
            error: None,
        };

        let combined = reducer.combine(left, right);

        assert_eq!(combined.max_width, 6);
        assert_eq!(combined.max_height, 4);
        assert_eq!(combined.active_positions, vec![(2, 3)]);
        assert!(matches!(
            combined.error,
            Some(ViprsError::ImageTooLarge {
                details: "labelregions final image exceeds addressable memory",
                ..
            })
        ));
    }

    #[test]
    fn finalize_returns_image_too_large_for_oversized_dimensions() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let partial = TileLabelsPartial {
            max_width: u32::MAX,
            max_height: u32::MAX,
            active_positions: Vec::new(),
            error: None,
        };

        let result = reducer.finalize(partial);

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }

    #[test]
    fn checked_region_extents_clips_negative_offset_wide_tiles_without_false_overflow() {
        let region = Region::new(i32::MIN, 0, u32::MAX, 2);

        let extents = LabelRegionsReducer::checked_region_extents(&region);

        assert!(matches!(extents, Ok((width, 2)) if width == i32::MAX as u32));
    }

    #[test]
    fn checked_region_extents_still_rejects_genuinely_oversized_regions() {
        let region = Region::new(0, 0, u32::MAX, u32::MAX);

        let extents = LabelRegionsReducer::checked_region_extents(&region).unwrap();
        let pixel_count = LabelRegionsReducer::checked_pixel_count(
            extents.0,
            extents.1,
            "labelregions final image exceeds addressable memory",
        );

        assert!(matches!(pixel_count, Err(ViprsError::ImageTooLarge { .. })));
    }

    #[test]
    fn finalize_returns_image_too_large_when_region_extent_overflows() {
        let reducer = LabelRegionsReducer { threshold: 0 };
        let region = Region::new(i32::MAX, 0, (u32::MAX - i32::MAX as u32) + 10, 0);
        let tile = Tile::<U8>::new(region, 1, &[]);

        let result = reducer.finalize(reducer.reduce_tile(&tile, &region));

        assert!(matches!(result, Err(ViprsError::ImageTooLarge { .. })));
    }
}
