//! Reducers for sampling exact pixel coordinates from a tiled image.

use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    reducer::TileReducer,
};

/// Collects the samples stored at a fixed set of image coordinates.
///
/// This reducer solves point queries without materializing an entire image traversal in the
/// caller, returning one sample vector per requested coordinate in band order.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::GetpointReducer,
/// };
///
/// let region = Region::new(0, 0, 2, 1);
/// let tile = Tile::<U8>::new(region, 1, &[10, 20]);
/// let reducer = GetpointReducer::new(vec![(1, 0)]);
/// let partial = reducer.reduce_tile(&tile, &region);
/// let samples = reducer.finalize(partial);
///
/// assert_eq!(samples, vec![vec![20]]);
/// ```
pub struct GetpointReducer {
    points: Vec<(u32, u32)>,
}

impl GetpointReducer {
    /// Creates a reducer that samples the supplied image coordinates.
    ///
    /// This constructor lets callers batch multiple point lookups into a single reduction pass
    /// and preserve the original request order in the output.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::GetpointReducer;
    ///
    /// let reducer = GetpointReducer::new(vec![(0, 0), (10, 20)]);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub const fn new(points: Vec<(u32, u32)>) -> Self {
        Self { points }
    }
}

impl<F> TileReducer<F> for GetpointReducer
where
    F: BandFormat,
    F::Sample: Copy,
{
    type Partial = Vec<Option<Vec<F::Sample>>>;
    type Output = Vec<Vec<F::Sample>>;
    /// Per-thread reusable sample slots for each requested point. Each inner vector is
    /// allocated lazily the first time that point is observed on the current thread and
    /// then cleared/reused across tiles.
    type Scratch = Vec<Vec<F::Sample>>;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = vec![None; self.points.len()];
        let left = i64::from(region.x);
        let top = i64::from(region.y);
        let right = left + i64::from(region.width);
        let bottom = top + i64::from(region.height);
        let row_stride = region.width as usize * tile.bands as usize;
        let pixel_stride = tile.bands as usize;

        for (idx, &(point_x, point_y)) in self.points.iter().enumerate() {
            let px = i64::from(point_x);
            let py = i64::from(point_y);

            if px < left || px >= right || py < top || py >= bottom {
                continue;
            }

            let local_x = (px - left) as usize;
            let local_y = (py - top) as usize;
            let offset = local_y * row_stride + local_x * pixel_stride;
            partial[idx] = Some(tile.data[offset..offset + pixel_stride].to_vec());
        }

        partial
    }

    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        let left = i64::from(region.x);
        let top = i64::from(region.y);
        let right = left + i64::from(region.width);
        let bottom = top + i64::from(region.height);
        let row_stride = region.width as usize * tile.bands as usize;
        let pixel_stride = tile.bands as usize;

        scratch.resize_with(self.points.len(), Vec::new);
        for samples in scratch.iter_mut() {
            samples.clear();
        }

        for (idx, &(point_x, point_y)) in self.points.iter().enumerate() {
            let px = i64::from(point_x);
            let py = i64::from(point_y);

            if px < left || px >= right || py < top || py >= bottom {
                continue;
            }

            let local_x = (px - left) as usize;
            let local_y = (py - top) as usize;
            let offset = local_y * row_stride + local_x * pixel_stride;
            scratch[idx].extend_from_slice(&tile.data[offset..offset + pixel_stride]);
        }

        let accumulated = partial.get_or_insert_with(|| vec![None; self.points.len()]);
        for (entry, samples) in accumulated.iter_mut().zip(scratch.iter()) {
            if entry.is_none() && !samples.is_empty() {
                *entry = Some(samples.clone());
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.iter_mut().zip(b) {
            if left.is_none() {
                *left = right;
            }
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        combined
            .into_iter()
            .map(std::option::Option::unwrap_or_default)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::U8,
        image::{Region, Tile},
    };

    #[test]
    fn known_point_returns_expected_samples() {
        let region = Region::new(0, 0, 2, 2);
        let data = vec![
            9u8, 10, 11, 12, //
            13, 14, 15, 16,
        ];
        let tile = Tile::<U8>::new(region, 2, &data);
        let reducer = GetpointReducer::new(vec![(0, 0), (1, 1)]);
        let partial = <GetpointReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let points = <GetpointReducer as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(points, vec![vec![9, 10], vec![15, 16]]);
    }

    #[test]
    fn combine_preserves_points_found_in_different_tiles() {
        let reducer = GetpointReducer::new(vec![(0, 0), (3, 0)]);
        let left_tile_region = Region::new(0, 0, 2, 1);
        let right_tile_region = Region::new(2, 0, 2, 1);
        let left_tile = Tile::<U8>::new(left_tile_region, 1, &[7, 8]);
        let right_tile = Tile::<U8>::new(right_tile_region, 1, &[9, 10]);

        let left = <GetpointReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &left_tile,
            &left_tile_region,
        );
        let right = <GetpointReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &right_tile,
            &right_tile_region,
        );
        let combined = <GetpointReducer as TileReducer<U8>>::combine(&reducer, left, right);

        assert_eq!(
            <GetpointReducer as TileReducer<U8>>::finalize(&reducer, combined),
            vec![vec![7], vec![10]]
        );
    }

    #[test]
    fn missing_points_finalize_to_empty_vectors() {
        let region = Region::new(0, 0, 1, 1);
        let tile = Tile::<U8>::new(region, 1, &[42]);
        let reducer = GetpointReducer::new(vec![(2, 2)]);
        let partial = <GetpointReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);

        assert_eq!(
            <GetpointReducer as TileReducer<U8>>::finalize(&reducer, partial),
            vec![Vec::<u8>::new()]
        );
    }

    #[test]
    fn accumulate_into_reuses_scratch_and_matches_reduce_tile_path() {
        let reducer = GetpointReducer::new(vec![(0, 0), (3, 0)]);
        let left_region = Region::new(0, 0, 2, 1);
        let right_region = Region::new(2, 0, 2, 1);
        let left_tile = Tile::<U8>::new(left_region, 1, &[7, 8]);
        let right_tile = Tile::<U8>::new(right_region, 1, &[9, 10]);

        let expected = <GetpointReducer as TileReducer<U8>>::finalize(
            &reducer,
            <GetpointReducer as TileReducer<U8>>::combine(
                &reducer,
                reducer.reduce_tile(&left_tile, &left_region),
                reducer.reduce_tile(&right_tile, &right_region),
            ),
        );

        let mut partial = None;
        let mut scratch = Vec::with_capacity(2);
        let scratch_ptr = scratch.as_ptr();
        reducer.accumulate_into(&left_tile, &left_region, &mut scratch, &mut partial);
        reducer.accumulate_into(&right_tile, &right_region, &mut scratch, &mut partial);

        assert_eq!(scratch_ptr, scratch.as_ptr());
        let actual = <GetpointReducer as TileReducer<U8>>::finalize(
            &reducer,
            partial.expect("partial should exist after accumulate_into"),
        );
        assert_eq!(actual, expected);
    }
}
