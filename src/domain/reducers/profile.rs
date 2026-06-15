//! Reducers for finding the first non-zero sample along rows and columns.

use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

/// Stores the first non-zero row and column hit for each band.
///
/// This result type solves image profiling by exposing the earliest occupied coordinate per
/// column and per row, which downstream code can use to reason about object extents.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProfileOp,
/// };
///
/// let reducer = ProfileOp::new(2, 2, 1);
/// let region = Region::new(0, 0, 2, 2);
/// let tile = Tile::<U8>::new(region, 1, &[0, 1, 2, 0]);
/// let profile = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(profile.columns, vec![1, 0]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileResult {
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `columns` value for this item.
    pub columns: Vec<u32>,
    /// Number of rows associated with this configuration.
    pub rows: Vec<u32>,
}

/// Accumulates the earliest non-zero coordinates for one tile.
///
/// This partial value solves tile-local profile aggregation so parallel reducers can merge
/// earliest-hit information before producing the final profile image summary.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProfileOp,
/// };
///
/// let reducer = ProfileOp::new(1, 1, 1);
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 1, &[1]);
/// let partial = reducer.reduce_tile(&tile, &region);
///
/// let _ = partial;
/// ```
#[derive(Clone, Default)]
pub struct ProfilePartial {
    columns: Vec<u32>,
    rows: Vec<u32>,
}

impl ProfilePartial {
    fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            columns: vec![height; width as usize * bands as usize],
            rows: vec![width; height as usize * bands as usize],
        }
    }

    fn reset(&mut self, width: u32, height: u32) {
        self.columns.fill(height);
        self.rows.fill(width);
    }
}

/// Computes the first non-zero sample position for every row and column.
///
/// This reducer solves fast extent profiling by recording the earliest occupied coordinate in
/// each axis instead of materializing a full mask or scanline analysis result.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProfileOp,
/// };
///
/// let reducer = ProfileOp::new(3, 1, 1);
/// let region = Region::new(0, 0, 3, 1);
/// let tile = Tile::<U8>::new(region, 1, &[0, 5, 0]);
/// let profile = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(profile.rows, vec![1]);
/// ```
pub struct ProfileOp {
    width: u32,
    height: u32,
    bands: u32,
}

impl ProfileOp {
    /// Creates a profile reducer for an image with the given shape.
    ///
    /// This constructor fixes the output buffer sizes up front so every tile can reuse the same
    /// per-axis storage layout while searching for first-hit coordinates.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::ProfileOp;
    ///
    /// let reducer = ProfileOp::new(640, 480, 3);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32, bands: u32) -> Self {
        debug_assert!(bands > 0, "ProfileOp: bands must be at least 1");
        Self {
            width,
            height,
            bands,
        }
    }

    fn accumulate_tile_partial<F>(
        &self,
        tile: &Tile<F>,
        region: &Region,
        partial: &mut ProfilePartial,
    ) where
        F: BandFormat,
        F::Sample: ToF64,
    {
        let bands = self.bands as usize;
        let tile_width = region.width as usize;
        let x_start = region.x.max(0) as u32;
        let y_start = region.y.max(0) as u32;
        let x_end =
            (i64::from(region.x) + i64::from(region.width)).clamp(0, i64::from(self.width)) as u32;
        let y_end = (i64::from(region.y) + i64::from(region.height))
            .clamp(0, i64::from(self.height)) as u32;

        for y in y_start..y_end {
            let row = (i64::from(y) - i64::from(region.y)) as usize;
            for x in x_start..x_end {
                let col = (i64::from(x) - i64::from(region.x)) as usize;
                let base = (row * tile_width + col) * bands;
                for band in 0..bands {
                    if tile.data[base + band].to_f64() != 0.0 {
                        let col_idx = x as usize * bands + band;
                        let row_idx = y as usize * bands + band;
                        partial.columns[col_idx] = partial.columns[col_idx].min(y);
                        partial.rows[row_idx] = partial.rows[row_idx].min(x);
                    }
                }
            }
        }
    }
}

impl<F> TileReducer<F> for ProfileOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ProfilePartial;
    type Output = ProfileResult;
    type Scratch = ProfilePartial;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = ProfilePartial::new(self.width, self.height, self.bands);
        self.accumulate_tile_partial(tile, region, &mut partial);
        partial
    }

    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        if scratch.columns.len() != self.width as usize * self.bands as usize
            || scratch.rows.len() != self.height as usize * self.bands as usize
        {
            *scratch = ProfilePartial::new(self.width, self.height, self.bands);
        } else {
            scratch.reset(self.width, self.height);
        }

        self.accumulate_tile_partial(tile, region, scratch);

        match partial {
            Some(existing) => {
                for (left, right) in existing
                    .columns
                    .iter_mut()
                    .zip(scratch.columns.iter().copied())
                {
                    *left = (*left).min(right);
                }
                for (left, right) in existing.rows.iter_mut().zip(scratch.rows.iter().copied()) {
                    *left = (*left).min(right);
                }
            }
            None => {
                *partial = Some(scratch.clone());
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.columns.iter_mut().zip(b.columns) {
            *left = (*left).min(right);
        }
        for (left, right) in a.rows.iter_mut().zip(b.rows) {
            *left = (*left).min(right);
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        ProfileResult {
            width: self.width,
            height: self.height,
            bands: self.bands,
            columns: combined.columns,
            rows: combined.rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, reducer::TileReducer};
    use proptest::prelude::*;

    fn expected_profile(data: &[u8], width: u32, height: u32, bands: u32) -> ProfileResult {
        let mut columns = vec![height; width as usize * bands as usize];
        let mut rows = vec![width; height as usize * bands as usize];
        let width_usize = width as usize;
        let bands_usize = bands as usize;

        for y in 0..height as usize {
            for x in 0..width_usize {
                let base = (y * width_usize + x) * bands_usize;
                for band in 0..bands_usize {
                    if data[base + band] != 0 {
                        let col_idx = x * bands_usize + band;
                        let row_idx = y * bands_usize + band;
                        columns[col_idx] = columns[col_idx].min(y as u32);
                        rows[row_idx] = rows[row_idx].min(x as u32);
                    }
                }
            }
        }

        ProfileResult {
            width,
            height,
            bands,
            columns,
            rows,
        }
    }

    #[test]
    fn profile_finds_top_and_left_edges_per_band() {
        let region = Region::new(0, 0, 3, 2);
        let tile = Tile::<U8>::new(region, 1, &[0, 5, 0, 7, 0, 9]);
        let reducer = ProfileOp::new(3, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let profile = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(profile.columns, vec![1, 0, 1]);
        assert_eq!(profile.rows, vec![1, 0]);
    }

    proptest! {
        #[test]
        fn profile_identity_for_all_zero_image(width in 1u32..=8, height in 1u32..=8, bands in 1u32..=3) {
            let data = vec![0u8; width as usize * height as usize * bands as usize];
            let region = Region::new(0, 0, width, height);
            let tile = Tile::<U8>::new(region, bands, &data);
            let reducer = ProfileOp::new(width, height, bands);
            let partial = reducer.reduce_tile(&tile, &region);
            let actual = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);
            let expected = expected_profile(&data, width, height, bands);
            prop_assert_eq!(actual, expected);
        }

        fn profile_matches_reference(
            (width, height, bands, data) in (1u32..=8, 1u32..=8, 1u32..=3)
                .prop_flat_map(|(width, height, bands)| {
                    let len = width as usize * height as usize * bands as usize;
                    (Just(width), Just(height), Just(bands), proptest::collection::vec(any::<u8>(), len))
                })
        ) {
            let region = Region::new(0, 0, width, height);
            let tile = Tile::<U8>::new(region, bands, &data);
            let reducer = ProfileOp::new(width, height, bands);
            let partial = reducer.reduce_tile(&tile, &region);
            let actual = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);
            let expected = expected_profile(&data, width, height, bands);
            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn profile_boundary_detects_edge_pixels() {
        let region = Region::new(0, 0, 4, 3);
        let tile = Tile::<U8>::new(region, 1, &[1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 4]);
        let reducer = ProfileOp::new(4, 3, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let profile = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);
        assert_eq!(profile.columns, vec![0, 3, 3, 0]);
        assert_eq!(profile.rows, vec![0, 4, 0]);
    }

    #[test]
    fn profile_original_negative_origin_repro_no_longer_panics() {
        let region = Region::new(-1, -1, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 0, 0, 0]);
        let reducer = ProfileOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let profile = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(profile.columns, vec![2, 2]);
        assert_eq!(profile.rows, vec![2, 2]);
    }

    #[test]
    fn profile_clips_negative_origin_tiles_before_indexing() {
        let region = Region::new(-1, -1, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 2, 3, 4]);
        let reducer = ProfileOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let profile = <ProfileOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(profile.columns, vec![0, 2]);
        assert_eq!(profile.rows, vec![0, 2]);
    }

    #[test]
    fn expected_profile_tracks_non_zero_samples_per_band() {
        let profile = expected_profile(&[0, 1, 2, 0, 0, 0, 3, 4], 2, 2, 2);

        assert_eq!(profile.columns, vec![2, 0, 0, 1]);
        assert_eq!(profile.rows, vec![1, 0, 1, 1]);
    }

    #[test]
    fn profile_accumulate_into_reinitializes_mismatched_scratch_and_merges_partials() {
        let region = Region::new(0, 0, 3, 2);
        let first_tile = Tile::<U8>::new(region, 1, &[0, 5, 0, 0, 0, 0]);
        let second_tile = Tile::<U8>::new(region, 1, &[7, 0, 0, 0, 0, 9]);
        let reducer = ProfileOp::new(3, 2, 1);
        let mut scratch = ProfilePartial::default();
        let mut partial = None;

        reducer.accumulate_into(&first_tile, &region, &mut scratch, &mut partial);

        let first = partial
            .clone()
            .expect("first partial should be initialized");
        assert_eq!(first.columns, vec![2, 0, 2]);
        assert_eq!(first.rows, vec![1, 3]);

        reducer.accumulate_into(&second_tile, &region, &mut scratch, &mut partial);

        let profile = <ProfileOp as TileReducer<U8>>::finalize(
            &reducer,
            partial.expect("merged partial should exist"),
        );
        assert_eq!(profile.columns, vec![0, 0, 1]);
        assert_eq!(profile.rows, vec![0, 2]);
    }

    #[test]
    fn profile_accumulate_into_resets_reusable_scratch_before_cloning() {
        let region = Region::new(0, 0, 3, 2);
        let marked_tile = Tile::<U8>::new(region, 1, &[0, 5, 0, 7, 0, 9]);
        let zero_tile = Tile::<U8>::new(region, 1, &[0, 0, 0, 0, 0, 0]);
        let reducer = ProfileOp::new(3, 2, 1);
        let mut scratch = reducer.reduce_tile(&marked_tile, &region);
        let mut partial = None;

        reducer.accumulate_into(&zero_tile, &region, &mut scratch, &mut partial);

        let profile = <ProfileOp as TileReducer<U8>>::finalize(
            &reducer,
            partial.expect("zero tile should still produce a partial"),
        );
        assert_eq!(profile.columns, vec![2, 2, 2]);
        assert_eq!(profile.rows, vec![3, 3]);
    }

    #[test]
    fn profile_combine_keeps_the_earliest_non_zero_coordinate_from_each_partial() {
        let region = Region::new(0, 0, 3, 2);
        let first_tile = Tile::<U8>::new(region, 1, &[0, 5, 0, 0, 0, 0]);
        let second_tile = Tile::<U8>::new(region, 1, &[7, 0, 0, 0, 0, 9]);
        let reducer = ProfileOp::new(3, 2, 1);
        let first = reducer.reduce_tile(&first_tile, &region);
        let second = reducer.reduce_tile(&second_tile, &region);

        let combined = <ProfileOp as TileReducer<U8>>::combine(&reducer, first, second);
        let profile = <ProfileOp as TileReducer<U8>>::finalize(&reducer, combined);

        assert_eq!(profile.columns, vec![0, 0, 1]);
        assert_eq!(profile.rows, vec![0, 2]);
    }
}
