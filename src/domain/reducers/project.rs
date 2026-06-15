//! Reducers for projecting sample sums onto image rows and columns.

use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

/// Stores the per-axis sample sums for a projection reduction.
///
/// This result type solves projection queries by returning one summed value per column and row
/// for every band in the source image.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProjectOp,
/// };
///
/// let reducer = ProjectOp::new(2, 1, 1);
/// let region = Region::new(0, 0, 2, 1);
/// let tile = Tile::<U8>::new(region, 1, &[1, 2]);
/// let project = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(project.columns, vec![1.0, 2.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectResult {
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Stores the `columns` value for this item.
    pub columns: Vec<f64>,
    /// Number of rows associated with this configuration.
    pub rows: Vec<f64>,
}

/// Accumulates row and column sums for a single tile.
///
/// This partial value solves parallel projection by letting tile-local sums be merged before the
/// caller consumes the final per-axis totals.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProjectOp,
/// };
///
/// let reducer = ProjectOp::new(1, 1, 1);
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 1, &[7]);
/// let partial = reducer.reduce_tile(&tile, &region);
///
/// let _ = partial;
/// ```
#[derive(Clone, Default)]
pub struct ProjectPartial {
    columns: Vec<f64>,
    rows: Vec<f64>,
}

impl ProjectPartial {
    fn new(width: u32, height: u32, bands: u32) -> Self {
        Self {
            columns: vec![0.0; width as usize * bands as usize],
            rows: vec![0.0; height as usize * bands as usize],
        }
    }

    fn reset(&mut self) {
        self.columns.fill(0.0);
        self.rows.fill(0.0);
    }
}

/// Computes projection sums across columns and rows.
///
/// This reducer solves profile-style intensity measurements by collapsing an image into the total
/// signal carried by each axis and band.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::ProjectOp,
/// };
///
/// let reducer = ProjectOp::new(2, 2, 1);
/// let region = Region::new(0, 0, 2, 2);
/// let tile = Tile::<U8>::new(region, 1, &[1, 2, 3, 4]);
/// let project = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(project.rows, vec![3.0, 7.0]);
/// ```
pub struct ProjectOp {
    width: u32,
    height: u32,
    bands: u32,
}

impl ProjectOp {
    /// Creates a projection reducer for an image with the given shape.
    ///
    /// This constructor solves output sizing once so each tile can reuse the same row and column
    /// accumulation layout throughout the reduction.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::ProjectOp;
    ///
    /// let reducer = ProjectOp::new(640, 480, 3);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub fn new(width: u32, height: u32, bands: u32) -> Self {
        debug_assert!(bands > 0, "ProjectOp: bands must be at least 1");
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
        partial: &mut ProjectPartial,
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
                    let value = tile.data[base + band].to_f64();
                    partial.columns[x as usize * bands + band] += value;
                    partial.rows[y as usize * bands + band] += value;
                }
            }
        }
    }
}

impl<F> TileReducer<F> for ProjectOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ProjectPartial;
    type Scratch = ProjectPartial;
    type Output = ProjectResult;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = ProjectPartial::new(self.width, self.height, self.bands);
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
            *scratch = ProjectPartial::new(self.width, self.height, self.bands);
        } else {
            scratch.reset();
        }

        self.accumulate_tile_partial(tile, region, scratch);

        match partial {
            Some(existing) => {
                for (left, right) in existing
                    .columns
                    .iter_mut()
                    .zip(scratch.columns.iter().copied())
                {
                    *left += right;
                }
                for (left, right) in existing.rows.iter_mut().zip(scratch.rows.iter().copied()) {
                    *left += right;
                }
            }
            None => {
                *partial = Some(scratch.clone());
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (left, right) in a.columns.iter_mut().zip(b.columns) {
            *left += right;
        }
        for (left, right) in a.rows.iter_mut().zip(b.rows) {
            *left += right;
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        ProjectResult {
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

    fn expected_project(data: &[u8], width: u32, height: u32, bands: u32) -> ProjectResult {
        let mut columns = vec![0.0; width as usize * bands as usize];
        let mut rows = vec![0.0; height as usize * bands as usize];
        let width_usize = width as usize;
        let bands_usize = bands as usize;

        for y in 0..height as usize {
            for x in 0..width_usize {
                let base = (y * width_usize + x) * bands_usize;
                for band in 0..bands_usize {
                    let value = f64::from(data[base + band]);
                    columns[x * bands_usize + band] += value;
                    rows[y * bands_usize + band] += value;
                }
            }
        }

        ProjectResult {
            width,
            height,
            bands,
            columns,
            rows,
        }
    }

    #[test]
    fn project_sums_columns_and_rows() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 2, 3, 4]);
        let reducer = ProjectOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(project.columns, vec![4.0, 6.0]);
        assert_eq!(project.rows, vec![3.0, 7.0]);
    }

    proptest! {
        #[test]
        fn project_identity_for_all_zero_image(width in 1u32..=8, height in 1u32..=8, bands in 1u32..=3) {
            let data = vec![0u8; width as usize * height as usize * bands as usize];
            let region = Region::new(0, 0, width, height);
            let tile = Tile::<U8>::new(region, bands, &data);
            let reducer = ProjectOp::new(width, height, bands);
            let partial = reducer.reduce_tile(&tile, &region);
            let actual = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);
            let expected = expected_project(&data, width, height, bands);
            prop_assert_eq!(actual, expected);
        }

        fn project_matches_reference(
            (width, height, bands, data) in (1u32..=8, 1u32..=8, 1u32..=3)
                .prop_flat_map(|(width, height, bands)| {
                    let len = width as usize * height as usize * bands as usize;
                    (Just(width), Just(height), Just(bands), proptest::collection::vec(any::<u8>(), len))
                })
        ) {
            let region = Region::new(0, 0, width, height);
            let tile = Tile::<U8>::new(region, bands, &data);
            let reducer = ProjectOp::new(width, height, bands);
            let partial = reducer.reduce_tile(&tile, &region);
            let actual = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);
            let expected = expected_project(&data, width, height, bands);
            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn project_boundary_sums_maximum_values() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[u8::MAX, 0, 0, u8::MAX]);
        let reducer = ProjectOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);
        assert_eq!(project.columns, vec![255.0, 255.0]);
        assert_eq!(project.rows, vec![255.0, 255.0]);
    }

    #[test]
    fn project_original_negative_origin_repro_no_longer_panics() {
        let region = Region::new(-1, -1, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 0, 0, 0]);
        let reducer = ProjectOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(project.columns, vec![0.0, 0.0]);
        assert_eq!(project.rows, vec![0.0, 0.0]);
    }

    #[test]
    fn project_clips_negative_origin_tiles_before_indexing() {
        let region = Region::new(-1, -1, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 2, 3, 4]);
        let reducer = ProjectOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(project.columns, vec![4.0, 0.0]);
        assert_eq!(project.rows, vec![4.0, 0.0]);
    }

    #[test]
    fn project_clips_tiles_past_right_and_bottom_edges() {
        let region = Region::new(1, 1, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[1, 2, 3, 4]);
        let reducer = ProjectOp::new(2, 2, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(project.columns, vec![0.0, 1.0]);
        assert_eq!(project.rows, vec![0.0, 1.0]);
    }

    #[test]
    fn project_accumulate_into_reuses_scratch_and_matches_reduce_tile_path() {
        let reducer = ProjectOp::new(4, 1, 1);
        let left_region = Region::new(0, 0, 2, 1);
        let right_region = Region::new(2, 0, 2, 1);
        let left_tile = Tile::<U8>::new(left_region, 1, &[1, 2]);
        let right_tile = Tile::<U8>::new(right_region, 1, &[3, 4]);

        let expected = <ProjectOp as TileReducer<U8>>::finalize(
            &reducer,
            <ProjectOp as TileReducer<U8>>::combine(
                &reducer,
                reducer.reduce_tile(&left_tile, &left_region),
                reducer.reduce_tile(&right_tile, &right_region),
            ),
        );

        let mut partial = None;
        let mut scratch = ProjectPartial {
            columns: vec![99.0; 4],
            rows: vec![99.0; 1],
        };
        let columns_ptr = scratch.columns.as_ptr();
        let rows_ptr = scratch.rows.as_ptr();

        reducer.accumulate_into(&left_tile, &left_region, &mut scratch, &mut partial);
        reducer.accumulate_into(&right_tile, &right_region, &mut scratch, &mut partial);

        assert_eq!(columns_ptr, scratch.columns.as_ptr());
        assert_eq!(rows_ptr, scratch.rows.as_ptr());
        let actual = <ProjectOp as TileReducer<U8>>::finalize(
            &reducer,
            partial.expect("partial should exist after accumulate_into"),
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn project_accumulate_into_resizes_mismatched_scratch_before_accumulating() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 2, &[1, 10, 2, 20, 3, 30, 4, 40]);
        let reducer = ProjectOp::new(2, 2, 2);
        let mut partial = None;
        let mut scratch = ProjectPartial::default();

        reducer.accumulate_into(&tile, &region, &mut scratch, &mut partial);

        assert_eq!(scratch.columns.len(), 4);
        assert_eq!(scratch.rows.len(), 4);
        let actual = <ProjectOp as TileReducer<U8>>::finalize(
            &reducer,
            partial.expect("partial should exist after accumulate_into"),
        );
        let expected = expected_project(tile.data, 2, 2, 2);
        assert_eq!(actual, expected);
    }

    #[test]
    fn project_combine_adds_partials_from_different_tiles() {
        let reducer = ProjectOp::new(2, 2, 1);
        let top_region = Region::new(0, 0, 2, 1);
        let bottom_region = Region::new(0, 1, 2, 1);
        let top_tile = Tile::<U8>::new(top_region, 1, &[1, 2]);
        let bottom_tile = Tile::<U8>::new(bottom_region, 1, &[3, 4]);

        let combined = <ProjectOp as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&top_tile, &top_region),
            reducer.reduce_tile(&bottom_tile, &bottom_region),
        );

        let project = <ProjectOp as TileReducer<U8>>::finalize(&reducer, combined);
        assert_eq!(project.columns, vec![4.0, 6.0]);
        assert_eq!(project.rows, vec![3.0, 7.0]);
    }
}
