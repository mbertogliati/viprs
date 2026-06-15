use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

/// Mean statistics for the current simplified 1×1 measurement grid.
#[derive(Debug, Clone, PartialEq)]
pub struct MeasureResult {
    /// Number of rows associated with this configuration.
    pub rows: u32,
    /// Number of columns associated with this configuration.
    pub cols: u32,
    /// Stores the `mean` value for this item.
    pub mean: Vec<f64>,
}

#[derive(Debug, Clone)]
/// Represents a measure partial.
pub struct MeasurePartial {
    sums: Vec<f64>,
    count: u64,
}

/// Whole-image mean reducer.
///
/// `MeasureOp` currently supports only a single 1×1 cell over the whole image.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::measure::MeasureOp;
///
/// let op = MeasureOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MeasureOp {
    bands: usize,
}

impl MeasureOp {
    #[must_use]
    /// Creates a new `MeasureOp`.
    pub fn new(bands: usize) -> Self {
        debug_assert!(bands > 0, "MeasureOp: bands must be at least 1");
        Self { bands }
    }
}

impl<F> TileReducer<F> for MeasureOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = MeasurePartial;
    type Output = MeasureResult;
    /// Pre-allocated per-band sums buffer, reused across tiles per rayon thread.
    /// Eliminates the `vec![0.0; bands]` allocation per tile.
    type Scratch = Vec<f64>;

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> Self::Partial {
        debug_assert_eq!(tile.bands as usize, self.bands);

        let mut sums = vec![0.0; self.bands];
        for (index, sample) in tile.data.iter().enumerate() {
            sums[index % self.bands] += sample.to_f64();
        }

        MeasurePartial {
            sums,
            count: tile.region.pixel_count() as u64,
        }
    }

    /// Zero-allocation tile accumulation using a pre-allocated sums buffer.
    ///
    /// `scratch` is resized on the first call and zeroed at the start of each tile.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        _region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        debug_assert_eq!(tile.bands as usize, self.bands);

        scratch.resize(self.bands, 0.0f64);
        scratch.fill(0.0f64);

        for (index, sample) in tile.data.iter().enumerate() {
            scratch[index % self.bands] += sample.to_f64();
        }

        let accumulated = partial.get_or_insert_with(|| MeasurePartial {
            sums: vec![0.0; self.bands],
            count: 0,
        });
        for (dst, src) in accumulated.sums.iter_mut().zip(scratch.iter().copied()) {
            *dst += src;
        }
        accumulated.count += tile.region.pixel_count() as u64;
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (lhs, rhs) in a.sums.iter_mut().zip(b.sums.iter()) {
            *lhs += rhs;
        }
        a.count += b.count;
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        let mean = if combined.count == 0 {
            vec![0.0; self.bands]
        } else {
            combined
                .sums
                .iter()
                .map(|sum| *sum / combined.count as f64)
                .collect()
        };

        MeasureResult {
            rows: 1,
            cols: 1,
            mean,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, reducer::TileReducer};

    const ALLOC_ENV: &str = "VIPRS_MEASURE_ALLOC_CHILD";
    const ALLOC_CHILD_TEST: &str = "domain::ops::arithmetic::measure::tests::accumulate_into_reuses_partial_storage_after_first_tile_child";

    fn measure_accumulate_into_alloc_stats() -> crate::test_support::AllocStats {
        let reducer = MeasureOp::new(2);
        let region = Region::new(0, 0, 2, 1);
        let first_data = vec![10u8, 100, 20, 120];
        let second_data = vec![30u8, 140, 40, 160];
        let first_tile = Tile::<U8>::new(region, 2, &first_data);
        let second_tile = Tile::<U8>::new(region, 2, &second_data);
        let mut scratch = Vec::new();
        let mut partial = None;

        reducer.accumulate_into(&first_tile, &region, &mut scratch, &mut partial);
        crate::test_support::reset_alloc_stats();
        reducer.accumulate_into(&second_tile, &region, &mut scratch, &mut partial);

        crate::test_support::alloc_stats()
    }

    #[test]
    fn measures_whole_image_mean() {
        let reducer = MeasureOp::new(1);
        let region = Region::new(0, 0, 4, 1);
        let data = vec![0u8, 10, 20, 30];
        let tile = Tile::<U8>::new(region, 1, &data);
        let partial = reducer.reduce_tile(&tile, &region);
        let result = <MeasureOp as TileReducer<U8>>::finalize(&reducer, partial);
        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 1);
        assert_eq!(result.mean, vec![15.0]);
    }

    #[test]
    fn combine_accumulates_multiple_tiles() {
        let reducer = MeasureOp::new(1);
        let region = Region::new(0, 0, 2, 1);
        let left_data = vec![10u8, 20];
        let right_data = vec![30u8, 40];
        let left_tile = Tile::<U8>::new(region, 1, &left_data);
        let right_tile = Tile::<U8>::new(region, 1, &right_data);

        let combined = <MeasureOp as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &region),
            reducer.reduce_tile(&right_tile, &region),
        );
        let result = <MeasureOp as TileReducer<U8>>::finalize(&reducer, combined);
        assert_eq!(result.mean, vec![25.0]);
    }

    #[test]
    fn accumulate_into_matches_reduce_tile_combine_across_tiles() {
        let reducer = MeasureOp::new(2);
        let region = Region::new(0, 0, 2, 1);
        let left_data = vec![10u8, 100, 20, 120];
        let right_data = vec![30u8, 140, 40, 160];
        let left_tile = Tile::<U8>::new(region, 2, &left_data);
        let right_tile = Tile::<U8>::new(region, 2, &right_data);

        let combined = <MeasureOp as TileReducer<U8>>::combine(
            &reducer,
            reducer.reduce_tile(&left_tile, &region),
            reducer.reduce_tile(&right_tile, &region),
        );
        let expected = <MeasureOp as TileReducer<U8>>::finalize(&reducer, combined);

        let mut scratch = vec![999.0, 888.0, 777.0];
        let mut partial = None;
        reducer.accumulate_into(&left_tile, &region, &mut scratch, &mut partial);
        assert_eq!(scratch, vec![30.0, 220.0]);

        reducer.accumulate_into(&right_tile, &region, &mut scratch, &mut partial);
        assert_eq!(scratch, vec![70.0, 300.0]);

        let accumulated =
            <MeasureOp as TileReducer<U8>>::finalize(&reducer, partial.expect("partial exists"));
        assert_eq!(accumulated, expected);
    }

    #[test]
    fn accumulate_into_reuses_partial_storage_after_first_tile() {
        let stats = crate::test_support::run_alloc_stats_child(ALLOC_CHILD_TEST, ALLOC_ENV);

        assert_eq!(
            stats.alloc_count, 0,
            "MeasureOp::accumulate_into should reuse its partial buffer after warmup: {stats:?}"
        );
    }

    #[test]
    fn accumulate_into_reuses_partial_storage_after_first_tile_child() {
        if !crate::test_support::should_run_alloc_stats_child(ALLOC_ENV) {
            return;
        }

        crate::test_support::emit_alloc_stats(measure_accumulate_into_alloc_stats());
    }

    #[test]
    fn reduce_tile_tracks_multiband_ordering() {
        let reducer = MeasureOp::new(3);
        let region = Region::new(0, 0, 2, 1);
        let data = vec![10u8, 1, 100, 20, 2, 200];
        let tile = Tile::<U8>::new(region, 3, &data);

        let partial = reducer.reduce_tile(&tile, &region);

        assert_eq!(partial.sums, vec![30.0, 3.0, 300.0]);
        assert_eq!(partial.count, 2);
    }

    #[test]
    fn finalize_returns_zeroes_for_zero_count_partial() {
        let reducer = MeasureOp::new(2);
        let partial = MeasurePartial {
            sums: vec![123.0, 456.0],
            count: 0,
        };

        let result = <MeasureOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(result.rows, 1);
        assert_eq!(result.cols, 1);
        assert_eq!(result.mean, vec![0.0, 0.0]);
    }
}
