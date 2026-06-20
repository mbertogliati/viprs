use viprs_core::{
    format::BandFormat,
    image::{Region, Tile},
    reducer::TileReducer,
    shared_ops::sample_conv::ToF64,
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
/// use viprs_ops_pixel::arithmetic::measure::MeasureOp;
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
    use viprs_core::{format::U8, reducer::TileReducer};

    // Allocation tests require the root crate test_support (global allocator).
    // Run via: cargo test -p viprs --lib

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
