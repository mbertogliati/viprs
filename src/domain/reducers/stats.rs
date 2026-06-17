//! Reducers and helpers for per-band image statistics.

use crate::{
    domain::reducer::TileReducer,
    domain::{
        error::ViprsError,
        format::BandFormat,
        image::{Image, Region, Tile},
        stats::ImageStats,
    },
};

/// Intermediate accumulator for per-band statistics.
///
/// Tracks min, max, sum, and sum-of-squares per band without heap allocation
/// per pixel. Final mean and stddev are derived in `finalize`.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::StatsReducer,
/// };
///
/// let reducer = StatsReducer::new(1);
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 1, &[7]);
/// let partial = reducer.reduce_tile(&tile, &region);
///
/// assert_eq!(partial.per_band.len(), 1);
/// ```
#[derive(Clone)]
pub struct PartialStats {
    /// One entry per band.
    pub per_band: Vec<BandPartial>,
}

/// Stores the running statistics for one output band.
///
/// This type solves stable parallel aggregation by carrying the extrema and moment sums needed
/// to derive the final mean and population standard deviation.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::StatsReducer,
/// };
///
/// let reducer = StatsReducer::new(1);
/// let region = Region::new(0, 0, 1, 1);
/// let tile = Tile::<U8>::new(region, 1, &[42]);
/// let partial = reducer.reduce_tile(&tile, &region);
///
/// assert_eq!(partial.per_band[0].count, 1);
/// ```
#[derive(Clone)]
pub struct BandPartial {
    /// Stores the `min` value for this item.
    pub min: f64,
    /// Stores the `max` value for this item.
    pub max: f64,
    /// Stores the `sum` value for this item.
    pub sum: f64,
    /// Stores the `sum_sq` value for this item.
    pub sum_sq: f64,
    /// Stores the `count` value for this item.
    pub count: u64,
}

impl BandPartial {
    const fn identity() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }
}

/// Computes per-band `min`, `max`, `mean`, and `stddev` over an image.
///
/// Works for any `F: BandFormat` where `F::Sample` converts to `f64`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::StatsReducer,
/// };
///
/// let reducer = StatsReducer::new(1);
/// let region = Region::new(0, 0, 3, 1);
/// let tile = Tile::<U8>::new(region, 1, &[0, 10, 20]);
/// let stats = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(stats.mean, vec![10.0]);
/// ```
pub struct StatsReducer {
    bands: u32,
}

impl StatsReducer {
    /// Creates a reducer that computes per-band summary statistics.
    ///
    /// This constructor fixes the band count up front so tile reductions can aggregate into a
    /// stable per-band layout with no runtime shape discovery.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::StatsReducer;
    ///
    /// let reducer = StatsReducer::new(3);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub const fn new(bands: u32) -> Self {
        Self { bands }
    }
}

fn stats_for_image<F>(image: &Image<F>) -> Result<ImageStats, ViprsError>
where
    F: BandFormat,
    F::Sample: Into<f64> + Copy,
{
    if image.bands() == 0 {
        return Err(ViprsError::Scheduler(
            "stats reducers require at least one band".into(),
        ));
    }

    let reducer = StatsReducer::new(image.bands());
    let region = Region::new(0, 0, image.width(), image.height());
    let tile = Tile::<F>::new(region, image.bands(), image.pixels());

    Ok(<StatsReducer as TileReducer<F>>::finalize(
        &reducer,
        <StatsReducer as TileReducer<F>>::reduce_tile(&reducer, &tile, &region),
    ))
}

/// Computes the per-band arithmetic mean for a materialized image.
///
/// This helper solves the common “average intensity per band” query without requiring callers to
/// manually instantiate a reducer or unpack the full statistics payload.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::Image,
///     reducers::image_avg,
/// };
///
/// let image = Image::<U8>::from_buffer(2, 1, 1, vec![10, 30]).unwrap();
/// let mean = image_avg(&image).unwrap();
///
/// assert_eq!(mean, vec![20.0]);
/// ```
pub fn image_avg<F>(image: &Image<F>) -> Result<Vec<f64>, ViprsError>
where
    F: BandFormat,
    F::Sample: Into<f64> + Copy,
{
    Ok(stats_for_image(image)?.mean)
}

/// Computes the per-band minimum sample value for a materialized image.
///
/// This helper solves fast low-end range inspection when callers only need minima and do not want
/// to inspect the rest of the statistics bundle.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::Image,
///     reducers::image_min,
/// };
///
/// let image = Image::<U8>::from_buffer(3, 1, 1, vec![9, 4, 7]).unwrap();
/// let min = image_min(&image).unwrap();
///
/// assert_eq!(min, vec![4.0]);
/// ```
pub fn image_min<F>(image: &Image<F>) -> Result<Vec<f64>, ViprsError>
where
    F: BandFormat,
    F::Sample: Into<f64> + Copy,
{
    Ok(stats_for_image(image)?.min)
}

/// Computes the per-band maximum sample value for a materialized image.
///
/// This helper solves peak-value inspection while reusing the same reduction path that powers the
/// broader statistics API.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::Image,
///     reducers::image_max,
/// };
///
/// let image = Image::<U8>::from_buffer(3, 1, 1, vec![9, 4, 7]).unwrap();
/// let max = image_max(&image).unwrap();
///
/// assert_eq!(max, vec![9.0]);
/// ```
pub fn image_max<F>(image: &Image<F>) -> Result<Vec<f64>, ViprsError>
where
    F: BandFormat,
    F::Sample: Into<f64> + Copy,
{
    Ok(stats_for_image(image)?.max)
}

/// Computes the per-band population standard deviation for a materialized image.
///
/// This helper solves spread estimation for already materialized images without forcing callers to
/// derive standard deviations from the lower-level sum and mean values themselves.
///
/// # Examples
/// ```rust
/// use viprs::domain::{
///     format::U8,
///     image::Image,
///     reducers::image_deviate,
/// };
///
/// let image = Image::<U8>::from_buffer(2, 1, 1, vec![0, 2]).unwrap();
/// let stddev = image_deviate(&image).unwrap();
///
/// assert_eq!(stddev, vec![1.0]);
/// ```
pub fn image_deviate<F>(image: &Image<F>) -> Result<Vec<f64>, ViprsError>
where
    F: BandFormat,
    F::Sample: Into<f64> + Copy,
{
    Ok(stats_for_image(image)?.stddev)
}

impl<F: BandFormat> TileReducer<F> for StatsReducer
where
    F::Sample: Into<f64> + Copy,
{
    type Partial = PartialStats;
    type Output = ImageStats;
    /// Pre-allocated per-band scratch. Reused across tiles on the same thread.
    type Scratch = Vec<BandPartial>;

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> PartialStats {
        let bands = self.bands as usize;
        let mut per_band: Vec<BandPartial> = (0..bands).map(|_| BandPartial::identity()).collect();

        for (i, &sample) in tile.data.iter().enumerate() {
            let band_idx = i % bands;
            let v: f64 = sample.into();
            let b = &mut per_band[band_idx];
            if v < b.min {
                b.min = v;
            }
            if v > b.max {
                b.max = v;
            }
            b.sum += v;
            b.sum_sq = v.mul_add(v, b.sum_sq);
            b.count += 1;
        }

        PartialStats { per_band }
    }

    /// Zero-allocation tile accumulation using pre-allocated per-band scratch.
    ///
    /// On the first call, `scratch` is an empty `Vec`; it is resized to `bands`
    /// entries of identity values. Subsequent calls reset each entry to identity
    /// and accumulate, reusing the existing allocation.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        _region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        let bands = self.bands as usize;

        // Resize only on first call; noop if capacity already matches.
        scratch.resize_with(bands, BandPartial::identity);
        // Reset each band's accumulators to identity without deallocating.
        for entry in scratch.iter_mut() {
            *entry = BandPartial::identity();
        }

        for (i, &sample) in tile.data.iter().enumerate() {
            let band_idx = i % bands;
            let v: f64 = sample.into();
            let b = &mut scratch[band_idx];
            if v < b.min {
                b.min = v;
            }
            if v > b.max {
                b.max = v;
            }
            b.sum += v;
            b.sum_sq = v.mul_add(v, b.sum_sq);
            b.count += 1;
        }

        let tile_partial = PartialStats {
            per_band: scratch.clone(),
        };

        *partial = Some(match partial.take() {
            Some(existing) => <Self as TileReducer<F>>::combine(self, existing, tile_partial),
            None => tile_partial,
        });
    }

    fn combine(&self, mut a: PartialStats, b: PartialStats) -> PartialStats {
        for (ab, bb) in a.per_band.iter_mut().zip(b.per_band.iter()) {
            if bb.min < ab.min {
                ab.min = bb.min;
            }
            if bb.max > ab.max {
                ab.max = bb.max;
            }
            ab.sum += bb.sum;
            ab.sum_sq += bb.sum_sq;
            ab.count += bb.count;
        }
        a
    }

    fn finalize(&self, combined: PartialStats) -> ImageStats {
        let bands = self.bands;
        let mut min = Vec::with_capacity(bands as usize);
        let mut max = Vec::with_capacity(bands as usize);
        let mut mean = Vec::with_capacity(bands as usize);
        let mut stddev = Vec::with_capacity(bands as usize);

        for b in &combined.per_band {
            let n = b.count as f64;
            let m = if n > 0.0 { b.sum / n } else { 0.0 };
            // Population variance: E[x²] - (E[x])²
            let variance = if n > 0.0 {
                (b.sum_sq / n) - (m * m)
            } else {
                0.0
            };
            min.push(b.min);
            max.push(b.max);
            mean.push(m);
            stddev.push(variance.max(0.0).sqrt());
        }

        ImageStats {
            bands,
            min,
            max,
            mean,
            stddev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        error::ViprsError,
        format::U8,
        image::{Image, Region},
    };

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            assert!((lhs - rhs).abs() < 1e-9, "expected {rhs}, got {lhs}");
        }
    }

    #[test]
    fn stats_reducer_single_band_known_values() {
        // 1×4 image, single band: [0, 100, 200, 100]
        let region = Region::new(0, 0, 4, 1);
        let data = vec![0u8, 100, 200, 100];
        let tile = Tile::<U8>::new(region, 1, &data);
        let reducer = StatsReducer::new(1);
        let partial = <StatsReducer as crate::domain::reducer::TileReducer<U8>>::reduce_tile(
            &reducer, &tile, &region,
        );
        let stats =
            <StatsReducer as crate::domain::reducer::TileReducer<U8>>::finalize(&reducer, partial);

        assert!((stats.min[0] - 0.0).abs() < f64::EPSILON);
        assert!((stats.max[0] - 200.0).abs() < f64::EPSILON);
        assert!((stats.mean[0] - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn image_avg_single_band_image() {
        let image = Image::<U8>::from_buffer(4, 1, 1, vec![0, 100, 200, 100]).unwrap();

        let avg = image_avg(&image).unwrap();

        assert_close(&avg, &[100.0]);
    }

    #[test]
    fn image_avg_multi_band_image() {
        let image = Image::<U8>::from_buffer(3, 1, 2, vec![10, 100, 20, 200, 30, 50]).unwrap();

        let avg = image_avg(&image).unwrap();

        assert_close(&avg, &[20.0, 350.0 / 3.0]);
    }

    #[test]
    fn image_min_single_band_image() {
        let image = Image::<U8>::from_buffer(4, 1, 1, vec![0, 100, 200, 100]).unwrap();

        let min = image_min(&image).unwrap();

        assert_close(&min, &[0.0]);
    }

    #[test]
    fn image_min_multi_band_image() {
        let image = Image::<U8>::from_buffer(3, 1, 2, vec![10, 100, 20, 200, 30, 50]).unwrap();

        let min = image_min(&image).unwrap();

        assert_close(&min, &[10.0, 50.0]);
    }

    #[test]
    fn image_max_single_band_image() {
        let image = Image::<U8>::from_buffer(4, 1, 1, vec![0, 100, 200, 100]).unwrap();

        let max = image_max(&image).unwrap();

        assert_close(&max, &[200.0]);
    }

    #[test]
    fn image_max_multi_band_image() {
        let image = Image::<U8>::from_buffer(3, 1, 2, vec![10, 100, 20, 200, 30, 50]).unwrap();

        let max = image_max(&image).unwrap();

        assert_close(&max, &[30.0, 200.0]);
    }

    #[test]
    fn image_deviate_single_band_image() {
        let image = Image::<U8>::from_buffer(4, 1, 1, vec![0, 100, 200, 100]).unwrap();

        let stddev = image_deviate(&image).unwrap();

        assert_close(&stddev, &[5_000.0_f64.sqrt()]);
    }

    #[test]
    fn image_deviate_multi_band_image() {
        let image = Image::<U8>::from_buffer(3, 1, 2, vec![10, 100, 20, 200, 30, 50]).unwrap();

        let stddev = image_deviate(&image).unwrap();

        assert_close(
            &stddev,
            &[(200.0_f64 / 3.0).sqrt(), (35_000.0_f64 / 9.0).sqrt()],
        );
    }

    #[test]
    fn image_avg_rejects_zero_band_images() {
        let image = Image::<U8>::from_buffer(1, 1, 0, Vec::new()).unwrap();

        let error = image_avg(&image).unwrap_err();

        assert!(
            matches!(error, ViprsError::Scheduler(message) if message == "stats reducers require at least one band")
        );
    }
}
