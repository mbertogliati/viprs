use crate::domain::{
    format::BandFormat,
    image::{Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents an extrema result.
pub struct ExtremaResult {
    /// Value associated with this item.
    pub value: f64,
    /// Horizontal factor associated with this condition.
    pub x: u32,
    /// Vertical factor associated with this condition.
    pub y: u32,
    /// Stores the `band` value for this item.
    pub band: u32,
}

#[derive(Debug, Clone, Copy)]
/// Represents an extrema partial.
pub struct ExtremaPartial {
    found: bool,
    value: f64,
    x: u32,
    y: u32,
    band: u32,
}

impl ExtremaPartial {
    const fn empty_min() -> Self {
        Self {
            found: false,
            value: f64::INFINITY,
            x: 0,
            y: 0,
            band: 0,
        }
    }

    const fn empty_max() -> Self {
        Self {
            found: false,
            value: f64::NEG_INFINITY,
            x: 0,
            y: 0,
            band: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
/// Represents a scalar stats.
pub struct ScalarStats {
    /// Stores the `sum` value for this item.
    pub sum: f64,
    /// Stores the `sum_sq` value for this item.
    pub sum_sq: f64,
    /// Stores the `count` value for this item.
    pub count: u64,
}

impl ScalarStats {
    const fn empty() -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }

    fn combine(mut self, rhs: Self) -> Self {
        self.sum += rhs.sum;
        self.sum_sq += rhs.sum_sq;
        self.count += rhs.count;
        self
    }
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::AvgOp;
///
/// let op = AvgOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct AvgOp;

impl AvgOp {
    #[must_use]
    /// Creates a new `AvgOp`.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for AvgOp {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> TileReducer<F> for AvgOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ScalarStats;
    type Output = f64;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> Self::Partial {
        let mut partial = ScalarStats::empty();
        for &sample in tile.data {
            let value = sample.to_f64();
            partial.sum += value;
            partial.count += 1;
        }
        partial
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        a.combine(b)
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        if combined.count == 0 {
            0.0
        } else {
            combined.sum / combined.count as f64
        }
    }
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::DeviateOp;
///
/// let op = DeviateOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct DeviateOp;

impl DeviateOp {
    #[must_use]
    /// Creates a new `DeviateOp`.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for DeviateOp {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> TileReducer<F> for DeviateOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ScalarStats;
    type Output = f64;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, _region: &Region) -> Self::Partial {
        let mut partial = ScalarStats::empty();
        for &sample in tile.data {
            let value = sample.to_f64();
            partial.sum += value;
            partial.sum_sq = value.mul_add(value, partial.sum_sq);
            partial.count += 1;
        }
        partial
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        a.combine(b)
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        if combined.count <= 1 {
            return 0.0;
        }

        let count = combined.count as f64;
        (f64::abs(combined.sum_sq - (combined.sum * combined.sum / count)) / (count - 1.0)).sqrt()
    }
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::MinOp;
///
/// let op = MinOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MinOp;

impl MinOp {
    #[must_use]
    /// Creates a new `MinOp`.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for MinOp {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> TileReducer<F> for MinOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ExtremaPartial;
    type Output = Option<ExtremaResult>;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = ExtremaPartial::empty_min();
        let bands = tile.bands as usize;
        let width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..width {
                let base = (row * width + col) * bands;
                for band in 0..bands {
                    let value = tile.data[base + band].to_f64();
                    if value.is_nan() {
                        continue;
                    }
                    let x = region.x.saturating_add(col as i32) as u32;
                    let y = region.y.saturating_add(row as i32) as u32;
                    if !partial.found
                        || value < partial.value
                        || (value == partial.value && tie_before(x, y, partial))
                    {
                        partial = ExtremaPartial {
                            found: true,
                            value,
                            x,
                            y,
                            band: band as u32,
                        };
                    }
                }
            }
        }

        partial
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        match (a.found, b.found) {
            (false | true, false) => a,
            (false, true) => b,
            (true, true) => {
                if b.value < a.value || (b.value == a.value && tie_before(b.x, b.y, a)) {
                    b
                } else {
                    a
                }
            }
        }
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        combined.found.then_some(ExtremaResult {
            value: combined.value,
            x: combined.x,
            y: combined.y,
            band: combined.band,
        })
    }
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::MaxOp;
///
/// let op = MaxOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MaxOp;

impl MaxOp {
    #[must_use]
    /// Creates a new `MaxOp`.
    pub const fn new() -> Self {
        Self
    }
}

impl Default for MaxOp {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> TileReducer<F> for MaxOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = ExtremaPartial;
    type Output = Option<ExtremaResult>;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut partial = ExtremaPartial::empty_max();
        let bands = tile.bands as usize;
        let width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..width {
                let base = (row * width + col) * bands;
                for band in 0..bands {
                    let value = tile.data[base + band].to_f64();
                    if value.is_nan() {
                        continue;
                    }
                    let x = region.x.saturating_add(col as i32) as u32;
                    let y = region.y.saturating_add(row as i32) as u32;
                    if !partial.found
                        || value > partial.value
                        || (value == partial.value && tie_before(x, y, partial))
                    {
                        partial = ExtremaPartial {
                            found: true,
                            value,
                            x,
                            y,
                            band: band as u32,
                        };
                    }
                }
            }
        }

        partial
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        match (a.found, b.found) {
            (false | true, false) => a,
            (false, true) => b,
            (true, true) => {
                if b.value > a.value || (b.value == a.value && tie_before(b.x, b.y, a)) {
                    b
                } else {
                    a
                }
            }
        }
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        combined.found.then_some(ExtremaResult {
            value: combined.value,
            x: combined.x,
            y: combined.y,
            band: combined.band,
        })
    }
}

const fn tie_before(x: u32, y: u32, current: ExtremaPartial) -> bool {
    y < current.y || (y == current.y && x <= current.x)
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a stats row.
pub struct StatsRow {
    /// Stores the `min` value for this item.
    pub min: f64,
    /// Stores the `max` value for this item.
    pub max: f64,
    /// Stores the `sum` value for this item.
    pub sum: f64,
    /// Stores the `sum_sq` value for this item.
    pub sum_sq: f64,
    /// Stores the `avg` value for this item.
    pub avg: f64,
    /// Stores the `stddev` value for this item.
    pub stddev: f64,
    /// Stores the `x_min` value for this item.
    pub x_min: u32,
    /// Stores the `y_min` value for this item.
    pub y_min: u32,
    /// Stores the `x_max` value for this item.
    pub x_max: u32,
    /// Stores the `y_max` value for this item.
    pub y_max: u32,
}

#[derive(Debug, Clone, PartialEq)]
/// Represents a stats result.
pub struct StatsResult {
    /// Number of bands associated with this item.
    pub bands: u32,
    /// Number of rows associated with this configuration.
    pub rows: Vec<StatsRow>,
}

#[derive(Clone)]
/// Represents a stats partial.
pub struct StatsPartial {
    rows: Vec<StatsAccumulator>,
}

#[derive(Clone)]
struct StatsAccumulator {
    found: bool,
    min: f64,
    max: f64,
    sum: f64,
    sum_sq: f64,
    count: u64,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
}

impl StatsAccumulator {
    const fn empty() -> Self {
        Self {
            found: false,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            x_min: 0,
            y_min: 0,
            x_max: 0,
            y_max: 0,
        }
    }

    fn add(&mut self, value: f64, x: u32, y: u32) {
        if value.is_nan() {
            return;
        }

        if !self.found || value < self.min || (value == self.min && tie_before_min(x, y, self)) {
            self.min = value;
            self.x_min = x;
            self.y_min = y;
        }
        if !self.found || value > self.max || (value == self.max && tie_before_max(x, y, self)) {
            self.max = value;
            self.x_max = x;
            self.y_max = y;
        }
        self.found = true;
        self.sum += value;
        self.sum_sq = value.mul_add(value, self.sum_sq);
        self.count += 1;
    }

    fn combine(mut self, rhs: Self) -> Self {
        if !rhs.found {
            return self;
        }
        if !self.found {
            return rhs;
        }
        if rhs.min < self.min
            || (rhs.min == self.min && tie_before_min(rhs.x_min, rhs.y_min, &self))
        {
            self.min = rhs.min;
            self.x_min = rhs.x_min;
            self.y_min = rhs.y_min;
        }
        if rhs.max > self.max
            || (rhs.max == self.max && tie_before_max(rhs.x_max, rhs.y_max, &self))
        {
            self.max = rhs.max;
            self.x_max = rhs.x_max;
            self.y_max = rhs.y_max;
        }
        self.sum += rhs.sum;
        self.sum_sq += rhs.sum_sq;
        self.count += rhs.count;
        self
    }

    fn finalize(&self, divisor: u64) -> StatsRow {
        let avg = if divisor == 0 {
            0.0
        } else {
            self.sum / divisor as f64
        };
        let stddev = if divisor <= 1 {
            0.0
        } else {
            (f64::abs(self.sum_sq - (self.sum * self.sum / divisor as f64))
                / (divisor as f64 - 1.0))
                .sqrt()
        };

        StatsRow {
            min: self.min,
            max: self.max,
            sum: self.sum,
            sum_sq: self.sum_sq,
            avg,
            stddev,
            x_min: self.x_min,
            y_min: self.y_min,
            x_max: self.x_max,
            y_max: self.y_max,
        }
    }
}

const fn tie_before_min(x: u32, y: u32, current: &StatsAccumulator) -> bool {
    y < current.y_min || (y == current.y_min && x <= current.x_min)
}

const fn tie_before_max(x: u32, y: u32, current: &StatsAccumulator) -> bool {
    y < current.y_max || (y == current.y_max && x <= current.x_max)
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::StatsOp;
///
/// let op = StatsOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct StatsOp {
    bands: u32,
}

impl StatsOp {
    #[must_use]
    /// Creates a new `StatsOp`.
    pub fn new(bands: u32) -> Self {
        debug_assert!(bands > 0, "StatsOp: bands must be at least 1");
        Self { bands }
    }
}

impl<F> TileReducer<F> for StatsOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = StatsPartial;
    type Output = StatsResult;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let bands = self.bands as usize;
        let mut rows = vec![StatsAccumulator::empty(); bands + 1];
        let width = region.width as usize;

        for row in 0..region.height as usize {
            for col in 0..width {
                let x = region.x.saturating_add(col as i32) as u32;
                let y = region.y.saturating_add(row as i32) as u32;
                let base = (row * width + col) * bands;
                for band in 0..bands {
                    let value = tile.data[base + band].to_f64();
                    rows[band + 1].add(value, x, y);
                    rows[0].add(value, x, y);
                }
            }
        }

        StatsPartial { rows }
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        let rows = a
            .rows
            .into_iter()
            .zip(b.rows)
            .map(|(left, right)| left.combine(right))
            .collect();
        StatsPartial { rows }
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        let rows = combined
            .rows
            .iter()
            .map(|row| row.finalize(row.count))
            .collect();

        StatsResult {
            bands: self.bands,
            rows,
        }
    }
}

/// Applies the `reduce facades` arithmetic operation to image samples. It preserves image
/// geometry while updating sample values according to the operation parameters.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::reduce_facades::GetpointOp;
///
/// let op = GetpointOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct GetpointOp {
    x: u32,
    y: u32,
}

impl GetpointOp {
    #[must_use]
    /// Creates a new `GetpointOp`.
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }
}

impl<F> TileReducer<F> for GetpointOp
where
    F: BandFormat,
    F::Sample: ToF64,
{
    type Partial = Option<Vec<f64>>;
    type Output = Vec<f64>;
    type Scratch = ();

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let left = i64::from(region.x);
        let top = i64::from(region.y);
        let right = left + i64::from(region.width);
        let bottom = top + i64::from(region.height);
        let x = i64::from(self.x);
        let y = i64::from(self.y);

        if x < left || x >= right || y < top || y >= bottom {
            return None;
        }

        let local_x = (x - left) as usize;
        let local_y = (y - top) as usize;
        let bands = tile.bands as usize;
        let offset = (local_y * region.width as usize + local_x) * bands;
        Some(
            tile.data[offset..offset + bands]
                .iter()
                .map(|sample| sample.to_f64())
                .collect(),
        )
    }

    fn combine(&self, a: Self::Partial, b: Self::Partial) -> Self::Partial {
        a.or(b)
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        combined.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8},
        reducer::TileReducer,
    };

    #[test]
    fn avg_matches_all_samples() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[10, 20, 30, 40]);
        let reducer = AvgOp::new();
        let partial = reducer.reduce_tile(&tile, &region);
        assert_eq!(
            <AvgOp as TileReducer<U8>>::finalize(&reducer, partial),
            25.0
        );
    }

    #[test]
    fn avg_empty_tile_finalizes_to_zero() {
        let region = Region::new(0, 0, 0, 0);
        let tile = Tile::<U8>::new(region, 1, &[]);
        let reducer = AvgOp::new();
        let partial = reducer.reduce_tile(&tile, &region);
        assert_eq!(<AvgOp as TileReducer<U8>>::finalize(&reducer, partial), 0.0);
    }

    #[test]
    fn deviate_uses_sample_standard_deviation() {
        let region = Region::new(0, 0, 4, 1);
        let tile = Tile::<U8>::new(region, 1, &[2, 4, 4, 4]);
        let reducer = DeviateOp::new();
        let partial = reducer.reduce_tile(&tile, &region);
        let out = <DeviateOp as TileReducer<U8>>::finalize(&reducer, partial);
        assert!((out - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn min_and_max_report_coordinates() {
        let region = Region::new(10, 20, 3, 1);
        let tile = Tile::<U8>::new(region, 1, &[9, 2, 12]);
        let min = MinOp::new();
        let max = MaxOp::new();
        let min_partial = min.reduce_tile(&tile, &region);
        let max_partial = max.reduce_tile(&tile, &region);

        assert_eq!(
            <MinOp as TileReducer<U8>>::finalize(&min, min_partial),
            Some(ExtremaResult {
                value: 2.0,
                x: 11,
                y: 20,
                band: 0,
            })
        );
        assert_eq!(
            <MaxOp as TileReducer<U8>>::finalize(&max, max_partial),
            Some(ExtremaResult {
                value: 12.0,
                x: 12,
                y: 20,
                band: 0,
            })
        );
    }

    #[test]
    fn stats_row_zero_is_global_and_following_rows_are_per_band() {
        let region = Region::new(0, 0, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[1, 10, 3, 30]);
        let reducer = StatsOp::new(2);
        let partial = reducer.reduce_tile(&tile, &region);
        let stats = <StatsOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(stats.rows.len(), 3);
        assert_eq!(stats.rows[0].sum, 44.0);
        assert_eq!(stats.rows[1].sum, 4.0);
        assert_eq!(stats.rows[2].sum, 40.0);
        assert_eq!(stats.rows[1].x_min, 0);
        assert_eq!(stats.rows[2].x_max, 1);
    }

    #[test]
    fn stats_rows_expose_full_libvips_column_set() {
        let region = Region::new(10, 20, 2, 1);
        let tile = Tile::<U8>::new(region, 2, &[1, 10, 3, 30]);
        let reducer = StatsOp::new(2);
        let partial = reducer.reduce_tile(&tile, &region);
        let stats = <StatsOp as TileReducer<U8>>::finalize(&reducer, partial);

        assert_eq!(
            stats.rows[0],
            StatsRow {
                min: 1.0,
                max: 30.0,
                sum: 44.0,
                sum_sq: 1_010.0,
                avg: 11.0,
                stddev: (526.0f64 / 3.0).sqrt(),
                x_min: 10,
                y_min: 20,
                x_max: 11,
                y_max: 20,
            }
        );
        assert_eq!(
            stats.rows[1],
            StatsRow {
                min: 1.0,
                max: 3.0,
                sum: 4.0,
                sum_sq: 10.0,
                avg: 2.0,
                stddev: 2.0f64.sqrt(),
                x_min: 10,
                y_min: 20,
                x_max: 11,
                y_max: 20,
            }
        );
        assert_eq!(
            stats.rows[2],
            StatsRow {
                min: 10.0,
                max: 30.0,
                sum: 40.0,
                sum_sq: 1_000.0,
                avg: 20.0,
                stddev: 200.0f64.sqrt(),
                x_min: 10,
                y_min: 20,
                x_max: 11,
                y_max: 20,
            }
        );
    }

    #[test]
    fn getpoint_returns_pixel_bands_as_f64() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let reducer = GetpointOp::new(1, 1);
        let partial = reducer.reduce_tile(&tile, &region);
        assert_eq!(
            <GetpointOp as TileReducer<U8>>::finalize(&reducer, partial),
            vec![7.0, 8.0]
        );
    }

    #[test]
    fn getpoint_reduce_tile_returns_none_for_out_of_bounds_coords() {
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
        let reducer = GetpointOp::new(2, 1);
        assert_eq!(reducer.reduce_tile(&tile, &region), None);
    }

    #[test]
    fn avg_and_deviate_combine_tile_partials() {
        let left_region = Region::new(0, 0, 2, 1);
        let right_region = Region::new(2, 0, 2, 1);
        let left = Tile::<U8>::new(left_region, 1, &[2, 4]);
        let right = Tile::<U8>::new(right_region, 1, &[4, 4]);

        let avg = AvgOp::new();
        let avg_partial = <AvgOp as TileReducer<U8>>::combine(
            &avg,
            avg.reduce_tile(&left, &left_region),
            avg.reduce_tile(&right, &right_region),
        );
        assert_eq!(<AvgOp as TileReducer<U8>>::finalize(&avg, avg_partial), 3.5);

        let deviate = DeviateOp::new();
        let dev_partial = <DeviateOp as TileReducer<U8>>::combine(
            &deviate,
            deviate.reduce_tile(&left, &left_region),
            deviate.reduce_tile(&right, &right_region),
        );
        assert!(
            (<DeviateOp as TileReducer<U8>>::finalize(&deviate, dev_partial) - 1.0).abs()
                < f64::EPSILON
        );
        assert_eq!(
            <DeviateOp as TileReducer<U8>>::finalize(
                &deviate,
                ScalarStats {
                    sum: 9.0,
                    sum_sq: 81.0,
                    count: 1,
                }
            ),
            0.0
        );
    }

    #[test]
    fn min_and_max_ignore_nan_and_break_ties_by_earlier_coordinates() {
        let region = Region::new(4, 7, 2, 2);
        let tile = Tile::<F32>::new(region, 1, &[f32::NAN, 2.0, 2.0, 9.0]);
        let min = MinOp::new();
        let max = MaxOp::new();

        assert_eq!(
            <MinOp as TileReducer<F32>>::finalize(&min, min.reduce_tile(&tile, &region)),
            Some(ExtremaResult {
                value: 2.0,
                x: 5,
                y: 7,
                band: 0,
            })
        );
        assert_eq!(
            <MaxOp as TileReducer<F32>>::finalize(&max, max.reduce_tile(&tile, &region)),
            Some(ExtremaResult {
                value: 9.0,
                x: 5,
                y: 8,
                band: 0,
            })
        );
    }

    #[test]
    fn extrema_combine_handles_empty_and_tied_partials() {
        let min = MinOp::new();
        let max = MaxOp::new();
        let empty_min = ExtremaPartial::empty_min();
        let empty_max = ExtremaPartial::empty_max();
        let late = ExtremaPartial {
            found: true,
            value: 3.0,
            x: 8,
            y: 10,
            band: 0,
        };
        let early = ExtremaPartial {
            found: true,
            value: 3.0,
            x: 6,
            y: 9,
            band: 0,
        };

        assert_eq!(
            <MinOp as TileReducer<F32>>::combine(&min, empty_min, late).x,
            8
        );
        assert_eq!(<MinOp as TileReducer<F32>>::combine(&min, late, early).x, 6);
        assert_eq!(
            <MaxOp as TileReducer<F32>>::combine(&max, empty_max, late).x,
            8
        );
        assert_eq!(<MaxOp as TileReducer<F32>>::combine(&max, late, early).x, 6);
        assert_eq!(<MinOp as TileReducer<F32>>::finalize(&min, empty_min), None);
        assert_eq!(<MaxOp as TileReducer<F32>>::finalize(&max, empty_max), None);
    }

    #[test]
    fn stats_combine_merges_rows_and_empty_stats_finalize_to_zeroes() {
        let reducer = StatsOp::new(1);
        let left_region = Region::new(10, 20, 1, 1);
        let right_region = Region::new(11, 20, 1, 1);
        let left = Tile::<F32>::new(left_region, 1, &[2.0]);
        let right = Tile::<F32>::new(right_region, 1, &[2.0]);

        let merged = <StatsOp as TileReducer<F32>>::combine(
            &reducer,
            reducer.reduce_tile(&left, &left_region),
            reducer.reduce_tile(&right, &right_region),
        );
        let stats = <StatsOp as TileReducer<F32>>::finalize(&reducer, merged);
        assert_eq!(stats.rows[0].min, 2.0);
        assert_eq!(stats.rows[0].x_min, 10);
        assert_eq!(stats.rows[0].x_max, 10);
        assert_eq!(stats.rows[0].stddev, 0.0);

        let empty = StatsAccumulator::empty().finalize(0);
        assert_eq!(empty.avg, 0.0);
        assert_eq!(empty.stddev, 0.0);
    }

    #[test]
    fn getpoint_combine_prefers_first_hit_and_defaults_missing_points() {
        let reducer = GetpointOp::new(1, 1);
        assert_eq!(
            <GetpointOp as TileReducer<F32>>::combine(
                &reducer,
                Some(vec![1.0, 2.0]),
                Some(vec![3.0])
            ),
            Some(vec![1.0, 2.0])
        );
        assert_eq!(
            <GetpointOp as TileReducer<F32>>::combine(&reducer, None, Some(vec![3.0])),
            Some(vec![3.0])
        );
        assert_eq!(
            <GetpointOp as TileReducer<F32>>::finalize(&reducer, None),
            Vec::<f64>::new()
        );
    }
}
