//! Reducers for Hough-space line and circle voting.

use crate::domain::{
    error::{HoughError, ViprsError},
    format::{BandFormat, U32},
    image::{InMemoryImage, Region, Tile},
    ops::resample::sample_conv::ToF64,
    reducer::TileReducer,
};

/// Builds a Hough line accumulator image from thresholded input pixels.
///
/// This reducer solves straight-line detection by converting bright pixels into votes across
/// a `(theta, rho)` parameter space that can be searched for peaks later.
///
/// # Examples
/// ```ignore
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::HoughLineReducer,
/// };
///
/// let reducer = HoughLineReducer::new(32, 32, 4, 4, 0.0);
/// let region = Region::new(0, 0, 4, 4);
/// let tile = Tile::<U8>::new(region, 1, &[255; 16]);
/// let image = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(image.width(), 32);
/// ```
pub struct HoughLineReducer {
    width: usize,
    height: usize,
    input_width: u32,
    input_height: u32,
    threshold: f64,
    sin_lut: Vec<f64>,
}

impl HoughLineReducer {
    /// Creates a line-voting reducer with a fixed accumulator geometry.
    ///
    /// This constructor precomputes the sine lookup table needed during voting so each tile can
    /// transform edge pixels into Hough space without recomputing trigonometry.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HoughLineReducer;
    ///
    /// let reducer = HoughLineReducer::new(180, 64, 640, 480, 128.0);
    /// let _ = reducer;
    /// ```
    #[must_use]
    pub fn new(
        width: usize,
        height: usize,
        input_width: u32,
        input_height: u32,
        threshold: f64,
    ) -> Self {
        let mut sin_lut = vec![0.0; 2 * width];
        for (index, value) in sin_lut.iter_mut().enumerate() {
            *value = (2.0 * std::f64::consts::PI * index as f64 / (2 * width) as f64).sin();
        }

        Self {
            width,
            height,
            input_width,
            input_height,
            threshold,
            sin_lut,
        }
    }
}

impl<F> TileReducer<F> for HoughLineReducer
where
    F: BandFormat,
    F::Sample: ToF64 + Copy,
{
    type Partial = Vec<u32>;
    type Output = InMemoryImage<U32>;
    /// Pre-allocated Hough line accumulator. The full `width × height` buffer is
    /// allocated once per rayon thread and zeroed at the start of each tile, saving
    /// up to 46 KB per tile for a 180×64 accumulator.
    type Scratch = Vec<u32>;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut accumulator = vec![0u32; self.width * self.height];
        let diagonal = f64::from(self.input_width).hypot(f64::from(self.input_height));
        let bands = tile.bands as usize;
        let width_half_turn = self.width / 2;

        for local_y in 0..region.height as usize {
            let y = region.y + local_y as i32;
            if y < 0 || y >= self.input_height as i32 {
                continue;
            }

            for local_x in 0..region.width as usize {
                let x = region.x + local_x as i32;
                if x < 0 || x >= self.input_width as i32 {
                    continue;
                }

                let pixel_index = (local_y * region.width as usize + local_x) * bands;
                if tile.data[pixel_index].to_f64() <= self.threshold {
                    continue;
                }

                let xd = f64::from(x) / diagonal;
                let yd = f64::from(y) / diagonal;

                for theta in 0..self.width {
                    let theta_90 = theta + width_half_turn;
                    let rho = yd.mul_add(self.sin_lut[theta], xd * self.sin_lut[theta_90]);
                    let rho_index = ((rho + 1.0) * (self.height as f64 / 2.0)) as isize;

                    if (0..self.height as isize).contains(&rho_index) {
                        accumulator[theta + rho_index as usize * self.width] += 1;
                    }
                }
            }
        }

        accumulator
    }

    /// Zero-allocation tile accumulation using a pre-allocated `width × height` buffer.
    ///
    /// On first call, `scratch` is an empty `Vec`; it is resized to `width * height`
    /// (one alloc). All subsequent calls zero the scratch and accumulate votes in place
    /// without touching the allocator.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        scratch.resize(self.width * self.height, 0u32);
        scratch.fill(0u32);

        let diagonal = f64::from(self.input_width).hypot(f64::from(self.input_height));
        let bands = tile.bands as usize;
        let width_half_turn = self.width / 2;

        for local_y in 0..region.height as usize {
            let y = region.y + local_y as i32;
            if y < 0 || y >= self.input_height as i32 {
                continue;
            }

            for local_x in 0..region.width as usize {
                let x = region.x + local_x as i32;
                if x < 0 || x >= self.input_width as i32 {
                    continue;
                }

                let pixel_index = (local_y * region.width as usize + local_x) * bands;
                if tile.data[pixel_index].to_f64() <= self.threshold {
                    continue;
                }

                let xd = f64::from(x) / diagonal;
                let yd = f64::from(y) / diagonal;

                for theta in 0..self.width {
                    let theta_90 = theta + width_half_turn;
                    let rho = yd.mul_add(self.sin_lut[theta], xd * self.sin_lut[theta_90]);
                    let rho_index = ((rho + 1.0) * (self.height as f64 / 2.0)) as isize;

                    if (0..self.height as isize).contains(&rho_index) {
                        scratch[theta + rho_index as usize * self.width] += 1;
                    }
                }
            }
        }

        match partial {
            Some(acc) => {
                for (lhs, rhs) in acc.iter_mut().zip(scratch.iter()) {
                    *lhs += rhs;
                }
            }
            None => {
                *partial = Some(scratch.clone());
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (lhs, rhs) in a.iter_mut().zip(b.iter()) {
            *lhs += rhs;
        }
        a
    }

    fn finalize(&self, combined: Self::Partial) -> Self::Output {
        InMemoryImage::<U32>::from_buffer(self.width as u32, self.height as u32, 1, combined)
            .unwrap_or_else(|error| {
                debug_assert!(
                    false,
                    "hough line accumulator dimensions are internally consistent: {error}"
                );
                // SAFETY: `combined` is allocated from the same checked dimensions used to build the image shape.
                unsafe { std::hint::unreachable_unchecked() }
            })
    }
}

/// Builds a multi-band Hough accumulator for circle-center and radius voting.
///
/// This reducer solves circle detection by storing one accumulator band per candidate radius,
/// allowing later stages to search for both center position and radius peaks.
///
/// # Examples
/// ```ignore
/// use viprs::domain::{
///     format::U8,
///     image::{Region, Tile},
///     reducer::TileReducer,
///     reducers::HoughCircleReducer,
/// };
///
/// let reducer = HoughCircleReducer::new(1, 2, 4, 8, 8, 0.0).unwrap();
/// let region = Region::new(0, 0, 8, 8);
/// let tile = Tile::<U8>::new(region, 1, &[255; 64]);
/// let image = reducer.finalize(reducer.reduce_tile(&tile, &region));
///
/// assert_eq!(image.height(), 8);
/// ```
pub struct HoughCircleReducer {
    scale: u32,
    min_radius: u32,
    max_radius: u32,
    width: usize,
    height: usize,
    bands: usize,
    threshold: f64,
}

impl HoughCircleReducer {
    /// Creates a circle-voting reducer for the requested radius range.
    ///
    /// This constructor validates the radius configuration and derives the accumulator layout
    /// that later tile reductions reuse without additional shape checks.
    ///
    /// # Examples
    /// ```rust
    /// use viprs::domain::reducers::HoughCircleReducer;
    ///
    /// let reducer = HoughCircleReducer::new(2, 4, 10, 64, 64, 32.0).unwrap();
    /// let _ = reducer;
    /// ```
    pub fn new(
        scale: u32,
        min_radius: u32,
        max_radius: u32,
        input_width: u32,
        input_height: u32,
        threshold: f64,
    ) -> Result<Self, ViprsError> {
        if scale == 0 {
            return Err(HoughError::ZeroScale.into());
        }
        if max_radius <= min_radius {
            return Err(HoughError::InvalidRadiusRange {
                min_radius,
                max_radius,
            }
            .into());
        }

        let range = max_radius - min_radius;
        Ok(Self {
            scale,
            min_radius,
            max_radius,
            width: (input_width / scale) as usize,
            height: (input_height / scale) as usize,
            bands: (1 + range / scale) as usize,
            threshold,
        })
    }

    #[inline(always)]
    fn vote_point(&self, accumulator: &mut [u32], band: usize, x: i32, y: i32) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }

        let pixel_index = (y as usize * self.width + x as usize) * self.bands + band;
        accumulator[pixel_index] += 1;
    }

    #[inline(always)]
    fn vote_endpoints(&self, accumulator: &mut [u32], band: usize, y: i32, x1: i32, x2: i32) {
        self.vote_point(accumulator, band, x1, y);
        self.vote_point(accumulator, band, x2, y);
    }

    fn vote_circle_band(
        &self,
        accumulator: &mut [u32],
        center_x: i32,
        center_y: i32,
        radius: i32,
        band: usize,
    ) {
        let mut y = radius;
        let mut d = 3_i64 - 2_i64 * i64::from(radius);
        let mut x = 0_i32;

        while x < y {
            self.vote_endpoints(accumulator, band, center_y + y, center_x - x, center_x + x);
            self.vote_endpoints(accumulator, band, center_y - y, center_x - x, center_x + x);
            self.vote_endpoints(accumulator, band, center_y + x, center_x - y, center_x + y);
            self.vote_endpoints(accumulator, band, center_y - x, center_x - y, center_x + y);

            if d < 0 {
                d += i64::from(4 * x + 6);
            } else {
                d += i64::from(4 * (x - y) + 10);
                y -= 1;
            }
            x += 1;
        }

        if x == y {
            self.vote_endpoints(accumulator, band, center_y + y, center_x - x, center_x + x);
            self.vote_endpoints(accumulator, band, center_y - y, center_x - x, center_x + x);
            self.vote_endpoints(accumulator, band, center_y + x, center_x - y, center_x + y);
            self.vote_endpoints(accumulator, band, center_y - x, center_x - y, center_x + y);
        }
    }
}

impl<F> TileReducer<F> for HoughCircleReducer
where
    F: BandFormat,
    F::Sample: ToF64 + Copy,
{
    type Partial = Vec<u32>;
    type Output = InMemoryImage<U32>;
    /// Pre-allocated circle accumulator `width × height × radius_bands` reused
    /// across tiles per rayon thread, eliminating per-tile alloc.
    type Scratch = Vec<u32>;

    fn reduce_tile(&self, tile: &Tile<F>, region: &Region) -> Self::Partial {
        let mut accumulator = vec![0u32; self.width * self.height * self.bands];
        let bands = tile.bands as usize;

        for local_y in 0..region.height as usize {
            let y = region.y + local_y as i32;
            if y < 0 {
                continue;
            }

            for local_x in 0..region.width as usize {
                let x = region.x + local_x as i32;
                if x < 0 {
                    continue;
                }

                let pixel_index = (local_y * region.width as usize + local_x) * bands;
                if tile.data[pixel_index].to_f64() <= self.threshold {
                    continue;
                }

                let center_x = x / self.scale as i32;
                let center_y = y / self.scale as i32;

                for band in 0..self.bands {
                    let radius = band as i32 + (self.min_radius / self.scale) as i32;
                    self.vote_circle_band(&mut accumulator, center_x, center_y, radius, band);
                }
            }
        }

        accumulator
    }

    /// Zero-allocation accumulation. `scratch` holds the full `width × height × bands`
    /// buffer, resized once per rayon thread on the first tile, then zeroed and reused.
    fn accumulate_into(
        &self,
        tile: &Tile<F>,
        region: &Region,
        scratch: &mut Self::Scratch,
        partial: &mut Option<Self::Partial>,
    ) {
        scratch.resize(self.width * self.height * self.bands, 0u32);
        scratch.fill(0u32);

        let bands = tile.bands as usize;

        for local_y in 0..region.height as usize {
            let y = region.y + local_y as i32;
            if y < 0 {
                continue;
            }

            for local_x in 0..region.width as usize {
                let x = region.x + local_x as i32;
                if x < 0 {
                    continue;
                }

                let pixel_index = (local_y * region.width as usize + local_x) * bands;
                if tile.data[pixel_index].to_f64() <= self.threshold {
                    continue;
                }

                let center_x = x / self.scale as i32;
                let center_y = y / self.scale as i32;

                for band in 0..self.bands {
                    let radius = band as i32 + (self.min_radius / self.scale) as i32;
                    self.vote_circle_band(scratch, center_x, center_y, radius, band);
                }
            }
        }

        match partial {
            Some(acc) => {
                for (lhs, rhs) in acc.iter_mut().zip(scratch.iter()) {
                    *lhs += rhs;
                }
            }
            None => {
                *partial = Some(scratch.clone());
            }
        }
    }

    fn combine(&self, mut a: Self::Partial, b: Self::Partial) -> Self::Partial {
        for (lhs, rhs) in a.iter_mut().zip(b.iter()) {
            *lhs += rhs;
        }
        a
    }

    fn finalize(&self, mut combined: Self::Partial) -> Self::Output {
        let max_circumference = 2.0 * std::f64::consts::PI * f64::from(self.max_radius);

        for band in 0..self.bands {
            let radius = band as u32 * self.scale + self.min_radius;
            let circumference = 2.0 * std::f64::consts::PI * f64::from(radius);
            let ratio = max_circumference / circumference;

            for index in (band..combined.len()).step_by(self.bands) {
                combined[index] = (f64::from(combined[index]) * ratio) as u32;
            }
        }

        InMemoryImage::<U32>::from_buffer(
            self.width as u32,
            self.height as u32,
            self.bands as u32,
            combined,
        )
        .unwrap_or_else(|error| {
            debug_assert!(
                false,
                "hough circle accumulator dimensions are internally consistent: {error}"
            );
            // SAFETY: `combined` is allocated from the same checked dimensions used to build the image shape.
            unsafe { std::hint::unreachable_unchecked() }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::U8, reducer::TileReducer};

    fn accumulator_index(width: usize, bands: usize, x: usize, y: usize, band: usize) -> usize {
        (y * width + x) * bands + band
    }

    fn line_accumulator_len(reducer: &HoughLineReducer) -> usize {
        reducer.width * reducer.height
    }

    fn circle_accumulator_len(reducer: &HoughCircleReducer) -> usize {
        reducer.width * reducer.height * reducer.bands
    }

    #[test]
    fn line_single_pixel_votes_expected_theta_bins() {
        let reducer = HoughLineReducer::new(180, 64, 8, 8, 0.0);
        let region = Region::new(0, 0, 8, 8);
        let mut data = vec![0u8; 64];
        data[4 + 2 * 8] = 255;
        let tile = Tile::<U8>::new(region, 1, &data);

        let partial = <HoughLineReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let diagonal = 8.0_f64.hypot(8.0_f64);
        let rho_x = (4.0 / diagonal + 1.0) * (64.0 / 2.0);
        let rho_y = (2.0 / diagonal + 1.0) * (64.0 / 2.0);

        assert_eq!(partial[rho_x as usize * 180], 1);
        assert_eq!(partial[90 + rho_y as usize * 180], 1);
        assert_eq!(partial.iter().sum::<u32>(), 180);
    }

    #[test]
    fn line_empty_image_accumulator_is_zero() {
        let reducer = HoughLineReducer::new(64, 64, 8, 8, 0.0);
        let region = Region::new(0, 0, 8, 8);
        let data = vec![0u8; 64];
        let tile = Tile::<U8>::new(region, 1, &data);

        let partial = <HoughLineReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let image = <HoughLineReducer as TileReducer<U8>>::finalize(&reducer, partial.clone());

        assert!(partial.iter().all(|vote| *vote == 0));
        assert_eq!(image.width(), 64);
        assert_eq!(image.height(), 64);
        assert!(image.pixels().iter().all(|vote| *vote == 0));
    }

    #[test]
    fn line_combine_adds_partial_accumulators() {
        let reducer = HoughLineReducer::new(2, 2, 4, 4, 0.0);
        let combined = <HoughLineReducer as TileReducer<U8>>::combine(
            &reducer,
            vec![1, 2, 3, 4],
            vec![4, 3, 2, 1],
        );

        assert_eq!(combined, vec![5, 5, 5, 5]);
    }

    #[test]
    fn line_negative_region_matches_equivalent_in_bounds_pixel() {
        let reducer = HoughLineReducer::new(32, 32, 4, 4, 0.0);
        let negative_region = Region::new(-1, -1, 2, 2);
        let negative_tile = Tile::<U8>::new(negative_region, 1, &[0, 0, 0, 255]);
        let in_bounds_region = Region::new(0, 0, 1, 1);
        let in_bounds_tile = Tile::<U8>::new(in_bounds_region, 1, &[255]);

        let negative = <HoughLineReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &negative_tile,
            &negative_region,
        );
        let in_bounds = <HoughLineReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &in_bounds_tile,
            &in_bounds_region,
        );

        assert_eq!(negative, in_bounds);
    }

    #[test]
    fn line_accumulate_into_initializes_and_combines_partial_votes() {
        let reducer = HoughLineReducer::new(32, 32, 4, 4, 0.0);
        let region = Region::new(0, 0, 4, 4);
        let mut data = vec![0u8; 16];
        data[1 + 2 * 4] = 255;
        let tile = Tile::<U8>::new(region, 1, &data);
        let expected = <HoughLineReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let mut scratch = vec![u32::MAX; 3];
        let mut partial = None;

        <HoughLineReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        assert_eq!(scratch.len(), line_accumulator_len(&reducer));
        assert_eq!(partial, Some(expected.clone()));

        scratch.fill(u32::MAX);
        <HoughLineReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        let doubled: Vec<u32> = expected.iter().map(|vote| vote * 2).collect();
        assert_eq!(partial, Some(doubled));
    }

    #[test]
    fn line_accumulate_into_discards_stale_scratch_for_empty_tile() {
        let reducer = HoughLineReducer::new(16, 16, 2, 2, 0.0);
        let region = Region::new(0, 0, 2, 2);
        let tile = Tile::<U8>::new(region, 1, &[0, 0, 0, 0]);
        let mut scratch = vec![7u32; line_accumulator_len(&reducer)];
        let mut partial = Some(vec![11u32; line_accumulator_len(&reducer)]);

        <HoughLineReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        assert!(scratch.iter().all(|vote| *vote == 0));
        assert_eq!(partial, Some(vec![11u32; line_accumulator_len(&reducer)]));
    }

    #[test]
    fn circle_single_pixel_votes_cardinal_centers_for_radius_one() {
        let reducer = HoughCircleReducer::new(1, 1, 2, 7, 7, 0.0).unwrap();
        let region = Region::new(0, 0, 7, 7);
        let mut data = vec![0u8; 49];
        data[3 + 3 * 7] = 255;
        let tile = Tile::<U8>::new(region, 1, &data);

        let partial =
            <HoughCircleReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);

        assert_eq!(partial[accumulator_index(7, 2, 3, 4, 0)], 2);
        assert_eq!(partial[accumulator_index(7, 2, 3, 2, 0)], 2);
        assert_eq!(partial[accumulator_index(7, 2, 2, 3, 0)], 2);
        assert_eq!(partial[accumulator_index(7, 2, 4, 3, 0)], 2);
    }

    #[test]
    fn circle_empty_image_accumulator_is_zero() {
        let reducer = HoughCircleReducer::new(1, 2, 4, 8, 8, 0.0).unwrap();
        let region = Region::new(0, 0, 8, 8);
        let data = vec![0u8; 64];
        let tile = Tile::<U8>::new(region, 1, &data);

        let partial =
            <HoughCircleReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let image = <HoughCircleReducer as TileReducer<U8>>::finalize(&reducer, partial.clone());

        assert!(partial.iter().all(|vote| *vote == 0));
        assert_eq!(image.bands(), 3);
        assert!(image.pixels().iter().all(|vote| *vote == 0));
    }

    #[test]
    fn circle_combine_adds_partial_accumulators() {
        let reducer = HoughCircleReducer::new(1, 1, 2, 2, 2, 0.0).unwrap();
        let combined = <HoughCircleReducer as TileReducer<U8>>::combine(
            &reducer,
            vec![1, 2, 3, 4],
            vec![4, 3, 2, 1],
        );

        assert_eq!(combined, vec![5, 5, 5, 5]);
    }

    #[test]
    fn circle_negative_region_matches_equivalent_in_bounds_pixel() {
        let reducer = HoughCircleReducer::new(1, 1, 2, 5, 5, 0.0).unwrap();
        let negative_region = Region::new(-1, -1, 2, 2);
        let negative_tile = Tile::<U8>::new(negative_region, 1, &[0, 0, 0, 255]);
        let in_bounds_region = Region::new(0, 0, 1, 1);
        let in_bounds_tile = Tile::<U8>::new(in_bounds_region, 1, &[255]);

        let negative = <HoughCircleReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &negative_tile,
            &negative_region,
        );
        let in_bounds = <HoughCircleReducer as TileReducer<U8>>::reduce_tile(
            &reducer,
            &in_bounds_tile,
            &in_bounds_region,
        );

        assert_eq!(negative, in_bounds);
    }

    #[test]
    fn circle_vote_circle_band_clips_out_of_bounds_points_and_hits_diagonal_case() {
        let reducer = HoughCircleReducer::new(1, 1, 4, 5, 5, 0.0).unwrap();
        let mut accumulator = vec![0u32; 5 * 5 * 4];

        reducer.vote_circle_band(&mut accumulator, 0, 0, 3, 0);

        assert_eq!(accumulator[accumulator_index(5, 4, 3, 0, 0)], 2);
        assert_eq!(accumulator[accumulator_index(5, 4, 0, 3, 0)], 2);
    }

    #[test]
    fn circle_finalize_normalizes_smaller_radii_against_max_radius() {
        let reducer = HoughCircleReducer::new(1, 1, 2, 2, 1, 0.0).unwrap();
        let image = <HoughCircleReducer as TileReducer<U8>>::finalize(&reducer, vec![1, 1, 0, 0]);

        assert_eq!(image.width(), 2);
        assert_eq!(image.height(), 1);
        assert_eq!(image.bands(), 2);
        assert_eq!(image.pixels(), &[2, 1, 0, 0]);
    }

    #[test]
    fn circle_rejects_invalid_parameters() {
        assert!(matches!(
            HoughCircleReducer::new(0, 1, 3, 8, 8, 0.0),
            Err(ViprsError::Hough(HoughError::ZeroScale))
        ));
        assert!(matches!(
            HoughCircleReducer::new(1, 3, 3, 8, 8, 0.0),
            Err(ViprsError::Hough(HoughError::InvalidRadiusRange {
                min_radius: 3,
                max_radius: 3
            }))
        ));
    }

    #[test]
    fn circle_accumulate_into_initializes_and_combines_partial_votes() {
        let reducer = HoughCircleReducer::new(1, 1, 3, 5, 5, 0.0).unwrap();
        let region = Region::new(0, 0, 5, 5);
        let mut data = vec![0u8; 25];
        data[2 + 2 * 5] = 255;
        let tile = Tile::<U8>::new(region, 1, &data);
        let expected =
            <HoughCircleReducer as TileReducer<U8>>::reduce_tile(&reducer, &tile, &region);
        let mut scratch = vec![u32::MAX; 5];
        let mut partial = None;

        <HoughCircleReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        assert_eq!(scratch.len(), circle_accumulator_len(&reducer));
        assert_eq!(partial, Some(expected.clone()));

        scratch.fill(u32::MAX);
        <HoughCircleReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        let doubled: Vec<u32> = expected.iter().map(|vote| vote * 2).collect();
        assert_eq!(partial, Some(doubled));
    }

    #[test]
    fn circle_accumulate_into_discards_stale_scratch_for_empty_tile() {
        let reducer = HoughCircleReducer::new(1, 1, 3, 4, 4, 0.0).unwrap();
        let region = Region::new(0, 0, 4, 4);
        let tile = Tile::<U8>::new(region, 1, &[0; 16]);
        let mut scratch = vec![9u32; circle_accumulator_len(&reducer)];
        let mut partial = Some(vec![13u32; circle_accumulator_len(&reducer)]);

        <HoughCircleReducer as TileReducer<U8>>::accumulate_into(
            &reducer,
            &tile,
            &region,
            &mut scratch,
            &mut partial,
        );

        assert!(scratch.iter().all(|vote| *vote == 0));
        assert_eq!(partial, Some(vec![13u32; circle_accumulator_len(&reducer)]));
    }
}
