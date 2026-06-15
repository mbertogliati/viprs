#![allow(clippy::unused_self)]
// REASON: stdif helpers stay on the op for consistency with the rest of the histogram API.
#![allow(clippy::suspicious_operation_groupings)]
// REASON: the grouped arithmetic mirrors libvips' local-contrast formula exactly.

use std::marker::PhantomData;

use crate::domain::{
    error::ViprsError,
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op},
};

/// Integral-image local standard-deviation filter for contrast enhancement.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::stdif::StdifOp;
///
/// let op = StdifOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct StdifOp<F: BandFormat> {
    /// Width associated with this item.
    pub width: u32,
    /// Height associated with this item.
    pub height: u32,
    /// Stores the `a` value for this item.
    pub a: f64,
    /// Stores the `m0` value for this item.
    pub m0: f64,
    /// Stores the `b` value for this item.
    pub b: f64,
    /// Stores the `s0` value for this item.
    pub s0: f64,
    _phantom: PhantomData<F>,
}

/// Represents a stdif state.
pub struct StdifState {
    integral_sum: Vec<u64>,
    integral_sum_sq: Vec<u64>,
}

const DEFAULT_TILE_SIDE: u32 = 128;
const MAX_LIBVIPS_WINDOW: u32 = 256;

impl<F: BandFormat> StdifOp<F> {
    /// Creates a new `StdifOp`.
    pub fn new(
        width: u32,
        height: u32,
        a: f64,
        m0: f64,
        b: f64,
        s0: f64,
    ) -> Result<Self, ViprsError> {
        if width == 0 || height == 0 {
            return Err(ViprsError::Scheduler(
                "StdifOp window dimensions must be greater than zero".into(),
            ));
        }
        if width > MAX_LIBVIPS_WINDOW || height > MAX_LIBVIPS_WINDOW {
            return Err(ViprsError::Scheduler(
                "StdifOp window dimensions must be <= 256 to match libvips".into(),
            ));
        }
        if !a.is_finite() || !(0.0..=1.0).contains(&a) {
            return Err(ViprsError::Scheduler(
                "StdifOp a must be finite and in the range [0, 1]".into(),
            ));
        }
        if !m0.is_finite() {
            return Err(ViprsError::Scheduler("StdifOp m0 must be finite".into()));
        }
        if !b.is_finite() || !(0.0..=2.0).contains(&b) {
            return Err(ViprsError::Scheduler(
                "StdifOp b must be finite and in the range [0, 2]".into(),
            ));
        }
        if !s0.is_finite() {
            return Err(ViprsError::Scheduler("StdifOp s0 must be finite".into()));
        }

        Ok(Self {
            width,
            height,
            a,
            m0,
            b,
            s0,
            _phantom: PhantomData,
        })
    }

    #[inline(always)]
    const fn radius_x(&self) -> u32 {
        self.width / 2
    }

    #[inline(always)]
    const fn radius_y(&self) -> u32 {
        self.height / 2
    }

    #[inline(always)]
    const fn integral_len(&self, input_width: usize, input_height: usize, bands: usize) -> usize {
        (input_width + 1) * (input_height + 1) * bands
    }

    fn state_for_tile(&self, tile_w: u32, tile_h: u32, bands: u32) -> StdifState {
        let spec = NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x(),
            input_tile_h: tile_h + 2 * self.radius_y(),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        };
        let integral_len = self.integral_len(
            spec.input_tile_w as usize,
            spec.input_tile_h as usize,
            bands as usize,
        );

        StdifState {
            integral_sum: vec![0u64; integral_len],
            integral_sum_sq: vec![0u64; integral_len],
        }
    }
}

#[inline(always)]
const fn integral_offset(y: usize, x: usize, band: usize, stride: usize, bands: usize) -> usize {
    (y * stride + x) * bands + band
}

#[inline(always)]
fn integral_window_sum(
    integral: &[u64],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    band: usize,
    stride: usize,
    bands: usize,
) -> u64 {
    let x1 = x + width;
    let y1 = y + height;
    let bottom_right = i128::from(integral[integral_offset(y1, x1, band, stride, bands)]);
    let top_left = i128::from(integral[integral_offset(y, x, band, stride, bands)]);
    let bottom_left = i128::from(integral[integral_offset(y1, x, band, stride, bands)]);
    let top_right = i128::from(integral[integral_offset(y, x1, band, stride, bands)]);
    (bottom_right + top_left - bottom_left - top_right) as u64
}

impl Op for StdifOp<U8> {
    type Input = U8;
    type Output = U8;
    type State = StdifState;

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FatStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x() as i32,
            output.y - self.radius_y() as i32,
            output.width + 2 * self.radius_x(),
            output.height + 2 * self.radius_y(),
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x(),
            input_tile_h: tile_h + 2 * self.radius_y(),
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {
        self.state_for_tile(DEFAULT_TILE_SIDE, DEFAULT_TILE_SIDE, 1)
    }

    fn start_with_tile(&self, tile_w: u32, tile_h: u32) -> Self::State {
        self.start_with_tile_and_bands(tile_w, tile_h, 1)
    }

    fn start_with_tile_and_bands(&self, tile_w: u32, tile_h: u32, bands: u32) -> Self::State {
        self.state_for_tile(tile_w, tile_h, bands)
    }

    #[inline]
    fn process_region(&self, state: &mut Self::State, input: &Tile<U8>, output: &mut TileMut<U8>) {
        let bands = input.bands as usize;
        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let in_h = input.region.height as usize;
        let win_w = self.width as usize;
        let win_h = self.height as usize;
        let radius_x = self.radius_x() as usize;
        let radius_y = self.radius_y() as usize;
        let window_area = f64::from(self.width * self.height);
        let integral_stride = in_w + 1;
        let integral_len = self.integral_len(in_w, in_h, bands);
        let f1 = self.a * self.m0;
        let f2 = 1.0 - self.a;
        let f3 = self.b * self.s0;

        assert!(
            state.integral_sum.len() >= integral_len && state.integral_sum_sq.len() >= integral_len,
            "StdifOp scratch must be pre-sized with start_with_tile_and_bands()"
        );

        {
            let integral_sum_slice = &mut state.integral_sum[..integral_len];
            let integral_sum_sq_slice = &mut state.integral_sum_sq[..integral_len];
            integral_sum_slice.fill(0);
            integral_sum_sq_slice.fill(0);

            for y in 0..in_h {
                let prev_row = y * integral_stride;
                let current_row = (y + 1) * integral_stride;
                let input_row = y * in_w * bands;

                for x in 0..in_w {
                    let input_base = input_row + x * bands;
                    let left = current_row + x;
                    let top = prev_row + x + 1;
                    let top_left = prev_row + x;
                    let cell = current_row + x + 1;

                    for band in 0..bands {
                        let sample = u64::from(input.data[input_base + band]);
                        let sample_sq = sample * sample;
                        let dst = cell * bands + band;
                        integral_sum_slice[dst] = sample
                            + integral_sum_slice[left * bands + band]
                            + integral_sum_slice[top * bands + band]
                            - integral_sum_slice[top_left * bands + band];
                        integral_sum_sq_slice[dst] = sample_sq
                            + integral_sum_sq_slice[left * bands + band]
                            + integral_sum_sq_slice[top * bands + band]
                            - integral_sum_sq_slice[top_left * bands + band];
                    }
                }
            }

            for oy in 0..out_h {
                for ox in 0..out_w {
                    let center_base = ((oy + radius_y) * in_w + (ox + radius_x)) * bands;
                    let out_base = (oy * out_w + ox) * bands;

                    for band in 0..bands {
                        let sum = integral_window_sum(
                            integral_sum_slice,
                            ox,
                            oy,
                            win_w,
                            win_h,
                            band,
                            integral_stride,
                            bands,
                        ) as f64;
                        let sum_sq = integral_window_sum(
                            integral_sum_sq_slice,
                            ox,
                            oy,
                            win_w,
                            win_h,
                            band,
                            integral_stride,
                            bands,
                        ) as f64;
                        let mean = sum / window_area;
                        let variance = (sum_sq / window_area - mean * mean).max(0.0);
                        let stddev = variance.sqrt();
                        let center = f64::from(input.data[center_base + band]);
                        let result = (center - mean)
                            .mul_add(f3 / self.b.mul_add(stddev, self.s0), f1 + f2 * mean);

                        output.data[out_base + band] = stdif_to_u8(result);
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn stdif_to_u8(value: f64) -> u8 {
    if value < 0.0 {
        0
    } else if value >= 256.0 {
        255
    } else {
        (value + 0.5) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::Region;
    use proptest::prelude::*;

    fn clamped_index(coord: i32, limit: usize) -> usize {
        coord.clamp(0, limit as i32 - 1) as usize
    }

    fn expanded_input(
        data: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        window_w: u32,
        window_h: u32,
    ) -> Vec<u8> {
        let radius_x = (window_w / 2) as i32;
        let radius_y = (window_h / 2) as i32;
        let out_w = width + 2 * radius_x as usize;
        let out_h = height + 2 * radius_y as usize;
        let mut expanded = vec![0u8; out_w * out_h * bands];

        for y in 0..out_h {
            for x in 0..out_w {
                let src_x = clamped_index(x as i32 - radius_x, width);
                let src_y = clamped_index(y as i32 - radius_y, height);
                let src_base = (src_y * width + src_x) * bands;
                let dst_base = (y * out_w + x) * bands;
                expanded[dst_base..dst_base + bands]
                    .copy_from_slice(&data[src_base..src_base + bands]);
            }
        }

        expanded
    }

    fn run_naive(
        data: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        window_w: u32,
        window_h: u32,
        a: f64,
        m0: f64,
        b: f64,
        s0: f64,
    ) -> Vec<u8> {
        let expanded = expanded_input(data, width, height, bands, window_w, window_h);
        let out_w = width;
        let out_h = height;
        let in_w = width + (window_w / 2) as usize * 2;
        let win_w = window_w as usize;
        let win_h = window_h as usize;
        let radius_x = (window_w / 2) as usize;
        let radius_y = (window_h / 2) as usize;
        let window_area = f64::from(window_w * window_h);
        let mut output = vec![0u8; width * height * bands];

        for oy in 0..out_h {
            for ox in 0..out_w {
                let center_base = ((oy + radius_y) * in_w + (ox + radius_x)) * bands;
                let out_base = (oy * out_w + ox) * bands;

                for band in 0..bands {
                    let mut sum = 0.0f64;
                    let mut sum_sq = 0.0f64;

                    for wy in 0..win_h {
                        let row_base = (oy + wy) * in_w * bands;
                        for wx in 0..win_w {
                            let sample = f64::from(expanded[row_base + (ox + wx) * bands + band]);
                            sum += sample;
                            sum_sq += sample * sample;
                        }
                    }

                    let mean = sum / window_area;
                    let variance = (sum_sq / window_area - mean * mean).max(0.0);
                    let stddev = variance.sqrt();
                    let center = f64::from(expanded[center_base + band]);
                    let result = a * m0
                        + (1.0 - a) * mean
                        + (center - mean) * ((b * s0) / (s0 + b * stddev));

                    output[out_base + band] = stdif_to_u8(result);
                }
            }
        }

        output
    }

    fn run_op(
        data: &[u8],
        width: usize,
        height: usize,
        bands: usize,
        window_w: u32,
        window_h: u32,
        a: f64,
        m0: f64,
        b: f64,
        s0: f64,
    ) -> Vec<u8> {
        let op = StdifOp::<U8>::new(window_w, window_h, a, m0, b, s0).unwrap();
        let input_region = Region::new(
            0,
            0,
            width as u32 + 2 * (window_w / 2),
            height as u32 + 2 * (window_h / 2),
        );
        let output_region = Region::new(0, 0, width as u32, height as u32);
        let expanded = expanded_input(data, width, height, bands, window_w, window_h);
        let input = Tile::<U8>::new(input_region, bands as u32, &expanded);
        let mut output_data = vec![0u8; width * height * bands];
        let mut output = TileMut::<U8>::new(output_region, bands as u32, &mut output_data);
        let mut state = op.start_with_tile_and_bands(width as u32, height as u32, bands as u32);
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    prop_compose! {
        fn stdif_case()
            (
                width in 1usize..=6usize,
                height in 1usize..=6usize,
                bands in 1usize..=3usize,
                window_w in 1u32..=5u32,
                window_h in 1u32..=5u32,
                a in 0.0f64..1.0f64,
                m0 in 0.0f64..255.0f64,
                b in 0.0f64..2.0f64,
                s0 in 1.0f64..96.0f64,
            )
            (
                data in prop::collection::vec(any::<u8>(), width * height * bands),
                width in Just(width),
                height in Just(height),
                bands in Just(bands),
                window_w in Just(window_w),
                window_h in Just(window_h),
                a in Just(a),
                m0 in Just(m0),
                b in Just(b),
                s0 in Just(s0),
            ) -> (Vec<u8>, usize, usize, usize, u32, u32, f64, f64, f64, f64) {
                (data, width, height, bands, window_w, window_h, a, m0, b, s0)
            }
    }

    #[test]
    fn stdif_flat_image_is_identity() {
        let input = vec![77u8; 16];
        let output = run_op(&input, 4, 4, 1, 3, 3, 0.0, 128.0, 1.0, 32.0);
        assert_eq!(output, input);
    }

    #[test]
    fn stdif_mean_weight_one_returns_target_mean_on_boundary_values() {
        let input = vec![0u8, 255, 255, 0];
        let output = run_op(&input, 2, 2, 1, 3, 3, 1.0, 37.0, 0.0, 32.0);
        assert_eq!(output, vec![37, 37, 37, 37]);
    }

    #[test]
    fn stdif_integral_matches_naive_on_16x16_gradient() {
        let input: Vec<u8> = (0..16usize)
            .flat_map(|y| (0..16usize).map(move |x| ((x * 13 + y * 7) % 256) as u8))
            .collect();

        let naive = run_naive(&input, 16, 16, 1, 5, 5, 0.25, 128.0, 1.5, 32.0);
        let integral = run_op(&input, 16, 16, 1, 5, 5, 0.25, 128.0, 1.5, 32.0);

        assert_eq!(integral, naive);
    }

    #[test]
    fn stdif_multiband_matches_libvips_formula_reference() {
        let input = vec![0u8, 255, 64, 128, 192, 32, 255, 0];

        let naive = run_naive(&input, 2, 2, 2, 3, 3, 0.5, 128.0, 0.5, 50.0);
        let integral = run_op(&input, 2, 2, 2, 3, 3, 0.5, 128.0, 0.5, 50.0);

        assert_eq!(integral, naive);
        assert_eq!(integral, vec![85, 165, 106, 125, 150, 93, 171, 79]);
    }

    #[test]
    fn stdif_window_larger_than_image_matches_naive() {
        let input = vec![0u8, 32, 64, 96, 128, 160];

        let naive = run_naive(&input, 3, 2, 1, 7, 7, 0.5, 128.0, 0.5, 50.0);
        let integral = run_op(&input, 3, 2, 1, 7, 7, 0.5, 128.0, 0.5, 50.0);

        assert_eq!(integral, naive);
    }

    #[test]
    fn stdif_single_pixel_window_is_identity() {
        let input = vec![0u8, 64, 128, 255];
        let output = run_op(&input, 2, 2, 1, 1, 1, 0.0, 128.0, 1.0, 32.0);
        assert_eq!(output, input);
    }

    #[test]
    fn stdif_rejects_invalid_window_dimensions() {
        assert!(StdifOp::<U8>::new(0, 3, 0.5, 128.0, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 0, 0.5, 128.0, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(257, 3, 0.5, 128.0, 0.5, 50.0).is_err());
    }

    #[test]
    fn stdif_rejects_invalid_libvips_parameters() {
        assert!(StdifOp::<U8>::new(3, 3, f64::NAN, 128.0, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, -0.1, 128.0, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, 1.1, 128.0, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, 0.5, f64::INFINITY, 0.5, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, 0.5, 128.0, -0.1, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, 0.5, 128.0, 2.1, 50.0).is_err());
        assert!(StdifOp::<U8>::new(3, 3, 0.5, 128.0, 0.5, f64::NAN).is_err());
    }

    #[test]
    fn stdif_required_region_and_node_spec_include_window_radius() {
        let op = StdifOp::<U8>::new(5, 3, 0.5, 128.0, 0.5, 50.0).unwrap();
        let output = Region::new(10, 20, 32, 16);
        let required = op.required_input_region(&output);
        assert_eq!(required, Region::new(8, 19, 36, 18));

        let spec = op.node_spec(64, 8);
        assert_eq!(spec.input_tile_w, 68);
        assert_eq!(spec.input_tile_h, 10);
        assert_eq!(spec.output_tile_w, 64);
        assert_eq!(spec.output_tile_h, 8);
        assert_eq!(op.demand_hint(), DemandHint::FatStrip);
        op.start();
    }

    proptest! {
        #[test]
        fn stdif_flat_image_identity_prop(value in 0u8..=255u8, width in 1usize..=6usize, height in 1usize..=6usize) {
            let input = vec![value; width * height];
            let output = run_op(&input, width, height, 1, 3, 3, 0.0, 128.0, 1.0, 32.0);
            prop_assert_eq!(output, input);
        }

        #[test]
        fn integral_image_matches_naive_small_inputs(
            (data, width, height, bands, window_w, window_h, a, m0, b, s0) in stdif_case()
        ) {
            let naive = run_naive(&data, width, height, bands, window_w, window_h, a, m0, b, s0);
            let integral = run_op(&data, width, height, bands, window_w, window_h, a, m0, b, s0);

            prop_assert_eq!(integral, naive);
        }
    }
}
