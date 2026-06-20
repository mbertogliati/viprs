//! Quadratic geometric warp matching libvips `resample/quadratic.c`.
//!
//! The mapping is evaluated in backward-mapping form over output pixel
//! coordinates `(x, y)`:
//!
//! ```text
//! x_in = x + a + b*x + c*y + d*x*y + e*x² + f*y²
//! y_in = y + g + h*x + i*y + j*x*y + k*x² + l*y²
//! ```
//!
//! Coefficients are supplied in the same row order as libvips:
//!
//! ```text
//! a g
//! b h
//! c i
//! d j
//! e k
//! f l
//! ```

use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{NodeSpec, Op},
};

use super::{
    affine::{Affine, ExtendMode},
    sample_conv::{FromF64, ToF64},
};

const COEFF_COUNT: usize = 12;
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
/// Represents a quadratic coefficients.
pub struct QuadraticCoefficients {
    values: [f64; COEFF_COUNT],
}

impl QuadraticCoefficients {
    #[must_use]
    /// Returns or performs identity.
    pub fn identity() -> Self {
        Self::default()
    }

    #[must_use]
    /// Creates this value from order0.
    pub const fn from_order0(row0: [f64; 2]) -> Self {
        let mut values = [0.0; COEFF_COUNT];
        values[0] = row0[0];
        values[1] = row0[1];
        Self { values }
    }

    #[must_use]
    /// Creates this value from order1.
    pub const fn from_order1(rows: [[f64; 2]; 3]) -> Self {
        let mut coeffs = Self::from_order0(rows[0]);
        coeffs.values[2] = rows[1][0];
        coeffs.values[3] = rows[1][1];
        coeffs.values[4] = rows[2][0];
        coeffs.values[5] = rows[2][1];
        coeffs
    }

    #[must_use]
    /// Creates this value from order2.
    pub const fn from_order2(rows: [[f64; 2]; 4]) -> Self {
        let mut coeffs = Self::from_order1([rows[0], rows[1], rows[2]]);
        coeffs.values[6] = rows[3][0];
        coeffs.values[7] = rows[3][1];
        coeffs
    }

    #[must_use]
    /// Creates this value from order3.
    pub const fn from_order3(rows: [[f64; 2]; 6]) -> Self {
        let mut coeffs = Self::from_order2([rows[0], rows[1], rows[2], rows[3]]);
        coeffs.values[8] = rows[4][0];
        coeffs.values[9] = rows[4][1];
        coeffs.values[10] = rows[5][0];
        coeffs.values[11] = rows[5][1];
        coeffs
    }

    #[inline]
    const fn x_const(self) -> f64 {
        self.values[0]
    }

    #[inline]
    const fn y_const(self) -> f64 {
        self.values[1]
    }

    #[inline]
    const fn x_x(self) -> f64 {
        self.values[2]
    }

    #[inline]
    const fn y_x(self) -> f64 {
        self.values[3]
    }

    #[inline]
    const fn x_y(self) -> f64 {
        self.values[4]
    }

    #[inline]
    const fn y_y(self) -> f64 {
        self.values[5]
    }

    #[inline]
    const fn x_xy(self) -> f64 {
        self.values[6]
    }

    #[inline]
    const fn y_xy(self) -> f64 {
        self.values[7]
    }

    #[inline]
    const fn x_x2(self) -> f64 {
        self.values[8]
    }

    #[inline]
    const fn y_x2(self) -> f64 {
        self.values[9]
    }

    #[inline]
    const fn x_y2(self) -> f64 {
        self.values[10]
    }

    #[inline]
    const fn y_y2(self) -> f64 {
        self.values[11]
    }
}

#[derive(Debug, Clone, Copy)]
struct AxisPolynomial {
    constant: f64,
    linear_x: f64,
    linear_y: f64,
    cross_xy: f64,
    quad_x2: f64,
    quad_y2: f64,
}

impl AxisPolynomial {
    #[inline]
    fn eval(self, x: f64, y: f64) -> f64 {
        (self.quad_y2 * y).mul_add(
            y,
            (self.quad_x2 * x).mul_add(
                x,
                (self.cross_xy * x).mul_add(
                    y,
                    self.linear_y
                        .mul_add(y, self.linear_x.mul_add(x, self.constant)),
                ),
            ),
        )
    }

    #[inline]
    fn deriv_x(self, x: f64, y: f64) -> f64 {
        (2.0 * self.quad_x2).mul_add(x, self.cross_xy.mul_add(y, self.linear_x))
    }

    #[inline]
    fn deriv_y(self, x: f64, y: f64) -> f64 {
        (2.0 * self.quad_y2).mul_add(y, self.cross_xy.mul_add(x, self.linear_y))
    }

    fn bounds_on_rect(self, x0: f64, x1: f64, y0: f64, y1: f64) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut consider = |x: f64, y: f64| {
            let value = self.eval(x, y);
            min = min.min(value);
            max = max.max(value);
        };

        consider(x0, y0);
        consider(x0, y1);
        consider(x1, y0);
        consider(x1, y1);

        if self.quad_x2.abs() > EPSILON {
            for y in [y0, y1] {
                let x = -self.cross_xy.mul_add(y, self.linear_x) / (2.0 * self.quad_x2);
                if x >= x0 && x <= x1 {
                    consider(x, y);
                }
            }
        }

        if self.quad_y2.abs() > EPSILON {
            for x in [x0, x1] {
                let y = -self.cross_xy.mul_add(x, self.linear_y) / (2.0 * self.quad_y2);
                if y >= y0 && y <= y1 {
                    consider(x, y);
                }
            }
        }

        let determinant = self
            .cross_xy
            .mul_add(-self.cross_xy, 4.0 * self.quad_x2 * self.quad_y2);
        if determinant.abs() > EPSILON {
            let x = (2.0 * self.quad_y2).mul_add(-self.linear_x, self.cross_xy * self.linear_y)
                / determinant;
            let y = (2.0 * self.quad_x2).mul_add(-self.linear_y, self.cross_xy * self.linear_x)
                / determinant;
            if x >= x0 && x <= x1 && y >= y0 && y <= y1 {
                consider(x, y);
            }
        }

        (min, max)
    }

    #[inline]
    fn max_abs_deriv_x_on_rect(self, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        [
            self.deriv_x(x0, y0).abs(),
            self.deriv_x(x0, y1).abs(),
            self.deriv_x(x1, y0).abs(),
            self.deriv_x(x1, y1).abs(),
        ]
        .into_iter()
        .fold(0.0, f64::max)
    }

    #[inline]
    fn max_abs_deriv_y_on_rect(self, x0: f64, x1: f64, y0: f64, y1: f64) -> f64 {
        [
            self.deriv_y(x0, y0).abs(),
            self.deriv_y(x0, y1).abs(),
            self.deriv_y(x1, y0).abs(),
            self.deriv_y(x1, y1).abs(),
        ]
        .into_iter()
        .fold(0.0, f64::max)
    }
}

#[derive(Debug, Clone, Copy)]
struct RowState {
    x_in: f64,
    y_in: f64,
    dx: f64,
    dy: f64,
    ddx: f64,
    ddy: f64,
}

/// Applies the `quadratic` resampling operation to the image. Use it to change geometry, scale,
/// or sampling density while preserving image content.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::resample::quadratic::Quadratic;
///
/// let op = Quadratic::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Quadratic<F: BandFormat> {
    coeffs: QuadraticCoefficients,
    kernel: InterpolationKernel,
    output_w: u32,
    output_h: u32,
    background: f64,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Quadratic<F> {
    #[must_use]
    /// Creates a new `Quadratic`.
    pub const fn new(
        coeffs: QuadraticCoefficients,
        kernel: InterpolationKernel,
        output_w: u32,
        output_h: u32,
    ) -> Self {
        Self {
            coeffs,
            kernel,
            output_w,
            output_h,
            background: 0.0,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns this value configured with background.
    pub const fn with_background(mut self, background: f64) -> Self {
        self.background = background;
        self
    }

    #[inline]
    fn x_poly(&self) -> AxisPolynomial {
        AxisPolynomial {
            constant: self.coeffs.x_const(),
            linear_x: 1.0 + self.coeffs.x_x(),
            linear_y: self.coeffs.x_y(),
            cross_xy: self.coeffs.x_xy(),
            quad_x2: self.coeffs.x_x2(),
            quad_y2: self.coeffs.x_y2(),
        }
    }

    #[inline]
    fn y_poly(&self) -> AxisPolynomial {
        AxisPolynomial {
            constant: self.coeffs.y_const(),
            linear_x: self.coeffs.y_x(),
            linear_y: 1.0 + self.coeffs.y_y(),
            cross_xy: self.coeffs.y_xy(),
            quad_x2: self.coeffs.y_x2(),
            quad_y2: self.coeffs.y_y2(),
        }
    }

    #[inline]
    const fn kernel_padding(&self) -> (i32, i32) {
        self.kernel.affine_padding()
    }

    #[inline]
    fn row_state(&self, x_out: f64, y_out: f64) -> RowState {
        let x_poly = self.x_poly();
        let y_poly = self.y_poly();

        RowState {
            x_in: x_poly.eval(x_out, y_out),
            y_in: y_poly.eval(x_out, y_out),
            dx: self.coeffs.x_x2().mul_add(
                2.0f64.mul_add(x_out, 1.0),
                self.coeffs.x_xy().mul_add(y_out, 1.0 + self.coeffs.x_x()),
            ),
            dy: self.coeffs.y_x2().mul_add(
                2.0f64.mul_add(x_out, 1.0),
                self.coeffs.y_xy().mul_add(y_out, self.coeffs.y_x()),
            ),
            ddx: 2.0 * self.coeffs.x_x2(),
            ddy: 2.0 * self.coeffs.y_x2(),
        }
    }

    #[inline]
    fn sampler(&self) -> Affine<F> {
        Affine::new(
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            self.kernel,
            self.output_w,
            self.output_h,
        )
        .with_extend(ExtendMode::Copy)
    }
}

impl<F: BandFormat> Op for Quadratic<F>
where
    F::Sample: ToF64 + FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        if output.is_empty() {
            return Region::new(output.x, output.y, 0, 0);
        }

        let x0 = f64::from(output.x);
        let y0 = f64::from(output.y);
        let x1 = f64::from(output.x) + f64::from(output.width.saturating_sub(1));
        let y1 = f64::from(output.y) + f64::from(output.height.saturating_sub(1));

        let (min_x, max_x) = self.x_poly().bounds_on_rect(x0, x1, y0, y1);
        let (min_y, max_y) = self.y_poly().bounds_on_rect(x0, x1, y0, y1);
        let (left_pad, right_pad) = self.kernel_padding();

        let rx0 = min_x.floor() as i32 - left_pad;
        let rx1 = max_x.floor() as i32 + right_pad;
        let ry0 = min_y.floor() as i32 - left_pad;
        let ry1 = max_y.floor() as i32 + right_pad;

        Region::new(rx0, ry0, (rx1 - rx0 + 1) as u32, (ry1 - ry0 + 1) as u32)
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        let x1 = f64::from(self.output_w.saturating_sub(1));
        let y1 = f64::from(self.output_h.saturating_sub(1));
        let x_poly = self.x_poly();
        let y_poly = self.y_poly();
        let span_w = f64::from(tile_w.saturating_sub(1));
        let span_h = f64::from(tile_h.saturating_sub(1));
        let (left_pad, right_pad) = self.kernel_padding();
        let halo = (left_pad + right_pad) as u32;

        let input_span_x = x_poly.max_abs_deriv_y_on_rect(0.0, x1, 0.0, y1).mul_add(
            span_h,
            x_poly.max_abs_deriv_x_on_rect(0.0, x1, 0.0, y1) * span_w,
        );
        let input_span_y = y_poly.max_abs_deriv_y_on_rect(0.0, x1, 0.0, y1).mul_add(
            span_h,
            y_poly.max_abs_deriv_x_on_rect(0.0, x1, 0.0, y1) * span_w,
        );

        NodeSpec {
            input_tile_w: input_span_x.ceil() as u32 + 1 + halo,
            input_tile_h: input_span_y.ceil() as u32 + 1 + halo,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        let sampler = self.sampler();
        let out_h = output.region.height as usize;
        let out_w = output.region.width as usize;
        let bands = input.bands as usize;
        let x_start = f64::from(output.region.x);
        let clip_x0 = i64::from(input.region.x);
        let clip_y0 = i64::from(input.region.y);
        let clip_x1 = clip_x0 + i64::from(input.region.width);
        let clip_y1 = clip_y0 + i64::from(input.region.height);
        let background = F::Sample::from_f64(self.background);

        for y_local in 0..out_h {
            let y_out = f64::from(output.region.y) + y_local as f64;
            let mut row = self.row_state(x_start, y_out);

            for x_local in 0..out_w {
                let out_base = (y_local * out_w + x_local) * bands;
                let xi = row.x_in as i64;
                let yi = row.y_in as i64;

                if xi < clip_x0 || yi < clip_y0 || xi >= clip_x1 || yi >= clip_y1 {
                    output.data[out_base..out_base + bands].fill(background);
                } else {
                    for band in 0..bands {
                        let value = sampler.sample_at(input, row.x_in, row.y_in, band);
                        output.data[out_base + band] = F::Sample::from_f64(value);
                    }
                }

                row.x_in += row.dx;
                row.y_in += row.dy;
                row.dx += row.ddx;
                row.dy += row.ddy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::U8,
        image::{Region, Tile, TileMut},
        kernel::InterpolationKernel,
    };

    fn run_quadratic_u8(
        input_data: &[u8],
        in_region: Region,
        out_region: Region,
        coeffs: QuadraticCoefficients,
        kernel: InterpolationKernel,
    ) -> Vec<u8> {
        let mut output_data = vec![0u8; out_region.pixel_count()];
        let input = Tile::<U8>::new(in_region, 1, input_data);
        let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
        let mut state = ();
        Quadratic::<U8>::new(coeffs, kernel, out_region.width, out_region.height).process_region(
            &mut state,
            &input,
            &mut output,
        );
        output_data
    }

    #[test]
    fn identity_nearest_preserves_pixels() {
        let data: Vec<u8> = (0u8..16).collect();
        let region = Region::new(0, 0, 4, 4);
        let result = run_quadratic_u8(
            &data,
            region,
            region,
            QuadraticCoefficients::identity(),
            InterpolationKernel::Nearest,
        );

        assert_eq!(result, data);
    }

    #[test]
    fn order0_translation_shifts_samples_and_backgrounds_edge() {
        let data = vec![0u8, 1, 2, 3, 4, 5];
        let in_region = Region::new(0, 0, 3, 2);
        let out_region = Region::new(0, 0, 3, 2);
        let coeffs = QuadraticCoefficients::from_order0([1.0, 0.0]);

        let result = run_quadratic_u8(
            &data,
            in_region,
            out_region,
            coeffs,
            InterpolationKernel::Nearest,
        );

        assert_eq!(result, vec![1, 2, 0, 4, 5, 0]);
    }

    #[test]
    fn order2_cross_term_warps_second_row() {
        let data: Vec<u8> = (0u8..9).collect();
        let region = Region::new(0, 0, 3, 3);
        let coeffs =
            QuadraticCoefficients::from_order2([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]);

        let result = run_quadratic_u8(&data, region, region, coeffs, InterpolationKernel::Nearest);

        assert_eq!(result, vec![0, 1, 2, 3, 5, 0, 6, 0, 0]);
    }

    #[test]
    fn order3_quadratic_term_warps_last_column() {
        let data: Vec<u8> = (0u8..9).collect();
        let region = Region::new(0, 0, 3, 3);
        let coeffs = QuadraticCoefficients::from_order3([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]);

        let result = run_quadratic_u8(&data, region, region, coeffs, InterpolationKernel::Nearest);

        assert_eq!(result, vec![0, 2, 0, 3, 5, 0, 6, 8, 0]);
    }

    #[test]
    fn required_input_region_identity_bilinear_matches_affine_halo() {
        let op = Quadratic::<U8>::new(
            QuadraticCoefficients::identity(),
            InterpolationKernel::Bilinear,
            64,
            64,
        );

        assert_eq!(
            op.required_input_region(&Region::new(0, 0, 64, 64)),
            Region::new(0, 0, 65, 65)
        );
    }

    #[test]
    fn order1_bilinear_matches_libvips_golden_for_coeff_row_order_and_right_border() {
        let data = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80];
        let region = Region::new(0, 0, 3, 3);
        let coeffs = QuadraticCoefficients::from_order1([[0.25, 0.1], [0.5, 0.25], [0.75, 0.5]]);

        let result = run_quadratic_u8(&data, region, region, coeffs, InterpolationKernel::Bilinear);

        assert_eq!(result, vec![6, 28, 0, 58, 76, 0, 0, 0, 0]);
    }

    #[test]
    fn order2_bilinear_matches_libvips_golden_for_bilinear_cross_term_and_zero_fill() {
        let data = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80];
        let region = Region::new(0, 0, 3, 3);
        let coeffs =
            QuadraticCoefficients::from_order2([[0.1, 0.0], [0.2, 0.05], [0.3, 0.4], [0.5, 0.25]]);

        let result = run_quadratic_u8(&data, region, region, coeffs, InterpolationKernel::Bilinear);

        assert_eq!(result, vec![1, 15, 23, 46, 71, 0, 67, 0, 0]);
    }

    #[test]
    fn order3_bilinear_matches_libvips_golden_for_quadratic_terms_and_border_clip() {
        let data = vec![0u8, 10, 20, 30, 40, 50, 60, 70, 80];
        let region = Region::new(0, 0, 3, 3);
        let coeffs = QuadraticCoefficients::from_order3([
            [0.0, 0.2],
            [0.1, 0.15],
            [0.25, 0.35],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.4, 0.05],
        ]);

        let result = run_quadratic_u8(&data, region, region, coeffs, InterpolationKernel::Bilinear);

        assert_eq!(result, vec![6, 31, 0, 55, 80, 0, 0, 0, 0]);
    }

    #[test]
    fn coefficient_constructors_populate_expected_slots() {
        let order0 = QuadraticCoefficients::from_order0([1.0, 2.0]);
        let order1 = QuadraticCoefficients::from_order1([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let order2 =
            QuadraticCoefficients::from_order2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let order3 = QuadraticCoefficients::from_order3([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]);

        assert_eq!(order0.values[..2], [1.0, 2.0]);
        assert_eq!(order1.values[..6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(order2.values[..8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            order3.values,
            [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }

    #[test]
    fn required_input_region_tracks_quadratic_extrema_inside_output_rect() {
        let op = Quadratic::<U8>::new(
            QuadraticCoefficients::from_order3([
                [0.0, 0.0],
                [-4.0, 0.0],
                [0.0, -2.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]),
            InterpolationKernel::Nearest,
            4,
            4,
        );

        assert_eq!(
            op.required_input_region(&Region::new(0, 0, 4, 4)),
            Region::new(-3, -1, 4, 8)
        );
    }

    #[test]
    fn node_spec_expands_for_quadratic_derivatives() {
        let op = Quadratic::<U8>::new(
            QuadraticCoefficients::from_order3([
                [0.0, 0.0],
                [1.0, -1.0],
                [0.5, 0.25],
                [0.25, -0.5],
                [0.5, 0.0],
                [0.0, 0.75],
            ]),
            InterpolationKernel::Bilinear,
            8,
            6,
        );

        let spec = op.node_spec(3, 2);

        assert!(spec.input_tile_w > 3);
        assert!(spec.input_tile_h > 2);
        assert_eq!(spec.output_tile_w, 3);
        assert_eq!(spec.output_tile_h, 2);
    }

    proptest! {
        #[test]
        fn identity_coefficients_preserve_random_pixels(
            width in 1u32..=16,
            height in 1u32..=16,
            pixels in prop::collection::vec(any::<u8>(), 1..=256),
            kernel in prop_oneof![
                Just(InterpolationKernel::Nearest),
                Just(InterpolationKernel::Bilinear),
                Just(InterpolationKernel::Quadratic),
                Just(InterpolationKernel::Lanczos3),
            ],
        ) {
            let expected_len = (width * height) as usize;
            prop_assume!(pixels.len() >= expected_len);
            let input = pixels[..expected_len].to_vec();
            let region = Region::new(0, 0, width, height);

            let result = run_quadratic_u8(
                &input,
                region,
                region,
                QuadraticCoefficients::identity(),
                kernel,
            );

            prop_assert_eq!(result, input);
        }

        #[test]
        fn order0_shift_right_fills_last_column_with_background(
            width in 1u32..=16,
            height in 1u32..=16,
            pixels in prop::collection::vec(any::<u8>(), 1..=256),
        ) {
            let expected_len = (width * height) as usize;
            prop_assume!(pixels.len() >= expected_len);
            let input = pixels[..expected_len].to_vec();
            let region = Region::new(0, 0, width, height);
            let result = run_quadratic_u8(
                &input,
                region,
                region,
                QuadraticCoefficients::from_order0([1.0, 0.0]),
                InterpolationKernel::Nearest,
            );

            for y in 0..height as usize {
                for x in 0..width as usize {
                    let idx = y * width as usize + x;
                    if x + 1 < width as usize {
                        prop_assert_eq!(result[idx], input[idx + 1]);
                    } else {
                        prop_assert_eq!(result[idx], 0);
                    }
                }
            }
        }
    }
}
