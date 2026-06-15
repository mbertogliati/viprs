#![allow(dead_code)]
// REASON: radius metadata is reserved for pending public tuning hooks.

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        format::{BandFormat, F32},
        image::{DemandHint, Region, Tile, TileMut},
        ops::convolution::gauss_blur::gaussian_kernel_1d_float,
    },
};

use super::common::{ToF64, convolve_separable_at};

/// libvips-style Canny edge detector:
/// Gaussian blur, 2×2 gradients, polar conversion, then non-maximum suppression.
pub struct Canny<F: BandFormat> {
    kernel: Vec<f64>,
    radius: usize,
    halo: usize,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Canny<F>
where
    F::Sample: ToF64 + Pod,
{
    #[must_use]
    /// Creates a new `Canny`.
    pub fn new(sigma: f32) -> Self {
        let kernel = gaussian_kernel_1d_float(sigma);
        let radius = (kernel.len() - 1) / 2;
        let halo = radius + 2;
        Self {
            kernel,
            radius,
            halo,
            _format: PhantomData,
        }
    }

    #[inline(always)]
    fn blurred_at(
        &self,
        input: &Tile<F>,
        in_w: usize,
        bands: usize,
        x: usize,
        y: usize,
        band: usize,
    ) -> f64 {
        convolve_separable_at(input, in_w, bands, x, y, band, &self.kernel)
    }

    #[inline(always)]
    fn gradient_at(
        &self,
        input: &Tile<F>,
        in_w: usize,
        bands: usize,
        x: usize,
        y: usize,
        band: usize,
    ) -> (f64, f64) {
        // libvips computes the 2x2 gradient with `conv()` and the mask:
        // [-1, 1]
        // [-1, 1]
        //
        // Even-sized masks carry an X/Y offset of `-size/2`, so the output pixel
        // is centered on the bottom-right corner of the 2x2 support rather than
        // the top-left. Sampling the blurred image with the 2x2 window ending at
        // `(x, y)` keeps Viprs aligned with libvips pixel centers.
        let a = self.blurred_at(input, in_w, bands, x - 1, y - 1, band);
        let b = self.blurred_at(input, in_w, bands, x, y - 1, band);
        let c = self.blurred_at(input, in_w, bands, x - 1, y, band);
        let d = self.blurred_at(input, in_w, bands, x, y, band);
        let gx = -a + b - c + d;
        let gy = -a - b + c + d;
        (gx, gy)
    }

    #[inline(always)]
    fn polar_at(
        &self,
        input: &Tile<F>,
        in_w: usize,
        bands: usize,
        x: usize,
        y: usize,
        band: usize,
    ) -> (f64, f64) {
        let (gx, gy) = self.gradient_at(input, in_w, bands, x, y, band);
        let magnitude = (gy.mul_add(gy, gx * gx) + 256.0) / 512.0;
        let theta = gx.atan2(gy).to_degrees().rem_euclid(360.0) * 256.0 / 360.0;
        (magnitude, theta)
    }
}

impl<F> Op for Canny<F>
where
    F: BandFormat,
    F::Sample: ToF64 + Pod,
{
    type Input = F;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::SmallTile
    }

    fn required_input_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.halo as i32,
            output.y - self.halo as i32,
            output.width + 2 * self.halo as u32,
            output.height + 2 * self.halo as u32,
        )
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.halo as u32,
            input_tile_h: tile_h + 2 * self.halo as u32,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    fn start(&self) -> Self::State {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F32>) {
        const DIRS: [(isize, isize); 8] = [
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
        ];

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;
        let bands = input.bands as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                let x = ox + self.halo;
                let y = oy + self.halo;
                for band in 0..bands {
                    let (center_magnitude, theta) = self.polar_at(input, in_w, bands, x, y, band);
                    let low_theta = ((theta / 32.0) as usize) & 0x7;
                    let high_theta = (low_theta + 1) & 0x7;
                    let residual = (low_theta as f64).mul_add(-32.0, theta);

                    let sample_mag = |dir: usize| -> f64 {
                        let (dx, dy) = DIRS[dir];
                        let px = x.saturating_add_signed(dx);
                        let py = y.saturating_add_signed(dy);
                        self.polar_at(input, in_w, bands, px, py, band).0
                    };

                    let lowa = sample_mag(low_theta);
                    let lowb = sample_mag(high_theta);
                    let low = (lowa * (32.0 - residual) + lowb * residual) / 32.0;

                    let higha = sample_mag((low_theta + 4) & 0x7);
                    let highb = sample_mag((high_theta + 4) & 0x7);
                    let high = (higha * (32.0 - residual) + highb * residual) / 32.0;

                    let out_idx = (oy * out_w + ox) * bands + band;
                    output.data[out_idx] = if center_magnitude <= low || center_magnitude < high {
                        0.0
                    } else {
                        center_magnitude as f32
                    };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::F32,
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn zero_input_stays_zero(width in 1usize..4, height in 1usize..4) {
            let op = Canny::<F32>::new(1.4);
            let halo = op.halo;
            let in_region = Region::new(0, 0, (width + 2 * halo) as u32, (height + 2 * halo) as u32);
            let out_region = Region::new(0, 0, width as u32, height as u32);
            let input_data = vec![0.0f32; (width + 2 * halo) * (height + 2 * halo)];
            let input = Tile::<F32>::new(in_region, 1, &input_data);
            let mut output_data = vec![1.0f32; width * height];
            let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            prop_assert!(output_data.iter().all(|value| value.abs() < 1e-6));
        }
    }

    #[test]
    fn constant_field_has_no_edges() {
        let op = Canny::<F32>::new(1.4);
        let halo = op.halo;
        let in_region = Region::new(0, 0, (3 + 2 * halo) as u32, (3 + 2 * halo) as u32);
        let out_region = Region::new(0, 0, 3, 3);
        let input_data = vec![5.0f32; (3 + 2 * halo) * (3 + 2 * halo)];
        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; 9];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert!(output_data.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn constructor_sets_radius_halo_and_contracts_from_sigma() {
        let op = Canny::<F32>::new(1.4);
        let out_region = Region::new(10, 20, 3, 4);

        assert_eq!(op.halo, op.radius + 2);
        assert_eq!(op.demand_hint(), DemandHint::SmallTile);
        assert_eq!(
            op.required_input_region(&out_region),
            Region::new(
                out_region.x - op.halo as i32,
                out_region.y - op.halo as i32,
                out_region.width + 2 * op.halo as u32,
                out_region.height + 2 * op.halo as u32,
            )
        );
        assert_eq!(
            op.node_spec(7, 9),
            NodeSpec {
                input_tile_w: 7 + 2 * op.halo as u32,
                input_tile_h: 9 + 2 * op.halo as u32,
                output_tile_w: 7,
                output_tile_h: 9,
                coordinate_driven_source: None,
            }
        );
    }

    #[test]
    fn zero_multiband_input_stays_zero() {
        let op = Canny::<F32>::new(1.4);
        let halo = op.halo;
        let width = 3usize;
        let height = 2usize;
        let bands = 2usize;
        let in_region = Region::new(0, 0, (width + 2 * halo) as u32, (height + 2 * halo) as u32);
        let out_region = Region::new(0, 0, width as u32, height as u32);
        let input_data = vec![0.0f32; (width + 2 * halo) * (height + 2 * halo) * bands];
        let input = Tile::<F32>::new(in_region, bands as u32, &input_data);
        let mut output_data = vec![1.0f32; width * height * bands];
        let mut output = TileMut::<F32>::new(out_region, bands as u32, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert!(output_data.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn gradient_matches_bottom_right_anchored_lipvips_mask() {
        let op = Canny::<F32>::new(0.5);
        let region = Region::new(0, 0, 4, 4);
        let input_data = vec![
            0.0f32, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 6.0,
        ];
        let input = Tile::<F32>::new(region, 1, &input_data);

        let (gx, gy) = op.gradient_at(&input, 4, 1, 1, 1, 0);
        let a = op.blurred_at(&input, 4, 1, 0, 0, 0);
        let b = op.blurred_at(&input, 4, 1, 1, 0, 0);
        let c = op.blurred_at(&input, 4, 1, 0, 1, 0);
        let d = op.blurred_at(&input, 4, 1, 1, 1, 0);

        assert!((gx - (-a + b - c + d)).abs() < 1e-9);
        assert!((gy - (-a - b + c + d)).abs() < 1e-9);
    }

    #[test]
    fn step_edge_peak_is_centered_on_libvips_column() {
        let op = Canny::<F32>::new(1.4);
        let halo = op.halo;
        let width = 16usize;
        let height = 10usize;
        let edge_x = width / 2;
        let padded_width = width + 2 * halo;
        let padded_height = height + 2 * halo;
        let in_region = Region::new(0, 0, padded_width as u32, padded_height as u32);
        let out_region = Region::new(0, 0, width as u32, height as u32);
        let mut input_data = vec![0.0f32; padded_width * padded_height];

        for py in 0..padded_height {
            for px in 0..padded_width {
                let sx = (px.saturating_sub(halo)).min(width - 1);
                if sx >= edge_x {
                    input_data[py * padded_width + px] = 255.0;
                }
            }
        }

        let input = Tile::<F32>::new(in_region, 1, &input_data);
        let mut output_data = vec![0.0f32; width * height];
        let mut output = TileMut::<F32>::new(out_region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        for row in output_data.chunks_exact(width) {
            let peak = row
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(peak, edge_x);
        }
    }
}
