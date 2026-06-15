use std::sync::Arc;

use crate::{
    domain::op::{NodeSpec, Op, PixelLocalOp},
    domain::{
        format::{FloatFormat, FloatSample, Math2Sample},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Enumerates the available math2 mode values.
pub enum Math2Mode {
    /// Uses the `Pow` variant of `Math2Mode`.
    Pow,
    /// Uses the `Wop` variant of `Math2Mode`.
    Wop,
    /// Uses the `Atan2` variant of `Math2Mode`.
    Atan2,
}

/// Binary math operation with a synchronized right-hand-side image.
///
/// Matches libvips `math2` semantics for `pow`, `wop`, and `atan2` over
/// float images. The right-hand side may be single-band or match input bands.
pub struct Math2<F: FloatFormat>
where
    F::Sample: FloatSample + Math2Sample,
{
    rhs: Arc<[F::Sample]>,
    image_width: usize,
    rhs_bands: usize,
    mode: Math2Mode,
}

impl<F: FloatFormat> Math2<F>
where
    F::Sample: FloatSample + Math2Sample,
{
    #[must_use]
    /// Creates a new `Math2`.
    pub fn new<R>(rhs: R, image_width: u32, bands: u32, mode: Math2Mode) -> Self
    where
        R: Into<Arc<[F::Sample]>>,
    {
        Self {
            rhs: rhs.into(),
            image_width: image_width as usize,
            rhs_bands: bands as usize,
            mode,
        }
    }

    #[inline(always)]
    fn apply(&self, left: F::Sample, right: F::Sample) -> F::Sample {
        match self.mode {
            Math2Mode::Pow => left.s_pow2(right),
            Math2Mode::Wop => left.s_wop(right),
            Math2Mode::Atan2 => left.s_atan2(right),
        }
    }
}

impl<F> Op for Math2<F>
where
    F: FloatFormat,
    F::Sample: FloatSample + Math2Sample,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.data.len(), output.data.len());
        debug_assert_eq!(input.bands, output.bands);
        debug_assert!(input.region.x >= 0 && input.region.y >= 0);

        let output_bands = input.bands as usize;
        let rhs_bands = self.rhs_bands;
        let tile_width = input.region.width as usize;
        let tile_height = input.region.height as usize;
        let tile_row_samples = tile_width * output_bands;
        let rhs_row_samples = tile_width * rhs_bands;
        let rhs_row_stride = self.image_width * rhs_bands;
        let start_x = input.region.x as usize * rhs_bands;
        let start_y = input.region.y as usize;
        let required_rhs_len = start_y
            .checked_add(tile_height)
            .and_then(|rows| rows.checked_mul(rhs_row_stride))
            .unwrap_or(usize::MAX);

        debug_assert!(rhs_bands == 1 || rhs_bands == output_bands);
        debug_assert!(start_x <= rhs_row_stride);
        debug_assert!(rhs_row_samples <= rhs_row_stride.saturating_sub(start_x));
        debug_assert!(self.rhs.len() >= required_rhs_len);

        for row in 0..tile_height {
            let tile_offset = row * tile_row_samples;
            let rhs_offset = (start_y + row) * rhs_row_stride + start_x;
            let src_row = &input.data[tile_offset..tile_offset + tile_row_samples];
            let dst_row = &mut output.data[tile_offset..tile_offset + tile_row_samples];
            let rhs_row = &self.rhs[rhs_offset..rhs_offset + rhs_row_samples];

            if rhs_bands == output_bands {
                for ((sample, rhs), dst) in
                    src_row.iter().zip(rhs_row.iter()).zip(dst_row.iter_mut())
                {
                    *dst = self.apply(*sample, *rhs);
                }
            } else {
                for (pixel, rhs) in rhs_row.iter().copied().enumerate().take(tile_width) {
                    let src_base = pixel * output_bands;
                    for band in 0..output_bands {
                        let idx = src_base + band;
                        dst_row[idx] = self.apply(src_row[idx], rhs);
                    }
                }
            }
        }
    }
}

impl<F> PixelLocalOp for Math2<F>
where
    F: FloatFormat,
    F::Sample: FloatSample + Math2Sample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{format::F32, image::Region};
    use proptest::prelude::*;

    fn run(op: &Math2<F32>, input_data: &[f32], width: u32, height: u32, bands: u32) -> Vec<f32> {
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F32>::new(region, bands, input_data);
        let mut output_data = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, bands, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    proptest! {
        #[test]
        fn pow_identity_with_rhs_one(pixels in proptest::collection::vec(0.01f32..=10.0f32, 1..=64)) {
            let len = pixels.len();
            let op = Math2::<F32>::new(vec![1.0f32; len], len as u32, 1, Math2Mode::Pow);
            let output = run(&op, &pixels, len as u32, 1, 1);
            for (actual, expected) in output.iter().zip(pixels.iter()) {
                prop_assert!((*actual - *expected).abs() < 1e-6);
            }
        }

        #[test]
        fn atan2_matches_reference(
            (left, right) in (1usize..=64).prop_flat_map(|len| {
                (
                    proptest::collection::vec(-10.0f32..=10.0f32, len),
                    proptest::collection::vec(-10.0f32..=10.0f32, len),
                )
            })
        ) {
            let len = left.len();
            let op = Math2::<F32>::new(right.clone(), len as u32, 1, Math2Mode::Atan2);
            let output = run(&op, &left, len as u32, 1, 1);
            for ((actual, lhs), rhs) in output.iter().zip(left.iter()).zip(right.iter()) {
                let expected = lhs.atan2(*rhs).to_degrees().rem_euclid(360.0);
                prop_assert!((*actual - expected).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn pow_zero_base_with_negative_exponent_stays_zero() {
        let op = Math2::<F32>::new(vec![-1.0f32], 1, 1, Math2Mode::Pow);
        let output = run(&op, &[0.0f32], 1, 1, 1);
        assert_eq!(output, vec![0.0f32]);
    }

    #[test]
    fn atan2_boundary_angles_match_libvips_quadrants() {
        let op = Math2::<F32>::new(vec![1.0, 0.0, -1.0, 0.0], 4, 1, Math2Mode::Atan2);
        let output = run(&op, &[0.0, 1.0, 0.0, -1.0], 4, 1, 1);
        let expected = [0.0, 90.0, 180.0, 270.0];
        for (actual, expected) in output.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() < 1e-5,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn wop_reverses_pow_operands() {
        let op = Math2::<F32>::new(vec![2.0f32, 4.0], 2, 1, Math2Mode::Wop);
        let output = run(&op, &[3.0f32, 0.5], 2, 1, 1);
        assert!((output[0] - 8.0).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn pow_uses_multiband_rhs_without_broadcasting() {
        let op = Math2::<F32>::new(vec![1.0, 2.0, 3.0, 2.0], 2, 2, Math2Mode::Pow);
        let output = run(&op, &[2.0, 3.0, 4.0, 5.0], 2, 1, 2);
        assert_eq!(output, vec![2.0, 9.0, 64.0, 25.0]);
    }

    #[test]
    fn single_band_rhs_broadcasts_across_all_bands_and_rows() {
        let rhs = vec![1.0, 2.0, 3.0, 4.0, 8.0, 9.0];
        let op = Math2::<F32>::new(rhs, 2, 1, Math2Mode::Pow);
        let region = Region::new(0, 1, 2, 2);
        let input_data = [2.0, 3.0, 4.0, 5.0, 2.0, 2.0, 3.0, 3.0];
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output_data = vec![0.0f32; input_data.len()];
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(
            output_data,
            vec![8.0, 27.0, 256.0, 625.0, 256.0, 256.0, 19_683.0, 19_683.0]
        );
    }

    #[test]
    fn metadata_helpers_match_identity_pixel_local_contract() {
        let op = Math2::<F32>::new(vec![1.0, 1.0, 1.0], 3, 1, Math2Mode::Pow);
        let region = Region::new(4, 5, 3, 2);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(8, 9), NodeSpec::identity(8, 9));
        op.start();
    }
}
