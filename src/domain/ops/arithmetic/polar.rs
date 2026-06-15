use crate::{
    domain::op::{Op, PixelLocalOp},
    domain::{
        format::F32,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

/// Convert interleaved `(re, im)` samples into `(magnitude, angle_radians)`.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::polar::PolarOp;
///
/// let op = PolarOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct PolarOp;

impl Op for PolarOp {
    type Input = F32;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        debug_assert_eq!(input.bands % 2, 0, "PolarOp: input bands must be even");
        debug_assert_eq!(input.bands, output.bands);

        let bands = input.bands as usize;
        let pixels = input.region.pixel_count();
        for pixel in 0..pixels {
            let base = pixel * bands;
            for band in (0..bands).step_by(2) {
                let re = input.data[base + band];
                let im = input.data[base + band + 1];
                output.data[base + band] = re.hypot(im);
                output.data[base + band + 1] = im.atan2(re);
            }
        }
    }
}

impl PixelLocalOp for PolarOp {}

#[cfg(test)]
mod tests {
    use super::super::rect::RectOp;
    use super::*;
    use crate::domain::image::Region;
    use proptest::prelude::*;

    #[test]
    fn polar_matches_axis_aligned_vectors() {
        let op = PolarOp;
        let input_data = [1.0f32, 0.0, 0.0, 1.0];
        let mut output_data = [0.0f32; 4];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!(output_data[1].abs() < 1e-6);
        assert!((output_data[2] - 1.0).abs() < 1e-6);
        assert!((output_data[3] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn polar_handles_negative_real_axis() {
        let op = PolarOp;
        let input_data = [-2.0f32, 0.0];
        let mut output_data = [0.0f32; 2];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert!((output_data[0] - 2.0).abs() < 1e-6);
        assert!((output_data[1] - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn reports_identity_region_metadata() {
        let op = PolarOp;
        let region = Region::new(2, 3, 4, 5);

        op.start();
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn rect_round_trip_recovers_complex_input(
            re in -100.0f32..100.0,
            im in -100.0f32..100.0,
        ) {
            let polar = PolarOp;
            let rect = RectOp;
            let region = Region::new(0, 0, 1, 1);
            let input_data = [re, im];
            let mut polar_data = [0.0f32; 2];
            let mut rect_data = [0.0f32; 2];

            let input = Tile::<F32>::new(region, 2, &input_data);
            let mut polar_output = TileMut::<F32>::new(region, 2, &mut polar_data);
            let mut state = ();
            polar.process_region(&mut state, &input, &mut polar_output);

            let polar_tile = Tile::<F32>::new(region, 2, &polar_data);
            let mut rect_output = TileMut::<F32>::new(region, 2, &mut rect_data);
            rect.process_region(&mut state, &polar_tile, &mut rect_output);

            prop_assert!((rect_data[0] - re).abs() < 1e-4);
            prop_assert!((rect_data[1] - im).abs() < 1e-4);
        }
    }
}
