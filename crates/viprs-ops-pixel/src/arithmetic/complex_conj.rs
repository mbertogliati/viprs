use viprs_core::{
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Complex conjugate over interleaved `(re, im)` samples.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::complex_conj::ComplexConjOp;
///
/// let op = ComplexConjOp;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ComplexConjOp;

impl Op for ComplexConjOp {
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
        debug_assert_eq!(
            input.bands % 2,
            0,
            "ComplexConjOp: input bands must be even"
        );
        debug_assert_eq!(input.bands, output.bands);

        let bands = input.bands as usize;
        let pixel_count = input.region.pixel_count();
        for pixel in 0..pixel_count {
            let base = pixel * bands;
            for band in (0..bands).step_by(2) {
                output.data[base + band] = input.data[base + band];
                output.data[base + band + 1] = -input.data[base + band + 1];
            }
        }
    }
}

impl PixelLocalOp for ComplexConjOp {}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::Region;

    #[test]
    fn conjugates_single_complex_value() {
        let op = ComplexConjOp;
        let input_data = [3.0f32, 4.0];
        let mut output_data = [0.0f32; 2];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [3.0, -4.0]);
    }

    #[test]
    fn zero_imaginary_part_is_identity() {
        let op = ComplexConjOp;
        let input_data = [1.0f32, 0.0, -2.0, 0.0];
        let mut output_data = [0.0f32; 4];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output = TileMut::<F32>::new(region, 2, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn conjugates_all_complex_pairs_across_pixels() {
        let op = ComplexConjOp;
        let input_data = [1.0f32, 2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0];
        let mut output_data = [0.0f32; 8];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::<F32>::new(region, 4, &input_data);
        let mut output = TileMut::<F32>::new(region, 4, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, [1.0, -2.0, 3.0, -4.0, -5.0, -6.0, 7.0, 8.0]);
    }

    #[test]
    fn complex_conj_reports_pixel_local_geometry_contract() {
        let op = ComplexConjOp;
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }
}
