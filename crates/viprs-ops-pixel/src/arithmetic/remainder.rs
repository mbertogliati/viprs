use crate::arithmetic::rhs_broadcast::{RhsLayout, detect_rhs_layout};
use viprs_core::{
    format::{BandFormat, RemSample},
    image::{DemandHint, Region, Tile, TileMut},
    op::{NodeSpec, Op, PixelLocalOp},
};

/// Element-wise remainder against a fixed right-hand-side buffer.
///
/// The rhs buffer is allocated once at construction time and may contain full-tile,
/// scalar, per-band, or single-band-image data.
pub struct Remainder<F: BandFormat>
where
    F::Sample: RemSample,
{
    rhs: Vec<F::Sample>,
}

impl<F: BandFormat> Remainder<F>
where
    F::Sample: RemSample,
{
    #[must_use]
    /// Creates a new `Remainder`.
    pub const fn new(rhs: Vec<F::Sample>) -> Self {
        Self { rhs }
    }
}

impl<F: BandFormat> Op for Remainder<F>
where
    F::Sample: RemSample,
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
        let bands = input.bands as usize;
        let layout = detect_rhs_layout(self.rhs.len(), input.data.len(), bands);
        debug_assert!(
            layout.is_some(),
            "Remainder: rhs must match full tile, scalar, per-band, or single-band-image layout"
        );
        debug_assert_eq!(
            input.data.len(),
            output.data.len(),
            "Remainder: input and output must have same length"
        );

        match layout {
            Some(RhsLayout::Direct) => {
                for ((sample, rhs), out) in input
                    .data
                    .iter()
                    .zip(self.rhs.iter())
                    .zip(output.data.iter_mut())
                {
                    *out = sample.s_remainder(*rhs);
                }
            }
            Some(RhsLayout::Scalar) => {
                let rhs = self.rhs[0];
                for (sample, out) in input.data.iter().zip(output.data.iter_mut()) {
                    *out = sample.s_remainder(rhs);
                }
            }
            Some(RhsLayout::PerBand) => {
                for ((index, sample), out) in
                    input.data.iter().enumerate().zip(output.data.iter_mut())
                {
                    *out = sample.s_remainder(self.rhs[index % bands]);
                }
            }
            Some(RhsLayout::SingleBandImage) => {
                for ((index, sample), out) in
                    input.data.iter().enumerate().zip(output.data.iter_mut())
                {
                    *out = sample.s_remainder(self.rhs[index / bands]);
                }
            }
            None => {}
        }
    }
}

impl<F: BandFormat> PixelLocalOp for Remainder<F> where F::Sample: RemSample {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, I32, U16},
        image::Region,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    #[test]
    fn remainder_i32_known_values() {
        let op = Remainder::<I32>::new(vec![2, 2, 0]);
        let r = make_region(3, 1);
        let input_data = vec![5, -5, 7];
        let mut output_data = vec![0; 3];
        let input = Tile::<I32>::new(r, 1, &input_data);
        let mut output = TileMut::<I32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![1, -1, -1]);
    }

    #[test]
    fn remainder_f32_known_values() {
        let op = Remainder::<F32>::new(vec![2.0, 2.0, 0.0]);
        let r = make_region(3, 1);
        let input_data = vec![5.5f32, -5.5, 7.0];
        let mut output_data = vec![0.0f32; 3];
        let input = Tile::<F32>::new(r, 1, &input_data);
        let mut output = TileMut::<F32>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert!((output_data[0] - 1.5).abs() < f32::EPSILON);
        assert!((output_data[1] - 0.5).abs() < f32::EPSILON);
        assert_eq!(output_data[2], -1.0);
    }

    #[test]
    fn remainder_u16_zero_divisor_returns_max() {
        let op = Remainder::<U16>::new(vec![0, 6, 0]);
        let r = make_region(3, 1);
        let input_data = vec![5u16, 17, 9];
        let mut output_data = vec![0u16; 3];
        let input = Tile::<U16>::new(r, 1, &input_data);
        let mut output = TileMut::<U16>::new(r, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![u16::MAX, 5, u16::MAX]);
    }

    #[test]
    fn remainder_single_band_rhs_expands_across_multiband_input() {
        let op = Remainder::<U16>::new(vec![5, 7]);
        let r = make_region(2, 1);
        let input_data = vec![9u16, 10, 11, 13, 14, 15];
        let mut output_data = vec![0u16; 6];
        let input = Tile::<U16>::new(r, 3, &input_data);
        let mut output = TileMut::<U16>::new(r, 3, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![4u16, 0, 1, 6, 0, 1]);
    }

    #[test]
    fn remainder_metadata_matches_identity_geometry() {
        let op = Remainder::<F32>::new(vec![1.0, 1.0, 1.0]);
        let region = make_region(3, 1);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(3, 1), viprs_core::op::NodeSpec::identity(3, 1));
    }

    #[test]
    fn remainder_scalar_and_per_band_layouts_cover_all_rhs_dispatch_modes() {
        let scalar = Remainder::<U16>::new(vec![5]);
        let per_band = Remainder::<U16>::new(vec![5, 7, 9]);
        let region = make_region(2, 1);
        let input_data = vec![9u16, 10, 11, 13, 14, 15];

        let mut scalar_out = vec![0u16; 6];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output = TileMut::<U16>::new(region, 3, &mut scalar_out);
        scalar.process_region(&mut (), &input, &mut output);
        assert_eq!(scalar_out, vec![4, 0, 1, 3, 4, 0]);

        let mut per_band_out = vec![0u16; 6];
        let input = Tile::<U16>::new(region, 3, &input_data);
        let mut output = TileMut::<U16>::new(region, 3, &mut per_band_out);
        per_band.process_region(&mut (), &input, &mut output);
        assert_eq!(per_band_out, vec![4, 3, 2, 3, 0, 6]);
    }

    proptest! {
        #[test]
        fn remainder_f32_by_self_is_zero(
            pixels in proptest::collection::vec(0.001f32..=1000.0f32, 1..=32)
        ) {
            let len = pixels.len();
            let op = Remainder::<F32>::new(pixels.clone());
            let r = Region::new(0, 0, len as u32, 1);
            let mut result = vec![1.0f32; len];
            {
                let input = Tile::<F32>::new(r, 1, &pixels);
                let mut output = TileMut::<F32>::new(r, 1, &mut result);
                let mut state = ();
                op.process_region(&mut state, &input, &mut output);
            }
            for value in &result {
                prop_assert!(value.abs() < 1e-5, "x % x != 0: {}", value);
            }
        }
    }
}
