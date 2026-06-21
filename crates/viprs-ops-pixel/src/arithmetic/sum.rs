use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, Op},
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Sum all bands of each pixel into a single-band output.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::sum::SumOp;
///
/// let op = SumOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SumOp<F: BandFormat> {
    input_bands: usize,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> SumOp<F> {
    #[must_use]
    /// Creates a new `SumOp`.
    pub fn new(input_bands: usize) -> Self {
        debug_assert!(input_bands > 0, "SumOp: input_bands must be at least 1");
        Self {
            input_bands,
            _phantom: PhantomData,
        }
    }
}

impl<F> Op for SumOp<F>
where
    F: BandFormat,
    F::Sample: FromF64 + ToF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(input.bands as usize, self.input_bands);
        debug_assert_eq!(output.bands, 1);

        let pixels = input.region.pixel_count();
        for pixel in 0..pixels {
            let src_base = pixel * self.input_bands;
            let sum = input.data[src_base..src_base + self.input_bands]
                .iter()
                .fold(0.0, |acc, sample| acc + sample.to_f64());
            output.data[pixel] = F::Sample::from_f64(sum);
        }
    }
}

impl<F> DynOperation for SumOp<F>
where
    F: BandFormat,
    F::Sample: FromF64 + ToF64,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        input_region: Region,
        output_region: Region,
    ) {
        let input_samples: &[F::Sample] = if let Ok(samples) = bytemuck::try_cast_slice(input) {
            samples
        } else {
            debug_assert!(false, "SumOp: input cast failed");
            return;
        };
        let output_samples: &mut [F::Sample] =
            if let Ok(samples) = bytemuck::try_cast_slice_mut(output) {
                samples
            } else {
                debug_assert!(false, "SumOp: output cast failed");
                return;
            };

        let input_tile = Tile::<F>::new(input_region, self.input_bands as u32, input_samples);
        let mut output_tile = TileMut::<F>::new(output_region, 1, output_samples);
        let mut state = ();
        self.process_region(&mut state, &input_tile, &mut output_tile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{U8, U16},
        image::Region,
    };

    #[test]
    fn sums_three_band_pixel() {
        let op = SumOp::<U8>::new(3);
        let input_data = [10u8, 20, 30];
        let mut output_data = [0u8; 1];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [60]);
    }

    #[test]
    fn single_band_is_identity() {
        let op = SumOp::<U8>::new(1);
        let input_data = [1u8, 20, 255];
        let mut output_data = [0u8; 3];
        let region = Region::new(0, 0, 3, 1);
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn boundary_clamps_integer_overflow() {
        let op = SumOp::<U8>::new(2);
        let input_data = [255u8, 255];
        let mut output_data = [0u8; 1];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<U8>::new(region, 2, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [255]);
    }

    #[test]
    fn dyn_operation_reports_single_output_band() {
        let op = SumOp::<U16>::new(4);
        assert_eq!(op.bands(), 1);
    }

    #[test]
    fn sums_zero_and_max_pixels() {
        let zero_op = SumOp::<U8>::new(2);
        let zero_region = Region::new(0, 0, 1, 1);
        let zero_input = [0u8, 0];
        let mut zero_output = [1u8; 1];
        let input = Tile::<U8>::new(zero_region, 2, &zero_input);
        let mut output = TileMut::<U8>::new(zero_region, 1, &mut zero_output);
        let mut state = ();
        zero_op.process_region(&mut state, &input, &mut output);
        assert_eq!(zero_output, [0]);

        let max_op = SumOp::<U16>::new(2);
        let max_input = [u16::MAX, u16::MAX];
        let mut max_output = [0u16; 1];
        let input = Tile::<U16>::new(zero_region, 2, &max_input);
        let mut output = TileMut::<U16>::new(zero_region, 1, &mut max_output);
        max_op.process_region(&mut state, &input, &mut output);
        assert_eq!(max_output, [u16::MAX]);
    }

    #[test]
    fn dyn_process_region_sums_each_pixel() {
        let op = SumOp::<U16>::new(2);
        let region = Region::new(0, 0, 2, 1);
        let input_data = [3u16, 4, 10, 20];
        let mut output_data = [0u16; 2];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(&input_data),
            bytemuck::cast_slice_mut(&mut output_data),
            region,
            region,
        );

        assert_eq!(output_data, [7, 30]);
        assert_eq!(Op::required_input_region(&op, &region), region);
        assert_eq!(DynOperation::required_input_region(&op, &region), region);
        assert_eq!(Op::demand_hint(&op), DemandHint::ThinStrip);
        assert_eq!(DynOperation::demand_hint(&op), DemandHint::ThinStrip);
    }

    proptest! {
        #[test]
        fn sums_two_band_pixels(
            pixels in proptest::collection::vec((0u16..=1000u16, 0u16..=1000u16), 1..=16)
        ) {
            let op = SumOp::<U16>::new(2);
            let region = Region::new(0, 0, pixels.len() as u32, 1);
            let input_data: Vec<u16> = pixels
                .iter()
                .flat_map(|&(left, right)| [left, right])
                .collect();
            let expected: Vec<u16> = pixels
                .iter()
                .map(|&(left, right)| left.saturating_add(right))
                .collect();
            let mut output_data = vec![0u16; pixels.len()];
            let input = Tile::<U16>::new(region, 2, &input_data);
            let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input, &mut output);

            prop_assert_eq!(output_data, expected);
        }
    }
}
