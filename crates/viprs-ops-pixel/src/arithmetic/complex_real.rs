use std::{any::Any, marker::PhantomData};

use viprs_core::{
    format::{BandFormatId, F32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{DynOperation, Op},
};

/// Extract the real component from interleaved `(re, im)` samples.
///
/// # Examples
/// ```ignore
/// use viprs_ops_pixel::arithmetic::complex_real::ComplexRealOp;
///
/// let op = ComplexRealOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ComplexRealOp {
    input_bands: usize,
    _phantom: PhantomData<F32>,
}

impl ComplexRealOp {
    #[must_use]
    /// Creates a new `ComplexRealOp`.
    pub fn new(input_bands: usize) -> Self {
        debug_assert!(
            input_bands > 0 && input_bands.is_multiple_of(2),
            "ComplexRealOp: input_bands must be a positive even number"
        );
        Self {
            input_bands,
            _phantom: PhantomData,
        }
    }

    const fn output_bands(&self) -> u32 {
        (self.input_bands / 2) as u32
    }
}

impl Op for ComplexRealOp {
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
        debug_assert_eq!(input.bands as usize, self.input_bands);
        debug_assert_eq!(output.bands, self.output_bands());

        let complex_bands = self.input_bands / 2;
        let pixel_count = input.region.pixel_count();
        for pixel in 0..pixel_count {
            let src_base = pixel * self.input_bands;
            let dst_base = pixel * complex_bands;
            for band in 0..complex_bands {
                output.data[dst_base + band] = input.data[src_base + band * 2];
            }
        }
    }
}

impl DynOperation for ComplexRealOp {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::F32
    }

    fn bands(&self) -> u32 {
        self.output_bands()
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
        let input_samples: &[f32] = if let Ok(samples) = bytemuck::try_cast_slice(input) {
            samples
        } else {
            debug_assert!(false, "ComplexRealOp: input cast failed");
            return;
        };
        let output_samples: &mut [f32] = if let Ok(samples) = bytemuck::try_cast_slice_mut(output) {
            samples
        } else {
            debug_assert!(false, "ComplexRealOp: output cast failed");
            return;
        };

        let input_tile = Tile::<F32>::new(input_region, self.input_bands as u32, input_samples);
        let mut output_tile =
            TileMut::<F32>::new(output_region, self.output_bands(), output_samples);
        let mut state = ();
        self.process_region(&mut state, &input_tile, &mut output_tile);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::image::Region;

    #[test]
    fn extracts_real_component() {
        let op = ComplexRealOp::new(2);
        let input_data = [3.0f32, 4.0];
        let mut output_data = [0.0f32; 1];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::<F32>::new(region, 2, &input_data);
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [3.0]);
    }

    #[test]
    fn dyn_operation_reports_half_band_count() {
        let op = ComplexRealOp::new(6);
        assert_eq!(op.bands(), 3);
    }

    #[test]
    fn dyn_process_region_extracts_each_real_component() {
        let op = ComplexRealOp::new(4);
        let input_data = [1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let mut output_data = [0.0f32; 4];
        let region = Region::new(0, 0, 2, 1);

        let mut state = op.dyn_start();
        op.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(&input_data),
            bytemuck::cast_slice_mut(&mut output_data),
            region,
            region,
        );

        assert_eq!(output_data, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Op::required_input_region(&op, &region), region);
        assert_eq!(DynOperation::required_input_region(&op, &region), region);
        assert_eq!(Op::demand_hint(&op), DemandHint::ThinStrip);
        assert_eq!(DynOperation::demand_hint(&op), DemandHint::ThinStrip);
    }

    #[test]
    fn metadata_reports_f32_formats_and_half_band_count() {
        let op = ComplexRealOp::new(6);
        assert_eq!(op.input_format(), BandFormatId::F32);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.bands(), 3);
    }

    #[test]
    fn extracts_multiple_real_bands_per_pixel() {
        let op = ComplexRealOp::new(6);
        let input_data = [
            1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, //
            4.0, 40.0, 5.0, 50.0, 6.0, 60.0,
        ];
        let mut output_data = [0.0f32; 6];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::<F32>::new(region, 6, &input_data);
        let mut output = TileMut::<F32>::new(region, 3, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    proptest! {
        #[test]
        fn extracts_real_component_for_each_pixel(
            re0 in -100.0f32..100.0,
            im0 in -100.0f32..100.0,
            re1 in -100.0f32..100.0,
            im1 in -100.0f32..100.0,
        ) {
            let op = ComplexRealOp::new(2);
            let input_data = [re0, im0, re1, im1];
            let mut output_data = [0.0f32; 2];
            let region = Region::new(0, 0, 2, 1);
            let input = Tile::<F32>::new(region, 2, &input_data);
            let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            prop_assert_eq!(output_data, [re0, re1]);
        }
    }
}
