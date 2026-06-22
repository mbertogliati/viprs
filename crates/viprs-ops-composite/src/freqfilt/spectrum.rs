use std::{
    any::Any,
    marker::PhantomData,
    ops::{Add, Mul},
};

use crate::freqfilt::COMPLEX_BANDS;

use viprs_core::{
    format::{BandFormat, FloatFormat, FloatSample},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Magnitude of a Fourier-domain image.
///
/// The libvips implementation finishes with `wrap()` to center the DC component.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::freqfilt::spectrum::SpectrumOp;
///
/// let op = SpectrumOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SpectrumOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

#[inline(always)]
const fn fftshift_destination(x: usize, y: usize, width: usize, height: usize) -> usize {
    let shifted_x = (x + width / 2) % width;
    let shifted_y = (y + height / 2) % height;
    shifted_y * width + shifted_x
}

impl<F: BandFormat> SpectrumOp<F> {
    #[must_use]
    /// Creates a new `SpectrumOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<F> DynOperation for SpectrumOp<F>
where
    F: FloatFormat,
    F::Sample:
        FloatSample + Add<Output = F::Sample> + Mul<Output = F::Sample> + Copy + bytemuck::Pod,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::FullImage
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn dyn_start(&self) -> Box<dyn Any + Send> {
        Box::new(())
    }

    #[inline]
    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        output_region: Region,
    ) {
        let Ok(input_samples) = bytemuck::try_cast_slice::<u8, F::Sample>(input) else {
            debug_assert!(false, "SpectrumOp: cast failed on input");
            output.fill(0);
            return;
        };
        let Ok(output_samples) = bytemuck::try_cast_slice_mut::<u8, F::Sample>(output) else {
            debug_assert!(false, "SpectrumOp: cast failed on output");
            return;
        };

        let width = output_region.width as usize;
        let height = output_region.height as usize;
        let pixel_count = output_region.pixel_count();
        debug_assert_eq!(
            input_samples.len(),
            pixel_count * COMPLEX_BANDS as usize,
            "SpectrumOp: input size mismatch"
        );
        debug_assert_eq!(
            output_samples.len(),
            pixel_count,
            "SpectrumOp: output size mismatch"
        );

        for pixel in 0..pixel_count {
            let idx = pixel * COMPLEX_BANDS as usize;
            let re = input_samples[idx];
            let im = input_samples[idx + 1];
            let shifted = fftshift_destination(pixel % width, pixel / width, width, height);
            output_samples[shifted] = (re * re + im * im).s_sqrt();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freqfilt::COMPLEX_BANDS;

    use proptest::prelude::*;
    use viprs_core::{
        format::F32,
        image::{DemandHint, Region, Tile, TileMut},
        op::{DynOperation, NodeSpec, Op},
    };

    fn make_region(width: usize, height: usize) -> Region {
        Region::new(0, 0, width as u32, height as u32)
    }

    fn run_spectrum(width: usize, height: usize, input: &[f32]) -> Vec<f32> {
        let op = SpectrumOp::<F32>::new();
        let pixel_count = width * height;
        let region = make_region(width, height);
        let mut output = vec![0.0f32; pixel_count];
        let mut state = op.dyn_start();

        op.dyn_process_region(
            state.as_mut(),
            bytemuck::cast_slice(input),
            bytemuck::cast_slice_mut(output.as_mut_slice()),
            region,
            region,
        );

        output
    }

    proptest! {
        #[test]
        fn dc_component_moves_to_center_after_fftshift(
            width in 1usize..=8,
            height in 1usize..=8,
            dc in -64.0f32..64.0f32,
        ) {
            let mut input = vec![0.0f32; width * height * COMPLEX_BANDS as usize];
            input[0] = dc;

            let output = run_spectrum(width, height, &input);
            let center = (height / 2) * width + (width / 2);

            for (index, value) in output.iter().enumerate() {
                if index == center {
                    prop_assert!((*value - dc.abs()).abs() <= 1e-6);
                } else {
                    prop_assert!(value.abs() <= 1e-6);
                }
            }
        }
    }

    #[test]
    fn fftshift_swaps_quadrants_for_two_by_two_input() {
        let input = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];

        let output = run_spectrum(2, 2, &input);

        assert_eq!(output, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn zero_complex_signal_has_zero_magnitude() {
        let input = vec![0.0f32, 0.0, 0.0, 0.0];

        let output = run_spectrum(2, 1, &input);

        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn spectrum_reports_single_band_full_image_contract() {
        let op = SpectrumOp::<F32>::new();
        let region = Region::new(4, 5, 6, 7);

        assert_eq!(op.input_format(), op.output_format());
        assert_eq!(op.bands(), 1);
        assert_eq!(op.demand_hint(), DemandHint::FullImage);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(12, 14), NodeSpec::identity(12, 14));
    }

    #[test]
    fn output_dimensions_match_input_pixels() {
        let width = 3;
        let height = 2;
        let input = vec![
            1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0, //
            4.0, 0.0, 5.0, 0.0, 6.0, 0.0,
        ];

        let output = run_spectrum(width, height, &input);

        assert_eq!(output.len(), width * height);
    }

    #[test]
    fn output_is_single_real_band() {
        let width = 2;
        let height = 2;
        let input = vec![
            3.0f32, 4.0, -5.0, 12.0, //
            8.0, -15.0, 7.0, 24.0,
        ];

        let output = run_spectrum(width, height, &input);

        assert_eq!(output.len(), width * height);
        assert_eq!(output, vec![25.0f32, 8.0f32.hypot(-15.0f32), 13.0, 5.0,]);
    }

    #[cfg(feature = "fft")]
    fn run_fwfft(width: u32, height: u32, input_data: &[f32]) -> Vec<f32> {
        let op =
            crate::freqfilt::FwFftOp::<F32>::new(width, height).expect("FwFftOp should construct");
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F32>::new(region, 1, input_data);
        let mut output_data =
            vec![0.0f32; width as usize * height as usize * COMPLEX_BANDS as usize];
        let mut output = TileMut::<F32>::new(region, COMPLEX_BANDS, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[cfg(feature = "fft")]
    #[test]
    fn constant_image_energy_moves_to_center_after_fftshift() {
        let width = 4;
        let height = 4;
        let input = vec![3.0f32; width * height];
        let complex_spectrum = run_fwfft(width as u32, height as u32, &input);

        let output = run_spectrum(width, height, &complex_spectrum);
        let center = (height / 2) * width + (width / 2);

        for (index, value) in output.iter().enumerate() {
            if index == center {
                assert!((*value - 3.0).abs() <= 1e-4);
            } else {
                assert!(value.abs() <= 1e-4);
            }
        }
    }
}
