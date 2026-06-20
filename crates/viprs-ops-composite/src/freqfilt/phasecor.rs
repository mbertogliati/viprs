#![allow(clippy::unused_self)]
// REASON: helper methods remain instance methods for symmetry with other frequency-filter ops.

use std::{any::Any, marker::PhantomData};

use crate::freqfilt::COMPLEX_BANDS;

use viprs_core::{
    error::{FreqfiltError, ViprsError},
    format::{BandFormat, FloatFormat, FloatSample},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Normalized cross-power spectrum for phase correlation.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::freqfilt::phasecor::PhasecorOp;
///
/// let op = PhasecorOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PhasecorOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> PhasecorOp<F> {
    #[must_use]
    /// Creates a new `PhasecorOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    const fn process_single_input_checked(&self) -> Result<(), ViprsError> {
        Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
            op: "PhasecorOp",
            expected: 2,
            actual: 1,
        }))
    }
}

impl<F> PhasecorOp<F>
where
    F: FloatFormat,
    F::Sample: FloatSample + ToF64 + FromF64 + Copy + bytemuck::Pod,
{
    fn process_region_multi_checked(
        &self,
        inputs: &[&[u8]],
        output: &mut [u8],
        output_region: Region,
    ) -> Result<(), ViprsError> {
        if inputs.len() != 2 {
            return Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "PhasecorOp",
                expected: 2,
                actual: inputs.len(),
            }));
        }

        let (Some(&lhs_bytes), Some(&rhs_bytes)) = (inputs.first(), inputs.get(1)) else {
            return Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "PhasecorOp",
                expected: 2,
                actual: inputs.len(),
            }));
        };

        let lhs = bytemuck::try_cast_slice::<u8, F::Sample>(lhs_bytes).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "PhasecorOp",
                buffer: "input[0]",
            })
        })?;
        let rhs = bytemuck::try_cast_slice::<u8, F::Sample>(rhs_bytes).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "PhasecorOp",
                buffer: "input[1]",
            })
        })?;
        let out = bytemuck::try_cast_slice_mut::<u8, F::Sample>(output).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "PhasecorOp",
                buffer: "output",
            })
        })?;

        let pixel_count = output_region.pixel_count();
        let expected_len = pixel_count * COMPLEX_BANDS as usize;

        if lhs.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "PhasecorOp",
                    buffer: "input[0]",
                    expected: expected_len,
                    actual: lhs.len(),
                },
            ));
        }
        if rhs.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "PhasecorOp",
                    buffer: "input[1]",
                    expected: expected_len,
                    actual: rhs.len(),
                },
            ));
        }
        if out.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "PhasecorOp",
                    buffer: "output",
                    expected: expected_len,
                    actual: out.len(),
                },
            ));
        }

        for pixel in 0..pixel_count {
            let idx = pixel * COMPLEX_BANDS as usize;
            let a_re = lhs[idx].to_f64();
            let a_im = lhs[idx + 1].to_f64();
            let b_re = rhs[idx].to_f64();
            let b_im = rhs[idx + 1].to_f64();

            let cross_re = a_im.mul_add(b_im, a_re * b_re);
            let cross_im = a_re.mul_add(-b_im, a_im * b_re);
            let magnitude = cross_re.hypot(cross_im);

            if magnitude == 0.0 {
                out[idx] = F::Sample::from_f64(0.0);
                out[idx + 1] = F::Sample::from_f64(0.0);
            } else {
                out[idx] = F::Sample::from_f64(cross_re / magnitude);
                out[idx + 1] = F::Sample::from_f64(cross_im / magnitude);
            }
        }

        Ok(())
    }
}

impl<F> DynOperation for PhasecorOp<F>
where
    F: FloatFormat,
    F::Sample: FloatSample + ToF64 + FromF64 + Copy + bytemuck::Pod,
{
    fn input_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn output_format(&self) -> viprs_core::format::BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        COMPLEX_BANDS
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        2
    }

    fn required_input_region_slot(&self, output: &Region, _slot: usize) -> Region {
        *output
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

    fn validate_region_contract(
        &self,
        input_region: Region,
        input_bands: u32,
        output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        if input_region != output_region
            || input_bands != COMPLEX_BANDS
            || output_bands != COMPLEX_BANDS
        {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "PhasecorOp",
                    buffer: "region/bands contract",
                    expected: COMPLEX_BANDS as usize,
                    actual: input_bands.max(output_bands) as usize,
                },
            ));
        }

        output_region
            .checked_pixel_count()
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width: output_region.width,
                height: output_region.height,
                bands: COMPLEX_BANDS,
                bytes: u128::from(output_region.width)
                    * u128::from(output_region.height)
                    * u128::from(COMPLEX_BANDS),
                limit_bytes: usize::MAX as u128,
                details: "phasecor sample count exceeds addressable memory",
            })?;
        Ok(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        _input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        if self.process_single_input_checked().is_err() {
            output.fill(0);
        }
    }

    #[inline]
    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        _input_regions: &[Region],
        output_region: Region,
    ) {
        if self
            .process_region_multi_checked(inputs, output, output_region)
            .is_err()
        {
            output.fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::freqfilt::COMPLEX_BANDS;

    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, F64},
        image::{DemandHint, Region, Tile, TileMut},
        op::{DynOperation, NodeSpec, Op},
    };

    fn make_region(pixel_count: usize) -> Region {
        Region::new(0, 0, pixel_count as u32, 1)
    }

    fn run_phasecor(a: &[f32], b: &[f32]) -> Vec<f32> {
        run_phasecor_typed::<F32>(a, b)
    }

    fn run_phasecor_typed<Fmt>(a: &[Fmt::Sample], b: &[Fmt::Sample]) -> Vec<Fmt::Sample>
    where
        Fmt: FloatFormat,
        Fmt::Sample: FloatSample + ToF64 + FromF64 + Copy + bytemuck::Pod,
    {
        let op = PhasecorOp::<Fmt>::new();
        let pixel_count = a.len() / COMPLEX_BANDS as usize;
        let region = make_region(pixel_count);
        let inputs: [&[u8]; 2] = [bytemuck::cast_slice(a), bytemuck::cast_slice(b)];
        let input_regions = [region; 2];
        let mut output = vec![Fmt::Sample::from_f64(0.0); a.len()];
        let mut state = op.dyn_start();

        op.dyn_process_region_multi(
            state.as_mut(),
            &inputs,
            bytemuck::cast_slice_mut(output.as_mut_slice()),
            &input_regions,
            region,
        );

        output
    }

    #[test]
    fn tiny_cross_power_remains_normalized_in_f64() {
        let lhs = [1.0e-150_f64, 1.0e-150, -1.0e-150, 1.0e-150];
        let rhs = [1.0e-150_f64, -1.0e-150, 1.0e-150, 1.0e-150];

        let output = run_phasecor_typed::<F64>(&lhs, &rhs);

        for pair in output.chunks_exact(COMPLEX_BANDS as usize) {
            let magnitude = pair[0].hypot(pair[1]);
            assert!((magnitude - 1.0).abs() <= 1e-12);
        }
    }

    prop_compose! {
        fn non_zero_complex_line()
            (width in 1usize..=16)
            (
                samples in prop::collection::vec((-64.0f32..64.0f32, -64.0f32..64.0f32), width),
            ) -> Vec<f32> {
                samples
                    .into_iter()
                    .flat_map(|(re, im)| {
                        let re = if re == 0.0 && im == 0.0 { 1.0 } else { re };
                        [re, im]
                    })
                    .collect()
            }
    }

    proptest! {
        #[test]
        fn identical_inputs_produce_unit_phase(samples in non_zero_complex_line()) {
            let output = run_phasecor(&samples, &samples);

            for pair in output.chunks_exact(COMPLEX_BANDS as usize) {
                let magnitude = (pair[0] * pair[0] + pair[1] * pair[1]).sqrt();
                prop_assert!((1.0 - magnitude).abs() <= 1e-5);
                prop_assert!(pair[1].abs() <= 1e-5);
            }
        }
    }

    #[test]
    fn self_correlation_matches_requested_example() {
        let input = vec![2.0f32, 1.0, -3.0, 4.0];

        let output = run_phasecor(&input, &input);

        assert!((output[0] - 1.0).abs() <= 1e-6);
        assert!(output[1].abs() <= 1e-6);
        assert!((output[2] - 1.0).abs() <= 1e-6);
        assert!(output[3].abs() <= 1e-6);
    }

    #[test]
    fn zero_cross_power_returns_zero_pair() {
        let zeros = vec![0.0f32, 0.0, 0.0, 0.0];

        let output = run_phasecor(&zeros, &zeros);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn phasecor_reports_two_input_complex_contract() {
        let op = PhasecorOp::<F32>::new();
        let region = Region::new(1, 2, 3, 4);

        assert_eq!(op.input_format(), op.output_format());
        assert_eq!(op.bands(), COMPLEX_BANDS);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region_slot(&region, 1), region);
        assert_eq!(op.node_spec(9, 6), NodeSpec::identity(9, 6));
    }

    #[test]
    fn single_input_dispatch_returns_typed_error() {
        let op = PhasecorOp::<F32>::new();
        let err = op.process_single_input_checked().unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "PhasecorOp",
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn missing_secondary_input_returns_typed_error() {
        let op = PhasecorOp::<F32>::new();
        let region = make_region(1);
        let lhs = vec![1.0f32, 2.0];
        let inputs: [&[u8]; 1] = [bytemuck::cast_slice(&lhs)];
        let mut output = vec![6.0f32; 2];
        let err = op
            .process_region_multi_checked(
                &inputs,
                bytemuck::cast_slice_mut(output.as_mut_slice()),
                region,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "PhasecorOp",
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn invalid_input_bytes_return_typed_error() {
        let op = PhasecorOp::<F32>::new();
        let region = make_region(1);
        let lhs = [0_u8; 1];
        let rhs = [0_u8; 1];
        let inputs = [&lhs[..], &rhs[..]];
        let mut output = vec![4.0f32; 2];
        let err = op
            .process_region_multi_checked(
                &inputs,
                bytemuck::cast_slice_mut(output.as_mut_slice()),
                region,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "PhasecorOp",
                buffer: "input[0]"
            })
        ));
    }

    #[test]
    fn mismatched_input_lengths_return_typed_error() {
        let op = PhasecorOp::<F32>::new();
        let region = make_region(1);
        let lhs = vec![1.0f32, 2.0, 3.0, 4.0];
        let rhs = vec![5.0f32, 6.0];
        let inputs = [
            bytemuck::cast_slice(lhs.as_slice()),
            bytemuck::cast_slice(rhs.as_slice()),
        ];
        let mut output = vec![0.0f32; 4];
        let err = op
            .process_region_multi_checked(
                &inputs,
                bytemuck::cast_slice_mut(output.as_mut_slice()),
                region,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferLength {
                op: "PhasecorOp",
                buffer: "input[0]",
                expected: 2,
                actual: 4
            })
        ));
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
    fn run_invfft(width: u32, height: u32, input_data: &[f32]) -> Vec<f32> {
        let op = crate::freqfilt::InvFftOp::<F32>::new(width, height)
            .expect("InvFftOp should construct");
        let region = Region::new(0, 0, width, height);
        let input = Tile::<F32>::new(region, COMPLEX_BANDS, input_data);
        let mut output_data = vec![0.0f32; width as usize * height as usize];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[cfg(feature = "fft")]
    fn circular_shift(
        input: &[f32],
        width: usize,
        height: usize,
        dx: usize,
        dy: usize,
    ) -> Vec<f32> {
        let mut shifted = vec![0.0f32; input.len()];

        for y in 0..height {
            for x in 0..width {
                let src_x = (x + width - dx % width) % width;
                let src_y = (y + height - dy % height) % height;
                shifted[y * width + x] = input[src_y * width + src_x];
            }
        }

        shifted
    }

    #[cfg(feature = "fft")]
    fn peak_position(image: &[f32], width: usize) -> (usize, usize, f32) {
        let (index, &value) = image
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .expect("phase correlation output must not be empty");

        (index % width, index / width, value)
    }

    #[cfg(feature = "fft")]
    fn run_phasecor_peak_image(width: u32, height: u32, lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
        let lhs_fft = run_fwfft(width, height, lhs);
        let rhs_fft = run_fwfft(width, height, rhs);
        let cross_power = run_phasecor(&lhs_fft, &rhs_fft);

        run_invfft(width, height, &cross_power)
    }

    #[cfg(feature = "fft")]
    #[test]
    fn identical_images_peak_at_origin() {
        let width = 4;
        let height = 4;
        let image = vec![
            1.0f32, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];

        let correlation = run_phasecor_peak_image(width, height, &image, &image);
        let (peak_x, peak_y, peak_value) = peak_position(&correlation, width as usize);

        assert_eq!((peak_x, peak_y), (0, 0));
        assert!((peak_value - (width * height) as f32).abs() <= 1e-4);
    }

    #[cfg(feature = "fft")]
    #[test]
    fn shifted_image_peak_matches_offset() {
        let width = 8;
        let height = 8;
        let dx = 2;
        let dy = 3;
        let mut image = vec![0.0f32; width * height];
        image[1] = 1.0;
        image[width + 4] = 0.5;
        image[height * width - 3] = 0.25;
        let shifted = circular_shift(&image, width, height, dx, dy);

        let correlation = run_phasecor_peak_image(width as u32, height as u32, &shifted, &image);
        let (peak_x, peak_y, peak_value) = peak_position(&correlation, width);

        assert_eq!((peak_x, peak_y), (dx, dy));
        assert!((peak_value - (width * height) as f32).abs() <= 1e-4);
    }
}
