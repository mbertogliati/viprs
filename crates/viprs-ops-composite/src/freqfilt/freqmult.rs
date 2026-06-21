#![allow(clippy::unused_self)]
// REASON: helper methods remain instance methods for symmetry with other frequency-filter ops.

use std::{
    any::Any,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use crate::freqfilt::COMPLEX_BANDS;

use viprs_core::{
    error::{FreqfiltError, ViprsError},
    format::{BandFormat, FloatFormat, FloatSample},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Point-wise complex multiplication of two Fourier-domain images.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::freqfilt::freqmult::FreqMultOp;
///
/// let op = FreqMultOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct FreqMultOp<F: BandFormat> {
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> FreqMultOp<F> {
    #[must_use]
    /// Creates a new `FreqMultOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    #[inline]
    fn expected_sample_len(output_region: Region) -> Result<usize, ViprsError> {
        output_region
            .checked_pixel_count()
            .and_then(|n| n.checked_mul(COMPLEX_BANDS as usize))
            .ok_or_else(|| ViprsError::ImageTooLarge {
                width: output_region.width,
                height: output_region.height,
                bands: COMPLEX_BANDS,
                bytes: u128::from(output_region.width)
                    * u128::from(output_region.height)
                    * u128::from(COMPLEX_BANDS),
                limit_bytes: usize::MAX as u128,
                details: "freqmult sample count exceeds addressable memory",
            })
    }

    const fn process_single_input_checked(&self) -> Result<(), ViprsError> {
        Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
            op: "FreqMultOp",
            expected: 2,
            actual: 1,
        }))
    }
}

impl<F> FreqMultOp<F>
where
    F: FloatFormat,
    F::Sample: FloatSample
        + Add<Output = F::Sample>
        + Mul<Output = F::Sample>
        + Sub<Output = F::Sample>
        + Copy
        + bytemuck::Pod,
{
    fn process_region_multi_checked(
        &self,
        inputs: &[&[u8]],
        output: &mut [u8],
        output_region: Region,
    ) -> Result<(), ViprsError> {
        if inputs.len() != 2 {
            return Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "FreqMultOp",
                expected: 2,
                actual: inputs.len(),
            }));
        }

        let (Some(&lhs_bytes), Some(&rhs_bytes)) = (inputs.first(), inputs.get(1)) else {
            return Err(ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "FreqMultOp",
                expected: 2,
                actual: inputs.len(),
            }));
        };

        let lhs = bytemuck::try_cast_slice::<u8, F::Sample>(lhs_bytes).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "FreqMultOp",
                buffer: "input[0]",
            })
        })?;
        let rhs = bytemuck::try_cast_slice::<u8, F::Sample>(rhs_bytes).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "FreqMultOp",
                buffer: "input[1]",
            })
        })?;
        let out = bytemuck::try_cast_slice_mut::<u8, F::Sample>(output).map_err(|_| {
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "FreqMultOp",
                buffer: "output",
            })
        })?;

        let expected_len = Self::expected_sample_len(output_region)?;
        if lhs.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "FreqMultOp",
                    buffer: "input[0]",
                    expected: expected_len,
                    actual: lhs.len(),
                },
            ));
        }
        if rhs.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "FreqMultOp",
                    buffer: "input[1]",
                    expected: expected_len,
                    actual: rhs.len(),
                },
            ));
        }
        if out.len() != expected_len {
            return Err(ViprsError::Freqfilt(
                FreqfiltError::MultiInputBufferLength {
                    op: "FreqMultOp",
                    buffer: "output",
                    expected: expected_len,
                    actual: out.len(),
                },
            ));
        }

        for pixel in 0..output_region.pixel_count() {
            let idx = pixel * COMPLEX_BANDS as usize;
            let a_re = lhs[idx];
            let a_im = lhs[idx + 1];
            let b_re = rhs[idx];
            let b_im = rhs[idx + 1];

            out[idx] = a_re * b_re - a_im * b_im;
            out[idx + 1] = a_re * b_im + a_im * b_re;
        }

        Ok(())
    }
}

impl<F> DynOperation for FreqMultOp<F>
where
    F: FloatFormat,
    F::Sample: FloatSample
        + Add<Output = F::Sample>
        + Mul<Output = F::Sample>
        + Sub<Output = F::Sample>
        + Copy
        + bytemuck::Pod,
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
                    op: "FreqMultOp",
                    buffer: "region/bands contract",
                    expected: COMPLEX_BANDS as usize,
                    actual: input_bands.max(output_bands) as usize,
                },
            ));
        }

        let _ = Self::expected_sample_len(output_region)?;
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
        format::F32,
        image::{DemandHint, Region},
        op::{DynOperation, NodeSpec},
    };

    fn make_region(pixel_count: usize) -> Region {
        Region::new(0, 0, pixel_count as u32, 1)
    }

    fn run_freqmult(a: &[f32], b: &[f32]) -> Vec<f32> {
        let op = FreqMultOp::<F32>::new();
        let pixel_count = a.len() / COMPLEX_BANDS as usize;
        let region = make_region(pixel_count);
        let inputs: [&[u8]; 2] = [bytemuck::cast_slice(a), bytemuck::cast_slice(b)];
        let input_regions = [region; 2];
        let mut output = vec![0.0f32; a.len()];
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

    prop_compose! {
        fn complex_line()
            (width in 1usize..=16)
            (
                samples in prop::collection::vec(-64.0f32..64.0f32, width * COMPLEX_BANDS as usize),
            ) -> Vec<f32> {
                samples
            }
    }

    proptest! {
        #[test]
        fn multiplying_by_unit_complex_is_identity(samples in complex_line()) {
            let unit = (0..samples.len() / COMPLEX_BANDS as usize)
                .flat_map(|_| [1.0f32, 0.0f32])
                .collect::<Vec<_>>();

            let output = run_freqmult(&samples, &unit);

            for (expected, actual) in samples.iter().zip(output.iter()) {
                prop_assert!((*expected - *actual).abs() <= 1e-6);
            }
        }
    }

    #[test]
    fn multiplying_pure_real_signal_by_itself_squares_real_part() {
        let input = vec![2.0f32, 0.0, -3.0, 0.0, 4.5, 0.0];

        let output = run_freqmult(&input, &input);

        assert_eq!(output, vec![4.0, 0.0, 9.0, -0.0, 20.25, 0.0]);
    }

    #[test]
    fn zero_complex_inputs_stay_zero() {
        let input = vec![0.0f32, 0.0, 0.0, 0.0];

        let output = run_freqmult(&input, &input);

        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn multiplying_by_complex_conjugate_returns_magnitude_squared() {
        let lhs = vec![3.0f32, 4.0, -5.0, 12.0];
        let rhs = vec![3.0f32, -4.0, -5.0, -12.0];

        let output = run_freqmult(&lhs, &rhs);

        assert_eq!(output, vec![25.0, 0.0, 169.0, 0.0]);
    }

    #[test]
    fn freqmult_reports_two_input_complex_contract() {
        let op = FreqMultOp::<F32>::new();
        let region = Region::new(2, 3, 5, 7);

        assert_eq!(op.input_format(), op.output_format());
        assert_eq!(op.bands(), COMPLEX_BANDS);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.input_slot_count(), 2);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region_slot(&region, 1), region);
        assert_eq!(op.node_spec(11, 13), NodeSpec::identity(11, 13));
    }

    #[test]
    fn single_input_dispatch_returns_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let err = op.process_single_input_checked().unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputArity {
                op: "FreqMultOp",
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn missing_secondary_input_returns_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let region = make_region(1);
        let lhs = vec![1.0f32, 2.0];
        let inputs: [&[u8]; 1] = [bytemuck::cast_slice(&lhs)];
        let mut output = vec![7.0f32; 2];
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
                op: "FreqMultOp",
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn invalid_input_bytes_return_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let region = make_region(1);
        let lhs = [0_u8; 1];
        let rhs = [0_u8; 1];
        let inputs = [&lhs[..], &rhs[..]];
        let mut output = vec![5.0f32; 2];
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
                op: "FreqMultOp",
                buffer: "input[0]"
            })
        ));
    }

    #[test]
    fn invalid_rhs_bytes_return_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let region = make_region(1);
        let lhs = bytemuck::cast_slice(&[1.0f32, 2.0f32]).to_vec();
        let rhs = [0_u8; 1];
        let inputs = [lhs.as_slice(), &rhs[..]];
        let mut output = vec![5.0f32; 2];
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
                op: "FreqMultOp",
                buffer: "input[1]"
            })
        ));
    }

    #[test]
    fn mismatched_input_lengths_return_typed_error() {
        let op = FreqMultOp::<F32>::new();
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
                op: "FreqMultOp",
                buffer: "input[0]",
                expected: 2,
                actual: 4
            })
        ));
    }

    #[test]
    fn invalid_output_bytes_return_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let region = make_region(1);
        let lhs = vec![1.0f32, 2.0];
        let rhs = vec![3.0f32, 4.0];
        let inputs = [
            bytemuck::cast_slice(lhs.as_slice()),
            bytemuck::cast_slice(rhs.as_slice()),
        ];
        let mut output = [0_u8; 1];
        let err = op
            .process_region_multi_checked(&inputs, &mut output, region)
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::Freqfilt(FreqfiltError::MultiInputBufferCast {
                op: "FreqMultOp",
                buffer: "output"
            })
        ));
    }

    #[test]
    fn short_output_buffer_returns_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let region = make_region(1);
        let lhs = vec![1.0f32, 2.0];
        let rhs = vec![3.0f32, 4.0];
        let inputs = [
            bytemuck::cast_slice(lhs.as_slice()),
            bytemuck::cast_slice(rhs.as_slice()),
        ];
        let mut output = vec![0.0f32; 1];
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
                op: "FreqMultOp",
                buffer: "output",
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn overflowing_output_region_returns_typed_error() {
        let op = FreqMultOp::<F32>::new();
        let huge = Region::new(0, 0, u32::MAX, u32::MAX);
        let lhs = Vec::<f32>::new();
        let rhs = Vec::<f32>::new();
        let inputs = [
            bytemuck::cast_slice(lhs.as_slice()),
            bytemuck::cast_slice(rhs.as_slice()),
        ];
        let mut output = Vec::<f32>::new();
        let err = op
            .process_region_multi_checked(
                &inputs,
                bytemuck::cast_slice_mut(output.as_mut_slice()),
                huge,
            )
            .unwrap_err();
        assert!(matches!(
            err,
            ViprsError::ImageTooLarge {
                width: u32::MAX,
                height: u32::MAX,
                bands: COMPLEX_BANDS,
                ..
            }
        ));
    }
}
