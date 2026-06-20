use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    format::{BandFormat, BandFormatId, U8, U16, U32},
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Defines the contract for hist norm default output.
pub trait HistNormDefaultOutput: BandFormat {
    /// Associated type for output.
    type Output: BandFormat;
}

impl HistNormDefaultOutput for U8 {
    type Output = Self;
}

impl HistNormDefaultOutput for U16 {
    type Output = Self;
}

impl HistNormDefaultOutput for U32 {
    type Output = Self;
}

/// Normalize a histogram image band-by-band.
///
/// libvips scales each band so its maximum value becomes `pixel_count - 1`
/// before casting to the narrowest unsigned integer format that can hold the
/// result. `HistNormOp<F>` keeps the default libvips-compatible output for
/// canonical histogram domains (`U8 → U8`, `U16 → U16`, `U32 → U32`), while
/// `HistNormTypedOp<F, O>` lets callers preserve parity after an earlier type
/// promotion such as `hist_cum` (`U32 → U8` for a 256-bin cumulative histogram).
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::histogram::hist_norm::HistNormTypedOp;
///
/// let op = HistNormTypedOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistNormTypedOp<F: BandFormat, O: BandFormat> {
    _phantom: PhantomData<F>,
    _output: PhantomData<O>,
}

/// Type alias for hist norm op.
pub type HistNormOp<F> = HistNormTypedOp<F, <F as HistNormDefaultOutput>::Output>;

#[must_use]
/// Returns or performs hist norm promoted format.
pub const fn hist_norm_promoted_format(pixel_count: u32) -> BandFormatId {
    let new_max = pixel_count.saturating_sub(1);
    if new_max <= u8::MAX as u32 {
        BandFormatId::U8
    } else if new_max <= u16::MAX as u32 {
        BandFormatId::U16
    } else {
        BandFormatId::U32
    }
}

impl<F: BandFormat, O: BandFormat> HistNormTypedOp<F, O> {
    #[must_use]
    /// Creates a new `HistNormTypedOp`.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
            _output: PhantomData,
        }
    }
}

impl<F: BandFormat, O: BandFormat> Default for HistNormTypedOp<F, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, O> Op for HistNormTypedOp<F, O>
where
    F: BandFormat,
    O: BandFormat,
    F::Sample: ToF64 + FromF64 + Pod,
    O::Sample: FromF64 + Pod,
{
    type Input = F;
    type Output = O;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<O>) {
        let bands = input.bands as usize;
        let pixel_count = input.region.pixel_count();
        let scale_max = pixel_count.saturating_sub(1) as f64;

        for band in 0..bands {
            let mut band_max = 0.0f64;
            for idx in (band..input.data.len()).step_by(bands) {
                band_max = band_max.max(input.data[idx].to_f64());
            }

            for idx in (band..input.data.len()).step_by(bands) {
                let value = if band_max > 0.0 {
                    (input.data[idx].to_f64() / band_max) * scale_max
                } else {
                    0.0
                };
                output.data[idx] = O::Sample::from_f64(value);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormatId, F32, U8, U16, U32},
        image::Region,
        op::{DynOperation, OperationBridge},
    };

    fn run_op<F, O>(input_data: &[F::Sample]) -> Vec<O::Sample>
    where
        F: BandFormat,
        O: BandFormat,
        F::Sample: ToF64 + FromF64 + Pod + Copy + Default,
        O::Sample: FromF64 + Pod + Copy + Default,
    {
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<F>::new(region, 1, input_data);
        let mut output_data = vec![O::Sample::default(); input_data.len()];
        let mut output = TileMut::<O>::new(region, 1, &mut output_data);
        let mut state = ();
        HistNormTypedOp::<F, O>::new().process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn hist_norm_all_zero_input_stays_zero() {
        let output = run_op::<F32, F32>(&[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn hist_norm_float_output_scales_to_pixel_count_minus_one() {
        let output = run_op::<F32, F32>(&[7.0, 7.0, 7.0, 7.0]);
        assert_eq!(output, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn hist_norm_u8_histograms_default_to_u8_output() {
        let bridge = OperationBridge::new(HistNormOp::<U8>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::U8);
        assert_eq!(bridge.output_format(), BandFormatId::U8);

        let output = run_op::<U8, U8>(&[0, 4, 2, 4]);
        assert_eq!(output, vec![0, 3, 2, 3]);
    }

    #[test]
    fn hist_norm_u16_histograms_default_to_u16_output() {
        let bridge = OperationBridge::new(HistNormOp::<U16>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::U16);
        assert_eq!(bridge.output_format(), BandFormatId::U16);
    }

    #[test]
    fn hist_norm_supports_explicit_u32_to_u8_cast_for_cumulative_u8_histograms() {
        let bridge = OperationBridge::new(HistNormTypedOp::<U32, U8>::new(), 1);
        assert_eq!(bridge.input_format(), BandFormatId::U32);
        assert_eq!(bridge.output_format(), BandFormatId::U8);

        let output = run_op::<U32, U8>(&[0, 4, 2, 4]);
        assert_eq!(output, vec![0, 3, 2, 3]);
    }

    #[test]
    fn hist_norm_promoted_format_matches_libvips_thresholds() {
        assert_eq!(hist_norm_promoted_format(256), BandFormatId::U8);
        assert_eq!(hist_norm_promoted_format(65_536), BandFormatId::U16);
        assert_eq!(hist_norm_promoted_format(65_537), BandFormatId::U32);
    }

    #[test]
    fn hist_norm_accumulates_band_maxima_independently() {
        let region = Region::new(0, 0, 3, 1);
        let input_data = vec![0u8, 2, 5, 4, 10, 8];
        let input = Tile::<U8>::new(region, 2, &input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, 2, &mut output_data);
        let mut state = ();

        HistNormOp::<U8>::default().process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, vec![0, 1, 1, 1, 2, 2]);
    }

    #[test]
    fn hist_norm_region_contract_is_identity() {
        let op = HistNormTypedOp::<F32, F32>::new();
        let region = Region::new(4, -6, 9, 3);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_norm_uniform_histogram_is_flat(value in 1u16..=1024u16, len in 1usize..=64usize) {
            let input = vec![f32::from(value); len];
            let output = run_op::<F32, F32>(&input);
            for value in output {
                prop_assert!((value - (len.saturating_sub(1) as f32)).abs() < f32::EPSILON);
            }
        }
    }
}
