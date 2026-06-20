use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Applies a per-tile gain before mosaic blending.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::mosaicing::remosaic::RemosaicOp;
///
/// let op = RemosaicOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct RemosaicOp<F: BandFormat> {
    gain: f64,
    _format: PhantomData<F>,
}

impl<F: BandFormat> RemosaicOp<F> {
    /// Creates a new `RemosaicOp`.
    pub fn new(gain: f64) -> Result<Self, ViprsError> {
        if !gain.is_finite() {
            return Err(ViprsError::Scheduler(
                "RemosaicOp gain must be finite".into(),
            ));
        }
        Ok(Self {
            gain,
            _format: PhantomData,
        })
    }

    /// Creates this value from gains.
    pub fn from_gains(gains: &[f64], tile_index: usize) -> Result<Self, ViprsError> {
        let Some(&gain) = gains.get(tile_index) else {
            return Err(ViprsError::Scheduler(format!(
                "RemosaicOp tile index {tile_index} is out of range for {} gains",
                gains.len()
            )));
        };
        Self::new(gain)
    }

    #[must_use]
    /// Returns or performs gain.
    pub const fn gain(&self) -> f64 {
        self.gain
    }
}

impl<F> Op for RemosaicOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        if (self.gain - 1.0).abs() <= f64::EPSILON {
            output.data.copy_from_slice(input.data);
            return;
        }

        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = F::Sample::from_f64(src.to_f64() * self.gain);
        }
    }
}

impl<F> PixelLocalOp for RemosaicOp<F>
where
    F: BandFormat,
    F::Sample: ToF64 + FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, U8},
        image::Region,
    };

    #[test]
    fn unit_gain_preserves_input() {
        let op = RemosaicOp::<U8>::new(1.0).unwrap();
        let region = Region::new(0, 0, 3, 2);
        let input_data = vec![1u8, 2, 3, 4, 5, 6];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, input_data);
    }

    #[test]
    fn clips_scaled_integer_values() {
        let op = RemosaicOp::<U8>::new(1.5).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let input_data = vec![100u8, 250u8];
        let mut output_data = vec![0u8; input_data.len()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, vec![150u8, 255u8]);
    }

    #[test]
    fn builder_and_contract_cover_error_paths() {
        let op = RemosaicOp::<U8>::from_gains(&[0.5, 1.5], 1).unwrap();
        let region = Region::new(2, 3, 4, 5);

        assert_eq!(op.gain(), 1.5);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert!(RemosaicOp::<U8>::new(f64::NAN).is_err());
        assert!(RemosaicOp::<U8>::from_gains(&[1.0], 4).is_err());
    }

    proptest! {
        #[test]
        fn proptest_identity_for_unit_gain(values in prop::collection::vec(any::<u8>(), 1..128)) {
            let region = Region::new(0, 0, values.len() as u32, 1);
            let op = RemosaicOp::<U8>::new(1.0).unwrap();
            let input = Tile::<U8>::new(region, 1, &values);
            let mut output_data = vec![0u8; values.len()];
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);

            op.process_region(&mut (), &input, &mut output);

            prop_assert_eq!(output_data, values);
        }

        #[test]
        fn reciprocal_gains_round_trip_float_tiles(
            values in prop::collection::vec(-1000.0f32..1000.0f32, 1..128),
            gain in 0.1f64..4.0,
        ) {
            let region = Region::new(0, 0, values.len() as u32, 1);
            let forward = RemosaicOp::<F32>::new(gain).unwrap();
            let inverse = RemosaicOp::<F32>::new(1.0 / gain).unwrap();
            let input = Tile::<F32>::new(region, 1, &values);

            let mut balanced_data = vec![0.0f32; values.len()];
            let mut restored_data = vec![0.0f32; values.len()];
            let mut balanced = TileMut::<F32>::new(region, 1, &mut balanced_data);
            forward.process_region(&mut (), &input, &mut balanced);

            let balanced_tile = Tile::<F32>::new(region, 1, &balanced_data);
            let mut restored = TileMut::<F32>::new(region, 1, &mut restored_data);
            inverse.process_region(&mut (), &balanced_tile, &mut restored);

            for (original, restored) in values.iter().zip(restored_data.iter()) {
                prop_assert!((f64::from(*original) - f64::from(*restored)).abs() <= 1e-4);
            }
        }
    }
}
