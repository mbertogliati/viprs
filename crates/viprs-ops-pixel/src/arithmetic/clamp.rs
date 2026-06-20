use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::{FromF64, ToF64},
};

/// Clamp every sample to the inclusive `[min, max]` range.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::arithmetic::clamp::ClampOp;
///
/// let op = ClampOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ClampOp<F: BandFormat> {
    /// Stores the `min` value for this item.
    pub min: f64,
    /// Stores the `max` value for this item.
    pub max: f64,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> ClampOp<F> {
    #[must_use]
    /// Creates a new `ClampOp`.
    pub fn new(min: f64, max: f64) -> Self {
        debug_assert!(min <= max, "ClampOp: min must be <= max");
        Self {
            min,
            max,
            _phantom: PhantomData,
        }
    }
}

impl<F> Op for ClampOp<F>
where
    F: BandFormat,
    F::Sample: FromF64 + ToF64,
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
        let min = self.min;
        let max = self.max;

        for (sample, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = F::Sample::from_f64(sample.to_f64().clamp(min, max));
        }
    }
}

impl<F> PixelLocalOp for ClampOp<F>
where
    F: BandFormat,
    F::Sample: FromF64 + ToF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, I32},
        image::Region,
    };

    fn make_region(samples: usize) -> Region {
        Region::new(0, 0, samples as u32, 1)
    }

    #[test]
    fn clamps_known_i32_values() {
        let op = ClampOp::<I32>::new(0.0, 255.0);
        let input_data = [-100i32, 0, 100, 200];
        let mut output_data = [0i32; 4];
        let region = make_region(input_data.len());
        let input = Tile::<I32>::new(region, 1, &input_data);
        let mut output = TileMut::<I32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [0, 0, 100, 200]);
    }

    #[test]
    fn boundary_values_are_clamped() {
        let op = ClampOp::<I32>::new(-10.0, 10.0);
        let input_data = [i32::MIN, -10, 0, 10, i32::MAX];
        let mut output_data = [0i32; 5];
        let region = make_region(input_data.len());
        let input = Tile::<I32>::new(region, 1, &input_data);
        let mut output = TileMut::<I32>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, [-10, -10, 0, 10, 10]);
    }

    #[test]
    fn equal_bounds_collapse_every_sample() {
        let op = ClampOp::<I32>::new(7.0, 7.0);
        let input_data = [-3i32, 0, 100];
        let mut output_data = [0i32; 3];
        let region = make_region(input_data.len());
        let input = Tile::<I32>::new(region, 1, &input_data);
        let mut output = TileMut::<I32>::new(region, 1, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input, &mut output);

        assert_eq!(output_data, [7, 7, 7]);
    }

    #[test]
    fn clamp_reports_pixel_local_geometry_contract() {
        let op = ClampOp::<F32>::new(-1.0, 1.0);
        let region = Region::new(2, 3, 4, 5);

        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn identity_within_range(
            pixels in proptest::collection::vec(-1_000.0f32..=1_000.0f32, 1..=64)
        ) {
            let op = ClampOp::<F32>::new(-1_000.0, 1_000.0);
            let mut output_data = vec![0.0f32; pixels.len()];
            let region = make_region(pixels.len());
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }

        #[test]
        fn clamped_values_stay_in_bounds(
            pixels in proptest::collection::vec(-10_000.0f32..=10_000.0f32, 1..=64),
            min in -500.0f64..=500.0f64,
            span in 0.0f64..=500.0f64,
        ) {
            let max = min + span;
            let op = ClampOp::<F32>::new(min, max);
            let mut output_data = vec![0.0f32; pixels.len()];
            let region = make_region(pixels.len());
            let input = Tile::<F32>::new(region, 1, &pixels);
            let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);

            for (src, value) in pixels.iter().zip(output_data.iter()) {
                let expected = f64::from(*src).clamp(min, max) as f32;
                prop_assert_eq!(*value, expected);
            }
        }
    }
}
