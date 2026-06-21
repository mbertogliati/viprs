use std::marker::PhantomData;

use viprs_core::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::FromF64,
};

/// Fill the output tile with zeros.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::create::black::BlackOp;
///
/// let op = BlackOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct BlackOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> BlackOp<F> {
    #[must_use]
    /// Creates a new `BlackOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F> Op for BlackOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
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

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        output.data.fill(F::Sample::from_f64(0.0));
    }
}

impl<F> PixelLocalOp for BlackOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::U8,
        image::{Region, Tile, TileMut},
    };

    #[test]
    fn black_op_clears_every_band() {
        let op = BlackOp::<U8>::new();
        let region = Region::new(0, 0, 2, 2);
        let input_data = [7u8; 12];
        let mut output_data = [255u8; 12];
        let input = Tile::<U8>::new(region, 3, &input_data);
        let mut output = TileMut::<U8>::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0u8; 12]);
    }

    #[test]
    fn black_op_handles_single_pixel_tiles() {
        let op = BlackOp::<U8>::new();
        let region = Region::new(0, 0, 1, 1);
        let input_data = [99u8; 4];
        let mut output_data = [1u8, 2, 3, 4];
        let input = Tile::<U8>::new(region, 4, &input_data);
        let mut output = TileMut::<U8>::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0u8; 4]);
    }

    #[test]
    fn black_op_metadata_is_identity() {
        let op = BlackOp::<U8>::new();
        let region = Region::new(3, 4, 5, 6);

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn prop_black_op_always_outputs_zero(
            width in 1u32..=16,
            height in 1u32..=16,
            bands in 1u32..=4,
            seed in proptest::collection::vec(any::<u8>(), 1..=1024),
        ) {
            let len = width as usize * height as usize * bands as usize;
            let mut input_data = vec![0u8; len];
            for (dst, src) in input_data.iter_mut().zip(seed.iter().cycle()) {
                *dst = *src;
            }
            let mut output_data = vec![255u8; len];
            let region = Region::new(0, 0, width, height);
            let input = Tile::<U8>::new(region, bands, &input_data);
            let mut output = TileMut::<U8>::new(region, bands, &mut output_data);

            BlackOp::<U8>::new().process_region(&mut (), &input, &mut output);

            prop_assert!(output_data.iter().all(|sample| *sample == 0));
        }
    }
}
