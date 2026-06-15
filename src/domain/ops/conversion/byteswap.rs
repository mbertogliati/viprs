use std::marker::PhantomData;

use crate::domain::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Per-sample byte-order swap.
pub trait ByteswapSample: Copy {
    #[must_use]
    /// Returns or performs byteswap.
    fn byteswap(self) -> Self;
}

impl ByteswapSample for u8 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        self
    }
}

impl ByteswapSample for u16 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        self.swap_bytes()
    }
}

impl ByteswapSample for i16 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        self.swap_bytes()
    }
}

impl ByteswapSample for u32 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        self.swap_bytes()
    }
}

impl ByteswapSample for i32 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        self.swap_bytes()
    }
}

impl ByteswapSample for f32 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        Self::from_bits(self.to_bits().swap_bytes())
    }
}

impl ByteswapSample for f64 {
    #[inline(always)]
    fn byteswap(self) -> Self {
        Self::from_bits(self.to_bits().swap_bytes())
    }
}

/// Swap byte order for every sample in the tile.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::conversion::byteswap::ByteswapOp;
///
/// let op = ByteswapOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ByteswapOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> ByteswapOp<F>
where
    F::Sample: ByteswapSample,
{
    #[must_use]
    /// Creates a new `ByteswapOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for ByteswapOp<F>
where
    F::Sample: ByteswapSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for ByteswapOp<F>
where
    F: BandFormat,
    F::Sample: ByteswapSample,
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
        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = src.byteswap();
        }
    }
}

impl<F> PixelLocalOp for ByteswapOp<F>
where
    F: BandFormat,
    F::Sample: ByteswapSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, U8, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_byteswap_u16(input_data: &[u16]) -> Vec<u16> {
        let op = ByteswapOp::<U16>::new();
        let region = Region::new(0, 0, input_data.len() as u32, 1);
        let input = Tile::<U16>::new(region, 1, input_data);
        let mut output_data = vec![0u16; input_data.len()];
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        output_data
    }

    #[test]
    fn swaps_known_u16_values() {
        assert_eq!(
            run_byteswap_u16(&[0x1234, 0xabcd, 0x00ff]),
            vec![0x3412, 0xcdab, 0xff00]
        );
    }

    #[test]
    fn u8_is_identity_boundary_values() {
        let op = ByteswapOp::<U8>::new();
        let region = Region::new(0, 0, 2, 1);
        let input_data = [0u8, u8::MAX];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output_data = [0u8; 2];
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn f32_swaps_bits_not_numeric_value() {
        let value = f32::from_bits(0x1234_5678);
        let op = ByteswapOp::<F32>::new();
        let region = Region::new(0, 0, 1, 1);
        let input_data = [value];
        let input = Tile::<F32>::new(region, 1, &input_data);
        let mut output_data = [0.0f32; 1];
        let mut output = TileMut::<F32>::new(region, 1, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data[0].to_bits(), 0x7856_3412);
    }

    #[test]
    fn f64_and_multiband_u32_swap_entire_samples() {
        let value = f64::from_bits(0x0123_4567_89ab_cdef);
        assert_eq!(value.byteswap().to_bits(), 0xefcd_ab89_6745_2301);

        let op = ByteswapOp::<crate::domain::format::U32>::new();
        let region = Region::new(0, 0, 2, 1);
        let input_data = [0x0102_0304u32, 0x1122_3344, 0xaabb_ccdd, 0x0bad_f00d];
        let input = Tile::new(region, 2, &input_data);
        let mut output_data = [0u32; 4];
        let mut output = TileMut::new(region, 2, &mut output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(
            output_data,
            [0x0403_0201, 0x4433_2211, 0xddcc_bbaa, 0x0df0_ad0b]
        );
    }

    #[test]
    fn byteswap_reports_thin_strip_demand_hint() {
        let op = ByteswapOp::<U16>::new();
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
    }

    proptest! {
        #[test]
        fn swapping_twice_is_identity(samples in proptest::collection::vec(any::<u16>(), 1..=128)) {
            let once = run_byteswap_u16(&samples);
            let twice = run_byteswap_u16(&once);
            prop_assert_eq!(twice, samples);
        }

        #[test]
        fn required_region_is_identity(width in 1u32..=16, height in 1u32..=16) {
            let op = ByteswapOp::<U16>::new();
            let region = Region::new(2, 3, width, height);
            prop_assert_eq!(op.required_input_region(&region), region);
        }
    }
}
