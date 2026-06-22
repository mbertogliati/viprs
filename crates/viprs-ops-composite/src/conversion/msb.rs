use std::marker::PhantomData;

use viprs_core::{
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Extracts the most-significant byte from every multibyte integer sample.
pub trait MsbSample: Copy {
    /// Returns or performs msb byte.
    fn msb_byte(self) -> u8;
}

impl MsbSample for u8 {
    #[inline(always)]
    fn msb_byte(self) -> u8 {
        self
    }
}

impl MsbSample for u16 {
    #[inline(always)]
    fn msb_byte(self) -> u8 {
        let bytes = self.to_ne_bytes();
        bytes[msb_index(bytes.len())]
    }
}

impl MsbSample for i16 {
    #[inline(always)]
    fn msb_byte(self) -> u8 {
        let bytes = self.to_ne_bytes();
        bytes[msb_index(bytes.len())] ^ 0x80
    }
}

impl MsbSample for u32 {
    #[inline(always)]
    fn msb_byte(self) -> u8 {
        let bytes = self.to_ne_bytes();
        bytes[msb_index(bytes.len())]
    }
}

impl MsbSample for i32 {
    #[inline(always)]
    fn msb_byte(self) -> u8 {
        let bytes = self.to_ne_bytes();
        bytes[msb_index(bytes.len())] ^ 0x80
    }
}

#[inline(always)]
const fn msb_index(byte_len: usize) -> usize {
    if cfg!(target_endian = "big") {
        0
    } else {
        byte_len - 1
    }
}

/// Libvips-style `msb`: convert integer samples to a single U8 byte per band.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::msb::MsbOp;
///
/// let op = MsbOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct MsbOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> MsbOp<F>
where
    F::Sample: MsbSample,
{
    #[must_use]
    /// Creates a new `MsbOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for MsbOp<F>
where
    F::Sample: MsbSample,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> Op for MsbOp<F>
where
    F: BandFormat,
    F::Sample: MsbSample,
{
    type Input = F;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<U8>) {
        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = src.msb_byte();
        }
    }
}

impl<F> PixelLocalOp for MsbOp<F>
where
    F: BandFormat,
    F::Sample: MsbSample,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{I16, U16, U32},
        image::{Region, Tile, TileMut},
        op::{DynOperation, OperationBridge},
    };

    fn run_msb<F: BandFormat>(input_data: &[F::Sample], bands: u32) -> Vec<u8>
    where
        F::Sample: MsbSample,
    {
        let op = MsbOp::<F>::new();
        let region = Region::new(0, 0, (input_data.len() / bands as usize) as u32, 1);
        let input = Tile::<F>::new(region, bands, input_data);
        let mut output_data = vec![0u8; input_data.len()];
        let mut output = TileMut::<U8>::new(region, bands, &mut output_data);
        op.start();
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn u16_extracts_high_byte() {
        assert_eq!(
            run_msb::<U16>(&[0x1234, 0xabcd, 0x00ff, 0xff00], 1),
            vec![0x12, 0xab, 0x00, 0xff]
        );
    }

    #[test]
    fn signed_formats_flip_the_sign_bit_like_libvips() {
        assert_eq!(
            run_msb::<I16>(&[i16::MIN, -1, 0, i16::MAX], 1),
            vec![0x00, 0x7f, 0x80, 0xff]
        );
    }

    #[test]
    fn bridge_reports_u8_output() {
        let bridge = OperationBridge::new_pixel_local(MsbOp::<U32>::new(), 3);
        assert_eq!(bridge.input_format(), viprs_core::format::BandFormatId::U32);
        assert_eq!(bridge.output_format(), viprs_core::format::BandFormatId::U8);
        assert_eq!(bridge.bands(), 3);
    }

    #[test]
    fn u8_is_identity_and_i32_flips_sign_bit() {
        assert_eq!(
            run_msb::<viprs_core::format::U8>(&[0, 127, 255], 1),
            vec![0, 127, 255]
        );
        assert_eq!(
            run_msb::<viprs_core::format::I32>(&[i32::MIN, -1, 0, i32::MAX], 1),
            vec![0x00, 0x7f, 0x80, 0xff]
        );
    }

    #[test]
    fn msb_reports_thin_strip_demand_hint() {
        let op = MsbOp::<U16>::new();
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        op.start();
    }

    proptest! {
        #[test]
        fn u32_output_stays_within_u8_range(samples in proptest::collection::vec(any::<u32>(), 1..=128)) {
            let output = run_msb::<U32>(&samples, 1);
            prop_assert_eq!(output.len(), samples.len());
        }

        #[test]
        fn required_region_is_identity(width in 1u32..=32, height in 1u32..=8) {
            let op = MsbOp::<U16>::new();
            let region = Region::new(3, -2, width, height);
            prop_assert_eq!(op.required_input_region(&region), region);
        }
    }
}
