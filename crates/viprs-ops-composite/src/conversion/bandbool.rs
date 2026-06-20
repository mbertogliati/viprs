use std::{
    marker::PhantomData,
    ops::{BitAnd, BitOr, BitXor},
};

use viprs_core::{
    format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, OperationBridge, PixelLocalOp},
};

/// Boolean reduction applied across the bands of each pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    /// Uses the `And` variant of `BoolOp`.
    And,
    /// Uses the `Or` variant of `BoolOp`.
    Or,
    /// Uses the `Eor` variant of `BoolOp`.
    Eor,
    /// Uses the `Xor` variant of `BoolOp`.
    Xor,
}

/// Reduce the bands of each pixel with a boolean operation.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::bandbool::BandboolOp;
///
/// let op = BandboolOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct BandboolOp<F: BandFormat> {
    operation: BoolOp,
    input_bands: usize,
    _f: PhantomData<F>,
}

/// Type alias for band bool.
pub type BandBool<F> = BandboolOp<F>;
/// Type alias for band bool op.
pub type BandBoolOp = BoolOp;

impl<F: BandFormat> BandboolOp<F> {
    /// Construct a `BandboolOp` reducer for `input_bands`.
    #[must_use]
    pub fn new(operation: BoolOp, input_bands: usize) -> Self {
        debug_assert!(input_bands > 0, "BandBool: input_bands must be at least 1");
        Self {
            operation,
            input_bands,
            _f: PhantomData,
        }
    }

    #[inline(always)]
    fn reduce_integer<S>(&self, lhs: S, rhs: S) -> S
    where
        S: BandBoolIntSample,
    {
        match self.operation {
            BoolOp::And => lhs & rhs,
            BoolOp::Or => lhs | rhs,
            BoolOp::Eor | BoolOp::Xor => lhs ^ rhs,
        }
    }
}

impl<F> BandboolOp<F>
where
    F: BandBoolFormat,
    F::Sample: bytemuck::Pod,
    <F::Output as BandFormat>::Sample: bytemuck::Pod + BandBoolIntSample,
{
    /// Build an `OperationBridge` configured with the fixed 1-band output.
    #[must_use]
    pub fn into_bridge(self) -> OperationBridge<Self> {
        OperationBridge::new_pixel_local(self, 1)
    }
}

impl<F> Op for BandboolOp<F>
where
    F: BandBoolFormat,
    <F::Output as BandFormat>::Sample: BandBoolIntSample,
{
    type Input = F;
    type Output = F::Output;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<F>, output: &mut TileMut<F::Output>) {
        debug_assert_eq!(
            input.bands as usize, self.input_bands,
            "BandBool input tile band count must match constructor input_bands"
        );
        debug_assert_eq!(
            output.bands, 1,
            "BandBool output tile must have exactly 1 band"
        );

        let pixel_count = input.region.pixel_count();
        for px in 0..pixel_count {
            let src_base = px * self.input_bands;
            let mut acc = F::cast_for_bool(input.data[src_base]);
            for &sample in &input.data[src_base + 1..src_base + self.input_bands] {
                acc = self.reduce_integer(acc, F::cast_for_bool(sample));
            }
            output.data[px] = acc;
        }
    }
}

impl<F> PixelLocalOp for BandboolOp<F>
where
    F: BandBoolFormat,
    <F::Output as BandFormat>::Sample: BandBoolIntSample,
{
}

/// Defines the contract for band bool int sample.
pub trait BandBoolIntSample:
    Copy + 'static + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self>
{
}

impl BandBoolIntSample for u8 {}
impl BandBoolIntSample for u16 {}
impl BandBoolIntSample for i16 {}
impl BandBoolIntSample for u32 {}
impl BandBoolIntSample for i32 {}

/// Defines the contract for band bool format.
pub trait BandBoolFormat: BandFormat {
    /// Associated type for output.
    type Output: BandFormat;

    /// Returns or performs cast for bool.
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample;
}

impl BandBoolFormat for U8 {
    type Output = Self;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample
    }
}

impl BandBoolFormat for U16 {
    type Output = Self;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample
    }
}

impl BandBoolFormat for I16 {
    type Output = Self;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample
    }
}

impl BandBoolFormat for U32 {
    type Output = Self;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample
    }
}

impl BandBoolFormat for I32 {
    type Output = Self;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample
    }
}

impl BandBoolFormat for F32 {
    type Output = I32;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample as i32
    }
}

impl BandBoolFormat for F64 {
    type Output = I32;

    #[inline(always)]
    fn cast_for_bool(sample: Self::Sample) -> <Self::Output as BandFormat>::Sample {
        sample as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{F32, F64, I16, I32, U8, U16, U32},
        image::{Region, Tile, TileMut},
        op::DynOperation,
    };

    fn make_region(w: u32, h: u32) -> Region {
        Region::new(0, 0, w, h)
    }

    fn run_bandbool_u8(
        operation: BoolOp,
        input_bands: usize,
        input_data: &[u8],
        output_data: &mut [u8],
        pixels: usize,
    ) {
        let region = make_region(pixels as u32, 1);
        let op = BandboolOp::<U8>::new(operation, input_bands);
        let input = Tile::<U8>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<U8>::new(region, 1, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    fn run_bandbool_f32(
        operation: BoolOp,
        input_bands: usize,
        input_data: &[f32],
        output_data: &mut [i32],
        pixels: usize,
    ) {
        let region = make_region(pixels as u32, 1);
        let op = BandboolOp::<F32>::new(operation, input_bands);
        let input = Tile::<F32>::new(region, input_bands as u32, input_data);
        let mut output = TileMut::<I32>::new(region, 1, output_data);
        let mut state = op.start();
        op.process_region(&mut state, &input, &mut output);
    }

    #[test]
    fn into_bridge_reports_one_band() {
        let bridge = BandboolOp::<U8>::new(BoolOp::And, 4).into_bridge();
        assert_eq!(bridge.bands(), 1);
    }

    #[test]
    fn single_band_is_identity() {
        let input = [0u8, 127, 255];
        let mut output = [0u8; 3];
        run_bandbool_u8(BoolOp::Eor, 1, &input, &mut output, 3);
        assert_eq!(output, input);
    }

    #[test]
    fn float_input_casts_to_i32_before_boolean() {
        let input = [1.9f32, 2.1, 4.0];
        let mut output = [0i32; 1];
        run_bandbool_f32(BoolOp::Or, 3, &input, &mut output, 1);
        assert_eq!(output, [7]);
    }

    #[test]
    fn and_on_duplicated_pixels_is_identity() {
        let input = [5u8, 5, 9, 9, 255, 255];
        let mut output = [0u8; 3];
        run_bandbool_u8(BoolOp::And, 2, &input, &mut output, 3);
        assert_eq!(output, [5, 9, 255]);
    }

    #[test]
    fn or_of_zeros_and_ones_is_ones() {
        let input = [0u8, 1, 0, 1, 0, 1];
        let mut output = [0u8; 3];
        run_bandbool_u8(BoolOp::Or, 2, &input, &mut output, 3);
        assert_eq!(output, [1, 1, 1]);
    }

    #[test]
    fn bandbool_format_dispatch_covers_all_supported_input_types() {
        assert_eq!(<U8 as BandBoolFormat>::cast_for_bool(5), 5);
        assert_eq!(<U16 as BandBoolFormat>::cast_for_bool(6), 6);
        assert_eq!(<I16 as BandBoolFormat>::cast_for_bool(-7), -7);
        assert_eq!(<U32 as BandBoolFormat>::cast_for_bool(8), 8);
        assert_eq!(<I32 as BandBoolFormat>::cast_for_bool(-9), -9);
        assert_eq!(<F32 as BandBoolFormat>::cast_for_bool(3.9), 3);
        assert_eq!(<F64 as BandBoolFormat>::cast_for_bool(4.9), 4);
    }

    #[test]
    fn bandbool_metadata_matches_identity_geometry() {
        let op = BandboolOp::<U8>::new(BoolOp::And, 3);
        let region = make_region(4, 2);
        assert_eq!(op.demand_hint(), viprs_core::image::DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn and_all_ones_stays_all_ones(
            pixels in 1usize..=64,
            bands in 1usize..=8,
        ) {
            let input = vec![0xFFu8; pixels * bands];
            let mut output = vec![0u8; pixels];
            run_bandbool_u8(BoolOp::And, bands, &input, &mut output, pixels);
            prop_assert!(output.iter().all(|&sample| sample == 0xFF));
        }
    }
}
