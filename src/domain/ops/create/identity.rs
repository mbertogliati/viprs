use std::marker::PhantomData;

use crate::domain::{
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

/// Generate a one-band identity LUT image.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::create::identity::IdentityOp;
///
/// let op = IdentityOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
#[derive(Debug)]
pub struct IdentityOp<F: BandFormat> {
    ushort: bool,
    _format: PhantomData<F>,
}

impl<F: BandFormat> Copy for IdentityOp<F> {}

impl<F: BandFormat> Clone for IdentityOp<F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F: BandFormat> IdentityOp<F> {
    #[must_use]
    /// Creates a new `IdentityOp`.
    pub const fn new(ushort: bool) -> Self {
        Self {
            ushort,
            _format: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs ushort.
    pub const fn ushort(&self) -> bool {
        self.ushort
    }

    #[must_use]
    /// Returns or performs width.
    pub const fn width(&self) -> u32 {
        if self.ushort { 65_536 } else { 256 }
    }

    #[must_use]
    /// Returns or performs height.
    pub const fn height(&self) -> u32 {
        1
    }
}

impl<F> Op for IdentityOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
    type Input = F;
    type Output = F;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), _input: &Tile<F>, output: &mut TileMut<F>) {
        debug_assert_eq!(output.bands, 1, "IdentityOp output must be single-band");
        debug_assert!(output.region.x >= 0 && output.region.y >= 0);
        debug_assert!(output.region.x as u32 + output.region.width <= self.width());
        debug_assert!(output.region.y as u32 + output.region.height <= self.height());

        let region_width = output.region.width as usize;
        for row in 0..output.region.height as usize {
            for col in 0..region_width {
                let x = output.region.x as u32 + col as u32;
                output.data[row * region_width + col] = F::Sample::from_f64(f64::from(x));
            }
        }
    }
}

impl<F> PixelLocalOp for IdentityOp<F>
where
    F: BandFormat,
    F::Sample: FromF64,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{U8, U16},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn render_u8(op: IdentityOp<U8>) -> Vec<u8> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0u8; region.pixel_count()];
        let mut output_data = vec![0u8; region.pixel_count()];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn render_u16(op: IdentityOp<U16>) -> Vec<u16> {
        let region = Region::new(0, 0, op.width(), op.height());
        let input_data = vec![0u16; region.pixel_count()];
        let mut output_data = vec![0u16; region.pixel_count()];
        let input = Tile::<U16>::new(region, 1, &input_data);
        let mut output = TileMut::<U16>::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    #[test]
    fn dimensions_match_selected_mode() {
        let byte = IdentityOp::<U8>::new(false);
        let ushort = IdentityOp::<U16>::new(true);

        assert_eq!(byte.width(), 256);
        assert_eq!(byte.height(), 1);
        assert_eq!(ushort.width(), 65_536);
        assert_eq!(ushort.height(), 1);
    }

    #[test]
    fn byte_mode_emits_byte_indices() {
        let samples = render_u8(IdentityOp::<U8>::new(false));

        assert_eq!(samples.len(), 256);
        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], 1);
        assert_eq!(samples[255], 255);
    }

    #[test]
    fn ushort_mode_emits_full_16bit_range() {
        let samples = render_u16(IdentityOp::<U16>::new(true));

        assert_eq!(samples.len(), 65_536);
        assert_eq!(samples[0], 0);
        assert_eq!(samples[1], 1);
        assert_eq!(samples[65_535], 65_535);
    }

    #[test]
    fn ushort_accessor_matches_mode() {
        assert!(!IdentityOp::<U8>::new(false).ushort());
        assert!(IdentityOp::<U16>::new(true).ushort());
    }

    #[test]
    fn partial_region_uses_absolute_x_offsets() {
        let op = IdentityOp::<U16>::new(true);
        let input_region = Region::new(0, 0, op.width(), op.height());
        let output_region = Region::new(10, 0, 4, 1);
        let input_data = vec![0u16; input_region.pixel_count()];
        let mut output_data = vec![0u16; output_region.pixel_count()];
        let input = Tile::<U16>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U16>::new(output_region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, vec![10u16, 11, 12, 13]);
    }

    #[test]
    fn identity_reports_any_demand_and_identity_region_contract() {
        let op = IdentityOp::<U8>::new(false);
        let region = Region::new(8, 0, 4, 1);

        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    #[test]
    fn byte_mode_partial_region_emits_requested_indices() {
        let op = IdentityOp::<U8>::new(false);
        let input_region = Region::new(0, 0, op.width(), op.height());
        let output_region = Region::new(250, 0, 6, 1);
        let input_data = vec![0u8; input_region.pixel_count()];
        let mut output_data = vec![0u8; output_region.pixel_count()];
        let input = Tile::<U8>::new(input_region, 1, &input_data);
        let mut output = TileMut::<U8>::new(output_region, 1, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, vec![250u8, 251, 252, 253, 254, 255]);
    }

    proptest! {
        #[test]
        fn prop_output_has_expected_dimensions_and_range(ushort in any::<bool>()) {
            if ushort {
                let op = IdentityOp::<U16>::new(true);
                let samples = render_u16(op);
                prop_assert_eq!(samples.len(), op.width() as usize * op.height() as usize);
                prop_assert_eq!(samples.first().copied(), Some(0));
                prop_assert_eq!(samples.last().copied(), Some(65_535));
                prop_assert!(samples.iter().enumerate().all(|(index, sample)| *sample == index as u16));
            } else {
                let op = IdentityOp::<U8>::new(false);
                let samples = render_u8(op);
                prop_assert_eq!(samples.len(), op.width() as usize * op.height() as usize);
                prop_assert_eq!(samples.first().copied(), Some(0));
                prop_assert_eq!(samples.last().copied(), Some(255));
                prop_assert!(samples.iter().enumerate().all(|(index, sample)| *sample == index as u8));
            }
        }
    }
}
