use std::marker::PhantomData;

use viprs_core::{
    error::ViprsError,
    format::{BandFormat, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Apply a precomputed histogram-equalization LUT.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::histogram::hist_equal::HistEqualOp;
///
/// let op = HistEqualOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct HistEqualOp<F: BandFormat> {
    lut: [u8; 256],
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> HistEqualOp<F> {
    /// Creates this value from lut.
    pub fn from_lut<L>(lut: L) -> Result<Self, ViprsError>
    where
        L: TryInto<[u8; 256]>,
    {
        let lut: [u8; 256] = lut
            .try_into()
            .map_err(|_| ViprsError::Scheduler("HistEqualOp requires a 256-entry LUT".into()))?;

        Ok(Self {
            lut,
            _phantom: PhantomData,
        })
    }

    #[must_use]
    /// Returns or performs lut.
    pub const fn lut(&self) -> &[u8; 256] {
        &self.lut
    }
}

impl Op for HistEqualOp<U8> {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn preferred_tile_geometry(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, _state: &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        for (src, dst) in input.data.iter().zip(output.data.iter_mut()) {
            *dst = self.lut[*src as usize];
        }
    }
}

impl PixelLocalOp for HistEqualOp<U8> {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::image::Region;

    fn identity_lut() -> [u8; 256] {
        std::array::from_fn(|idx| idx as u8)
    }

    #[test]
    fn hist_equal_identity_lut_is_no_op() {
        let op = HistEqualOp::<U8>::from_lut(identity_lut()).unwrap();
        let region = Region::new(0, 0, 4, 1);
        let input_data = vec![0u8, 64, 128, 255];
        let mut output_data = vec![0u8; 4];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, input_data);
    }

    #[test]
    fn hist_equal_boundary_values_follow_lut() {
        let mut lut = [0u8; 256];
        lut[0] = 17;
        lut[255] = 231;
        let op = HistEqualOp::<U8>::from_lut(lut).unwrap();
        let region = Region::new(0, 0, 2, 1);
        let input_data = vec![0u8, 255u8];
        let mut output_data = vec![0u8; 2];
        let input = Tile::<U8>::new(region, 1, &input_data);
        let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input, &mut output);
        assert_eq!(output_data, vec![17, 231]);
    }

    #[test]
    fn hist_equal_rejects_wrong_lut_length() {
        match HistEqualOp::<U8>::from_lut(vec![0u8; 255]) {
            Err(ViprsError::Scheduler(message)) => assert!(message.contains("256-entry LUT")),
            _ => panic!("expected 256-entry LUT validation error"),
        }
    }

    #[test]
    fn hist_equal_lut_accessor_returns_fixed_table() {
        let mut lut = identity_lut();
        lut[42] = 7;
        let op = HistEqualOp::<U8>::from_lut(lut).unwrap();
        assert_eq!(op.lut()[42], 7);
    }

    #[test]
    fn hist_equal_region_contract_is_pixel_local() {
        let op = HistEqualOp::<U8>::from_lut(identity_lut()).unwrap();
        let region = Region::new(-3, 5, 11, 7);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }

    proptest! {
        #[test]
        fn hist_equal_identity_prop(pixels in proptest::collection::vec(0u8..=255u8, 1..=128)) {
            let op = HistEqualOp::<U8>::from_lut(identity_lut()).unwrap();
            let region = Region::new(0, 0, pixels.len() as u32, 1);
            let input = Tile::<U8>::new(region, 1, &pixels);
            let mut output_data = vec![0u8; pixels.len()];
            let mut output = TileMut::<U8>::new(region, 1, &mut output_data);
            let mut state = ();
            op.process_region(&mut state, &input, &mut output);
            prop_assert_eq!(output_data, pixels);
        }
    }
}
