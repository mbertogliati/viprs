use std::{any::Any, marker::PhantomData};

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Per-pixel image selection over a packed input tile.
///
/// Input layout per pixel is:
/// `[index, image0_0, .. image0_n, image1_0, .. image{N-1}_n]`
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::switch::SwitchOp;
///
/// let op = SwitchOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SwitchOp<F: BandFormat> {
    choice_count: u32,
    candidate_bands: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> SwitchOp<F> {
    #[must_use]
    /// Creates a new `SwitchOp`.
    pub fn new(choice_count: u32, candidate_bands: u32) -> Self {
        debug_assert!(choice_count > 0, "SwitchOp: choice_count must be >= 1");
        debug_assert!(
            candidate_bands > 0,
            "SwitchOp: candidate_bands must be >= 1"
        );
        Self {
            choice_count,
            candidate_bands,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs combined bands.
    pub const fn combined_bands(&self) -> u32 {
        1 + self.choice_count * self.candidate_bands
    }
}

impl<F> DynOperation for SwitchOp<F>
where
    F: BandFormat,
    F::Sample: Pod + SwitchIndex,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.candidate_bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
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

    #[inline]
    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        output_region: Region,
    ) {
        let pixel_count = output_region.pixel_count();
        self.process_typed::<F::Sample>(input, output, pixel_count);
    }
}

impl<F> SwitchOp<F>
where
    F: BandFormat,
    F::Sample: Pod + SwitchIndex,
{
    fn process_typed<T>(&self, input: &[u8], output: &mut [u8], pixel_count: usize)
    where
        T: Pod + Copy + SwitchIndex,
    {
        let src = cast_slice::<u8, T>(input);
        let dst = cast_slice_mut::<u8, T>(output);
        let candidate_bands = self.candidate_bands as usize;
        let combined_bands = self.combined_bands() as usize;
        let choice_count = self.choice_count as usize;

        debug_assert_eq!(src.len(), pixel_count * combined_bands);
        debug_assert_eq!(dst.len(), pixel_count * candidate_bands);

        for px in 0..pixel_count {
            let src_base = px * combined_bands;
            let dst_base = px * candidate_bands;
            let choice = src[src_base].to_index() % choice_count;
            let candidate_base = src_base + 1 + choice * candidate_bands;
            dst[dst_base..dst_base + candidate_bands]
                .copy_from_slice(&src[candidate_base..candidate_base + candidate_bands]);
        }
    }
}

/// Defines the contract for switch index.
pub trait SwitchIndex {
    /// Converts this value to index.
    fn to_index(self) -> usize;
}

impl SwitchIndex for u8 {
    fn to_index(self) -> usize {
        self as usize
    }
}

impl SwitchIndex for u16 {
    fn to_index(self) -> usize {
        self as usize
    }
}

impl SwitchIndex for i16 {
    fn to_index(self) -> usize {
        self.max(0) as usize
    }
}

impl SwitchIndex for u32 {
    fn to_index(self) -> usize {
        self as usize
    }
}

impl SwitchIndex for i32 {
    fn to_index(self) -> usize {
        self.max(0) as usize
    }
}

impl SwitchIndex for f32 {
    fn to_index(self) -> usize {
        self.max(0.0) as usize
    }
}

impl SwitchIndex for f64 {
    fn to_index(self) -> usize {
        self.max(0.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{format::U8, image::Region, op::DynOperation};

    fn run_switch(
        input: &[u8],
        output: &mut [u8],
        choice_count: u32,
        candidate_bands: u32,
        pixel_count: usize,
    ) {
        let op = SwitchOp::<U8>::new(choice_count, candidate_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region(state.as_mut(), input, output, region, region);
    }

    #[test]
    fn index_zero_selects_first_candidate() {
        let input = [
            0u8, 10, 11, 20, 21, //
            0, 30, 31, 40, 41,
        ];
        let mut output = [0u8; 4];
        run_switch(&input, &mut output, 2, 2, 2);
        assert_eq!(output, [10, 11, 30, 31]);
    }

    #[test]
    fn index_one_selects_second_candidate() {
        let input = [
            1u8, 10, 11, 20, 21, //
            1, 30, 31, 40, 41,
        ];
        let mut output = [0u8; 4];
        run_switch(&input, &mut output, 2, 2, 2);
        assert_eq!(output, [20, 21, 40, 41]);
    }

    #[test]
    fn metadata_matches_packed_layout() {
        let op = SwitchOp::<U8>::new(3, 2);
        let region = Region::new(4, 5, 6, 7);
        let spec = op.node_spec(32, 16);

        assert_eq!(op.input_format(), BandFormatId::U8);
        assert_eq!(op.output_format(), BandFormatId::U8);
        assert_eq!(op.bands(), 2);
        assert_eq!(op.combined_bands(), 7);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(spec, NodeSpec::identity(32, 16));
        let _state = op.dyn_start();
    }

    #[test]
    fn switch_index_conversions_cover_supported_types() {
        assert_eq!(u8::to_index(7), 7);
        assert_eq!(u16::to_index(8), 8);
        assert_eq!(i16::to_index(-3), 0);
        assert_eq!(u32::to_index(9), 9);
        assert_eq!(i32::to_index(-4), 0);
        assert_eq!(f32::to_index(5.9), 5);
        assert_eq!(f64::to_index(-1.0), 0);
    }

    proptest! {
        #[test]
        fn index_wraps_modulo_choice_count(index in 0u8..=255, first in any::<u8>(), second in any::<u8>(), third in any::<u8>()) {
            let input = [index, first, second, third];
            let mut output = [0u8; 1];
            run_switch(&input, &mut output, 3, 1, 1);
            let expected = match index as usize % 3 {
                0 => first,
                1 => second,
                _ => third,
            };
            prop_assert_eq!(output[0], expected);
        }
    }
}
