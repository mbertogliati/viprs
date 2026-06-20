use std::any::Any;
use std::marker::PhantomData;

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Conditional pixel selection over three input slots.
///
/// Slot 0 is a one-band U8 condition image. Non-zero selects slot 1 (then);
/// zero selects slot 2 (else). Then/else slots use the output format `F`.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::ifthenelse::IfThenElseOp;
///
/// let op = IfThenElseOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct IfThenElseOp<F: BandFormat> {
    branch_bands: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> IfThenElseOp<F> {
    #[must_use]
    /// Creates a new `IfThenElseOp`.
    pub fn new(branch_bands: u32) -> Self {
        debug_assert!(
            branch_bands > 0,
            "IfThenElseOp requires at least one branch band"
        );
        Self {
            branch_bands,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs combined bands.
    pub const fn combined_bands(&self) -> u32 {
        1 + self.branch_bands * 2
    }
}

impl<F> DynOperation for IfThenElseOp<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
    fn input_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.branch_bands
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn input_slot_count(&self) -> usize {
        3
    }

    fn input_format_slot(&self, slot: usize) -> BandFormatId {
        match slot {
            0 => BandFormatId::U8,
            1 | 2 => F::ID,
            _ => {
                debug_assert!(false, "IfThenElseOp input slot out of range");
                BandFormatId::U8
            }
        }
    }

    fn input_bands_slot(&self, slot: usize) -> u32 {
        match slot {
            0 => 1,
            1 | 2 => self.branch_bands,
            _ => {
                debug_assert!(false, "IfThenElseOp input slot out of range");
                self.branch_bands
            }
        }
    }

    fn required_input_region_slot(&self, output: &Region, slot: usize) -> Region {
        debug_assert!(slot < self.input_slot_count());
        *output
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

    fn dyn_process_region(
        &self,
        _state: &mut dyn Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        output_region: Region,
    ) {
        let pixel_count = output_region.pixel_count();
        match F::ID {
            BandFormatId::U8 => self.process_typed::<u8>(input, output, pixel_count),
            BandFormatId::U16 => self.process_typed::<u16>(input, output, pixel_count),
            BandFormatId::I16 => self.process_typed::<i16>(input, output, pixel_count),
            BandFormatId::U32 => self.process_typed::<u32>(input, output, pixel_count),
            BandFormatId::I32 => self.process_typed::<i32>(input, output, pixel_count),
            BandFormatId::F32 => self.process_typed::<f32>(input, output, pixel_count),
            BandFormatId::F64 => self.process_typed::<f64>(input, output, pixel_count),
        }
    }

    fn dyn_process_region_multi(
        &self,
        _state: &mut dyn Any,
        inputs: &[&[u8]],
        output: &mut [u8],
        input_regions: &[Region],
        output_region: Region,
    ) {
        debug_assert_eq!(inputs.len(), self.input_slot_count());
        debug_assert_eq!(input_regions.len(), self.input_slot_count());
        debug_assert!(input_regions.iter().all(|region| *region == output_region));

        let Some((&cond, rest)) = inputs.split_first() else {
            return;
        };
        let [then_input, else_input] = rest else {
            return;
        };

        self.process_multi_typed::<F::Sample>(
            cond,
            then_input,
            else_input,
            output,
            output_region.pixel_count(),
        );
    }
}

impl<F> IfThenElseOp<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
    fn process_typed<T>(&self, input: &[u8], output: &mut [u8], pixel_count: usize)
    where
        T: Pod + Copy + Truthy,
    {
        let src = cast_slice::<u8, T>(input);
        let dst = cast_slice_mut::<u8, T>(output);
        let branch_bands = self.branch_bands as usize;
        let combined_bands = self.combined_bands() as usize;

        debug_assert_eq!(src.len(), pixel_count * combined_bands);
        debug_assert_eq!(dst.len(), pixel_count * branch_bands);

        for px in 0..pixel_count {
            let src_base = px * combined_bands;
            let dst_base = px * branch_bands;
            let cond = src[src_base];
            let then_base = src_base + 1;
            let else_base = then_base + branch_bands;
            let chosen = if cond.truthy() {
                &src[then_base..then_base + branch_bands]
            } else {
                &src[else_base..else_base + branch_bands]
            };
            dst[dst_base..dst_base + branch_bands].copy_from_slice(chosen);
        }
    }

    fn process_multi_typed<T>(
        &self,
        cond: &[u8],
        then_input: &[u8],
        else_input: &[u8],
        output: &mut [u8],
        pixel_count: usize,
    ) where
        T: Pod + Copy,
    {
        let then_samples = cast_slice::<u8, T>(then_input);
        let else_samples = cast_slice::<u8, T>(else_input);
        let dst = cast_slice_mut::<u8, T>(output);
        let branch_bands = self.branch_bands as usize;

        debug_assert_eq!(cond.len(), pixel_count);
        debug_assert_eq!(then_samples.len(), pixel_count * branch_bands);
        debug_assert_eq!(else_samples.len(), pixel_count * branch_bands);
        debug_assert_eq!(dst.len(), pixel_count * branch_bands);

        for (px, &condition) in cond.iter().take(pixel_count).enumerate() {
            let base = px * branch_bands;
            let chosen = if condition != 0 {
                &then_samples[base..base + branch_bands]
            } else {
                &else_samples[base..base + branch_bands]
            };
            dst[base..base + branch_bands].copy_from_slice(chosen);
        }
    }
}

trait Truthy {
    fn truthy(self) -> bool;
}

impl Truthy for u8 {
    fn truthy(self) -> bool {
        self != 0
    }
}

impl Truthy for u16 {
    fn truthy(self) -> bool {
        self != 0
    }
}

impl Truthy for i16 {
    fn truthy(self) -> bool {
        self != 0
    }
}

impl Truthy for u32 {
    fn truthy(self) -> bool {
        self != 0
    }
}

impl Truthy for i32 {
    fn truthy(self) -> bool {
        self != 0
    }
}

impl Truthy for f32 {
    fn truthy(self) -> bool {
        self != 0.0
    }
}

impl Truthy for f64 {
    fn truthy(self) -> bool {
        self != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::format::{BandFormat, F32, F64, I16, I32, U8, U16, U32};

    fn run_ifthenelse(input: &[u8], output: &mut [u8], branch_bands: u32, pixel_count: usize) {
        let op = IfThenElseOp::<U8>::new(branch_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region(state.as_mut(), input, output, region, region);
    }

    fn run_ifthenelse_typed<F>(
        input: &[F::Sample],
        output: &mut [F::Sample],
        branch_bands: u32,
        pixel_count: usize,
    ) where
        F: BandFormat,
        F::Sample: Pod + Copy,
    {
        let op = IfThenElseOp::<F>::new(branch_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region(
            state.as_mut(),
            cast_slice(input),
            cast_slice_mut(output),
            region,
            region,
        );
    }

    fn run_ifthenelse_multi_f32(
        cond: &[u8],
        then_input: &[f32],
        else_input: &[f32],
        output: &mut [f32],
        branch_bands: u32,
        pixel_count: usize,
    ) {
        let op = IfThenElseOp::<F32>::new(branch_bands);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let mut state = op.dyn_start();
        let inputs = [
            cond,
            cast_slice::<f32, u8>(then_input),
            cast_slice::<f32, u8>(else_input),
        ];
        let input_regions = [region, region, region];
        op.dyn_process_region_multi(
            state.as_mut(),
            &inputs,
            cast_slice_mut::<f32, u8>(output),
            &input_regions,
            region,
        );
    }

    #[test]
    fn per_slot_metadata_describes_u8_condition_and_typed_branches() {
        let op = IfThenElseOp::<F32>::new(3);

        assert_eq!(op.input_slot_count(), 3);
        assert_eq!(op.input_format_slot(0), BandFormatId::U8);
        assert_eq!(op.input_format_slot(1), BandFormatId::F32);
        assert_eq!(op.input_format_slot(2), BandFormatId::F32);
        assert_eq!(op.input_bands_slot(0), 1);
        assert_eq!(op.input_bands_slot(1), 3);
        assert_eq!(op.input_bands_slot(2), 3);
    }

    #[test]
    fn all_true_returns_then_image() {
        let input = [
            255u8, 10, 20, 30, 40, //
            255, 50, 60, 70, 80,
        ];
        let mut output = [0u8; 4];
        run_ifthenelse(&input, &mut output, 2, 2);
        assert_eq!(output, [10, 20, 50, 60]);
    }

    #[test]
    fn all_false_returns_else_image() {
        let input = [
            0u8, 10, 20, 30, 40, //
            0, 50, 60, 70, 80,
        ];
        let mut output = [0u8; 4];
        run_ifthenelse(&input, &mut output, 2, 2);
        assert_eq!(output, [30, 40, 70, 80]);
    }

    #[test]
    fn metadata_reports_combined_bands_and_identity_geometry() {
        let op = IfThenElseOp::<F32>::new(2);
        let region = Region::new(4, 5, 6, 7);

        assert_eq!(op.combined_bands(), 5);
        assert_eq!(op.input_format(), BandFormatId::U8);
        assert_eq!(op.output_format(), BandFormatId::F32);
        assert_eq!(op.bands(), 2);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.required_input_region_slot(&region, 0), region);
        assert_eq!(op.required_input_region_slot(&region, 2), region);
        assert_eq!(op.node_spec(8, 9), NodeSpec::identity(8, 9));
    }

    #[test]
    fn truthy_is_non_zero_for_all_supported_samples() {
        assert!(1u8.truthy());
        assert!(!0u8.truthy());
        assert!(1u16.truthy());
        assert!(!0u16.truthy());
        assert!((-1i16).truthy());
        assert!(!0i16.truthy());
        assert!(1u32.truthy());
        assert!(!0u32.truthy());
        assert!((-1i32).truthy());
        assert!(!0i32.truthy());
        assert!(0.5f32.truthy());
        assert!(!0.0f32.truthy());
        assert!(0.5f64.truthy());
        assert!(!0.0f64.truthy());
    }

    #[test]
    fn dyn_process_region_supports_all_numeric_branch_formats() {
        let mut out_u16 = [0u16; 1];
        run_ifthenelse_typed::<U16>(&[1u16, 7, 9], &mut out_u16, 1, 1);
        assert_eq!(out_u16, [7]);

        let mut out_i16 = [0i16; 1];
        run_ifthenelse_typed::<I16>(&[0i16, -4, 12], &mut out_i16, 1, 1);
        assert_eq!(out_i16, [12]);

        let mut out_u32 = [0u32; 1];
        run_ifthenelse_typed::<U32>(&[5u32, 11, 13], &mut out_u32, 1, 1);
        assert_eq!(out_u32, [11]);

        let mut out_i32 = [0i32; 1];
        run_ifthenelse_typed::<I32>(&[0i32, -8, 21], &mut out_i32, 1, 1);
        assert_eq!(out_i32, [21]);

        let mut out_f64 = [0.0f64; 1];
        run_ifthenelse_typed::<F64>(&[1.0f64, 0.25, 0.75], &mut out_f64, 1, 1);
        assert_eq!(out_f64, [0.25]);
    }

    proptest! {
        #[test]
        fn boundary_single_band_selection(
            cond in prop_oneof![Just(0u8), Just(255u8)],
            then_sample in any::<u8>(),
            else_sample in any::<u8>(),
        ) {
            let input = [cond, then_sample, else_sample];
            let mut output = [0u8; 1];
            run_ifthenelse(&input, &mut output, 1, 1);
            prop_assert_eq!(output[0], if cond == 0 { else_sample } else { then_sample });
        }
    }

    #[test]
    fn multi_input_f32_uses_u8_condition_slot() {
        let cond = [0u8, 1, 255];
        let then_input = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
        let else_input = [-10.0f32, -20.0, -30.0, -40.0, -50.0, -60.0];
        let mut output = [0.0f32; 6];

        run_ifthenelse_multi_f32(&cond, &then_input, &else_input, &mut output, 2, 3);

        assert_eq!(output, [-10.0, -20.0, 30.0, 40.0, 50.0, 60.0]);
    }
}
