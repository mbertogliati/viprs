use std::any::Any;
use std::marker::PhantomData;

use bytemuck::{Pod, cast_slice, cast_slice_mut};

use viprs_core::{
    format::{BandFormat, BandFormatId},
    image::{DemandHint, Region},
    op::{DynOperation, NodeSpec},
};

/// Append a fully-opaque alpha band to an image.
///
/// # Examples
/// ```ignore
/// use viprs_ops_composite::conversion::addalpha::AddAlphaOp;
///
/// let op = AddAlphaOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct AddAlphaOp<F: BandFormat> {
    /// Stores the `alpha_value` value for this item.
    pub alpha_value: f64,
    input_bands: u32,
    _phantom: PhantomData<F>,
}

impl<F: BandFormat> AddAlphaOp<F> {
    #[must_use]
    /// Creates a new `AddAlphaOp`.
    pub fn new(input_bands: u32, alpha_value: f64) -> Self {
        debug_assert!(
            input_bands > 0,
            "AddAlphaOp requires at least one input band"
        );
        Self {
            alpha_value,
            input_bands,
            _phantom: PhantomData,
        }
    }

    #[must_use]
    /// Returns or performs output bands.
    pub const fn output_bands(&self) -> u32 {
        self.input_bands + 1
    }
}

impl<F> DynOperation for AddAlphaOp<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
    fn input_format(&self) -> BandFormatId {
        F::ID
    }

    fn output_format(&self) -> BandFormatId {
        F::ID
    }

    fn bands(&self) -> u32 {
        self.output_bands()
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
}

impl<F> AddAlphaOp<F>
where
    F: BandFormat,
    F::Sample: Pod,
{
    fn process_typed<T>(&self, input: &[u8], output: &mut [u8], pixel_count: usize)
    where
        T: Pod + Copy + AlphaCast,
    {
        let src = cast_slice::<u8, T>(input);
        let dst = cast_slice_mut::<u8, T>(output);
        let input_bands = self.input_bands as usize;
        let output_bands = self.output_bands() as usize;
        let alpha = T::from_alpha_value(self.alpha_value);

        debug_assert_eq!(src.len(), pixel_count * input_bands);
        debug_assert_eq!(dst.len(), pixel_count * output_bands);

        for px in 0..pixel_count {
            let src_base = px * input_bands;
            let dst_base = px * output_bands;
            dst[dst_base..dst_base + input_bands]
                .copy_from_slice(&src[src_base..src_base + input_bands]);
            dst[dst_base + input_bands] = alpha;
        }
    }
}

trait AlphaCast: Copy + Pod {
    fn from_alpha_value(value: f64) -> Self;
}

impl AlphaCast for u8 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for u16 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for i16 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for u32 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for i32 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for f32 {
    fn from_alpha_value(value: f64) -> Self {
        value as Self
    }
}

impl AlphaCast for f64 {
    fn from_alpha_value(value: f64) -> Self {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
        image::Region,
    };

    fn run_addalpha_u8(input_bands: u32, input: &[u8], output: &mut [u8], pixel_count: usize) {
        let op = AddAlphaOp::<U8>::new(input_bands, 255.0);
        let region = Region::new(0, 0, pixel_count as u32, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region(state.as_mut(), input, output, region, region);
    }

    fn run_addalpha_typed<F>(
        input_bands: u32,
        alpha_value: f64,
        input: &[F::Sample],
        output: &mut [F::Sample],
        pixel_count: usize,
    ) where
        F: BandFormat,
        F::Sample: Pod,
    {
        let op = AddAlphaOp::<F>::new(input_bands, alpha_value);
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

    #[test]
    fn adds_opaque_alpha_to_rgb() {
        let input = [10u8, 20, 30, 40, 50, 60];
        let mut output = [0u8; 8];
        run_addalpha_u8(3, &input, &mut output, 2);
        assert_eq!(output, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn adds_unit_alpha_for_f32() {
        let op = AddAlphaOp::<F32>::new(2, 1.0);
        let input = [0.25f32, 0.5, 0.75, 1.0];
        let mut output = [0.0f32; 6];
        let region = Region::new(0, 0, 2, 1);
        let mut state = op.dyn_start();
        op.dyn_process_region(
            state.as_mut(),
            cast_slice(&input),
            cast_slice_mut(&mut output),
            region,
            region,
        );
        assert_eq!(output, [0.25, 0.5, 1.0, 0.75, 1.0, 1.0]);
    }

    #[test]
    fn metadata_reports_identity_geometry_and_output_band_count() {
        let op = AddAlphaOp::<U16>::new(2, 65_535.0);
        let region = Region::new(7, 3, 4, 5);

        assert_eq!(op.input_format(), BandFormatId::U16);
        assert_eq!(op.output_format(), BandFormatId::U16);
        assert_eq!(op.bands(), 3);
        assert_eq!(op.demand_hint(), DemandHint::ThinStrip);
        assert_eq!(op.required_input_region(&region), region);
        assert_eq!(op.node_spec(8, 9), NodeSpec::identity(8, 9));
    }

    #[test]
    fn alpha_cast_converts_all_supported_sample_types() {
        assert_eq!(<u8 as AlphaCast>::from_alpha_value(255.9), 255);
        assert_eq!(<u16 as AlphaCast>::from_alpha_value(513.0), 513);
        assert_eq!(<i16 as AlphaCast>::from_alpha_value(-7.9), -7);
        assert_eq!(<u32 as AlphaCast>::from_alpha_value(4_096.0), 4_096);
        assert_eq!(<i32 as AlphaCast>::from_alpha_value(-4_096.0), -4_096);
        assert_eq!(<f32 as AlphaCast>::from_alpha_value(0.25), 0.25f32);
        assert_eq!(<f64 as AlphaCast>::from_alpha_value(0.5), 0.5f64);
    }

    #[test]
    fn dyn_process_region_supports_all_numeric_output_formats() {
        let mut output_u16 = [0u16; 4];
        run_addalpha_typed::<U16>(1, 512.0, &[3u16, 7], &mut output_u16, 2);
        assert_eq!(output_u16, [3, 512, 7, 512]);

        let mut output_i16 = [0i16; 4];
        run_addalpha_typed::<I16>(1, -9.0, &[2i16, -4], &mut output_i16, 2);
        assert_eq!(output_i16, [2, -9, -4, -9]);

        let mut output_u32 = [0u32; 4];
        run_addalpha_typed::<U32>(1, 7.0, &[11u32, 13], &mut output_u32, 2);
        assert_eq!(output_u32, [11, 7, 13, 7]);

        let mut output_i32 = [0i32; 4];
        run_addalpha_typed::<I32>(1, -3.0, &[5i32, -8], &mut output_i32, 2);
        assert_eq!(output_i32, [5, -3, -8, -3]);

        let mut output_f64 = [0.0f64; 4];
        run_addalpha_typed::<F64>(1, 0.75, &[0.1f64, 0.9], &mut output_f64, 2);
        assert_eq!(output_f64, [0.1, 0.75, 0.9, 0.75]);
    }

    #[test]
    fn addalpha_supports_single_band_and_dyn_start_with_tile() {
        let op = AddAlphaOp::<U8>::new(1, 255.0);
        let region = Region::new(0, 0, 3, 1);
        let mut state = op.dyn_start_with_tile(3, 1);
        let input = [10u8, 20, 30];
        let mut output = [0u8; 6];
        op.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
        assert_eq!(output, [10, 255, 20, 255, 30, 255]);
    }

    #[test]
    fn metadata_reports_input_slot_shape_for_rg_format() {
        let op = AddAlphaOp::<F32>::new(2, 1.0);
        assert_eq!(op.input_slot_count(), 1);
        assert_eq!(op.input_bands_slot(0), 3);
        assert_eq!(op.output_bands(), 3);
    }

    proptest! {
        #[test]
        fn identity_preserves_existing_bands(
            pixels in proptest::collection::vec(any::<u8>(), 3..96)
        ) {
            let pixel_count = pixels.len() / 3;
            prop_assume!(pixel_count > 0);
            let input = &pixels[..pixel_count * 3];
            let mut output = vec![0u8; pixel_count * 4];
            run_addalpha_u8(3, input, &mut output, pixel_count);

            for px in 0..pixel_count {
                let src = px * 3;
                let dst = px * 4;
                prop_assert_eq!(&output[dst..dst + 3], &input[src..src + 3]);
                prop_assert_eq!(output[dst + 3], 255);
            }
        }
    }
}
