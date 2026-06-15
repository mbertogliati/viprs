use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    domain::op::{NodeSpec, Op},
    domain::{
        error::ViprsError,
        format::{BandFormat, F32, F64, I16, I32, U8, U16, U32},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::common::{FromF64, ToF64};

/// Fast unnormalized correlation surface: sum of squared differences.
///
/// The output format follows the libvips `fastcor` promotion table: integer
/// inputs output U32, F32 inputs output F32, and F64 inputs output F64.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::convolution::fastcor::FastCorOp;
///
/// let op = FastCorOp::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct FastCorOp<F: BandFormat> {
    reference: Box<[f64]>,
    ref_width: usize,
    ref_height: usize,
    ref_bands: usize,
    radius_x: u32,
    radius_y: u32,
    _format: PhantomData<F>,
}

impl<F: BandFormat> FastCorOp<F>
where
    F::Sample: ToF64,
{
    /// Creates a new `FastCorOp`.
    pub fn new(
        reference: Vec<F::Sample>,
        ref_width: u32,
        ref_height: u32,
        ref_bands: u32,
    ) -> Result<Self, ViprsError> {
        let expected = ref_width as usize * ref_height as usize * ref_bands as usize;
        if reference.len() != expected || ref_width == 0 || ref_height == 0 || ref_bands == 0 {
            return Err(ViprsError::Codec(
                "FastCorOp: reference dimensions must match the supplied buffer".to_owned(),
            ));
        }

        let ref_width = ref_width as usize;
        let ref_height = ref_height as usize;
        let ref_bands = ref_bands as usize;
        let reference = reference
            .into_iter()
            .map(ToF64::to_f64)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            reference,
            ref_width,
            ref_height,
            ref_bands,
            radius_x: (ref_width / 2) as u32,
            radius_y: (ref_height / 2) as u32,
            _format: PhantomData,
        })
    }
}

impl<F: BandFormat> FastCorOp<F>
where
    F::Sample: ToF64 + Pod,
{
    const fn required_region(&self, output: &Region) -> Region {
        Region::new(
            output.x - self.radius_x as i32,
            output.y - self.radius_y as i32,
            output.width + 2 * self.radius_x,
            output.height + 2 * self.radius_y,
        )
    }

    const fn spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec {
            input_tile_w: tile_w + 2 * self.radius_x,
            input_tile_h: tile_h + 2 * self.radius_y,
            output_tile_w: tile_w,
            output_tile_h: tile_h,
            coordinate_driven_source: None,
        }
    }

    #[inline]
    fn process_as<O>(&self, input: &Tile<F>, output: &mut TileMut<O>)
    where
        O: BandFormat,
        O::Sample: FromF64 + Pod,
    {
        let input_bands = input.bands as usize;
        let output_bands = output.bands as usize;
        if !bands_are_compatible(input_bands, self.ref_bands, output_bands) {
            debug_assert!(
                false,
                "FastCorOp band counts require each input to be one-band or match output bands"
            );
            return;
        }

        let out_w = output.region.width as usize;
        let out_h = output.region.height as usize;
        let in_w = input.region.width as usize;

        for oy in 0..out_h {
            for ox in 0..out_w {
                for band in 0..output_bands {
                    let input_band = if input_bands == 1 { 0 } else { band };
                    let ref_band = if self.ref_bands == 1 { 0 } else { band };
                    let mut sum = 0.0f64;

                    for ky in 0..self.ref_height {
                        for kx in 0..self.ref_width {
                            let input_idx = ((oy + ky) * in_w + ox + kx) * input_bands + input_band;
                            let ref_idx = ((ky * self.ref_width + kx) * self.ref_bands) + ref_band;
                            let difference =
                                input.data[input_idx].to_f64() - self.reference[ref_idx];
                            sum = difference.mul_add(difference, sum);
                        }
                    }

                    let out_idx = (oy * out_w + ox) * output_bands + band;
                    output.data[out_idx] = O::Sample::from_f64(sum);
                }
            }
        }
    }
}

#[inline]
const fn bands_are_compatible(input_bands: usize, ref_bands: usize, output_bands: usize) -> bool {
    output_bands > 0
        && (input_bands == 1 || input_bands == output_bands)
        && (ref_bands == 1 || ref_bands == output_bands)
}

macro_rules! impl_fastcor_output {
    ($output:ty; $($input:ty),+ $(,)?) => {
        $(
            impl Op for FastCorOp<$input> {
                type Input = $input;
                type Output = $output;
                type State = ();

                fn demand_hint(&self) -> DemandHint {
                    DemandHint::FatStrip
                }

                fn required_input_region(&self, output: &Region) -> Region {
                    self.required_region(output)
                }

                fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
                    self.spec(tile_w, tile_h)
                }

                fn start(&self) {}

                #[inline]
                fn process_region(
                    &self,
                    _state: &mut Self::State,
                    input: &Tile<Self::Input>,
                    output: &mut TileMut<Self::Output>,
                ) {
                    self.process_as(input, output);
                }
            }
        )+
    };
}

impl_fastcor_output!(U32; U8, U16, I16, U32, I32);
impl_fastcor_output!(F32; F32);
impl_fastcor_output!(F64; F64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        format::{F32, F64, U16, U32},
        image::{Region, Tile, TileMut},
    };
    use proptest::prelude::*;

    fn run_f32(reference: Vec<f32>, input: &[f32]) -> f32 {
        let op = FastCorOp::<F32>::new(reference, 3, 3, 1).unwrap();
        let input_tile = Tile::<F32>::new(Region::new(0, 0, 3, 3), 1, input);
        let mut output_data = vec![1.0f32; 1];
        let mut output = TileMut::<F32>::new(Region::new(0, 0, 1, 1), 1, &mut output_data);
        let mut state = ();
        op.process_region(&mut state, &input_tile, &mut output);
        output_data[0]
    }

    #[test]
    fn matching_patch_has_zero_difference() {
        let patch = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(run_f32(patch.clone(), &patch), 0.0);
    }

    #[test]
    fn one_band_reference_expands_across_input_bands() {
        let reference = vec![2u16; 9];
        let op = FastCorOp::<U16>::new(reference, 3, 3, 1).unwrap();
        fn output_is_u32<O: Op<Input = U16, Output = U32>>(_: &O) {}
        output_is_u32(&op);
        let input = vec![3u16, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5];
        let input_tile = Tile::<U16>::new(Region::new(0, 0, 3, 3), 2, &input);
        let mut output_data = vec![0u32; 2];
        let mut output = TileMut::<U32>::new(Region::new(0, 0, 1, 1), 2, &mut output_data);
        let mut state = ();

        op.process_region(&mut state, &input_tile, &mut output);

        assert_eq!(output_data, vec![9, 81]);
    }

    #[test]
    fn metadata_expands_by_reference_radius() {
        let op = FastCorOp::<F32>::new(vec![0.0; 15], 5, 3, 1).unwrap();
        let output = Region::new(7, 9, 11, 13);
        assert_eq!(op.demand_hint(), DemandHint::FatStrip);
        assert_eq!(op.required_input_region(&output), Region::new(5, 8, 15, 15));
        let spec = op.node_spec(11, 13);
        assert_eq!(spec.input_tile_w, 15);
        assert_eq!(spec.input_tile_h, 15);
        assert_eq!(spec.output_tile_w, 11);
        assert_eq!(spec.output_tile_h, 13);
    }

    proptest! {
        #[test]
        fn identical_reference_and_patch_have_zero_ssd(
            values in prop::collection::vec(-20.0f32..20.0, 9),
        ) {
            let score = run_f32(values.clone(), &values);
            prop_assert!(score.abs() < 1e-6);
        }

        #[test]
        fn zero_reference_accumulates_squares(value in 0u16..1024u16) {
            let reference = vec![0u16; 9];
            let input = vec![value; 9];
            let op = FastCorOp::<U16>::new(reference, 3, 3, 1).unwrap();
            let input_tile = Tile::<U16>::new(Region::new(0, 0, 3, 3), 1, &input);
            let mut output_data = vec![0u32; 1];
            let mut output = TileMut::<U32>::new(Region::new(0, 0, 1, 1), 1, &mut output_data);
            let mut state = ();

            op.process_region(&mut state, &input_tile, &mut output);

            let expected = 9 * u32::from(value) * u32::from(value);
            prop_assert_eq!(output_data[0], expected);
        }
    }

    #[test]
    fn f64_input_preserves_f64_output() {
        fn output_is_f64<O: Op<Input = F64, Output = F64>>(_: &O) {}
        let op = FastCorOp::<F64>::new(vec![1.0; 9], 3, 3, 1).unwrap();
        output_is_f64(&op);
    }
}
