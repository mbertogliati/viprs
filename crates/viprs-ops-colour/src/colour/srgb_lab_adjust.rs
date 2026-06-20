use std::sync::Arc;

use crate::colour::{LabToSRgb, SRgbToLab};
use viprs_ops_pixel::arithmetic::Linear;

use viprs_core::{
    colour::ColourConvert,
    error::BuildError,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `srgb lab adjust` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::srgb_lab_adjust::SRgbLabAdjust;
///
/// let op = SRgbLabAdjust::new(/* operation parameters */);
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbLabAdjust {
    lut: Arc<[u8]>,
    brightness_scale: f64,
    contrast_shift: f64,
}

impl Clone for SRgbLabAdjust {
    fn clone(&self) -> Self {
        Self {
            lut: Arc::clone(&self.lut),
            brightness_scale: self.brightness_scale,
            contrast_shift: self.contrast_shift,
        }
    }
}

impl SRgbLabAdjust {
    /// Callers that reuse the same parameters across many pipelines should cache the
    /// constructed op instance at the call site.
    pub fn new(scale: f64, offset: f64) -> Result<Self, BuildError> {
        if !scale.is_finite() || !offset.is_finite() {
            return Err(BuildError::InvalidLinearParameters { scale, offset });
        }

        Ok(Self {
            lut: lab_adjust_lut(scale, offset)?,
            brightness_scale: scale,
            contrast_shift: offset,
        })
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        (self.brightness_scale - 1.0_f64).abs() < f64::EPSILON
            && self.contrast_shift.abs() < f64::EPSILON
    }
}

const fn validate_adjust_bands(input_bands: u32, output_bands: u32) -> Result<(), BuildError> {
    if input_bands < 3 || output_bands != input_bands {
        return Err(BuildError::InvalidOperationBands {
            op: "SRgbLabAdjust",
            input_bands,
            output_bands,
            expected: "at least 3 bands",
            expected_output: "same as input",
        });
    }

    Ok(())
}

const RGB_LUT_STRIDE: usize = 3;
const RGB_LUT_SIZE: usize =
    (u8::MAX as usize + 1) * (u8::MAX as usize + 1) * (u8::MAX as usize + 1) * RGB_LUT_STRIDE;

#[inline(always)]
const fn roundtrip_index(r: u8, g: u8, b: u8) -> usize {
    ((((r as usize) << 8) | g as usize) << 8 | b as usize) * RGB_LUT_STRIDE
}

fn lab_adjust_lut(scale: f64, offset: f64) -> Result<Arc<[u8]>, BuildError> {
    let to_lab = SRgbToLab;
    let linear = Linear::<F32>::new(scale, offset)?;
    let to_srgb = LabToSRgb;
    let region = Region::new(0, 0, 256, 1);
    let mut input_row = [0_u8; 256 * RGB_LUT_STRIDE];
    let mut lab_row = [0.0_f32; 256 * RGB_LUT_STRIDE];
    let mut adjusted_row = [0.0_f32; 256 * RGB_LUT_STRIDE];
    let mut output_row = [0_u8; 256 * RGB_LUT_STRIDE];
    let mut lut = vec![0_u8; RGB_LUT_SIZE];

    to_lab.start();
    linear.start();
    to_srgb.start();

    for r in 0..=u8::MAX {
        for g in 0..=u8::MAX {
            for (b, pixel) in input_row.chunks_exact_mut(RGB_LUT_STRIDE).enumerate() {
                pixel[0] = r;
                pixel[1] = g;
                pixel[2] = b as u8;
            }

            let input = Tile::new(region, 3, &input_row);
            let mut lab_tile = TileMut::new(region, 3, &mut lab_row);
            to_lab.convert_region(&mut (), &input, &mut lab_tile);

            let lab_input = Tile::new(region, 3, &lab_row);
            let mut adjusted_tile = TileMut::new(region, 3, &mut adjusted_row);
            linear.process_region(&mut (), &lab_input, &mut adjusted_tile);

            let adjusted_input = Tile::new(region, 3, &adjusted_row);
            let mut output_tile = TileMut::new(region, 3, &mut output_row);
            to_srgb.convert_region(&mut (), &adjusted_input, &mut output_tile);

            let index = roundtrip_index(r, g, 0);
            lut[index..index + output_row.len()].copy_from_slice(&output_row);
        }
    }
    Ok(Arc::from(lut.into_boxed_slice()))
}

impl Op for SRgbLabAdjust {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        validate_adjust_bands(input_bands, output_bands)
    }

    fn validate_region_contract(
        &self,
        _input_region: Region,
        input_bands: u32,
        _output_region: Region,
        output_bands: u32,
    ) -> Result<(), viprs_core::error::ViprsError> {
        validate_adjust_bands(input_bands, output_bands)
            .map_err(viprs_core::error::ViprsError::from)
    }

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        let stride = input.bands as usize;
        debug_assert!(stride >= 3, "SRgbLabAdjust requires at least 3 bands");

        if self.is_identity() {
            output.data.copy_from_slice(input.data);
            return;
        }

        let lut = &self.lut;

        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(stride)
            .zip(output.data.chunks_exact_mut(stride))
        {
            let index = roundtrip_index(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lut[index];
            pixel_out[1] = lut[index + 1];
            pixel_out[2] = lut[index + 2];
            if stride > 3 {
                pixel_out[3..].copy_from_slice(&pixel_in[3..]);
            }
        }
    }
}

impl PixelLocalOp for SRgbLabAdjust {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    use crate::colour::{LabToSRgb, SRgbToLab};
    use viprs_ops_pixel::arithmetic::Linear;

    use proptest::prelude::*;
    use viprs_core::{colour::ColourConvert, image::TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn flatten_rgb(pixels: &[[u8; 3]]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect()
    }

    fn default_op() -> &'static SRgbLabAdjust {
        static OP: OnceLock<SRgbLabAdjust> = OnceLock::new();
        OP.get_or_init(|| SRgbLabAdjust::new(1.05, -3.5).unwrap())
    }

    fn chained_adjust(rgb_data: &[u8], scale: f64, offset: f64) -> Vec<u8> {
        let to_lab = SRgbToLab;
        let linear = Linear::<viprs_core::format::F32>::new(scale, offset).unwrap();
        let to_srgb = LabToSRgb;
        let region = make_region(rgb_data.len() / 3);
        let input = Tile::new(region, 3, rgb_data);

        let mut lab = vec![0.0_f32; rgb_data.len()];
        let mut lab_tile = TileMut::new(region, 3, &mut lab);
        to_lab.convert_region(&mut (), &input, &mut lab_tile);

        let mut adjusted = vec![0.0_f32; rgb_data.len()];
        let lab_input = Tile::new(region, 3, &lab);
        let mut adjusted_tile = TileMut::new(region, 3, &mut adjusted);
        linear.process_region(&mut (), &lab_input, &mut adjusted_tile);

        let mut roundtrip = vec![0_u8; rgb_data.len()];
        let adjusted_input = Tile::new(region, 3, &adjusted);
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip);
        to_srgb.convert_region(&mut (), &adjusted_input, &mut roundtrip_tile);

        roundtrip
    }

    #[test]
    fn matches_chained_lab_linear_roundtrip_for_boundary_pixels() {
        let op = default_op();
        let region = make_region(4);
        let input_data = [0_u8, 0, 0, 255, 255, 255, 255, 0, 0, 12, 34, 56];
        let input = Tile::new(region, 3, &input_data);
        let mut output_data = vec![0_u8; input_data.len()];
        let mut output = TileMut::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, chained_adjust(&input_data, 1.05, -3.5));
    }

    #[test]
    fn preserves_alpha_for_rgba_input() {
        let op = default_op();
        let region = make_region(2);
        let input_data = [10_u8, 20, 30, 200, 40, 50, 60, 99];
        let input = Tile::new(region, 4, &input_data);
        let mut output_data = vec![0_u8; input_data.len()];
        let mut output = TileMut::new(region, 4, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        let rgb_expected = chained_adjust(&[10, 20, 30, 40, 50, 60], 1.05, -3.5);
        assert_eq!(&output_data[..3], &rgb_expected[..3]);
        assert_eq!(&output_data[4..7], &rgb_expected[3..6]);
        assert_eq!(output_data[3], input_data[3]);
        assert_eq!(output_data[7], input_data[7]);
    }

    #[test]
    fn execute_region_rejects_tiles_with_fewer_than_three_bands() {
        let op = default_op();
        let region = make_region(1);
        let input_data = [42_u8, 7];
        let input = Tile::new(region, 2, &input_data);
        let mut output_data = vec![0_u8; input_data.len()];
        let mut output = TileMut::new(region, 2, &mut output_data);

        let err = <SRgbLabAdjust as Op>::execute_region(&op, &mut (), &input, &mut output)
            .expect_err("2-band tiles must be rejected before processing");

        assert!(matches!(
            err,
            viprs_core::error::ViprsError::Build(BuildError::InvalidOperationBands {
                op: "SRgbLabAdjust",
                input_bands: 2,
                output_bands: 2,
                expected: "at least 3 bands",
                expected_output: "same as input",
            })
        ));
    }

    #[test]
    fn new_builds_a_fresh_lut_instead_of_reusing_a_global_cache() {
        let first = SRgbLabAdjust::new(1.05, -3.5).unwrap();
        let second = SRgbLabAdjust::new(1.05, -3.5).unwrap();

        assert!(
            !Arc::ptr_eq(&first.lut, &second.lut),
            "SRgbLabAdjust::new should not retain a process-global LUT"
        );
    }

    #[test]
    fn identity_params_are_pixel_exact_for_all_rgb_values() {
        let op = SRgbLabAdjust::new(1.0, 0.0).unwrap();
        let region = make_region(256);
        let mut input_data = [0_u8; 256 * RGB_LUT_STRIDE];
        let mut output_data = [0_u8; 256 * RGB_LUT_STRIDE];

        for r in 0_u8..=u8::MAX {
            for g in 0_u8..=u8::MAX {
                for (b, pixel) in input_data.chunks_exact_mut(RGB_LUT_STRIDE).enumerate() {
                    pixel[0] = r;
                    pixel[1] = g;
                    pixel[2] = b as u8;
                }

                let input = Tile::new(region, 3, &input_data);
                let mut output = TileMut::new(region, 3, &mut output_data);
                op.process_region(&mut (), &input, &mut output);

                assert_eq!(
                    output_data, input_data,
                    "identity drifted for row r={r}, g={g}"
                );
            }
        }
    }

    #[test]
    fn identity_params_bypass_the_lut() {
        let mut op = SRgbLabAdjust::new(1.0, 0.0).unwrap();
        Arc::make_mut(&mut op.lut)[..RGB_LUT_STRIDE].copy_from_slice(&[255, 0, 0]);

        let region = make_region(1);
        let input_data = [0_u8, 0, 0];
        let input = Tile::new(region, 3, &input_data);
        let mut output_data = [0_u8; RGB_LUT_STRIDE];
        let mut output = TileMut::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, input_data);
    }

    #[test]
    fn chained_adjust_singleton_matches_vectorized_batch_for_boundary_pixel() {
        let pixel = [1_u8, 185, 228];
        let singleton = chained_adjust(&pixel, 1.05, -3.5);
        let batch = [pixel, pixel, pixel, pixel].concat();
        let batched = chained_adjust(&batch, 1.05, -3.5);

        assert_eq!(singleton, batched[..RGB_LUT_STRIDE]);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn matches_chained_lab_linear_roundtrip_for_rgb_tiles(
            pixels in proptest::collection::vec(any::<[u8; 3]>(), 1..=64)
        ) {
            let op = default_op();
            let input_data = flatten_rgb(&pixels);
            let region = make_region(pixels.len());
            let input = Tile::new(region, 3, &input_data);
            let mut output_data = vec![0_u8; input_data.len()];
            let mut output = TileMut::new(region, 3, &mut output_data);
            op.process_region(&mut (), &input, &mut output);

            let chained = chained_adjust(&input_data, 1.05, -3.5);
            prop_assert_eq!(output_data.len(), chained.len());

            for (index, (&actual_channel, &expected_channel)) in output_data.iter().zip(&chained).enumerate() {
                let delta = i16::from(actual_channel) - i16::from(expected_channel);
                prop_assert!(
                    delta.abs() <= 1,
                    "channel mismatch at index {index}: actual={actual_channel}, expected={expected_channel}, delta={delta}"
                );
            }
        }
    }
}
