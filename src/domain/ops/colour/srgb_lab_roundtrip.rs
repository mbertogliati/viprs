use std::sync::OnceLock;

use crate::domain::{
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

use super::math::{
    lab_to_xyz_components, scrgb_to_srgb_u8_components, scrgb_to_xyz_components, srgb_decode_u8,
    xyz_to_lab_components, xyz_to_scrgb_components,
};

/// Applies the `srgb lab roundtrip` colour transform to image pixels. Use it when a pipeline
/// needs to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::srgb_lab_roundtrip::SRgbLabRoundtrip;
///
/// let op = SRgbLabRoundtrip;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbLabRoundtrip;

const RGB_LUT_STRIDE: usize = 3;
const RGB_LUT_SIZE: usize =
    (u8::MAX as usize + 1) * (u8::MAX as usize + 1) * (u8::MAX as usize + 1) * RGB_LUT_STRIDE;

#[inline(always)]
fn roundtrip_pixel(r: u8, g: u8, b: u8) -> [u8; 3] {
    let red = srgb_decode_u8(r);
    let green = srgb_decode_u8(g);
    let blue = srgb_decode_u8(b);
    let (x, y, z) = scrgb_to_xyz_components(red, green, blue);
    let (l, a, b_star) = xyz_to_lab_components(x, y, z);
    let (x, y, z) = lab_to_xyz_components(l, a, b_star);
    let (red, green, blue) = xyz_to_scrgb_components(x, y, z);
    scrgb_to_srgb_u8_components(red, green, blue)
}

#[inline(always)]
fn roundtrip_lut() -> &'static [u8] {
    static LUT: OnceLock<Box<[u8]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = vec![0_u8; RGB_LUT_SIZE];
        for r in 0..=u8::MAX {
            for g in 0..=u8::MAX {
                for b in 0..=u8::MAX {
                    let index = roundtrip_index(r, g, b);
                    let [out_r, out_g, out_b] = roundtrip_pixel(r, g, b);
                    lut[index] = out_r;
                    lut[index + 1] = out_g;
                    lut[index + 2] = out_b;
                }
            }
        }
        lut.into_boxed_slice()
    })
}

#[inline(always)]
const fn roundtrip_index(r: u8, g: u8, b: u8) -> usize {
    ((((r as usize) << 8) | g as usize) << 8 | b as usize) * RGB_LUT_STRIDE
}

impl Op for SRgbLabRoundtrip {
    type Input = U8;
    type Output = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        let stride = input.bands as usize;
        let lut = roundtrip_lut();
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

impl PixelLocalOp for SRgbLabRoundtrip {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        colour::ColourConvert,
        ops::colour::{LabToSRgb, SRgbToLab},
    };
    use proptest::prelude::*;

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn flatten_rgb(pixels: &[[u8; 3]]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|pixel| pixel.iter().copied())
            .collect()
    }

    fn flatten_rgba(pixels: &[(u8, u8, u8, u8)]) -> Vec<u8> {
        pixels
            .iter()
            .flat_map(|&(r, g, b, a)| [r, g, b, a])
            .collect()
    }

    fn chained_roundtrip(rgb_data: &[u8]) -> Vec<u8> {
        let to_lab = SRgbToLab;
        let to_srgb = LabToSRgb;
        let region = make_region(rgb_data.len() / 3);
        let input = Tile::new(region, 3, rgb_data);

        let mut lab = vec![0.0_f32; rgb_data.len()];
        let mut lab_tile = TileMut::new(region, 3, &mut lab);
        to_lab.convert_region(&mut (), &input, &mut lab_tile);

        let mut roundtrip = vec![0_u8; rgb_data.len()];
        let lab_input = Tile::new(region, 3, &lab);
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip);
        to_srgb.convert_region(&mut (), &lab_input, &mut roundtrip_tile);

        roundtrip
    }

    #[test]
    fn roundtrip_matches_two_stage_chain_for_rgb_and_rgba() {
        let op = SRgbLabRoundtrip;
        let to_lab = SRgbToLab;
        let to_srgb = LabToSRgb;
        let rgb_region = Region::new(0, 0, 2, 1);
        let rgb_input_data = [255_u8, 0, 0, 12, 34, 56];
        let rgb_input = Tile::new(rgb_region, 3, &rgb_input_data);
        let mut rgb_direct = rgb_input_data;
        let mut rgb_direct_tile = TileMut::new(rgb_region, 3, &mut rgb_direct);
        op.process_region(&mut (), &rgb_input, &mut rgb_direct_tile);

        let mut rgb_lab = [0.0_f32; 6];
        let mut rgb_lab_tile = TileMut::new(rgb_region, 3, &mut rgb_lab);
        to_lab.convert_region(&mut (), &rgb_input, &mut rgb_lab_tile);

        let mut rgb_chained = [0_u8; 6];
        let rgb_lab_input = Tile::new(rgb_region, 3, &rgb_lab);
        let mut rgb_chained_tile = TileMut::new(rgb_region, 3, &mut rgb_chained);
        to_srgb.convert_region(&mut (), &rgb_lab_input, &mut rgb_chained_tile);
        assert_eq!(rgb_direct, rgb_chained);

        let rgba_region = Region::new(0, 0, 2, 1);
        let rgba_input_data = [10_u8, 20, 30, 200, 40, 50, 60, 99];
        let rgba_input = Tile::new(rgba_region, 4, &rgba_input_data);
        let mut rgba_direct = rgba_input_data;
        let mut rgba_direct_tile = TileMut::new(rgba_region, 4, &mut rgba_direct);
        op.process_region(&mut (), &rgba_input, &mut rgba_direct_tile);

        let rgb_from_rgba = [
            rgba_input_data[0],
            rgba_input_data[1],
            rgba_input_data[2],
            rgba_input_data[4],
            rgba_input_data[5],
            rgba_input_data[6],
        ];
        let rgb_from_rgba_input = Tile::new(rgba_region, 3, &rgb_from_rgba);
        let mut rgba_lab = [0.0_f32; 6];
        let mut rgba_lab_tile = TileMut::new(rgba_region, 3, &mut rgba_lab);
        to_lab.convert_region(&mut (), &rgb_from_rgba_input, &mut rgba_lab_tile);

        let mut rgba_chained = [0_u8; 6];
        let rgba_lab_input = Tile::new(rgba_region, 3, &rgba_lab);
        let mut rgba_chained_tile = TileMut::new(rgba_region, 3, &mut rgba_chained);
        to_srgb.convert_region(&mut (), &rgba_lab_input, &mut rgba_chained_tile);

        assert_eq!(&rgba_direct[..3], &rgba_chained[..3]);
        assert_eq!(&rgba_direct[4..7], &rgba_chained[3..6]);
        assert_eq!(rgba_direct[3], rgba_input_data[3]);
        assert_eq!(rgba_direct[7], rgba_input_data[7]);
    }

    #[test]
    fn lookup_table_matches_scalar_roundtrip() {
        let lut = roundtrip_lut();
        for [r, g, b] in [[0_u8, 0, 0], [12, 34, 56], [118, 118, 118], [255, 255, 255]] {
            let index = roundtrip_index(r, g, b);
            assert_eq!(
                [lut[index], lut[index + 1], lut[index + 2]],
                roundtrip_pixel(r, g, b)
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn srgb_lab_srgb_roundtrip_preserves_rgb_within_rounding_tolerance(
            pixels in proptest::collection::vec(any::<[u8; 3]>(), 1..=64)
        ) {
            let op = SRgbLabRoundtrip;
            let input_data = flatten_rgb(&pixels);
            let region = make_region(pixels.len());
            let input = Tile::new(region, 3, &input_data);
            let mut output_data = vec![0_u8; input_data.len()];
            let mut output = TileMut::new(region, 3, &mut output_data);
            op.process_region(&mut (), &input, &mut output);

            let chained = chained_roundtrip(&input_data);

            for ((actual, expected_chain), expected_input) in output_data
                .iter()
                .zip(chained.iter())
                .zip(input_data.iter())
            {
                prop_assert_eq!(*actual, *expected_chain);
                prop_assert!((i32::from(*actual) - i32::from(*expected_input)).abs() <= 1);
            }
        }

        #[test]
        fn srgb_lab_srgb_roundtrip_preserves_rgba_and_alpha(
            pixels in proptest::collection::vec((any::<u8>(), any::<u8>(), any::<u8>(), any::<u8>()), 1..=64)
        ) {
            let op = SRgbLabRoundtrip;
            let input_data = flatten_rgba(&pixels);
            let region = make_region(pixels.len());
            let input = Tile::new(region, 4, &input_data);
            let mut output_data = vec![0_u8; input_data.len()];
            let mut output = TileMut::new(region, 4, &mut output_data);
            op.process_region(&mut (), &input, &mut output);

            for (input_pixel, output_pixel) in input_data
                .chunks_exact(4)
                .zip(output_data.chunks_exact(4))
            {
                prop_assert_eq!(output_pixel[3], input_pixel[3]);
                prop_assert!((i32::from(output_pixel[0]) - i32::from(input_pixel[0])).abs() <= 1);
                prop_assert!((i32::from(output_pixel[1]) - i32::from(input_pixel[1])).abs() <= 1);
                prop_assert!((i32::from(output_pixel[2]) - i32::from(input_pixel[2])).abs() <= 1);
            }
        }
    }
}
