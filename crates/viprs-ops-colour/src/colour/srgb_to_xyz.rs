use viprs_core::{
    colorspace::{SRgb, Xyz},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::{scrgb_to_xyz_components, srgb_decode_u8};

/// Convert one sRGB pixel (u8 × 3) to CIE XYZ D65 (f32 × 3).
#[inline(always)]
fn srgb_u8_to_xyz_f32(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let red = srgb_decode_u8(r);
    let green = srgb_decode_u8(g);
    let blue = srgb_decode_u8(b);
    scrgb_to_xyz_components(red, green, blue)
}

/// Applies the `srgb to xyz` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::srgb_to_xyz::SRgbToXyz;
///
/// let op = SRgbToXyz;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbToXyz;

impl ColourConvert<SRgb, Xyz> for SRgbToXyz {
    type InputFormat = U8;
    type OutputFormat = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (x, y, z) = srgb_u8_to_xyz_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = x;
            pixel_out[1] = y;
            pixel_out[2] = z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn red_pixel_xyz() {
        let converter = SRgbToXyz;
        let input_data: [u8; 3] = [255, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // sRGB(255,0,0) → XYZ D65 ≈ (0.4124, 0.2127, 0.0193)
        assert!(
            (output_data[0] - 0.4124).abs() < 1e-3,
            "X={}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 0.2127).abs() < 1e-3,
            "Y={}",
            output_data[1]
        );
        assert!(
            (output_data[2] - 0.0193).abs() < 1e-3,
            "Z={}",
            output_data[2]
        );
    }

    #[test]
    fn white_pixel_xyz() {
        let converter = SRgbToXyz;
        let input_data: [u8; 3] = [255, 255, 255];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // White → XYZ D65 illuminant: X≈0.9505, Y≈1.0, Z≈1.0888
        assert!(
            (output_data[0] - 0.9505).abs() < 1e-3,
            "X={}",
            output_data[0]
        );
        assert!(
            (output_data[1] - 1.0000).abs() < 1e-3,
            "Y={}",
            output_data[1]
        );
        assert!(
            (output_data[2] - 1.0888).abs() < 1e-3,
            "Z={}",
            output_data[2]
        );
    }

    #[test]
    fn black_pixel_xyz() {
        let converter = SRgbToXyz;
        let input_data: [u8; 3] = [0, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert!(output_data[0].abs() < 1e-6);
        assert!(output_data[1].abs() < 1e-6);
        assert!(output_data[2].abs() < 1e-6);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(SRgbToXyz.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(SRgbToXyz.required_input_region(&r), r);
    }

    proptest! {
        #[test]
        fn srgb_xyz_srgb_round_trip_proptest(red in any::<u8>(), green in any::<u8>(), blue in any::<u8>()) {
            use super::super::xyz_to_srgb::XyzToSRgb;
            use viprs_core::colour::ColourConvert;

            let converter = SRgbToXyz;
            let inverse = XyzToSRgb;
            let input_data = [red, green, blue];
            let region = make_region(1);
            let input = Tile::new(region, 3, &input_data);

            let mut xyz_data = [0.0_f32; 3];
            let mut xyz_output = TileMut::new(region, 3, &mut xyz_data);
            converter.convert_region(&mut (), &input, &mut xyz_output);

            let xyz_input = Tile::new(region, 3, &xyz_data);
            let mut srgb_data = [0_u8; 3];
            let mut srgb_output = TileMut::new(region, 3, &mut srgb_data);
            inverse.convert_region(&mut (), &xyz_input, &mut srgb_output);

            prop_assert!((srgb_data[0] as i32 - red as i32).unsigned_abs() <= 1);
            prop_assert!((srgb_data[1] as i32 - green as i32).unsigned_abs() <= 1);
            prop_assert!((srgb_data[2] as i32 - blue as i32).unsigned_abs() <= 1);
        }
    }
}
