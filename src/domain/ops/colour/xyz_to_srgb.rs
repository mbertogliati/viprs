#![allow(clippy::tuple_array_conversions)]
// REASON: the tuple destructuring mirrors the small fixed-size matrix math used by libvips.

use crate::{
    domain::colour::ColourConvert,
    domain::{
        colorspace::{SRgb, Xyz},
        format::{F32, U8},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::math::{scrgb_to_srgb_u8_components, xyz_to_scrgb_components};

/// Convert one CIE XYZ D65 pixel (f32 × 3) to sRGB (u8 × 3).
#[inline(always)]
fn xyz_f32_to_srgb_u8(x: f32, y: f32, z: f32) -> (u8, u8, u8) {
    let (red, green, blue) = xyz_to_scrgb_components(x, y, z);
    let [red, green, blue] = scrgb_to_srgb_u8_components(red, green, blue);
    (red, green, blue)
}

/// Applies the `xyz to srgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::xyz_to_srgb::XyzToSRgb;
///
/// let op = XyzToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct XyzToSRgb;

impl ColourConvert<Xyz, SRgb> for XyzToSRgb {
    type InputFormat = F32;
    type OutputFormat = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<U8>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (r, g, b) = xyz_f32_to_srgb_u8(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = r;
            pixel_out[1] = g;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn red_xyz_to_srgb() {
        let converter = XyzToSRgb;
        // XYZ of sRGB red D65
        let input_data: [f32; 3] = [0.412_456_4, 0.212_672_9, 0.019_333_9];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data[0], 255);
        assert!(output_data[1] <= 1);
        assert!(output_data[2] <= 1);
    }

    #[test]
    fn white_xyz_to_srgb() {
        let converter = XyzToSRgb;
        let input_data: [f32; 3] = [0.950_47, 1.0, 1.088_83];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert!(output_data[0] >= 254);
        assert!(output_data[1] >= 254);
        assert!(output_data[2] >= 254);
    }

    /// Round-trip sRGB → XYZ → sRGB must be identity within ±1 u8.
    #[test]
    fn round_trip_srgb_xyz_srgb() {
        use super::super::srgb_to_xyz::SRgbToXyz;
        let fwd = SRgbToXyz;
        let inv = XyzToSRgb;

        let originals: &[[u8; 3]] = &[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [255, 255, 255],
            [0, 0, 0],
            [100, 149, 237],
        ];

        for orig in originals {
            let region = Region::new(0, 0, 1, 1);
            let mut xyz_buf = [0.0_f32; 3];
            let input = Tile::new(region, 3, orig.as_slice());
            let mut xyz_tile = TileMut::new(region, 3, &mut xyz_buf);
            fwd.convert_region(&mut (), &input, &mut xyz_tile);

            let mut srgb_buf = [0_u8; 3];
            let xyz_in = Tile::new(region, 3, &xyz_buf);
            let mut srgb_out = TileMut::new(region, 3, &mut srgb_buf);
            inv.convert_region(&mut (), &xyz_in, &mut srgb_out);

            let diff_r = (srgb_buf[0] as i32 - orig[0] as i32).unsigned_abs();
            let diff_g = (srgb_buf[1] as i32 - orig[1] as i32).unsigned_abs();
            let diff_b = (srgb_buf[2] as i32 - orig[2] as i32).unsigned_abs();
            assert!(
                diff_r <= 1,
                "R mismatch: {} vs {} for {:?}",
                srgb_buf[0],
                orig[0],
                orig
            );
            assert!(
                diff_g <= 1,
                "G mismatch: {} vs {} for {:?}",
                srgb_buf[1],
                orig[1],
                orig
            );
            assert!(
                diff_b <= 1,
                "B mismatch: {} vs {} for {:?}",
                srgb_buf[2],
                orig[2],
                orig
            );
        }
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(XyzToSRgb.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(XyzToSRgb.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        // Covers the `fn start(&self) {}` line.
        XyzToSRgb.start();
    }

    /// Very small XYZ input exercises the linear branch of `srgb_gamma_encode`
    /// (x <= 0.003_130_8 → 12.92 * x), which applies to near-black colors.
    #[test]
    fn near_black_xyz_linear_gamma_branch() {
        let converter = XyzToSRgb;
        // XYZ(0.001, 0.001, 0.001): r_lin ≈ 3.24*0.001 - 1.54*0.001 - 0.50*0.001 ≈ 0.0012
        // which is < 0.003_130_8, so srgb_gamma_encode uses the linear branch.
        let input_data: [f32; 3] = [0.001, 0.001, 0.001];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // Near-black: output should be very small but >= 0
        assert!(output_data[0] <= 10);
        assert!(output_data[1] <= 10);
        assert!(output_data[2] <= 10);
    }
}
