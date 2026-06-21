use viprs_core::{
    colorspace::{Lab, SRgb},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::{
    scrgb_to_xyz_components, srgb_decode_table, xyz_to_lab_components_with_table, xyz_to_lab_table,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmulq_n_f32, vst1q_f32};

/// Applies the `srgb to lab` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::srgb_to_lab::SRgbToLab;
///
/// let op = SRgbToLab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbToLab;

impl ColourConvert<SRgb, Lab> for SRgbToLab {
    type InputFormat = U8;
    type OutputFormat = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {
        let _ = srgb_decode_table();
        let _ = xyz_to_lab_table();
    }

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<F32>) {
        let decode = srgb_decode_table();
        let lab_table = xyz_to_lab_table();
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: aarch64 targets always provide NEON, and the helper only touches the
            // input/output slice bounds derived from the tile lengths.
            unsafe { convert_region_neon(decode, lab_table, input.data, output.data) };
        }

        #[cfg(not(target_arch = "aarch64"))]
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let red = decode[pixel_in[0] as usize];
            let green = decode[pixel_in[1] as usize];
            let blue = decode[pixel_in[2] as usize];

            let (x, y, z) = scrgb_to_xyz_components(red, green, blue);
            let (l, a, b) = xyz_to_lab_components_with_table(lab_table, x, y, z);

            pixel_out[0] = l;
            pixel_out[1] = a;
            pixel_out[2] = b;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_region_neon(
    decode: &[f32; 256],
    lab_table: &[f32; 100_000],
    input: &[u8],
    output: &mut [f32],
) {
    let pixel_count = input.len() / 3;
    let simd_pixels = pixel_count / 4;
    let simd_input_len = simd_pixels * 12;
    let simd_output_len = simd_pixels * 12;

    let mut input_offset = 0;
    let mut output_offset = 0;

    for _ in 0..simd_pixels {
        let mut red_lanes = [0.0_f32; 4];
        let mut green_lanes = [0.0_f32; 4];
        let mut blue_lanes = [0.0_f32; 4];

        for lane in 0..4 {
            let pixel_offset = input_offset + lane * 3;
            red_lanes[lane] = decode[input[pixel_offset] as usize];
            green_lanes[lane] = decode[input[pixel_offset + 1] as usize];
            blue_lanes[lane] = decode[input[pixel_offset + 2] as usize];
        }

        let mut x_lanes = [0.0_f32; 4];
        let mut y_lanes = [0.0_f32; 4];
        let mut z_lanes = [0.0_f32; 4];

        // SAFETY: all NEON loads/stores operate on 4-lane stack arrays and on output arrays
        // sized for exactly four f32 values.
        unsafe {
            let red = vld1q_f32(red_lanes.as_ptr());
            let green = vld1q_f32(green_lanes.as_ptr());
            let blue = vld1q_f32(blue_lanes.as_ptr());

            let x = vaddq_f32(
                vmulq_n_f32(red, 0.4124),
                vaddq_f32(vmulq_n_f32(green, 0.3576), vmulq_n_f32(blue, 0.1805)),
            );
            let y = vaddq_f32(
                vmulq_n_f32(red, 0.2126),
                vaddq_f32(vmulq_n_f32(green, 0.7152), vmulq_n_f32(blue, 0.0722)),
            );
            let z = vaddq_f32(
                vmulq_n_f32(red, 0.0193),
                vaddq_f32(vmulq_n_f32(green, 0.1192), vmulq_n_f32(blue, 0.9505)),
            );

            vst1q_f32(x_lanes.as_mut_ptr(), x);
            vst1q_f32(y_lanes.as_mut_ptr(), y);
            vst1q_f32(z_lanes.as_mut_ptr(), z);
        }

        for lane in 0..4 {
            let (lightness, a, b) = xyz_to_lab_components_with_table(
                lab_table,
                x_lanes[lane],
                y_lanes[lane],
                z_lanes[lane],
            );
            let dst = &mut output[output_offset + lane * 3..output_offset + lane * 3 + 3];
            dst[0] = lightness;
            dst[1] = a;
            dst[2] = b;
        }

        input_offset += 12;
        output_offset += 12;
    }

    for (pixel_in, pixel_out) in input[simd_input_len..]
        .chunks_exact(3)
        .zip(output[simd_output_len..].chunks_exact_mut(3))
    {
        let (x, y, z) = scrgb_to_xyz_components(
            decode[pixel_in[0] as usize],
            decode[pixel_in[1] as usize],
            decode[pixel_in[2] as usize],
        );
        let (lightness, a, b) = xyz_to_lab_components_with_table(lab_table, x, y, z);
        pixel_out[0] = lightness;
        pixel_out[1] = a;
        pixel_out[2] = b;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::math::{scrgb_to_xyz_components, xyz_to_lab_components};
    use viprs_core::image::{Region, Tile, TileMut};

    const LAB_TOLERANCE_TIGHT: f32 = 0.1;
    const LAB_TOLERANCE_RED_REFERENCE: f32 = 0.01;

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn convert_single_pixel(rgb: [u8; 3]) -> [f32; 3] {
        let converter = SRgbToLab;
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &rgb);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        output_data
    }

    fn assert_lab_close(actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
        assert!(
            (actual[0] - expected[0]).abs() < tolerance,
            "L={} expected {} ±{}",
            actual[0],
            expected[0],
            tolerance
        );
        assert!(
            (actual[1] - expected[1]).abs() < tolerance,
            "a={} expected {} ±{}",
            actual[1],
            expected[1],
            tolerance
        );
        assert!(
            (actual[2] - expected[2]).abs() < tolerance,
            "b={} expected {} ±{}",
            actual[2],
            expected[2],
            tolerance
        );
    }

    fn srgb_u8_to_lab_f32_gamma_22(r: u8, g: u8, b: u8) -> [f32; 3] {
        let red = (r as f32 / 255.0).powf(2.2);
        let green = (g as f32 / 255.0).powf(2.2);
        let blue = (b as f32 / 255.0).powf(2.2);
        let (x, y, z) = scrgb_to_xyz_components(red, green, blue);
        let (l, a, b) = xyz_to_lab_components(x, y, z);
        [l, a, b]
    }

    #[test]
    fn white_to_lab() {
        // libvips getpoint(colourspace(srgb(255,255,255),lab)) reference
        let output = convert_single_pixel([255, 255, 255]);
        assert_lab_close(
            output,
            [100.0, 0.005_245_208_7, -0.010_609_627],
            LAB_TOLERANCE_TIGHT,
        );
    }

    #[test]
    fn black_to_lab() {
        // libvips getpoint(colourspace(srgb(0,0,0),lab)) reference
        let output = convert_single_pixel([0, 0, 0]);
        assert_lab_close(output, [-5.960_464_5e-08, 0.0, 0.0], LAB_TOLERANCE_TIGHT);
    }

    #[test]
    fn red_to_lab() {
        // libvips getpoint(colourspace(srgb(255,0,0),lab)) reference
        let output = convert_single_pixel([255, 0, 0]);
        assert_lab_close(
            output,
            [53.232_883, 80.109_33, 67.220_024],
            LAB_TOLERANCE_RED_REFERENCE,
        );
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(SRgbToLab.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(SRgbToLab.required_input_region(&r), r);
    }

    /// Ported from libvips test_colour.py::test_colourspace.
    ///
    /// libvips test: mid-grey Lab(50,0,0) converted from sRGB(118,118,118).
    /// Checks that L* is close to 50.0 and a*, b* are near 0.
    /// Reference: mid-grey in sRGB ≈ (118,118,118) → Lab ≈ (50,0,0).
    #[test]
    fn mid_grey_srgb_to_lab_matches_libvips_reference() {
        // libvips getpoint(colourspace(srgb(118,118,118),lab)) reference
        let output = convert_single_pixel([118, 118, 118]);
        assert_lab_close(
            output,
            [49.637_012, 0.002_980_232_2, -0.005_996_227_3],
            LAB_TOLERANCE_TIGHT,
        );
    }

    #[test]
    fn non_primary_reference_detects_gamma_22_regression() {
        // libvips getpoint(colourspace(srgb(64,32,192),lab)) reference
        let expected = [29.232_079, 56.875_214, -76.697_23];
        let output = convert_single_pixel([64, 32, 192]);
        assert_lab_close(output, expected, LAB_TOLERANCE_TIGHT);

        // A 2.2 power-law approximation would drift well beyond tight bounds.
        let wrong_gamma = srgb_u8_to_lab_f32_gamma_22(64, 32, 192);
        let max_error = wrong_gamma
            .iter()
            .zip(expected.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_error > LAB_TOLERANCE_TIGHT,
            "2.2 gamma approximation drift ({max_error}) must exceed tolerance ({LAB_TOLERANCE_TIGHT})"
        );
    }

    /// Ported from libvips test_colour.py::test_colourspace.
    ///
    /// libvips test: Lab(50,0,0) → XYZ ≈ [17.5064, 18.4187, 20.0547].
    /// Checked against http://www.brucelindbloom.com
    ///
    /// This test verifies that `SRgbToLab` produces a Lab output that, when
    /// forwarded through `LabToXyz`, lands within the expected XYZ neighbourhood
    /// of sRGB(118,118,118). We verify L* proximity to 50 and then a loose XYZ
    /// check to confirm the full pipeline direction is correct.
    #[test]
    fn srgb_mid_grey_lab_to_xyz_is_in_expected_range() {
        use super::super::lab_to_xyz::LabToXyz;
        use viprs_core::colour::ColourConvert;

        let to_lab = SRgbToLab;
        let to_xyz = LabToXyz;
        // sRGB(118,118,118) is approximately L*≈47-49 in CIE Lab (D65).
        let input_data: [u8; 3] = [118, 118, 118];
        let region = make_region(1);

        let mut lab_buf = [0.0f32; 3];
        let srgb_in = Tile::new(region, 3, &input_data);
        let mut lab_out = TileMut::new(region, 3, &mut lab_buf);
        to_lab.convert_region(&mut (), &srgb_in, &mut lab_out);

        // L* must be in the 45-52 range for sRGB mid-grey
        assert!(
            lab_buf[0] > 45.0 && lab_buf[0] < 52.0,
            "L*={} for sRGB(118,118,118) is out of expected range [45,52]",
            lab_buf[0]
        );

        let mut xyz_buf = [0.0f32; 3];
        let lab_in = Tile::new(region, 3, &lab_buf);
        let mut xyz_out = TileMut::new(region, 3, &mut xyz_buf);
        to_xyz.convert_region(&mut (), &lab_in, &mut xyz_out);

        // For near-mid-grey XYZ must be in the 0.15–0.22 range per channel
        assert!(xyz_buf[0] > 0.15 && xyz_buf[0] < 0.22, "X={}", xyz_buf[0]);
        assert!(xyz_buf[1] > 0.15 && xyz_buf[1] < 0.22, "Y={}", xyz_buf[1]);
        assert!(xyz_buf[2] > 0.15 && xyz_buf[2] < 0.24, "Z={}", xyz_buf[2]);
    }

    /// Ported from libvips test_colour.py::test_colourspace.
    ///
    /// libvips test: convert a 2-pixel strip with different colours and check
    /// that each pixel is transformed independently.
    #[test]
    fn two_pixel_strip_transforms_independently() {
        let converter = SRgbToLab;
        // Pixel 0: white, Pixel 1: black
        let input_data: [u8; 6] = [255, 255, 255, 0, 0, 0];
        let mut output_data = [0.0_f32; 6];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // Pixel 0: white → L* ≈ 100
        assert!(
            (output_data[0] - 100.0).abs() < 0.5,
            "pixel 0 L*={} expected ≈100",
            output_data[0]
        );
        // Pixel 1: black → L* ≈ 0
        assert!(
            output_data[3].abs() < 0.5,
            "pixel 1 L*={} expected ≈0",
            output_data[3]
        );
    }

    #[test]
    fn five_pixel_strip_matches_single_pixel_reference() {
        let converter = SRgbToLab;
        let input_data: [u8; 15] = [
            255, 255, 255, 255, 0, 0, 64, 32, 192, 118, 118, 118, 0, 0, 0,
        ];
        let mut output_data = [0.0_f32; 15];
        let region = Region::new(0, 0, 5, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        for (index, rgb) in input_data.chunks_exact(3).enumerate() {
            let expected = convert_single_pixel([rgb[0], rgb[1], rgb[2]]);
            let actual = [
                output_data[index * 3],
                output_data[index * 3 + 1],
                output_data[index * 3 + 2],
            ];
            assert_lab_close(actual, expected, 1e-6);
        }
    }
}
