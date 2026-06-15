use crate::{
    domain::colour::ColourConvert,
    domain::{
        colorspace::{Lab, SRgb},
        format::{F32, U8},
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use super::math::{
    lab_to_xyz_components, scrgb_to_srgb_u8_component_with_table, scrgb_to_srgb_u8_table,
    xyz_to_scrgb_components,
};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vaddq_f32, vld1q_f32, vmulq_n_f32, vst1q_f32};

/// Applies the `lab to srgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::lab_to_srgb::LabToSRgb;
///
/// let op = LabToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabToSRgb;

impl ColourConvert<Lab, SRgb> for LabToSRgb {
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
        let encode = scrgb_to_srgb_u8_table();
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: aarch64 targets always provide NEON, and the helper only accesses
            // slice ranges derived from the tile lengths.
            unsafe { convert_region_neon(encode, input.data, output.data) };
        }

        #[cfg(not(target_arch = "aarch64"))]
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (x, y, z) = lab_to_xyz_components(pixel_in[0], pixel_in[1], pixel_in[2]);
            let (red, green, blue) = xyz_to_scrgb_components(x, y, z);
            pixel_out[0] = scrgb_to_srgb_u8_component_with_table(encode, red);
            pixel_out[1] = scrgb_to_srgb_u8_component_with_table(encode, green);
            pixel_out[2] = scrgb_to_srgb_u8_component_with_table(encode, blue);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn convert_region_neon(encode: &[u8; 257], input: &[f32], output: &mut [u8]) {
    let pixel_count = input.len() / 3;
    let simd_pixels = pixel_count / 4;
    let simd_len = simd_pixels * 12;

    let mut input_offset = 0;
    let mut output_offset = 0;

    for _ in 0..simd_pixels {
        let mut x_lanes = [0.0_f32; 4];
        let mut y_lanes = [0.0_f32; 4];
        let mut z_lanes = [0.0_f32; 4];

        for lane in 0..4 {
            let pixel_offset = input_offset + lane * 3;
            let (x, y, z) = lab_to_xyz_components(
                input[pixel_offset],
                input[pixel_offset + 1],
                input[pixel_offset + 2],
            );
            x_lanes[lane] = x;
            y_lanes[lane] = y;
            z_lanes[lane] = z;
        }

        let mut red_lanes = [0.0_f32; 4];
        let mut green_lanes = [0.0_f32; 4];
        let mut blue_lanes = [0.0_f32; 4];

        // SAFETY: all NEON loads/stores operate on fixed-size 4-lane stack arrays sized for
        // exactly one vector each.
        unsafe {
            let x = vld1q_f32(x_lanes.as_ptr());
            let y = vld1q_f32(y_lanes.as_ptr());
            let z = vld1q_f32(z_lanes.as_ptr());

            let red = vaddq_f32(
                vmulq_n_f32(x, 3.240_625),
                vaddq_f32(vmulq_n_f32(y, -1.537_208), vmulq_n_f32(z, -0.498_629)),
            );
            let green = vaddq_f32(
                vmulq_n_f32(x, -0.968_931),
                vaddq_f32(vmulq_n_f32(y, 1.875_756), vmulq_n_f32(z, 0.041_518)),
            );
            let blue = vaddq_f32(
                vmulq_n_f32(x, 0.055_71),
                vaddq_f32(vmulq_n_f32(y, -0.204_021), vmulq_n_f32(z, 1.056_996)),
            );

            vst1q_f32(red_lanes.as_mut_ptr(), red);
            vst1q_f32(green_lanes.as_mut_ptr(), green);
            vst1q_f32(blue_lanes.as_mut_ptr(), blue);
        }

        for lane in 0..4 {
            let dst = &mut output[output_offset + lane * 3..output_offset + lane * 3 + 3];
            dst[0] = scrgb_to_srgb_u8_component_with_table(encode, red_lanes[lane]);
            dst[1] = scrgb_to_srgb_u8_component_with_table(encode, green_lanes[lane]);
            dst[2] = scrgb_to_srgb_u8_component_with_table(encode, blue_lanes[lane]);
        }

        input_offset += 12;
        output_offset += 12;
    }

    for (pixel_in, pixel_out) in input[simd_len..]
        .chunks_exact(3)
        .zip(output[simd_len..].chunks_exact_mut(3))
    {
        let (x, y, z) = lab_to_xyz_components(pixel_in[0], pixel_in[1], pixel_in[2]);
        let (red, green, blue) = xyz_to_scrgb_components(x, y, z);
        pixel_out[0] = scrgb_to_srgb_u8_component_with_table(encode, red);
        pixel_out[1] = scrgb_to_srgb_u8_component_with_table(encode, green);
        pixel_out[2] = scrgb_to_srgb_u8_component_with_table(encode, blue);
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
    fn white_lab_to_srgb() {
        let converter = LabToSRgb;
        // Lab(100, 0, 0) → white
        let input_data: [f32; 3] = [100.0, 0.0, 0.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert!(output_data[0] >= 254);
        assert!(output_data[1] >= 254);
        assert!(output_data[2] >= 254);
    }

    #[test]
    fn black_lab_to_srgb() {
        let converter = LabToSRgb;
        // Lab(0, 0, 0) → black
        let input_data: [f32; 3] = [0.0, 0.0, 0.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert!(output_data[0] <= 1);
        assert!(output_data[1] <= 1);
        assert!(output_data[2] <= 1);
    }

    /// Round-trip sRGB → Lab → sRGB must be identity within ±1 u8.
    #[test]
    fn round_trip_srgb_lab_srgb() {
        use super::super::srgb_to_lab::SRgbToLab;
        let fwd = SRgbToLab;
        let inv = LabToSRgb;

        let originals: &[[u8; 3]] = &[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [255, 255, 255],
            [0, 0, 0],
            [200, 100, 50],
        ];

        for orig in originals {
            let region = Region::new(0, 0, 1, 1);
            let mut lab_buf = [0.0_f32; 3];
            let input = Tile::new(region, 3, orig.as_slice());
            let mut lab_tile = TileMut::new(region, 3, &mut lab_buf);
            fwd.convert_region(&mut (), &input, &mut lab_tile);

            let mut srgb_buf = [0_u8; 3];
            let lab_in = Tile::new(region, 3, &lab_buf);
            let mut srgb_out = TileMut::new(region, 3, &mut srgb_buf);
            inv.convert_region(&mut (), &lab_in, &mut srgb_out);

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
        assert_eq!(LabToSRgb.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(LabToSRgb.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        // Covers the `fn start(&self) {}` line.
        LabToSRgb.start();
    }

    /// Lab near-black with very small L exercises the linear branch of
    /// `lab_f_inv` (t <= 6/29) and the linear branch of `srgb_gamma_encode`
    /// (x <= 0.003_130_8). Both branches are needed for correct near-black output.
    #[test]
    fn near_black_lab_linear_branches() {
        let converter = LabToSRgb;
        // Lab(1.0, 0.0, 0.0): fy = (1+16)/116 = 0.147, which is < 6/29 ≈ 0.207
        // so lab_f_inv takes the linear branch.
        // The resulting XYZ values are tiny, so srgb_gamma_encode also takes
        // its linear branch (x <= 0.003_130_8).
        let input_data: [f32; 3] = [1.0, 0.0, 0.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        // Near-black: all channels should be close to 0
        assert!(output_data[0] <= 5);
        assert!(output_data[1] <= 5);
        assert!(output_data[2] <= 5);
    }

    #[test]
    fn extreme_lab_boundaries_roundtrip_without_nan() {
        use super::super::srgb_to_lab::SRgbToLab;
        use crate::domain::colour::ColourConvert;

        let to_srgb = LabToSRgb;
        let to_lab = SRgbToLab;

        for input_data in [
            [0.0_f32, -127.0, -127.0],
            [0.0, -127.0, 127.0],
            [0.0, 127.0, -127.0],
            [0.0, 127.0, 127.0],
            [100.0, -127.0, -127.0],
            [100.0, -127.0, 127.0],
            [100.0, 127.0, -127.0],
            [100.0, 127.0, 127.0],
        ] {
            let region = make_region(1);
            let lab_input = Tile::new(region, 3, &input_data);
            let mut srgb_data = [0_u8; 3];
            let mut srgb_tile = TileMut::new(region, 3, &mut srgb_data);
            to_srgb.convert_region(&mut (), &lab_input, &mut srgb_tile);

            let srgb_input = Tile::new(region, 3, &srgb_data);
            let mut roundtrip_lab = [0.0_f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_lab);
            to_lab.convert_region(&mut (), &srgb_input, &mut roundtrip_tile);

            assert!(
                roundtrip_lab.iter().all(|value| value.is_finite()),
                "roundtrip Lab must stay finite for input {input_data:?}: {roundtrip_lab:?}"
            );
            assert!(
                (0.0..=100.0).contains(&roundtrip_lab[0]),
                "roundtrip lightness out of range for input {input_data:?}: {}",
                roundtrip_lab[0]
            );
        }
    }

    #[test]
    fn five_pixel_strip_matches_single_pixel_reference() {
        let converter = LabToSRgb;
        let input_data: [f32; 15] = [
            100.0,
            0.0,
            0.0,
            53.232_883,
            80.109_33,
            67.220_024,
            49.637_012,
            0.002_980_232_2,
            -0.005_996_227_3,
            29.232_079,
            56.875_214,
            -76.697_23,
            0.0,
            0.0,
            0.0,
        ];
        let mut output_data = [0_u8; 15];
        let region = Region::new(0, 0, 5, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        for (index, lab) in input_data.chunks_exact(3).enumerate() {
            let mut expected_pixel = [0_u8; 3];
            let pixel_region = make_region(1);
            let pixel_input = Tile::new(pixel_region, 3, lab);
            let mut pixel_output = TileMut::new(pixel_region, 3, &mut expected_pixel);
            converter.convert_region(&mut (), &pixel_input, &mut pixel_output);
            assert_eq!(
                &output_data[index * 3..index * 3 + 3],
                expected_pixel.as_slice()
            );
        }
    }
}
