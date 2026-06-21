use viprs_core::{
    colorspace::{Lab, Xyz},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::lab_to_xyz_components;

/// Applies the `lab to xyz` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::lab_to_xyz::LabToXyz;
///
/// let op = LabToXyz;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabToXyz;

impl ColourConvert<Lab, Xyz> for LabToXyz {
    type InputFormat = F32;
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
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (x, y, z) = lab_to_xyz_components(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = x;
            pixel_out[1] = y;
            pixel_out[2] = z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::{srgb_to_lab::SRgbToLab, xyz_to_lab::XyzToLab};
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    proptest! {
        #[test]
        fn lab_xyz_lab_round_trip_proptest(red in any::<u8>(), green in any::<u8>(), blue in any::<u8>()) {
            let to_lab = SRgbToLab;
            let forward = LabToXyz;
            let inverse = XyzToLab;
            let region = make_region(1);
            let srgb_data = [red, green, blue];
            let srgb_input = Tile::new(region, 3, &srgb_data);

            let mut lab_data = [0.0f32; 3];
            let mut lab_tile = TileMut::new(region, 3, &mut lab_data);
            to_lab.convert_region(&mut (), &srgb_input, &mut lab_tile);

            let input = Tile::new(region, 3, &lab_data);

            let mut xyz_data = [0.0f32; 3];
            let mut xyz_tile = TileMut::new(region, 3, &mut xyz_data);
            forward.convert_region(&mut (), &input, &mut xyz_tile);

            let xyz_input = Tile::new(region, 3, &xyz_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &xyz_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - lab_data[0]).abs() <= 0.02);
            prop_assert!((roundtrip_data[1] - lab_data[1]).abs() <= 0.05);
            prop_assert!((roundtrip_data[2] - lab_data[2]).abs() <= 0.05);
        }
    }

    #[test]
    fn neutral_white_lab_maps_to_d65_white() {
        let converter = LabToXyz;
        let input_data = [100.0_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 0.950_47).abs() < 1e-4);
        assert!((output_data[1] - 1.0).abs() < 1e-4);
        assert!((output_data[2] - 1.088_83).abs() < 1e-4);
    }

    /// Ported from libvips test_colour.py::test_colourspace.
    ///
    /// libvips test: Lab(50,0,0) → XYZ ≈ [17.5064, 18.4187, 20.0547].
    /// Checked against http://www.brucelindbloom.com for D65 illuminant.
    /// Values are absolute (not %-of-white-point) in viprs convention:
    /// X/Xn, Y/Yn, Z/Zn where Xn≈0.9505, Yn=1.0, Zn≈1.0890.
    #[test]
    fn mid_grey_lab_maps_to_brucelindbloom_xyz() {
        let converter = LabToXyz;
        // Lab(50, 0, 0): L*=50, achromatic mid-grey.
        let input_data = [50.0_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        // BruceLindbloom for D65: Lab(50,0,0) → XYZ = [17.5064, 18.4187, 20.0547] / 100
        assert!(
            (output_data[0] - 0.175_064).abs() < 0.002,
            "X={} expected ≈0.17506",
            output_data[0]
        );
        assert!(
            (output_data[1] - 0.184_187).abs() < 0.002,
            "Y={} expected ≈0.18419",
            output_data[1]
        );
        assert!(
            (output_data[2] - 0.200_547).abs() < 0.002,
            "Z={} expected ≈0.20055",
            output_data[2]
        );
    }

    /// Ported from libvips test_colour.py::test_colourspace.
    ///
    /// libvips test: achromatic input (a*=0, b*=0) must map to achromatic XYZ
    /// (X/Y and Z/Y must equal the D65 white-point ratios).
    /// Verifies that the operation does not introduce chromaticity for neutral greys.
    #[test]
    fn achromatic_lab_maps_to_achromatic_xyz() {
        let converter = LabToXyz;
        // Several neutral L* values from 10 to 90.
        let l_values = [10.0f32, 30.0, 50.0, 70.0, 90.0];
        for l in l_values {
            let input_data = [l, 0.0_f32, 0.0_f32];
            let mut output_data = [0.0_f32; 3];
            let region = make_region(1);
            let input = Tile::new(region, 3, &input_data);
            let mut output = TileMut::new(region, 3, &mut output_data);
            converter.convert_region(&mut (), &input, &mut output);

            let x = output_data[0];
            let y = output_data[1];
            let z = output_data[2];

            // For achromatic Lab, X/Y ≈ Xn and Z/Y ≈ Zn (D65 white-point ratios).
            // D65: Xn/Yn ≈ 0.9505, Zn/Yn ≈ 1.0890
            if y > 1e-4 {
                assert!(
                    ((x / y) - 0.9505).abs() < 0.005,
                    "L*={l}: X/Y={} expected ≈0.9505",
                    x / y
                );
                assert!(
                    ((z / y) - 1.0890).abs() < 0.005,
                    "L*={l}: Z/Y={} expected ≈1.0890",
                    z / y
                );
            }
        }
    }
}
