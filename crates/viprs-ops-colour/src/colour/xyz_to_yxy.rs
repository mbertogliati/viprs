use viprs_core::{
    colorspace::{Xyz, Yxy},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `xyz to yxy` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::xyz_to_yxy::XyzToYxy;
///
/// let op = XyzToYxy;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct XyzToYxy;

#[inline(always)]
fn xyz_f32_to_yxy_f32(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let sum = x + y + z;
    if sum == 0.0 {
        (y, 0.0, 0.0)
    } else {
        (y, x / sum, y / sum)
    }
}

impl ColourConvert<Xyz, Yxy> for XyzToYxy {
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
            let (yy, x, y) = xyz_f32_to_yxy_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = yy;
            pixel_out[1] = x;
            pixel_out[2] = y;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::yxy_to_xyz::YxyToXyz;
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn d65_white_maps_to_expected_chromaticity() {
        let converter = XyzToYxy;
        let input_data = [0.950_47_f32, 1.0, 1.088_83];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!(
            (output_data[1] - 0.312_726_62).abs() < 1e-5,
            "x={}",
            output_data[1]
        );
        assert!(
            (output_data[2] - 0.329_023_15).abs() < 1e-5,
            "y={}",
            output_data[2]
        );
    }

    #[test]
    fn zero_xyz_maps_to_zero_chromaticity() {
        let converter = XyzToYxy;
        let input_data = [0.0_f32, 0.0, 0.0];
        let mut output_data = [1.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data, [0.0, 0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn xyz_yxy_round_trip_proptest(
            x in 0.0f32..1.0,
            y in 0.0f32..1.0,
            z in 0.0f32..1.0,
        ) {
            let forward = XyzToYxy;
            let inverse = YxyToXyz;
            let region = make_region(1);
            let input_data = [x, y, z];
            let mut yxy_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut yxy_tile = TileMut::new(region, 3, &mut yxy_data);
            forward.convert_region(&mut (), &input, &mut yxy_tile);

            let mut roundtrip_data = [0.0f32; 3];
            let yxy_input = Tile::new(region, 3, &yxy_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &yxy_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - x).abs() < 2e-5);
            prop_assert!((roundtrip_data[1] - y).abs() < 2e-5);
            prop_assert!((roundtrip_data[2] - z).abs() < 2e-5);
        }
    }
}
