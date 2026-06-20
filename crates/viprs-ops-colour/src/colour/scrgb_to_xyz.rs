use viprs_core::{
    colorspace::{ScRgb, Xyz},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::scrgb_to_xyz_components;

/// Applies the `scrgb to xyz` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::scrgb_to_xyz::ScRgbToXyz;
///
/// let op = ScRgbToXyz;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ScRgbToXyz;

impl ColourConvert<ScRgb, Xyz> for ScRgbToXyz {
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
            let (x, y, z) = scrgb_to_xyz_components(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = x;
            pixel_out[1] = y;
            pixel_out[2] = z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::xyz_to_scrgb::XyzToScRgb;
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
        fn scrgb_xyz_scrgb_round_trip_proptest(
            red in 0.0f32..=2.5,
            green in 0.0f32..=2.5,
            blue in 0.0f32..=2.5
        ) {
            let forward = ScRgbToXyz;
            let inverse = XyzToScRgb;
            let region = make_region(1);
            let input_data = [red, green, blue];
            let input = Tile::new(region, 3, &input_data);

            let mut xyz_data = [0.0f32; 3];
            let mut xyz_tile = TileMut::new(region, 3, &mut xyz_data);
            forward.convert_region(&mut (), &input, &mut xyz_tile);

            let xyz_input = Tile::new(region, 3, &xyz_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &xyz_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - red).abs() <= 5e-4);
            prop_assert!((roundtrip_data[1] - green).abs() <= 5e-4);
            prop_assert!((roundtrip_data[2] - blue).abs() <= 5e-4);
        }
    }

    #[test]
    fn unit_white_maps_to_d65_white() {
        let converter = ScRgbToXyz;
        let input_data = [1.0_f32, 1.0, 1.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 0.9505).abs() < 5e-4);
        assert!((output_data[1] - 1.0).abs() < 5e-4);
        assert!((output_data[2] - 1.0890).abs() < 5e-4);
    }
}
