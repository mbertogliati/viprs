use viprs_core::{
    colorspace::{ScRgb, Xyz},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::xyz_to_scrgb_components;

/// Applies the `xyz to scrgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::xyz_to_scrgb::XyzToScRgb;
///
/// let op = XyzToScRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct XyzToScRgb;

impl ColourConvert<Xyz, ScRgb> for XyzToScRgb {
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
            let (red, green, blue) = xyz_to_scrgb_components(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = red;
            pixel_out[1] = green;
            pixel_out[2] = blue;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::scrgb_to_xyz::ScRgbToXyz;
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
        fn xyz_scrgb_xyz_round_trip_proptest(
            x in 0.0f32..=2.5,
            y in 0.0f32..=2.5,
            z in 0.0f32..=2.5
        ) {
            let forward = XyzToScRgb;
            let inverse = ScRgbToXyz;
            let region = make_region(1);
            let input_data = [x, y, z];
            let input = Tile::new(region, 3, &input_data);

            let mut scrgb_data = [0.0f32; 3];
            let mut scrgb_tile = TileMut::new(region, 3, &mut scrgb_data);
            forward.convert_region(&mut (), &input, &mut scrgb_tile);

            let scrgb_input = Tile::new(region, 3, &scrgb_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &scrgb_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - x).abs() <= 1e-3);
            prop_assert!((roundtrip_data[1] - y).abs() <= 1e-3);
            prop_assert!((roundtrip_data[2] - z).abs() <= 1e-3);
        }
    }

    #[test]
    fn d65_white_maps_to_unit_scrgb() {
        let converter = XyzToScRgb;
        let input_data = [0.950_47_f32, 1.0, 1.088_83];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 5e-4);
        assert!((output_data[1] - 1.0).abs() < 5e-4);
        assert!((output_data[2] - 1.0).abs() < 5e-4);
    }
}
