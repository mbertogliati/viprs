use viprs_core::{
    colorspace::{SRgb, ScRgb},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::srgb_gamma_decode;

/// Applies the `sRGB to scRGB` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::srgb_to_scrgb::SRgbToScRgb;
///
/// let op = SRgbToScRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbToScRgb;

#[inline(always)]
fn srgb_u8_to_scrgb_f32(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    (
        srgb_gamma_decode(f32::from(r) / 255.0),
        srgb_gamma_decode(f32::from(g) / 255.0),
        srgb_gamma_decode(f32::from(b) / 255.0),
    )
}

impl ColourConvert<SRgb, ScRgb> for SRgbToScRgb {
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
            let (r, g, b) = srgb_u8_to_scrgb_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = r;
            pixel_out[1] = g;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::scrgb_to_srgb::ScRgbToSRgb;
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
        fn srgb_scrgb_round_trip_proptest(r in any::<u8>(), g in any::<u8>(), b in any::<u8>()) {
            let forward = SRgbToScRgb;
            let inverse = ScRgbToSRgb;
            let region = make_region(1);

            let input_data = [r, g, b];
            let mut scrgb_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut scrgb_tile = TileMut::new(region, 3, &mut scrgb_data);
            forward.convert_region(&mut (), &input, &mut scrgb_tile);

            let mut roundtrip_data = [0u8; 3];
            let scrgb_input = Tile::new(region, 3, &scrgb_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &scrgb_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] as i32 - r as i32).abs() <= 1);
            prop_assert!((roundtrip_data[1] as i32 - g as i32).abs() <= 1);
            prop_assert!((roundtrip_data[2] as i32 - b as i32).abs() <= 1);
        }
    }

    #[test]
    fn white_maps_to_unit_linear() {
        let converter = SRgbToScRgb;
        let input_data = [255_u8, 255, 255];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 1e-6);
        assert!((output_data[1] - 1.0).abs() < 1e-6);
        assert!((output_data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn low_u8_values_use_linear_decode_branch() {
        let converter = SRgbToScRgb;
        let input_data = [10_u8, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        let expected = (10.0 / 255.0) / 12.92;
        assert!((output_data[0] - expected).abs() < 1e-6);
        assert_eq!(output_data[1], 0.0);
        assert_eq!(output_data[2], 0.0);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(SRgbToScRgb.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(SRgbToScRgb.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        SRgbToScRgb.start();
    }
}
