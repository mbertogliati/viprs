use crate::{
    domain::colour::ColourConvert,
    domain::{
        colorspace::{Oklab, Oklch},
        format::F32,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use crate::domain::ops::colour::math::ab_to_hue_degrees;

/// Applies the `oklab to oklch` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::oklab::oklab_to_oklch::OklabToOklch;
///
/// let op = OklabToOklch;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct OklabToOklch;

#[inline(always)]
fn oklab_f32_to_oklch_f32(lightness: f32, a: f32, b: f32) -> (f32, f32, f32) {
    (lightness, a.hypot(b), ab_to_hue_degrees(a, b))
}

impl ColourConvert<Oklab, Oklch> for OklabToOklch {
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
            let (lightness, chroma, hue) =
                oklab_f32_to_oklch_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lightness;
            pixel_out[1] = chroma;
            pixel_out[2] = hue;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
        ops::colour::oklab::OklchToOklab,
    };
    use proptest::prelude::*;

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn zero_chroma_maps_to_zero_hue() {
        let converter = OklabToOklch;
        let input_data = [0.7_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0.7, 0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn oklab_oklch_round_trip_proptest(
            lightness in 0.0f32..1.0,
            a in -0.4f32..0.4,
            b in -0.4f32..0.4,
        ) {
            let forward = OklabToOklch;
            let inverse = OklchToOklab;
            let region = make_region(1);
            let input_data = [lightness, a, b];
            let mut oklch_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut oklch_tile = TileMut::new(region, 3, &mut oklch_data);
            forward.convert_region(&mut (), &input, &mut oklch_tile);

            let mut roundtrip_data = [0.0f32; 3];
            let oklch_input = Tile::new(region, 3, &oklch_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &oklch_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - lightness).abs() < 1e-5);
            prop_assert!((roundtrip_data[1] - a).abs() < 1e-5);
            prop_assert!((roundtrip_data[2] - b).abs() < 1e-5);
        }
    }
}
