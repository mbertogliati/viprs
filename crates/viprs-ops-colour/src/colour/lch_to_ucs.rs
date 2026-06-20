use viprs_core::{
    colorspace::{Lch, Ucs},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use crate::colour::math::{c_to_ucs, ch_to_hucs, l_to_ucs};

/// Applies the `lch to ucs` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::lch_to_ucs::LchToUcs;
///
/// let op = LchToUcs;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LchToUcs;

#[inline(always)]
fn lch_f32_to_ucs_f32(lightness: f32, chroma: f32, hue: f32) -> (f32, f32, f32) {
    (
        l_to_ucs(lightness),
        c_to_ucs(chroma),
        ch_to_hucs(chroma, hue),
    )
}

impl ColourConvert<Lch, Ucs> for LchToUcs {
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
            let (l_ucs, c_ucs, h_ucs) = lch_f32_to_ucs_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = l_ucs;
            pixel_out[1] = c_ucs;
            pixel_out[2] = h_ucs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::ucs_to_lch::UcsToLch;
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn zero_chroma_preserves_zero_hue() {
        let converter = LchToUcs;
        let input_data = [50.0_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!(output_data[1].abs() < 1e-5);
        assert_eq!(output_data[2], 0.0);
    }

    #[test]
    fn ucs_inverse_preserves_non_boundary_hue() {
        let forward = LchToUcs;
        let inverse = UcsToLch;
        let input_data = [45.0_f32, 60.0, 120.0];
        let region = make_region(1);

        let input = Tile::new(region, 3, &input_data);
        let mut ucs_data = [0.0_f32; 3];
        let mut ucs_tile = TileMut::new(region, 3, &mut ucs_data);
        forward.convert_region(&mut (), &input, &mut ucs_tile);

        let ucs_input = Tile::new(region, 3, &ucs_data);
        let mut roundtrip_data = [0.0_f32; 3];
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
        inverse.convert_region(&mut (), &ucs_input, &mut roundtrip_tile);

        let hue_delta = (roundtrip_data[2] - input_data[2])
            .rem_euclid(360.0)
            .min((input_data[2] - roundtrip_data[2]).rem_euclid(360.0));
        assert!(hue_delta < 5.0, "hue delta={}", hue_delta);
    }

    proptest! {
        #[test]
        fn lch_ucs_round_trip_proptest(
            lightness in 0.0f32..100.0,
            chroma in 0.0f32..120.0,
            hue in 0.0f32..360.0,
        ) {
            let forward = LchToUcs;
            let inverse = UcsToLch;
            let region = make_region(1);
            let input_data = [lightness, chroma, hue];
            let mut ucs_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut ucs_tile = TileMut::new(region, 3, &mut ucs_data);
            forward.convert_region(&mut (), &input, &mut ucs_tile);

            let mut roundtrip_data = [0.0f32; 3];
            let ucs_input = Tile::new(region, 3, &ucs_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &ucs_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - lightness).abs() < 0.25);
            prop_assert!((roundtrip_data[1] - chroma).abs() < 0.25);
        }
    }
}
