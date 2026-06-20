use viprs_core::{
    colorspace::{Lab, Lch},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::ab_to_hue_degrees;

/// Applies the `lab to lch` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::lab_to_lch::LabToLch;
///
/// let op = LabToLch;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabToLch;

#[inline(always)]
fn lab_f32_to_lch_f32(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    (l, a.hypot(b), ab_to_hue_degrees(a, b))
}

impl ColourConvert<Lab, Lch> for LabToLch {
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
            let (l, chroma, hue) = lab_f32_to_lch_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = l;
            pixel_out[1] = chroma;
            pixel_out[2] = hue;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::lch_to_lab::LchToLab;
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
        fn lab_lch_round_trip_proptest(
            l in 0.0f32..100.0,
            a in -128.0f32..128.0,
            b in -128.0f32..128.0,
        ) {
            let forward = LabToLch;
            let inverse = LchToLab;
            let region = make_region(1);

            let input_data = [l, a, b];
            let mut lch_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut lch_tile = TileMut::new(region, 3, &mut lch_data);
            forward.convert_region(&mut (), &input, &mut lch_tile);

            let mut roundtrip_data = [0.0f32; 3];
            let lch_input = Tile::new(region, 3, &lch_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &lch_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - l).abs() < 1e-3);
            prop_assert!((roundtrip_data[1] - a).abs() < 1e-3);
            prop_assert!((roundtrip_data[2] - b).abs() < 1e-3);
        }
    }

    #[test]
    fn zero_chroma_maps_to_zero_hue() {
        let converter = LabToLch;
        let input_data = [42.0_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [42.0, 0.0, 0.0]);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabToLch.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(LabToLch.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        LabToLch.start();
    }
}
