use viprs_core::{
    colorspace::{Oklab, Xyz},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `xyz to oklab` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::oklab::xyz_to_oklab::XyzToOklab;
///
/// let op = XyzToOklab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct XyzToOklab;

#[inline(always)]
fn xyz_f32_to_oklab_f32(x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let l = 0.128_859_71f32.mul_add(-z, 0.361_866_74f32.mul_add(y, 0.818_933 * x));
    let m = 0.036_145_64f32.mul_add(z, 0.929_311_9f32.mul_add(y, 0.032_984_544 * x));
    let s = 0.633_851_7f32.mul_add(z, 0.264_366_27f32.mul_add(y, 0.048_200_3 * x));

    let lp = l.cbrt();
    let mp = m.cbrt();
    let sp = s.cbrt();

    (
        0.004_072_047f32.mul_add(-sp, 0.793_617_8f32.mul_add(mp, 0.210_454_26 * lp)),
        0.450_593_7f32.mul_add(sp, 2.428_592_2f32.mul_add(-mp, 1.977_998_5 * lp)),
        0.808_675_77f32.mul_add(-sp, 0.782_771_77f32.mul_add(mp, 0.025_904_037 * lp)),
    )
}

impl ColourConvert<Xyz, Oklab> for XyzToOklab {
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
            let (l, a, b) = xyz_f32_to_oklab_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = l;
            pixel_out[1] = a;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::oklab::OklabToXyz;
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn d65_white_maps_to_neutral_oklab() {
        let converter = XyzToOklab;
        let input_data = [0.950_47_f32, 1.0, 1.088_83];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 1.0).abs() < 2e-3, "L={}", output_data[0]);
        assert!(output_data[1].abs() < 3e-4, "a={}", output_data[1]);
        assert!(output_data[2].abs() < 3e-4, "b={}", output_data[2]);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(XyzToOklab.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(3, 4, 5, 6);
        assert_eq!(XyzToOklab.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn xyz_oklab_round_trip_proptest(
            x in 0.0f32..1.0,
            y in 0.0f32..1.0,
            z in 0.0f32..1.0,
        ) {
            let forward = XyzToOklab;
            let inverse = OklabToXyz;
            let region = make_region(1);
            let input_data = [x, y, z];
            let mut oklab_data = [0.0f32; 3];
            let input = Tile::new(region, 3, &input_data);
            let mut oklab_tile = TileMut::new(region, 3, &mut oklab_data);
            forward.convert_region(&mut (), &input, &mut oklab_tile);

            let mut roundtrip_data = [0.0f32; 3];
            let oklab_input = Tile::new(region, 3, &oklab_data);
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &oklab_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - x).abs() < 5e-5);
            prop_assert!((roundtrip_data[1] - y).abs() < 5e-5);
            prop_assert!((roundtrip_data[2] - z).abs() < 5e-5);
        }
    }
}
