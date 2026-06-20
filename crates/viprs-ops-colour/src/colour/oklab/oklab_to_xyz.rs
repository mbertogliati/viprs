use viprs_core::{
    colorspace::{Oklab, Xyz},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `oklab to xyz` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::oklab::oklab_to_xyz::OklabToXyz;
///
/// let op = OklabToXyz;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct OklabToXyz;

#[inline(always)]
fn oklab_f32_to_xyz_f32(lightness: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let lp = 0.215_803_76f32.mul_add(b, 0.396_337_78f32.mul_add(a, lightness));
    let mp = 0.063_854_17f32.mul_add(-b, 0.105_561_346f32.mul_add(-a, lightness));
    let sp = 1.291_485_5f32.mul_add(-b, 0.089_484_18f32.mul_add(-a, lightness));

    let l = lp * lp * lp;
    let m = mp * mp * mp;
    let s = sp * sp * sp;

    (
        0.281_256_14f32.mul_add(s, 0.557_799_94f32.mul_add(-m, 1.227_013_8 * l)),
        0.071_676_68f32.mul_add(-s, 1.112_256_9f32.mul_add(m, -0.040_580_18 * l)),
        1.586_163_3f32.mul_add(s, 0.421_481_97f32.mul_add(-m, -0.076_381_28 * l)),
    )
}

impl ColourConvert<Oklab, Xyz> for OklabToXyz {
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
            let (x, y, z) = oklab_f32_to_xyz_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = x;
            pixel_out[1] = y;
            pixel_out[2] = z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::oklab::XyzToOklab;
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn neutral_oklab_maps_to_d65_white() {
        let converter = OklabToXyz;
        let input_data = [1.0_f32, 0.0, 0.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!(
            (output_data[0] - 0.950_47).abs() < 2e-4,
            "X={}",
            output_data[0]
        );
        assert!((output_data[1] - 1.0).abs() < 2e-4, "Y={}", output_data[1]);
        assert!(
            (output_data[2] - 1.088_83).abs() < 1e-3,
            "Z={}",
            output_data[2]
        );
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(OklabToXyz.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(5, 6, 7, 8);
        assert_eq!(OklabToXyz.required_input_region(&region), region);
    }

    proptest! {
        #[test]
        fn oklab_xyz_round_trip_proptest(
            x in 0.0f32..1.0,
            y in 0.0f32..1.0,
            z in 0.0f32..1.0,
        ) {
            let to_oklab = XyzToOklab;
            let to_xyz = OklabToXyz;
            let region = make_region(1);
            let xyz_data = [x, y, z];
            let xyz_input = Tile::new(region, 3, &xyz_data);
            let mut oklab_data = [0.0f32; 3];
            let mut oklab_tile = TileMut::new(region, 3, &mut oklab_data);
            to_oklab.convert_region(&mut (), &xyz_input, &mut oklab_tile);

            let oklab_input = Tile::new(region, 3, &oklab_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            to_xyz.convert_region(&mut (), &oklab_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - x).abs() < 5e-5);
            prop_assert!((roundtrip_data[1] - y).abs() < 5e-5);
            prop_assert!((roundtrip_data[2] - z).abs() < 5e-5);
        }
    }
}
