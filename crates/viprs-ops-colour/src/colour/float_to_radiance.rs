use viprs_core::{
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

const RADIANCE_EXP_BIAS: i32 = 128;

/// Applies the `float to radiance` colour transform to image pixels. Use it when a pipeline
/// needs to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::float_to_radiance::FloatToRadiance;
///
/// let op = FloatToRadiance;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct FloatToRadiance;

#[inline(always)]
fn float_to_radiance_components(red: f32, green: f32, blue: f32) -> [u8; 4] {
    let max_component = red.max(green).max(blue);
    if max_component <= 1e-32 {
        return [0, 0, 0, 0];
    }

    let exponent = max_component.log2().floor() as i32 + 1;
    let scale = 255.9999_f32 * 2.0_f32.powi(-exponent);

    [
        if red > 0.0 { (red * scale) as u8 } else { 0 },
        if green > 0.0 {
            (green * scale) as u8
        } else {
            0
        },
        if blue > 0.0 { (blue * scale) as u8 } else { 0 },
        (exponent + RADIANCE_EXP_BIAS) as u8,
    ]
}

impl Op for FloatToRadiance {
    type Input = F32;
    type Output = U8;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(4);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<U8>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(4))
        {
            pixel_out.copy_from_slice(&float_to_radiance_components(
                pixel_in[0],
                pixel_in[1],
                pixel_in[2],
            ));
        }
    }
}

impl PixelLocalOp for FloatToRadiance {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::radiance_to_float::RadianceToFloat;
    use proptest::prelude::*;
    use viprs_core::{
        image::{Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    proptest! {
        #[test]
        fn float_radiance_float_reencodes_exactly(
            red in 0.0f32..=4096.0,
            green in 0.0f32..=4096.0,
            blue in 0.0f32..=4096.0
        ) {
            let forward = FloatToRadiance;
            let inverse = RadianceToFloat;
            let region = make_region(1);
            let input_data = [red, green, blue];
            let input = Tile::new(region, 3, &input_data);

            let mut radiance_data = [0u8; 4];
            let mut radiance_tile = TileMut::new(region, 4, &mut radiance_data);
            forward.process_region(&mut (), &input, &mut radiance_tile);

            let radiance_input = Tile::new(region, 4, &radiance_data);
            let mut float_data = [0.0f32; 3];
            let mut float_tile = TileMut::new(region, 3, &mut float_data);
            inverse.process_region(&mut (), &radiance_input, &mut float_tile);

            let float_input = Tile::new(region, 3, &float_data);
            let mut reencoded = [0u8; 4];
            let mut reencoded_tile = TileMut::new(region, 4, &mut reencoded);
            forward.process_region(&mut (), &float_input, &mut reencoded_tile);

            prop_assert_eq!(reencoded, radiance_data);
        }
    }

    #[test]
    fn zero_rgb_maps_to_zero_radiance() {
        let op = FloatToRadiance;
        let input_data = [0.0_f32, 0.0, 0.0];
        let mut output_data = [255_u8; 4];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 0, 0]);
    }

    #[test]
    fn operation_bridge_forces_four_output_bands() {
        let bridge = OperationBridge::new_pixel_local(FloatToRadiance, 3);
        assert_eq!(bridge.bands, 4);
    }

    #[test]
    fn mixed_sign_rgb_clamps_negative_channels_to_zero() {
        let op = FloatToRadiance;
        let input_data = [-1.0_f32, 2.0, 0.0];
        let mut output_data = [255_u8; 4];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 127, 0, 130]);
    }

    #[test]
    fn metadata_helpers_match_pixel_local_contract() {
        let op = FloatToRadiance;
        let region = Region::new(3, -2, 2, 4);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }
}
