use viprs_core::{
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

const RADIANCE_EXP_BIAS: i32 = 128;

/// Applies the `radiance to float` colour transform to image pixels. Use it when a pipeline
/// needs to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::radiance_to_float::RadianceToFloat;
///
/// let op = RadianceToFloat;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct RadianceToFloat;

#[inline(always)]
fn radiance_to_float_components(red: u8, green: u8, blue: u8, exponent: u8) -> [f32; 3] {
    if exponent == 0 {
        return [0.0, 0.0, 0.0];
    }

    let factor = 2.0_f32.powi(i32::from(exponent) - (RADIANCE_EXP_BIAS + 8));
    [
        (f32::from(red) + 0.5) * factor,
        (f32::from(green) + 0.5) * factor,
        (f32::from(blue) + 0.5) * factor,
    ]
}

impl Op for RadianceToFloat {
    type Input = U8;
    type Output = F32;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(3);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(4)
            .zip(output.data.chunks_exact_mut(3))
        {
            let decoded =
                radiance_to_float_components(pixel_in[0], pixel_in[1], pixel_in[2], pixel_in[3]);
            pixel_out[0] = decoded[0];
            pixel_out[1] = decoded[1];
            pixel_out[2] = decoded[2];
        }
    }
}

impl PixelLocalOp for RadianceToFloat {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::float_to_radiance::FloatToRadiance;
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
        fn radiance_float_radiance_round_trip_proptest(
            red in 0.0f32..=4096.0,
            green in 0.0f32..=4096.0,
            blue in 0.0f32..=4096.0
        ) {
            let forward = RadianceToFloat;
            let inverse = FloatToRadiance;
            let region = make_region(1);
            let float_input_data = [red, green, blue];
            let float_input = Tile::new(region, 3, &float_input_data);

            let mut radiance_data = [0u8; 4];
            let mut radiance_tile = TileMut::new(region, 4, &mut radiance_data);
            inverse.process_region(&mut (), &float_input, &mut radiance_tile);

            let mut float_data = [0.0f32; 3];
            let mut float_tile = TileMut::new(region, 3, &mut float_data);
            let radiance_input = Tile::new(region, 4, &radiance_data);
            forward.process_region(&mut (), &radiance_input, &mut float_tile);

            let float_input = Tile::new(region, 3, &float_data);
            let mut roundtrip_data = [0u8; 4];
            let mut roundtrip_tile = TileMut::new(region, 4, &mut roundtrip_data);
            inverse.process_region(&mut (), &float_input, &mut roundtrip_tile);

            prop_assert_eq!(roundtrip_data, radiance_data);
        }
    }

    #[test]
    fn zero_exponent_decodes_to_black() {
        let op = RadianceToFloat;
        let input_data = [255_u8, 255, 255, 0];
        let mut output_data = [1.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn operation_bridge_forces_three_output_bands() {
        let bridge = OperationBridge::new_pixel_local(RadianceToFloat, 4);
        assert_eq!(bridge.bands, 3);
    }

    #[test]
    fn nonzero_exponent_uses_half_lsb_bias() {
        let decoded = radiance_to_float_components(1, 2, 3, 129);
        assert_eq!(decoded, [0.01171875, 0.01953125, 0.02734375]);
    }

    #[test]
    fn metadata_helpers_match_pixel_local_contract() {
        let op = RadianceToFloat;
        let region = Region::new(-4, 1, 3, 2);
        assert_eq!(op.demand_hint(), DemandHint::Any);
        assert_eq!(op.required_input_region(&region), region);
        op.start();
    }
}
