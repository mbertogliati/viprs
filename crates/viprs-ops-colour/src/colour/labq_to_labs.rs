use viprs_core::{
    format::{I16, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `labq to labs` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::labq_to_labs::LabQToLabS;
///
/// let op = LabQToLabS;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabQToLabS;

#[inline(always)]
fn labq_bytes_to_labs_i16(lightness: u8, a: u8, b: u8, ext: u8) -> (i16, i16, i16) {
    let lightness = (i32::from(lightness) << 7) | (i32::from(ext & 0xc0) >> 1);
    let a = (i32::from(i8::from_ne_bytes([a])) << 8) | (i32::from(ext & 0x38) << 2);
    let b = (i32::from(i8::from_ne_bytes([b])) << 8) | (i32::from(ext & 0x7) << 5);

    (lightness as i16, a as i16, b as i16)
}

impl Op for LabQToLabS {
    type Input = U8;
    type Output = I16;
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
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<I16>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(4)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (lightness, a, b) =
                labq_bytes_to_labs_i16(pixel_in[0], pixel_in[1], pixel_in[2], pixel_in[3]);
            pixel_out[0] = lightness;
            pixel_out[1] = a;
            pixel_out[2] = b;
        }
    }
}

impl PixelLocalOp for LabQToLabS {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::labs_to_labq::LabSToLabQ;
    use proptest::prelude::*;
    use viprs_core::{
        image::{Image, Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[inline(always)]
    fn quantized_lightness(lightness: i16) -> i16 {
        ((i32::from(lightness) + 16).clamp(0, i32::from(i16::MAX)) >> 5) as i16 * 32
    }

    #[inline(always)]
    fn quantized_chroma(chroma: i16) -> i16 {
        let chroma = if chroma >= 0 {
            i32::from(chroma) + 16
        } else {
            i32::from(chroma) - 16
        }
        .clamp(i32::from(i16::MIN), i32::from(i16::MAX));

        ((chroma >> 5) << 5) as i16
    }

    fn decode_labq(input_data: Vec<u8>) -> Vec<i16> {
        let pixels = input_data.len() / 4;
        let input_image = Image::<U8>::from_buffer(pixels as u32, 1, 4, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 4, input_image.pixels());
        let mut output_data = vec![0_i16; pixels * 3];
        let mut output = TileMut::new(region, 3, &mut output_data);

        LabQToLabS.process_region(&mut (), &input, &mut output);

        output_data
    }

    fn encode_labs(input_data: Vec<i16>) -> Vec<u8> {
        let pixels = input_data.len() / 3;
        let input_image = Image::<I16>::from_buffer(pixels as u32, 1, 3, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = vec![0_u8; pixels * 4];
        let mut output = TileMut::new(region, 4, &mut output_data);

        LabSToLabQ.process_region(&mut (), &input, &mut output);

        output_data
    }

    proptest! {
        #[test]
        fn encoded_labs_decode_to_quantized_samples(
            lightness in 0i16..=i16::MAX,
            a in any::<i16>(),
            b in any::<i16>(),
        ) {
            let decoded = decode_labq(encode_labs(vec![lightness, a, b]));

            prop_assert_eq!(decoded, vec![
                quantized_lightness(lightness),
                quantized_chroma(a),
                quantized_chroma(b),
            ]);
        }
    }

    #[test]
    fn packed_extrema_decode_to_expected_labs_values() {
        let op = LabQToLabS;
        let input_data = [0_u8, 128, 128, 0, 255, 127, 127, 255];
        let mut output_data = [0_i16; 6];
        let region = make_region(2);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(&output_data[..3], &[0, i16::MIN, i16::MIN]);
        assert_eq!(&output_data[3..], &[32_736, 32_736, 32_736]);
    }

    #[test]
    fn labs_labq_labs_round_trip_recovers_quantized_extrema() {
        let forward = LabSToLabQ;
        let inverse = LabQToLabS;
        let input_data = [0_i16, i16::MIN, i16::MAX];
        let region = make_region(1);

        let input = Tile::new(region, 3, &input_data);
        let mut labq_data = [0_u8; 4];
        let mut labq_tile = TileMut::new(region, 4, &mut labq_data);
        forward.process_region(&mut (), &input, &mut labq_tile);

        let labq_input = Tile::new(region, 4, &labq_data);
        let mut roundtrip_data = [0_i16; 3];
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
        inverse.process_region(&mut (), &labq_input, &mut roundtrip_tile);

        assert_eq!(roundtrip_data, [0, i16::MIN, 32_736]);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabQToLabS.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(1, 2, 3, 4);
        assert_eq!(LabQToLabS.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabQToLabS.start();
    }

    #[test]
    fn operation_bridge_forces_three_output_bands() {
        let bridge = OperationBridge::new_pixel_local(LabQToLabS, 4);
        assert_eq!(bridge.bands, 3);
    }
}
