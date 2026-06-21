use viprs_core::{
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `lab to labq` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::lab_to_labq::LabToLabQ;
///
/// let op = LabToLabQ;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabToLabQ;

#[inline(always)]
fn lab_f32_to_labq_bytes(lightness: f32, a: f32, b: f32) -> [u8; 4] {
    let lightness = (10.23 * lightness).round_ties_even().clamp(0.0, 1023.0) as i32;
    let mut lsbs = ((lightness & 0x3) << 6) as u8;
    let l_high = (lightness >> 2) as u8;

    let a_val = (8.0 * a).round_ties_even().clamp(-1024.0, 1023.0) as i32;
    lsbs |= ((a_val & 0x7) << 3) as u8;
    let a_high = (a_val >> 3) as i8 as u8;

    let b_val = (8.0 * b).round_ties_even().clamp(-1024.0, 1023.0) as i32;
    lsbs |= (b_val & 0x7) as u8;
    let b_high = (b_val >> 3) as i8 as u8;

    [l_high, a_high, b_high, lsbs]
}

impl Op for LabToLabQ {
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
            pixel_out.copy_from_slice(&lab_f32_to_labq_bytes(
                pixel_in[0],
                pixel_in[1],
                pixel_in[2],
            ));
        }
    }
}

impl PixelLocalOp for LabToLabQ {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::labq_to_lab::LabQToLab;
    use proptest::prelude::*;
    use viprs_core::{
        image::{Image, Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn encode_lab(input_data: Vec<f32>) -> Vec<u8> {
        let pixels = input_data.len() / 3;
        let input_image = Image::<F32>::from_buffer(pixels as u32, 1, 3, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = vec![0_u8; pixels * 4];
        let mut output = TileMut::new(region, 4, &mut output_data);

        LabToLabQ.process_region(&mut (), &input, &mut output);

        output_data
    }

    fn decode_labq(input_data: Vec<u8>) -> Vec<f32> {
        let pixels = input_data.len() / 4;
        let input_image = Image::<U8>::from_buffer(pixels as u32, 1, 4, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 4, input_image.pixels());
        let mut output_data = vec![0.0_f32; pixels * 3];
        let mut output = TileMut::new(region, 3, &mut output_data);

        LabQToLab.process_region(&mut (), &input, &mut output);

        output_data
    }

    proptest! {
        #[test]
        fn quantized_lab_samples_round_trip_exactly(
            lightness_code in 0i32..=1023,
            a_code in -1024i32..=1023,
            b_code in -1024i32..=1023,
        ) {
            let input_data = vec![
                lightness_code as f32 * (100.0 / 1023.0),
                a_code as f32 * 0.125,
                b_code as f32 * 0.125,
            ];

            let roundtrip_data = decode_labq(encode_lab(input_data.clone()));

            prop_assert!((roundtrip_data[0] - input_data[0]).abs() < 1e-5);
            prop_assert_eq!(roundtrip_data[1], input_data[1]);
            prop_assert_eq!(roundtrip_data[2], input_data[2]);
        }
    }

    #[test]
    fn round_trip_stays_within_labq_quantization_error() {
        let forward = LabToLabQ;
        let inverse = LabQToLab;
        let input_data = [63.5_f32, -12.25, 80.625];
        let region = make_region(1);

        let input = Tile::new(region, 3, &input_data);
        let mut labq_data = [0_u8; 4];
        let mut labq_tile = TileMut::new(region, 4, &mut labq_data);
        forward.process_region(&mut (), &input, &mut labq_tile);

        let labq_input = Tile::new(region, 4, &labq_data);
        let mut roundtrip_data = [0.0_f32; 3];
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
        inverse.process_region(&mut (), &labq_input, &mut roundtrip_tile);

        assert!((roundtrip_data[0] - input_data[0]).abs() < 0.15);
        assert!((roundtrip_data[1] - input_data[1]).abs() <= 0.125);
        assert!((roundtrip_data[2] - input_data[2]).abs() <= 0.125);
    }

    #[test]
    fn ties_round_to_even_and_extrema_clamp() {
        let output_data = encode_lab(vec![2.5 / 10.23, 0.3125, 0.3125, 200.0, -200.0, 200.0]);

        assert_eq!(&output_data[..4], &[0, 0, 0, 146]);
        assert_eq!(&output_data[4..], &[255, 128, 127, 199]);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabToLabQ.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(4, 6, 8, 10);
        assert_eq!(LabToLabQ.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabToLabQ.start();
    }

    #[test]
    fn operation_bridge_forces_four_output_bands() {
        let bridge = OperationBridge::new_pixel_local(LabToLabQ, 3);
        assert_eq!(bridge.bands, 4);
    }
}
