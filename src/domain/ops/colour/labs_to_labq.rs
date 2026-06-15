use crate::domain::{
    format::{I16, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `labs to labq` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::labs_to_labq::LabSToLabQ;
///
/// let op = LabSToLabQ;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabSToLabQ;

#[inline(always)]
fn round_labs_lightness_to_labq(lightness: i16) -> i32 {
    (i32::from(lightness) + 16).clamp(0, i32::from(i16::MAX)) >> 5
}

#[inline(always)]
fn round_labs_chroma_to_labq(chroma: i16) -> i32 {
    let chroma = if chroma >= 0 {
        i32::from(chroma) + 16
    } else {
        i32::from(chroma) - 16
    };

    chroma.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) >> 5
}

#[inline(always)]
fn labs_i16_to_labq_bytes(lightness: i16, a: i16, b: i16) -> [u8; 4] {
    let lightness = round_labs_lightness_to_labq(lightness);
    let a = round_labs_chroma_to_labq(a);
    let b = round_labs_chroma_to_labq(b);

    [
        (lightness >> 2) as u8,
        (a >> 3) as i8 as u8,
        (b >> 3) as i8 as u8,
        (((lightness << 6) & 0xc0) | ((a << 3) & 0x38) | (b & 0x7)) as u8,
    ]
}

impl Op for LabSToLabQ {
    type Input = I16;
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
    fn process_region(&self, (): &mut (), input: &Tile<I16>, output: &mut TileMut<U8>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(4))
        {
            pixel_out.copy_from_slice(&labs_i16_to_labq_bytes(
                pixel_in[0],
                pixel_in[1],
                pixel_in[2],
            ));
        }
    }
}

impl PixelLocalOp for LabSToLabQ {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        image::{Region, Tile, TileMut},
        op::OperationBridge,
        ops::colour::labq_to_labs::LabQToLabS,
    };
    use proptest::prelude::*;

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[inline(always)]
    fn quantized_lightness(lightness: i16) -> i16 {
        (round_labs_lightness_to_labq(lightness) << 5) as i16
    }

    #[inline(always)]
    fn quantized_chroma(chroma: i16) -> i16 {
        (round_labs_chroma_to_labq(chroma) << 5) as i16
    }

    proptest! {
        #[test]
        fn labs_labq_labs_round_trip_matches_quantized_samples(
            lightness in 0i16..=i16::MAX,
            a in any::<i16>(),
            b in any::<i16>(),
        ) {
            let forward = LabSToLabQ;
            let inverse = LabQToLabS;
            let region = make_region(1);
            let input_data = [lightness, a, b];

            let input = Tile::new(region, 3, &input_data);
            let mut labq_data = [0_u8; 4];
            let mut labq_tile = TileMut::new(region, 4, &mut labq_data);
            forward.process_region(&mut (), &input, &mut labq_tile);

            let labq_input = Tile::new(region, 4, &labq_data);
            let mut roundtrip_data = [0_i16; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.process_region(&mut (), &labq_input, &mut roundtrip_tile);

            prop_assert_eq!(roundtrip_data[0], quantized_lightness(lightness));
            prop_assert_eq!(roundtrip_data[1], quantized_chroma(a));
            prop_assert_eq!(roundtrip_data[2], quantized_chroma(b));
        }
    }

    #[test]
    fn extrema_pack_to_expected_labq_bytes() {
        let op = LabSToLabQ;
        let input_data = [0_i16, i16::MIN, i16::MIN, i16::MAX, i16::MAX, i16::MAX];
        let mut output_data = [0_u8; 8];
        let region = make_region(2);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(&output_data[..4], &[0, 128, 128, 0]);
        assert_eq!(&output_data[4..], &[255, 127, 127, 255]);
    }

    #[test]
    fn half_step_rounding_moves_away_from_zero_for_labs_quantization() {
        let input_image =
            crate::domain::image::Image::<I16>::from_buffer(1, 1, 3, vec![16, 16, -16]).unwrap();
        let region = make_region(1);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = [0_u8; 4];
        let mut output = TileMut::new(region, 4, &mut output_data);

        LabSToLabQ.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 255, 79]);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabSToLabQ.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(8, 6, 4, 2);
        assert_eq!(LabSToLabQ.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabSToLabQ.start();
    }

    #[test]
    fn operation_bridge_forces_four_output_bands() {
        let bridge = OperationBridge::new_pixel_local(LabSToLabQ, 3);
        assert_eq!(bridge.bands, 4);
    }
}
