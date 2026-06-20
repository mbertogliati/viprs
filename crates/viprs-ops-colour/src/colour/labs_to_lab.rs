use viprs_core::{
    format::{F32, I16},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    shared_ops::sample_conv::ToF64,
};

/// Applies the `labs to lab` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::labs_to_lab::LabSToLab;
///
/// let op = LabSToLab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabSToLab;

#[inline(always)]
fn labs_i16_to_lab_f32(lightness: i16, a: i16, b: i16) -> (f32, f32, f32) {
    (
        f32::from(lightness) / (32_767.0 / 100.0),
        f32::from(a) / (32_768.0 / 128.0),
        f32::from(b) / (32_768.0 / 128.0),
    )
}

impl Op for LabSToLab {
    type Input = I16;
    type Output = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<I16>, output: &mut TileMut<F32>) {
        let bands = input.bands as usize;
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(bands)
            .zip(output.data.chunks_exact_mut(bands))
        {
            let (lightness, a, b) = labs_i16_to_lab_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lightness;
            pixel_out[1] = a;
            pixel_out[2] = b;
            for band in 3..bands {
                pixel_out[band] = pixel_in[band].to_f64() as f32;
            }
        }
    }
}

impl PixelLocalOp for LabSToLab {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::lab_to_labs::LabToLabS;
    use proptest::prelude::*;
    use viprs_core::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn zero_values_decode_to_zero_lab() {
        let op = LabSToLab;
        let input_data = [0_i16, 0, 0];
        let mut output_data = [1.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0.0, 0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn labs_lab_labs_round_trip_preserves_quantized_samples(
            lightness in 0i16..=i16::MAX,
            a in any::<i16>(),
            b in any::<i16>(),
        ) {
            let forward = LabSToLab;
            let inverse = LabToLabS;
            let region = make_region(1);
            let input_data = [lightness, a, b];

            let input = Tile::new(region, 3, &input_data);
            let mut lab_data = [0.0_f32; 3];
            let mut lab_tile = TileMut::new(region, 3, &mut lab_data);
            forward.process_region(&mut (), &input, &mut lab_tile);

            let lab_input = Tile::new(region, 3, &lab_data);
            let mut roundtrip_data = [0_i16; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.process_region(&mut (), &lab_input, &mut roundtrip_tile);

            prop_assert!((i32::from(roundtrip_data[0]) - i32::from(lightness)).abs() <= 1);
            prop_assert!((i32::from(roundtrip_data[1]) - i32::from(a)).abs() <= 1);
            prop_assert!((i32::from(roundtrip_data[2]) - i32::from(b)).abs() <= 1);
        }
    }

    #[test]
    fn extrema_decode_to_expected_lab_ranges() {
        let op = LabSToLab;
        let input_data = [i16::MAX, i16::MIN, i16::MAX];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 100.0).abs() < 1e-4);
        assert!((output_data[1] + 128.0).abs() < 1e-6);
        assert!((output_data[2] - 127.99609).abs() < 1e-4);
    }

    #[test]
    fn preserves_extra_bands() {
        let op = LabSToLab;
        let input_data = [i16::MAX, i16::MIN, i16::MAX, 200];
        let mut output_data = [0.0_f32; 4];
        let region = make_region(1);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data[3], 200.0);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabSToLab.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(5, 7, 11, 13);
        assert_eq!(LabSToLab.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabSToLab.start();
    }
}
