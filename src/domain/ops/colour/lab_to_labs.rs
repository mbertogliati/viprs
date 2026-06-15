use crate::domain::{
    format::{F32, I16},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
    ops::resample::sample_conv::FromF64,
};

/// Applies the `lab to labs` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::lab_to_labs::LabToLabS;
///
/// let op = LabToLabS;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabToLabS;

#[inline(always)]
fn lab_f32_to_labs_i16(lightness: f32, a: f32, b: f32) -> (i16, i16, i16) {
    let lightness = (lightness * (32_767.0 / 100.0)).clamp(0.0, f32::from(i16::MAX));
    let a = (a * (32_768.0 / 128.0)).clamp(f32::from(i16::MIN), f32::from(i16::MAX));
    let b = (b * (32_768.0 / 128.0)).clamp(f32::from(i16::MIN), f32::from(i16::MAX));
    (lightness as i16, a as i16, b as i16)
}

impl Op for LabToLabS {
    type Input = F32;
    type Output = I16;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<I16>) {
        let bands = input.bands as usize;
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(bands)
            .zip(output.data.chunks_exact_mut(bands))
        {
            let (lightness, a, b) = lab_f32_to_labs_i16(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lightness;
            pixel_out[1] = a;
            pixel_out[2] = b;
            for band in 3..bands {
                pixel_out[band] = i16::from_f64(f64::from(pixel_in[band]));
            }
        }
    }
}

impl PixelLocalOp for LabToLabS {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        image::{Region, Tile, TileMut},
        ops::colour::labs_to_lab::LabSToLab,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn round_trip_stays_within_quantization_error() {
        let forward = LabToLabS;
        let inverse = LabSToLab;
        let input_data = [42.5_f32, -12.25, 63.5];
        let region = make_region(1);

        let input = Tile::new(region, 3, &input_data);
        let mut labs_data = [0_i16; 3];
        let mut labs_tile = TileMut::new(region, 3, &mut labs_data);
        forward.process_region(&mut (), &input, &mut labs_tile);

        let labs_input = Tile::new(region, 3, &labs_data);
        let mut roundtrip_data = [0.0_f32; 3];
        let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
        inverse.process_region(&mut (), &labs_input, &mut roundtrip_tile);

        assert!((roundtrip_data[0] - input_data[0]).abs() < 0.01);
        assert!((roundtrip_data[1] - input_data[1]).abs() < 0.01);
        assert!((roundtrip_data[2] - input_data[2]).abs() < 0.01);
    }

    #[test]
    fn clips_out_of_range_values() {
        let forward = LabToLabS;
        let input_data = [120.0_f32, -300.0, 300.0];
        let mut output_data = [0_i16; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        forward.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data[0], i16::MAX);
        assert_eq!(output_data[1], i16::MIN);
        assert_eq!(output_data[2], i16::MAX);
    }

    #[test]
    fn preserves_extra_bands() {
        let op = LabToLabS;
        let input_data = [50.0_f32, 10.0, -20.0, 200.0];
        let mut output_data = [0_i16; 4];
        let region = make_region(1);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        op.process_region(&mut (), &input, &mut output);

        assert_eq!(output_data[3], 200);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabToLabS.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(5, 7, 11, 13);
        assert_eq!(LabToLabS.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabToLabS.start();
    }
}
