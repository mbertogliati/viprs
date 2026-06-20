use viprs_core::{
    colorspace::{SRgb, ScRgb},
    colour::ColourConvert,
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::srgb_gamma_encode;

/// Applies the `scRGB to sRGB` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::scrgb_to_srgb::ScRgbToSRgb;
///
/// let op = ScRgbToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ScRgbToSRgb;

#[inline(always)]
fn scrgb_f32_to_srgb_u8(r: f32, g: f32, b: f32) -> (u8, u8, u8) {
    let r = srgb_gamma_encode(r.clamp(0.0, 1.0));
    let g = srgb_gamma_encode(g.clamp(0.0, 1.0));
    let b = srgb_gamma_encode(b.clamp(0.0, 1.0));

    (
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        (b * 255.0).round() as u8,
    )
}

impl ColourConvert<ScRgb, SRgb> for ScRgbToSRgb {
    type InputFormat = F32;
    type OutputFormat = U8;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<U8>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (r, g, b) = scrgb_f32_to_srgb_u8(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = r;
            pixel_out[1] = g;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn unit_linear_maps_to_white() {
        let converter = ScRgbToSRgb;
        let input_data = [1.0_f32, 1.0, 1.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [255, 255, 255]);
    }

    #[test]
    fn out_of_range_values_are_clamped() {
        let converter = ScRgbToSRgb;
        let input_data = [-0.5_f32, 0.0, 1.5];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 255]);
    }

    #[test]
    fn near_black_uses_linear_encode_branch() {
        let converter = ScRgbToSRgb;
        let input_data = [0.001_f32, 0.0, 0.0];
        let mut output_data = [0_u8; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        let expected = (12.92_f32 * 0.001_f32 * 255.0_f32).round() as u8;
        assert_eq!(output_data[0], expected);
        assert_eq!(output_data[1], 0);
        assert_eq!(output_data[2], 0);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(ScRgbToSRgb.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(ScRgbToSRgb.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        ScRgbToSRgb.start();
    }
}
