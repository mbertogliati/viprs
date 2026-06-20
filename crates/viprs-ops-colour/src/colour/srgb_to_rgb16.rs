use viprs_core::{
    colorspace::{Rgb16, SRgb},
    colour::ColourConvert,
    format::{U8, U16},
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `srgb to rgb16` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::srgb_to_rgb16::SRgbToRgb16;
///
/// let op = SRgbToRgb16;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct SRgbToRgb16;

#[inline]
fn process_tile(input: &Tile<U8>, output: &mut TileMut<U16>) {
    for (pixel_in, pixel_out) in input
        .data
        .chunks_exact(3)
        .zip(output.data.chunks_exact_mut(3))
    {
        pixel_out[0] = u16::from(pixel_in[0]) * 257;
        pixel_out[1] = u16::from(pixel_in[1]) * 257;
        pixel_out[2] = u16::from(pixel_in[2]) * 257;
    }
}

impl ColourConvert<SRgb, Rgb16> for SRgbToRgb16 {
    type InputFormat = U8;
    type OutputFormat = U16;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U16>) {
        process_tile(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::{Region, Tile, TileMut};
    use viprs_core::op::DemandHint;

    #[test]
    fn srgb_upshifts_into_rgb16() {
        let converter = SRgbToRgb16;
        let input_data = [0_u8, 1, 255];
        let mut output_data = [0_u16; 3];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 257, 65535]);
    }

    #[test]
    fn metadata_methods_match_colour_contract() {
        let converter = SRgbToRgb16;
        let region = Region::new(-2, 3, 4, 1);
        assert_eq!(converter.demand_hint(), DemandHint::Any);
        assert_eq!(converter.required_input_region(&region), region);
        converter.start();
    }

    #[test]
    fn multiple_pixels_upsample_every_channel() {
        let converter = SRgbToRgb16;
        let input_data = [0_u8, 17, 255, 1, 2, 3];
        let mut output_data = [0_u16; 6];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 4369, 65535, 257, 514, 771]);
    }
}
