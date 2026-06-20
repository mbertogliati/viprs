use viprs_core::{
    colorspace::{Rgb16, SRgb},
    colour::ColourConvert,
    format::{U8, U16},
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `rgb16 to srgb` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::rgb16_to_srgb::Rgb16ToSRgb;
///
/// let op = Rgb16ToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct Rgb16ToSRgb;

#[inline]
fn process_tile(input: &Tile<U16>, output: &mut TileMut<U8>) {
    for (pixel_in, pixel_out) in input
        .data
        .chunks_exact(3)
        .zip(output.data.chunks_exact_mut(3))
    {
        pixel_out[0] = (pixel_in[0] >> 8) as u8;
        pixel_out[1] = (pixel_in[1] >> 8) as u8;
        pixel_out[2] = (pixel_in[2] >> 8) as u8;
    }
}

impl ColourConvert<Rgb16, SRgb> for Rgb16ToSRgb {
    type InputFormat = U16;
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
    fn convert_region(&self, (): &mut (), input: &Tile<U16>, output: &mut TileMut<U8>) {
        process_tile(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::{Region, Tile, TileMut};
    use viprs_core::op::DemandHint;

    #[test]
    fn rgb16_downshifts_into_srgb() {
        let converter = Rgb16ToSRgb;
        let input_data = [0_u16, 257, 65535];
        let mut output_data = [0_u8; 3];
        let region = Region::new(0, 0, 1, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 1, 255]);
    }

    #[test]
    fn metadata_methods_match_colour_contract() {
        let converter = Rgb16ToSRgb;
        let region = Region::new(2, -1, 4, 3);
        assert_eq!(converter.demand_hint(), DemandHint::Any);
        assert_eq!(converter.required_input_region(&region), region);
        converter.start();
    }

    #[test]
    fn multiple_rgb16_pixels_downshift_per_channel() {
        let converter = Rgb16ToSRgb;
        let input_data = [0x1200_u16, 0x34ff, 0xab00, 0x0101, 0x0202, 0x0303];
        let mut output_data = [0_u8; 6];
        let region = Region::new(0, 0, 2, 1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0x12, 0x34, 0xab, 0x01, 0x02, 0x03]);
    }
}
