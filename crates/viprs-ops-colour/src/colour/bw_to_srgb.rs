use viprs_core::{
    colorspace::{Greyscale, SRgb},
    colour::ColourConvert,
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `grayscale to sRGB` colour transform to image pixels. Use it when a pipeline
/// needs to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::bw_to_srgb::BwToSRgb;
///
/// let op = BwToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct BwToSRgb;

#[inline]
fn process_tile(input: &Tile<U8>, output: &mut TileMut<U8>) {
    for (sample, pixel_out) in input.data.iter().zip(output.data.chunks_exact_mut(3)) {
        pixel_out[0] = *sample;
        pixel_out[1] = *sample;
        pixel_out[2] = *sample;
    }
}

impl ColourConvert<Greyscale, SRgb> for BwToSRgb {
    type InputFormat = U8;
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
    fn convert_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        process_tile(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use viprs_core::image::{Region, Tile, TileMut};
    use viprs_core::op::DemandHint;

    #[test]
    fn grey_pixels_expand_to_triplets() {
        let converter = BwToSRgb;
        let input_data = [0_u8, 64, 255];
        let mut output_data = [0_u8; 9];
        let region = Region::new(0, 0, 3, 1);
        let input = Tile::new(region, 1, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 0, 64, 64, 64, 255, 255, 255]);
    }

    #[test]
    fn metadata_methods_match_pixel_local_identity_contract() {
        let converter = BwToSRgb;
        let region = Region::new(-3, 4, 2, 5);
        assert_eq!(converter.demand_hint(), DemandHint::Any);
        assert_eq!(converter.required_input_region(&region), region);
        converter.start();
    }

    #[test]
    fn multiple_rows_keep_pixels_interleaved() {
        let converter = BwToSRgb;
        let input_data = [1_u8, 2, 3, 4];
        let mut output_data = [0_u8; 12];
        let region = Region::new(0, 0, 2, 2);
        let input = Tile::new(region, 1, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]);
    }
}
