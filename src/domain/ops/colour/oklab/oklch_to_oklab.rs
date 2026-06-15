use crate::{
    domain::colour::ColourConvert,
    domain::{
        colorspace::{Oklab, Oklch},
        format::F32,
        image::{DemandHint, Region, Tile, TileMut},
    },
};

use crate::domain::ops::colour::math::chroma_hue_to_ab;

/// Applies the `oklch to oklab` colour transform to image pixels. Use it when a pipeline needs
/// to move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::ops::colour::oklab::oklch_to_oklab::OklchToOklab;
///
/// let op = OklchToOklab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct OklchToOklab;

#[inline(always)]
fn oklch_f32_to_oklab_f32(lightness: f32, chroma: f32, hue: f32) -> (f32, f32, f32) {
    let (a, b) = chroma_hue_to_ab(chroma, hue);
    (lightness, a, b)
}

impl ColourConvert<Oklch, Oklab> for OklchToOklab {
    type InputFormat = F32;
    type OutputFormat = F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (lightness, a, b) = oklch_f32_to_oklab_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = lightness;
            pixel_out[1] = a;
            pixel_out[2] = b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::image::{Region, Tile, TileMut};

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    #[test]
    fn ninety_degree_hue_maps_to_positive_b() {
        let converter = OklchToOklab;
        let input_data = [0.6_f32, 0.2, 90.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 0.6).abs() < 1e-6);
        assert!(output_data[1].abs() < 1e-4);
        assert!((output_data[2] - 0.2).abs() < 1e-4);
    }
}
