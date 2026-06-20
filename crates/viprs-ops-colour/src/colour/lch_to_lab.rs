use viprs_core::{
    colorspace::{Lab, Lch},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

use super::math::chroma_hue_to_ab;

/// Applies the `lch to lab` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::lch_to_lab::LchToLab;
///
/// let op = LchToLab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LchToLab;

#[inline(always)]
fn lch_f32_to_lab_f32(l: f32, chroma: f32, hue: f32) -> (f32, f32, f32) {
    let (a, b) = chroma_hue_to_ab(chroma, hue);
    (l, a, b)
}

impl ColourConvert<Lch, Lab> for LchToLab {
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
            let (l, a, b) = lch_f32_to_lab_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = l;
            pixel_out[1] = a;
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
    fn ninety_degree_hue_maps_to_positive_b() {
        let converter = LchToLab;
        let input_data = [50.0_f32, 40.0, 90.0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 50.0).abs() < 1e-6);
        assert!(output_data[1].abs() < 1e-4);
        assert!((output_data[2] - 40.0).abs() < 1e-4);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LchToLab.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let r = Region::new(10, 20, 30, 40);
        assert_eq!(LchToLab.required_input_region(&r), r);
    }

    #[test]
    fn start_returns_unit() {
        LchToLab.start();
    }
}
