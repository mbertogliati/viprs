use viprs_core::{
    colorspace::{Xyz, Yxy},
    colour::ColourConvert,
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
};

/// Applies the `yxy to xyz` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::yxy_to_xyz::YxyToXyz;
///
/// let op = YxyToXyz;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct YxyToXyz;

#[inline(always)]
fn yxy_f32_to_xyz_f32(y_luma: f32, x: f32, y: f32) -> (f32, f32, f32) {
    if x == 0.0 || y == 0.0 {
        (0.0, y_luma, 0.0)
    } else {
        let x_val = y_luma * x / y;
        let z_val = y_luma * (1.0 - x - y) / y;
        (x_val, y_luma, z_val)
    }
}

impl ColourConvert<Yxy, Xyz> for YxyToXyz {
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
            let (x, y, z) = yxy_f32_to_xyz_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
            pixel_out[0] = x;
            pixel_out[1] = y;
            pixel_out[2] = z;
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
    fn zero_y_coordinate_avoids_division_by_zero() {
        let converter = YxyToXyz;
        let input_data = [0.4_f32, 0.2, 0.0];
        let mut output_data = [1.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        converter.convert_region(&mut (), &input, &mut output);
        assert_eq!(output_data, [0.0, 0.4, 0.0]);
    }
}
