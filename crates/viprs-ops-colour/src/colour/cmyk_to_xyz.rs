use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    colorspace::{Cmyk, Xyz},
    colour::ColourConvert,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    shared_ops::cast_sample::CastSample,
};

use super::math::{D65_X0, D65_Y0, D65_Z0};

/// Represents a cmyk to xyz.
pub struct CmykToXyz<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> CmykToXyz<F> {
    #[must_use]
    /// Creates a new `CmykToXyz`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for CmykToXyz<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ColourConvert<Cmyk, Xyz> for CmykToXyz<F>
where
    F: BandFormat,
    F::Sample: CastSample<f32> + Pod,
{
    type InputFormat = F;
    type OutputFormat = viprs_core::format::F32;
    type State = ();

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn convert_region(
        &self,
        (): &mut (),
        input: &Tile<Self::InputFormat>,
        output: &mut TileMut<Self::OutputFormat>,
    ) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(4)
            .zip(output.data.chunks_exact_mut(3))
        {
            let cyan = pixel_in[0].cast_to();
            let magenta = pixel_in[1].cast_to();
            let yellow = pixel_in[2].cast_to();
            let key = pixel_in[3].cast_to();

            let red = 1.0 - cyan.mul_add(1.0 - key, key);
            let green = 1.0 - magenta.mul_add(1.0 - key, key);
            let blue = 1.0 - yellow.mul_add(1.0 - key, key);

            pixel_out[0] = D65_X0 * red;
            pixel_out[1] = D65_Y0 * green;
            pixel_out[2] = D65_Z0 * blue;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        format::U8,
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    proptest! {
        #[test]
        fn cmyk_to_xyz_matches_fallback_formula_proptest(
            cyan in any::<u8>(),
            magenta in any::<u8>(),
            yellow in any::<u8>(),
            key in any::<u8>()
        ) {
            let direct = CmykToXyz::<U8>::new();
            let region = make_region(1);
            let input_data = [cyan, magenta, yellow, key];
            let input = Tile::new(region, 4, &input_data);

            let mut direct_xyz = [0.0f32; 3];
            let mut direct_tile = TileMut::new(region, 3, &mut direct_xyz);
            direct.convert_region(&mut (), &input, &mut direct_tile);

            let cyan = f32::from(cyan) / 255.0;
            let magenta = f32::from(magenta) / 255.0;
            let yellow = f32::from(yellow) / 255.0;
            let key = f32::from(key) / 255.0;
            let red = 1.0 - (cyan * (1.0 - key) + key);
            let green = 1.0 - (magenta * (1.0 - key) + key);
            let blue = 1.0 - (yellow * (1.0 - key) + key);

            prop_assert!((direct_xyz[0] - D65_X0 * red).abs() <= 1e-6);
            prop_assert!((direct_xyz[1] - D65_Y0 * green).abs() <= 1e-6);
            prop_assert!((direct_xyz[2] - D65_Z0 * blue).abs() <= 1e-6);
        }
    }

    #[test]
    fn white_cmyk_maps_to_d65_white() {
        let converter = CmykToXyz::<U8>::new();
        let input_data = [0_u8, 0, 0, 0];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert!((output_data[0] - D65_X0).abs() < 1e-6);
        assert!((output_data[1] - D65_Y0).abs() < 1e-6);
        assert!((output_data[2] - D65_Z0).abs() < 1e-6);
    }
}
