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

/// Applies the `xyz to cmyk` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::xyz_to_cmyk::XyzToCmyk;
///
/// let op = XyzToCmyk { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct XyzToCmyk<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> XyzToCmyk<F> {
    #[must_use]
    /// Creates a new `XyzToCmyk`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for XyzToCmyk<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F> ColourConvert<Xyz, Cmyk> for XyzToCmyk<F>
where
    F: BandFormat,
    F::Sample: Pod,
    f32: CastSample<F::Sample>,
{
    type InputFormat = viprs_core::format::F32;
    type OutputFormat = F;
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
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(4))
        {
            let red = pixel_in[0] / D65_X0;
            let green = pixel_in[1] / D65_Y0;
            let blue = pixel_in[2] / D65_Z0;

            let cyan = 1.0 - red;
            let magenta = 1.0 - green;
            let yellow = 1.0 - blue;
            let key = cyan.min(magenta).min(yellow);
            let inverse_key = 1.0 - key;

            let (cyan, magenta, yellow, key) = if inverse_key < 0.000_01 {
                (1.0, 1.0, 1.0, 1.0)
            } else {
                (
                    ((cyan - key) / inverse_key).clamp(0.0, 1.0),
                    ((magenta - key) / inverse_key).clamp(0.0, 1.0),
                    ((yellow - key) / inverse_key).clamp(0.0, 1.0),
                    key.clamp(0.0, 1.0),
                )
            };

            pixel_out[0] = cyan.cast_to();
            pixel_out[1] = magenta.cast_to();
            pixel_out[2] = yellow.cast_to();
            pixel_out[3] = key.cast_to();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::cmyk_to_xyz::CmykToXyz;
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
        fn xyz_cmyk_xyz_round_trip_proptest(
            x in 0.0f32..=0.95047,
            y in 0.0f32..=1.0,
            z in 0.0f32..=1.08883
        ) {
            let forward = XyzToCmyk::<U8>::new();
            let inverse = CmykToXyz::<U8>::new();
            let region = make_region(1);
            let input_data = [x, y, z];
            let input = Tile::new(region, 3, &input_data);

            let mut cmyk_data = [0u8; 4];
            let mut cmyk_tile = TileMut::new(region, 4, &mut cmyk_data);
            forward.convert_region(&mut (), &input, &mut cmyk_tile);

            let cmyk_input = Tile::new(region, 4, &cmyk_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &cmyk_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - x).abs() <= 0.01);
            prop_assert!((roundtrip_data[1] - y).abs() <= 0.01);
            prop_assert!((roundtrip_data[2] - z).abs() <= 0.01);
        }
    }

    #[test]
    fn d65_white_maps_to_zero_ink() {
        let converter = XyzToCmyk::<U8>::new();
        let input_data = [D65_X0, D65_Y0, D65_Z0];
        let mut output_data = [0_u8; 4];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 4, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(output_data, [0, 0, 0, 0]);
    }
}
