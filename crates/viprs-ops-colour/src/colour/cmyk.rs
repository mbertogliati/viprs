use std::marker::PhantomData;

use bytemuck::Pod;

use viprs_core::{
    colorspace::{Cmyk, SRgb},
    colour::ColourConvert,
    format::BandFormat,
    image::{DemandHint, Region, Tile, TileMut},
    shared_ops::cast_sample::CastSample,
};

/// Applies the `cmyk` colour transform to image pixels. Use it when a pipeline needs to move
/// between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::cmyk::CmykToRgbOp;
///
/// let op = CmykToRgbOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct CmykToRgbOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> CmykToRgbOp<F> {
    #[must_use]
    /// Creates a new `CmykToRgbOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for CmykToRgbOp<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies the `cmyk` colour transform to image pixels. Use it when a pipeline needs to move
/// between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::cmyk::RgbToCmykOp;
///
/// let op = RgbToCmykOp { /* operation parameters */ };
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct RgbToCmykOp<F: BandFormat> {
    _format: PhantomData<F>,
}

impl<F: BandFormat> RgbToCmykOp<F> {
    #[must_use]
    /// Creates a new `RgbToCmykOp`.
    pub const fn new() -> Self {
        Self {
            _format: PhantomData,
        }
    }
}

impl<F: BandFormat> Default for RgbToCmykOp<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn cmyk_to_rgb_components(cyan: f32, magenta: f32, yellow: f32, key: f32) -> (f32, f32, f32) {
    let key_scale = 1.0 - key.clamp(0.0, 1.0);
    (
        (1.0 - cyan.clamp(0.0, 1.0)) * key_scale,
        (1.0 - magenta.clamp(0.0, 1.0)) * key_scale,
        (1.0 - yellow.clamp(0.0, 1.0)) * key_scale,
    )
}

#[inline(always)]
fn rgb_to_cmyk_components(red: f32, green: f32, blue: f32) -> (f32, f32, f32, f32) {
    let red = red.clamp(0.0, 1.0);
    let green = green.clamp(0.0, 1.0);
    let blue = blue.clamp(0.0, 1.0);
    let key = 1.0 - red.max(green).max(blue);

    if key >= 1.0 {
        return (0.0, 0.0, 0.0, 1.0);
    }

    let denom = 1.0 - key;
    (
        ((1.0 - red - key) / denom).clamp(0.0, 1.0),
        ((1.0 - green - key) / denom).clamp(0.0, 1.0),
        ((1.0 - blue - key) / denom).clamp(0.0, 1.0),
        key.clamp(0.0, 1.0),
    )
}

impl<F> ColourConvert<Cmyk, SRgb> for CmykToRgbOp<F>
where
    F: BandFormat,
    F::Sample: CastSample<f32> + Pod,
    f32: CastSample<F::Sample>,
{
    type InputFormat = F;
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
    fn convert_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(4)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (red, green, blue) = cmyk_to_rgb_components(
                pixel_in[0].cast_to(),
                pixel_in[1].cast_to(),
                pixel_in[2].cast_to(),
                pixel_in[3].cast_to(),
            );
            pixel_out[0] = red.cast_to();
            pixel_out[1] = green.cast_to();
            pixel_out[2] = blue.cast_to();
        }
    }
}

impl<F> ColourConvert<SRgb, Cmyk> for RgbToCmykOp<F>
where
    F: BandFormat,
    F::Sample: CastSample<f32> + Pod,
    f32: CastSample<F::Sample>,
{
    type InputFormat = F;
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
    fn convert_region(&self, (): &mut (), input: &Tile<F>, output: &mut TileMut<F>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(3)
            .zip(output.data.chunks_exact_mut(4))
        {
            let (cyan, magenta, yellow, key) = rgb_to_cmyk_components(
                pixel_in[0].cast_to(),
                pixel_in[1].cast_to(),
                pixel_in[2].cast_to(),
            );
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
    use proptest::prelude::*;
    use viprs_core::{
        colour::ColourConvert,
        format::{F32, U8},
        image::{Region, Tile, TileMut},
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    proptest! {
        #[test]
        fn rgb_cmyk_rgb_round_trip_proptest(
            red in 0.0f32..=1.0,
            green in 0.0f32..=1.0,
            blue in 0.0f32..=1.0
        ) {
            let forward = RgbToCmykOp::<F32>::new();
            let inverse = CmykToRgbOp::<F32>::new();
            let region = make_region(1);
            let input_data = [red, green, blue];
            let input = Tile::new(region, 3, &input_data);

            let mut cmyk_data = [0.0f32; 4];
            let mut cmyk_tile = TileMut::new(region, 4, &mut cmyk_data);
            forward.convert_region(&mut (), &input, &mut cmyk_tile);

            let cmyk_input = Tile::new(region, 4, &cmyk_data);
            let mut roundtrip_data = [0.0f32; 3];
            let mut roundtrip_tile = TileMut::new(region, 3, &mut roundtrip_data);
            inverse.convert_region(&mut (), &cmyk_input, &mut roundtrip_tile);

            prop_assert!((roundtrip_data[0] - red).abs() <= 1e-6);
            prop_assert!((roundtrip_data[1] - green).abs() <= 1e-6);
            prop_assert!((roundtrip_data[2] - blue).abs() <= 1e-6);
        }
    }

    #[test]
    fn rgb_to_cmyk_boundaries_match_expected_u8_values() {
        let converter = RgbToCmykOp::<U8>::new();
        let region = make_region(3);
        let input_data = [
            255_u8, 255, 255, // white
            0, 0, 0, // black
            255, 0, 0, // red
        ];
        let input = Tile::new(region, 3, &input_data);
        let mut output_data = [0u8; 12];
        let mut output = TileMut::new(region, 4, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(&output_data[0..4], &[0, 0, 0, 0]);
        assert_eq!(&output_data[4..8], &[0, 0, 0, 255]);
        assert_eq!(&output_data[8..12], &[0, 255, 255, 0]);
    }

    #[test]
    fn cmyk_to_rgb_boundaries_match_expected_u8_values() {
        let converter = CmykToRgbOp::<U8>::new();
        let region = make_region(3);
        let input_data = [
            0_u8, 0, 0, 0, // white
            0, 0, 0, 255, // black
            0, 255, 255, 0, // red
        ];
        let input = Tile::new(region, 4, &input_data);
        let mut output_data = [0u8; 9];
        let mut output = TileMut::new(region, 3, &mut output_data);

        converter.convert_region(&mut (), &input, &mut output);

        assert_eq!(&output_data[0..3], &[255, 255, 255]);
        assert_eq!(&output_data[3..6], &[0, 0, 0]);
        assert_eq!(&output_data[6..9], &[255, 0, 0]);
    }

    #[test]
    fn converters_are_pixel_local() {
        let region = Region::new(4, 8, 16, 32);
        assert_eq!(RgbToCmykOp::<U8>::new().demand_hint(), DemandHint::Any);
        assert_eq!(CmykToRgbOp::<U8>::new().demand_hint(), DemandHint::Any);
        assert_eq!(
            RgbToCmykOp::<U8>::new().required_input_region(&region),
            region
        );
        assert_eq!(
            CmykToRgbOp::<U8>::new().required_input_region(&region),
            region
        );
    }
}
