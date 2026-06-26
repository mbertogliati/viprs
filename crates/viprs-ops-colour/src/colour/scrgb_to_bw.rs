use viprs_core::{
    colorspace::{Greyscale, ScRgb},
    colour::ColourConvert,
    error::{BuildError, ViprsError},
    format::F32,
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `scrgb to bw` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::scrgb_to_bw::ScRgbToBw;
///
/// let op = ScRgbToBw;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct ScRgbToBw;
const SCRGB_TO_BW_INPUT_BANDS: u32 = 3;
const SCRGB_TO_BW_OUTPUT_BANDS: u32 = 1;

#[inline(always)]
fn scrgb_to_bw_f32(r: f32, g: f32, b: f32) -> f32 {
    0.0722f32.mul_add(b, 0.7152f32.mul_add(g, 0.2126 * r))
}

#[inline]
fn process_tile(input: &Tile<F32>, output: &mut TileMut<F32>) {
    for (pixel_in, pixel_out) in input.data.chunks_exact(3).zip(output.data.iter_mut()) {
        *pixel_out = scrgb_to_bw_f32(pixel_in[0], pixel_in[1], pixel_in[2]);
    }
}

impl ColourConvert<ScRgb, Greyscale> for ScRgbToBw {
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
        process_tile(input, output);
    }
}

impl Op for ScRgbToBw {
    type Input = F32;
    type Output = F32;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(1);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    fn validate_build_contract(
        &self,
        input_bands: u32,
        output_bands: u32,
    ) -> Result<(), BuildError> {
        if input_bands == SCRGB_TO_BW_INPUT_BANDS && output_bands == SCRGB_TO_BW_OUTPUT_BANDS {
            Ok(())
        } else {
            Err(BuildError::InvalidOperationBands {
                op: "ScRgbToBw",
                input_bands,
                output_bands,
                expected: "3 bands",
                expected_output: "1 band",
            })
        }
    }

    fn validate_region_contract(
        &self,
        _input_region: Region,
        input_bands: u32,
        _output_region: Region,
        output_bands: u32,
    ) -> Result<(), ViprsError> {
        self.validate_build_contract(input_bands, output_bands)
            .map_err(ViprsError::from)
    }

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<F32>, output: &mut TileMut<F32>) {
        process_tile(input, output);
    }
}

impl PixelLocalOp for ScRgbToBw {}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        format::F32,
        image::{Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn run_process_region(input_data: Vec<f32>) -> Vec<f32> {
        let pixels = input_data.len() / 3;
        let input_image =
            viprs_core::image::InMemoryImage::<F32>::from_buffer(pixels as u32, 1, 3, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = vec![0.0_f32; pixels];
        let mut output = TileMut::new(region, 1, &mut output_data);

        ScRgbToBw.process_region(&mut (), &input, &mut output);

        output_data
    }

    fn run_convert_region(input_data: Vec<f32>) -> Vec<f32> {
        let pixels = input_data.len() / 3;
        let input_image =
            viprs_core::image::InMemoryImage::<F32>::from_buffer(pixels as u32, 1, 3, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = vec![0.0_f32; pixels];
        let mut output = TileMut::new(region, 1, &mut output_data);

        <ScRgbToBw as ColourConvert<ScRgb, Greyscale>>::convert_region(
            &ScRgbToBw,
            &mut (),
            &input,
            &mut output,
        );

        output_data
    }

    proptest! {
        #[test]
        fn neutral_scrgb_pixels_are_identity_in_greyscale(value in -4.0_f32..4.0) {
            let output = run_process_region(vec![value, value, value]);
            prop_assert!((output[0] - value).abs() < 1e-6, "output={} input={value}", output[0]);
        }
    }

    #[test]
    fn white_maps_to_one() {
        let op = ScRgbToBw;
        let input_data = [1.0_f32, 1.0, 1.0];
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        assert!((output_data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn green_uses_luminance_weight() {
        let op = ScRgbToBw;
        let input_data = [0.0_f32, 1.0, 0.0];
        let mut output_data = [0.0_f32; 1];
        let region = make_region(1);
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);
        op.process_region(&mut (), &input, &mut output);
        assert!((output_data[0] - 0.7152).abs() < 1e-6);
    }

    #[test]
    fn red_and_blue_channels_use_expected_weights() {
        let output = run_process_region(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert!((output[0] - 0.2126).abs() < 1e-6);
        assert!((output[1] - 0.0722).abs() < 1e-6);
    }

    #[test]
    fn colour_convert_and_op_paths_match() {
        let input_data = vec![0.25, -0.5, 1.5, 1.0, 0.5, 0.0];
        assert_eq!(
            run_convert_region(input_data.clone()),
            run_process_region(input_data)
        );
    }

    #[test]
    fn op_demand_hint_is_any() {
        assert_eq!(<ScRgbToBw as Op>::demand_hint(&ScRgbToBw), DemandHint::Any);
    }

    #[test]
    fn colour_convert_demand_hint_is_any() {
        assert_eq!(
            <ScRgbToBw as ColourConvert<ScRgb, Greyscale>>::demand_hint(&ScRgbToBw),
            DemandHint::Any
        );
    }

    #[test]
    fn required_input_region_is_identity_for_op_and_converter() {
        let region = Region::new(9, 7, 5, 3);
        assert_eq!(
            <ScRgbToBw as Op>::required_input_region(&ScRgbToBw, &region),
            region
        );
        assert_eq!(
            <ScRgbToBw as ColourConvert<ScRgb, Greyscale>>::required_input_region(
                &ScRgbToBw, &region,
            ),
            region
        );
    }

    #[test]
    fn start_returns_unit_for_op_and_converter() {
        <ScRgbToBw as Op>::start(&ScRgbToBw);
        <ScRgbToBw as ColourConvert<ScRgb, Greyscale>>::start(&ScRgbToBw);
    }

    #[test]
    fn operation_bridge_forces_single_output_band() {
        let bridge = OperationBridge::new_pixel_local(ScRgbToBw, 3);
        assert_eq!(bridge.bands, 1);
    }

    #[test]
    fn scrgb_to_bw_rejects_underspecified_band_count() {
        let region = make_region(1);
        let input_data = [0.5_f32, 0.25];
        let mut output_data = [9.0_f32; 1];
        let input = Tile::new(region, 2, &input_data);
        let mut output = TileMut::new(region, 1, &mut output_data);

        let err = <ScRgbToBw as Op>::execute_region(&ScRgbToBw, &mut (), &input, &mut output)
            .expect_err("ScRgbToBw must reject inputs with fewer than 3 bands");

        assert!(
            matches!(
                err,
                viprs_core::error::ViprsError::Build(
                    viprs_core::error::BuildError::InvalidOperationBands {
                        op: "ScRgbToBw",
                        input_bands: 2,
                        output_bands: 1,
                        ..
                    }
                )
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(output_data, [9.0]);
    }
}
