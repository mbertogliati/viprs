use viprs_core::{
    error::{BuildError, ViprsError},
    format::{F32, U8},
    image::{DemandHint, Region, Tile, TileMut},
    op::{Op, PixelLocalOp},
};

/// Applies the `labq to lab` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs_ops_colour::colour::labq_to_lab::LabQToLab;
///
/// let op = LabQToLab;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabQToLab;
const LABQ_TO_LAB_INPUT_BANDS: u32 = 4;
const LABQ_TO_LAB_OUTPUT_BANDS: u32 = 3;

#[inline(always)]
fn labq_bytes_to_lab_f32(lightness: u8, a: u8, b: u8, lsbs: u8) -> (f32, f32, f32) {
    let lightness = ((i32::from(lightness) << 2) | (i32::from(lsbs) >> 6)) as f32;
    let a_high = i32::from(i8::from_ne_bytes([a]));
    let b_high = i32::from(i8::from_ne_bytes([b]));
    let a_val = (a_high << 3) | ((i32::from(lsbs) >> 3) & 0x7);
    let b_val = (b_high << 3) | (i32::from(lsbs) & 0x7);

    (
        lightness * (100.0 / 1023.0),
        a_val as f32 * 0.125,
        b_val as f32 * 0.125,
    )
}

impl Op for LabQToLab {
    type Input = U8;
    type Output = F32;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(3);

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
        if input_bands == LABQ_TO_LAB_INPUT_BANDS && output_bands == LABQ_TO_LAB_OUTPUT_BANDS {
            Ok(())
        } else {
            Err(BuildError::InvalidOperationBands {
                op: "LabQToLab",
                input_bands,
                output_bands,
                expected: "4 bands",
                expected_output: "3 bands",
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
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<F32>) {
        for (pixel_in, pixel_out) in input
            .data
            .chunks_exact(4)
            .zip(output.data.chunks_exact_mut(3))
        {
            let (lightness, a, b) =
                labq_bytes_to_lab_f32(pixel_in[0], pixel_in[1], pixel_in[2], pixel_in[3]);
            pixel_out[0] = lightness;
            pixel_out[1] = a;
            pixel_out[2] = b;
        }
    }
}

impl PixelLocalOp for LabQToLab {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::colour::lab_to_labq::LabToLabQ;
    use proptest::prelude::*;
    use viprs_core::{
      image::{InMemoryImage, Region, Tile, TileMut},
      op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn decode_labq(input_data: Vec<u8>) -> Vec<f32> {
        let pixels = input_data.len() / 4;
        let input_image = InMemoryImage::<U8>::from_buffer(pixels as u32, 1, 4, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 4, input_image.pixels());
        let mut output_data = vec![0.0_f32; pixels * 3];
        let mut output = TileMut::new(region, 3, &mut output_data);

        LabQToLab.process_region(&mut (), &input, &mut output);

        output_data
    }

    fn encode_lab(input_data: Vec<f32>) -> Vec<u8> {
        let pixels = input_data.len() / 3;
        let input_image = InMemoryImage::<F32>::from_buffer(pixels as u32, 1, 3, input_data).unwrap();
        let region = make_region(pixels);
        let input = Tile::new(region, 3, input_image.pixels());
        let mut output_data = vec![0_u8; pixels * 4];
        let mut output = TileMut::new(region, 4, &mut output_data);

        LabToLabQ.process_region(&mut (), &input, &mut output);

        output_data
    }

    proptest! {
        #[test]
        fn labq_lab_labq_round_trip_preserves_original_bytes(
            lightness in any::<u8>(),
            a in any::<u8>(),
            b in any::<u8>(),
            lsbs in any::<u8>(),
        ) {
            let input_data = vec![lightness, a, b, lsbs];
            let decoded = decode_labq(input_data.clone());

            prop_assert_eq!(encode_lab(decoded), input_data);
        }
    }

    #[test]
    fn packed_neutral_labq_decodes_correctly() {
        let op = LabQToLab;
        let input_data = [255_u8, 0_u8, 0_u8, 192_u8];
        let mut output_data = [0.0_f32; 3];
        let region = make_region(1);
        let input = Tile::new(region, 4, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);
        op.process_region(&mut (), &input, &mut output);

        assert!((output_data[0] - 100.0).abs() < 0.1);
        assert!(output_data[1].abs() < 0.125);
        assert!(output_data[2].abs() < 0.125);
    }

    #[test]
    fn packed_extrema_decode_to_expected_lab_values() {
        let output_data = decode_labq(vec![0, 128, 128, 0, 255, 127, 127, 255]);

        assert_eq!(&output_data[..3], &[0.0, -128.0, -128.0]);
        assert!((output_data[3] - 100.0).abs() < 1e-6);
        assert!((output_data[4] - 127.875).abs() < 1e-6);
        assert!((output_data[5] - 127.875).abs() < 1e-6);
    }

    #[test]
    fn demand_hint_is_any() {
        assert_eq!(LabQToLab.demand_hint(), DemandHint::Any);
    }

    #[test]
    fn required_input_region_is_identity() {
        let region = Region::new(3, 5, 7, 11);
        assert_eq!(LabQToLab.required_input_region(&region), region);
    }

    #[test]
    fn start_returns_unit() {
        LabQToLab.start();
    }

    #[test]
    fn operation_bridge_forces_three_output_bands() {
        let bridge = OperationBridge::new_pixel_local(LabQToLab, 4);
        assert_eq!(bridge.bands, 3);
    }

    #[test]
    fn labq_to_lab_rejects_underspecified_band_count() {
        let region = make_region(1);
        let input_data = [255_u8, 0_u8, 0_u8];
        let mut output_data = [7.0_f32; 3];
        let input = Tile::new(region, 3, &input_data);
        let mut output = TileMut::new(region, 3, &mut output_data);

        let err = LabQToLab
            .execute_region(&mut (), &input, &mut output)
            .expect_err("LabQToLab must reject inputs with fewer than 4 bands");

        assert!(
            matches!(
                err,
                viprs_core::error::ViprsError::Build(
                    viprs_core::error::BuildError::InvalidOperationBands {
                        op: "LabQToLab",
                        input_bands: 3,
                        output_bands: 3,
                        ..
                    }
                )
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(output_data, [7.0, 7.0, 7.0]);
    }
}
