use std::sync::OnceLock;

use viprs_core::{
    format::U8,
    image::{DemandHint, Region, Tile, TileMut},
    op::Op,
};

use super::math::{lab_to_xyz_components, scrgb_to_srgb_u8_components, xyz_to_scrgb_components};

const LABQ_INDEX_SIDE: usize = 64;
const LABQ_TABLE_SIZE: usize = LABQ_INDEX_SIDE * LABQ_INDEX_SIDE * LABQ_INDEX_SIDE;

struct LabQToSRgbTables {
    red: Box<[u8; LABQ_TABLE_SIZE]>,
    green: Box<[u8; LABQ_TABLE_SIZE]>,
    blue: Box<[u8; LABQ_TABLE_SIZE]>,
}

static LABQ_TO_SRGB_TABLES: OnceLock<LabQToSRgbTables> = OnceLock::new();

/// Applies the `labq to srgb` colour transform to image pixels. Use it when a pipeline needs to
/// move between colour spaces or encoded representations.
///
/// # Examples
/// ```ignore
/// use viprs::domain::crate::colour::labq_to_srgb::LabQToSRgb;
///
/// let op = LabQToSRgb;
/// // Run `op` through a compiled image pipeline.
/// ```
pub struct LabQToSRgb;

#[inline(always)]
const fn labq_index(lightness: usize, a: usize, b: usize) -> usize {
    lightness + (a << 6) + (b << 12)
}

fn boxed_zero_array() -> Box<[u8; LABQ_TABLE_SIZE]> {
    vec![0u8; LABQ_TABLE_SIZE]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| {
            debug_assert!(false, "fixed-size LabQ LUT allocation must match");
            // SAFETY: the boxed slice length is constructed from `LABQ_TABLE_SIZE`, so the conversion cannot fail.
            unsafe { std::hint::unreachable_unchecked() }
        })
}

fn labq_tables() -> &'static LabQToSRgbTables {
    LABQ_TO_SRGB_TABLES.get_or_init(|| {
        let mut red = boxed_zero_array();
        let mut green = boxed_zero_array();
        let mut blue = boxed_zero_array();

        for lightness_index in 0..LABQ_INDEX_SIDE {
            for a_index in 0..LABQ_INDEX_SIDE {
                for b_index in 0..LABQ_INDEX_SIDE {
                    let lightness = ((lightness_index << 2) as f32) * (100.0 / 256.0);
                    let a = f32::from(i8::from_ne_bytes([((a_index << 2) & 0xff) as u8]));
                    let b = f32::from(i8::from_ne_bytes([((b_index << 2) & 0xff) as u8]));
                    let (x, y, z) = lab_to_xyz_components(lightness, a, b);
                    let (red_f, green_f, blue_f) = xyz_to_scrgb_components(x, y, z);
                    let rgb = scrgb_to_srgb_u8_components(red_f, green_f, blue_f);
                    let index = labq_index(lightness_index, a_index, b_index);
                    red[index] = rgb[0];
                    green[index] = rgb[1];
                    blue[index] = rgb[2];
                }
            }
        }

        LabQToSRgbTables { red, green, blue }
    })
}

impl Op for LabQToSRgb {
    type Input = U8;
    type Output = U8;
    type State = ();

    const OUTPUT_BANDS: Option<usize> = Some(3);

    fn demand_hint(&self) -> DemandHint {
        DemandHint::Any
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn start(&self) {}

    #[inline]
    fn process_region(&self, (): &mut (), input: &Tile<U8>, output: &mut TileMut<U8>) {
        let tables = labq_tables();
        let width = input.region.width as usize;
        let input_row_stride = width * 4;
        let output_row_stride = width * 3;

        for (input_row, output_row) in input
            .data
            .chunks_exact(input_row_stride)
            .zip(output.data.chunks_exact_mut(output_row_stride))
        {
            let mut lightness_error = 0i32;
            let mut a_error = 0i32;
            let mut b_error = 0i32;

            for (pixel_in, pixel_out) in input_row
                .chunks_exact(4)
                .zip(output_row.chunks_exact_mut(3))
            {
                let mut lightness = i32::from(pixel_in[0]) + lightness_error;
                let mut a = i32::from(i8::from_ne_bytes([pixel_in[1]])) + a_error;
                let mut b = i32::from(i8::from_ne_bytes([pixel_in[2]])) + b_error;

                if lightness > 255 {
                    lightness = 255;
                }
                if a > 127 {
                    a = 127;
                }
                if b > 127 {
                    b = 127;
                }

                lightness_error = lightness & 3;
                a_error = a & 3;
                b_error = b & 3;

                let index = labq_index(
                    ((lightness >> 2) & 63) as usize,
                    ((a >> 2) & 63) as usize,
                    ((b >> 2) & 63) as usize,
                );

                pixel_out[0] = tables.red[index];
                pixel_out[1] = tables.green[index];
                pixel_out[2] = tables.blue[index];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use viprs_core::{
        image::{Region, Tile, TileMut},
        op::OperationBridge,
    };

    fn make_region(pixels: usize) -> Region {
        Region::new(0, 0, pixels as u32, 1)
    }

    fn run_labq_to_srgb(input_data: &[u8]) -> Vec<u8> {
        let op = LabQToSRgb;
        let region = make_region(input_data.len() / 4);
        let input = Tile::new(region, 4, input_data);
        let mut output_data = vec![0_u8; (input_data.len() / 4) * 3];
        let mut output = TileMut::new(region, 3, &mut output_data);

        op.process_region(&mut (), &input, &mut output);
        output_data
    }

    fn scalar_reference_row(input_data: &[u8]) -> Vec<u8> {
        let mut output_data = vec![0_u8; (input_data.len() / 4) * 3];
        let mut lightness_error = 0i32;
        let mut a_error = 0i32;
        let mut b_error = 0i32;

        for (pixel_in, pixel_out) in input_data
            .chunks_exact(4)
            .zip(output_data.chunks_exact_mut(3))
        {
            let mut lightness = i32::from(pixel_in[0]) + lightness_error;
            let mut a = i32::from(i8::from_ne_bytes([pixel_in[1]])) + a_error;
            let mut b = i32::from(i8::from_ne_bytes([pixel_in[2]])) + b_error;

            lightness = lightness.min(255);
            a = a.min(127);
            b = b.min(127);

            lightness_error = lightness & 3;
            a_error = a & 3;
            b_error = b & 3;

            let lightness = ((lightness >> 2) << 2) as f32 * (100.0 / 256.0);
            let a = i8::from_ne_bytes([(((a >> 2) & 63) as u8) << 2]) as f32;
            let b = i8::from_ne_bytes([(((b >> 2) & 63) as u8) << 2]) as f32;
            let (x, y, z) = lab_to_xyz_components(lightness, a, b);
            let (red_f, green_f, blue_f) = xyz_to_scrgb_components(x, y, z);
            pixel_out.copy_from_slice(&scrgb_to_srgb_u8_components(red_f, green_f, blue_f));
        }

        output_data
    }

    proptest! {
        #[test]
        fn labq_to_srgb_matches_scalar_reference_proptest(pixels in prop::collection::vec(any::<[u8; 4]>(), 1..=16)) {
            let input_data = pixels
                .into_iter()
                .flat_map(|pixel| pixel.into_iter())
                .collect::<Vec<_>>();

            let actual = run_labq_to_srgb(&input_data);
            let expected = scalar_reference_row(&input_data);

            prop_assert_eq!(actual, expected);
        }
    }

    #[test]
    fn libvips_reference_pixels_match_expected_srgb_bytes() {
        // libvips references generated with:
        //   vips rawload sample.raw sample.v 1 1 4 --format uchar --interpretation labq
        //   vips copy sample.v sample-coded.v --coding labq --interpretation labq
        //   vips LabQ2sRGB sample-coded.v sample-srgb.v
        //   vips getpoint sample-srgb.v 0 0
        let cases = [
            ([255_u8, 0, 0, 192], [251_u8, 251, 251]),
            ([0, 0, 0, 0], [0, 0, 0]),
            ([128, 0, 0, 64], [119, 119, 119]),
            ([196, 64, 32, 255], [255, 133, 133]),
            ([96, 224, 240, 17], [0, 102, 113]),
            ([160, 96, 224, 88], [255, 13, 210]),
            ([144, 208, 96, 201], [79, 153, 0]),
        ];

        for (input, expected) in cases {
            assert_eq!(run_labq_to_srgb(&input), expected);
        }
    }

    #[test]
    fn libvips_boundary_row_matches_clipped_error_diffusion_reference() {
        // libvips reference row exercises upper clipping on the second pixel and
        // propagated quantization error on the fourth pixel.
        let input_data = [
            255_u8, 127, 127, 0, 255, 127, 127, 255, 0, 255, 255, 12, 1, 255, 255, 99,
        ];

        assert_eq!(
            run_labq_to_srgb(&input_data),
            vec![255_u8, 75, 0, 255, 75, 0, 0, 0, 0, 6, 6, 6]
        );
    }

    #[test]
    fn operation_bridge_forces_three_output_bands() {
        let bridge = OperationBridge::new(LabQToSRgb, 4);
        assert_eq!(bridge.bands, 3);
    }
}
