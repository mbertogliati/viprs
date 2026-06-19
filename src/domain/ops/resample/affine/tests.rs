use super::*;
use crate::domain::{
    format::{F64, U8},
    image::{Region, Tile, TileMut},
    kernel::InterpolationKernel,
    op::{NodeSpec, Op},
};
use proptest::prelude::*;

/// Run an `Affine<U8>` op with a 1-band input and return the output buffer.
fn run_affine_u8(
    input_data: &[u8],
    in_region: Region,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
) -> Vec<u8> {
    let op = Affine::<U8>::new(matrix, tx, ty, kernel, out_region.width, out_region.height);
    let out_len = (out_region.width * out_region.height) as usize;
    let mut output_data = vec![0u8; out_len];
    let input = Tile::<U8>::new(in_region, 1, input_data);
    let mut output = TileMut::<U8>::new(out_region, 1, &mut output_data);
    let mut state = ();
    op.process_region(&mut state, &input, &mut output);
    output_data
}

fn run_affine_u8_bands(
    input_data: &[u8],
    in_region: Region,
    bands: u32,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
) -> Vec<u8> {
    let op = Affine::<U8>::new(matrix, tx, ty, kernel, out_region.width, out_region.height);
    let out_len = (out_region.width * out_region.height * bands) as usize;
    let mut output_data = vec![0u8; out_len];
    let input = Tile::<U8>::new(in_region, bands, input_data);
    let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
    let mut state = ();
    op.process_region(&mut state, &input, &mut output);
    output_data
}

fn run_affine_u8_bands_with_premultiplied(
    input_data: &[u8],
    in_region: Region,
    bands: u32,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
    premultiplied: bool,
) -> Vec<u8> {
    let op = Affine::<U8>::new(matrix, tx, ty, kernel, out_region.width, out_region.height)
        .with_premultiplied(premultiplied);
    let out_len = (out_region.width * out_region.height * bands) as usize;
    let mut output_data = vec![0u8; out_len];
    let input = Tile::<U8>::new(in_region, bands, input_data);
    let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
    let mut state = ();
    op.process_region(&mut state, &input, &mut output);
    output_data
}

fn run_affine_u8_bands_with_extend(
    input_data: &[u8],
    in_region: Region,
    bands: u32,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
    extend: crate::domain::ops::conversion::embed::ExtendMode,
) -> Vec<u8> {
    let op = Affine::<U8>::new(matrix, tx, ty, kernel, out_region.width, out_region.height)
        .with_extend(extend);
    let out_len = (out_region.width * out_region.height * bands) as usize;
    let mut output_data = vec![0u8; out_len];
    let input = Tile::<U8>::new(in_region, bands, input_data);
    let mut output = TileMut::<U8>::new(out_region, bands, &mut output_data);
    let mut state = ();
    op.process_region(&mut state, &input, &mut output);
    output_data
}

fn run_affine_u8_scalar_reference(
    input_data: &[u8],
    in_region: Region,
    bands: u32,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
) -> Vec<u8> {
    let op = Affine::<U8>::new(matrix, tx, ty, kernel, out_region.width, out_region.height);
    let out_w = out_region.width as usize;
    let out_h = out_region.height as usize;
    let bands = bands as usize;
    let mut output_data = vec![0u8; out_w * out_h * bands];
    let input = Tile::<U8>::new(in_region, bands as u32, input_data);
    let mut row_x = matrix[0] * out_region.x as f64 + matrix[1] * out_region.y as f64 + tx;
    let mut row_y = matrix[2] * out_region.x as f64 + matrix[3] * out_region.y as f64 + ty;

    for y_local in 0..out_h {
        let mut x_in = row_x;
        let mut y_in = row_y;
        let row_base = y_local * out_w * bands;

        for x_local in 0..out_w {
            let out_base = row_base + x_local * bands;
            op.sample_pixel_at(
                &input,
                x_in,
                y_in,
                &mut output_data[out_base..out_base + bands],
            );
            x_in += matrix[0];
            y_in += matrix[2];
        }

        row_x += matrix[1];
        row_y += matrix[3];
    }

    output_data
}

fn run_affine_f64(
    input_data: &[f64],
    in_region: Region,
    out_region: Region,
    matrix: [f64; 4],
    tx: f64,
    ty: f64,
    kernel: InterpolationKernel,
) -> Vec<f64> {
    let op = Affine::<F64>::new(matrix, tx, ty, kernel, out_region.width, out_region.height);
    let out_len = (out_region.width * out_region.height) as usize;
    let mut output_data = vec![0.0_f64; out_len];
    let input = Tile::<F64>::new(in_region, 1, input_data);
    let mut output = TileMut::<F64>::new(out_region, 1, &mut output_data);
    let mut state = ();
    op.process_region(&mut state, &input, &mut output);
    output_data
}

#[test]
fn output_tile_fully_outside_input_fills_background() {
    let input = vec![10_u8, 20, 30, 40];
    let output = run_affine_u8(
        &input,
        Region::new(0, 0, 2, 2),
        Region::new(0, 0, 4, 4),
        [1.0, 0.0, 0.0, 1.0],
        10.0,
        10.0,
        InterpolationKernel::Bilinear,
    );

    assert_eq!(output, vec![0; 16]);
}

fn nohalo_anchor_and_sign_for_test(phase: f64) -> (usize, i32) {
    if phase < 0.5 { (2, 1) } else { (3, -1) }
}

fn nohalo_local_bounds(samples: &[f64], x_phase: f64, y_phase: f64) -> (f64, f64) {
    let (anchor_x, sign_x) = nohalo_anchor_and_sign_for_test(x_phase);
    let (anchor_y, sign_y) = nohalo_anchor_and_sign_for_test(y_phase);
    let idx = |anchor: usize, sign: i32, offset: i32| -> usize {
        (anchor as i32 + offset * sign) as usize
    };
    let rows = [
        idx(anchor_y, sign_y, -2),
        idx(anchor_y, sign_y, -1),
        idx(anchor_y, sign_y, 0),
        idx(anchor_y, sign_y, 1),
        idx(anchor_y, sign_y, 2),
    ];
    let cols = [
        idx(anchor_x, sign_x, -2),
        idx(anchor_x, sign_x, -1),
        idx(anchor_x, sign_x, 0),
        idx(anchor_x, sign_x, 1),
        idx(anchor_x, sign_x, 2),
    ];

    let sample = |row: usize, col: usize| -> f64 { samples[row * 6 + col] };
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for &col in &cols[1..4] {
        let value = sample(rows[0], col);
        min = min.min(value);
        max = max.max(value);
    }
    for &row in &rows[1..4] {
        for &col in &cols {
            let value = sample(row, col);
            min = min.min(value);
            max = max.max(value);
        }
    }
    for &col in &cols[1..4] {
        let value = sample(rows[4], col);
        min = min.min(value);
        max = max.max(value);
    }

    (min, max)
}

/// Identity transform (matrix = I, tx = ty = 0): each output pixel must equal
/// the input pixel at the same absolute coordinate.
#[test]
fn identity_nearest() {
    // 4×4 input at image origin; output at same region.
    let data: Vec<u8> = (0u8..16).collect();
    let region = Region::new(0, 0, 4, 4);
    let result = run_affine_u8(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
    );
    assert_eq!(result, data, "identity nearest: {result:?}");
}

#[test]
fn identity_bilinear() {
    let data: Vec<u8> = (0u8..16).collect();
    let region = Region::new(0, 0, 4, 4);
    let result = run_affine_u8(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bilinear,
    );
    assert_eq!(result, data, "identity bilinear: {result:?}");
}

#[test]
fn identity_nearest_two_band_preserves_samples_without_alpha_unpremultiplication() {
    let data = vec![11u8, 64, 42, 95, 203, 0, 7, 255];
    let region = Region::new(0, 0, 2, 2);
    let result = run_affine_u8_bands(
        &data,
        region,
        2,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
    );
    assert_eq!(result, data);
}

#[test]
fn axis_aligned_rgb_bilinear_matches_scalar_reference() {
    let in_region = Region::new(0, 0, 6, 5);
    let out_region = Region::new(0, 0, 4, 3);
    let bands = 3;
    let input: Vec<u8> = (0..(in_region.width * in_region.height * bands))
        .map(|index| ((index * 37 + 11) % 251) as u8)
        .collect();
    let matrix = [1.25, 0.0, 0.0, 1.5];
    let tx = 0.35;
    let ty = 0.2;

    let fast = run_affine_u8_bands(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        ty,
        InterpolationKernel::Bilinear,
    );
    let scalar = run_affine_u8_scalar_reference(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        ty,
        InterpolationKernel::Bilinear,
    );

    assert_eq!(fast, scalar);
}

#[test]
fn axis_aligned_rgb_nearest_matches_scalar_reference() {
    let in_region = Region::new(3, 7, 5, 4);
    let out_region = Region::new(0, 0, 4, 3);
    let bands = 3;
    let input: Vec<u8> = (0..(in_region.width * in_region.height * bands))
        .map(|index| ((index * 29 + 5) % 253) as u8)
        .collect();
    let matrix = [0.75, 0.0, 0.0, 1.0];
    let tx = 3.4;
    let ty = 7.0;

    let fast = run_affine_u8_bands(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        ty,
        InterpolationKernel::Nearest,
    );
    let scalar = run_affine_u8_scalar_reference(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        ty,
        InterpolationKernel::Nearest,
    );

    assert_eq!(fast, scalar);
}

#[test]
fn identity_bicubic() {
    let data: Vec<u8> = (0u8..16).collect();
    let region = Region::new(0, 0, 4, 4);
    let result = run_affine_u8(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bicubic,
    );
    assert_eq!(result, data, "identity bicubic: {result:?}");
}

#[test]
fn identity_lbb() {
    let data: Vec<u8> = (0u8..16).collect();
    let region = Region::new(0, 0, 4, 4);
    let result = run_affine_u8(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Lbb,
    );
    assert_eq!(result, data, "identity lbb: {result:?}");
}

#[test]
fn lbb_fractional_sample_stays_within_local_stencil_bounds() {
    let data = [
        -1.0, 2.0, 9.0, 3.0, //
        4.0, 6.0, 8.0, 5.0, //
        7.0, 0.0, 1.0, 11.0, //
        10.0, 12.0, -3.0, 13.0,
    ];
    let input = Tile::<F64>::new(Region::new(0, 0, 4, 4), 1, &data);
    let op = Affine::<F64>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Lbb,
        1,
        1,
    );

    let value = op.interp_lbb(&input, 1.35, 1.65, 0);
    let min = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    assert!(value >= min - 1e-12, "value={value} min={min}");
    assert!(value <= max + 1e-12, "value={value} max={max}");
}

/// Out-of-bounds pixels must produce `background` (default 0).
#[test]
fn oob_fills_background() {
    // Identity transform, output region = [0,0,4,4], input entirely OOB.
    let data = vec![255u8; 16];
    // Input tile is at x=1000 — none of the output pixels will map into it.
    let in_region = Region::new(1000, 0, 4, 4);
    let out_region = Region::new(0, 0, 4, 4);
    let result = run_affine_u8(
        &data,
        in_region,
        out_region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
    );
    assert!(
        result.iter().all(|&v| v == 0),
        "OOB must fill with background=0: {result:?}"
    );
}

#[test]
fn extend_modes_distinguish_black_white_background_copy_repeat_and_mirror() {
    use crate::domain::ops::conversion::embed::ExtendMode;

    let in_region = Region::new(0, 0, 3, 1);
    let out_region = Region::new(0, 0, 7, 1);
    let bands = 4;
    let input = [
        10u8, 11, 12, 13, //
        20, 21, 22, 23, //
        30, 31, 32, 33,
    ];
    let matrix = [1.0, 0.0, 0.0, 1.0];
    let tx = -2.0;

    let black = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::Black,
    );
    let white = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::White,
    );
    let background = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::Background(vec![1.0, 2.0, 3.0, 4.0]),
    );
    let copy = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::Copy,
    );
    let repeat = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::Repeat,
    );
    let mirror = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        bands,
        out_region,
        matrix,
        tx,
        0.0,
        InterpolationKernel::Nearest,
        ExtendMode::Mirror,
    );

    assert_eq!(
        black,
        vec![
            0, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 0, 0, 0, 0, 0,
            0, 0, 0
        ]
    );
    assert_eq!(white[..8], [255, 255, 255, 255, 255, 255, 255, 255]);
    assert_eq!(white[20..], [255, 255, 255, 255, 255, 255, 255, 255]);
    assert_eq!(
        background,
        vec![
            1, 2, 3, 4, 1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 1, 2, 3, 4, 1,
            2, 3, 4
        ]
    );
    assert_eq!(
        copy,
        vec![
            10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 30, 31,
            32, 33, 30, 31, 32, 33
        ]
    );
    assert_eq!(
        repeat,
        vec![
            20, 21, 22, 23, 30, 31, 32, 33, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 10, 11,
            12, 13, 20, 21, 22, 23
        ]
    );
    assert_eq!(
        mirror,
        vec![
            20, 21, 22, 23, 10, 11, 12, 13, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33, 30, 31,
            32, 33, 20, 21, 22, 23
        ]
    );
}

#[test]
fn bilinear_background_extend_uses_per_band_fill_outside_edges() {
    use crate::domain::ops::conversion::embed::ExtendMode;

    let in_region = Region::new(0, 0, 2, 1);
    let out_region = Region::new(0, 0, 1, 1);
    let input = [10u8, 20, 30, 40, 50, 60];

    let result = run_affine_u8_bands_with_extend(
        &input,
        in_region,
        3,
        out_region,
        [1.0, 0.0, 0.0, 1.0],
        1.5,
        0.0,
        InterpolationKernel::Bilinear,
        ExtendMode::Background(vec![100.0, 110.0, 120.0]),
    );

    assert_eq!(result, vec![70, 80, 90]);
}

#[test]
fn identity_lbb_preserves_constant_field_for_f64() {
    let data = vec![7.5_f64; 16];
    let region = Region::new(0, 0, 4, 4);
    let result = run_affine_f64(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Lbb,
    );
    assert_eq!(result, data);
}

#[test]
fn required_input_region_identity_nearest_matches_output() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
        64,
        64,
    );
    let out = Region::new(0, 0, 64, 64);
    assert_eq!(op.required_input_region(&out), out);
}

#[test]
fn required_input_region_identity_bilinear_adds_one_pixel_on_bottom_right() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bilinear,
        64,
        64,
    );
    let out = Region::new(0, 0, 64, 64);
    assert_eq!(op.required_input_region(&out), Region::new(0, 0, 65, 65));
}

#[test]
fn node_spec_identity_bilinear_allocates_interpolation_halo() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bilinear,
        512,
        512,
    );
    assert_eq!(
        op.node_spec(128, 128),
        NodeSpec {
            input_tile_w: 129,
            input_tile_h: 129,
            output_tile_w: 128,
            output_tile_h: 128,
            coordinate_driven_source: None,
        }
    );
}

#[test]
fn required_input_region_identity_nohalo_matches_six_tap_footprint() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nohalo,
        64,
        64,
    );
    let out = Region::new(0, 0, 64, 64);
    assert_eq!(op.required_input_region(&out), Region::new(-2, -2, 69, 69));
}

#[test]
fn node_spec_identity_nohalo_allocates_six_tap_halo() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nohalo,
        512,
        512,
    );
    assert_eq!(
        op.node_spec(128, 128),
        NodeSpec {
            input_tile_w: 133,
            input_tile_h: 133,
            output_tile_w: 128,
            output_tile_h: 128,
            coordinate_driven_source: None,
        }
    );
}

#[test]
fn node_spec_caps_input_tile_to_source_bounds() {
    let op = Affine::<U8>::new(
        [1000.0, 0.0, 0.0, 1000.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
        16,
        16,
    )
    .with_source_bounds(Region::new(0, 0, 17, 13));

    assert_eq!(
        op.node_spec(128, 128),
        NodeSpec {
            input_tile_w: 17,
            input_tile_h: 13,
            output_tile_w: 128,
            output_tile_h: 128,
            coordinate_driven_source: None,
        }
    );
}

#[test]
fn vsqbs_affine_matches_reference_copy_extend_2x2_to_4x4_upscale() {
    let in_region = Region::new(-1, -1, 5, 5);
    let out_region = Region::new(0, 0, 4, 4);
    let data = vec![
        0u8, 0, 64, 64, 64, 0, 0, 64, 64, 64, 128, 128, 255, 255, 255, 128, 128, 255, 255, 255,
        128, 128, 255, 255, 255,
    ];
    let result = run_affine_u8(
        &data,
        in_region,
        out_region,
        [0.5, 0.0, 0.0, 0.5],
        0.25,
        0.25,
        InterpolationKernel::Vsqbs,
    );

    assert_eq!(
        result,
        vec![
            57, 96, 115, 118, 124, 169, 198, 201, 159, 214, 245, 249, 164, 219, 251, 255
        ]
    );
}

#[test]
fn try_new_rejects_non_finite_and_singular_matrices() {
    assert!(
        Affine::<U8>::try_new(
            [f64::NAN, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            InterpolationKernel::Nearest,
            4,
            4,
        )
        .is_err()
    );
    assert!(
        Affine::<U8>::try_new(
            [1.0, 2.0, 2.0, 4.0],
            0.0,
            0.0,
            InterpolationKernel::Nearest,
            4,
            4,
        )
        .is_err()
    );
}

#[test]
fn skew_transform_uses_generic_path_and_matches_scalar_reference() {
    let in_region = Region::new(0, 0, 6, 5);
    let out_region = Region::new(0, 0, 4, 3);
    let input: Vec<u8> = (0..(in_region.width * in_region.height * 3))
        .map(|index| ((index * 19 + 7) % 251) as u8)
        .collect();
    let matrix = [1.0, 0.25, 0.1, 1.0];

    let generic = run_affine_u8_bands(
        &input,
        in_region,
        3,
        out_region,
        matrix,
        0.2,
        0.4,
        InterpolationKernel::Nearest,
    );
    let scalar = run_affine_u8_scalar_reference(
        &input,
        in_region,
        3,
        out_region,
        matrix,
        0.2,
        0.4,
        InterpolationKernel::Nearest,
    );

    assert_eq!(generic, scalar);
}

#[test]
fn bicubic_fast_path_matches_scalar_reference_for_rgba_and_two_band_u8() {
    let in_region = Region::new(0, 0, 7, 6);
    let out_region = Region::new(0, 0, 3, 3);
    let rgba: Vec<u8> = (0..(in_region.width * in_region.height * 4))
        .map(|index| ((index * 13 + 5) % 255) as u8)
        .collect();
    let two_band: Vec<u8> = (0..(in_region.width * in_region.height * 2))
        .map(|index| ((index * 17 + 9) % 255) as u8)
        .collect();
    let matrix = [0.8, 0.0, 0.0, 0.75];

    let rgba_fast = run_affine_u8_bands(
        &rgba,
        in_region,
        4,
        out_region,
        matrix,
        0.35,
        0.1,
        InterpolationKernel::Bicubic,
    );
    let rgba_scalar = run_affine_u8_scalar_reference(
        &rgba,
        in_region,
        4,
        out_region,
        matrix,
        0.35,
        0.1,
        InterpolationKernel::Bicubic,
    );
    assert_eq!(rgba_fast, rgba_scalar);

    let generic_fast = run_affine_u8_bands(
        &two_band,
        in_region,
        2,
        out_region,
        matrix,
        0.35,
        0.1,
        InterpolationKernel::Bicubic,
    );
    let generic_scalar = run_affine_u8_scalar_reference(
        &two_band,
        in_region,
        2,
        out_region,
        matrix,
        0.35,
        0.1,
        InterpolationKernel::Bicubic,
    );
    assert_eq!(generic_fast, generic_scalar);
}

#[test]
fn bilinear_rgba_sampling_premultiplies_transparent_edges_before_interpolation() {
    let in_region = Region::new(0, 0, 2, 1);
    let out_region = Region::new(0, 0, 1, 1);
    let input = vec![255u8, 0, 0, 0, 0, 0, 255, 255];

    let output = run_affine_u8_bands(
        &input,
        in_region,
        4,
        out_region,
        [1.0, 0.0, 0.0, 1.0],
        0.5,
        0.0,
        InterpolationKernel::Bilinear,
    );

    assert_eq!(output, vec![0, 0, 255, 128]);
}

#[test]
fn bilinear_rgba_sampling_respects_explicit_premultiplied_inputs() {
    let in_region = Region::new(0, 0, 2, 1);
    let out_region = Region::new(0, 0, 1, 1);
    let input = vec![0u8, 0, 0, 0, 0, 0, 255, 255];

    let output = run_affine_u8_bands_with_premultiplied(
        &input,
        in_region,
        4,
        out_region,
        [1.0, 0.0, 0.0, 1.0],
        0.5,
        0.0,
        InterpolationKernel::Bilinear,
        true,
    );

    assert_eq!(output, vec![0, 0, 128, 128]);
}

#[test]
fn negative_output_origin_falls_back_to_generic_path() {
    let region = Region::new(-2, -1, 3, 2);
    let data: Vec<u8> = (0u8..6).collect();
    let result = run_affine_u8(
        &data,
        region,
        region,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
    );
    assert_eq!(result, data);
}

#[test]
fn sample_pixel_at_non_finite_coordinates_fill_background_for_general_kernels() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Lanczos3,
        2,
        2,
    )
    .with_background(23.0);
    let input = Tile::<U8>::new(Region::new(0, 0, 2, 2), 2, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let mut output = [0u8; 2];

    op.sample_pixel_at(&input, f64::NAN, 0.0, &mut output);

    assert_eq!(output, [23, 23]);
}

#[test]
fn required_input_region_handles_empty_output_and_background_only_detection() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        50.0,
        50.0,
        InterpolationKernel::Nearest,
        8,
        8,
    );
    let input = Tile::<U8>::new(Region::new(0, 0, 4, 4), 1, &[0u8; 16]);

    assert_eq!(
        op.required_input_region(&Region::new(3, 4, 0, 0)),
        Region::new(53, 54, 1, 1)
    );
    assert!(op.output_region_is_background_only(&input, &Region::new(0, 0, 2, 2)));
}

#[test]
fn required_input_region_clamps_to_declared_source_bounds() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        50.0,
        50.0,
        InterpolationKernel::Nearest,
        8,
        8,
    )
    .with_source_bounds(Region::new(0, 0, 4, 4));

    assert_eq!(
        op.required_input_region(&Region::new(0, 0, 2, 2)),
        Region::new(4, 4, 0, 0)
    );
}

#[test]
fn process_region_fast_path_rejects_tiles_outside_declared_output_bounds() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
        2,
        2,
    );
    let input = Tile::<U8>::new(Region::new(0, 0, 2, 2), 1, &[1u8, 2, 3, 4]);
    let mut output_data = [0u8; 4];
    let mut output = TileMut::<U8>::new(Region::new(1, 1, 2, 2), 1, &mut output_data);

    assert!(!op.process_region_fast_path(&input, &mut output));
}

#[test]
fn affine_helper_functions_cover_axis_alignment_bilinear_paths_and_background_checks() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bilinear,
        4,
        4,
    )
    .with_background(17.0);
    let input_pixels: Vec<u8> = (0u8..64).collect();
    let input = Tile::<U8>::new(Region::new(0, 0, 4, 4), 4, &input_pixels);

    assert!(op.is_axis_aligned());
    assert_eq!(Affine::<U8>::bilinear_fixed_weight(0.5), 1 << 14);
    assert_eq!(
        Affine::<U8>::bilinear_u8_coefficients(0, 0),
        (
            Affine::<U8>::BILINEAR_FIXED_SCALE * Affine::<U8>::BILINEAR_FIXED_SCALE,
            0,
            0,
            0,
        )
    );
    assert_eq!(
        Affine::<U8>::bilinear_u8_channel(10, 20, 30, 40, 1, 0, 0, 0),
        0
    );

    let mut nearest = [0u8; 4];
    op.sample_pixel_nearest_into(&input, f64::INFINITY, 0.0, &mut nearest);
    assert_eq!(nearest, [17; 4]);

    let mut bilinear = [0u8; 4];
    op.sample_pixel_bilinear_into(&input, 10.0, 10.0, &mut bilinear);
    assert_eq!(bilinear, [17; 4]);
    assert!(op.output_region_is_background_only(&input, &Region::new(20, 20, 1, 1)));
}

#[test]
fn resolved_pixel_base_rejects_source_coords_outside_current_tile_region() {
    let op = Affine::<U8>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Nearest,
        4,
        4,
    )
    .with_extend(ExtendMode::Edge)
    .with_source_bounds(Region::new(0, 0, 4, 4));
    let input = Tile::<U8>::new(Region::new(1, 1, 2, 2), 1, &[5u8, 6, 7, 8]);

    assert_eq!(op.resolve_sample_coords(&input, 0, 0), Some((0, 0)));
    assert_eq!(op.resolved_pixel_base(&input, 0, 0), None);
}

#[test]
fn affine_bicubic_fast_path_falls_back_to_sample_pixel_at_at_edges() {
    let op = Affine::<F64>::new(
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        0.0,
        InterpolationKernel::Bicubic,
        3,
        3,
    )
    .with_background(-1.0);
    let region = Region::new(0, 0, 4, 4);
    let input_data: Vec<f64> = (0..16).map(f64::from).collect();
    let input = Tile::<F64>::new(region, 1, &input_data);
    let output_region = Region::new(0, 0, 3, 3);
    let mut output_data = vec![0.0f64; output_region.pixel_count()];
    let mut output = TileMut::<F64>::new(output_region, 1, &mut output_data);
    let mut state = ();

    op.process_region(&mut state, &input, &mut output);

    assert!(output_data.iter().all(|value| value.is_finite()));
    assert_eq!(output_data[0], input_data[0]);
}

proptest! {
    #[test]
    fn identity_matrix_preserves_random_pixels(
        width in 1u32..=16,
        height in 1u32..=16,
        pixels in prop::collection::vec(any::<u8>(), 1..=256),
    ) {
        let expected_len = (width * height) as usize;
        prop_assume!(pixels.len() >= expected_len);
        let input = pixels[..expected_len].to_vec();
        let region = Region::new(0, 0, width, height);
        let result = run_affine_u8(
            &input,
            region,
            region,
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            InterpolationKernel::Lanczos3,
        );

        prop_assert_eq!(result, input);
    }

    /// A uniform image through identity transform must return the same uniform value
    /// for all four kernels.
    #[test]
    fn uniform_identity_all_kernels(
        val in 0u8..=200u8,
        size in 2u32..=8u32,
        kernel in prop_oneof![
            Just(InterpolationKernel::Nearest),
            Just(InterpolationKernel::Bilinear),
            Just(InterpolationKernel::Bicubic),
            Just(InterpolationKernel::Lbb),
            Just(InterpolationKernel::CatmullRom),
            Just(InterpolationKernel::Nohalo),
            Just(InterpolationKernel::Lanczos3),
        ],
    ) {
        let data = vec![val; (size * size) as usize];
        let region = Region::new(0, 0, size, size);
        let result = run_affine_u8(
            &data,
            region,
            region,
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            kernel,
        );
        for (i, &got) in result.iter().enumerate() {
            prop_assert_eq!(
                got, val,
                "uniform identity kernel={:?} pixel {}: expected {}, got {}",
                kernel, i, val, got
            );
        }
    }

    #[test]
    fn lbb_output_is_locally_bounded(
        samples in prop::array::uniform16(-32.0f64..32.0),
        fx in 0.0f64..1.0,
        fy in 0.0f64..1.0,
    ) {
        let input = Tile::<F64>::new(Region::new(0, 0, 4, 4), 1, &samples);
        let op = Affine::<F64>::new([1.0, 0.0, 0.0, 1.0], 0.0, 0.0, InterpolationKernel::Lbb, 1, 1);

        let value = op.interp_lbb(&input, 1.0 + fx, 1.0 + fy, 0);
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        prop_assert!(value >= min - 1e-9, "value={value} min={min} fx={fx} fy={fy}");
        prop_assert!(value <= max + 1e-9, "value={value} max={max} fx={fx} fy={fy}");
    }

    #[test]
    fn nohalo_uniform_identity_preserves_constant_field(
        value in -32.0f64..32.0,
        size in 2u32..=8u32,
    ) {
        let data = vec![value; (size * size) as usize];
        let region = Region::new(0, 0, size, size);
        let result = run_affine_f64(
            &data,
            region,
            region,
            [1.0, 0.0, 0.0, 1.0],
            0.0,
            0.0,
            InterpolationKernel::Nohalo,
        );

        for (index, got) in result.iter().enumerate() {
            prop_assert!(
                (*got - value).abs() < 1e-9,
                "index={index} expected={value} got={got}",
            );
        }
    }

    #[test]
    fn nohalo_output_is_locally_bounded(
        samples in prop::collection::vec(-32.0f64..32.0, 36),
        fx in 0.0f64..1.0,
        fy in 0.0f64..1.0,
    ) {
        let input = Tile::<F64>::new(Region::new(0, 0, 6, 6), 1, &samples);
        let op = Affine::<F64>::new([1.0, 0.0, 0.0, 1.0], 0.0, 0.0, InterpolationKernel::Nohalo, 1, 1);

        let value = op.interp_nohalo(&input, 2.0 + fx, 2.0 + fy, 0);
        let (min, max) = nohalo_local_bounds(&samples, fx, fy);

        prop_assert!(value >= min - 1e-9, "value={value} min={min} fx={fx} fy={fy}");
        prop_assert!(value <= max + 1e-9, "value={value} max={max} fx={fx} fy={fy}");
    }

    /// Integer translation: tx=dx, ty=dy with identity matrix shifts input coords.
    /// Output pixel at (x, y) maps to input at (x+dx, y+dy). If the translated
    /// coordinate is inside the input tile, the value must match.
    #[test]
    fn integer_translation_correct(
        val in 10u8..=200u8,
        dx in 0i32..=3i32,
        dy in 0i32..=3i32,
    ) {
        // 8×8 input tile; output 4×4 at (0,0).
        // Translation (dx, dy): out pixel (x,y) reads input at (x+dx, y+dy).
        let in_size = 8u32;
        let out_size = 4u32;
        let mut data = vec![0u8; (in_size * in_size) as usize];
        // Fill the region that will be sampled (dx..dx+4, dy..dy+4) with `val`.
        for row in (dy as u32)..(dy as u32 + out_size) {
            for col in (dx as u32)..(dx as u32 + out_size) {
                data[(row * in_size + col) as usize] = val;
            }
        }
        let in_region = Region::new(0, 0, in_size, in_size);
        let out_region = Region::new(0, 0, out_size, out_size);
        let result = run_affine_u8(
            &data,
            in_region,
            out_region,
            [1.0, 0.0, 0.0, 1.0],
            dx as f64,
            dy as f64,
            InterpolationKernel::Nearest,
        );
        for (i, &got) in result.iter().enumerate() {
            prop_assert_eq!(
                got, val,
                "translation dx={} dy={} pixel {}: expected {}, got {}",
                dx, dy, i, val, got
            );
        }
    }

    /// libvips floors NN source coordinates. A half-pixel translation must not
    /// shift samples one pixel to the right.
    #[test]
    fn nearest_half_pixel_translation_uses_floor(size in 2u32..=16u32) {
        let in_width = size + 1;
        let in_height = size;
        let mut data = vec![0u8; (in_width * in_height) as usize];
        for row in 0..in_height {
            for col in 0..in_width {
                data[(row * in_width + col) as usize] = col as u8;
            }
        }

        let in_region = Region::new(0, 0, in_width, in_height);
        let out_region = Region::new(0, 0, size, size);
        let result = run_affine_u8(
            &data,
            in_region,
            out_region,
            [1.0, 0.0, 0.0, 1.0],
            0.5,
            0.0,
            InterpolationKernel::Nearest,
        );

        for row in 0..size {
            for col in 0..size {
                let idx = (row * size + col) as usize;
                prop_assert_eq!(
                    result[idx],
                    col as u8,
                    "half-pixel NN translation must floor source coords at row={} col={}",
                    row,
                    col,
                );
            }
        }
    }
}
