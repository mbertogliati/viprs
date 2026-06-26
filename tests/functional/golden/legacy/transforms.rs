use super::super::support as golden;
use super::support::*;

#[test]
#[ignore = "see B-82: affine libvips parity gap is not closed yet"]
fn affine_rotate30_scale_half_gradient_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 32;
    let height = 24;
    let bands = 3;
    let case = "rgb_rotate30_scale0_5_bilinear";
    let kernel = InterpolationKernel::Bilinear;
    let source = rgb_source(width, height);
    let (matrix, tx, ty, output_width, output_height) =
        affine_rotate_scale_auto_canvas(0.5, 30.0, width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_affine(matrix, tx, ty, output_width, output_height, kernel)
    });
    let input = write_u8_input_spec(
        "affine_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let matrix_arg = format!(
        "{:.17},{:.17},{:.17},{:.17}",
        matrix[0], matrix[1], matrix[2], matrix[3]
    );
    let oarea = format!("0,0,{output_width},{output_height}");
    let idx = format!("{tx:.17}");
    let idy = format!("{ty:.17}");
    let cmd = vec![
        "affine".to_string(),
        input,
        "{output}".to_string(),
        matrix_arg,
        "--oarea".to_string(),
        oarea,
        "--idx".to_string(),
        idx,
        "--idy".to_string(),
        idy,
        "--interpolate".to_string(),
        interpolate_name(kernel).to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips("affine_libvips", case, &actual, &cmd_refs);
}

#[test]
#[ignore = "see B-82: affine libvips parity gap is not closed yet"]
fn affine_rotate30_scale_half_gradient_libvips_with_cache() {
    if skip_without_vips() {
        return;
    }

    let width = 32;
    let height = 24;
    let bands = 3;
    let case = "rgb_rotate30_scale0_5_bilinear_cache";
    let kernel = InterpolationKernel::Bilinear;
    let source = rgb_source(width, height);
    let (matrix, tx, ty, output_width, output_height) =
        affine_rotate_scale_auto_canvas(0.5, 30.0, width, height);
    let (_, _, actual, _) =
        run_cached_pipeline_u8_twice(source.clone(), width, height, bands, |builder| {
            let builder =
                builder.plan_affine(matrix, tx, ty, output_width, output_height, kernel)?;
            builder.cache_last_op(NonZeroUsize::new(64).unwrap())
        });
    let input = write_u8_input_spec(
        "affine_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let matrix_arg = format!(
        "{:.17},{:.17},{:.17},{:.17}",
        matrix[0], matrix[1], matrix[2], matrix[3]
    );
    let oarea = format!("0,0,{output_width},{output_height}");
    let idx = format!("{tx:.17}");
    let idy = format!("{ty:.17}");
    let cmd = vec![
        "affine".to_string(),
        input,
        "{output}".to_string(),
        matrix_arg,
        "--oarea".to_string(),
        oarea,
        "--idx".to_string(),
        idx,
        "--idy".to_string(),
        idy,
        "--interpolate".to_string(),
        interpolate_name(kernel).to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips("affine_libvips", case, &actual, &cmd_refs);
}

#[test]
#[ignore = "see B-82: similarity libvips parity gap is not closed yet"]
fn similarity_rotate45_scale0_75_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 40;
    let height = 28;
    let bands = 3;
    let case = "rgb_rotate45_scale0_75_bilinear";
    let kernel = InterpolationKernel::Bilinear;
    let source = rgb_source(width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_similarity(0.75, 45.0, kernel)
    });
    let input = write_u8_input_spec(
        "similarity_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = vec![
        "similarity".to_string(),
        input,
        "{output}".to_string(),
        "--scale".to_string(),
        "0.75".to_string(),
        "--angle".to_string(),
        "45".to_string(),
        "--interpolate".to_string(),
        interpolate_name(kernel).to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips("similarity_libvips", case, &actual, &cmd_refs);
}

#[test]
fn thumbnail_width128_preserves_square_aspect_ratio_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 512;
    let height = 512;
    let bands = 1;
    let case = "grayscale_512_to_width128";
    let source = smooth_grayscale_source(width, height);
    let (output_width, output_height, actual) =
        run_pipeline_u8(source.clone(), width, height, bands, |builder| {
            builder.plan_thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(128),
                InterpolationKernel::Lanczos3,
            ))
        });

    assert_eq!((output_width, output_height), (128, 128));

    let input = write_u8_input_spec(
        "thumbnail_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = vec![
        "thumbnail".to_string(),
        input,
        "{output}".to_string(),
        "128".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    let expected = golden::generate_vips_golden("thumbnail_libvips", case, &cmd_refs);
    golden::assert_golden_approx(&actual, &expected, 16);
}

#[test]
fn thumbnail_width128_preserves_square_aspect_ratio_libvips_with_cache() {
    if skip_without_vips() {
        return;
    }

    let width = 512;
    let height = 512;
    let bands = 1;
    let case = "grayscale_512_to_width128_cache";
    let source = smooth_grayscale_source(width, height);
    let (output_width, output_height, actual, _) =
        run_cached_pipeline_u8_twice(source.clone(), width, height, bands, |builder| {
            let builder = builder.plan_thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(128),
                InterpolationKernel::Lanczos3,
            ))?;
            builder.cache_last_op(NonZeroUsize::new(64).unwrap())
        });

    assert_eq!((output_width, output_height), (128, 128));

    let input = write_u8_input_spec(
        "thumbnail_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = vec![
        "thumbnail".to_string(),
        input,
        "{output}".to_string(),
        "128".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    let expected = golden::generate_vips_golden("thumbnail_libvips", case, &cmd_refs);
    golden::assert_golden_approx(&actual, &expected, 16);
}

pub(crate) fn assert_resize_matches_libvips(case: &str, scale: f64, kernel: InterpolationKernel) {
    let width = 4;
    let height = 4;
    let source = smooth_grayscale_source(width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.plan_resize(Resize::new(scale, scale, kernel))
    });
    let input = write_u8_input_spec(
        "resize_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = vec![
        "resize".to_string(),
        input,
        "{output}".to_string(),
        scale.to_string(),
        "--vscale".to_string(),
        scale.to_string(),
        "--kernel".to_string(),
        resize_kernel_name(kernel).to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips("resize_libvips", case, &actual, &cmd_refs);
}

pub(crate) fn psnr_u8(actual: &[u8], expected: &[u8]) -> f64 {
    assert_eq!(
        actual.len(),
        expected.len(),
        "PSNR inputs must have same length"
    );

    if actual.is_empty() {
        return f64::INFINITY;
    }

    let mse = actual
        .iter()
        .zip(expected.iter())
        .map(|(&got, &want)| {
            let diff = f64::from(got) - f64::from(want);
            diff * diff
        })
        .sum::<f64>()
        / actual.len() as f64;

    if mse == 0.0 {
        f64::INFINITY
    } else {
        20.0 * 255.0_f64.log10() - 10.0 * mse.log10()
    }
}

pub(crate) fn split_rgb_planes(interleaved: &[u8]) -> [Vec<u8>; 3] {
    let mut r = Vec::with_capacity(interleaved.len() / 3);
    let mut g = Vec::with_capacity(interleaved.len() / 3);
    let mut b = Vec::with_capacity(interleaved.len() / 3);
    for chunk in interleaved.chunks_exact(3) {
        r.push(chunk[0]);
        g.push(chunk[1]);
        b.push(chunk[2]);
    }
    [r, g, b]
}

pub(crate) fn interleave_rgb_planes(r: &[u8], g: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(r.len(), g.len());
    assert_eq!(r.len(), b.len());
    let mut out = Vec::with_capacity(r.len() * 3);
    for i in 0..r.len() {
        out.push(r[i]);
        out.push(g[i]);
        out.push(b[i]);
    }
    out
}
