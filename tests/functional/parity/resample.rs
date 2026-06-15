use super::support::*;

// Upstream: test/test-suite/test_resample.py::TestResample::test_resize
#[test]
fn libvips_parity_upstream_resample_resize_half_bilinear() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_resize";
    let op = "libvips_parity_resample";
    let case = "test_resize_half_bilinear_rgb";
    let width = 16;
    let height = 12;
    let bands = 3;
    let source = rgb_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, bands, |builder| {
            builder.resize(Resize::new(0.5, 0.5, InterpolationKernel::Bilinear))
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let command = vec![
        "resize".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "0.5".to_string(),
        "--kernel".to_string(),
        "linear".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (8, 6),
        &actual,
        &command,
        0,
    );
}

// Upstream: test/test-suite/test_resample.py::TestResample::test_resize
#[test]
fn libvips_parity_upstream_resample_resize_geometry_corner_case() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_resize";
    let op = "libvips_parity_resample";
    let case = "test_resize_height_one_rounding";
    let width = 100;
    let height = 1;
    let source = smooth_grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, 1, |builder| {
            builder.resize(Resize::new(0.5, 0.5, InterpolationKernel::Bilinear))
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let command = vec![
        "resize".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "0.5".to_string(),
        "--kernel".to_string(),
        "linear".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (50, 1),
        &actual,
        &command,
        0,
    );
}

// Upstream: test/test-suite/test_resample.py::TestResample::test_reduce
#[test]
fn libvips_parity_upstream_resample_reduceh_lanczos3() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_reduce";
    let op = "libvips_parity_resample";
    let case = "test_reduceh_factor_1_5_lanczos3";
    let width = 8;
    let height = 6;
    let source = smooth_grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, 1, |builder| {
            builder.reduce_h(1.5, InterpolationKernel::Lanczos3)
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let command = vec![
        "reduceh".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.5".to_string(),
        "--kernel".to_string(),
        "lanczos3".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (5, 6),
        &actual,
        &command,
        0,
    );
}

// Upstream: test/test-suite/test_resample.py::TestResample::test_reduce
#[test]
fn libvips_parity_upstream_resample_reducev_lanczos3() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_reduce";
    let op = "libvips_parity_resample";
    let case = "test_reducev_factor_1_5_lanczos3";
    let width = 6;
    let height = 8;
    let source = smooth_grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, 1, |builder| {
            builder.reduce_v(1.5, InterpolationKernel::Lanczos3)
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let command = vec![
        "reducev".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.5".to_string(),
        "--kernel".to_string(),
        "lanczos3".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (6, 5),
        &actual,
        &command,
        0,
    );
}

// Upstream: test/test-suite/test_resample.py::TestResample::test_shrink
#[test]
fn libvips_parity_upstream_resample_shrink_factor4() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_shrink";
    let op = "libvips_parity_resample";
    let case = "test_shrink_factor4_rgb";
    let width = 16;
    let height = 16;
    let bands = 3;
    let source = rgb_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, bands, |builder| {
            builder.shrink_v(4)?.shrink_h(4)
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let command = vec![
        "shrink".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "4".to_string(),
        "4".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (4, 4),
        &actual,
        &command,
        0,
    );
}

// Upstream: test/test-suite/test_resample.py::TestResample::test_thumbnail
#[test]
fn libvips_parity_upstream_resample_thumbnail_width_target() {
    ensure_vips();

    let upstream = "test/test-suite/test_resample.py::TestResample::test_thumbnail";
    let op = "libvips_parity_resample";
    let case = "test_thumbnail_width_128_square";
    let width = 512;
    let height = 512;
    let bands = 1;
    let source = smooth_grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, bands, |builder| {
            builder.thumbnail(Thumbnail::new(
                ThumbnailTarget::Width(128),
                InterpolationKernel::Lanczos3,
            ))
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let command = vec![
        "thumbnail".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "128".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (128, 128),
        &actual,
        &command,
        16,
    );
}
