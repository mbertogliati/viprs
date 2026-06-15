use super::support::*;

// Upstream: test/test-suite/test_colour.py::TestColour::test_colourspace
#[test]
fn libvips_parity_upstream_colour_xyz_to_lab() {
    ensure_vips();

    let upstream = "test/test-suite/test_colour.py::TestColour::test_colourspace";
    let op = "libvips_parity_colour";
    let case = "test_colourspace_xyz_to_lab";
    let source = colour_xyz_source();
    let (width, height, actual) = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Xyz)
            .colourspace::<Lab>()
    });
    let vips_source = scale_f32_pixels(&source, 100.0);
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &vips_source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let command = vec![
        "colourspace".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "lab".to_string(),
        "--source-space".to_string(),
        "xyz".to_string(),
    ];

    assert_f32_parity(
        upstream,
        op,
        case,
        (width, height),
        (WIDTH, HEIGHT),
        &actual,
        &command,
        2e-3,
    );
}

// Upstream: test/test-suite/test_convolution.py::TestConvolution::test_gaussblur
#[test]
fn libvips_parity_upstream_convolution_gaussblur_uniform_field() {
    ensure_vips();

    let upstream = "test/test-suite/test_convolution.py::TestConvolution::test_gaussblur";
    let op = "libvips_parity_convolution";
    let case = "test_gaussblur_uniform_field";
    let source = vec![37.5f32; (WIDTH * HEIGHT) as usize];
    let blur = GaussBlur::new(1.0);
    let (width, height, actual) = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder
            .then(Box::new(OperationBridge::new(blur.h, 1)))?
            .then(Box::new(OperationBridge::new(blur.v, 1)))
    });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "gaussblur".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.0".to_string(),
    ];

    assert_f32_parity(
        upstream,
        op,
        case,
        (width, height),
        (WIDTH, HEIGHT),
        &actual,
        &command,
        1e-6,
    );
}

// Upstream: test/test-suite/test_convolution.py::TestConvolution::test_gaussblur
#[test]
fn libvips_parity_upstream_convolution_gaussblur_u8_nonuniform_field() {
    ensure_vips();

    let upstream = "test/test-suite/test_convolution.py::TestConvolution::test_gaussblur";
    let op = "libvips_parity_convolution";
    let case = "test_gaussblur_u8_nonuniform_5x3";
    let width = 5;
    let height = 3;
    let source = grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, 1, |builder| {
            builder
                .then(Box::new(OperationBridge::new(
                    GaussBlurH::<U8>::new(1.0),
                    1,
                )))?
                .then(Box::new(OperationBridge::new(
                    GaussBlurV::<U8>::new(1.0),
                    1,
                )))
        });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let command = vec![
        "gaussblur".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.0".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (width, height),
        &actual,
        &command,
        1,
    );
}

// Upstream: test/test-suite/test_convolution.py::TestConvolution::test_gaussblur
#[test]
fn libvips_parity_upstream_convolution_gaussblur_u8_rgb_field() {
    ensure_vips();

    let upstream = "test/test-suite/test_convolution.py::TestConvolution::test_gaussblur";
    let op = "libvips_parity_convolution";
    let case = "test_gaussblur_u8_rgb_8x8";
    let source = rgb_source(WIDTH, HEIGHT);
    let (width, height, actual) = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .then(Box::new(OperationBridge::new(
                GaussBlurH::<U8>::new(1.0),
                3,
            )))?
            .then(Box::new(OperationBridge::new(
                GaussBlurV::<U8>::new(1.0),
                3,
            )))
    });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let command = vec![
        "gaussblur".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.0".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (width, height),
        (WIDTH, HEIGHT),
        &actual,
        &command,
        1,
    );
}

// Upstream: test/test-suite/test_convolution.py::TestConvolution::test_gaussblur
#[test]
fn libvips_parity_upstream_convolution_gaussblur_nonuniform_field() {
    ensure_vips();

    let upstream = "test/test-suite/test_convolution.py::TestConvolution::test_gaussblur";
    let op = "libvips_parity_convolution";
    let case = "test_gaussblur_nonuniform_5x3";
    let width = 5;
    let height = 3;
    let source = gauss_source(width, height);
    let blur = GaussBlur::new(1.0);
    let (out_width, out_height, actual) =
        run_pipeline_f32(source.clone(), width, height, 1, |builder| {
            builder
                .then(Box::new(OperationBridge::new(blur.h, 1)))?
                .then(Box::new(OperationBridge::new(blur.v, 1)))
        });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "gaussblur".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "1.0".to_string(),
    ];

    assert_f32_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (width, height),
        &actual,
        &command,
        1e-6,
    );
}
