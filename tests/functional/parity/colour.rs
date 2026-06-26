use super::support::*;

// Upstream: test/test-suite/test_resample.py::TestResample::test_thumbnail
#[test]
fn libvips_parity_upstream_resample_thumbnail_fit_box() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_resample.py::TestResample::test_thumbnail";
    let op = "libvips_parity_resample";
    let case = "test_thumbnail_fit_box_40x24";
    let width = 96;
    let height = 64;
    let bands = 1;
    let source = grayscale_source(width, height);
    let (out_width, out_height, actual) =
        run_pipeline_u8(source.clone(), width, height, bands, |builder| {
            builder.thumbnail_with(Thumbnail::new(
                ThumbnailTarget::FitBox {
                    width: 40,
                    height: 24,
                },
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
        "40".to_string(),
        "--height".to_string(),
        "24".to_string(),
    ];

    assert_u8_parity(
        upstream,
        op,
        case,
        (out_width, out_height),
        (36, 24),
        &actual,
        &command,
        20,
    );
}

// Upstream: test/test-suite/test_colour.py::TestColour::test_colourspace
#[test]
fn libvips_parity_upstream_colour_srgb_to_lab() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_colour.py::TestColour::test_colourspace";
    let op = "libvips_parity_colour";
    let case = "test_colourspace_srgb_to_lab";
    let source = rgb_source(WIDTH, HEIGHT);
    let (width, height, actual) = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Lab>()
    });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let command = vec![
        "colourspace".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "lab".to_string(),
        "--source-space".to_string(),
        "srgb".to_string(),
    ];

    assert_f32_parity(
        upstream,
        op,
        case,
        (width, height),
        (WIDTH, HEIGHT),
        &actual,
        &command,
        3e-2,
    );
}

// Upstream: test/test-suite/test_colour.py::TestColour::test_colourspace
#[test]
fn libvips_parity_upstream_colour_lab_to_srgb() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_colour.py::TestColour::test_colourspace";
    let op = "libvips_parity_colour";
    let case = "test_colourspace_lab_to_srgb";
    let source = colour_lab_source();
    let (width, height, actual) = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Lab)
            .colourspace::<SRgb>()
    });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let command = vec![
        "colourspace".to_string(),
        input,
        OUTPUT_PLACEHOLDER.to_string(),
        "srgb".to_string(),
        "--source-space".to_string(),
        "lab".to_string(),
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
