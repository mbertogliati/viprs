use super::super::support as golden;
use super::support_core::*;
use super::support_inputs::*;

#[test]
fn avg_reducer_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "rgb_gradient";
    let source = rgb_source(WIDTH, HEIGHT);
    let actual = run_reducer_u8(source.clone(), WIDTH, HEIGHT, 3, &AvgOp::new());
    let input = write_u8_input_spec(
        "avg_reducer_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let expected = if golden::fixtures_regeneration_requested() {
        let value = run_vips_scalar("avg", &input);
        write_f64_fixture("avg_reducer_libvips", case, value);
        value
    } else {
        read_f64_fixture("avg_reducer_libvips", case)
    };

    assert!((actual - expected).abs() < f64::EPSILON);
}

#[test]
fn deviate_reducer_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "rgb_gradient";
    let source = rgb_source(WIDTH, HEIGHT);
    let actual = run_reducer_u8(source.clone(), WIDTH, HEIGHT, 3, &DeviateOp::new());
    let input = write_u8_input_spec(
        "deviate_reducer_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let expected = if golden::fixtures_regeneration_requested() {
        let value = run_vips_scalar("deviate", &input);
        write_f64_fixture("deviate_reducer_libvips", case, value);
        value
    } else {
        read_f64_fixture("deviate_reducer_libvips", case)
    };

    assert!((actual - expected).abs() <= 1e-6);
}

#[test]
fn hist_find_reducer_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "rgb_gradient";
    let source = rgb_source(WIDTH, HEIGHT);
    let hist = run_reducer_u8(
        source.clone(),
        WIDTH,
        HEIGHT,
        3,
        &HistFindOp::for_format(3, None, u8::MAX as u32),
    );
    let actual = u64_bins_to_u32_bytes(&hist.bins);
    let input = write_u8_input_spec(
        "hist_find_reducer_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let cmd = ["hist_find", input.as_str(), "{output}"];

    golden::assert_golden_libvips("hist_find_reducer_libvips", case, &actual, &cmd);
}

#[test]
fn colourspace_srgb_to_lab_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "srgb_to_lab_rgb_gradient";
    let source = rgb_source(WIDTH, HEIGHT);
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Lab>()
    });
    let input = write_u8_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "lab",
        "--source-space",
        "srgb",
    ];

    assert_f32_golden_with_epsilon("colourspace_libvips", case, &actual, &cmd, 3e-2);
}

#[test]
fn colourspace_lab_to_srgb_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "lab_to_srgb_known_colours";
    let source = colour_lab_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Lab)
            .colourspace::<SRgb>()
    });
    let input = write_f32_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "srgb",
        "--source-space",
        "lab",
    ];

    assert_u8_golden_with_max_diff("colourspace_libvips", case, &actual, &cmd, 1);
}

#[test]
fn colourspace_srgb_to_xyz_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "srgb_to_xyz_rgb_gradient";
    let source = rgb_source(WIDTH, HEIGHT);
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Xyz>()
    });
    let input = write_u8_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "xyz",
        "--source-space",
        "srgb",
    ];

    assert_f32_golden_scaled("colourspace_libvips", case, &actual, &cmd, 0.01, 1e-4);
}

#[test]
fn colourspace_xyz_to_srgb_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "xyz_to_srgb_known_colours";
    let source = colour_xyz_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Xyz)
            .colourspace::<SRgb>()
    });
    let vips_source = scale_f32_pixels(&source, 100.0);
    let input = write_f32_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &vips_source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "srgb",
        "--source-space",
        "xyz",
    ];

    assert_u8_golden_with_max_diff("colourspace_libvips", case, &actual, &cmd, 1);
}

#[test]
fn colourspace_srgb_to_hsv_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "srgb_to_hsv_reference_palette";
    let source = colour_srgb_hsv_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::SRgb)
            .colourspace::<Hsv>()
    });
    let input = write_u8_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "hsv",
        "--source-space",
        "srgb",
    ];

    assert_hsv_golden_from_vips_u8("colourspace_libvips", case, &actual, &cmd, 1.5, 5e-3);
}

#[test]
fn colourspace_hsv_to_srgb_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "hsv_to_srgb_known_colours";
    let source = colour_hsv_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Hsv)
            .colourspace::<SRgb>()
    });
    let vips_source = encode_vips_hsv_input(&source);
    let input = write_u8_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &vips_source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::U8),
    );
    set_vips_interpretation(&input, "hsv");
    let cmd = ["colourspace", input.as_str(), "{output}", "srgb"];

    assert_u8_golden_with_max_diff("colourspace_libvips", case, &actual, &cmd, 4);
}

#[test]
fn colourspace_lab_to_xyz_d65_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "lab_to_xyz_d65_reference_pixels";
    let source = colour_lab_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Lab)
            .colourspace::<Xyz>()
    });
    let input = write_f32_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "xyz",
        "--source-space",
        "lab",
    ];

    assert_f32_golden_scaled("colourspace_libvips", case, &actual, &cmd, 0.01, 1e-4);
}

#[test]
fn colourspace_xyz_to_lab_d65_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "xyz_to_lab_d65_reference_pixels";
    let source = colour_xyz_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 3, |builder| {
        builder
            .with_colorspace(ColorspaceId::Xyz)
            .colourspace::<Lab>()
    });
    let vips_source = scale_f32_pixels(&source, 100.0);
    let input = write_f32_input_spec(
        "colourspace_libvips",
        case,
        "input",
        &vips_source,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let cmd = [
        "colourspace",
        input.as_str(),
        "{output}",
        "lab",
        "--source-space",
        "xyz",
    ];

    assert_f32_golden_with_epsilon("colourspace_libvips", case, &actual, &cmd, 2e-3);
}

#[test]
fn de00_known_pairs_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "sharma_reference_pairs";
    let (left, right, combined) = delta_e_known_pairs();
    let actual = run_pipeline_f32(combined, WIDTH, HEIGHT, 6, |builder| {
        builder.then(Box::new(OperationBridge::new_pixel_local(DE00, 6)))
    });
    let left_input = write_f32_input_spec(
        "de00_libvips",
        case,
        "left",
        &left,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let right_input = write_f32_input_spec(
        "de00_libvips",
        case,
        "right",
        &right,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    set_vips_interpretation(&left_input, "lab");
    set_vips_interpretation(&right_input, "lab");
    let cmd = [
        "dE00",
        left_input.as_str(),
        right_input.as_str(),
        "{output}",
    ];

    assert_f32_golden_with_epsilon("de00_libvips", case, &actual, &cmd, 5e-2);
}

#[test]
fn de76_known_pairs_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "sharma_reference_pairs";
    let (left, right, combined) = delta_e_known_pairs();
    let actual = run_pipeline_f32(combined, WIDTH, HEIGHT, 6, |builder| {
        builder.then(Box::new(OperationBridge::new_pixel_local(DE76, 6)))
    });
    let left_input = write_f32_input_spec(
        "de76_libvips",
        case,
        "left",
        &left,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    let right_input = write_f32_input_spec(
        "de76_libvips",
        case,
        "right",
        &right,
        ImageSpec::new(WIDTH, HEIGHT, 3, VipsBandFormat::F32),
    );
    set_vips_interpretation(&left_input, "lab");
    set_vips_interpretation(&right_input, "lab");
    let cmd = [
        "dE76",
        left_input.as_str(),
        right_input.as_str(),
        "{output}",
    ];

    assert_f32_golden_with_epsilon("de76_libvips", case, &actual, &cmd, 1e-5);
}
