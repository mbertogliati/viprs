use super::super::support as golden;
use super::support_core::*;
use super::support_inputs::*;

#[test]
fn flip_horizontal_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_flip_horizontal()
    });
    let input = write_u8_input("flip_horizontal_libvips", case, "input", &source);
    let cmd = ["flip", input.as_str(), "{output}", "horizontal"];

    golden::assert_golden_libvips("flip_horizontal_libvips", case, &actual, &cmd);
}

#[test]
fn flip_vertical_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_flip_vertical()
    });
    let input = write_u8_input("flip_vertical_libvips", case, "input", &source);
    let cmd = ["flip", input.as_str(), "{output}", "vertical"];

    golden::assert_golden_libvips("flip_vertical_libvips", case, &actual, &cmd);
}

#[test]
fn rotate90_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_rotate90()
    });
    let input = write_u8_input("rotate90_libvips", case, "input", &source);
    let cmd = ["rot", input.as_str(), "{output}", "d90"];

    golden::assert_golden_libvips("rotate90_libvips", case, &actual, &cmd);
}

#[test]
fn rotate180_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_rotate180()
    });
    let input = write_u8_input("rotate180_libvips", case, "input", &source);
    let cmd = ["rot", input.as_str(), "{output}", "d180"];

    golden::assert_golden_libvips("rotate180_libvips", case, &actual, &cmd);
}

#[test]
fn rotate270_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_rotate270()
    });
    let input = write_u8_input("rotate270_libvips", case, "input", &source);
    let cmd = ["rot", input.as_str(), "{output}", "d270"];

    golden::assert_golden_libvips("rotate270_libvips", case, &actual, &cmd);
}

#[test]
fn autorot_orientation_6_rgb_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 5;
    let height = 3;
    let bands = 3;
    let case = "orientation_6_rgb";
    let source = rgb_source(width, height);
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.append_dyn_op(Box::new(AutorotBridge::<U8>::new(width, height, bands, 6)))
    });
    let input = write_u8_input_spec(
        "autorot_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    // `.v` fixtures serialize metadata in XML, so patch the generated EXIF orientation before
    // invoking the libvips CLI to validate the metadata-driven autorot path.
    patch_vips_orientation_6(&input);
    let cmd = ["autorot", input.as_str(), "{output}"];

    golden::assert_golden_libvips("autorot_libvips", case, &actual, &cmd);
}

#[test]
fn replicate_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_2x3";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_replicate(2, 3)
    });
    let input = write_u8_input("replicate_libvips", case, "input", &source);
    let cmd = ["replicate", input.as_str(), "{output}", "2", "3"];

    golden::assert_golden_libvips("replicate_libvips", case, &actual, &cmd);
}

#[test]
fn subsample_non_point_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_2x3_non_point";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_subsample(2, 3)
    });
    let input = write_u8_input("subsample_libvips", case, "input", &source);
    let cmd = ["subsample", input.as_str(), "{output}", "2", "3"];

    golden::assert_golden_libvips("subsample_libvips", case, &actual, &cmd);
}

#[test]
fn gamma_default_2_4_boundary_pixels_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 4;
    let height = 2;
    let case = "boundary_pixels_default_2_4";
    let source = vec![0u8, u8::MAX, 0, u8::MAX, u8::MAX, 0, u8::MAX, 0];
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            GammaOp::<U8>::default(),
            1,
        )))
    });
    let input = write_u8_input_spec(
        "gamma_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = [
        "gamma",
        input.as_str(),
        "{output}",
        "--exponent",
        "0.4166666666666667",
    ];

    golden::assert_golden_libvips("gamma_libvips", case, &actual, &cmd);
}

#[test]
fn gamma_default_2_4_midtones_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 4;
    let height = 2;
    let case = "midtones_default_2_4";
    let source = vec![32u8, 64, 96, 128, 160, 192, 224, 255];
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            GammaOp::<U8>::default(),
            1,
        )))
    });
    let input = write_u8_input_spec(
        "gamma_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = [
        "gamma",
        input.as_str(),
        "{output}",
        "--exponent",
        "0.4166666666666667",
    ];

    golden::assert_golden_libvips("gamma_libvips", case, &actual, &cmd);
}

#[test]
fn zoom_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_2x3";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_zoom(2, 3)
    });
    let input = write_u8_input("zoom_libvips", case, "input", &source);
    let cmd = ["zoom", input.as_str(), "{output}", "2", "3"];

    golden::assert_golden_libvips("zoom_libvips", case, &actual, &cmd);
}

#[test]
fn embed_black_extend_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "offset_black";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_embed(12, 10, 2, 1, WIDTH, HEIGHT, ExtendMode::Black)
    });
    let input = write_u8_input("embed_libvips", case, "input", &source);
    let cmd = [
        "embed",
        input.as_str(),
        "{output}",
        "2",
        "1",
        "12",
        "10",
        "--extend",
        "black",
    ];

    golden::assert_golden_libvips("embed_libvips", case, &actual, &cmd);
}

#[test]
fn embed_copy_extend_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "offset_copy";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_embed(12, 10, 2, 1, WIDTH, HEIGHT, ExtendMode::Copy)
    });
    let input = write_u8_input("embed_libvips", case, "input", &source);
    let cmd = [
        "embed",
        input.as_str(),
        "{output}",
        "2",
        "1",
        "12",
        "10",
        "--extend",
        "copy",
    ];

    golden::assert_golden_libvips("embed_libvips", case, &actual, &cmd);
}

#[test]
fn embed_mirror_extend_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "offset_mirror";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_embed(12, 10, 2, 1, WIDTH, HEIGHT, ExtendMode::Mirror)
    });
    let input = write_u8_input("embed_libvips", case, "input", &source);
    let cmd = [
        "embed",
        input.as_str(),
        "{output}",
        "2",
        "1",
        "12",
        "10",
        "--extend",
        "mirror",
    ];

    golden::assert_golden_libvips("embed_libvips", case, &actual, &cmd);
}

#[test]
fn extract_area_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "center_crop";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_extract_area(1, 2, 4, 3)
    });
    let input = write_u8_input("extract_area_libvips", case, "input", &source);
    let cmd = ["crop", input.as_str(), "{output}", "1", "2", "4", "3"];

    golden::assert_golden_libvips("extract_area_libvips", case, &actual, &cmd);
}
