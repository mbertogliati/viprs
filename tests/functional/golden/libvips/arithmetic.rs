use super::super::support as golden;
use super::support_core::*;
use super::support_inputs::*;

#[test]
fn abs_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_ramp";
    let source = signed_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Abs::<F32>::new(),
            1,
        )))
    });
    let input = write_f32_input("abs_libvips", case, "input", &source);
    let cmd = ["abs", input.as_str(), "{output}"];

    golden::assert_golden_libvips("abs_libvips", case, &actual, &cmd);
}

#[test]
fn round_rint_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "fractional_gradient_rint";
    let source = fractional_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new(Round::<F32>::new(), 1)))
    });
    let input = write_f32_input("round_libvips", case, "input", &source);
    let cmd = ["round", input.as_str(), "{output}", "rint"];

    golden::assert_golden_libvips("round_libvips", case, &actual, &cmd);
}

#[test]
fn round_ceil_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "fractional_gradient_ceil";
    let source = fractional_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new(Ceil::<F32>::new(), 1)))
    });
    let input = write_f32_input("round_libvips", case, "input", &source);
    let cmd = ["round", input.as_str(), "{output}", "ceil"];

    golden::assert_golden_libvips("round_libvips", case, &actual, &cmd);
}

#[test]
fn round_floor_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "fractional_gradient_floor";
    let source = fractional_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new(Floor::<F32>::new(), 1)))
    });
    let input = write_f32_input("round_libvips", case, "input", &source);
    let cmd = ["round", input.as_str(), "{output}", "floor"];

    golden::assert_golden_libvips("round_libvips", case, &actual, &cmd);
}

#[test]
fn sign_u8_nonzero_pixels_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_nonzero_pixels";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Sign::<U8>::new(),
            1,
        )))
    });
    let input = write_u8_input("sign_libvips", case, "input", &source);
    let cmd = ["sign", input.as_str(), "{output}"];

    golden::assert_golden_libvips("sign_libvips", case, &actual, &cmd);
}

#[test]
fn add_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_plus_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Add::<F32>::new(rhs_for_op),
            1,
        )))
    });
    let input = write_f32_input("add_libvips", case, "input", &source);
    let rhs_path = write_f32_input("add_libvips", case, "rhs", &rhs);
    let cmd = ["add", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("add_libvips", case, &actual, &cmd);
}

#[test]
fn subtract_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_minus_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Subtract::<F32>::new(rhs_for_op),
            1,
        )))
    });
    let input = write_f32_input("subtract_libvips", case, "input", &source);
    let rhs_path = write_f32_input("subtract_libvips", case, "rhs", &rhs);
    let cmd = ["subtract", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("subtract_libvips", case, &actual, &cmd);
}

#[test]
fn multiply_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_times_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Multiply::<F32>::new(rhs_for_op),
            1,
        )))
    });
    let input = write_f32_input("multiply_libvips", case, "input", &source);
    let rhs_path = write_f32_input("multiply_libvips", case, "rhs", &rhs);
    let cmd = ["multiply", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("multiply_libvips", case, &actual, &cmd);
}

#[test]
fn merge_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "horizontal_zero_overlap_u8";
    let reference = vec![
        10u8, 20, 30, 40, //
        50, 60, 70, 80,
    ];
    let secondary = vec![
        100u8, 110, 120, 130, //
        140, 150, 160, 170,
    ];
    let op = MergeH::<U8>::new(4, 2, 4, 2, 4, 0, 2, 1);
    let actual = run_two_input_u8(&op, &reference, &secondary, Region::new(0, 0, 8, 2));
    let reference_path = golden::write_vips_input(
        "merge_libvips",
        case,
        "reference",
        &reference,
        ImageSpec::new(4, 2, 1, VipsBandFormat::U8),
    )
    .display()
    .to_string();
    let secondary_path = golden::write_vips_input(
        "merge_libvips",
        case,
        "secondary",
        &secondary,
        ImageSpec::new(4, 2, 1, VipsBandFormat::U8),
    )
    .display()
    .to_string();
    let cmd = [
        "merge",
        reference_path.as_str(),
        secondary_path.as_str(),
        "{output}",
        "horizontal",
        "4",
        "0",
        "--mblend",
        "2",
    ];

    golden::assert_golden_libvips("merge_libvips", case, &actual, &cmd);
}

#[test]
fn divide_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_divided_by_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Divide::<F32>::new(rhs_for_op, WIDTH, 1),
            1,
        )))
    });
    let input = write_f32_input("divide_libvips", case, "input", &source);
    let rhs_path = write_f32_input("divide_libvips", case, "rhs", &rhs);
    let cmd = ["divide", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("divide_libvips", case, &actual, &cmd);
}

#[test]
fn clamp_signed_f32_range_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_gradient_range";
    let source = signed_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            ClampOp::<F32>::new(-3.25, 4.5),
            1,
        )))
    });
    let input = write_f32_input("clamp_libvips", case, "input", &source);
    let cmd = [
        "clamp",
        input.as_str(),
        "{output}",
        "--min",
        "-3.25",
        "--max",
        "4.5",
    ];

    golden::assert_golden_libvips("clamp_libvips", case, &actual, &cmd);
}

#[test]
fn remainder_i32_with_zero_divisors_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_i32_zero_divisors";
    let source = signed_i32_source();
    let rhs = rhs_i32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_i32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Remainder::<I32>::new(rhs_for_op),
            1,
        )))
    });
    let input = write_i32_input("remainder_libvips", case, "input", &source);
    let rhs_path = write_i32_input("remainder_libvips", case, "rhs", &rhs);
    let cmd = ["remainder", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("remainder_libvips", case, &actual, &cmd);
}

#[test]
fn remainder_f32_fractional_rhs_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_f32_fractional_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Remainder::<F32>::new(rhs_for_op),
            1,
        )))
    });
    let input = write_f32_input("remainder_libvips", case, "input", &source);
    let rhs_path = write_f32_input("remainder_libvips", case, "rhs", &rhs);
    let cmd = ["remainder", input.as_str(), rhs_path.as_str(), "{output}"];

    golden::assert_golden_libvips("remainder_libvips", case, &actual, &cmd);
}

#[test]
fn invert_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "grayscale_ramp";
    let source = grayscale_source();
    let actual = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_invert()
    });
    let input = write_u8_input("invert_libvips", case, "input", &source);
    let cmd = ["invert", input.as_str(), "{output}"];

    golden::assert_golden_libvips("invert_libvips", case, &actual, &cmd);
}

#[test]
fn linear_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "signed_scaled_offset";
    let source = signed_f32_source();
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_linear(1.75, -2.5)
    });
    let input = write_f32_input("linear_libvips", case, "input", &source);
    let cmd = ["linear", input.as_str(), "{output}", "1.75", "--", "-2.5"];

    golden::assert_golden_libvips("linear_libvips", case, &actual, &cmd);
}

#[test]
fn bandjoin_two_single_bands_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "two_single_band_gradients";
    let primary = grayscale_source();
    let secondary = secondary_u8_source();
    let op = BandJoin::new(1, 1, viprs::BandFormatId::U8);
    let actual = run_two_input_u8(&op, &primary, &secondary, Region::new(0, 0, WIDTH, HEIGHT));
    let primary_path = write_u8_input("bandjoin_libvips", case, "primary", &primary);
    let secondary_path = write_u8_input("bandjoin_libvips", case, "secondary", &secondary);
    let joined_inputs = format!("{primary_path} {secondary_path}");
    let cmd = ["bandjoin", joined_inputs.as_str(), "{output}"];

    golden::assert_golden_libvips("bandjoin_libvips", case, &actual, &cmd);
}

#[test]
fn bandmean_rgb_gradient_libvips() {
    if skip_without_vips() {
        return;
    }

    let case = "rgb_gradient";
    let width = 6;
    let height = 5;
    let source = rgb_source(width, height);
    let actual = run_bandmean_u8(&source, width, height, 3);
    let input = write_u8_input_spec(
        "bandmean_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 3, VipsBandFormat::U8),
    );
    let cmd = ["bandmean", input.as_str(), "{output}"];

    golden::assert_golden_libvips("bandmean_libvips", case, &actual, &cmd);
}
