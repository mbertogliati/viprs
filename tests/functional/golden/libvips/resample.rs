use super::super::support as golden;
use super::support_core::*;
use super::support_inputs::*;

#[test]
fn gauss_blur_libvips() {
    if skip_without_vips() {
        return;
    }

    // Non-uniform parity currently diverges from libvips; see B-91.
    let case = "uniform_field";
    let source = gauss_source();
    let blur = GaussBlur::new(1.0);
    let actual = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder
            .append_dyn_op(Box::new(OperationBridge::new(blur.h, 1)))?
            .append_dyn_op(Box::new(OperationBridge::new(blur.v, 1)))
    });
    let input = write_f32_input("gauss_blur_libvips", case, "input", &source);
    let cmd = ["gaussblur", input.as_str(), "{output}", "1.0"];

    golden::assert_golden_libvips("gauss_blur_libvips", case, &actual, &cmd);
}

#[test]
fn gauss_blur_libvips_nonuniform_non_square() {
    if skip_without_vips() {
        return;
    }

    let width = 5;
    let height = 3;
    let case = "nonuniform_5x3";
    let source: Vec<f32> = (0..height)
        .flat_map(|y| (0..width).map(move |x| x as f32 * 1.75 - y as f32 * 2.25 + (x * y) as f32))
        .collect();
    let blur = GaussBlur::new(1.0);
    let actual = run_pipeline_f32(source.clone(), width, height, 1, |builder| {
        builder
            .append_dyn_op(Box::new(OperationBridge::new(blur.h, 1)))?
            .append_dyn_op(Box::new(OperationBridge::new(blur.v, 1)))
    });
    let input = write_f32_input_spec(
        "gauss_blur_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::F32),
    );
    let cmd = ["gaussblur", input.as_str(), "{output}", "1.0"];

    golden::assert_golden_libvips("gauss_blur_libvips", case, &actual, &cmd);
}

#[test]
fn reduceh_linear_factor2_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 8;
    let height = 6;
    let bands = 3;
    let case = "rgb_factor2_linear";
    let source = rgb_source(width, height);
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_reduce_h(2.0, viprs::domain::kernel::InterpolationKernel::Bilinear)
    });
    let input = write_u8_input_spec(
        "reduceh_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = [
        "reduceh",
        input.as_str(),
        "{output}",
        "2.0",
        "--kernel",
        "linear",
    ];

    golden::assert_golden_libvips("reduceh_libvips", case, &actual, &cmd);
}

#[test]
fn reducev_linear_factor2_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 6;
    let height = 8;
    let bands = 3;
    let case = "rgb_factor2_linear";
    let source = rgb_source(width, height);
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_reduce_v(2.0, viprs::domain::kernel::InterpolationKernel::Bilinear)
    });
    let input = write_u8_input_spec(
        "reducev_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = [
        "reducev",
        input.as_str(),
        "{output}",
        "2.0",
        "--kernel",
        "linear",
    ];

    golden::assert_golden_libvips("reducev_libvips", case, &actual, &cmd);
}

#[test]
fn reduceh_lanczos3_factor1_5_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 8;
    let height = 6;
    let bands = 1;
    let case = "grayscale_factor1_5_lanczos3";
    let source = grayscale_source()[..(width * height) as usize].to_vec();
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_reduce_h(1.5, viprs::domain::kernel::InterpolationKernel::Lanczos3)
    });
    let input = write_u8_input_spec(
        "reduceh_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = [
        "reduceh",
        input.as_str(),
        "{output}",
        "1.5",
        "--kernel",
        "lanczos3",
    ];

    golden::assert_golden_libvips("reduceh_libvips", case, &actual, &cmd);
}

#[test]
fn reducev_lanczos3_factor1_5_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 6;
    let height = 8;
    let bands = 1;
    let case = "grayscale_factor1_5_lanczos3";
    let source = grayscale_source()[..(width * height) as usize].to_vec();
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_reduce_v(1.5, viprs::domain::kernel::InterpolationKernel::Lanczos3)
    });
    let input = write_u8_input_spec(
        "reducev_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = [
        "reducev",
        input.as_str(),
        "{output}",
        "1.5",
        "--kernel",
        "lanczos3",
    ];

    golden::assert_golden_libvips("reducev_libvips", case, &actual, &cmd);
}

#[test]
fn shrink_factor4_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 16;
    let height = 16;
    let bands = 3;
    let case = "rgb_factor4";
    let source = rgb_source(width, height);
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.plan_shrink_v(4)?.plan_shrink_h(4)
    });
    let input = write_u8_input_spec(
        "shrink_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = ["shrink", input.as_str(), "{output}", "4", "4"];

    golden::assert_golden_libvips("shrink_libvips", case, &actual, &cmd);
}

#[test]
fn reduce_4x4_edge_pixels_match_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 4;
    let height = 4;
    let bands = 1;
    let case = "edge_4x4_factor2_linear";
    let source: Vec<u8> = (0..(width * height) as u8).collect();
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder
            .plan_reduce_v(2.0, viprs::domain::kernel::InterpolationKernel::Bilinear)?
            .plan_reduce_h(2.0, viprs::domain::kernel::InterpolationKernel::Bilinear)
    });
    let input = write_u8_input_spec(
        "reduce_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = [
        "reduce",
        input.as_str(),
        "{output}",
        "2.0",
        "2.0",
        "--kernel",
        "linear",
    ];

    golden::assert_golden_libvips("reduce_libvips", case, &actual, &cmd);
}
