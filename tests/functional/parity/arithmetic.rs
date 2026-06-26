use super::support::*;

#[test]
fn libvips_parity_upstream_arithmetic_add() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_add";
    let op = "libvips_parity_arithmetic";
    let case = "test_add_signed_plus_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let (width, height, actual) =
        run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Add::<F32>::new(rhs_for_op),
                1,
            )))
        });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let rhs_path = write_f32_input_spec(
        op,
        case,
        "rhs",
        &rhs,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "add".to_string(),
        input,
        rhs_path,
        OUTPUT_PLACEHOLDER.to_string(),
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

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_sub
#[test]
fn libvips_parity_upstream_arithmetic_subtract() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_sub";
    let op = "libvips_parity_arithmetic";
    let case = "test_sub_signed_minus_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let (width, height, actual) =
        run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Subtract::<F32>::new(rhs_for_op),
                1,
            )))
        });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let rhs_path = write_f32_input_spec(
        op,
        case,
        "rhs",
        &rhs,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "subtract".to_string(),
        input,
        rhs_path,
        OUTPUT_PLACEHOLDER.to_string(),
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

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_mul
#[test]
fn libvips_parity_upstream_arithmetic_multiply() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_mul";
    let op = "libvips_parity_arithmetic";
    let case = "test_mul_signed_times_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let (width, height, actual) =
        run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Multiply::<F32>::new(rhs_for_op),
                1,
            )))
        });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let rhs_path = write_f32_input_spec(
        op,
        case,
        "rhs",
        &rhs,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "multiply".to_string(),
        input,
        rhs_path,
        OUTPUT_PLACEHOLDER.to_string(),
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

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_div
#[test]
fn libvips_parity_upstream_arithmetic_divide() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_div";
    let op = "libvips_parity_arithmetic";
    let case = "test_div_signed_by_rhs";
    let source = signed_f32_source();
    let rhs = rhs_f32();
    let rhs_for_op = rhs.clone();
    let (width, height, actual) =
        run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, move |builder| {
            builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
                Divide::<F32>::new(rhs_for_op, WIDTH, 1),
                1,
            )))
        });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let rhs_path = write_f32_input_spec(
        op,
        case,
        "rhs",
        &rhs,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec![
        "divide".to_string(),
        input,
        rhs_path,
        OUTPUT_PLACEHOLDER.to_string(),
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

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_abs
#[test]
fn libvips_parity_upstream_arithmetic_abs() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_abs";
    let op = "libvips_parity_arithmetic";
    let case = "test_abs_signed_ramp";
    let source = signed_f32_source();
    let (width, height, actual) = run_pipeline_f32(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.append_dyn_op(Box::new(OperationBridge::new_pixel_local(
            Abs::<F32>::new(),
            1,
        )))
    });
    let input = write_f32_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    );
    let command = vec!["abs".to_string(), input, OUTPUT_PLACEHOLDER.to_string()];

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

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_invert
#[test]
fn libvips_parity_upstream_arithmetic_invert() {
    if skip_without_vips() {
        return;
    }

    let upstream = "test/test-suite/test_arithmetic.py::TestArithmetic::test_invert";
    let op = "libvips_parity_arithmetic";
    let case = "test_invert_grayscale_gradient";
    let source = grayscale_source(WIDTH, HEIGHT);
    let (width, height, actual) = run_pipeline_u8(source.clone(), WIDTH, HEIGHT, 1, |builder| {
        builder.plan_invert()
    });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::U8),
    );
    let command = vec!["invert".to_string(), input, OUTPUT_PLACEHOLDER.to_string()];

    assert_u8_parity(
        upstream,
        op,
        case,
        (width, height),
        (WIDTH, HEIGHT),
        &actual,
        &command,
        0,
    );
}
