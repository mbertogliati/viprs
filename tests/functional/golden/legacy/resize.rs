use super::super::support as golden;
use super::{
    support::*,
    transforms::{assert_resize_matches_libvips, interleave_rgb_planes, psnr_u8, split_rgb_planes},
};

fn resize_channel_4x_manual_lanczos3(input: Vec<u8>, width: u32, height: u32) -> Vec<u8> {
    let plan =
        Resize::new(0.25, 0.25, InterpolationKernel::Lanczos3).into_pipeline_nodes(width, height);
    let mut current_width = width;
    let mut current_height = height;
    let mut current = input;

    for node in plan.nodes {
        match node {
            ResizeNode::ShrinkH { factor } => {
                let source = MemorySource::<U8>::new(current_width, current_height, 1, current)
                    .expect("source");
                let op = ShrinkH::<U8>::new(factor).unwrap();
                let output_region = Region::new(0, 0, current_width / factor, current_height);
                let input_region = op.required_input_region(&output_region);
                let mut input_bytes = vec![0u8; input_region.pixel_count()];
                source
                    .read_region(input_region, &mut input_bytes)
                    .expect("read");
                let mut output = vec![0u8; output_region.pixel_count()];
                let input_tile = Tile::<U8>::new(input_region, 1, &input_bytes);
                let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
                let mut state =
                    op.start_with_tile_and_bands(output_region.width, output_region.height, 1);
                op.process_region(&mut state, &input_tile, &mut output_tile);
                current = output;
                current_width /= factor;
            }
            ResizeNode::ShrinkV { factor } => {
                let source = MemorySource::<U8>::new(current_width, current_height, 1, current)
                    .expect("source");
                let op = ShrinkV::<U8>::new(factor).unwrap();
                let output_region = Region::new(0, 0, current_width, current_height / factor);
                let input_region = op.required_input_region(&output_region);
                let mut input_bytes = vec![0u8; input_region.pixel_count()];
                source
                    .read_region(input_region, &mut input_bytes)
                    .expect("read");
                let mut output = vec![0u8; output_region.pixel_count()];
                let input_tile = Tile::<U8>::new(input_region, 1, &input_bytes);
                let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
                let mut state = op.start_with_tile(output_region.width, output_region.height);
                op.process_region(&mut state, &input_tile, &mut output_tile);
                current = output;
                current_height /= factor;
            }
            ResizeNode::ReduceH { factor, kernel } => {
                let source = MemorySource::<U8>::new(current_width, current_height, 1, current)
                    .expect("source");
                let op = ReduceH::<U8>::new(factor, kernel)
                    .expect("reduceh")
                    .with_input_width(current_width);
                let output_region = Region::new(
                    0,
                    0,
                    ((current_width as f64 / factor).round().max(1.0)) as u32,
                    current_height,
                );
                let input_region = op.required_input_region(&output_region);
                let mut input_bytes = vec![0u8; input_region.pixel_count()];
                source
                    .read_region(input_region, &mut input_bytes)
                    .expect("read");
                let mut output = vec![0u8; output_region.pixel_count()];
                let input_tile = Tile::<U8>::new(input_region, 1, &input_bytes);
                let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
                let mut state = op.start_with_tile(output_region.width, output_region.height);
                op.process_region(&mut state, &input_tile, &mut output_tile);
                current = output;
                current_width = output_region.width;
            }
            ResizeNode::ReduceV { factor, kernel } => {
                let source = MemorySource::<U8>::new(current_width, current_height, 1, current)
                    .expect("source");
                let op = ReduceV::<U8>::new(factor, kernel)
                    .expect("reducev")
                    .with_input_height(current_height);
                let output_region = Region::new(
                    0,
                    0,
                    current_width,
                    ((current_height as f64 / factor).round().max(1.0)) as u32,
                );
                let input_region = op.required_input_region(&output_region);
                let mut input_bytes = vec![0u8; input_region.pixel_count()];
                source
                    .read_region(input_region, &mut input_bytes)
                    .expect("read");
                let mut output = vec![0u8; output_region.pixel_count()];
                let input_tile = Tile::<U8>::new(input_region, 1, &input_bytes);
                let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
                let mut state = op.start_with_tile(output_region.width, output_region.height);
                op.process_region(&mut state, &input_tile, &mut output_tile);
                current = output;
                current_height = output_region.height;
            }
            other => panic!("unexpected resize node in 4x downscale test: {other:?}"),
        }
    }

    current
}

#[test]
fn resize_nearest_small_image_libvips() {
    if skip_without_vips() {
        return;
    }

    assert_resize_matches_libvips(
        "smooth_gradient_nearest_2x",
        2.0,
        InterpolationKernel::Nearest,
    );
}

#[test]
fn resize_bilinear_small_image_libvips() {
    if skip_without_vips() {
        return;
    }

    assert_resize_matches_libvips(
        "smooth_gradient_linear_0_75x",
        0.75,
        InterpolationKernel::Bilinear,
    );
}

#[test]
fn resize_bilinear_small_image_libvips_with_cache() {
    if skip_without_vips() {
        return;
    }

    let width = 4;
    let height = 4;
    let scale = 0.75;
    let kernel = InterpolationKernel::Bilinear;
    let case = "smooth_gradient_linear_0_75x_cache";
    let source = smooth_grayscale_source(width, height);
    let (_, _, actual, _) =
        run_cached_pipeline_u8_twice(source.clone(), width, height, 1, |builder| {
            let builder = builder.resize(Resize::new(scale, scale, kernel))?;
            builder.cache_last_op(NonZeroUsize::new(64).unwrap())
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

#[test]
fn resize_bicubic_small_image_libvips() {
    if skip_without_vips() {
        return;
    }

    assert_resize_matches_libvips(
        "smooth_gradient_cubic_0_75x",
        0.75,
        InterpolationKernel::Bicubic,
    );
}

#[test]
fn resize_rgb_4x_downscale_psnr_against_libvips_default_gap() {
    if skip_without_vips() {
        return;
    }

    let width = 8;
    let height = 8;
    let bands = 3;
    let scale = 0.25;
    let case = "rgb_lanczos3_0_25x_psnr";
    let source = rgb_source(width, height);
    let [r, g, b] = split_rgb_planes(&source);
    let resized_r = resize_channel_4x_manual_lanczos3(r, width, height);
    let resized_g = resize_channel_4x_manual_lanczos3(g, width, height);
    let resized_b = resize_channel_4x_manual_lanczos3(b, width, height);
    let actual = interleave_rgb_planes(&resized_r, &resized_g, &resized_b);

    let input = write_u8_input_spec(
        "resize_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
    );
    let cmd = vec![
        "resize".to_string(),
        input,
        "{output}".to_string(),
        scale.to_string(),
        "--vscale".to_string(),
        scale.to_string(),
        "--kernel".to_string(),
        "lanczos3".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    let expected = golden::generate_vips_golden("resize_libvips", case, &cmd_refs);
    let psnr = psnr_u8(&actual, &expected);
    assert!(
        psnr >= 45.0,
        "PSNR against libvips is too low for resize 4x downscale: {psnr:.2} dB"
    );
}

#[test]
fn reduce_factor2_lanczos3_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 8;
    let height = 8;
    let case = "smooth_grayscale_factor2_lanczos3";
    let source = smooth_grayscale_source(width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.reduce(2.0, 2.0, InterpolationKernel::Lanczos3)
    });
    let input = write_u8_input_spec(
        "reduce_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = vec![
        "reduce".to_string(),
        input,
        "{output}".to_string(),
        "2.0".to_string(),
        "2.0".to_string(),
        "--kernel".to_string(),
        "lanczos3".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    let expected = golden::generate_vips_golden("reduce_libvips", case, &cmd_refs);
    golden::assert_golden_approx(&actual, &expected, 4);
}

fn assert_axis_reduce_matches_libvips(
    op: &str,
    case: &str,
    width: u32,
    height: u32,
    reduce_h: bool,
) {
    let source = grayscale_source(width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        if reduce_h {
            builder.reduce_h(1.5, InterpolationKernel::Bicubic)
        } else {
            builder.reduce_v(1.5, InterpolationKernel::Bicubic)
        }
    });
    let input = write_u8_input_spec(
        op,
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = vec![
        if reduce_h { "reduceh" } else { "reducev" }.to_string(),
        input,
        "{output}".to_string(),
        "1.5".to_string(),
        "--kernel".to_string(),
        "cubic".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips(op, case, &actual, &cmd_refs);
}

#[test]
fn reduceh_factor1_5_bicubic_libvips() {
    if skip_without_vips() {
        return;
    }

    assert_axis_reduce_matches_libvips("reduceh_libvips", "grayscale_factor1_5_cubic", 8, 6, true);
}

#[test]
fn reducev_factor1_5_bicubic_libvips() {
    if skip_without_vips() {
        return;
    }

    assert_axis_reduce_matches_libvips("reducev_libvips", "grayscale_factor1_5_cubic", 6, 8, false);
}

#[test]
fn subsample_non_point_3x2_libvips() {
    if skip_without_vips() {
        return;
    }

    let width = 9;
    let height = 8;
    let case = "grayscale_3x2";
    let source = grayscale_source(width, height);
    let (_, _, actual) = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.subsample(3, 2)
    });
    let input = write_u8_input_spec(
        "subsample_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    );
    let cmd = vec![
        "subsample".to_string(),
        input,
        "{output}".to_string(),
        "3".to_string(),
        "2".to_string(),
    ];
    let cmd_refs = cmd.iter().map(String::as_str).collect::<Vec<_>>();

    golden::assert_golden_libvips("subsample_libvips", case, &actual, &cmd_refs);
}
