use super::support as golden;

#[allow(dead_code)]
use bytemuck::cast_slice;
use std::{mem::size_of, process::Command};
use viprs::{
    BuildError, OperationBridge, PipelineBuilder, TileScheduler,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        format::{F32, U8},
        ops::{
            convolution::{Canny, ConvSep, Sharpen, Sobel, gauss_blur::gaussian_kernel_1d},
            morphology::{Dilate, Erode, LabelRegionsOp, RankOp},
        },
    },
};

use golden::{ImageSpec, VipsBandFormat};

fn ensure_vips() {
    golden::require_vips();
}

fn run_pipeline_u8<S: viprs::pipeline::Flush>(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
) -> Vec<u8> {
    let source = MemorySource::<U8>::new(width, height, bands, source_pixels).unwrap();
    let pipeline = configure(PipelineBuilder::from_source(source))
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    sink.into_buffer()
}

fn run_pipeline_f32<S: viprs::pipeline::Flush>(
    source_pixels: Vec<f32>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
) -> Vec<u8> {
    let source = MemorySource::<F32>::new(width, height, bands, source_pixels).unwrap();
    let pipeline = configure(PipelineBuilder::from_source(source))
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
    sink.into_buffer()
}

fn write_u8_input_spec(op: &str, case: &str, name: &str, pixels: &[u8], spec: ImageSpec) -> String {
    golden::write_vips_input(op, case, name, pixels, spec)
        .display()
        .to_string()
}

fn write_u8_input(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
) -> String {
    write_u8_input_spec(
        op,
        case,
        name,
        pixels,
        ImageSpec::new(width, height, 1, VipsBandFormat::U8),
    )
}

fn write_f32_input(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[f32],
    width: u32,
    height: u32,
) -> String {
    golden::write_vips_input(
        op,
        case,
        name,
        cast_slice(pixels),
        ImageSpec::new(width, height, 1, VipsBandFormat::F32),
    )
    .display()
    .to_string()
}

fn rgb_gray_gradient_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let value = ((x * 13 + y * 7 + 19) % 256) as u8;
            pixels.extend_from_slice(&[value, value, value]);
        }
    }
    pixels
}

fn step_edge_source(width: u32, height: u32, edge_x: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in edge_x..width {
            pixels[(y * width + x) as usize] = 255;
        }
    }
    pixels
}

fn convsep_source(width: u32, height: u32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| {
            (0..width).map(move |x| x as f32 * 1.75 - y as f32 * 2.25 + (x * y) as f32 * 0.5)
        })
        .collect()
}

fn morphology_source() -> Vec<u8> {
    vec![
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 0, 255, 0, 0, 255, 0, 0, //
        0, 255, 255, 255, 0, 255, 255, 0, //
        0, 0, 255, 0, 0, 0, 255, 0, //
        0, 0, 0, 0, 255, 0, 0, 0, //
        0, 255, 0, 255, 255, 255, 0, 0, //
        0, 0, 255, 0, 255, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, //
    ]
}

fn rank_source() -> Vec<u8> {
    vec![
        10, 10, 200, 10, 10, 10, 10, //
        10, 255, 30, 40, 255, 10, 10, //
        10, 20, 80, 90, 25, 220, 10, //
        10, 35, 95, 5, 105, 45, 10, //
        10, 255, 50, 115, 60, 255, 10, //
        10, 10, 210, 55, 215, 10, 10, //
        10, 10, 10, 10, 10, 10, 10, //
    ]
}

fn labelregions_source() -> Vec<u8> {
    vec![
        0, 0, 0, 0, 0, 0, 0, 0, //
        0, 255, 255, 0, 0, 0, 255, 0, //
        0, 255, 255, 0, 0, 0, 255, 0, //
        0, 0, 0, 0, 255, 255, 0, 0, //
        0, 255, 0, 0, 255, 255, 0, 0, //
        0, 255, 0, 0, 0, 0, 0, 0, //
        0, 0, 0, 255, 255, 0, 255, 255, //
        0, 0, 0, 255, 255, 0, 255, 255, //
    ]
}

fn run_vips_command(args: &[&str]) {
    let output = Command::new("/opt/homebrew/bin/vips")
        .args(args)
        .output()
        .unwrap_or_else(|err| panic!("failed to run vips {:?}: {err}", args));
    assert!(
        output.status.success(),
        "vips {:?} failed\nstdout:\n{}\nstderr:\n{}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_interpreted_u8_input(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[u8],
    spec: ImageSpec,
    interpretation: &str,
) -> String {
    let raw = write_u8_input_spec(op, case, name, pixels, spec);
    let interpreted = golden::case_dir(op, case).join(format!("{name}-{interpretation}.v"));
    let interpreted_string = interpreted.display().to_string();
    run_vips_command(&[
        "copy",
        raw.as_str(),
        interpreted_string.as_str(),
        "--interpretation",
        interpretation,
    ]);
    interpreted_string
}

fn write_gaussmat_mask(op: &str, case: &str, sigma: f32) -> String {
    let mask_path = golden::case_dir(op, case).join("mask.v");
    let mask_string = mask_path.display().to_string();
    let sigma_arg = sigma.to_string();
    run_vips_command(&[
        "gaussmat",
        mask_string.as_str(),
        sigma_arg.as_str(),
        "0.2",
        "--separable",
        "--precision",
        "integer",
    ]);
    mask_string
}

fn crop_bytes(
    bytes: &[u8],
    width: u32,
    bands: u32,
    left: u32,
    top: u32,
    crop_w: u32,
    crop_h: u32,
    bytes_per_sample: usize,
) -> Vec<u8> {
    let row_bytes = width as usize * bands as usize * bytes_per_sample;
    let crop_row_bytes = crop_w as usize * bands as usize * bytes_per_sample;
    let start_x = left as usize * bands as usize * bytes_per_sample;
    let mut cropped = Vec::with_capacity(crop_row_bytes * crop_h as usize);

    for row in top as usize..(top + crop_h) as usize {
        let row_start = row * row_bytes + start_x;
        cropped.extend_from_slice(&bytes[row_start..row_start + crop_row_bytes]);
    }

    cropped
}

fn u8_to_f32_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut converted = Vec::with_capacity(bytes.len() * size_of::<f32>());
    for &value in bytes {
        converted.extend_from_slice(&f32::from(value).to_le_bytes());
    }
    converted
}

fn ordered_f32_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

fn assert_f32_bytes_ulp(actual: &[u8], expected: &[u8], max_ulp: u32) {
    assert_eq!(actual.len(), expected.len(), "f32 output length mismatch");
    for (idx, (got, want)) in actual
        .chunks_exact(size_of::<f32>())
        .zip(expected.chunks_exact(size_of::<f32>()))
        .enumerate()
    {
        let got = f32::from_le_bytes(got.try_into().expect("actual f32 bytes"));
        let want = f32::from_le_bytes(want.try_into().expect("expected f32 bytes"));
        let ulp = ordered_f32_bits(got).abs_diff(ordered_f32_bits(want));
        assert!(
            ulp <= max_ulp,
            "sample {idx}: viprs={got} libvips={want} ulp={ulp} > {max_ulp}"
        );
    }
}

fn assert_f32_matches_u8(actual: &[u8], expected: &[u8], max_abs_diff: f32) {
    assert_eq!(
        actual.len(),
        expected.len() * size_of::<f32>(),
        "u8/f32 output length mismatch"
    );
    for (idx, (got, want)) in actual
        .chunks_exact(size_of::<f32>())
        .zip(expected)
        .enumerate()
    {
        let got = f32::from_le_bytes(got.try_into().expect("actual f32 bytes"));
        let want = f32::from(*want);
        let diff = (got - want).abs();
        assert!(
            diff <= max_abs_diff,
            "sample {idx}: viprs={got} libvips={want} abs_diff={diff} > {max_abs_diff}"
        );
    }
}

#[test]
fn sharpen_libvips_gradient_rgb_center_crop() {
    ensure_vips();

    let width = 12;
    let height = 10;
    let bands = 3;
    let case = "rgb_gradient_center_crop";
    let source = rgb_gray_gradient_source(width, height);
    let actual = run_pipeline_u8(source.clone(), width, height, bands, |builder| {
        builder.then(Box::new(OperationBridge::new(
            Sharpen::<U8>::new(0.5, 3.0),
            bands,
        )))
    });
    let actual_cropped = crop_bytes(&actual, width, bands, 1, 1, width - 2, height - 2, 4);
    let input = write_interpreted_u8_input(
        "sharpen_libvips",
        case,
        "input",
        &source,
        ImageSpec::new(width, height, bands, VipsBandFormat::U8),
        "srgb",
    );
    let expected = golden::generate_vips_golden(
        "sharpen_libvips",
        case,
        &["sharpen", input.as_str(), "{output}"],
    );
    let expected_cropped = crop_bytes(&expected, width, bands, 1, 1, width - 2, height - 2, 1);

    assert_f32_bytes_ulp(&actual_cropped, &u8_to_f32_bytes(&expected_cropped), 1);
}

#[test]
fn sobel_libvips_step_edge() {
    ensure_vips();

    let width = 16;
    let height = 10;
    let case = "step_edge_u8";
    let source = step_edge_source(width, height, width / 2);
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(Sobel::<U8>::new(), 1)))
    });
    let input = write_u8_input("sobel_libvips", case, "input", &source, width, height);
    let expected = golden::generate_vips_golden(
        "sobel_libvips",
        case,
        &["sobel", input.as_str(), "{output}"],
    );
    let actual_cropped = crop_bytes(&actual, width, 1, 1, 1, width - 2, height - 2, 4);
    let expected_cropped = crop_bytes(&expected, width, 1, 1, 1, width - 2, height - 2, 1);

    assert_f32_matches_u8(&actual_cropped, &expected_cropped, 1.0);
}

#[test]
fn canny_libvips_step_edge_sigma_1_4() {
    ensure_vips();

    let width = 16;
    let height = 10;
    let case = "step_edge_sigma_1_4";
    let source = step_edge_source(width, height, width / 2);
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(Canny::<U8>::new(1.4), 1)))
    });
    let input = write_u8_input("canny_libvips", case, "input", &source, width, height);
    let expected = golden::generate_vips_golden(
        "canny_libvips",
        case,
        &["canny", input.as_str(), "{output}", "--sigma", "1.4"],
    );

    assert_f32_bytes_ulp(&actual, &expected, 1);
}

#[test]
fn convsep_libvips_matches_gaussblur() {
    ensure_vips();

    let width = 9;
    let height = 7;
    let case = "gaussian_sigma_1_0";
    let source = convsep_source(width, height);
    let convsep = ConvSep::new(gaussian_kernel_1d(1.0)).unwrap();
    let actual = run_pipeline_f32(source.clone(), width, height, 1, |builder| {
        builder
            .then(Box::new(OperationBridge::new(convsep.h, 1)))?
            .then(Box::new(OperationBridge::new(convsep.v, 1)))
    });
    let input = write_f32_input("convsep_libvips", case, "input", &source, width, height);
    let _mask = write_gaussmat_mask("convsep_libvips", case, 1.0);
    let expected = golden::generate_vips_golden(
        "convsep_libvips",
        case,
        &["gaussblur", input.as_str(), "{output}", "1.0"],
    );

    assert_f32_bytes_ulp(&actual, &expected, 1);
}

#[test]
fn dilate_libvips_rect_3x3() {
    ensure_vips();

    let width = 8;
    let height = 8;
    let case = "rect_3x3_binary";
    let source = morphology_source();
    let mask = [255u8; 9];
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(Dilate::rect(3).unwrap(), 1)))
    });
    let input = write_u8_input("dilate_libvips", case, "input", &source, width, height);
    let mask_path = write_u8_input_spec(
        "dilate_libvips",
        case,
        "mask",
        &mask,
        ImageSpec::new(3, 3, 1, VipsBandFormat::U8),
    );
    let cmd = [
        "morph",
        input.as_str(),
        "{output}",
        mask_path.as_str(),
        "dilate",
    ];

    golden::assert_golden_libvips("dilate_libvips", case, &actual, &cmd);
}

#[test]
fn erode_libvips_rect_3x3() {
    ensure_vips();

    let width = 8;
    let height = 8;
    let case = "rect_3x3_binary";
    let source = morphology_source();
    let mask = [255u8; 9];
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(Erode::rect(3).unwrap(), 1)))
    });
    let input = write_u8_input("erode_libvips", case, "input", &source, width, height);
    let mask_path = write_u8_input_spec(
        "erode_libvips",
        case,
        "mask",
        &mask,
        ImageSpec::new(3, 3, 1, VipsBandFormat::U8),
    );
    let cmd = [
        "morph",
        input.as_str(),
        "{output}",
        mask_path.as_str(),
        "erode",
    ];

    golden::assert_golden_libvips("erode_libvips", case, &actual, &cmd);
}

#[test]
fn rank_libvips_median_rect_3x3() {
    ensure_vips();

    let width = 7;
    let height = 7;
    let case = "median_3x3_rank_4";
    let source = rank_source();
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(
            RankOp::<U8>::new(3, 3, 4).unwrap(),
            1,
        )))
    });
    let input = write_u8_input("rank_libvips", case, "input", &source, width, height);
    let cmd = ["rank", input.as_str(), "{output}", "3", "3", "4"];

    golden::assert_golden_libvips("rank_libvips", case, &actual, &cmd);
}

#[test]
fn labelregions_libvips_binary_components() {
    ensure_vips();

    let width = 8;
    let height = 8;
    let case = "binary_components";
    let source = labelregions_source();
    let actual = run_pipeline_u8(source.clone(), width, height, 1, |builder| {
        builder.then(Box::new(OperationBridge::new(LabelRegionsOp::new(), 1)))
    });
    let input = write_u8_input(
        "labelregions_libvips",
        case,
        "input",
        &source,
        width,
        height,
    );
    let cmd = ["labelregions", input.as_str(), "{output}"];

    golden::assert_golden_libvips("labelregions_libvips", case, &actual, &cmd);
}
