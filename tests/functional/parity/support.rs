#[allow(dead_code)]
#[path = "../golden/support.rs"]
mod golden;

pub(crate) use bytemuck::cast_slice;
use std::{
    fs,
    mem::size_of,
    path::{Path, PathBuf},
    process::Command,
};
pub(crate) use viprs::{
    Add, BuildError, Multiply, OperationBridge, PipelineBuilder, Subtract, TileScheduler,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Lab, SRgb},
        format::{F32, U8},
        kernel::InterpolationKernel,
        ops::{
            arithmetic::{Abs, Divide},
            convolution::{GaussBlur, GaussBlurH, GaussBlurV},
            resample::{Resize, Thumbnail, thumbnail::ThumbnailTarget},
        },
    },
};

pub(crate) use golden::{ImageSpec, VipsBandFormat};

pub(crate) const WIDTH: u32 = 8;
pub(crate) const HEIGHT: u32 = 8;
pub(crate) const OUTPUT_PLACEHOLDER: &str = "{output}";

pub(crate) fn ensure_vips() {
    golden::require_vips();
}

fn fixture_metadata(upstream: &str, op: &str, case: &str) -> String {
    format!("upstream: {upstream}\nop: {op}\ncase: {case}")
}

pub(crate) fn run_pipeline_u8<S: viprs::pipeline::Flush>(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
) -> (u32, u32, Vec<u8>) {
    let source =
        MemorySource::<U8>::new(width, height, bands, source_pixels).expect("MemorySource");
    let pipeline = configure(PipelineBuilder::from_source(source))
        .expect("pipeline step")
        .build()
        .expect("pipeline build");

    let output_width = pipeline.width;
    let output_height = pipeline.height;
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .expect("scheduler")
        .run(&pipeline, &mut sink)
        .expect("pipeline run");

    (output_width, output_height, sink.into_buffer())
}

pub(crate) fn run_pipeline_f32<S: viprs::pipeline::Flush>(
    source_pixels: Vec<f32>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
) -> (u32, u32, Vec<u8>) {
    let source =
        MemorySource::<F32>::new(width, height, bands, source_pixels).expect("MemorySource");
    let pipeline = configure(PipelineBuilder::from_source(source))
        .expect("pipeline step")
        .build()
        .expect("pipeline build");

    let output_width = pipeline.width;
    let output_height = pipeline.height;
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .expect("scheduler")
        .run(&pipeline, &mut sink)
        .expect("pipeline run");

    (output_width, output_height, sink.into_buffer())
}

pub(crate) fn grayscale_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 17 + y * 13 + 5) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn rgb_source(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 29 + y * 7 + 3) % 256) as u8);
            pixels.push(((x * 11 + y * 23 + 17) % 256) as u8);
            pixels.push(((x * 5 + y * 13 + 29) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn smooth_grayscale_source(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| ((x * 4 + y * 3) % 256) as u8))
        .collect()
}

pub(crate) fn signed_f32_source() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(x as f32 * 0.75 - y as f32 * 1.1 - 7.25);
        }
    }
    pixels
}

pub(crate) fn rhs_f32() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(((x % 5) as f32 + 1.0) * 0.5 + (y % 4) as f32 * 0.25);
        }
    }
    pixels
}

pub(crate) fn colour_lab_source() -> Vec<f32> {
    const PIXELS: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [53.232_883, 80.109_33, 67.220_02],
        [87.737_04, -86.184_64, 83.181_17],
        [32.302_586, 79.196_66, -107.863_686],
        [60.0, -20.0, 30.0],
        [75.0, 10.0, -40.0],
        [25.0, 40.0, 20.0],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn colour_xyz_source() -> Vec<f32> {
    const PIXELS: [[f32; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [0.950_47, 1.0, 1.088_83],
        [0.412_456_4, 0.212_672_9, 0.019_333_9],
        [0.357_576_1, 0.715_152_2, 0.119_192],
        [0.180_437_5, 0.072_175, 0.950_304_1],
        [0.203_44, 0.214_04, 0.233_09],
        [0.538_01, 0.787_33, 0.131_78],
        [0.114, 0.082, 0.401],
    ];

    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT * 3) as usize);
    for idx in 0..(WIDTH * HEIGHT) as usize {
        pixels.extend_from_slice(&PIXELS[idx % PIXELS.len()]);
    }
    pixels
}

pub(crate) fn gauss_source(width: u32, height: u32) -> Vec<f32> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| x as f32 * 1.75 - y as f32 * 2.25 + (x * y) as f32))
        .collect()
}

pub(crate) fn scale_f32_pixels(pixels: &[f32], factor: f32) -> Vec<f32> {
    pixels.iter().map(|value| value * factor).collect()
}

pub(crate) fn write_u8_input_spec(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[u8],
    spec: ImageSpec,
) -> String {
    golden::write_vips_input(op, case, name, pixels, spec)
        .display()
        .to_string()
}

pub(crate) fn write_f32_input_spec(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[f32],
    spec: ImageSpec,
) -> String {
    golden::write_vips_input(op, case, name, cast_slice(pixels), spec)
        .display()
        .to_string()
}

pub(crate) fn render_command(args: &[String]) -> String {
    let mut rendered = String::from("vips");
    for arg in args {
        rendered.push(' ');
        rendered.push_str(arg);
    }
    rendered
}

pub(crate) fn write_expected_raw(output_image: &Path, output_raw: &Path) {
    if !golden::fixtures_regeneration_requested() {
        return;
    }
    let output = Command::new("vips")
        .args([
            "rawsave",
            output_image.to_str().expect("expected image path utf8"),
            output_raw.to_str().expect("expected raw path utf8"),
        ])
        .output()
        .unwrap_or_else(|err| panic!("failed to run vips rawsave for {output_image:?}: {err}"));

    assert!(
        output.status.success(),
        "vips rawsave failed for {output_image:?}\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

pub(crate) fn generate_vips_expected(
    op: &str,
    case: &str,
    command: &[String],
) -> (Vec<u8>, String, PathBuf) {
    golden::require_vips();
    let output_image = golden::case_dir(op, case).join("expected.v");
    let output_raw = golden::case_dir(op, case).join("expected.raw");
    let resolved = command
        .iter()
        .map(|arg| {
            if arg == OUTPUT_PLACEHOLDER {
                output_image.display().to_string()
            } else {
                arg.clone()
            }
        })
        .collect::<Vec<_>>();

    if let Some(parent) = output_image.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|err| panic!("failed to create runtime dir {parent:?}: {err}"));
    }

    let output = Command::new("vips")
        .args(&resolved)
        .output()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", render_command(&resolved)));

    assert!(
        output.status.success(),
        "{} failed\nstdout:\n{}\nstderr:\n{}",
        render_command(&resolved),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    write_expected_raw(&output_image, &output_raw);

    (
        fs::read(&output_raw).unwrap_or_else(|err| {
            panic!("failed to read generated raw output {output_raw:?}: {err}")
        }),
        render_command(&resolved),
        output_image,
    )
}

pub(crate) fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
        .collect()
}

pub(crate) fn parity_metadata(
    upstream: &str,
    op: &str,
    case: &str,
    command: &str,
    output_image: &Path,
) -> String {
    format!(
        "upstream: {upstream}\nop: {op}\ncase: {case}\ncommand: {command}\nexpected_image: {}",
        output_image.display()
    )
}

pub(crate) fn assert_u8_parity(
    upstream: &str,
    op: &str,
    case: &str,
    actual_dims: (u32, u32),
    expected_dims: (u32, u32),
    actual: &[u8],
    command: &[String],
    max_diff: u8,
) {
    if golden::fixtures_regeneration_requested() {
        let (expected, rendered, output_image) = generate_vips_expected(op, case, command);
        golden::write_fixture(op, case, &expected);
        let metadata = parity_metadata(upstream, op, case, &rendered, &output_image);

        assert_eq!(
            actual_dims, expected_dims,
            "dimension mismatch\n{metadata}\nactual_dims={actual_dims:?}\nexpected_dims={expected_dims:?}"
        );
        assert_eq!(
            actual.len(),
            expected.len(),
            "byte length mismatch\n{metadata}\nactual_len={}\nexpected_len={}",
            actual.len(),
            expected.len()
        );

        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (i16::from(got) - i16::from(want)).unsigned_abs() as u8;
            assert!(
                diff <= max_diff,
                "byte mismatch at index {idx}\n{metadata}\nviprs={got}\nlibvips={want}\ndiff={diff}\nallowed_diff={max_diff}"
            );
        }
        return;
    }

    let expected = golden::read_fixture(op, case);
    let metadata = fixture_metadata(upstream, op, case);

    assert_eq!(
        actual_dims, expected_dims,
        "dimension mismatch\n{metadata}\nactual_dims={actual_dims:?}\nexpected_dims={expected_dims:?}"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "byte length mismatch\n{metadata}\nactual_len={}\nexpected_len={}",
        actual.len(),
        expected.len()
    );

    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (i16::from(got) - i16::from(want)).unsigned_abs() as u8;
        assert!(
            diff <= max_diff,
            "byte mismatch at index {idx}\n{metadata}\nviprs={got}\nfixture={want}\ndiff={diff}\nallowed_diff={max_diff}"
        );
    }
}

pub(crate) fn assert_f32_parity(
    upstream: &str,
    op: &str,
    case: &str,
    actual_dims: (u32, u32),
    expected_dims: (u32, u32),
    actual: &[u8],
    command: &[String],
    epsilon: f32,
) {
    if golden::fixtures_regeneration_requested() {
        let (expected, rendered, output_image) = generate_vips_expected(op, case, command);
        golden::write_fixture(op, case, &expected);
        let metadata = parity_metadata(upstream, op, case, &rendered, &output_image);
        let actual = decode_f32(actual);
        let expected = decode_f32(&expected);

        assert_eq!(
            actual_dims, expected_dims,
            "dimension mismatch\n{metadata}\nactual_dims={actual_dims:?}\nexpected_dims={expected_dims:?}"
        );
        assert_eq!(
            actual.len(),
            expected.len(),
            "sample length mismatch\n{metadata}\nactual_len={}\nexpected_len={}",
            actual.len(),
            expected.len()
        );

        for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(
                diff <= epsilon,
                "sample mismatch at index {idx}\n{metadata}\nviprs={got}\nlibvips={want}\ndiff={diff}\nallowed_diff={epsilon}"
            );
        }
        return;
    }

    let metadata = fixture_metadata(upstream, op, case);
    let actual = decode_f32(actual);
    let expected = decode_f32(&golden::read_fixture(op, case));

    assert_eq!(
        actual_dims, expected_dims,
        "dimension mismatch\n{metadata}\nactual_dims={actual_dims:?}\nexpected_dims={expected_dims:?}"
    );
    assert_eq!(
        actual.len(),
        expected.len(),
        "sample length mismatch\n{metadata}\nactual_len={}\nexpected_len={}",
        actual.len(),
        expected.len()
    );

    for (idx, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(
            diff <= epsilon,
            "sample mismatch at index {idx}\n{metadata}\nviprs={got}\nfixture={want}\ndiff={diff}\nallowed_diff={epsilon}"
        );
    }
}

// Upstream: test/test-suite/test_arithmetic.py::TestArithmetic::test_add
