use super::super::support as golden;

pub(crate) use bytemuck::cast_slice;
use std::{fs, mem::size_of, process::Command};
pub(crate) use viprs::{
    Add, AvgOp, BuildError, DeviateOp, HistFindOp, Multiply, Op, OperationBridge, PipelineBuilder,
    Subtract, TileScheduler,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::{
        colorspace::{ColorspaceId, Hsv, Lab, SRgb, Xyz},
        format::{F32, I32, U8},
        image::Region,
        op::DynOperation,
        ops::{
            arithmetic::{Abs, Ceil, ClampOp, Divide, Floor, Remainder, Round, Sign},
            colour::{DE00, DE76},
            conversion::{AutorotBridge, BandJoin, BandMean, GammaOp, embed::ExtendMode},
            convolution::GaussBlur,
            mosaicing::MergeH,
        },
    },
    ports::scheduler::ReducingScheduler,
};

pub(crate) use golden::{ImageSpec, VipsBandFormat};

pub(crate) const WIDTH: u32 = 8;
pub(crate) const HEIGHT: u32 = 8;

pub(crate) fn skip_without_vips() -> bool {
    golden::skip_without_vips()
}

pub(crate) fn grayscale_source() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(((x * 17 + y * 13 + 5) % 256) as u8);
        }
    }
    pixels
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

pub(crate) fn fractional_f32_source() -> Vec<f32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(x as f32 * 0.5 - y as f32 * 0.75 - 3.125);
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

pub(crate) fn signed_i32_source() -> Vec<i32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(x as i32 * 7 - y as i32 * 5 - 23);
        }
    }
    pixels
}

pub(crate) fn rhs_i32() -> Vec<i32> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(match (x + y) % 5 {
                0 => 0,
                1 => 3,
                2 => -4,
                3 => 5,
                _ => -6,
            });
        }
    }
    pixels
}

pub(crate) fn secondary_u8_source() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((WIDTH * HEIGHT) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            pixels.push(((x * 9 + y * 21 + 11) % 256) as u8);
        }
    }
    pixels
}

pub(crate) fn gauss_source() -> Vec<f32> {
    vec![37.5; (WIDTH * HEIGHT) as usize]
}

pub(crate) fn run_pipeline_u8<S: viprs::pipeline::Flush>(
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

pub(crate) fn run_pipeline_f32<S: viprs::pipeline::Flush>(
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

pub(crate) fn run_pipeline_i32<S: viprs::pipeline::Flush>(
    source_pixels: Vec<i32>,
    width: u32,
    height: u32,
    bands: u32,
    configure: impl FnOnce(PipelineBuilder) -> Result<PipelineBuilder<S>, BuildError>,
) -> Vec<u8> {
    let source = MemorySource::<I32>::new(width, height, bands, source_pixels).unwrap();
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

pub(crate) fn write_u8_input(op: &str, case: &str, name: &str, pixels: &[u8]) -> String {
    write_u8_input_spec(
        op,
        case,
        name,
        pixels,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::U8),
    )
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

pub(crate) fn write_f32_input(op: &str, case: &str, name: &str, pixels: &[f32]) -> String {
    write_f32_input_spec(
        op,
        case,
        name,
        pixels,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    )
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

pub(crate) fn write_i32_input(op: &str, case: &str, name: &str, pixels: &[i32]) -> String {
    write_i32_input_spec(
        op,
        case,
        name,
        pixels,
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::I32),
    )
}

pub(crate) fn write_i32_input_spec(
    op: &str,
    case: &str,
    name: &str,
    pixels: &[i32],
    spec: ImageSpec,
) -> String {
    golden::write_vips_input(op, case, name, cast_slice(pixels), spec)
        .display()
        .to_string()
}

pub(crate) fn set_vips_interpretation(image_path: &str, interpretation: &str) {
    if !golden::fixtures_regeneration_requested() {
        return;
    }
    let rewritten = format!("{image_path}.{interpretation}.rewrite.v");
    let output = Command::new("vips")
        .args([
            "copy",
            image_path,
            rewritten.as_str(),
            "--interpretation",
            interpretation,
        ])
        .output()
        .unwrap_or_else(|err| panic!("failed to set interpretation {interpretation}: {err}"));
    assert!(
        output.status.success(),
        "vips copy --interpretation {interpretation} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    fs::rename(&rewritten, image_path).unwrap_or_else(|err| {
        panic!("failed to replace interpreted image {image_path} with {rewritten}: {err}")
    });
}

pub(crate) fn decode_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 bytes")))
        .collect()
}

pub(crate) fn assert_f32_golden_with_epsilon(
    op: &str,
    case: &str,
    actual: &[u8],
    vips_cmd: &[&str],
    epsilon: f32,
) {
    let expected = golden::generate_vips_golden(op, case, vips_cmd);
    let actual = decode_f32_le(actual);
    let expected = decode_f32_le(&expected);

    assert_eq!(
        actual.len(),
        expected.len(),
        "f32 differential golden length mismatch for op={op} case={case}"
    );

    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= epsilon,
            "f32 differential golden mismatch for op={op} case={case} at sample {idx}: got {got}, want {want}, epsilon {epsilon}"
        );
    }
}

pub(crate) fn assert_f32_golden_scaled(
    op: &str,
    case: &str,
    actual: &[u8],
    vips_cmd: &[&str],
    expected_scale: f32,
    epsilon: f32,
) {
    let expected = golden::generate_vips_golden(op, case, vips_cmd);
    let actual = decode_f32_le(actual);
    let expected = decode_f32_le(&expected)
        .into_iter()
        .map(|value| value * expected_scale)
        .collect::<Vec<_>>();

    assert_eq!(
        actual.len(),
        expected.len(),
        "scaled f32 differential golden length mismatch for op={op} case={case}"
    );

    for (idx, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= epsilon,
            "scaled f32 differential golden mismatch for op={op} case={case} at sample {idx}: got {got}, want {want}, epsilon {epsilon}"
        );
    }
}

pub(crate) fn assert_u8_golden_with_max_diff(
    op: &str,
    case: &str,
    actual: &[u8],
    vips_cmd: &[&str],
    max_diff: u8,
) {
    let expected = golden::generate_vips_golden(op, case, vips_cmd);
    golden::assert_golden_approx(actual, &expected, max_diff);
}

pub(crate) fn assert_hsv_golden_from_vips_u8(
    op: &str,
    case: &str,
    actual: &[u8],
    vips_cmd: &[&str],
    hue_epsilon: f32,
    sv_epsilon: f32,
) {
    let expected = golden::generate_vips_golden(op, case, vips_cmd);
    let actual = decode_f32_le(actual);

    assert_eq!(
        actual.len(),
        expected.len(),
        "HSV differential golden sample length mismatch for op={op} case={case}"
    );

    for (pixel_idx, (got, want)) in actual
        .chunks_exact(3)
        .zip(expected.chunks_exact(3))
        .enumerate()
    {
        let expected_h = f32::from(want[0]) * (360.0 / 255.0);
        let expected_s = f32::from(want[1]) / 255.0;
        let expected_v = f32::from(want[2]) / 255.0;

        assert!(
            (got[0] - expected_h).abs() <= hue_epsilon,
            "HSV hue mismatch for op={op} case={case} at pixel {pixel_idx}: got {}, want {}, epsilon {}",
            got[0],
            expected_h,
            hue_epsilon
        );
        assert!(
            (got[1] - expected_s).abs() <= sv_epsilon,
            "HSV saturation mismatch for op={op} case={case} at pixel {pixel_idx}: got {}, want {}, epsilon {}",
            got[1],
            expected_s,
            sv_epsilon
        );
        assert!(
            (got[2] - expected_v).abs() <= sv_epsilon,
            "HSV value mismatch for op={op} case={case} at pixel {pixel_idx}: got {}, want {}, epsilon {}",
            got[2],
            expected_v,
            sv_epsilon
        );
    }
}

pub(crate) fn patch_vips_orientation_6(image_path: &str) {
    if !golden::fixtures_regeneration_requested() {
        return;
    }
    let bytes =
        fs::read(image_path).unwrap_or_else(|err| panic!("failed to read {image_path}: {err}"));
    let xml_start = bytes
        .windows(5)
        .position(|window| window == b"<?xml")
        .unwrap_or_else(|| panic!("missing XML payload in {image_path}"));
    let mut xml = String::from_utf8(bytes[xml_start..].to_vec())
        .unwrap_or_else(|err| panic!("invalid XML payload in {image_path}: {err}"));

    xml = xml.replacen(
        "name=\"exif-ifd0-Orientation\">1 (Top-left, Short, 1 components, 2 bytes)",
        "name=\"exif-ifd0-Orientation\">6 (Right-top, Short, 1 components, 2 bytes)",
        1,
    );
    xml = xml.replacen(
        "name=\"orientation\">1</field>",
        "name=\"orientation\">6</field>",
        1,
    );

    let mut patched = bytes[..xml_start].to_vec();
    patched.extend_from_slice(xml.as_bytes());
    fs::write(image_path, patched)
        .unwrap_or_else(|err| panic!("failed to write {image_path}: {err}"));
}

pub(crate) fn run_reducer_u8<R>(
    source_pixels: Vec<u8>,
    width: u32,
    height: u32,
    bands: u32,
    reducer: &R,
) -> R::Output
where
    R: viprs::domain::reducer::TileReducer<U8>,
{
    let source = MemorySource::<U8>::new(width, height, bands, source_pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply(viprs::Linear::new(1.0, 0.0))
        .unwrap()
        .build()
        .unwrap();
    let sink = MemorySink::for_pipeline(&pipeline).unwrap();

    RayonScheduler::new(1)
        .unwrap()
        .run_with_reducer::<U8, R>(&pipeline, &sink, reducer)
        .unwrap()
}

pub(crate) fn run_vips_scalar(command: &str, input: &str) -> f64 {
    if !golden::fixtures_regeneration_requested() {
        panic!("run_vips_scalar requires fixture regeneration mode");
    }
    let output = Command::new("vips")
        .args([command, input])
        .output()
        .unwrap_or_else(|err| panic!("failed to run vips {command}: {err}"));
    assert!(
        output.status.success(),
        "vips {command} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout)
        .expect("vips scalar output is utf8")
        .trim()
        .parse::<f64>()
        .expect("vips scalar output parses as f64")
}

pub(crate) fn read_f64_fixture(op: &str, case: &str) -> f64 {
    let bytes = golden::read_fixture(op, case);
    let raw: [u8; 8] = bytes
        .as_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("f64 fixture for op={op} case={case} must be exactly 8 bytes"));
    f64::from_le_bytes(raw)
}

pub(crate) fn write_f64_fixture(op: &str, case: &str, value: f64) {
    golden::write_fixture(op, case, &value.to_le_bytes());
}

pub(crate) fn u64_bins_to_u32_bytes(bins: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(bins.len() * size_of::<u32>());
    for &count in bins {
        bytes.extend_from_slice(&(count as u32).to_le_bytes());
    }
    bytes
}
