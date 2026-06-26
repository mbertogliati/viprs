use super::support as golden;

// GOLDEN: compared against fixtures generated from libvips.
//
// These tests run a 4x4 F32 image through the Subtract operation and compare
// the output byte-for-byte against stored fixtures generated from libvips.
// When the libvips CLI is available, the same cases are also checked
// differentially against a fresh libvips run.
//
// F32 is used because U8 subtraction wraps on underflow (see Subtract docstring
// and B-014). F32 produces well-defined results for the test values chosen here.
//
// The fixture format is raw row-major bytes (little-endian IEEE 754 f32).

use bytemuck::cast_slice;

use golden::{ImageSpec, VipsBandFormat};
use viprs::{
    OperationBridge, Subtract,
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::format::F32,
    ports::scheduler::TileScheduler,
};

const WIDTH: u32 = 4;
const HEIGHT: u32 = 4;

/// Build, run the pipeline, and return the raw output bytes.
///
/// `source_pixels` — 4x4 F32 image (16 f32 values, row-major).
/// `rhs` — 16-element rhs buffer passed to `Subtract::new`.
fn run_subtract(source_pixels: Vec<f32>, rhs: Vec<f32>) -> Vec<u8> {
    let source = MemorySource::<F32>::new(WIDTH, HEIGHT, 1, source_pixels).unwrap();
    let sub_op = Subtract::<F32>::new(rhs);
    let dyn_op = Box::new(OperationBridge::new(sub_op, 1u32));

    let pipeline = viprs_runtime::pipeline::PipelineBuilder::from_source(source)
        .then(dyn_op)
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

fn write_f32_input(case: &str, name: &str, pixels: &[f32]) -> String {
    golden::write_vips_input(
        "subtract",
        case,
        name,
        cast_slice(pixels),
        ImageSpec::new(WIDTH, HEIGHT, 1, VipsBandFormat::F32),
    )
    .display()
    .to_string()
}

fn assert_subtract_case(case: &str, source: Vec<f32>, rhs: Vec<f32>) {
    let output = run_subtract(source.clone(), rhs.clone());
    golden::assert_golden("subtract", case, &output);

    if golden::skip_without_vips() {
        return;
    }

    let input = write_f32_input(case, "input", &source);
    let rhs_path = write_f32_input(case, "rhs", &rhs);
    let cmd = ["subtract", input.as_str(), rhs_path.as_str(), "{output}"];
    golden::assert_golden_libvips("subtract", case, &output, &cmd);
}

/// Case: every source pixel is 10.0, rhs is a uniform constant 3.0.
/// Expected output: all pixels equal to 7.0.
#[test]
fn subtract_uniform_constant() {
    assert_subtract_case("uniform_constant", vec![10.0f32; 16], vec![3.0f32; 16]);
}

/// Case: source pixels are 1.0, 2.0, … 16.0 (sequential), rhs is 0.5 for all.
/// Expected output: 0.5, 1.5, … 15.5.
#[test]
fn subtract_sequential_minus_half() {
    let source: Vec<f32> = (1u8..=16).map(|v| v as f32).collect();
    assert_subtract_case("sequential_minus_half", source, vec![0.5f32; 16]);
}
