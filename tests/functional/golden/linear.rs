use super::support as golden;

// GOLDEN: compared against fixtures generated from libvips.
//
// These tests run a 4x4 F32 image through the Linear operation (output = input
// * scale + offset) and compare the output byte-for-byte against a stored
// fixture generated from libvips. All runs must continue to match exactly.
//
// The fixture format is raw row-major bytes (little-endian IEEE 754 f32).

use viprs::{
    adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    },
    domain::format::F32,
    ports::scheduler::TileScheduler,
};

/// Build, run the pipeline, and return the raw output bytes.
///
/// `source_pixels` — 4x4 F32 image (16 f32 values, row-major).
/// `scale` and `offset` are `f64` as required by `viprs_runtime::pipeline::internal::PipelinePlan::linear`.
fn run_linear(source_pixels: Vec<f32>, scale: f64, offset: f64) -> Vec<u8> {
    let source = MemorySource::<F32>::new(4, 4, 1, source_pixels).unwrap();

    let pipeline = viprs_runtime::pipeline::internal::PipelinePlan::from_source(source)
        .plan_linear(scale, offset)
        .unwrap()
        .compile()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    sink.into_buffer()
}

/// Case: uniform source 2.0, scale=3.0, offset=1.0.
/// Expected output: all pixels equal to 7.0 (2*3 + 1).
#[test]
fn linear_uniform_scale_and_offset() {
    let output = run_linear(vec![2.0f32; 16], 3.0_f64, 1.0_f64);
    golden::assert_golden("linear", "uniform_scale_and_offset", &output);
}

/// Case: sequential source 1.0..16.0, scale=2.0, offset=0.0.
/// Expected output: 2.0, 4.0, 6.0, … 32.0 (each value doubled).
#[test]
fn linear_sequential_doubled() {
    let source: Vec<f32> = (1u8..=16).map(|v| v as f32).collect();
    let output = run_linear(source, 2.0_f64, 0.0_f64);
    golden::assert_golden("linear", "sequential_doubled", &output);
}
