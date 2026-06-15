use super::support as golden;

// GOLDEN: compared against fixtures generated from libvips.
//
// These tests run a 4x4 U8 image through the Add operation and compare the
// output byte-for-byte against a stored fixture generated from libvips.
//
// The fixture format is raw row-major pixel bytes with no header.

use viprs::{
    Add, OperationBridge,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::format::U8,
    ports::scheduler::TileScheduler,
};

/// Build, run the pipeline, and return the output buffer.
///
/// `source_pixels` — 4x4 U8 image (16 bytes, row-major).
/// `rhs` — 16-element rhs buffer passed to `Add::new`.
fn run_add(source_pixels: Vec<u8>, rhs: Vec<u8>) -> Vec<u8> {
    let source = MemorySource::<U8>::new(4, 4, 1, source_pixels).unwrap();
    let add_op = Add::<U8>::new(rhs);
    let dyn_op = Box::new(OperationBridge::new(add_op, 1u32));

    let pipeline = PipelineBuilder::from_source(source)
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

/// Case: every source pixel is 10, rhs is a uniform constant 5 for all pixels.
/// Expected output: all pixels equal to 15.
#[test]
fn add_uniform_constant() {
    let output = run_add(vec![10u8; 16], vec![5u8; 16]);
    golden::assert_golden("add", "uniform_constant", &output);
}

/// Case: source pixels are 0..15 (the sequence), rhs is 1 for all pixels.
/// Expected output: each pixel incremented by 1 (1..16).
#[test]
fn add_sequential_plus_one() {
    let source: Vec<u8> = (0u8..16).collect();
    let rhs = vec![1u8; 16];
    let output = run_add(source, rhs);
    golden::assert_golden("add", "sequential_plus_one", &output);
}
