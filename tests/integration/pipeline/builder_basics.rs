//! Integration tests: dynamic pipeline (PipelineBuilder) connected to real sources.
//!
//! All tests use `MemorySource` with known pixel data to verify the full
//! Source → Op → Sink contract. See ADR-011 for the rationale behind removing the
//! static pipeline API.

// ── from_source without explicit Box::new ────────────────────────────────────

#[test]
fn memory_source_add_constant_end_to_end() {
    // Verifies that PipelineBuilder::from_source accepts a concrete type directly
    // (no Box::new at the call site) and that the full pipeline produces correct output.
    use viprs::{
        Add, OperationBridge,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    // 4x4 single-band image where every pixel is 10.
    let source = MemorySource::<U8>::new(4, 4, 1, vec![10u8; 16]).unwrap();
    let add_op = Add::<U8>::new(vec![5u8; 16]);
    let dyn_op = Box::new(OperationBridge::new(add_op, 1u32));

    let pipeline = PipelineBuilder::from_source(source)
        .then(dyn_op)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert!(
        output.iter().all(|&b| b == 15),
        "Expected all 15s (10 + 5), got: {:?}",
        output
    );
}

#[test]
fn memory_source_clamp_to_edge_does_not_panic() {
    // Verify that MemorySource handles clamp-to-edge correctly when the scheduler
    // requests a region that exactly matches the image boundary.
    use viprs::{
        Add, OperationBridge,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    // 4x4 image with distinct pixel values to catch any coordinate confusion.
    let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
    let add_op = Add::<U8>::new(vec![0u8; 16]); // identity (add zero)
    let dyn_op = Box::new(OperationBridge::new(add_op, 1u32));

    let pipeline = PipelineBuilder::from_source(source)
        .then(dyn_op)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::new(4, 4, 1, 1).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    let expected: Vec<u8> = (0u8..16).collect();
    assert_eq!(
        output, expected,
        "Source pixels were not reproduced correctly"
    );
}

// ── Convenience methods ───────────────────────────────────────────────────────

#[test]
fn convenience_invert_end_to_end() {
    // PipelineBuilder::invert() builds correctly and produces inverted pixels.
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    // 4x4 single-band: all pixels = 0, inverted → 255.
    let source = MemorySource::<U8>::new(4, 4, 1, vec![0u8; 16]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert!(
        output.iter().all(|&b| b == 255),
        "Invert(0) = 255, got: {:?}",
        output
    );
}

#[test]
fn convenience_linear_end_to_end() {
    // Validates per-pixel linear math across distinct samples so a constant-filled buffer
    // cannot hide indexing or broadcast bugs.
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::F32,
        ports::scheduler::TileScheduler,
    };

    let input = [1.0f32, 2.0, 3.0, 4.0];
    let expected = [4.0f32, 7.0, 10.0, 13.0];
    let source = MemorySource::<F32>::new(4, 1, 1, input.to_vec()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .linear(3.0, 1.0)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    let floats: &[f32] = bytemuck::cast_slice(&output);
    assert_eq!(floats.len(), expected.len());
    for (i, (&got, &expected)) in floats.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-5,
            "pixel {i}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn convenience_cast_u8_to_f32_end_to_end() {
    // Validates normalization across multiple U8 samples so the cast path must handle
    // low, mid, and max values correctly instead of a single white pixel.
    use viprs::{
        BandFormatId,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let input = [0u8, 64, 127, 255];
    let expected = [0.0f32, 64.0 / 255.0, 127.0 / 255.0, 1.0];
    let source = MemorySource::<U8>::new(4, 1, 1, input.to_vec()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .cast(BandFormatId::F32)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    let floats: &[f32] = bytemuck::cast_slice(&output);
    assert_eq!(floats.len(), expected.len());
    for (i, (&got, &expected)) in floats.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-6,
            "pixel {i}: expected {expected}, got {got}"
        );
    }
}

#[test]
fn chained_invert_twice_is_identity() {
    // Two consecutive invert() calls must cancel out.
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let source_data: Vec<u8> = (0u8..16).collect();
    let expected = source_data.clone();
    let source = MemorySource::<U8>::new(4, 4, 1, source_data).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(output, expected, "Double-invert must be identity");
}
