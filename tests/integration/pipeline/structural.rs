// ── Structural ops ───────────────────────────────────────────────────────────

#[test]
fn extract_area_produces_correct_subregion() {
    // Source: 4x4 image where pixel value = y * 4 + x (values 0..15).
    // Extract a 2x2 subregion starting at (1, 1).
    // Expected pixels: (1,1)=5, (2,1)=6, (1,2)=9, (2,2)=10.
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let data: Vec<u8> = (0u8..16).collect();
    let source = MemorySource::<U8>::new(4, 4, 1, data).unwrap();

    let pipeline = PipelineBuilder::from_source(source)
        .extract_area(1, 1, 2, 2)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(
        pipeline.width, 2,
        "pipeline width must be the extract width"
    );
    assert_eq!(
        pipeline.height, 2,
        "pipeline height must be the extract height"
    );

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(
        output,
        vec![5u8, 6, 9, 10],
        "extracted pixels must match source sub-region"
    );
}

#[test]
fn flip_horizontal_reverses_columns() {
    // Source: 1x4 image (single row) with values [1, 2, 3, 4].
    // After horizontal flip: [4, 3, 2, 1].
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let source = MemorySource::<U8>::new(4, 1, 1, vec![1u8, 2, 3, 4]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .flip_horizontal()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(
        output,
        vec![4u8, 3, 2, 1],
        "horizontal flip must reverse columns"
    );
}

#[test]
fn flip_horizontal_twice_is_identity() {
    // Two consecutive flip_horizontal calls cancel out.
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
        .flip_horizontal()
        .unwrap()
        .flip_horizontal()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(output, expected, "double flip_horizontal must be identity");
}

#[test]
fn flip_vertical_reverses_rows() {
    // Source: 4x1 image (single column) with values [10, 20, 30, 40].
    // After vertical flip: [40, 30, 20, 10].
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let source = MemorySource::<U8>::new(1, 4, 1, vec![10u8, 20, 30, 40]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .flip_vertical()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(
        output,
        vec![40u8, 30, 20, 10],
        "vertical flip must reverse rows"
    );
}

#[test]
fn flip_vertical_twice_is_identity() {
    // Two consecutive flip_vertical calls cancel out.
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
        .flip_vertical()
        .unwrap()
        .flip_vertical()
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    assert_eq!(output, expected, "double flip_vertical must be identity");
}

// ── Convolution ──────────────────────────────────────────────────────────────

/// Integration test: MemorySource<F32> → Conv2d (1×1 identity kernel) → MemorySink
///
/// A 1×1 identity kernel has radius 0, so the input and output tiles have the same
/// dimensions. The output must equal the input within f32 precision.
#[test]
fn conv2d_identity_kernel_end_to_end() {
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::F32,
        ports::scheduler::TileScheduler,
    };

    // 4×4 single-band F32 image with distinct values.
    let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let source = MemorySource::<F32>::new(4, 4, 1, input.clone()).unwrap();

    let identity_kernel = vec![vec![1.0f64]];
    let pipeline = PipelineBuilder::from_source(source)
        .conv2d(identity_kernel)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    let output_f32: &[f32] = bytemuck::cast_slice(&output);

    for (i, (got, expected)) in output_f32.iter().zip(input.iter()).enumerate() {
        assert!(
            (got - expected).abs() < 1e-4,
            "pixel {i}: expected {expected}, got {got}"
        );
    }
}

/// Integration test: MemorySource<F32> → Conv2d (3×3 box filter on step-edge image) → MemorySink
///
/// A step edge must be smoothed by the box filter around the boundary, so the
/// center pixel can detect incorrect kernels or passthrough behaviour.
#[test]
fn conv2d_box_filter_uniform_image_end_to_end() {
    // Validates that conv2d actually averages across a neighborhood by smoothing a
    // step edge; a uniform image would pass even if the kernel were ignored.
    use viprs::{
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::F32,
        ports::scheduler::TileScheduler,
    };

    let width = 8usize;
    let height = 8usize;
    let dark = 0.0f32;
    let bright = 9.0f32;
    let mut input = Vec::with_capacity(width * height);
    for _y in 0..height {
        for x in 0..width {
            input.push(if x < width / 2 { dark } else { bright });
        }
    }
    let source = MemorySource::<F32>::new(width as u32, height as u32, 1, input.clone()).unwrap();

    let w = 1.0f64 / 9.0;
    let box_3x3 = vec![vec![w, w, w], vec![w, w, w], vec![w, w, w]];
    let pipeline = PipelineBuilder::from_source(source)
        .conv2d(box_3x3)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(2)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    let output = sink.into_buffer();
    let output_f32: &[f32] = bytemuck::cast_slice(&output);
    let center_index = (height / 2) * width + (width / 2);
    let center = output_f32[center_index];

    assert!(
        (output_f32[0] - dark).abs() < 1e-4,
        "far-left pixel should remain dark, got {}",
        output_f32[0]
    );
    assert!(
        (output_f32[width - 1] - bright).abs() < 1e-4,
        "far-right pixel should remain bright, got {}",
        output_f32[width - 1]
    );
    assert!(
        center > dark && center < bright,
        "center pixel should be smoothed between {dark} and {bright}, got {center}"
    );
    assert!(
        (center - input[center_index]).abs() > 1e-4,
        "center pixel should differ from the original edge value {}, got {center}",
        input[center_index]
    );
}

// ── AnySource ────────────────────────────────────────────────────────────────

#[test]
fn any_source_u8_variant_works_end_to_end() {
    // AnySource::U8 passes pixels through the pipeline correctly.
    use viprs::{
        AnySource,
        adapters::{
            pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
            sinks::memory::MemorySink, sources::memory::MemorySource,
        },
        domain::format::U8,
        ports::scheduler::TileScheduler,
    };

    let mem_source = MemorySource::<U8>::new(2, 2, 1, vec![100u8; 4]).unwrap();
    let any = AnySource::U8(mem_source);

    let pipeline = PipelineBuilder::from_source(any)
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
    // invert(100) = 155 for U8.
    assert!(
        output.iter().all(|&b| b == 155),
        "Expected all 155s (255 - 100), got: {:?}",
        output
    );
}
