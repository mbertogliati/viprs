use super::*;

#[test]
fn pipeline_with_zero_ops_is_identity() {
    let pixels = vec![19_u8, 7, 191, 36, 18, 196, 32, 36, 210, 49, 47, 215];
    let source = MemorySource::<U8>::new(2, 2, 3, pixels.clone()).unwrap();
    let pipeline = PipelineBuilder::from_source(source).build().unwrap();
    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(output.pixels(), pixels.as_slice());
    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 3);
}

#[test]
fn invert_rejects_zero_band_sources() {
    let result = PipelineBuilder::from_source(zero_band_source()).invert();

    assert!(matches!(
        result,
        Err(BuildError::InvalidImage {
            reason: "zero-band image"
        })
    ));
}

#[test]
fn flip_horizontal_rejects_zero_band_sources() {
    let result = PipelineBuilder::from_source(zero_band_source()).flip_horizontal();

    assert!(matches!(
        result,
        Err(BuildError::InvalidImage {
            reason: "zero-band image"
        })
    ));
}

#[test]
fn rotate90_rejects_zero_band_sources() {
    let result = PipelineBuilder::from_source(zero_band_source()).rotate90();

    assert!(matches!(
        result,
        Err(BuildError::InvalidImage {
            reason: "zero-band image"
        })
    ));
}

fn point_mode_pipeline_handles_zero_width_source_after_upstream_op() {
    use crate::{
        adapters::sinks::memory::MemorySink,
        domain::ops::{arithmetic::Invert, conversion::subsample::SubsampleBridge},
    };

    let source = MemorySource::<U8>::new(0, 1, 1, vec![]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .then(Box::new(OperationBridge::new_pixel_local(
            Invert::<U8>::new(),
            1,
        )))
        .unwrap()
        .then(Box::new(
            SubsampleBridge::<U8>::with_point(12, 5, 1, true).unwrap(),
        ))
        .unwrap()
        .build()
        .unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();
}

#[test]
fn source_region_sizing_uses_coordinate_driven_source_plan() {
    let node = CompiledNode::new(
        CompiledOp::Transform(Box::new(CoordinateDrivenSourceStub {
            bands: 1,
            source_halo: 1,
            full_source: Region::new(0, 0, 1024, 1024),
        })),
        vec![0, 1],
        vec![None, Some(0)],
        vec![None, Some(0)],
        2,
        None,
    )
    .expect("coordinate-driven compiled node");

    let source_region = source_region_for_scheduler_tile(&[node], 128, 128);

    assert_eq!(source_region, Region::new(0, 0, 129, 129));
}

#[test]
fn flatten_rgb_is_noop() {
    let pixels = vec![19_u8, 7, 191, 36, 18, 196, 32, 36, 210, 49, 47, 215];
    let source = MemorySource::<U8>::new(2, 2, 3, pixels.clone()).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .flatten([0.0, 0.0, 0.0, 1.0])
        .unwrap()
        .build()
        .unwrap();
    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(pipeline.output_bands, 3);
    assert_eq!(output.pixels(), pixels.as_slice());
}

#[test]
fn flatten_rgba_removes_alpha() {
    let pixels = vec![10_u8, 20, 30, 0, 40, 50, 60, 255];
    let source = MemorySource::<U8>::new(2, 1, 4, pixels).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .flatten([0.0, 0.0, 0.0, 1.0])
        .unwrap()
        .build()
        .unwrap();
    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(pipeline.output_bands, 3);
    assert_eq!(output.pixels(), &[0u8, 0, 0, 40, 50, 60]);
}

#[test]
fn premultiply_u16_rgb16_uses_interpretation_max_alpha() {
    let mut metadata = ImageMetadata::default();
    metadata.interpretation = Some(Interpretation::Rgb16);
    let source = MemorySource::<U16>::new(1, 1, 4, vec![65535, 32768, 16384, 32768])
        .unwrap()
        .with_metadata(metadata);
    let pipeline = PipelineBuilder::from_source(source)
        .premultiply()
        .unwrap()
        .build()
        .unwrap();
    let output = pipeline
        .run_to_image::<U16, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert_eq!(output.pixels(), &[32768, 16384, 8192, 32768]);
}

#[test]
fn pipeline_arena_connect_format_mismatch() {
    let mut arena = PipelineArena::new(100, 100);
    let u8_op = arena.add_node(Box::new(OperationBridge::new(
        PassThrough { bands: 1 },
        1u32,
    )));
    let f32_op = arena.add_node(Box::new(OperationBridge::new(F32PassThrough, 1u32)));
    let result = arena.connect(u8_op, f32_op);
    assert!(matches!(result, Err(BuildError::FormatMismatch { .. })));
}

#[test]
fn pipeline_arena_connect_validates_each_input_slot_format() {
    use crate::domain::ops::conversion::IfThenElseOp;

    let mut arena = PipelineArena::new(100, 100);
    let u8_op = arena.add_node(pass_op(1));
    let f32_op = arena.add_node(Box::new(OperationBridge::new(F32PassThrough, 1u32)));
    let merge = arena.add_node(Box::new(IfThenElseOp::<F32>::new(1)));

    let result = arena.connect_to_slot(f32_op, merge, 0);
    assert!(matches!(result, Err(BuildError::FormatMismatch { .. })));

    let result = arena.connect_to_slot(u8_op, merge, 1);
    assert!(matches!(result, Err(BuildError::FormatMismatch { .. })));

    arena.connect_to_slot(f32_op, merge, 1).unwrap();
}

#[test]
fn pipeline_arena_connect_rejects_invalid_input_slot() {
    use crate::domain::ops::conversion::IfThenElseOp;

    let mut arena = PipelineArena::new(100, 100);
    let upstream = arena.add_node(pass_op(1));
    let merge = arena.add_node(Box::new(IfThenElseOp::<F32>::new(1)));

    let result = arena.connect_to_slot(upstream, merge, 3);
    assert!(matches!(result, Err(BuildError::InvalidInputSlot { .. })));
}

#[test]
fn similarity_auto_canvas_updates_pipeline_dimensions() {
    let pipeline = PipelineBuilder::new(4, 2)
        .similarity(1.0, 90.0, InterpolationKernel::Bilinear)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!((pipeline.width, pipeline.height), (2, 4));
}

#[test]
fn pipeline_arena_connect_invalid_index() {
    let mut arena = PipelineArena::new(100, 100);
    let idx = arena.add_node(pass_op(1));
    let result = arena.connect(idx, 999);
    assert!(matches!(result, Err(BuildError::InvalidNodeIndex(999))));
}

#[test]
fn pipeline_linear_3_nodes_topological_order() {
    let builder = PipelineBuilder::new(64, 64);
    let builder = builder.then(pass_op(3)).unwrap();
    let builder = builder.then(pass_op(3)).unwrap();
    let builder = builder.then(pass_op(3)).unwrap();
    let pipeline = builder.build().unwrap();
    assert_eq!(pipeline.nodes.len(), 3);
}

#[test]
fn pipeline_empty_builds_identity_pipeline() {
    let pipeline = PipelineBuilder::new(2, 2).build().unwrap();
    let output = pipeline
        .run_to_image::<U8, _>(&RayonScheduler::new(1).unwrap())
        .unwrap();

    assert!(pipeline.nodes.is_empty());
    assert_eq!(output.pixels(), &[0, 0, 0, 0]);
}

#[test]
fn thread_buffer_pool_correct_sizes() {
    let builder = PipelineBuilder::new(128, 128);
    let builder = builder.then(pass_op(1)).unwrap();
    let pipeline = builder.build().unwrap();
    let pool = ThreadBufferPool::new(&pipeline);
    assert!(!pool.buffers.is_empty());
    assert!(pool.buffers.iter().all(Vec::is_empty));
    assert_eq!(pool.buffers.len(), pipeline.buffer_sizes.len());
}

#[test]
fn pipeline_from_source_uses_source_dimensions() {
    let source = ZeroSource::<U8>::new(32, 16, 1);
    let builder = PipelineBuilder::from_source(source);
    let builder = builder.then(pass_op(1)).unwrap();
    let pipeline = builder.build().unwrap();
    assert_eq!(pipeline.width, 32);
    assert_eq!(pipeline.height, 16);
}

#[test]
fn pipeline_from_source_tracks_format() {
    // A F32 source must set current_format to F32.
    let source = ZeroSource::<F32>::new(8, 8, 1);
    let builder = PipelineBuilder::from_source(source);
    assert_eq!(builder.current_format(), BandFormatId::F32);
}

#[test]
fn gauss_blur_preserves_u8_output_format() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let builder = PipelineBuilder::from_source(source)
        .gauss_blur(1.5)
        .unwrap();
    assert_eq!(builder.current_format(), BandFormatId::U8);
}

#[test]
fn gauss_blur_promotes_non_u8_output_format_to_f32() {
    let source = ZeroSource::<U16>::new(8, 8, 1);
    let builder = PipelineBuilder::from_source(source)
        .gauss_blur(1.5)
        .unwrap();
    assert_eq!(builder.current_format(), BandFormatId::F32);
}

#[test]
fn then_rejects_mismatched_format() {
    // ZeroSource<U8> → F32 op must fail with FormatMismatch.
    let source = ZeroSource::<U8>::new(4, 4, 1);
    let builder = PipelineBuilder::from_source(source);
    let f32_op = Box::new(OperationBridge::new(F32PassThrough, 1u32));
    let result = builder.then(f32_op);
    assert!(matches!(result, Err(BuildError::FormatMismatch { .. })));
}

#[test]
fn convenience_linear_builds_pipeline() {
    let source = ZeroSource::<F32>::new(4, 4, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .linear(2.0, 0.5)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::F32);
}

#[test]
fn convenience_linear_statically_fuses_adjacent_linear_ops() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    let chained = PipelineBuilder::from_source(
        MemorySource::<F32>::new(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .linear(2.0, 10.0)
    .unwrap()
    .linear(3.0, 5.0)
    .unwrap()
    .build()
    .unwrap();
    assert_eq!(chained.nodes.len(), 1);

    let fused = PipelineBuilder::from_source(
        MemorySource::<F32>::new(4, 1, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .linear(6.0, 35.0)
    .unwrap()
    .build()
    .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut chained_sink = MemorySink::for_pipeline(&chained).unwrap();
    scheduler.run(&chained, &mut chained_sink).unwrap();
    let mut fused_sink = MemorySink::for_pipeline(&fused).unwrap();
    scheduler.run(&fused, &mut fused_sink).unwrap();

    assert_eq!(chained_sink.into_buffer(), fused_sink.into_buffer());
}

#[test]
fn convenience_linear_u8_clips_like_libvips() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 10, 250, 255]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .linear(1.5, 10.9)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(sink.into_buffer(), vec![10, 25, 255, 255]);
}

#[test]
fn convenience_linear_rejects_nan_scale() {
    let source = ZeroSource::<F32>::new(1, 1, 1);
    let result = PipelineBuilder::from_source(source).linear(f64::NAN, 0.0);
    assert!(matches!(
        result,
        Err(BuildError::InvalidLinearParameters { scale, offset })
            if scale.is_nan() && offset == 0.0
    ));
}

#[test]
fn convenience_linear_rejects_infinite_scale() {
    let source = ZeroSource::<F32>::new(1, 1, 1);
    let result = PipelineBuilder::from_source(source).linear(f64::INFINITY, 0.0);
    assert!(matches!(
        result,
        Err(BuildError::InvalidLinearParameters { scale, offset })
            if scale.is_infinite() && offset == 0.0
    ));
}

#[test]
fn convenience_linear_rejects_nan_offset() {
    let source = ZeroSource::<F32>::new(1, 1, 1);
    let result = PipelineBuilder::from_source(source).linear(1.0, f64::NAN);
    assert!(matches!(
        result,
        Err(BuildError::InvalidLinearParameters { scale, offset })
            if scale == 1.0 && offset.is_nan()
    ));
}

#[test]
fn convenience_linear_accepts_zero_scale_and_produces_black() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };

    let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 10, 250, 255]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .linear(0.0, 0.0)
        .unwrap()
        .build()
        .unwrap();

    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    RayonScheduler::new(1)
        .unwrap()
        .run(&pipeline, &mut sink)
        .unwrap();

    assert_eq!(sink.into_buffer(), vec![0, 0, 0, 0]);
}

#[test]
fn convenience_invert_builds_pipeline() {
    let source = ZeroSource::<U8>::new(4, 4, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::U8);
}

#[test]
fn convenience_cast_u8_to_f32() {
    let source = ZeroSource::<U8>::new(4, 4, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .cast(BandFormatId::F32)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::F32);
}

#[test]
fn convenience_cast_unsupported_returns_error() {
    // U8 → I32 has no CastSample impl; must return UnsupportedFormat.
    let source = ZeroSource::<U8>::new(4, 4, 1);
    let result = PipelineBuilder::from_source(source).cast(BandFormatId::I32);
    assert!(matches!(result, Err(BuildError::UnsupportedFormat { .. })));
}

#[test]
fn shrink_h_with_ceil_rounds_output_width_up() {
    let source = ZeroSource::<U8>::new(10, 1, 1);
    let floor = PipelineBuilder::from_source(source)
        .shrink_h(3)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(floor.width, 3);

    let source = ZeroSource::<U8>::new(10, 1, 1);
    let ceil = PipelineBuilder::from_source(source)
        .shrink_h_with_ceil(3, true)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(ceil.width, 4);
}

#[test]
fn shrink_v_with_ceil_rounds_output_height_up() {
    let source = ZeroSource::<U8>::new(1, 10, 1);
    let floor = PipelineBuilder::from_source(source)
        .shrink_v(3)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(floor.height, 3);

    let source = ZeroSource::<U8>::new(1, 10, 1);
    let ceil = PipelineBuilder::from_source(source)
        .shrink_v_with_ceil(3, true)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(ceil.height, 4);
}

#[test]
fn shrink_v_zero_factor_returns_typed_error() {
    let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
    let result = PipelineBuilder::from_source(source).shrink_v(0);

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "shrink_v",
            ..
        })
    ));
}

#[test]
fn shrink_h_zero_factor_returns_typed_error() {
    let source = MemorySource::<U8>::new(4, 4, 1, (0u8..16).collect()).unwrap();
    let result = PipelineBuilder::from_source(source).shrink_h(0);

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "shrink_h",
            ..
        })
    ));
}

#[test]
fn shrink_v_zero_band_source_returns_typed_error() {
    let source = MemorySource::<U8>::new(8, 8, 0, vec![]).unwrap();
    let result = PipelineBuilder::from_source(source).shrink_v(2);

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "shrink_v",
            ..
        })
    ));
}

#[test]
fn resize_zero_band_source_returns_typed_error() {
    let source = MemorySource::<U8>::new(8, 8, 0, vec![]).unwrap();
    let result = PipelineBuilder::from_source(source).resize(Resize::new(
        1.5,
        1.5,
        InterpolationKernel::Lanczos3,
    ));

    assert!(matches!(
        result,
        Err(BuildError::SourceHint {
            context: "resize",
            ..
        })
    ));
}

#[test]
fn resize_rejects_vsqbs_downscale_reduce_path() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).resize(Resize::new(
        0.6,
        1.0,
        InterpolationKernel::Vsqbs,
    ));
    assert!(matches!(
        result,
        Err(BuildError::InvalidKernel {
            op: "reduceh",
            kernel: InterpolationKernel::Vsqbs,
            ..
        })
    ));
}

#[test]
fn convenience_msb_builds_u8_pipeline() {
    let source = ZeroSource::<U16>::new(4, 4, 2);
    let pipeline = PipelineBuilder::from_source(source)
        .msb()
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.output_bands, 2);
}

#[test]
fn convenience_msb_rejects_float_formats() {
    let source = ZeroSource::<F32>::new(4, 4, 1);
    let result = PipelineBuilder::from_source(source).msb();
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedFormat { op: "msb", .. })
    ));
}

#[test]
fn convenience_rot45_builds_for_odd_square_images() {
    let source = ZeroSource::<U8>::new(5, 5, 1);
    let pipeline = PipelineBuilder::from_source(source)
        .rot45(Angle45::D45)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_format, BandFormatId::U8);
    assert_eq!(pipeline.width, 7);
    assert_eq!(pipeline.height, 7);
}

#[test]
fn rot45_non_square_canvas_correct() {
    let pipeline = PipelineBuilder::from_source(ZeroSource::<U8>::new(100, 200, 1))
        .rot45(Angle45::D45)
        .unwrap()
        .build()
        .unwrap();
    let expected = 212;
    assert_eq!(pipeline.width, expected);
    assert_eq!(pipeline.height, expected);
}

#[test]
fn rot45_square_canvas_correct() {
    let pipeline = PipelineBuilder::from_source(ZeroSource::<U8>::new(100, 100, 1))
        .rot45(Angle45::D45)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.width, 141);
    assert_eq!(pipeline.height, 141);
}

#[test]
fn convenience_rot45_keeps_exact_right_angle_dimensions() {
    let pipeline = PipelineBuilder::from_source(ZeroSource::<U8>::new(5, 3, 1))
        .rot45(Angle45::D90)
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.width, 3);
    assert_eq!(pipeline.height, 5);
}

#[test]
fn reduce_h_rejects_lbb_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce_h(2.0, InterpolationKernel::Lbb);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce_h",
            kernel: InterpolationKernel::Lbb,
            ..
        })
    ));
}

#[test]
fn reduce_v_rejects_lbb_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce_v(2.0, InterpolationKernel::Lbb);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce_v",
            kernel: InterpolationKernel::Lbb,
            ..
        })
    ));
}

#[test]
fn reduce_rejects_lbb_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce(2.0, 2.0, InterpolationKernel::Lbb);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce",
            kernel: InterpolationKernel::Lbb,
            ..
        })
    ));
}

#[test]
fn reduce_h_rejects_nohalo_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce_h(2.0, InterpolationKernel::Nohalo);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce_h",
            kernel: InterpolationKernel::Nohalo,
            ..
        })
    ));
}

#[test]
fn reduce_v_rejects_nohalo_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce_v(2.0, InterpolationKernel::Nohalo);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce_v",
            kernel: InterpolationKernel::Nohalo,
            ..
        })
    ));
}

#[test]
fn reduce_rejects_nohalo_kernel() {
    let source = ZeroSource::<U8>::new(8, 8, 1);
    let result = PipelineBuilder::from_source(source).reduce(2.0, 2.0, InterpolationKernel::Nohalo);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedKernel {
            op: "reduce",
            kernel: InterpolationKernel::Nohalo,
            ..
        })
    ));
}

#[test]
fn pipeline_rgb_source_propagates_bands() {
    use crate::adapters::sources::memory::MemorySource;
    // 2x2 RGB: 4 pixels * 3 bands = 12 samples
    let source = MemorySource::<U8>::new(2, 2, 3, vec![0u8; 12]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();
    assert_eq!(pipeline.output_bands, 3);
    for node in &pipeline.nodes {
        let bands = match &node.op {
            CompiledOp::Transform(t) => t.bands(),
            CompiledOp::View(v) => v.bands(),
        };
        assert_eq!(bands, 3, "compiled node must carry the source band count");
    }
}

#[test]
fn pipeline_rgb_buffer_sizes_are_correct() {
    use crate::adapters::sources::memory::MemorySource;
    // 4x4 RGB U8: 4*4*3 = 48 samples
    let source = MemorySource::<U8>::new(4, 4, 3, vec![0u8; 48]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .build()
        .unwrap();
    let tile_w = pipeline.demand_hint.tile_width(pipeline.width) as usize;
    let tile_h = pipeline
        .demand_hint
        .tile_height(pipeline.width, pipeline.height) as usize;
    let expected = tile_w * tile_h * 3 * 1; // bands=3, bps=1 (U8)
    for &size in &pipeline.buffer_sizes {
        if size > 0 {
            assert_eq!(size, expected, "buffer size must account for 3 bands");
        }
    }
}

#[test]
fn linear_transform_chain_starts_without_input_scratch_buffers() {
    use crate::adapters::sources::memory::MemorySource;

    let source = MemorySource::<U8>::new(8, 8, 3, vec![0u8; 8 * 8 * 3]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .invert()
        .unwrap()
        .invert()
        .unwrap()
        .build()
        .unwrap();

    let pool = ThreadBufferPool::new(&pipeline);

    assert!(
        pool.input_scratch_buffers
            .iter()
            .flatten()
            .all(Vec::is_empty),
        "linear transform chains should not preallocate unused input scratch buffers"
    );
}
