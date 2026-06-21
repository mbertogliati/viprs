use super::*;

// ── Concretize apply integration tests ─────────────────────────────────

#[test]
fn apply_single_invert_u8() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::point::Invert as ConcretizeInvert;

    let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 100, 200, 255]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply(ConcretizeInvert)
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();

    let output = sink.into_buffer();
    assert_eq!(output, vec![255, 155, 55, 0]);
}

#[test]
fn apply_fused_chain_u8() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::point::{Invert as ConcretizeInvert, Linear};

    // Chain: invert then linear(2.0, -10.0)
    let source = MemorySource::<U8>::new(2, 1, 1, vec![100, 200]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply((ConcretizeInvert, Linear::new(2.0, -10.0)))
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();

    let output = sink.into_buffer();
    // invert(100)=155, linear(155,2,-10)=300→clamped 255
    // invert(200)=55, linear(55,2,-10)=100
    assert_eq!(output, vec![255, 100]);
}

#[test]
fn apply_matches_legacy_invert() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::point::Invert as ConcretizeInvert;

    let pixels: Vec<u8> = (0..=255).collect();

    // Legacy path
    let legacy =
        PipelineBuilder::from_source(MemorySource::<U8>::new(256, 1, 1, pixels.clone()).unwrap())
            .invert()
            .unwrap()
            .build()
            .unwrap();

    // Concretize path
    let concretize =
        PipelineBuilder::from_source(MemorySource::<U8>::new(256, 1, 1, pixels).unwrap())
            .apply(ConcretizeInvert)
            .unwrap()
            .build()
            .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut legacy_sink = MemorySink::for_pipeline(&legacy).unwrap();
    scheduler.run(&legacy, &mut legacy_sink).unwrap();
    let mut concretize_sink = MemorySink::for_pipeline(&concretize).unwrap();
    scheduler.run(&concretize, &mut concretize_sink).unwrap();

    assert_eq!(legacy_sink.into_buffer(), concretize_sink.into_buffer());
}

#[test]
fn apply_chain_builder_ergonomic() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::concretize::Chain;
    use crate::domain::format::U8;
    use crate::domain::ops::point::{Clamp, Invert as ConcretizeInvert, Linear};

    let chain = Chain::new()
        .then(ConcretizeInvert)
        .then(Linear::new(2.0, -10.0))
        .then(Clamp::new(0.0, 200.0));

    let source = MemorySource::<U8>::new(3, 1, 1, vec![100, 200, 50]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply(chain)
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();

    let output = sink.into_buffer();
    // 100 → invert(155) → linear(300) → clamp(200)
    // 200 → invert(55)  → linear(100) → clamp(100)
    // 50  → invert(205) → linear(400) → clamp(200)
    assert_eq!(output, vec![200, 100, 200]);
}

// ── Unified .apply() integration tests ───────────────────────────────────────

#[test]
fn apply_unified_point_op() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::point::Invert as ConcretizeInvert;

    let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 100, 200, 255]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply(ConcretizeInvert)
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();
    assert_eq!(sink.into_buffer(), vec![255, 155, 55, 0]);
}

#[test]
fn apply_unified_dyn_operation() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::arithmetic::invert::Invert;

    // Use a Box<dyn DynOperation> through .apply()
    let source = MemorySource::<U8>::new(4, 1, 1, vec![0, 100, 200, 255]).unwrap();
    let dyn_op: Box<dyn DynOperation> =
        Box::new(OperationBridge::new_pixel_local(Invert::<U8>::new(), 1));

    let pipeline = PipelineBuilder::from_source(source)
        .apply(dyn_op)
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();
    assert_eq!(sink.into_buffer(), vec![255, 155, 55, 0]);
}

#[test]
fn apply_unified_mixed_chain() {
    use crate::adapters::{
        scheduler::rayon_scheduler::RayonScheduler, sinks::memory::MemorySink,
        sources::memory::MemorySource,
    };
    use crate::domain::format::U8;
    use crate::domain::ops::point::{Invert as ConcretizeInvert, Linear};

    // Mix point ops via .apply() with chained calls
    let source = MemorySource::<U8>::new(2, 1, 1, vec![100, 200]).unwrap();
    let pipeline = PipelineBuilder::from_source(source)
        .apply(ConcretizeInvert)
        .unwrap()
        .apply(Linear::new(2.0, -10.0))
        .unwrap()
        .build()
        .unwrap();

    let scheduler = RayonScheduler::new(1).unwrap();
    let mut sink = MemorySink::for_pipeline(&pipeline).unwrap();
    scheduler.run(&pipeline, &mut sink).unwrap();

    let output = sink.into_buffer();
    // invert(100)=155, linear(155,2,-10)=300→255
    // invert(200)=55, linear(55,2,-10)=100
    assert_eq!(output, vec![255, 100]);
}

#[test]
fn apply_consecutive_point_ops_auto_fuse_into_single_node() {
    use crate::domain::format::U8;
    use crate::domain::ops::point::{Invert as ConcretizeInvert, Linear};
    use crate::sources::memory::MemorySource;

    let builder =
        PipelineBuilder::from_source(MemorySource::<U8>::new(2, 1, 1, vec![100, 200]).unwrap())
            .apply(ConcretizeInvert)
            .unwrap()
            .apply(Linear::new(2.0, -10.0))
            .unwrap();

    assert_eq!(
        builder.node_count(),
        0,
        "pending point ops should stay out of the arena until flush"
    );

    let pipeline = builder.build().unwrap();
    assert_eq!(
        pipeline.nodes.len(),
        1,
        "consecutive point ops must flush as one fused node"
    );
}

#[test]
fn convenience_point_ops_auto_fuse_into_single_node() {
    use crate::domain::format::U8;
    use crate::sources::memory::MemorySource;

    let builder =
        PipelineBuilder::from_source(MemorySource::<U8>::new(2, 1, 1, vec![100, 200]).unwrap())
            .invert()
            .unwrap()
            .linear(2.0, -10.0)
            .unwrap();

    assert_eq!(
        builder.node_count(),
        0,
        "convenience point ops should stay out of the arena until flush"
    );

    let pipeline = builder.build().unwrap();
    assert_eq!(
        pipeline.nodes.len(),
        1,
        "convenience point ops must flush as one fused node"
    );
}
