/// Benchmark: Concretize fusion vs legacy pipeline for multi-op chains.
///
/// Compares:
/// 1. Legacy path: N separate `.invert()` / `.linear()` calls (N DynOperation nodes)
/// 2. Concretize path: single `.apply(chain)` call (1 fused DynOperation node)
///
/// The key insight: Concretize eliminates N-1 intermediate memory passes.
/// For integer ops, this produces N× speedup as chain depth grows.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::{
    BandFormatId, ImageMetadata,
    adapters::{
        pipeline::PipelineBuilder, scheduler::rayon_scheduler::RayonScheduler,
        sinks::memory::MemorySink, sources::memory::MemorySource,
    },
    domain::{
        concretize::Chain,
        format::U8,
        image::Region,
        op::{DemandHint, DynOperation, NodeSpec},
        ops::point::{Invert as CInvert, Linear},
    },
    ports::scheduler::TileScheduler,
};

fn bench_fusion_chain(c: &mut Criterion) {
    let size = 2048u32;
    let pixel_count = (size as usize) * (size as usize);
    let pixels = vec![128u8; pixel_count];

    let mut group = c.benchmark_group("fusion_chain_u8_2048");

    // 2-op chain: invert + linear
    group.bench_function("legacy_2ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .linear(2.0, -10.0)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("concretize_2ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .apply((CInvert, Linear::new(2.0, -10.0)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("auto_apply_2ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .apply(CInvert)
                .unwrap()
                .apply(Linear::new(2.0, -10.0))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 4-op: separate apply calls (2 nodes, not fused together)
    group.bench_function("separate_4ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .apply((CInvert, Linear::new(1.5, -5.0)))
                .unwrap()
                .apply((CInvert, Linear::new(0.8, 10.0)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 4-op chain: invert + linear + invert + linear
    group.bench_function("legacy_4ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .linear(1.5, -5.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.8, 10.0)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("concretize_4ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let chain = (
                (CInvert, Linear::new(1.5, -5.0)),
                (CInvert, Linear::new(0.8, 10.0)),
            );
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 8-op chain: alternating invert + linear
    group.bench_function("legacy_8ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .linear(1.2, -3.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.9, 5.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(1.1, -2.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.8, 10.0)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("concretize_8ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let chain = (
                (
                    (CInvert, Linear::new(1.2, -3.0)),
                    (CInvert, Linear::new(0.9, 5.0)),
                ),
                (
                    (CInvert, Linear::new(1.1, -2.0)),
                    (CInvert, Linear::new(0.8, 10.0)),
                ),
            );
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // Same 8-op chain using Chain builder (ergonomic API, same perf)
    group.bench_function("chain_builder_8ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let chain = Chain::new()
                .then(CInvert)
                .then(Linear::new(1.2, -3.0))
                .then(CInvert)
                .then(Linear::new(0.9, 5.0))
                .then(CInvert)
                .then(Linear::new(1.1, -2.0))
                .then(CInvert)
                .then(Linear::new(0.8, 10.0));
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 8-op chain: all inverts (no f64 conversion — pure integer)
    group.bench_function("legacy_8_inverts", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("concretize_8_inverts", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let chain = (
                ((CInvert, CInvert), (CInvert, CInvert)),
                ((CInvert, CInvert), (CInvert, CInvert)),
            );
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 7-op chain: odd inverts (not identity — can't be optimized away)
    group.bench_function("legacy_7_inverts", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .invert()
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("concretize_7_inverts", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
            // 7 inverts = ((inv,inv),(inv,inv)), ((inv,inv), inv)
            let chain = (
                ((CInvert, CInvert), (CInvert, CInvert)),
                ((CInvert, CInvert), CInvert),
            );
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.finish();
    let mut group = c.benchmark_group("fusion_4ops_sizes");

    for &size in &[512u32, 2048, 8192] {
        let pixel_count = (size as usize) * (size as usize);
        let pixels = vec![128u8; pixel_count];

        group.bench_with_input(BenchmarkId::new("legacy", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let pipeline = PipelineBuilder::from_source(source)
                    .invert()
                    .unwrap()
                    .linear(1.5, -5.0)
                    .unwrap()
                    .invert()
                    .unwrap()
                    .linear(0.8, 10.0)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(1)
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });

        group.bench_with_input(BenchmarkId::new("concretize", size), &size, |b, &size| {
            b.iter(|| {
                let source = MemorySource::<U8>::new(size, size, 1, pixels.clone()).unwrap();
                let chain = (
                    (CInvert, Linear::new(1.5, -5.0)),
                    (CInvert, Linear::new(0.8, 10.0)),
                );
                let pipeline = PipelineBuilder::from_source(source)
                    .apply(chain)
                    .unwrap()
                    .build()
                    .unwrap();
                let mut sink = MemorySink::for_pipeline(&pipeline);
                RayonScheduler::new(1)
                    .unwrap()
                    .run(&pipeline, &mut sink)
                    .unwrap();
                black_box(sink.into_buffer())
            });
        });
    }

    group.finish();

    // ── Runtime interpreter vs static fusion vs N-node legacy ────────────────
    // This measures the three approaches for 4 and 8 ops:
    // 1. "static" = Concretize chain (1 node, SIMD)
    // 2. "interpreter" = single DynOperation with Vec<enum> loop (1 node, no SIMD)
    // 3. "legacy" = N separate nodes (N buffers, each individually simple)
    let mut group = c.benchmark_group("runtime_vs_static");

    let pixel_count = 2048usize * 2048;
    let pixels = vec![128u8; pixel_count];

    // 8 ops: invert, linear, invert, linear, invert, linear, invert, linear
    group.bench_function("static_8ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let chain = Chain::new()
                .then(CInvert)
                .then(Linear::new(1.2, -3.0))
                .then(CInvert)
                .then(Linear::new(0.9, 5.0))
                .then(CInvert)
                .then(Linear::new(1.1, -2.0))
                .then(CInvert)
                .then(Linear::new(0.8, 10.0));
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("interpreter_8ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let ops = vec![
                PointInstr::Invert,
                PointInstr::Linear(1.2, -3.0),
                PointInstr::Invert,
                PointInstr::Linear(0.9, 5.0),
                PointInstr::Invert,
                PointInstr::Linear(1.1, -2.0),
                PointInstr::Invert,
                PointInstr::Linear(0.8, 10.0),
            ];
            let pipeline = PipelineBuilder::from_source(source)
                .then(Box::new(InterpreterOp::new(ops)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("legacy_8ops_nodes", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .linear(1.2, -3.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.9, 5.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(1.1, -2.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.8, 10.0)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    // 16 ops — more exaggerated
    group.bench_function("static_16ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let chain = Chain::new()
                .then(CInvert)
                .then(Linear::new(1.2, -3.0))
                .then(CInvert)
                .then(Linear::new(0.9, 5.0))
                .then(CInvert)
                .then(Linear::new(1.1, -2.0))
                .then(CInvert)
                .then(Linear::new(0.8, 10.0))
                .then(CInvert)
                .then(Linear::new(1.3, -1.0))
                .then(CInvert)
                .then(Linear::new(0.7, 8.0))
                .then(CInvert)
                .then(Linear::new(1.05, -4.0))
                .then(CInvert)
                .then(Linear::new(0.95, 3.0));
            let pipeline = PipelineBuilder::from_source(source)
                .apply(chain)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("interpreter_16ops", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let ops = vec![
                PointInstr::Invert,
                PointInstr::Linear(1.2, -3.0),
                PointInstr::Invert,
                PointInstr::Linear(0.9, 5.0),
                PointInstr::Invert,
                PointInstr::Linear(1.1, -2.0),
                PointInstr::Invert,
                PointInstr::Linear(0.8, 10.0),
                PointInstr::Invert,
                PointInstr::Linear(1.3, -1.0),
                PointInstr::Invert,
                PointInstr::Linear(0.7, 8.0),
                PointInstr::Invert,
                PointInstr::Linear(1.05, -4.0),
                PointInstr::Invert,
                PointInstr::Linear(0.95, 3.0),
            ];
            let pipeline = PipelineBuilder::from_source(source)
                .then(Box::new(InterpreterOp::new(ops)))
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.bench_function("legacy_16ops_nodes", |b| {
        b.iter(|| {
            let source = MemorySource::<U8>::new(2048, 2048, 1, pixels.clone()).unwrap();
            let pipeline = PipelineBuilder::from_source(source)
                .invert()
                .unwrap()
                .linear(1.2, -3.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.9, 5.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(1.1, -2.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.8, 10.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(1.3, -1.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.7, 8.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(1.05, -4.0)
                .unwrap()
                .invert()
                .unwrap()
                .linear(0.95, 3.0)
                .unwrap()
                .build()
                .unwrap();
            let mut sink = MemorySink::for_pipeline(&pipeline);
            RayonScheduler::new(1)
                .unwrap()
                .run(&pipeline, &mut sink)
                .unwrap();
            black_box(sink.into_buffer())
        });
    });

    group.finish();
}

// ── Interpreter DynOperation for benchmarking ────────────────────────────────

/// Runtime point-op instruction (type-erased).
#[derive(Clone)]
enum PointInstr {
    Invert,
    Linear(f64, f64),
}

/// A single DynOperation that interprets a Vec of point ops in one loop.
/// Simulates what runtime fusion groups would do: 1 node, no SIMD fusion.
struct InterpreterOp {
    ops: Vec<PointInstr>,
}

impl InterpreterOp {
    fn new(ops: Vec<PointInstr>) -> Self {
        Self { ops }
    }
}

impl DynOperation for InterpreterOp {
    fn input_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn output_format(&self) -> BandFormatId {
        BandFormatId::U8
    }

    fn bands(&self) -> u32 {
        1
    }

    fn demand_hint(&self) -> DemandHint {
        DemandHint::ThinStrip
    }

    fn is_pixel_local(&self) -> bool {
        true
    }

    fn transform_metadata(&self, source: &ImageMetadata) -> ImageMetadata {
        source.clone()
    }

    fn node_spec(&self, tile_w: u32, tile_h: u32) -> NodeSpec {
        NodeSpec::identity(tile_w, tile_h)
    }

    fn required_input_region(&self, output: &Region) -> Region {
        *output
    }

    fn dyn_start(&self) -> Box<dyn std::any::Any + Send> {
        Box::new(())
    }

    fn dyn_process_region(
        &self,
        _state: &mut dyn std::any::Any,
        input: &[u8],
        output: &mut [u8],
        _input_region: Region,
        _output_region: Region,
    ) {
        for (d, s) in output.iter_mut().zip(input.iter()) {
            let mut v = *s as f64;
            for op in &self.ops {
                v = match op {
                    PointInstr::Invert => 255.0 - v,
                    PointInstr::Linear(scale, offset) => v * scale + offset,
                };
            }
            *d = v.clamp(0.0, 255.0) as u8;
        }
    }
}

// ── "Ideal wide loop" — theoretical ceiling for fused chains ─────────────────
//
// These loops simulate what LLVM SHOULD produce if we eliminate intermediate
// clamp+conversion between ops. No pipeline overhead, no DynOperation — just
// the raw pixel loop to measure auto-vectorization potential.

fn bench_wide_loop(c: &mut Criterion) {
    let size = 2048usize * 2048;
    let src: Vec<u8> = vec![128u8; size];
    let mut dst: Vec<u8> = vec![0u8; size];

    let mut group = c.benchmark_group("wide_loop_ceiling");

    // ── 8 ops: invert, linear(1.2,-3), invert, linear(0.9,5), ... ───────────
    // Same ops as runtime_vs_static benchmarks for direct comparison.

    // f32 wide: no clamp between ops, 4 NEON lanes expected
    group.bench_function("f32_wide_8ops", |b| {
        let ops: [(f32, f32); 8] = [
            (-1.0, 255.0), // invert
            (1.2, -3.0),
            (-1.0, 255.0), // invert
            (0.9, 5.0),
            (-1.0, 255.0), // invert
            (1.1, -2.0),
            (-1.0, 255.0), // invert
            (0.8, 10.0),
        ];
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut acc: f32 = *s as f32;
                for &(scale, offset) in &ops {
                    acc = acc.mul_add(scale, offset);
                }
                *d = acc.clamp(0.0, 255.0) as u8;
            }
            black_box(&dst);
        });
    });

    // i16 wide: all-integer coefs, 8 NEON lanes expected
    group.bench_function("i16_wide_8ops", |b| {
        // All inverts (scale=-1, offset=255) — guaranteed to fit i16
        let ops: [(i16, i16); 8] = [
            (-1, 255),
            (-1, 255),
            (-1, 255),
            (-1, 255),
            (-1, 255),
            (-1, 255),
            (-1, 255),
            (-1, 255),
        ];
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut acc: i16 = *s as i16;
                for &(scale, offset) in &ops {
                    acc = acc * scale + offset;
                }
                *d = acc.clamp(0, 255) as u8;
            }
            black_box(&dst);
        });
    });

    // i16 wide UNROLLED: same 8 inverts but no loop (static chain)
    group.bench_function("i16_unrolled_8ops", |b| {
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut acc: i16 = *s as i16;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                acc = acc * -1 + 255;
                *d = acc.clamp(0, 255) as u8;
            }
            black_box(&dst);
        });
    });

    // f32 wide UNROLLED: 8 mixed ops, no inner loop
    group.bench_function("f32_unrolled_8ops", |b| {
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut acc: f32 = *s as f32;
                acc = acc.mul_add(-1.0, 255.0); // invert
                acc = acc.mul_add(1.2, -3.0);
                acc = acc.mul_add(-1.0, 255.0); // invert
                acc = acc.mul_add(0.9, 5.0);
                acc = acc.mul_add(-1.0, 255.0); // invert
                acc = acc.mul_add(1.1, -2.0);
                acc = acc.mul_add(-1.0, 255.0); // invert
                acc = acc.mul_add(0.8, 10.0);
                *d = acc.clamp(0.0, 255.0) as u8;
            }
            black_box(&dst);
        });
    });

    // Baseline: current Concretize behavior (with intermediate clamp)
    // Simulates what pt_linear does today per-op
    group.bench_function("f32_with_clamp_8ops", |b| {
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                let mut acc: u8 = *s;
                // Each op: u8→f32→math→clamp→u8 (the current behavior)
                acc = (acc as f32).mul_add(-1.0, 255.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(1.2, -3.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(-1.0, 255.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(0.9, 5.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(-1.0, 255.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(1.1, -2.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(-1.0, 255.0).clamp(0.0, 255.0) as u8;
                acc = (acc as f32).mul_add(0.8, 10.0).clamp(0.0, 255.0) as u8;
                *d = acc;
            }
            black_box(&dst);
        });
    });

    // ── NEW: apply_wide via the real Concretize trait ────────────────────────
    // This tests the actual trait implementation, not hand-written loops.

    // apply_wide with f32 accumulator (mixed ops with fractional coefs)
    group.bench_function("trait_f32_wide_8ops", |b| {
        use viprs::domain::concretize::apply_chain_wide_u8;
        let chain = Chain::new()
            .then(CInvert)
            .then(Linear::new(1.2, -3.0))
            .then(CInvert)
            .then(Linear::new(0.9, 5.0))
            .then(CInvert)
            .then(Linear::new(1.1, -2.0))
            .then(CInvert)
            .then(Linear::new(0.8, 10.0));
        b.iter(|| {
            apply_chain_wide_u8::<f32, _>(&chain, &src, &mut dst);
            black_box(&dst);
        });
    });

    // apply_wide with i16 accumulator (all inverts — integer coefs)
    group.bench_function("trait_i16_wide_8ops", |b| {
        use viprs::domain::concretize::apply_chain_wide_u8;
        let chain = Chain::new()
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert);
        b.iter(|| {
            apply_chain_wide_u8::<i16, _>(&chain, &src, &mut dst);
            black_box(&dst);
        });
    });

    // apply_wide_auto: let the chain decide (should pick i16 for all-inverts)
    group.bench_function("trait_auto_wide_8inverts", |b| {
        use viprs::domain::concretize::apply_chain_wide_u8_auto;
        let chain = Chain::new()
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert)
            .then(CInvert);
        b.iter(|| {
            apply_chain_wide_u8_auto(&chain, &src, &mut dst);
            black_box(&dst);
        });
    });

    group.finish();
}

/// Benchmark: Core + Post-op absorption.
///
/// Simulates a realistic pipeline: resize(0.5) followed by 4 point ops.
/// Compares:
/// A. Two nodes with intermediate buffer (current pipeline behavior)
/// B. Single node with in-place post-apply (proposed optimization)
///
/// The core op (resize) is simulated as a memcpy-equivalent workload to isolate
/// the cost of the extra buffer read+write cycle.
fn bench_core_post(c: &mut Criterion) {
    use viprs::domain::concretize::{
        Concretize, Width, apply_chain_wide_u8, apply_chain_wide_u8_inplace,
    };

    // Simulate 2048² → 1024² resize (0.5× on each axis)
    let core_output_size = 1024usize * 1024; // resize output pixels
    let bands = 3u32;
    let bytes = core_output_size * bands as usize;

    // Pre-allocated buffers
    let core_input = vec![128u8; 2048 * 2048 * bands as usize];
    let mut intermediate_buf = vec![0u8; bytes];
    let mut output_buf = vec![0u8; bytes];

    // The post-chain: 4 point ops (invert, linear, invert, linear)
    let chain = (
        (CInvert, Linear::new(1.2, -3.0)),
        (CInvert, Linear::new(0.9, 5.0)),
    );

    let mut group = c.benchmark_group("core_post_absorption");

    // Simulate core op: naive 2× downsample (take every other pixel)
    #[inline(always)]
    fn simulate_resize(input: &[u8], output: &mut [u8], bands: usize) {
        // 2× shrink: skip every other pixel (bands bytes per pixel)
        let src_stride = bands * 2; // skip one pixel
        let mut out_idx = 0;
        let mut in_idx = 0;
        let in_len = input.len();
        while in_idx + bands <= in_len && out_idx + bands <= output.len() {
            output[out_idx..out_idx + bands].copy_from_slice(&input[in_idx..in_idx + bands]);
            out_idx += bands;
            in_idx += src_stride;
        }
    }

    // A. Two-node: core → intermediate → post-chain → output
    group.bench_function("two_nodes_with_buffer", |b| {
        b.iter(|| {
            // Step 1: Core writes to intermediate buffer
            simulate_resize(&core_input, &mut intermediate_buf, bands as usize);
            // Step 2: Post-chain reads intermediate, writes to output
            apply_chain_wide_u8::<i16, _>(&chain, &intermediate_buf, &mut output_buf);
            black_box(&output_buf);
        });
    });

    // B. Single-node: core → output → post-chain in-place on output
    group.bench_function("single_node_inplace", |b| {
        b.iter(|| {
            // Step 1: Core writes directly to output
            simulate_resize(&core_input, &mut output_buf, bands as usize);
            // Step 2: Post-chain applied in-place on same buffer (cache-hot)
            apply_chain_wide_u8_inplace::<i16, _>(&chain, &mut output_buf);
            black_box(&output_buf);
        });
    });

    // C. Baseline: core only (no post ops) — to measure post-chain cost in isolation
    group.bench_function("core_only_no_post", |b| {
        b.iter(|| {
            simulate_resize(&core_input, &mut output_buf, bands as usize);
            black_box(&output_buf);
        });
    });

    // D. Post-chain only (to measure raw chain cost)
    group.bench_function("post_chain_only", |b| {
        // Pre-fill intermediate with data
        intermediate_buf.fill(128);
        b.iter(|| {
            apply_chain_wide_u8::<i16, _>(&chain, &intermediate_buf, &mut output_buf);
            black_box(&output_buf);
        });
    });

    // ── Pixel-local non-point core: simulated colour conversion ──────────────
    // sRGB→Lab-like: reads 3 bands per pixel, writes 3 bands, but NOT a point op
    // (each output band depends on all input bands). This cannot be fused into
    // the Concretize chain.

    let colour_size = 1024usize * 1024;
    let colour_bytes = colour_size * 3; // 3 bands
    let colour_input_buf = vec![128u8; colour_bytes];
    let mut colour_intermediate = vec![0u8; colour_bytes];
    let mut colour_output = vec![0u8; colour_bytes];

    // Simulated sRGB→linear: per-pixel, cross-band (gamma decode + matrix)
    #[inline(never)]
    fn simulate_colour_convert(input: &[u8], output: &mut [u8]) {
        let pixels = input.len() / 3;
        for i in 0..pixels {
            let r = input[i * 3] as f32 / 255.0;
            let g = input[i * 3 + 1] as f32 / 255.0;
            let b = input[i * 3 + 2] as f32 / 255.0;
            let rl = r * r;
            let gl = g * g;
            let bl = b * b;
            let l = (0.4124 * rl + 0.3576 * gl + 0.1805 * bl).min(1.0);
            let a = (0.2126 * rl + 0.7152 * gl + 0.0722 * bl).min(1.0);
            let bb_val = (0.0193 * rl + 0.1192 * gl + 0.9505 * bl).min(1.0);
            output[i * 3] = (l * 255.0) as u8;
            output[i * 3 + 1] = (a * 255.0) as u8;
            output[i * 3 + 2] = (bb_val * 255.0) as u8;
        }
    }

    // E. colour_convert → buffer → post_chain → output (two nodes)
    group.bench_function("colour_two_nodes", |b| {
        b.iter(|| {
            simulate_colour_convert(&colour_input_buf, &mut colour_intermediate);
            apply_chain_wide_u8::<i16, _>(&chain, &colour_intermediate, &mut colour_output);
            black_box(&colour_output);
        });
    });

    // F. colour_convert → output → post_chain in-place (single node)
    group.bench_function("colour_single_inplace", |b| {
        b.iter(|| {
            simulate_colour_convert(&colour_input_buf, &mut colour_output);
            apply_chain_wide_u8_inplace::<i16, _>(&chain, &mut colour_output);
            black_box(&colour_output);
        });
    });

    // G. colour_convert only (baseline)
    group.bench_function("colour_only", |b| {
        b.iter(|| {
            simulate_colour_convert(&colour_input_buf, &mut colour_output);
            black_box(&colour_output);
        });
    });

    // ── Chain of pixel-local non-point ops ────────────────────────────────────
    // Three colour converts chained: measures buffer overhead between non-point ops

    let mut buf_a = vec![0u8; colour_bytes];
    let mut buf_b = vec![0u8; colour_bytes];

    // H. Three colour converts with 3 buffers (current pipeline behavior)
    group.bench_function("3_colour_separate_bufs", |b| {
        b.iter(|| {
            simulate_colour_convert(&colour_input_buf, &mut buf_a);
            simulate_colour_convert(&buf_a, &mut buf_b);
            simulate_colour_convert(&buf_b, &mut colour_output);
            black_box(&colour_output);
        });
    });

    // I. Three colour converts, 2-buffer ping-pong (minimal allocation)
    group.bench_function("3_colour_pingpong", |b| {
        b.iter(|| {
            simulate_colour_convert(&colour_input_buf, &mut buf_a);
            simulate_colour_convert(&buf_a, &mut buf_b);
            simulate_colour_convert(&buf_b, &mut colour_output);
            black_box(&colour_output);
        });
    });

    group.finish();
}

/// LUT-based fusion: precompute the entire chain as a 256-entry u8→u8 table.
/// Single `vqtbl1q_u8` processes 16 bytes per NEON instruction.
fn bench_lut_fusion(c: &mut Criterion) {
    let size = 1024usize * 1024 * 3; // 1024² × 3 bands (same as colour benchmark)
    let src = vec![128u8; size];
    let mut dst = vec![0u8; size];

    // Build LUT for chain: invert → linear(1.2, -3) → invert → linear(0.9, 5)
    let mut lut = [0u8; 256];
    for i in 0..256u16 {
        let mut v = i as f32;
        // invert
        v = 255.0 - v;
        // linear(1.2, -3)
        v = v * 1.2 + (-3.0);
        // invert
        v = 255.0 - v;
        // linear(0.9, 5)
        v = v * 0.9 + 5.0;
        lut[i as usize] = v.clamp(0.0, 255.0) as u8;
    }

    let mut group = c.benchmark_group("lut_vs_wide");

    // Scalar LUT
    group.bench_function("lut_scalar", |b| {
        b.iter(|| {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = lut[*s as usize];
            }
            black_box(&dst);
        });
    });

    // NEON LUT (vqtbl1q_u8 = 16 bytes per instruction, needs 256/16=16 table regs)
    // For u8→u8 with full 256 entries, use vqtbl4q_u8 (64-byte table, 4 passes)
    #[cfg(target_arch = "aarch64")]
    group.bench_function("lut_neon_tbl", |b| {
        use std::arch::aarch64::*;
        // Split LUT into 4 × 64-byte tables for vqtbl4q
        let tbl0: [u8; 64] = lut[0..64].try_into().unwrap();
        let tbl1: [u8; 64] = lut[64..128].try_into().unwrap();
        let tbl2: [u8; 64] = lut[128..192].try_into().unwrap();
        let tbl3: [u8; 64] = lut[192..256].try_into().unwrap();

        b.iter(|| {
            // SAFETY: aligned loads from our buffers, aarch64 target verified by cfg
            unsafe {
                let t0 = vld1q_u8_x4(tbl0.as_ptr());
                let t1 = vld1q_u8_x4(tbl1.as_ptr());
                let t2 = vld1q_u8_x4(tbl2.as_ptr());
                let t3 = vld1q_u8_x4(tbl3.as_ptr());
                let offset_64 = vdupq_n_u8(64);
                let offset_128 = vdupq_n_u8(128);
                let offset_192 = vdupq_n_u8(192);

                let chunks = size / 16;
                for i in 0..chunks {
                    let idx = i * 16;
                    let input = vld1q_u8(src.as_ptr().add(idx));

                    // 4-range lookup: [0,63], [64,127], [128,191], [192,255]
                    let r0 = vqtbl4q_u8(t0, input);
                    let idx1 = vsubq_u8(input, offset_64);
                    let r1 = vqtbl4q_u8(t1, idx1);
                    let idx2 = vsubq_u8(input, offset_128);
                    let r2 = vqtbl4q_u8(t2, idx2);
                    let idx3 = vsubq_u8(input, offset_192);
                    let r3 = vqtbl4q_u8(t3, idx3);

                    // Combine (out-of-range returns 0, so OR merges them)
                    let combined = vorrq_u8(vorrq_u8(r0, r1), vorrq_u8(r2, r3));
                    vst1q_u8(dst.as_mut_ptr().add(idx), combined);
                }
            }
            black_box(&dst);
        });
    });

    // WideAccum i16 (current best path) for comparison
    {
        use viprs::domain::concretize::apply_chain_wide_u8;
        use viprs::domain::ops::point::{Invert as CInvert, Linear};
        let chain = (
            (CInvert, Linear::new(1.2, -3.0)),
            (CInvert, Linear::new(0.9, 5.0)),
        );
        group.bench_function("wide_i16_chain", |b| {
            b.iter(|| {
                apply_chain_wide_u8::<i16, _>(&chain, &src, &mut dst);
                black_box(&dst);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches_all,
    bench_fusion_chain,
    bench_wide_loop,
    bench_core_post,
    bench_lut_fusion
);
criterion_main!(benches_all);
