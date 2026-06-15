#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Tile, TileMut},
    op::Op,
    ops::histogram::clahe::ClaheOp,
};

fn bench_clahe(c: &mut Criterion) {
    let mut group = c.benchmark_group("clahe_u8");

    for &size in &common::STANDARD_SIZES {
        let op = ClaheOp::<U8>::new(64, 64, 4.0).unwrap();
        let (input_region, output_region) = common::direct_tile_regions(&op, size);
        let input = vec![128u8; common::sample_count(input_region, 1)];
        let mut output = vec![0u8; common::sample_count(output_region, 1)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let input_tile = Tile::<U8>::new(input_region, 1, &input);
                let mut output_tile = TileMut::<U8>::new(output_region, 1, &mut output);
                let mut state = op.start();
                op.process_region(&mut state, &input_tile, &mut output_tile);
                black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_clahe);
criterion_main!(benches);
