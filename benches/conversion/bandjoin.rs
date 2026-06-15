#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{format::BandFormatId, op::DynOperation, ops::conversion::bandjoin::BandJoin};

fn bench_bandjoin(c: &mut Criterion) {
    let mut group = c.benchmark_group("bandjoin_u8");

    for &size in &common::STANDARD_SIZES {
        let op = BandJoin::new(3, 1, BandFormatId::U8);
        let output_region = common::tile_region(op.demand_hint(), size);
        let input_regions = [output_region; 2];
        let lhs = vec![96u8; common::sample_count(output_region, 3)];
        let rhs = vec![255u8; common::sample_count(output_region, 1)];
        let mut output = vec![0u8; common::sample_count(output_region, 4)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&lhs[..], &rhs[..]];
                let mut state = op.dyn_start();
                op.dyn_process_region_multi(
                    state.as_mut(),
                    &inputs,
                    &mut output,
                    &input_regions,
                    output_region,
                );
                black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_bandjoin);
criterion_main!(benches);
