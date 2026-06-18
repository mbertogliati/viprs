#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{format::BandFormatId, op::DynOperation, ops::conversion::ArrayJoinOp};

fn bench_arrayjoin(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrayjoin_u8");

    for &size in &common::STANDARD_SIZES {
        let op = ArrayJoinOp::new(4, 3, BandFormatId::U8);
        let output_region = common::tile_region(op.demand_hint(), size);
        let input_regions = [output_region; 4];
        let samples = common::sample_count(output_region, 3);
        let input0 = vec![32u8; samples];
        let input1 = vec![96u8; samples];
        let input2 = vec![160u8; samples];
        let input3 = vec![224u8; samples];
        let mut output = vec![0u8; common::sample_count(output_region, op.bands())];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&input0[..], &input1[..], &input2[..], &input3[..]];
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

criterion_group!(benches, bench_arrayjoin);
criterion_main!(benches);
