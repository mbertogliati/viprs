#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{format::U8, op::DynOperation, ops::conversion::IfThenElseOp};

fn bench_ifthenelse(c: &mut Criterion) {
    let mut group = c.benchmark_group("ifthenelse_u8_rgb");

    for &size in &common::STANDARD_SIZES {
        let op = IfThenElseOp::<U8>::new(3);
        let region = common::tile_region(op.demand_hint(), size);
        let pixels = common::sample_count(region, op.combined_bands());
        let input = (0..pixels)
            .map(|index| {
                if index % op.combined_bands() as usize == 0 {
                    if (index / op.combined_bands() as usize) % 2 == 0 {
                        255
                    } else {
                        0
                    }
                } else {
                    (index % 251) as u8
                }
            })
            .collect::<Vec<_>>();
        let mut output = vec![0u8; common::sample_count(region, op.bands())];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let mut state = op.dyn_start();
                op.dyn_process_region(state.as_mut(), &input, &mut output, region, region);
                black_box(&output);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ifthenelse);
criterion_main!(benches);
