#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::Region,
    op::DynOperation,
    ops::mosaicing::{MergeDirection, MergeOp},
};

fn bench_merge_h(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_h_u8");

    for &size in &common::STANDARD_SIZES {
        let op = MergeOp::<U8>::new(
            MergeDirection::Horizontal,
            size,
            size,
            size,
            size,
            -(size as i32 / 4),
            0,
            128,
            3,
        );
        let tile = common::tile_region(op.demand_hint(), size);
        let output_region =
            Region::new(0, 0, op.output_width(), tile.height.min(op.output_height()));
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let lhs = vec![96u8; common::sample_count(input_regions[0], 3)];
        let rhs = vec![160u8; common::sample_count(input_regions[1], 3)];
        let mut output = vec![0u8; common::sample_count(output_region, 3)];

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

fn bench_merge_v(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_v_u8");

    for &size in &common::STANDARD_SIZES {
        let op = MergeOp::<U8>::new(
            MergeDirection::Vertical,
            size,
            size,
            size,
            size,
            0,
            -(size as i32 / 4),
            128,
            3,
        );
        let tile = common::tile_region(op.demand_hint(), size);
        let output_region =
            Region::new(0, 0, op.output_width(), tile.height.min(op.output_height()));
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let top = vec![96u8; common::sample_count(input_regions[0], 3)];
        let bottom = vec![160u8; common::sample_count(input_regions[1], 3)];
        let mut output = vec![0u8; common::sample_count(output_region, 3)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&top[..], &bottom[..]];
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

criterion_group!(benches, bench_merge_h, bench_merge_v);
criterion_main!(benches);
