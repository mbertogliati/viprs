#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::BandFormatId,
    image::Region,
    op::DynOperation,
    ops::structural::join::{Join, JoinDirection},
};

fn bench_join_horizontal(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_horizontal_u8_rgb");

    for &size in &common::STANDARD_SIZES {
        let op = Join::new(
            JoinDirection::Horizontal,
            size,
            size,
            size / 2,
            size,
            3,
            BandFormatId::U8,
        );
        let tile = common::tile_region(op.demand_hint(), size);
        let output_region =
            Region::new(0, 0, op.output_width(), tile.height.min(op.output_height()));
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let left = vec![48u8; common::sample_count(input_regions[0], 3)];
        let right = vec![192u8; common::sample_count(input_regions[1], 3)];
        let mut output = vec![0u8; common::sample_count(output_region, 3)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&left[..], &right[..]];
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

fn bench_join_vertical(c: &mut Criterion) {
    let mut group = c.benchmark_group("join_vertical_u8_rgb");

    for &size in &common::STANDARD_SIZES {
        let op = Join::new(
            JoinDirection::Vertical,
            size,
            size,
            size,
            size / 2,
            3,
            BandFormatId::U8,
        );
        let tile = common::tile_region(op.demand_hint(), size);
        let output_region =
            Region::new(0, 0, op.output_width(), tile.height.min(op.output_height()));
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let top = vec![48u8; common::sample_count(input_regions[0], 3)];
        let bottom = vec![192u8; common::sample_count(input_regions[1], 3)];
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

criterion_group!(benches, bench_join_horizontal, bench_join_vertical);
criterion_main!(benches);
