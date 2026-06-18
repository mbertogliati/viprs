#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::BandFormatId, image::Region, op::DynOperation, ops::structural::insert::Insert,
};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert_u8_rgb");

    for &size in &common::STANDARD_SIZES {
        let op = Insert::new(
            size,
            size,
            size / 2,
            size / 2,
            size as i32 / 4,
            size as i32 / 4,
            false,
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
        let main = vec![32u8; common::sample_count(input_regions[0], 3)];
        let sub = vec![224u8; common::sample_count(input_regions[1], 3)];
        let mut output = vec![0u8; common::sample_count(output_region, 3)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&main[..], &sub[..]];
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

criterion_group!(benches, bench_insert);
criterion_main!(benches);
