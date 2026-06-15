#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::BandFormatId, op::DynOperation, ops::arithmetic::add_images::AddImages,
};

fn format_sample_size(id: BandFormatId) -> usize {
    match id {
        BandFormatId::U8 => 1,
        BandFormatId::U16 | BandFormatId::I16 => 2,
        BandFormatId::U32 | BandFormatId::I32 | BandFormatId::F32 => 4,
        BandFormatId::F64 => 8,
    }
}

fn bench_add_images(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_images_u8");

    for &size in &common::STANDARD_SIZES {
        let op = AddImages::new(3, BandFormatId::U8);
        let output_region = common::tile_region(op.demand_hint(), size);
        let input_regions = [output_region; 2];
        let lhs = vec![64u8; common::sample_count(output_region, 3)];
        let rhs = vec![32u8; common::sample_count(output_region, 3)];
        let mut output = vec![
            0u8;
            common::sample_count(output_region, 3)
                * format_sample_size(op.output_format())
        ];

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

criterion_group!(benches, bench_add_images);
criterion_main!(benches);
