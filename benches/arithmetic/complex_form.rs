#[path = "../common/mod.rs"]
mod common;

use bytemuck::cast_slice;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{op::DynOperation, ops::arithmetic::ComplexFormOp};

fn bench_complex_form(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_form_f32");

    for &size in &common::STANDARD_SIZES {
        let op = ComplexFormOp::new(1);
        let output_region = common::tile_region(op.demand_hint(), size);
        let input_regions = [output_region; 2];
        let real = (0..common::sample_count(output_region, 1))
            .map(|index| index as f32 * 0.25 + 1.0)
            .collect::<Vec<_>>();
        let imag = (0..common::sample_count(output_region, 1))
            .map(|index| index as f32 * 0.125 + 0.5)
            .collect::<Vec<_>>();
        let mut output =
            vec![0u8; common::sample_count(output_region, op.bands()) * std::mem::size_of::<f32>()];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [cast_slice(&real), cast_slice(&imag)];
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

criterion_group!(benches, bench_complex_form);
criterion_main!(benches);
