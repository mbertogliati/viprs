#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::Region,
    op::DynOperation,
    ops::mosaicing::{LrMerge, TbMerge},
};

const BANDS: u32 = 1;
const BLEND_WIDTH: u32 = 64;

fn patterned_pixels(width: u32, height: u32, origin_x: i64, origin_y: i64) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            let global_x = origin_x + i64::from(x);
            let global_y = origin_y + i64::from(y);
            let value =
                (global_x * 31 + global_y * 17 + global_x * global_y * 13).rem_euclid(251) + 1;
            pixels.push(value as u8);
        }
    }
    pixels
}

fn output_region(
    op: &dyn DynOperation,
    output_width: u32,
    output_height: u32,
    size: u32,
) -> Region {
    let tile_height = op.demand_hint().tile_height(size, size);
    Region::new(0, 0, output_width, tile_height.min(output_height))
}

fn bench_lrmerge(c: &mut Criterion) {
    let mut group = c.benchmark_group("lrmerge_u8");

    for &size in &common::STANDARD_SIZES {
        let overlap = (size / 4).max(BLEND_WIDTH);
        let shift_x = size - overlap;
        let op = LrMerge::<U8>::new(size, size, size, size, -(shift_x as i32), 0, overlap, BANDS);
        let output_region = output_region(&op, op.output_width(), op.output_height(), size);
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let reference = patterned_pixels(
            input_regions[0].width,
            input_regions[0].height,
            i64::from(input_regions[0].x),
            i64::from(input_regions[0].y),
        );
        let secondary = patterned_pixels(
            input_regions[1].width,
            input_regions[1].height,
            i64::from(shift_x) + i64::from(input_regions[1].x),
            i64::from(input_regions[1].y),
        );
        let mut output = vec![0u8; common::sample_count(output_region, BANDS)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&reference[..], &secondary[..]];
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

fn bench_tbmerge(c: &mut Criterion) {
    let mut group = c.benchmark_group("tbmerge_u8");

    for &size in &common::STANDARD_SIZES {
        let overlap = (size / 4).max(BLEND_WIDTH);
        let shift_y = size - overlap;
        let op = TbMerge::<U8>::new(size, size, size, size, 0, -(shift_y as i32), overlap, BANDS);
        let output_region = output_region(&op, op.output_width(), op.output_height(), size);
        let input_regions = [
            op.required_input_region_slot(&output_region, 0),
            op.required_input_region_slot(&output_region, 1),
        ];
        let reference = patterned_pixels(
            input_regions[0].width,
            input_regions[0].height,
            i64::from(input_regions[0].x),
            i64::from(input_regions[0].y),
        );
        let secondary = patterned_pixels(
            input_regions[1].width,
            input_regions[1].height,
            i64::from(input_regions[1].x),
            i64::from(shift_y) + i64::from(input_regions[1].y),
        );
        let mut output = vec![0u8; common::sample_count(output_region, BANDS)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let inputs = [&reference[..], &secondary[..]];
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

criterion_group!(benches, bench_lrmerge, bench_tbmerge);
criterion_main!(benches);
