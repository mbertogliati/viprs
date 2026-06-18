#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Region, Tile},
    op::DynOperation,
    ops::mosaicing::TbMosaicOp,
};

const BANDS: u32 = 1;
const BLEND_WIDTH: u32 = 64;
const SEARCH_RADIUS: u32 = 8;

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

// Criterion-only baseline: the xtask/libvips runner is wired for single-input fixture ops,
// while `TbMosaicOp` needs two synthetic inputs plus tie-point refinement before merge.
fn bench_tbmosaic(c: &mut Criterion) {
    let mut group = c.benchmark_group("tbmosaic_u8");

    for &size in &common::STANDARD_SIZES {
        let shift_x = (size / 32).max(1);
        let shift_y = (size / 4).max(BLEND_WIDTH);
        let op = TbMosaicOp::<U8>::new(
            size,
            size,
            size,
            size,
            shift_x as i32,
            shift_y as i32,
            0,
            0,
            SEARCH_RADIUS,
            BLEND_WIDTH,
            BANDS,
        );
        let reference = patterned_pixels(size, size, 0, 0);
        let secondary = patterned_pixels(size, size, i64::from(shift_x), i64::from(shift_y));
        let region = Region::new(0, 0, size, size);
        let reference_tile = Tile::<U8>::new(region, BANDS, &reference);
        let secondary_tile = Tile::<U8>::new(region, BANDS, &secondary);
        let (_, warm_merge) = op
            .detect_and_build_merge(&reference_tile, &secondary_tile)
            .unwrap_or_else(|err| panic!("tbmosaic warmup failed: {err}"));
        let output_region =
            Region::new(0, 0, warm_merge.output_width(), warm_merge.output_height());
        let input_regions = [
            warm_merge.required_input_region_slot(&output_region, 0),
            warm_merge.required_input_region_slot(&output_region, 1),
        ];
        let mut output = vec![0u8; common::sample_count(output_region, BANDS)];

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let reference_tile = Tile::<U8>::new(region, BANDS, &reference);
                let secondary_tile = Tile::<U8>::new(region, BANDS, &secondary);
                let (_, merge) = op
                    .detect_and_build_merge(&reference_tile, &secondary_tile)
                    .unwrap_or_else(|err| panic!("tbmosaic benchmark failed: {err}"));
                let mut state = ();
                merge.dyn_process_region_multi(
                    &mut state,
                    &[&reference, &secondary],
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

criterion_group!(benches, bench_tbmosaic);
criterion_main!(benches);
