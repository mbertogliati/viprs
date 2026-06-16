#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use viprs::domain::{
    format::U8,
    image::{Region, Tile},
    ops::mosaicing::{LrMosaicOp, TbMosaicOp},
};

const BANDS: u32 = 1;
const BLEND_WIDTH: u32 = 64;
const SEARCH_RADIUS: u32 = 8;

#[inline]
fn clamped_shift(size: u32, preferred: u32, floor: u32) -> u32 {
    if size <= 1 {
        0
    } else {
        preferred.max(floor).min(size - 1)
    }
}

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

// Criterion-only baseline: `AutoMosaicSearch` is internal (`pub(super)`), so the public
// left-right and top-bottom wrappers are the narrowest stable entrypoints for measuring the
// tie-point refinement stage. The xtask/libvips runner does not expose this search helper.
fn bench_auto_mosaic_lr(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_mosaic_lr_u8");

    for &size in &common::STANDARD_SIZES {
        let shift_x = clamped_shift(size, size / 4, BLEND_WIDTH);
        let shift_y = clamped_shift(size, size / 32, 1);
        let op = LrMosaicOp::<U8>::new(
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

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let reference_tile = Tile::<U8>::new(region, BANDS, &reference);
                let secondary_tile = Tile::<U8>::new(region, BANDS, &secondary);
                match op.detect_offset(&reference_tile, &secondary_tile) {
                    Ok(found) => black_box(found),
                    Err(err) => panic!("auto-mosaic LR benchmark failed: {err}"),
                }
            });
        });
    }

    group.finish();
}

fn bench_auto_mosaic_tb(c: &mut Criterion) {
    let mut group = c.benchmark_group("auto_mosaic_tb_u8");

    for &size in &common::STANDARD_SIZES {
        let shift_x = clamped_shift(size, size / 32, 1);
        let shift_y = clamped_shift(size, size / 4, BLEND_WIDTH);
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

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let reference_tile = Tile::<U8>::new(region, BANDS, &reference);
                let secondary_tile = Tile::<U8>::new(region, BANDS, &secondary);
                match op.detect_offset(&reference_tile, &secondary_tile) {
                    Ok(found) => black_box(found),
                    Err(err) => panic!("auto-mosaic TB benchmark failed: {err}"),
                }
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_auto_mosaic_lr, bench_auto_mosaic_tb);
criterion_main!(benches);
