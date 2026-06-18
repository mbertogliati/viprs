#![allow(missing_docs)]
#[path = "../common/mod.rs"]
mod common;

use criterion::{Criterion, criterion_group, criterion_main};
use viprs::domain::{
    colorspace::{Rgb16, SRgb},
    format::{U8, U16},
    ops::colour::srgb_to_rgb16::SRgbToRgb16,
};

fn bench_srgb_to_rgb16(c: &mut Criterion) {
    common::bench_colour_convert_with_regions::<SRgbToRgb16, SRgb, Rgb16, U8, U16, _, _, _>(
        c,
        "srgb_to_rgb16",
        3,
        3,
        || SRgbToRgb16,
        common::colour_convert_tile_regions::<SRgbToRgb16, SRgb, Rgb16>,
        |samples| {
            (0..samples)
                .map(|idx| ((idx * 31 + idx / 3 * 17) % 256) as u8)
                .collect()
        },
    );
}

criterion_group!(benches, bench_srgb_to_rgb16);
criterion_main!(benches);
